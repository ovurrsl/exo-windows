from __future__ import annotations

import ctypes
import importlib.util
import multiprocessing
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

if sys.platform != "win32":
    import resource

import loguru

if TYPE_CHECKING:
    from exo.shared.types.events import Event, RunnerEvent
    from exo.shared.types.tasks import Task, TaskId
    from exo.shared.types.worker.instances import BoundInstance, Instance
    from exo.utils.channels import MpReceiver, MpSender
    from exo.worker.engines.base import Builder

logger: loguru.Logger = loguru.logger

_DLL_DIRECTORY_HANDLES: list[object] = []
_REGISTERED_DLL_DIRS: set[str] = set()

_NVIDIA_CUDA_PACKAGES: tuple[str, ...] = (
    "cuda_runtime",
    "cublas",
    "cudnn",
    "cuda_nvrtc",
    "cufft",
    "cusolver",
    "nvjitlink",
    "cu13",
)


def _register_win32_dll_directories(directories: list[Path]) -> None:
    if sys.platform != "win32":
        return

    existing_path_entries = {
        os.path.normcase(entry)
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry
    }
    prepended: list[str] = []

    for directory in directories:
        if not directory.is_dir():
            continue
        dir_str = str(directory)
        normalized = os.path.normcase(dir_str)
        if normalized not in _REGISTERED_DLL_DIRS:
            _REGISTERED_DLL_DIRS.add(normalized)
            handle = os.add_dll_directory(dir_str)
            _DLL_DIRECTORY_HANDLES.append(handle)
        if normalized not in existing_path_entries:
            prepended.append(dir_str)
            existing_path_entries.add(normalized)

    if prepended:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = (
            f"{os.pathsep.join(prepended)}{os.pathsep}{current_path}"
            if current_path
            else os.pathsep.join(prepended)
        )


def _configure_cuda_environment() -> None:
    """Configure CUDA_HOME and Windows DLL search paths before importing MLX."""
    if sys.platform not in ("linux", "win32"):
        return

    nvidia_spec = importlib.util.find_spec("nvidia")
    nvidia_locations = (
        [Path(loc) for loc in nvidia_spec.submodule_search_locations]
        if nvidia_spec is not None and nvidia_spec.submodule_search_locations
        else []
    )

    if not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH"):
        # Check cuda_runtime (CUDA 12 wheels) first, then cu13 (CUDA 13 wheels).
        for subdir_name in ("cuda_runtime", "cu13"):
            found_cuda_home = False
            for location in nvidia_locations:
                candidate = location / subdir_name
                if (candidate / "include").is_dir():
                    os.environ["CUDA_HOME"] = str(candidate)
                    logger.info(
                        f"CUDA_HOME unset; using bundled CUDA headers at {candidate}"
                    )
                    found_cuda_home = True
                    break
            if found_cuda_home:
                break

    if sys.platform != "win32":
        return

    dll_directories: list[Path] = []

    try:
        mlx_spec = importlib.util.find_spec("mlx")
    except ImportError:
        mlx_spec = None
    if mlx_spec is not None:
        mlx_roots: list[Path] = []
        if mlx_spec.origin is not None:
            mlx_roots.append(Path(mlx_spec.origin).parent)
        if mlx_spec.submodule_search_locations:
            mlx_roots.extend(Path(loc) for loc in mlx_spec.submodule_search_locations)
        for root in mlx_roots:
            if (root / "mlx.dll").is_file() or root.is_dir():
                dll_directories.append(root)
            lib_dir = root / "lib"
            if lib_dir.is_dir():
                dll_directories.append(lib_dir)

    for location in nvidia_locations:
        for package_name in _NVIDIA_CUDA_PACKAGES:
            package_dir = location / package_name
            dll_directories.append(package_dir / "bin")
            dll_directories.append(package_dir / "bin" / "x86_64")

    _register_win32_dll_directories(dll_directories)


@dataclass(frozen=True)
class RunnerTerminationError:
    exception_type: str
    exception_message: str
    exception_repr: str
    traceback: str

    @classmethod
    def from_exception(cls, e: Exception) -> Self:
        return cls(
            exception_type=type(e).__qualname__,
            exception_message=str(e),
            exception_repr=repr(e),
            traceback="".join(
                traceback.TracebackException.from_exception(e).format(chain=True)
            ),
        )

    def __str__(self) -> str:
        return f"{self.exception_type}: {self.exception_message}\n{self.traceback}"


def use_fast_synch(instance: Instance, override: str | None) -> bool:
    """Whether MLX should synchronise its CPU and GPU work by spinning on shared memory.

    It saves latency on every hand-over between the GPU and the CPU, which an RDMA
    (JACCL) instance makes for each collective. Over the TCP ring it makes no measurable
    difference to a model's speed, and it breaks down when two runners share a Mac's GPU:
    both models slow to a crawl, the ring fails with EFAULT, or the GPU deadlocks.
    A model family can turn it off for its models (FamilySpec.fast_synch).
    `--fast-synch` / `--no-fast-synch` (EXO_FAST_SYNCH) override the choice.
    """
    if override == "true":
        return True
    if override == "false":
        return False

    from exo.shared.types.worker.instances import MlxJacclInstance
    from exo.worker.engines.mlx.models.spec_registry import fast_synch_allowed

    return isinstance(instance, MlxJacclInstance) and all(
        fast_synch_allowed(shard.model_card.model_type)
        for shard in instance.shard_assignments.runner_to_shard.values()
    )


def exit_with_parent() -> None:
    """End this process as soon as the process that started it (the node) is gone.

    A runner waits on pipes to its node, and holds both ends of them, so it never sees an end
    of file when the node dies: killed with SIGKILL (a crash, or the OS ending it), exo leaves
    its runners running for good, each holding its model's memory, and a restarted exo can't
    load the model. multiprocessing gives a spawned process the read end of a pipe whose write
    end only its parent holds, while the parent's Process object for it is open (it can't be
    closed, nor collected, while the child runs): `parent_process().join()` returns when the
    parent dies, and not before, so this can't fire while the node lives.

    The thread needs the GIL to exit. A runner inside native code that holds it exits only once
    that call returns: a lower rank orphaned in MLX's ring set-up, which waits in accept with
    the GIL held (JACCL's set-up may too), keeps its process and the instance's port, though no
    model memory. A watcher that doesn't need the GIL would close that gap."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        # No one is left to tell or to clean up for: the node's memory and GPU state go with
        # this process, and its resource tracker releases its semaphores once it is gone
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def _terminate_win32_runner(exit_code: int) -> None:
    """Flush streams, synchronize MLX CUDA work, and exit without CRT teardown.

    Avoids ``0xC0000409`` crashes in ``CommandEncoder::~CommandEncoder()``
    during ``DLL_PROCESS_DETACH`` when CUDA unloads before MLX static objects.
    """
    if sys.platform != "win32":
        return

    if "mlx.core" in sys.modules:
        try:
            import mlx.core as mx

            mx.synchronize()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Ignoring mx.synchronize error during runner exit: {exc}")

    logger.complete()
    if sys.stdout is not None:
        sys.stdout.flush()
    if sys.stderr is not None:
        sys.stderr.flush()

    ctypes.windll.kernel32.TerminateProcess(
        ctypes.c_void_p(-1),
        ctypes.c_uint(exit_code),
    )


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event | RunnerTerminationError],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: loguru.Logger,
) -> None:
    global logger
    logger = _logger

    _configure_cuda_environment()
    exit_with_parent()

    if sys.platform != "win32":
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch = use_fast_synch(
        bound_instance.instance, os.environ.get("EXO_FAST_SYNCH")
    )
    os.environ["MLX_METAL_FAST_SYNCH"] = "1" if fast_synch else "0"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    from exo.utils.channels import ClosedResourceError

    exit_code = 0
    try:
        event_sender_downcast: MpSender[RunnerEvent] = cast(
            "MpSender[RunnerEvent]", event_sender
        )

        from exo.worker.runner.runner import Runner

        builder: Builder
        if bound_instance.is_image_model:
            from exo.worker.engines.image.builder import MfluxBuilder

            builder = MfluxBuilder(
                event_sender_downcast, cancel_receiver, bound_instance.bound_shard
            )
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            from exo.worker.engines.mlx.builder import MlxBuilder

            # evil sharing of the event sender
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender_downcast,
                cancel_receiver=cancel_receiver,
            )

        runner = Runner(bound_instance, builder, event_sender_downcast, task_receiver)
        runner.main()
    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        exit_code = 1
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(RunnerTerminationError.from_exception(e))
        raise SystemExit(1) from e
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
            _terminate_win32_runner(exit_code)
