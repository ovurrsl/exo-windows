import importlib
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, Self, cast

import psutil

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel


class _NvmlMemoryInfo(Protocol):
    total: int
    free: int


class _NvmlModule(Protocol):
    def nvmlInit(self) -> None: ...  # noqa: N802
    def nvmlShutdown(self) -> None: ...  # noqa: N802
    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...  # noqa: N802
    def nvmlDeviceGetMemoryInfo(self, handle: object) -> _NvmlMemoryInfo: ...  # noqa: N802


def query_cuda_vram_bytes() -> tuple[int, int] | None:
    """Return ``(total_bytes, free_bytes)`` for GPU 0 via NVML or ``nvidia-smi``.

    Tries ``pynvml`` first (matching ``_has_nvml_cuda``) and falls back to
    ``nvidia-smi``. Never imports ``mlx.core`` so the main process does not load
    CUDA DLLs. Returns ``None`` if no CUDA GPU can be queried.
    """
    try:
        nvml = cast(_NvmlModule, cast(object, importlib.import_module("pynvml")))
    except ImportError:
        nvml = None

    if nvml is not None:
        try:
            nvml.nvmlInit()
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                total = int(info.total)
                free = int(info.free)
                if total > 0:
                    return total, max(free, 0)
            finally:
                nvml.nvmlShutdown()
        except Exception:
            pass

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None
    fields = lines[0].split(",")
    if len(fields) != 2:
        return None
    try:
        total_mebibytes = int(fields[0].strip())
        free_mebibytes = int(fields[1].strip())
    except ValueError:
        return None
    if total_mebibytes <= 0:
        return None

    mebibyte = 1024 * 1024
    return total_mebibytes * mebibyte, max(free_mebibytes, 0) * mebibyte


class MemoryUsage(FrozenModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory
    # The most memory a Mac's GPU should use (Metal's recommended working set, about 75% of
    # the RAM). macOS keeps the rest for itself, so a model can't use all of a Mac's free
    # memory. None where it isn't known.
    gpu_working_set: Memory | None = None

    @classmethod
    def from_bytes(
        cls,
        *,
        ram_total: int,
        ram_available: int,
        swap_total: int,
        swap_available: int,
        gpu_working_set: Memory | None = None,
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
            gpu_working_set=gpu_working_set,
        )

    @classmethod
    def from_psutil(
        cls, *, override_memory: int | None, gpu_working_set: Memory | None = None
    ) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
            gpu_working_set=gpu_working_set,
        )

    @classmethod
    def from_cuda(
        cls, *, override_memory: int | None, gpu_working_set: Memory | None = None
    ) -> Self | None:
        """Report a CUDA GPU's VRAM as the node's memory.

        On a discrete NVIDIA GPU the memory that actually bounds MLX inference is
        the GPU's VRAM, not system RAM (unlike Apple Silicon's unified memory).
        Returns None when no GPU/VRAM can be queried so the caller can fall back
        to :meth:`from_psutil`.
        """
        vram = query_cuda_vram_bytes()
        if vram is None:
            return None
        total_vram, free_vram = vram
        sm = psutil.swap_memory()
        return cls.from_bytes(
            ram_total=total_vram,
            ram_available=free_vram if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
            gpu_working_set=gpu_working_set
            if gpu_working_set is not None
            else Memory.from_bytes(total_vram),
        )


class DiskUsage(FrozenModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(FrozenModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(FrozenModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class NodeIdentity(FrozenModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"


class NodeNetworkInfo(FrozenModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(FrozenModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(FrozenModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool


class ThunderboltBridgeStatus(FrozenModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
