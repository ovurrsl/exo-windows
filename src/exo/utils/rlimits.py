import importlib.util
import sys
import types


def install_win32_resource_shim() -> None:
    """Install a minimal stdlib-only ``resource`` shim on Windows for ``mlx_lm``.

    Must avoid importing third-party libraries (e.g. ``mlx.core``) so importing
    ``exo`` from main/master never loads CUDA DLLs or logs before CLI parsing.
    """
    if sys.platform != "win32" or "resource" in sys.modules:
        return
    if importlib.util.find_spec("resource") is not None:
        return

    shim = types.ModuleType("resource")

    def _getrlimit(_resource_id: int) -> tuple[int, int]:
        return (4096, 4096)

    def _setrlimit(_resource_id: int, _limits: tuple[int, int]) -> None:
        return None

    setattr(shim, "RLIMIT_NOFILE", 0)  # noqa: B010
    setattr(shim, "getrlimit", _getrlimit)  # noqa: B010
    setattr(shim, "setrlimit", _setrlimit)  # noqa: B010
    sys.modules["resource"] = shim


def raise_nofile_limit(min_soft: int) -> None:
    """Raise the RLIMIT_NOFILE soft limit when the platform supports it.

    Windows does not expose POSIX resource limits; the process already inherits
    a large handle table, so this is a no-op there.
    """
    if sys.platform == "win32":
        return

    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, min_soft), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
