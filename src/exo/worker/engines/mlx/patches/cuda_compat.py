import sys

from loguru import logger

from exo.utils.rlimits import install_win32_resource_shim


def apply_cuda_compat_patches() -> None:
    """Apply MLX API compatibility shims for non-macOS backends."""
    if sys.platform == "darwin":
        return

    install_win32_resource_shim()

    try:
        import mlx.core as mx
    except ImportError:
        return

    if not hasattr(mx, "new_thread_local_stream") and hasattr(mx, "new_stream"):
        # mlx_lm.generate invokes mx.new_thread_local_stream at import time,
        # which is absent in non-Metal MLX builds.
        setattr(mx, "new_thread_local_stream", mx.new_stream)  # noqa: B010
        logger.debug("Patched mx.new_thread_local_stream -> mx.new_stream")
