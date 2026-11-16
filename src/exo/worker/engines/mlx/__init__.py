import sys

if sys.platform in ("win32", "linux"):
    from exo.utils.rlimits import install_win32_resource_shim
    from exo.worker.runner.bootstrap import (
        _configure_cuda_environment,  # pyright: ignore[reportPrivateUsage]
    )

    _configure_cuda_environment()
    install_win32_resource_shim()
