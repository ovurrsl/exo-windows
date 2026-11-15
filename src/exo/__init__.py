import sys
from importlib.metadata import version

if sys.platform == "win32":
    from exo.utils.rlimits import install_win32_resource_shim

    install_win32_resource_shim()

__version__ = version("exo")
