"""Shared tensor- and pipeline-parallel building blocks for mlx-lm models.

Nothing in this package names a model. Model-specific sharding lives in the model's
own folder under ``exo.worker.engines.mlx.models``.
"""

import sys

if sys.platform != "darwin":
    from exo.worker.engines.mlx.patches.cuda_compat import apply_cuda_compat_patches

    apply_cuda_compat_patches()

if sys.platform == "win32":
    from exo.worker.engines.mlx.win32_ring import install_win32_ring_patch

    install_win32_ring_patch()
