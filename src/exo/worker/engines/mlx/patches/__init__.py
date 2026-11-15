from exo.worker.engines.mlx.patches.cuda_compat import apply_cuda_compat_patches

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    # Must run before importing opt_batch_gen, which imports mlx_lm.generate at top level.
    apply_cuda_compat_patches()

    from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
    from exo.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope

    patch_yarn_rope()
    apply_batch_gen_patch()
