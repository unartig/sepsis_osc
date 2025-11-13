from beartype import beartype as typechecker


EPS = 1e-7


def setup_jax(*, simulation: bool = True) -> None:
    import os
    import jax

    # jax flags
    if simulation:
      jax.config.update("jax_enable_x64", val=True)  #  MATLAB defaults to double precision
    # jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_disable_jit", True)
    # jax.config.update("jax_debug_nans", val=True)
    # jax.config.update("jax_debug_infs", False)

    # cpu/gpu flags
    xla_flags = (
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_triton_gemm=true "
        "--xla_gpu_enable_cublaslt=true "
        "--xla_gpu_enable_command_buffer='' "
    )
    if simulation:
        xla_flags += (
            "--xla_gpu_exhaustive_tiling_search=true "
            "--xla_gpu_autotune_level=4 "  # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-autotune
        )
    os.environ["XLA_FLAGS"] = xla_flags
