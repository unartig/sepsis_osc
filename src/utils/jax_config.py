def setup_jax():
    import os
    import jax

    #### Configurations

    # jax flags
    jax.config.update("jax_enable_x64", True)  #  MATLAB defaults to double precision
    # jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_debug_infs", False)

    # cpu/gpu flags
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_triton_gemm=true "
        "--xla_gpu_enable_cublaslt=true "
        "--xla_gpu_autotune_level=4 "  # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-autotune
        "--xla_gpu_exhaustive_tiling_search=true "
    )
