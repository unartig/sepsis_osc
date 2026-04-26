import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt

    from sepsis_osc.utils.config import ALPHA, BETA_SPACE, SIGMA_SPACE, jax_random_seed, plt_params
    from sepsis_osc.storage.storage_interface import Storage
    from sepsis_osc.ldm.commons import build_lookup_table
    from sepsis_osc.ldm.lookup import LatentLookup, as_2d_indices
    from sepsis_osc.dnm.dynamic_network_model import DNMConfig, DNMMetrics, DNMState, DynamicNetworkModel

    from sepsis_osc.visualisations.viz_param_space import pretty_plot, space_plot


    return (
        ALPHA,
        BETA_SPACE,
        DNMConfig,
        DNMMetrics,
        SIGMA_SPACE,
        Storage,
        as_2d_indices,
        build_lookup_table,
        jax,
        jnp,
        np,
        plt,
    )


@app.cell
def _(
    ALPHA,
    BETA_SPACE,
    DNMConfig,
    DNMMetrics,
    SIGMA_SPACE,
    Storage,
    as_2d_indices,
    build_lookup_table,
    np,
):
    beta_space, sigma_space = np.arange(*BETA_SPACE), np.arange(*SIGMA_SPACE)

    n = round((BETA_SPACE[1] - BETA_SPACE[0]) / BETA_SPACE[2])
    m = round((SIGMA_SPACE[1] - SIGMA_SPACE[0]) / SIGMA_SPACE[2])

    print(n, m)

    beta_2d, sigma_2d = as_2d_indices(BETA_SPACE, SIGMA_SPACE)

    db_str = "DaisyFinal"
    sim_storage = Storage(
        key_dim=9,
        metrics_kv_name=f"data/{db_str}SepsisMetrics.db/",
        parameter_k_name=f"data/{db_str}SepsisParameters_index.bin",
        use_mem_cache=True,
    )
    sim_storage.close()
    lookup_table = build_lookup_table(sim_storage, alpha=ALPHA, beta_space=BETA_SPACE, sigma_space=SIGMA_SPACE)

    a = np.ones_like(beta_2d) * ALPHA
    params = DNMConfig.batch_as_index(a, beta_2d, sigma_2d, 0.2)
    metrics_3d, _ = sim_storage.read_multiple_results(params, proto_metric=DNMMetrics, threshold=0.0)
    metrics_2d = metrics_3d.reshape([1, *metrics_3d.shape["r_1"]]).squeeze()

    s1 = metrics_2d.s_1
    return beta_2d, beta_space, m, n, s1, sigma_2d, sigma_space


@app.cell
def _(beta_2d, jax, jnp, m, n, np, plt, s1, sigma_2d):
    def get_linear(x, y, k):
        xmid = x.min() + (x.max() - x.min()) / 2

        def linear(xs, ys):
            return jax.nn.sigmoid(k * (xs - xmid))

        return linear


    def get_radial(x, y, k):
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        def radial(xs, ys):
            x_symm = (xs - xmin) / (xmax - xmin)
            y_symm = (ys - ymin) / (ymax - ymin)
            r = jnp.sqrt((x_symm - 0.5) ** 2 + (y_symm - 0.5) ** 2)
            return jax.nn.sigmoid(k * (0.2 - r))

        return radial


    def bump(x):
        return x * jnp.exp(-x)

    def softstep(x):
        return 1 / (1 + jnp.exp(-10 * x))


    def get_approx(x, y):
        def approx(xs, ys):
            #return jnp.square(jnp.cos(jnp.exp(xs)) * jnp.sin(((jnp.sin(jnp.sin(jnp.sin(ys) / 0.17486112)) - xs) / xs) * 0.70644474)) + 0.017329557
            #return jnp.tanh(((xs + 0.45009452) ** (22.540365 ** jnp.tanh((ys + 0.40252033) ** 106.4002))) * 0.18739262) * 0.13808896
            #return (xs ** 5.1395) * jnp.tanh(jnp.exp(0.30081 / (-0.5962 + ys)))
            #return (jnp.sqrt((softstep(jnp.sin((ys * (xs / (0.1967987 - (ys / softstep((ys - 0.47228748) / 0.6887915))))) / 0.24941997)) + 0.011010214) * ys)) * 0.15
            #return (jnp.sin(jnp.square(jnp.square(jnp.sin(jnp.square((((softstep(ys / jnp.cos(jnp.cos(ys / -0.29963335))) * (softstep(xs) ** ys)) ** 89.80053) / -0.4105899) + jnp.sqrt(softstep(ys))))))) + (ys * 0.10159114)) * 0.15
            return (jnp.sin(jnp.square(jnp.square(jnp.square(jnp.sin(jnp.square(softstep(jnp.sqrt(ys)) + ((((softstep(xs) ** ys) * softstep(ys * 1.0825667)) ** 75.41125) * -2.4839172)))))) + 0.099605806)) * 0.15
            #return 0.15 * (jnp.sin((jnp.sin((jnp.sin(xs / (0.50825256 - bump(jnp.sin(jnp.sin((ys / 0.60103625) - (xs * 1.1470096))) + 0.4785785))) - 0.10480727)**4))**2 + (ys * 0.11145829)))
            #return 0.04 * ys + 0.11 * (jax.nn.sigmoid((xs - 0.58) / 0.03) * jax.nn.sigmoid((ys - 0.62) / 0.05))

        print(jax.make_jaxpr(approx)(1.0, 1.0))

        return approx


    def eval_grad(p, f):
        x, y = p
        val, grads = jax.value_and_grad(f, argnums=(0, 1))(x, y)
        return val, grads


    def eval_on_grid(points, getter, **kwargs):
        f = getter(points[:, 0], points[:, 1], **kwargs)
        return jax.vmap(eval_grad, in_axes=(0, None))(points, f)


    def show_val(x, y, v, k, ax, cbar=""):
        vmin = v.min() if v.min() > 0 else -np.abs(v).max()
        vmax = np.abs(v).max()
        im = ax.imshow(v.reshape(n, m).T, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", aspect="auto", cmap=cbar or None, vmin=vmin, vmax=vmax)
        if cbar:
            plt.colorbar(im, ax=ax, location="right", shrink=0.8, pad=0.05, cmap=cbar)

        ax.set_title(f"k={k}")
        ax.set_ylabel(f"value")


    def show_grad(x, y, dx, dy, ax, step=10, type="stream"):
        if type == "quiver":
            ax.quiver(
                x[::step, ::step],
                y[::step, ::step],
                dx.reshape(n, m)[::step, ::step],
                dy.reshape(n, m)[::step, ::step],
                color="white",
            )
        if type == "stream":
            mag = np.sqrt(dx**2 + dy**2).reshape(n, m).T
            ax.streamplot(
                x[:, 0],
                y[0, :],
                dx.reshape(n, m).T,
                dy.reshape(n, m).T,
                color=mag,
                linewidth=2 * mag / mag.max(),
                cmap="plasma",
                density=1.0,
            )
        ax.set_ylabel("gradient")


    rbetas, rsigmas, rs1 = beta_2d.ravel(), sigma_2d.ravel(), s1.ravel()
    stacked_points = jnp.stack([rbetas, rsigmas], axis=1)
    return (
        eval_on_grid,
        get_approx,
        get_linear,
        get_radial,
        rbetas,
        rs1,
        rsigmas,
        show_grad,
        show_val,
        stacked_points,
    )


@app.cell
def _(
    beta_2d,
    eval_on_grid,
    get_approx,
    jnp,
    np,
    plt,
    s1,
    show_val,
    sigma_2d,
    stacked_points,
):
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))

    _v, (_dx, _dy) = eval_on_grid(stacked_points, get_approx)
    show_val(beta_2d, sigma_2d, np.clip(_v, 1e-12, jnp.inf), k="", ax=_axs[0], cbar="viridis")
    show_val(beta_2d, sigma_2d, _v - s1.ravel(), k="", ax=_axs[1], cbar="RdBu")
    show_val(beta_2d, sigma_2d, s1.ravel(), k="", ax=_axs[2], cbar="viridis")
    print(np.abs(_v - s1.ravel()).sum())

    _fig
    return


@app.cell
def _(
    beta_2d,
    eval_on_grid,
    get_linear,
    plt,
    show_grad,
    show_val,
    sigma_2d,
    stacked_points,
):
    _fig, _axs = plt.subplots(2, 3)

    for _i, _k in enumerate((100, 25, 0.1)):
        _v, (_dx, _dy) = eval_on_grid(stacked_points, get_linear, k=_k)
        show_val(beta_2d, sigma_2d, _v, k=_k, ax=_axs[0, _i])
        show_grad(beta_2d, sigma_2d, _dx, _dy, _axs[1, _i])

    _fig
    return


@app.cell
def _(
    beta_2d,
    eval_on_grid,
    get_radial,
    plt,
    show_grad,
    show_val,
    sigma_2d,
    stacked_points,
):
    _fig, _axs = plt.subplots(2, 3)

    for _i, _k in enumerate((100, 25, 15)):
        _v, (_dx, _dy) = eval_on_grid(stacked_points, get_radial, k=_k)
        show_val(beta_2d, sigma_2d, _v, k=_k, ax=_axs[0, _i])
        show_grad(beta_2d, sigma_2d, _dx, _dy, _axs[1, _i])

    _fig
    return


@app.cell
def _(
    beta_2d,
    eval_on_grid,
    get_approx,
    jnp,
    np,
    plt,
    show_grad,
    show_val,
    sigma_2d,
    stacked_points,
):
    _fig, _axs = plt.subplots(2, 1)

    _v, (_dx, _dy) = eval_on_grid(stacked_points, get_approx)
    show_val(beta_2d, sigma_2d, np.clip(_v, 1e-12, jnp.inf), k="", ax=_axs[0])
    show_grad(beta_2d, sigma_2d, _dx, _dy, _axs[1])

    _fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    import sympy as sp

    def softstep_sp(x):
        return 1 / (1 + sp.exp(-10 * x))

    x, y = sp.symbols('x y')

    sp_expr = (sp.sin((((sp.sin((softstep_sp(sp.sqrt(y)) + ((((softstep_sp(x) ** y) * softstep_sp(y * 1.0825667)) ** 75.41125) * -2.4839172))**2))**2)**2)**2 + 0.099605806)) * 0.15

    print(sp.simplify(sp_expr))
    return


@app.cell
def _():
    return


@app.cell
def _():
    import optimistix as optx

    return (optx,)


@app.cell
def _(jnp):

    def model(x, y, theta):
        (a, b, c, d, e, f, g) = theta
        return x *  -jnp.exp(-x * b + c)*a + d*jnp.sin(x*e-f) + g
    theta_init = jnp.array([1, 1, 1, 1, 1, 1, 1])
    return model, theta_init


@app.cell
def _(jax, jnp, model, optx):
    model_batch = jax.jit(jax.vmap(model, in_axes=(0, 0, None)))

    @jax.jit
    def loss_fn(theta, data):
        xs, ys, z_target = data
        z_pred = model_batch(xs, ys, theta)
        return jnp.mean((z_pred - z_target) ** 2)

    def fit(theta_init, xs, ys, z_target, max_steps=300, rtol=1e-8, atol=1e-10):
        """
        Run BFGS via optimistix.

        Returns the optimised parameter array and the solver result object.
        """
        data = (xs, ys, z_target)

        # optimistix expects fn(y, args) -> scalar
        def fn(theta, args):
            return loss_fn(theta, args)

        solver = optx.BFGS(rtol=rtol, atol=atol)

        result = optx.minimise(
            fn=fn,
            solver=solver,
            y0=theta_init,
            args=data,
            max_steps=max_steps,
            throw=False,          # return result even if not fully converged
        )

        return result


    return fit, loss_fn


@app.cell
def _(fit, rbetas, rs1, rsigmas, theta_init):
    result = fit(theta_init, rbetas, rsigmas, rs1)
    return (result,)


@app.cell
def _(loss_fn, rbetas, result, rs1, rsigmas):
    theta_opt = result.value
    final_loss = loss_fn(theta_opt, (rbetas, rsigmas, rs1))
    return (theta_opt,)


@app.cell
def _(
    beta_2d,
    beta_space,
    c_,
    eval_on_grid,
    jnp,
    model,
    plt,
    s1,
    show_val,
    sigma_2d,
    sigma_space,
    stacked_points,
    theta_opt,
):
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))

    #Z_fit = model_batch(xs_flat, ys_flat, theta_opt).reshape(ny, nx)

    def get_approximation(x, y, theta):

        def app(xs, ys):
            return model(xs, ys, theta)

        return app

    _v, (_dx, _dy) = eval_on_grid(stacked_points, get_approximation, theta=theta_opt)

    show_val(beta_2d, sigma_2d, s1.ravel(), k="", ax=_axs[0], cbar="viridis")
    _axs[0].plot(jnp.clip(model(sigma_space, beta_space, theta_opt), beta_space.min(), beta_space.max()), sigma_space)
    _axs[1].plot(c_(sigma_space), sigma_space)

    _fig
    return


@app.cell
def _(beta_2d, beta_space, jnp, plt, s1, show_val, sigma_2d, sigma_space):
    _fig, _axs = plt.subplots(1, 2)

    def c_(s):
        # return -(s-0.6)**(1/32) + s * -jnp.exp(-s)/100 + 1.35 + jnp.sin(s - 0.11)/3.9
        return s *  -jnp.exp(-s) + jnp.sin(s)/4 + .5

    show_val(beta_2d, sigma_2d, s1.ravel(), k="", ax=_axs[0], cbar="viridis")
    _axs[0].plot(jnp.clip(c_(sigma_space), beta_space.min(), beta_space.max()), sigma_space)
    _axs[1].plot(c_(sigma_space), sigma_space)

    _fig
    return (c_,)


@app.cell
def _(beta_space):
    beta_space
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
