import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import pandas as pd
    import scipy
    import scipy.optimize
    import scipy.stats
    from scipy.special import expit
    from scipy.stats import truncnorm, beta as beta_dist
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.regression.mixed_linear_model import MixedLM
    from pyem import EMModel
    from pyem.utils import plotting

    # Defined at module (setup) scope, not nested in a cell function, because
    # joblib/loky (used internally by pyEM's EMModel for parallel per-subject
    # fitting) needs to pickle these across worker processes -- a closure
    # nested inside a cell function that itself calls a sibling closure (as
    # rw1a1b_fit calls norm2alpha/norm2beta) doesn't reliably survive that
    # round trip, while plain top-level functions always do.

    def softmax(v, beta_sm=1.0):
        """Compute softmax values for each v value."""
        return np.exp(v * beta_sm) / np.sum(np.exp(v * beta_sm), axis=0)

    def rw1a1b_sim(params, nblocks: int = 3, ntrials: int = 24, outcomes=None):
        """Simulate a simple Rescorla-Wagner model with one learning rate."""
        np.random.seed(42)
        nsubjects = params.shape[0]
        choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
        rewards = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
        EV = np.zeros((nsubjects, nblocks, ntrials + 1, 2), dtype=float)
        ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
        choices_A = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
        PE = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
        nll_sim = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

        rng = np.random.default_rng()
        this_block_probs = np.array([0.8, 0.2])

        all_beta = params[:, 0]
        all_alpha = params[:, 1]

        if not ((all_beta >= 1e-5) & (all_beta <= 20.0)).all():
            raise ValueError("Beta values out of bounds")
        if not ((all_alpha >= 0.0) & (all_alpha <= 1.0)).all():
            raise ValueError("Alpha values out of bounds")

        for s in range(nsubjects):
            beta_s = float(all_beta[s])
            alpha_s = float(all_alpha[s])
            for b in range(nblocks):
                for t in range(ntrials):
                    p = softmax(EV[s, b, t, :], beta_s)
                    ch_prob[s, b, t, :] = p
                    c = rng.choice([0, 1], p=p)
                    choices[s, b, t] = "A" if c == 0 else "B"
                    choices_A[s, b, t] = 1.0 if c == 0 else 0.0
                    if outcomes is None:
                        rew = rng.choice([1.0, 0.0], p=this_block_probs if c == 0 else this_block_probs[::-1])
                    else:
                        rew = float(outcomes[b, t, c])
                    rewards[s, b, t] = rew
                    PE[s, b, t] = rew - EV[s, b, t, c]
                    EV[s, b, t + 1, :] = EV[s, b, t, :]
                    EV[s, b, t + 1, c] = EV[s, b, t, c] + alpha_s * PE[s, b, t]
                    nll_sim[s, b, t] = -np.log(p[c] + 1e-12)

        return {
            "params": np.array([all_beta, all_alpha]).T,
            "choices": choices,
            "rewards": rewards,
            "EV": EV,
            "ch_prob": ch_prob,
            "choices_A": choices_A,
            "PE": PE,
            "nll": nll_sim,
        }

    def norm2alpha(x):
        return expit(np.asarray(x))

    def norm2beta(x, max_val=20.0):
        return max_val / (1.0 + np.exp(-np.asarray(x)))

    def calc_fval(negll, params, prior=None, output="npl"):
        """Returns objective value given a negative log-likelihood."""
        if output == "npl" and prior is not None and hasattr(prior, "logpdf"):
            fval = -(-negll + prior.logpdf(np.asarray(params)))
            if np.isinf(fval):
                fval = 1e7
            return fval
        return negll

    def rw1a1b_fit(params, choices, rewards, prior=None, output="npl"):
        """Fit a simple Rescorla-Wagner model with one learning rate."""
        beta_f = float(norm2beta(params[0]))
        alpha_f = float(norm2alpha(params[1]))

        if not (1e-5 <= beta_f <= 20.0):
            return 1e7
        if not (0.0 <= alpha_f <= 1.0):
            return 1e7

        fblocks, ftrials = rewards.shape
        EV_f = np.zeros((fblocks, ftrials + 1, 2))
        PE_f = np.zeros((fblocks, ftrials))
        nll_f = 0.0
        for fb in range(fblocks):
            EV_f[fb, 0, :] = 0
            for ft in range(ftrials):
                fc = 0 if choices[fb, ft] == "A" else 1
                fp = np.exp(EV_f[fb, ft, :] * beta_f) / np.sum(np.exp(EV_f[fb, ft, :] * beta_f), axis=0)
                fr = rewards[fb, ft]
                PE_f[fb, ft] = fr - EV_f[fb, ft, fc]
                EV_f[fb, ft + 1, :] = EV_f[fb, ft, :]
                EV_f[fb, ft + 1, fc] = EV_f[fb, ft, fc] + alpha_f * PE_f[fb, ft]
                nll_f += -np.log(fp[fc] + 1e-12)

        if output == "all":
            choices_A_f = (np.asarray(choices) == "A").astype(float)
            return {
                "params": [beta_f, alpha_f],
                "choices": choices,
                "choices_A": choices_A_f,
                "rewards": rewards,
                "EV": EV_f,
                "PE": PE_f,
                "nll": nll_f,
            }

        return calc_fval(nll_f, params, prior=prior, output=output)


@app.cell(hide_code=True)
def title_md():
    mo.md(r"""
    # Fitting models to behavioral data

    By [Shawn A. Rhoads, Ph.D.](https://shawnrhoads.github.io/) -- New York
    Computational Psychiatry Workshop 2025. Adapted from Rhoads' PSYC 347
    course, the [pyEM](https://github.com/shawnrhoads/pyEM) package examples,
    and the Neuromatch Academy tutorials. Marimo conversion keeps every real
    computation live; only the Colab badge and the `ipywidgets` cell (replaced
    with `mo.ui` sliders, since `ipywidgets` doesn't run in marimo) changed.

    **Goals**

    1. Understand Maximum Likelihood Estimation (**MLE**) and algorithms to
       maximize the log-likelihood (or minimize the negative log-likelihood).
    2. Apply **MLE** to fit a linear regression model via grid search and an
       optimization algorithm.
    3. Apply **MLE** to fit a reinforcement learning model via grid search and
       an optimization algorithm.
    4. Understand Maximum A Posteriori (**MAP**) estimation, hierarchical
       modeling, and the expectation-maximization (**EM**) algorithm.
    5. Apply **MAP** to fit a linear regression model using EM.
    6. Apply **MAP** to fit a reinforcement learning model using EM.
    """)
    return


@app.cell(hide_code=True)
def sec_mle_intro_md():
    mo.md(r"""
    ## What is Maximum Likelihood Estimation (MLE)?

    MLE estimates a model's parameters by finding the values that make the
    observed data most probable. In simple terms: *given the data we've
    observed, what parameter values maximize the probability of observing
    exactly this data?* We assume a specific form for the data's probability
    distribution and adjust the model's parameters until the likelihood is
    highest.

    ### Key concepts

    Given observed data $\mathcal{D}$ and parameters $\theta$, MLE maximizes
    the likelihood function:

    $$\mathcal{L}(\theta) = P(\mathcal{D} \mid \theta)$$

    MLE finds $\theta^\ast = \arg\max_\theta \log\mathcal{L}(\theta)$ -- the
    *log*-likelihood, because it's more numerically convenient (turns
    products of probabilities into sums, avoids underflow). Finding
    $\theta^\ast$ is an optimization problem, solvable by grid search or
    iterative optimization algorithms.
    """)
    return


@app.cell(hide_code=True)
def sec_linreg_intro_md():
    mo.md(r"""
    ### Example 1: Linear Regression Model

    A multivariate linear regression model:

    $$Y = X\beta + \epsilon$$

    - $Y$: observed outcomes, shape $(n,1)$
    - $X$: predictor matrix, shape $(n,p)$
    - $\beta$: regression coefficients to estimate, shape $(p,1)$
    - $\epsilon$: error term, typically $\mathcal{N}(0,\sigma^2)$

    Expanded per observation $i$: $y_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip} + \epsilon_i$.
    """)
    return


@app.cell(hide_code=True)
def linreg_simulate():
    np.random.seed(42)
    n_lr = 100
    x1 = np.random.uniform(0, 10, n_lr)
    x2 = np.random.uniform(0, 10, n_lr)
    noise_lr = np.random.normal(0, 1, n_lr)
    b0 = 2.41  # intercept
    b1 = 1.64  # slope 1
    b2 = -3.27  # slope 2
    y = b0 + b1 * x1 + b2 * x2 + noise_lr
    return b0, b1, b2, n_lr, x1, x2, y


@app.cell(hide_code=True)
def linreg_plot_data(x1, x2, y):
    fig01, ax01 = plt.subplots()
    ax01.scatter(x1, y, label="x1")
    ax01.scatter(x2, y, label="x2")
    ax01.set_xlabel("x")
    ax01.set_ylabel("y")
    ax01.legend(loc="lower center")
    sns.despine()
    ax01.set_title("Simulated Data")
    plt.close(fig01)
    fig01
    return


@app.cell(hide_code=True)
def sec_linreg_gaussian_md():
    mo.md(r"""
    ---

    Now we estimate the free parameters `b0`, `b1`, `b2`. To do so we need
    the probability distribution of the errors. A Gaussian is the common
    choice: errors symmetric around zero, most values close to zero.

    $$P(\mathcal{D}\mid\beta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i - X_i\beta)^2}{2\sigma^2}\right)$$

    The normalization term keeps the total area under the curve equal to 1;
    the exponential term shrinks the likelihood of an observation as the gap
    between predicted ($X_i\beta$) and actual ($y_i$) grows.
    """)
    return


@app.cell(hide_code=True)
def normal_pdf_sliders():
    mu_s = mo.ui.slider(-2.0, 2.0, step=0.1, value=0.0, label="μ")
    sigma_s = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label="σ")
    mo.hstack([mu_s, sigma_s])
    return mu_s, sigma_s


@app.cell(hide_code=True)
def normal_pdf_plot(mu_s, sigma_s):
    mu_pdf, sigma_pdf = mu_s.value, sigma_s.value
    x_pdf = np.linspace(-5, 5, 100)
    y_pdf = scipy.stats.norm.pdf(x_pdf, mu_pdf, sigma_pdf)
    samples_pdf = scipy.stats.norm.rvs(mu_pdf, sigma_pdf, size=1000, random_state=0)

    fig02, ax02 = plt.subplots()
    ax02.hist(samples_pdf, 20, density=True, color="g", histtype="stepfilled", alpha=0.8, label="histogram")
    ax02.plot(x_pdf, y_pdf, color="orange", linewidth=3, label="pdf")
    ax02.vlines(mu_pdf, 0, scipy.stats.norm.pdf(mu_pdf, mu_pdf, sigma_pdf), color="red", linewidth=3, label=r"$y_i - X_i\hat{\beta}$")
    ax02.vlines([mu_pdf - sigma_pdf, mu_pdf + sigma_pdf], 0,
                scipy.stats.norm.pdf([mu_pdf - sigma_pdf, mu_pdf + sigma_pdf], mu_pdf, sigma_pdf),
                color="b", linewidth=3, label=r"$\sigma$")
    ax02.set(xlabel=r"$y_i - X_i\hat{\beta}$", ylabel="probability density", xlim=[-5, 5], ylim=[0, 1.0])
    ax02.legend()
    plt.close(fig02)
    fig02
    return


@app.cell(hide_code=True)
def sec_loglik_md():
    mo.md(r"""
    In practice we maximize the *log*-likelihood (numerical stability,
    products become sums, handles large datasets by additivity) -- or,
    equivalently, minimize the *negative* log-likelihood, since most
    optimizers minimize by default.

    > **Note:** linear regression has an analytic solution for $\beta$. We
    > use MLE here anyway to demonstrate an approach that generalizes to
    > models with no closed-form solution.
    """)
    return


@app.cell(hide_code=True)
def negll_lm_func():
    def negll_lm(params, y_data, x1_data, x2_data):
        nb0, nb1, nb2 = params
        y_pred = nb0 + nb1 * x1_data + nb2 * x2_data
        likelihoods = scipy.stats.norm.logpdf(y_data, y_pred)
        return -np.sum(likelihoods)

    return (negll_lm,)


@app.cell(hide_code=True)
def sec_optim_algos_md():
    mo.md(r"""
    ---

    ### Algorithms for Maximizing Likelihood

    - **Grid search** -- evaluate the likelihood over a grid of parameter
      values, pick the best. Simple but expensive for high-dimensional
      parameter spaces.
    - **Gradient-based optimization** (e.g. **BFGS**, `scipy.optimize.minimize`'s
      default) -- iteratively follow the likelihood's gradient to converge on
      the optimum. Far more efficient than grid search.
    """)
    return


@app.cell(hide_code=True)
def linreg_grid_search(negll_lm, x1, x2, y):
    b0_grid = np.linspace(-10, 10, 50)
    b1_grid = np.linspace(-10, 10, 50)
    b2_grid = np.linspace(-10, 10, 50)
    nll_lr = np.zeros((50, 50, 50))
    for gi in range(50):
        for gj in range(50):
            for gk in range(50):
                nll_lr[gi, gj, gk] = negll_lm([b0_grid[gi], b1_grid[gj], b2_grid[gk]], y, x1, x2)

    gi_opt, gj_opt, gk_opt = np.unravel_index(nll_lr.argmin(), nll_lr.shape)
    b0_opt = b0_grid[gi_opt]
    b1_opt = b1_grid[gj_opt]
    b2_opt = b2_grid[gk_opt]
    return b0_grid, b0_opt, b1_grid, b1_opt, b2_grid, b2_opt, gi_opt, gj_opt, gk_opt, nll_lr


@app.cell(hide_code=True)
def linreg_grid_heatmap_plot(b1_grid, b2_grid, gi_opt, gj_opt, gk_opt, nll_lr):
    fig03, ax03 = plt.subplots()
    sns.heatmap(nll_lr[gi_opt, :, :], ax=ax03, cmap="Spectral")
    ax03.set_xlabel(r"$b_2$")
    ax03.set_xticks(np.linspace(0, 50, 6))
    ax03.set_xticklabels(np.linspace(-10, 10, 6))
    ax03.set_ylabel(r"$b_1$")
    ax03.set_yticks(np.linspace(0, 50, 6))
    ax03.set_yticklabels(np.linspace(-10, 10, 6))
    ax03.invert_yaxis()
    ax03.scatter(gk_opt, gj_opt, color="black", marker="x", s=100)
    ax03.collections[0].colorbar.set_label("Negative Log-Likelihood", rotation=270, labelpad=20)
    plt.close(fig03)
    fig03
    return


@app.cell(hide_code=True)
def linreg_grid_3d_plot(b1_grid, b1_opt, b2_grid, b2_opt, gi_opt, gj_opt, gk_opt, nll_lr):
    fig04 = plt.figure()
    ax04 = fig04.add_subplot(111, projection="3d")
    X_mesh, Y_mesh = np.meshgrid(b1_grid, b2_grid)
    ax04.plot_surface(X_mesh, Y_mesh, nll_lr[gi_opt, :, :], cmap="Spectral", alpha=0.75)
    ax04.set_xlabel(r"$b_2$")
    ax04.set_ylabel(r"$b_1$")
    ax04.set_zlabel("Negative Log-Likelihood")
    ax04.scatter(b2_opt, b1_opt, nll_lr[gi_opt, gj_opt, gk_opt], color="black", marker="x", s=100)
    plt.tight_layout()
    plt.close(fig04)
    fig04
    return


@app.cell(hide_code=True)
def linreg_grid_print(b0, b0_opt, b1, b1_opt, b2, b2_opt):
    mo.md(
        f"""
        | Parameter | Actual | Estimated (grid search) |
        |---|---|---|
        | b0 | {b0:.2f} | {b0_opt:.2f} |
        | b1 | {b1:.2f} | {b1_opt:.2f} |
        | b2 | {b2:.2f} | {b2_opt:.2f} |
        """
    )
    return


@app.cell(hide_code=True)
def linreg_scipy_optimize(negll_lm, x1, x2, y):
    linreg_result = scipy.optimize.minimize(negll_lm, [0, 0, 0], args=(y, x1, x2))
    b0_hat, b1_hat, b2_hat = linreg_result.x
    return b0_hat, b1_hat, b2_hat


@app.cell(hide_code=True)
def linreg_scipy_print(b0, b0_hat, b1, b1_hat, b2, b2_hat):
    mo.md(
        f"""
        | Parameter | Actual | Estimated (scipy.optimize) |
        |---|---|---|
        | b0 | {b0:.2f} | {b0_hat:.2f} |
        | b1 | {b1:.2f} | {b1_hat:.2f} |
        | b2 | {b2:.2f} | {b2_hat:.2f} |
        """
    )
    return


@app.cell(hide_code=True)
def linreg_fit_plot(b0_hat, b1_hat, b2_hat, x1, x2, y):
    fig05, ax05 = plt.subplots()
    y_hat_lr = b0_hat + b1_hat * np.mean(x1) + b2_hat * x2
    ax05.plot(x2, y_hat_lr, label="Predicted Data", color="black")
    ax05.scatter(x2, y, label="True Simulated Data", alpha=0.5, s=50)
    ax05.set_xlabel("x2")
    ax05.set_ylabel("y")
    ax05.legend(loc="lower center")
    sns.despine()
    plt.close(fig05)
    fig05
    return


@app.cell(hide_code=True)
def sec_linreg_summary_md():
    mo.md(r"""
    Both methods recovered parameters close to, but not exactly matching,
    the true values -- expected, given noise in the data (models only ever
    approximate the true data-generating process). `scipy.optimize.minimize`
    (BFGS) generally outperformed grid search: it follows gradient
    information instead of exhaustively evaluating a fixed grid.

    ---

    ### Example 2: Temporal Difference (TD) Learning Model

    TD learning estimates state/action values to predict long-term reward,
    updating iteratively from observed experience:

    $$Q_{t+1}(S_t,A_t) = Q_t(S_t,A_t) + \alpha\big[R_{t+1} + \gamma\max_a Q_t(S_{t+1},a) - Q_t(S_t,A_t)\big]$$

    - $Q_t(S_t,A_t)$: value of action $A_t$ in state $S_t$ at time $t$
    - $\alpha \in (0,1)$: learning rate
    - $R_{t+1}$: reward received
    - $\gamma$: discount factor
    - $\max_a Q_t(S_{t+1},a)$: best estimated value of the next state

    Action selection uses **softmax**: $P(a) = \dfrac{e^{\beta Q(S_t,a)}}{\sum_{a'} e^{\beta Q(S_t,a')}}$,
    with inverse-temperature $\beta$ controlling choice randomness (high
    $\beta$ = deterministic, low $\beta$ = exploratory). Two free parameters
    to estimate: $\alpha$ (learning rate) and $\beta$ (inverse temperature).

    #### Two-Armed Bandit Task

    Choose between two options, each rewarding with some fixed probability
    (e.g. 80% vs 20%); learn which is better through exploration and
    exploitation. With no state transitions, the TD update drops the
    discounted next-state term:

    $$Q_{t+1} = Q_t + \alpha\big[R_t - Q_t\big]$$
    """)
    return


@app.cell(hide_code=True)
def rw_sim_params():
    nsubjects, nblocks, ntrials = 80, 1, 100
    betamin, betamax = 0.75, 10  # inverse temperature
    alphamin, alphamax = 0.05, 0.95  # learning rate

    np.random.seed(42)
    beta_rv = truncnorm((betamin - 0) / 1, (betamax - 0) / 1, loc=0.2, scale=2.2).rvs(nsubjects)
    a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], 1.1, 1.1)
    alpha_rv = beta_dist.ppf(a_lo + np.random.rand(nsubjects) * (a_hi - a_lo), 1.1, 1.1)
    rl_params = np.column_stack((beta_rv, alpha_rv))
    rl_param_names = ["beta", "alpha"]
    return alpha_rv, beta_rv, nblocks, nsubjects, ntrials, rl_param_names, rl_params


@app.cell(hide_code=True)
def rw_sim_run(nblocks, ntrials, rl_params):
    sim_output = rw1a1b_sim(rl_params, nblocks=nblocks, ntrials=ntrials)
    return (sim_output,)


@app.cell(hide_code=True)
def rw_sim_plot_choices(ntrials, sim_output):
    fig06, ax06 = plt.subplots()
    ax06.scatter(range(ntrials), sim_output["rewards"][0, 0, :], label="Reward", color="blue", alpha=0.3)
    ax06.set_xlabel("Trials")
    ax06.set_ylabel("Rewards (Blue)", color="blue")
    ax06.set_ylim(-0.1, 1.1)

    ax06b = ax06.twinx()
    ax06b.plot(range(ntrials), [1 if v == 1 else 0 for v in sim_output["choices_A"][0, 0, :]],
               label="Choice", color="red", linestyle="--")
    ax06b.set_ylim(-0.1, 1.1)
    ax06b.set_yticks([0, 1])
    ax06b.set_yticklabels(["Left", "Right"])
    ax06b.set_ylabel("Choices (Red)", color="red", rotation=270, labelpad=15)

    ax06.set_title("Choices and rewards over trials")
    plt.close(fig06)
    fig06
    return


@app.cell(hide_code=True)
def rw_sim_plot_qvalues(ntrials, sim_output):
    fig07, ax07 = plt.subplots()
    ax07.plot(range(ntrials), sim_output["EV"][0, 0, 1:, 0], label=r"$Q_{1}$", color="blue")
    ax07.plot(range(ntrials), sim_output["EV"][0, 0, 1:, 1], label=r"$Q_{2}$", color="red", linestyle="--")
    ax07.set_xlabel("Trials")
    ax07.set_ylabel("Q-Value")
    ax07.legend()
    sns.despine()
    plt.close(fig07)
    fig07
    return


@app.cell(hide_code=True)
def sec_mle_rl_md():
    mo.md(r"""
    ---

    ### Estimating Parameters with MLE

    Now estimate $\alpha$ and $\beta$ that best explain the observed choices
    and rewards, again via grid search and `scipy.optimize.minimize`:

    $$-\log\mathcal{L}(\beta,\alpha,\gamma) = -\sum_{t=1}^T \log P(A_t\mid S_t,\beta,\alpha)$$
    """)
    return


@app.cell(hide_code=True)
def rw_grid_search(sim_output):
    beta_grid_norm = np.linspace(-5, -1, 50)
    beta_grid = norm2beta(beta_grid_norm)
    alpha_grid_norm = np.linspace(-2, 3, 50)
    alpha_grid = norm2alpha(alpha_grid_norm)
    nll_rw = np.zeros((50, 50))

    for ri in range(50):
        for rj in range(50):
            nll_rw[ri, rj] = rw1a1b_fit(
                [beta_grid_norm[ri], alpha_grid_norm[rj]],
                sim_output["choices"][0, :, :], sim_output["rewards"][0, :, :],
                prior=None, output="nll",
            )

    ri_opt, rj_opt = np.unravel_index(nll_rw.argmin(), nll_rw.shape)
    beta_opt = beta_grid[ri_opt]
    alpha_opt = alpha_grid[rj_opt]
    return alpha_grid, alpha_opt, beta_grid, nll_rw, ri_opt, rj_opt


@app.cell(hide_code=True)
def rw_grid_heatmap(alpha_grid, beta_grid, nll_rw, ri_opt, rj_opt):
    fig08, ax08 = plt.subplots()
    sns.heatmap(nll_rw, ax=ax08, cmap="Spectral")
    ax08.set_ylabel(r"$\beta$")
    ax08.set_xlabel(r"$\alpha$")

    num_ticks = 6
    beta_tick_idx = np.linspace(0, len(beta_grid) - 1, num_ticks).astype(int)
    alpha_tick_idx = np.linspace(0, len(alpha_grid) - 1, num_ticks).astype(int)
    ax08.set_yticks(beta_tick_idx + 0.5)
    ax08.set_xticks(alpha_tick_idx + 0.5)
    ax08.set_yticklabels([f"{beta_grid[i]:.2f}" for i in beta_tick_idx])
    ax08.set_xticklabels([f"{alpha_grid[j]:.2f}" for j in alpha_tick_idx])
    ax08.invert_yaxis()
    ax08.scatter(rj_opt, ri_opt, color="black", marker="x", s=100)
    ax08.collections[0].colorbar.set_label("Negative Log-Likelihood", rotation=270, labelpad=20)
    plt.close(fig08)
    fig08
    return


@app.cell(hide_code=True)
def rw_grid_print(alpha_opt, beta_opt, sim_output):
    mo.md(
        f"""
        | Parameter | Actual | Estimated (grid search) |
        |---|---|---|
        | beta | {sim_output['params'][0, 0]:.2f} | {beta_opt:.2f} |
        | alpha | {sim_output['params'][0, 1]:.2f} | {alpha_opt:.2f} |
        """
    )
    return


@app.cell(hide_code=True)
def rw_scipy_optimize(sim_output):
    rw_result = scipy.optimize.minimize(
        rw1a1b_fit, [0.2, 1],
        args=(sim_output["choices"][0, :, :], sim_output["rewards"][0, :, :], None, "nll"),
    )
    beta_hat, alpha_hat = norm2beta(rw_result.x[0]), norm2alpha(rw_result.x[1])
    return alpha_hat, beta_hat


@app.cell(hide_code=True)
def rw_scipy_print(alpha_hat, beta_hat, sim_output):
    mo.md(
        f"""
        | Parameter | Actual | Estimated (scipy.optimize) |
        |---|---|---|
        | beta | {sim_output['params'][0, 0]:.2f} | {beta_hat:.2f} |
        | alpha | {sim_output['params'][0, 1]:.2f} | {alpha_hat:.2f} |
        """
    )
    return


@app.cell(hide_code=True)
def rw_params_dist_plot(rl_param_names, rl_params):
    fig09, ax09 = plt.subplots(1, 2, figsize=(8, 4))
    for pi in range(len(rl_param_names)):
        sns.histplot(rl_params[:, pi], kde=True, ax=ax09[pi])
        ax09[pi].set_title(rl_param_names[pi])
    plt.tight_layout()
    plt.close(fig09)
    fig09
    return


@app.cell(hide_code=True)
def rw_avg_q_plot(nsubjects, ntrials, sim_output):
    q_data = [[qx[:, :-1, 0].ravel(), qx[:, :-1, 1].ravel()] for qx in sim_output["EV"]]
    rew_df = pd.DataFrame()
    for qs in range(nsubjects):
        rew_df = pd.concat([rew_df, pd.DataFrame({
            "Subject": np.ones(ntrials) * qs, "Trial": np.arange(ntrials),
            "Q": q_data[qs][0], "Arm": np.ones(ntrials) * 0,
        })])
        rew_df = pd.concat([rew_df, pd.DataFrame({
            "Subject": np.ones(ntrials) * qs, "Trial": np.arange(ntrials),
            "Q": q_data[qs][1], "Arm": np.ones(ntrials),
        })])

    fig10, ax10 = plt.subplots()
    sns.lineplot(data=rew_df, x="Trial", y="Q", hue="Arm", dashes=False, alpha=0.6, ax=ax10)
    ax10.set_ylim(-0.1, 1.1)
    plt.close(fig10)
    fig10
    return


@app.cell(hide_code=True)
def rw_fit_all_subjects(nsubjects, rl_params, sim_output):
    rl_est_params = np.zeros((nsubjects, 2))
    for fs in range(nsubjects):
        fs_guess = np.random.normal(-1, 1, 2)
        fs_result = scipy.optimize.minimize(
            rw1a1b_fit, fs_guess,
            args=(sim_output["choices"][fs, :, :], sim_output["rewards"][fs, :, :], None, "nll"),
        )
        rl_est_params[fs, :] = norm2beta(fs_result.x[0]), norm2alpha(fs_result.x[1])
    return (rl_est_params,)


@app.cell(hide_code=True)
def rw_recovery_plot(rl_est_params, rl_param_names, rl_params):
    fig11, ax11 = plt.subplots(1, 2, figsize=(8, 4))
    for pj in range(2):
        ax11[pj].scatter(rl_params[:, pj], rl_est_params[:, pj], color="black", alpha=0.3, s=50)
        ax11[pj].set_xlabel("Simulated")
        ax11[pj].set_ylabel("Estimated")
        ax11[pj].set_title(rl_param_names[pj])

        if pj == 1:
            ax11[pj].plot([0, 1], [0, 1], "k--")
        else:
            ax11[pj].plot([0, 6], [0, 6], "k--")

        r_pj, _ = scipy.stats.pearsonr(rl_params[:, pj], rl_est_params[:, pj])
        ax11[pj].annotate(rf"$r={r_pj:.2f}$", xy=(0.01, 1.01), xycoords="axes fraction")

    sns.despine()
    plt.tight_layout()
    plt.close(fig11)
    fig11
    return


@app.cell(hide_code=True)
def sec_mle_limitations_md():
    mo.md(r"""
    Both grid search and `scipy.optimize.minimize` failed to perfectly
    recover the true parameters, and `scipy.optimize.minimize` again
    outperformed grid search. MLE alone struggles to capture individual
    agent variability -- next: hierarchical modeling.

    ---

    ## Hierarchical Modeling, MAP Estimation, and the EM Algorithm

    Hierarchical modeling estimates parameters at multiple levels at once --
    individual-level parameters (e.g. learning rate, inverse temperature)
    and group-level "hyper" parameters (e.g. their mean, variance) --
    capturing individual differences while pooling information across
    subjects.

    ### What is Maximum A Posteriori (MAP) Estimation?

    MAP combines a prior belief about parameters, $P(\theta)$, with the data
    likelihood to estimate the parameters that maximize the *posterior*:

    $$\theta^\ast = \arg\max_\theta P(\theta\mid\mathcal{D}) = \arg\max_\theta P(\mathcal{D}\mid\theta)\cdot P(\theta)$$
    """)
    return


@app.cell(hide_code=True)
def lm_simulate_hierarchical_funcs():
    def simulate_lm(params, trials):
        """Simulate Y ~ b0 + b1*X1 + b2*X2 + ... with params.shape[1] features."""
        hsubjects, hparams = params.shape
        Y_out = np.zeros((hsubjects, trials))
        X_out = np.zeros((hsubjects, trials, hparams))
        for hsubj in range(hsubjects):
            np.random.seed(2021)
            X_out[hsubj, :, :] = np.concatenate(
                (np.ones((trials, 1)), np.random.normal(size=(trials, hparams - 1))), axis=1
            )
            Y_out[hsubj, :] = np.dot(X_out[hsubj, :], params[hsubj, :]) + np.random.normal(size=(trials,))
        return X_out, Y_out

    return (simulate_lm,)


@app.cell(hide_code=True)
def lm_simulate_hierarchical_run(simulate_lm):
    np.random.seed(42)
    param_names = ["b0", "b1", "b2"]
    nparams = len(param_names)
    hnsubjects = 100
    hntrials = 100

    hparams_arr = np.ones((hnsubjects, nparams))
    for hsimS in range(hnsubjects):
        for hparP in range(nparams):
            hparams_arr[hsimS, hparP] = scipy.stats.norm.rvs(0, scale=1)

    X_out, Y_out = simulate_lm(hparams_arr, hntrials)
    all_data = [[ay, ax] for ay, ax in zip(Y_out, X_out)]
    return X_out, Y_out, all_data, hnsubjects, hntrials, hparams_arr, nparams, param_names


@app.cell(hide_code=True)
def lm_hierarchical_subject_plot(X_out, Y_out, hparams_arr):
    subject_test = 21
    fig12, ax12 = plt.subplots()
    ax12.scatter(np.dot(X_out[subject_test, :, :], hparams_arr[subject_test, :]),
                 Y_out[subject_test], alpha=0.5, s=50)
    ax12.set_ylabel("Predicted (Y)")
    ax12.set_xlabel("Simulated (Y)")
    ax12.set_title(f"Subject {subject_test}; b1={hparams_arr[subject_test, 1]:.2f}")
    sns.despine()
    plt.close(fig12)
    fig12
    return (subject_test,)


@app.cell(hide_code=True)
def map_lm_func():
    def map_lm(params, y_data, X_data, prior=None, output="npl"):
        y_pred = np.dot(X_data, params)
        likelihoods = scipy.stats.norm.logpdf(y_data, y_pred)
        negll = -np.sum(likelihoods)

        if output == "npl" and prior is not None:
            return -(-negll + prior.logpdf(np.asarray(params)))
        if output == "nll":
            return negll
        if output == "all":
            return {
                "params": params,
                "y_pred": y_pred,
                "negll": negll,
                "BIC": len(params) * np.log(len(y_data)) + 2 * negll,
            }
        return None

    return (map_lm,)


@app.cell(hide_code=True)
def sec_em_intro_md():
    mo.md(r"""
    ---

    ### What is Expectation-Maximization (EM)?

    EM iteratively estimates model parameters in two steps:

    1. **Expectation (E) step** -- estimate the expected value of latent
       variables given the observed data and current parameter estimates.
    2. **Maximization (M) step** -- maximize the likelihood with respect to
       the parameters, using those expected values.

    Repeat until convergence -- this finds parameters maximizing the
    likelihood of the observed data even with unobserved components.

    ### Hierarchical Modeling with EM

    EM estimates both individual- and group-level parameters: E-step infers
    individual-level parameters given the group-level parameters as a prior;
    M-step updates the group-level parameters from those individual
    estimates (used as the next E-step's prior). Repeat until convergence.

    ```
    1. Initialize the group-level parameters
    2. Repeat until convergence:
       a. E-step: estimate individual-level parameters given
          group-level parameters as prior
       b. M-step: update group-level parameters from individual-level
          estimates (used as new prior)
    3. Return the individual and group-level estimates
    ```

    We use the [pyEM](https://github.com/shawnrhoads/pyEM) package (already
    a project dependency here) to run this.
    """)
    return


@app.cell(hide_code=True)
def lm_em_fit(all_data, hparams_arr, map_lm, nparams):
    lm_em_model = EMModel(all_data=all_data, fit_func=map_lm, param_names=[f"b{pi}" for pi in range(nparams)])
    lm_fit_result = lm_em_model.fit(verbose=0)

    lm_em_figs = []
    for param_idx in range(nparams):
        simulated_param = hparams_arr[:, param_idx]
        estimated_param = lm_em_model.outfit["params"][:, param_idx]
        lm_em_figs.append(
            plotting.plot_scatter(simulated_param, f"Simulated b_{param_idx}", estimated_param, f"Estimated b_{param_idx}")
        )
    return lm_em_model, lm_fit_result


@app.cell(hide_code=True)
def sec_overfitting_check_md():
    mo.md(r"""
    This is close to perfect recovery of the true parameters -- hierarchical
    modeling and EM capturing individual differences while improving
    parameter estimation.

    Are we overfitting? MAP estimation mitigates this: the prior acts as a
    regularizer, preventing the model from fitting noise and keeping
    estimates plausible. A quick check: compare predicted vs. actual $y$ for
    a subject -- an overfit model would show either extreme, near-perfect
    matches (fitting noise) or an overly tight central-trend fit
    (under-regularized).
    """)
    return


@app.cell(hide_code=True)
def lm_em_overfit_check_plot(X_out, Y_out, hparams_arr, subject_test):
    fig13, ax13 = plt.subplots()
    ax13.scatter(np.dot(X_out[subject_test, :, :], hparams_arr[subject_test, :]),
                 Y_out[subject_test], alpha=0.3, s=50)
    ax13.set_ylabel("Predicted Y")
    ax13.set_xlabel("Simulated Y")
    ax13.set_title(f"Subject {subject_test}; b1={hparams_arr[subject_test, 1]:.2f}")
    sns.despine()
    plt.close(fig13)
    fig13
    return


@app.cell(hide_code=True)
def sec_overfitting_explanation_md():
    mo.md(r"""
    A strong correspondence without excessive variance supports good
    generalization -- the hierarchical structure and priors prevent both
    overfitting extremes, balancing flexibility against regularization for
    robust parameter recovery.

    ---

    This algorithm is a **mixed-effects linear model** in disguise. The
    individual subject parameters (`b0`, `b1`, `b2`) are akin to **random
    effects**; the group-level means (`posterior_mu` from pyEM) are akin to
    **fixed effects**. EM combines both levels of estimation for robust
    parameter estimates that capture individual differences *and*
    group-level trends. Let's confirm this with `statsmodels`.
    """)
    return


@app.cell(hide_code=True)
def statsmodels_mixedlm_fit(X_out, Y_out, hnsubjects, hntrials):
    mlm_rows = []
    for mlm_subj in range(hnsubjects):
        mlm_X = X_out[mlm_subj, :, :]
        mlm_Y = Y_out[mlm_subj, :]
        mlm_rows.append(pd.DataFrame(
            np.hstack((np.ones((hntrials, 1)) * mlm_subj, mlm_X, mlm_Y[:, np.newaxis])),
            columns=["subject", "x0", "x1", "x2", "Y"],
        ))
    mlm_df = pd.concat(mlm_rows, ignore_index=True)

    mlm_model = MixedLM.from_formula("Y ~ 1 + x1 + x2", mlm_df, groups="subject")
    mlm_result = mlm_model.fit()
    mo.md(f"```\n{mlm_result.summary()}\n```")
    return (mlm_result,)


@app.cell(hide_code=True)
def compare_fixed_vs_posterior_print(lm_fit_result, mlm_result):
    mo.md(
        f"""
        | Parameter | statsmodels fixed effect | pyEM posterior mean |
        |---|---|---|
        | b0 | {mlm_result.params['Intercept']:.3f} | {lm_fit_result.posterior_mu[0]:.3f} |
        | b1 | {mlm_result.params['x1']:.3f} | {lm_fit_result.posterior_mu[1]:.3f} |
        | b2 | {mlm_result.params['x2']:.3f} | {lm_fit_result.posterior_mu[2]:.3f} |
        """
    )
    return


@app.cell(hide_code=True)
def sec_rl_hierarchical_md():
    mo.md(r"""
    ---

    Now apply MAP estimation and EM to a reinforcement learning model:
    estimate individual- and group-level parameters together, using EM to
    iterate between them.
    """)
    return


@app.cell(hide_code=True)
def rw_em_recovery(nblocks, ntrials, rl_params):
    rw_em_model = EMModel(
        all_data=None, fit_func=rw1a1b_fit,
        param_names=["beta", "alpha"], param_xform=[norm2beta, norm2alpha],
        simulate_func=rw1a1b_sim,
    )
    rw_recovery = rw_em_model.recover(rl_params, pr_inputs=["choices", "rewards"], nblocks=nblocks, ntrials=ntrials)
    rw_recovery_fig = rw_em_model.plot_recovery(rw_recovery)
    return


@app.cell(hide_code=True)
def sec_summary_md():
    mo.md(r"""
    ---

    ### Summary

    We covered (1) Maximum Likelihood Estimation and (2) hierarchical
    Maximum A Posteriori estimation with the Expectation-Maximization
    algorithm, fitting a linear model and reinforcement learning models to
    behavioral data -- MLE for parameter estimation, hierarchical modeling
    for capturing individual differences and improving estimation.

    *"All models are wrong, but some are useful." -- George Box*

    ### Model Comparison

    Once you've fit a model, compare it against alternatives:

    - **AIC** (Akaike Information Criterion) -- balances fit and complexity;
      lower is better.
    - **BIC** (Bayesian Information Criterion) -- like AIC, penalizes
      complexity more strongly, favoring simpler models.
    - **LME** (Log Model Evidence) -- Bayesian log-evidence for direct model
      comparison.
    - **WAIC** (Widely Applicable Information Criterion) -- Bayesian
      estimate of out-of-sample predictive accuracy.
    - **Cross-validation** -- assess generalization by splitting data into
      training/test sets.

    ### Additional Resources

    - Wilson, R. C., & Collins, A. G. (2019). Ten simple rules for the
      computational modeling of behavioral data. *eLife*, 8, e49547.
    - Daw, N. D. (2011). Trial-by-trial data analysis using computational
      models.
    - Rhoads, S. A. (2023). pyEM: Expectation Maximization with MAP
      estimation in Python. https://github.com/shawnrhoads/pyEM
    - Rhoads, S. A. & Gan, L. (2022). Computational models of human social
      behavior and neuroscience. *JOSE*, 5(47), 146.
    """)
    return


if __name__ == "__main__":
    app.run()
