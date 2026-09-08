import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", sql_output="polars")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    return math, mo, np, plt


@app.cell(hide_code=True)
def title_md(mo):
    mo.md(r"""
    # A helper notebook for the Poisson Cognitive Hierarchy model

    A companion to Camerer, Ho & Chong (2004), *A Cognitive Hierarchy Model of Games*
    (*Quarterly Journal of Economics*, 119(3), 861-898) alongside Yoshida, Dolan & Friston (2008),
    *Game Theory of Mind*, which arrives at a structurally similar idea from
    neuroscience rather than economics.

    The paper's opening move: players in a game are "in equilibrium" only if they are
    rational *and* correctly predict everyone else's strategy.
    Empirically, people are usually rational but not mutually consistent -- beliefs about what others will
    do are often wrong, in a patterned way.
    The Poisson-CH model replaces "equilibrium" with an explicit, inductively defined hierarchy of bounded reasoning depths.
    This notebook builds the model directly so you can manipulate it, rather than just read
    about it.
    """)
    return


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## 1 &middot; The model

    **Step 0** players don't reason about opponents at all -- they choose uniformly at
    random over their $m$ available strategies. Camerer et al. are explicit that this
    is "a placeholder assumption... which could easily be relaxed": whatever level-0 is
    *assumed* to do silently determines the predictions of every level above it, which
    is the single most common way these models get mis-specified in later work.

    **Step $k$** players ($k \ge 1$) best-respond to a belief about the population:
    they assume opponents are distributed over steps $0, \dots, k-1$ -- never $k$ or
    higher. This isn't modesty so much as a specific, falsifiable claim about limited
    working memory: a step-$k$ player must simulate every step below it to compute its
    own best response, so the computational burden grows with $k$ regardless of whether
    the assumption about who's "below" is correct.

    The **true** population frequency of step-$k$ thinkers is assumed Poisson,
    $f(k) = e^{-\tau}\tau^k / k!$ -- one free parameter $\tau$, which is simultaneously
    its mean and its variance. A step-$k$ thinker's *belief* about the relative
    frequency of step-$h$ opponents ($h < k$) is the same Poisson shape, truncated at
    $k$ and renormalized:

    $$g_k(h) = \frac{f(h)}{\sum_{l=0}^{k-1} f(l)}, \qquad h < k.$$

    This has a property the paper calls **increasingly rational expectations**: as $k$
    grows, the truncated $g_k$ converges to the true $f$, so higher-step players are
    *less* wrong about the population than lower-step players are -- without ever
    becoming exactly right, since $g_k(k)$ is always forced to zero (nobody thinks
    anyone else is exactly as sophisticated as they are).

    Fit to 24 *p*-beauty-contest datasets, the median $\hat\tau$ is **1.61**; across
    roughly 100 games of very different structure, $\tau \approx 1.5$ gives reliable
    predictions. That's the default this notebook uses.
    """)
    return


@app.cell(hide_code=True)
def core_funcs(math, np):
    def poisson_pmf(tau: float, k_max: int) -> np.ndarray:
        """f(k) for k = 0..k_max: the population's true frequency of step-k thinkers.

        One free parameter tau (both the mean and the variance of the distribution).
        """
        return np.array([math.exp(-tau) * tau**k / math.factorial(k) for k in range(k_max + 1)])


    def truncated_belief(f: np.ndarray, k: int) -> np.ndarray:
        """g_k(h): a step-k thinker's belief about the proportion of step-h opponents.

        Truncated to h < k and renormalized -- step-k players don\'t imagine anyone
        thinking as much as, or more than, they do (g_k(k) = 0 by construction). As k
        grows this converges to the true f: "increasingly rational expectations."
        """
        g = np.zeros_like(f)
        if k > 0:
            g[:k] = f[:k] / f[:k].sum()
        return g

    return poisson_pmf, truncated_belief


@app.cell(hide_code=True)
def bc_md(mo):
    mo.md(r"""
    ## 2 &middot; Worked example: the *p*-beauty contest

    Keynes's newspaper contest, formalized: everyone picks a number in $[0, 100]$;
    whoever picks closest to $p$ times the *group average* wins. For $p < 1$ the
    unique Nash equilibrium is 0 -- reached only by iterating "the others will reason
    about this too" infinitely many times. People don't: one-shot $p = 2/3$ experiments
    typically average 20-35, not 0.

    Because the payoff only rewards proximity to $p \times$ (others' average guess), a
    step-$k$ player's best response collapses to a single number rather than a full
    mixed strategy -- exactly the arithmetic Camerer et al. use for their own example:
    *"Step 2 players... assume the other players are a combination of step 0 players
    (average guess 50) and step 1 players (who guess &#8532; of 50)."*

    $$s_0 = 50, \qquad s_k = p \sum_{h=0}^{k-1} g_k(h)\, s_h \;\; (k \ge 1).$$
    """)
    return


@app.cell(hide_code=True)
def bc_solver(np, poisson_pmf, truncated_belief):
    def solve_beauty_contest(
        p: float, tau: float, k_max: int = 15, lo: float = 0.0, hi: float = 100.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recursively solve step-k guesses s_k for the p-beauty contest.

        s_0 is the mean of level-0's uniform guess over [lo, hi]. Each s_k folds in the
        belief-weighted mix of every guess below it -- solved bottom-up, one level at a
        time, never as a fixed point.
        """
        f = poisson_pmf(tau, k_max)
        s = np.zeros(k_max + 1)
        s[0] = (lo + hi) / 2
        for k in range(1, k_max + 1):
            g = truncated_belief(f, k)
            s[k] = p * np.dot(g[:k], s[:k])
        return f, s

    return (solve_beauty_contest,)


@app.cell(hide_code=True)
def bc_sliders(mo):
    p_slider = mo.ui.slider(0.3, 1.5, step=0.01, value=2 / 3, label="p")
    tau_slider = mo.ui.slider(0.0, 5.0, step=0.05, value=1.61, label="\u03c4")
    mo.hstack([p_slider, tau_slider], justify="start", gap=2)
    return p_slider, tau_slider


@app.cell(hide_code=True)
def bc_summary(mo, np, p_slider, solve_beauty_contest, tau_slider):
    p_val, tau_val = p_slider.value, tau_slider.value
    f_dist, s_levels = solve_beauty_contest(p_val, tau_val)

    pop_mean = float(np.dot(f_dist, s_levels))
    pop_std = float(np.sqrt(np.dot(f_dist, (s_levels - pop_mean) ** 2)))
    nash = 0.0 if p_val < 1 else (100.0 if p_val > 1 else None)
    nash_text = f"{nash:.0f}" if nash is not None else "any common guess (p = 1 is degenerate)"

    mo.md(
        f"""
        With **p = {p_val:.2f}**, **τ = {tau_val:.2f}**:

        | | |
        |---|---|
        | Nash equilibrium | {nash_text} |
        | step-1 guess | {s_levels[1]:.1f} |
        | step-2 guess | {s_levels[2]:.1f} |
        | population-predicted average guess | **{pop_mean:.1f}** (std {pop_std:.1f}) |

        (Actual one-shot p=2/3 experiments typically average 20-35 -- try nudging τ to
        see how much of that gap a single reasoning-depth parameter can close.)
        """
    )
    return f_dist, nash, p_val, pop_mean, s_levels, tau_val


@app.cell(hide_code=True)
def bc_plot1(nash, np, p_val, plt, pop_mean, s_levels, tau_val):
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ks = np.arange(len(s_levels))
    ax1.plot(ks, s_levels, "o-", color="#2f6f4f", label="step-$k$ guess $s_k$")
    ax1.axhline(pop_mean, color="#c0392b", ls="--", lw=1.5,
                label=f"population-predicted mean ({pop_mean:.1f})")
    if nash is not None:
        ax1.axhline(nash, color="0.4", ls=":", lw=1.5, label=f"Nash equilibrium ({nash:.0f})")
    ax1.set_xlabel("thinking step $k$")
    ax1.set_ylabel("guess")
    ax1.set_title(f"convergence of step-$k$ guesses (p = {p_val:.2f}, τ = {tau_val:.2f})")
    ax1.legend(fontsize=8, loc="best")
    ax1.set_xticks(ks[::2])
    fig1
    return (ks,)


@app.cell(hide_code=True)
def bc_plot2(f_dist, ks, plt, tau_val):
    fig2, ax2 = plt.subplots(figsize=(6, 2.6))
    ax2.bar(ks, f_dist, color="#4a6fa5")
    ax2.set_xlabel("step $k$")
    ax2.set_ylabel("$f(k)$")
    ax2.set_title(f"population frequency of step-$k$ thinkers (τ = {tau_val:.2f})")
    ax2.set_xticks(ks[::2])
    fig2
    return


@app.cell(hide_code=True)
def matrix_md(mo):
    mo.md(r"""
    ## 3 &middot; Beyond point guesses: a discrete matrix game

    The beauty contest's payoff only depends on the *mean* of the belief, which is why
    a step-$k$ "belief" could collapse to a single number above. Most of the games this
    reading list actually cares about -- coordination and joint-planning problems --
    don't have that shortcut: the payoff depends on the full probability a partner puts
    on each action, so a step-$k$ player has to best-respond to a genuine mixed-strategy
    belief $\sum_{h<k} g_k(h)\, P_h$, not just its expectation.

    A small stag hunt makes this concrete, and it's the running example of a good chunk
    of this reading list's threads A and D:

    | | Stag | Hare-A | Hare-B |
    |---|---|---|---|
    | **Stag** | 10 | 0 | 0 |
    | **Hare-A** | 7 | 7 | 7 |
    | **Hare-B** | 6 | 6 | 6 |

    (row = your action, column = opponent's action, cell = your payoff.) Matching on
    Stag is payoff-dominant; either Hare is risk-dominant -- safe regardless of what the
    opponent does. A step-0 population puts a third of its mass on each action, which
    already makes Stag look risky to a step-1 best-responder.
    """)
    return


@app.cell(hide_code=True)
def matrix_solver(np, poisson_pmf, truncated_belief):
    def solve_matrix_game(payoff: np.ndarray, tau: float, k_max: int = 15) -> list[np.ndarray]:
        """Solve step-k mixed strategies P_k for a symmetric 2-player normal-form game.

        payoff[i, j] = row player\'s payoff from action i against column action j. Unlike
        the beauty contest, the best response here needs the full belief distribution over
        actions, not just its mean, so P_k is a probability vector (ties in expected payoff
        are split evenly across the tied best responses, per Camerer et al.).
        """
        m = payoff.shape[0]
        f = poisson_pmf(tau, k_max)
        strategy_levels = [np.full(m, 1.0 / m)]  # step 0: uniform
        for k in range(1, k_max + 1):
            g = truncated_belief(f, k)
            belief = sum(g[h] * strategy_levels[h] for h in range(k))
            expected_payoff = payoff @ belief
            best = np.isclose(expected_payoff, expected_payoff.max())
            strategy_levels.append(best / best.sum())
        return strategy_levels

    return (solve_matrix_game,)


@app.cell(hide_code=True)
def matrix_table(mo, np, solve_matrix_game, tau_slider):
    stag_hunt = np.array([
        [10.0, 0.0, 0.0],
        [7.0, 7.0, 7.0],
        [6.0, 6.0, 6.0],
    ])
    actions = ["Stag", "Hare-A", "Hare-B"]

    strategy_levels = solve_matrix_game(stag_hunt, tau_slider.value, k_max=6)
    _rows = "\n".join(
        f"| step {k} | " + " | ".join(f"{p:.2f}" for p in strategy_levels[k]) + " |"
        for k in range(len(strategy_levels))
    )
    mo.md(
        f"""
        Step-$k$ mixed strategies at **τ = {tau_slider.value:.2f}** (reusing the slider above):

        | | {" | ".join(actions)} |
        |---|---|---|---|
        {_rows}
        """
    )
    return actions, strategy_levels


@app.cell(hide_code=True)
def matrix_plot(actions, plt, poisson_pmf, strategy_levels, tau_slider):
    f_stag = poisson_pmf(tau_slider.value, k_max=6)
    pop_play = sum(f_stag[k] * strategy_levels[k] for k in range(len(strategy_levels)))

    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.bar(actions, pop_play, color=["#2f6f4f", "#4a6fa5", "#4a6fa5"])
    ax3.set_ylabel("population-predicted probability")
    ax3.set_title(f"predicted play (τ = {tau_slider.value:.2f})")
    fig3
    return


@app.cell(hide_code=True)
def ll_md(mo):
    mo.md(r"""
    ## 4 &middot; How much does this actually buy you?

    Table IV of the paper reports log-likelihoods for the Poisson-CH model against
    Nash equilibrium, fit to five published datasets (Stahl & Wilson matrix games,
    Cooper & Van Huyck coordination games, Costa-Gomes et al. matrix games, a set of
    mixed-strategy games, and the entry game described in the paper). Higher
    (less negative) is a better fit.
    """)
    return


@app.cell(hide_code=True)
def ll_plot(np, plt):
    # Table IV of Camerer, Ho & Chong (2004): log-likelihood, CH (game-specific tau) vs. Nash.
    datasets = ["Stahl &\nWilson", "Cooper &\nVan Huyck", "Costa-Gomes\net al.", "Mixed", "Entry"]
    ll_ch = np.array([-360, -838, -264, -824, -150])
    ll_nash = np.array([-1823, -5422, -1819, -1270, -154])

    fig4, ax4 = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(datasets))
    width = 0.35
    ax4.bar(x - width / 2, ll_ch, width, label="Cognitive hierarchy", color="#2f6f4f")
    ax4.bar(x + width / 2, ll_nash, width, label="Nash equilibrium", color="#a5a5a5")
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, fontsize=8)
    ax4.set_ylabel("log-likelihood (higher = better fit)")
    ax4.set_title("CH vs. Nash: fit to five published datasets")
    ax4.legend(fontsize=8)
    fig4
    return


@app.cell(hide_code=True)
def closing_md(mo):
    mo.md(r"""
    ## Notes and caveats

    - This notebook reproduces the *mechanics* of the Poisson-CH model, not every
      figure in the paper -- Table II's 24 game-specific $\hat\tau$ estimates and their
      bootstrapped confidence intervals are worth reading directly rather than
      re-derived here, since several outlying subject pools (professional stock
      traders, game theorists) have $\tau$ estimates 3-5&times; the median.
    - Level-0 is *assumed* uniform throughout. Camerer et al. flag this as their
      biggest hidden modeling choice -- swap `s[0] = (lo + hi) / 2` or the first row
      of `strategy_levels` for something else (a salience-weighted guess, a focal
      point) and every level above it changes.
    - Yoshida, Dolan & Friston (2008) -- the other new arrival at the top of Week 1 --
      solve the same "how deep does recursive reasoning go" problem with a hard
      sophistication ceiling $K$ instead of a Poisson-weighted population; Devaine et
      al. (2014, thread C) builds directly on their k-ToM formalism. Worth comparing
      the two bottoming-out mechanisms once both are read.
    """)
    return


if __name__ == "__main__":
    app.run()
