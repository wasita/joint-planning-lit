import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def setup():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def title_md(mo):
    mo.md(r"""
    # Finite Markov Decision Processes (Sutton & Barto, Chapter 3)

    A working notebook version of the chapter summary -- every worked example
    that has concrete enough parameters to simulate (the recycling robot, the
    gridworld, the returns exercise) is implemented as live, interactive code
    rather than just described. Sections without a natural simulation (goals
    and rewards, unified notation, optimality and approximation) stay as
    markdown.
    """)
    return


@app.cell(hide_code=True)
def sec31_md(mo):
    mo.md(r"""
    ## 3.1 -- The Agent-Environment Interface

    Agent and environment interact over discrete time steps. At each step $t$
    the agent receives a state $S_t \in \mathcal{S}$, selects an action
    $A_t \in \mathcal{A}(s)$, and receives a reward $R_{t+1}$ and a new state
    $S_{t+1}$:

    $$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots$$

    A **finite MDP** has finite $\mathcal{S}$, $\mathcal{A}$, $\mathcal{R}$,
    and the dynamics are fully captured by one four-argument function:

    $$p(s', r \mid s, a) \doteq \Pr\{S_t=s', R_t=r \mid S_{t-1}=s, A_{t-1}=a\}$$

    Probabilities depending only on the immediately preceding state/action
    (not the full history) is the **Markov property**. The agent-environment
    boundary is not a robot's physical body -- the rule is: anything the agent
    cannot change arbitrarily counts as environment. The boundary marks the
    limit of the agent's *control*, not its *knowledge*.

    **Example 3.3 -- Recycling Robot** (the running example used again in
    Section 3.6 below): two battery states, `high`/`low`. At `high` the agent
    can `search` or `wait`; at `low` it can also `recharge`. Searching risks
    draining the battery (a costly rescue, reward $-3$); recharging is safe
    but forgoes a step of searching.

    | $s$ | $a$ | $s'$ | $p(s'\mid s,a)$ | $r(s,a,s')$ |
    |---|---|---|---|---|
    | high | search | high | $\alpha$ | $r_\text{search}$ |
    | high | search | low | $1-\alpha$ | $r_\text{search}$ |
    | high | wait | high | $1$ | $r_\text{wait}$ |
    | low | search | low | $\beta$ | $r_\text{search}$ |
    | low | search | high | $1-\beta$ | $-3$ |
    | low | wait | low | $1$ | $r_\text{wait}$ |
    | low | recharge | high | $1$ | $0$ |

    Try your own parameters below and see which action is optimal in each
    state (this instantiates the Bellman optimality equation worked out
    symbolically in Example 3.9, Section 3.6).
    """)
    return


@app.cell(hide_code=True)
def rr_sliders(mo):
    alpha_s = mo.ui.slider(0.0, 1.0, step=0.05, value=0.7, label="α (search succeeds at high)")
    beta_s = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="β (search succeeds at low)")
    rsearch_s = mo.ui.slider(0.0, 5.0, step=0.1, value=2.0, label="r_search")
    rwait_s = mo.ui.slider(0.0, 5.0, step=0.1, value=0.5, label="r_wait")
    rr_gamma_s = mo.ui.slider(0.0, 0.99, step=0.01, value=0.9, label="γ")
    mo.vstack([alpha_s, beta_s, rsearch_s, rwait_s, rr_gamma_s])
    return alpha_s, beta_s, rr_gamma_s, rsearch_s, rwait_s


@app.cell(hide_code=True)
def rr_solve(alpha_s, beta_s, mo, rr_gamma_s, rsearch_s, rwait_s):
    alpha, beta = alpha_s.value, beta_s.value
    r_search, r_wait, gamma_rr = rsearch_s.value, rwait_s.value, rr_gamma_s.value

    rr_transitions = {
        "high": {
            "search": [(alpha, "high", r_search), (1 - alpha, "low", r_search)],
            "wait": [(1.0, "high", r_wait)],
        },
        "low": {
            "search": [(beta, "low", r_search), (1 - beta, "high", -3.0)],
            "wait": [(1.0, "low", r_wait)],
            "recharge": [(1.0, "high", 0.0)],
        },
    }

    v_rr = {"high": 0.0, "low": 0.0}
    for _ in range(500):
        new_v = {}
        for s, actions in rr_transitions.items():
            qs = {a: sum(p * (r + gamma_rr * v_rr[s2]) for p, s2, r in outcomes)
                  for a, outcomes in actions.items()}
            new_v[s] = max(qs.values())
        if max(abs(new_v[s] - v_rr[s]) for s in v_rr) < 1e-10:
            v_rr = new_v
            break
        v_rr = new_v

    rr_rows = []
    for s, actions in rr_transitions.items():
        qs = {a: sum(p * (r + gamma_rr * v_rr[s2]) for p, s2, r in outcomes)
              for a, outcomes in actions.items()}
        best = max(qs, key=qs.get)
        rr_rows.append(f"| {s} | **{best}** | {v_rr[s]:.2f} | " +
                        ", ".join(f"{a}={q:.2f}" for a, q in qs.items()) + " |")

    mo.md(
        "**Solved via value iteration on the Bellman optimality equation:**\n\n"
        "| state | optimal action | $v_*(s)$ | $q_*(s,a)$ for each action |\n"
        "|---|---|---|---|\n" + "\n".join(rr_rows)
    )
    return


@app.cell(hide_code=True)
def sec32_md(mo):
    mo.md(r"""
    ## 3.2 -- Goals and Rewards

    The **reward hypothesis**: *"All of what we mean by goals and purposes
    can be well thought of as the maximization of the expected value of the
    cumulative sum of a received scalar signal (reward)."*

    The reward signal should communicate **what** you want achieved, not
    **how** to achieve it. Rewarding subgoals (e.g., a chess agent rewarded
    for capturing pieces) risks the agent finding a way to hit the subgoal
    without achieving the real goal. Prior knowledge about *how* to solve a
    task belongs in the initial policy or value function, not the reward.
    """)
    return


@app.cell(hide_code=True)
def sec33_md(mo):
    mo.md(r"""
    ## 3.3 -- Returns and Episodes

    The agent maximizes the expected **return** $G_t$. Undiscounted, for
    **episodic** tasks with a final step $T$:

    $$G_t \doteq R_{t+1} + R_{t+2} + \dots + R_T$$

    For **continuing** tasks ($T=\infty$), use **discounting** instead:

    $$G_t \doteq \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}, \qquad \gamma \in [0,1]$$

    Returns at successive steps relate recursively -- this is the identity
    almost everything later in the book leans on:

    $$G_t = R_{t+1} + \gamma G_{t+1}$$

    **Exercise 3.8, worked live below:** $\gamma=0.5$ and reward sequence
    $R_1=-1, R_2=2, R_3=6, R_4=3, R_5=2$, with $T=5$. Compute
    $G_0, \dots, G_5$ by working *backwards* from $G_T \doteq 0$.
    """)
    return


@app.cell(hide_code=True)
def returns_slider(mo):
    returns_gamma_s = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="γ")
    returns_gamma_s
    return (returns_gamma_s,)


@app.cell(hide_code=True)
def returns_compute(mo, np, returns_gamma_s):
    R = np.array([-1.0, 2.0, 6.0, 3.0, 2.0])  # R_1 .. R_5
    T = len(R)
    gamma_ret = returns_gamma_s.value

    G = np.zeros(T + 1)  # G[t] for t = 0..T, with G[T] = 0
    for t in range(T - 1, -1, -1):
        G[t] = R[t] + gamma_ret * G[t + 1]

    rows = "\n".join(f"| $G_{t}$ | {G[t]:.3f} |" for t in range(T + 1))
    mo.md(f"**γ = {gamma_ret:.2f}**\n\n| return | value |\n|---|---|\n{rows}")
    return


@app.cell(hide_code=True)
def sec34_md(mo):
    mo.md(r"""
    ## 3.4 -- Unified Notation for Episodic and Continuing Tasks

    Episode termination is reframed as entering a special **absorbing
    state** that transitions only to itself and always emits reward $0$.
    This lets one formula cover both task types:

    $$G_t \doteq \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k$$

    allowing $T=\infty$ or $\gamma=1$ (but not both) -- no simulation needed
    here, it's purely a notational unification.
    """)
    return


@app.cell(hide_code=True)
def sec35_md(mo):
    mo.md(r"""
    ## 3.5 -- Policies and Value Functions

    A **policy** $\pi(a\mid s)$ maps states to action probabilities.

    $$v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] \qquad
    q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$$

    The **Bellman equation for $v_\pi$** -- the value of a state equals
    expected immediate reward plus discounted value of the next state,
    averaged over the policy's action probabilities and the environment's
    dynamics:

    $$v_\pi(s) = \sum_a \pi(a\mid s) \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma v_\pi(s')\big]$$

    **Example 3.5 -- Gridworld**, solved live below: a 5x5 grid. From state
    A, every action yields reward $+10$ and teleports to A'. From state B,
    every action yields reward $+5$ and teleports to B'. Any action that
    would leave the grid keeps you in place with reward $-1$; all other
    moves give reward $0$. Solved here for the **equiprobable random
    policy** (each of the 4 actions with probability $0.25$), by directly
    solving the linear system $v_\pi = (I - \gamma P_\pi)^{-1} r_\pi$.
    """)
    return


@app.cell(hide_code=True)
def gridworld_setup(np):
    GW_N = 5
    GW_A, GW_APRIME, GW_AREWARD = (0, 1), (4, 1), 10.0
    GW_B, GW_BPRIME, GW_BREWARD = (0, 3), (2, 3), 5.0
    GW_ACTIONS = ["north", "south", "east", "west"]
    GW_DELTAS = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}

    def gw_step(s, a):
        if s == GW_A:
            return GW_APRIME, GW_AREWARD
        if s == GW_B:
            return GW_BPRIME, GW_BREWARD
        dr, dc = GW_DELTAS[a]
        r, c = s[0] + dr, s[1] + dc
        if r < 0 or r >= GW_N or c < 0 or c >= GW_N:
            return s, -1.0
        return (r, c), 0.0

    def gw_idx(s):
        return s[0] * GW_N + s[1]

    return GW_A, GW_ACTIONS, GW_B, GW_N, gw_idx, gw_step


@app.cell(hide_code=True)
def gridworld_gamma_slider(mo):
    gw_gamma_s = mo.ui.slider(0.0, 0.99, step=0.01, value=0.9, label="γ (gridworld)")
    gw_gamma_s
    return (gw_gamma_s,)


@app.cell(hide_code=True)
def gridworld_random_policy(GW_ACTIONS, GW_N, gw_gamma_s, gw_idx, gw_step, np, plt):
    gw_gamma = gw_gamma_s.value
    P_pi = np.zeros((GW_N * GW_N, GW_N * GW_N))
    r_pi = np.zeros(GW_N * GW_N)
    for row in range(GW_N):
        for col in range(GW_N):
            i = gw_idx((row, col))
            for a in GW_ACTIONS:
                s2, rew = gw_step((row, col), a)
                P_pi[i, gw_idx(s2)] += 0.25
                r_pi[i] += 0.25 * rew

    v_random = np.linalg.solve(np.eye(GW_N * GW_N) - gw_gamma * P_pi, r_pi).reshape(GW_N, GW_N)

    fig_gw1, ax_gw1 = plt.subplots(figsize=(4.5, 4.5))
    ax_gw1.imshow(v_random, cmap="Blues")
    for row in range(GW_N):
        for col in range(GW_N):
            ax_gw1.text(col, row, f"{v_random[row, col]:.1f}", ha="center", va="center")
    ax_gw1.set_xticks([])
    ax_gw1.set_yticks([])
    ax_gw1.set_title(f"$v_\\pi$, equiprobable random policy (γ={gw_gamma:.2f})")
    plt.close(fig_gw1)
    fig_gw1
    return


@app.cell(hide_code=True)
def sec36_md(mo):
    mo.md(r"""
    ## 3.6 -- Optimal Policies and Optimal Value Functions

    $$v_*(s) \doteq \max_\pi v_\pi(s) \qquad q_*(s,a) \doteq \max_\pi q_\pi(s,a)$$

    Because $v_*$ is optimal, its self-consistency condition can be written
    without reference to any specific policy -- a max over actions instead
    of an expectation over $\pi$. This is the **Bellman optimality
    equation**:

    $$v_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma v_*(s')\big]$$

    $$q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\Big[r+\gamma \max_{a'} q_*(s',a')\Big]$$

    A policy that is **greedy** with respect to $v_*$ is automatically
    optimal, because $v_*$ already accounts for all future consequences of
    each immediate choice -- no lookahead needed. Solved below for the same
    gridworld via **value iteration** (repeatedly applying the Bellman
    optimality backup until convergence), replicating Figure 3.5: the
    optimal value function and the resulting greedy policy (arrows -- more
    than one per cell where actions tie).
    """)
    return


@app.cell(hide_code=True)
def gridworld_optimal(GW_ACTIONS, GW_N, gw_gamma_s, gw_idx, gw_step, np, plt):
    gw_gamma_star = gw_gamma_s.value
    v_star = np.zeros(GW_N * GW_N)
    for _ in range(1000):
        v_new = np.zeros(GW_N * GW_N)
        for orow in range(GW_N):
            for ocol in range(GW_N):
                qs2 = [gw_step((orow, ocol), act)[1] + gw_gamma_star * v_star[gw_idx(gw_step((orow, ocol), act)[0])]
                       for act in GW_ACTIONS]
                v_new[gw_idx((orow, ocol))] = max(qs2)
        if np.max(np.abs(v_new - v_star)) < 1e-10:
            v_star = v_new
            break
        v_star = v_new

    arrow = {"north": (0, -0.35), "south": (0, 0.35), "east": (0.35, 0), "west": (-0.35, 0)}
    fig_gw2, ax_gw2 = plt.subplots(figsize=(4.5, 4.5))
    ax_gw2.imshow(v_star.reshape(GW_N, GW_N), cmap="Greens")
    for orow in range(GW_N):
        for ocol in range(GW_N):
            val = v_star[gw_idx((orow, ocol))]
            ax_gw2.text(ocol, orow - 0.25, f"{val:.1f}", ha="center", va="center", fontsize=8)
            qs2 = {act: gw_step((orow, ocol), act)[1] + gw_gamma_star * v_star[gw_idx(gw_step((orow, ocol), act)[0])]
                   for act in GW_ACTIONS}
            best_q = max(qs2.values())
            for act, qval in qs2.items():
                if np.isclose(qval, best_q, atol=1e-6):
                    dx, dy = arrow[act]
                    ax_gw2.arrow(ocol, orow + 0.15, dx, dy, head_width=0.08, color="black")
    ax_gw2.set_xticks([])
    ax_gw2.set_yticks([])
    ax_gw2.set_title(f"$v_*$ and greedy $\\pi_*$ (γ={gw_gamma_star:.2f})")
    plt.close(fig_gw2)
    fig_gw2
    return


@app.cell(hide_code=True)
def sec37_md(mo):
    mo.md(r"""
    ## 3.7 -- Optimality and Approximation

    Explicitly solving the Bellman optimality equation is rarely useful: it
    amounts to exhaustive search and needs (1) accurately known dynamics,
    (2) enough compute, and (3) the Markov property -- rarely all true at
    once. Backgammon has roughly $10^{20}$ states; even with fully known
    dynamics, exact solution would take thousands of years on today's
    hardware. In practice RL settles for **approximate** solutions, and RL's
    *online* nature lets an agent spend more approximation effort on
    frequently-encountered states and less on rare ones (TD-Gammon plays
    with exceptional skill despite poor decisions in board configurations
    that almost never arise against real opponents).
    """)
    return


@app.cell(hide_code=True)
def sec38_md(mo):
    mo.md(r"""
    ## 3.8 -- Summary

    RL is learning from interaction to achieve a goal, formalized through
    three signals: actions, states, rewards. When this satisfies the Markov
    property, it's a (finite) MDP. The **return** aggregates future reward,
    unified across episodic and continuing tasks. **Value functions**
    ($v_\pi$, $q_\pi$) give expected return under a policy; **optimal value
    functions** ($v_*$, $q_*$) give the best achievable expected return and
    satisfy the **Bellman optimality equations**, from which an optimal
    policy follows by acting greedily.
    """)
    return


@app.cell(hide_code=True)
def key_equations_md(mo):
    mo.md(r"""
    ## Key equations at a glance

    | # | Name | Equation |
    |---|------|----------|
    | 3.2 | Dynamics function | $p(s',r\mid s,a) \doteq \Pr\{S_t=s',R_t=r\mid S_{t-1}=s,A_{t-1}=a\}$ |
    | 3.8 | Discounted return | $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$ |
    | 3.9 | Recursive return | $G_t = R_{t+1} + \gamma G_{t+1}$ |
    | 3.12 | State-value function | $v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t=s]$ |
    | 3.13 | Action-value function | $q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t \mid S_t=s,A_t=a]$ |
    | 3.14 | Bellman equation for $v_\pi$ | $v_\pi(s) = \sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)[r+\gamma v_\pi(s')]$ |
    | 3.19 | Bellman optimality eq. for $v_*$ | $v_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)[r+\gamma v_*(s')]$ |
    | 3.20 | Bellman optimality eq. for $q_*$ | $q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)[r+\gamma \max_{a'} q_*(s',a')]$ |
    """)
    return


@app.cell(hide_code=True)
def ch_bridge_md(mo):
    mo.md(r"""
    ## Note: connection to the Poisson-CH reading

    Worth reading alongside the Camerer, Ho & Chong cognitive hierarchy
    equations (`ch_equations_reference.py`). $q_\pi(s,a)$ and $Q_k(a_i)$
    (the CH-to-RL rewrite of Equation 5) share the same shape -- an
    expected value computed by averaging a payoff/reward over a
    distribution (policy vs. belief-weighted opponent mixture). The key
    structural difference: Bellman's $v_*$/$q_*$ bootstrap off a state's
    *own* future self through *time* (via $S_{t+1}$), while Poisson-CH's
    $g_k(h)$ bootstraps off *other agents'* lower reasoning *levels* --
    recursion through depth-of-reasoning rather than through time. Both
    frameworks solve the "avoid infinite regress" problem, just along
    different axes (temporal horizon vs. reasoning depth).
    """)
    return


if __name__ == "__main__":
    app.run()
