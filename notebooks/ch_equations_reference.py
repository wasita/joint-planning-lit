import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", sql_output="polars")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def title_md(mo):
    mo.md(r"""
    # Poisson Cognitive Hierarchy: Equation Reference

    A term-by-term reference for every equation from Camerer, Ho & Chong (2004),
    *A Cognitive Hierarchy Model of Games*, including what each symbol means, how it fits the broader CH story, and how
    it plays out in the two running examples: the *p*-beauty contest and the
    stag hunt.
    """)
    return


@app.cell(hide_code=True)
def eq1_step0(mo):
    mo.md(r"""
    ## Equation 1 -- the step-0 decision rule

    $$P_0\left(s_i^j\right) = \frac{1}{m_i} \quad \forall j$$

    **Terms**

    - $P_0(\cdot)$ -- the step-0 choice-probability function.
    - $s_i^j$ -- player $i$'s $j$-th strategy.
    - $m_i$ -- the total number of strategies available to player $i$.
    - $\forall j$ -- holds for *every* strategy, which is what makes the
      distribution uniform.

    **What it means for cognitive hierarchy theory**

    Step 0 is the model's placeholder for zero strategic reasoning: a step-0
    player doesn't think about opponents, payoffs, or the structure of the game
    at all -- they just pick uniformly at random. Every level built on top of the
    hierarchy is defined *relative to* this non-strategic baseline, which is why
    Camerer et al. flag it as the single most consequential (and most easily
    mis-specified) modeling choice in the whole framework.

    **Beauty contest:** the step-0 guess is uniform over $[0, 100]$, with mean
    $50$ -- this seeds $s_0 = 50$ in the guess recursion (Equation 7).

    **Stag hunt:** the step-0 population randomizes with full support over
    {Stag, Hare}. That full-support randomization is exactly what guarantees a
    1-step thinker never plays a weakly dominated strategy (see the dominance
    note near Equation 7).
    """)
    return


@app.cell(hide_code=True)
def eq1_step0_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 1

    $$\pi_0(a_i) = \frac{1}{|\mathcal{A}_i|} \quad \forall a_i \in \mathcal{A}_i$$

    Uniform random policy -- an untrained, no-lookahead policy with no value
    estimation behind it at all. Direct RL analogue of the step-0 decision
    rule: same math, agent-policy framing instead of player-strategy framing.
    """)
    return


@app.cell(hide_code=True)
def eq2_belief_bound(mo):
    mo.md(r"""
    ## Equation 2 -- the belief-bound assumption

    $$g_k(h) = 0,\ \forall h \ge k+1 \qquad \text{and} \qquad g_k(k) = 0$$

    **Terms**

    - $g_k(h)$ -- a step-$k$ player's belief about the proportion of the
      population doing exactly $h$ steps of thinking.
    - $k$ -- the believer's own thinking level.
    - $h$ -- the level being believed about.
    - the first clause -- a step-$k$ player assigns zero probability to anyone
      reasoning *more* than they do.
    - the second clause -- a step-$k$ player also assigns zero probability to
      anyone reasoning at *exactly* their own level (overconfidence).

    **What it means for cognitive hierarchy theory**

    Together these two conditions bound the recursion: a step-$k$ player's
    beliefs only ever put weight on levels $0$ through $k-1$. This is what makes
    the model solvable by simple bottom-up substitution rather than by finding a
    fixed point -- each level's problem only ever depends on levels that have
    already been solved. It's also the model's formalization of a bounded
    working-memory constraint: nobody can conceive of a mind more sophisticated
    than (or exactly as sophisticated as) their own.

    **Beauty contest:** a step-$k$ guesser never imagines an opponent doing $k$
    or more steps, so the guess recursion (Equation 7) only ever reaches down,
    never sideways or up.

    **Stag hunt:** the same bound is what keeps the matrix-game solver a clean
    recursion (step 0, then step 1, then step 2, ...) instead of requiring an
    equilibrium-style fixed-point search.
    """)
    return


@app.cell(hide_code=True)
def eq2_belief_bound_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 2

    $$b_k(h) = 0,\ \forall h \ge k+1 \qquad \text{and} \qquad b_k(k) = 0$$

    $b_k$ -- an agent's belief distribution over which recursion-depth "tier"
    the opponent's policy belongs to. Truncating at $h < k$ is bounded
    opponent modeling: it prevents infinite recursive self-simulation ("I
    model them modeling me modeling them...").
    """)
    return


@app.cell(hide_code=True)
def eq3_poisson(mo):
    mo.md(r"""
    ## Equation 3 -- the Poisson population distribution

    $$f(k) = \frac{e^{-\tau}\tau^k}{k!}$$

    **Terms**

    - $f(k)$ -- the *true* share of the population doing exactly $k$ steps of
      thinking.
    - $\tau$ -- the model's single free parameter: simultaneously the mean and
      the variance of the distribution.
    - $\tau^k$ -- grows with $k$; reaching a deeper level requires more of
      whatever $\tau$ represents.
    - $k!$ -- the combinatorial cost of nested reasoning: a step-$k$ thinker
      must redo every computation a step-$(k-1)$ thinker does, then add one
      more, so the "cost" of going one level deeper compounds.
    - $e^{-\tau}$ -- the normalizing constant. Note that plugging in $k=0$ gives
      $f(0) = e^{-\tau}$ directly -- this constant *is* the model's predicted
      share of purely non-strategic (step-0) players.

    **What it means for cognitive hierarchy theory**

    This is the generative model of the whole population: a single number,
    $\tau$, stands in for how strategically sophisticated a population is on
    average. Because $\tau$ is both the mean and the variance, a population with
    a higher average reasoning depth is also predicted to be more spread out in
    depth, not just uniformly shifted upward.

    **Beauty contest:** fit to 24 *p*-beauty-contest datasets, the median
    $\hat\tau$ is $1.61$. That's why the empirical average lands around 20-35
    instead of converging all the way to the Nash prediction of 0.

    **Stag hunt:** $\tau$ sets how much population mass sits at each level,
    which feeds directly into the belief weights ($g_k(h)$, Equation 4) used to
    predict how much of the group ends up playing Stag versus Hare, in both the
    two-player and three-player versions of the game.
    """)
    return


@app.cell(hide_code=True)
def eq3_poisson_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 3

    $$p(k) = \frac{e^{-\tau}\tau^k}{k!}$$

    Prior over how many rounds of best-response iteration (think: search or
    rollout depth) an agent in the population runs before committing to a
    policy. $\tau$ is the population's average planning depth -- like a prior
    on lookahead budget.
    """)
    return


@app.cell(hide_code=True)
def eq4_truncated_belief(mo):
    mo.md(r"""
    ## Equation 4 -- the truncated, renormalized belief

    $$g_k(h) = \frac{f(h)}{\sum_{l=0}^{k-1} f(l)}, \qquad h < k$$

    **Terms**

    - $f(h)$ -- the true population frequency at level $h$.
    - $\sum_{l=0}^{k-1} f(l)$ -- the total true mass sitting at every level below
      $k$; this renormalizes the truncated slice back into a valid probability
      distribution.

    **What it means for cognitive hierarchy theory**

    This is the piece that makes the model specifically *Poisson-CH* rather than
    generic "cognitive hierarchy": a step-$k$ player's belief isn't a made-up
    shape, it's the actual truth, just chopped off above $k-1$ and rescaled. As
    $k$ grows, that truncated slice converges toward the true $f$ -- the paper's
    "increasingly rational expectations" property. This is the direct opposite
    of the alternative ("level-$k$") models, which instead concentrate all
    belief on the single level directly below ($g_k(k-1) = 1$), producing
    *increasingly* irrational expectations as $k$ grows.

    **Beauty contest:** supplies the weights $g_k(h)$ used on each lower-level
    guess $s_h$ inside the recursion in Equation 7.

    **Stag hunt:** supplies the weights used to mix step-0 through step-$(k-1)$
    strategies into the belief a step-$k$ player best-responds to, when deciding
    between Stag and Hare.
    """)
    return


@app.cell(hide_code=True)
def eq4_truncated_belief_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 4

    $$b_k(h) = \frac{p(h)}{\sum_{l=0}^{k-1} p(l)}, \qquad h < k$$

    Bayesian belief over the opponent's policy tier, renormalized after
    truncating out tiers $\ge k$. This is the mixture weighting used to build
    the opponent model in Equation 5's RL rewrite below.
    """)
    return


@app.cell(hide_code=True)
def eq5_expected_payoff(mo):
    mo.md(r"""
    ## Equation 5 -- expected payoff

    $$E_k\left(\pi_i\left(s_i^j\right)\right) = \sum_{j'=1}^{m_{-i}} \pi_i\left(s_i^j, s_{-i}^{j'}\right)\left\{\sum_{h=0}^{k-1} g_k(h)\, P_h\left(s_{-i}^{j'}\right)\right\}$$

    **Terms**

    - $E_k(\cdot)$ -- the expected-payoff operator for a step-$k$ thinker.
    - $\pi_i\left(s_i^j, s_{-i}^{j'}\right)$ -- player $i$'s payoff from playing
      strategy $j$ when the opponent plays strategy $j'$.
    - $m_{-i}$ -- the number of strategies available to the opponent.
    - $P_h\left(s_{-i}^{j'}\right)$ -- the probability that a step-$h$ opponent
      plays strategy $j'$.
    - $g_k(h)$ -- the belief weight (Equation 4) placed on opponents being at
      level $h$.

    **What it means for cognitive hierarchy theory**

    This is general machinery, shared by every model in the CH family regardless
    of what specific $g_k$ or $f(k)$ you plug in -- it's just "average payoff
    across every opponent strategy, weighted by how likely each opponent type is
    to play it, weighted by how likely each opponent type is to exist."

    **Beauty contest:** because the payoff there depends only on the *mean* of
    the opponents' guesses, this whole double sum collapses to a single number
    -- see the simplified point-value form in Equation 7.

    **Stag hunt:** no such shortcut exists -- payoffs depend on the full
    probability an opponent puts on Stag versus Hare, not just an average, so
    the full form of this equation is needed. That's precisely why the paper
    introduces a genuine matrix-game version of CH for coordination games.
    """)
    return


@app.cell(hide_code=True)
def eq5_expected_payoff_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 5

    $$Q_k(a_i) = \sum_{a_{-i}} R_i(a_i, a_{-i})\, \bar\pi_{-i}^{(k)}(a_{-i}), \qquad \bar\pi_{-i}^{(k)}(a_{-i}) = \sum_{h=0}^{k-1} b_k(h)\, \pi_h(a_{-i})$$

    $\bar\pi_{-i}^{(k)}$ -- a belief-marginalized opponent policy: a mixture
    of lower-tier opponent policies weighted by belief. $Q_k(a_i)$ is the
    expected return of action $a_i$ against that mixture -- the standard
    opponent-modeling Q-value used in multi-agent RL. ($R_i$ replaces econ's
    $\pi_i$ here to avoid clashing with RL's own use of $\pi$ for policy.)
    """)
    return


@app.cell(hide_code=True)
def eq6_best_response(mo):
    mo.md(r"""
    ## Equation 6 -- the best-response rule

    $$P_k\left(s_i^{\ast}\right) = 1 \quad \text{iff} \quad s_i^{\ast} = \operatorname*{arg\,max}_{s_i^j} E_k\left(\pi_i\left(s_i^j\right)\right)$$

    **Terms**

    - $P_k(\cdot)$ -- the step-$k$ decision rule.
    - $s_i^{\ast}$ -- the strategy chosen with certainty.
    - $\operatorname*{arg\,max}_{s_i^j}$ -- whichever strategy maximizes expected
      payoff (Equation 5); ties are split evenly across the tied maximizers.

    **What it means for cognitive hierarchy theory**

    This closes the loop of the whole model: bounded belief (Equations 2 and 4)
    feeds into an expected-payoff calculation (Equation 5), and the step-$k$
    player simply best-responds to it. Nothing here is boundedly rational in
    itself -- the *only* bounded-rationality assumption in the entire model is
    in what a player believes about others, not in how they optimize given that
    belief.

    **Beauty contest:** picks the single number minimizing distance to $p \times$
    the believed average guess.

    **Stag hunt:** picks Stag or Hare (or splits evenly on a tie) by comparing
    expected payoffs under the belief-weighted mix from Equation 5.
    """)
    return


@app.cell(hide_code=True)
def eq6_best_response_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 6

    $$\pi_k(a_i) = \mathbb{1}\left[a_i = \operatorname*{arg\,max}_{a_i'} Q_k(a_i')\right]$$

    Greedy policy improvement with respect to $Q_k$ -- the standard "improve"
    step of policy iteration, ties split uniformly.
    """)
    return


@app.cell(hide_code=True)
def eq7_beauty_contest_applied(mo):
    mo.md(r"""
    ## Equation 7 -- applied to the beauty contest (derived from Equations 4-6)

    $$s_0 = 50, \qquad s_k = p\sum_{h=0}^{k-1} g_k(h)\, s_h \quad (k \ge 1)$$

    This is the point-value instantiation of Equations 4-6 for a game whose
    payoff depends only on the mean of the guess distribution. It solves purely
    by substitution, bottom-up from $s_0$.

    **Dominance-solvability:** iterating this recursion converges to the Nash
    prediction of $0$ as $\tau \to \infty$. At the empirically fitted
    $\tau \approx 1.5$, it stops well short of that -- matching the "converges
    to ~20-35, not 0" pattern seen in real one-shot experiments.

    A related result used to prove this: $f(k-1)/f(k-2) = \tau/(k-1)$. For
    $k \ll \tau$, this puts almost all belief-weight on the $k-1$ type directly
    below (i.e., $g_k(k-1) \approx 1-\epsilon$), which is what lets Camerer et
    al. show that step-by-step CH reasoning mimics, level by level, the
    classical *iterated deletion of dominated strategies* -- without ever
    building "delete the dominated strategy" into the rules by hand.

    **This machinery does not apply to the stag hunt.** The stag hunt is *not*
    dominance-solvable -- neither Stag nor Hare is ever weakly dominated, since
    which one pays better flips depending on what the rest of the group does.
    Dominance reasoning alone can't narrow the stag hunt down to a unique
    prediction; the full belief-weighted machinery of Equations 4-6 has to carry
    the entire predictive burden there instead.
    """)
    return


@app.cell(hide_code=True)
def eq7_beauty_contest_applied_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 7

    $$V_0 = 50, \qquad V_k = p\sum_{h=0}^{k-1} b_k(h)\, V_h$$

    A finite-horizon, non-self-referential value backup: each tier's value
    bootstraps off lower tiers' values only, weighted by belief -- like a
    depth-limited expectimax backup with no recursion into your own value.
    """)
    return


@app.cell(hide_code=True)
def stag_hunt_applied(mo):
    mo.md(r"""
    ## Stag hunt: applying Equations 3, 4, and 6 directly

    No new general equation is needed here -- the stag hunt is solved with the
    same population distribution (Equation 3), the same belief rule
    (Equation 4), and the same best-response rule (Equation 6) used everywhere
    else in the model. Only the payoff structure changes.

    **Payoffs:** choosing Hare guarantees payoff $x$ regardless of what anyone
    else does. Choosing Stag pays $1$ if *everyone* chooses Stag, and $0$ if
    even one other player chooses Hare.

    **Two-player game:** a step-1 thinker best-responds to a single step-0
    (randomizing) opponent, and picks Stag iff $x < 1/2$, Hare iff $x > 1/2$.

    **Three-player game:** a step-1 thinker now faces two independent step-0
    opponents, so the chance at least one of them picks Hare is $0.75$ rather
    than $0.5$. That tougher bar for coordination means Stag is only chosen iff
    $x \le 0.25$.

    **Predicted group play:** the population-level probability of Hare works
    out to $1 - f(0)/2$ -- about $89\%$ at $\tau = 1.5$ -- illustrating the
    paper's point that larger groups get pulled toward the safe, inefficient
    equilibrium purely because coordination gets statistically harder as the
    number of people you have to simultaneously coordinate with grows.
    """)
    return


if __name__ == "__main__":
    app.run()
