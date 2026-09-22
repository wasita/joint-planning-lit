---
title: "Finite Markov Decision Processes (Chapter 3 summary)"
source: "Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 3"
type: textbook-chapter
tags: [reference, RL, MDP]
---

# Chapter 3 — Finite Markov Decision Processes

Chapter 3 formalizes the reinforcement learning problem as a **finite Markov
decision process (MDP)**. Where Chapter 2's bandit problems only required
estimating the value of each action, MDPs add an *associative* element:
actions are chosen in different *situations* (states), and actions influence
not just immediate reward but future states and future reward. This is the
mathematically idealized framework the rest of the book builds on.

## 3.1 — The Agent–Environment Interface

The **agent** (learner/decision-maker) and the **environment** (everything
outside the agent) interact over discrete time steps $t = 0, 1, 2, \dots$.
At each step the agent receives a state $S_t \in \mathcal{S}$, selects an
action $A_t \in \mathcal{A}(s)$, and one step later receives a reward
$R_{t+1} \in \mathcal{R}$ and lands in a new state $S_{t+1}$. This produces a
trajectory:

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots \tag{3.1}$$

In a **finite** MDP, $\mathcal{S}$, $\mathcal{A}$, and $\mathcal{R}$ are all
finite sets, and the dynamics are fully captured by a single four-argument
function:

$$p(s', r \mid s, a) \doteq \Pr\{S_t=s', R_t=r \mid S_{t-1}=s, A_{t-1}=a\} \tag{3.2}$$

This is a deterministic function $p: \mathcal{S}\times\mathcal{R}\times\mathcal{S}\times\mathcal{A}\to[0,1]$
that defines a valid probability distribution for every $(s,a)$ pair
(Eq. 3.3). Because probabilities at time $t$ depend *only* on the
immediately preceding state and action — not on the full history — the
state is said to have the **Markov property**. This is a restriction on
what the state must encode (everything from the past that matters for the
future), not on the environment itself.

From $p$, other useful quantities can be derived: state-transition
probabilities $p(s'\mid s,a)$ (3.4), expected reward for a state-action pair
$r(s,a)$ (3.5), and expected reward for a state-action-next-state triple
$r(s,a,s')$ (3.6).

**The agent–environment boundary is not the physical boundary of a
robot/animal's body.** The general rule: anything the agent cannot change
arbitrarily is considered part of the environment (e.g., a robot's motors
and sensors, an animal's muscles and sensory organs, and the reward
computation itself are all "environment," even if the agent has full
knowledge of how they work). The boundary represents the limit of the
agent's *control*, not its *knowledge* — an agent can know exactly how a
Rubik's cube works and still face a hard RL problem.

**Worked examples:** a bioreactor (continuous sensor-based states, target
temperature/stirring actions), a pick-and-place robot arm (joint angle/
velocity states, motor voltage actions, smoothness-penalizing rewards), and
the **recycling robot** (two states, `high`/`low` battery; actions `search`,
`wait`, `recharge`; a small worked finite MDP with an explicit transition
table and transition graph used throughout the rest of the chapter).

## 3.2 — Goals and Rewards

The **reward hypothesis**: *"All of what we mean by goals and purposes can
be well thought of as the maximization of the expected value of the
cumulative sum of a received scalar signal (reward)."* This is one of RL's
most distinctive and, at first glance, limiting-seeming design choices — but
it has proven broadly flexible in practice.

Critically, the reward signal should communicate **what** you want achieved,
not **how** to achieve it. Rewarding subgoals (e.g., a chess agent rewarded
for capturing pieces) risks the agent finding a way to hit the subgoal
without accomplishing the real goal (e.g., trading pieces at the cost of the
game). Prior knowledge about *how* to solve a task belongs in the initial
policy or value function, not the reward signal.

## 3.3 — Returns and Episodes

The agent's objective is formalized as maximizing the **expected return**
$G_t$, a function of the future reward sequence. In the simplest case, the
return is just the sum of rewards:

$$G_t \doteq R_{t+1} + R_{t+2} + \dots + R_T \tag{3.7}$$

This works for **episodic tasks** — tasks that break naturally into
episodes with a terminal state (games, trips through a maze), after which
the environment resets. For **continuing tasks** (no natural endpoint, e.g.
an ongoing process-control task), $T=\infty$ and the undiscounted sum can
diverge, so **discounting** is introduced instead:

$$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1} \tag{3.8}$$

where $\gamma \in [0,1]$ is the **discount rate**. $\gamma=0$ makes the
agent "myopic" (only $R_{t+1}$ matters); as $\gamma\to1$ the agent becomes
increasingly farsighted. Returns at successive steps relate recursively —
a fact central to nearly everything that follows:

$$G_t = R_{t+1} + \gamma G_{t+1} \tag{3.9}$$

For a constant reward of $+1$ forever, $G_t = \sum_{k=0}^\infty \gamma^k = \dfrac{1}{1-\gamma}$ (3.10).

## 3.4 — Unified Notation for Episodic and Continuing Tasks

To write one set of equations that covers both task types, episode
termination is reframed as entering a special **absorbing state** that
transitions only to itself and always emits reward $0$. This makes the
finite sum (3.7) and infinite discounted sum (3.8) special cases of one
general definition:

$$G_t \doteq \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k \tag{3.11}$$

allowing $T=\infty$ or $\gamma=1$ (but not both).

## 3.5 — Policies and Value Functions

A **policy** $\pi$ is a mapping from states to probabilities of selecting
each possible action: $\pi(a\mid s)$.

The **state-value function** $v_\pi(s)$ is the expected return starting
from $s$ and following $\pi$ thereafter:

$$v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] \tag{3.12}$$

The **action-value function** $q_\pi(s,a)$ is the expected return starting
from $s$, taking action $a$, and thereafter following $\pi$:

$$q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a] \tag{3.13}$$

Both can in principle be estimated from experience by averaging observed
returns (**Monte Carlo methods**, covered in Chapter 5), or by fitting a
parameterized approximation when there are too many states to track
individually.

**The Bellman equation for $v_\pi$** is the chapter's central recursive
identity — the value of a state equals the expected immediate reward plus
the discounted value of whatever state comes next, averaged over the
policy's action probabilities and the environment's dynamics:

$$v_\pi(s) = \sum_a \pi(a\mid s) \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma v_\pi(s')\big] \tag{3.14}$$

This is visualized with a **backup diagram**: an open circle (state) branches
into solid circles (state-action pairs) under $\pi$, each of which branches
into possible next states weighted by $p$. $v_\pi$ is the *unique* solution
to its Bellman equation.

**Worked example — Gridworld:** a 5×5 grid with two special states (A, B)
that teleport the agent to A′, B′ with rewards +10, +5 respectively; running
off the grid costs −1. Under the equiprobable random policy, the resulting
$v_\pi$ (solved via the linear system 3.14) shows that A is valued *less*
than its immediate +10 reward (because A′ is edge-adjacent and risky),
while B is valued *more* than +5 (because B′ is more central).

## 3.6 — Optimal Policies and Optimal Value Functions

Value functions define a partial ordering over policies: $\pi \ge \pi'$ iff
$v_\pi(s) \ge v_{\pi'}(s)$ for all states. There is always at least one
**optimal policy** $\pi_*$ (possibly more than one), and all optimal
policies share the same **optimal state-value function**:

$$v_*(s) \doteq \max_\pi v_\pi(s) \tag{3.15}$$

and the same **optimal action-value function**:

$$q_*(s,a) \doteq \max_\pi q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t=s, A_t=a] \tag{3.16, 3.17}$$

Because $v_*$ is the value function for a (optimal) policy, it must satisfy
a self-consistency condition — but since it's optimal, that condition can
be written *without reference to any specific policy*, as a max over
actions instead of an expectation over $\pi$. This is the **Bellman
optimality equation**:

$$v_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma v_*(s')\big] \tag{3.19}$$

$$q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\Big[r+\gamma \max_{a'} q_*(s',a')\Big] \tag{3.20}$$

Backup diagrams for $v_*$/$q_*$ add an explicit **max** arc at the agent's
choice points, in contrast to the plain expectation used for $v_\pi$/$q_\pi$.

**Why $q_*$ is so useful:** once you have it, choosing optimally requires
*no* lookahead or model of the environment at all — just pick
$\arg\max_a q_*(s,a)$. $q_*$ effectively caches the result of a one-step
search for every state, at the cost of representing a function over
state-*action* pairs instead of just states. A policy that is **greedy**
with respect to $v_*$ (or $q_*$) is automatically an optimal policy, because
$v_*$ already accounts for all future reward consequences of each
immediate choice.

**Worked examples:** the gridworld's optimal value function and optimal
policy (multiple optimal actions in several cells), and the recycling
robot's explicit two-equation Bellman optimality system for $v_*(\texttt{high})$
and $v_*(\texttt{low})$ — for any valid choice of the reward/transition
parameters, there is exactly one pair of values solving both equations
simultaneously.

## 3.7 — Optimality and Approximation

Explicitly solving the Bellman optimality equation is rarely useful in
practice — it amounts to an exhaustive search and relies on three
assumptions that are rarely all true: (1) the environment's dynamics are
accurately known, (2) enough computational resources exist to complete the
calculation, and (3) the Markov property holds. Backgammon, with roughly
$10^{20}$ states, would take thousands of years to solve exactly on
present-day hardware even though its dynamics are fully known — computation
is the binding constraint there, not information.

In practice, RL settles for **approximate** solutions, and the *online*
nature of RL creates a useful opportunity: an agent can spend more
approximation effort on frequently-encountered states and less on rare
ones (Tesauro's TD-Gammon plays with exceptional skill despite making poor
decisions in board configurations that almost never arise against real
opponents). This is a key property distinguishing RL from other approaches
to approximately solving MDPs.

## 3.8 — Summary

RL is about learning from interaction to achieve a goal, formalized through
three signals passed between agent and environment: actions (agent's
choices), states (basis for those choices), and rewards (definition of the
goal). When this interaction satisfies the Markov property, it constitutes
a (finite) MDP. The **return** aggregates future reward, with episodic
(undiscounted, natural endpoint) and continuing (discounted, indefinite)
formulations unified under one notation. **Value functions** ($v_\pi$,
$q_\pi$) assign expected return to states/state-action pairs under a given
policy; **optimal value functions** ($v_*$, $q_*$) give the best achievable
expected return, are unique for a given MDP even when multiple optimal
policies exist, and satisfy the **Bellman optimality equations** — special
self-consistency conditions that can, in principle, be solved for the
optimal value functions, from which an optimal policy follows by acting
greedily.

---

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

## Note: connection to the Poisson-CH reading

The Bellman/best-response structure here is worth reading alongside the
Camerer, Ho & Chong cognitive hierarchy equations (see
`ch_equations_reference.py`). $q_\pi(s,a)$ and $Q_k(a_i)$ (the CH-to-RL
rewrite of Equation 5) share the same shape — an expected value computed by
averaging a payoff/reward over a distribution (policy vs. belief-weighted
opponent mixture). The key structural difference: Bellman's $v_*$/$q_*$
bootstrap off a state's *own* future self through *time* (via $S_{t+1}$),
while Poisson-CH's $g_k(h)$ bootstraps off *other agents'* lower reasoning
*levels* — recursion through depth-of-reasoning rather than through time.
Both frameworks solve the "avoid infinite regress" problem, just along
different axes (temporal horizon vs. reasoning depth).
