import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", sql_output="polars")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    return mo, np, plt


@app.cell(hide_code=True)
def eq_md(mo):
    mo.md(r"""
    ## Step 0: the non-strategic baseline

    $$P_0\left(s_i^j\right) = \frac{1}{m_i} \quad \forall j$$
    """)
    return


@app.cell(hide_code=True)
def explain_md(mo):
    mo.md(r"""
    ### Reading the equation

    **Input:** a strategy index $j$ -- one specific option out of everything player $i$ could choose.

    **Output:** a probability -- how likely a step-0 player is to pick that particular strategy.

    **Meaning:** step 0 represents *not thinking about others at all*. A step-0 player
    isn't reasoning about payoffs, opponents' likely moves, or the structure of the
    game -- they simply pick uniformly at random among whatever options they have.
    It's the model's placeholder for zero strategic sophistication, and every level
    built on top of it (step 1, step 2, ...) is defined *relative to* this baseline.

    **Term by term:**

    - $P_0(\cdot)$ -- the step-0 *decision rule*: a function mapping a strategy to
      the probability a step-0 player assigns it.
    - $s_i^j$ -- player $i$'s $j$-th strategy: one specific action out of the full
      set of options available to player $i$.
    - $m_i$ -- the total number of strategies player $i$ has to choose from.
    - $\dfrac{1}{m_i}$ -- the probability assigned to *that* strategy: with no
      reasoning to favor any option, probability mass is split evenly across all
      of them.
    - $\forall j$ -- "for every $j$": this holds for *every* strategy in the set,
      not just one -- which is exactly what makes the distribution uniform rather
      than concentrated on some subset.
    """)
    return


@app.cell
def m_slider_cell(mo):
    m_slider = mo.ui.slider(2, 20, value=5, step=1, label="$m_i$ (number of strategies)", show_value=True)
    m_slider
    return (m_slider,)


@app.cell
def uniform_plot(m_slider, mo, np, plt):
    m_i = m_slider.value
    strategies = np.arange(1, m_i + 1)
    probs = np.full(m_i, 1 / m_i)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(strategies, probs, color="#4a6fa5")
    ax.axhline(1 / m_i, color="#c0392b", ls="--", lw=1, label=f"$1/m_i$ = {1 / m_i:.3f}")
    ax.set_ylim(0, 1)
    ax.set_xticks(strategies)
    ax.set_xlabel("strategy $j$")
    ax.set_ylabel("$P_0(s_i^j)$")
    ax.set_title(f"step-0 (uniform) distribution over $m_i$ = {m_i} strategies")
    ax.legend(fontsize=8)
    plt.close(fig)

    mo.vstack([
        fig,
        mo.md(
            f"Every one of the {m_i} strategies gets the same probability: "
            f"**{1 / m_i:.4f}**. Move the slider and watch each bar shrink as "
            f"$m_i$ grows -- the bars always sum to 1."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
