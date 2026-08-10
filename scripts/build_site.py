#!/usr/bin/env python3
"""
papers.json (+ this file's prose)  ->  index.html

The public, publishable reading list. Regenerate after editing papers.json:

    python3 scripts/build_site.py data/papers.json index.html
"""
import html, json, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))

THREAD_ORDER = list("ABCDEFGH")

THREAD_LEADIN = {
 "G": "Nearly everything in threads A–F is <b>discrete and turn-based</b>: one simultaneous "
      "move, or a step through a gridworld. This thread is <b>continuous-time and spatial</b> — "
      "agents move freely, watch each other move, and commitment emerges over a trajectory "
      "rather than in a single choice. The distinction is not cosmetic. It changes which "
      "equilibrium gets selected, which models are even expressible, and what a task has to "
      "log in order to measure anything.",
 "H": "Adjacent rather than central, and worth being precise about why. This work models "
      "<em>social learning</em> — extracting information from others — rather than <em>joint "
      "planning</em>, reasoning about a partner's plan toward a shared goal. The foraging "
      "paradigms here are congestion problems: peers are simultaneously information sources "
      "and competitors for the same resource, with no payoff-dominant joint equilibrium to "
      "coordinate onto. That mutual-expectation problem is the entire content of a stag hunt. "
      "What this thread does supply, and supplies better than anything in A–G, is the "
      "methodological craft for running and fitting real-time multi-agent tasks.",
}

SIM_NOTE = """
<p>Papers in this list are frequently described as running &ldquo;simulation experiments&rdquo; to
compare models of coordination. The phrase covers at least five distinct activities, and
conflating them is a common source of confusion.</p>

<h3>1 · Forward simulation — do the models even differ?</h3>
<p>Implement three or four competing accounts — level-k / cognitive hierarchy, joint-intention
planning, virtual bargaining, plain best-response — run them as agents in a candidate
environment, and look at what comes out. The first question is basic and constantly skipped:
<em>do these models produce different behavior at all?</em> In many environments, level-2
reasoning and team reasoning prescribe the same action, and the equations alone will not
reveal it.</p>

<h3>2 · Model recovery</h3>
<p>Simulate a dataset from Model A, then fit <em>every</em> model to that synthetic dataset and
see which wins. Repeat with B as generator, then C, then D. The result is a confusion matrix:
generating model × recovered model. If A's data is best fit by B, <b>the models are not
identifiable in that task</b>, and no quantity of human data will fix it. The same exercise at
the parameter level asks whether known parameters can be recovered from the number of trials a
single participant realistically produces.</p>

<h3>3 · Design optimization</h3>
<p>Once models can be simulated and recovered, the environment becomes something to search over.
Sweep layouts, payoff structures, visibility conditions, time pressure; score each candidate by
how far apart the models' predictions fall, or by expected information gain about which model is
true. This is optimal experimental design, and it is why simulation properly precedes task
construction rather than following it.</p>

<h3>4 · Cross-play</h3>
<p>The genuinely multi-agent part. In single-agent modeling, simulating a model means running one
agent; here behavior is a joint product of <em>two</em> policies, which changes the object being
evaluated. What is needed is an N×N pairing matrix — A with A, A with B, B with A. Self-play is
famously misleading: the Overcooked result is precisely that an agent optimized in self-play can
be near-useless with a partner it was not trained against. Human data is <em>always</em>
cross-play, since people are never paired with copies of themselves. A related decision silently
breaks model fits if left implicit: <b>when fitting one participant's choices, what is assumed
about their partner?</b> Fitting each person as though their partner were optimal will
systematically mis-attribute coordination failures to the wrong parameter.</p>

<h3>5 · Power analysis</h3>
<p>Simulate N participants × T trials, run the planned model-comparison analysis on the synthetic
data, and count how often it selects the true model. That curve over N is a sample-size
justification that can be preregistered. It matters more here than in a typical study because
synchronous two-player recruitment is lossy — a meaningful fraction of recruited participants
produce no usable data.</p>

<h3>A second sense of the phrase</h3>
<p>&ldquo;Simulation experiments&rdquo; sometimes instead means agent-based or evolutionary
simulation, where the question is not <em>which model fits humans</em> but <em>which coordination
mechanism is stable, learnable, or evolvable</em>. Kleiman-Weiner et al. (2025, <i>PNAS</i>) is
the current exemplar. That is a legitimate research program in its own right and requires no
human data at all.</p>
"""

EARLY_BANDS = [
 ("Joint planning as a planning problem",
  "Closest to the core of this list: two agents forming and acting on a shared plan, modelled as "
  "planning rather than as equilibrium selection or third-party attribution.",
  ["Nastaran Arfaei", "Tan Zhi-Xuan", "Luke McEllin", "Georgina Török", "Yiling Yun"]),
 ("Building and instrumenting real-time multi-agent tasks",
  "The methodological half. Synchronous multiplayer at scale, continuous trajectories, what a task "
  "has to log, and the tooling for writing recursive partner models.",
  ["Veronica Boyce", "Chase McDonald", "Cecilia De Vicariis", "Thomas Wolf", "Kartik Chandra"]),
 ("Human-facing multi-agent AI",
  "Coordination with partners who were not trained against you — the cross-play problem, and how to "
  "evaluate it against real people rather than proxies.",
  ["Yancheng Liang", "Bidipta Sarkar", "Tin Dizdarević", "Dhara Yu"]),
 ("Interacting brains",
  "The neural side of coordination: dual-brain and intracranial approaches to two agents acting "
  "together, and the individual differences that modulate it.",
  ["Martina Fanghella", "Qianying Wu", "Setayesh Radkani"]),
 ("Neighbouring theory",
  "Adjacent problems that share the machinery — legibility, teaching a bounded partner, role "
  "allocation, and commitment at population scale.",
  ["Amanda Royka", "Huang Ham", "Max E. Potter", "Julien Lie-Panis"]),
]

EARLY = [
 # ── closest to joint planning proper ──
 ("Nastaran Arfaei", "NYU Psychology, w/ Wei Ji Ma",
  "<span class='faint'>ccneuro.org</span>",
  "<b>E</b> · Collaborative planning over long horizons; heuristic tree search that simulates a "
  "partner's next move using the same value function",
  "The closest person found to joint planning treated as a <em>planning</em> problem rather than a "
  "game or an attribution. Also the nearest one geographically to anyone at NYU."),
 ("Tan Zhi-Xuan", "Assistant Professor, National University of Singapore (from Aug 2025) · MIT PhD w/ Mansinghka &amp; Tenenbaum",
  "<a href='https://ztangent.github.io/' target='_blank' rel='noopener'>ztangent.github.io</a>",
  "<b>C · F</b> · Inverse planning over multi-agent goals, cooperative language-guided assistance, "
  "norm induction in Markov games, probabilistic programming",
  "Newly independent — runs the Cooperative Intelligence &amp; Systems lab. Thesis was literally "
  "<i>Scaling Cooperative Intelligence via Inverse Planning and Probabilistic Programming</i>, and "
  "she connects the MIT inference machinery to virtual bargaining via the Levine collaboration."),
 ("Luke McEllin", "Postdoc, Social Mind Center, Central European University",
  "<a href='https://somby.ceu.edu/luke-mcellin' target='_blank' rel='noopener'>somby.ceu.edu</a>",
  "<b>D</b> · Coordination as a generator of commitment; sensorimotor communication, trust and "
  "generosity; synchrony and public-goods contributions",
  "The best-positioned early-career person bridging the Sebanz/Knoblich joint-action tradition and "
  "the Michael commitment tradition, with an economic-games extension."),
 ("Thomas Wolf", "Postdoc, Social Mind Center, Central European University",
  "<a href='https://somby.ceu.edu/thomas-wolf' target='_blank' rel='noopener'>somby.ceu.edu</a>",
  "<b>G</b> · Real-time temporal coordination; expert–novice prediction asymmetries; joint rushing; "
  "work songs and musical improvisation",
  "The strongest early-career person on the continuous/real-time side, and his work on "
  "&ldquo;dissensus&rdquo; is a useful counterweight to the field's synchrony-is-good default."),
 ("Georgina Török", "TU Munich (position unconfirmed) · CEU PhD w/ Sebanz &amp; Csibra",
  "<a href='https://somby.ceu.edu/georgina-torok' target='_blank' rel='noopener'>somby.ceu.edu</a>",
  "<b>E</b> · &ldquo;Coefficiency&rdquo; — co-actors minimizing aggregate dyadic cost; computing "
  "joint action costs; efficiency versus fairness in distributing joint actions",
  "The coefficiency result is the strongest empirical rationality principle for joint plans from "
  "the psychology side. Recent output is thin and the current post is unconfirmed."),
 # ── computational ToM & cooperative communication ──
 ("Kartik Chandra", "MIT CSAIL / CoCoSci / Saxelab, w/ Ragan-Kelley &amp; Tenenbaum",
  "<a href='https://cs.stanford.edu/~kach/' target='_blank' rel='noopener'>kach</a>",
  "<b>C</b> · Cooperative explanation as rational communication; storytelling as inverse inverse "
  "planning; <code>memo</code>, a probabilistic language for reasoning about reasoning",
  "Acting <em>so as to be understood</em> — the communication-side instance of joint planning. "
  "<code>memo</code> is separately the most practical piece of infrastructure on this list."),
 ("Amanda Royka", "PhD candidate, Yale, w/ Jara-Ettinger &amp; Santos",
  "<a href='https://amandaroyka.github.io/' target='_blank' rel='noopener'>amandaroyka.github.io</a>",
  "<b>C</b> · Legibility and ostension — how agents make their own behaviour intelligible; "
  "signalling intentions through efficient action; comparative and cross-cultural work",
  "The legibility half of joint action: modifying one's own plan so a partner can recover the "
  "intention behind it. Same recursive machinery as coordination, one agent at a time."),
 ("Yiling Yun", "PhD candidate, UCLA (Tao Gao's orbit)",
  "<span class='faint'>arXiv:2504.21224</span>",
  "<b>C · B</b> · Joint utility versus pragmatic reasoning in cooperative communication; "
  "ambiguous signal choice in cooperative gridworlds",
  "Finds joint utility explains human signal choice better than standard RSA pragmatics — which is "
  "the seam between communication and joint planning. One paper so far; a bet rather than a fixture."),
 ("Huang Ham", "PhD candidate, Princeton, w/ Vélez &amp; Griffiths",
  "<a href='https://huangham.github.io/' target='_blank' rel='noopener'>huangham.github.io</a>",
  "<b>H · E</b> · Higher-order epistemic reasoning; teaching against a model of a <em>bounded</em> "
  "partner; how collective representations emerge from individual minds",
  "The epistemic-reasoning work models nested &ldquo;I know that you know&rdquo; of the kind "
  "coordination games require; the teaching work is planning against a resource-limited partner."),
 # ── human-facing multi-agent AI ──
 ("Dhara Yu", "PhD student, UC Berkeley, w/ Bill Thompson · Cooperative AI Fellow",
  "<a href='https://dharakyu.com/' target='_blank' rel='noopener'>dharakyu.com</a>",
  "<b>D · H</b> · Two-player games with human participants: how pairs trade off joint reward, "
  "fairness and rule complexity; adaptive vagueness; emergent communication",
  "Rare combination — runs human behavioural experiments <em>and</em> multi-agent simulation on the "
  "same questions."),
 ("Yancheng Liang", "PhD student, University of Washington, w/ Du &amp; Jaques",
  "<a href='https://liangyancheng.com/' target='_blank' rel='noopener'>liangyancheng.com</a>",
  "<b>F</b> · Generative models of human partner strategies for human-AI coordination; adversarial "
  "training for robust cooperation",
  "Evaluates with real human teammates rather than proxies, which is less common in this literature "
  "than it should be. The direct successor line to the Overcooked result."),
 ("Bidipta Sarkar", "DPhil student, Oxford FLAIR, w/ Foerster &amp; Whiteson",
  "<a href='https://bsarkar321.github.io/' target='_blank' rel='noopener'>bsarkar321.github.io</a>",
  "<b>F</b> · Diverse conventions for human-AI collaboration; emergent discussion in social-deduction "
  "games",
  "Generating <em>multiple</em> self-play conventions so an agent can recognize whichever one a human "
  "partner happens to use — convention selection as an explicit design problem."),
 ("Tin Dizdarević", "Oxford FLAIR, w/ Foerster",
  "<span class='faint'>docs.ah2ac2.com</span>",
  "<b>F</b> · Reproducible human-facing evaluation of coordination; human-proxy partners distilled "
  "from a large Hanabi corpus",
  "Worth knowing for the methodological argument about how human-facing coordination should be "
  "measured, not only for the benchmark."),
 # ── methods & infrastructure for real-time multiplayer ──
 ("Veronica Boyce", "NSF Postdoctoral Fellow, MIT, w/ Roger Levy · Stanford PhD w/ Frank",
  "<a href='https://vboyce.github.io/' target='_blank' rel='noopener'>vboyce.github.io</a>",
  "<b>H</b> · Large synchronous online multiplayer reference games; how interaction structure "
  "constrains whether conventions emerge; Refbank",
  "The best living example of running synchronous multiplayer behavioural experiments at scale, and "
  "of aggregating them for meta-analysis."),
 ("Chase McDonald", "Riot Games · CMU PhD w/ Coty Gonzalez (DDMLab)",
  "<a href='https://chasemcd.com/' target='_blank' rel='noopener'>chasemcd.com</a>",
  "<b>G · F</b> · Real-time browser-based human-AI experiments; subjective as well as objective "
  "measures of complementarity; builds Interactive Gym, CoGrid, MUG",
  "Has left academia, but the tooling is live and derived from the Overcooked demo — the closest "
  "thing to an off-the-shelf harness for real-time human-agent studies."),
 ("Cecilia De Vicariis", "Università di Genova, w/ Sanguineti",
  "<span class='faint'>PLOS Comp Biol 2024</span>",
  "<b>G</b> · Physically coupled human dyads in continuous joint reaching, modelled as a "
  "differential game",
  "One of very few people giving continuous-time coordination with real humans an explicit "
  "game-theoretic treatment. Bioengineering edge rather than the ML edge."),
 # ── neighbouring, worth knowing ──
 ("Martina Fanghella", "Postdoc &amp; Co-PI, Cognition in Action Unit, Milan, w/ Michael &amp; Sinigaglia",
  "<span class='faint'>unimi.it</span>",
  "<b>D</b> · Dual-EEG contrasts of joint versus side-by-side action; the psychophysiology of "
  "commitment",
  "The person most directly welding the commitment tradition to dual-brain electrophysiology — the "
  "seam between the behavioural and neural strands."),
 ("Setayesh Radkani", "PhD candidate, MIT Saxelab, w/ Saxe",
  "<span class='faint'>saxelab.mit.edu</span>",
  "<b>D</b> · Punishment as rational communicative action; what people learn from being punished; "
  "perceived legitimacy of authority",
  "An explicit inverse-planning model of acting to change another agent's beliefs about norms, with "
  "a coordination-failure result. Authority-to-observer rather than co-planners."),
 ("Max E. Potter", "PhD student, UC San Diego, w/ Rossano",
  "<span class='faint'>cogsci.ucsd.edu</span>",
  "<b>D</b> · Coordinative difficulty as a driver of role-taking in group tasks",
  "Role differentiation as a response to coordination cost is exactly the phenomenon a joint-planning "
  "model has to predict. Comparative/behavioural tradition, so the phenomenon travels, not the model."),
 ("Julien Lie-Panis", "Postdoc, joining SMILE at ENS (Sept 2026)",
  "<a href='https://jliep.github.io/' target='_blank' rel='noopener'>jliep.github.io</a>",
  "<b>D</b> · Institutions as commitment devices; why moral rules are rigid; cooperation as a signal "
  "of time preferences",
  "The population-scale analogue of commitment — evolutionary game theory rather than planning "
  "representations, but the rigidity-of-rules work is directly usable."),
 ("Qianying Wu", "PhD candidate, Caltech, w/ Adolphs &amp; O'Doherty",
  "<a href='https://wuqy052.github.io/' target='_blank' rel='noopener'>wuqy052.github.io</a>",
  "<b>—</b> · Social attention and social learning; goal inference from observation; individual "
  "differences and autistic traits; eye-tracking, behavioural modelling, fMRI",
  "Adjacent rather than central: goal inference and social learning rather than joint planning. The "
  "individual-differences and computational-psychiatry angle is the closer fit."),
]

GROUPS = [
 ("Nick Chater", "Warwick",
  "Julia Misyak · Tigran Melkonyan · Hossam Zeitoun · Arthur Le Pargneux",
  "Virtual bargaining, tacit commitments, instantaneous conventions",
  "The longest-running formal program on tacitly agreed joint plans, and still active."),
 ("Fiery Cushman", "Harvard",
  "Sydney Levine · Arthur Le Pargneux · Diego Trujillo",
  "Resource-rational contractualism; bargaining power in moral judgment",
  "Where the bargaining tradition and resource-rationality have already met. Le Pargneux's own "
  "output is the nearest published work to joint planning proper; his page runs ahead of the databases."),
 ("Max Kleiman-Weiner", "University of Washington",
  "Sarah A. Wu · Kunal Jha",
  "Joint intentions, Bayesian delegation, evolving cooperation, minds-as-code",
  "The through-line of computational joint planning from 2016 to the present runs through here."),
 ("Robert Hawkins", "Stanford",
  "Jinyi Kuang · Erik Brockbank",
  "Real-time convention formation; commitment inference; continuous-time stag hunt",
  "Three of the most on-point items on this list. The 2016 PLOS ONE paper is the methodological "
  "precedent for treating movement trajectories as coordination data."),
 ("Tao Gao", "UCLA",
  "Ning Tang · Stephanie Stacy · Minglu Zhao · Siyi Gong · Aishni Parab",
  "&ldquo;Imagined We,&rdquo; joint commitment, cooperative hunting, psychophysics of animacy",
  "The formalization of shared intentionality currently winning model comparisons in "
  "continuous-time coordination — and the one place cooperation and communication are derived "
  "from a single framework."),
 ("John Michael", "Central European University / Warwick",
  "Francesca Bonalumi",
  "The sense of commitment; commitment cues; coordination as a source of commitment",
  "The commitment literature proper. Mostly descriptive rather than formal, which is precisely "
  "the gap computational accounts have started to fill."),
 ("Charley Wu", "TU Darmstadt (from Tübingen, 2025)",
  "Charley M. Wu · Dominik Deffner · Alexandra Witt",
  "Real-time multi-agent foraging, social learning strategies, GP generalization",
  "Unusual depth of experience running real-time spatial multi-agent tasks <em>with a fitted "
  "moment-to-moment choice model</em> rather than aggregate outcomes."),
 ("Natalia Vélez", "UC San Diego (from Princeton, 2026)",
  "Elise Mieczkowski · Ross Mon-Williams · Yang Xiang",
  "Division of labor, collective intelligence, competence and effort inference",
  "Mostly one abstraction level above dyadic coordination; the Griffiths-lab multi-agent RL "
  "papers are the directly applicable part."),
 ("Julian Jara-Ettinger", "Yale",
  "Zoe Wang · Ilona Davis",
  "Naïve utility calculus; &ldquo;restricted scope models&rdquo;",
  "The social-cognition analogue of simplified task representations in planning. The overlap "
  "is largely unclaimed."),
 ("Joshua Tenenbaum", "MIT CoCoSci",
  "Tan Zhi-Xuan · Lance Ying · Katherine Collins · Kelsey Allen",
  "Bayesian ToM, cooperative language-guided inverse planning, norm induction",
  "Where much of the inference machinery gets built."),
 ("Robert Sugden · Andrew Colman · Natalie Gold", "UEA, Leicester",
  "Nicholas Bardsley · Andrea Isoni",
  "Team reasoning, focal points, the Bacharach line",
  "Economics has been arguing about &ldquo;we-mode&rdquo; since the 1990s. A good deal of what "
  "reads as new in cognitive science is a rediscovery."),
 ("Michael Tomasello", "Duke / MPI EVA",
  "Maria Gräfenhain · Felix Warneken · Barbara Siposova",
  "Shared intentionality; developmental and comparative evidence on joint commitment",
  "The developmental work is where commitment is manipulated rather than described — which "
  "makes it more useful to a modeler than the framework papers."),
 ("Natalie Sebanz &amp; Günther Knoblich", "Central European University",
  "Cordula Vesper",
  "The empirical joint-action programme",
  "A separate tradition — sensorimotor, real-time, embodied — whose citation graph barely "
  "touches the modeling one."),
 ("Jakob Foerster · Cooperative AI Foundation", "Oxford FLAIR",
  "Hengyuan Hu · Johannes Treutlein · Nolan Bard",
  "Zero-shot coordination, other-play, Hanabi, emergent communication",
  "The AI-side audience: different venues, different standards of evidence."),
 ("Pascal Vrtička · Stefanie Hoehl", "Essex, Vienna",
  "Sara De Felice",
  "Hyperscanning and dual-brain approaches to real-time interaction",
  "The neural counterpart to everything in thread G, and now engaging the "
  "mechanism-versus-epiphenomenon question rather than cataloguing synchrony."),
]

FOOTER = """

<p>Roughly half these entries are CogSci proceedings with no DOI — worth knowing before building
anything that keys on one. Citations were verified against publisher pages, author-hosted PDFs,
arXiv/OpenReview, or lab publication lists; errors that remain are the compiler's.</p>

<p>Annotations are editorial judgments, not consensus positions. Corrections and additions
welcome.</p>
"""

def yr(p):
    """First 4-digit year in the field; 0 if absent (handles '2024/2026')."""
    m = re.search(r"\d{4}", str(p.get("year") or ""))
    return int(m.group()) if m else 0

def esc(s):
    return html.escape(s or "", quote=False)

def tier_tag(t):
    if t == 1:
        return '<span class="tag t1">Tier 1</span>'
    if t == 2:
        return '<span class="tag t2">Tier 2</span>'
    if t == 3:
        return '<span class="tag">Tier 3</span>'
    return ""

def entry(p, li_class="lib"):
    bits = [f'<div class="{li_class}">']
    bits.append(f'<p class="cite">{p["citation_html"]}</p>')
    if p.get("why"):
        bits.append(f'<p class="why">{esc(p["why"])}</p>')
    meta = [tier_tag(p.get("tier"))]
    if p.get("est_hours"):
        h = p["est_hours"]
        meta.append(f'<span class="tag hrs">{h:g} h</span>')
    for l in p.get("links") or []:
        meta.append(f'<a class="tag" href="{esc(l["url"])}" target="_blank" rel="noopener">'
                    f'{esc(l["label"])}</a>')
    for f in p.get("flags") or []:
        meta.append(f'<span class="tag warn">{esc(f)}</span>')
    bits.append('<div class="meta">' + "".join(m for m in meta if m) + "</div>")
    bits.append("</div>")
    return "\n".join(bits)

def build(papers, style, meta=None):
    meta = meta or {}
    by_thread = collections.defaultdict(list)
    weeks = collections.defaultdict(list)
    for p in papers:
        t = (p.get("thread") or {}).get("key")
        if t:
            by_thread[t].append(p)
        w = (p.get("week") or {}).get("n")
        if w:
            weeks[w].append(p)

    n_t1 = sum(1 for p in papers if p.get("tier") == 1)

    o = ['<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width, initial-scale=1">',
         '<title>Joint planning &amp; coordination — an annotated reading list</title>',
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;'
         '0,9..40,500;0,9..40,700;1,9..40,400;1,9..40,500&family=Caveat:wght@400;600&display=swap" '
         'rel="stylesheet">',
         style, '</head><body><div class="wrap">']

    o.append('<header class="top">')
    o.append('<p class="eyebrow label">annotated reading list · '
             f'{len(papers)} papers · 8 threads · v{meta.get("version","1.0")} · '
             f'updated {meta.get("updated","")}</p>')
    o.append('<h1>joint planning &amp;<br>coordination</h1>')
    o.append('<p class="dek">Where computational cognitive science, behavioral game theory, and '
             'multi-agent AI meet on the problem of two or more agents forming and acting on a '
             'shared plan. Every entry carries a tier, a note on why it earns its place, and — '
             'where relevant — a flag for a common citation trap.</p>')
    o.append('</header>')

    o.append('<div class="taped"><span class="label">how to read the tiers</span>')
    o.append(f'<p><b>Tier 1</b> ({n_t1} papers) — read in full. <b>Tier 2</b> — skim the '
             'introduction and discussion. <b>Tier 3</b> — know it exists, cite when relevant. '
             'Tiers reflect value to someone building computational models of coordination, not '
             'the importance of the work in its own field; several Tier 3 entries are classics '
             'that simply are not the fastest route in.</p>')
    o.append('<p>The threads are not mutually exclusive and the boundaries are arguable — '
             'thread <b>G</b> in particular cuts across <b>A</b> and <b>D</b> on the axis of '
             'continuous versus discrete time, which turns out to matter more than the '
             'disciplinary split.</p></div>')

    if weeks:
        o.append('<div class="sec"><span class="num">01</span><h2>a four-week path</h2></div>')
        o.append('<p class="sec-note">Sixteen of the papers below, sequenced as roughly four '
                 'hours a week for a month. The social and joint-planning material comes first '
                 'deliberately: the single-agent resource-rational work is easier to absorb '
                 'later, whereas these shape which questions look worth asking.</p>')
        for n in sorted(weeks):
            ps = weeks[n]
            wt = (ps[0].get("week") or {}).get("title") or ""
            hrs = sum(p.get("est_hours") or 0 for p in ps)
            o.append('<div class="week"><div class="week-head">'
                     f'<span class="week-num">{n:02d}</span>'
                     f'<span class="week-title">{esc(wt)}</span>'
                     f'<span class="count label">{len(ps)} papers · ~{hrs:g} h</span></div>')
            for p in sorted(ps, key=lambda x: (x.get("tier") or 9)):
                o.append(entry(p))
            o.append('</div>')

    o.append('<div class="sec"><span class="num">02</span><h2>the library, by thread</h2></div>')
    for k in THREAD_ORDER:
        ps = by_thread.get(k)
        if not ps:
            continue
        label = (ps[0].get("thread") or {}).get("label") or k
        open_attr = " open" if k in ("G", "H") else ""
        o.append(f'<details class="thread"{open_attr}><summary>{esc(label)}'
                 f'<span class="cnt">{len(ps)} items</span></summary><div class="inner">')
        if k in THREAD_LEADIN:
            o.append(f'<p class="lead-in">{THREAD_LEADIN[k]}</p>')
        for p in sorted(ps, key=lambda x: (x.get("tier") or 9, -yr(x))):
            o.append(entry(p))
        o.append('</div></details>')

    o.append('<div class="sec"><span class="num">03</span>'
             '<h2>on &ldquo;simulation experiments&rdquo;</h2></div>')
    o.append(SIM_NOTE)

    o.append('<div class="sec"><span class="num">04</span><h2>where this work lives</h2></div>')
    o.append('<p class="sec-note">Groups whose output this list draws on most heavily. Senior authors anchor the columns because that is how the areas are usually named, but the second column is the more useful one for alerts — first authors are who is actually producing the work, and several are on the job market or moving.</p>')
    o.append('<table><thead><tr><th>senior author / group</th>'
             '<th>first authors to follow</th><th>focus</th><th>note</th></tr></thead><tbody>')
    for name, where, firsts, focus, note in GROUPS:
        o.append(f'<tr><td><b>{name}</b><br><span class="faint">{where}</span></td>'
                 f'<td>{firsts}</td><td>{focus}</td><td>{note}</td></tr>')
    o.append('</tbody></table>')

    o.append('<h3>Early career — worth following directly</h3>')
    o.append('<p class="sec-note">People without their own labs yet, whose output is close enough '
             'to this list to be worth watching in its own right rather than through a senior '
             'author. Grouped by what they contribute, most central first; relevance is stated '
             'rather than assumed.</p>')
    by_name = {e[0]: e for e in EARLY}
    placed = set()
    for band, blurb, names in EARLY_BANDS:
        o.append(f'<h4 style="margin:30px 0 4px">{band}</h4>')
        o.append(f'<p class="sec-note" style="margin-bottom:10px">{blurb}</p>')
        o.append('<table><thead><tr><th>name</th><th>where</th><th>focus</th>'
                 '<th>relevance</th></tr></thead><tbody>')
        for n in names:
            if n not in by_name:
                continue
            name, where, site, focus, note = by_name[n]
            placed.add(n)
            o.append(f'<tr><td><b>{name}</b><br><span class="faint">{site}</span></td>'
                     f'<td class="faint">{where}</td><td>{focus}</td><td>{note}</td></tr>')
        o.append('</tbody></table>')
    leftover = [e for e in EARLY if e[0] not in placed]
    if leftover:
        o.append('<h4 style="margin:30px 0 10px">Unsorted</h4>')
        o.append('<table><tbody>')
        for name, where, site, focus, note in leftover:
            o.append(f'<tr><td><b>{name}</b><br><span class="faint">{site}</span></td>'
                     f'<td class="faint">{where}</td><td>{focus}</td><td>{note}</td></tr>')
        o.append('</tbody></table>')

    o.append('<div class="foot">'
             f'<p><b>Version {meta.get("version","1.0")}</b>, last updated '
             f'{meta.get("updated","")}. {len(papers)} entries.</p>'
             f'{FOOTER}</div>')
    o.append('</div></body></html>')
    return "\n".join(o)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/papers.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "index.html"
    style_path = os.path.join(HERE, "style.html")
    style = open(style_path, encoding="utf-8").read()
    data = json.load(open(src, encoding="utf-8"))
    papers = data["papers"]
    out = build(papers, style, data)
    open(dst, "w", encoding="utf-8").write(out)
    print(f"{len(papers)} papers -> {dst} ({len(out)//1024} KB)")
