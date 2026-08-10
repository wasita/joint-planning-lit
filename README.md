# joint-planning-lit

A curated, annotated reading list on **joint planning and multi-agent coordination** —
where computational cognitive science, behavioral game theory, and multi-agent AI meet.

87 papers across eight threads, each with a tier (read in full / skim / know it exists),
an annotation on why it matters, and flags for common citation traps.

## Threads

| | |
|---|---|
| **A** | behavioral game theory & coordination — focal points, level-k, team reasoning |
| **B** | virtual bargaining & the current synthesis frontier |
| **C** | computational theory of mind & inverse planning |
| **D** | joint action, shared agency, commitment, norms |
| **E** | resource-rational & hierarchical planning |
| **F** | multi-agent AI & cooperative AI |
| **G** | continuous-time & real-time spatial coordination |
| **H** | collective intelligence & social learning |

A four-week reading plan (~4 hrs/week) picks 16 of them as a spine.

## Read it in a browser

`index.html` is the whole list as a single self-contained page — threads, tiers,
annotations, flags, the four-week path, and a note on what "simulation experiments" means
in this literature. Open it locally, or serve it with GitHub Pages
(Settings → Pages → deploy from `main`, root).

Regenerate after editing the data:

```bash
python3 scripts/build_site.py data/papers.json index.html
```

## Use it as an Obsidian vault

1. Install [Obsidian](https://obsidian.md) (free).
2. **Open folder as vault** → point it at `vault/`.
3. Settings → Community plugins → Browse → install and enable **Dataview**.
4. Open `Reading dashboard.md`.

Notes are plain markdown on disk. Track state by editing frontmatter:

```yaml
status: to-read | reading | skimmed | read
rating: 1-5
started: 2026-09-02
finished: 2026-09-03
tags: [paper, stag-hunt, bayesian-tom]
```

The dashboard tables update automatically. Write under `## Notes`. Link papers with
`[[Wikilinks]]` under `## Connections` and Obsidian's graph view (⌘G) draws the network.

## Use it as data

`data/papers.json` is the machine-readable source of record — 87 records with title,
authors, year, venue, tier, thread, week, annotation, links, and flags.

```bash
python3 scripts/make_vault.py data/papers.json vault/   # regenerate; skips existing notes
```

## Caveats

Annotations are editorial judgments, not consensus positions — they reflect one reading
of the literature and are meant to be argued with. Several entries carry explicit flags
where a paper is commonly miscited, where a venue or author list differs across versions,
or where a result is weaker than its abstract suggests. Corrections welcome.

Roughly half these papers are CogSci proceedings with no DOI, which is worth knowing
before building anything that keys on one.

## Files

- `index.html` — the reading list as a single self-contained page
- `vault/` — Obsidian vault: `papers/` + `Reading dashboard.md`
- `data/papers.json` — machine-readable source of record
- `scripts/make_vault.py` — regenerates the vault from the JSON
- `scripts/build_site.py` — regenerates `index.html` from the JSON
