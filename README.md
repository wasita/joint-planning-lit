# joint-planning-lit

A curated, annotated reading list on **joint planning and multi-agent coordination** —
where computational cognitive science, behavioral game theory, and multi-agent AI meet.

123 papers across eight threads, each with a tier (read in full / skim / know it exists)
and a note on why it earns its place.

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

`data/papers.json` is the machine-readable source of record — 123 records with title,
authors, year, venue, tier, thread, week, annotation, links, and flags.

```bash
python3 scripts/make_vault.py data/papers.json vault/   # regenerate; skips existing notes
```

## Caveats

Annotations are editorial judgments, not consensus positions — they reflect one reading
of the literature and are meant to be argued with. Corrections welcome.

Roughly half these papers are CogSci proceedings with no DOI, which is worth knowing
before building anything that keys on one.

## Files

- `vault/` — Obsidian vault: `papers/` + `Reading dashboard.md`
- `data/papers.json` — machine-readable source of record
- `scripts/make_vault.py` — regenerates the vault from the JSON
