#!/usr/bin/env python3
"""
papers.json  ->  an Obsidian vault of one markdown note per paper.

Frontmatter carries the queryable fields (tier, thread, status, week, hours);
the body carries the annotation and an empty Notes section for you to write in.
Dataview turns the frontmatter into Notion-style tables.

Usage:  python3 make_vault.py data/papers.json vault/
"""
import json, os, re, sys

def slug(s, n=70):
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s[:n].strip("-") or "untitled"

def esc(s):
    """Quote-safe YAML scalar."""
    if s is None:
        return '""'
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'

def note(p):
    thread = p.get("thread") or {}
    week = p.get("week") or {}
    authors = p.get("authors_raw") or ""
    fname = f"{p.get('year') or 'nd'} {authors.split('&')[0].split(',')[0].strip()} - {p['title']}"

    fm = ["---"]
    fm.append(f"title: {esc(p['title'])}")
    fm.append(f"authors: {esc(authors)}")
    fm.append(f"year: {p.get('year') or ''}")
    fm.append(f"venue: {esc(p.get('venue'))}")
    fm.append(f"tier: {p.get('tier') or ''}")
    fm.append(f"thread: {esc(thread.get('key'))}")
    fm.append(f"thread_label: {esc(thread.get('label'))}")
    fm.append(f"week: {week.get('n') or ''}")
    fm.append(f"week_title: {esc(week.get('title'))}")
    fm.append(f"week_dates: {esc(week.get('dates'))}")
    fm.append(f"est_hours: {p.get('est_hours') or ''}")
    fm.append("status: to-read")          # to-read | reading | skimmed | read
    fm.append("rating: ")
    fm.append("started: ")
    fm.append("finished: ")
    fm.append("tags: [paper]")
    links = p.get("links") or []
    if links:
        fm.append(f"url: {esc(links[0]['url'])}")
    fm.append("---\n")

    b = []
    b.append(f"# {p['title']}\n")
    b.append(f"*{authors} ({p.get('year') or 'n.d.'}).*"
             + (f" {p['venue']}." if p.get("venue") else "") + "\n")

    if links:
        b.append(" · ".join(f"[{l['label']}]({l['url']})" for l in links) + "\n")

    if p.get("why"):
        b.append("## Why it's here\n")
        b.append(p["why"] + "\n")

    for f in p.get("flags") or []:
        b.append(f"> [!warning] {f}\n")

    b.append("## Notes\n")
    b.append("<!-- write here. `## 2026-09-03` headings if you want a running log. -->\n")
    b.append("\n## Connections\n")
    b.append("<!-- [[Other Paper]] — rebuts / extends / same-model-as / method-for -->\n")

    return fname, "\n".join(fm) + "\n".join(b)


DASHBOARD = """---
tags: [dashboard]
---
# Reading dashboard

> [!tip] Requires the **Dataview** community plugin (Settings → Community plugins → Browse → Dataview).

## This week's plan

```dataview
TABLE WITHOUT ID
  file.link AS Paper, tier AS Tier, est_hours AS Hrs, status AS Status
FROM "papers"
WHERE week
SORT week ASC, tier ASC
GROUP BY "Week " + week + "  ·  " + week_dates + "  —  " + week_title
```

## Unread tier-1

```dataview
TABLE WITHOUT ID
  file.link AS Paper, year AS Year, thread AS Thread, est_hours AS Hrs
FROM "papers"
WHERE tier = 1 AND status = "to-read"
SORT year DESC
```

## Everything, by thread

```dataview
TABLE WITHOUT ID
  file.link AS Paper, year AS Year, tier AS Tier, status AS Status
FROM "papers"
WHERE thread
SORT tier ASC, year DESC
GROUP BY thread + " · " + thread_label
```

## Done — with dates and ratings

```dataview
TABLE WITHOUT ID
  file.link AS Paper, finished AS Finished, rating AS Rating, thread AS Thread
FROM "papers"
WHERE status = "read"
SORT finished DESC
```

## Progress

```dataview
TABLE WITHOUT ID
  status AS Status, length(rows) AS Count, sum(rows.est_hours) AS Hours
FROM "papers"
GROUP BY status
SORT length(rows) DESC
```

## Started but not finished

```dataview
LIST
FROM "papers"
WHERE status = "reading"
SORT started ASC
```
"""

README = """# joint-planning-lit

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
"""

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/papers.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "vault"
    papers = json.load(open(src, encoding="utf-8"))["papers"]

    pdir = os.path.join(out, "papers")
    os.makedirs(pdir, exist_ok=True)

    written = skipped = 0
    for p in papers:
        fname, body = note(p)
        path = os.path.join(pdir, slug(fname, 90) + ".md")
        if os.path.exists(path):
            skipped += 1
            continue
        open(path, "w", encoding="utf-8").write(body)
        written += 1

    dash = os.path.join(out, "Reading dashboard.md")
    if not os.path.exists(dash):
        open(dash, "w", encoding="utf-8").write(DASHBOARD)

    open(os.path.join(out, "..", "README.md"), "w", encoding="utf-8").write(README)
    print(f"{written} notes written, {skipped} skipped (already existed) -> {pdir}")
