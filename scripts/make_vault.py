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
    b.append("<!-- write here. `## 2026-08-14` headings if you want a running log. -->\n")
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
GROUP BY "Week " + week + " — " + week_title
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

Reading database for the joint-planning / multi-agent coordination literature.

## Open it

1. Install [Obsidian](https://obsidian.md) (free).
2. **Open folder as vault** → point it at this `vault/` directory.
3. Settings → Community plugins → Browse → install and enable **Dataview**.
4. Open `Reading dashboard.md`.

That's the whole setup. Notes are plain markdown on disk, versioned in this git repo.

## Using it

Each paper is one note in `papers/`. Edit the frontmatter to track state:

```yaml
status: to-read | reading | skimmed | read
rating: 1-5
started: 2026-08-05
finished: 2026-08-06
tags: [paper, stag-hunt, bayesian-tom]
```

The dashboard tables update automatically. Write freely under `## Notes`.

Link papers to each other with `[[Wikilinks]]` under `## Connections` — Obsidian's graph
view (⌘G) then shows the citation/argument network for free.

## Regenerating

`data/papers.json` is the seed, extracted from `reading-guide.html`:

```bash
python3 scripts/extract_papers.py reading-guide.html data/papers.json
python3 scripts/make_vault.py data/papers.json vault/     # will NOT overwrite existing notes
```

`make_vault.py` skips any note that already exists, so re-running is safe once you've
started writing.

## Files

- `reading-guide.html` — the curated guide (the readable artifact)
- `vault/` — the Obsidian vault: `papers/` + `Reading dashboard.md`
- `data/papers.json` — machine-readable seed, 87 records
- `scripts/` — extractor and vault generator
- `PLAN.md` — architecture notes for a custom web app, if this ever outgrows Obsidian
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
