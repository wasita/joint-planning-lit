---
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
