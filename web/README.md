# xwalk workbench (`/app`)

The React source of the workbench that `xwalk ui` serves at **`/app`**, next to the
classic explorer at `/`. It started from the *xwalk Design System* (Claude Design handoff,
14 Oct 2026) and was then redesigned after a reference mock-up (the "Cartographer's
Workbench"): Runs, Results as a source ⟶ target lineage with an inspector drawer, and the
Review queue, plus the Export and Resume dialogs.

## Visual language

- **Archival neutrals, one signal.** Canvas `#F5F5F2`, cells `#FAF9F6`, header strips
  `#EFECE6`, ink `#1A1A17`; signal orange `#EE6327` only for the selected strand, focus,
  filters and the single *Save and next*. Hairline grids instead of cards and shadows; 2px
  corners.
- **Marks, not pills.** Model output is a 6px square (`■ NEEDS REVIEW`); a human decision is
  a blue diamond (`◆ REPLACED`), so the two layers never read alike. Run state sits in a
  hairline box and blinks a signal square while running.
- **Mono capitals for structure** (`SOURCE // 4 RECORDS`, column heads, buttons), sans for
  record text people read.
- **The strand.** Results draws each source to its target: solid ink (matched), dashed
  amber (needs_review), cut at `∅` (unmatched), broken with `×` (failed), dotted
  (pending), blue (a reviewer's decision). Hovering one dims the rest. Review draws the same
  strand between the source and the proposal and redraws it as you choose a decision.
- **Motion with a reason, all off under reduced motion:** rows draw in once, strands wipe
  from source to target, a refused save shakes the decision bar, a changed result strobes
  the record header, a recorded decision flashes its queue row.
- **The CLI beside the click.** Where the page does something the CLI can, it shows the
  exact command (`$ xwalk explain --run … --full s3`), copyable. Only real commands.

The build is committed under `src/xwalk/ui/static/app/`, so `pip install xwalk` never
needs Node. CI (`web` job) rebuilds it and fails when the committed files differ.

## Working on it

```bash
cd web
npm ci
npm test              # vitest: components and app logic
npm run typecheck
npm run build         # writes ../src/xwalk/ui/static/app — commit the result
```

Live reload against a running server (the dev page has no token, so the proxy adds it):

```bash
xwalk ui --root ~/mappings --no-browser            # port 8765
export XWALK_TOKEN=$(curl -s localhost:8765/ | sed -n 's/.*name="xwalk-token" content="\([^"]*\)".*/\1/p')
npm run dev                                         # http://localhost:5173/app/
```

The page runs under the same policy as the classic one (`default-src 'self'`, no inline
script or style elements, no third-party hosts): fonts and icons are bundled, styles are
CSS modules, and runtime-computed values go through React's `style` prop (CSSOM, which
the policy allows).

## Layout

| Path | What |
|---|---|
| `src/styles/tokens/` | The design system's tokens, copied verbatim (`colors`, `typography`, `spacing`, `effects`, `base`). `fonts.css` bundles Geist and Geist Mono (OFL, `@fontsource`) instead of loading Google Fonts. |
| `src/components/` | 32 primitives (the design system's 30, plus `CommandLine` and `Drawer`), grouped: `core/` Button, IconButton, Icon, Kbd, Tooltip · `status/` StatusBadge, RunStateBadge, ReviewBadge, Tag · `data/` DataTable, Identifier, ScoreMeter, KeyValue, FilterChip, CommandLine · `forms/` Input, Textarea, Select, Checkbox, Switch, SegmentedControl, DecisionPicker · `navigation/` Tabs, SidebarNav · `feedback/` Banner, EmptyState, Skeleton, Toast · `overlays/` Dialog, Inspector, Drawer. |
| `src/api/` | A typed client over `xwalk.ui.api`. Field names are the backend's own. |
| `src/app/` | The shell and screens. Routes are hash routes: `#/runs`, `#/results?dir=…&sel=…`, `#/review?dir=…&sel=…`. |

`Icon` wraps `lucide-react` (bundled, tree-shaken) with the design system's size and
stroke contract and its fixed glyph meanings; add a glyph to its registry rather than
importing Lucide elsewhere.

## Semantics the UI keeps

- Matching, status derivation and review policy stay in Python. The page never derives a
  final status: the Reviewed view and the inspector read `review.adjudicated` through
  `GET /api/reviews` and `GET /api/explain`.
- Model output, the reviewer overlay and the adjudicated answer are separate layers on
  screen. Superseded and stale decisions stay visible in history and are never applied.
- "Model score" is the self-reported number with the policy's `accept_at` and
  `review_floor` ticks — never a percentage. Unknown tokens read "unknown", never 0; no
  money conversion is shown.
- A failed save says why and does not advance the queue; a result that changed since the
  queue loaded (`SnapshotMismatch`, `not_reviewable`) is shown as a conflict to reload.

## Control → operation

| Control | Operation | Status |
|---|---|---|
| Runs list, state, counts, calls · tokens | `GET /api/workspace` (manifest `run_state`, `counts`, `usage`) + `GET /api/tasks` | existing (usage and fingerprint added to the listing) |
| Resume run | `POST /api/tasks` with the run's job and directory, `resume` | existing |
| Cancel a running task | `POST /api/tasks/<id>/cancel` | existing |
| Create demo job (empty workspace) | `POST /api/init`, then an offline `POST /api/tasks` | existing |
| Map a file · New mapping job · Jobs · Ontology library · Clustering | the classic explorer at `/` | existing, not redesigned yet |
| Results table | `GET /api/mapping` (pages of `ops.list_results` with source text, context and target labels) | existing (`source_context` added) |
| Model output / Reviewed views, Human review column | `GET /api/reviews` (`review.review_history` + `review.adjudicated`) | **new read route** |
| History view | download `GET /api/export?view=history` | a table read route is **proposed** |
| Status chips | server-side `status` filter (model output); over loaded rows (reviewed) | existing |
| Inspector: decision, candidates, attempts, history | `GET /api/explain` (`ops.explain(full=True)`) | existing |
| Shared target hint | manifest `duplicate_targets` via `GET /api/run` | existing (passed through) |
| Review queue | `GET /api/review` (the worksheet's `needs_review` rows) | existing |
| Save and next | `POST /api/review` with one decision (`ops.review_apply`, validated against the current result; the decided result is looked up alone); the next record's `GET /api/explain` is prefetched while the current one is read | existing |
| Spot-check matched records | — | **proposed**: the worksheet holds `needs_review` rows only |
| Corrected target id check | `GET /api/target` (`ops.find_targets`, exact id); `POST /api/review` refuses a `replace` to an unknown id (422 `unknown_target`) | **new read route** |
| Worksheet | `GET /api/review-sheet` | existing |
| Export dialog | `GET /api/export?view=raw\|reviewed\|history` | existing |
| Summary, Evaluate tabs | the classic explorer's run page | existing, not redesigned yet |

## Departures from the prototype

- Data is live, not fixtures; the prototype's sidebar "state simulator" and fixture tag
  are gone. Every state it simulated (loading, empty, load error, failed save, conflict)
  is reached by the real one.
- The redesign replaced the sidebar with a top bar and a status line (server, running
  tasks, the keys that work on the screen), the results table and right inspector with
  the lineage view and a bottom drawer, status pills with squares and diamonds, and the
  status filter chips with the metric cells (each cell filters).
- The Results tab strip is gone: it repeated the Review and Export buttons. Summary and
  Evaluate are links to the classic explorer.
- The saved-decision toast is now a line in the decision bar plus a flash on the queue row.
- From the reference mock-up, its invented figures (latencies, "confidence 100%", SQL/CEL
  transforms) were not carried over: the score stays self-reported and never a percentage.
- The Runs table's second column is **Job** (name, job file, record count): the listing
  does not carry source and reference collection names.
- `ReviewBadge` for a stale or superseded decision leads with the state ("Stale ·
  accepted"); the prototype printed the decision twice.
- The "Include matched (spot-check)" switch is shown disabled and tagged *proposed*.
