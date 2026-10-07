# Frontend refinement — implementation and validation

Validated 7 October 2026. This change is confined to the frontend and documentation.
Backend source, Python tests, API contracts, models, prompts, persistence and
retained Phase I–VI output schemas are unchanged.

## A. Baseline

Before edits, ESLint, TypeScript, all 4 playback tests and the production build
passed. Baseline assets: JS 245.27 kB (75.87 kB gzip), CSS 19.40 kB (5.04 kB gzip).
The retained two-speaker meeting loaded successfully. Overview and raw transcript
controls were inspected in the browser, and existing Phase VII screenshots and
validation records covered upload, history, evidence, downloads and failures.

## B. UI stack

| Dependency | Exact installed version | Purpose |
|---|---|---|
| React / React DOM | 19.3.0, unchanged | Application |
| Vite | 8.3.3, unchanged | Build/dev server |
| TypeScript | 5.9.3, unchanged | Strict type checking |
| Tailwind / @tailwindcss/vite | 4.3.3 | CSS utilities and Vite integration |
| Lucide React | 1.52.0 | Consistent icons |
| Radix UI | 1.7.0 | Accessible shadcn primitives |
| class-variance-authority | 0.7.1 | Component variants |
| clsx | 2.1.1 | Conditional classes |
| tailwind-merge | 3.7.0 | Local `cn` utility |
| tw-animate-css | 1.4.0 | Small sheet/menu transitions |
| Sonner | 2.0.8 | Download feedback |
| Testing Library React | 16.3.3 | Component interaction tests |
| Testing Library user-event | 14.6.7 | Keyboard/pointer interactions |
| Testing Library jest-dom | 7.0.1 | DOM assertions |
| jsdom | 30.1.2 | Test DOM only |
| @types/node | 26.6.4 | Vite/test configuration types |

`components.json` configures shadcn's New York/Radix style with local TSX
components and the `@` alias. The official CLI generated Button, Badge, Card,
Tabs, Sheet, DropdownMenu, Alert, Input, Skeleton, Progress and Separator.
Generated `cn` imports were redirected to the existing local utility; no separate
`cn` runtime package remains. No large UI, charting or animation framework was added.
Exact dependency resolution is retained in `frontend/package-lock.json`.

Setup follows [Tailwind's Vite integration](https://tailwindcss.com/docs/installation/using-vite)
and [shadcn's Tailwind v4 guidance](https://ui.shadcn.com/docs/tailwind-v4).

## C. Design system

CSS variables and semantic Tailwind tokens cover background, surface,
surface-muted, border, foreground, muted-foreground, primary, primary-foreground,
accent, warning, danger and success. Main colors: warm background `#f7f8f4`,
forest green `#285a48`, sage `#e5eddf`, white surfaces and restrained borders.
Speaker/type tones are centralized, muted and supplementary to visible labels.
The existing sans-serif stack and upload hero's selective serif accent remain;
no remote font or image downloads are needed.

Titles are compact, transcript body text is 14 px, and metadata is smaller.
Cards use two surface levels, 18–22 px padding and 10–24 px gaps. Major cards
have subtle borders without heavy shadows; interactions use approximately 180 ms.

## D. Main layout and architecture

The shell retains upload/history/local routing. The header and meeting tabs are
sticky; title/status/date/duration/speakers are compact. Keyboard-operable Radix
tabs show decision/action counts, including zero. Downloads use a dropdown.

`main.tsx` now contains the shell and upload/history/status pages. The larger
workspace is separated into `pages/workspace.tsx`, reusable meeting cards,
transcript, evidence sheet, download menu, audio player, common states and
processing stepper. `hooks/useAudioPlayback.ts` owns playback; `api.ts` remains
the single API layer. `types.ts` and the original playback module/tests are unchanged.

## E. Conversation transcript

Utterances use source IDs, readable conversation bubbles and stable speaker
styling. Bubble position depends on the retained speaker ID, not turn alternation;
actual anonymous labels remain visible. Raw/refined switching preserves the same
utterances. Applied edits have a Refined badge and expandable original ASR.

Timestamp/play controls seek to the utterance start and play the shared recording.
Active highlighting uses retained start/end bounds, with no inferred utterance
during silence. Follow audio is off by default and yields for four seconds after
wheel/touch/navigation-key input. Literal search highlights all occurrences,
shows matching utterance count and supports previous/next navigation. Original
ASR segments and word timestamps remain accessible in Raw mode.

## F. Evidence

The shadcn Sheet is a nonmodal right drawer on desktop/tablet and a modal bottom
sheet below 768 px. Mobile focus is trapped; Escape/Close work. The selected item,
all returned source spans, speaker labels and timestamps are rendered directly
from the existing deterministic API. No quote is generated in React.

Evidence play seeks/plays the central element and pauses near the existing end
timestamp. The sheet shows playing state and interval progress. Its original-ASR
disclosure includes retained refined/raw wording and source references.
Jump to transcript closes the sheet, switches tabs, clears conflicting raw/search
state, focuses the exact utterance and highlights it temporarily.

## G. Intelligence views

Overview has a compact metric strip, summary cards and short decision/action
previews. Minutes retain topic grouping and Proposal/Decision/Information/Action
badges, with styles available for Concern/Discussion. Decisions use restrained
green emphasis. Actions show task, owner and deadline with icons; null fields
display Unassigned / Not specified. Evidence source counts use retained IDs.
Empty summaries, minutes, decisions and tasks use concise icon states.
No owners, dates, confidence values or evidence timestamps are inferred.

## H. Audio

Exactly one audio element serves the unchanged canonical-audio URL. Its controller
owns current time, duration, playing/rate state and one evidence boundary.
Controls include play/pause, seek, ±5 seconds and 1×/1.25×/1.5×/2×. Manual seeking
or skipping cancels evidence bounds; pausing/resuming preserves the current bound.
The player displays the active speaker or evidence ID and interval. Metadata reload
restores the selected speed. Playback errors provide a retry action.

Actual retained-fixture browser checks: timestamp playback began at **22.480 s**,
highlighted **utt_000005**, and used **1.5×**. Evidence **22.480–28.560 s** stopped
at **28.584 s** (24 ms overshoot). Jump focused **utt_000005**. These are browser
timestamp checks, not forced alignment or frame-exact playback claims.

## I. Downloads

Primary: meeting JSON, minutes Markdown, refined JSON/TXT, raw JSON/TXT and evidence
manifest. Advanced: speaker JSON/TXT, diarization, grounding JSON/TXT, edit log and
canonical WAV. All original logical IDs/URLs remain unchanged. A one-byte range
preflight detects unavailable artifacts, then the browser streams the normal
download; large WAVs are not loaded into a JavaScript blob. Sonner reports start/error.
Actual meeting JSON download and its start toast were observed in the browser.

## J. Upload, processing, history and errors

The existing upload hero remains. Drag hover, selected filename/size, disabled
controls and removal are retained. XMLHttpRequest reports actual upload byte
progress using the unchanged single `file` multipart contract. Sending 100% is
labelled as waiting for server acceptance, not pipeline completion.

Processing uses a vertical six-stage timeline, source status, completed stage
counts, stage durations and elapsed time. No model-level percentage is invented.
Loading uses skeleton cards; failures use Alert cards and existing safe messages.
History uses compact rows with date, size, state and failed-stage labels.

An actual corrupt browser upload failed in INGESTING with `unsupported_media`,
zero completed stages and no later model calls. An isolated fake runner at port
8001 exercised processing and empty/history states under `.validation/ui-polish/`.
Those controlled states are not ML inference or an accuracy dataset. The isolated
server was stopped after validation; the main server remains on port 8000.

## K. Responsive validation

Actual meeting Overview, Transcript, Minutes, Decisions, Actions and evidence were
reviewed at desktop, tablet and phone sizes; all five views had no page horizontal
overflow at **768×1024** and **390×844**. Desktop **1366×900** and intermediate
**1024×900** were also checked. Upload, controlled processing, history and empty
states were captured at the three required resolutions. Mobile tabs scroll
horizontally; the player uses two compact rows. Viewport overrides were reset.

## L. Accessibility

Visible focus rings, labelled icon buttons, keyboard tabs/menus, Escape/Close,
mobile sheet focus containment, transcript focus after jumps, native seek/speed
controls, semantic headings and reduced-motion overrides are implemented.
Color is supplementary to text labels. This was a targeted keyboard/browser review,
not a full screen-reader or formal WCAG audit.

## M. Final checks

- **21/21 frontend tests passed**: original 4 playback tests plus 17 new tests.
- New checks cover bubble playback, corrections, raw/refined/search, active timing,
  one audio element, speed/skip, bounded end/manual seek, evidence open/Escape,
  multi-span display, jump/clear-search, auto-follow/manual-scroll interaction,
  unavailable evidence, dropdown grouping, streamed download preflight/errors,
  absent owner/deadline, empty views and upload progress/server/network failures.
- ESLint, strict TypeScript and production build passed.
- Final bundle: **437.59 kB JS / 134.72 kB gzip**, **73.68 kB CSS / 15.46 kB gzip**.
  No chart, waveform or large animation package is present. Local shadcn components
  and Radix/Lucide imports are bundled/tree-shaken.
- Final retained-meeting browser console contained no errors.
- SHA-256 audit found **zero changes to backend source and Python tests**.
  Python regression rerun was not required for this frontend-only change; the
  preceding Phase VII run was 542 tests, 531 passed and 11 skipped.
- `.env`, node_modules, built assets and validation/runtime artifacts remain ignored.
  The repository still has no tracked files/commits, so ordinary Git diff is empty;
  content hashes provide the backend preservation check. No commit was created.

## N. Screenshots

Screenshots are retained in ignored `.validation/ui-polish/`, including:

```text
final-workspace.jpg
transcript-desktop.jpg / transcript-tablet.jpg / transcript-mobile.jpg
evidence-desktop.jpg / evidence-tablet.jpg / evidence-mobile.jpg
overview-1024.jpg / overview-tablet.jpg / overview-mobile.jpg
minutes-desktop.jpg / minutes-tablet.jpg / minutes-mobile.jpg
decisions-desktop.jpg / decisions-tablet.jpg / decisions-mobile.jpg
actions-tablet.jpg / actions-mobile.jpg
downloads-desktop.jpg / failure-desktop.jpg
upload-desktop.jpg / upload-tablet.jpg / upload-mobile.jpg
processing-controlled-desktop.jpg / processing-controlled-tablet.jpg / processing-controlled-mobile.jpg
history-controlled-desktop.jpg / history-controlled-tablet.jpg / history-controlled-mobile.jpg
empty-actions-controlled-desktop.jpg / empty-actions-controlled-tablet.jpg / empty-actions-controlled-mobile.jpg
```

## O. Backend changes

**None.** Routes, schemas, model settings, evidence resolution, serializers and
persistence are unchanged. No inference artifacts were patched.

## P. Genuine limitations

- Large histories/transcripts are not paginated/virtualized. Follow audio is optional;
  scrolling precision and playback stopping depend on the browser.
- History metadata does not include duration/decision/action counts; the UI avoids
  fetching every full meeting record just to populate history rows.
- Download preflight verifies availability at request time, but a later native
  download can still fail due to a connection interruption.
- Existing provider/ML quality and network/quota limitations remain. Semantic
  verification, identity resolution, editing, PDF generation and local model
  migration remain separate future work.

## Run locally

```powershell
cd frontend
npm ci
npm run lint
npm run typecheck
npm test
npm run build
cd ..
python -m meeting_assistant.web
```

Open http://127.0.0.1:8000. For development, run `npm run dev` in `frontend` with
the existing backend running; the original Vite API proxy is unchanged.
