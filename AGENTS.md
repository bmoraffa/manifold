# Manifold contributor guidance

## Purpose and philosophy

Manifold teaches machine learning through interactive pictures, plain-language intuition,
and mathematics. The simulation, explanation, and controls should help a learner understand
the same idea. Favor clarity for learners and maintainers over architectural novelty.

- Keep the website a native HTML, CSS, and JavaScript single-page application.
- Serve committed source directly from GitHub Pages. No build, bundler, frontend framework,
  backend, or additional publishing workflow is required.
- Prefer existing shared code and browser capabilities over new dependencies. Development
  tooling may use justified test dependencies; they must not enter the browser runtime or
  become a publishing prerequisite.
- Preserve useful modularization. A lesson is a cohesive module with its own simulation;
  shared controls, numerical helpers, and specialized lesson families remain shared.
- Improve readability by separating responsibilities, not by creating many tiny files,
  generic engines, speculative extension points, or a custom framework.
- Keep uploaded data and computation in the browser. Do not introduce telemetry, server
  uploads, or persistence as part of this architecture work.

## Architecture status

Distinguish the current implementation from the architectural direction below. Inspect the
checkout before assuming a proposed module, directory, or command exists, and keep this
guidance accurate as the architecture evolves.

### Current implementation

- `index.html` loads classic scripts in a specific order.
- `js/core.js` owns shared drawing, controls, animation, and mathematical helpers.
- Topic files such as `js/gaussian.js` and `js/classification.js` contain lesson arrays,
  HTML strings, simulations, and some topic-level helpers.
- `js/extra-lessons.js` appends lessons to topics and changes their final ordering.
- `js/app.js` owns the topic catalog, home animation, page rendering, menu, and hash routing.
- `js/about.js` contains the four About pages and citation behavior.
- `css/style.css` owns the existing visual identity and both themes.
- Tests currently use `tests/helpers/load-site.mjs` to load classic scripts into a Node VM.
  `eslint.config.mjs` records ownership of cross-script globals.

Until a caller has migrated, preserve its script-order dependencies and existing helper
interface. Do not remove a classic source file merely because a replacement exists.

### Approved target

```text
index.html                         SPA shell and native-module entry
css/style.css                      Shared styling and themes
js/app.js                          Application entry and composition
js/app/                            Catalog, loading, rendering, routing, hero, transitions
js/core/                           Cohesive shared drawing, animation, and utility modules
js/ui/                             Shared controls and native HTML templates
js/math/                           Numerical code genuinely shared across topics
lessons/<topic>/index.js            Topic metadata and explicit lesson order
lessons/<topic>/<id>/lesson.html    Readable teaching content and view fragments
lessons/<topic>/<id>/lesson.js      State, algorithms, drawing, and control connections
lessons/<topic>/shared.js           Reused topic behavior, where needed
lessons/<topic>/shared.html         Reused lesson-family layout, where needed
pages/about/                       Existing informational pages and citation behavior
tests/                             Regression, markup, loading, and navigation checks
```

Create shared files only when there is real shared behavior. Keep topic-local helpers with
their topic. A topic index is the authoritative order; filenames do not determine order.
Move later-added lessons into their owning topics and remove append/splice registration.
Adding a lesson should require its files and one topic entry, without editing the router.

Use explicit ES module imports and exports after migration. Topic indexes contain lightweight
metadata, module-relative HTML URLs, and explicit lazy import functions; importing the catalog
must not import or initialize all simulations. Use `new URL(relativePath, import.meta.url)`
for assets. GitHub Pages serves this project beneath `/manifold/`; origin-root paths and
fetch destinations constructed from arbitrary hash values are incorrect.

The proposed local preview uses a static HTTP server because modules and HTML loading require
HTTP. This is a development command, not a build or deployment service. Direct `file://`
opening works in the current implementation but is not a target requirement.

## Lesson and shell responsibilities

- Preserve the lesson interface: `init(root)` returns `draw()` and optional `stop()`,
  `sync()`, and `auto()` methods. Importing a lesson must not initialize it.
- The shell owns navigation, loading, root/runtime caches, mathematical typesetting,
  next-lesson navigation, and page transitions. Lessons own their simulation and controls.
- Create and initialize a lesson once per application session. Reuse its root and runtime
  when returning so control values, uploaded data, and simulation progress survive.
- Stop outgoing playback. Synchronize and draw the active lesson after mounting; autoplay
  only when eligible, visible, and permitted by the reduced-motion preference.
- Keep metadata separate from mutable runtime state. Preserve intentional shared mutable
  state as one module instance, not a copied object per importing lesson.
- Preserve specialized families such as `clfLesson`, `gdLesson`, `bagLesson`, and the
  unsupervised helpers. Extract their shared layout and behavior instead of duplicating
  an entire family in each lesson.
- Reuse successful and in-flight resource loads. Evict rejected application cache entries
  so failures can be retried. Browser-cached module evaluation errors may require reload;
  do not cache-bust imports in a way that duplicates shared module instances.
- Use a navigation token or equivalent guard so a slow earlier load, typesetting callback,
  animation completion, or scheduled frame cannot replace or autoplay a newer destination.
- Keep the last usable page available during loading/failure, keep navigation responsive,
  and provide concise inline recovery. URL, selected tab, and committed content must agree.

## HTML and control sharing

Author teaching prose, equations, figures, and static lesson layout in formatted `.html`
files. Use named native `<template data-part="...">` fragments for the shared renderer;
do not move unreadable nested HTML strings into smaller JavaScript files.

Repeated controls should have one shared template and behavior implementation. Lessons use
small native HTML declarations, such as `template[data-control="slider"]`, or explicit
family slots. Shared code expands them to the existing native HTML and removes the temporary
declarations. Preserve labels, IDs, `for` relationships, outputs, classes, and defaults.

Keep this mechanism limited to real control/layout reuse. Do not introduce a general-purpose
template language, evaluate expressions from HTML attributes, add shadow DOM, or create a
parallel custom-element system. Missing required parts and unknown control types should
fail clearly. Dynamic simulation output may still use DOM operations.

## Visual and behavioral compatibility

For the approved refactor, the settled site must retain its existing layout, typography,
colors, spacing, responsive behavior, prose, equations, control defaults, algorithms, and
interactions. Preserve existing hashes, final topic/lesson order, and history behavior.
The migration baseline is 115 educational lessons across 12 topics plus four About pages;
these counts are a parity check, not a permanent prohibition on adding lessons.

The approved additions are:

- Subtle slide plus fade within a topic: later lessons enter from the right and earlier
  lessons from the left, based on the explicit topic order.
- A simple fade between topics and to/from home or About.
- The existing pinned navigation stays stationary. Content scrolls normally underneath it.
- A gradual theme-aware soft edge extends 64px below navigation while scrolled. It overlays
  content, adds no layout height, and does not dim the initial heading at scroll position zero.
- Starting motion values: 30px movement, 280ms slide/fade, 220ms simple fade, 180ms outgoing
  fade. These are shared tuning values, not separate settings repeated in lessons.

Keep transitions cancellable and temporary snapshots inert. Copy canvas pixels if a snapshot
is used; never initialize a second simulation for an outgoing visual. Remove temporary
layers on completion, cancellation, and errors. Navigation remains usable during transitions.

Respect reduced motion on initial load and live preference changes, with immediate navigation
and no unwanted autoplay. Keep keyboard focus and scroll targets visible below wrapped
navigation and the fading region. The gradient must not intercept pointer input. Use normal
document scrolling, not a new viewport-height lesson scroller. Preview-only controls are not
part of the production interface.

Retain the existing CSS variables as the visual source of truth. Record approved durable
visual decisions in `DESIGN.md` when it is introduced; do not restyle the site to match a
generic design system.

## General coding principles

- Read callers, shared owners, and relevant tests before changing an interface. Make the
  smallest complete change that solves the actual problem.
- Keep interfaces small and explicit. Put complexity behind shared modules so a fix happens
  in one place. Prefer plain functions and data to unnecessary classes, wrappers, or registries.
- Keep files that change together close together. Keep a cohesive algorithm readable in one
  lesson file instead of splitting it solely to reduce line counts.
- Extract proven reuse; do not generalize on the assumption that a second use might appear.
  Do not copy an existing helper when importing its real owner is sufficient.
- During structural migration, move working logic without rewriting it. Keep error handling,
  numerical safeguards, run budgets, upload limits, and validation intact.
- Treat uploaded values and labels as untrusted data. Use `textContent` or correct escaping
  for text; never interpolate uploaded names into executable markup. Trusted repository HTML
  and untrusted learner input are different trust boundaries.
- Use native semantic controls, working label associations, visible keyboard focus, and
  accessible status feedback. Motion and visual polish must not weaken accessibility.
- Avoid side effects at module import, hidden cross-script globals, circular dependencies,
  and initialization order requirements that an explicit interface could express.
- Comment why a non-obvious decision exists, not what straightforward code does. Name real
  limitations precisely; avoid speculative TODOs and unrelated cleanup.
- Preserve user changes and task scope. Do not publish, push, or introduce a workflow simply
  because a local change is ready.

## Verification and tooling

Use the existing Node range (`^22.22.0 || >=24.8.0`) and locked development dependencies.
Prettier owns formatting: two spaces, semicolons, single-quoted JavaScript, trailing commas,
a 100-column target, and LF endings. Preserve meaningful prose whitespace and TeX notation.

Current commands:

| Command                | Purpose                                 |
| ---------------------- | --------------------------------------- |
| `npm ci`               | Install locked development dependencies |
| `npm test`             | Existing focused regressions            |
| `npm run format:check` | Check formatting                        |
| `npm run lint:js`      | JavaScript linting                      |
| `npm run lint:css`     | CSS linting                             |
| `npm run lint:html`    | HTML validation                         |
| `npm run check`        | Formatting, tests, and linters          |

While editing one document or module, use targeted formatter commands rather than formatting
the entire checkout and its throwaway previews. Keep `.superpowers/` artifacts out of commits.

For structural refactors, capture catalog and rendered-HTML evidence before changing the
implementation. Verify representative standalone lessons, shared families, and informational
pages against that baseline. Do not regenerate expected output merely to accept a regression.

Keep existing regression assertions when adapting tests to modules. Add focused tests for
new asynchronous loading, failure recovery, state caching, template expansion, and transition
cancellation. Test real shared interfaces rather than duplicating their implementation in
the test. Documentation-only changes need formatting checks, not new behavioral tests.

Use a real browser for canvas rendering, controls, uploads, equations, motion, and visual
parity. Check desktop/narrow widths, both themes, reduced motion, interrupted navigation,
and direct hash links/reloads beneath `/manifold/`. Check filename/import casing explicitly;
Windows can hide mistakes that fail on GitHub Pages. A Node DOM test cannot prove layout
or canvas correctness.

The README documents pre-existing lint debt. Record a baseline, fix regressions introduced
by the task, and distinguish inherited failures from new ones. Report commands actually run
and their outcomes; do not claim tests, browser checks, or a migration have passed by inference.
