# Add or edit a lesson

Find the topic under `lessons/`. Its `index.js` lists the lessons in display order.
For a new lesson, create a folder named for the hash ID and add `lesson.html` and
`lesson.js`. Register the descriptor once in the topic index.

```js
{
  id: 'galton',
  tab: 'Galton board',
  title: 'A board of pegs makes a bell',
  viewUrl: new URL('./galton/lesson.html', import.meta.url),
  load: () => import('./galton/lesson.js'),
}
```

Use native `<template data-part="…">` fragments for `lede`, optional `intro`, `stage`,
`tryit`, `math`, `question`, and `answer`. Keep existing element IDs when editing an
interactive stage. Write TeX directly, using `&lt;` for a literal less-than sign in HTML.
The shell owns the title, disclosure, next link, and MathJax typesetting.

For a repeated control, use its declaration in `js/ui/controls.html` and
`js/ui/controls.js`. For example, a slider declaration looks like this.

```html
<template
  data-control="slider"
  data-id="gb-rows"
  data-label="Rows of pegs"
  data-min="3"
  data-max="16"
  data-step="1"
  data-value="12"
></template>
```

The output is a native label, range input, and output element. For an existing family,
follow its `shared.html` slot names and use the shared factory instead of copying its stage.
Keep the simulation in the lesson module. Its default export has `init(root)`, which binds
events and returns `draw()` and optional `stop()`, `sync()`, and `auto()` methods.
The shell calls `init` once and keeps the runtime and DOM when the learner switches away.
Use explicit imports and keep browser-dependent work inside `init`.

Run `npm ci`, then `npm run dev` and open `http://127.0.0.1:4173/manifold/`.
Open the direct hash for the lesson and test its controls before and after revisiting it.
Run `npm test` and `npm run lint:html` before submitting changes. The migration fixture
records the original 119 entries. When intentionally adding or updating lessons, retain
the original evidence and adjust the scoped parity assertions to reflect the approved change.
