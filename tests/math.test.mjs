import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parseHTML } from 'linkedom';
import { loadSite } from './helpers/load-site.mjs';
import { TOPICS } from '../js/app/catalog.js';
import { createLessonLoader } from '../js/app/loader.js';
import { renderLesson } from '../js/app/lesson-view.js';

async function mountMathLesson(t, id) {
  const { document, window } = parseHTML('<html><body></body></html>');
  const canvasContext = new Proxy(
    { measureText: () => ({ width: 0 }) },
    { get: (context, key) => context[key] ?? (() => {}) },
  );
  t.mock.method(window.HTMLCanvasElement.prototype, 'getContext', () => canvasContext);
  const previousDocument = globalThis.document;
  const previousWindow = globalThis.window;
  const previousStyle = globalThis.getComputedStyle;
  globalThis.document = document;
  globalThis.window = window;
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => '#eee' });
  t.after(() => {
    globalThis.document = previousDocument;
    globalThis.window = previousWindow;
    globalThis.getComputedStyle = previousStyle;
  });
  const entry = TOPICS.flatMap((topic) => topic.lessons).find((lesson) => lesson.id === id);
  const loader = createLessonLoader({ fetchText: (url) => readFile(url, 'utf8') });
  const loaded = await loader.load(entry);
  const root = await renderLesson(loaded, { document, loadText: loader.loadText });
  document.body.append(root);
  for (const canvas of root.querySelectorAll('canvas')) {
    canvas.getBoundingClientRect = () => ({ width: 480, height: 300 });
  }
  return { root, runtime: loaded.definition.init(root) };
}

test('shared log-gamma preserves known positive-input values', () => {
  const site = loadSite();
  for (const [x, want] of [
    [1, 0],
    [2, 0],
    [0.5, Math.log(Math.sqrt(Math.PI))],
    [5, Math.log(24)],
  ]) {
    assert.ok(Math.abs(site.evaluate(`lgam(${x})`) - want) < 1e-10);
  }
});

test('kernel PCA renders an explained variance readout from its eigenpairs', async (t) => {
  const { root, runtime } = await mountMathLesson(t, 'kernel-pca');
  runtime.draw();
  assert.match(root.querySelector('[data-r="var"]').textContent, /^\d+(?:\.\d+)?%$/);
});

test('the optimizer comparison shows a loss for enabled optimizers', async (t) => {
  const { root, runtime } = await mountMathLesson(t, 'optim');
  runtime.draw();
  assert.match(root.querySelector('[data-r="Lmom"]').textContent, /^\d+(?:\.\d+)?$/);
});
