import assert from 'node:assert/strict';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { createMathTypesetter } from '../js/app/mathjax.js';

test('MathJax processes mounted roots one at a time and skips detached work', async () => {
  const { document } = parseHTML(
    '<html><body><div><p class="prose">A</p></div><div><p class="prose">B</p></div></body></html>',
  );
  const [a, b] = document.body.children;
  const first = Promise.withResolvers();
  const calls = [];
  const math = createMathTypesetter({
    window: {
      MathJax: {
        typesetPromise(prose) {
          calls.push(prose[0].textContent);
          return first.promise;
        },
      },
    },
  });
  const detached = document.createElement('div');
  detached.innerHTML = '<p class="prose">Detached</p>';
  await math.prepare(detached);
  assert.deepEqual(calls, []);
  const pendingA = math.prepare(a);
  const pendingB = math.prepare(b);
  await Promise.resolve();
  assert.deepEqual(calls, ['A']);
  first.resolve();
  await Promise.all([pendingA, pendingB]);
  assert.deepEqual(calls, ['A', 'B']);
});
