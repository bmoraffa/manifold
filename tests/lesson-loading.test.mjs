import assert from 'node:assert/strict';
import test from 'node:test';
import { createLessonLoader } from '../js/app/loader.js';

test('concurrent requests share work and a failed fetch can be retried', async () => {
  const viewUrl = new URL('https://example.test/manifold/lessons/gaussian/galton/lesson.html');
  let reads = 0;
  const loader = createLessonLoader({
    fetchText: async (url) => {
      if (url.href !== viewUrl.href) return '';
      if (++reads === 1) throw new Error('Unavailable');
      return '<template data-part="lede"></template>';
    },
  });
  const entry = { viewUrl, load: async () => ({ default: { init() {} } }) };
  const first = loader.load(entry);
  assert.equal(loader.load(entry), first);
  await assert.rejects(first, /Unavailable/);
  assert.match((await loader.load(entry)).html, /data-part/);
  assert.equal(reads, 2);
});
