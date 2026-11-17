import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import baseline from './fixtures/legacy-lessons.json' with { type: 'json' };
import { TOPICS } from '../js/app/catalog.js';
import { createLessonLoader } from '../js/app/loader.js';
import { renderLesson } from '../js/app/lesson-view.js';

// Keep inline text and TeX intact. Whitespace between block elements is formatting.
const blocks = new Set(
  'div section p pre h1 h2 h3 h4 figure figcaption fieldset legend dl dt dd ol ul li details summary'.split(
    ' ',
  ),
);
function structure(parent) {
  parent.normalize();
  return [...parent.childNodes].flatMap((node, index, siblings) => {
    if (node.nodeType === 3) {
      const text = node.nodeValue;
      if (
        !text.trim() &&
        (blocks.has(siblings[index - 1]?.localName) || blocks.has(siblings[index + 1]?.localName))
      )
        return [];
      return [['text', parent.localName === 'pre' ? text : text.replace(/[\t\r\n ]+/g, ' ')]];
    }
    if (node.nodeType !== 1) return [];
    const attributes = node
      .getAttributeNames()
      .sort()
      .map((name) => [name, node.getAttribute(name)])
      .filter(([name, value]) => name !== 'class' || value !== '');
    return [[node.localName, attributes, structure(node)]];
  });
}

test('all 119 native entries preserve catalog metadata and generated HTML', async () => {
  const { document } = parseHTML('<html><body></body></html>');
  const loader = createLessonLoader({ fetchText: (url) => readFile(url, 'utf8') });
  assert.deepEqual(
    TOPICS.map(({ id, title, isAbout, lessons }) => ({
      id,
      t: title,
      ...(isAbout ? { about: isAbout } : {}),
      ids: lessons.map((l) => l.id),
    })),
    baseline.topics,
  );
  assert.equal(baseline.entries.length, 119);
  const mismatches = [];
  function mismatch(actual, expected, path = '') {
    if (Array.isArray(actual) && Array.isArray(expected)) {
      for (let i = 0; i < Math.max(actual.length, expected.length); i++) {
        const failed = mismatch(actual[i], expected[i], path + '/' + i);
        if (failed) return failed;
      }
      return;
    }
    if (actual !== expected) return [path, actual, expected];
  }
  for (const topic of TOPICS) {
    for (const [i, entry] of topic.lessons.entries()) {
      const original = baseline.entries.find((l) => l.id === entry.id);
      assert.deepEqual(
        [entry.tab, entry.title, topic.id],
        [original.tab, original.title, original.topicId],
        entry.id,
      );
      const expected = document.createElement('div');
      // Keep the captured source immutable. In the HTML files TeX's less-than sign
      // is encoded as an HTML entity so the browser doesn't treat it as a tag.
      expected.innerHTML = original.html.replace(/\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$/g, (equation) =>
        equation.replaceAll('<', '&lt;'),
      );
      if (entry.id === 'acknowledgements') {
        // Dachun Sun is now credited as a co-author in the citation.
        const author = [...expected.querySelectorAll('p > b')].find(
          (name) => name.textContent === 'Professor Dachun Sun',
        );
        author.parentElement.remove();
      }
      try {
        const actual = await renderLesson(await loader.load(entry), {
          document,
          nextEntry: topic.lessons[i + 1],
          loadText: loader.loadText,
        });
        const difference = mismatch(structure(actual), structure(expected));
        if (difference) mismatches.push([entry.id, ...difference]);
      } catch (error) {
        mismatches.push([entry.id, error.stack]);
      }
    }
  }
  assert.equal(
    mismatches.length,
    0,
    mismatches.map((item) => JSON.stringify(item).slice(0, 500)).join('\n'),
  );
});
