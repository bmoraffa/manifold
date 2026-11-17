import test from 'node:test';
import assert from 'node:assert/strict';
import { Parser, Config } from 'html-validate';
import { loadSite } from './helpers/load-site.mjs';
test('lesson controls have unique IDs and the repaired sliders have matching labels', () => {
  const { lessons } = loadSite();
  assert.equal(lessons.length, 119);
  assert.equal(new Set(lessons.map((l) => l.id)).size, 119);
  const controls = new Map([
    ['svm-cost', '0'],
    ['rs-depth', '20'],
    ['lo-rank', '2'],
  ]);
  for (const lesson of lessons) {
    const tree = new Parser(Config.empty()).parseHtml(lesson.stage || '');
    const ids = tree.querySelectorAll('[id]').map((n) => n.id);
    assert.equal(new Set(ids).size, ids.length, lesson.id);
    for (const [id, value] of controls) {
      const input = tree.querySelector('#' + id);
      if (!input) continue;
      assert.equal(input.tagName, 'input');
      assert.equal(input.getAttributeValue('value'), value);
      assert.ok(tree.querySelector(`label[for="${id}"]`));
      assert.ok(tree.querySelector(`output[for="${id}"]`));
      controls.delete(id);
    }
  }
  assert.equal(controls.size, 0);
});
