import test from 'node:test';
import assert from 'node:assert/strict';
import { Parser, Config } from 'html-validate';
import { loadSite } from './helpers/load-site.mjs';

const site = loadSite();
const call = (name, ...args) => site.evaluate(`${name}(...${JSON.stringify(args)})`);
test('uploaded names remain literal text', () => {
  assert.equal(call('escapeHTML', '&<>"\''), '&amp;&lt;&gt;&quot;&#39;');
  const escaped = call('escapeHTML', '<svg onload=alert() />');
  assert.equal(escaped, '&lt;svg onload=alert() /&gt;');
  assert.equal(call('escapeHTML', 'floor area'), 'floor area');
  const tree = new Parser(Config.empty()).parseHtml(`<span>${escaped}</span>`);
  assert.equal(tree.querySelectorAll('svg').length, 0);
});
test('XGBoost drops missing selected values, retaining zero', () => {
  const rows = Array.from({ length: 40 }, (_, i) => `${i},${i + 2}`);
  rows[1] = ',3';
  rows[2] = '2,   ';
  rows[3] = '3';
  const result = call('parseXgbCSV', 'feature,target\n' + rows.join('\n'));
  assert.equal(result.rows.length, 37);
  assert.deepEqual(Array.from(result.rows[0]), [0, 2]);
  assert.equal(call('parseXgbCSV', rows.slice(4).join('\n')).rows.length, 36);
  assert.equal(typeof call('parseXgbCSV', rows.slice(0, 32).join('\n')), 'string');
  assert.equal(typeof call('parseXgbCSV', 'a,b\n' + 'hello,world\n'.repeat(40)), 'string');
  assert.equal(typeof call('parseXgbCSV', 'x,y\n' + '1,2\n'.repeat(29) + '3,"   "'), 'string');
});
test('numeric imports reject dates and numeric prefixes and preserve valid numbers', () => {
  const rows = Array.from({ length: 5 }, (_, i) => `2026-11-0${i + 1},${170 + i},${60 + i}`);
  assert.deepEqual(
    Array.from(call('parseXY', 'date,height,weight\n' + rows.join('\n'))[0]),
    [170, 60],
  );
  for (const sep of [',', ';', '\t']) {
    const input = [
      'bad,x,y',
      ...['0', '-1.5', '+2', '1e-3', '3'].map((v) => ['5kg', v, '2'].join(sep)),
    ].join('\r\n');
    assert.deepEqual(
      Array.from(call('parseXY', input), (r) => Array.from(r)),
      [
        [0, 2],
        [-1.5, 2],
        [2, 2],
        [0.001, 2],
        [3, 2],
      ],
    );
  }
  assert.equal(typeof call('parseXY', '1,2\n'.repeat(4)), 'string');
  const gaps = 'x,y\n,2\n' + '1,2\n'.repeat(5);
  assert.equal(call('parseXY', gaps).length, 5);
});
test('large numeric uploads respect both caps without argument overflow', () => {
  const input = '1,2\n'.repeat(150000);
  assert.equal(call('parseXY', input).length, 2000);
  assert.equal(call('parseXY', input, 5000).length, 5000);
  const source = site.lessons.find((l) => l.id === 'pca-table').init.toString();
  const parse = site.evaluate(
    '(' + source.match(/const parse = (\(text\) => \{[\s\S]*?\n {6}\});/)[1] + ')',
  );
  assert.equal(parse('1,2,3\n'.repeat(150000)).X.length, 2000);
  assert.equal(parse('x,y,z\n' + '1,2,3\n'.repeat(10) + '1,2,"   "').X.length, 10);
});
