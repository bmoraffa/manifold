import test from 'node:test';
import assert from 'node:assert/strict';
import { loadSite } from './helpers/load-site.mjs';
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
