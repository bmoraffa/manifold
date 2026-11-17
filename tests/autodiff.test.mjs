import assert from 'node:assert/strict';
import test from 'node:test';
import { AD } from '../js/math/autodiff.js';

test('the standalone autodiff module includes the operations formerly registered by topic scripts', () => {
  for (const name of ['addb', 'mlp', 'mean', 'pgl']) {
    assert.equal(typeof AD[name], 'function', name);
  }
});
