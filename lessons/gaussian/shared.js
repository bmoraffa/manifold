/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { randnFrom, rng } from '../../js/core/shared.js';

const Z250 = (() => {
  const r = randnFrom(rng(11)),
    a = [];
  for (let i = 0; i < 250; i++) a.push([r(), r()]);
  return a;
})();

export { Z250 };
