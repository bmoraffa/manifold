import assert from 'node:assert/strict';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { mountThemeSwitches } from '../js/app/theme.js';

test('theme switches share the saved choice and crossfade only when motion is permitted', (t) => {
  const { document } = parseHTML(`<html><body>
    <span data-theme-anchor></span><span data-theme-anchor></span>
    <template id="theme-switch"><button aria-pressed="false"></button></template>
  </body></html>`);
  const scheme = new EventTarget();
  scheme.matches = true;
  const motion = { matches: false };
  const transition = (document.startViewTransition = t.mock.fn((update) => {
    update();
    return { ready: Promise.resolve() };
  }));
  const preferences = new Map();
  const window = {
    matchMedia: (query) => (query.includes('reduced-motion') ? motion : scheme),
    localStorage: {
      getItem: (key) => preferences.get(key),
      setItem: (key, value) => preferences.set(key, value),
    },
  };
  mountThemeSwitches({ document, window });
  const switches = [...document.body.querySelectorAll('button')];
  const states = () => switches.map((control) => control.getAttribute('aria-pressed'));
  assert.deepEqual(states(), ['true', 'true']);

  scheme.matches = false;
  scheme.dispatchEvent(new Event('change'));
  assert.deepEqual(states(), ['false', 'false']);

  switches[0].click();
  assert.equal(transition.mock.callCount(), 1);
  assert.equal(document.documentElement.dataset.theme, 'dark');
  assert.deepEqual(states(), ['true', 'true']);
  motion.matches = true;
  switches[1].click();
  assert.equal(transition.mock.callCount(), 1);
  assert.equal(document.documentElement.dataset.theme, 'light');
  assert.deepEqual(states(), ['false', 'false']);
  assert.equal(preferences.get('manifold-theme'), 'light');

  scheme.matches = true;
  scheme.dispatchEvent(new Event('change'));
  assert.deepEqual(states(), ['false', 'false']);
});
