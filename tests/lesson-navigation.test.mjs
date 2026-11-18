import assert from 'node:assert/strict';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { mountApp } from '../js/app/router.js';
import { createTransitions, lessonMotion } from '../js/app/transitions.js';

function harness({ withProse = false } = {}) {
  const { document, window } = parseHTML(`<html><body>
    <aside id="side"></aside><button id="menu-b"></button><main><section id="home"></section>
    <header id="modbar" hidden><span id="mod-name"></span><nav id="ltabs"></nav></header>
    <section id="lesson" hidden></section></main></body></html>`);
  const media = new EventTarget();
  media.matches = false;
  window.matchMedia = () => media;
  window.scrollTo = () => {};
  window.requestAnimationFrame = (callback) => callback();
  window.location = { hash: '' };
  window.history = {
    replaceState(_state, _title, hash) {
      window.location.hash = hash;
    },
  };
  const entries = ['a', 'b', 'c'].map((id) => ({ id, tab: id, title: id, isAboutPage: true }));
  const pending = new Map();
  const calls = [];
  const loader = {
    load: (entry) => pending.get(entry.id)?.promise ?? Promise.resolve(make(entry)),
  };
  function make(entry) {
    return {
      entry,
      html:
        '<template data-part="lede">Intro</template><template data-part="body"><input>' +
        (withProse ? '<p class="prose">Equation</p>' : '') +
        '</template>',
      controlsHTML: '',
      definition: {
        init(root) {
          calls.push('init:' + entry.id);
          return {
            draw() {},
            auto() {
              calls.push('auto:' + entry.id);
            },
            stop() {
              calls.push('stop:' + entry.id);
            },
            root,
          };
        },
      },
    };
  }
  function delay(id) {
    const task = Promise.withResolvers();
    pending.set(id, task);
    return {
      resolve: () => {
        pending.delete(id);
        task.resolve(make(entries.find((entry) => entry.id === id)));
      },
      reject: () => {
        pending.delete(id);
        task.reject(new Error('Offline'));
      },
    };
  }
  const app = mountApp({
    document,
    window,
    catalog: [{ id: 'topic', title: 'Topic', lessons: entries }],
    loader,
    hero: { start() {}, stop() {}, draw() {}, rebuild() {} },
  });
  return { ...app, document, window, delay, calls };
}

test('latest route wins and revisits keep the same lesson root', async () => {
  const app = harness();
  await app.navigate('topic');
  const input = app.document.querySelector('#lesson input');
  input.value = 'learner data';
  const wait = app.delay('b');
  const older = app.navigate('b');
  await app.navigate('c');
  wait.resolve();
  await older;
  assert.equal(app.getActiveId(), 'c');
  assert.equal(app.window.location.hash, '#c');
  assert.ok(!app.calls.includes('init:b') && !app.calls.includes('auto:b'));
  await app.navigate('a');
  assert.equal(app.document.querySelector('#lesson input'), input);
  assert.equal(input.value, 'learner data');
  assert.equal(app.calls.filter((call) => call === 'init:a').length, 1);
  app.destroy();
});

test('a failed route keeps the selected lesson and can be retried', async () => {
  const app = harness();
  await app.navigate('a');
  const wait = app.delay('b');
  const navigation = app.navigate('b');
  wait.reject();
  await navigation;
  assert.equal(app.getActiveId(), 'a');
  assert.equal(app.window.location.hash, '#a');
  assert.equal(app.document.querySelector('[aria-selected="true"]').getAttribute('data-go'), 'a');
  assert.match(app.document.querySelector('.navigation-status').textContent, /Retry/);
  await app.navigate('b');
  assert.equal(app.getActiveId(), 'b');
  app.destroy();
});

test('the outgoing lesson stays usable while loading and stops when it leaves the page', async () => {
  const app = harness();
  await app.navigate('a');
  const wait = app.delay('b');
  const pending = app.navigate('b');
  assert.equal(app.getActiveId(), 'a');
  assert.ok(!app.calls.includes('stop:a'));
  wait.resolve();
  await pending;
  assert.ok(app.calls.indexOf('stop:a') < app.calls.indexOf('init:b'));
  assert.equal(app.calls.filter((call) => call === 'stop:a').length, 1);
  app.destroy();
});

test(
  'late MathJax still typesets the visible lesson when a replacement fails',
  { timeout: 2000 },
  async () => {
    const app = harness({ withProse: true });
    const savedMath = app.window.MathJax;
    app.window.MathJax = undefined;
    try {
      await app.navigate('a');
      const wait = app.delay('b');
      const navigation = app.navigate('b');
      wait.reject();
      await navigation;
      const rendered = Promise.withResolvers();
      app.window.MathJax = {
        typesetPromise(nodes) {
          rendered.resolve(nodes);
          return Promise.resolve();
        },
      };
      const [prose] = await rendered.promise;
      assert.equal(prose, app.document.querySelector('#lesson .prose'));
    } finally {
      app.destroy();
      app.window.MathJax = savedMath;
    }
  },
);

test('movement follows lesson order and switches to a fade between topics', () => {
  const a = { topic: { id: 'one' }, index: 1 };
  assert.deepEqual(lessonMotion(a, { ...a, index: 2 }), { offsetXPx: 30, duration: 280 });
  assert.deepEqual(lessonMotion(a, { ...a, index: 0 }), { offsetXPx: -30, duration: 280 });
  assert.deepEqual(lessonMotion(a, { topic: { id: 'two' }, index: 1 }), {
    offsetXPx: 0,
    duration: 220,
  });
  assert.deepEqual(lessonMotion(a, { ...a, topic: { id: 'one', isAbout: true }, index: 2 }), {
    offsetXPx: 0,
    duration: 220,
  });
});

test('cancelling a transition removes its snapshot and restores instant navigation', async () => {
  const { document, window } = parseHTML(
    '<html><body><header></header><section><button id="control">Run</button></section></body></html>',
  );
  const nav = document.querySelector('header');
  const content = document.querySelector('section');
  const media = new EventTarget();
  media.matches = false;
  window.matchMedia = () => media;
  const animations = [];
  Object.getPrototypeOf(content).animate = function animate(frames) {
    const deferred = Promise.withResolvers();
    const result = {
      frames,
      done: false,
      finished: deferred.promise,
      cancel() {
        this.done = true;
        deferred.reject(new Error('Cancelled'));
      },
    };
    animations.push(result);
    return result;
  };
  const motion = createTransitions({ window, document, nav, content });
  const topic = { id: 'test' };
  const from = { id: 'a', index: 0, topic, element: content };
  const to = { id: 'b', index: 1, topic, element: content };
  const pending = motion.run({ from, to, isCurrent: () => true, commit() {} });
  const snapshot = document.querySelector('.navigation-snapshot');
  assert.ok(snapshot);
  assert.equal(snapshot.inert, true);
  assert.equal(snapshot.querySelector('[id]'), null);
  media.matches = true;
  media.dispatchEvent(new Event('change'));
  await pending;
  assert.equal(document.querySelector('.navigation-snapshot'), null);
  assert.equal(animations.length, 2);
  assert.ok(animations.every((animation) => animation.done));
  let committed = false;
  await motion.run({
    from,
    to,
    isCurrent: () => true,
    commit() {
      committed = true;
    },
  });
  assert.equal(committed, true);
  assert.equal(animations.length, 2);
  motion.destroy();
  delete Object.getPrototypeOf(content).animate;
});
