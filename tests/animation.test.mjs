import test from 'node:test';
import assert from 'node:assert/strict';
import { loadSite } from './helpers/load-site.mjs';
function run(limit = '1', initial = 20) {
  let frame,
    count = initial,
    complete = false;
  const button = {
    textContent: 'Run',
    classList: { add() {}, remove() {} },
    addEventListener(_, fn) {
      this.click = fn;
    },
  };
  const note = { textContent: '' };
  const max = { value: limit };
  const root = { querySelector: (s) => (s.endsWith('-maxnote') ? note : max) };
  const site = loadSite({
    globals: {
      button,
      root,
      requestAnimationFrame: (fn) => {
        frame = fn;
        return 1;
      },
      cancelAnimationFrame: () => {
        frame = null;
      },
      counter: () => count,
      completed: () => complete,
    },
  });
  site.evaluate(
    "var playback = playBtn(button, Anim(()=>{})); var check = attachRunLimit(playback,root,'test',{count:counter,done:completed,restart:()=>{},what:'steps'});",
  );
  return {
    site,
    button,
    note,
    max,
    frame: () => frame?.(),
    add: (n) => {
      count += n;
    },
    reset: () => {
      count = 0;
    },
    done: () => {
      complete = true;
    },
  };
}
test('playback state follows start, stop, and button clicks', () => {
  const { site, button } = run();
  assert.equal(site.evaluate('playback.on'), false);
  button.click();
  assert.equal(site.evaluate('playback.on'), true);
  button.click();
  assert.equal(site.evaluate('playback.on'), false);
  site.evaluate('playback.start()');
  assert.equal(button.textContent, 'Pause');
  site.evaluate('playback.stop()');
  assert.equal(button.textContent, 'Run');
});
test('run budgets are additional, resumable, and survive counter resets', () => {
  const r = run('5');
  assert.equal(r.site.evaluate('check.remaining()'), Infinity);
  r.site.evaluate('playback.start()');
  r.add(4);
  assert.equal(r.site.evaluate('check.remaining()'), 1);
  r.add(1);
  assert.equal(r.site.evaluate('check()'), true);
  assert.equal(r.site.evaluate('playback.on'), false);
  assert.match(r.note.textContent, /Stopped after 5/);
  assert.doesNotMatch(r.note.textContent, /converged/i);
  r.site.evaluate('playback.start()');
  assert.equal(r.site.evaluate('check.remaining()'), 5);
  r.add(2);
  r.site.evaluate('check.rebase(0)');
  r.reset();
  assert.equal(r.site.evaluate('check.remaining()'), 3);
  r.done();
  assert.equal(r.site.evaluate('check()'), true);
  assert.equal(r.note.textContent, '');
  r.max.value = '';
  r.site.evaluate('playback.start()');
  assert.equal(r.site.evaluate('check.remaining()'), Infinity);
});

// Exercise production frame callbacks without emulating a browser or canvas.
// Atomic workers stand in for expensive training; the real callback must cap them.
function callbackRun(id, state, workers, count) {
  const r = run('1');
  const lesson = r.site.lessons.find((l) => l.id === id);
  const callback = lesson.init.toString().match(/const anim = Anim\(([\s\S]*?\n {4,6}\})\);/)[1];
  r.site.evaluate(`${state}; var draw=()=>{}, say=()=>{}; ${workers};
    var checkRunStop; var animation = Anim(${callback}); playback = playBtn(button, animation);
    checkRunStop = attachRunLimit(playback,root,'test',{done:()=>false,restart:()=>{},count:()=>${count},what:'tasks'}); playback.start();`);
  return r;
}
test('Few-shot frames cap batches and resume with the same additional budget', () => {
  const r = callbackRun('few-shot', 'var ep=0,fin=false', 'var train=k=>{ep+=k;}', 'ep');
  r.frame();
  assert.equal(r.site.evaluate('ep'), 1);
  assert.equal(r.site.evaluate('playback.on'), false);
  assert.match(r.note.textContent, /Stopped after 1 more task,/);
  r.max.value = '5';
  r.site.evaluate('playback.start()');
  r.frame();
  r.frame();
  assert.equal(r.site.evaluate('ep'), 6);
  assert.equal(r.site.evaluate('playback.on'), false);
});
test('t-SNE stops at the next counted boundary', () => {
  const ts = callbackRun(
    'tsne',
    'var it=40,frameCount=0',
    'var fin=()=>false,iter=()=>{it++;}',
    'it',
  );
  ts.frame();
  assert.equal(ts.site.evaluate('it'), 41);
});
test('fast Q-learning stops at the next episode boundary', () => {
  const q = callbackRun(
    'q-learning',
    'var ep=20,frameCount=0,fin=false,settings={sp:4},lastL=null,transitions=0',
    'var one=()=>{if(++transitions%6===0)ep++;}',
    'ep',
  );
  q.frame();
  assert.equal(q.site.evaluate('ep'), 21);
});
test('automatic neuron growth cannot renew the run budget', () => {
  const r = callbackRun(
    'universal',
    'var ep=40,frameCount=3,conv=false,grow=true,settings={H:1},LL=[],hb={set(){}},run=0,msg=""',
    'var train=k=>{ep+=k;conv=true;LL[ep-1]=1;},build=()=>{ep=0;conv=false;}',
    'ep',
  );
  r.max.value = '5';
  for (let i = 0; i < 12; i++) {
    if (r.site.evaluate('playback.on')) r.frame();
  }
  assert.equal(r.site.evaluate('playback.on'), false);
  assert.match(r.note.textContent, /Stopped after 5/);
});

test('SVM caps actual optimizer iterations', () => {
  const site = loadSite();
  const model = site.lessons
    .find((l) => l.id === 'svm')
    .setup({
      S: { C: 0 },
      train: [
        { x: -1, y: -1, c: 0 },
        { x: 1, y: 1, c: 1 },
      ],
      bind() {},
    });
  model.reset();
  model.tick(1);
  assert.equal(model.count(), 1);
});

test('embedding initialization completes only for reduced motion', () => {
  for (const reducedMotion of [false, true]) {
    const site = loadSite({ reducedMotion });
    for (const id of ['tsne', 'umap']) {
      // Run the real initialization up to animation wiring, without drawing.
      const source = site.lessons.find((l) => l.id === id).init.toString();
      const prefix = source
        .slice(0, source.indexOf('const anim = Anim('))
        .replace(/^init\(root\)\s*\{/, 'function(root) {');
      const result = site.evaluate(
        `(${prefix} return { count:${id === 'tsne' ? 'it' : 'ep'},metric:${id === 'tsne' ? 'KL[KL.length-1]' : 'kept'},done:fin()}; })({querySelector:()=>({})})`,
      );
      if (reducedMotion) {
        assert.equal(result.done, true);
        assert.ok(Number.isFinite(result.metric));
      } else assert.equal(result.count, 0, id);
    }
  }
});
