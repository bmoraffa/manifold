/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { MONO, clamp, f2, fit, mix } from '../../../js/core/shared.js';
import { clfLesson, sig } from '../../classification/shared.js';
import { errPlot, growTree, pct, treeP } from '../shared.js';

export default clfLesson({
  id: 'adaboost',
  fromZero: true,
  ds: ['moons', 'circles', 'overlap', 'blobs', 'xor'],
  stepBtn: true,
  S: { B: 40, depth: 1 },
  n: 120,
  nc: 46,

  setup(api) {
    const S = api.S;
    let H = [],
      wt = [],
      eTr = [],
      eTe = [],
      wait = 0,
      fr = 0,
      msg = '',
      fresh = true,
      last = null;
    const F = (x, y) => {
      let s = 0;
      for (const q of H) s += q.a * (treeP(q.t, x, y) >= 0.5 ? 1 : -1);
      return s;
    };
    const reset = () => {
      H = [];
      const n = api.train.length;
      wt = api.train.map(() => 1 / n);
      eTr = [];
      eTe = [];
      wait = 0;
      last = null;
      msg = 'Every point has the same weight. Press Add learners or One step.';
    };
    const add = () => {
      const tr = api.train;
      if (tr.length < 4) return;
      if (wt.length !== tr.length) wt = tr.map(() => 1 / tr.length);
      const tree = growTree(
        tr.map((p, i) => ({ x: p.x, y: p.y, c: p.c, w: wt[i] })),
        { depth: S.depth, minleaf: 1, force: true },
      );
      let e = 0;
      const hv = tr.map((p) => (treeP(tree, p.x, p.y) >= 0.5 ? 1 : -1));
      tr.forEach((p, i) => {
        if (hv[i] !== (p.c ? 1 : -1)) e += wt[i];
      });
      const ee = clamp(e, 1e-6, 1 - 1e-6),
        a = 0.5 * Math.log((1 - ee) / ee);
      let z = 0;
      wt = wt.map((w, i) => {
        const v = w * Math.exp(-a * (tr[i].c ? 1 : -1) * hv[i]);
        z += v;
        return v;
      });
      wt = wt.map((w) => w / z);
      H.push({ t: tree, a, e });
      last = tree;
      const er = (P) =>
        P.length ? P.filter((p) => (F(p.x, p.y) >= 0 ? 1 : 0) !== p.c).length / P.length : NaN;
      eTr.push(er(tr));
      eTe.push(er(api.test));
      msg =
        'Learner ' +
        H.length +
        ': ' +
        (tree.sp && S.depth === 1
          ? 'the best single cut on the weighted points is “' +
            tree.sp.f +
            ' ≤ ' +
            tree.sp.thr.toFixed(2) +
            '”. '
          : 'a depth-' + S.depth + ' tree fitted to the weighted points. ') +
        'It gets ' +
        (100 * e).toFixed(1) +
        '% of the weight wrong, so its vote is α = ½ ln(' +
        (1 - ee).toFixed(2) +
        ' / ' +
        ee.toFixed(2) +
        ') = ' +
        f2(a) +
        '. The points it got wrong are made ' +
        f2(Math.exp(2 * a)) +
        ' times heavier relative to the ones it got right. Training error of the combined vote: ' +
        pct(eTr[eTr.length - 1]) +
        '.';
    };
    const rebuild = () => {
      reset();
      for (let i = 0; i < S.B; i++) add();
      api.redraw();
    };
    api.bind('B', 'B', (v) => String(v), rebuild);
    api.bind('depth', 'depth', (v) => String(v), rebuild);
    return {
      reset,
      fit() {
        const k = fresh ? S.B : H.length;
        fresh = false;
        reset();
        for (let i = 0; i < k; i++) add();
      },
      p: (x, y) => (H.length ? sig(1.6 * F(x, y)) : 0.5),
      stepOnce() {
        if (H.length >= S.B) reset();
        add();
      },
      done: () => wait > 0,
      tick() {
        fr++;
        if (wait > 0) return false;
        if (fr % (H.length < 6 ? 75 : 30)) return false;
        if (H.length < S.B) add();
        else wait = 240;
        return true;
      },
      status: () => msg,
      cur: () => (!H.length ? 0 : 4),
      overlay(g, X, Y, t) {
        const tr = api.train,
          n = tr.length;
        g.strokeStyle = t.ink;
        g.lineWidth = 1.2;
        tr.forEach((p, i) => {
          const r = 4 + Math.min(22, 5 * Math.sqrt(Math.max(0, (wt[i] || 0) * n - 0.6)));
          if (r > 6) {
            g.beginPath();
            g.arc(X(p.x), Y(p.y), r, 0, 7);
            g.stroke();
          }
        });
        if (last && last.sp && S.depth === 1) {
          g.setLineDash([7, 5]);
          g.lineWidth = 2.4;
          g.beginPath();
          if (last.sp.f === 'x') {
            g.moveTo(X(last.sp.thr), Y(-3));
            g.lineTo(X(last.sp.thr), Y(3));
          } else {
            g.moveTo(X(-3), Y(last.sp.thr));
            g.lineTo(X(3), Y(last.sp.thr));
          }
          g.stroke();
          g.setLineDash([]);
        }
      },
      right(t) {
        const { g, w, h } = fit(api.cr),
          h1 = h * 0.42,
          m = { l: 38, r: 12 },
          n = Math.max(8, S.B),
          bw = (w - m.l - m.r) / n;
        let am = 0.3;
        for (const q of H) am = Math.max(am, Math.abs(q.a));
        const base = h1 - 16,
          Yb = (v) => base - (v / am) * (base - 18);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, base);
        g.lineTo(w - m.r, base);
        g.stroke();
        H.forEach((q, i) => {
          g.fillStyle = i === H.length - 1 ? t.gold : mix(t.gold, t.gold, 0, 0.45);
          g.fillRect(
            m.l + i * bw + 0.5,
            Math.min(base, Yb(q.a)),
            Math.max(1, bw - 1.5),
            Math.abs(base - Yb(q.a)),
          );
        });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('vote weight α of each learner (latest in bold gold)', m.l, 11);
        errPlot(
          g,
          w,
          h,
          h1,
          t,
          S.B,
          [
            [eTr, t.accent, 'training error'],
            [eTe, t.ok, 'test error'],
          ],
          'learners',
        );
      },
      read(set) {
        const k = H.length,
          n = wt.length;
        set('k', k + ' of ' + S.B);
        set('er', k ? pct(H[k - 1].e) : '–');
        set('al', k ? f2(H[k - 1].a) : '–');
        set('mw', n ? (Math.max(...wt) * n).toFixed(1) + ' ×' : '–');
      },
    };
  },
});
