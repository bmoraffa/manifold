/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { MONO, SANS, f2, f3, fit } from '../../../js/core/shared.js';
import { clfLesson } from '../shared.js';

export default clfLesson({
  id: 'softmax',
  maxWhat: 'steps',

  ds: ['three'],
  multi: true,
  probe: true,
  stepBtn: true,
  S: { lr: 0.5 },
  n: 150,
  setup(api) {
    const S = api.S;
    let W,
      hist,
      wait = 0;
    const sm = (x, y) => {
      const z = W.map((w) => w[0] + w[1] * x + w[2] * y),
        m = Math.max(...z),
        e = z.map((v) => Math.exp(v - m)),
        s = e[0] + e[1] + e[2];
      return { z, p: e.map((v) => v / s) };
    };
    const reset = () => {
        W = [0, 1, 2].map(() => [0, (Math.random() - 0.5) * 2, (Math.random() - 0.5) * 2]);
        hist = [];
        wait = 0;
      },
      loss = () => {
        const T = api.train;
        return (
          -T.reduce((s, p) => s + Math.log(sm(p.x, p.y).p[p.c] + 1e-12), 0) / Math.max(1, T.length)
        );
      };
    const step = () => {
      const T = api.train,
        G = W.map(() => [0, 0, 0]);
      for (const p of T) {
        const q = sm(p.x, p.y).p;
        for (let k = 0; k < 3; k++) {
          const e = q[k] - (p.c === k ? 1 : 0);
          G[k][0] += e;
          G[k][1] += e * p.x;
          G[k][2] += e * p.y;
        }
      }
      for (let k = 0; k < 3; k++)
        for (let j = 0; j < 3; j++) W[k][j] -= (S.lr * G[k][j]) / Math.max(1, T.length);
      hist.push(loss());
    };
    api.bind('lr', 'lr', f2, () => {});
    return {
      reset,
      p: (x, y) => sm(x, y).p,
      stepOnce: step,
      done: () => wait > 0,
      tick() {
        if (wait > 0) return false;
        step();
        const n = hist.length;
        if (n > 500 || (n > 30 && Math.abs(hist[n - 1] - hist[n - 21]) < 1e-5)) wait = 1;
        return true;
      },
      count: () => hist.length,
      overlay(g, X, Y, t) {
        const q = api.probe;
        g.strokeStyle = t.ink;
        g.lineWidth = 2.5;
        g.beginPath();
        g.arc(X(q[0]), Y(q[1]), 9, 0, 7);
        g.stroke();
        g.beginPath();
        g.moveTo(X(q[0]) - 14, Y(q[1]));
        g.lineTo(X(q[0]) + 14, Y(q[1]));
        g.moveTo(X(q[0]), Y(q[1]) - 14);
        g.lineTo(X(q[0]), Y(q[1]) + 14);
        g.stroke();
      },
      right(t) {
        const { g, w, h } = fit(api.cr),
          q = api.probe,
          r = sm(q[0], q[1]),
          cc = [t.accent, t.gold, t.ok],
          bw = (w - 80) / 3,
          base = h - 46,
          top = 40;
        g.font = MONO;
        g.textAlign = 'center';
        g.strokeStyle = t.muted;
        g.beginPath();
        g.moveTo(30, base);
        g.lineTo(w - 20, base);
        g.stroke();
        [0.5, 1].forEach((v) => {
          g.strokeStyle = t.line;
          g.beginPath();
          g.moveTo(30, base - (base - top) * v);
          g.lineTo(w - 20, base - (base - top) * v);
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText(String(v), 26, base - (base - top) * v + 4);
          g.textAlign = 'center';
        });
        r.p.forEach((v, k) => {
          const x = 50 + k * bw;
          g.fillStyle = cc[k];
          g.fillRect(x, base - (base - top) * v, bw - 22, (base - top) * v);
          g.fillStyle = t.ink;
          g.font = SANS;
          g.fillText((100 * v).toFixed(1) + '%', x + (bw - 22) / 2, base - (base - top) * v - 7);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.fillText('class ' + 'ABC'[k], x + (bw - 22) / 2, base + 15);
          g.fillText('score ' + f2(r.z[k]), x + (bw - 22) / 2, base + 30);
        });
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('pointer at (' + f2(q[0]) + ', ' + f2(q[1]) + ')', 30, 16);
      },
      status() {
        const n = hist.length;
        if (!n) return 'Step 0: random weights. Press Train or One step.';
        return (
          'Step ' +
          n +
          ': all nine weights moved against their gradients. Cross-entropy ' +
          (n > 1 ? f3(hist[n - 2]) + ' → ' : '') +
          f3(hist[n - 1]) +
          '.' +
          (wait > 0 ? ' The loss has stopped falling: training is finished.' : '')
        );
      },
      cur() {
        return wait > 0 ? 4 : hist.length ? 3 : 0;
      },
      read(set) {
        set('ll', f3(loss()));
        set('k', String(hist.length));
      },
    };
  },
});
