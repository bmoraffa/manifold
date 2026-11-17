/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { MONO, clamp, clipPlot, f2, fit, mix, plot } from '../../../js/core/shared.js';
import { clfLesson, lineAt, sig } from '../shared.js';

export default clfLesson({
  id: 'svm',
  maxWhat: 'iterations',
  stepBtn: true,

  ds: ['blobs', 'overlap', 'moons'],
  S: { C: 0 },
  n: 80,
  setup(api) {
    const S = api.S;
    let w,
      t0,
      wait = 0;
    const C = () => Math.pow(10, S.C),
      f = (p) => w[0] + w[1] * p.x + w[2] * p.y,
      reset = () => {
        w = [0, 0.01, 0.01];
        t0 = 0;
        wait = 0;
      };
    const it = () => {
      const T = api.train,
        g = [0, w[1], w[2]],
        c = C();
      for (const p of T) {
        const y = p.c ? 1 : -1;
        if (y * f(p) < 1) {
          g[0] -= c * y;
          g[1] -= c * y * p.x;
          g[2] -= c * y * p.y;
        }
      }
      const lr = 0.25 / (1 + c * T.length) / (1 + 0.003 * t0);
      t0++;
      for (let j = 0; j < 3; j++) w[j] -= lr * g[j];
    };
    api.bind(
      'cost',
      'C',
      (v) => (Math.pow(10, v) < 10 ? Math.pow(10, v).toFixed(2) : Math.pow(10, v).toFixed(0)),
      () => {
        reset();
        api.refit();
      },
    );
    return {
      reset,
      fit() {
        for (let k = 0; k < 400; k++) it();
      },
      stepOnce() {
        for (let k = 0; k < 12; k++) it();
      },
      status() {
        const T = api.train;
        let sv = 0;
        for (const p of T) if ((p.c ? 1 : -1) * f(p) < 1) sv++;
        return (
          'Iteration ' +
          t0 +
          ': ' +
          sv +
          ' of ' +
          T.length +
          ' points are inside the street or on the wrong side, and only those pull on the line. The street is ' +
          f2(2 / (Math.hypot(w[1], w[2]) || 1e-9)) +
          ' wide. The line is ' +
          f2(w[0]) +
          ' + ' +
          f2(w[1]) +
          '·x + ' +
          f2(w[2]) +
          '·y = 0.' +
          (wait > 0 ? ' The pull and the shrink are in balance: training is finished.' : '')
        );
      },
      cur() {
        return wait > 0 ? 4 : 2;
      },
      p: (x, y) => sig(2.2 * (w[0] + w[1] * x + w[2] * y)),
      done: () => wait > 0,
      tick(budget = Infinity) {
        if (wait > 0) return false;
        for (let k = 0; k < Math.min(12, budget); k++) it();
        if (t0 > 2600) wait = 200;
        return true;
      },
      count: () => t0,
      overlay(g, X, Y, t) {
        g.setLineDash([6, 5]);
        g.strokeStyle = t.ink;
        g.lineWidth = 1.5;
        lineAt(g, X, Y, w, 1);
        lineAt(g, X, Y, w, -1);
        g.setLineDash([]);
        g.strokeStyle = t.ink;
        g.lineWidth = 2.4;
        for (const p of api.train) {
          if ((p.c ? 1 : -1) * f(p) <= 1.03) {
            g.beginPath();
            g.arc(X(p.x), Y(p.y), 9, 0, 7);
            g.stroke();
          }
        }
      },
      right(t) {
        const { g, w: cw, h } = fit(api.cr),
          P = plot(g, cw, h, [-3, 5], [0, 4], t, {
            yticks: [1, 2, 3],
            xlabel: 'y · f(x): distance on the correct side',
          });
        clipPlot(g, P);
        g.fillStyle = mix(t.ok, t.ok, 0, 0.08);
        g.fillRect(P.X(-3), P.Y(4), P.X(1) - P.X(-3), P.Y(0) - P.Y(4));
        g.strokeStyle = t.ink;
        g.lineWidth = 2.6;
        g.beginPath();
        g.moveTo(P.X(-3), P.Y(4));
        g.lineTo(P.X(1), P.Y(0));
        g.lineTo(P.X(5), P.Y(0));
        g.stroke();
        g.setLineDash([4, 4]);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        [0, 1].forEach((v) => {
          g.beginPath();
          g.moveTo(P.X(v), P.Y(0));
          g.lineTo(P.X(v), P.Y(4));
          g.stroke();
        });
        g.setLineDash([]);
        for (const p of api.train) {
          const m = clamp((p.c ? 1 : -1) * f(p), -2.95, 4.95);
          g.fillStyle = p.c ? t.gold : t.accent;
          g.beginPath();
          g.arc(P.X(m), P.Y(Math.max(0, 1 - m)), 4, 0, 7);
          g.fill();
        }
        g.restore();
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('wrong side', P.X(-1.5), P.m.t + 8);
        g.fillText('in the street', P.X(0.5), P.m.t + 22);
        g.fillText('safe: costs nothing', P.X(3), P.m.t + 8);
      },
      read(set) {
        const T = api.train;
        let hl = 0,
          sv = 0;
        for (const p of T) {
          const m = (p.c ? 1 : -1) * f(p);
          hl += Math.max(0, 1 - m);
          if (m <= 1.03) sv++;
        }
        set('mw', f2(2 / (Math.hypot(w[1], w[2]) || 1e-9)));
        set('sv', sv + ' of ' + T.length);
        set('hl', f2(hl));
      },
    };
  },
});
