/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { MONO, clamp, clipPlot, f2, prepareCanvas, mix, plot } from '../../../js/core/shared.js';
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
      iterationCount,
      trainingFinished = 0;
    const C = () => Math.pow(10, S.C),
      f = (p) => w[0] + w[1] * p.x + w[2] * p.y,
      reset = () => {
        w = [0, 0.01, 0.01];
        iterationCount = 0;
        trainingFinished = 0;
      };
    const gradientStep = () => {
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
      const lr = 0.25 / (1 + c * T.length) / (1 + 0.003 * iterationCount);
      iterationCount++;
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
        for (let k = 0; k < 400; k++) gradientStep();
      },
      stepOnce() {
        for (let k = 0; k < 12; k++) gradientStep();
      },
      status() {
        const T = api.train;
        let sv = 0;
        for (const p of T) if ((p.c ? 1 : -1) * f(p) < 1) sv++;
        return (
          'Iteration ' +
          iterationCount +
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
          (trainingFinished > 0
            ? ' The pull and the shrink are in balance: training is finished.'
            : '')
        );
      },
      cur() {
        return trainingFinished > 0 ? 4 : 2;
      },
      p: (x, y) => sig(2.2 * (w[0] + w[1] * x + w[2] * y)),
      done: () => trainingFinished > 0,
      tick(budget = Infinity) {
        if (trainingFinished > 0) return false;
        for (let k = 0; k < Math.min(12, budget); k++) gradientStep();
        if (iterationCount > 2600) trainingFinished = 200;
        return true;
      },
      count: () => iterationCount,
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
        const { g: ctx, w: cw, h } = prepareCanvas(api.cr),
          P = plot(ctx, cw, h, [-3, 5], [0, 4], t, {
            yticks: [1, 2, 3],
            xlabel: 'y · f(x): distance on the correct side',
          });
        clipPlot(ctx, P);
        ctx.fillStyle = mix(t.ok, t.ok, 0, 0.08);
        ctx.fillRect(P.X(-3), P.Y(4), P.X(1) - P.X(-3), P.Y(0) - P.Y(4));
        ctx.strokeStyle = t.ink;
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(P.X(-3), P.Y(4));
        ctx.lineTo(P.X(1), P.Y(0));
        ctx.lineTo(P.X(5), P.Y(0));
        ctx.stroke();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = t.muted;
        ctx.lineWidth = 1;
        [0, 1].forEach((v) => {
          ctx.beginPath();
          ctx.moveTo(P.X(v), P.Y(0));
          ctx.lineTo(P.X(v), P.Y(4));
          ctx.stroke();
        });
        ctx.setLineDash([]);
        for (const p of api.train) {
          const m = clamp((p.c ? 1 : -1) * f(p), -2.95, 4.95);
          ctx.fillStyle = p.c ? t.gold : t.accent;
          ctx.beginPath();
          ctx.arc(P.X(m), P.Y(Math.max(0, 1 - m)), 4, 0, 7);
          ctx.fill();
        }
        ctx.restore();
        ctx.font = MONO;
        ctx.fillStyle = t.muted;
        ctx.textAlign = 'center';
        ctx.fillText('wrong side', P.X(-1.5), P.m.t + 8);
        ctx.fillText('in the street', P.X(0.5), P.m.t + 22);
        ctx.fillText('safe: costs nothing', P.X(3), P.m.t + 8);
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
