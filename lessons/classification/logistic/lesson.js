/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  MONO,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  prepareCanvas,
  mix,
  plot,
} from '../../../js/core/shared.js';
import { clfLesson, sig } from '../shared.js';
import { dot } from '../../regression/shared.js';

export default clfLesson({
  id: 'logistic',
  maxWhat: 'steps',

  ds: ['blobs', 'overlap', 'moons', 'circles', 'xor'],
  stepBtn: true,
  S: { lr: 0.5, lam: 0, quad: false },
  setup(api) {
    const S = api.S;
    let w,
      lossHistory,
      trainingFinished = 0;
    const phi = (x, y) => (S.quad ? [1, x, y, x * x, x * y, y * y] : [1, x, y]);
    const reset = () => {
      w = phi(0, 0).map((_, j) => (j ? (Math.random() - 0.5) * 2.4 : 0));
      lossHistory = [];
      trainingFinished = 0;
    };
    const loss = () => {
      const T = api.train;
      let L = 0;
      for (const p of T) {
        const q = sig(dot(w, phi(p.x, p.y)));
        L -= p.c ? Math.log(q + 1e-12) : Math.log(1 - q + 1e-12);
      }
      return L / Math.max(1, T.length);
    };
    const step = () => {
      const T = api.train,
        g = w.map(() => 0);
      for (const p of T) {
        const f = phi(p.x, p.y),
          e = sig(dot(w, f)) - p.c;
        for (let j = 0; j < w.length; j++) g[j] += e * f[j];
      }
      for (let j = 0; j < w.length; j++)
        w[j] -= S.lr * (g[j] / Math.max(1, T.length) + (j ? S.lam * w[j] : 0));
      lossHistory.push(loss());
    };
    api.bind('lr', 'lr', f2, () => {});
    api.bind('lam', 'lam', f3, () => {});
    $('#logistic-quad', api.root).onchange = (e) => {
      S.quad = e.target.checked;
      reset();
      api.redraw();
    };
    return {
      reset,
      p: (x, y) => sig(dot(w, phi(x, y))),
      stepOnce: step,
      done: () => trainingFinished > 0,
      tick() {
        if (trainingFinished > 0) return false;
        step();
        const n = lossHistory.length;
        if (n > 600 || (n > 30 && Math.abs(lossHistory[n - 1] - lossHistory[n - 21]) < 1e-5))
          trainingFinished = 1;
        return true;
      },
      count: () => lossHistory.length,
      right(t) {
        const { g: ctx, w: cw, h } = prepareCanvas(api.cr),
          P = plot(ctx, cw, h, [-8, 8], [-0.12, 1.12], t, { xstep: 2, xlabel: 'score z' });
        ctx.font = MONO;
        ctx.fillStyle = t.muted;
        ctx.textAlign = 'right';
        [0, 0.5, 1].forEach((v) => {
          ctx.strokeStyle = t.line;
          ctx.beginPath();
          ctx.moveTo(P.m.l, P.Y(v));
          ctx.lineTo(cw - P.m.r, P.Y(v));
          ctx.stroke();
          ctx.fillText(String(v), P.m.l - 6, P.Y(v) + 4);
        });
        clipPlot(ctx, P);
        for (const p of api.train) {
          const z = clamp(dot(w, phi(p.x, p.y)), -7.9, 7.9),
            col = p.c ? t.gold : t.accent;
          ctx.strokeStyle = mix(col, col, 0, 0.35);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(P.X(z), P.Y(p.c));
          ctx.lineTo(P.X(z), P.Y(sig(z)));
          ctx.stroke();
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.arc(P.X(z), P.Y(p.c), 3.6, 0, 7);
          ctx.fill();
        }
        curve(ctx, P, sig, -8, 8);
        ctx.strokeStyle = t.ink;
        ctx.lineWidth = 2.6;
        ctx.stroke();
        ctx.restore();
        ctx.textAlign = 'left';
        ctx.fillStyle = t.muted;
        ctx.fillText('dots at their true label; line = how wrong p is', P.m.l + 8, P.Y(0.5) - 6);
      },
      status() {
        const n = lossHistory.length;
        if (!n) return 'Step 0: a random starting line. Press Train or One step.';
        const d = n > 1 ? lossHistory[n - 2] - lossHistory[n - 1] : 0;
        return (
          'Step ' +
          n +
          ': every weight moved against its gradient. Log loss ' +
          (n > 1 ? f3(lossHistory[n - 2]) + ' → ' : '') +
          f3(lossHistory[n - 1]) +
          (n > 1
            ? d >= 0
              ? ', down by ' + d.toFixed(4)
              : ', UP by ' + (-d).toFixed(4) + ' (the step is too large)'
            : '') +
          '. The line is now ' +
          f2(w[0]) +
          ' + ' +
          f2(w[1]) +
          '·x + ' +
          f2(w[2]) +
          '·y = 0.' +
          (trainingFinished > 0 ? ' The loss has stopped falling: training is finished.' : '')
        );
      },
      cur() {
        return trainingFinished > 0 ? 4 : lossHistory.length ? 3 : 0;
      },
      read(set) {
        set('ll', f3(loss()));
        set('k', String(lossHistory.length));
        set('w', w.slice(0, 3).map(f2).join(', ') + (w.length > 3 ? ' …' : ''));
      },
    };
  },
});
