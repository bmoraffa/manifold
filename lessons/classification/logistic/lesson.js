/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  MONO,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  fit,
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
      hist,
      wait = 0;
    const phi = (x, y) => (S.quad ? [1, x, y, x * x, x * y, y * y] : [1, x, y]);
    const reset = () => {
      w = phi(0, 0).map((_, j) => (j ? (Math.random() - 0.5) * 2.4 : 0));
      hist = [];
      wait = 0;
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
      hist.push(loss());
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
      done: () => wait > 0,
      tick() {
        if (wait > 0) return false;
        step();
        const n = hist.length;
        if (n > 600 || (n > 30 && Math.abs(hist[n - 1] - hist[n - 21]) < 1e-5)) wait = 1;
        return true;
      },
      count: () => hist.length,
      right(t) {
        const { g, w: cw, h } = fit(api.cr),
          P = plot(g, cw, h, [-8, 8], [-0.12, 1.12], t, { xstep: 2, xlabel: 'score z' });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        [0, 0.5, 1].forEach((v) => {
          g.strokeStyle = t.line;
          g.beginPath();
          g.moveTo(P.m.l, P.Y(v));
          g.lineTo(cw - P.m.r, P.Y(v));
          g.stroke();
          g.fillText(String(v), P.m.l - 6, P.Y(v) + 4);
        });
        clipPlot(g, P);
        for (const p of api.train) {
          const z = clamp(dot(w, phi(p.x, p.y)), -7.9, 7.9),
            col = p.c ? t.gold : t.accent;
          g.strokeStyle = mix(col, col, 0, 0.35);
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(P.X(z), P.Y(p.c));
          g.lineTo(P.X(z), P.Y(sig(z)));
          g.stroke();
          g.fillStyle = col;
          g.beginPath();
          g.arc(P.X(z), P.Y(p.c), 3.6, 0, 7);
          g.fill();
        }
        curve(g, P, sig, -8, 8);
        g.strokeStyle = t.ink;
        g.lineWidth = 2.6;
        g.stroke();
        g.restore();
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('dots at their true label; line = how wrong p is', P.m.l + 8, P.Y(0.5) - 6);
      },
      status() {
        const n = hist.length;
        if (!n) return 'Step 0: a random starting line. Press Train or One step.';
        const d = n > 1 ? hist[n - 2] - hist[n - 1] : 0;
        return (
          'Step ' +
          n +
          ': every weight moved against its gradient. Log loss ' +
          (n > 1 ? f3(hist[n - 2]) + ' → ' : '') +
          f3(hist[n - 1]) +
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
          (wait > 0 ? ' The loss has stopped falling: training is finished.' : '')
        );
      },
      cur() {
        return wait > 0 ? 4 : hist.length ? 3 : 0;
      },
      read(set) {
        set('ll', f3(loss()));
        set('k', String(hist.length));
        set('w', w.slice(0, 3).map(f2).join(', ') + (w.length > 3 ? ' …' : ''));
      },
    };
  },
});
