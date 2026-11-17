/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $, MONO, clipPlot, fit, mix, plot } from '../../../js/core/shared.js';
import { clfLesson } from '../shared.js';

export default clfLesson({
  id: 'knn',
  still:
    'There is nothing to train: the model is the stored training data. All the work happens when a prediction is asked for, as you can see by moving the pointer or your test point.',

  ds: ['moons', 'blobs', 'overlap', 'circles', 'xor'],
  probe: true,
  S: { k: 5, man: false },
  nc: 44,
  setup(api) {
    const S = api.S;
    let etr = [],
      ete = [],
      dir = 1,
      fr = 0,
      ks;
    const dist = (p, x, y) =>
      S.man ? Math.abs(p.x - x) + Math.abs(p.y - y) : Math.hypot(p.x - x, p.y - y);
    const near = (x, y, k) => {
      const d = api.train.map((p) => [dist(p, x, y), p]);
      d.sort((a, b) => a[0] - b[0]);
      return d.slice(0, k);
    };
    const fitc = () => {
      const T = api.train,
        KM = Math.min(41, T.length);
      etr = [];
      ete = [];
      const run = (P, E) => {
        const cnt = new Array(KM + 1).fill(0);
        for (const p of P) {
          const d = T.map((q) => [dist(q, p.x, p.y), q.c]);
          d.sort((a, b) => a[0] - b[0]);
          let s = 0;
          for (let k = 1; k <= KM; k++) {
            s += d[k - 1][1];
            if ((s / k >= 0.5 ? 1 : 0) !== p.c) cnt[k]++;
          }
        }
        for (let k = 1; k <= KM; k++) E[k] = P.length ? cnt[k] / P.length : NaN;
      };
      run(T, etr);
      run(api.test, ete);
    };
    ks = api.bind(
      'k',
      'k',
      (v) => String(v),
      () => api.redraw(),
    );
    $('#knn-m', api.root).onchange = (e) => {
      S.man = e.target.value === 'm';
      fitc();
      api.redraw();
    };
    return {
      fit: fitc,
      p: (x, y) => {
        const nb = near(x, y, S.k);
        return nb.reduce((s, q) => s + q[1].c, 0) / Math.max(1, nb.length);
      },
      tick() {
        if (++fr % 26) return false;
        let k = S.k + dir * 2;
        if (k > 41) {
          dir = -1;
          k = 39;
        }
        if (k < 1) {
          dir = 1;
          k = 3;
        }
        S.k = k;
        ks.set(k);
        return true;
      },
      overlay(g, X, Y, t, w) {
        const q = api.probe,
          nb = near(q[0], q[1], S.k),
          rad = nb.length ? nb[nb.length - 1][0] : 0,
          s = w / 6;
        g.strokeStyle = t.ink;
        g.lineWidth = 1.6;
        g.beginPath();
        if (S.man) {
          g.moveTo(X(q[0]) + rad * s, Y(q[1]));
          g.lineTo(X(q[0]), Y(q[1]) - rad * s);
          g.lineTo(X(q[0]) - rad * s, Y(q[1]));
          g.lineTo(X(q[0]), Y(q[1]) + rad * s);
          g.closePath();
        } else g.arc(X(q[0]), Y(q[1]), rad * s, 0, 7);
        g.stroke();
        g.lineWidth = 1;
        for (const n of nb) {
          g.beginPath();
          g.moveTo(X(q[0]), Y(q[1]));
          g.lineTo(X(n[1].x), Y(n[1].y));
          g.stroke();
        }
        g.fillStyle = t.ink;
        g.beginPath();
        g.arc(X(q[0]), Y(q[1]), 5, 0, 7);
        g.fill();
      },
      right(t) {
        const { g, w, h } = fit(api.cr),
          KM = etr.length - 1;
        let ym = 0.1;
        for (let k = 1; k <= KM; k++) ym = Math.max(ym, etr[k], ete[k] || 0);
        const P = plot(g, w, h, [0, KM + 1], [0, ym * 1.15], t, { xstep: 5, xlabel: 'k' });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        [0.1, 0.2, 0.3, 0.4, 0.5].forEach((v) => {
          if (v < ym * 1.15) {
            g.strokeStyle = t.line;
            g.beginPath();
            g.moveTo(P.m.l, P.Y(v));
            g.lineTo(w - P.m.r, P.Y(v));
            g.stroke();
            g.fillText(100 * v + '%', P.m.l - 4, P.Y(v) + 4);
          }
        });
        clipPlot(g, P);
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
        g.lineWidth = 9;
        g.beginPath();
        g.moveTo(P.X(S.k), P.Y(0));
        g.lineTo(P.X(S.k), P.Y(ym * 1.15));
        g.stroke();
        [
          [etr, t.accent],
          [ete, t.ok],
        ].forEach(([E, cl]) => {
          g.strokeStyle = cl;
          g.lineWidth = 2.4;
          g.beginPath();
          for (let k = 1; k <= KM; k++)
            k > 1 ? g.lineTo(P.X(k), P.Y(E[k])) : g.moveTo(P.X(k), P.Y(E[k]));
          g.stroke();
        });
        g.restore();
        g.textAlign = 'left';
        g.fillStyle = t.accent;
        g.fillText('train error', P.m.l + 46, P.m.t + 8);
        g.fillStyle = t.ok;
        g.fillText('test error', P.m.l + 46, P.m.t + 23);
      },
      read(set) {
        const q = api.probe,
          nb = near(q[0], q[1], S.k),
          b = nb.reduce((s, n) => s + n[1].c, 0);
        let bk = 1;
        for (let k = 2; k < ete.length; k++) if (ete[k] < ete[bk]) bk = k;
        set('k', String(S.k));
        set('vote', nb.length - b + ' for A, ' + b + ' for B');
        set('bk', String(bk));
      },
    };
  },
});
