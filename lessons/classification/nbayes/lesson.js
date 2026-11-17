/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  area,
  clipPlot,
  curve,
  ellPath,
  f2,
  prepareCanvas,
  mix,
  npdf,
  plot,
} from '../../../js/core/shared.js';
import { clfLesson, gfit, glog, sig, sweepTick } from '../shared.js';

export default clfLesson({
  id: 'nbayes',
  still:
    'Training is one pass of arithmetic with no iterations: two averages and two variances per class, and the share of each class.',

  ds: ['blobs', 'overlap', 'moons'],
  S: { corr: 0 },
  setup(api) {
    const S = api.S;
    let G = [],
      pr = 0.5,
      correlationSlider,
      advanceCorrelationSweep;
    const naiveBayesLogScore = (k, x, y) =>
      -0.5 * Math.log(G[k].a * G[k].d) -
      0.5 * ((x - G[k].mx) ** 2 / G[k].a + (y - G[k].my) ** 2 / G[k].d);
    correlationSlider = api.bind('corr', 'corr', f2, () => api.regen());
    advanceCorrelationSweep = sweepTick(
      () => S.corr,
      (v) => {
        S.corr = v;
        correlationSlider.set(v);
        api.regen();
      },
      -0.9,
      0.9,
      9,
    );
    return {
      fit() {
        const T = api.train;
        G = [gfit(T, 0), gfit(T, 1)];
        pr = G[1].n / Math.max(1, T.length);
      },
      p: (x, y) =>
        sig(
          Math.log(pr / (1 - pr) + 1e-12) +
            naiveBayesLogScore(1, x, y) -
            naiveBayesLogScore(0, x, y),
        ),
      tick() {
        advanceCorrelationSweep();
        return false;
      },
      overlay(g, X, Y, t) {
        [t.accent, t.cut].forEach((cl, k) => {
          g.setLineDash([6, 4]);
          g.strokeStyle = cl;
          g.lineWidth = 2;
          [1, 2].forEach((r) => {
            ellPath(g, X, Y, G[k].mx, G[k].my, G[k].a, 0, G[k].d, r);
            g.stroke();
          });
          g.setLineDash([]);
        });
      },
      right(t) {
        const { g: ctx, w, h } = prepareCanvas(api.cr),
          hh = h / 2;
        [
          ['x', (k) => [G[k].mx, Math.sqrt(G[k].a)]],
          ['y', (k) => [G[k].my, Math.sqrt(G[k].d)]],
        ].forEach(([nm, fn], r) => {
          ctx.save();
          ctx.translate(0, r * hh);
          let ym = 0;
          [0, 1].forEach((k) => {
            ym = Math.max(ym, npdf(0, 0, fn(k)[1]));
          });
          const P = plot(ctx, w, hh, [-3, 3], [0, ym * 1.2], t, { xlabel: 'feature ' + nm });
          clipPlot(ctx, P);
          [t.accent, t.gold].forEach((cl, k) => {
            const a = fn(k);
            area(ctx, P, (x) => npdf(x, a[0], a[1]), -3, 3);
            ctx.fillStyle = mix(cl, cl, 0, 0.22);
            ctx.fill();
            curve(ctx, P, (x) => npdf(x, a[0], a[1]), -3, 3);
            ctx.strokeStyle = cl;
            ctx.lineWidth = 2.4;
            ctx.stroke();
          });
          ctx.restore();
          ctx.restore();
        });
      },
      read(set) {
        const te = api.test,
          T = api.train,
          Q = [gfit(T, 0), gfit(T, 1)],
          ok = te.filter(
            (p) =>
              (Math.log(pr / (1 - pr) + 1e-12) + glog(Q[1], p.x, p.y) - glog(Q[0], p.x, p.y) >= 0
                ? 1
                : 0) === p.c,
          ).length;
        set('q', te.length ? ((100 * ok) / te.length).toFixed(1) + '%' : '–');
        set('pr', (100 * pr).toFixed(0) + '%');
      },
    };
  },
});
