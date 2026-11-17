/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $, clamp, ellPath, f2, prepareCanvas, mix } from '../../../js/core/shared.js';
import { Orbit } from '../../regression/shared.js';
import { clfLesson, gfit, glog, sig, sweepTick } from '../shared.js';

export default clfLesson({
  id: 'lda',
  still:
    'Training is one pass of arithmetic with no iterations: a mean per class and a covariance, shared (LDA) or one per class (QDA).',

  ds: ['blobs', 'overlap', 'moons', 'circles'],
  r3d: true,
  S: { kind: 'lda', sb: 1, corr: 0.5 },
  setup(api) {
    const S = api.S,
      root = api.root;
    let classGaussians = [],
      alternateClassGaussians = [],
      pr = 0.5,
      ss,
      tk;
    const orb = Orbit(api.cr, () => api.redraw(), root.querySelector('.viewbar'), {
      az: -0.5,
      el: 0.6,
      zoom: 1,
    });
    ss = api.bind('sb', 'sb', f2, () => api.regen());
    api.bind('corr', 'corr', f2, () => api.regen());
    tk = sweepTick(
      () => S.sb,
      (v) => {
        S.sb = v;
        ss.set(v);
        api.regen();
      },
      0.45,
      2.3,
      9,
    );
    $('#lda-kind', root).addEventListener('click', (e) => {
      const k = e.target.dataset && e.target.dataset.k;
      if (k) {
        S.kind = k;
        api.refit();
      }
    });
    const model = (kind) => {
        const T = api.train,
          A = gfit(T, 0),
          B = gfit(T, 1);
        if (kind === 'lda') {
          const n = A.n + B.n || 1,
            a = (A.n * A.a + B.n * B.a) / n,
            b = (A.n * A.b + B.n * B.b) / n,
            d = (A.n * A.d + B.n * B.d) / n;
          return [
            { ...A, a, b, d },
            { ...B, a, b, d },
          ];
        }
        return [A, B];
      },
      classLogOdds = (Q, x, y) =>
        Math.log(pr / (1 - pr) + 1e-12) + glog(Q[1], x, y) - glog(Q[0], x, y);
    return {
      fit() {
        const T = api.train;
        pr = clamp(T.filter((p) => p.c).length / Math.max(1, T.length), 0.01, 0.99);
        classGaussians = model(S.kind);
        alternateClassGaussians = model(S.kind === 'lda' ? 'qda' : 'lda');
      },
      p: (x, y) => sig(classLogOdds(classGaussians, x, y)),
      tick() {
        tk();
        return false;
      },
      start() {
        orb.spin(true);
      },
      stop() {
        orb.spin(false);
      },
      overlay(g, X, Y, t) {
        [t.accent, t.cut].forEach((cl, k) => {
          g.setLineDash([6, 4]);
          g.strokeStyle = cl;
          g.lineWidth = 2;
          [1, 2].forEach((r) => {
            ellPath(
              g,
              X,
              Y,
              classGaussians[k].mx,
              classGaussians[k].my,
              classGaussians[k].a,
              classGaussians[k].b,
              classGaussians[k].d,
              r,
            );
            g.stroke();
          });
          g.setLineDash([]);
        });
      },
      right(t) {
        const { g: ctx, w, h } = prepareCanvas(api.cr),
          pr3 = orb.cam(w, h, 8.6),
          N = 34,
          R = 3,
          den = (k, x, y) =>
            ((k ? pr : 1 - pr) * Math.exp(glog(classGaussians[k], x, y))) / (2 * Math.PI);
        let mx = 1e-9;
        const Z = [],
          Pp = [],
          Wn = [];
        for (let i = 0; i <= N; i++) {
          Z.push([]);
          Wn.push([]);
          for (let j = 0; j <= N; j++) {
            const x = -R + (2 * R * i) / N,
              y = -R + (2 * R * j) / N,
              a = den(0, x, y),
              b = den(1, x, y);
            Z[i].push(Math.max(a, b));
            Wn[i].push(b > a ? 1 : 0);
            mx = Math.max(mx, a, b);
          }
        }
        for (let i = 0; i <= N; i++) {
          Pp.push([]);
          for (let j = 0; j <= N; j++)
            Pp[i].push(pr3(-R + (2 * R * i) / N, -R + (2 * R * j) / N, (2.6 * Z[i][j]) / mx - 0.9));
        }
        const Q = [];
        for (let i = 0; i < N; i++)
          for (let j = 0; j < N; j++) Q.push({ i, j, d: (Pp[i][j][2] + Pp[i + 1][j + 1][2]) / 2 });
        Q.sort((a, b) => b.d - a.d);
        ctx.lineWidth = 0.6;
        for (const q of Q) {
          const z = (Z[q.i][q.j] + Z[q.i + 1][q.j + 1]) / (2 * mx),
            cl = Wn[q.i][q.j] ? t.gold : t.accent;
          ctx.beginPath();
          ctx.moveTo(Pp[q.i][q.j][0], Pp[q.i][q.j][1]);
          ctx.lineTo(Pp[q.i + 1][q.j][0], Pp[q.i + 1][q.j][1]);
          ctx.lineTo(Pp[q.i + 1][q.j + 1][0], Pp[q.i + 1][q.j + 1][1]);
          ctx.lineTo(Pp[q.i][q.j + 1][0], Pp[q.i][q.j + 1][1]);
          ctx.closePath();
          ctx.fillStyle = mix(t.bg, cl, 0.18 + 0.82 * Math.pow(z, 0.6));
          ctx.fill();
          ctx.strokeStyle = mix(t.ink, t.ink, 0, 0.14);
          ctx.stroke();
        }
      },
      read(set) {
        const te = api.test,
          ok = te.filter(
            (p) => (classLogOdds(alternateClassGaussians, p.x, p.y) >= 0 ? 1 : 0) === p.c,
          ).length;
        set('np', S.kind === 'lda' ? '4 means + 3 shared = 7' : '4 means + 3 + 3 = 10');
        set(
          'ot',
          (S.kind === 'lda' ? 'QDA: ' : 'LDA: ') +
            (te.length ? ((100 * ok) / te.length).toFixed(1) + '%' : '–'),
        );
        root
          .querySelectorAll('#lda-kind button')
          .forEach((b) =>
            b.setAttribute('aria-pressed', b.dataset.k === S.kind ? 'true' : 'false'),
          );
      },
    };
  },
});
