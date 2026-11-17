/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $, MONO, clamp, prepareCanvas, mix } from '../../../js/core/shared.js';
import { Orbit, dot, solve } from '../../regression/shared.js';
import { clfLesson, sig } from '../shared.js';

export default clfLesson({
  id: 'kernel',
  still:
    'Training here is ordinary logistic regression, run on three coordinates instead of two (or, for the RBF kernel, one linear solve for the bump weights). The lesson is in the lift, not the fitting.',

  ds: ['circles', 'xor', 'moons', 'blobs'],
  r3d: true,
  S: { lift: 'r2', gam: 1 },
  n: 140,
  nc: 46,
  setup(api) {
    const S = api.S,
      root = api.root;
    let w = [0, 0, 0, 0],
      al = [],
      Tr = [],
      linearWeights = [0, 0, 0],
      u = 0,
      wait = 0;
    const orb = Orbit(api.cr, () => api.redraw(), root.querySelector('.viewbar'), {
      az: -0.55,
      el: 0.36,
      zoom: 1,
    });
    const liftedFeature = (x, y) =>
        S.lift === 'r2'
          ? (x * x + y * y) / 3.4
          : S.lift === 'xy'
            ? (x * y) / 2.4
            : S.lift === 'bump'
              ? 2.4 * Math.exp(-(x * x + y * y) / 1.6)
              : 0,
      gam = () => Math.pow(10, S.gam);
    const logit = (F, T) => {
      const v = F(0, 0).map(() => 0);
      for (let k = 0; k < 700; k++) {
        const g = v.map(() => 0);
        for (const p of T) {
          const f = F(p.x, p.y),
            e = sig(dot(v, f)) - p.c;
          for (let j = 0; j < v.length; j++) g[j] += e * f[j];
        }
        for (let j = 0; j < v.length; j++) v[j] -= 0.6 * (g[j] / T.length + (j ? 0.002 * v[j] : 0));
      }
      return v;
    };
    const kernelScore = (x, y) => {
      let s = 0;
      const gm = gam();
      for (let i = 0; i < Tr.length; i++)
        s += al[i] * Math.exp(-gm * ((Tr[i].x - x) ** 2 + (Tr[i].y - y) ** 2));
      return s;
    };
    $('#kernel-lift', root).onchange = (e) => {
      S.lift = e.target.value;
      u = 0;
      wait = 0;
      api.refit();
    };
    api.bind('gam', 'gam', (v) => Math.pow(10, v).toFixed(2));
    return {
      fit() {
        Tr = api.train;
        if (!Tr.length) return;
        linearWeights = logit((x, y) => [1, x, y], Tr);
        if (S.lift === 'kern') {
          const n = Tr.length,
            gm = gam(),
            K = Tr.map((a, i) =>
              Tr.map(
                (b, j) =>
                  Math.exp(-gm * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2)) + (i === j ? 0.15 : 0),
              ),
            );
          al = solve(
            K,
            Tr.map((p) => (p.c ? 1 : -1)),
          );
        } else
          w =
            S.lift === 'none'
              ? [...linearWeights, 0]
              : logit((x, y) => [1, x, y, liftedFeature(x, y)], Tr);
      },
      p: (x, y) =>
        S.lift === 'kern'
          ? sig(3.5 * kernelScore(x, y))
          : sig(w[0] + w[1] * x + w[2] * y + w[3] * liftedFeature(x, y)),
      tick() {
        if (wait > 0) {
          wait--;
          if (!wait) u = 0;
          return true;
        }
        u = Math.min(1, u + 0.007);
        if (u >= 1) wait = 260;
        return true;
      },
      start() {
        orb.spin(true);
      },
      stop() {
        orb.spin(false);
      },
      right(t) {
        const { g: ctx, w: cw, h } = prepareCanvas(api.cr),
          pr = orb.cam(cw, h, 8.4),
          R = 3,
          e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u),
          kern = S.lift === 'kern',
          hz = (x, y) => (kern ? clamp(kernelScore(x, y), -1.6, 1.6) * e : liftedFeature(x, y) * e),
          path = (ps) => {
            ctx.beginPath();
            ps.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
          };
        ctx.strokeStyle = t.line;
        ctx.lineWidth = 1;
        for (let k = -3; k <= 3; k++) {
          path([pr(k, -R, 0), pr(k, R, 0)]);
          ctx.stroke();
          path([pr(-R, k, 0), pr(R, k, 0)]);
          ctx.stroke();
        }
        const N = kern ? 26 : 16,
          Q = [];
        for (let i = 0; i < N; i++)
          for (let j = 0; j < N; j++) {
            const a0 = -R + (2 * R * i) / N,
              a1 = -R + (2 * R * (i + 1)) / N,
              b0 = -R + (2 * R * j) / N,
              b1 = -R + (2 * R * (j + 1)) / N,
              z = hz((a0 + a1) / 2, (b0 + b1) / 2),
              ps = [
                pr(a0, b0, hz(a0, b0)),
                pr(a1, b0, hz(a1, b0)),
                pr(a1, b1, hz(a1, b1)),
                pr(a0, b1, hz(a0, b1)),
              ];
            Q.push({ ps, z, d: (ps[0][2] + ps[2][2]) / 2 });
          }
        Q.sort((a, b) => b.d - a.d);
        for (const q of Q) {
          path(q.ps);
          ctx.closePath();
          if (kern) {
            ctx.fillStyle = mix(q.z >= 0 ? t.gold : t.accent, t.bg, 0.35, 0.55);
            ctx.fill();
          }
          ctx.strokeStyle = mix(t.ink, t.ink, 0, 0.16);
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
        const zp = (x, y) =>
            kern ? 0 : Math.abs(w[3]) > 1e-6 ? -(w[0] + w[1] * x + w[2] * y) / w[3] : NaN,
          pa = clamp((e - 0.72) / 0.28, 0, 1);
        if (pa > 0 && (kern || isFinite(zp(0, 0)))) {
          const M2 = 8;
          for (let i = 0; i < M2; i++)
            for (let j = 0; j < M2; j++) {
              const a0 = -R + (2 * R * i) / M2,
                a1 = -R + (2 * R * (i + 1)) / M2,
                b0 = -R + (2 * R * j) / M2,
                b1 = -R + (2 * R * (j + 1)) / M2,
                zs = [zp(a0, b0), zp(a1, b0), zp(a1, b1), zp(a0, b1)];
              if (zs.some((v) => v < -1.8 || v > 3.4)) continue;
              path([pr(a0, b0, zs[0]), pr(a1, b0, zs[1]), pr(a1, b1, zs[2]), pr(a0, b1, zs[3])]);
              ctx.closePath();
              ctx.fillStyle = mix(t.ok, t.ok, 0, 0.2 * pa);
              ctx.fill();
              ctx.strokeStyle = mix(t.ok, t.ok, 0, 0.5 * pa);
              ctx.lineWidth = 1;
              ctx.stroke();
            }
        }
        api.pts
          .map((p) => ({ p, q: pr(p.x, p.y, hz(p.x, p.y) + 0.03) }))
          .sort((a, b) => b.q[2] - a.q[2])
          .forEach((o) => {
            ctx.fillStyle = o.p.c ? t.gold : t.accent;
            ctx.strokeStyle = t.bg;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(o.q[0], o.q[1], 4, 0, 7);
            ctx.fill();
            ctx.stroke();
          });
        ctx.font = MONO;
        ctx.fillStyle = t.ok;
        ctx.textAlign = 'left';
        if (pa > 0)
          ctx.fillText(
            kern ? 'burgundy sheet: height zero, the boundary' : 'burgundy sheet: the flat cut',
            8,
            14,
          );
      },
      read(set) {
        const te = api.test,
          ok = te.filter(
            (p) =>
              (linearWeights[0] + linearWeights[1] * p.x + linearWeights[2] * p.y >= 0 ? 1 : 0) ===
              p.c,
          ).length;
        set('lin', te.length ? ((100 * ok) / te.length).toFixed(1) + '%' : '–');
        set(
          'what',
          S.lift === 'none'
            ? 'A line (no lift)'
            : S.lift === 'kern'
              ? 'Where a sum of ' + Tr.length + ' bumps crosses zero'
              : 'A flat plane in 3D',
        );
      },
    };
  },
});
