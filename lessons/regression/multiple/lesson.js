/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  T,
  bind,
  clamp,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  ramp,
  randnFrom,
  rng,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { Orbit, dot, fpval, lsq, pfmt } from '../shared.js';

export default {
  init(root) {
    const c = $('#mr-c', root),
      ce = $('#mr-e', root),
      S = { b: [0.5, 1.2, -0.8], noise: 0.8, r: 0, n: 60, res: true };
    let seed = 5,
      X = [],
      y = [],
      beta = [0, 0, 0],
      ym = 0,
      u = 0,
      wait = 0,
      sim = [],
      pb,
      draw;
    const orb = Orbit(c, () => draw(), $('.viewbar', root));
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r),
        q = Math.sqrt(1 - S.r * S.r);
      X = [];
      y = [];
      for (let i = 0; i < S.n; i++) {
        const a = clamp(rn(), -2.4, 2.4),
          b = clamp(S.r * a + q * rn(), -2.4, 2.4);
        X.push([1, a, b]);
        y.push(S.b[0] + S.b[1] * a + S.b[2] * b + S.noise * rn());
      }
      beta = lsq(X, y, 1e-9);
      ym = y.reduce((s, v) => s + v, 0) / y.length;
      const r2 = rng(77),
        rn2 = randnFrom(r2);
      sim = [];
      for (let k = 0; k < 200; k++) {
        const yy = X.map((row) => S.b[0] + S.b[1] * row[1] + S.b[2] * row[2] + S.noise * rn2());
        const bb = lsq(X, yy, 1e-9);
        sim.push([bb[1], bb[2]]);
      }
    };
    gen();
    const tp = testPoint(
      root,
      'mr',
      ['Your x₁', 'Your x₂'],
      [1, -1],
      [
        [-2.5, 2.5],
        [-2.5, 2.5],
      ],
    );
    tp.on = () => draw();
    draw = () => {
      const t = T(),
        e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u),
        cb = [ym + (beta[0] - ym) * e, beta[1] * e, beta[2] * e],
        ZS = 0.42;
      {
        const { g, w, h } = fit(c),
          pr = orb.cam(w, h, 5.6),
          R = 2.5,
          pl = (a, b) => clamp((cb[0] + cb[1] * a + cb[2] * b) * ZS, -3.2, 3.2),
          path = (ps) => {
            g.beginPath();
            ps.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
          };
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        for (let k = -2; k <= 2; k++) {
          path([pr(k * 1.25, -R, -2), pr(k * 1.25, R, -2)]);
          g.stroke();
          path([pr(-R, k * 1.25, -2), pr(R, k * 1.25, -2)]);
          g.stroke();
        }
        const N = 10,
          Q = [];
        for (let i = 0; i < N; i++)
          for (let j = 0; j < N; j++) {
            const a0 = -R + (2 * R * i) / N,
              a1 = -R + (2 * R * (i + 1)) / N,
              b0 = -R + (2 * R * j) / N,
              b1 = -R + (2 * R * (j + 1)) / N,
              ps = [
                pr(a0, b0, pl(a0, b0)),
                pr(a1, b0, pl(a1, b0)),
                pr(a1, b1, pl(a1, b1)),
                pr(a0, b1, pl(a0, b1)),
              ];
            Q.push({ ps, d: (ps[0][2] + ps[2][2]) / 2, z: (pl(a0, b0) + pl(a1, b1)) / 2 });
          }
        Q.sort((a, b) => b.d - a.d);
        for (const o of Q) {
          path(o.ps);
          g.closePath();
          g.fillStyle = ramp(t, clamp(0.5 + o.z / 5, 0, 1), 0.42);
          g.fill();
          g.strokeStyle = mix(t.ink, t.ink, 0, 0.14);
          g.lineWidth = 0.7;
          g.stroke();
        }
        const D = X.map((row, i) => {
          const z = clamp(y[i] * ZS, -3.4, 3.4),
            zp = pl(row[1], row[2]),
            p = pr(row[1], row[2], z);
          return { p, q: pr(row[1], row[2], zp), up: z >= zp };
        });
        D.sort((a, b) => b.p[2] - a.p[2]);
        for (const o of D) {
          if (S.res) {
            g.strokeStyle = mix(t.ok, t.ok, 0, 0.7);
            g.lineWidth = 1.2;
            path([o.p, o.q]);
            g.stroke();
          }
          g.fillStyle = o.up ? t.accent : mix(t.accent, t.bg, 0.45);
          g.beginPath();
          g.arc(o.p[0], o.p[1], 3.8, 0, 7);
          g.fill();
        }
        {
          const q0 = pr(tp.v[0], tp.v[1], -2),
            q1 = pr(tp.v[0], tp.v[1], pl(tp.v[0], tp.v[1]));
          g.strokeStyle = t.ink;
          g.lineWidth = 1.6;
          g.setLineDash([3, 3]);
          path([q0, q1]);
          g.stroke();
          g.setLineDash([]);
          markTP(g, q1[0], q1[1], t);
        }
        g.font = SANS;
        g.fillStyle = t.ink;
        g.textAlign = 'center';
        let a = pr(0, -R * 1.18, -2);
        g.fillText('x₁', a[0], a[1]);
        a = pr(-R * 1.18, 0, -2);
        g.fillText('x₂', a[0], a[1]);
        a = pr(-R, -R, 2.4);
        g.fillText('y', a[0], a[1]);
      }
      {
        const { g, w, h } = fit(ce),
          m = { l: 40, r: 10, t: 10, b: 30 },
          LR = [-3, 3],
          Xf = (v) => m.l + ((v - LR[0]) / 6) * (w - m.l - m.r),
          Yf = (v) => h - m.b - ((v - LR[0]) / 6) * (h - m.t - m.b);
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.font = MONO;
        g.fillStyle = t.muted;
        for (let v = -3; v <= 3; v++) {
          g.beginPath();
          g.moveTo(Xf(v), m.t);
          g.lineTo(Xf(v), h - m.b);
          g.moveTo(m.l, Yf(v));
          g.lineTo(w - m.r, Yf(v));
          g.stroke();
          g.textAlign = 'center';
          g.fillText(String(v), Xf(v), h - m.b + 14);
          g.textAlign = 'right';
          g.fillText(String(v), m.l - 6, Yf(v) + 4);
        }
        g.textAlign = 'center';
        g.fillText('estimated slope for x₁', w / 2 + m.l / 2, h - 3);
        g.save();
        g.translate(11, h / 2);
        g.rotate(-Math.PI / 2);
        g.fillText('estimated slope for x₂', 0, 0);
        g.restore();
        g.save();
        g.beginPath();
        g.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
        g.clip();
        g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
        for (const s of sim) {
          g.beginPath();
          g.arc(Xf(s[0]), Yf(s[1]), 2.4, 0, 7);
          g.fill();
        }
        const tx = Xf(S.b[1]),
          ty = Yf(S.b[2]);
        g.strokeStyle = t.ink;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(tx - 8, ty);
        g.lineTo(tx + 8, ty);
        g.moveTo(tx, ty - 8);
        g.lineTo(tx, ty + 8);
        g.stroke();
        g.fillStyle = t.gold;
        g.strokeStyle = t.bg;
        g.beginPath();
        g.arc(Xf(beta[1]), Yf(beta[2]), 7, 0, 7);
        g.fill();
        g.stroke();
        g.restore();
        g.textAlign = 'left';
        g.fillStyle = t.ink;
        g.fillText('+ truth', m.l + 8, m.t + 12);
        g.fillStyle = t.cut;
        g.fillText('● this sample', m.l + 8, m.t + 27);
      }
      const n = y.length,
        p = 2;
      let rss = 0,
        tss = 0;
      for (let i = 0; i < n; i++) {
        rss += (y[i] - dot(beta, X[i])) ** 2;
        tss += (y[i] - ym) ** 2;
      }
      const r2 = tss > 1e-12 ? 1 - rss / tss : 0,
        F = rss > 1e-12 ? (tss - rss) / p / (rss / (n - p - 1)) : Infinity;
      let s1 = 0,
        s2 = 0,
        s11 = 0,
        s22 = 0,
        s12 = 0;
      for (const row of X) {
        s1 += row[1];
        s2 += row[2];
        s11 += row[1] * row[1];
        s22 += row[2] * row[2];
        s12 += row[1] * row[2];
      }
      const rr = (s12 - (s1 * s2) / n) / Math.sqrt((s11 - (s1 * s1) / n) * (s22 - (s2 * s2) / n));
      tp.out.textContent =
        'At x₁ = ' +
        f2(tp.v[0]) +
        ', x₂ = ' +
        f2(tp.v[1]) +
        ' the fitted plane predicts y = ' +
        f2(beta[0]) +
        ' + (' +
        f2(beta[1]) +
        ' × ' +
        f2(tp.v[0]) +
        ') + (' +
        f2(beta[2]) +
        ' × ' +
        f2(tp.v[1]) +
        ') = ' +
        f2(dot(beta, [1, tp.v[0], tp.v[1]])) +
        '.';
      setR(root, 'b', f2(beta[0]) + ', ' + f2(beta[1]) + ', ' + f2(beta[2]));
      setR(root, 'r2', f3(r2));
      setR(root, 'ar2', f3(1 - ((1 - r2) * (n - 1)) / (n - p - 1)));
      setR(root, 'f', isFinite(F) ? F.toFixed(1) : '∞');
      setR(root, 'p', isFinite(F) ? pfmt(fpval(F, p, n - p - 1)) : '< 0.0001');
      setR(root, 'vif', (1 / (1 - rr * rr)).toFixed(2));
    };
    const anim = Anim(() => {
      if (wait > 0) {
        wait--;
        if (!wait) u = 0;
        return;
      }
      u = Math.min(1, u + 0.012);
      if (u >= 1) wait = 200;
      draw();
    });
    pb = playBtn($('#mr-play', root), anim);
    $('#mr-play', root).addEventListener('click', () => {
      if (anim.on && u >= 1) {
        u = 0;
        wait = 0;
      }
    });
    const re = () => {
      gen();
      draw();
    };
    bind(root, 'mr-b1', (v) => {
      S.b[1] = v;
      re();
    });
    bind(root, 'mr-b2', (v) => {
      S.b[2] = v;
      re();
    });
    bind(root, 'mr-s', (v) => {
      S.noise = v;
      re();
    });
    bind(root, 'mr-r', (v) => {
      S.r = v;
      re();
    });
    bind(
      root,
      'mr-n',
      (v) => {
        S.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#mr-new', root).onclick = () => {
      seed++;
      u = 0;
      wait = 0;
      re();
      pb.start();
    };
    $('#mr-res', root).onchange = (e) => {
      S.res = e.target.checked;
      draw();
    };
    return {
      draw,
      stop() {
        pb.stop();
        orb.spin(false);
      },
      auto() {
        pb.start();
        orb.spin(true);
      },
    };
  },
};
