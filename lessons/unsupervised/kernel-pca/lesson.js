/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { dotAt, segBind, uCur, uView, xyUp } from '../shared.js';
import { g2Note } from '../../generative/shared.js';
import { laTop } from '../../../js/math/linear-algebra.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#kp-c', root),
      cr = $('#kp-r', root),
      live = $('#kp-live', root),
      S = { d: 'rings', k: 'rbf', g: 0.8, q: 0 },
      GW = 44,
      GH = 33;
    let D = [],
      lab = [],
      Z = [],
      ZL = [],
      al = null,
      Km = null,
      Kt = 0,
      fld = null,
      fmin = 0,
      fmax = 1,
      st = {},
      u = 0,
      seed = 1,
      msg = '',
      cu = 0,
      pb,
      V = null,
      hasLab = true;
    const kern = (a, b) =>
      S.k === 'rbf'
        ? Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * S.g * S.g))
        : S.k === 'poly'
          ? (1 + a[0] * b[0] + a[1] * b[1]) ** 2
          : a[0] * b[0] + a[1] * b[1];
    const gen = () => {
      const r = rng(seed * 11 + S.d.length),
        rn = randnFrom(r);
      D = [];
      lab = [];
      hasLab = true;
      for (let i = 0; i < 110; i++) {
        const k = i % 2;
        let p;
        if (S.d === 'rings') {
          const a = 2 * Math.PI * r(),
            rad = k ? 2.2 : 0.7;
          p = [rad * Math.cos(a) + 0.13 * rn(), rad * Math.sin(a) + 0.13 * rn()];
        } else if (S.d === 'moons') {
          const a = Math.PI * r();
          p = k
            ? [1.6 * Math.cos(a) - 0.8 + 0.03 * rn(), 1.6 * Math.sin(a) - 0.4 + 0.03 * rn()]
            : [0.8 - 1.6 * Math.cos(a) + 0.03 * rn(), 0.4 - 1.6 * Math.sin(a) + 0.03 * rn()];
        } else p = [(k ? 1.5 : -1.5) + 0.6 * rn(), (k ? 0.6 : -0.6) + 0.6 * rn()];
        D.push(p);
        lab.push(k);
      }
    };
    const proj = (x) => {
      const n = D.length,
        kx = D.map((d) => kern(x, d)),
        mk = kx.reduce((s, v) => s + v, 0) / n;
      return al.map((a) => {
        let s = 0;
        for (let i = 0; i < n; i++) s += a[i] * (kx[i] - mk - Km[i] + Kt);
        return s;
      });
    };
    const cut = (vals) => {
      if (!hasLab) return NaN;
      const o = vals.map((v, i) => [v, lab[i]]).sort((a, b) => a[0] - b[0]),
        n = o.length,
        n1 = lab.reduce((s, v) => s + v, 0);
      let best = 0,
        l1 = 0;
      for (let i = 0; i < n; i++) {
        l1 += o[i][1];
        const acc = Math.max(l1 + (n - i - 1) - (n1 - l1), i + 1 - l1 + (n1 - l1)) / n;
        best = Math.max(best, acc);
      }
      return best;
    };
    const cut2 = (P) => {
      let b = 0;
      for (let a = 0; a < 36; a++) {
        const cs = Math.cos((a * Math.PI) / 36),
          sn = Math.sin((a * Math.PI) / 36);
        b = Math.max(b, cut(P.map((p) => p[0] * cs + p[1] * sn)));
      }
      return b;
    };
    const solve = () => {
      const n = D.length,
        K = D.map((a) => D.map((b) => kern(a, b)));
      Km = K.map((r) => r.reduce((s, v) => s + v, 0) / n);
      Kt = Km.reduce((s, v) => s + v, 0) / n;
      const Kc = K.map((r, i) => r.map((v, j) => v - Km[i] - Km[j] + Kt)),
        { vals, vecs } = laTop(Kc, 3),
        tr = Kc.reduce((s, r, i) => s + r[i], 0);
      al = vecs.map((v, m) => v.map((x) => x / Math.sqrt(Math.max(vals[m], 1e-9))));
      Z = D.map((d) => proj(d));
      for (let m = 0; m < 3; m++) {
        const mx = Math.max(...Z.map((z) => Math.abs(z[m]))) || 1;
        Z.forEach((z) => (z[m] = z[m] / mx));
      }
      let cxx = 0,
        cxy = 0,
        cyy = 0,
        mx = 0,
        my = 0;
      D.forEach((d) => {
        mx += d[0] / n;
        my += d[1] / n;
      });
      D.forEach((d) => {
        cxx += (d[0] - mx) ** 2;
        cxy += (d[0] - mx) * (d[1] - my);
        cyy += (d[1] - my) ** 2;
      });
      const th = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
      ZL = D.map((d) => (d[0] - mx) * Math.cos(th) + (d[1] - my) * Math.sin(th));
      st = { v: (vals[0] + vals[1]) / tr, s1: cut(Z.map((z) => z[0])), s12: cut2(Z), lin: cut2(D) };
      field();
      u = RM ? 1 : 0;
      cu = 4;
      const b = st.s12;
      msg = hasLab
        ? b > 0.97
          ? 'In the new coordinates a single straight cut tells the two groups apart (' +
            pct(b) +
            ' right' +
            (st.s1 > 0.97 ? ', and component 1 alone is enough' : '') +
            '). In the original picture the best straight cut gets ' +
            pct(st.lin) +
            '.'
          : 'Even in the new coordinates no straight cut separates the groups cleanly (best ' +
            pct(b) +
            '). In the original picture the best straight cut gets ' +
            pct(st.lin) +
            '.'
        : 'Components computed for your data.';
    };
    const field = () => {
      fld = new Float64Array(GW * GH);
      fmin = 1e9;
      fmax = -1e9;
      for (let r = 0; r < GH; r++)
        for (let q = 0; q < GW; q++) {
          const v = proj([-4 + (8 * (q + 0.5)) / GW, 3 - (6 * (r + 0.5)) / GH])[S.q];
          fld[r * GW + q] = v;
          if (v < fmin) fmin = v;
          if (v > fmax) fmax = v;
        }
    };
    const tp = testPoint(
      root,
      'kp',
      ['Your x', 'Your y'],
      [0, 0.7],
      [
        [-4, 4],
        [-3, 3],
      ],
    );
    const draw = () => {
      const t = T(),
        cs = [t.accent, t.gold];
      V = uView(c);
      {
        const { g, w, h, X, Y } = V;
        for (let r = 0; r < GH; r++)
          for (let q = 0; q < GW; q++) {
            const v = (fld[r * GW + q] - fmin) / (fmax - fmin || 1);
            g.fillStyle = mix(t.bg, t.ink, v * 0.42);
            const xa = Math.round(X(-4 + (8 * q) / GW)),
              xb = Math.round(X(-4 + (8 * (q + 1)) / GW)),
              ya = Math.round(Y(3 - (6 * r) / GH)),
              yb = Math.round(Y(3 - (6 * (r + 1)) / GH));
            g.fillRect(xa, ya, xb - xa, yb - ya);
          }
        D.forEach((d, i) => {
          g.lineWidth = 1.2;
          dotAt(g, X(d[0]), Y(d[1]), 4, hasLab ? cs[lab[i]] : t.accent, t.bg);
        });
        markTP(g, X(tp.v[0]), Y(tp.v[1]), t);
        g2Note(g, t, h, 'shading: component ' + (S.q + 1) + ', dark = high');
      }
      const zt = proj(tp.v),
        mxs = [0, 1].map((m) => 1);
      {
        const { g, w, h } = fit(cr),
          s = u * u * (3 - 2 * u),
          sc = Math.min(w / 8, h / 6),
          X = (x) => w / 2 + x * sc,
          Y = (y) => h / 2 - y * sc;
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X(-4), Y(0));
        g.lineTo(X(4), Y(0));
        g.moveTo(X(0), Y(-3));
        g.lineTo(X(0), Y(3));
        g.stroke();
        D.forEach((d, i) => {
          const x = d[0] + (Z[i][0] * 3.3 - d[0]) * s,
            y = d[1] + (Z[i][1] * 2.4 - d[1]) * s;
          g.lineWidth = 1.2;
          dotAt(g, X(x), Y(y), 4, hasLab ? cs[lab[i]] : t.accent, t.bg);
        });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText(s > 0.5 ? 'component 1 →' : 'x →', w - 6, h - 6);
        g.textAlign = 'left';
        g.fillText(s > 0.5 ? '↑ component 2' : '↑ y', 6, 12);
      }
      tp.out.textContent =
        'Your point (' +
        f2(tp.v[0]) +
        ', ' +
        f2(tp.v[1]) +
        ') has component 1 = ' +
        f3(zt[0]) +
        ', component 2 = ' +
        f3(zt[1]) +
        ' and component 3 = ' +
        f3(zt[2]) +
        ', found by comparing it with all ' +
        D.length +
        ' data points.';
      live.textContent = msg;
      uCur(root, 'kp', cu);
      setR(root, 'var', pct(st.v));
      setR(root, 's1', hasLab ? pct(st.s1) : '–');
      setR(root, 's12', hasLab ? pct(st.s12) : '–');
      setR(root, 'lin', hasLab ? pct(st.lin) : '–');
    };
    tp.on = () => draw();
    gen();
    solve();
    const anim = Anim(() => {
      u = clamp(u + 0.012, 0, 1);
      if (u >= 1) pb.stop();
      draw();
    });
    pb = playBtn($('#kp-play', root), anim);
    {
      const s0 = pb.start;
      pb.start = () => {
        if (u >= 1) u = 0;
        s0();
      };
    }
    const re = () => {
      pb.stop();
      solve();
      draw();
      pb.start();
    };
    $('#kp-step', root).onclick = () => {
      pb.stop();
      u = u < 0.5 ? 1 : 0;
      draw();
    };
    $('#kp-reset', root).onclick = () => {
      pb.stop();
      u = 0;
      draw();
    };
    $('#kp-new', root).onclick = () => {
      seed++;
      gen();
      re();
    };
    bind(root, 'kp-g', (v) => {
      S.g = v;
      pb.stop();
      solve();
      u = 1;
      draw();
    });
    segBind(root, 'kp-d', S.d, (v) => {
      S.d = v;
      gen();
      re();
    });
    segBind(root, 'kp-k', S.k, (v) => {
      S.k = v;
      re();
    });
    segBind(root, 'kp-q', '0', (v) => {
      S.q = +v;
      field();
      draw();
    });
    xyUp(root, 'kp', (q) => {
      D = q.slice(0, 160);
      lab = D.map(() => 0);
      hasLab = false;
      re();
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        u = 0;
        draw();
        pb.start();
      },
    };
  },
};
