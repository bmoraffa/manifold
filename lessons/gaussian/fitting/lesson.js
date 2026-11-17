/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  ellPath,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  randnFrom,
  rng,
  setR,
} from '../../../js/core/shared.js';
import { parseXY } from '../../regression/shared.js';

export default {
  init(root) {
    const c = $('#fit-c', root),
      st = $('#fit-status', root),
      sel = $('#fit-ds', root);
    const DS = {
      tilted: { mu: [1, -0.5], sx: 1.5, sy: 0.8, rho: 0.7 },
      round: { mu: [0, 0], sx: 1, sy: 1, rho: 0 },
      narrow: { mu: [-0.5, 0.5], sx: 1.6, sy: 1.2, rho: -0.92 },
    };
    const Pm = { mu: [1, -0.5], sx: 1.5, sy: 0.8, rho: 0.7, n: 220 };
    let seed = 7,
      train = [],
      test = [],
      shown = 0,
      truth = null,
      acc = 0,
      V = {},
      custom = null,
      pb;
    const mle = (p) => {
      const n = p.length;
      let mx = 0,
        my = 0;
      for (const q of p) {
        mx += q[0];
        my += q[1];
      }
      mx /= n;
      my /= n;
      let a = 0,
        b = 0,
        d = 0;
      for (const q of p) {
        const u = q[0] - mx,
          v = q[1] - my;
        a += u * u;
        b += u * v;
        d += v * v;
      }
      return { mx, my, a: a / n, b: b / n, d: d / n, n };
    };
    const ll = (p, F) => {
      const det = F.a * F.d - F.b * F.b;
      if (!(det > 1e-12) || !p.length) return NaN;
      let s = 0;
      for (const q of p) {
        const u = q[0] - F.mx,
          v = q[1] - F.my;
        s += (F.d * u * u - 2 * F.b * u * v + F.a * v * v) / det;
      }
      return -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - (0.5 * s) / p.length;
    };
    const load = (pts, tr, fixed) => {
      const r = rng(5),
        p = pts.slice();
      for (let i = p.length - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [p[i], p[j]] = [p[j], p[i]];
      }
      const k = Math.round(p.length * 0.7);
      train = p.slice(0, k);
      test = p.slice(k);
      truth = tr;
      if (fixed) V = fixed;
      else {
        let x0 = 1 / 0,
          x1 = -1 / 0,
          y0 = 1 / 0,
          y1 = -1 / 0;
        for (const q of p) {
          x0 = Math.min(x0, q[0]);
          x1 = Math.max(x1, q[0]);
          y0 = Math.min(y0, q[1]);
          y1 = Math.max(y1, q[1]);
        }
        V = {
          cx: (x0 + x1) / 2,
          cy: (y0 + y1) / 2,
          hx: ((x1 - x0) / 2) * 1.15 || 1,
          hy: ((y1 - y0) / 2) * 1.15 || 1,
        };
      }
    };
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r),
        p = [];
      if (sel.value === 'clumps') {
        for (let i = 0; i < Pm.n; i++) {
          const k = r() < 0.5;
          p.push([(k ? -2 : 2) + 0.6 * rn(), (k ? -1 : 1.2) + 0.6 * rn()]);
        }
        load(p, null, { cx: 0, cy: 0, hx: 7.5, hy: 3.6 });
      } else if (sel.value === 'csv' && custom) load(custom, null);
      else {
        const q = Math.sqrt(1 - Pm.rho * Pm.rho);
        for (let i = 0; i < Pm.n; i++) {
          const a = rn(),
            b = rn();
          p.push([Pm.mu[0] + Pm.sx * a, Pm.mu[1] + Pm.sy * (Pm.rho * a + q * b)]);
        }
        load(
          p,
          {
            mx: Pm.mu[0],
            my: Pm.mu[1],
            a: Pm.sx * Pm.sx,
            b: Pm.rho * Pm.sx * Pm.sy,
            d: Pm.sy * Pm.sy,
          },
          { cx: Pm.mu[0], cy: Pm.mu[1], hx: 7.5, hy: 3.6 },
        );
      }
    };
    gen();
    const draw = () => {
      const t = T(),
        { g, w, h } = fit(c),
        m = { l: 44, r: 14, t: 12, b: 26 };
      const sc = Math.min((w - m.l - m.r) / (2 * V.hx), (h - m.t - m.b) / (2 * V.hy)),
        ox = m.l + (w - m.l - m.r) / 2,
        oy = m.t + (h - m.t - m.b) / 2;
      const X = (x) => ox + (x - V.cx) * sc,
        Y = (y) => oy - (y - V.cy) * sc;
      g.font = MONO;
      g.fillStyle = t.muted;
      g.textAlign = 'center';
      const hx = (w - m.l - m.r) / 2 / sc,
        hy = (h - m.t - m.b) / 2 / sc;
      g.strokeStyle = t.line;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(m.l, h - m.b);
      g.lineTo(w - m.r, h - m.b);
      g.moveTo(m.l, m.t);
      g.lineTo(m.l, h - m.b);
      g.stroke();
      [-1, 0, 1].forEach((k) => {
        g.fillText((V.cx + k * hx * 0.9).toPrecision(3), X(V.cx + k * hx * 0.9), h - m.b + 16);
      });
      g.textAlign = 'right';
      [-1, 0, 1].forEach((k) => {
        g.fillText((V.cy + k * hy * 0.85).toPrecision(3), m.l - 5, Y(V.cy + k * hy * 0.85) + 4);
      });
      g.save();
      g.beginPath();
      g.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
      g.clip();
      g.strokeStyle = t.muted;
      g.lineWidth = 1.2;
      for (const q of test) {
        g.beginPath();
        g.arc(X(q[0]), Y(q[1]), 3.2, 0, 7);
        g.stroke();
      }
      g.fillStyle = mix(t.accent, t.accent, 0, 0.85);
      for (let i = 0; i < shown; i++) {
        g.beginPath();
        g.arc(
          X(train[i][0]),
          Y(train[i][1]),
          i === shown - 1 && shown < train.length ? 6 : 3.4,
          0,
          7,
        );
        g.fill();
      }
      if (truth) {
        g.setLineDash([6, 4]);
        g.strokeStyle = t.ink;
        g.lineWidth = 1.6;
        ellPath(g, X, Y, truth.mx, truth.my, truth.a, truth.b, truth.d, 2);
        g.stroke();
        g.setLineDash([]);
      }
      let F = null;
      if (shown >= 3) {
        F = mle(train.slice(0, shown));
        if (F.a * F.d - F.b * F.b > 1e-12) {
          g.strokeStyle = t.ok;
          g.lineWidth = 2.8;
          ellPath(g, X, Y, F.mx, F.my, F.a, F.b, F.d, 1);
          g.stroke();
          ellPath(g, X, Y, F.mx, F.my, F.a, F.b, F.d, 2);
          g.stroke();
          g.fillStyle = mix(t.ok, t.ok, 0, 0.07);
          g.fill();
          g.beginPath();
          g.moveTo(X(F.mx) - 7, Y(F.my));
          g.lineTo(X(F.mx) + 7, Y(F.my));
          g.moveTo(X(F.mx), Y(F.my) - 7);
          g.lineTo(X(F.mx), Y(F.my) + 7);
          g.stroke();
        } else F = null;
      }
      g.restore();
      g.textAlign = 'left';
      g.fillStyle = t.ok;
      g.fillText('fit: 1σ and 2σ', m.l + 10, m.t + 12);
      if (truth) {
        g.fillStyle = t.ink;
        g.fillText('truth: 2σ, dashed', m.l + 10, m.t + 27);
      }
      g.fillStyle = t.muted;
      g.fillText('hollow = test', m.l + 10, m.t + (truth ? 42 : 27));
      setR(root, 'n', shown + ' of ' + train.length);
      if (F) {
        const sx = Math.sqrt(F.a),
          sy = Math.sqrt(F.d);
        setR(root, 'mu', '(' + F.mx.toPrecision(3) + ', ' + F.my.toPrecision(3) + ')');
        setR(
          root,
          'cov',
          sx.toPrecision(3) + ', ' + sy.toPrecision(3) + ', ' + f2(F.b / (sx * sy)),
        );
        setR(root, 'tr', f3(ll(train.slice(0, shown), F)));
        setR(root, 'te', f3(ll(test, F)));
      } else ['mu', 'cov', 'tr', 'te'].forEach((k) => setR(root, k, '–'));
    };
    const sp = bind(
      root,
      'fit-speed',
      () => {},
      (v) => String(v),
    );
    const anim = Anim(() => {
      acc += +sp.el.value / 12;
      const k = Math.floor(acc);
      acc -= k;
      shown = Math.min(train.length, shown + k);
      draw();
      if (shown >= train.length) pb.stop();
    });
    pb = playBtn($('#fit-play', root), anim);
    $('#fit-play', root).addEventListener('click', () => {
      if (anim.on && shown >= train.length) {
        shown = 0;
        acc = 0;
      }
    });
    $('#fit-step', root).onclick = () => {
      shown = Math.min(train.length, shown + 1);
      draw();
    };
    $('#fit-reset', root).onclick = () => {
      pb.stop();
      shown = 0;
      acc = 0;
      draw();
    };
    const sx = bind(root, 'fit-sx', (v) => {
        Pm.sx = v;
        live();
      }),
      sy = bind(root, 'fit-sy', (v) => {
        Pm.sy = v;
        live();
      }),
      sr = bind(root, 'fit-rho', (v) => {
        Pm.rho = v;
        live();
      }),
      sn = bind(
        root,
        'fit-n',
        (v) => {
          Pm.n = v;
          live(true);
        },
        (v) => String(v),
      );
    function live(isN) {
      pb.stop();
      if (!(isN && sel.value === 'clumps') && sel.value !== 'custom') {
        sel.value = 'custom';
        Pm.mu = [0, 0];
      }
      gen();
      shown = train.length;
      draw();
    }
    sel.onchange = () => {
      st.textContent = '';
      const d = DS[sel.value];
      if (d) {
        Object.assign(Pm, { mu: d.mu.slice(), sx: d.sx, sy: d.sy, rho: d.rho });
        sx.set(Pm.sx);
        sy.set(Pm.sy);
        sr.set(Pm.rho);
      }
      if (sel.value === 'custom') Pm.mu = [0, 0];
      if (sel.value === 'csv' && !custom) sel.value = 'custom';
      gen();
      shown = 0;
      acc = 0;
      draw();
      pb.start();
    };
    $('#fit-new', root).onclick = () => {
      seed++;
      gen();
      shown = 0;
      acc = 0;
      draw();
      pb.start();
    };
    $('#fit-file', root).onchange = (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const fr = new FileReader();
      fr.onerror = () => {
        st.textContent = 'That file could not be read. Try saving it again as plain CSV.';
      };
      fr.onload = () => {
        const p = parseXY(fr.result, 5000);
        if (typeof p === 'string') {
          st.textContent = p;
          return;
        }
        custom = p;
        let o = $('option[value="csv"]', sel);
        if (!o) {
          o = document.createElement('option');
          o.value = 'csv';
          sel.appendChild(o);
        }
        o.textContent = 'Your CSV: ' + f.name.slice(0, 24);
        sel.value = 'csv';
        st.textContent = '';
        gen();
        shown = 0;
        acc = 0;
        draw();
        pb.start();
      };
      fr.readAsText(f);
      e.target.value = '';
    };
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
