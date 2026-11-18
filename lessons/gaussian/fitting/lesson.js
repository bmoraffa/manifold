/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  bind,
  ellPath,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  setReadout,
} from '../../../js/core/shared.js';
import { parseXY } from '../../regression/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#fit-c', root),
      st = $('#fit-status', root),
      sel = $('#fit-ds', root);
    const DS = {
      tilted: { mu: [1, -0.5], sx: 1.5, sy: 0.8, rho: 0.7 },
      round: { mu: [0, 0], sx: 1, sy: 1, rho: 0 },
      narrow: { mu: [-0.5, 0.5], sx: 1.6, sy: 1.2, rho: -0.92 },
    };
    const generatorParams = { mu: [1, -0.5], sx: 1.5, sy: 0.8, rho: 0.7, n: 220 };
    let seed = 7,
      train = [],
      test = [],
      shown = 0,
      truth = null,
      acc = 0,
      plotView = {},
      custom = null,
      playback;
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
    const meanLogLikelihood = (p, F) => {
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
      if (fixed) plotView = fixed;
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
        plotView = {
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
        for (let i = 0; i < generatorParams.n; i++) {
          const k = r() < 0.5;
          p.push([(k ? -2 : 2) + 0.6 * rn(), (k ? -1 : 1.2) + 0.6 * rn()]);
        }
        load(p, null, { cx: 0, cy: 0, hx: 7.5, hy: 3.6 });
      } else if (sel.value === 'csv' && custom) load(custom, null);
      else {
        const q = Math.sqrt(1 - generatorParams.rho * generatorParams.rho);
        for (let i = 0; i < generatorParams.n; i++) {
          const a = rn(),
            b = rn();
          p.push([
            generatorParams.mu[0] + generatorParams.sx * a,
            generatorParams.mu[1] + generatorParams.sy * (generatorParams.rho * a + q * b),
          ]);
        }
        load(
          p,
          {
            mx: generatorParams.mu[0],
            my: generatorParams.mu[1],
            a: generatorParams.sx * generatorParams.sx,
            b: generatorParams.rho * generatorParams.sx * generatorParams.sy,
            d: generatorParams.sy * generatorParams.sy,
          },
          { cx: generatorParams.mu[0], cy: generatorParams.mu[1], hx: 7.5, hy: 3.6 },
        );
      }
    };
    gen();
    const draw = () => {
      const colors = readThemeColors(),
        { g: ctx, w, h } = prepareCanvas(mainCanvas),
        m = { l: 44, r: 14, t: 12, b: 26 };
      const sc = Math.min((w - m.l - m.r) / (2 * plotView.hx), (h - m.t - m.b) / (2 * plotView.hy)),
        ox = m.l + (w - m.l - m.r) / 2,
        oy = m.t + (h - m.t - m.b) / 2;
      const X = (x) => ox + (x - plotView.cx) * sc,
        Y = (y) => oy - (y - plotView.cy) * sc;
      ctx.font = MONO;
      ctx.fillStyle = colors.muted;
      ctx.textAlign = 'center';
      const hx = (w - m.l - m.r) / 2 / sc,
        hy = (h - m.t - m.b) / 2 / sc;
      ctx.strokeStyle = colors.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(m.l, h - m.b);
      ctx.lineTo(w - m.r, h - m.b);
      ctx.moveTo(m.l, m.t);
      ctx.lineTo(m.l, h - m.b);
      ctx.stroke();
      [-1, 0, 1].forEach((k) => {
        ctx.fillText(
          (plotView.cx + k * hx * 0.9).toPrecision(3),
          X(plotView.cx + k * hx * 0.9),
          h - m.b + 16,
        );
      });
      ctx.textAlign = 'right';
      [-1, 0, 1].forEach((k) => {
        ctx.fillText(
          (plotView.cy + k * hy * 0.85).toPrecision(3),
          m.l - 5,
          Y(plotView.cy + k * hy * 0.85) + 4,
        );
      });
      ctx.save();
      ctx.beginPath();
      ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
      ctx.clip();
      ctx.strokeStyle = colors.muted;
      ctx.lineWidth = 1.2;
      for (const q of test) {
        ctx.beginPath();
        ctx.arc(X(q[0]), Y(q[1]), 3.2, 0, 7);
        ctx.stroke();
      }
      ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.85);
      for (let i = 0; i < shown; i++) {
        ctx.beginPath();
        ctx.arc(
          X(train[i][0]),
          Y(train[i][1]),
          i === shown - 1 && shown < train.length ? 6 : 3.4,
          0,
          7,
        );
        ctx.fill();
      }
      if (truth) {
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.6;
        ellPath(ctx, X, Y, truth.mx, truth.my, truth.a, truth.b, truth.d, 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      let F = null;
      if (shown >= 3) {
        F = mle(train.slice(0, shown));
        if (F.a * F.d - F.b * F.b > 1e-12) {
          ctx.strokeStyle = colors.ok;
          ctx.lineWidth = 2.8;
          ellPath(ctx, X, Y, F.mx, F.my, F.a, F.b, F.d, 1);
          ctx.stroke();
          ellPath(ctx, X, Y, F.mx, F.my, F.a, F.b, F.d, 2);
          ctx.stroke();
          ctx.fillStyle = mix(colors.ok, colors.ok, 0, 0.07);
          ctx.fill();
          ctx.beginPath();
          ctx.moveTo(X(F.mx) - 7, Y(F.my));
          ctx.lineTo(X(F.mx) + 7, Y(F.my));
          ctx.moveTo(X(F.mx), Y(F.my) - 7);
          ctx.lineTo(X(F.mx), Y(F.my) + 7);
          ctx.stroke();
        } else F = null;
      }
      ctx.restore();
      ctx.textAlign = 'left';
      ctx.fillStyle = colors.ok;
      ctx.fillText('fit: 1σ and 2σ', m.l + 10, m.t + 12);
      if (truth) {
        ctx.fillStyle = colors.ink;
        ctx.fillText('truth: 2σ, dashed', m.l + 10, m.t + 27);
      }
      ctx.fillStyle = colors.muted;
      ctx.fillText('hollow = test', m.l + 10, m.t + (truth ? 42 : 27));
      setReadout(root, 'n', shown + ' of ' + train.length);
      if (F) {
        const sx = Math.sqrt(F.a),
          sy = Math.sqrt(F.d);
        setReadout(root, 'mu', '(' + F.mx.toPrecision(3) + ', ' + F.my.toPrecision(3) + ')');
        setReadout(
          root,
          'cov',
          sx.toPrecision(3) + ', ' + sy.toPrecision(3) + ', ' + f2(F.b / (sx * sy)),
        );
        setReadout(root, 'tr', f3(meanLogLikelihood(train.slice(0, shown), F)));
        setReadout(root, 'te', f3(meanLogLikelihood(test, F)));
      } else ['mu', 'cov', 'tr', 'te'].forEach((k) => setReadout(root, k, '–'));
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
      if (shown >= train.length) playback.stop();
    });
    playback = playBtn($('#fit-play', root), anim);
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
      playback.stop();
      shown = 0;
      acc = 0;
      draw();
    };
    const sx = bind(root, 'fit-sx', (v) => {
        generatorParams.sx = v;
        live();
      }),
      sy = bind(root, 'fit-sy', (v) => {
        generatorParams.sy = v;
        live();
      }),
      sr = bind(root, 'fit-rho', (v) => {
        generatorParams.rho = v;
        live();
      }),
      sn = bind(
        root,
        'fit-n',
        (v) => {
          generatorParams.n = v;
          live(true);
        },
        (v) => String(v),
      );
    function live(isN) {
      playback.stop();
      if (!(isN && sel.value === 'clumps') && sel.value !== 'custom') {
        sel.value = 'custom';
        generatorParams.mu = [0, 0];
      }
      gen();
      shown = train.length;
      draw();
    }
    sel.onchange = () => {
      st.textContent = '';
      const d = DS[sel.value];
      if (d) {
        Object.assign(generatorParams, { mu: d.mu.slice(), sx: d.sx, sy: d.sy, rho: d.rho });
        sx.set(generatorParams.sx);
        sy.set(generatorParams.sy);
        sr.set(generatorParams.rho);
      }
      if (sel.value === 'custom') generatorParams.mu = [0, 0];
      if (sel.value === 'csv' && !custom) sel.value = 'custom';
      gen();
      shown = 0;
      acc = 0;
      draw();
      playback.start();
    };
    $('#fit-new', root).onclick = () => {
      seed++;
      gen();
      shown = 0;
      acc = 0;
      draw();
      playback.start();
    };
    $('#fit-file', root).onchange = (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const fileReader = new FileReader();
      fileReader.onerror = () => {
        st.textContent = 'That file could not be read. Try saving it again as plain CSV.';
      };
      fileReader.onload = () => {
        const p = parseXY(fileReader.result, 5000);
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
        playback.start();
      };
      fileReader.readAsText(f);
      e.target.value = '';
    };
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
