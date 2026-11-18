/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  bind,
  clamp,
  f2,
  f3,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  randn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import {
  canvasXY,
  dotAt,
  segBind,
  highlightLessonStep,
  uFrame,
  drawSeriesChart,
  uView,
  xyUp,
} from '../../unsupervised/shared.js';
import { laBck, laChol, laFwd } from '../../../js/math/linear-algebra.js';

export default {
  init(root) {
    const c = $('#gp-c', root),
      cr = $('#gp-r', root),
      live = $('#gp-live', root),
      settings = { d: 'wave', k: 'rbf', l: 1, f: 1, n: 0.2, w: true },
      G = 97,
      gx = Array.from({ length: G }, (_, i) => -4 + (8 * i) / (G - 1));
    let D = [],
      mu = null,
      sd = null,
      Lp = null,
      za = null,
      zb = null,
      sampleAnimationPhase = 0,
      ml = NaN,
      dataFitTerm = NaN,
      complexityTerm = NaN,
      hist = [],
      fin = false,
      msg = '',
      activeStepIndex = 0,
      V = null,
      playback,
      checkRunStop = () => false,
      lB,
      fB,
      nB,
      ad = null,
      it = 0;
    const kern = (a, b, p = settings) => {
      const r = Math.abs(a - b),
        f2_ = p.f * p.f;
      return p.k === 'rbf'
        ? f2_ * Math.exp((-r * r) / (2 * p.l * p.l))
        : p.k === 'mat'
          ? f2_ * (1 + (Math.sqrt(3) * r) / p.l) * Math.exp((-Math.sqrt(3) * r) / p.l)
          : p.k === 'per'
            ? f2_ * Math.exp((-2 * Math.sin((Math.PI * r) / 3) ** 2) / (p.l * p.l))
            : f2_ * (1 + a * b);
    };
    const data = () => {
      const r = rng(settings.d === 'wave' ? 5 : settings.d === 'gap' ? 9 : 13),
        rn = randnFrom(r);
      D = [];
      if (settings.d === 'wave')
        for (let i = 0; i < 9; i++) {
          const x = -3.5 + (7 * (i + 0.5 * r())) / 9;
          D.push([x, 1.3 * Math.sin(1.3 * x) + 0.2 * rn()]);
        }
      else if (settings.d === 'gap')
        for (let i = 0; i < 12; i++) {
          const x = i < 6 ? -3.7 + 2.2 * r() : 1.5 + 2.2 * r();
          D.push([x, 1.2 * Math.sin(1.5 * x) + 0.4 * x * 0.3 + 0.15 * rn()]);
        }
      else if (settings.d === 'line')
        for (let i = 0; i < 14; i++) {
          const x = -3.6 + 7.2 * r();
          D.push([x, 0.45 * x + 0.3 + 0.45 * rn()]);
        }
    };
    const score = (p) => {
      const n = D.length;
      if (!n) return [0, 0, 0];
      const K = D.map((a, i) =>
          D.map((b, j) => kern(a[0], b[0], p) + (i === j ? p.n * p.n + 1e-8 : 0)),
        ),
        L = laChol(K),
        y = D.map((d) => d[1]),
        al = laBck(L, laFwd(L, y));
      let a = 0,
        b = 0;
      for (let i = 0; i < n; i++) {
        a -= 0.5 * y[i] * al[i];
        b -= Math.log(L[i][i]);
      }
      return [a + b - 0.5 * n * Math.log(2 * Math.PI), a, b];
    };
    const solve = () => {
      const n = D.length;
      mu = new Float64Array(G);
      sd = new Float64Array(G);
      const C = gx.map((a, i) => gx.map((b, j) => kern(a, b) + (i === j ? 1e-6 : 0)));
      if (n) {
        const K = D.map((a, i) =>
            D.map((b, j) => kern(a[0], b[0]) + (i === j ? settings.n * settings.n + 1e-8 : 0)),
          ),
          L = laChol(K),
          al = laBck(
            L,
            laFwd(
              L,
              D.map((d) => d[1]),
            ),
          ),
          Vs = gx.map((x) =>
            laFwd(
              L,
              D.map((d) => kern(x, d[0])),
            ),
          );
        for (let i = 0; i < G; i++) {
          let m = 0;
          for (let j = 0; j < n; j++) m += kern(gx[i], D[j][0]) * al[j];
          mu[i] = m;
          for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let k = 0; k < n; k++) s += Vs[i][k] * Vs[j][k];
            C[i][j] -= s;
            C[j][i] = C[i][j];
          }
        }
      }
      for (let i = 0; i < G; i++) {
        C[i][i] += 1e-6;
        sd[i] = Math.sqrt(Math.max(0, C[i][i]));
      }
      Lp = laChol(C);
      [ml, dataFitTerm, complexityTerm] = score(settings);
    };
    const newZ = () => {
      za = [0, 1, 2].map(() => Float64Array.from({ length: G }, randn));
      zb = [0, 1, 2].map(() => Float64Array.from({ length: G }, randn));
    };
    const reset = () => {
      hist = [];
      fin = false;
      it = 0;
      ad = null;
      activeStepIndex = D.length ? 3 : 1;
      solve();
      msg = D.length
        ? 'With ' +
          D.length +
          ' measurements the belief is pinched around each one. Change the kernel settings by hand, or press Fit the settings.'
        : 'No data: this is the prior. Every curve the kernel allows is still possible. Click in the picture to add a measurement.';
    };
    const predict = (x) => {
      const n = D.length;
      if (!n) return [0, Math.sqrt(kern(x, x))];
      const K = D.map((a, i) =>
          D.map((b, j) => kern(a[0], b[0]) + (i === j ? settings.n * settings.n + 1e-8 : 0)),
        ),
        L = laChol(K),
        ks = D.map((d) => kern(x, d[0])),
        al = laBck(
          L,
          laFwd(
            L,
            D.map((d) => d[1]),
          ),
        ),
        v = laFwd(L, ks);
      let m = 0,
        s = kern(x, x);
      for (let j = 0; j < n; j++) {
        m += ks[j] * al[j];
        s -= v[j] * v[j];
      }
      return [m, Math.sqrt(Math.max(0, s))];
    };
    const fitStep = () => {
      if (fin || D.length < 2) {
        fin = true;
        msg = 'Add at least two measurements before fitting.';
        return;
      }
      if (!ad) ad = { m: [0, 0, 0], v: [0, 0, 0], t: 0 };
      const P = (t) => ({ k: settings.k, l: Math.exp(t[0]), f: Math.exp(t[1]), n: Math.exp(t[2]) });
      let th = [Math.log(settings.l), Math.log(settings.f), Math.log(settings.n)];
      if (it === 0) {
        let bs = score(P(th))[0] + 0.5;
        for (const l of [0.3, 0.6, 1.2, 2.4])
          for (const f of [0.5, 1, 2])
            for (const n of [0.1, 0.3, 0.8]) {
              const q = [Math.log(l), Math.log(f), Math.log(n)],
                v = score(P(q))[0];
              if (v > bs) {
                bs = v;
                th = q;
              }
            }
      }
      const P0 = (t) => ({
          k: settings.k,
          l: Math.exp(t[0]),
          f: Math.exp(t[1]),
          n: Math.exp(t[2]),
        }),
        s0 = score(P(th))[0],
        gr = th.map((_, i) => {
          const a = th.slice(),
            b = th.slice();
          a[i] += 1e-4;
          b[i] -= 1e-4;
          return (score(P(a))[0] - score(P(b))[0]) / 2e-4;
        });
      ad.t++;
      for (let i = 0; i < 3; i++) {
        ad.m[i] = 0.8 * ad.m[i] + 0.2 * gr[i];
        ad.v[i] = 0.99 * ad.v[i] + 0.01 * gr[i] * gr[i];
        th[i] +=
          (0.06 * (ad.m[i] / (1 - 0.8 ** ad.t))) / (Math.sqrt(ad.v[i] / (1 - 0.99 ** ad.t)) + 1e-8);
      }
      settings.l = clamp(Math.exp(th[0]), 0.15, 3);
      settings.f = clamp(Math.exp(th[1]), 0.2, 3);
      settings.n = clamp(Math.exp(th[2]), 0.02, 1.5);
      lB.set(settings.l);
      fB.set(settings.f);
      nB.set(settings.n);
      solve();
      it++;
      if (!hist.length) hist.push(s0);
      hist.push(ml);
      activeStepIndex = 5;
      const gain = ml - hist[hist.length - 2];
      msg =
        'Fitting step ' +
        it +
        ': score ' +
        f2(ml) +
        ' (' +
        (gain >= 0 ? '+' : '') +
        f3(gain) +
        '). Length scale ' +
        f2(settings.l) +
        ', height ' +
        f2(settings.f) +
        ', noise ' +
        f2(settings.n) +
        '.';
      if (
        (it >= 12 &&
          Math.abs(gain) < 5e-4 &&
          Math.abs(ml - hist[Math.max(0, hist.length - 6)]) < 3e-3) ||
        it >= 400
      ) {
        fin = true;
        msg =
          'Stopped after ' +
          it +
          ' steps: the score has stopped improving at ' +
          f2(ml) +
          '. Fitted settings: length scale ' +
          f2(settings.l) +
          ', height ' +
          f2(settings.f) +
          ', noise ' +
          f2(settings.n) +
          '. Move a slider to try to beat it.';
      }
    };
    const tp = testPoint(root, 'gp', ['Your x'], [1.5], [[-4, 4]]);
    const draw = () => {
      const colors = readThemeColors();
      V = uView(c);
      const { g, w, h, X, Y } = V;
      uFrame(V, colors, 'x', 'y');
      if (mu) {
        g.fillStyle = mix(colors.accent, colors.accent, 0, 0.16);
        g.beginPath();
        for (let i = 0; i < G; i++) g.lineTo(X(gx[i]), Y(mu[i] + 1.96 * sd[i]));
        for (let i = G - 1; i >= 0; i--) g.lineTo(X(gx[i]), Y(mu[i] - 1.96 * sd[i]));
        g.fill();
        const cs = [colors.gold, colors.ok, '#4a7fb5'];
        for (let s = 0; s < 3; s++) {
          g.strokeStyle = cs[s];
          g.lineWidth = 1.1;
          g.globalAlpha = 0.8;
          g.beginPath();
          const ca = Math.cos(sampleAnimationPhase + s),
            sa = Math.sin(sampleAnimationPhase + s);
          for (let i = 0; i < G; i++) {
            let v = mu[i];
            const Li = Lp[i];
            for (let j = 0; j <= i; j++) v += Li[j] * (ca * za[s][j] + sa * zb[s][j]);
            i ? g.lineTo(X(gx[i]), Y(v)) : g.moveTo(X(gx[i]), Y(v));
          }
          g.stroke();
        }
        g.globalAlpha = 1;
        g.strokeStyle = colors.accent;
        g.lineWidth = 2.6;
        g.beginPath();
        for (let i = 0; i < G; i++) i ? g.lineTo(X(gx[i]), Y(mu[i])) : g.moveTo(X(gx[i]), Y(mu[i]));
        g.stroke();
      }
      for (const d of D) {
        g.lineWidth = 2;
        dotAt(g, X(d[0]), Y(d[1]), 5, colors.ink, colors.bg);
      }
      const [pm, ps] = predict(tp.v[0]);
      g.strokeStyle = colors.ink;
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(X(tp.v[0]), Y(pm - 1.96 * ps));
      g.lineTo(X(tp.v[0]), Y(pm + 1.96 * ps));
      g.stroke();
      markTP(g, X(tp.v[0]), Y(pm), colors, false);
      tp.out.textContent =
        'At x = ' +
        f2(tp.v[0]) +
        ' the prediction is ' +
        f2(pm) +
        ' ± ' +
        f2(1.96 * ps) +
        ' (95% range ' +
        f2(pm - 1.96 * ps) +
        ' to ' +
        f2(pm + 1.96 * ps) +
        ').';
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          ks = Array.from({ length: 60 }, (_, i) => kern(0, (4 * i) / 59));
        const A = drawSeriesChart(ctx, colors, 0, 0, w, h1, ks, {
          title: 'kernel: tie between two points, against their distance (0 to 4)',
          ym: Math.max(...ks) * 1.05,
          y1: Math.min(0, ...ks),
        });
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ks.forEach((v, i) => (i ? ctx.lineTo(A.X(i), A.Y(v)) : ctx.moveTo(A.X(i), A.Y(v))));
        ctx.stroke();
        if (hist.length > 1) {
          const lo = Math.min(...hist),
            hi = Math.max(...hist),
            B = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, hist, {
              title: 'score during fitting',
              n: Math.max(20, hist.length),
              y1: lo - 0.05 * (hi - lo + 1e-6),
              ym: hi + 0.05 * (hi - lo + 1e-6),
            });
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 2;
          ctx.beginPath();
          hist.forEach((v, i) => (i ? ctx.lineTo(B.X(i), B.Y(v)) : ctx.moveTo(B.X(i), B.Y(v))));
          ctx.stroke();
        } else {
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText('score during fitting: press Fit the settings', 8, h1 + 13);
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'gp', activeStepIndex);
      setReadout(root, 'n', D.length);
      setReadout(root, 'ml', D.length ? f2(ml) : '–');
      setReadout(root, 'fit', D.length ? f2(dataFitTerm) : '–');
      setReadout(root, 'cx', D.length ? f2(complexityTerm) : '–');
      setReadout(root, 'set', f2(settings.l) + ' / ' + f2(settings.f) + ' / ' + f2(settings.n));
    };
    tp.on = () => draw();
    data();
    newZ();
    reset();
    const anim = Anim(() => {
      if (fin) {
        playback.stop();
        return;
      }
      sampleAnimationPhase += 0.02;
      fitStep();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#gp-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'gp', {
      done: () => fin,
      restart: () => {
        hist = [];
        fin = false;
        it = 0;
        ad = null;
      },
      count: () => it,
      what: 'steps',
    });
    const wig = Anim(() => {
      if (anim.on || !settings.w) return;
      sampleAnimationPhase += 0.02;
      draw();
    });
    $('#gp-step', root).onclick = () => {
      playback.stop();
      if (fin) {
        hist = [];
        fin = false;
        it = 0;
        ad = null;
      }
      fitStep();
      draw();
    };
    $('#gp-reset', root).onclick = () => {
      playback.stop();
      settings.l = 1;
      settings.f = 1;
      settings.n = 0.2;
      lB.set(1);
      fB.set(1);
      nB.set(0.2);
      data();
      newZ();
      reset();
      draw();
    };
    const ch = () => {
      playback.stop();
      reset();
      draw();
    };
    lB = bind(root, 'gp-l', (v) => {
      settings.l = v;
      ch();
    });
    fB = bind(root, 'gp-f', (v) => {
      settings.f = v;
      ch();
    });
    nB = bind(root, 'gp-n', (v) => {
      settings.n = v;
      ch();
    });
    segBind(root, 'gp-d', settings.d, (v) => {
      settings.d = v;
      data();
      ch();
    });
    segBind(root, 'gp-k', settings.k, (v) => {
      settings.k = v;
      ch();
    });
    $('#gp-w', root).onchange = (e) => {
      settings.w = e.target.checked;
    };
    xyUp(root, 'gp', (q) => {
      D = q.slice(0, 60);
      ch();
    });
    c.style.cursor = 'crosshair';
    c.addEventListener('click', (e) => {
      if (!V) return;
      const [x, y] = canvasXY(c, V, e),
        k = D.findIndex((d) => Math.hypot((d[0] - x) * V.s, (d[1] - y) * V.s) < 10);
      if (k >= 0) D.splice(k, 1);
      else if (D.length < 60) D.push([clamp(x, -4, 4), clamp(y, -3, 3)]);
      ch();
    });
    return {
      draw,
      stop: () => {
        playback.stop();
        wig.stop();
      },
      auto: () => {
        wig.start();
      },
    };
  },
};
