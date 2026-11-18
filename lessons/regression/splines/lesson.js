/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  bind,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  prepareCanvas,
  grab,
  markTP,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { dot, lsq, splineBasis } from '../shared.js';

export default {
  init(root) {
    const mainCanvas = $('#sp-c', root),
      cd = $('#sp-d', root),
      cap = $('#sp-cap', root),
      settings = { kind: 'natural', K: 4, li: 18, noise: 0.25, n: 40, f: 'sine' },
      F = {
        sine: (x) => Math.sin(2.8 * x),
        cubic: (x) => 2.4 * x * x * x - 1.3 * x,
        bump: (x) => 1.6 * Math.exp(-9 * x * x) - 0.6,
        line: (x) => 0.9 * x,
      },
      LAM = (i) => Math.pow(10, -7 + (8 * i) / 40);
    let seed = 8,
      pts = [],
      b = [],
      trainMse = 0,
      testMse = 0,
      bestK = 0,
      playback;
    const effectiveKnotCount = () => (settings.kind === 'pen' ? 14 : settings.K),
      bas = (x) => splineBasis(settings.kind, x, effectiveKnotCount());
    const fitK = (kind, K, lam) => {
      const tr = pts.filter((p) => !p.test),
        X = tr.map((p) => splineBasis(kind, p.x, K)),
        y = tr.map((p) => p.y);
      let rg = 1e-9;
      if (kind === 'pen') rg = X[0].map((_, j) => (j < 4 ? 1e-9 : lam));
      return lsq(X, y, rg);
    };
    const err = (P, kind, K, bb) =>
      P.length
        ? P.reduce((s, p) => s + (p.y - dot(bb, splineBasis(kind, p.x, K))) ** 2, 0) / P.length
        : NaN;
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < settings.n; i++) {
        const x = r() * 2 - 1;
        pts.push({ x, y: F[settings.f](x) + settings.noise * rn(), test: r() < 0.3 });
      }
    };
    const refit = () => {
      b = fitK(settings.kind, effectiveKnotCount(), LAM(settings.li));
      const tr = pts.filter((p) => !p.test),
        te = pts.filter((p) => p.test);
      trainMse = err(tr, settings.kind, effectiveKnotCount(), b);
      testMse = err(te, settings.kind, effectiveKnotCount(), b);
      bestK = -1;
      if (settings.kind !== 'pen') {
        let be = 1 / 0;
        for (let k = 0; k <= 12; k++) {
          const bb = fitK(settings.kind, k),
            e = err(te, settings.kind, k, bb);
          if (e < be) {
            be = e;
            bestK = k;
          }
        }
      }
    };
    gen();
    refit();
    const tp = testPoint(root, 'sp', ['Your x (try beyond ±1)'], [0.4], [[-1.4, 1.4]]);
    tp.on = () => draw();
    const CAP = {
      step: 'Steps: a separate constant on each stretch. The pieces do not even meet, so the fit jumps at every knot.',
      linear:
        'Linear spline: a straight piece on each stretch, joined end to end. Continuous, but with a corner at every knot, so the curvature is a spike there and zero elsewhere.',
      cubic:
        'Cubic spline: a cubic on each stretch, matching value, slope and curvature at every knot. The curvature on the right is one unbroken zig-zag line. Outside the data the fit runs away.',
      natural:
        'Natural cubic spline: a cubic spline whose second derivative is forced to zero at both ends. Look right: the curvature starts and finishes at exactly zero, and outside the data the fit continues as a straight line.',
      pen: 'Smoothing spline: many knots, with a price λ on curvature. Raise λ and the curvature on the right is pressed toward zero until only a straight line is left.',
    };
    const f = (x) => dot(b, bas(x)),
      d2 = (x) => {
        const e = 0.004;
        return (f(x + e) - 2 * f(x) + f(x - e)) / (e * e);
      };
    const draw = () => {
      const colors = readThemeColors(),
        K = effectiveKnotCount(),
        kn = [];
      for (let j = 1; j <= K; j++) kn.push(-1 + (2 * j) / (K + 1));
      const XR = [-1.4, 1.4];
      const deco = (g, P, yr) => {
        g.fillStyle = mix(colors.muted, colors.muted, 0, 0.1);
        g.fillRect(P.X(XR[0]), P.Y(yr[1]), P.X(-1) - P.X(XR[0]), P.Y(yr[0]) - P.Y(yr[1]));
        g.fillRect(P.X(1), P.Y(yr[1]), P.X(XR[1]) - P.X(1), P.Y(yr[0]) - P.Y(yr[1]));
        g.setLineDash([2, 4]);
        g.strokeStyle = colors.muted;
        g.lineWidth = 1;
        for (const k of kn) {
          g.beginPath();
          g.moveTo(P.X(k), P.Y(yr[0]));
          g.lineTo(P.X(k), P.Y(yr[1]));
          g.stroke();
        }
        g.setLineDash([]);
      };
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          YR = [-2.4, 2.4],
          P = plot(ctx, w, h, XR, YR, colors, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: 'x',
          });
        clipPlot(ctx, P);
        deco(ctx, P, YR);
        ctx.setLineDash([6, 5]);
        curve(ctx, P, F[settings.f], XR[0], XR[1]);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
        curve(ctx, P, (x) => clamp(f(x), -9, 9), XR[0], XR[1], 560);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 3.2;
        ctx.stroke();
        ctx.fillStyle = colors.gold;
        for (const k of kn) {
          ctx.beginPath();
          ctx.arc(P.X(k), P.Y(clamp(f(k), -9, 9)), 3.4, 0, 7);
          ctx.fill();
        }
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 1.5;
        for (const p of pts)
          if (p.test) {
            ctx.beginPath();
            ctx.arc(P.X(p.x), P.Y(p.y), 4, 0, 7);
            ctx.stroke();
          }
        ctx.fillStyle = colors.accent;
        for (const p of pts)
          if (!p.test) {
            ctx.beginPath();
            ctx.arc(P.X(p.x), P.Y(p.y), 4.2, 0, 7);
            ctx.fill();
          }
        markTP(ctx, P.X(tp.v[0]), P.Y(clamp(f(tp.v[0]), -9, 9)), colors);
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.muted;
        ctx.fillText('dotted lines = knots, dashed = truth', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cd);
        let mx = 1;
        for (let i = 0; i <= 200; i++) {
          const v = Math.abs(d2(-1 + (2 * i) / 200));
          if (isFinite(v)) mx = Math.max(mx, v);
        }
        mx = Math.min(mx * 1.15, 400);
        const YR = [-mx, mx],
          P = plot(ctx, w, h, XR, YR, colors, { xstep: 0.5, xlabel: 'x' });
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        [-mx / 1.15, 0, mx / 1.15].forEach((v) =>
          ctx.fillText(v.toFixed(0), P.m.l - 6, P.Y(v) + 4),
        );
        clipPlot(ctx, P);
        deco(ctx, P, YR);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(P.X(XR[0]), P.Y(0));
        ctx.lineTo(P.X(XR[1]), P.Y(0));
        ctx.stroke();
        curve(ctx, P, (x) => clamp(d2(x), -mx * 2, mx * 2), XR[0], XR[1], 700);
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 2.6;
        ctx.stroke();
        if (settings.kind === 'natural' || settings.kind === 'pen' || settings.kind === 'cubic') {
          ctx.fillStyle = colors.gold;
          ctx.strokeStyle = colors.bg;
          ctx.lineWidth = 2;
          [-1, 1].forEach((x) => {
            ctx.beginPath();
            ctx.arc(P.X(x), P.Y(clamp(2 * d2(x * 0.99) - d2(x * 0.98), -mx, mx)), 6, 0, 7);
            ctx.fill();
            ctx.stroke();
          });
        }
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.ok;
        ctx.fillText("f''(x)", P.m.l + 8, P.m.t + 8);
        ctx.fillStyle = colors.cut;
        ctx.fillText('● the two ends of the data', P.m.l + 8, P.m.t + 23);
      }
      cap.textContent = CAP[settings.kind];
      const e1 = 2 * d2(-0.99) - d2(-0.98),
        e2 = 2 * d2(0.99) - d2(0.98),
        z = (v) => (Math.abs(v) < 0.05 ? '0.00' : v.toFixed(1));
      tp.out.textContent =
        'At x = ' +
        f2(tp.v[0]) +
        ' this spline predicts y = ' +
        f2(f(tp.v[0])) +
        '. The true curve there is ' +
        f2(F[settings.f](tp.v[0])) +
        '.' +
        (Math.abs(tp.v[0]) > 1
          ? ' This point is outside the data, so the prediction is an extrapolation.'
          : '');
      setReadout(root, 'df', String(b.length) + (settings.kind === 'pen' ? ' (penalised)' : ''));
      setReadout(root, 'tr', f3(trainMse));
      setReadout(
        root,
        'te',
        isFinite(testMse) ? (testMse > 999 ? testMse.toExponential(1) : f3(testMse)) : '–',
      );
      setReadout(
        root,
        'end',
        settings.kind === 'step' || settings.kind === 'linear'
          ? 'not defined at knots'
          : z(e1) + ' and ' + z(e2),
      );
      setReadout(root, 'best', bestK < 0 ? 'set by λ here' : String(bestK));
      root
        .querySelectorAll('#sp-kind button')
        .forEach((x) =>
          x.setAttribute('aria-pressed', x.dataset.k === settings.kind ? 'true' : 'false'),
        );
    };
    const ks = bind(
        root,
        'sp-k',
        (v) => {
          settings.K = v;
          refit();
          draw();
        },
        (v) => String(v),
      ),
      ls = bind(
        root,
        'sp-l',
        (v) => {
          settings.li = v;
          refit();
          draw();
        },
        (v) => (-7 + (8 * v) / 40).toFixed(1),
      );
    let frameCount = 0,
      dir = 1;
    const anim = Anim(() => {
      if (settings.kind === 'pen') {
        if (++frameCount % 4) return;
        let v = settings.li + dir;
        if (v > 40) {
          dir = -1;
          v = 39;
        }
        if (v < 0) {
          dir = 1;
          v = 1;
        }
        settings.li = v;
        ls.set(v);
      } else {
        if (++frameCount % 40) return;
        let v = settings.K + dir;
        if (v > 12) {
          dir = -1;
          v = 11;
        }
        if (v < 0) {
          dir = 1;
          v = 1;
        }
        settings.K = v;
        ks.set(v);
      }
      refit();
      draw();
    });
    playback = playBtn($('#sp-play', root), anim);
    grab(ks.el, playback);
    grab(ls.el, playback);
    $('#sp-kind', root).addEventListener('click', (e) => {
      const k = e.target.dataset && e.target.dataset.k;
      if (!k) return;
      settings.kind = k;
      refit();
      draw();
    });
    const re = () => {
      gen();
      refit();
      draw();
    };
    $('#sp-f', root).onchange = (e) => {
      settings.f = e.target.value;
      re();
    };
    bind(root, 'sp-s', (v) => {
      settings.noise = v;
      re();
    });
    bind(
      root,
      'sp-n',
      (v) => {
        settings.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#sp-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
