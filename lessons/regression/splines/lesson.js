/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  fit,
  grab,
  markTP,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { dot, lsq, splineBasis } from '../shared.js';

export default {
  init(root) {
    const c = $('#sp-c', root),
      cd = $('#sp-d', root),
      cap = $('#sp-cap', root),
      S = { kind: 'natural', K: 4, li: 18, noise: 0.25, n: 40, f: 'sine' },
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
      etr = 0,
      ete = 0,
      bestK = 0,
      pb;
    const kx = () => (S.kind === 'pen' ? 14 : S.K),
      bas = (x) => splineBasis(S.kind, x, kx());
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
      for (let i = 0; i < S.n; i++) {
        const x = r() * 2 - 1;
        pts.push({ x, y: F[S.f](x) + S.noise * rn(), test: r() < 0.3 });
      }
    };
    const refit = () => {
      b = fitK(S.kind, kx(), LAM(S.li));
      const tr = pts.filter((p) => !p.test),
        te = pts.filter((p) => p.test);
      etr = err(tr, S.kind, kx(), b);
      ete = err(te, S.kind, kx(), b);
      bestK = -1;
      if (S.kind !== 'pen') {
        let be = 1 / 0;
        for (let k = 0; k <= 12; k++) {
          const bb = fitK(S.kind, k),
            e = err(te, S.kind, k, bb);
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
      const t = T(),
        K = kx(),
        kn = [];
      for (let j = 1; j <= K; j++) kn.push(-1 + (2 * j) / (K + 1));
      const XR = [-1.4, 1.4];
      const deco = (g, P, yr) => {
        g.fillStyle = mix(t.muted, t.muted, 0, 0.1);
        g.fillRect(P.X(XR[0]), P.Y(yr[1]), P.X(-1) - P.X(XR[0]), P.Y(yr[0]) - P.Y(yr[1]));
        g.fillRect(P.X(1), P.Y(yr[1]), P.X(XR[1]) - P.X(1), P.Y(yr[0]) - P.Y(yr[1]));
        g.setLineDash([2, 4]);
        g.strokeStyle = t.muted;
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
        const { g, w, h } = fit(c),
          YR = [-2.4, 2.4],
          P = plot(g, w, h, XR, YR, t, { yticks: [-2, -1, 0, 1, 2], xstep: 0.5, xlabel: 'x' });
        clipPlot(g, P);
        deco(g, P, YR);
        g.setLineDash([6, 5]);
        curve(g, P, F[S.f], XR[0], XR[1]);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.5;
        g.stroke();
        g.setLineDash([]);
        curve(g, P, (x) => clamp(f(x), -9, 9), XR[0], XR[1], 560);
        g.strokeStyle = t.gold;
        g.lineWidth = 3.2;
        g.stroke();
        g.fillStyle = t.gold;
        for (const k of kn) {
          g.beginPath();
          g.arc(P.X(k), P.Y(clamp(f(k), -9, 9)), 3.4, 0, 7);
          g.fill();
        }
        g.strokeStyle = t.ok;
        g.lineWidth = 1.5;
        for (const p of pts)
          if (p.test) {
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 4, 0, 7);
            g.stroke();
          }
        g.fillStyle = t.accent;
        for (const p of pts)
          if (!p.test) {
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 4.2, 0, 7);
            g.fill();
          }
        markTP(g, P.X(tp.v[0]), P.Y(clamp(f(tp.v[0]), -9, 9)), t);
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('dotted lines = knots, dashed = truth', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g, w, h } = fit(cd);
        let mx = 1;
        for (let i = 0; i <= 200; i++) {
          const v = Math.abs(d2(-1 + (2 * i) / 200));
          if (isFinite(v)) mx = Math.max(mx, v);
        }
        mx = Math.min(mx * 1.15, 400);
        const YR = [-mx, mx],
          P = plot(g, w, h, XR, YR, t, { xstep: 0.5, xlabel: 'x' });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        [-mx / 1.15, 0, mx / 1.15].forEach((v) => g.fillText(v.toFixed(0), P.m.l - 6, P.Y(v) + 4));
        clipPlot(g, P);
        deco(g, P, YR);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(P.X(XR[0]), P.Y(0));
        g.lineTo(P.X(XR[1]), P.Y(0));
        g.stroke();
        curve(g, P, (x) => clamp(d2(x), -mx * 2, mx * 2), XR[0], XR[1], 700);
        g.strokeStyle = t.ok;
        g.lineWidth = 2.6;
        g.stroke();
        if (S.kind === 'natural' || S.kind === 'pen' || S.kind === 'cubic') {
          g.fillStyle = t.gold;
          g.strokeStyle = t.bg;
          g.lineWidth = 2;
          [-1, 1].forEach((x) => {
            g.beginPath();
            g.arc(P.X(x), P.Y(clamp(2 * d2(x * 0.99) - d2(x * 0.98), -mx, mx)), 6, 0, 7);
            g.fill();
            g.stroke();
          });
        }
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.ok;
        g.fillText("f''(x)", P.m.l + 8, P.m.t + 8);
        g.fillStyle = t.cut;
        g.fillText('● the two ends of the data', P.m.l + 8, P.m.t + 23);
      }
      cap.textContent = CAP[S.kind];
      const e1 = 2 * d2(-0.99) - d2(-0.98),
        e2 = 2 * d2(0.99) - d2(0.98),
        z = (v) => (Math.abs(v) < 0.05 ? '0.00' : v.toFixed(1));
      tp.out.textContent =
        'At x = ' +
        f2(tp.v[0]) +
        ' this spline predicts y = ' +
        f2(f(tp.v[0])) +
        '. The true curve there is ' +
        f2(F[S.f](tp.v[0])) +
        '.' +
        (Math.abs(tp.v[0]) > 1
          ? ' This point is outside the data, so the prediction is an extrapolation.'
          : '');
      setR(root, 'df', String(b.length) + (S.kind === 'pen' ? ' (penalised)' : ''));
      setR(root, 'tr', f3(etr));
      setR(root, 'te', isFinite(ete) ? (ete > 999 ? ete.toExponential(1) : f3(ete)) : '–');
      setR(
        root,
        'end',
        S.kind === 'step' || S.kind === 'linear' ? 'not defined at knots' : z(e1) + ' and ' + z(e2),
      );
      setR(root, 'best', bestK < 0 ? 'set by λ here' : String(bestK));
      root
        .querySelectorAll('#sp-kind button')
        .forEach((x) => x.setAttribute('aria-pressed', x.dataset.k === S.kind ? 'true' : 'false'));
    };
    const ks = bind(
        root,
        'sp-k',
        (v) => {
          S.K = v;
          refit();
          draw();
        },
        (v) => String(v),
      ),
      ls = bind(
        root,
        'sp-l',
        (v) => {
          S.li = v;
          refit();
          draw();
        },
        (v) => (-7 + (8 * v) / 40).toFixed(1),
      );
    let fr = 0,
      dir = 1;
    const anim = Anim(() => {
      if (S.kind === 'pen') {
        if (++fr % 4) return;
        let v = S.li + dir;
        if (v > 40) {
          dir = -1;
          v = 39;
        }
        if (v < 0) {
          dir = 1;
          v = 1;
        }
        S.li = v;
        ls.set(v);
      } else {
        if (++fr % 40) return;
        let v = S.K + dir;
        if (v > 12) {
          dir = -1;
          v = 11;
        }
        if (v < 0) {
          dir = 1;
          v = 1;
        }
        S.K = v;
        ks.set(v);
      }
      refit();
      draw();
    });
    pb = playBtn($('#sp-play', root), anim);
    grab(ks.el, pb);
    grab(ls.el, pb);
    $('#sp-kind', root).addEventListener('click', (e) => {
      const k = e.target.dataset && e.target.dataset.k;
      if (!k) return;
      S.kind = k;
      refit();
      draw();
    });
    const re = () => {
      gen();
      refit();
      draw();
    };
    $('#sp-f', root).onchange = (e) => {
      S.f = e.target.value;
      re();
    };
    bind(root, 'sp-s', (v) => {
      S.noise = v;
      re();
    });
    bind(
      root,
      'sp-n',
      (v) => {
        S.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#sp-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
