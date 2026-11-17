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
import { dot, legendre, lsq } from '../shared.js';

export default {
  init(root) {
    const c = $('#pr-c', root),
      ce = $('#pr-e', root),
      S = { d: 3, noise: 0.25, n: 30, f: 'sine' },
      F = {
        sine: (x) => Math.sin(2.8 * x),
        cubic: (x) => 2.4 * x * x * x - 1.3 * x,
        bump: (x) => 1.6 * Math.exp(-9 * x * x) - 0.6,
        line: (x) => 0.9 * x,
      },
      DM = 14;
    let seed = 8,
      pts = [],
      fits = [],
      etr = [],
      ete = [],
      pb;
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < S.n; i++) {
        const x = r() * 2 - 1;
        pts.push({ x, y: F[S.f](x) + S.noise * rn(), test: r() < 0.3 });
      }
      const tr = pts.filter((p) => !p.test),
        te = pts.filter((p) => p.test);
      fits = [];
      etr = [];
      ete = [];
      for (let d = 1; d <= DM; d++) {
        const b = lsq(
          tr.map((p) => legendre(p.x, d)),
          tr.map((p) => p.y),
          1e-9,
        );
        fits[d] = b;
        const e = (P) =>
          P.length
            ? P.reduce((s, p) => s + (p.y - dot(b, legendre(p.x, d))) ** 2, 0) / P.length
            : NaN;
        etr[d] = e(tr);
        ete[d] = e(te);
      }
    };
    gen();
    const tp = testPoint(root, 'pr', ['Your x'], [0.4], [[-1, 1]]);
    tp.on = () => draw();
    const draw = () => {
      const t = T(),
        b = fits[S.d];
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, [-1.05, 1.05], [-2.2, 2.2], t, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: 'x',
          });
        clipPlot(g, P);
        g.setLineDash([6, 5]);
        curve(g, P, F[S.f], -1, 1);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.6;
        g.stroke();
        g.setLineDash([]);
        curve(g, P, (x) => clamp(dot(b, legendre(x, S.d)), -9, 9), -1, 1, 400);
        g.strokeStyle = t.gold;
        g.lineWidth = 3.2;
        g.stroke();
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
        markTP(g, P.X(tp.v[0]), P.Y(clamp(dot(b, legendre(tp.v[0], S.d)), -9, 9)), t);
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('dashed = true curve, hollow = test', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g, w, h } = fit(ce),
          LY = [-2.6, 1.2],
          P = plot(g, w, h, [0.5, DM + 0.5], LY, t, { xlabel: 'degree' }),
          lg = (v) => clamp(Math.log10(Math.max(v, 1e-9)), LY[0], LY[1]);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        [-2, -1, 0, 1].forEach((v) => {
          g.strokeStyle = t.line;
          g.beginPath();
          g.moveTo(P.m.l, P.Y(v));
          g.lineTo(w - P.m.r, P.Y(v));
          g.stroke();
          g.fillText(
            v === 0 ? '1' : v === 1 ? '10' : v === -1 ? '0.1' : '0.01',
            P.m.l - 6,
            P.Y(v) + 4,
          );
        });
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
        g.lineWidth = 10;
        g.beginPath();
        g.moveTo(P.X(S.d), P.Y(LY[0]));
        g.lineTo(P.X(S.d), P.Y(LY[1]));
        g.stroke();
        [
          [etr, t.accent],
          [ete, t.ok],
        ].forEach(([E, col]) => {
          g.strokeStyle = col;
          g.fillStyle = col;
          g.lineWidth = 2.4;
          g.beginPath();
          for (let d = 1; d <= DM; d++)
            d > 1 ? g.lineTo(P.X(d), P.Y(lg(E[d]))) : g.moveTo(P.X(d), P.Y(lg(E[d])));
          g.stroke();
          for (let d = 1; d <= DM; d++) {
            g.beginPath();
            g.arc(P.X(d), P.Y(lg(E[d])), d === S.d ? 5.5 : 3, 0, 7);
            g.fill();
          }
        });
        g.textAlign = 'left';
        g.fillStyle = t.accent;
        g.fillText('train', P.m.l + 8, P.m.t + 8);
        g.fillStyle = t.ok;
        g.fillText('test', P.m.l + 8, P.m.t + 23);
      }
      let bd = 1;
      for (let d = 2; d <= DM; d++) if (ete[d] < ete[bd]) bd = d;
      tp.out.textContent =
        'At x = ' +
        f2(tp.v[0]) +
        ' the degree-' +
        S.d +
        ' curve predicts y = ' +
        f2(dot(b, legendre(tp.v[0], S.d))) +
        '. The true curve there is ' +
        f2(F[S.f](tp.v[0])) +
        '.';
      setR(root, 'd', S.d + ', ' + (S.d + 1));
      setR(root, 'tr', f3(etr[S.d]));
      setR(
        root,
        'te',
        isFinite(ete[S.d]) ? (ete[S.d] > 999 ? ete[S.d].toExponential(1) : f3(ete[S.d])) : '–',
      );
      setR(root, 'best', String(bd));
    };
    const ds = bind(
      root,
      'pr-d',
      (v) => {
        S.d = v;
        draw();
      },
      (v) => String(v),
    );
    let fr = 0,
      dir = 1;
    const anim = Anim(() => {
      if (++fr % 34) return;
      let d = S.d + dir;
      if (d > DM) {
        dir = -1;
        d = DM - 1;
      }
      if (d < 1) {
        dir = 1;
        d = 2;
      }
      S.d = d;
      ds.set(d);
      draw();
    });
    pb = playBtn($('#pr-play', root), anim);
    grab(ds.el, pb);
    const re = () => {
      gen();
      draw();
    };
    $('#pr-f', root).onchange = (e) => {
      S.f = e.target.value;
      re();
    };
    bind(root, 'pr-s', (v) => {
      S.noise = v;
      re();
    });
    bind(
      root,
      'pr-n',
      (v) => {
        S.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#pr-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
