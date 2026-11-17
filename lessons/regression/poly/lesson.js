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
import { dot, legendre, lsq } from '../shared.js';

export default {
  init(root) {
    const mainCanvas = $('#pr-c', root),
      ce = $('#pr-e', root),
      settings = { d: 3, noise: 0.25, n: 30, f: 'sine' },
      F = {
        sine: (x) => Math.sin(2.8 * x),
        cubic: (x) => 2.4 * x * x * x - 1.3 * x,
        bump: (x) => 1.6 * Math.exp(-9 * x * x) - 0.6,
        line: (x) => 0.9 * x,
      },
      MAX_DEGREE = 14;
    let seed = 8,
      pts = [],
      fits = [],
      trainMseByDegree = [],
      testMseByDegree = [],
      playback;
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < settings.n; i++) {
        const x = r() * 2 - 1;
        pts.push({ x, y: F[settings.f](x) + settings.noise * rn(), test: r() < 0.3 });
      }
      const tr = pts.filter((p) => !p.test),
        te = pts.filter((p) => p.test);
      fits = [];
      trainMseByDegree = [];
      testMseByDegree = [];
      for (let d = 1; d <= MAX_DEGREE; d++) {
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
        trainMseByDegree[d] = e(tr);
        testMseByDegree[d] = e(te);
      }
    };
    gen();
    const tp = testPoint(root, 'pr', ['Your x'], [0.4], [[-1, 1]]);
    tp.on = () => draw();
    const draw = () => {
      const colors = readThemeColors(),
        b = fits[settings.d];
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = plot(ctx, w, h, [-1.05, 1.05], [-2.2, 2.2], colors, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: 'x',
          });
        clipPlot(ctx, P);
        ctx.setLineDash([6, 5]);
        curve(ctx, P, F[settings.f], -1, 1);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.6;
        ctx.stroke();
        ctx.setLineDash([]);
        curve(ctx, P, (x) => clamp(dot(b, legendre(x, settings.d)), -9, 9), -1, 1, 400);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 3.2;
        ctx.stroke();
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
        markTP(ctx, P.X(tp.v[0]), P.Y(clamp(dot(b, legendre(tp.v[0], settings.d)), -9, 9)), colors);
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.muted;
        ctx.fillText('dashed = true curve, hollow = test', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(ce),
          LY = [-2.6, 1.2],
          P = plot(ctx, w, h, [0.5, MAX_DEGREE + 0.5], LY, colors, { xlabel: 'degree' }),
          lg = (v) => clamp(Math.log10(Math.max(v, 1e-9)), LY[0], LY[1]);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        [-2, -1, 0, 1].forEach((v) => {
          ctx.strokeStyle = colors.line;
          ctx.beginPath();
          ctx.moveTo(P.m.l, P.Y(v));
          ctx.lineTo(w - P.m.r, P.Y(v));
          ctx.stroke();
          ctx.fillText(
            v === 0 ? '1' : v === 1 ? '10' : v === -1 ? '0.1' : '0.01',
            P.m.l - 6,
            P.Y(v) + 4,
          );
        });
        ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.5);
        ctx.lineWidth = 10;
        ctx.beginPath();
        ctx.moveTo(P.X(settings.d), P.Y(LY[0]));
        ctx.lineTo(P.X(settings.d), P.Y(LY[1]));
        ctx.stroke();
        [
          [trainMseByDegree, colors.accent],
          [testMseByDegree, colors.ok],
        ].forEach(([E, col]) => {
          ctx.strokeStyle = col;
          ctx.fillStyle = col;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          for (let d = 1; d <= MAX_DEGREE; d++)
            d > 1 ? ctx.lineTo(P.X(d), P.Y(lg(E[d]))) : ctx.moveTo(P.X(d), P.Y(lg(E[d])));
          ctx.stroke();
          for (let d = 1; d <= MAX_DEGREE; d++) {
            ctx.beginPath();
            ctx.arc(P.X(d), P.Y(lg(E[d])), d === settings.d ? 5.5 : 3, 0, 7);
            ctx.fill();
          }
        });
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.accent;
        ctx.fillText('train', P.m.l + 8, P.m.t + 8);
        ctx.fillStyle = colors.ok;
        ctx.fillText('test', P.m.l + 8, P.m.t + 23);
      }
      let bd = 1;
      for (let d = 2; d <= MAX_DEGREE; d++) if (testMseByDegree[d] < testMseByDegree[bd]) bd = d;
      tp.out.textContent =
        'At x = ' +
        f2(tp.v[0]) +
        ' the degree-' +
        settings.d +
        ' curve predicts y = ' +
        f2(dot(b, legendre(tp.v[0], settings.d))) +
        '. The true curve there is ' +
        f2(F[settings.f](tp.v[0])) +
        '.';
      setReadout(root, 'd', settings.d + ', ' + (settings.d + 1));
      setReadout(root, 'tr', f3(trainMseByDegree[settings.d]));
      setReadout(
        root,
        'te',
        isFinite(testMseByDegree[settings.d])
          ? testMseByDegree[settings.d] > 999
            ? testMseByDegree[settings.d].toExponential(1)
            : f3(testMseByDegree[settings.d])
          : '–',
      );
      setReadout(root, 'best', String(bd));
    };
    const ds = bind(
      root,
      'pr-d',
      (v) => {
        settings.d = v;
        draw();
      },
      (v) => String(v),
    );
    let frameCount = 0,
      dir = 1;
    const anim = Anim(() => {
      if (++frameCount % 34) return;
      let d = settings.d + dir;
      if (d > MAX_DEGREE) {
        dir = -1;
        d = MAX_DEGREE - 1;
      }
      if (d < 1) {
        dir = 1;
        d = 2;
      }
      settings.d = d;
      ds.set(d);
      draw();
    });
    playback = playBtn($('#pr-play', root), anim);
    grab(ds.el, playback);
    const re = () => {
      gen();
      draw();
    };
    $('#pr-f', root).onchange = (e) => {
      settings.f = e.target.value;
      re();
    };
    bind(root, 'pr-s', (v) => {
      settings.noise = v;
      re();
    });
    bind(
      root,
      'pr-n',
      (v) => {
        settings.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#pr-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
