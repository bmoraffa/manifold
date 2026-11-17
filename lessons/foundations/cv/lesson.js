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
  f3,
  prepareCanvas,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  setReadout,
} from '../../../js/core/shared.js';
import { dot, legendre, lsq } from '../../regression/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#cv-c', root),
      ce = $('#cv-e', root),
      settings = { k: 5, d: 3, n: 40, noise: 0.3 },
      Fq = (x) => Math.sin(2.8 * x);
    let seed = 14,
      pts = [],
      fold = 0,
      FE = [],
      cvMseByDegree = [],
      testMseByDegree = [],
      trainMseByDegree = [],
      bcur = null,
      playback;
    const fitOn = (P, d) =>
        lsq(
          P.map((p) => legendre(p.x, d)),
          P.map((p) => p.y),
          1e-8,
        ),
      er = (P, b, d) =>
        P.reduce((s, p) => s + (p.y - clamp(dot(b, legendre(p.x, d)), -20, 20)) ** 2, 0) / P.length;
    const cvFor = (d) => {
      const fe = [];
      for (let j = 0; j < settings.k; j++) {
        const tr = pts.filter((p) => p.f !== j),
          va = pts.filter((p) => p.f === j);
        fe.push(va.length && tr.length > d ? er(va, fitOn(tr, d), d) : NaN);
      }
      return fe;
    };
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < settings.n; i++) {
        const x = r() * 2 - 1;
        pts.push({ x, y: Fq(x) + settings.noise * rn() });
      }
      const ix = pts.map((_, i) => i);
      for (let i = ix.length - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [ix[i], ix[j]] = [ix[j], ix[i]];
      }
      ix.forEach((q, i) => (pts[q].f = i % settings.k));
      const big = [];
      for (let i = 0; i < 3000; i++) {
        const x = r() * 2 - 1;
        big.push({ x, y: Fq(x) + settings.noise * rn() });
      }
      cvMseByDegree = [];
      testMseByDegree = [];
      trainMseByDegree = [];
      for (let d = 1; d <= 12; d++) {
        const fe = cvFor(d).filter(isFinite);
        cvMseByDegree[d] = fe.length ? fe.reduce((a, b) => a + b, 0) / fe.length : NaN;
        const b = fitOn(pts, d);
        testMseByDegree[d] = er(big, b, d);
        trainMseByDegree[d] = er(pts, b, d);
      }
    };
    const turn = () => {
      FE = cvFor(settings.d);
      bcur = fitOn(
        pts.filter((p) => p.f !== fold),
        settings.d,
      );
    };
    gen();
    turn();
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = plot(ctx, w, h, [-1.05, 1.05], [-2.2, 2.6], colors, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: 'x',
          }),
          bw = (w - P.m.l - P.m.r) / settings.k;
        for (let j = 0; j < settings.k; j++) {
          ctx.fillStyle = j === fold ? colors.ok : mix(colors.accent, colors.accent, 0, 0.55);
          ctx.fillRect(P.m.l + j * bw + 1, P.m.t - 4, bw - 2, 9);
        }
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('blocks: burgundy is held out this turn', P.m.l, P.m.t + 19);
        clipPlot(ctx, P);
        ctx.setLineDash([6, 5]);
        curve(ctx, P, Fq, -1, 1);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.setLineDash([]);
        curve(ctx, P, (x) => clamp(dot(bcur, legendre(x, settings.d)), -9, 9), -1, 1, 300);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 3.2;
        ctx.stroke();
        for (const p of pts) {
          if (p.f === fold) {
            const yh = clamp(dot(bcur, legendre(p.x, settings.d)), -9, 9);
            ctx.strokeStyle = mix(colors.ok, colors.ok, 0, 0.6);
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(P.X(p.x), P.Y(p.y));
            ctx.lineTo(P.X(p.x), P.Y(yh));
            ctx.stroke();
            ctx.strokeStyle = colors.ok;
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            ctx.arc(P.X(p.x), P.Y(p.y), 5, 0, 7);
            ctx.stroke();
          } else {
            ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.85);
            ctx.beginPath();
            ctx.arc(P.X(p.x), P.Y(p.y), 3.8, 0, 7);
            ctx.fill();
          }
        }
        ctx.restore();
      }
      {
        const { g: ctx, w, h } = prepareCanvas(ce),
          h1 = h * 0.42,
          fe = FE.filter(isFinite),
          mean = fe.length ? fe.reduce((a, b) => a + b, 0) / fe.length : 0,
          mx = Math.max(0.05, ...fe) * 1.15,
          m = { l: 38, r: 12 };
        ctx.font = MONO;
        const bw = (w - m.l - m.r) / settings.k,
          Yb = (v) => h1 - 12 - (Math.min(v, mx) / mx) * (h1 - 30);
        for (let j = 0; j < settings.k; j++) {
          if (!isFinite(FE[j])) continue;
          ctx.fillStyle = j === fold ? colors.ok : mix(colors.ok, colors.ok, 0, 0.35);
          ctx.fillRect(m.l + j * bw + 3, Yb(FE[j]), bw - 6, h1 - 12 - Yb(FE[j]));
        }
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(m.l, Yb(mean));
        ctx.lineTo(w - m.r, Yb(mean));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colors.ink;
        ctx.textAlign = 'right';
        ctx.fillText('average = CV estimate ' + f3(mean), w - m.r, Math.max(10, Yb(mean) - 5));
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, h1 - 12);
        ctx.lineTo(w - m.r, h1 - 12);
        ctx.stroke();
        ctx.save();
        ctx.translate(0, h1);
        const hh = h - h1,
          LY = [-2.3, 0.8],
          P = plot(ctx, w, hh, [0.5, 12.5], LY, colors, { xlabel: 'degree' }),
          lg = (v) => clamp(Math.log10(Math.max(v, 1e-9)), LY[0], LY[1]);
        clipPlot(ctx, P);
        ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.5);
        ctx.lineWidth = 9;
        ctx.beginPath();
        ctx.moveTo(P.X(settings.d), P.Y(LY[0]));
        ctx.lineTo(P.X(settings.d), P.Y(LY[1]));
        ctx.stroke();
        [
          [trainMseByDegree, colors.accent],
          [cvMseByDegree, colors.ok],
          [testMseByDegree, colors.ink],
        ].forEach(([E, cl]) => {
          ctx.strokeStyle = cl;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          let on = false;
          for (let d = 1; d <= 12; d++) {
            if (!isFinite(E[d])) {
              on = false;
              continue;
            }
            on ? ctx.lineTo(P.X(d), P.Y(lg(E[d]))) : ctx.moveTo(P.X(d), P.Y(lg(E[d])));
            on = true;
          }
          ctx.stroke();
        });
        ctx.restore();
        ctx.textAlign = 'left';
        let lx = P.m.l + 8;
        [
          [colors.accent, 'training'],
          [colors.ok, 'cross-validation'],
          [colors.ink, 'true test'],
        ].forEach(([cl, sx]) => {
          ctx.fillStyle = cl;
          ctx.fillText(sx, lx, P.m.t + 6);
          lx += ctx.measureText(sx).width + 16;
        });
        ctx.restore();
      }
      const fe = FE.filter(isFinite),
        mean = fe.reduce((a, b) => a + b, 0) / (fe.length || 1),
        sd = Math.sqrt(fe.reduce((a, b) => a + (b - mean) ** 2, 0) / (fe.length || 1));
      let pk = 1;
      for (let d = 2; d <= 12; d++) if (cvMseByDegree[d] < cvMseByDegree[pk]) pk = d;
      setReadout(root, 'cv', f3(mean));
      setReadout(root, 'sd', f3(sd));
      setReadout(root, 'tr', f3(trainMseByDegree[settings.d]));
      setReadout(root, 'tt', f3(testMseByDegree[settings.d]));
      setReadout(root, 'pick', String(pk));
    };
    let frameCount = 0;
    const anim = Anim(() => {
      if (++frameCount % 75) return;
      fold = (fold + 1) % settings.k;
      turn();
      draw();
    });
    playback = playBtn($('#cv-play', root), anim);
    const re = () => {
      fold = 0;
      gen();
      turn();
      draw();
    };
    bind(
      root,
      'cv-k',
      (v) => {
        settings.k = v;
        re();
      },
      (v) => String(v),
    );
    bind(
      root,
      'cv-d',
      (v) => {
        settings.d = v;
        turn();
        draw();
      },
      (v) => String(v),
    );
    bind(
      root,
      'cv-n',
      (v) => {
        settings.n = v;
        re();
      },
      (v) => String(v),
    );
    bind(root, 'cv-s', (v) => {
      settings.noise = v;
      re();
    });
    $('#cv-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
