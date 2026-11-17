/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  readThemeColors,
  bind,
  clamp,
  clipPlot,
  curve,
  f3,
  prepareCanvas,
  npdf,
  playBtn,
  plot,
  ramp,
  randn,
  setReadout,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#sm-c', root),
      xr = [-6, 6],
      settings = { mu: 0, sig: 1, nb: 48 };
    let binCounts, n, s1, s2, fallingSamples, bw;
    const reset = () => {
      bw = 12 / settings.nb;
      binCounts = new Array(settings.nb).fill(0);
      n = 0;
      s1 = 0;
      s2 = 0;
      fallingSamples = [];
    };
    reset();
    const land = (x) => {
      n++;
      s1 += x;
      s2 += x * x;
      const b = Math.floor((x - xr[0]) / bw);
      if (b >= 0 && b < settings.nb) binCounts[b]++;
    };
    const drawSample = () => settings.mu + settings.sig * randn(),
      sp = bind(
        root,
        'sm-speed',
        () => {},
        (v) => String(v),
      );
    const draw = () => {
      const colors = readThemeColors(),
        { g: ctx, w, h } = prepareCanvas(mainCanvas),
        pk = npdf(0, 0, settings.sig),
        P = plot(ctx, w, h, xr, [0, Math.max(0.5, pk * 1.5)], colors, {
          yticks: [0.2, 0.4, 0.6, 0.8, 1],
          xlabel: 'x',
        });
      clipPlot(ctx, P);
      for (let b = 0; b < settings.nb; b++) {
        if (!binCounts[b]) continue;
        const d = binCounts[b] / (n * bw),
          x0 = P.X(xr[0] + b * bw),
          x1 = P.X(xr[0] + (b + 1) * bw);
        ctx.fillStyle = ramp(colors, clamp(0.25 + (0.75 * d) / pk, 0, 1), 0.9);
        ctx.fillRect(x0 + 0.5, P.Y(d), x1 - x0 - 1, P.Y(0) - P.Y(d));
      }
      curve(ctx, P, (x) => npdf(x, settings.mu, settings.sig), xr[0], xr[1]);
      ctx.strokeStyle = colors.ink;
      ctx.lineWidth = 2.2;
      ctx.stroke();
      ctx.fillStyle = colors.gold;
      for (const p of fallingSamples) {
        ctx.beginPath();
        ctx.arc(P.X(p.x), P.m.t + (P.Y(0) - P.m.t) * p.y * p.y, 4, 0, 7);
        ctx.fill();
      }
      ctx.restore();
      const m = n ? s1 / n : 0;
      setReadout(root, 'n', String(n));
      setReadout(root, 'm', n ? f3(m) : '–');
      setReadout(root, 's', n > 1 ? f3(Math.sqrt(Math.max(0, (s2 - n * m * m) / (n - 1)))) : '–');
      setReadout(root, 'se', n ? f3(settings.sig / Math.sqrt(n)) : '–');
    };
    const anim = Anim(() => {
      const r = +sp.el.value;
      if (r <= 8) for (let i = 0; i < r; i++) fallingSamples.push({ x: drawSample(), y: 0 });
      else for (let i = 0; i < r; i++) land(drawSample());
      fallingSamples = fallingSamples.filter((p) => {
        p.y += 0.05;
        if (p.y >= 1) {
          land(p.x);
          return false;
        }
        return true;
      });
      draw();
    });
    const playback = playBtn($('#sm-play', root), anim);
    bind(
      root,
      'sm-mu',
      (v) => {
        settings.mu = v;
        reset();
        draw();
      },
      (v) => (+v).toFixed(1),
    );
    bind(root, 'sm-sig', (v) => {
      settings.sig = v;
      reset();
      draw();
    });
    bind(
      root,
      'sm-bins',
      (v) => {
        settings.nb = v;
        reset();
        draw();
      },
      (v) => String(v),
    );
    $('#sm-step', root).onclick = () => {
      land(drawSample());
      draw();
    };
    $('#sm-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    return { draw, stop: playback.stop, auto: playback.start };
  },
};
