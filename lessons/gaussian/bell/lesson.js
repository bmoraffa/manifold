/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  SANS,
  readThemeColors,
  area,
  bind,
  clipPlot,
  curve,
  erf,
  f2,
  f3,
  prepareCanvas,
  grab,
  mix,
  npdf,
  playBtn,
  plot,
  setReadout,
  sweeper,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#bell-c', root),
      settings = { mu: 0, sig: 1, k: 1 };
    const draw = () => {
      const colors = readThemeColors(),
        { g: ctx, w, h } = prepareCanvas(mainCanvas),
        xr = [-8, 8],
        P = plot(ctx, w, h, xr, [0, 1.05], colors, {
          yticks: [0.2, 0.4, 0.6, 0.8, 1],
          xlabel: 'x',
        }),
        f = (x) => npdf(x, settings.mu, settings.sig);
      clipPlot(ctx, P);
      [3, 2, 1].forEach((j) => {
        area(
          ctx,
          P,
          f,
          Math.max(xr[0], settings.mu - j * settings.sig),
          Math.min(xr[1], settings.mu + j * settings.sig),
        );
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.1);
        ctx.fill();
      });
      const a = Math.max(xr[0], settings.mu - settings.k * settings.sig),
        b = Math.min(xr[1], settings.mu + settings.k * settings.sig);
      area(ctx, P, f, a, b);
      ctx.fillStyle = mix(colors.ok, colors.ok, 0, 0.28);
      ctx.fill();
      ctx.strokeStyle = colors.ok;
      ctx.lineWidth = 1.5;
      [a, b].forEach((x) => {
        ctx.beginPath();
        ctx.moveTo(P.X(x), P.Y(0));
        ctx.lineTo(P.X(x), P.Y(f(x)));
        ctx.stroke();
      });
      curve(ctx, P, f, xr[0], xr[1]);
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 3.2;
      ctx.stroke();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = colors.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(P.X(settings.mu), P.Y(0));
      ctx.lineTo(P.X(settings.mu), P.Y(f(settings.mu)));
      ctx.stroke();
      ctx.setLineDash([]);
      const sigmaMarkerHeight = f(settings.mu + settings.sig);
      ctx.strokeStyle = colors.ink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(P.X(settings.mu), P.Y(sigmaMarkerHeight));
      ctx.lineTo(P.X(settings.mu + settings.sig), P.Y(sigmaMarkerHeight));
      ctx.stroke();
      ctx.font = SANS;
      ctx.fillStyle = colors.ink;
      ctx.textAlign = 'center';
      ctx.fillText('σ', P.X(settings.mu + settings.sig / 2), P.Y(sigmaMarkerHeight) - 6);
      ctx.fillText('μ', P.X(settings.mu), P.Y(f(settings.mu)) - 8);
      ctx.restore();
      setReadout(root, 'p', (100 * erf(settings.k / Math.SQRT2)).toFixed(1) + '%');
      setReadout(
        root,
        'int',
        '[' +
          f2(settings.mu - settings.k * settings.sig) +
          ', ' +
          f2(settings.mu + settings.k * settings.sig) +
          ']',
      );
      setReadout(root, 'var', f2(settings.sig * settings.sig));
      setReadout(root, 'peak', f3(f(settings.mu)));
    };
    bind(
      root,
      'bell-mu',
      (v) => {
        settings.mu = v;
        draw();
      },
      (v) => (+v).toFixed(1),
    );
    const sigmaControl = bind(root, 'bell-sig', (v) => {
      settings.sig = v;
      draw();
    });
    bind(
      root,
      'bell-k',
      (v) => {
        settings.k = v;
        draw();
      },
      (v) => (+v).toFixed(1),
    );
    const playback = playBtn($('#bell-play', root), sweeper(sigmaControl.el, 0.4, 3));
    grab(sigmaControl.el, playback);
    return { draw, stop: playback.stop, auto: playback.start };
  },
};
