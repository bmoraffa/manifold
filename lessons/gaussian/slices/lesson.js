/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  sharedGaussianShape,
  Hill,
  MONO,
  readThemeColors,
  area,
  bind,
  clipPlot,
  curve,
  f2,
  f3,
  prepareCanvas,
  grab,
  mix,
  npdf,
  playBtn,
  plot,
  setReadout,
  shapeBind,
  sweeper,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c3 = $('#sl-c3', root),
      densityCanvas = $('#sl-c1', root),
      S = { x0: 1 };
    let draw;
    const hill = Hill(c3, () => draw(), $('.viewbar', root), { marg: true });
    draw = () => {
      const colors = readThemeColors();
      hill.draw(colors, sharedGaussianShape, { v: S.x0 });
      const m =
          ((sharedGaussianShape.rho * sharedGaussianShape.sy) / sharedGaussianShape.sx) * S.x0,
        sd =
          sharedGaussianShape.sy * Math.sqrt(1 - sharedGaussianShape.rho * sharedGaussianShape.rho);
      const { g: ctx, w, h } = prepareCanvas(densityCanvas),
        ym = Math.max(1, npdf(0, 0, sd) * 1.15),
        P = plot(ctx, w, h, [-4, 4], [0, ym], colors, { xlabel: 'y' });
      clipPlot(ctx, P);
      ctx.setLineDash([6, 4]);
      curve(ctx, P, (y) => npdf(y, 0, sharedGaussianShape.sy), -4, 4);
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 2.4;
      ctx.stroke();
      ctx.setLineDash([]);
      area(ctx, P, (y) => npdf(y, m, sd), -4, 4);
      ctx.fillStyle = mix(colors.ok, colors.ok, 0, 0.24);
      ctx.fill();
      curve(ctx, P, (y) => npdf(y, m, sd), -4, 4);
      ctx.strokeStyle = colors.ok;
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.restore();
      ctx.font = MONO;
      ctx.textAlign = 'left';
      ctx.fillStyle = colors.ok;
      ctx.fillText('p(y | x₀)', P.m.l + 8, P.m.t + 10);
      ctx.fillStyle = colors.accent;
      ctx.fillText('p(y), dashed', P.m.l + 8, P.m.t + 25);
      setReadout(root, 'm', f2(m));
      setReadout(root, 's', f3(sd));
      setReadout(root, 'ms', f3(sharedGaussianShape.sy));
      setReadout(
        root,
        'r2',
        (100 * sharedGaussianShape.rho * sharedGaussianShape.rho).toFixed(0) + '%',
      );
    };
    const slicePositionControl = bind(root, 'sl-x', (v) => {
        S.x0 = v;
        draw();
      }),
      shapeControls = shapeBind(root, 'sl', sharedGaussianShape, () => draw());
    const playback = playBtn($('#sl-play', root), sweeper(slicePositionControl.el, -2.6, 2.6, 7));
    grab(slicePositionControl.el, playback);
    return {
      draw,
      stop() {
        playback.stop();
        hill.spin(false);
      },
      auto: playback.start,
      sync: shapeControls.sync,
    };
  },
};
