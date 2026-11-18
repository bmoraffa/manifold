/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  sharedGaussianShape,
  Hill,
  readThemeColors,
  bind,
  drawMap,
  f2,
  f3,
  grab,
  playBtn,
  setReadout,
  shapeBind,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c3 = $('#hill-c3', root),
      c2 = $('#hill-c2', root),
      settings = { h: 0.9, trace: [] };
    let draw,
      wait = 0;
    const hill = Hill(c3, () => draw(), $('.viewbar', root));
    draw = () => {
      const colors = readThemeColors(),
        r = Math.sqrt(-2 * Math.log(settings.h));
      hill.draw(colors, sharedGaussianShape, { h: settings.h });
      drawMap(c2, colors, sharedGaussianShape, { rings: settings.trace, ring: r });
      const pk =
        1 /
        (2 *
          Math.PI *
          sharedGaussianShape.sx *
          sharedGaussianShape.sy *
          Math.sqrt(1 - sharedGaussianShape.rho * sharedGaussianShape.rho));
      setReadout(root, 'h', f3(settings.h * pk));
      setReadout(root, 'r', f2(r));
      setReadout(root, 'in', (100 * (1 - settings.h)).toFixed(0) + '%');
    };
    const sliceHeightControl = bind(root, 'hill-h', (v) => {
        settings.h = v;
        draw();
      }),
      shapeControls = shapeBind(root, 'hill', sharedGaussianShape, () => draw()),
      contourLevels = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.05];
    const anim = Anim(() => {
      if (wait > 0) {
        wait--;
        if (!wait) {
          settings.h = 0.97;
          settings.trace = [];
          sliceHeightControl.set(settings.h);
          draw();
        }
        return;
      }
      const nh = settings.h - 0.0035;
      contourLevels.forEach((l) => {
        if (settings.h > l && nh <= l) settings.trace.push(Math.sqrt(-2 * Math.log(l)));
      });
      if (nh < 0.035) {
        settings.h = 0.035;
        wait = 110;
      } else settings.h = nh;
      sliceHeightControl.set(settings.h);
      draw();
    });
    const playback = playBtn($('#hill-play', root), anim);
    grab(sliceHeightControl.el, playback);
    $('#hill-clear', root).onclick = () => {
      settings.trace = [];
      draw();
    };
    return {
      draw,
      stop() {
        playback.stop();
        hill.spin(false);
      },
      auto() {
        playback.start();
        hill.spin(true);
      },
      sync: shapeControls.sync,
    };
  },
};
