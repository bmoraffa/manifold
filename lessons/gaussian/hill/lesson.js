/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  G,
  Hill,
  T,
  bind,
  drawMap,
  f2,
  f3,
  grab,
  playBtn,
  setR,
  shapeBind,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c3 = $('#hill-c3', root),
      c2 = $('#hill-c2', root),
      S = { h: 0.9, trace: [] };
    let draw,
      wait = 0;
    const hill = Hill(c3, () => draw(), $('.viewbar', root));
    draw = () => {
      const t = T(),
        r = Math.sqrt(-2 * Math.log(S.h));
      hill.draw(t, G, { h: S.h });
      drawMap(c2, t, G, { rings: S.trace, ring: r });
      const pk = 1 / (2 * Math.PI * G.sx * G.sy * Math.sqrt(1 - G.rho * G.rho));
      setR(root, 'h', f3(S.h * pk));
      setR(root, 'r', f2(r));
      setR(root, 'in', (100 * (1 - S.h)).toFixed(0) + '%');
    };
    const hs = bind(root, 'hill-h', (v) => {
        S.h = v;
        draw();
      }),
      sh = shapeBind(root, 'hill', G, () => draw()),
      LV = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.05];
    const anim = Anim(() => {
      if (wait > 0) {
        wait--;
        if (!wait) {
          S.h = 0.97;
          S.trace = [];
          hs.set(S.h);
          draw();
        }
        return;
      }
      const nh = S.h - 0.0035;
      LV.forEach((l) => {
        if (S.h > l && nh <= l) S.trace.push(Math.sqrt(-2 * Math.log(l)));
      });
      if (nh < 0.035) {
        S.h = 0.035;
        wait = 110;
      } else S.h = nh;
      hs.set(S.h);
      draw();
    });
    const pb = playBtn($('#hill-play', root), anim);
    grab(hs.el, pb);
    $('#hill-clear', root).onclick = () => {
      S.trace = [];
      draw();
    };
    return {
      draw,
      stop() {
        pb.stop();
        hill.spin(false);
      },
      auto() {
        pb.start();
        hill.spin(true);
      },
      sync: sh.sync,
    };
  },
};
