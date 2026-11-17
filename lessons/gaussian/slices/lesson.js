/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  G,
  Hill,
  MONO,
  T,
  area,
  bind,
  clipPlot,
  curve,
  f2,
  f3,
  fit,
  grab,
  mix,
  npdf,
  playBtn,
  plot,
  setR,
  shapeBind,
  sweeper,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c3 = $('#sl-c3', root),
      c1 = $('#sl-c1', root),
      S = { x0: 1 };
    let draw;
    const hill = Hill(c3, () => draw(), $('.viewbar', root), { marg: true });
    draw = () => {
      const t = T();
      hill.draw(t, G, { v: S.x0 });
      const m = ((G.rho * G.sy) / G.sx) * S.x0,
        sd = G.sy * Math.sqrt(1 - G.rho * G.rho);
      const { g, w, h } = fit(c1),
        ym = Math.max(1, npdf(0, 0, sd) * 1.15),
        P = plot(g, w, h, [-4, 4], [0, ym], t, { xlabel: 'y' });
      clipPlot(g, P);
      g.setLineDash([6, 4]);
      curve(g, P, (y) => npdf(y, 0, G.sy), -4, 4);
      g.strokeStyle = t.accent;
      g.lineWidth = 2.4;
      g.stroke();
      g.setLineDash([]);
      area(g, P, (y) => npdf(y, m, sd), -4, 4);
      g.fillStyle = mix(t.ok, t.ok, 0, 0.24);
      g.fill();
      curve(g, P, (y) => npdf(y, m, sd), -4, 4);
      g.strokeStyle = t.ok;
      g.lineWidth = 3;
      g.stroke();
      g.restore();
      g.font = MONO;
      g.textAlign = 'left';
      g.fillStyle = t.ok;
      g.fillText('p(y | x₀)', P.m.l + 8, P.m.t + 10);
      g.fillStyle = t.accent;
      g.fillText('p(y), dashed', P.m.l + 8, P.m.t + 25);
      setR(root, 'm', f2(m));
      setR(root, 's', f3(sd));
      setR(root, 'ms', f3(G.sy));
      setR(root, 'r2', (100 * G.rho * G.rho).toFixed(0) + '%');
    };
    const xs = bind(root, 'sl-x', (v) => {
        S.x0 = v;
        draw();
      }),
      sh = shapeBind(root, 'sl', G, () => draw());
    const pb = playBtn($('#sl-play', root), sweeper(xs.el, -2.6, 2.6, 7));
    grab(xs.el, pb);
    return {
      draw,
      stop() {
        pb.stop();
        hill.spin(false);
      },
      auto: pb.start,
      sync: sh.sync,
    };
  },
};
