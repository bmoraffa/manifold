/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  SANS,
  T,
  area,
  bind,
  clipPlot,
  curve,
  erf,
  f2,
  f3,
  fit,
  grab,
  mix,
  npdf,
  playBtn,
  plot,
  setR,
  sweeper,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#bell-c', root),
      S = { mu: 0, sig: 1, k: 1 };
    const draw = () => {
      const t = T(),
        { g, w, h } = fit(c),
        xr = [-8, 8],
        P = plot(g, w, h, xr, [0, 1.05], t, { yticks: [0.2, 0.4, 0.6, 0.8, 1], xlabel: 'x' }),
        f = (x) => npdf(x, S.mu, S.sig);
      clipPlot(g, P);
      [3, 2, 1].forEach((j) => {
        area(g, P, f, Math.max(xr[0], S.mu - j * S.sig), Math.min(xr[1], S.mu + j * S.sig));
        g.fillStyle = mix(t.accent, t.accent, 0, 0.1);
        g.fill();
      });
      const a = Math.max(xr[0], S.mu - S.k * S.sig),
        b = Math.min(xr[1], S.mu + S.k * S.sig);
      area(g, P, f, a, b);
      g.fillStyle = mix(t.ok, t.ok, 0, 0.28);
      g.fill();
      g.strokeStyle = t.ok;
      g.lineWidth = 1.5;
      [a, b].forEach((x) => {
        g.beginPath();
        g.moveTo(P.X(x), P.Y(0));
        g.lineTo(P.X(x), P.Y(f(x)));
        g.stroke();
      });
      curve(g, P, f, xr[0], xr[1]);
      g.strokeStyle = t.accent;
      g.lineWidth = 3.2;
      g.stroke();
      g.setLineDash([4, 4]);
      g.strokeStyle = t.muted;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(P.X(S.mu), P.Y(0));
      g.lineTo(P.X(S.mu), P.Y(f(S.mu)));
      g.stroke();
      g.setLineDash([]);
      const yi = f(S.mu + S.sig);
      g.strokeStyle = t.ink;
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(P.X(S.mu), P.Y(yi));
      g.lineTo(P.X(S.mu + S.sig), P.Y(yi));
      g.stroke();
      g.font = SANS;
      g.fillStyle = t.ink;
      g.textAlign = 'center';
      g.fillText('σ', P.X(S.mu + S.sig / 2), P.Y(yi) - 6);
      g.fillText('μ', P.X(S.mu), P.Y(f(S.mu)) - 8);
      g.restore();
      setR(root, 'p', (100 * erf(S.k / Math.SQRT2)).toFixed(1) + '%');
      setR(root, 'int', '[' + f2(S.mu - S.k * S.sig) + ', ' + f2(S.mu + S.k * S.sig) + ']');
      setR(root, 'var', f2(S.sig * S.sig));
      setR(root, 'peak', f3(f(S.mu)));
    };
    bind(
      root,
      'bell-mu',
      (v) => {
        S.mu = v;
        draw();
      },
      (v) => (+v).toFixed(1),
    );
    const sg = bind(root, 'bell-sig', (v) => {
      S.sig = v;
      draw();
    });
    bind(
      root,
      'bell-k',
      (v) => {
        S.k = v;
        draw();
      },
      (v) => (+v).toFixed(1),
    );
    const pb = playBtn($('#bell-play', root), sweeper(sg.el, 0.4, 3));
    grab(sg.el, pb);
    return { draw, stop: pb.stop, auto: pb.start };
  },
};
