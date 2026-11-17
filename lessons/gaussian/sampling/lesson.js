/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  T,
  bind,
  clamp,
  clipPlot,
  curve,
  f3,
  fit,
  npdf,
  playBtn,
  plot,
  ramp,
  randn,
  setR,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#sm-c', root),
      xr = [-6, 6],
      S = { mu: 0, sig: 1, nb: 48 };
    let cnt, n, s1, s2, parts, bw;
    const reset = () => {
      bw = 12 / S.nb;
      cnt = new Array(S.nb).fill(0);
      n = 0;
      s1 = 0;
      s2 = 0;
      parts = [];
    };
    reset();
    const land = (x) => {
      n++;
      s1 += x;
      s2 += x * x;
      const b = Math.floor((x - xr[0]) / bw);
      if (b >= 0 && b < S.nb) cnt[b]++;
    };
    const smp = () => S.mu + S.sig * randn(),
      sp = bind(
        root,
        'sm-speed',
        () => {},
        (v) => String(v),
      );
    const draw = () => {
      const t = T(),
        { g, w, h } = fit(c),
        pk = npdf(0, 0, S.sig),
        P = plot(g, w, h, xr, [0, Math.max(0.5, pk * 1.5)], t, {
          yticks: [0.2, 0.4, 0.6, 0.8, 1],
          xlabel: 'x',
        });
      clipPlot(g, P);
      for (let b = 0; b < S.nb; b++) {
        if (!cnt[b]) continue;
        const d = cnt[b] / (n * bw),
          x0 = P.X(xr[0] + b * bw),
          x1 = P.X(xr[0] + (b + 1) * bw);
        g.fillStyle = ramp(t, clamp(0.25 + (0.75 * d) / pk, 0, 1), 0.9);
        g.fillRect(x0 + 0.5, P.Y(d), x1 - x0 - 1, P.Y(0) - P.Y(d));
      }
      curve(g, P, (x) => npdf(x, S.mu, S.sig), xr[0], xr[1]);
      g.strokeStyle = t.ink;
      g.lineWidth = 2.2;
      g.stroke();
      g.fillStyle = t.gold;
      for (const p of parts) {
        g.beginPath();
        g.arc(P.X(p.x), P.m.t + (P.Y(0) - P.m.t) * p.y * p.y, 4, 0, 7);
        g.fill();
      }
      g.restore();
      const m = n ? s1 / n : 0;
      setR(root, 'n', String(n));
      setR(root, 'm', n ? f3(m) : '–');
      setR(root, 's', n > 1 ? f3(Math.sqrt(Math.max(0, (s2 - n * m * m) / (n - 1)))) : '–');
      setR(root, 'se', n ? f3(S.sig / Math.sqrt(n)) : '–');
    };
    const anim = Anim(() => {
      const r = +sp.el.value;
      if (r <= 8) for (let i = 0; i < r; i++) parts.push({ x: smp(), y: 0 });
      else for (let i = 0; i < r; i++) land(smp());
      parts = parts.filter((p) => {
        p.y += 0.05;
        if (p.y >= 1) {
          land(p.x);
          return false;
        }
        return true;
      });
      draw();
    });
    const pb = playBtn($('#sm-play', root), anim);
    bind(
      root,
      'sm-mu',
      (v) => {
        S.mu = v;
        reset();
        draw();
      },
      (v) => (+v).toFixed(1),
    );
    bind(root, 'sm-sig', (v) => {
      S.sig = v;
      reset();
      draw();
    });
    bind(
      root,
      'sm-bins',
      (v) => {
        S.nb = v;
        reset();
        draw();
      },
      (v) => String(v),
    );
    $('#sm-step', root).onclick = () => {
      land(smp());
      draw();
    };
    $('#sm-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    return { draw, stop: pb.stop, auto: pb.start };
  },
};
