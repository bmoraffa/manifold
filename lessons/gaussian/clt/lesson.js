/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  clipPlot,
  curve,
  f3,
  fit,
  mix,
  npdf,
  playBtn,
  plot,
  ramp,
  randn,
  setR,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#clt-c', root),
      NS = [1, 2, 3, 5, 10, 30, 100];
    const SRC = {
      die: {
        mu: 3.5,
        sd: Math.sqrt(35 / 12),
        xr: [1, 6],
        disc: true,
        draw: () => 1 + Math.floor(Math.random() * 6),
      },
      exp: { mu: 1, sd: 1, xr: [0, 4], draw: () => -Math.log(1 - Math.random()) },
      two: {
        mu: 0,
        sd: Math.sqrt(4.25),
        xr: [-4, 4],
        draw: () => (Math.random() < 0.5 ? -2 : 2) + 0.5 * randn(),
      },
      uni: { mu: 0, sd: Math.sqrt(3), xr: [-3, 3], draw: () => Math.random() * 6 - 3 },
    };
    const S = { src: 'die', n: 2 };
    let cnt, N, s1, s2, lo, bw, nb;
    const reset = () => {
      const sr = SRC[S.src];
      if (sr.disc && S.n <= 30) {
        bw = 1 / S.n;
        lo = 1 - bw / 2;
        nb = 5 * S.n + 1;
      } else {
        nb = 80;
        lo = sr.xr[0];
        bw = (sr.xr[1] - sr.xr[0]) / nb;
      }
      cnt = new Array(nb).fill(0);
      N = 0;
      s1 = 0;
      s2 = 0;
    };
    reset();
    const add = () => {
      const sr = SRC[S.src];
      let a = 0;
      for (let i = 0; i < S.n; i++) a += sr.draw();
      a /= S.n;
      N++;
      s1 += a;
      s2 += a * a;
      const b = Math.floor((a - lo) / bw);
      if (b >= 0 && b < nb) cnt[b]++;
    };
    const draw = () => {
      const t = T(),
        { g, w, h } = fit(c),
        sr = SRC[S.src],
        sd = sr.sd / Math.sqrt(S.n),
        gp = npdf(0, 0, sd);
      let ym = 1.35 * gp;
      if (N > 300) {
        let mx = 0;
        for (const k of cnt) mx = Math.max(mx, k / (N * bw));
        ym = Math.max(ym, mx * 1.1);
      }
      const xr = [sr.xr[0] - 0.5, sr.xr[1] + 0.5],
        P = plot(g, w, h, xr, [0, ym], t, { xlabel: 'average of n draws' });
      clipPlot(g, P);
      for (let b = 0; b < nb; b++) {
        if (!cnt[b]) continue;
        const d = cnt[b] / (N * bw),
          x0 = P.X(lo + b * bw),
          x1 = P.X(lo + (b + 1) * bw);
        g.fillStyle = ramp(t, clamp(0.25 + (0.75 * d) / gp, 0, 1), 0.9);
        g.fillRect(x0 + 0.5, P.Y(d), Math.max(1, x1 - x0 - 1), P.Y(0) - P.Y(d));
      }
      g.setLineDash([7, 5]);
      curve(g, P, (x) => npdf(x, sr.mu, sd), xr[0], xr[1]);
      g.strokeStyle = t.ok;
      g.lineWidth = 2.4;
      g.stroke();
      g.setLineDash([]);
      g.restore();
      const iw = 128,
        ih = 58,
        ix = P.m.l + 10,
        iy = P.m.t + 2;
      g.font = MONO;
      g.fillStyle = t.muted;
      g.textAlign = 'left';
      g.fillText('one draw looks like', ix, iy + 9);
      g.fillStyle = mix(t.ink, t.ink, 0, 0.55);
      if (S.src === 'die') for (let k = 0; k < 6; k++) g.fillRect(ix + k * 18, iy + 26, 13, 26);
      else {
        g.beginPath();
        g.moveTo(ix, iy + ih - 6);
        for (let k = 0; k <= 60; k++) {
          const u = k / 60,
            v =
              S.src === 'exp'
                ? Math.exp(-3.2 * u)
                : S.src === 'uni'
                  ? 0.7
                  : Math.exp(-((u - 0.25) ** 2) / 0.008) + Math.exp(-((u - 0.75) ** 2) / 0.008);
          g.lineTo(ix + u * (iw - 16), iy + ih - 6 - v * 34);
        }
        g.lineTo(ix + iw - 16, iy + ih - 6);
        g.closePath();
        g.fill();
      }
      const m = N ? s1 / N : 0;
      setR(root, 'N', String(N));
      setR(root, 'sd', N > 1 ? f3(Math.sqrt(Math.max(0, s2 / N - m * m))) : '–');
      setR(root, 'pred', f3(sd));
    };
    const sp = bind(
      root,
      'clt-speed',
      () => {},
      (v) => String(v),
    );
    const anim = Anim(() => {
      const r = +sp.el.value;
      for (let i = 0; i < r; i++) add();
      draw();
    });
    const pb = playBtn($('#clt-play', root), anim);
    bind(
      root,
      'clt-n',
      (v) => {
        S.n = NS[v];
        reset();
        draw();
      },
      (v) => String(NS[v]),
    );
    $('#clt-src', root).onchange = (e) => {
      S.src = e.target.value;
      reset();
      draw();
    };
    $('#clt-reset', root).onclick = () => {
      reset();
      draw();
    };
    return { draw, stop: pb.stop, auto: pb.start };
  },
};
