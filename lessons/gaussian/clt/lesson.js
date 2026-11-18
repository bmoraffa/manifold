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
  npdf,
  playBtn,
  plot,
  ramp,
  randn,
  setReadout,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#clt-c', root),
      NS = [1, 2, 3, 5, 10, 30, 100];
    const sourceDistributions = {
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
    const settings = { src: 'die', n: 2 };
    let binCounts, N, s1, s2, lo, bw, nb;
    const reset = () => {
      const sr = sourceDistributions[settings.src];
      if (sr.disc && settings.n <= 30) {
        bw = 1 / settings.n;
        lo = 1 - bw / 2;
        nb = 5 * settings.n + 1;
      } else {
        nb = 80;
        lo = sr.xr[0];
        bw = (sr.xr[1] - sr.xr[0]) / nb;
      }
      binCounts = new Array(nb).fill(0);
      N = 0;
      s1 = 0;
      s2 = 0;
    };
    reset();
    const recordSampleMean = () => {
      const sr = sourceDistributions[settings.src];
      let a = 0;
      for (let i = 0; i < settings.n; i++) a += sr.draw();
      a /= settings.n;
      N++;
      s1 += a;
      s2 += a * a;
      const b = Math.floor((a - lo) / bw);
      if (b >= 0 && b < nb) binCounts[b]++;
    };
    const draw = () => {
      const colors = readThemeColors(),
        { g: ctx, w, h } = prepareCanvas(mainCanvas),
        sr = sourceDistributions[settings.src],
        sd = sr.sd / Math.sqrt(settings.n),
        gp = npdf(0, 0, sd);
      let ym = 1.35 * gp;
      if (N > 300) {
        let mx = 0;
        for (const k of binCounts) mx = Math.max(mx, k / (N * bw));
        ym = Math.max(ym, mx * 1.1);
      }
      const xr = [sr.xr[0] - 0.5, sr.xr[1] + 0.5],
        P = plot(ctx, w, h, xr, [0, ym], colors, { xlabel: 'average of n draws' });
      clipPlot(ctx, P);
      for (let b = 0; b < nb; b++) {
        if (!binCounts[b]) continue;
        const d = binCounts[b] / (N * bw),
          x0 = P.X(lo + b * bw),
          x1 = P.X(lo + (b + 1) * bw);
        ctx.fillStyle = ramp(colors, clamp(0.25 + (0.75 * d) / gp, 0, 1), 0.9);
        ctx.fillRect(x0 + 0.5, P.Y(d), Math.max(1, x1 - x0 - 1), P.Y(0) - P.Y(d));
      }
      ctx.setLineDash([7, 5]);
      curve(ctx, P, (x) => npdf(x, sr.mu, sd), xr[0], xr[1]);
      ctx.strokeStyle = colors.ok;
      ctx.lineWidth = 2.4;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
      const iw = 128,
        ih = 58,
        ix = P.m.l + 10,
        iy = P.m.t + 2;
      ctx.font = MONO;
      ctx.fillStyle = colors.muted;
      ctx.textAlign = 'left';
      ctx.fillText('one draw looks like', ix, iy + 9);
      ctx.fillStyle = mix(colors.ink, colors.ink, 0, 0.55);
      if (settings.src === 'die')
        for (let k = 0; k < 6; k++) ctx.fillRect(ix + k * 18, iy + 26, 13, 26);
      else {
        ctx.beginPath();
        ctx.moveTo(ix, iy + ih - 6);
        for (let k = 0; k <= 60; k++) {
          const u = k / 60,
            v =
              settings.src === 'exp'
                ? Math.exp(-3.2 * u)
                : settings.src === 'uni'
                  ? 0.7
                  : Math.exp(-((u - 0.25) ** 2) / 0.008) + Math.exp(-((u - 0.75) ** 2) / 0.008);
          ctx.lineTo(ix + u * (iw - 16), iy + ih - 6 - v * 34);
        }
        ctx.lineTo(ix + iw - 16, iy + ih - 6);
        ctx.closePath();
        ctx.fill();
      }
      const m = N ? s1 / N : 0;
      setReadout(root, 'N', String(N));
      setReadout(root, 'sd', N > 1 ? f3(Math.sqrt(Math.max(0, s2 / N - m * m))) : '–');
      setReadout(root, 'pred', f3(sd));
    };
    const sp = bind(
      root,
      'clt-speed',
      () => {},
      (v) => String(v),
    );
    const anim = Anim(() => {
      const r = +sp.el.value;
      for (let i = 0; i < r; i++) recordSampleMean();
      draw();
    });
    const playback = playBtn($('#clt-play', root), anim);
    bind(
      root,
      'clt-n',
      (v) => {
        settings.n = NS[v];
        reset();
        draw();
      },
      (v) => String(NS[v]),
    );
    $('#clt-src', root).onchange = (e) => {
      settings.src = e.target.value;
      reset();
      draw();
    };
    $('#clt-reset', root).onclick = () => {
      reset();
      draw();
    };
    return { draw, stop: playback.stop, auto: playback.start };
  },
};
