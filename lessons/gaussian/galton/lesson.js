/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  bind,
  clamp,
  f2,
  prepareCanvas,
  npdf,
  playBtn,
  ramp,
  setReadout,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#gb-c', root),
      settings = { rows: 12, p: 0.5, rate: 14 };
    let balls,
      binCounts,
      n,
      spawnAccumulator = 0;
    const reset = () => {
      balls = [];
      binCounts = new Array(settings.rows + 1).fill(0);
      n = 0;
    };
    reset();
    const draw = () => {
      const colors = readThemeColors(),
        { g: ctx, w, h } = prepareCanvas(mainCanvas),
        top = 14,
        pegH = h * 0.5,
        dy = pegH / settings.rows,
        dx = Math.min((w - 30) / (settings.rows + 2), dy * 1.5),
        cx = w / 2,
        binTop = top + pegH + 8,
        binH = h - binTop - 24,
        br = clamp(dx * 0.2, 2.2, 5.5);
      ctx.fillStyle = colors.muted;
      for (let r = 0; r < settings.rows; r++)
        for (let i = 0; i <= r; i++) {
          ctx.beginPath();
          ctx.arc(cx + (i - r / 2) * dx, top + (r + 0.5) * dy, Math.min(2.6, dx * 0.09), 0, 7);
          ctx.fill();
        }
      let mx = 1;
      for (const k of binCounts) mx = Math.max(mx, k);
      const unit = Math.min(br * 1.6, binH / mx),
        bw = dx * 0.78;
      for (let k = 0; k <= settings.rows; k++) {
        const x = cx + (k - settings.rows / 2) * dx,
          hh = binCounts[k] * unit;
        ctx.fillStyle = ramp(colors, 0.25 + (0.75 * binCounts[k]) / mx);
        ctx.fillRect(x - bw / 2, binTop + binH - hh, bw, hh);
      }
      ctx.strokeStyle = colors.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - (settings.rows / 2 + 0.6) * dx, binTop + binH);
      ctx.lineTo(cx + (settings.rows / 2 + 0.6) * dx, binTop + binH);
      ctx.stroke();
      if (n > 20) {
        const mu = settings.rows * settings.p,
          sd = Math.sqrt(settings.rows * settings.p * (1 - settings.p));
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const k = -0.6 + ((settings.rows + 1.2) * i) / 120,
            y = binTop + binH - n * npdf(k, mu, sd) * unit,
            x = cx + (k - settings.rows / 2) * dx;
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 2.6;
        ctx.stroke();
      }
      ctx.font = MONO;
      ctx.fillStyle = colors.muted;
      ctx.textAlign = 'center';
      for (let k = 0; k <= settings.rows; k += settings.rows > 10 ? 2 : 1)
        ctx.fillText(String(k), cx + (k - settings.rows / 2) * dx, binTop + binH + 15);
      ctx.fillStyle = colors.gold;
      for (const b of balls) {
        let x, y;
        if (b.r < settings.rows) {
          const r0 = Math.floor(b.r),
            f = b.r - r0,
            k0 = b.pre[r0],
            x0 = cx + (k0 - r0 / 2) * dx,
            y0 = top + (r0 + 0.5) * dy - br - 3,
            e = f * f * (3 - 2 * f);
          x = x0 + (b.dirs[r0] - 0.5) * dx * e;
          y = y0 + dy * f - Math.sin(Math.PI * f) * dy * 0.28;
        } else {
          const f = Math.min(1, (b.r - settings.rows) / 3);
          x = cx + (b.k - settings.rows / 2) * dx;
          const ya = top + (settings.rows + 0.5) * dy - br - 3;
          y = ya + (binTop + binH - binCounts[b.k] * unit - br - ya) * f * f;
        }
        ctx.beginPath();
        ctx.arc(x, y, br, 0, 7);
        ctx.fill();
      }
      let s1 = 0,
        s2 = 0;
      for (let k = 0; k <= settings.rows; k++) {
        s1 += k * binCounts[k];
        s2 += k * k * binCounts[k];
      }
      const m = n ? s1 / n : 0;
      setReadout(root, 'n', String(n));
      setReadout(root, 'm', n ? f2(m) : '–');
      setReadout(root, 's', n > 1 ? f2(Math.sqrt(Math.max(0, s2 / n - m * m))) : '–');
      setReadout(root, 'pm', f2(settings.rows * settings.p));
      setReadout(root, 'ps', f2(Math.sqrt(settings.rows * settings.p * (1 - settings.p))));
    };
    const spawn = () => {
      const dirs = [],
        rightMovesByRow = [0];
      let k = 0;
      for (let r = 0; r < settings.rows; r++) {
        const d = Math.random() < settings.p ? 1 : 0;
        dirs.push(d);
        k += d;
        rightMovesByRow.push(k);
      }
      balls.push({ r: 0, k, dirs, pre: rightMovesByRow });
    };
    const anim = Anim(() => {
      spawnAccumulator += settings.rate / 60;
      while (spawnAccumulator >= 1) {
        spawnAccumulator--;
        if (balls.length < 500) spawn();
      }
      balls = balls.filter((b) => {
        b.r += 0.085;
        if (b.r >= settings.rows + 3) {
          binCounts[b.k]++;
          n++;
          return false;
        }
        return true;
      });
      draw();
    });
    const playback = playBtn($('#gb-play', root), anim);
    bind(
      root,
      'gb-rate',
      (v) => {
        settings.rate = v;
      },
      (v) => String(v),
    );
    bind(
      root,
      'gb-rows',
      (v) => {
        settings.rows = v;
        reset();
        draw();
      },
      (v) => String(v),
    );
    bind(root, 'gb-p', (v) => {
      settings.p = v;
      reset();
      draw();
    });
    $('#gb-reset', root).onclick = () => {
      reset();
      draw();
    };
    return { draw, stop: playback.stop, auto: playback.start };
  },
};
