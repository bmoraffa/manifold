/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  f2,
  fit,
  npdf,
  playBtn,
  ramp,
  setR,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#gb-c', root),
      S = { rows: 12, p: 0.5, rate: 14 };
    let balls,
      cnt,
      n,
      acc = 0;
    const reset = () => {
      balls = [];
      cnt = new Array(S.rows + 1).fill(0);
      n = 0;
    };
    reset();
    const draw = () => {
      const t = T(),
        { g, w, h } = fit(c),
        top = 14,
        pegH = h * 0.5,
        dy = pegH / S.rows,
        dx = Math.min((w - 30) / (S.rows + 2), dy * 1.5),
        cx = w / 2,
        binTop = top + pegH + 8,
        binH = h - binTop - 24,
        br = clamp(dx * 0.2, 2.2, 5.5);
      g.fillStyle = t.muted;
      for (let r = 0; r < S.rows; r++)
        for (let i = 0; i <= r; i++) {
          g.beginPath();
          g.arc(cx + (i - r / 2) * dx, top + (r + 0.5) * dy, Math.min(2.6, dx * 0.09), 0, 7);
          g.fill();
        }
      let mx = 1;
      for (const k of cnt) mx = Math.max(mx, k);
      const unit = Math.min(br * 1.6, binH / mx),
        bw = dx * 0.78;
      for (let k = 0; k <= S.rows; k++) {
        const x = cx + (k - S.rows / 2) * dx,
          hh = cnt[k] * unit;
        g.fillStyle = ramp(t, 0.25 + (0.75 * cnt[k]) / mx);
        g.fillRect(x - bw / 2, binTop + binH - hh, bw, hh);
      }
      g.strokeStyle = t.muted;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(cx - (S.rows / 2 + 0.6) * dx, binTop + binH);
      g.lineTo(cx + (S.rows / 2 + 0.6) * dx, binTop + binH);
      g.stroke();
      if (n > 20) {
        const mu = S.rows * S.p,
          sd = Math.sqrt(S.rows * S.p * (1 - S.p));
        g.beginPath();
        for (let i = 0; i <= 120; i++) {
          const k = -0.6 + ((S.rows + 1.2) * i) / 120,
            y = binTop + binH - n * npdf(k, mu, sd) * unit,
            x = cx + (k - S.rows / 2) * dx;
          i ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        g.strokeStyle = t.ok;
        g.lineWidth = 2.6;
        g.stroke();
      }
      g.font = MONO;
      g.fillStyle = t.muted;
      g.textAlign = 'center';
      for (let k = 0; k <= S.rows; k += S.rows > 10 ? 2 : 1)
        g.fillText(String(k), cx + (k - S.rows / 2) * dx, binTop + binH + 15);
      g.fillStyle = t.gold;
      for (const b of balls) {
        let x, y;
        if (b.r < S.rows) {
          const r0 = Math.floor(b.r),
            f = b.r - r0,
            k0 = b.pre[r0],
            x0 = cx + (k0 - r0 / 2) * dx,
            y0 = top + (r0 + 0.5) * dy - br - 3,
            e = f * f * (3 - 2 * f);
          x = x0 + (b.dirs[r0] - 0.5) * dx * e;
          y = y0 + dy * f - Math.sin(Math.PI * f) * dy * 0.28;
        } else {
          const f = Math.min(1, (b.r - S.rows) / 3);
          x = cx + (b.k - S.rows / 2) * dx;
          const ya = top + (S.rows + 0.5) * dy - br - 3;
          y = ya + (binTop + binH - cnt[b.k] * unit - br - ya) * f * f;
        }
        g.beginPath();
        g.arc(x, y, br, 0, 7);
        g.fill();
      }
      let s1 = 0,
        s2 = 0;
      for (let k = 0; k <= S.rows; k++) {
        s1 += k * cnt[k];
        s2 += k * k * cnt[k];
      }
      const m = n ? s1 / n : 0;
      setR(root, 'n', String(n));
      setR(root, 'm', n ? f2(m) : '–');
      setR(root, 's', n > 1 ? f2(Math.sqrt(Math.max(0, s2 / n - m * m))) : '–');
      setR(root, 'pm', f2(S.rows * S.p));
      setR(root, 'ps', f2(Math.sqrt(S.rows * S.p * (1 - S.p))));
    };
    const spawn = () => {
      const dirs = [],
        pre = [0];
      let k = 0;
      for (let r = 0; r < S.rows; r++) {
        const d = Math.random() < S.p ? 1 : 0;
        dirs.push(d);
        k += d;
        pre.push(k);
      }
      balls.push({ r: 0, k, dirs, pre });
    };
    const anim = Anim(() => {
      acc += S.rate / 60;
      while (acc >= 1) {
        acc--;
        if (balls.length < 500) spawn();
      }
      balls = balls.filter((b) => {
        b.r += 0.085;
        if (b.r >= S.rows + 3) {
          cnt[b.k]++;
          n++;
          return false;
        }
        return true;
      });
      draw();
    });
    const pb = playBtn($('#gb-play', root), anim);
    bind(
      root,
      'gb-rate',
      (v) => {
        S.rate = v;
      },
      (v) => String(v),
    );
    bind(
      root,
      'gb-rows',
      (v) => {
        S.rows = v;
        reset();
        draw();
      },
      (v) => String(v),
    );
    bind(root, 'gb-p', (v) => {
      S.p = v;
      reset();
      draw();
    });
    $('#gb-reset', root).onclick = () => {
      reset();
      draw();
    };
    return { draw, stop: pb.stop, auto: pb.start };
  },
};
