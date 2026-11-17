/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  clamp,
  f2,
  fit,
  mix,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { segBind, uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#cv-c', root),
      live = $('#cv-live', root),
      eq = $('#cv-eq', root),
      N = 18,
      M = 16,
      S = { k: 'vert', img: 'seven', relu: true },
      KS = {
        vert: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
        horiz: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
        outline: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        blur: [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9),
        sharp: [0, -1, 0, -1, 5, -1, 0, -1, 0],
      };
    let img = new Float32Array(N * N),
      out = new Float32Array(M * M),
      pos = 0,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      GEO = null,
      paint = false;
    const load = () => {
      const o = document.createElement('canvas');
      o.width = o.height = 72;
      const g = o.getContext('2d');
      g.fillStyle = '#000';
      g.fillRect(0, 0, 72, 72);
      g.strokeStyle = g.fillStyle = '#fff';
      g.lineWidth = 7;
      g.lineCap = 'round';
      g.lineJoin = 'round';
      if (S.img === 'seven') {
        g.beginPath();
        g.moveTo(18, 16);
        g.lineTo(54, 16);
        g.lineTo(32, 58);
        g.stroke();
      } else if (S.img === 'box') {
        g.strokeRect(18, 18, 36, 36);
      } else if (S.img === 'cross') {
        g.beginPath();
        g.moveTo(36, 12);
        g.lineTo(36, 60);
        g.moveTo(12, 36);
        g.lineTo(60, 36);
        g.stroke();
      } else if (S.img === 'face') {
        g.lineWidth = 5;
        g.beginPath();
        g.arc(36, 36, 24, 0, 7);
        g.stroke();
        g.beginPath();
        g.arc(27, 29, 3, 0, 7);
        g.arc(45, 29, 3, 0, 7);
        g.fill();
        g.beginPath();
        g.arc(36, 36, 13, 0.5, 2.64);
        g.stroke();
      } else {
        g.lineWidth = 11;
        g.beginPath();
        g.moveTo(8, 62);
        g.lineTo(62, 8);
        g.stroke();
      }
      const d = g.getImageData(0, 0, 72, 72).data;
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
          let s = 0;
          for (let a = 0; a < 4; a++)
            for (let b = 0; b < 4; b++) s += d[((y * 4 + a) * 72 + x * 4 + b) * 4];
          img[y * N + x] = Math.round((s / 16 / 255) * 10) / 10;
        }
    };
    const K = () => KS[S.k],
      at = (p) => {
        const i = Math.floor(p / M),
          j = p % M,
          k = K();
        let s = 0;
        const pr = [];
        for (let a = 0; a < 3; a++)
          for (let b = 0; b < 3; b++) {
            const x = img[(i + a) * N + j + b];
            pr.push([x, k[a * 3 + b]]);
            s += x * k[a * 3 + b];
          }
        return { i, j, s, pr };
      };
    const comp = () => {
      for (let p = 0; p < M * M; p++) {
        const s = at(p).s;
        out[p] = S.relu ? Math.max(0, s) : s;
      }
    };
    const END = M * M + 64,
      reset = () => {
        pos = 0;
        cu = 0;
        msg =
          'The window is about to start at the top-left corner. Press Slide the window or One position.';
        eq.innerHTML = '';
      };
    const nf = (v) =>
      (v < 0 ? '(−' + Math.abs(v).toFixed(2) + ')' : v.toFixed(2)).replace('.00', '');
    const say = () => {
      if (pos <= M * M && pos > 0) {
        const q = at(pos - 1);
        cu = pos === 1 ? 1 : pos < M * M ? 3 : 2;
        msg =
          'Position ' +
          pos +
          ' of ' +
          M * M +
          ' (row ' +
          (q.i + 1) +
          ', column ' +
          (q.j + 1) +
          '): the nine products add up to ' +
          f2(q.s) +
          '.' +
          (S.relu && q.s < 0
            ? ' Negative, so ReLU writes 0.'
            : q.s > 1
              ? ' A strong match with the pattern.'
              : Math.abs(q.s) < 0.05
                ? ' No match: this patch is flat.'
                : '');
        eq.innerHTML =
          [0, 1, 2]
            .map(
              (a) =>
                '<div class="on">' +
                [0, 1, 2]
                  .map((b) => nf(q.pr[a * 3 + b][0]) + '·' + nf(q.pr[a * 3 + b][1]))
                  .join('  +  ') +
                (a < 2 ? '  +' : '') +
                '</div>',
            )
            .join('') +
          '<div class="on">= <b>' +
          f2(q.s) +
          '</b>' +
          (S.relu ? '  →  after ReLU: <b>' + f2(Math.max(0, q.s)) + '</b>' : '') +
          '</div>';
      } else if (pos > M * M) {
        const b = pos - M * M - 1,
          bi = Math.floor(b / 8),
          bj = b % 8,
          vals = [0, 1, 2, 3].map((q) => out[(2 * bi + (q >> 1)) * M + 2 * bj + (q & 1)]);
        cu = 4;
        msg =
          'Pooling block ' +
          (b + 1) +
          ' of 64: the four answers are ' +
          vals.map(f2).join(', ') +
          '. Keep the largest, ' +
          f2(Math.max(...vals)) +
          '.' +
          (pos >= END
            ? ' Finished: an 18 × 18 picture became a 16 × 16 map of where the pattern is, and then an 8 × 8 summary.'
            : '');
        eq.innerHTML =
          '<div class="on">max( ' +
          vals.map(f2).join(' ,  ') +
          ' ) = <b>' +
          f2(Math.max(...vals)) +
          '</b></div>';
      }
    };
    const step = () => {
      if (pos >= END) {
        reset();
        return;
      }
      pos++;
      say();
    };
    const draw = () => {
      const t = T(),
        { g, w, h } = fit(c),
        A = Math.min(h - 34, w * 0.36),
        ca = A / N,
        x0 = 8,
        y0 = 22,
        B = Math.min(h - 34, w * 0.3),
        cb = B / M,
        xo = w - B - w * 0.13 - 12,
        xp = w - w * 0.12 - 4,
        Pp = w * 0.12,
        cp = Pp / 8,
        xm = x0 + A + 18,
        wm = xo - xm - 18;
      GEO = { x0, y0, ca };
      g.font = MONO;
      const gray = (v) => mix(t.bg, t.ink, 0.07 + 0.93 * clamp(v, 0, 1)),
        div = (v, m) =>
          v >= 0 ? mix(t.bg, t.gold, clamp(v / m, 0, 1)) : mix(t.bg, t.ok, clamp(-v / m, 0, 1));
      let om = 1e-6;
      for (const v of out) om = Math.max(om, Math.abs(v));
      g.fillStyle = t.muted;
      g.textAlign = 'left';
      g.fillText('picture, 18 × 18', x0, 14);
      g.fillText('output, 16 × 16', xo, 14);
      g.fillText('pooled, 8 × 8', xp, 14);
      for (let p = 0; p < N * N; p++) {
        g.fillStyle = gray(img[p]);
        g.fillRect(x0 + (p % N) * ca, y0 + Math.floor(p / N) * ca, ca - 0.6, ca - 0.6);
      }
      const shown = Math.min(pos, M * M);
      for (let p = 0; p < M * M; p++) {
        g.fillStyle = p < shown ? div(out[p], om) : mix(t.ink, t.ink, 0, 0.05);
        g.fillRect(xo + (p % M) * cb, y0 + Math.floor(p / M) * cb, cb - 0.6, cb - 0.6);
      }
      const np = Math.max(0, pos - M * M);
      for (let b = 0; b < 64; b++) {
        const bi = Math.floor(b / 8),
          bj = b % 8;
        let v = -1e9;
        for (let q = 0; q < 4; q++)
          v = Math.max(v, out[(2 * bi + (q >> 1)) * M + 2 * bj + (q & 1)]);
        g.fillStyle = b < np ? div(v, om) : mix(t.ink, t.ink, 0, 0.05);
        g.fillRect(xp + bj * cp, y0 + bi * cp, cp - 0.6, cp - 0.6);
      }
      if (pos > 0 && pos <= M * M) {
        const q = at(pos - 1);
        g.strokeStyle = t.gold;
        g.lineWidth = 3;
        g.strokeRect(x0 + q.j * ca - 1, y0 + q.i * ca - 1, 3 * ca + 1.4, 3 * ca + 1.4);
        g.strokeRect(xo + q.j * cb - 1, y0 + q.i * cb - 1, cb + 1.4, cb + 1.4);
        const cz = Math.min(wm / 3.3, (h - 70) / 10.2),
          mx = xm + (wm - 3 * cz) / 2,
          k = K(),
          km = Math.max(...k.map(Math.abs));
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('patch under the window', xm + wm / 2, y0 + 6);
        for (let a = 0; a < 3; a++)
          for (let b = 0; b < 3; b++) {
            const X = mx + b * cz,
              Y1 = y0 + 12 + a * cz,
              Y2 = y0 + 30 + 3 * cz + a * cz,
              Y3 = y0 + 48 + 6 * cz + a * cz,
              pv = q.pr[a * 3 + b];
            g.fillStyle = gray(pv[0]);
            g.fillRect(X, Y1, cz - 2, cz - 2);
            g.fillStyle = pv[0] > 0.5 ? t.bg : t.ink;
            g.fillText(nf(pv[0]), X + cz / 2 - 1, Y1 + cz / 2 + 3);
            g.fillStyle = div(pv[1], km);
            g.fillRect(X, Y2, cz - 2, cz - 2);
            g.strokeStyle = t.line;
            g.lineWidth = 1;
            g.strokeRect(X, Y2, cz - 2, cz - 2);
            g.fillStyle = t.ink;
            g.fillText(nf(pv[1]), X + cz / 2 - 1, Y2 + cz / 2 + 3);
            const pd = pv[0] * pv[1];
            g.fillStyle = div(pd, Math.max(1, km));
            g.fillRect(X, Y3, cz - 2, cz - 2);
            g.strokeStyle = t.line;
            g.strokeRect(X, Y3, cz - 2, cz - 2);
            g.fillStyle = t.ink;
            g.fillText(nf(pd), X + cz / 2 - 1, Y3 + cz / 2 + 3);
          }
        g.fillStyle = t.muted;
        g.fillText('× the kernel', xm + wm / 2, y0 + 24 + 3 * cz);
        g.fillText('= products, which add up to', xm + wm / 2, y0 + 42 + 6 * cz);
        g.font = '600 20px "Hanken Grotesk",sans-serif';
        g.fillStyle = t.cut;
        g.fillText(f2(q.s), xm + wm / 2, Math.min(h - 6, y0 + 72 + 9 * cz));
        g.font = MONO;
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.6);
        g.lineWidth = 1.2;
        g.beginPath();
        g.moveTo(x0 + (q.j + 3) * ca, y0 + (q.i + 1.5) * ca);
        g.lineTo(mx, y0 + 12 + 1.5 * cz);
        g.moveTo(mx + 3 * cz, y0 + 48 + 7.5 * cz);
        g.lineTo(xo + q.j * cb, y0 + (q.i + 0.5) * cb);
        g.stroke();
      } else if (pos > M * M) {
        const b = pos - M * M - 1,
          bi = Math.floor(b / 8),
          bj = b % 8;
        g.strokeStyle = t.gold;
        g.lineWidth = 3;
        g.strokeRect(xo + 2 * bj * cb - 1, y0 + 2 * bi * cb - 1, 2 * cb + 1.4, 2 * cb + 1.4);
        g.strokeRect(xp + bj * cp - 1, y0 + bi * cp - 1, cp + 1.4, cp + 1.4);
        g.lineWidth = 1.2;
        g.beginPath();
        g.moveTo(xo + (2 * bj + 2) * cb, y0 + (2 * bi + 1) * cb);
        g.lineTo(xp + bj * cp, y0 + (bi + 0.5) * cp);
        g.stroke();
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('keep the largest of each 2 × 2 block →', xm + wm / 2, h / 2);
      } else {
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('the 3 × 3 window will stop', xm + wm / 2, h / 2 - 8);
        g.fillText('at 256 positions', xm + wm / 2, h / 2 + 8);
      }
      g.fillStyle = t.muted;
      g.textAlign = 'left';
      g.fillText(
        'gold: pattern found.  red: its opposite found.',
        xo,
        Math.min(h - 4, y0 + B + 14),
      );
      live.textContent = msg;
      uCur(root, 'cv', cu);
      const q = pos > 0 && pos <= M * M ? at(pos - 1) : null;
      setR(root, 'pos', pos > M * M ? 'pooling ' + (pos - M * M) + ' of 64' : pos + ' of ' + M * M);
      setR(root, 'sum', q ? f2(q.s) : '–');
      setR(root, 'sz', '18 × 18 → 16 × 16 → 8 × 8');
      setR(root, 'np', '9 weights + 1 bias');
      setR(root, 'st', pos >= END ? 'Finished' : pos ? 'Sliding' : 'Not started');
    };
    load();
    comp();
    pos = END;
    say();
    const anim = Anim(() => {
      fr++;
      const slow = pos < 8 || (pos > M * M && pos < M * M + 5);
      if (fr % (slow ? 32 : pos > M * M ? 5 : 2)) return;
      step();
      draw();
      if (pos >= END) pb.stop();
    });
    pb = playBtn($('#cv-play', root), anim);
    runCtl(pb, root, 'cv', { done: () => pos >= END, restart: () => reset() });
    $('#cv-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#cv-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    const redo = () => {
      pb.stop();
      comp();
      reset();
      draw();
      pb.start();
    };
    segBind(root, 'cv-k', S.k, (v) => {
      S.k = v;
      redo();
    });
    $('#cv-img', root).onchange = (e) => {
      S.img = e.target.value;
      load();
      redo();
    };
    $('#cv-relu', root).onchange = (e) => {
      S.relu = e.target.checked;
      redo();
    };
    $('#cv-clear', root).onclick = () => {
      img.fill(0);
      pb.stop();
      comp();
      pos = END;
      say();
      draw();
    };
    const pt = (e) => {
      if (!GEO) return;
      const r = c.getBoundingClientRect(),
        x = Math.floor((e.clientX - r.left - GEO.x0) / GEO.ca),
        y = Math.floor((e.clientY - r.top - GEO.y0) / GEO.ca);
      if (x < 0 || y < 0 || x >= N || y >= N) return false;
      img[y * N + x] = 1;
      return true;
    };
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', (e) => {
      if (pt(e) === false) return;
      pb.stop();
      paint = true;
      c.setPointerCapture(e.pointerId);
      comp();
      pos = END;
      say();
      draw();
    });
    c.addEventListener('pointermove', (e) => {
      if (!paint) return;
      pt(e);
      comp();
      draw();
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      c.addEventListener(n, () => {
        if (paint) {
          paint = false;
          say();
          draw();
        }
      }),
    );
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        pb.start();
      },
    };
  },
};
