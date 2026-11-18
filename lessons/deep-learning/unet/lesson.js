/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { pct } from '../../ensembles/shared.js';
import { sig } from '../../classification/shared.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#un-c', root),
      cr = $('#un-r', root),
      live = $('#un-live', root),
      S = { nz: 0.3 },
      MAXS = 700;
    let plainNet,
      skipNet,
      ts = 0,
      IA = [],
      IB = [],
      TE = [],
      pad = new Float64Array(256),
      previewMaskProbs = [],
      done = false,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      geo = null,
      paint = false;
    const mkNet = (skip, seed) => {
      const q = randnFrom(rng(seed)),
        cw = (o, ci) => AD.param([o, ci, 3, 3], Math.sqrt(2 / (ci * 9)), q),
        cb = (o) => AD.param([o], 0, q),
        P = {
          w1: cw(4, 1),
          b1: cb(4),
          w2: cw(6, 4),
          b2: cb(6),
          w3: cw(8, 6),
          b3: cb(8),
          w4: cw(6, skip ? 14 : 8),
          b4: cb(6),
          w5: cw(4, skip ? 10 : 6),
          b5: cb(4),
          w6: cw(1, 4),
          b6: cb(1),
        },
        ps = Object.values(P);
      return {
        ps,
        fwd: (img, tg) => {
          const x = AD.input(img, [1, 16, 16]),
            e1 = AD.relu(AD.conv3(x, P.w1, P.b1)),
            e2 = AD.relu(AD.conv3(AD.pool(e1), P.w2, P.b2)),
            bt = AD.relu(AD.conv3(AD.pool(e2), P.w3, P.b3)),
            u2 = AD.up(bt),
            d2 = AD.relu(AD.conv3(skip ? AD.cat(u2, e2) : u2, P.w4, P.b4)),
            u1 = AD.up(d2),
            d1 = AD.relu(AD.conv3(skip ? AD.cat(u1, e1) : u1, P.w5, P.b5)),
            L = AD.conv3(d1, P.w6, P.b6);
          return { L, loss: tg ? AD.bce(L, tg) : null };
        },
      };
    };
    const shape = (r, rn) => {
      const m = new Float64Array(256),
        k = 1 + Math.floor(r() * 2);
      for (let s = 0; s < k; s++) {
        const ty = Math.floor(r() * 3),
          cx = 3 + r() * 10,
          cy = 3 + r() * 10,
          rd = 2 + r() * 3.5;
        for (let y = 0; y < 16; y++)
          for (let x = 0; x < 16; x++) {
            const dx = x - cx,
              dy = y - cy;
            if (ty === 0) {
              if (Math.abs(Math.hypot(dx, dy) - rd) < 0.8) m[y * 16 + x] = 1;
            } else if (ty === 1) {
              if (Math.abs(Math.max(Math.abs(dx), Math.abs(dy)) - rd) < 0.7) m[y * 16 + x] = 1;
            } else if (
              Math.abs(dx * Math.cos(rd) + dy * Math.sin(rd)) < 0.7 &&
              Math.hypot(dx, dy) < rd + 2.5
            )
              m[y * 16 + x] = 1;
          }
      }
      return { m, x: m.map((v) => v + S.nz * rn()) };
    };
    const R = () => Math.random(),
      RN = randnFrom(Math.random);
    const iou = (N) => {
      let s = 0;
      for (const q of TE) {
        const o = N.fwd(q.x);
        AD.clear();
        let a = 0,
          b = 0;
        for (let i = 0; i < 256; i++) {
          const p = o.L.v[i] > 0 ? 1 : 0;
          if (p && q.m[i]) a++;
          if (p || q.m[i]) b++;
        }
        s += b ? a / b : 1;
      }
      return s / TE.length;
    };
    const outs = () => {
      previewMaskProbs = [];
      const rows = TE.slice(0, 3)
        .map((q) => q.x)
        .concat([pad]);
      for (const x of rows) {
        const a = plainNet.fwd(x),
          b = skipNet.fwd(x);
        AD.clear();
        previewMaskProbs.push([Float64Array.from(a.L.v, sig), Float64Array.from(b.L.v, sig)]);
      }
    };
    const build = () => {
      plainNet = mkNet(false, run * 7 + 1);
      skipNet = mkNet(true, run * 7 + 1);
      const r = rng(run * 5 + 9),
        rn = randnFrom(r);
      TE = [];
      for (let i = 0; i < 20; i++) TE.push(shape(r, rn));
      ts = 0;
      IA = [];
      IB = [];
      done = false;
      activeStepIndex = 0;
      outs();
      msg = 'Two networks with random weights. Both answers are grey fog. Press Train both.';
    };
    const train = (k) => {
      for (let q = 0; q < k && !done; q++) {
        for (let b = 0; b < 5; b++) {
          const s = shape(R, RN);
          for (const N of [plainNet, skipNet]) {
            const o = N.fwd(s.x, s.m);
            AD.backward(o.loss);
          }
        }
        ts++;
        AD.adam(plainNet.ps, 0.006, ts, 5);
        AD.adam(skipNet.ps, 0.006, ts, 5);
        if (ts % 10 === 0 || ts === 1) {
          IA.push(iou(plainNet));
          IB.push(iou(skipNet));
        }
        if (ts >= MAXS) done = true;
      }
      outs();
      activeStepIndex = done ? 4 : 1 + (ts % 3);
      const a = IA[IA.length - 1],
        b = IB[IB.length - 1];
      msg =
        'Step ' +
        ts +
        ' of ' +
        MAXS +
        ': both networks saw 5 fresh noisy pictures and adjusted their convolution kernels. On 20 test pictures, the answer overlaps the truth by ' +
        pct(a) +
        ' without skips and ' +
        pct(b) +
        ' with skips.' +
        (done
          ? ' Both have had their ' +
            MAXS +
            ' steps. ' +
            (b > a + 0.1
              ? 'The skips make the difference between a blur and a clean outline.'
              : 'They finish close: with this little noise or this few thin lines, the bottleneck was enough.')
          : '');
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          sz = Math.min((w - 30) / 4.3, (h - 34) / 4.25),
          gx = (w - sz * 4.3) / 2,
          cz = sz / 16,
          hd = ['noisy input', 'true shape', 'without skips', 'with skips'],
          rows = TE.slice(0, 3);
        ctx.font = MONO;
        hd.forEach((s, q) => {
          ctx.fillStyle = q === 2 ? colors.ok : q === 3 ? colors.accent : colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText(s, gx + q * sz * 1.1 + sz / 2, 12);
        });
        const pic = (A, x, y, cl) => {
          for (let i = 0; i < 256; i++) {
            ctx.fillStyle = mix(colors.bg, colors.ink, 0.05 + 0.95 * clamp(A[i], 0, 1));
            ctx.fillRect(x + (i % 16) * cz, y + Math.floor(i / 16) * cz, cz + 0.3, cz + 0.3);
          }
          ctx.strokeStyle = cl || colors.line;
          ctx.lineWidth = cl ? 2 : 1;
          ctx.strokeRect(x - 0.5, y - 0.5, sz, sz);
        };
        for (let r = 0; r < 4; r++) {
          const y = 20 + r * sz * 1.06,
            mine = r === 3;
          pic(mine ? pad : rows[r].x, gx, y, mine ? colors.gold : null);
          if (mine) {
            ctx.fillStyle = colors.muted;
            ctx.textAlign = 'center';
            ctx.fillText('draw in the gold box', gx + sz * 1.1 + sz / 2, y + sz / 2);
            geo = { x: gx, y, cz };
          } else pic(rows[r].m, gx + sz * 1.1, y);
          pic(previewMaskProbs[r][0], gx + 2 * sz * 1.1, y, colors.ok);
          pic(previewMaskProbs[r][1], gx + 3 * sz * 1.1, y, colors.accent);
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          ph = (frameCount / 90) % 1,
          L = [
            [0.1, 0.16, 34, '16 × 16', '4 maps'],
            [0.27, 0.5, 24, '8 × 8', '6 maps'],
            [0.5, 0.82, 15, '4 × 4', '8 maps'],
            [0.73, 0.5, 24, '8 × 8', '6 maps'],
            [0.9, 0.16, 34, '16 × 16', '4 maps'],
          ].map((q) => [q[0] * w, q[1] * (h1 - 30) + 14, q[2], q[3], q[4]]);
        ctx.font = MONO;
        for (let q = 0; q < 4; q++) {
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(L[q][0], L[q][1]);
          ctx.lineTo(L[q + 1][0], L[q + 1][1]);
          ctx.stroke();
        }
        [
          [0, 4],
          [1, 3],
        ].forEach(([a, b]) => {
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 2.6;
          ctx.beginPath();
          ctx.moveTo(L[a][0] + L[a][2] / 2 + 4, L[a][1]);
          ctx.lineTo(L[b][0] - L[b][2] / 2 - 4, L[b][1]);
          ctx.stroke();
          ctx.fillStyle = colors.accent;
          ctx.textAlign = 'center';
          ctx.fillText('skip', (L[a][0] + L[b][0]) / 2, L[a][1] - 6);
          if (playback && playback.on) {
            ctx.beginPath();
            ctx.arc(L[a][0] + (L[b][0] - L[a][0]) * ph, L[a][1], 4.5, 0, 7);
            ctx.fill();
          }
        });
        L.forEach((q, i) => {
          ctx.fillStyle = i === 2 ? colors.gold : colors.bg;
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.rect(q[0] - q[2] / 2, q[1] - q[2] / 2, q[2], q[2]);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText(q[3], q[0], q[1] + q[2] / 2 + 13);
          ctx.fillText(q[4], q[0], q[1] + q[2] / 2 + 25);
        });
        if (playback && playback.on) {
          const k = Math.min(3, Math.floor(ph * 4)),
            uu = ph * 4 - k;
          ctx.fillStyle = colors.gold;
          ctx.beginPath();
          ctx.arc(
            L[k][0] + (L[k + 1][0] - L[k][0]) * uu,
            L[k][1] + (L[k + 1][1] - L[k][1]) * uu,
            5,
            0,
            7,
          );
          ctx.fill();
        }
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('down: convolve, shrink', 8, h1 - 6);
        ctx.textAlign = 'right';
        ctx.fillText('up: enlarge, convolve', w - 8, h1 - 6);
        const U = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, IB, {
          title: 'overlap with the truth on test pictures: with skips (green), without (red)',
          n: MAXS / 10 + 1,
          ym: 1,
          col: colors.accent,
        });
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        IA.forEach((v, i) => (i ? ctx.lineTo(U.X(i), U.Y(v)) : ctx.moveTo(U.X(i), U.Y(v))));
        ctx.stroke();
      }
      live.textContent = msg;
      highlightLessonStep(root, 'un', activeStepIndex);
      const cnt = (N) => N.ps.reduce((a, p) => a + p.v.length, 0);
      setReadout(root, 'step', ts + ' of ' + MAXS);
      setReadout(root, 'i0', IA.length ? pct(IA[IA.length - 1]) : '–');
      setReadout(root, 'i1', IB.length ? pct(IB[IB.length - 1]) : '–');
      setReadout(root, 'np', cnt(plainNet) + ' / ' + cnt(skipNet));
      setReadout(root, 'st', done ? 'Finished' : ts ? 'Training' : 'Not started');
    };
    build();
    if (prefersReducedMotion) train(MAXS);
    const anim = Anim(() => {
      frameCount++;
      if (ts < 10 && frameCount % 6) {
        draw();
        return;
      }
      train(Math.min(1, checkRunStop.remaining()));
      draw();
      checkRunStop();
    });
    playback = playBtn($('#un-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'un', {
      done: () => done,
      restart: () => {
        run++;
        build();
      },
      count: () => ts,
      what: 'steps',
    });
    $('#un-step', root).onclick = () => {
      playback.stop();
      if (done) {
        run++;
        build();
      } else train(10);
      draw();
    };
    $('#un-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    bind(root, 'un-nz', (v) => {
      S.nz = v;
      playback.stop();
      run++;
      build();
      draw();
    });
    $('#un-clear', root).onclick = () => {
      pad.fill(0);
      outs();
      draw();
    };
    const pt = (e) => {
      if (!geo) return false;
      const r = mainCanvas.getBoundingClientRect(),
        x = Math.floor((e.clientX - r.left - geo.x) / geo.cz),
        y = Math.floor((e.clientY - r.top - geo.y) / geo.cz);
      if (x < 0 || y < 0 || x >= 16 || y >= 16) return false;
      pad[y * 16 + x] = 1;
      return true;
    };
    mainCanvas.style.touchAction = 'none';
    mainCanvas.addEventListener('pointerdown', (e) => {
      if (pt(e) === false) return;
      paint = true;
      mainCanvas.setPointerCapture(e.pointerId);
      outs();
      if (!playback.on) draw();
    });
    mainCanvas.addEventListener('pointermove', (e) => {
      if (!paint) return;
      pt(e);
      outs();
      if (!playback.on) draw();
    });
    ['pointerup', 'pointercancel'].forEach((n2) =>
      mainCanvas.addEventListener(n2, () => {
        if (paint) {
          paint = false;
          msg =
            'Your drawing went through both networks: compare the two boxes to its right. (Your drawing has no noise, so it is easier than the training pictures.)';
          draw();
        }
      }),
    );
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        run++;
        build();
        draw();
        playback.start();
      },
    };
  },
};
