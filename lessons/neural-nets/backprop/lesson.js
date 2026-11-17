/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  SANS,
  readThemeColors,
  bind,
  clamp,
  eqShow,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { UC, segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';
import { sig } from '../../classification/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#bp-c', root),
      cr = $('#bp-r', root),
      live = $('#bp-live', root),
      tab = $('#bp-tab', root),
      eq = $('#bp-eq', root),
      settings = { x: [0.8, 0.6], y: 1, lr: 1.5, slow: true };
    let w,
      b,
      v,
      cc,
      ph = 0,
      passState = {},
      hist = [],
      parameterHistory = [],
      upd = 0,
      msg = '',
      frameCount = 0,
      t0 = 0,
      playback,
      checkRunStop = () => false,
      beforeUpdate = null,
      nl = 1;
    const s4 = (q) => (q >= 0 ? '+' : '−') + Math.abs(q).toFixed(3),
      n3 = (q) => (q < 0 ? '(−' + Math.abs(q).toFixed(3) + ')' : q.toFixed(3)),
      n2 = (q) => (q < 0 ? '(−' + Math.abs(q).toFixed(2) + ')' : q.toFixed(2)),
      B = (q) => '<b>' + q + '</b>';
    const all = () => [w[0][0], w[0][1], b[0], w[1][0], w[1][1], b[1], v[0], v[1], cc],
      NM = ['w₁₁', 'w₁₂', 'b₁', 'w₂₁', 'w₂₂', 'b₂', 'v₁', 'v₂', 'c'],
      LB = [
        'hidden 1 ← cloud cover',
        'hidden 1 ← humidity',
        'hidden 1 bias',
        'hidden 2 ← cloud cover',
        'hidden 2 ← humidity',
        'hidden 2 bias',
        'output ← hidden 1',
        'output ← hidden 2',
        'output bias',
      ];
    const calc = () => {
      const z = [0, 1].map((j) => w[j][0] * settings.x[0] + w[j][1] * settings.x[1] + b[j]),
        h = z.map(Math.tanh),
        zo = v[0] * h[0] + v[1] * h[1] + cc,
        o = sig(zo),
        L = 0.5 * (o - settings.y) ** 2,
        dO = (o - settings.y) * o * (1 - o),
        dH = [0, 1].map((j) => dO * v[j] * (1 - h[j] * h[j]));
      passState = {
        z,
        h,
        zo,
        o,
        L,
        dO,
        dH,
        G: [
          dH[0] * settings.x[0],
          dH[0] * settings.x[1],
          dH[0],
          dH[1] * settings.x[0],
          dH[1] * settings.x[1],
          dH[1],
          dO * h[0],
          dO * h[1],
          dO,
        ],
      };
    };
    const show = (lines) => {
      nl = lines.length;
      eqShow(eq, lines, settings.slow ? 480 : 0);
      t0 = frameCount;
    };
    const start = () => {
      w = [
        [0.5, -0.4],
        [-0.3, 0.6],
      ];
      b = [0.1, -0.1];
      v = [0.4, -0.5];
      cc = -0.2;
      ph = 0;
      hist = [];
      upd = 0;
      beforeUpdate = null;
      calc();
      hist.push(passState.L);
      parameterHistory = [all()];
      msg =
        'The example: cloud cover ' +
        f2(settings.x[0]) +
        ', humidity ' +
        f2(settings.x[1]) +
        '. What really happened: ' +
        (settings.y ? 'it rained' : 'it stayed dry') +
        '. The weights are an arbitrary starting guess. Press Run or One step.';
      show([
        'x₁ = ' + B(f2(settings.x[0])) + '   (cloud cover)',
        'x₂ = ' + B(f2(settings.x[1])) + '   (humidity)',
        'y  = ' + B(settings.y) + '   (what really happened)',
      ]);
    };
    const done = () => (passState.L < 5e-4 && ph === 0 && upd > 0) || upd >= 600;
    const gk = () => (ph >= 10 ? 9 : ph >= 9 ? [0, 1, 2, 6, 7, 8] : ph >= 6 ? [6, 7, 8] : []);
    const step = () => {
      calc();
      ph = ph >= 11 ? 1 : ph + 1;
      const x = settings.x,
        q = passState;
      if (ph === 1 || ph === 2) {
        const j = ph - 1,
          s = j ? '₂' : '₁';
        msg =
          'Hidden neuron ' +
          (j + 1) +
          ' multiplies each input by its weight, adds them up with its bias, and squashes the total with tanh.';
        show([
          'z' + s + ' = w' + s + '₁·x₁ + w' + s + '₂·x₂ + b' + s,
          '   = ' +
            n2(w[j][0]) +
            '·' +
            f2(x[0]) +
            ' + ' +
            n2(w[j][1]) +
            '·' +
            f2(x[1]) +
            ' + ' +
            n2(b[j]),
          '   = ' + B(n3(q.z[j])),
          'h' + s + ' = tanh(z' + s + ') = tanh(' + n3(q.z[j]) + ') = ' + B(n3(q.h[j])),
        ]);
      } else if (ph === 3) {
        msg =
          'The output neuron does the same with the two hidden values, then a sigmoid turns the total into a probability: a ' +
          (100 * q.o).toFixed(0) +
          '% chance of rain.';
        show([
          'z = v₁·h₁ + v₂·h₂ + c',
          '  = ' +
            n2(v[0]) +
            '·' +
            n3(q.h[0]) +
            ' + ' +
            n2(v[1]) +
            '·' +
            n3(q.h[1]) +
            ' + ' +
            n2(cc),
          '  = ' + B(n3(q.zo)),
          'ŷ = 1 / (1 + e^(−z)) = ' + B(f3(q.o)),
        ]);
      } else if (ph === 4) {
        msg =
          'The loss measures how far the answer is from the truth. Everything after this asks: which weights caused it?';
        show([
          'L = ½ (ŷ − y)²',
          '  = ½ (' + f3(q.o) + ' − ' + settings.y + ')²',
          '  = ' + B(f3(q.L)),
        ]);
      } else if (ph === 5) {
        msg =
          'Blame at the output: the error times the slope of the sigmoid. ' +
          (q.dO < 0
            ? 'It is negative, so the output’s input should go up.'
            : 'It is positive, so the output’s input should go down.');
        show([
          'δ = (ŷ − y) · ŷ · (1 − ŷ)',
          '  = (' + f3(q.o) + ' − ' + settings.y + ') · ' + f3(q.o) + ' · ' + f3(1 - q.o),
          '  = ' + B(s4(q.dO)),
        ]);
      } else if (ph === 6) {
        msg =
          'A weight’s gradient is the blame of the neuron it feeds times the value that travelled along it. That gives the three gradients at the output.';
        show([
          '∂L/∂v₁ = δ · h₁ = ' + n3(q.dO) + ' · ' + n3(q.h[0]) + ' = ' + B(s4(q.G[6])),
          '∂L/∂v₂ = δ · h₂ = ' + n3(q.dO) + ' · ' + n3(q.h[1]) + ' = ' + B(s4(q.G[7])),
          '∂L/∂c  = δ = ' + B(s4(q.G[8])),
        ]);
      } else if (ph === 7 || ph === 8) {
        const j = ph - 7,
          s = j ? '₂' : '₁';
        msg =
          'The blame travels back to hidden neuron ' +
          (j + 1) +
          ' along the weight that connects it to the output, and is scaled by the slope of that neuron’s tanh.';
        show([
          'δ' + s + ' = δ · v' + s + ' · (1 − h' + s + '²)',
          '   = ' + n3(q.dO) + ' · ' + n2(v[j]) + ' · (1 − ' + n3(q.h[j]) + '²)',
          '   = ' + n3(q.dO) + ' · ' + n2(v[j]) + ' · ' + f3(1 - q.h[j] ** 2),
          '   = ' + B(s4(q.dH[j])),
        ]);
      } else if (ph === 9 || ph === 10) {
        const j = ph - 9,
          s = j ? '₂' : '₁';
        msg =
          'Blame of hidden neuron ' +
          (j + 1) +
          ' times each input gives the gradients of the three weights that feed it.';
        show([
          '∂L/∂w' +
            s +
            '₁ = δ' +
            s +
            ' · x₁ = ' +
            n3(q.dH[j]) +
            ' · ' +
            f2(x[0]) +
            ' = ' +
            B(s4(q.G[3 * j])),
          '∂L/∂w' +
            s +
            '₂ = δ' +
            s +
            ' · x₂ = ' +
            n3(q.dH[j]) +
            ' · ' +
            f2(x[1]) +
            ' = ' +
            B(s4(q.G[3 * j + 1])),
          '∂L/∂b' + s + '  = δ' + s + ' = ' + B(s4(q.G[3 * j + 2])),
        ]);
      } else {
        const o0 = q.o,
          L0 = q.L,
          A0 = all();
        beforeUpdate = { A: A0, G: q.G.slice() };
        for (let j = 0; j < 2; j++) {
          v[j] -= settings.lr * q.G[6 + j];
          b[j] -= settings.lr * q.G[3 * j + 2];
          for (let i = 0; i < 2; i++) w[j][i] -= settings.lr * q.G[3 * j + i];
        }
        cc -= settings.lr * q.G[8];
        upd++;
        calc();
        hist.push(passState.L);
        const A1 = all();
        parameterHistory.push(A1);
        ph = 0;
        msg =
          'Update ' +
          upd +
          ': every weight moved against its gradient. Running the same example forward again, the answer goes from ' +
          f3(o0) +
          ' to ' +
          f3(passState.o) +
          ' and the loss from ' +
          f3(L0) +
          ' to ' +
          f3(passState.L) +
          '.' +
          (done()
            ? upd >= 600
              ? ' Reached the limit of 600 updates.'
              : ' Converged: the answer is within a few points of the truth, so it stops here.'
            : '');
        show(
          ['new weight = old weight − η · gradient,   η = ' + f2(settings.lr)].concat(
            NM.map(
              (nm, i) =>
                nm +
                (nm.length < 3 ? ' ' : '') +
                ' : ' +
                n3(A0[i]) +
                ' − ' +
                f2(settings.lr) +
                '·' +
                n3(beforeUpdate.G[i]) +
                ' = ' +
                B(n3(A1[i])),
            ),
          ),
        );
      }
    };
    const full = () => {
      const sl = settings.slow;
      settings.slow = false;
      do {
        step();
      } while (ph !== 0);
      settings.slow = sl;
    };
    const draw = () => {
      const colors = readThemeColors(),
        known = gk(),
        kn = (i) => known === 9 || (Array.isArray(known) && known.includes(i));
      {
        const { g: ctx, w: W, h } = prepareCanvas(mainCanvas),
          P = {
            i: [
              [W * 0.13, h * 0.3],
              [W * 0.13, h * 0.72],
            ],
            h: [
              [W * 0.5, h * 0.25],
              [W * 0.5, h * 0.77],
            ],
            o: [W * 0.87, h * 0.51],
          },
          R = 27,
          u = prefersReducedMotion ? 1 : clamp((frameCount - t0) / 70, 0, 1),
          e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u);
        ctx.font = MONO;
        const edge = (a, b2, wt, gi, act, dir, pk) => {
          ctx.strokeStyle = act ? colors.ink : mix(colors.ink, colors.ink, 0, 0.3);
          ctx.lineWidth = act ? 2.6 : 1.2;
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b2[0], b2[1]);
          ctx.stroke();
          const lx = a[0] + (b2[0] - a[0]) * 0.28,
            ly = a[1] + (b2[1] - a[1]) * 0.28;
          ctx.font = MONO;
          ctx.fillStyle = colors.bg;
          ctx.fillRect(lx - 21, ly - 9, 42, 17);
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText(f2(wt), lx, ly + 4);
          if (kn(gi)) {
            const gx2 = a[0] + (b2[0] - a[0]) * 0.7,
              gy = a[1] + (b2[1] - a[1]) * 0.7;
            ctx.fillStyle = colors.bg;
            ctx.fillRect(gx2 - 27, gy - 9, 54, 17);
            ctx.fillStyle = colors.ok;
            ctx.fillText(s4(passState.G[gi]), gx2, gy + 4);
          }
          if (act && u < 1) {
            const q = dir > 0 ? e : 1 - e,
              px = a[0] + (b2[0] - a[0]) * q,
              py = a[1] + (b2[1] - a[1]) * q,
              cl = dir > 0 ? colors.gold : colors.ok;
            ctx.font = SANS;
            const tw = ctx.measureText(pk).width + 14;
            ctx.fillStyle = cl;
            ctx.beginPath();
            ctx.rect(px - tw / 2, py - 11, tw, 22);
            ctx.fill();
            ctx.fillStyle = colors.bg;
            ctx.fillText(pk, px, py + 5);
          }
        };
        for (let j = 0; j < 2; j++)
          for (let i = 0; i < 2; i++) {
            const fw = ph === 1 + j,
              bk = ph === 9 + j;
            edge(
              P.i[i],
              P.h[j],
              w[j][i],
              3 * j + i,
              fw || bk,
              bk ? -1 : 1,
              bk ? s4(passState.dH[j]) : f2(settings.x[i]) + '×' + f2(w[j][i]),
            );
          }
        for (let j = 0; j < 2; j++) {
          const fw = ph === 3,
            bk = ph === 6 || ph === 7 + j;
          edge(
            P.h[j],
            P.o,
            v[j],
            6 + j,
            fw || bk,
            bk ? -1 : 1,
            bk ? s4(passState.dO) : f3(passState.h[j]) + '×' + f2(v[j]),
          );
        }
        const node = (p, val, lab, on, blame) => {
          ctx.fillStyle = on ? mix(colors.gold, colors.bg, 0, 0.9) : colors.bg;
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = on ? 2.8 : 1.5;
          ctx.beginPath();
          ctx.arc(p[0], p[1], on ? R + 3 * Math.sin(Math.PI * u) : R, 0, 7);
          ctx.fill();
          ctx.stroke();
          ctx.font = SANS;
          ctx.fillStyle = on ? colors.bg : colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText(val, p[0], p[1] + 5);
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.fillText(lab, p[0], p[1] - R - 9);
          if (blame != null) {
            ctx.fillStyle = colors.ok;
            ctx.fillText('blame ' + s4(blame), p[0], p[1] + R + 16);
          }
        };
        const seen = ph === 0 && upd > 0;
        node(P.i[0], f2(settings.x[0]), 'cloud cover', ph === 0 && !upd);
        node(P.i[1], f2(settings.x[1]), 'humidity', ph === 0 && !upd);
        node(
          P.h[0],
          ph >= 1 || seen ? f3(passState.h[0]) : '?',
          'hidden 1',
          (ph === 1 && u > 0.6) || ph === 7,
          ph >= 7 ? passState.dH[0] : null,
        );
        node(
          P.h[1],
          ph >= 2 || seen ? f3(passState.h[1]) : '?',
          'hidden 2',
          (ph === 2 && u > 0.6) || ph === 8,
          ph >= 8 ? passState.dH[1] : null,
        );
        node(
          P.o,
          ph >= 3 || seen ? f3(passState.o) : '?',
          'chance of rain',
          (ph === 3 && u > 0.6) || ph === 4 || ph === 5,
          ph >= 5 ? passState.dO : null,
        );
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          ph >= 1 && ph <= 4
            ? 'forward pass →'
            : ph >= 5
              ? '← backward pass'
              : seen
                ? 'weights updated'
                : '',
          8,
          h - 8,
        );
        ctx.textAlign = 'right';
        ctx.fillText('black on lines: weight.  red: its gradient', W - 8, h - 8);
      }
      {
        const { g: ctx, w: W, h } = prepareCanvas(cr),
          o = passState.o,
          x0 = 24,
          bw = W - 48,
          y0 = 30;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('the network’s answer (gold) and the truth (black line)', 8, 13);
        ctx.fillStyle = mix(colors.ink, colors.ink, 0, 0.1);
        ctx.fillRect(x0, y0, bw, 20);
        ctx.fillStyle = colors.gold;
        ctx.fillRect(x0, y0, bw * o, 20);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x0 + bw * settings.y, y0 - 6);
        ctx.lineTo(x0 + bw * settings.y, y0 + 26);
        ctx.stroke();
        ctx.fillStyle = colors.ink;
        ctx.textAlign = 'center';
        ctx.fillText(
          (100 * o).toFixed(0) + '% rain',
          clamp(x0 + bw * o, x0 + 30, x0 + bw - 30),
          y0 + 38,
        );
        const ya = 84,
          ha = h * 0.3,
          rh = (ha - 18) / 9,
          G = ph === 0 && beforeUpdate ? beforeUpdate.G : passState.G,
          gm = Math.max(0.02, ...G.map(Math.abs)),
          xc = W * 0.58,
          sx = W * 0.36;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          ph === 0 && beforeUpdate
            ? 'gradients used in the last update'
            : 'gradient of each weight (appears when computed)',
          8,
          ya,
        );
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xc, ya + 8);
        ctx.lineTo(xc, ya + 10 + 9 * rh);
        ctx.stroke();
        for (let i = 0; i < 9; i++) {
          const y = ya + 10 + i * rh,
            on = (ph === 0 && beforeUpdate) || kn(i);
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText(NM[i], xc - sx - 6, y + rh * 0.7);
          if (on) {
            const q = G[i];
            ctx.fillStyle = q < 0 ? colors.accent : colors.ok;
            ctx.fillRect(q < 0 ? xc + (q / gm) * sx : xc, y + 2, (Math.abs(q) / gm) * sx, rh - 4);
            ctx.fillStyle = colors.muted;
            ctx.textAlign = q < 0 ? 'right' : 'left';
            ctx.fillText(s4(q), xc + (q / gm) * sx + (q < 0 ? -4 : 4), y + rh * 0.7);
          }
        }
        ctx.fillStyle = colors.accent;
        ctx.textAlign = 'left';
        ctx.fillText('← raise this weight', xc - sx, ya + 18 + 9 * rh);
        ctx.fillStyle = colors.ok;
        ctx.textAlign = 'right';
        ctx.fillText('lower it →', xc + sx, ya + 18 + 9 * rh);
        const yb = ya + ha + 22,
          hb = (h - yb) / 2;
        drawSeriesChart(ctx, colors, 0, yb, W, hb, hist, {
          title: 'loss after each update',
          n: Math.max(20, hist.length),
          col: colors.ok,
          dots: hist.length < 40,
          ym: Math.max(0.02, hist[0] || 0.1),
        });
        let lo = 1e9,
          hi = -1e9;
        for (const r of parameterHistory)
          for (const q of r) {
            lo = Math.min(lo, q);
            hi = Math.max(hi, q);
          }
        const pad = (hi - lo) * 0.08 + 0.05,
          col = UC(colors);
        const U = drawSeriesChart(ctx, colors, 0, yb + hb, W, hb, [], {
          title: 'the nine weights, update by update',
          n: Math.max(20, parameterHistory.length),
          y1: lo - pad,
          ym: hi + pad,
        });
        for (let i = 0; i < 9; i++) {
          ctx.strokeStyle = col[i % 8];
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          parameterHistory.forEach((r, k) =>
            k ? ctx.lineTo(U.X(k), U.Y(r[i])) : ctx.moveTo(U.X(k), U.Y(r[i])),
          );
          ctx.stroke();
        }
      }
      const O = ph === 0 && beforeUpdate ? beforeUpdate : null,
        A = all();
      tab.innerHTML =
        '<table><thead><tr><th>weight</th><th></th><th>' +
        (O ? 'before' : 'value now') +
        '</th><th>gradient</th><th>' +
        (O ? 'after: before − ' + f2(settings.lr) + ' × gradient' : '') +
        '</th></tr></thead><tbody>' +
        LB.map(
          (lb, i) =>
            '<tr><td>' +
            NM[i] +
            '</td><td>' +
            lb +
            '</td><td>' +
            f3(O ? O.A[i] : A[i]) +
            '</td><td>' +
            (O ? s4(O.G[i]) : kn(i) ? s4(passState.G[i]) : 'not known yet') +
            '</td><td>' +
            (O ? f3(A[i]) : '') +
            '</td></tr>',
        ).join('') +
        '</tbody></table>';
      live.textContent = msg;
      highlightLessonStep(
        root,
        'bp',
        ph === 0 ? (upd ? 8 : 0) : [0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7][ph],
      );
      setReadout(root, 'o', (100 * passState.o).toFixed(1) + '%');
      setReadout(root, 'y', settings.y ? 'It rained (1)' : 'It stayed dry (0)');
      setReadout(
        root,
        'L',
        f3(passState.L) + (passState.L < 0.001 ? ' (' + passState.L.toExponential(1) + ')' : ''),
      );
      setReadout(root, 'u', String(upd));
      setReadout(root, 'st', done() ? 'Converged' : upd ? 'Learning' : 'Not started');
    };
    const fx = Anim(() => {
      frameCount++;
      draw();
      if (frameCount - t0 > 80) fx.stop();
    });
    start();
    const anim = Anim(() => {
      frameCount++;
      if (settings.slow && upd < 2) {
        if (frameCount - t0 > Math.max(150, nl * 30 + 110)) {
          step();
          if (checkRunStop()) {
            draw();
            return;
          }
        }
        draw();
        return;
      }
      if (frameCount % (upd < 10 ? 10 : 3) === 0) {
        full();
        draw();
        checkRunStop();
        return;
      }
    });
    playback = playBtn($('#bp-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'bp', {
      done,
      restart: () => start(),
      count: () => upd,
      what: 'updates',
    });
    $('#bp-step', root).onclick = () => {
      playback.stop();
      if (done()) start();
      else step();
      fx.start();
    };
    $('#bp-reset', root).onclick = () => {
      playback.stop();
      start();
      draw();
    };
    const re = () => {
      playback.stop();
      start();
      draw();
    };
    bind(root, 'bp-x1', (q) => {
      settings.x[0] = q;
      re();
    });
    bind(root, 'bp-x2', (q) => {
      settings.x[1] = q;
      re();
    });
    bind(
      root,
      'bp-lr',
      (q) => {
        settings.lr = q;
      },
      f2,
    );
    segBind(root, 'bp-y', String(settings.y), (q) => {
      settings.y = +q;
      re();
    });
    $('#bp-slow', root).onchange = (e) => {
      settings.slow = e.target.checked;
    };
    return {
      draw,
      stop: () => {
        playback.stop();
        fx.stop();
      },
      auto: () => {
        start();
        draw();
        playback.start();
      },
    };
  },
};
