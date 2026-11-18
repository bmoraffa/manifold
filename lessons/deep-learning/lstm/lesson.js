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
  f2,
  prepareCanvas,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { sci } from '../../neural-nets/shared.js';
import { highlightLessonStep } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#ls-c', root),
      cr = $('#ls-r', root),
      cd = $('#ls-d', root),
      live = $('#ls-live', root),
      eq = $('#ls-eq', root),
      settings = { f: 0.98, i: 0.03, w: 0.9 },
      WD = [
        'The',
        'keys',
        ',',
        'which',
        'I',
        'left',
        'on',
        'the',
        'table',
        'next',
        'to',
        'the',
        'lamp',
        ',',
        '___',
      ],
      X = WD.map((w) => (w === 'keys' ? 1 : w === 'table' || w === 'lamp' ? -1 : 0)),
      tokenCount = WD.length;
    let H = [],
      C = [],
      plainGradients = [],
      lstmGradients = [],
      k = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      t0 = 0,
      playback;
    const comp = () => {
      H = [0];
      C = [0];
      for (let t = 0; t < tokenCount; t++) {
        const x = X[t];
        H.push(Math.tanh(settings.w * H[t] + 1.2 * x));
        const ig = x > 0 ? 1 : x < 0 ? settings.i : 0;
        C.push((t === 0 ? 0 : settings.f) * C[t] + ig * x);
      }
      plainGradients = new Array(tokenCount + 1).fill(1);
      lstmGradients = new Array(tokenCount + 1).fill(1);
      for (let t = tokenCount - 1; t >= 2; t--) {
        plainGradients[t] = plainGradients[t + 1] * settings.w * (1 - H[t + 1] ** 2);
        lstmGradients[t] = lstmGradients[t + 1] * settings.f;
      }
    };
    const verdict = (v) => (v > 0.3 ? '“are” (correct)' : v < -0.3 ? '“is” (wrong)' : 'unsure');
    const step = () => {
      if (k >= tokenCount) {
        k = 0;
        activeStepIndex = 0;
        eq.innerHTML = '';
        msg = 'Start again.';
        return;
      }
      k++;
      t0 = frameCount;
      const t = k - 1,
        x = X[t],
        ig = x > 0 ? 1 : x < 0 ? settings.i : 0;
      if (x > 0) {
        activeStepIndex = 0;
        msg =
          '“keys”: both networks store the fact that the subject is plural. Plain network memory ' +
          f2(H[k]) +
          ', LSTM memory ' +
          f2(C[k]) +
          '.';
      } else if (x < 0) {
        activeStepIndex = 3;
        msg =
          '“' +
          WD[t] +
          '” is singular, a distraction. The plain network has to mix it in: ' +
          f2(H[k - 1]) +
          ' → ' +
          f2(H[k]) +
          '. The LSTM’s input gate is ' +
          (settings.i < 0.2 ? 'nearly shut' : 'open ' + f2(settings.i)) +
          ': ' +
          f2(C[k - 1]) +
          ' → ' +
          f2(C[k]) +
          '.';
      } else if (k >= tokenCount) {
        activeStepIndex = 4;
        msg =
          'The blank. Plain network memory ' +
          f2(H[k]) +
          ': it says ' +
          verdict(H[k]) +
          '. LSTM memory ' +
          f2(C[k]) +
          ': it says ' +
          verdict(C[k]) +
          '.';
      } else {
        activeStepIndex = k % 2 ? 1 : 2;
        msg =
          '“' +
          WD[t] +
          '”: nothing to store. Plain network: ' +
          f2(H[k - 1]) +
          ' → ' +
          f2(H[k]) +
          ' (multiplied by its weight and squashed). LSTM: ' +
          f2(C[k - 1]) +
          ' → ' +
          f2(C[k]) +
          ' (multiplied by the forget gate, ' +
          f2(settings.f) +
          ').';
      }
      eq.innerHTML =
        '<div class="on">plain:  h = tanh( w·h + u·x ) = tanh( ' +
        f2(settings.w) +
        '·' +
        f2(H[k - 1]) +
        ' + 1.2·' +
        x +
        ' ) = <b>' +
        f2(H[k]) +
        '</b></div><div class="on">LSTM:   c = f·c + i·x = ' +
        f2(t === 0 ? 0 : settings.f) +
        '·' +
        f2(C[k - 1]) +
        ' + ' +
        f2(ig) +
        '·' +
        x +
        ' = <b>' +
        f2(C[k]) +
        '</b></div>';
    };
    const draw = () => {
      const colors = readThemeColors(),
        u = prefersReducedMotion ? 1 : clamp((frameCount - t0) / 26, 0, 1);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          m = 34,
          cw = (w - 2 * m) / tokenCount,
          rows = [
            [H, h * 0.3, colors.ok, 'plain recurrent network'],
            [C, h * 0.72, colors.accent, 'LSTM'],
          ];
        ctx.font = MONO;
        rows.forEach(([A, y, cl, lab], ri) => {
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(lab, m, y - h * 0.17);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = ri ? 3 : 1.2;
          ctx.beginPath();
          ctx.moveTo(m, y);
          ctx.lineTo(w - m, y);
          ctx.stroke();
          for (let q = 0; q < tokenCount; q++) {
            const x = m + (q + 0.5) * cw;
            ctx.font = q < k ? SANS : MONO;
            ctx.fillStyle =
              q === k - 1 ? colors.ink : q < k ? colors.muted : mix(colors.ink, colors.ink, 0, 0.3);
            ctx.textAlign = 'center';
            ctx.fillText(WD[q], x, y + h * 0.2);
            if (ri === 1) {
              const ig = X[q] > 0 ? 1 : X[q] < 0 ? settings.i : 0;
              ctx.strokeStyle = X[q] ? colors.ink : colors.line;
              ctx.lineWidth = 1.4;
              ctx.beginPath();
              ctx.moveTo(x - 7, y + h * 0.11);
              ctx.lineTo(x - 7, y + 12);
              ctx.moveTo(x + 7, y + h * 0.11);
              ctx.lineTo(x + 7, y + 12);
              ctx.stroke();
              ctx.fillStyle = X[q] ? colors.ok : colors.line;
              ctx.fillRect(x - 7, y + 12, 14 * (1 - ig), 3);
            } else {
              ctx.strokeStyle = colors.line;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.arc(x, y, 9, 0, 7);
              ctx.stroke();
            }
            if (q < k) {
              const v = A[q + 1],
                al = q === k - 1 ? u : 1,
                pv = A[q];
              const vv = pv + (v - pv) * al;
              ctx.fillStyle = vv >= 0 ? cl : colors.muted;
              ctx.globalAlpha = q === k - 1 ? 1 : 0.28;
              ctx.beginPath();
              ctx.arc(x - (1 - al) * cw, y, 2 + 15 * Math.min(1, Math.abs(vv)), 0, 7);
              ctx.fill();
              ctx.globalAlpha = 1;
              if (q === k - 1) {
                ctx.fillStyle = colors.ink;
                ctx.fillText(f2(v), x, y - h * 0.11);
              }
            }
          }
        });
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('grey dot: the memory has flipped to “singular”', w - m, h - 6);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          m = { l: 40, r: 12, t: 16, b: 22 },
          X2 = (q) => m.l + (q / tokenCount) * (w - m.l - m.r),
          Y2 = (v) => h - m.b - ((clamp(v, -1.1, 1.1) + 1.1) / 2.2) * (h - m.t - m.b);
        ctx.font = MONO;
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        [-1, 0, 1].forEach((v) => {
          ctx.beginPath();
          ctx.moveTo(m.l, Y2(v));
          ctx.lineTo(w - m.r, Y2(v));
          ctx.stroke();
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText(String(v), m.l - 5, Y2(v) + 4);
        });
        [
          [H, colors.ok],
          [C, colors.accent],
        ].forEach(([A, cl]) => {
          ctx.strokeStyle = cl;
          ctx.lineWidth = 2.6;
          ctx.beginPath();
          for (let q = 1; q <= k; q++)
            q > 1 ? ctx.lineTo(X2(q), Y2(A[q])) : ctx.moveTo(X2(q), Y2(A[q]));
          ctx.stroke();
          if (k) {
            ctx.fillStyle = cl;
            ctx.beginPath();
            ctx.arc(X2(k), Y2(A[k]), 5, 0, 7);
            ctx.fill();
          }
        });
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.ok;
        ctx.fillText('red: plain network', m.l + 6, m.t);
        ctx.fillStyle = colors.accent;
        ctx.fillText('green: LSTM', m.l + 150, m.t);
        ctx.fillStyle = colors.muted;
        ctx.fillText('+1 = “plural”, −1 = “singular”', m.l + 6, h - 6);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cd),
          y = h * 0.34,
          bx = (x, yy, tx, cl) => {
            ctx.fillStyle = colors.bg;
            ctx.strokeStyle = cl || colors.ink;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.rect(x - 46, yy - 14, 92, 28);
            ctx.fill();
            ctx.stroke();
            ctx.font = MONO;
            ctx.fillStyle = cl || colors.ink;
            ctx.textAlign = 'center';
            ctx.fillText(tx, x, yy + 4);
          };
        ctx.font = MONO;
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(16, y);
        ctx.lineTo(w - 16, y);
        ctx.stroke();
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('old memory c', 16, y - 12);
        ctx.textAlign = 'right';
        ctx.fillText('new memory c', w - 16, y - 12);
        const xf = w * 0.32,
          xi = w * 0.62;
        [
          [xf, '×'],
          [xi, '+'],
        ].forEach(([x, s]) => {
          ctx.fillStyle = colors.bg;
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(x, y, 12, 0, 7);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'center';
          ctx.font = SANS;
          ctx.fillText(s, x, y + 5);
          ctx.font = MONO;
          ctx.strokeStyle = colors.ink;
          ctx.beginPath();
          ctx.moveTo(x, y + 12);
          ctx.lineTo(x, h * 0.66 - 14);
          ctx.stroke();
        });
        bx(xf, h * 0.66, 'forget ' + f2(settings.f));
        bx(xi, h * 0.66, 'input × new', colors.ok);
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        ctx.fillText('how much to keep', xf, h * 0.66 + 30);
        ctx.fillText('whether to write', xi, h * 0.66 + 30);
        ctx.fillText('current word and previous output feed every gate', w / 2, h - 8);
        if (playback && playback.on) {
          const p = (frameCount / 80) % 1;
          ctx.fillStyle = colors.gold;
          ctx.beginPath();
          ctx.arc(16 + (w - 32) * p, y, 6, 0, 7);
          ctx.fill();
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'ls', activeStepIndex);
      setReadout(root, 'r', f2(H[tokenCount]) + ', ' + verdict(H[tokenCount]));
      setReadout(root, 'l', f2(C[tokenCount]) + ', ' + verdict(C[tokenCount]));
      setReadout(root, 'rg', sci(Math.abs(plainGradients[2])));
      setReadout(root, 'lg', sci(lstmGradients[2]));
    };
    comp();
    k = tokenCount;
    const anim = Anim(() => {
      frameCount++;
      if (frameCount - t0 > (k < 3 ? 90 : 48) && k < tokenCount) step();
      draw();
      if (k >= tokenCount && frameCount - t0 > 30) playback.stop();
    });
    playback = playBtn($('#ls-play', root), anim);
    attachRunLimit(playback, root, 'ls', {
      done: () => k >= tokenCount,
      restart: () => {
        k = 0;
      },
    });
    $('#ls-step', root).onclick = () => {
      playback.stop();
      step();
      t0 = frameCount - 100;
      draw();
    };
    $('#ls-reset', root).onclick = () => {
      playback.stop();
      k = 0;
      activeStepIndex = 0;
      eq.innerHTML = '';
      msg = 'Press Read the sentence or One word.';
      draw();
    };
    const re = () => {
      playback.stop();
      comp();
      k = tokenCount;
      msg =
        'With these settings the plain network ends at ' +
        f2(H[tokenCount]) +
        ' (' +
        verdict(H[tokenCount]) +
        ') and the LSTM at ' +
        f2(C[tokenCount]) +
        ' (' +
        verdict(C[tokenCount]) +
        ').';
      draw();
    };
    bind(root, 'ls-f', (v) => {
      settings.f = v;
      re();
    });
    bind(root, 'ls-i', (v) => {
      settings.i = v;
      re();
    });
    bind(root, 'ls-w', (v) => {
      settings.w = v;
      re();
    });
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        comp();
        k = 0;
        draw();
        playback.start();
      },
    };
  },
};
