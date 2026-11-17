/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  SANS,
  T,
  bind,
  clamp,
  f2,
  fit,
  mix,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { sci } from '../../neural-nets/shared.js';
import { uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#ls-c', root),
      cr = $('#ls-r', root),
      cd = $('#ls-d', root),
      live = $('#ls-live', root),
      eq = $('#ls-eq', root),
      S = { f: 0.98, i: 0.03, w: 0.9 },
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
      T2 = WD.length;
    let H = [],
      C = [],
      GR = [],
      GL = [],
      k = 0,
      msg = '',
      cu = 0,
      fr = 0,
      t0 = 0,
      pb;
    const comp = () => {
      H = [0];
      C = [0];
      for (let t = 0; t < T2; t++) {
        const x = X[t];
        H.push(Math.tanh(S.w * H[t] + 1.2 * x));
        const ig = x > 0 ? 1 : x < 0 ? S.i : 0;
        C.push((t === 0 ? 0 : S.f) * C[t] + ig * x);
      }
      GR = new Array(T2 + 1).fill(1);
      GL = new Array(T2 + 1).fill(1);
      for (let t = T2 - 1; t >= 2; t--) {
        GR[t] = GR[t + 1] * S.w * (1 - H[t + 1] ** 2);
        GL[t] = GL[t + 1] * S.f;
      }
    };
    const verdict = (v) => (v > 0.3 ? '“are” (correct)' : v < -0.3 ? '“is” (wrong)' : 'unsure');
    const step = () => {
      if (k >= T2) {
        k = 0;
        cu = 0;
        eq.innerHTML = '';
        msg = 'Start again.';
        return;
      }
      k++;
      t0 = fr;
      const t = k - 1,
        x = X[t],
        ig = x > 0 ? 1 : x < 0 ? S.i : 0;
      if (x > 0) {
        cu = 0;
        msg =
          '“keys”: both networks store the fact that the subject is plural. Plain network memory ' +
          f2(H[k]) +
          ', LSTM memory ' +
          f2(C[k]) +
          '.';
      } else if (x < 0) {
        cu = 3;
        msg =
          '“' +
          WD[t] +
          '” is singular, a distraction. The plain network has to mix it in: ' +
          f2(H[k - 1]) +
          ' → ' +
          f2(H[k]) +
          '. The LSTM’s input gate is ' +
          (S.i < 0.2 ? 'nearly shut' : 'open ' + f2(S.i)) +
          ': ' +
          f2(C[k - 1]) +
          ' → ' +
          f2(C[k]) +
          '.';
      } else if (k >= T2) {
        cu = 4;
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
        cu = k % 2 ? 1 : 2;
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
          f2(S.f) +
          ').';
      }
      eq.innerHTML =
        '<div class="on">plain:  h = tanh( w·h + u·x ) = tanh( ' +
        f2(S.w) +
        '·' +
        f2(H[k - 1]) +
        ' + 1.2·' +
        x +
        ' ) = <b>' +
        f2(H[k]) +
        '</b></div><div class="on">LSTM:   c = f·c + i·x = ' +
        f2(t === 0 ? 0 : S.f) +
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
      const t = T(),
        u = RM ? 1 : clamp((fr - t0) / 26, 0, 1);
      {
        const { g, w, h } = fit(c),
          m = 34,
          cw = (w - 2 * m) / T2,
          rows = [
            [H, h * 0.3, t.ok, 'plain recurrent network'],
            [C, h * 0.72, t.accent, 'LSTM'],
          ];
        g.font = MONO;
        rows.forEach(([A, y, cl, lab], ri) => {
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(lab, m, y - h * 0.17);
          g.strokeStyle = t.muted;
          g.lineWidth = ri ? 3 : 1.2;
          g.beginPath();
          g.moveTo(m, y);
          g.lineTo(w - m, y);
          g.stroke();
          for (let q = 0; q < T2; q++) {
            const x = m + (q + 0.5) * cw;
            g.font = q < k ? SANS : MONO;
            g.fillStyle = q === k - 1 ? t.ink : q < k ? t.muted : mix(t.ink, t.ink, 0, 0.3);
            g.textAlign = 'center';
            g.fillText(WD[q], x, y + h * 0.2);
            if (ri === 1) {
              const ig = X[q] > 0 ? 1 : X[q] < 0 ? S.i : 0;
              g.strokeStyle = X[q] ? t.ink : t.line;
              g.lineWidth = 1.4;
              g.beginPath();
              g.moveTo(x - 7, y + h * 0.11);
              g.lineTo(x - 7, y + 12);
              g.moveTo(x + 7, y + h * 0.11);
              g.lineTo(x + 7, y + 12);
              g.stroke();
              g.fillStyle = X[q] ? t.ok : t.line;
              g.fillRect(x - 7, y + 12, 14 * (1 - ig), 3);
            } else {
              g.strokeStyle = t.line;
              g.lineWidth = 1;
              g.beginPath();
              g.arc(x, y, 9, 0, 7);
              g.stroke();
            }
            if (q < k) {
              const v = A[q + 1],
                al = q === k - 1 ? u : 1,
                pv = A[q];
              const vv = pv + (v - pv) * al;
              g.fillStyle = vv >= 0 ? cl : t.muted;
              g.globalAlpha = q === k - 1 ? 1 : 0.28;
              g.beginPath();
              g.arc(x - (1 - al) * cw, y, 2 + 15 * Math.min(1, Math.abs(vv)), 0, 7);
              g.fill();
              g.globalAlpha = 1;
              if (q === k - 1) {
                g.fillStyle = t.ink;
                g.fillText(f2(v), x, y - h * 0.11);
              }
            }
          }
        });
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('grey dot: the memory has flipped to “singular”', w - m, h - 6);
      }
      {
        const { g, w, h } = fit(cr),
          m = { l: 40, r: 12, t: 16, b: 22 },
          X2 = (q) => m.l + (q / T2) * (w - m.l - m.r),
          Y2 = (v) => h - m.b - ((clamp(v, -1.1, 1.1) + 1.1) / 2.2) * (h - m.t - m.b);
        g.font = MONO;
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        [-1, 0, 1].forEach((v) => {
          g.beginPath();
          g.moveTo(m.l, Y2(v));
          g.lineTo(w - m.r, Y2(v));
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText(String(v), m.l - 5, Y2(v) + 4);
        });
        [
          [H, t.ok],
          [C, t.accent],
        ].forEach(([A, cl]) => {
          g.strokeStyle = cl;
          g.lineWidth = 2.6;
          g.beginPath();
          for (let q = 1; q <= k; q++)
            q > 1 ? g.lineTo(X2(q), Y2(A[q])) : g.moveTo(X2(q), Y2(A[q]));
          g.stroke();
          if (k) {
            g.fillStyle = cl;
            g.beginPath();
            g.arc(X2(k), Y2(A[k]), 5, 0, 7);
            g.fill();
          }
        });
        g.textAlign = 'left';
        g.fillStyle = t.ok;
        g.fillText('red: plain network', m.l + 6, m.t);
        g.fillStyle = t.accent;
        g.fillText('green: LSTM', m.l + 150, m.t);
        g.fillStyle = t.muted;
        g.fillText('+1 = “plural”, −1 = “singular”', m.l + 6, h - 6);
      }
      {
        const { g, w, h } = fit(cd),
          y = h * 0.34,
          bx = (x, yy, tx, cl) => {
            g.fillStyle = t.bg;
            g.strokeStyle = cl || t.ink;
            g.lineWidth = 1.5;
            g.beginPath();
            g.rect(x - 46, yy - 14, 92, 28);
            g.fill();
            g.stroke();
            g.font = MONO;
            g.fillStyle = cl || t.ink;
            g.textAlign = 'center';
            g.fillText(tx, x, yy + 4);
          };
        g.font = MONO;
        g.strokeStyle = t.accent;
        g.lineWidth = 4;
        g.beginPath();
        g.moveTo(16, y);
        g.lineTo(w - 16, y);
        g.stroke();
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('old memory c', 16, y - 12);
        g.textAlign = 'right';
        g.fillText('new memory c', w - 16, y - 12);
        const xf = w * 0.32,
          xi = w * 0.62;
        [
          [xf, '×'],
          [xi, '+'],
        ].forEach(([x, s]) => {
          g.fillStyle = t.bg;
          g.strokeStyle = t.ink;
          g.lineWidth = 1.5;
          g.beginPath();
          g.arc(x, y, 12, 0, 7);
          g.fill();
          g.stroke();
          g.fillStyle = t.ink;
          g.textAlign = 'center';
          g.font = SANS;
          g.fillText(s, x, y + 5);
          g.font = MONO;
          g.strokeStyle = t.ink;
          g.beginPath();
          g.moveTo(x, y + 12);
          g.lineTo(x, h * 0.66 - 14);
          g.stroke();
        });
        bx(xf, h * 0.66, 'forget ' + f2(S.f));
        bx(xi, h * 0.66, 'input × new', t.ok);
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('how much to keep', xf, h * 0.66 + 30);
        g.fillText('whether to write', xi, h * 0.66 + 30);
        g.fillText('current word and previous output feed every gate', w / 2, h - 8);
        if (pb && pb.on) {
          const p = (fr / 80) % 1;
          g.fillStyle = t.gold;
          g.beginPath();
          g.arc(16 + (w - 32) * p, y, 6, 0, 7);
          g.fill();
        }
      }
      live.textContent = msg;
      uCur(root, 'ls', cu);
      setR(root, 'r', f2(H[T2]) + ', ' + verdict(H[T2]));
      setR(root, 'l', f2(C[T2]) + ', ' + verdict(C[T2]));
      setR(root, 'rg', sci(Math.abs(GR[2])));
      setR(root, 'lg', sci(GL[2]));
    };
    comp();
    k = T2;
    const anim = Anim(() => {
      fr++;
      if (fr - t0 > (k < 3 ? 90 : 48) && k < T2) step();
      draw();
      if (k >= T2 && fr - t0 > 30) pb.stop();
    });
    pb = playBtn($('#ls-play', root), anim);
    runCtl(pb, root, 'ls', {
      done: () => k >= T2,
      restart: () => {
        k = 0;
      },
    });
    $('#ls-step', root).onclick = () => {
      pb.stop();
      step();
      t0 = fr - 100;
      draw();
    };
    $('#ls-reset', root).onclick = () => {
      pb.stop();
      k = 0;
      cu = 0;
      eq.innerHTML = '';
      msg = 'Press Read the sentence or One word.';
      draw();
    };
    const re = () => {
      pb.stop();
      comp();
      k = T2;
      msg =
        'With these settings the plain network ends at ' +
        f2(H[T2]) +
        ' (' +
        verdict(H[T2]) +
        ') and the LSTM at ' +
        f2(C[T2]) +
        ' (' +
        verdict(C[T2]) +
        ').';
      draw();
    };
    bind(root, 'ls-f', (v) => {
      S.f = v;
      re();
    });
    bind(root, 'ls-i', (v) => {
      S.i = v;
      re();
    });
    bind(root, 'ls-w', (v) => {
      S.w = v;
      re();
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        comp();
        k = 0;
        draw();
        pb.start();
      },
    };
  },
};
