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
import { ACT, sci } from '../../neural-nets/shared.js';
import { highlightLessonStep } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#rs-c', root),
      cr = $('#rs-r', root),
      cd = $('#rs-d', root),
      live = $('#rs-live', root),
      settings = { d: 20, v: 0.8, act: 'tanh' };
    let plainGradientProfile = [],
      residualGradientProfile = [],
      k = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      t0 = 0,
      playback;
    const comp = () => {
      const n = 30,
        bt = 30,
        D = settings.d,
        r = randnFrom(rng(5)),
        f = ACT[settings.act][0],
        df = ACT[settings.act][1],
        W = [],
        sdv = settings.v / Math.sqrt(n),
        bta = 0.5;
      for (let l = 0; l < D; l++)
        W.push(Array.from({ length: n }, () => Float64Array.from({ length: n }, () => sdv * r())));
      const computeGradientProfile = (res) => {
        let A = [];
        for (let b = 0; b < bt; b++) A.push(Float64Array.from({ length: n }, () => r()));
        const Zs = [];
        for (let l = 0; l < D; l++) {
          const Z = [],
            NA = [];
          for (let b = 0; b < bt; b++) {
            const z = new Float64Array(n),
              a = new Float64Array(n);
            for (let j = 0; j < n; j++) {
              let s = 0;
              for (let i = 0; i < n; i++) s += W[l][j][i] * A[b][i];
              z[j] = s;
              a[j] = res ? A[b][j] + bta * f(s) : f(s);
            }
            Z.push(z);
            NA.push(a);
          }
          Zs.push(Z);
          A = NA;
        }
        let Dl = [];
        for (let b = 0; b < bt; b++) Dl.push(Float64Array.from({ length: n }, () => r()));
        const G = new Array(D + 1).fill(0),
          rms = (Q) => {
            let s = 0;
            for (const q of Q) for (const u of q) s += u * u;
            return Math.sqrt(s / (n * bt));
          };
        G[D] = rms(Dl);
        for (let l = D - 1; l >= 0; l--) {
          const ND = [];
          for (let b = 0; b < bt; b++) {
            const da = new Float64Array(n);
            for (let i = 0; i < n; i++) {
              let q = 0;
              for (let j = 0; j < n; j++) q += W[l][j][i] * Dl[b][j] * df(Zs[l][b][j]);
              da[i] = res ? Dl[b][i] + bta * q : q;
            }
            ND.push(da);
          }
          Dl = ND;
          G[l] = rms(Dl);
        }
        return G.map((q) => q / G[D]);
      };
      plainGradientProfile = computeGradientProfile(false);
      residualGradientProfile = computeGradientProfile(true);
      k = 0;
      activeStepIndex = 0;
      msg =
        'Two stacks of ' +
        settings.d +
        ' layers are ready, built from the same weights. Press Send the signal back or One layer.';
    };
    const step = () => {
      if (k >= settings.d) {
        k = 0;
        activeStepIndex = 0;
        return;
      }
      k++;
      t0 = frameCount;
      const l = settings.d - k;
      activeStepIndex = k === 1 ? 2 : k >= settings.d ? 4 : 3;
      msg =
        'Back to layer ' +
        (l + 1) +
        '. Plain: the signal was multiplied by ' +
        sci(plainGradientProfile[l] / plainGradientProfile[l + 1]) +
        ' and is now ' +
        sci(plainGradientProfile[l]) +
        ' of its starting strength. With shortcuts: multiplied by ' +
        sci(residualGradientProfile[l] / residualGradientProfile[l + 1]) +
        ', now ' +
        sci(residualGradientProfile[l]) +
        '.' +
        (k >= settings.d
          ? ' At the first layer the plain stack has ' +
            sci(plainGradientProfile[0]) +
            ' left and the stack with shortcuts has ' +
            sci(residualGradientProfile[0]) +
            '.'
          : '');
    };
    const draw = () => {
      const colors = readThemeColors(),
        D = settings.d,
        u = prefersReducedMotion ? 1 : clamp((frameCount - t0) / 22, 0, 1);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          m = 30,
          bw = (w - 2 * m) / D,
          rowY = [h * 0.27, h * 0.72],
          rad = (v) => clamp(13 + 3.4 * Math.log10(v + 1e-300), 1.5, 24);
        ctx.font = MONO;
        [
          [plainGradientProfile, 'plain stack', colors.ok, false],
          [residualGradientProfile, 'with shortcuts', colors.accent, true],
        ].forEach(([G, lab, cl, res], ri) => {
          const y = rowY[ri];
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(lab, m, y - h * 0.2);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(m, y);
          ctx.lineTo(w - m, y);
          ctx.stroke();
          for (let l = 0; l < D; l++) {
            const x = m + l * bw + bw * 0.2,
              ww = bw * 0.6;
            ctx.fillStyle = colors.bg;
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.rect(x, y - 11, ww, 22);
            ctx.fill();
            ctx.stroke();
            if (res) {
              ctx.strokeStyle = mix(cl, cl, 0, 0.8);
              ctx.lineWidth = 1.4;
              ctx.beginPath();
              ctx.moveTo(m + l * bw + bw * 0.06, y);
              ctx.quadraticCurveTo(
                m + l * bw + bw * 0.5,
                y - h * 0.17,
                m + (l + 1) * bw - bw * 0.06,
                y,
              );
              ctx.stroke();
            }
            if (D <= 24 || l % 5 === 0) {
              ctx.fillStyle = colors.muted;
              ctx.textAlign = 'center';
              ctx.fillText(String(l + 1), x + ww / 2, y + 28);
            }
          }
          if (k > 0) {
            const pos = D - k + (1 - u),
              li = Math.min(D, Math.round(pos)),
              v = G[D - k] * u + G[Math.min(D, D - k + 1)] * (1 - u),
              x = m + pos * bw;
            ctx.fillStyle = cl;
            ctx.beginPath();
            ctx.arc(x, y, rad(v), 0, 7);
            ctx.fill();
            ctx.fillStyle = colors.ink;
            ctx.textAlign = 'center';
            ctx.fillText(sci(G[D - k]), clamp(x, 60, w - 60), y + h * 0.19);
          } else {
            ctx.fillStyle = cl;
            ctx.beginPath();
            ctx.arc(w - m, y, rad(1), 0, 7);
            ctx.fill();
          }
        });
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('← the learning signal travels this way', w - m, h - 6);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          m = { l: 54, r: 10, t: 16, b: 22 },
          bw = (w - m.l - m.r) / D;
        let lo = -3,
          hi = 1;
        [plainGradientProfile, residualGradientProfile].forEach((G) =>
          G.forEach((v) => {
            const q = Math.log10(v + 1e-300);
            lo = Math.min(lo, Math.floor(q));
            hi = Math.max(hi, Math.ceil(q));
          }),
        );
        lo = Math.max(lo, -16);
        hi = Math.min(hi, 12);
        const Y = (v) =>
          h - m.b - ((clamp(Math.log10(v + 1e-300), lo, hi) - lo) / (hi - lo)) * (h - m.t - m.b);
        ctx.font = MONO;
        for (let e = lo; e <= hi; e += Math.max(1, Math.round((hi - lo) / 6))) {
          ctx.strokeStyle = e === 0 ? colors.muted : colors.line;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(m.l, Y(10 ** e));
          ctx.lineTo(w - m.r, Y(10 ** e));
          ctx.stroke();
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText(e === 0 ? '1' : '10^' + e, m.l - 5, Y(10 ** e) + 4);
        }
        for (let l = 0; l < D; l++) {
          const on = l >= D - k,
            x = m.l + l * bw;
          [
            [plainGradientProfile, colors.ok, 0.08],
            [residualGradientProfile, colors.accent, 0.5],
          ].forEach(([G, cl, o]) => {
            const y1 = Y(1),
              y2 = on ? Y(G[l]) : y1;
            ctx.fillStyle = on ? cl : mix(colors.ink, colors.ink, 0, 0.07);
            ctx.fillRect(
              x + bw * o,
              Math.min(y1, y2) - (on ? 0 : 1),
              bw * 0.4,
              Math.max(2, Math.abs(y2 - y1)),
            );
          });
        }
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.ok;
        ctx.fillText('red: plain', m.l + 6, m.t + 2);
        ctx.fillStyle = colors.accent;
        ctx.fillText('green: with shortcuts', m.l + 90, m.t + 2);
        ctx.fillStyle = colors.muted;
        ctx.fillText('first layer', m.l, h - 5);
        ctx.textAlign = 'right';
        ctx.fillText('last layer', w - m.r, h - 5);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cd),
          bx = (x, y, tx) => {
            ctx.fillStyle = colors.bg;
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.rect(x - 58, y - 15, 116, 30);
            ctx.fill();
            ctx.stroke();
            ctx.font = MONO;
            ctx.fillStyle = colors.ink;
            ctx.textAlign = 'center';
            ctx.fillText(tx, x, y + 4);
          },
          ar = (x1, y1, x2, y2) => {
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          },
          ph = (frameCount / 70) % 1;
        ctx.font = MONO;
        const xa = w * 0.25,
          xb = w * 0.72;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        ctx.fillText('plain', xa, 14);
        ctx.fillText('with a shortcut', xb, 14);
        ar(xa, 28, xa, h - 24);
        bx(xa, h * 0.5, 'weights, activation');
        ctx.fillStyle = colors.ink;
        ctx.fillText('input a', xa, 28);
        ctx.fillText('output F(a)', xa, h - 10);
        ar(xb, 28, xb, h - 24);
        bx(xb, h * 0.45, 'weights, activation');
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xb, 40);
        ctx.bezierCurveTo(xb + 120, 40, xb + 120, h * 0.74, xb + 10, h * 0.74);
        ctx.stroke();
        ctx.fillStyle = colors.bg;
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(xb, h * 0.74, 10, 0, 7);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = colors.ink;
        ctx.fillText('+', xb, h * 0.74 + 4);
        ctx.fillText('input a', xb, 28);
        ctx.fillText('output a + F(a)', xb, h - 10);
        ctx.fillStyle = colors.accent;
        ctx.textAlign = 'left';
        ctx.fillText('shortcut: a, unchanged', Math.min(w - 150, xb + 62), h * 0.3);
        if (playback && playback.on) {
          ctx.fillStyle = colors.gold;
          [xa, xb].forEach((x) => {
            ctx.beginPath();
            ctx.arc(x, h - 24 - (h - 52) * ph, 5, 0, 7);
            ctx.fill();
          });
          const q = 1 - ph,
            bz = (a, b2, cc, d) =>
              q ** 3 * a + 3 * q * q * ph * b2 + 3 * q * ph * ph * cc + ph ** 3 * d;
          ctx.fillStyle = colors.accent;
          ctx.beginPath();
          ctx.arc(bz(xb + 10, xb + 120, xb + 120, xb), bz(h * 0.74, h * 0.74, 40, 40), 5, 0, 7);
          ctx.fill();
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'rs', activeStepIndex);
      const fin = k >= D;
      setReadout(root, 'pl', fin ? sci(plainGradientProfile[0]) : '–');
      setReadout(root, 'rs', fin ? sci(residualGradientProfile[0]) : '–');
      setReadout(root, 'pp', '× ' + sci(Math.pow(plainGradientProfile[0], 1 / D)));
      setReadout(root, 'rp', '× ' + sci(Math.pow(residualGradientProfile[0], 1 / D)));
    };
    comp();
    k = settings.d;
    const anim = Anim(() => {
      frameCount++;
      if (frameCount - t0 >= (settings.d > 30 ? 10 : 24) && k < settings.d) step();
      draw();
      if (k >= settings.d && frameCount - t0 > 26) playback.stop();
    });
    playback = playBtn($('#rs-play', root), anim);
    attachRunLimit(playback, root, 'rs', {
      done: () => k >= settings.d,
      restart: () => {
        k = 0;
      },
    });
    $('#rs-step', root).onclick = () => {
      playback.stop();
      step();
      t0 = frameCount - 100;
      draw();
    };
    $('#rs-reset', root).onclick = () => {
      playback.stop();
      comp();
      draw();
    };
    const redo = () => {
      playback.stop();
      comp();
      draw();
      playback.start();
    };
    bind(
      root,
      'rs-depth',
      (v) => {
        settings.d = v;
        playback.stop();
        comp();
        k = settings.d;
        draw();
      },
      (v) => String(v),
    );
    bind(root, 'rs-v', (v) => {
      settings.v = v;
      playback.stop();
      comp();
      k = settings.d;
      draw();
    });
    $('#rs-act', root).onchange = (e) => {
      settings.act = e.target.value;
      redo();
    };
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        comp();
        draw();
        playback.start();
      },
    };
  },
};
