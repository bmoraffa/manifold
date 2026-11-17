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
  playBtn,
  plot,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { ACT, ACTN, sci, sigFlow } from '../shared.js';
import { segBind, highlightLessonStep } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#af-c', root),
      cr = $('#af-r', root),
      live = $('#af-live', root),
      settings = { act: 'sigmoid', d: 12, w: 1 };
    let signalFlow = null,
      shown = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      probePhase = 0;
    const comp = () => {
      signalFlow = sigFlow(settings.act, settings.d, settings.w * settings.w, 11);
      shown = 0;
      activeStepIndex = 0;
      msg =
        'A stack of ' +
        settings.d +
        ' layers with ' +
        ACTN[settings.act] +
        ' is ready, and data has been sent forward through it. Press Send the signal back or One layer.';
    };
    const relativeGradientRms = (l) =>
      signalFlow.gradientRms[l] / signalFlow.gradientRms[settings.d - 1];
    const step = () => {
      if (shown >= settings.d) {
        shown = 0;
        activeStepIndex = 0;
        return;
      }
      shown++;
      const l = settings.d - shown;
      activeStepIndex = shown === 1 ? 1 : shown >= settings.d ? 4 : shown === 2 ? 2 : 3;
      msg =
        shown === 1
          ? 'The signal starts at layer ' + settings.d + ', the last one, with strength 1.'
          : 'Layer ' +
            (l + 1) +
            ': going back one layer multiplied the signal by ' +
            sci(signalFlow.gradientRms[l] / signalFlow.gradientRms[l + 1]) +
            '. It is now ' +
            sci(relativeGradientRms(l)) +
            ' of what it was at the last layer.';
      if (shown >= settings.d)
        msg +=
          ' That is what reaches the first layer after ' +
          settings.d +
          ' layers: ' +
          (relativeGradientRms(0) < 0.01
            ? 'almost nothing. The first layers of this network would learn extremely slowly.'
            : relativeGradientRms(0) > 100
              ? 'far too much. This network’s first updates would be wildly large.'
              : 'a healthy share. Every layer of this network can learn.');
    };
    const draw = () => {
      const colors = readThemeColors(),
        f = ACT[settings.act][0],
        df = ACT[settings.act][1],
        z = 4.6 * Math.sin(probePhase);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = plot(
            ctx,
            w,
            h,
            [-5, 5],
            [-1.3, settings.act === 'relu' || settings.act === 'leaky' ? 3.2 : 1.3],
            colors,
            {
              xstep: 1,
              yticks: [-1, 0, 1],
              xlabel: 'total input z',
            },
          );
        clipPlot(ctx, P);
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(P.X(0), P.Y(-2));
        ctx.lineTo(P.X(0), P.Y(4));
        ctx.stroke();
        curve(ctx, P, df, -5, 5, 400);
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 2.2;
        ctx.stroke();
        curve(ctx, P, f, -5, 5, 400);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 3;
        ctx.stroke();
        const s = df(z);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(P.X(z - 1.1), P.Y(f(z) - 1.1 * s));
        ctx.lineTo(P.X(z + 1.1), P.Y(f(z) + 1.1 * s));
        ctx.stroke();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(P.X(z), P.Y(f(z)));
        ctx.lineTo(P.X(z), P.Y(s));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colors.accent;
        ctx.beginPath();
        ctx.arc(P.X(z), P.Y(f(z)), 6.5, 0, 7);
        ctx.fill();
        ctx.fillStyle = colors.ok;
        ctx.beginPath();
        ctx.arc(P.X(z), P.Y(s), 5, 0, 7);
        ctx.fill();
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.accent;
        ctx.fillText('green: ' + ACTN[settings.act] + '(z)', P.m.l + 8, P.m.t + 8);
        ctx.fillStyle = colors.ok;
        ctx.fillText('red: its slope = share of the signal let through', P.m.l + 8, P.m.t + 22);
        ctx.fillStyle = colors.cut;
        ctx.fillText('gold: the tangent at the dot', P.m.l + 8, P.m.t + 36);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          m = { l: 54, r: 12, t: 20, b: 36 },
          D = settings.d,
          bw = (w - m.l - m.r) / D;
        let lo = -6,
          hi = 1;
        for (let l = 0; l < D; l++) {
          const v = Math.log10(relativeGradientRms(l) + 1e-300);
          lo = Math.min(lo, Math.floor(v));
          hi = Math.max(hi, Math.ceil(v));
        }
        lo = Math.max(lo, -14);
        const Y = (v) =>
          h - m.b - ((clamp(Math.log10(v + 1e-300), lo, hi) - lo) / (hi - lo)) * (h - m.t - m.b);
        ctx.font = MONO;
        ctx.textAlign = 'right';
        for (let e = lo; e <= hi; e += Math.max(1, Math.round((hi - lo) / 7))) {
          ctx.strokeStyle = e === 0 ? colors.muted : colors.line;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(m.l, Y(10 ** e));
          ctx.lineTo(w - m.r, Y(10 ** e));
          ctx.stroke();
          ctx.fillStyle = colors.muted;
          ctx.fillText(e === 0 ? '1' : '10^' + e, m.l - 5, Y(10 ** e) + 4);
        }
        for (let l = 0; l < D; l++) {
          const on = l >= D - shown,
            x = m.l + l * bw + bw * 0.14,
            v = relativeGradientRms(l),
            y1 = Y(1),
            y2 = Y(v);
          ctx.fillStyle = on
            ? l === D - shown
              ? colors.gold
              : v < 0.01
                ? colors.ok
                : colors.accent
            : mix(colors.ink, colors.ink, 0, 0.07);
          if (on) ctx.fillRect(x, Math.min(y1, y2), bw * 0.72, Math.max(3, Math.abs(y2 - y1)));
          else ctx.fillRect(x, y1 - 2, bw * 0.72, 4);
          if (D <= 16 || l % 2 === 0) {
            ctx.fillStyle = colors.muted;
            ctx.textAlign = 'center';
            ctx.fillText(String(l + 1), x + bw * 0.36, h - m.b + 14);
          }
        }
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('first layer', m.l, h - 5);
        ctx.textAlign = 'right';
        ctx.fillText('last layer: the signal starts here ←', w - m.r, h - 5);
        ctx.textAlign = 'left';
        ctx.fillText('strength compared with the start (each line is ×10)', m.l, 12);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'af', activeStepIndex);
      setReadout(root, 'f', f3(f(z)));
      setReadout(root, 's', f3(df(z)));
      setReadout(root, 'mx', settings.act === 'sigmoid' ? '0.25' : '1');
      setReadout(
        root,
        'g1',
        shown >= settings.d
          ? sci(relativeGradientRms(0)) + ' of the start'
          : 'send the signal back to see',
      );
      setReadout(
        root,
        'pl',
        '× ' + sci(Math.pow(relativeGradientRms(0), 1 / Math.max(1, settings.d - 1))),
      );
    };
    comp();
    shown = settings.d;
    const anim = Anim(() => {
      frameCount++;
      probePhase += 0.02;
      if (frameCount % 34 === 0 && shown < settings.d) step();
      draw();
      if (shown >= settings.d && frameCount % 34 === 0) playback.stop();
    });
    playback = playBtn($('#af-play', root), anim);
    attachRunLimit(playback, root, 'af', {
      done: () => shown >= settings.d,
      restart: () => {
        shown = 0;
      },
    });
    $('#af-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#af-reset', root).onclick = () => {
      playback.stop();
      comp();
      draw();
    };
    segBind(root, 'af-act', settings.act, (v) => {
      settings.act = v;
      playback.stop();
      comp();
      draw();
      playback.start();
    });
    bind(
      root,
      'af-d',
      (v) => {
        settings.d = v;
        playback.stop();
        comp();
        shown = settings.d;
        draw();
      },
      (v) => String(v),
    );
    bind(root, 'af-w', (v) => {
      settings.w = v;
      playback.stop();
      comp();
      shown = settings.d;
      draw();
    });
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
