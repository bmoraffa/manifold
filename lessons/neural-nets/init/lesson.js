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
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { sci, sigFlow } from '../shared.js';
import { highlightLessonStep } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#wi-c', root),
      cf = $('#wi-f', root),
      cb = $('#wi-b', root),
      live = $('#wi-live', root),
      settings = { act: 'tanh', v: 1, d: 10 },
      varianceMultipliers = { small: 0.2, xavier: 1, he: 2, large: 6 };
    let signalFlow = null,
      playbackStep = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback;
    const comp = () => {
      signalFlow = sigFlow(settings.act, settings.d, settings.v, 7);
      playbackStep = 0;
      activeStepIndex = 0;
      msg =
        'Weights drawn at random with variance ' +
        f2(settings.v) +
        ' ÷ (number of inputs). Nothing has been sent through yet. Press Send the signal through or One layer.';
    };
    const end = () => 2 * settings.d,
      rel = (l) => signalFlow.gradientRms[l] / signalFlow.gradientRms[settings.d - 1];
    const verdict = () => {
      const a = signalFlow.sd[settings.d - 1],
        b = rel(0),
        sat = settings.act !== 'relu' && a > 0.85 * (settings.act === 'sigmoid' ? 0.5 : 1);
      return a < 0.02
        ? 'The signal has faded away'
        : a > 30
          ? 'The signal has exploded'
          : sat
            ? 'Neurons are jammed at their limits (saturated)'
            : b < 0.01
              ? 'Outputs look fine but the learning signal dies'
              : b > 100
                ? 'The learning signal explodes'
                : 'Healthy: the signal survives in both directions';
    };
    const step = () => {
      if (playbackStep >= end()) {
        playbackStep = 0;
        activeStepIndex = 0;
        msg = 'Start again.';
        return;
      }
      playbackStep++;
      if (playbackStep <= settings.d) {
        const l = playbackStep - 1;
        activeStepIndex = playbackStep === 1 ? 1 : 2;
        msg =
          'Forward, layer ' +
          playbackStep +
          ': the outputs have spread ' +
          sci(signalFlow.sd[l]) +
          (l
            ? ' (the layer before had ' +
              sci(signalFlow.sd[l - 1]) +
              ', so this layer multiplied it by ' +
              f2(signalFlow.sd[l] / signalFlow.sd[l - 1]) +
              ')'
            : '') +
          '.';
      } else {
        const l = 2 * settings.d - playbackStep;
        activeStepIndex = playbackStep >= end() ? 4 : 3;
        msg =
          'Backward, layer ' +
          (l + 1) +
          ': the learning signal is ' +
          sci(rel(l)) +
          ' of its strength at the last layer.';
        if (playbackStep >= end()) msg += ' Done. Verdict: ' + verdict() + '.';
      }
    };
    const draw = () => {
      const colors = readThemeColors(),
        D = settings.d,
        nf = Math.min(playbackStep, D),
        nbk = Math.max(0, playbackStep - D);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          cw = w / D,
          bounded = settings.act !== 'relu',
          lo = settings.act === 'sigmoid' ? 0 : bounded ? -1 : 0;
        ctx.font = MONO;
        for (let l = 0; l < D; l++) {
          const x0 = l * cw + 6,
            ww = cw - 12,
            y1 = h - 30,
            on = l < nf;
          ctx.strokeStyle = colors.line;
          ctx.lineWidth = 1;
          ctx.strokeRect(x0, 14, ww, y1 - 14);
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText('layer ' + (l + 1), x0 + ww / 2, h - 14);
          if (!on) continue;
          const a = signalFlow.acts[l],
            nb = 22;
          let hi = 1;
          if (!bounded) {
            hi = 1e-9;
            for (const v of a) hi = Math.max(hi, v);
            hi = Math.max(hi, 1e-6);
          }
          const cnt = new Array(nb).fill(0);
          for (const v of a) cnt[clamp(Math.floor(((v - lo) / (hi - lo)) * nb), 0, nb - 1)]++;
          const cm = Math.max(...cnt);
          ctx.fillStyle = l === nf - 1 && playbackStep <= D ? colors.gold : colors.accent;
          for (let q = 0; q < nb; q++) {
            const bh = (cnt[q] / cm) * (y1 - 24);
            ctx.fillRect(x0 + (q * ww) / nb, y1 - bh, ww / nb - 0.6, bh);
          }
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(bounded ? String(lo) : '0', x0, h - 2);
          ctx.textAlign = 'right';
          ctx.fillText(bounded ? '1' : sci(hi), x0 + ww, h - 2);
        }
      }
      const bars = (cv, vals, n, from, ttl, ref) => {
        const { g: ctx, w, h } = prepareCanvas(cv),
          m = { l: 54, r: 10, t: 20, b: 24 },
          bw = (w - m.l - m.r) / D;
        let lo = -3,
          hi = 1;
        vals.forEach((v) => {
          const q = Math.log10(v + 1e-300);
          lo = Math.min(lo, Math.floor(q));
          hi = Math.max(hi, Math.ceil(q));
        });
        lo = Math.max(lo, -14);
        hi = Math.min(hi, 14);
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
          const on = from === 'L' ? l < n : l >= D - n,
            x = m.l + l * bw + bw * 0.15;
          ctx.fillStyle = on
            ? ref
              ? colors.ok
              : colors.accent
            : mix(colors.ink, colors.ink, 0, 0.07);
          const y1 = Y(1),
            y2 = on ? Y(vals[l]) : y1;
          ctx.fillRect(
            x,
            Math.min(y1, y2) - (on ? 0 : 2),
            bw * 0.7,
            Math.max(4, Math.abs(y2 - y1)),
          );
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText(String(l + 1), x + bw * 0.35, h - m.b + 14);
        }
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(ttl, m.l, 12);
      };
      bars(cf, signalFlow.sd, nf, 'L', 'spread of the outputs (each line is ×10) →');
      bars(
        cb,
        signalFlow.gradientRms.map((_, l) => rel(l)),
        nbk,
        'R',
        '← learning signal, relative to the last layer',
        true,
      );
      live.textContent = msg;
      highlightLessonStep(root, 'wi', activeStepIndex);
      setReadout(root, 'v', f2(settings.v));
      setReadout(root, 's1', sci(signalFlow.sd[0]));
      setReadout(root, 'sL', nf >= D ? sci(signalFlow.sd[D - 1]) : '–');
      setReadout(root, 'g', playbackStep >= end() ? sci(rel(0)) : '–');
      setReadout(root, 'vd', playbackStep >= end() ? verdict() : '–');
      root
        .querySelectorAll('#wi-mode button')
        .forEach((b) =>
          b.setAttribute(
            'aria-pressed',
            Math.abs(varianceMultipliers[b.dataset.v] - settings.v) < 0.01 ? 'true' : 'false',
          ),
        );
    };
    comp();
    playbackStep = end();
    const anim = Anim(() => {
      frameCount++;
      if (frameCount % 26) return;
      step();
      draw();
      if (playbackStep >= end()) playback.stop();
    });
    playback = playBtn($('#wi-play', root), anim);
    attachRunLimit(playback, root, 'wi', {
      done: () => playbackStep >= end(),
      restart: () => {
        playbackStep = 0;
      },
    });
    $('#wi-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#wi-reset', root).onclick = () => {
      playback.stop();
      comp();
      draw();
    };
    const vb = bind(root, 'wi-v', (v) => {
      settings.v = v;
      playback.stop();
      comp();
      playbackStep = end();
      draw();
    });
    bind(
      root,
      'wi-d',
      (v) => {
        settings.d = v;
        playback.stop();
        comp();
        playbackStep = end();
        draw();
      },
      (v) => String(v),
    );
    $('#wi-act', root).onchange = (e) => {
      settings.act = e.target.value;
      playback.stop();
      comp();
      draw();
      playback.start();
    };
    $('#wi-mode', root).addEventListener('click', (e) => {
      const v = e.target.dataset && e.target.dataset.v;
      if (!v) return;
      settings.v = varianceMultipliers[v];
      vb.set(settings.v);
      playback.stop();
      comp();
      draw();
      playback.start();
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
