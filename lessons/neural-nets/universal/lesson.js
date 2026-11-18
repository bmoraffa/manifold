/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  clipPlot,
  f2,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  plot,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { ACT, NNet, sci } from '../shared.js';
import { UC, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#ua-c', root),
      cr = $('#ua-r', root),
      live = $('#ua-live', root),
      st = $('#ua-st', root),
      settings = { H: 5, act: 'tanh', tg: 'wave' },
      NP = 100,
      targetFunctions = {
        wave: (x) => 0.8 * Math.sin(4.2 * x) + 0.3 * x,
        hill: (x) => 1.3 * Math.exp(-7 * x * x) - 0.5,
        step: (x) => (x < -0.5 ? -0.8 : x < 0 ? -0.1 : x < 0.5 ? 0.7 : 0.1),
        sky: (x) =>
          [0.2, 0.9, 0.4, -0.3, 0.6, -0.6, 0.1, 0.8, -0.1, 0.5][
            Math.min(9, Math.floor((x + 1) * 5))
          ],
      };
    let targetSamples = new Float64Array(NP),
      net,
      ep = 0,
      LL = [],
      conv = false,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      grow = false,
      bestLossByWidth = {},
      painting = false,
      lastI = -1;
    const xi = (i) => -1 + (2 * i) / (NP - 1),
      setT = () => {
        if (settings.tg !== 'own')
          for (let i = 0; i < NP; i++) targetSamples[i] = targetFunctions[settings.tg](xi(i));
      };
    const build = () => {
      net = NNet([1, settings.H, 1], {
        act: settings.act,
        out: 'lin',
        seed: run * 31 + settings.H * 7,
        scale: settings.act === 'relu' ? 1.2 : 2.2,
      });
      for (let j = 0; j < settings.H; j++) net.B[0][j] = (Math.random() - 0.5) * 3;
      ep = 0;
      LL = [];
      conv = false;
      activeStepIndex = 0;
      msg =
        settings.H +
        ' neuron' +
        (settings.H === 1 ? '' : 's') +
        ' with random settings: the pieces are scattered and their sum looks nothing like the target. Press Train.';
    };
    const X = Array.from({ length: NP }, (_, i) => [xi(i) * 3]);
    const train = (k) => {
      for (let q = 0; q < k && !conv; q++) {
        const l = 2 * net.train(X, targetSamples, 0.02).loss;
        ep++;
        LL.push(l);
        if (
          (ep > 300 && Math.abs(LL[ep - 1] - LL[ep - 201]) < 0.01 * LL[ep - 1] + 1e-7) ||
          ep >= 6000
        )
          conv = true;
      }
      const l = LL[ep - 1];
      bestLossByWidth[settings.H] = Math.min(
        bestLossByWidth[settings.H] == null ? 1e9 : bestLossByWidth[settings.H],
        l,
      );
      activeStepIndex = conv ? 4 : 3;
      msg =
        'Step ' +
        ep +
        ': the ' +
        settings.H +
        ' piece' +
        (settings.H === 1 ? '' : 's') +
        ' shifted a little. Mean squared error ' +
        sci(l) +
        '.' +
        (conv
          ? (ep >= 6000
              ? ' Reached the limit of 6000 steps.'
              : ' Converged: the error has stopped falling with ' +
                settings.H +
                ' neuron' +
                (settings.H === 1 ? '' : 's') +
                '.') +
            (l > 0.004
              ? ' The fit is still rough: there are not enough pieces for this curve.'
              : ' The gold line follows the target closely.')
          : '');
    };
    const draw = () => {
      const colors = readThemeColors(),
        col = UC(colors);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = plot(ctx, w, h, [-1, 1], [-1.6, 1.6], colors, {
            xstep: 0.5,
            yticks: [-1, 0, 1],
            xlabel: 'x',
          });
        clipPlot(ctx, P);
        const f = ACT[settings.act][0];
        for (let j = 0; j < settings.H; j++) {
          ctx.strokeStyle = mix(col[j % 8], col[j % 8], 0, 0.75);
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          for (let i = 0; i < NP; i++) {
            const v = net.W[1][0][j] * f(net.W[0][j][0] * xi(i) * 3 + net.B[0][j]);
            i ? ctx.lineTo(P.X(xi(i)), P.Y(v)) : ctx.moveTo(P.X(xi(i)), P.Y(v));
          }
          ctx.stroke();
        }
        ctx.setLineDash([7, 5]);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < NP; i++)
          i
            ? ctx.lineTo(P.X(xi(i)), P.Y(targetSamples[i]))
            : ctx.moveTo(P.X(xi(i)), P.Y(targetSamples[i]));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 3.4;
        ctx.beginPath();
        for (let i = 0; i < NP; i++) {
          const v = net.predictScalar(X[i]);
          i ? ctx.lineTo(P.X(xi(i)), P.Y(v)) : ctx.moveTo(P.X(xi(i)), P.Y(v));
        }
        ctx.stroke();
        {
          const x = tp.v[0],
            v = net.predictScalar([x * 3]);
          markTP(ctx, P.X(x), P.Y(v), colors);
        }
        ctx.restore();
        mainCanvas._P = P;
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5;
        const lo = LL.length ? Math.min(...LL) : 0;
        drawSeriesChart(
          ctx,
          colors,
          0,
          0,
          w,
          h1,
          LL.map((v) => Math.log10(v + 1e-9)),
          {
            title: 'error by training step (log scale): 10^…',
            n: Math.max(100, LL.length),
            y1: Math.min(-3, Math.floor(Math.log10(lo + 1e-9))),
            ym: 0.5,
            col: colors.ok,
          },
        );
        const m = { l: 44, r: 12, t: 22, b: 22 },
          bw = (w - m.l - m.r) / 24,
          Y = (v) =>
            h - m.b - ((clamp(Math.log10(v + 1e-9), -4, 0) + 4) / 4) * (h - h1 - m.t - m.b);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('best error reached with each number of neurons', 8, h1 + 13);
        ctx.textAlign = 'right';
        ctx.fillText('1', m.l - 5, Y(1) + 4);
        ctx.fillText('10^-4', m.l - 5, Y(1e-4) + 4);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, h - m.b);
        ctx.lineTo(w - m.r, h - m.b);
        ctx.stroke();
        for (let k = 1; k <= 24; k++) {
          const x = m.l + (k - 1) * bw + bw * 0.15;
          if (bestLossByWidth[k] != null) {
            ctx.fillStyle = k === settings.H ? colors.gold : colors.accent;
            ctx.fillRect(x, Y(bestLossByWidth[k]), bw * 0.7, h - m.b - Y(bestLossByWidth[k]));
          }
          if (k % 2 === 1) {
            ctx.fillStyle = colors.muted;
            ctx.textAlign = 'center';
            ctx.fillText(String(k), x + bw * 0.35, h - m.b + 14);
          }
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'ua', activeStepIndex);
      const x = tp.v[0],
        gi = Math.round(((x + 1) / 2) * (NP - 1));
      tp.out.textContent =
        'At x = ' +
        f2(x) +
        ' the network outputs ' +
        f2(net.predictScalar([x * 3])) +
        ' and the target is ' +
        f2(targetSamples[gi]) +
        '.';
      setReadout(root, 'H', String(settings.H));
      setReadout(root, 'np', String(3 * settings.H + 1));
      setReadout(root, 'ep', String(ep));
      setReadout(root, 'L', LL.length ? sci(LL[ep - 1]) : '–');
      setReadout(
        root,
        'st',
        grow ? 'Adding neurons' : conv ? 'Converged' : ep ? 'Training' : 'Not started',
      );
    };
    const tp = testPoint(root, 'ua', ['Your x'], [0.3], [[-1, 1]]);
    tp.on = () => draw();
    setT();
    build();
    if (prefersReducedMotion) train(6000);
    const anim = Anim(() => {
      frameCount++;
      if (ep < 40 && frameCount % 4) return;
      train(Math.min(ep < 40 ? 2 : ep < 400 ? 4 : 10, checkRunStop.remaining()));
      if (conv && grow) {
        if (settings.H < 12 && LL[ep - 1] > 3e-4) {
          settings.H++;
          hb.set(settings.H);
          run++;
          checkRunStop.rebase(0);
          build();
          msg = 'Added a neuron: now ' + settings.H + '. Training again from a fresh random start.';
        } else {
          grow = false;
          msg += ' Stopped adding neurons at ' + settings.H + '.';
        }
      }
      draw();
      checkRunStop();
    });
    playback = playBtn($('#ua-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'ua', {
      done: () => conv && !grow,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'steps',
    });
    const redo = () => {
      playback.stop();
      grow = false;
      run++;
      build();
      draw();
    };
    $('#ua-step', root).onclick = () => {
      playback.stop();
      grow = false;
      if (conv) {
        run++;
        build();
      } else train(20);
      draw();
    };
    $('#ua-reset', root).onclick = redo;
    $('#ua-grow', root).onclick = () => {
      playback.stop();
      bestLossByWidth = {};
      settings.H = 1;
      hb.set(1);
      run++;
      build();
      grow = true;
      playback.start();
    };
    const hb = bind(
      root,
      'ua-H',
      (v) => {
        settings.H = v;
        redo();
      },
      (v) => String(v),
    );
    $('#ua-act', root).onchange = (e) => {
      settings.act = e.target.value;
      bestLossByWidth = {};
      redo();
    };
    $('#ua-tg', root).onchange = (e) => {
      settings.tg = e.target.value;
      bestLossByWidth = {};
      if (settings.tg === 'own') st.textContent = 'Drag across the left picture to draw the curve.';
      else st.textContent = '';
      setT();
      redo();
    };
    const paint = (e) => {
      const P = mainCanvas._P;
      if (!P) return;
      const r = mainCanvas.getBoundingClientRect(),
        x = clamp(((e.clientX - r.left - P.m.l) / (P.w - P.m.l - P.m.r)) * 2 - 1, -1, 1),
        y = clamp(
          -((e.clientY - r.top - (P.h - P.m.b)) / (P.h - P.m.t - P.m.b)) * 3.2 - 1.6,
          -1.5,
          1.5,
        ),
        i = Math.round(((x + 1) / 2) * (NP - 1));
      if (lastI >= 0 && lastI !== i) {
        const a = Math.min(i, lastI),
          b = Math.max(i, lastI),
          ya = targetSamples[lastI];
        for (let q = a; q <= b; q++)
          targetSamples[q] =
            ya + (y - ya) * (i === lastI ? 1 : Math.abs(q - lastI) / Math.abs(i - lastI));
      } else targetSamples[i] = y;
      lastI = i;
    };
    mainCanvas.style.touchAction = 'none';
    mainCanvas.addEventListener('pointerdown', (e) => {
      playback.stop();
      grow = false;
      painting = true;
      lastI = -1;
      settings.tg = 'own';
      $('#ua-tg', root).value = 'own';
      st.textContent = 'Your own curve. Press Train, or Add Neurons One at a Time.';
      mainCanvas.setPointerCapture(e.pointerId);
      paint(e);
      draw();
    });
    mainCanvas.addEventListener('pointermove', (e) => {
      if (!painting) return;
      paint(e);
      draw();
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      mainCanvas.addEventListener(n, () => {
        if (painting) {
          painting = false;
          bestLossByWidth = {};
          run++;
          build();
          draw();
        }
      }),
    );
    return {
      draw,
      stop: () => {
        playback.stop();
        grow = false;
      },
      auto: () => {
        run++;
        build();
        draw();
        playback.start();
      },
    };
  },
};
