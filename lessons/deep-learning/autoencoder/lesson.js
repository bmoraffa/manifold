/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  f3,
  prepareCanvas,
  letterData,
  markTP,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { NNet } from '../../neural-nets/shared.js';
import { UC, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';
import { pct } from '../../ensembles/shared.js';
import { sig } from '../../classification/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#ae-c', root),
      cr = $('#ae-r', root),
      live = $('#ae-live', root),
      settings = { hw: 14, flip: 0.04, col: true, seed: 1 };
    let X = [],
      lab = [],
      LT = [],
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
      selectedLatentPoint = [0.2, 0.2],
      drag = false,
      codes = [];
    const gen = () => {
        const d = letterData(settings.seed, settings.flip);
        X = d.X.map((x) => Float64Array.from(x, (v) => clamp(Math.round(v), 0, 1)));
        lab = d.lab;
        LT = d.CH;
      },
      build = () => {
        net = NNet([35, settings.hw, 2, settings.hw, 35], {
          act: 'tanh',
          out: 'sig',
          seed: run * 9 + settings.seed,
        });
        ep = 0;
        LL = [];
        conv = false;
        activeStepIndex = 0;
        updateLatentCodes();
        msg =
          'Random weights: every picture lands near the middle of the map and comes back as a grey blur. Press Train.';
      };
    const updateLatentCodes = () => {
        codes = X.map((x) => net.fwd(x).A[2]);
      },
      dec = (z) => {
        let a = z;
        for (let l = 2; l < 4; l++) {
          const m = net.sizes[l + 1],
            o = new Float64Array(m);
          for (let j = 0; j < m; j++) {
            let s = net.B[l][j];
            for (let i = 0; i < a.length; i++) s += net.W[l][j][i] * a[i];
            o[j] = l === 3 ? sig(s) : Math.tanh(s);
          }
          a = o;
        }
        return a;
      };
    const pixelAccuracy = () => {
      let ok = 0;
      X.forEach((x) => {
        const o = net.predictVector(x);
        for (let k = 0; k < 35; k++) if ((o[k] >= 0.5 ? 1 : 0) === x[k]) ok++;
      });
      return ok / (X.length * 35);
    };
    const train = (k) => {
      for (let i = 0; i < k && !conv; i++) {
        const l = net.train(X, X, 0.01).loss / 35;
        ep++;
        LL.push(l);
        if ((ep > 250 && Math.abs(LL[ep - 1] - LL[ep - 151]) < 0.012 * l) || ep >= 3000)
          conv = true;
      }
      updateLatentCodes();
      activeStepIndex = conv ? 4 : ep < 4 ? 1 : 3;
      msg =
        'Epoch ' +
        ep +
        ': all ' +
        X.length +
        ' pictures were squeezed to two numbers and rebuilt. Error per pixel ' +
        f3(LL[ep - 1]) +
        '; ' +
        pct(pixelAccuracy()) +
        ' of pixels come back right.' +
        (conv
          ? ep >= 3000
            ? ' Reached the limit of 3000 epochs.'
            : ' Converged: the error has stopped falling, so training stops.'
          : '');
    };
    const glyph = (g, v, x, y, cz, t, al = 1) => {
      for (let k = 0; k < 35; k++) {
        g.fillStyle = mix(t.bg, t.ink, clamp(v[k], 0, 1), al);
        g.fillRect(x + (k % 5) * cz, y + Math.floor(k / 5) * cz, cz - 0.4, cz - 0.4);
      }
    };
    const draw = () => {
      const colors = readThemeColors(),
        col = UC(colors);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          X2 = (v) => w / 2 + (v * (w / 2 - 14)) / 1.08,
          Y2 = (v) => h / 2 - (v * (h / 2 - 14)) / 1.08,
          NGd = 8;
        for (let a = 0; a < NGd; a++)
          for (let b = 0; b < NGd; b++) {
            const zx = -1 + (2 * (b + 0.5)) / NGd,
              zy = 1 - (2 * (a + 0.5)) / NGd;
            glyph(ctx, dec([zx, zy]), X2(zx) - 10, Y2(zy) - 14, 4, colors, 0.22);
          }
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.strokeRect(X2(-1), Y2(1), X2(1) - X2(-1), Y2(-1) - Y2(1));
        codes.forEach((z, i) => {
          const x = X2(z[0]),
            y = Y2(z[1]);
          if (settings.col) {
            ctx.font = '600 13px "Hanken Grotesk",sans-serif';
            ctx.fillStyle = col[lab[i] % 8];
            ctx.textAlign = 'center';
            ctx.fillText(LT[lab[i]], x, y + 4);
          } else {
            ctx.fillStyle = mix(colors.ink, colors.ink, 0, 0.6);
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 7);
            ctx.fill();
          }
        });
        markTP(ctx, X2(selectedLatentPoint[0]), Y2(selectedLatentPoint[1]), colors, false);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('first number of the code →', X2(-1) + 4, Y2(-1) - 5);
        mainCanvas._g = { X2, Y2, w, h };
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          cz = Math.min((w - 40) / 6 / 6, (h * 0.21) / 7.4),
          gw = 5 * cz,
          x0 = (w - 6 * gw * 1.25) / 2;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('in', 6, 16 + 3.5 * cz);
        ctx.fillText('out', 6, 28 + 10.5 * cz);
        for (let k = 0; k < 6; k++) {
          const i = k * 25,
            x = x0 + k * gw * 1.25;
          glyph(ctx, X[i], x, 10, cz, colors);
          glyph(ctx, net.predictVector(X[i]), x, 22 + 7 * cz, cz, colors);
        }
        const y3 = 38 + 14 * cz,
          pz = Math.min(9, (h - y3 - 8) / 7.2);
        glyph(ctx, dec(selectedLatentPoint), 14, y3 + 10, pz, colors);
        ctx.fillStyle = colors.muted;
        ctx.fillText('the diamond', 14, y3 + 2);
        drawSeriesChart(ctx, colors, 5 * pz + 50, y3 - 4, w - 5 * pz - 54, h - y3 + 4, LL, {
          title: 'error per pixel',
          n: Math.max(100, LL.length),
          col: colors.ok,
          ym: Math.max(0.7, LL[0] || 0.7),
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'ae', activeStepIndex);
      setReadout(root, 'ep', String(ep));
      setReadout(root, 'L', ep ? f3(LL[ep - 1]) : '–');
      setReadout(root, 'px', pct(pixelAccuracy()));
      setReadout(root, 'cmp', '35 → 2');
      setReadout(root, 'st', conv ? 'Converged' : ep ? 'Training' : 'Not started');
    };
    gen();
    build();
    if (prefersReducedMotion) train(3000);
    const redo = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      frameCount++;
      if (ep < 30 && frameCount % 4) return;
      train(Math.min(ep < 30 ? 1 : ep < 300 ? 3 : 6, checkRunStop.remaining()));
      draw();
      checkRunStop();
    });
    playback = playBtn($('#ae-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'ae', {
      done: () => conv,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'epochs',
    });
    $('#ae-step', root).onclick = () => {
      playback.stop();
      if (conv) {
        run++;
        build();
      } else train(10);
      draw();
    };
    $('#ae-reset', root).onclick = redo;
    bind(
      root,
      'ae-hw',
      (v) => {
        settings.hw = v;
        redo();
      },
      (v) => String(v),
    );
    bind(root, 'ae-flip', (v) => {
      settings.flip = v;
      gen();
      redo();
    });
    $('#ae-col', root).onchange = (e) => {
      settings.col = e.target.checked;
      draw();
    };
    $('#ae-new', root).onclick = () => {
      settings.seed++;
      gen();
      redo();
    };
    const mv = (e) => {
      const q = mainCanvas._g;
      if (!q) return;
      const r = mainCanvas.getBoundingClientRect();
      selectedLatentPoint = [
        clamp((e.clientX - r.left - q.w / 2) / ((q.w / 2 - 14) / 1.08), -1, 1),
        clamp(-(e.clientY - r.top - q.h / 2) / ((q.h / 2 - 14) / 1.08), -1, 1),
      ];
      if (!playback.on) draw();
    };
    mainCanvas.style.touchAction = 'none';
    mainCanvas.addEventListener('pointerdown', (e) => {
      drag = true;
      mainCanvas.setPointerCapture(e.pointerId);
      mv(e);
    });
    mainCanvas.addEventListener('pointermove', (e) => {
      if (drag) mv(e);
    });
    ['pointerup', 'pointercancel'].forEach((n2) =>
      mainCanvas.addEventListener(n2, () => {
        drag = false;
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
