/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  f2,
  prepareCanvas,
  letterData,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { UC, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';
import { sig } from '../../classification/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#va-c', root),
      cr = $('#va-r', root),
      live = $('#va-live', root),
      settings = { beta: 1, flip: 0.04, col: true, seed: 1 },
      R = 3.2;
    let X,
      flatTrainingImages,
      lab,
      LT,
      enc,
      dec,
      ps,
      trainingSteps = 0,
      reconstructionLossHistory = [],
      KL = [],
      conv = false,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      MU = [],
      SD = [],
      pr = [0.5, 0.5],
      zs = [],
      drag = false;
    const gen = () => {
        const d = letterData(settings.seed, settings.flip);
        X = d.X.map((x) => x.map((v) => clamp(Math.round(v), 0, 1)));
        lab = d.lab;
        LT = d.CH;
        flatTrainingImages = new Float64Array(X.length * 35);
        X.forEach((x, i) =>
          x.forEach((v, j) => {
            flatTrainingImages[i * 35 + j] = v;
          }),
        );
      },
      newz = () => {
        const rn = randnFrom(Math.random);
        zs = Array.from({ length: 8 }, () => [rn(), rn()]);
      };
    const build = () => {
      const q = randnFrom(rng(run * 7 + 2));
      enc = AD.mlp([35, 24, 4], q, 'tanh');
      dec = AD.mlp([2, 24, 35], q, 'tanh');
      ps = enc.ps.concat(dec.ps);
      trainingSteps = 0;
      reconstructionLossHistory = [];
      KL = [];
      conv = false;
      activeStepIndex = 0;
      codes();
      msg =
        'Random weights. Every picture’s cloud sits near the middle and everything decodes to grey. Press Train.';
    };
    const codes = () => {
      const o = enc.fwd(AD.input(flatTrainingImages, [X.length, 35]));
      AD.clear();
      MU = [];
      SD = [];
      for (let i = 0; i < X.length; i++) {
        MU.push([o.v[i * 4], o.v[i * 4 + 1]]);
        SD.push([Math.exp(0.5 * o.v[i * 4 + 2]), Math.exp(0.5 * o.v[i * 4 + 3])]);
      }
    };
    const decode = (z) => {
      const o = dec.fwd(AD.input(z, [1, 2]));
      AD.clear();
      return Float64Array.from(o.v, sig);
    };
    const train = (k) => {
      const n = X.length,
        rn = randnFrom(Math.random);
      for (let q = 0; q < k && !conv; q++) {
        const h = enc.fwd(AD.input(flatTrainingImages, [n, 35])),
          mu = AD.cols(h, 0, 2),
          lv = AD.cols(h, 2, 4),
          z = AD.reparam(
            mu,
            lv,
            Float64Array.from({ length: 2 * n }, () => rn()),
          ),
          rec = AD.bce(dec.fwd(z), flatTrainingImages),
          kl = AD.kl(mu, lv, n),
          loss = AD.wsum(rec, kl, settings.beta / 35);
        reconstructionLossHistory.push(rec.v[0] * 35);
        KL.push(kl.v[0]);
        AD.backward(loss);
        AD.adam(ps, 0.006, ++trainingSteps, 1);
        if (
          (trainingSteps > 300 &&
            Math.abs(
              reconstructionLossHistory[trainingSteps - 1] +
                settings.beta * KL[trainingSteps - 1] -
                reconstructionLossHistory[trainingSteps - 201] -
                settings.beta * KL[trainingSteps - 201],
            ) <
              0.01 *
                (reconstructionLossHistory[trainingSteps - 1] +
                  settings.beta * KL[trainingSteps - 1])) ||
          trainingSteps >= 2500
        )
          conv = true;
      }
      codes();
      activeStepIndex = conv ? 4 : trainingSteps % 4;
      msg =
        'Epoch ' +
        trainingSteps +
        ': every picture became a cloud, one point was drawn from each and decoded. Reconstruction error ' +
        f2(reconstructionLossHistory[trainingSteps - 1]) +
        ' per picture; distance of the clouds from the centre ' +
        f2(KL[trainingSteps - 1]) +
        '.' +
        (conv
          ? ' Converged: the total has stopped falling. The second row on the right shows pictures made from random points.'
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
          sc = (Math.min(w, h) / 2 / R) * 0.96,
          X2 = (v) => w / 2 + v * sc,
          Y2 = (v) => h / 2 - v * sc,
          NG = 9;
        for (let a = 0; a < NG; a++)
          for (let b = 0; b < NG; b++) {
            const zx = -R + (2 * R * (b + 0.5)) / NG,
              zy = R - (2 * R * (a + 0.5)) / NG;
            glyph(ctx, decode([zx, zy]), X2(zx) - 9, Y2(zy) - 12, 3.6, colors, 0.22);
          }
        [1, 2].forEach((r) => {
          ctx.strokeStyle = mix(colors.gold, colors.gold, 0, r === 1 ? 0.9 : 0.5);
          ctx.lineWidth = 1.6;
          ctx.setLineDash(r === 2 ? [5, 5] : []);
          ctx.beginPath();
          ctx.arc(X2(0), Y2(0), r * sc, 0, 7);
          ctx.stroke();
        });
        ctx.setLineDash([]);
        MU.forEach((m, i) => {
          const x = X2(m[0]),
            y = Y2(m[1]);
          if (x < 0 || y < 0 || x > w || y > h) return;
          ctx.fillStyle = mix(col[lab[i] % 8], col[lab[i] % 8], 0, 0.12);
          ctx.beginPath();
          ctx.ellipse(x, y, Math.max(1, SD[i][0] * sc), Math.max(1, SD[i][1] * sc), 0, 0, 7);
          ctx.fill();
          if (settings.col) {
            ctx.font = '600 12px "Hanken Grotesk",sans-serif';
            ctx.fillStyle = col[lab[i] % 8];
            ctx.textAlign = 'center';
            ctx.fillText(LT[lab[i]], x, y + 4);
          } else {
            ctx.fillStyle = mix(colors.ink, colors.ink, 0, 0.6);
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, 7);
            ctx.fill();
          }
        });
        markTP(ctx, X2(pr[0]), Y2(pr[1]), colors, false);
        ctx.font = MONO;
        ctx.fillStyle = colors.cut;
        ctx.textAlign = 'left';
        ctx.fillText('rings: where random draws come from (1 and 2 standard deviations)', 8, h - 8);
        mainCanvas._g = { w, h, sc };
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          cz = Math.min((w - 60) / 8 / 6.2, (h * 0.15) / 7.4),
          gw = cz * 6.2,
          x0 = 50;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('in', 6, 12 + 3.5 * cz);
        ctx.fillText('out', 6, 18 + 10.5 * cz);
        ctx.fillStyle = colors.cut;
        ctx.fillText('new', 6, 26 + 17.5 * cz);
        const o = enc.fwd(AD.input(flatTrainingImages, [X.length, 35]));
        AD.clear();
        for (let k = 0; k < 6; k++) {
          const i = k * 25;
          glyph(ctx, X[i], x0 + k * gw, 6, cz, colors);
          glyph(ctx, decode([o.v[i * 4], o.v[i * 4 + 1]]), x0 + k * gw, 12 + 7 * cz, cz, colors);
        }
        glyph(ctx, decode(pr), x0 + 6.6 * gw, 6, cz * 1.5, colors);
        ctx.fillStyle = colors.muted;
        ctx.fillText('diamond', x0 + 6.6 * gw, 18 + 10.5 * cz + 4);
        zs.forEach((z, k) => {
          glyph(ctx, decode(z), x0 + k * gw, 20 + 14 * cz, cz, colors);
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 1;
          ctx.strokeRect(x0 + k * gw - 1.5, 18.5 + 14 * cz, 5 * cz + 2, 7 * cz + 2);
        });
        const y3 = 34 + 21 * cz,
          U = drawSeriesChart(ctx, colors, 0, y3, w, h - y3, reconstructionLossHistory, {
            title: 'reconstruction error (red) and β × cloud distance (green)',
            n: Math.max(200, reconstructionLossHistory.length),
            col: colors.ok,
            ym: Math.max(24, reconstructionLossHistory[0] || 24),
          });
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        KL.forEach((v, i) =>
          i
            ? ctx.lineTo(U.X(i), U.Y(settings.beta * v))
            : ctx.moveTo(U.X(i), U.Y(settings.beta * v)),
        );
        ctx.stroke();
      }
      live.textContent = msg;
      highlightLessonStep(root, 'va', activeStepIndex);
      setReadout(root, 'ep', String(trainingSteps));
      setReadout(
        root,
        'rec',
        trainingSteps ? f2(reconstructionLossHistory[trainingSteps - 1]) : '–',
      );
      setReadout(root, 'kl', trainingSteps ? f2(KL[trainingSteps - 1]) : '–');
      setReadout(
        root,
        'in',
        MU.filter((m) => Math.hypot(m[0], m[1]) < 2).length + ' of ' + MU.length,
      );
      setReadout(root, 'st', conv ? 'Converged' : trainingSteps ? 'Training' : 'Not started');
    };
    gen();
    newz();
    build();
    if (prefersReducedMotion) while (!conv) train(50);
    const redo = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      frameCount++;
      if (trainingSteps < 30 && frameCount % 4) return;
      train(
        Math.min(trainingSteps < 30 ? 1 : trainingSteps < 300 ? 2 : 5, checkRunStop.remaining()),
      );
      draw();
      checkRunStop();
    });
    playback = playBtn($('#va-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'va', {
      done: () => conv,
      restart: () => {
        run++;
        build();
      },
      count: () => trainingSteps,
      what: 'epochs',
    });
    $('#va-step', root).onclick = () => {
      playback.stop();
      if (conv) {
        run++;
        build();
      } else train(10);
      draw();
    };
    $('#va-reset', root).onclick = redo;
    $('#va-samp', root).onclick = () => {
      newz();
      if (!playback.on) draw();
    };
    bind(
      root,
      'va-beta',
      (v) => {
        settings.beta = v;
        playback.stop();
        run++;
        build();
        draw();
        playback.start();
      },
      f2,
    );
    bind(root, 'va-flip', (v) => {
      settings.flip = v;
      gen();
      redo();
    });
    $('#va-col', root).onchange = (e) => {
      settings.col = e.target.checked;
      draw();
    };
    const mv = (e) => {
      const q = mainCanvas._g;
      if (!q) return;
      const r = mainCanvas.getBoundingClientRect();
      pr = [
        clamp((e.clientX - r.left - q.w / 2) / q.sc, -R, R),
        clamp(-(e.clientY - r.top - q.h / 2) / q.sc, -R, R),
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
