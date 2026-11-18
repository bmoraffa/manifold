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
  letterData,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { NNet } from '../../neural-nets/shared.js';
import { pct } from '../../ensembles/shared.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#tl-c', root),
      cr = $('#tl-r', root),
      live = $('#tl-live', root),
      settings = { k: 1, fl: 0.2, seed: 1 },
      Hn = 24,
      PE = 200,
      FE = 260,
      RP = 4,
      VW = [1, 1, 0, 1, 0, 0];
    let A,
      targetTrainSets = [],
      Bte,
      pre,
      scratchNets = [],
      transferNets = [],
      ep = 0,
      fe = 0,
      pa = 0,
      AS = [],
      AT = [],
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      done = false;
    const bin = (d) => ({
        X: d.X.map((x) => Float64Array.from(x, (v) => clamp(Math.round(v), 0, 1))),
        lab: d.lab,
      }),
      oh = (l) => Float64Array.from({ length: 6 }, (_, q) => (q === l ? 1 : 0));
    const gen = () => {
      const a = bin(letterData(settings.seed * 3 + 1, settings.fl)),
        b = bin(letterData(settings.seed * 3 + 2, settings.fl));
      A = { X: a.X.concat(b.X), Y: a.lab.concat(b.lab).map(oh), lab: a.lab.concat(b.lab) };
      Bte = bin(letterData(settings.seed * 3 + 900, settings.fl));
      targetTrainSets = [];
      for (let r = 0; r < RP; r++) {
        const d = bin(letterData(settings.seed * 7 + 50 + r, settings.fl)),
          X = [],
          Y = [];
        for (let l = 0; l < 6; l++)
          for (let i = 0; i < settings.k; i++) {
            X.push(d.X[l * 25 + i]);
            Y.push(Float64Array.of(VW[l]));
          }
        targetTrainSets.push({ X, Y });
      }
    };
    const build = () => {
      pre = NNet([35, Hn, 6], { act: 'relu', out: 'sig', seed: run * 5 + settings.seed });
      scratchNets = [];
      transferNets = [];
      ep = 0;
      fe = 0;
      pa = 0;
      AS = [];
      AT = [];
      done = false;
      activeStepIndex = 0;
      msg =
        'Stage 1: a network with random weights is about to study ' +
        A.X.length +
        ' labelled letter pictures. Press Run.';
    };
    const acc6 = () => {
        let ok = 0;
        A.X.forEach((x, i) => {
          const o = pre.predictVector(x);
          if (o.indexOf(Math.max(...o)) === A.lab[i]) ok++;
        });
        return ok / A.X.length;
      },
      accB = (N) => {
        let ok = 0;
        Bte.X.forEach((x, i) => {
          if ((N.predictVector(x)[0] >= 0.5 ? 1 : 0) === VW[Bte.lab[i]]) ok++;
        });
        return ok / Bte.X.length;
      };
    const startB = () => {
      for (let r = 0; r < RP; r++) {
        const s = NNet([35, Hn, 1], { act: 'relu', out: 'sig', seed: run * 3 + r * 7 + 2 }),
          tn = NNet([35, Hn, 1], { act: 'relu', out: 'sig', seed: run * 3 + r * 7 + 2 });
        for (let j = 0; j < Hn; j++) {
          tn.W[0][j].set(pre.W[0][j]);
          tn.B[0][j] = pre.B[0][j];
        }
        scratchNets.push(s);
        transferNets.push(tn);
      }
    };
    const train = (k) => {
      for (let i = 0; i < k && !done; i++) {
        if (ep < PE) {
          pre.train(A.X, A.Y, 0.01);
          ep++;
          if (ep % 10 === 0 || ep >= PE) pa = acc6();
          activeStepIndex = 0;
          msg =
            'Stage 1, pretraining: epoch ' +
            ep +
            ' of ' +
            PE +
            '. It names ' +
            pct(pa) +
            ' of the ' +
            A.X.length +
            ' letter pictures correctly. Watch the top row of small pictures turn from noise into strokes.';
          if (ep >= PE) {
            startB();
            activeStepIndex = 1;
            msg =
              'Pretraining is done: ' +
              pct(pa) +
              ' correct. Now the new task: vowel or not, from ' +
              6 * settings.k +
              ' example' +
              (6 * settings.k === 1 ? '' : 's') +
              '. Learner A starts from random weights. Learner B starts with the first layer you see at the top, frozen.';
          }
        } else {
          let a = 0,
            b = 0;
          for (let r = 0; r < RP; r++) {
            scratchNets[r].train(targetTrainSets[r].X, targetTrainSets[r].Y, 0.01);
            transferNets[r].train(targetTrainSets[r].X, targetTrainSets[r].Y, 0.01, { freeze: 1 });
            a += accB(scratchNets[r]) / RP;
            b += accB(transferNets[r]) / RP;
          }
          AS.push(a);
          AT.push(b);
          fe++;
          activeStepIndex = fe % 2 ? 2 : 3;
          if (fe >= FE) done = true;
          msg =
            'Stage 2, the new task: epoch ' +
            fe +
            ' of ' +
            FE +
            '. On ' +
            Bte.X.length +
            ' new pictures, from scratch gets ' +
            pct(a) +
            ' right and transfer gets ' +
            pct(b) +
            '.' +
            (done
              ? ' Both have had their ' +
                FE +
                ' epochs. ' +
                (b > a + 0.02
                  ? 'Transfer wins by ' +
                    (100 * (b - a)).toFixed(0) +
                    ' points: with so few examples, borrowed features beat features learned from almost nothing.'
                  : a > b + 0.02
                    ? 'From scratch wins here: there was enough data for it to learn its own features, and the frozen ones were a limit.'
                    : 'They finish level: with this many examples, starting from scratch catches up.')
              : '');
          if (done) activeStepIndex = 4;
        }
      }
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          cz = Math.min((w - 20) / 12 / 6.2, (h / 2 - 34) / 2 / 8),
          gw = 5 * cz,
          gh = 7 * cz,
          rowsOf = (N, y0, lab, al) => {
            ctx.font = MONO;
            ctx.fillStyle = colors.muted;
            ctx.textAlign = 'left';
            ctx.fillText(lab, 8, y0 - 6);
            if (!N) {
              ctx.fillText('(not started yet)', 8, y0 + 20);
              return;
            }
            for (let j = 0; j < Hn; j++) {
              const x = 8 + (j % 12) * (gw + cz * 1.2),
                y = y0 + Math.floor(j / 12) * (gh + cz * 1.5),
                wv = N.W[0][j];
              let m = 1e-9;
              for (const v of wv) m = Math.max(m, Math.abs(v));
              for (let q = 0; q < 35; q++) {
                const v = wv[q] / m;
                ctx.fillStyle = mix(colors.bg, v >= 0 ? colors.gold : colors.ok, Math.abs(v));
                ctx.fillRect(x + (q % 5) * cz, y + Math.floor(q / 5) * cz, cz - 0.4, cz - 0.4);
              }
            }
          };
        rowsOf(
          pre,
          22,
          'first layer after pretraining on ' +
            A.X.length +
            ' pictures (copied to learner B, frozen)',
        );
        rowsOf(
          scratchNets[0],
          h / 2 + 26,
          'first layer of learner A, trained from scratch on ' +
            6 * settings.k +
            ' example' +
            (6 * settings.k === 1 ? '' : 's'),
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          B0 = targetTrainSets[0],
          sh = Math.min(B0.X.length, 18),
          cz = Math.min((w - 20) / sh / 6.2, (h * 0.2) / 7.6);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('the new task’s examples (first try): gold frame = vowel', 8, 13);
        for (let i = 0; i < sh; i++) {
          const x = 10 + i * cz * 6.2,
            y = 22;
          for (let q = 0; q < 35; q++) {
            ctx.fillStyle = mix(colors.bg, colors.ink, 0.06 + 0.94 * B0.X[i][q]);
            ctx.fillRect(x + (q % 5) * cz, y + Math.floor(q / 5) * cz, cz - 0.4, cz - 0.4);
          }
          ctx.strokeStyle = B0.Y[i][0] ? colors.gold : colors.muted;
          ctx.lineWidth = B0.Y[i][0] ? 2.4 : 1;
          ctx.strokeRect(x - 1.5, y - 1.5, 5 * cz + 2, 7 * cz + 2);
        }
        const y0 = h * 0.36,
          U = drawSeriesChart(ctx, colors, 0, y0, w, h - y0, AT, {
            title: 'accuracy on new pictures: transfer (green), from scratch (red)',
            n: FE,
            y1: 0.4,
            ym: 1,
            col: colors.accent,
          });
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        AS.forEach((v, i) => (i ? ctx.lineTo(U.X(i), U.Y(v)) : ctx.moveTo(U.X(i), U.Y(v))));
        ctx.stroke();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(U.X(0), U.Y(0.5));
        ctx.lineTo(U.X(FE - 1), U.Y(0.5));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('dashed: a coin toss', w - 14, U.Y(0.5) - 4);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'tl', activeStepIndex);
      setReadout(root, 'ph', done ? 'Finished' : ep < PE ? 'Pretraining' : 'New task');
      setReadout(root, 'pa', ep ? pct(pa) : '–');
      setReadout(root, 'sc', fe ? pct(AS[fe - 1]) : '–');
      setReadout(root, 'tf', fe ? pct(AT[fe - 1]) : '–');
      setReadout(root, 'n', String(6 * settings.k));
      setReadout(root, 'w', 35 * Hn + Hn + Hn + 1 + ' / ' + (Hn + 1));
    };
    gen();
    build();
    if (prefersReducedMotion) train(PE + FE);
    const redo = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      frameCount++;
      if (ep < PE) {
        if (ep < 30 && frameCount % 3) return;
        train(Math.min(ep < 30 ? 1 : 2, checkRunStop.remaining()));
        draw();
        checkRunStop();
        return;
      }
      if (frameCount % (fe < 6 ? 16 : 2)) return;
      train(Math.min(1, checkRunStop.remaining()));
      draw();
      checkRunStop();
    });
    playback = playBtn($('#tl-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'tl', {
      done: () => done,
      restart: () => {
        run++;
        build();
      },
      count: () => ep + fe,
      what: 'epochs',
    });
    $('#tl-step', root).onclick = () => {
      playback.stop();
      if (done) {
        run++;
        build();
      } else train(10);
      draw();
    };
    $('#tl-reset', root).onclick = redo;
    bind(
      root,
      'tl-k',
      (v) => {
        settings.k = v;
        gen();
        redo();
      },
      (v) => String(v),
    );
    bind(root, 'tl-fl', (v) => {
      settings.fl = v;
      gen();
      redo();
    });
    $('#tl-new', root).onclick = () => {
      settings.seed++;
      gen();
      redo();
    };
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
