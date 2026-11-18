/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  SANS,
  readThemeColors,
  bind,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { NNet, drawEdges, netPos, nnData } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#dr-c', root),
      cr = $('#dr-r', root),
      live = $('#dr-live', root),
      settings = { dropoutProbability: 0.5, hw: 24, n: 60, fl: 0.12, seed: 2 },
      EPM = 500;
    let tr = [],
      te = [],
      baselineNet,
      dropoutNet,
      ep = 0,
      lossHistories = [[], [], [], []],
      mask = null,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1;
    const gen = () => {
      const a = nnData('moons', Math.round((settings.n * 4) / 3) + 2, 0.28, settings.seed)
          .filter((p) => !p.test)
          .slice(0, settings.n),
        r = rng(settings.seed * 5 + 1);
      tr = a.map((p) => ({ ...p, bad: r() < settings.fl }));
      tr.forEach((p) => {
        if (p.bad) p.c = 1 - p.c;
      });
      te = nnData('moons', 400, 0.28, settings.seed + 77);
    };
    const build = () => {
      const sz = [2, settings.hw, settings.hw, 1],
        sd = settings.seed * 3 + run * 17;
      baselineNet = NNet(sz, { act: 'relu', seed: sd });
      dropoutNet = NNet(sz, { act: 'relu', seed: sd });
      ep = 0;
      lossHistories = [[], [], [], []];
      mask = null;
      activeStepIndex = 0;
      msg =
        'Two networks with the same ' +
        (2 * settings.hw + settings.hw * settings.hw + settings.hw + 2 * settings.hw + 1) +
        ' starting weights and the same ' +
        tr.length +
        ' training points, ' +
        tr.filter((p) => p.bad).length +
        ' of which carry the wrong label. Press Train both or One epoch.';
    };
    const ll = (N, P) =>
        -P.reduce((s, p) => {
          const o = N.predictScalar([p.x, p.y]);
          return s + (p.c ? Math.log(o + 1e-9) : Math.log(1 - o + 1e-9));
        }, 0) / P.length,
      ac = (N, P) =>
        P.filter((p) => (N.predictScalar([p.x, p.y]) >= 0.5 ? 1 : 0) === p.c).length / P.length;
    const epoch = () => {
      const X = tr.map((p) => [p.x, p.y]),
        Y = tr.map((p) => p.c),
        a = baselineNet.train(X, Y, 0.02),
        b = dropoutNet.train(X, Y, 0.02, { drop: settings.dropoutProbability });
      mask = b.mask;
      ep++;
      lossHistories[0].push(a.loss);
      lossHistories[1].push(b.loss);
      lossHistories[2].push(ll(baselineNet, te));
      lossHistories[3].push(ll(dropoutNet, te));
      activeStepIndex = ep < 3 ? 2 : 3;
      const off = mask ? mask.reduce((s, m) => s + m.filter((v) => v === 0).length, 0) : 0;
      msg =
        'Epoch ' +
        ep +
        ': on the last example ' +
        off +
        ' of the ' +
        2 * settings.hw +
        ' hidden neurons in the dropout network were switched off. Loss on new data: normal ' +
        f3(lossHistories[2][ep - 1]) +
        ', dropout ' +
        f3(lossHistories[3][ep - 1]) +
        '.' +
        (ep >= EPM
          ? ' Both have had their ' +
            EPM +
            ' epochs, so training stops. ' +
            (lossHistories[3][ep - 1] < lossHistories[2][ep - 1]
              ? 'Dropout ends with the lower error on new data.'
              : 'Here dropout did not help: there was little overfitting to cure, or p is too large.')
          : '');
      if (ep >= EPM) activeStepIndex = 4;
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          hw = w / 2,
          cs = 7;
        [baselineNet, dropoutNet].forEach((N, q) => {
          const x0 = q * hw,
            X = (x) => x0 + hw / 2 + (x * hw) / 6.4,
            Y = (y) => h / 2 - (y * h) / 6.4;
          ctx.save();
          ctx.beginPath();
          ctx.rect(x0, 0, hw, h);
          ctx.clip();
          for (let py = 0; py < h; py += cs)
            for (let px = 0; px < hw; px += cs) {
              const p = N.predictScalar([
                  ((px + cs / 2 - hw / 2) * 6.4) / hw,
                  (-(py + cs / 2 - h / 2) * 6.4) / h,
                ]),
                s = Math.abs(p - 0.5) * 2;
              ctx.fillStyle = mix(
                p < 0.5 ? colors.accent : colors.gold,
                colors.bg,
                0,
                0.12 + 0.42 * s,
              );
              ctx.fillRect(x0 + px, py, cs, cs);
            }
          for (const p of tr) {
            ctx.beginPath();
            ctx.arc(X(p.x), Y(p.y), 3.6, 0, 7);
            ctx.fillStyle = p.c ? colors.gold : colors.accent;
            ctx.fill();
            ctx.strokeStyle = p.bad ? colors.ink : colors.bg;
            ctx.lineWidth = p.bad ? 2 : 0.8;
            ctx.stroke();
          }
          ctx.restore();
          ctx.font = SANS;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'left';
          ctx.fillText(
            q ? 'With dropout, p = ' + f2(settings.dropoutProbability) : 'Without dropout',
            x0 + 10,
            18,
          );
        });
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(hw, 0);
        ctx.lineTo(hw, h);
        ctx.stroke();
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('black ring = wrong label', w - 8, h - 8);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          sh = Math.min(10, settings.hw),
          pos = netPos([2, sh, sh, 1], 30, 4, w - 60, h1 - 20);
        drawEdges(
          ctx,
          [
            dropoutNet.W[0].slice(0, sh),
            dropoutNet.W[1].slice(0, sh).map((q) => q.slice(0, sh)),
            [dropoutNet.W[2][0].slice(0, sh)],
          ],
          pos,
          colors,
          playback && playback.on ? frameCount / 45 : null,
        );
        pos.forEach((col, l) =>
          col.forEach((q, j) => {
            const off = mask && l > 0 && l < 3 && mask[l - 1][j] === 0;
            ctx.fillStyle = off ? colors.bg : colors.ink;
            ctx.strokeStyle = off ? colors.muted : colors.ink;
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.arc(q[0], q[1], 6.5, 0, 7);
            ctx.fill();
            ctx.stroke();
            if (off) {
              ctx.strokeStyle = colors.ok;
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(q[0] - 6, q[1] - 6);
              ctx.lineTo(q[0] + 6, q[1] + 6);
              ctx.moveTo(q[0] + 6, q[1] - 6);
              ctx.lineTo(q[0] - 6, q[1] + 6);
              ctx.stroke();
            }
          }),
        );
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        ctx.fillText(
          settings.hw > sh
            ? 'first ' + sh + ' of ' + settings.hw + ' neurons in each hidden layer'
            : 'the dropout network',
          w / 2,
          h1 + 2,
        );
        const ym = 1.6,
          U = drawSeriesChart(
            ctx,
            colors,
            0,
            h1 + 8,
            w,
            h - h1 - 8,
            lossHistories[2].map((v) => Math.min(v, ym)),
            {
              title:
                'loss on new data: normal (red), dropout (green). Dashed: on the training points',
              n: EPM,
              ym,
              col: colors.ok,
            },
          );
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        lossHistories[3].forEach((v, i) =>
          i ? ctx.lineTo(U.X(i), U.Y(Math.min(v, ym))) : ctx.moveTo(U.X(i), U.Y(Math.min(v, ym))),
        );
        ctx.stroke();
        [
          [lossHistories[0], colors.ok],
          [lossHistories[1], colors.accent],
        ].forEach((q) => {
          ctx.strokeStyle = mix(q[1], q[1], 0, 0.4);
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          q[0].forEach((v, i) =>
            i ? ctx.lineTo(U.X(i), U.Y(Math.min(v, ym))) : ctx.moveTo(U.X(i), U.Y(Math.min(v, ym))),
          );
          ctx.stroke();
          ctx.setLineDash([]);
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'dr', activeStepIndex);
      setReadout(root, 'ep', ep + ' of ' + EPM);
      setReadout(root, 'a0', pct(ac(baselineNet, tr)) + ' / ' + pct(ac(baselineNet, te)));
      setReadout(root, 'a1', pct(ac(dropoutNet, tr)) + ' / ' + pct(ac(dropoutNet, te)));
      setReadout(root, 'l0', ep ? f3(lossHistories[2][ep - 1]) : '–');
      setReadout(root, 'l1', ep ? f3(lossHistories[3][ep - 1]) : '–');
    };
    gen();
    build();
    if (prefersReducedMotion) for (let i = 0; i < EPM; i++) epoch();
    const redo = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      frameCount++;
      if (ep < 12 && frameCount % 8) return;
      for (let i = 0; i < (ep < 12 ? 1 : 2) && ep < EPM && checkRunStop.remaining() > 0; i++)
        epoch();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#dr-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'dr', {
      done: () => ep >= EPM,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'epochs',
    });
    $('#dr-step', root).onclick = () => {
      playback.stop();
      if (ep >= EPM) {
        run++;
        build();
      } else epoch();
      draw();
    };
    $('#dr-reset', root).onclick = redo;
    bind(root, 'dr-p', (v) => {
      settings.dropoutProbability = v;
      redo();
    });
    bind(
      root,
      'dr-hw',
      (v) => {
        settings.hw = v;
        redo();
      },
      (v) => String(v),
    );
    bind(
      root,
      'dr-n',
      (v) => {
        settings.n = v;
        gen();
        redo();
      },
      (v) => String(v),
    );
    bind(root, 'dr-fl', (v) => {
      settings.fl = v;
      gen();
      redo();
    });
    $('#dr-new', root).onclick = () => {
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
