/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  f2,
  f3,
  prepareCanvas,
  hex,
  markTP,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { NNet, drawEdges, netPos, nnData } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#nn-c', root),
      cr = $('#nn-r', root),
      live = $('#nn-live', root),
      settings = { hl: 2, hw: 5, act: 'tanh', lr: 0.015, ds: 'circles', nz: 0.2, n: 200, seed: 3 };
    let pts = [],
      net,
      ep = 0,
      trainLossHistory = [],
      testLossHistory = [],
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      conv = false,
      run = 1;
    const tr = () => pts.filter((p) => !p.test),
      te = () => pts.filter((p) => p.test),
      binaryCrossEntropy = (P) =>
        P.length
          ? -P.reduce((s, p) => {
              const o = net.predictScalar([p.x, p.y]);
              return s + (p.c ? Math.log(o + 1e-9) : Math.log(1 - o + 1e-9));
            }, 0) / P.length
          : NaN,
      ac = (P) =>
        P.length
          ? P.filter((p) => (net.predictScalar([p.x, p.y]) >= 0.5 ? 1 : 0) === p.c).length /
            P.length
          : NaN;
    const build = () => {
      net = NNet([2, ...Array(settings.hl).fill(settings.hw), 1], {
        act: settings.act,
        seed: settings.seed * 7 + run * 13,
      });
      ep = 0;
      trainLossHistory = [];
      testLossHistory = [];
      conv = false;
      activeStepIndex = 0;
      msg =
        'Random weights: the network has not seen any data yet, and its answer is a smear. Press Train or One epoch.';
    };
    const epoch = () => {
      const T = tr(),
        l = net.train(
          T.map((p) => [p.x, p.y]),
          T.map((p) => p.c),
          settings.lr,
        ).loss;
      ep++;
      trainLossHistory.push(l);
      testLossHistory.push(binaryCrossEntropy(te()));
      activeStepIndex = 3;
      if (
        (ep > 120 &&
          Math.abs(trainLossHistory[ep - 1] - trainLossHistory[ep - 61]) <
            4e-3 * trainLossHistory[ep - 1]) ||
        l < 0.012 ||
        ep >= 4000
      )
        conv = true;
      msg =
        'Epoch ' +
        ep +
        ': all ' +
        T.length +
        ' training points went forward through the network, the loss came out at ' +
        f3(l) +
        (ep > 1 ? ' (it was ' + f3(trainLossHistory[ep - 2]) + ')' : '') +
        ', and every weight was nudged to lower it. Training accuracy ' +
        pct(ac(T)) +
        ', test accuracy ' +
        pct(ac(te())) +
        '.' +
        (conv
          ? ep >= 4000
            ? ' Reached the limit of 4000 epochs.'
            : ' Converged: the loss has stopped falling, so training stops here.'
          : '');
      if (conv) activeStepIndex = 4;
    };
    const tp = testPoint(
      root,
      'nn',
      ['Your x', 'Your y'],
      [0, 0],
      [
        [-3, 3],
        [-3, 3],
      ],
    );
    const draw = () => {
      const colors = readThemeColors(),
        ca = hex(colors.accent),
        cg = hex(colors.gold);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          cs = 8,
          X = (x) => w / 2 + (x * w) / 6,
          Y = (y) => h / 2 - (y * h) / 6;
        for (let py = 0; py < h; py += cs)
          for (let px = 0; px < w; px += cs) {
            const p = net.predictScalar([
                ((px + cs / 2 - w / 2) * 6) / w,
                (-(py + cs / 2 - h / 2) * 6) / h,
              ]),
              q = Math.abs(p - 0.5) * 2;
            ctx.fillStyle = mix(
              p < 0.5 ? colors.accent : colors.gold,
              colors.bg,
              0,
              0.1 + 0.42 * q,
            );
            ctx.fillRect(px, py, cs, cs);
          }
        for (const p of pts) {
          const cl = p.c ? colors.gold : colors.accent;
          ctx.beginPath();
          ctx.arc(X(p.x), Y(p.y), p.test ? 3.4 : 3.6, 0, 7);
          if (p.test) {
            ctx.fillStyle = colors.bg;
            ctx.fill();
            ctx.strokeStyle = cl;
            ctx.lineWidth = 1.8;
            ctx.stroke();
          } else {
            ctx.fillStyle = cl;
            ctx.fill();
            ctx.strokeStyle = colors.bg;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
        markTP(ctx, X(tp.v[0]), Y(tp.v[1]), colors);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.62,
          pos = netPos(net.sizes, 34, 8, w - 68, h1 - 16),
          NC = 9,
          act = [];
        drawEdges(ctx, net.W, pos, colors, playback && playback.on ? frameCount / 50 : null);
        for (let a = 0; a < NC; a++)
          for (let b = 0; b < NC; b++)
            act.push(net.fwd([-3 + (6 * (b + 0.5)) / NC, 3 - (6 * (a + 0.5)) / NC]).A);
        pos.forEach((col, l) =>
          col.forEach((q, j) => {
            const sz = l === 0 ? 22 : 30,
              x0 = q[0] - sz / 2,
              y0 = q[1] - sz / 2,
              cz = sz / NC;
            let lo = 1e9,
              hi = -1e9;
            if (l > 0 && l < net.L)
              for (const A of act) {
                lo = Math.min(lo, A[l][j]);
                hi = Math.max(hi, A[l][j]);
              }
            for (let k = 0; k < NC * NC; k++) {
              let v =
                l === 0
                  ? j === 0
                    ? ((k % NC) + 0.5) / NC
                    : 1 - (Math.floor(k / NC) + 0.5) / NC
                  : l === net.L
                    ? act[k][l][j]
                    : (act[k][l][j] - lo) / (hi - lo || 1);
              ctx.fillStyle =
                'rgb(' + ca.map((u, i) => Math.round(u + (cg[i] - u) * v)).join(',') + ')';
              ctx.fillRect(x0 + (k % NC) * cz, y0 + Math.floor(k / NC) * cz, cz + 0.6, cz + 0.6);
            }
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.3;
            ctx.strokeRect(x0, y0, sz, sz);
          }),
        );
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        ctx.fillText('x, y', pos[0][0][0], h1 - 1);
        ctx.fillText('answer', pos[net.L][0][0], h1 - 1);
        if (net.L > 1) ctx.fillText('hidden layers', w / 2, h1 - 1);
        const ym = Math.max(0.72, ...trainLossHistory.slice(0, 3), ...testLossHistory.slice(0, 3)),
          U = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, trainLossHistory, {
            title: 'loss: training (green) and test (red)',
            n: Math.max(60, trainLossHistory.length),
            ym,
            col: colors.accent,
          });
        if (testLossHistory.length) {
          ctx.strokeStyle = colors.ok;
          ctx.lineWidth = 2;
          ctx.beginPath();
          testLossHistory.forEach((v, i) =>
            i ? ctx.lineTo(U.X(i), U.Y(Math.min(v, ym))) : ctx.moveTo(U.X(i), U.Y(Math.min(v, ym))),
          );
          ctx.stroke();
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'nn', activeStepIndex);
      const o = net.predictScalar(tp.v);
      tp.out.textContent =
        'At (' +
        f2(tp.v[0]) +
        ', ' +
        f2(tp.v[1]) +
        ') the network outputs ' +
        f3(o) +
        ', so it says ' +
        (o >= 0.5 ? 'gold' : 'green') +
        ' with ' +
        (100 * Math.max(o, 1 - o)).toFixed(1) +
        '% confidence.';
      let np = 0;
      net.W.forEach((wl, l) => {
        np += wl.length * (wl[0].length + 1);
      });
      setReadout(root, 'ep', String(ep));
      setReadout(root, 'lt', trainLossHistory.length ? f3(trainLossHistory[ep - 1]) : '–');
      setReadout(root, 'le', testLossHistory.length ? f3(testLossHistory[ep - 1]) : '–');
      setReadout(root, 'at', pct(ac(tr())));
      setReadout(root, 'ae', pct(ac(te())));
      setReadout(root, 'np', String(np));
      setReadout(root, 'st', conv ? 'Converged' : ep ? 'Training' : 'Not started');
    };
    tp.on = () => draw();
    const gen = () => {
        pts = nnData(settings.ds, settings.n, settings.nz, settings.seed);
      },
      redo = () => {
        playback.stop();
        run++;
        build();
        draw();
      };
    gen();
    build();
    if (prefersReducedMotion) for (let i = 0; i < 4000 && !conv; i++) epoch();
    const anim = Anim(() => {
      frameCount++;
      if (frameCount % (ep < 20 ? 8 : ep < 90 ? 3 : 1)) {
        draw();
        return;
      }
      for (let i = 0; i < (ep < 400 ? 1 : 3) && !conv && checkRunStop.remaining() > 0; i++) epoch();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#nn-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'nn', {
      done: () => conv,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'epochs',
    });
    $('#nn-step', root).onclick = () => {
      playback.stop();
      if (conv) {
        run++;
        build();
      } else epoch();
      draw();
    };
    $('#nn-reset', root).onclick = redo;
    bind(
      root,
      'nn-hl',
      (v) => {
        settings.hl = v;
        redo();
      },
      (v) => String(v),
    );
    bind(
      root,
      'nn-hw',
      (v) => {
        settings.hw = v;
        redo();
      },
      (v) => String(v),
    );
    $('#nn-act', root).onchange = (e) => {
      settings.act = e.target.value;
      redo();
    };
    bind(
      root,
      'nn-lr',
      (v) => {
        settings.lr = v;
      },
      f3,
    );
    $('#nn-ds', root).onchange = (e) => {
      settings.ds = e.target.value;
      gen();
      redo();
    };
    bind(root, 'nn-nz', (v) => {
      settings.nz = v;
      gen();
      redo();
    });
    bind(
      root,
      'nn-n',
      (v) => {
        settings.n = v;
        gen();
        redo();
      },
      (v) => String(v),
    );
    $('#nn-new', root).onclick = () => {
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
