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
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { NNet } from '../shared.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#mc-c', root),
      cr = $('#mc-r', root),
      live = $('#mc-live', root),
      settings = { dropoutProbability: 0.1, hw: 48, gap: 2.2, nz: 0.12, seed: 2 },
      NG = 110,
      EPM = 1200,
      f = (x) => Math.sin(1.6 * x) + 0.25 * x;
    let D = [],
      net,
      ep = 0,
      passes = [],
      sum,
      sq,
      sampleCount = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      deterministicCurve = null,
      last = 0,
      stable = 0,
      done = false,
      probePredictions = [];
    const gx = (i) => -6 + (12 * i) / (NG - 1),
      gen = () => {
        const r = rng(settings.seed * 9 + 4),
          rn = randnFrom(r);
        D = [];
        for (let i = 0; i < 46; i++) {
          const left = i % 2 === 0,
            x = left
              ? -2.6 + r() * (2.6 - settings.gap / 2 - 0.15)
              : settings.gap / 2 + 0.15 + r() * (2.6 - settings.gap / 2 - 0.15);
          D.push({ x, y: f(x) + settings.nz * rn() });
        }
      };
    const build = () => {
      net = NNet([1, settings.hw, settings.hw, 1], {
        act: 'relu',
        out: 'lin',
        seed: settings.seed * 5 + run * 11,
      });
      ep = 0;
      passes = [];
      sum = new Float64Array(NG);
      sq = new Float64Array(NG);
      sampleCount = 0;
      deterministicCurve = null;
      done = false;
      stable = 0;
      probePredictions = [];
      activeStepIndex = 0;
      msg =
        'Stage 1: training a network with dropout (p = ' +
        f2(settings.dropoutProbability) +
        ') on ' +
        D.length +
        ' measurements. Nothing lies in the gap between x = ' +
        f2(-settings.gap / 2) +
        ' and ' +
        f2(settings.gap / 2) +
        '.';
    };
    const sdAt = (i) =>
        sampleCount > 1
          ? Math.sqrt(Math.max(0, (sq[i] - (sum[i] * sum[i]) / sampleCount) / (sampleCount - 1)))
          : 0,
      avg = (a, b) => {
        let s = 0,
          n = 0;
        for (let i = 0; i < NG; i++) {
          const x = gx(i);
          if (x >= a && x <= b) {
            s += sdAt(i);
            n++;
          }
        }
        return n ? s / n : 0;
      };
    const pass = () => {
      const mask = [0, 1].map(() =>
          Float64Array.from({ length: settings.hw }, () =>
            Math.random() < settings.dropoutProbability ? 0 : 1 / (1 - settings.dropoutProbability),
          ),
        ),
        ft = new Float32Array(NG);
      for (let i = 0; i < NG; i++) {
        const v = net.fwd([gx(i)], mask).A[3][0];
        ft[i] = v;
        sum[i] += v;
        sq[i] += v * v;
      }
      passes.push(ft);
      if (passes.length > 50) passes.shift();
      sampleCount++;
      probePredictions.push(net.fwd([tp.v[0]], mask).A[3][0]);
      activeStepIndex = sampleCount < 2 ? 2 : sampleCount < 5 ? 3 : 4;
      const tot = avg(-6, 6);
      if (sampleCount > 12 && Math.abs(tot - last) < 0.012 * tot) stable++;
      else stable = 0;
      last = tot;
      if (stable >= 14 || sampleCount >= 400) done = true;
      msg =
        'Pass ' +
        sampleCount +
        ': a random ' +
        mask.reduce((s, m) => s + m.filter((q) => !q).length, 0) +
        ' of the ' +
        2 * settings.hw +
        ' hidden neurons were switched off and the rest drew this curve. So far the answers disagree by ±' +
        f2(avg(-2.5, -settings.gap / 2 - 0.2)) +
        ' over the data and by ±' +
        f2(avg(-settings.gap / 2 + 0.1, settings.gap / 2 - 0.1) || 0) +
        ' in the gap.' +
        (done
          ? ' More passes no longer change the band, so it stops after ' + sampleCount + ' passes.'
          : '');
    };
    const tr = (n) => {
      for (let i = 0; i < n && ep < EPM; i++) {
        net.train(
          D.map((q) => [q.x]),
          D.map((q) => q.y),
          0.01,
          { drop: settings.dropoutProbability },
        );
        ep++;
      }
      if (ep >= EPM && !deterministicCurve) {
        deterministicCurve = new Float32Array(NG);
        for (let i = 0; i < NG; i++) deterministicCurve[i] = net.predictScalar([gx(i)]);
        activeStepIndex = 1;
        msg =
          'Stage 2: training is finished. The black line is the ordinary prediction, with every neuron on: one confident answer for every x, including across the gap. Now dropout is left on.';
      } else if (ep < EPM) msg = 'Stage 1: training with dropout, epoch ' + ep + ' of ' + EPM + '.';
    };
    const step = () => {
      if (done) {
        run++;
        build();
        return;
      }
      if (ep < EPM) tr(EPM);
      else pass();
    };
    const tp = testPoint(root, 'mc', ['Your x'], [0], [[-6, 6]]);
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = plot(ctx, w, h, [-6, 6], [-2.6, 2.6], colors, {
            xstep: 2,
            yticks: [-2, 0, 2],
            xlabel: 'x',
          });
        clipPlot(ctx, P);
        ctx.fillStyle = mix(colors.ok, colors.ok, 0, 0.07);
        ctx.fillRect(
          P.X(-settings.gap / 2),
          P.Y(2.6),
          P.X(settings.gap / 2) - P.X(-settings.gap / 2),
          P.Y(-2.6) - P.Y(2.6),
        );
        ctx.fillRect(P.X(-6), P.Y(2.6), P.X(-2.6) - P.X(-6), P.Y(-2.6) - P.Y(2.6));
        ctx.fillRect(P.X(2.6), P.Y(2.6), P.X(6) - P.X(2.6), P.Y(-2.6) - P.Y(2.6));
        if (sampleCount > 1) {
          ctx.beginPath();
          for (let i = 0; i < NG; i++) {
            const y = sum[i] / sampleCount + 1.96 * sdAt(i);
            i ? ctx.lineTo(P.X(gx(i)), P.Y(y)) : ctx.moveTo(P.X(gx(i)), P.Y(y));
          }
          for (let i = NG - 1; i >= 0; i--)
            ctx.lineTo(P.X(gx(i)), P.Y(sum[i] / sampleCount - 1.96 * sdAt(i)));
          ctx.closePath();
          ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.2);
          ctx.fill();
        }
        passes.forEach((ft, q) => {
          const lastOne = q === passes.length - 1;
          ctx.strokeStyle = mix(colors.accent, colors.accent, 0, lastOne ? 1 : 0.22);
          ctx.lineWidth = lastOne ? 2 : 1;
          ctx.beginPath();
          for (let i = 0; i < NG; i++)
            i ? ctx.lineTo(P.X(gx(i)), P.Y(ft[i])) : ctx.moveTo(P.X(gx(i)), P.Y(ft[i]));
          ctx.stroke();
        });
        if (ep < EPM) {
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 2;
          ctx.beginPath();
          for (let i = 0; i < NG; i++) {
            const y = net.predictScalar([gx(i)]);
            i ? ctx.lineTo(P.X(gx(i)), P.Y(y)) : ctx.moveTo(P.X(gx(i)), P.Y(y));
          }
          ctx.stroke();
        }
        if (deterministicCurve) {
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 2;
          ctx.setLineDash(sampleCount ? [6, 4] : []);
          ctx.beginPath();
          for (let i = 0; i < NG; i++)
            i
              ? ctx.lineTo(P.X(gx(i)), P.Y(deterministicCurve[i]))
              : ctx.moveTo(P.X(gx(i)), P.Y(deterministicCurve[i]));
          ctx.stroke();
          ctx.setLineDash([]);
        }
        if (sampleCount > 1) {
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 3;
          ctx.beginPath();
          for (let i = 0; i < NG; i++)
            i
              ? ctx.lineTo(P.X(gx(i)), P.Y(sum[i] / sampleCount))
              : ctx.moveTo(P.X(gx(i)), P.Y(sum[i] / sampleCount));
          ctx.stroke();
        }
        ctx.fillStyle = colors.ink;
        for (const q of D) {
          ctx.beginPath();
          ctx.arc(P.X(q.x), P.Y(q.y), 3.2, 0, 7);
          ctx.fill();
        }
        const px = P.X(tp.v[0]);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(px, P.Y(-2.6));
        ctx.lineTo(px, P.Y(2.6));
        ctx.stroke();
        ctx.restore();
        ctx.font = MONO;
        ctx.fillStyle = colors.ok;
        ctx.textAlign = 'center';
        ctx.fillText('no data here', P.X(0), P.m.t + 10);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          m = { l: 40, r: 12, t: 24, b: 20 };
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          'answers given at x = ' + f2(tp.v[0]) + ' by the ' + probePredictions.length + ' passes',
          8,
          13,
        );
        const X = (v) => m.l + ((v + 2.6) / 5.2) * (w - m.l - m.r),
          nb = 40,
          cnt = new Array(nb).fill(0);
        for (const v of probePredictions)
          cnt[clamp(Math.floor(((v + 2.6) / 5.2) * nb), 0, nb - 1)]++;
        const cm = Math.max(4, ...cnt);
        ctx.fillStyle = colors.accent;
        cnt.forEach((q, i) =>
          ctx.fillRect(
            m.l + (i * (w - m.l - m.r)) / nb,
            h1 - m.b - (q / cm) * (h1 - m.t - m.b),
            (w - m.l - m.r) / nb - 1,
            (q / cm) * (h1 - m.t - m.b),
          ),
        );
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, h1 - m.b);
        ctx.lineTo(w - m.r, h1 - m.b);
        ctx.stroke();
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        [-2, -1, 0, 1, 2].forEach((v) => ctx.fillText(String(v), X(v), h1 - m.b + 13));
        if (deterministicCurve) {
          const gi = Math.round(((tp.v[0] + 6) / 12) * (NG - 1));
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 4]);
          ctx.beginPath();
          ctx.moveTo(X(deterministicCurve[gi]), m.t);
          ctx.lineTo(X(deterministicCurve[gi]), h1 - m.b);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.strokeStyle = colors.ok;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(X(f(tp.v[0])), m.t);
          ctx.lineTo(X(f(tp.v[0])), h1 - m.b);
          ctx.stroke();
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText('dashed: ordinary answer', w - m.r, m.t + 12);
          ctx.fillText('red: the truth', w - m.r, m.t + 25);
        }
        const sds = Array.from({ length: NG }, (_, i) => sdAt(i)),
          U = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, sds, {
            title: 'disagreement between passes (±), at every x',
            n: NG,
            ym: Math.max(0.3, ...sds) * 1.1,
            col: colors.accent,
          });
        ctx.fillStyle = colors.ink;
        for (const q of D) {
          ctx.fillRect(U.X(((q.x + 6) / 12) * (NG - 1)) - 1, h - 17, 2, 7);
        }
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('ticks: where the training data is', w - 12, h - 3);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'mc', activeStepIndex);
      const gi = Math.round(((tp.v[0] + 6) / 12) * (NG - 1));
      tp.out.textContent = !deterministicCurve
        ? 'The network is still training.'
        : sampleCount < 2
          ? 'Ordinary prediction at x = ' +
            f2(tp.v[0]) +
            ': ' +
            f2(deterministicCurve[gi]) +
            ', with nothing to say how sure it is.'
          : 'At x = ' +
            f2(tp.v[0]) +
            ' the ordinary network says ' +
            f2(deterministicCurve[gi]) +
            '. The ' +
            sampleCount +
            ' dropout passes average ' +
            f2(sum[gi] / sampleCount) +
            ' and disagree by ±' +
            f2(sdAt(gi)) +
            ', so a 95% range is ' +
            f2(sum[gi] / sampleCount - 1.96 * sdAt(gi)) +
            ' to ' +
            f2(sum[gi] / sampleCount + 1.96 * sdAt(gi)) +
            '. The truth is ' +
            f2(f(tp.v[0])) +
            '.';
      let ins = 0;
      if (sampleCount > 1)
        for (const q of D) {
          const i = Math.round(((q.x + 6) / 12) * (NG - 1));
          if (Math.abs(q.y - sum[i] / sampleCount) <= 1.96 * sdAt(i) + 1e-9) ins++;
        }
      setReadout(
        root,
        'ph',
        ep < EPM
          ? 'Training (' + ep + ' of ' + EPM + ' epochs)'
          : done
            ? 'Finished'
            : sampleCount
              ? 'Polling with dropout on'
              : 'Trained',
      );
      setReadout(root, 'k', String(sampleCount));
      setReadout(
        root,
        'sd',
        sampleCount > 1
          ? f2((avg(-2.5, -settings.gap / 2 - 0.2) + avg(settings.gap / 2 + 0.2, 2.5)) / 2)
          : '–',
      );
      setReadout(
        root,
        'sg',
        sampleCount > 1 && settings.gap > 0.3
          ? f2(avg(-settings.gap / 2 + 0.1, settings.gap / 2 - 0.1))
          : '–',
      );
      setReadout(root, 'so', sampleCount > 1 ? f2((avg(-6, -4.5) + avg(4.5, 6)) / 2) : '–');
      setReadout(root, 'ins', sampleCount > 1 ? ins + ' of ' + D.length : '–');
    };
    tp.on = () => {
      probePredictions = [];
      if (deterministicCurve && sampleCount) {
        /* re-poll this x with fresh masks */ for (let q = 0; q < Math.min(sampleCount, 200); q++) {
          const mask = [0, 1].map(() =>
            Float64Array.from({ length: settings.hw }, () =>
              Math.random() < settings.dropoutProbability
                ? 0
                : 1 / (1 - settings.dropoutProbability),
            ),
          );
          probePredictions.push(net.fwd([tp.v[0]], mask).A[3][0]);
        }
      }
      draw();
    };
    gen();
    build();
    if (prefersReducedMotion) {
      tr(EPM);
      for (let i = 0; i < 80; i++) pass();
    }
    const redo = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      frameCount++;
      if (ep < EPM) {
        tr(12);
        draw();
        return;
      }
      if (frameCount % (sampleCount < 8 ? 34 : 5)) return;
      pass();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#mc-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'mc', {
      done: () => done,
      restart: () => {
        run++;
        build();
      },
      count: () => sampleCount,
      what: 'passes',
    });
    $('#mc-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#mc-reset', root).onclick = redo;
    bind(root, 'mc-p', (v) => {
      settings.dropoutProbability = v;
      redo();
    });
    bind(
      root,
      'mc-hw',
      (v) => {
        settings.hw = v;
        redo();
      },
      (v) => String(v),
    );
    bind(root, 'mc-gap', (v) => {
      settings.gap = v;
      gen();
      redo();
    });
    bind(root, 'mc-nz', (v) => {
      settings.nz = v;
      gen();
      redo();
    });
    $('#mc-new', root).onclick = () => {
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
