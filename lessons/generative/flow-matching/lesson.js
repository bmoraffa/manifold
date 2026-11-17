/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import {
  vectorFieldGrid,
  g2Arrows,
  drawSamplerBenchmark,
  g2Data,
  g2Emb,
  fractionNearData,
  g2Note,
  g2Panel,
  g2Pred,
  g2Pts,
  g2Samp,
  g2Trails,
} from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#fm-c', root),
      cr = $('#fm-r', root),
      live = $('#fm-live', root),
      settings = { ds: 'ring', ni: 5, sol: 'euler' },
      MAXS = 2500,
      M = 400,
      NS = [1, 2, 3, 4, 6, 8, 12, 20, 40],
      RN = randnFrom(Math.random);
    let net,
      ts = 0,
      LL = [],
      sm = [],
      phase = 'train',
      data = [],
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      z0 = null,
      xs = null,
      si = 0,
      done = false,
      tr = [],
      nearDataRatesBySolver = [[], []],
      bi = 0,
      nearV = NaN,
      samplePathLengths = null,
      calls = 0,
      meanPathStraightness = NaN,
      pairs = [];
    const pred = (P, u) => g2Pred(net, P, g2Emb(u));
    const build = () => {
      net = AD.mlp([10, 64, 64, 2], randnFrom(rng(run * 7 + 4)), 'relu');
      ts = 0;
      LL = [];
      sm = [];
      phase = 'train';
      xs = null;
      done = false;
      activeStepIndex = 0;
      nearDataRatesBySolver = [NS.map(() => NaN), NS.map(() => NaN)];
      bi = 0;
      z0 = null;
      nearV = NaN;
      meanPathStraightness = NaN;
      tr = [];
      si = 0;
      calls = 0;
      data = Array.from({ length: 500 }, () => g2Samp(settings.ds, RN));
      pairs = Array.from({ length: 9 }, (_, i) => ({ a: [RN(), RN()], b: data[i] }));
      msg = 'A network with random weights: its arrows mean nothing yet. Press Run to train it.';
    };
    const train = (k) => {
      const B = 128;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(B * 10),
          E = new Float64Array(B * 2);
        for (let i = 0; i < B; i++) {
          const p = g2Samp(settings.ds, RN),
            u = Math.pow(Math.random(), 0.6),
            a0 = RN(),
            a1 = RN(),
            e = g2Emb(u);
          X[i * 10] = (1 - u) * a0 + u * p[0];
          X[i * 10 + 1] = (1 - u) * a1 + u * p[1];
          for (let j = 0; j < 8; j++) X[i * 10 + 2 + j] = e[j];
          E[2 * i] = p[0] - a0;
          E[2 * i + 1] = p[1] - a1;
        }
        const l = AD.mse(net.fwd(AD.input(X, [B, 10])), E);
        LL.push(l.v[0]);
        AD.backward(l);
        AD.adam(net.ps, 0.004 * Math.pow(0.1, ts / MAXS), ++ts, 1);
        if (ts % 20 === 0) {
          let s = 0;
          for (let j = ts - 20; j < ts; j++) s += LL[j];
          sm.push(s / 20);
        }
      }
      activeStepIndex = Math.floor(ts / 100) % 3;
      msg =
        'Training step ' +
        ts +
        ' of ' +
        MAXS +
        ': 128 noise points were each joined to a random real point. The network was asked for the velocity at a random place along each line and compared with the true one. Average error over the last 20 steps: ' +
        f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
        '.';
      if (ts >= MAXS) {
        phase = 'bench';
        bi = 0;
        activeStepIndex = 3;
        msg =
          'Training is finished. Next the network is used with nine different numbers of steps and with both solvers.';
      }
    };
    const ode = (x, j, N, sol) => {
      const dt = 1 / N,
        u = j / N,
        v = pred(x, u);
      if (sol === 'mid') {
        const xm = x.map((q, i) => q + (v[i] * dt) / 2),
          v2 = pred(xm, u + dt / 2);
        for (let i = 0; i < x.length; i++) x[i] += v2[i] * dt;
      } else for (let i = 0; i < x.length; i++) x[i] += v[i] * dt;
    };
    const sampleAll = (N, sol, Mm) => {
      const x = Float64Array.from({ length: 2 * Mm }, () => RN());
      for (let j = 0; j < N; j++) ode(x, j, N, sol);
      return fractionNearData(x, Mm, data, 0.1);
    };
    const startGen = (fresh) => {
      if (fresh || !z0) z0 = Float64Array.from({ length: 2 * M }, () => RN());
      xs = Float64Array.from(z0);
      si = 0;
      done = false;
      nearV = NaN;
      meanPathStraightness = NaN;
      calls = 0;
      samplePathLengths = new Float64Array(M);
      tr = Array.from({ length: 14 }, (_, i) => [[xs[2 * i], xs[2 * i + 1]]]);
    };
    const bench = () => {
      const j = bi >> 1,
        e = bi & 1;
      nearDataRatesBySolver[e][j] = sampleAll(NS[j], e ? 'mid' : 'euler', 200);
      bi++;
      activeStepIndex = 3;
      msg =
        'Measuring: ' +
        NS[j] +
        ' step' +
        (NS[j] > 1 ? 's' : '') +
        ' of ' +
        (e
          ? 'Midpoint (' + 2 * NS[j] + ' network calls)'
          : 'Euler (' + NS[j] + ' network call' + (NS[j] > 1 ? 's' : '') + ')') +
        ' put ' +
        pct(nearDataRatesBySolver[e][j]) +
        ' of 200 new points on the shape.';
      if (bi >= 2 * NS.length) {
        phase = 'gen';
        startGen(true);
        activeStepIndex = 4;
        msg =
          'All settings are measured: see the chart at the bottom right. Now watch one run: ' +
          M +
          ' points of pure noise at time 0.';
      }
    };
    const gstep = () => {
      if (done || !xs) return;
      const N = NS[settings.ni],
        o = Float64Array.from(xs);
      ode(xs, si, N, settings.sol);
      si++;
      calls += settings.sol === 'mid' ? 2 : 1;
      for (let i = 0; i < M; i++)
        samplePathLengths[i] += Math.hypot(xs[2 * i] - o[2 * i], xs[2 * i + 1] - o[2 * i + 1]);
      tr.forEach((p, i) => p.push([xs[2 * i], xs[2 * i + 1]]));
      activeStepIndex = 4;
      msg =
        'Step ' +
        si +
        ' of ' +
        N +
        ': the network gave a velocity for every point at time ' +
        f2((si - 1) / N) +
        (settings.sol === 'mid' ? ', each point looked again from half a step ahead,' : '') +
        ' and every point moved along it for a time of ' +
        f2(1 / N) +
        '. The time is now ' +
        f2(si / N) +
        '.';
      if (si >= N) {
        done = true;
        activeStepIndex = 5;
        nearV = fractionNearData(xs, M, data, 0.1);
        let s = 0;
        for (let i = 0; i < M; i++)
          s +=
            Math.hypot(xs[2 * i] - z0[2 * i], xs[2 * i + 1] - z0[2 * i + 1]) /
            (samplePathLengths[i] || 1);
        meanPathStraightness = s / M;
        msg +=
          ' Arrived: ' +
          pct(nearV) +
          ' of the points lie on the shape, after ' +
          calls +
          ' call' +
          (calls > 1 ? 's' : '') +
          ' of the network.';
      }
    };
    const step = () => {
      if (phase === 'train') train(50);
      else if (phase === 'bench') bench();
      else if (done) {
        startGen(false);
        activeStepIndex = 4;
        msg = 'The same starting noise again, at time 0. Press Run or One step.';
      } else gstep();
    };
    const draw = () => {
      const colors = readThemeColors(),
        N = NS[settings.ni],
        u = phase === 'gen' ? Math.min(1, si / N) : (Math.floor(frameCount / 3) % 100) / 100;
      {
        const V = uView(c),
          { g, h } = V;
        g2Data(g, V, colors, data);
        if (ts > 0)
          g2Arrows(g, V, pred(vectorFieldGrid, Math.min(u, 0.999)), colors.accent, 1, 0.55, 6);
        if (xs) {
          g2Trails(g, V, colors, tr);
          g2Pts(g, V, colors, xs, M, colors.gold);
        }
        g2Note(
          g,
          colors,
          h,
          (ts > 0 ? 'arrows shown for time ' + f2(u) : 'not trained yet') +
            (phase === 'gen'
              ? ', ' +
                N +
                ' step' +
                (N > 1 ? 's' : '') +
                ' of ' +
                (settings.sol === 'mid' ? 'Midpoint' : 'Euler')
              : ''),
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          P = g2Panel(
            ctx,
            colors,
            0,
            0,
            w,
            h1,
            'training pairs at time ' + f2(u) + ': the arrow is the right answer at the gold dot',
          );
        for (const p of pairs) {
          const ax = P.X(p.a[0]),
            ay = P.Y(p.a[1]),
            bx = P.X(p.b[0]),
            by = P.Y(p.b[1]),
            mx = ax + (bx - ax) * u,
            my = ay + (by - ay) * u,
            d = Math.hypot(bx - ax, by - ay) || 1;
          ctx.strokeStyle = mix(colors.ink, colors.ink, 0, 0.3);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.fillStyle = colors.bg;
          ctx.strokeStyle = colors.ink;
          ctx.beginPath();
          ctx.arc(ax, ay, 3.5, 0, 7);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = colors.ink;
          ctx.beginPath();
          ctx.arc(bx, by, 3.5, 0, 7);
          ctx.fill();
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(mx, my);
          ctx.lineTo(mx + ((bx - ax) / d) * 16, my + ((by - ay) / d) * 16);
          ctx.stroke();
          ctx.fillStyle = colors.gold;
          ctx.beginPath();
          ctx.arc(mx, my, 4, 0, 7);
          ctx.fill();
        }
        ctx.restore();
        if (phase === 'train')
          drawSeriesChart(ctx, colors, 0, h1, w, h - h1, sm, {
            title: 'error at guessing the velocity (average of every 20 steps)',
            n: MAXS / 20,
            col: colors.ok,
            ym: Math.max(2.2, sm[0] || 2.2),
          });
        else
          drawSamplerBenchmark(
            ctx,
            colors,
            0,
            h1,
            w,
            h - h1,
            NS,
            [
              {
                ys: nearDataRatesBySolver[1],
                col: colors.cut,
                lab: 'Midpoint: two calls per step',
              },
              { ys: nearDataRatesBySolver[0], col: colors.accent, lab: 'Euler: one call per step' },
            ],
            phase === 'gen' ? settings.ni : -1,
            'share of points on the shape, by number of steps',
          );
      }
      live.textContent = msg;
      highlightLessonStep(root, 'fm', activeStepIndex);
      setReadout(
        root,
        'ph',
        phase === 'train'
          ? ts
            ? 'Training'
            : 'Not started'
          : phase === 'bench'
            ? 'Measuring'
            : done
              ? 'Finished'
              : 'Generating',
      );
      setReadout(root, 'ts', ts + ' of ' + MAXS);
      setReadout(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
      setReadout(root, 'nc', phase === 'gen' ? String(calls) : '–');
      setReadout(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
      setReadout(root, 'st', isFinite(meanPathStraightness) ? f2(meanPathStraightness) : '–');
    };
    build();
    if (prefersReducedMotion) {
      train(MAXS);
      while (phase === 'bench') bench();
      while (!done) gstep();
    }
    const anim = Anim(() => {
      frameCount++;
      if (phase === 'train') {
        train(Math.min(ts < 60 ? 1 : 5, checkRunStop.remaining()));
        draw();
        checkRunStop();
        return;
      }
      if (phase === 'bench') {
        if (frameCount % 4 === 0) {
          bench();
          draw();
        }
        return;
      }
      if (frameCount % clamp(Math.round(200 / NS[settings.ni]), 5, 40) === 0 && !done) gstep();
      draw();
      if (done) playback.stop();
    });
    playback = playBtn($('#fm-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'fm', {
      done: () => phase === 'gen' && done,
      restart: () => startGen(false),
      count: () => ts,
      what: 'training steps',
    });
    const regen = (fresh) => {
      if (phase !== 'gen') {
        msg = 'The sampler is used once training and measuring are finished. Press Run.';
        draw();
        return;
      }
      playback.stop();
      startGen(fresh);
      draw();
      playback.start();
    };
    $('#fm-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#fm-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    $('#fm-ds', root).onchange = (e) => {
      settings.ds = e.target.value;
      playback.stop();
      run++;
      build();
      draw();
      playback.start();
    };
    $('#fm-same', root).onclick = () => regen(false);
    $('#fm-new', root).onclick = () => regen(true);
    bind(
      root,
      'fm-n',
      (v) => {
        settings.ni = +v;
        if (playback && phase === 'gen') regen(false);
      },
      (v) => String(NS[+v]),
    );
    segBind(root, 'fm-sol', settings.sol, (v) => {
      settings.sol = v;
      if (phase === 'gen') regen(false);
    });
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
