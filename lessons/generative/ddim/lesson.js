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
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import {
  drawSamplerBenchmark,
  g2Data,
  g2Emb,
  g2Levels,
  fractionNearData,
  g2Note,
  g2Panel,
  g2Pred,
  g2Pts,
  g2Samp,
  g2Trails,
} from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { highlightLessonStep, drawSeriesChart, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#dm-c', root),
      cr = $('#dm-r', root),
      live = $('#dm-live', root),
      settings = { ds: 'ring', ni: 6, eta: 0 },
      Tn = 100,
      MAXS = 2500,
      M = 400,
      NS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100],
      BE = Array.from({ length: Tn }, (_, i) => 0.0008 + (0.0792 * i) / (Tn - 1)),
      AB = [],
      RN = randnFrom(Math.random);
    {
      let p = 1;
      for (const b of BE) {
        p *= 1 - b;
        AB.push(p);
      }
    }
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
      x0h = null,
      seq = [],
      si = 0,
      done = false,
      tr = [],
      previousSamples = null,
      moved = NaN,
      nearDataRatesBySampler = [[], []],
      benchmarkIndex = 0,
      nearV = NaN;
    const ab = (k) => (k > 0 ? AB[k - 1] : 1),
      pred = (P, k) => g2Pred(net, P, g2Emb(k / Tn));
    const build = () => {
      net = AD.mlp([10, 64, 64, 2], randnFrom(rng(run * 7 + 4)), 'relu');
      ts = 0;
      LL = [];
      sm = [];
      phase = 'train';
      xs = null;
      x0h = null;
      done = false;
      activeStepIndex = 0;
      nearDataRatesBySampler = [NS.map(() => NaN), NS.map(() => NaN)];
      benchmarkIndex = 0;
      z0 = null;
      previousSamples = null;
      moved = NaN;
      nearV = NaN;
      tr = [];
      si = 0;
      data = Array.from({ length: 500 }, () => g2Samp(settings.ds, RN));
      msg =
        'A network with random weights. Press Run: it trains first, then the page tries it with different numbers of sampler steps.';
    };
    const train = (k) => {
      const B = 128;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(B * 10),
          E = new Float64Array(B * 2);
        for (let i = 0; i < B; i++) {
          const p = g2Samp(settings.ds, RN),
            tt = 1 + Math.floor(Math.random() * Tn),
            e0 = RN(),
            e1 = RN(),
            s = Math.sqrt(ab(tt)),
            nq = Math.sqrt(1 - ab(tt)),
            e = g2Emb(tt / Tn);
          X[i * 10] = s * p[0] + nq * e0;
          X[i * 10 + 1] = s * p[1] + nq * e1;
          for (let j = 0; j < 8; j++) X[i * 10 + 2 + j] = e[j];
          E[2 * i] = e0;
          E[2 * i + 1] = e1;
        }
        const l = AD.mse(net.fwd(AD.input(X, [B, 10])), E);
        LL.push(l.v[0]);
        AD.backward(l);
        AD.adam(net.ps, 0.003 * Math.pow(0.1, ts / MAXS), ++ts, 1);
        if (ts % 20 === 0) {
          let s = 0;
          for (let j = ts - 20; j < ts; j++) s += LL[j];
          sm.push(s / 20);
        }
      }
      activeStepIndex = 0;
      msg =
        'Training step ' +
        ts +
        ' of ' +
        MAXS +
        ': 128 real points were given random amounts of noise, from level 1 to level ' +
        Tn +
        ', and the network guessed the noise in each. Average error over the last 20 steps: ' +
        f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
        '.';
      if (ts >= MAXS) {
        phase = 'bench';
        benchmarkIndex = 0;
        activeStepIndex = 1;
        msg =
          'Training is finished and the network will not change again. Next it is used with eleven different numbers of sampler steps, with and without fresh randomness.';
      }
    };
    const mkSeq = (N) => {
      const s = [];
      for (let j = N; j >= 0; j--) s.push(Math.round((j * Tn) / N));
      return s;
    };
    const sstep = (x, k, kp, eta) => {
      const e = pred(x, k),
        a = ab(k),
        ap = ab(kp),
        sa = Math.sqrt(a),
        sn = Math.sqrt(1 - a),
        sig = kp > 0 ? eta * Math.sqrt((1 - ap) / (1 - a)) * Math.sqrt(1 - a / ap) : 0,
        dir = Math.sqrt(Math.max(0, 1 - ap - sig * sig)),
        sp = Math.sqrt(ap),
        h = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) {
        const x0 = clamp((x[i] - sn * e[i]) / sa, -2.5, 2.5),
          e2 = (x[i] - sa * x0) / sn;
        h[i] = x0;
        x[i] = sp * x0 + dir * e2 + (sig ? sig * RN() : 0);
      }
      return h;
    };
    const sampleAll = (N, eta, Mm) => {
      const x = Float64Array.from({ length: 2 * Mm }, () => RN()),
        s = mkSeq(N);
      for (let j = 0; j < N; j++) sstep(x, s[j], s[j + 1], eta);
      return fractionNearData(x, Mm, data, 0.1);
    };
    const startGen = (fresh) => {
      if (fresh || !z0) {
        z0 = Float64Array.from({ length: 2 * M }, () => RN());
        previousSamples = null;
        moved = NaN;
      }
      xs = Float64Array.from(z0);
      x0h = null;
      seq = mkSeq(NS[settings.ni]);
      si = 0;
      done = false;
      nearV = NaN;
      tr = Array.from({ length: 14 }, (_, i) => [[xs[2 * i], xs[2 * i + 1]]]);
    };
    const bench = () => {
      const j = benchmarkIndex >> 1,
        e = benchmarkIndex & 1;
      nearDataRatesBySampler[e][j] = sampleAll(NS[j], e, 200);
      benchmarkIndex++;
      activeStepIndex = 1;
      msg =
        'Measuring: ' +
        NS[j] +
        ' step' +
        (NS[j] > 1 ? 's' : '') +
        ' with η = ' +
        e +
        (e ? ' (fresh noise at every step, the DDPM rule)' : ' (no fresh noise, the DDIM rule)') +
        ' put ' +
        pct(nearDataRatesBySampler[e][j]) +
        ' of 200 new points on the shape.';
      if (benchmarkIndex >= 2 * NS.length) {
        phase = 'gen';
        startGen(true);
        activeStepIndex = 2;
        msg =
          'All settings are measured: see the chart at the bottom right. Now watch one run. ' +
          M +
          ' points of pure noise will visit these levels: ' +
          seq.join(', ') +
          '.';
      }
    };
    const gstep = () => {
      if (done || !xs) return;
      const k = seq[si],
        kp = seq[si + 1];
      x0h = sstep(xs, k, kp, settings.eta);
      si++;
      tr.forEach((p, i) => p.push([xs[2 * i], xs[2 * i + 1]]));
      activeStepIndex = 4;
      msg =
        'Step ' +
        si +
        ' of ' +
        NS[settings.ni] +
        ': the network guessed the noise at level ' +
        k +
        '. Removing it gave the guessed finished points (top right). They were put back at level ' +
        kp +
        (kp === 0
          ? ', which is no noise at all.'
          : settings.eta > 0
            ? ', using the guessed noise mixed with fresh random noise.'
            : ', using the same guessed noise and nothing random.');
      if (si >= seq.length - 1) {
        done = true;
        activeStepIndex = 5;
        nearV = fractionNearData(xs, M, data, 0.1);
        if (previousSamples) {
          let d = 0;
          for (let i = 0; i < M; i++)
            d += Math.hypot(
              xs[2 * i] - previousSamples[2 * i],
              xs[2 * i + 1] - previousSamples[2 * i + 1],
            );
          moved = d / M;
        }
        previousSamples = Float64Array.from(xs);
        msg +=
          ' Finished after ' +
          si +
          ' call' +
          (si > 1 ? 's' : '') +
          ' of the network: ' +
          pct(nearV) +
          ' of the points lie on the shape.' +
          (isFinite(moved)
            ? ' Compared with the last run from the same starting noise, each point ended ' +
              f2(moved) +
              ' away on average.'
            : '');
      }
    };
    const step = () => {
      if (phase === 'train') train(50);
      else if (phase === 'bench') bench();
      else if (done) {
        startGen(false);
        activeStepIndex = 2;
        msg = 'The same starting noise again. Press Run or One step.';
      } else gstep();
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const V = uView(c),
          { g, h } = V;
        g2Data(g, V, colors, data);
        if (xs) {
          g2Trails(g, V, colors, tr);
          g2Pts(g, V, colors, xs, M, colors.gold);
        }
        g2Note(
          g,
          colors,
          h,
          phase === 'gen'
            ? NS[settings.ni] +
                ' step' +
                (NS[settings.ni] > 1 ? 's' : '') +
                ', η = ' +
                f2(settings.eta) +
                (xs ? ', now at noise level ' + seq[si] : '')
            : phase === 'bench'
              ? 'measuring each number of steps'
              : 'training the network',
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.52;
        if (phase === 'gen' && x0h) {
          const P = g2Panel(
            ctx,
            colors,
            0,
            0,
            w,
            h1,
            'the network’s guess of where each point will finish, at this step',
            1.22,
          );
          g2Data(ctx, P, colors, data, 0.22, 1.5);
          g2Pts(ctx, P, colors, x0h, M, colors.accent, 2);
          ctx.restore();
        } else
          g2Levels(
            ctx,
            colors,
            w,
            h1,
            data,
            [0, 25, 50, 100].map((k) => ({
              a: Math.sqrt(ab(k)),
              s: Math.sqrt(1 - ab(k)),
              lab: 'level ' + k,
            })),
            'the real data at four of the 100 noise levels: what training sees',
          );
        if (phase === 'train')
          drawSeriesChart(ctx, colors, 0, h1, w, h - h1, sm, {
            title: 'error at guessing the noise (average of every 20 steps)',
            n: MAXS / 20,
            col: colors.ok,
            ym: Math.max(1.05, sm[0] || 1.05),
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
                ys: nearDataRatesBySampler[1],
                col: colors.cut,
                lab: 'η = 1: fresh noise each step',
              },
              { ys: nearDataRatesBySampler[0], col: colors.accent, lab: 'η = 0: DDIM' },
            ],
            phase === 'gen' ? settings.ni : -1,
            'share of points on the shape, by number of sampler steps',
          );
      }
      live.textContent = msg;
      highlightLessonStep(root, 'dm', activeStepIndex);
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
      setReadout(root, 'nc', phase === 'gen' ? si + ' of ' + NS[settings.ni] : '–');
      setReadout(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
      setReadout(root, 'mv', isFinite(moved) && done ? f2(moved) : '–');
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
      if (frameCount % clamp(Math.round(220 / NS[settings.ni]), 4, 40) === 0 && !done) gstep();
      draw();
      if (done) playback.stop();
    });
    playback = playBtn($('#dm-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'dm', {
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
    $('#dm-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#dm-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    $('#dm-ds', root).onchange = (e) => {
      settings.ds = e.target.value;
      playback.stop();
      run++;
      build();
      draw();
      playback.start();
    };
    $('#dm-same', root).onclick = () => regen(false);
    $('#dm-new', root).onclick = () => regen(true);
    bind(
      root,
      'dm-n',
      (v) => {
        settings.ni = +v;
        if (playback && phase === 'gen') regen(false);
      },
      (v) => String(NS[+v]),
    );
    bind(
      root,
      'dm-eta',
      (v) => {
        settings.eta = +v;
        if (playback && phase === 'gen') regen(false);
      },
      f2,
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
