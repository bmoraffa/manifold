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
  f3,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import {
  G2S,
  ebE,
  ebLang,
  g2Data,
  fractionNearData,
  g2Note,
  g2Pts,
  g2Samp,
  g2Uni,
} from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { highlightLessonStep, drawSeriesChart, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#eb-c', root),
      cr = $('#eb-r', root),
      live = $('#eb-live', root),
      settings = { ds: 'moons', sl: 0.25 },
      MAXS = 800,
      NP = 96,
      M = 300,
      GW = 64,
      GH = 48,
      A = 0.03,
      SGM = 0.05,
      GENS = 200,
      RN = randnFrom(Math.random);
    let net,
      ts = 0,
      energyGapHistory = [],
      smoothedEnergyGapHistory = [],
      phase = 'train',
      data = [],
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      negativeParticles = null,
      xs = null,
      si = 0,
      done = false,
      EG = null,
      ED = null,
      emin = 0,
      emax = 1,
      ecap = 1,
      nearV = NaN,
      edm = NaN,
      esm = NaN,
      lastE = -99;
    const refreshE = () => {
      const G = new Float64Array(2 * GW * GH);
      for (let r = 0; r < GH; r++)
        for (let q = 0; q < GW; q++) {
          G[2 * (r * GW + q)] = (-4 + (8 * (q + 0.5)) / GW) / G2S;
          G[2 * (r * GW + q) + 1] = (3 - (6 * (r + 0.5)) / GH) / G2S;
        }
      EG = Float64Array.from(ebE(net, G, GW * GH));
      emin = Math.min(...EG);
      emax = Math.max(emin + 1e-6, ...EG);
      ecap = Math.max(emin + 1e-6, Float64Array.from(EG).sort()[Math.floor(0.5 * EG.length)]);
      const D = new Float64Array(2 * data.length);
      data.forEach((p, i) => {
        D[2 * i] = p[0];
        D[2 * i + 1] = p[1];
      });
      ED = Float64Array.from(ebE(net, D, data.length));
      lastE = ts;
    };
    const build = () => {
      net = AD.mlp([2, 48, 48, 1], randnFrom(rng(run * 3 + 5)), 'relu');
      ts = 0;
      energyGapHistory = [];
      smoothedEnergyGapHistory = [];
      phase = 'train';
      xs = null;
      done = false;
      activeStepIndex = 0;
      nearV = NaN;
      edm = NaN;
      esm = NaN;
      si = 0;
      data = Array.from({ length: 500 }, () => g2Samp(settings.ds, RN));
      negativeParticles = g2Uni(NP);
      refreshE();
      msg =
        'A network with random weights gives a landscape that has nothing to do with the data. The small dots are the model’s own points, scattered at random. Press Run.';
    };
    const train = (k) => {
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(2 * NP);
        for (let i = 0; i < NP; i++) {
          const p = g2Samp(settings.ds, RN);
          X[2 * i] = p[0] + 0.03 * RN();
          X[2 * i + 1] = p[1] + 0.03 * RN();
        }
        for (let i = 0; i < NP; i++)
          if (Math.random() < 0.05) {
            negativeParticles[2 * i] = (Math.random() * 2 - 1) * 2.2;
            negativeParticles[2 * i + 1] = (Math.random() * 2 - 1) * 1.65;
          }
        ebLang(net, negativeParticles, NP, 5, A, SGM, RN);
        const Ed = net.fwd(AD.input(X, [NP, 2])),
          Es = net.fwd(AD.input(negativeParticles, [NP, 2])),
          l = AD.ebl(Ed, Es, 0.1);
        let md = 0,
          ms = 0;
        for (let i = 0; i < NP; i++) {
          md += Ed.v[i] / NP;
          ms += Es.v[i] / NP;
        }
        AD.backward(l);
        AD.adam(net.ps, 0.003, ++ts, 1);
        energyGapHistory.push(ms - md);
        edm = md;
        esm = ms;
        if (ts % 10 === 0) {
          let s = 0;
          for (let j = ts - 10; j < ts; j++) s += energyGapHistory[j];
          smoothedEnergyGapHistory.push(s / 10);
        }
      }
      if (ts - lastE >= 8 || ts >= MAXS) refreshE();
      activeStepIndex = 1 + (Math.floor(ts / 60) % 3);
      msg =
        'Training step ' +
        ts +
        ' of ' +
        MAXS +
        ': the model’s 96 points rolled downhill for five small steps. Then the landscape was lowered under 96 real points and raised under the model’s points. Average energy: ' +
        f3(edm) +
        ' at the real points, ' +
        f3(esm) +
        ' at the model’s.';
      if (ts >= MAXS) {
        phase = 'gen';
        startGen();
        activeStepIndex = 4;
        msg =
          'Training is finished and the landscape is fixed. ' +
          M +
          ' new points have been scattered at random. Press Run to let them roll.';
      }
    };
    const startGen = () => {
      xs = g2Uni(M);
      si = 0;
      done = false;
      nearV = NaN;
    };
    const gstep = (k) => {
      if (done || !xs) return;
      ebLang(net, xs, M, k, A, SGM, RN);
      si += k;
      activeStepIndex = 4;
      msg =
        'Rolling step ' +
        si +
        ' of ' +
        GENS +
        ': every point moved a short way downhill and was jittered.';
      if (si >= GENS) {
        ebLang(net, xs, M, 20, A, 0, RN);
        done = true;
        activeStepIndex = 5;
        nearV = fractionNearData(xs, M, data);
        msg =
          'Finished: ' +
          GENS +
          ' downhill steps with jitter, then 20 without. ' +
          pct(nearV) +
          ' of the points came to rest on the shape. They all began scattered at random.';
      }
    };
    const step = () => {
      if (phase === 'train') train(20);
      else if (done) {
        startGen();
        activeStepIndex = 4;
        msg = 'A fresh scatter of random points. Press Run or One step.';
      } else gstep(10);
    };
    const tp = testPoint(
      root,
      'eb',
      ['Your x', 'Your y'],
      [0, 0],
      [
        [-2.2, 2.2],
        [-1.6, 1.6],
      ],
    );
    const draw = () => {
      const colors = readThemeColors();
      {
        const V = uView(c),
          { g, w, h } = V,
          cw = w / GW,
          ch = h / GH;
        if (EG)
          for (let r = 0; r < GH; r++)
            for (let q = 0; q < GW; q++) {
              const u = 1 - clamp((EG[r * GW + q] - emin) / (ecap - emin), 0, 1);
              if (u < 0.03) continue;
              g.fillStyle = mix(colors.accent, colors.accent, 0, 0.8 * Math.pow(u, 1.2));
              const xa = Math.round(V.X(-4 + (8 * q) / GW)),
                xb = Math.round(V.X(-4 + (8 * (q + 1)) / GW)),
                ya = Math.round(V.Y(3 - (6 * r) / GH)),
                yb = Math.round(V.Y(3 - (6 * (r + 1)) / GH));
              g.fillRect(xa, ya, xb - xa, yb - ya);
            }
        g.setLineDash([6, 5]);
        g.strokeStyle = colors.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(V.X(-4), V.Y(settings.sl * G2S));
        g.lineTo(V.X(4), V.Y(settings.sl * G2S));
        g.stroke();
        g.setLineDash([]);
        g2Data(g, V, colors, data, 0.4);
        if (phase === 'train' && negativeParticles)
          g2Pts(g, V, colors, negativeParticles, NP, colors.cut, 2.4);
        if (xs) g2Pts(g, V, colors, xs, M, colors.gold);
        markTP(g, V.X(tp.v[0] * G2S), V.Y(tp.v[1] * G2S), colors);
        g2Note(
          g,
          colors,
          h,
          phase === 'train'
            ? 'small dots: the model’s own 96 points'
            : 'gold dots: ' + M + ' generated points',
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          NPF = 70,
          F = new Float64Array(2 * NPF);
        for (let i = 0; i < NPF; i++) {
          F[2 * i] = -2.25 + (4.5 * i) / (NPF - 1);
          F[2 * i + 1] = settings.sl;
        }
        const pe = Array.from(ebE(net, F, NPF)),
          lo = Math.min(...pe),
          hi = Math.max(lo + 1e-6, ...pe);
        const Lp = drawSeriesChart(ctx, colors, 0, 0, w, h1, pe, {
          title: 'energy along the dashed line: valleys are where the model expects data',
          col: colors.accent,
          y1: lo,
          ym: hi,
        });
        ctx.fillStyle = colors.ink;
        for (const p of data)
          if (Math.abs(p[1] - settings.sl) < 0.07) {
            const x = Lp.X(((p[0] + 2.25) / 4.5) * (NPF - 1));
            ctx.fillRect(x - 0.8, h1 - Lp.m.b - 7, 1.6, 7);
          }
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('ticks: real data near the line', w - Lp.m.r, h1 - 5);
        const G2 = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, smoothedEnergyGapHistory, {
          title: 'energy: model’s points minus real points (average of 10 steps)',
          n: MAXS / 10,
          col: colors.ok,
          y1: Math.min(-0.05, ...smoothedEnergyGapHistory),
          ym: Math.max(0.1, ...smoothedEnergyGapHistory),
        });
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(G2.m.l, G2.Y(0));
        ctx.lineTo(w - G2.m.r, G2.Y(0));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colors.cut;
        ctx.textAlign = 'right';
        ctx.fillText('dashed line: zero, where samples look like data', w - G2.m.r, h - G2.m.b - 7);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'eb', activeStepIndex);
      {
        const e = ebE(net, Float64Array.of(tp.v[0], tp.v[1]), 1)[0];
        let lo = 0;
        if (ED) for (const v of ED) if (v < e) lo++;
        tp.out.textContent = ts
          ? 'The energy at your point is ' +
            f3(e) +
            '. ' +
            (ED ? ((100 * lo) / ED.length).toFixed(0) : '–') +
            '% of the real data points have lower energy than that. On this landscape the lowest energy anywhere in view is ' +
            f3(emin) +
            ' and the highest is ' +
            f3(emax) +
            '.'
          : 'Train the model first: press Run.';
      }
      setReadout(
        root,
        'ph',
        phase === 'train' ? (ts ? 'Training' : 'Not started') : done ? 'Finished' : 'Generating',
      );
      setReadout(root, 'ts', ts + ' of ' + MAXS);
      setReadout(root, 'ed', isFinite(edm) ? f3(edm) : '–');
      setReadout(root, 'es', isFinite(esm) ? f3(esm) : '–');
      setReadout(root, 'gs', phase === 'gen' ? Math.min(si, GENS) + ' of ' + GENS : '–');
      setReadout(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
    };
    tp.on = () => draw();
    build();
    if (prefersReducedMotion) {
      train(MAXS);
      while (!done) gstep(10);
    }
    const anim = Anim(() => {
      frameCount++;
      if (phase === 'train') {
        train(Math.min(ts < 30 ? 1 : 2, checkRunStop.remaining()));
        draw();
        checkRunStop();
        return;
      }
      if (!done) gstep(1);
      draw();
      if (done) playback.stop();
    });
    playback = playBtn($('#eb-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'eb', {
      done: () => phase === 'gen' && done,
      restart: () => startGen(),
      count: () => ts,
      what: 'training steps',
    });
    $('#eb-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#eb-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    $('#eb-ds', root).onchange = (e) => {
      settings.ds = e.target.value;
      playback.stop();
      run++;
      build();
      draw();
      playback.start();
    };
    $('#eb-gen', root).onclick = () => {
      if (phase !== 'gen') {
        msg = 'Points can be generated once training is finished. Press Run.';
        draw();
        return;
      }
      playback.stop();
      startGen();
      draw();
      playback.start();
    };
    bind(
      root,
      'eb-sl',
      (v) => {
        settings.sl = +v;
        if (playback) draw();
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
