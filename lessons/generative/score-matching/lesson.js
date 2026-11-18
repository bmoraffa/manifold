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
  g2Data,
  g2Emb,
  fractionNearData,
  g2Note,
  g2Panel,
  g2Pred,
  g2Pts,
  g2Samp,
  g2Uni,
} from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#sm-c', root),
      cr = $('#sm-r', root),
      live = $('#sm-live', root),
      settings = { ds: 'blobs', lv: 4, mode: 'ann' },
      L = 10,
      SG = Array.from({ length: L }, (_, i) => 1.2 * Math.pow(0.05 / 1.2, i / (L - 1))),
      MAXS = 2500,
      M = 400,
      KS = 12,
      CC = 0.2,
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
      xs = null,
      si = 0,
      done = false,
      scoreAgreementByLevel = [],
      nearV = NaN,
      curL = 0,
      cachedExactScores = null,
      exactScoreCacheKey = '';
    const eps = (P, i) => g2Pred(net, P, g2Emb(i / (L - 1)));
    const exact = (P, sg) => {
      const n = P.length / 2,
        o = new Float64Array(2 * n),
        iv = 1 / (2 * sg * sg),
        d = new Float64Array(data.length);
      for (let i = 0; i < n; i++) {
        const x = P[2 * i],
          y = P[2 * i + 1];
        let mn = 1e99;
        for (let j = 0; j < data.length; j++) {
          d[j] = ((data[j][0] - x) ** 2 + (data[j][1] - y) ** 2) * iv;
          if (d[j] < mn) mn = d[j];
        }
        let W = 0,
          sx = 0,
          sy = 0;
        for (let j = 0; j < data.length; j++) {
          const w = Math.exp(mn - d[j]);
          W += w;
          sx += w * (data[j][0] - x);
          sy += w * (data[j][1] - y);
        }
        o[2 * i] = sx / W / (sg * sg);
        o[2 * i + 1] = sy / W / (sg * sg);
      }
      return o;
    };
    const exFor = (i) => {
      const k = run + '/' + i;
      if (exactScoreCacheKey !== k) {
        exactScoreCacheKey = k;
        cachedExactScores = exact(vectorFieldGrid, SG[i]);
      }
      return cachedExactScores;
    };
    const agree = (i) => {
      const e = eps(vectorFieldGrid, i),
        ex = exact(vectorFieldGrid, SG[i]),
        n = vectorFieldGrid.length / 2;
      let ok = 0;
      for (let k = 0; k < n; k++) {
        const ax = -e[2 * k],
          ay = -e[2 * k + 1],
          bx = ex[2 * k],
          by = ex[2 * k + 1];
        if ((ax * bx + ay * by) / (Math.hypot(ax, ay) * Math.hypot(bx, by) || 1) > 0.906) ok++;
      }
      return ok / n;
    };
    const build = () => {
      net = AD.mlp([10, 64, 64, 2], randnFrom(rng(run * 7 + 4)), 'relu');
      ts = 0;
      LL = [];
      sm = [];
      phase = 'train';
      xs = null;
      done = false;
      activeStepIndex = 0;
      scoreAgreementByLevel = [];
      nearV = NaN;
      si = 0;
      curL = 0;
      data = Array.from({ length: 500 }, () => g2Samp(settings.ds, RN));
      msg =
        'A network with random weights: its arrows point nowhere useful yet. The top-right picture shows what they should become. Press Run to train it.';
    };
    const train = (k) => {
      const B = 128;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(B * 10),
          E = new Float64Array(B * 2);
        for (let i = 0; i < B; i++) {
          const p = g2Samp(settings.ds, RN),
            lv = Math.floor(Math.random() * L),
            e0 = RN(),
            e1 = RN(),
            e = g2Emb(lv / (L - 1));
          X[i * 10] = p[0] + SG[lv] * e0;
          X[i * 10 + 1] = p[1] + SG[lv] * e1;
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
      activeStepIndex = Math.floor(ts / 100) % 2;
      msg =
        'Training step ' +
        ts +
        ' of ' +
        MAXS +
        ': 128 real points were given noise at random levels and the network guessed the way back for each. Average error over the last 20 steps: ' +
        f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
        '.';
      if (ts >= MAXS) {
        scoreAgreementByLevel = SG.map((_, i) => agree(i));
        phase = 'gen';
        startGen();
        activeStepIndex = 3;
        msg =
          'Training is finished. The bars at the bottom right compare the learned arrows with the exact ones at each level. Now generate: ' +
          M +
          ' points have been scattered at random.';
      }
    };
    const startGen = () => {
      xs = g2Uni(M);
      si = 0;
      done = false;
      nearV = NaN;
      curL = settings.mode === 'ann' ? 0 : settings.lv;
    };
    const gstep = () => {
      if (done || !xs) return;
      const i = settings.mode === 'ann' ? Math.min(L - 1, Math.floor(si / KS)) : settings.lv,
        sg = SG[i],
        e = eps(xs, i),
        a = (CC * sg) / 2,
        b = sg * Math.sqrt(CC);
      for (let j = 0; j < 2 * M; j++)
        xs[j] = clamp(xs[j] - a * e[j] + b * RN(), j % 2 ? -1.68 : -2.25, j % 2 ? 1.68 : 2.25);
      si++;
      curL = i;
      activeStepIndex = settings.mode === 'ann' && si % KS === 0 ? 5 : 4;
      msg =
        'Langevin step ' +
        si +
        ' of ' +
        L * KS +
        ', using the arrows for noise level ' +
        (i + 1) +
        ' (σ = ' +
        f2(sg) +
        '): every point moved a short way along its arrow and was jittered.' +
        (settings.mode === 'ann' && si % KS === 0 && si < L * KS
          ? ' Twelve steps are done at this level, so the next step moves to level ' + (i + 2) + '.'
          : '');
      if (si >= L * KS) {
        const e2 = eps(xs, i);
        for (let j = 0; j < 2 * M; j++) xs[j] -= sg * e2[j];
        done = true;
        nearV = fractionNearData(xs, M, data);
        let wall = 0;
        for (let j = 0; j < M; j++)
          if (Math.abs(xs[2 * j]) > 2.2 || Math.abs(xs[2 * j + 1]) > 1.63) wall++;
        msg =
          'Finished after ' +
          L * KS +
          ' Langevin steps' +
          (settings.mode === 'ann' ? ' through all ten levels' : ' at level ' + (i + 1) + ' only') +
          ', with the last of the jitter removed at the end: ' +
          pct(nearV) +
          ' of the points lie on the shape' +
          (wall ? ', and ' + wall + ' are stuck at the edge of the picture' : '') +
          '.' +
          (settings.mode === 'one' ? ' Compare with All Ten Levels.' : '');
      }
    };
    const step = () => {
      if (phase === 'train') train(50);
      else if (done) {
        startGen();
        activeStepIndex = 3;
        msg = 'A fresh scatter of random points. Press Run or One step.';
      } else gstep();
    };
    const draw = () => {
      const colors = readThemeColors(),
        lv =
          phase === 'gen' && xs && si > 0 && !done && settings.mode === 'ann' ? curL : settings.lv;
      {
        const V = uView(c),
          { g, h } = V;
        g2Data(g, V, colors, data);
        if (ts > 0) g2Arrows(g, V, eps(vectorFieldGrid, lv), colors.accent, -1, 0.6, 7);
        if (xs) g2Pts(g, V, colors, xs, M, colors.gold);
        g2Note(
          g,
          colors,
          h,
          (ts > 0 ? 'learned' : 'untrained') +
            ' arrows for noise level ' +
            (lv + 1) +
            ' of ' +
            L +
            ', σ = ' +
            f2(SG[lv]),
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.56,
          P = g2Panel(
            ctx,
            colors,
            0,
            0,
            w,
            h1,
            'the exact score at noise level ' + (lv + 1) + ', computed from the 500 data points',
          );
        g2Data(ctx, P, colors, data, 0.25, 1.5);
        g2Arrows(ctx, P, exFor(lv), colors.ok, 1, 0.7, 7 * SG[lv], 9);
        ctx.restore();
        if (phase === 'train')
          drawSeriesChart(ctx, colors, 0, h1, w, h - h1, sm, {
            title: 'error at guessing the noise (average of every 20 steps)',
            n: MAXS / 20,
            col: colors.ok,
            ym: Math.max(1.05, sm[0] || 1.05),
          });
        else {
          const y0 = h1,
            H = h - h1,
            m = { l: 44, r: 12, t: 26, b: 22 },
            bw = (w - m.l - m.r) / L;
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText('learned arrows within 25° of the exact ones, by noise level', 8, y0 + 13);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(m.l, y0 + H - m.b);
          ctx.lineTo(w - m.r, y0 + H - m.b);
          ctx.stroke();
          ctx.textAlign = 'right';
          ctx.fillText('100%', m.l - 5, y0 + m.t + 4);
          ctx.fillText('0%', m.l - 5, y0 + H - m.b + 4);
          scoreAgreementByLevel.forEach((v, i) => {
            const bh = v * (H - m.t - m.b);
            ctx.fillStyle = i === lv ? colors.gold : mix(colors.accent, colors.accent, 0, 0.75);
            ctx.fillRect(m.l + i * bw + 3, y0 + H - m.b - bh, bw - 6, bh);
            ctx.fillStyle = i === lv ? colors.cut : colors.muted;
            ctx.textAlign = 'center';
            ctx.fillText(String(i + 1), m.l + i * bw + bw / 2, y0 + H - 6);
            if (bh > 16) {
              ctx.fillStyle = colors.bg;
              ctx.fillText((100 * v).toFixed(0), m.l + i * bw + bw / 2, y0 + H - m.b - bh + 13);
            }
          });
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'sm', activeStepIndex);
      setReadout(
        root,
        'ph',
        phase === 'train' ? (ts ? 'Training' : 'Not started') : done ? 'Finished' : 'Generating',
      );
      setReadout(root, 'ts', ts + ' of ' + MAXS);
      setReadout(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
      setReadout(root, 'sg', lv + 1 + ' of ' + L + ', σ = ' + f2(SG[lv]));
      setReadout(root, 'ag', scoreAgreementByLevel.length ? pct(scoreAgreementByLevel[lv]) : '–');
      setReadout(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
    };
    build();
    if (prefersReducedMotion) {
      train(MAXS);
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
      if (frameCount % 3 === 0 && !done) gstep();
      draw();
      if (done) playback.stop();
    });
    playback = playBtn($('#sm-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'sm', {
      done: () => phase === 'gen' && done,
      restart: () => startGen(),
      count: () => ts,
      what: 'training steps',
    });
    const regen = () => {
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
    $('#sm-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#sm-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    $('#sm-ds', root).onchange = (e) => {
      settings.ds = e.target.value;
      playback.stop();
      run++;
      build();
      draw();
      playback.start();
    };
    $('#sm-gen', root).onclick = regen;
    segBind(root, 'sm-mode', settings.mode, (v) => {
      settings.mode = v;
      if (phase === 'gen') regen();
    });
    bind(
      root,
      'sm-lv',
      (v) => {
        settings.lv = +v - 1;
        if (!playback) return;
        if (phase === 'gen' && settings.mode === 'one') regen();
        else draw();
      },
      (v) => String(v),
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
