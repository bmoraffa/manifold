/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  bind,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { dot, lsq, splineBasis } from '../shared.js';

export default {
  init(root) {
    const c1 = $('#am-1', root),
      c2 = $('#am-2', root),
      cap = $('#am-cap', root),
      settings = { K: 4, noise: 0.3, r: 0, n: 160, f: 'a' };
    const TF = {
        a: [(x) => Math.sin(3 * x), (x) => 2 * x * x - 2 / 3],
        b: [(x) => 2 * Math.abs(x) - 1, (x) => 2 * x * x * x],
        c: [(x) => 0, (x) => 0],
      },
      gen2 = (a, b) =>
        settings.f === 'c' ? 2.2 * a * b : TF[settings.f][0](a) + TF[settings.f][1](b);
    let seed = 6,
      pts = [],
      al = 0,
      B = [null, null],
      off = [0, 0],
      G = [new Array(61).fill(0), new Array(61).fill(0)],
      previousCurveSamples = [G[0].slice(), G[1].slice()],
      curveAnimationProgress = 1,
      turn = 0,
      wait = 0,
      frameCount = 0,
      linearTestMse = 0,
      delta = 1,
      playback;
    const fj = (j, x) => (B[j] ? dot(B[j], splineBasis('natural', x, settings.K)) - off[j] : 0);
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r),
        q = Math.sqrt(1 - settings.r * settings.r);
      pts = [];
      for (let i = 0; i < settings.n; i++) {
        const a = r() * 2 - 1,
          b = clamp(settings.r * a + q * (r() * 2 - 1), -1, 1);
        pts.push({ x: [a, b], y: gen2(a, b) + settings.noise * rn(), test: r() < 0.3 });
      }
      const tr = pts.filter((p) => !p.test),
        te = pts.filter((p) => p.test);
      al = tr.reduce((s, p) => s + p.y, 0) / tr.length;
      const bl = lsq(
        tr.map((p) => [1, p.x[0], p.x[1]]),
        tr.map((p) => p.y),
        1e-9,
      );
      linearTestMse = te.length
        ? te.reduce((s, p) => s + (p.y - bl[0] - bl[1] * p.x[0] - bl[2] * p.x[1]) ** 2, 0) /
          te.length
        : NaN;
    };
    const reset = () => {
      B = [null, null];
      off = [0, 0];
      G = [new Array(61).fill(0), new Array(61).fill(0)];
      previousCurveSamples = [G[0].slice(), G[1].slice()];
      curveAnimationProgress = 1;
      turn = 0;
      wait = 0;
      delta = 1;
    };
    const takeTurn = () => {
      const j = turn % 2,
        o = 1 - j,
        tr = pts.filter((p) => !p.test),
        res = tr.map((p) => p.y - al - fj(o, p.x[o]));
      B[j] = lsq(
        tr.map((p) => splineBasis('natural', p.x[j], settings.K)),
        res,
        1e-8,
      );
      off[j] = 0;
      off[j] = tr.reduce((s, p) => s + fj(j, p.x[j]), 0) / tr.length;
      previousCurveSamples[j] = G[j].slice();
      G[j] = G[j].map((_, i) => fj(j, -1 + i / 30));
      delta = Math.max(...G[j].map((v, i) => Math.abs(v - previousCurveSamples[j][i])));
      turn++;
      curveAnimationProgress = 0;
    };
    gen();
    const tp = testPoint(
      root,
      'am',
      ['Your x₁', 'Your x₂'],
      [0.5, -0.5],
      [
        [-1, 1],
        [-1, 1],
      ],
    );
    tp.on = () => draw();
    const draw = () => {
      const colors = readThemeColors(),
        e =
          curveAnimationProgress < 0.5
            ? 2 * curveAnimationProgress * curveAnimationProgress
            : 1 - 2 * (1 - curveAnimationProgress) * (1 - curveAnimationProgress),
        act = turn ? (turn - 1) % 2 : -1;
      [c1, c2].forEach((c, j) => {
        const o = 1 - j,
          { g: ctx, w, h } = prepareCanvas(c),
          YR = [-2.6, 2.6],
          P = plot(ctx, w, h, [-1.05, 1.05], YR, colors, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: j ? 'x₂' : 'x₁',
          });
        clipPlot(ctx, P);
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.55);
        for (const p of pts)
          if (!p.test) {
            ctx.beginPath();
            ctx.arc(P.X(p.x[j]), P.Y(p.y - al - fj(o, p.x[o])), 2.8, 0, 7);
            ctx.fill();
          }
        if (settings.f !== 'c') {
          ctx.setLineDash([6, 5]);
          curve(ctx, P, TF[settings.f][j], -1, 1);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.beginPath();
        for (let i = 0; i <= 60; i++) {
          const v =
              j === act
                ? previousCurveSamples[j][i] + (G[j][i] - previousCurveSamples[j][i]) * e
                : G[j][i],
            x = P.X(-1 + i / 30),
            y = P.Y(clamp(v, -9, 9));
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.strokeStyle = j === act ? colors.gold : colors.ok;
        ctx.lineWidth = j === act ? 3.6 : 2.6;
        ctx.stroke();
        markTP(ctx, P.X(tp.v[j]), P.Y(clamp(fj(j, tp.v[j]), -9, 9)), colors, false);
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = j === act ? colors.cut : colors.muted;
        ctx.fillText(j === act ? 'being updated' : 'held fixed', P.m.l + 8, P.m.t + 8);
      });
      const tr = pts.filter((p) => !p.test),
        te = pts.filter((p) => p.test),
        er = (P) =>
          P.length
            ? P.reduce((s, p) => s + (p.y - al - fj(0, p.x[0]) - fj(1, p.x[1])) ** 2, 0) / P.length
            : NaN;
      cap.textContent = !turn
        ? 'Both curves start flat. The dots are the data with only the overall average removed.'
        : 'Turn ' +
          turn +
          ': f' +
          (act ? '₂' : '₁') +
          ' was just refitted to what f' +
          (act ? '₁' : '₂') +
          ' leaves unexplained.' +
          (delta < 0.004 ? ' The curves have stopped moving: backfitting has converged.' : '');
      {
        const a1 = fj(0, tp.v[0]),
          a2 = fj(1, tp.v[1]);
        tp.out.textContent =
          'At x₁ = ' +
          f2(tp.v[0]) +
          ', x₂ = ' +
          f2(tp.v[1]) +
          ' the model predicts y = average + f₁ + f₂ = ' +
          f2(al) +
          ' + (' +
          f2(a1) +
          ') + (' +
          f2(a2) +
          ') = ' +
          f2(al + a1 + a2) +
          '. The true value is ' +
          f2(gen2(tp.v[0], tp.v[1])) +
          '.';
      }
      setReadout(root, 'it', String(turn));
      setReadout(root, 'tr', f3(er(tr)));
      setReadout(root, 'te', f3(er(te)));
      setReadout(root, 'lin', f3(linearTestMse));
      setReadout(root, 's2', f3(settings.noise * settings.noise));
    };
    const anim = Anim(() => {
      frameCount++;
      if (curveAnimationProgress < 1) {
        curveAnimationProgress = Math.min(1, curveAnimationProgress + 0.03);
        draw();
        return;
      }
      if (wait > 0) {
        wait--;
        if (!wait) {
          reset();
          draw();
        }
        return;
      }
      if (frameCount % 50) return;
      if ((turn >= 2 && delta < 0.004) || turn >= 30) {
        fin = true;
        playback.stop();
        return;
      }
      takeTurn();
      draw();
      checkRunStop();
    });
    let fin = false,
      checkRunStop = () => false;
    playback = playBtn($('#am-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'am', {
      done: () => fin,
      restart: () => {
        fin = false;
        reset();
      },
      count: () => turn,
      what: 'turns',
    });
    $('#am-step', root).onclick = () => {
      playback.stop();
      takeTurn();
      curveAnimationProgress = 1;
      draw();
    };
    $('#am-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    const re = () => {
      gen();
      reset();
      draw();
    };
    bind(
      root,
      'am-k',
      (v) => {
        settings.K = v;
        reset();
        draw();
      },
      (v) => String(v),
    );
    $('#am-f', root).onchange = (e) => {
      settings.f = e.target.value;
      re();
    };
    bind(root, 'am-s', (v) => {
      settings.noise = v;
      re();
    });
    bind(root, 'am-r', (v) => {
      settings.r = v;
      re();
    });
    bind(
      root,
      'am-n',
      (v) => {
        settings.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#am-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
