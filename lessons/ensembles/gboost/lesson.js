/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  bind,
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
import { errPlot } from '../shared.js';

export default {
  init(root) {
    const mainCanvas = $('#gbm-c', root),
      cr = $('#gbm-r', root),
      live = $('#gbm-live', root),
      settings = { nu: 0.2, depth: 2, B: 100, noise: 0.35, n: 90, f: 'wave' },
      FN = {
        wave: (x) => Math.sin(x) + 0.5 * Math.sin(2.3 * x) + 0.12 * x,
        steps: (x) => (x < 3 ? -1 : x < 6.5 ? 1.6 : 0.2),
        line: (x) => 0.35 * x - 1.2,
      };
    let seed = 9,
      pts = [],
      trees = [],
      F0 = 0,
      trainMseHistory = [],
      testMseHistory = [],
      lastRes = [],
      wait = 0,
      frameCount = 0,
      playback,
      msg = '';
    const regTree = (P, D) => {
      const mk = (Q, dp) => {
        let s = 0;
        for (const p of Q) s += p.r;
        const nd = { v: s / Q.length, sp: null };
        if (dp >= D || Q.length < 4) return nd;
        let best = null,
          ls = 0;
        for (let i = 1; i < Q.length; i++) {
          ls += Q[i - 1].r;
          if (Q[i].x === Q[i - 1].x || i < 2 || Q.length - i < 2) continue;
          const gn = (ls * ls) / i + ((s - ls) * (s - ls)) / (Q.length - i);
          if (!best || gn > best.gn) best = { gn, i };
        }
        if (!best) return nd;
        nd.sp = {
          thr: (Q[best.i].x + Q[best.i - 1].x) / 2,
          L: mk(Q.slice(0, best.i), dp + 1),
          R: mk(Q.slice(best.i), dp + 1),
        };
        return nd;
      };
      return mk(
        P.slice().sort((a, b) => a.x - b.x),
        0,
      );
    };
    const predictTree = (nd, x) => {
        while (nd.sp) nd = x <= nd.sp.thr ? nd.sp.L : nd.sp.R;
        return nd.v;
      },
      pred = (x) => {
        let s = F0;
        for (const tq of trees) s += settings.nu * predictTree(tq, x);
        return s;
      },
      lv = (nd) => (nd.sp ? lv(nd.sp.L) + lv(nd.sp.R) : 1);
    const mse = (P) =>
        P.length ? P.reduce((s, p) => s + (p.y - pred(p.x)) ** 2, 0) / P.length : NaN,
      tr = () => pts.filter((p) => !p.test),
      te = () => pts.filter((p) => p.test);
    const reset = () => {
      trees = [];
      const T = tr();
      F0 = T.reduce((s, p) => s + p.y, 0) / Math.max(1, T.length);
      trainMseHistory = [];
      testMseHistory = [];
      lastRes = T.map((p) => ({ x: p.x, r: p.y - F0 }));
      wait = 0;
      msg =
        'Round 0: the prediction is the average of y, ' +
        f2(F0) +
        ', everywhere. Train MSE ' +
        f3(mse(T)) +
        '. The dots on the right are what that leaves unexplained.';
    };
    const add = () => {
      const T = tr();
      if (T.length < 6) return;
      const before = mse(T);
      lastRes = T.map((p) => ({ x: p.x, r: p.y - pred(p.x) }));
      const tq = regTree(lastRes, settings.depth);
      trees.push(tq);
      trainMseHistory.push(mse(T));
      testMseHistory.push(mse(te()));
      const k = trees.length;
      msg =
        'Round ' +
        k +
        ': fitted a tree with ' +
        lv(tq) +
        ' leaves to the residuals and added ' +
        f2(settings.nu) +
        ' × that tree to the prediction. Train MSE ' +
        f3(before) +
        ' → ' +
        f3(trainMseHistory[k - 1]) +
        '. Test MSE ' +
        f3(testMseHistory[k - 1]) +
        (k > 5 && testMseHistory[k - 1] > Math.min(...testMseHistory) * 1.03
          ? ', which is now above its best value: the new trees are fitting noise.'
          : '.');
    };
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < settings.n; i++) {
        const x = r() * 10;
        pts.push({ x, y: FN[settings.f](x) + settings.noise * rn(), test: r() < 0.3 });
      }
    };
    gen();
    reset();
    for (let i = 0; i < 30; i++) add();
    const tp = testPoint(root, 'gbm', ['Your x'], [4], [[0, 10]]);
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          YR = [-2.8, 3.8],
          P = plot(ctx, w, h, [0, 10], YR, colors, { yticks: [-2, 0, 2], xlabel: 'x' });
        clipPlot(ctx, P);
        ctx.setLineDash([6, 5]);
        curve(ctx, P, FN[settings.f], 0, 10, 400);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 1.4;
        for (const p of pts)
          if (p.test) {
            ctx.beginPath();
            ctx.arc(P.X(p.x), P.Y(p.y), 3.6, 0, 7);
            ctx.stroke();
          }
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.85);
        for (const p of pts)
          if (!p.test) {
            ctx.beginPath();
            ctx.arc(P.X(p.x), P.Y(p.y), 3.8, 0, 7);
            ctx.fill();
          }
        curve(ctx, P, pred, 0, 10, 700);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 3;
        ctx.stroke();
        markTP(ctx, P.X(tp.v[0]), P.Y(pred(tp.v[0])), colors);
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.muted;
        ctx.fillText(
          'gold = sum of ' + trees.length + ' trees, dashed = truth',
          P.m.l + 8,
          P.m.t + 8,
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.55;
        let rm = 0.5;
        for (const q of lastRes) rm = Math.max(rm, Math.abs(q.r));
        ctx.save();
        const P = plot(ctx, w, h1, [0, 10], [-rm * 1.1, rm * 1.1], colors, { xlabel: 'x' });
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('0', P.m.l - 6, P.Y(0) + 4);
        clipPlot(ctx, P);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(P.X(0), P.Y(0));
        ctx.lineTo(P.X(10), P.Y(0));
        ctx.stroke();
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.7);
        for (const q of lastRes) {
          ctx.beginPath();
          ctx.arc(P.X(q.x), P.Y(q.r), 3, 0, 7);
          ctx.fill();
        }
        if (trees.length) {
          const tq = trees[trees.length - 1];
          curve(ctx, P, (x) => predictTree(tq, x), 0, 10, 700);
          ctx.strokeStyle = colors.ok;
          ctx.lineWidth = 2.8;
          ctx.stroke();
        }
        ctx.restore();
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.ok;
        ctx.fillText(
          trees.length
            ? 'tree ' + trees.length + ', fitted to these residuals'
            : 'residuals after predicting the average',
          P.m.l + 8,
          P.m.t + 8,
        );
        ctx.restore();
        errPlot(
          ctx,
          w,
          h,
          h1,
          colors,
          settings.B,
          [
            [trainMseHistory, colors.accent, 'training error'],
            [testMseHistory, colors.ok, 'test error'],
          ],
          'rounds',
        );
      }
      live.textContent = msg;
      const cu = !trees.length ? 0 : trees.length >= settings.B ? 4 : 3;
      root
        .querySelectorAll('#gbm-steps li')
        .forEach((e, i) => e.setAttribute('aria-current', i === cu ? 'true' : 'false'));
      tp.out.textContent =
        'At x = ' +
        f2(tp.v[0]) +
        ' the ' +
        trees.length +
        ' trees predict y = ' +
        f2(F0) +
        ' (average) + (' +
        f2(pred(tp.v[0]) - F0) +
        ') (corrections) = ' +
        f2(pred(tp.v[0])) +
        '. The true curve there is ' +
        f2(FN[settings.f](tp.v[0])) +
        '.';
      let bi = 0;
      testMseHistory.forEach((v, i) => {
        if (v < testMseHistory[bi]) bi = i;
      });
      setReadout(root, 'k', trees.length + ' of ' + settings.B);
      setReadout(root, 'tr', f3(mse(tr())));
      setReadout(root, 'te', f3(mse(te())));
      setReadout(root, 'best', testMseHistory.length ? String(bi + 1) : '–');
      setReadout(root, 's2', f3(settings.noise * settings.noise));
    };
    tp.on = () => draw();
    const anim = Anim(() => {
      frameCount++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          reset();
          draw();
        }
        return;
      }
      if (frameCount % (trees.length < 8 ? 26 : 5)) return;
      if (trees.length < settings.B) {
        add();
        draw();
      } else playback.stop();
    });
    playback = playBtn($('#gbm-play', root), anim);
    attachRunLimit(playback, root, 'gbm', {
      done: () => trees.length >= settings.B,
      restart: () => reset(),
    });
    $('#gbm-step', root).onclick = () => {
      playback.stop();
      if (trees.length >= settings.B) reset();
      else add();
      draw();
    };
    $('#gbm-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    const rs = () => {
        reset();
        draw();
      },
      re = () => {
        gen();
        reset();
        draw();
      };
    bind(root, 'gbm-nu', (v) => {
      settings.nu = v;
      rs();
    });
    bind(
      root,
      'gbm-depth',
      (v) => {
        settings.depth = v;
        rs();
      },
      (v) => String(v),
    );
    bind(
      root,
      'gbm-B',
      (v) => {
        settings.B = v;
        rs();
      },
      (v) => String(v),
    );
    $('#gbm-f', root).onchange = (e) => {
      settings.f = e.target.value;
      re();
    };
    bind(root, 'gbm-s', (v) => {
      settings.noise = v;
      re();
    });
    bind(
      root,
      'gbm-n',
      (v) => {
        settings.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#gbm-new', root).onclick = () => {
      seed++;
      re();
    };
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        reset();
        draw();
        playback.start();
      },
    };
  },
};
