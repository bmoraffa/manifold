/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clipPlot,
  curve,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { errPlot } from '../shared.js';

export default {
  init(root) {
    const c = $('#gbm-c', root),
      cr = $('#gbm-r', root),
      live = $('#gbm-live', root),
      S = { nu: 0.2, depth: 2, B: 100, noise: 0.35, n: 90, f: 'wave' },
      FN = {
        wave: (x) => Math.sin(x) + 0.5 * Math.sin(2.3 * x) + 0.12 * x,
        steps: (x) => (x < 3 ? -1 : x < 6.5 ? 1.6 : 0.2),
        line: (x) => 0.35 * x - 1.2,
      };
    let seed = 9,
      pts = [],
      trees = [],
      F0 = 0,
      eTr = [],
      eTe = [],
      lastRes = [],
      wait = 0,
      fr = 0,
      pb,
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
    const rp = (nd, x) => {
        while (nd.sp) nd = x <= nd.sp.thr ? nd.sp.L : nd.sp.R;
        return nd.v;
      },
      pred = (x) => {
        let s = F0;
        for (const tq of trees) s += S.nu * rp(tq, x);
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
      eTr = [];
      eTe = [];
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
      const tq = regTree(lastRes, S.depth);
      trees.push(tq);
      eTr.push(mse(T));
      eTe.push(mse(te()));
      const k = trees.length;
      msg =
        'Round ' +
        k +
        ': fitted a tree with ' +
        lv(tq) +
        ' leaves to the residuals and added ' +
        f2(S.nu) +
        ' × that tree to the prediction. Train MSE ' +
        f3(before) +
        ' → ' +
        f3(eTr[k - 1]) +
        '. Test MSE ' +
        f3(eTe[k - 1]) +
        (k > 5 && eTe[k - 1] > Math.min(...eTe) * 1.03
          ? ', which is now above its best value: the new trees are fitting noise.'
          : '.');
    };
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < S.n; i++) {
        const x = r() * 10;
        pts.push({ x, y: FN[S.f](x) + S.noise * rn(), test: r() < 0.3 });
      }
    };
    gen();
    reset();
    for (let i = 0; i < 30; i++) add();
    const tp = testPoint(root, 'gbm', ['Your x'], [4], [[0, 10]]);
    const draw = () => {
      const t = T();
      {
        const { g, w, h } = fit(c),
          YR = [-2.8, 3.8],
          P = plot(g, w, h, [0, 10], YR, t, { yticks: [-2, 0, 2], xlabel: 'x' });
        clipPlot(g, P);
        g.setLineDash([6, 5]);
        curve(g, P, FN[S.f], 0, 10, 400);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.4;
        g.stroke();
        g.setLineDash([]);
        g.strokeStyle = t.ok;
        g.lineWidth = 1.4;
        for (const p of pts)
          if (p.test) {
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 3.6, 0, 7);
            g.stroke();
          }
        g.fillStyle = mix(t.accent, t.accent, 0, 0.85);
        for (const p of pts)
          if (!p.test) {
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 3.8, 0, 7);
            g.fill();
          }
        curve(g, P, pred, 0, 10, 700);
        g.strokeStyle = t.gold;
        g.lineWidth = 3;
        g.stroke();
        markTP(g, P.X(tp.v[0]), P.Y(pred(tp.v[0])), t);
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText(
          'gold = sum of ' + trees.length + ' trees, dashed = truth',
          P.m.l + 8,
          P.m.t + 8,
        );
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.55;
        let rm = 0.5;
        for (const q of lastRes) rm = Math.max(rm, Math.abs(q.r));
        g.save();
        const P = plot(g, w, h1, [0, 10], [-rm * 1.1, rm * 1.1], t, { xlabel: 'x' });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('0', P.m.l - 6, P.Y(0) + 4);
        clipPlot(g, P);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(P.X(0), P.Y(0));
        g.lineTo(P.X(10), P.Y(0));
        g.stroke();
        g.fillStyle = mix(t.accent, t.accent, 0, 0.7);
        for (const q of lastRes) {
          g.beginPath();
          g.arc(P.X(q.x), P.Y(q.r), 3, 0, 7);
          g.fill();
        }
        if (trees.length) {
          const tq = trees[trees.length - 1];
          curve(g, P, (x) => rp(tq, x), 0, 10, 700);
          g.strokeStyle = t.ok;
          g.lineWidth = 2.8;
          g.stroke();
        }
        g.restore();
        g.textAlign = 'left';
        g.fillStyle = t.ok;
        g.fillText(
          trees.length
            ? 'tree ' + trees.length + ', fitted to these residuals'
            : 'residuals after predicting the average',
          P.m.l + 8,
          P.m.t + 8,
        );
        g.restore();
        errPlot(
          g,
          w,
          h,
          h1,
          t,
          S.B,
          [
            [eTr, t.accent, 'training error'],
            [eTe, t.ok, 'test error'],
          ],
          'rounds',
        );
      }
      live.textContent = msg;
      const cu = !trees.length ? 0 : trees.length >= S.B ? 4 : 3;
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
        f2(FN[S.f](tp.v[0])) +
        '.';
      let bi = 0;
      eTe.forEach((v, i) => {
        if (v < eTe[bi]) bi = i;
      });
      setR(root, 'k', trees.length + ' of ' + S.B);
      setR(root, 'tr', f3(mse(tr())));
      setR(root, 'te', f3(mse(te())));
      setR(root, 'best', eTe.length ? String(bi + 1) : '–');
      setR(root, 's2', f3(S.noise * S.noise));
    };
    tp.on = () => draw();
    const anim = Anim(() => {
      fr++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          reset();
          draw();
        }
        return;
      }
      if (fr % (trees.length < 8 ? 26 : 5)) return;
      if (trees.length < S.B) {
        add();
        draw();
      } else pb.stop();
    });
    pb = playBtn($('#gbm-play', root), anim);
    runCtl(pb, root, 'gbm', { done: () => trees.length >= S.B, restart: () => reset() });
    $('#gbm-step', root).onclick = () => {
      pb.stop();
      if (trees.length >= S.B) reset();
      else add();
      draw();
    };
    $('#gbm-reset', root).onclick = () => {
      pb.stop();
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
      S.nu = v;
      rs();
    });
    bind(
      root,
      'gbm-depth',
      (v) => {
        S.depth = v;
        rs();
      },
      (v) => String(v),
    );
    bind(
      root,
      'gbm-B',
      (v) => {
        S.B = v;
        rs();
      },
      (v) => String(v),
    );
    $('#gbm-f', root).onchange = (e) => {
      S.f = e.target.value;
      re();
    };
    bind(root, 'gbm-s', (v) => {
      S.noise = v;
      re();
    });
    bind(
      root,
      'gbm-n',
      (v) => {
        S.n = v;
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
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        pb.start();
      },
    };
  },
};
