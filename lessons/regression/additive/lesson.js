/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
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
import { dot, lsq, splineBasis } from '../shared.js';

export default {
  init(root) {
    const c1 = $('#am-1', root),
      c2 = $('#am-2', root),
      cap = $('#am-cap', root),
      S = { K: 4, noise: 0.3, r: 0, n: 160, f: 'a' };
    const TF = {
        a: [(x) => Math.sin(3 * x), (x) => 2 * x * x - 2 / 3],
        b: [(x) => 2 * Math.abs(x) - 1, (x) => 2 * x * x * x],
        c: [(x) => 0, (x) => 0],
      },
      gen2 = (a, b) => (S.f === 'c' ? 2.2 * a * b : TF[S.f][0](a) + TF[S.f][1](b));
    let seed = 6,
      pts = [],
      al = 0,
      B = [null, null],
      off = [0, 0],
      G = [new Array(61).fill(0), new Array(61).fill(0)],
      Gp = [G[0].slice(), G[1].slice()],
      u = 1,
      turn = 0,
      wait = 0,
      fr = 0,
      lin = 0,
      delta = 1,
      pb;
    const fj = (j, x) => (B[j] ? dot(B[j], splineBasis('natural', x, S.K)) - off[j] : 0);
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r),
        q = Math.sqrt(1 - S.r * S.r);
      pts = [];
      for (let i = 0; i < S.n; i++) {
        const a = r() * 2 - 1,
          b = clamp(S.r * a + q * (r() * 2 - 1), -1, 1);
        pts.push({ x: [a, b], y: gen2(a, b) + S.noise * rn(), test: r() < 0.3 });
      }
      const tr = pts.filter((p) => !p.test),
        te = pts.filter((p) => p.test);
      al = tr.reduce((s, p) => s + p.y, 0) / tr.length;
      const bl = lsq(
        tr.map((p) => [1, p.x[0], p.x[1]]),
        tr.map((p) => p.y),
        1e-9,
      );
      lin = te.length
        ? te.reduce((s, p) => s + (p.y - bl[0] - bl[1] * p.x[0] - bl[2] * p.x[1]) ** 2, 0) /
          te.length
        : NaN;
    };
    const reset = () => {
      B = [null, null];
      off = [0, 0];
      G = [new Array(61).fill(0), new Array(61).fill(0)];
      Gp = [G[0].slice(), G[1].slice()];
      u = 1;
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
        tr.map((p) => splineBasis('natural', p.x[j], S.K)),
        res,
        1e-8,
      );
      off[j] = 0;
      off[j] = tr.reduce((s, p) => s + fj(j, p.x[j]), 0) / tr.length;
      Gp[j] = G[j].slice();
      G[j] = G[j].map((_, i) => fj(j, -1 + i / 30));
      delta = Math.max(...G[j].map((v, i) => Math.abs(v - Gp[j][i])));
      turn++;
      u = 0;
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
      const t = T(),
        e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u),
        act = turn ? (turn - 1) % 2 : -1;
      [c1, c2].forEach((c, j) => {
        const o = 1 - j,
          { g, w, h } = fit(c),
          YR = [-2.6, 2.6],
          P = plot(g, w, h, [-1.05, 1.05], YR, t, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: j ? 'x₂' : 'x₁',
          });
        clipPlot(g, P);
        g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
        for (const p of pts)
          if (!p.test) {
            g.beginPath();
            g.arc(P.X(p.x[j]), P.Y(p.y - al - fj(o, p.x[o])), 2.8, 0, 7);
            g.fill();
          }
        if (S.f !== 'c') {
          g.setLineDash([6, 5]);
          curve(g, P, TF[S.f][j], -1, 1);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.5;
          g.stroke();
          g.setLineDash([]);
        }
        g.beginPath();
        for (let i = 0; i <= 60; i++) {
          const v = j === act ? Gp[j][i] + (G[j][i] - Gp[j][i]) * e : G[j][i],
            x = P.X(-1 + i / 30),
            y = P.Y(clamp(v, -9, 9));
          i ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        g.strokeStyle = j === act ? t.gold : t.ok;
        g.lineWidth = j === act ? 3.6 : 2.6;
        g.stroke();
        markTP(g, P.X(tp.v[j]), P.Y(clamp(fj(j, tp.v[j]), -9, 9)), t, false);
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = j === act ? t.cut : t.muted;
        g.fillText(j === act ? 'being updated' : 'held fixed', P.m.l + 8, P.m.t + 8);
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
      setR(root, 'it', String(turn));
      setR(root, 'tr', f3(er(tr)));
      setR(root, 'te', f3(er(te)));
      setR(root, 'lin', f3(lin));
      setR(root, 's2', f3(S.noise * S.noise));
    };
    const anim = Anim(() => {
      fr++;
      if (u < 1) {
        u = Math.min(1, u + 0.03);
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
      if (fr % 50) return;
      if ((turn >= 2 && delta < 0.004) || turn >= 30) {
        fin = true;
        pb.stop();
        return;
      }
      takeTurn();
      draw();
      chk();
    });
    let fin = false,
      chk = () => false;
    pb = playBtn($('#am-play', root), anim);
    chk = runCtl(pb, root, 'am', {
      done: () => fin,
      restart: () => {
        fin = false;
        reset();
      },
      count: () => turn,
      what: 'turns',
    });
    $('#am-step', root).onclick = () => {
      pb.stop();
      takeTurn();
      u = 1;
      draw();
    };
    $('#am-reset', root).onclick = () => {
      pb.stop();
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
        S.K = v;
        reset();
        draw();
      },
      (v) => String(v),
    );
    $('#am-f', root).onchange = (e) => {
      S.f = e.target.value;
      re();
    };
    bind(root, 'am-s', (v) => {
      S.noise = v;
      re();
    });
    bind(root, 'am-r', (v) => {
      S.r = v;
      re();
    });
    bind(
      root,
      'am-n',
      (v) => {
        S.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#am-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
