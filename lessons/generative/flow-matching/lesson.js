/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  RM,
  T,
  bind,
  clamp,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import {
  G2G,
  g2Arrows,
  g2Bench,
  g2Data,
  g2Emb,
  g2Near,
  g2Note,
  g2Panel,
  g2Pred,
  g2Pts,
  g2Samp,
  g2Trails,
} from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, uCur, uLine, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#fm-c', root),
      cr = $('#fm-r', root),
      live = $('#fm-live', root),
      S = { ds: 'ring', ni: 5, sol: 'euler' },
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
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      z0 = null,
      xs = null,
      si = 0,
      done = false,
      tr = [],
      Q = [[], []],
      bi = 0,
      nearV = NaN,
      plen = null,
      calls = 0,
      strt = NaN,
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
      cu = 0;
      Q = [NS.map(() => NaN), NS.map(() => NaN)];
      bi = 0;
      z0 = null;
      nearV = NaN;
      strt = NaN;
      tr = [];
      si = 0;
      calls = 0;
      data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
      pairs = Array.from({ length: 9 }, (_, i) => ({ a: [RN(), RN()], b: data[i] }));
      msg = 'A network with random weights: its arrows mean nothing yet. Press Run to train it.';
    };
    const train = (k) => {
      const B = 128;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(B * 10),
          E = new Float64Array(B * 2);
        for (let i = 0; i < B; i++) {
          const p = g2Samp(S.ds, RN),
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
      cu = Math.floor(ts / 100) % 3;
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
        cu = 3;
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
      return g2Near(x, Mm, data, 0.1);
    };
    const startGen = (fresh) => {
      if (fresh || !z0) z0 = Float64Array.from({ length: 2 * M }, () => RN());
      xs = Float64Array.from(z0);
      si = 0;
      done = false;
      nearV = NaN;
      strt = NaN;
      calls = 0;
      plen = new Float64Array(M);
      tr = Array.from({ length: 14 }, (_, i) => [[xs[2 * i], xs[2 * i + 1]]]);
    };
    const bench = () => {
      const j = bi >> 1,
        e = bi & 1;
      Q[e][j] = sampleAll(NS[j], e ? 'mid' : 'euler', 200);
      bi++;
      cu = 3;
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
        pct(Q[e][j]) +
        ' of 200 new points on the shape.';
      if (bi >= 2 * NS.length) {
        phase = 'gen';
        startGen(true);
        cu = 4;
        msg =
          'All settings are measured: see the chart at the bottom right. Now watch one run: ' +
          M +
          ' points of pure noise at time 0.';
      }
    };
    const gstep = () => {
      if (done || !xs) return;
      const N = NS[S.ni],
        o = Float64Array.from(xs);
      ode(xs, si, N, S.sol);
      si++;
      calls += S.sol === 'mid' ? 2 : 1;
      for (let i = 0; i < M; i++)
        plen[i] += Math.hypot(xs[2 * i] - o[2 * i], xs[2 * i + 1] - o[2 * i + 1]);
      tr.forEach((p, i) => p.push([xs[2 * i], xs[2 * i + 1]]));
      cu = 4;
      msg =
        'Step ' +
        si +
        ' of ' +
        N +
        ': the network gave a velocity for every point at time ' +
        f2((si - 1) / N) +
        (S.sol === 'mid' ? ', each point looked again from half a step ahead,' : '') +
        ' and every point moved along it for a time of ' +
        f2(1 / N) +
        '. The time is now ' +
        f2(si / N) +
        '.';
      if (si >= N) {
        done = true;
        cu = 5;
        nearV = g2Near(xs, M, data, 0.1);
        let s = 0;
        for (let i = 0; i < M; i++)
          s += Math.hypot(xs[2 * i] - z0[2 * i], xs[2 * i + 1] - z0[2 * i + 1]) / (plen[i] || 1);
        strt = s / M;
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
        cu = 4;
        msg = 'The same starting noise again, at time 0. Press Run or One step.';
      } else gstep();
    };
    const draw = () => {
      const t = T(),
        N = NS[S.ni],
        u = phase === 'gen' ? Math.min(1, si / N) : (Math.floor(fr / 3) % 100) / 100;
      {
        const V = uView(c),
          { g, h } = V;
        g2Data(g, V, t, data);
        if (ts > 0) g2Arrows(g, V, pred(G2G, Math.min(u, 0.999)), t.accent, 1, 0.55, 6);
        if (xs) {
          g2Trails(g, V, t, tr);
          g2Pts(g, V, t, xs, M, t.gold);
        }
        g2Note(
          g,
          t,
          h,
          (ts > 0 ? 'arrows shown for time ' + f2(u) : 'not trained yet') +
            (phase === 'gen'
              ? ', ' +
                N +
                ' step' +
                (N > 1 ? 's' : '') +
                ' of ' +
                (S.sol === 'mid' ? 'Midpoint' : 'Euler')
              : ''),
        );
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5,
          P = g2Panel(
            g,
            t,
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
          g.strokeStyle = mix(t.ink, t.ink, 0, 0.3);
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(ax, ay);
          g.lineTo(bx, by);
          g.stroke();
          g.fillStyle = t.bg;
          g.strokeStyle = t.ink;
          g.beginPath();
          g.arc(ax, ay, 3.5, 0, 7);
          g.fill();
          g.stroke();
          g.fillStyle = t.ink;
          g.beginPath();
          g.arc(bx, by, 3.5, 0, 7);
          g.fill();
          g.strokeStyle = t.accent;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(mx, my);
          g.lineTo(mx + ((bx - ax) / d) * 16, my + ((by - ay) / d) * 16);
          g.stroke();
          g.fillStyle = t.gold;
          g.beginPath();
          g.arc(mx, my, 4, 0, 7);
          g.fill();
        }
        g.restore();
        if (phase === 'train')
          uLine(g, t, 0, h1, w, h - h1, sm, {
            title: 'error at guessing the velocity (average of every 20 steps)',
            n: MAXS / 20,
            col: t.ok,
            ym: Math.max(2.2, sm[0] || 2.2),
          });
        else
          g2Bench(
            g,
            t,
            0,
            h1,
            w,
            h - h1,
            NS,
            [
              { ys: Q[1], col: t.cut, lab: 'Midpoint: two calls per step' },
              { ys: Q[0], col: t.accent, lab: 'Euler: one call per step' },
            ],
            phase === 'gen' ? S.ni : -1,
            'share of points on the shape, by number of steps',
          );
      }
      live.textContent = msg;
      uCur(root, 'fm', cu);
      setR(
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
      setR(root, 'ts', ts + ' of ' + MAXS);
      setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
      setR(root, 'nc', phase === 'gen' ? String(calls) : '–');
      setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
      setR(root, 'st', isFinite(strt) ? f2(strt) : '–');
    };
    build();
    if (RM) {
      train(MAXS);
      while (phase === 'bench') bench();
      while (!done) gstep();
    }
    const anim = Anim(() => {
      fr++;
      if (phase === 'train') {
        train(Math.min(ts < 60 ? 1 : 5, chk.remaining()));
        draw();
        chk();
        return;
      }
      if (phase === 'bench') {
        if (fr % 4 === 0) {
          bench();
          draw();
        }
        return;
      }
      if (fr % clamp(Math.round(200 / NS[S.ni]), 5, 40) === 0 && !done) gstep();
      draw();
      if (done) pb.stop();
    });
    pb = playBtn($('#fm-play', root), anim);
    chk = runCtl(pb, root, 'fm', {
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
      pb.stop();
      startGen(fresh);
      draw();
      pb.start();
    };
    $('#fm-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#fm-reset', root).onclick = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    $('#fm-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      pb.stop();
      run++;
      build();
      draw();
      pb.start();
    };
    $('#fm-same', root).onclick = () => regen(false);
    $('#fm-new', root).onclick = () => regen(true);
    bind(
      root,
      'fm-n',
      (v) => {
        S.ni = +v;
        if (pb && phase === 'gen') regen(false);
      },
      (v) => String(NS[+v]),
    );
    segBind(root, 'fm-sol', S.sol, (v) => {
      S.sol = v;
      if (phase === 'gen') regen(false);
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        run++;
        build();
        draw();
        pb.start();
      },
    };
  },
};
