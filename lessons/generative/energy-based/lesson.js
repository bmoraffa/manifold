/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { G2S, ebE, ebLang, g2Data, g2Near, g2Note, g2Pts, g2Samp, g2Uni } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { uCur, uLine, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#eb-c', root),
      cr = $('#eb-r', root),
      live = $('#eb-live', root),
      S = { ds: 'moons', sl: 0.25 },
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
      GAP = [],
      gsm = [],
      phase = 'train',
      data = [],
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      P = null,
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
      GAP = [];
      gsm = [];
      phase = 'train';
      xs = null;
      done = false;
      cu = 0;
      nearV = NaN;
      edm = NaN;
      esm = NaN;
      si = 0;
      data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
      P = g2Uni(NP);
      refreshE();
      msg =
        'A network with random weights gives a landscape that has nothing to do with the data. The small dots are the model’s own points, scattered at random. Press Run.';
    };
    const train = (k) => {
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(2 * NP);
        for (let i = 0; i < NP; i++) {
          const p = g2Samp(S.ds, RN);
          X[2 * i] = p[0] + 0.03 * RN();
          X[2 * i + 1] = p[1] + 0.03 * RN();
        }
        for (let i = 0; i < NP; i++)
          if (Math.random() < 0.05) {
            P[2 * i] = (Math.random() * 2 - 1) * 2.2;
            P[2 * i + 1] = (Math.random() * 2 - 1) * 1.65;
          }
        ebLang(net, P, NP, 5, A, SGM, RN);
        const Ed = net.fwd(AD.input(X, [NP, 2])),
          Es = net.fwd(AD.input(P, [NP, 2])),
          l = AD.ebl(Ed, Es, 0.1);
        let md = 0,
          ms = 0;
        for (let i = 0; i < NP; i++) {
          md += Ed.v[i] / NP;
          ms += Es.v[i] / NP;
        }
        AD.backward(l);
        AD.adam(net.ps, 0.003, ++ts, 1);
        GAP.push(ms - md);
        edm = md;
        esm = ms;
        if (ts % 10 === 0) {
          let s = 0;
          for (let j = ts - 10; j < ts; j++) s += GAP[j];
          gsm.push(s / 10);
        }
      }
      if (ts - lastE >= 8 || ts >= MAXS) refreshE();
      cu = 1 + (Math.floor(ts / 60) % 3);
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
        cu = 4;
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
      cu = 4;
      msg =
        'Rolling step ' +
        si +
        ' of ' +
        GENS +
        ': every point moved a short way downhill and was jittered.';
      if (si >= GENS) {
        ebLang(net, xs, M, 20, A, 0, RN);
        done = true;
        cu = 5;
        nearV = g2Near(xs, M, data);
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
        cu = 4;
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
      const t = T();
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
              g.fillStyle = mix(t.accent, t.accent, 0, 0.8 * Math.pow(u, 1.2));
              const xa = Math.round(V.X(-4 + (8 * q) / GW)),
                xb = Math.round(V.X(-4 + (8 * (q + 1)) / GW)),
                ya = Math.round(V.Y(3 - (6 * r) / GH)),
                yb = Math.round(V.Y(3 - (6 * (r + 1)) / GH));
              g.fillRect(xa, ya, xb - xa, yb - ya);
            }
        g.setLineDash([6, 5]);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(V.X(-4), V.Y(S.sl * G2S));
        g.lineTo(V.X(4), V.Y(S.sl * G2S));
        g.stroke();
        g.setLineDash([]);
        g2Data(g, V, t, data, 0.4);
        if (phase === 'train' && P) g2Pts(g, V, t, P, NP, t.cut, 2.4);
        if (xs) g2Pts(g, V, t, xs, M, t.gold);
        markTP(g, V.X(tp.v[0] * G2S), V.Y(tp.v[1] * G2S), t);
        g2Note(
          g,
          t,
          h,
          phase === 'train'
            ? 'small dots: the model’s own 96 points'
            : 'gold dots: ' + M + ' generated points',
        );
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5,
          NPF = 70,
          F = new Float64Array(2 * NPF);
        for (let i = 0; i < NPF; i++) {
          F[2 * i] = -2.25 + (4.5 * i) / (NPF - 1);
          F[2 * i + 1] = S.sl;
        }
        const pe = Array.from(ebE(net, F, NPF)),
          lo = Math.min(...pe),
          hi = Math.max(lo + 1e-6, ...pe);
        const Lp = uLine(g, t, 0, 0, w, h1, pe, {
          title: 'energy along the dashed line: valleys are where the model expects data',
          col: t.accent,
          y1: lo,
          ym: hi,
        });
        g.fillStyle = t.ink;
        for (const p of data)
          if (Math.abs(p[1] - S.sl) < 0.07) {
            const x = Lp.X(((p[0] + 2.25) / 4.5) * (NPF - 1));
            g.fillRect(x - 0.8, h1 - Lp.m.b - 7, 1.6, 7);
          }
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('ticks: real data near the line', w - Lp.m.r, h1 - 5);
        const G2 = uLine(g, t, 0, h1, w, h - h1, gsm, {
          title: 'energy: model’s points minus real points (average of 10 steps)',
          n: MAXS / 10,
          col: t.ok,
          y1: Math.min(-0.05, ...gsm),
          ym: Math.max(0.1, ...gsm),
        });
        g.setLineDash([5, 4]);
        g.strokeStyle = t.gold;
        g.lineWidth = 1.3;
        g.beginPath();
        g.moveTo(G2.m.l, G2.Y(0));
        g.lineTo(w - G2.m.r, G2.Y(0));
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = t.cut;
        g.textAlign = 'right';
        g.fillText('dashed line: zero, where samples look like data', w - G2.m.r, h - G2.m.b - 7);
      }
      live.textContent = msg;
      uCur(root, 'eb', cu);
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
      setR(
        root,
        'ph',
        phase === 'train' ? (ts ? 'Training' : 'Not started') : done ? 'Finished' : 'Generating',
      );
      setR(root, 'ts', ts + ' of ' + MAXS);
      setR(root, 'ed', isFinite(edm) ? f3(edm) : '–');
      setR(root, 'es', isFinite(esm) ? f3(esm) : '–');
      setR(root, 'gs', phase === 'gen' ? Math.min(si, GENS) + ' of ' + GENS : '–');
      setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
    };
    tp.on = () => draw();
    build();
    if (RM) {
      train(MAXS);
      while (!done) gstep(10);
    }
    const anim = Anim(() => {
      fr++;
      if (phase === 'train') {
        train(Math.min(ts < 30 ? 1 : 2, chk.remaining()));
        draw();
        chk();
        return;
      }
      if (!done) gstep(1);
      draw();
      if (done) pb.stop();
    });
    pb = playBtn($('#eb-play', root), anim);
    chk = runCtl(pb, root, 'eb', {
      done: () => phase === 'gen' && done,
      restart: () => startGen(),
      count: () => ts,
      what: 'training steps',
    });
    $('#eb-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#eb-reset', root).onclick = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    $('#eb-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      pb.stop();
      run++;
      build();
      draw();
      pb.start();
    };
    $('#eb-gen', root).onclick = () => {
      if (phase !== 'gen') {
        msg = 'Points can be generated once training is finished. Press Run.';
        draw();
        return;
      }
      pb.stop();
      startGen();
      draw();
      pb.start();
    };
    bind(
      root,
      'eb-sl',
      (v) => {
        S.sl = +v;
        if (pb) draw();
      },
      f2,
    );
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
