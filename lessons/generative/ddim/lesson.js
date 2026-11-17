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
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import {
  g2Bench,
  g2Data,
  g2Emb,
  g2Levels,
  g2Near,
  g2Note,
  g2Panel,
  g2Pred,
  g2Pts,
  g2Samp,
  g2Trails,
} from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { uCur, uLine, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#dm-c', root),
      cr = $('#dm-r', root),
      live = $('#dm-live', root),
      S = { ds: 'ring', ni: 6, eta: 0 },
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
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      z0 = null,
      xs = null,
      x0h = null,
      seq = [],
      si = 0,
      done = false,
      tr = [],
      prevOut = null,
      moved = NaN,
      Q = [[], []],
      bi = 0,
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
      cu = 0;
      Q = [NS.map(() => NaN), NS.map(() => NaN)];
      bi = 0;
      z0 = null;
      prevOut = null;
      moved = NaN;
      nearV = NaN;
      tr = [];
      si = 0;
      data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
      msg =
        'A network with random weights. Press Run: it trains first, then the page tries it with different numbers of sampler steps.';
    };
    const train = (k) => {
      const B = 128;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(B * 10),
          E = new Float64Array(B * 2);
        for (let i = 0; i < B; i++) {
          const p = g2Samp(S.ds, RN),
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
      cu = 0;
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
        bi = 0;
        cu = 1;
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
      return g2Near(x, Mm, data, 0.1);
    };
    const startGen = (fresh) => {
      if (fresh || !z0) {
        z0 = Float64Array.from({ length: 2 * M }, () => RN());
        prevOut = null;
        moved = NaN;
      }
      xs = Float64Array.from(z0);
      x0h = null;
      seq = mkSeq(NS[S.ni]);
      si = 0;
      done = false;
      nearV = NaN;
      tr = Array.from({ length: 14 }, (_, i) => [[xs[2 * i], xs[2 * i + 1]]]);
    };
    const bench = () => {
      const j = bi >> 1,
        e = bi & 1;
      Q[e][j] = sampleAll(NS[j], e, 200);
      bi++;
      cu = 1;
      msg =
        'Measuring: ' +
        NS[j] +
        ' step' +
        (NS[j] > 1 ? 's' : '') +
        ' with η = ' +
        e +
        (e ? ' (fresh noise at every step, the DDPM rule)' : ' (no fresh noise, the DDIM rule)') +
        ' put ' +
        pct(Q[e][j]) +
        ' of 200 new points on the shape.';
      if (bi >= 2 * NS.length) {
        phase = 'gen';
        startGen(true);
        cu = 2;
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
      x0h = sstep(xs, k, kp, S.eta);
      si++;
      tr.forEach((p, i) => p.push([xs[2 * i], xs[2 * i + 1]]));
      cu = 4;
      msg =
        'Step ' +
        si +
        ' of ' +
        NS[S.ni] +
        ': the network guessed the noise at level ' +
        k +
        '. Removing it gave the guessed finished points (top right). They were put back at level ' +
        kp +
        (kp === 0
          ? ', which is no noise at all.'
          : S.eta > 0
            ? ', using the guessed noise mixed with fresh random noise.'
            : ', using the same guessed noise and nothing random.');
      if (si >= seq.length - 1) {
        done = true;
        cu = 5;
        nearV = g2Near(xs, M, data, 0.1);
        if (prevOut) {
          let d = 0;
          for (let i = 0; i < M; i++)
            d += Math.hypot(xs[2 * i] - prevOut[2 * i], xs[2 * i + 1] - prevOut[2 * i + 1]);
          moved = d / M;
        }
        prevOut = Float64Array.from(xs);
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
        cu = 2;
        msg = 'The same starting noise again. Press Run or One step.';
      } else gstep();
    };
    const draw = () => {
      const t = T();
      {
        const V = uView(c),
          { g, h } = V;
        g2Data(g, V, t, data);
        if (xs) {
          g2Trails(g, V, t, tr);
          g2Pts(g, V, t, xs, M, t.gold);
        }
        g2Note(
          g,
          t,
          h,
          phase === 'gen'
            ? NS[S.ni] +
                ' step' +
                (NS[S.ni] > 1 ? 's' : '') +
                ', η = ' +
                f2(S.eta) +
                (xs ? ', now at noise level ' + seq[si] : '')
            : phase === 'bench'
              ? 'measuring each number of steps'
              : 'training the network',
        );
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.52;
        if (phase === 'gen' && x0h) {
          const P = g2Panel(
            g,
            t,
            0,
            0,
            w,
            h1,
            'the network’s guess of where each point will finish, at this step',
            1.22,
          );
          g2Data(g, P, t, data, 0.22, 1.5);
          g2Pts(g, P, t, x0h, M, t.accent, 2);
          g.restore();
        } else
          g2Levels(
            g,
            t,
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
          uLine(g, t, 0, h1, w, h - h1, sm, {
            title: 'error at guessing the noise (average of every 20 steps)',
            n: MAXS / 20,
            col: t.ok,
            ym: Math.max(1.05, sm[0] || 1.05),
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
              { ys: Q[1], col: t.cut, lab: 'η = 1: fresh noise each step' },
              { ys: Q[0], col: t.accent, lab: 'η = 0: DDIM' },
            ],
            phase === 'gen' ? S.ni : -1,
            'share of points on the shape, by number of sampler steps',
          );
      }
      live.textContent = msg;
      uCur(root, 'dm', cu);
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
      setR(root, 'nc', phase === 'gen' ? si + ' of ' + NS[S.ni] : '–');
      setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
      setR(root, 'mv', isFinite(moved) && done ? f2(moved) : '–');
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
      if (fr % clamp(Math.round(220 / NS[S.ni]), 4, 40) === 0 && !done) gstep();
      draw();
      if (done) pb.stop();
    });
    pb = playBtn($('#dm-play', root), anim);
    chk = runCtl(pb, root, 'dm', {
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
    $('#dm-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#dm-reset', root).onclick = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    $('#dm-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      pb.stop();
      run++;
      build();
      draw();
      pb.start();
    };
    $('#dm-same', root).onclick = () => regen(false);
    $('#dm-new', root).onclick = () => regen(true);
    bind(
      root,
      'dm-n',
      (v) => {
        S.ni = +v;
        if (pb && phase === 'gen') regen(false);
      },
      (v) => String(NS[+v]),
    );
    bind(
      root,
      'dm-eta',
      (v) => {
        S.eta = +v;
        if (pb && phase === 'gen') regen(false);
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
