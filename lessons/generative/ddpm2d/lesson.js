/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
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
import { pct } from '../../ensembles/shared.js';
import { uCur, uLine, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#d2-c', root),
      cr = $('#d2-r', root),
      live = $('#d2-live', root),
      S = { ds: 'moons' },
      Tn = 40,
      MAXS = 3000,
      M = 400,
      BE = Array.from({ length: Tn }, (_, i) => 0.002 + (0.198 * i) / (Tn - 1)),
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
      xs = null,
      lv = Tn,
      phase = 'train',
      data = [],
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      sm = [],
      done = false;
    const samp = () => {
      const R = Math.random;
      if (S.ds === 'moons') {
        const a = R() * Math.PI;
        return R() < 0.5
          ? [Math.cos(a) - 0.5 + 0.06 * RN(), -0.25 + Math.sin(a) * 0.9 + 0.06 * RN()]
          : [0.5 - Math.cos(a) + 0.06 * RN(), 0.25 - Math.sin(a) * 0.9 + 0.06 * RN()];
      }
      if (S.ds === 'ring') {
        const a = R() * 2 * Math.PI;
        return [1.1 * Math.cos(a) + 0.06 * RN(), 1.1 * Math.sin(a) + 0.06 * RN()];
      }
      if (S.ds === 'spiral') {
        const u = R(),
          a = u * 4 * Math.PI;
        return [
          (0.25 + 1.05 * u) * Math.cos(a) + 0.04 * RN(),
          (0.25 + 1.05 * u) * Math.sin(a) + 0.04 * RN(),
        ];
      }
      const k = Math.floor(R() * 3),
        m = [
          [-1, -0.5],
          [1, -0.5],
          [0, 0.9],
        ][k];
      return [m[0] + 0.18 * RN(), m[1] + 0.18 * RN()];
    };
    const emb = (k) => {
      const o = [];
      for (let q = 0; q < 4; q++) {
        const f = ((k + 1) / Tn) * Math.PI * 2 ** q * 0.5;
        o.push(Math.sin(f), Math.cos(f));
      }
      return o;
    };
    const pred = (P, k) => {
      const n = P.length / 2,
        X = new Float64Array(n * 10),
        e = emb(k);
      for (let i = 0; i < n; i++) {
        X[i * 10] = P[2 * i];
        X[i * 10 + 1] = P[2 * i + 1];
        for (let q = 0; q < 8; q++) X[i * 10 + 2 + q] = e[q];
      }
      const o = net.fwd(AD.input(X, [n, 10]));
      AD.clear();
      return o.v;
    };
    const build = () => {
      net = AD.mlp([10, 64, 64, 2], randnFrom(rng(run * 7 + 4)), 'relu');
      ts = 0;
      LL = [];
      sm = [];
      phase = 'train';
      xs = null;
      lv = Tn;
      done = false;
      cu = 0;
      data = Array.from({ length: 500 }, samp);
      msg =
        'A network with random weights. It cannot tell noise from data yet. Press Run to train it.';
    };
    const train = (k) => {
      const B = 128;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(B * 10),
          E = new Float64Array(B * 2);
        for (let i = 0; i < B; i++) {
          const p = samp(),
            tt = Math.floor(Math.random() * Tn),
            e0 = RN(),
            e1 = RN(),
            s = Math.sqrt(AB[tt]),
            nq = Math.sqrt(1 - AB[tt]),
            e = emb(tt);
          X[i * 10] = s * p[0] + nq * e0;
          X[i * 10 + 1] = s * p[1] + nq * e1;
          for (let j = 0; j < 8; j++) X[i * 10 + 2 + j] = e[j];
          E[2 * i] = e0;
          E[2 * i + 1] = e1;
        }
        const l = AD.mse(net.fwd(AD.input(X, [B, 10])), E);
        LL.push(l.v[0]);
        AD.backward(l);
        AD.adam(net.ps, 0.002, ++ts, 1);
        if (ts % 20 === 0) {
          let s = 0;
          for (let j = ts - 20; j < ts; j++) s += LL[j];
          sm.push(s / 20);
        }
      }
      cu = ts % 2 ? 0 : 1;
      msg =
        'Training step ' +
        ts +
        ' of ' +
        MAXS +
        ': 128 real points were given random amounts of noise, and the network guessed the noise in each. Average error over the last 20 steps: ' +
        f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
        '.';
      if (ts >= MAXS) {
        startGen();
        cu = 3;
        msg =
          'Training is finished. Now generate: ' +
          M +
          ' points of pure noise have been placed at the highest noise level. Watch them walk.';
      }
    };
    const startGen = () => {
      phase = 'gen';
      xs = Float64Array.from({ length: 2 * M }, () => RN());
      lv = Tn;
      done = false;
    };
    const gstep = () => {
      if (lv <= 0) return;
      const k = lv - 1,
        e = pred(xs, k),
        b = BE[k],
        c1 = b / Math.sqrt(1 - AB[k]),
        s1 = 1 / Math.sqrt(1 - b),
        s2 = k > 0 ? Math.sqrt(b) : 0;
      for (let i = 0; i < 2 * M; i++) xs[i] = s1 * (xs[i] - c1 * e[i]) + s2 * RN();
      lv--;
      cu = 4;
      msg =
        'Generating, noise level ' +
        lv +
        ' of ' +
        Tn +
        ': the network estimated the noise in each of the ' +
        M +
        ' points, a little of it was removed, and ' +
        (k > 0 ? 'a little fresh noise was added.' : 'no noise was added on this last step.') +
        (lv <= 0
          ? ' Finished: ' +
            pct(near()) +
            ' of the generated points lie on the shape. None of them is a copy of a training point.'
          : '');
      if (lv <= 0) done = true;
    };
    const near = () => {
      if (!xs) return NaN;
      let ok = 0;
      for (let i = 0; i < M; i++) {
        let b = 1e9;
        for (const q of data)
          b = Math.min(b, (q[0] - xs[2 * i]) ** 2 + (q[1] - xs[2 * i + 1]) ** 2);
        if (b < 0.15 * 0.15) ok++;
      }
      return ok / M;
    };
    const step = () => {
      if (phase === 'train') train(50);
      else if (done) {
        if (ts < MAXS) {
          phase = 'train';
          xs = null;
          train(50);
        } else {
          startGen();
          msg = 'A fresh cloud of pure noise. Press Run or One step.';
        }
      } else gstep();
    };
    const draw = () => {
      const t = T();
      {
        const V = uView(c),
          { g, w, h, X, Y } = V,
          sc = 1.75,
          X2 = (v) => X(v * sc),
          Y2 = (v) => Y(v * sc);
        g.fillStyle = mix(t.ink, t.ink, 0, 0.28);
        data.forEach((p) => {
          g.beginPath();
          g.arc(X2(p[0]), Y2(p[1]), 2, 0, 7);
          g.fill();
        });
        if (ts > 0) {
          const k = phase === 'gen' ? Math.max(0, lv - 1) : Math.floor((fr / 8) % Tn),
            G = [];
          for (let a = 0; a < 10; a++)
            for (let b = 0; b < 14; b++)
              G.push(-2.1 + (4.2 * (b + 0.5)) / 14, 1.55 - (3.1 * (a + 0.5)) / 10);
          const e = pred(Float64Array.from(G), k);
          g.strokeStyle = mix(t.accent, t.accent, 0, 0.6);
          g.lineWidth = 1.2;
          for (let i = 0; i < G.length / 2; i++) {
            const x = X2(G[2 * i]),
              y = Y2(G[2 * i + 1]),
              dx = -e[2 * i],
              dy = -e[2 * i + 1],
              m = Math.hypot(dx, dy) || 1,
              L = Math.min(15, 7 * m);
            g.beginPath();
            g.moveTo(x, y);
            g.lineTo(x + (dx / m) * L, y - (dy / m) * L);
            g.stroke();
            g.beginPath();
            g.arc(x + (dx / m) * L, y - (dy / m) * L, 1.6, 0, 7);
            g.fillStyle = t.accent;
            g.fill();
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('arrows shown for noise level ' + (k + 1), 8, h - 8);
        }
        if (xs) {
          g.fillStyle = t.gold;
          g.strokeStyle = t.bg;
          g.lineWidth = 0.7;
          for (let i = 0; i < M; i++) {
            const x = X2(xs[2 * i]),
              y = Y2(xs[2 * i + 1]);
            if (x < -5 || y < -5 || x > w + 5 || y > h + 5) continue;
            g.beginPath();
            g.arc(x, y, 3, 0, 7);
            g.fill();
            g.stroke();
          }
        }
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.42,
          LV = [0, 8, 20, 40],
          pw = (w - 20) / 4;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('the real data with more and more noise added: what training sees', 8, 13);
        LV.forEach((k, q) => {
          const x0 = 10 + q * pw,
            cx = x0 + pw / 2,
            cy = 20 + (h1 - 40) / 2,
            s = Math.min(pw, h1 - 40) / 5.6,
            a = k ? AB[k - 1] : 1,
            r = rng(q * 7 + 3),
            rn = randnFrom(r);
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.strokeRect(x0 + 2, 20, pw - 4, h1 - 40);
          g.fillStyle = mix(t.ink, t.ink, 0, 0.55);
          for (let i = 0; i < 160; i++) {
            const p = data[i],
              x = Math.sqrt(a) * p[0] + Math.sqrt(1 - a) * rn(),
              y = Math.sqrt(a) * p[1] + Math.sqrt(1 - a) * rn();
            if (Math.abs(x) > 2.7 || Math.abs(y) > 2.6) continue;
            g.beginPath();
            g.arc(cx + x * s, cy - y * s, 1.5, 0, 7);
            g.fill();
          }
          g.fillStyle = phase === 'gen' && Math.abs(lv - k) < 5 ? t.cut : t.muted;
          g.textAlign = 'center';
          g.fillText('level ' + k, cx, h1 - 6);
        });
        uLine(g, t, 0, h1, w, h - h1, sm, {
          title: 'error at guessing the noise (average of every 20 steps)',
          n: MAXS / 20,
          col: t.ok,
          ym: Math.max(1.05, sm[0] || 1.05),
        });
      }
      live.textContent = msg;
      uCur(root, 'd2', cu);
      setR(
        root,
        'ph',
        phase === 'train' ? (ts ? 'Training' : 'Not started') : done ? 'Finished' : 'Generating',
      );
      setR(root, 'ts', ts + ' of ' + MAXS);
      setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
      setR(root, 'lv', phase === 'gen' ? lv + ' of ' + Tn : '–');
      setR(root, 'near', phase === 'gen' && done ? pct(near()) : '–');
    };
    build();
    if (RM) {
      train(MAXS);
      while (!done) gstep();
    }
    const anim = Anim(() => {
      fr++;
      if (phase === 'train') {
        train(Math.min(ts < 60 ? 1 : 6, chk.remaining()));
        draw();
        chk();
        return;
      }
      if (fr % (lv > Tn - 4 ? 26 : 9) === 0 && !done) gstep();
      draw();
      if (done) pb.stop();
    });
    pb = playBtn($('#d2-play', root), anim);
    chk = runCtl(pb, root, 'd2', {
      done: () => phase === 'gen' && done,
      restart: () => {
        if (ts < MAXS) {
          phase = 'train';
          xs = null;
          done = false;
        } else startGen();
      },
      count: () => ts,
      what: 'training steps',
    });
    $('#d2-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#d2-reset', root).onclick = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    $('#d2-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      pb.stop();
      run++;
      build();
      draw();
      pb.start();
    };
    $('#d2-gen', root).onclick = () => {
      if (!ts) {
        msg = 'Train the network first.';
        draw();
        return;
      }
      pb.stop();
      startGen();
      msg =
        'A fresh cloud of pure noise' +
        (ts < MAXS ? ', using the network as trained so far (' + ts + ' steps).' : '.');
      draw();
      pb.start();
    };
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
