/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  RM,
  T,
  bind,
  f2,
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
import { sig } from '../../classification/shared.js';
import { uCur, uLine, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#ga-c', root),
      cr = $('#ga-r', root),
      live = $('#ga-live', root),
      S = { ds: 'ring', dk: 1, lr: 2 },
      MAXS = 3000,
      B = 64,
      RN = randnFrom(Math.random);
    let G,
      D,
      ts = 0,
      tg = 0,
      td = 0,
      LD = [],
      LG = [],
      AR = [],
      AF = [],
      fake = [],
      real = [],
      done = false,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1;
    const samp = () => {
      if (S.ds === 'ring') {
        const k = Math.floor(Math.random() * 6),
          a = (k / 6) * 2 * Math.PI;
        return [1.3 * Math.cos(a) + 0.09 * RN(), 1.3 * Math.sin(a) + 0.09 * RN()];
      }
      if (S.ds === 'moons') {
        const a = Math.random() * Math.PI;
        return Math.random() < 0.5
          ? [Math.cos(a) * 1.1 - 0.5 + 0.07 * RN(), -0.3 + Math.sin(a) * 1.0 + 0.07 * RN()]
          : [0.5 - Math.cos(a) * 1.1 + 0.07 * RN(), 0.3 - Math.sin(a) * 1.0 + 0.07 * RN()];
      }
      return [0.5 + 0.35 * RN(), 0.3 + 0.25 * RN()];
    };
    const build = () => {
      G = AD.mlp([2, 32, 32, 2], randnFrom(rng(run * 3 + 1)), 'relu');
      D = AD.mlp([2, 32, 32, 1], randnFrom(rng(run * 3 + 2)), 'relu');
      ts = tg = td = 0;
      LD = [];
      LG = [];
      AR = [];
      AF = [];
      done = false;
      cu = 0;
      real = Array.from({ length: 260 }, samp);
      view();
      msg =
        'The generator has random weights and has never seen the data: its gold dots are a clump in the wrong place. Press Train.';
    };
    const view = () => {
      const z = Float64Array.from({ length: 400 }, () => RN()),
        f = G.fwd(AD.input(z, [200, 2]));
      AD.clear();
      fake = [];
      for (let i = 0; i < 200; i++) fake.push([f.v[2 * i], f.v[2 * i + 1]]);
    };
    const dOn = (P) => {
      const x = new Float64Array(P.length * 2);
      P.forEach((p, i) => {
        x[2 * i] = p[0];
        x[2 * i + 1] = p[1];
      });
      const o = D.fwd(AD.input(x, [P.length, 2]));
      AD.clear();
      return o.v;
    };
    const modes = () => {
      if (S.ds !== 'ring') return null;
      const h = new Set();
      fake.forEach((p) => {
        for (let k = 0; k < 6; k++) {
          const a = (k / 6) * 2 * Math.PI;
          if (Math.hypot(p[0] - 1.3 * Math.cos(a), p[1] - 1.3 * Math.sin(a)) < 0.32) h.add(k);
        }
      });
      return h.size;
    };
    const train = (k) => {
      const ones = new Float64Array(B).fill(1),
        zeros = new Float64Array(B),
        lr = S.lr * 0.001;
      for (let q = 0; q < k && !done; q++) {
        let ld = 0;
        for (let d = 0; d < S.dk; d++) {
          const z = Float64Array.from({ length: 2 * B }, () => RN()),
            rl = new Float64Array(2 * B);
          for (let i = 0; i < B; i++) {
            const p = samp();
            rl[2 * i] = p[0];
            rl[2 * i + 1] = p[1];
          }
          const fk = G.fwd(AD.input(z, [B, 2]));
          AD.clear();
          let l = AD.bce(D.fwd(AD.input(rl, [B, 2])), ones);
          ld = l.v[0];
          AD.backward(l);
          l = AD.bce(D.fwd(AD.input(fk.v, [B, 2])), zeros);
          ld += l.v[0];
          AD.backward(l);
          AD.adam(D.ps, lr, ++td, 1);
        }
        const z2 = Float64Array.from({ length: 2 * B }, () => RN()),
          l = AD.bce(D.fwd(G.fwd(AD.input(z2, [B, 2]))), ones);
        AD.backward(l);
        AD.adam(G.ps, lr, ++tg, 1);
        for (const p of D.ps) p.g.fill(0);
        ts++;
        LD.push(ld);
        LG.push(l.v[0]);
        if (ts >= MAXS) done = true;
      }
      view();
      const a = dOn(real),
        b = dOn(fake);
      AR.push(a.filter((v) => v > 0).length / a.length);
      AF.push(b.filter((v) => v > 0).length / b.length);
      cu = done ? 4 : 1 + (ts % 3);
      const m = modes();
      msg =
        'Step ' +
        ts +
        ': the discriminator trained on 64 real and 64 fake points, then the generator trained to fool it. Right now the discriminator calls ' +
        pct(AR[AR.length - 1]) +
        ' of real points real and is fooled by ' +
        pct(AF[AF.length - 1]) +
        ' of the fakes.' +
        (m !== null ? ' The gold dots cover ' + m + ' of the 6 clusters.' : '') +
        (done
          ? ' It has had its ' +
            MAXS +
            ' steps and stops here. A GAN has no loss that says “finished”.'
          : '');
    };
    const draw = () => {
      const t = T();
      {
        const V = uView(c),
          { g, w, h, X, Y, s } = V,
          cs = 14,
          nx = Math.ceil(w / cs),
          ny = Math.ceil(h / cs),
          P = [];
        for (let a = 0; a < ny; a++)
          for (let b = 0; b < nx; b++)
            P.push([((b + 0.5) * cs - w / 2) / s / 1.9, -((a + 0.5) * cs - h / 2) / s / 1.9]);
        const dv = dOn(P);
        P.forEach((p, i) => {
          const pr2 = sig(dv[i]);
          g.fillStyle = mix(
            pr2 >= 0.5 ? t.accent : t.ok,
            t.bg,
            0,
            0.05 + 0.3 * Math.abs(pr2 - 0.5) * 2,
          );
          g.fillRect((i % nx) * cs, Math.floor(i / nx) * cs, cs, cs);
        });
        const X2 = (v) => X(v * 1.9),
          Y2 = (v) => Y(v * 1.9);
        g.fillStyle = mix(t.ink, t.ink, 0, 0.75);
        real.forEach((p) => {
          g.beginPath();
          g.arc(X2(p[0]), Y2(p[1]), 2.6, 0, 7);
          g.fill();
        });
        g.fillStyle = t.gold;
        g.strokeStyle = t.bg;
        g.lineWidth = 0.8;
        fake.forEach((p) => {
          g.beginPath();
          g.arc(X2(p[0]), Y2(p[1]), 3.4, 0, 7);
          g.fill();
          g.stroke();
        });
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5,
          ym = Math.max(2.5, ...LG.slice(-400), ...LD.slice(-400)),
          U = uLine(
            g,
            t,
            0,
            0,
            w,
            h1,
            LD.map((v) => Math.min(v, ym)),
            {
              title: 'losses: discriminator (green), generator (red)',
              n: Math.max(300, LD.length),
              col: t.accent,
              ym,
            },
          );
        g.strokeStyle = t.ok;
        g.lineWidth = 1.8;
        g.beginPath();
        LG.forEach((v, i) =>
          i ? g.lineTo(U.X(i), U.Y(Math.min(v, ym))) : g.moveTo(U.X(i), U.Y(Math.min(v, ym))),
        );
        g.stroke();
        const U2 = uLine(g, t, 0, h1, w, h - h1, AR, {
          title: 'discriminator says “real” for: real points (green), fakes (gold)',
          n: Math.max(100, AR.length),
          col: t.accent,
          ym: 1,
        });
        g.strokeStyle = t.gold;
        g.lineWidth = 2;
        g.beginPath();
        AF.forEach((v, i) => (i ? g.lineTo(U2.X(i), U2.Y(v)) : g.moveTo(U2.X(i), U2.Y(v))));
        g.stroke();
        g.setLineDash([4, 4]);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(U2.X(0), U2.Y(0.5));
        g.lineTo(w - 12, U2.Y(0.5));
        g.stroke();
        g.setLineDash([]);
      }
      live.textContent = msg;
      uCur(root, 'ga', cu);
      const m = modes();
      setR(root, 'step', ts + ' of ' + MAXS);
      setR(root, 'ld', ts ? f2(LD[ts - 1]) : '–');
      setR(root, 'lg', ts ? f2(LG[ts - 1]) : '–');
      setR(root, 'dr', AR.length ? pct(AR[AR.length - 1]) : '–');
      setR(root, 'df', AF.length ? pct(AF[AF.length - 1]) : '–');
      setR(root, 'md', m === null ? '(only for the ring)' : m + ' of 6');
    };
    build();
    if (RM) train(MAXS);
    const redo = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      fr++;
      if (ts < 40 && fr % 3) return;
      train(Math.min(ts < 40 ? 1 : 4, chk.remaining()));
      draw();
      chk();
    });
    pb = playBtn($('#ga-play', root), anim);
    chk = runCtl(pb, root, 'ga', {
      done: () => done,
      restart: () => {
        run++;
        build();
      },
      count: () => ts,
      what: 'steps',
    });
    $('#ga-step', root).onclick = () => {
      pb.stop();
      if (done) {
        run++;
        build();
      } else train(20);
      draw();
    };
    $('#ga-reset', root).onclick = redo;
    $('#ga-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      redo();
      pb.start();
    };
    bind(
      root,
      'ga-dk',
      (v) => {
        S.dk = v;
      },
      (v) => String(v),
    );
    bind(
      root,
      'ga-lr',
      (v) => {
        S.lr = v;
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
