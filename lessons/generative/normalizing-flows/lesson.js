/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  RM,
  T,
  bind,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  randnFrom,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import {
  G2S,
  g2Data,
  g2Near,
  g2Note,
  g2Panel,
  g2Pts,
  g2Samp,
  nfFwd,
  nfInv,
  nfLogp,
  nfMake,
} from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { uCur, uLine, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#nf-c', root),
      cr = $('#nf-r', root),
      live = $('#nf-live', root),
      S = { ds: 'moons' },
      K = 6,
      MAXS = 1500,
      M = 350,
      GW = 40,
      GH = 30,
      RN = randnFrom(Math.random);
    let F,
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
      Z,
      GZ,
      gn = 0,
      st = null,
      gs = null,
      gl = 2 * K,
      done = false,
      dens = null,
      dmax = 1,
      base = NaN,
      lpD = null,
      nearV = NaN,
      lastD = 0;
    const mkGrid = () => {
      const P = [];
      for (let a = -2; a <= 2.001; a += 0.5)
        for (let b = 0; b <= 48; b++) P.push(a, -2 + (4 * b) / 48);
      for (let a = -2; a <= 2.001; a += 0.5)
        for (let b = 0; b <= 48; b++) P.push(-2 + (4 * b) / 48, a);
      gn = P.length / 2;
      return Float64Array.from(P);
    };
    const refresh = (full) => {
      st = nfInv(F, Z, M);
      gs = nfInv(F, GZ, gn);
      if (full) {
        lastD = ts;
        const G = new Float64Array(2 * GW * GH);
        for (let r = 0; r < GH; r++)
          for (let q = 0; q < GW; q++) {
            G[2 * (r * GW + q)] = (-4 + (8 * (q + 0.5)) / GW) / G2S;
            G[2 * (r * GW + q) + 1] = (3 - (6 * (r + 0.5)) / GH) / G2S;
          }
        const lp = nfLogp(F, G, GW * GH);
        dens = lp.map(Math.exp);
        dmax = Math.max(1e-9, ...dens);
        const D = new Float64Array(2 * data.length);
        data.forEach((p, i) => {
          D[2 * i] = p[0];
          D[2 * i + 1] = p[1];
        });
        lpD = nfLogp(F, D, data.length);
      }
    };
    const build = () => {
      F = nfMake(run * 5 + 3, K, 32);
      ts = 0;
      LL = [];
      sm = [];
      phase = 'train';
      done = false;
      cu = 0;
      gl = 2 * K;
      nearV = NaN;
      data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
      Z = Float64Array.from({ length: 2 * M }, () => RN());
      GZ = mkGrid();
      let b = 0;
      for (const p of data) b += (0.5 * (p[0] * p[0] + p[1] * p[1])) / data.length;
      base = b + Math.log(2 * Math.PI);
      refresh(true);
      msg =
        'Six layers that do almost nothing yet: the gold cloud is still a round Gaussian and the grid is still square. Press Run to train them.';
    };
    const train = (k) => {
      const B = 96;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const X = new Float64Array(2 * B);
        for (let i = 0; i < B; i++) {
          const p = g2Samp(S.ds, RN);
          X[2 * i] = p[0];
          X[2 * i + 1] = p[1];
        }
        const { z, As } = nfFwd(F, X, B),
          l = AD.nfl(z, As);
        LL.push(l.v[0]);
        AD.backward(l);
        AD.adam(F.ps, ts < 1100 ? 0.004 : 0.0015, ++ts, 1);
        if (ts % 20 === 0) {
          let s = 0;
          for (let j = ts - 20; j < ts; j++) s += LL[j];
          sm.push(s / 20);
        }
      }
      refresh(ts - lastD >= 30 || ts >= MAXS);
      cu = Math.floor(ts / 60) % 3;
      msg =
        'Training step ' +
        ts +
        ' of ' +
        MAXS +
        ': 96 real points were pulled back through the layers toward the Gaussian, their exact log-probability was computed, and the layers were adjusted to raise it. Loss over the last 20 steps: ' +
        f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
        ' (a plain round Gaussian scores ' +
        f3(base) +
        ').';
      if (ts >= MAXS) {
        phase = 'gen';
        gl = 0;
        done = false;
        cu = 3;
        nearV = g2Near(st[2 * K], M, data);
        msg =
          'Training is finished. The left picture is back at the start: 350 points from a round Gaussian, with no layers applied. Press Run to send them through the six layers.';
      }
    };
    const gstep = () => {
      if (done) return;
      gl = Math.min(2 * K, Math.floor(gl + 1e-6) + 1);
      const ly = Math.ceil(gl / 2);
      cu = 4;
      msg =
        gl % 2
          ? 'Layer ' +
            ly +
            ' of ' +
            K +
            ', first half: every point slides and stretches up or down, by an amount that depends on how far left or right it is. Grid cells that shrink here are where the density rises.'
          : 'Layer ' +
            ly +
            ' of ' +
            K +
            ', second half: the whole picture is turned by 60°' +
            (ly < K ? ', so that the next layer stretches along a new direction' : '') +
            '. A turn does not change any area, so it does not change any density.';
      if (gl >= 2 * K) {
        done = true;
        cu = 5;
        msg += ' All six layers are applied: ' + pct(nearV) + ' of the points lie on the shape.';
      }
    };
    const step = () => {
      if (phase === 'train') train(50);
      else if (done) {
        gl = 0;
        done = false;
        cu = 3;
        msg = 'Back to the round Gaussian, with no layers applied. Press Run or One step.';
      } else gstep();
    };
    const at = (sn, i, n, o) => {
      const j = Math.min(2 * K - 1, Math.floor(gl)),
        u = gl >= 2 * K ? 1 : gl - j,
        A = sn[j],
        B = sn[j + 1];
      if (j % 2 === 0) {
        o[0] = A[2 * i] + (B[2 * i] - A[2 * i]) * u;
        o[1] = A[2 * i + 1] + (B[2 * i + 1] - A[2 * i + 1]) * u;
      } else {
        const th = (-Math.PI / 3) * u,
          cs = Math.cos(th),
          sn2 = Math.sin(th),
          x = A[2 * i],
          y = A[2 * i + 1];
        o[0] = x * cs - y * sn2;
        o[1] = x * sn2 + y * cs;
      }
      return o;
    };
    const tp = testPoint(
      root,
      'nf',
      ['Your x', 'Your y'],
      [0, 0],
      [
        [-2.2, 2.2],
        [-1.6, 1.6],
      ],
    );
    const draw = () => {
      const t = T(),
        o = [0, 0];
      {
        const V = uView(c),
          { g, h } = V;
        g2Data(g, V, t, data);
        if (gs) {
          g.strokeStyle = mix(t.accent, t.accent, 0, 0.38);
          g.lineWidth = 1;
          g.lineJoin = 'round';
          for (let ln = 0; ln < gn / 49; ln++) {
            g.beginPath();
            for (let b = 0; b < 49; b++) {
              at(gs, ln * 49 + b, gn, o);
              const x = V.X(o[0] * G2S),
                y = V.Y(o[1] * G2S);
              b ? g.lineTo(x, y) : g.moveTo(x, y);
            }
            g.stroke();
          }
        }
        if (st) {
          const P = new Float64Array(2 * M);
          for (let i = 0; i < M; i++) {
            at(st, i, M, o);
            P[2 * i] = o[0];
            P[2 * i + 1] = o[1];
          }
          g2Pts(g, V, t, P, M, t.gold);
        }
        markTP(g, V.X(tp.v[0] * G2S), V.Y(tp.v[1] * G2S), t);
        g2Note(
          g,
          t,
          h,
          phase === 'train'
            ? 'all ' + K + ' layers applied, as trained so far'
            : f2(gl / 2) + ' of ' + K + ' layers applied',
        );
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.56,
          P = g2Panel(
            g,
            t,
            0,
            0,
            w,
            h1,
            'exact density of the model: darker means more probable',
            1.2,
          );
        if (dens) {
          for (let r = 0; r < GH; r++)
            for (let q = 0; q < GW; q++) {
              const v = dens[r * GW + q] / dmax;
              if (v < 0.012) continue;
              g.fillStyle = mix(t.accent, t.accent, 0, Math.min(0.95, Math.pow(v, 0.6)));
              const xa = Math.round(P.X(-4 + (8 * q) / GW)),
                xb = Math.round(P.X(-4 + (8 * (q + 1)) / GW)),
                ya = Math.round(P.Y(3 - (6 * r) / GH)),
                yb = Math.round(P.Y(3 - (6 * (r + 1)) / GH));
              g.fillRect(xa, ya, xb - xa, yb - ya);
            }
        }
        g2Data(g, P, t, data, 0.3, 1.2);
        markTP(g, P.X(tp.v[0] * G2S), P.Y(tp.v[1] * G2S), t, false);
        g.restore();
        const Lp = uLine(g, t, 0, h1, w, h - h1, sm, {
          title: 'minus the average log-probability of the data (lower is better)',
          n: MAXS / 20,
          col: t.ok,
          ym: Math.max(base * 1.08, sm[0] || 0),
          y1: Math.min(0, ...sm),
        });
        g.setLineDash([5, 4]);
        g.strokeStyle = t.gold;
        g.lineWidth = 1.4;
        g.beginPath();
        g.moveTo(Lp.m.l, Lp.Y(base));
        g.lineTo(w - Lp.m.r, Lp.Y(base));
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = t.cut;
        g.textAlign = 'right';
        g.fillText('a plain round Gaussian', w - Lp.m.r, Lp.Y(base) + 14);
      }
      live.textContent = msg;
      uCur(root, 'nf', cu);
      {
        const lp = nfLogp(F, Float64Array.of(tp.v[0], tp.v[1]), 1)[0];
        let lo = 0;
        if (lpD) for (const v of lpD) if (v < lp) lo++;
        tp.out.textContent = ts
          ? 'The model’s probability density at your point is ' +
            (Math.exp(lp) < 0.001 ? Math.exp(lp).toExponential(1) : f3(Math.exp(lp))) +
            ' (log-density ' +
            f2(lp) +
            '). That is higher than the density at ' +
            (lpD ? ((100 * lo) / lpD.length).toFixed(0) : '–') +
            '% of the real data points.'
          : 'Train the model first: press Run.';
      }
      setR(
        root,
        'ph',
        phase === 'train'
          ? ts
            ? 'Training'
            : 'Not started'
          : done
            ? 'Finished'
            : 'Applying the layers',
      );
      setR(root, 'ts', ts + ' of ' + MAXS);
      setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
      setR(root, 'base', f3(base));
      setR(root, 'ly', (phase === 'train' ? K : f2(gl / 2)) + ' of ' + K);
      setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
    };
    tp.on = () => draw();
    build();
    if (RM) {
      train(MAXS);
      gl = 2 * K;
      done = true;
    }
    const anim = Anim(() => {
      fr++;
      if (phase === 'train') {
        train(Math.min(ts < 40 ? 1 : 4, chk.remaining()));
        draw();
        chk();
        return;
      }
      if (!done) {
        const n = gl + 1 / 45;
        if (Math.floor(n + 1e-9) > Math.floor(gl + 1e-9) || n >= 2 * K) {
          gstep();
        } else gl = n;
      }
      draw();
      if (done) pb.stop();
    });
    pb = playBtn($('#nf-play', root), anim);
    chk = runCtl(pb, root, 'nf', {
      done: () => phase === 'gen' && done,
      restart: () => {
        gl = 0;
        done = false;
      },
      count: () => ts,
      what: 'training steps',
    });
    $('#nf-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#nf-reset', root).onclick = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    $('#nf-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      pb.stop();
      run++;
      build();
      draw();
      pb.start();
    };
    $('#nf-gen', root).onclick = () => {
      if (phase !== 'gen') {
        msg = 'The layers can be played one by one once training is finished. Press Run.';
        draw();
        return;
      }
      pb.stop();
      gl = 0;
      done = false;
      draw();
      pb.start();
    };
    bind(
      root,
      'nf-ly',
      (v) => {
        if (!pb) return;
        if (phase !== 'gen') {
          draw();
          return;
        }
        pb.stop();
        gl = 2 * +v;
        done = gl >= 2 * K;
        cu = 4;
        msg = 'Showing the cloud after ' + v + ' of ' + K + ' layers.';
        draw();
      },
      (v) => String(v),
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
