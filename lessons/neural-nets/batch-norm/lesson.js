/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  eqShow,
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
import { dotAt, segBind, uCur, uLine } from '../../unsupervised/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#bn-c', root),
      cr = $('#bn-r', root),
      live = $('#bn-live', root),
      eq = $('#bn-eq', root),
      S = { d: 'spiral', w: 0.3, lr: 0.1, b: 16, L: 6 },
      H = 10;
    let D = [],
      Y = [],
      nets = [],
      ep = 0,
      fin = false,
      hist = [[], []],
      acc = [0, 0],
      rng_ = null,
      seed = 1,
      msg = '',
      cu = 0,
      pb,
      chk = () => false,
      fld = [null, null],
      GW = 30,
      GH = 30,
      work = null;
    const gen = () => {
      const r = rng(41 + S.d.length),
        rn = randnFrom(r);
      D = [];
      Y = [];
      for (let i = 0; i < 160; i++) {
        const k = i % 2;
        let p;
        if (S.d === 'spiral') {
          const t = 0.35 + 2.3 * r() * Math.PI * 0.75,
            s = k ? 1 : -1;
          p = [s * t * 0.55 * Math.cos(t) + 0.07 * rn(), s * t * 0.55 * Math.sin(t) + 0.07 * rn()];
        } else if (S.d === 'rings') {
          const a = 2 * Math.PI * r(),
            rad = k ? 2.1 : 0.7;
          p = [rad * Math.cos(a) + 0.14 * rn(), rad * Math.sin(a) + 0.14 * rn()];
        } else {
          const a = Math.PI * r();
          p = k
            ? [1.5 * Math.cos(a) - 0.7 + 0.12 * rn(), 1.5 * Math.sin(a) - 0.4 + 0.12 * rn()]
            : [0.7 - 1.5 * Math.cos(a) + 0.12 * rn(), 0.4 - 1.5 * Math.sin(a) + 0.12 * rn()];
        }
        D.push(p);
        Y.push(k);
      }
    };
    const mk = (bn) => {
      const r = randnFrom(rng(seed * 97 + 5)),
        sz = [2, ...Array(S.L).fill(H), 1],
        W = [],
        B = [],
        G = [],
        Be = [],
        RM_ = [],
        RV = [];
      for (let l = 0; l < sz.length - 1; l++) {
        const sd = S.w * Math.sqrt(2 / (sz[l] + sz[l + 1]));
        W.push(
          Array.from({ length: sz[l + 1] }, () => Array.from({ length: sz[l] }, () => sd * r())),
        );
        B.push(new Array(sz[l + 1]).fill(0));
        G.push(new Array(sz[l + 1]).fill(1));
        Be.push(new Array(sz[l + 1]).fill(0));
        RM_.push(new Array(sz[l + 1]).fill(0));
        RV.push(new Array(sz[l + 1]).fill(1));
      }
      return { bn, W, B, G, Be, RM: RM_, RV, pre: [] };
    };
    const fwd = (N, X, train, keep) => {
      const m = X.length,
        L = N.W.length,
        cache = { A: [X], Z: [], Zh: [], sd: [] };
      let A = X;
      N.pre = [];
      for (let l = 0; l < L; l++) {
        const no = N.W[l].length,
          Z = A.map((a) =>
            N.W[l].map((w, j) => {
              let s = N.B[l][j];
              for (let k = 0; k < w.length; k++) s += w[k] * a[k];
              return s;
            }),
          );
        cache.Z.push(Z);
        if (l === L - 1) {
          A = Z.map((z) => [1 / (1 + Math.exp(-clamp(z[0], -30, 30)))]);
          cache.A.push(A);
          break;
        }
        let U = Z;
        if (N.bn) {
          const mu = new Array(no).fill(0),
            va = new Array(no).fill(0);
          if (train) {
            for (const z of Z) for (let j = 0; j < no; j++) mu[j] += z[j] / m;
            for (const z of Z) for (let j = 0; j < no; j++) va[j] += (z[j] - mu[j]) ** 2 / m;
            for (let j = 0; j < no; j++) {
              N.RM[l][j] = 0.9 * N.RM[l][j] + 0.1 * mu[j];
              N.RV[l][j] = 0.9 * N.RV[l][j] + 0.1 * va[j];
            }
          } else
            for (let j = 0; j < no; j++) {
              mu[j] = N.RM[l][j];
              va[j] = N.RV[l][j];
            }
          const sd = va.map((v) => Math.sqrt(v + 1e-5)),
            Zh = Z.map((z) => z.map((v, j) => (v - mu[j]) / sd[j]));
          cache.Zh.push(Zh);
          cache.sd.push(sd);
          cache.mu = cache.mu || [];
          cache.mu.push(mu);
          U = Zh.map((z) => z.map((v, j) => N.G[l][j] * v + N.Be[l][j]));
        } else {
          cache.Zh.push(null);
          cache.sd.push(null);
        }
        if (keep) N.pre.push(U.flat());
        cache.U = cache.U || [];
        cache.U.push(U);
        A = U.map((u) => u.map(Math.tanh));
        cache.A.push(A);
      }
      return cache;
    };
    const train = (N, X, y) => {
      const m = X.length,
        L = N.W.length,
        C = fwd(N, X, true, false);
      let dZ = C.A[L].map((a, i) => [(a[0] - y[i]) / m]);
      for (let l = L - 1; l >= 0; l--) {
        const Ain = C.A[l],
          no = N.W[l].length,
          ni = N.W[l][0].length;
        let dA = null;
        if (l > 0) {
          dA = Ain.map(() => new Array(ni).fill(0));
          for (let i = 0; i < m; i++)
            for (let j = 0; j < no; j++) {
              const d = dZ[i][j],
                w = N.W[l][j];
              for (let k = 0; k < ni; k++) dA[i][k] += d * w[k];
            }
        }
        for (let j = 0; j < no; j++) {
          let gb = 0;
          const w = N.W[l][j],
            gw = new Array(ni).fill(0);
          for (let i = 0; i < m; i++) {
            const d = dZ[i][j];
            gb += d;
            for (let k = 0; k < ni; k++) gw[k] += d * Ain[i][k];
          }
          for (let k = 0; k < ni; k++) w[k] -= S.lr * clamp(gw[k], -5, 5);
          N.B[l][j] -= S.lr * clamp(gb, -5, 5);
        }
        if (l > 0) {
          const lp = l - 1,
            U = C.U[lp],
            dU = dA.map((r, i) => r.map((v, j) => v * (1 - Math.tanh(U[i][j]) ** 2)));
          if (N.bn) {
            const Zh = C.Zh[lp],
              sd = C.sd[lp],
              n2 = N.W[lp].length;
            dZ = dU.map((r) => r.slice());
            for (let j = 0; j < n2; j++) {
              let s1 = 0,
                s2 = 0,
                gg = 0,
                gb = 0;
              for (let i = 0; i < m; i++) {
                const dh = dU[i][j] * N.G[lp][j];
                s1 += dh;
                s2 += dh * Zh[i][j];
                gg += dU[i][j] * Zh[i][j];
                gb += dU[i][j];
              }
              for (let i = 0; i < m; i++) {
                const dh = dU[i][j] * N.G[lp][j];
                dZ[i][j] = (m * dh - s1 - Zh[i][j] * s2) / (m * sd[j]);
              }
              N.G[lp][j] -= S.lr * clamp(gg, -5, 5);
              N.Be[lp][j] -= S.lr * clamp(gb, -5, 5);
            }
          } else dZ = dU;
        }
      }
    };
    const evalN = (N) => {
      const C = fwd(N, D, false, true),
        P = C.A[N.W.length];
      let l = 0,
        a = 0;
      P.forEach((p, i) => {
        const q = clamp(p[0], 1e-7, 1 - 1e-7);
        l -= Y[i] * Math.log(q) + (1 - Y[i]) * Math.log(1 - q);
        if ((q > 0.5 ? 1 : 0) === Y[i]) a++;
      });
      return [l / D.length, a / D.length];
    };
    const field = (k) => {
      const X = [];
      for (let r = 0; r < GH; r++)
        for (let q = 0; q < GW; q++) X.push([-3 + (6 * (q + 0.5)) / GW, 3 - (6 * (r + 0.5)) / GH]);
      const keep = nets[k].pre;
      const P = fwd(nets[k], X, false, false).A[nets[k].W.length];
      nets[k].pre = keep;
      fld[k] = P.map((p) => p[0]);
    };
    const reset = () => {
      nets = [mk(false), mk(true)];
      for (let w = 0; w < 3; w++) fwd(nets[1], D, true, false);
      ep = 0;
      fin = false;
      hist = [[], []];
      rng_ = rng(seed * 3 + 1);
      for (const k of [0, 1]) {
        const [l, a] = evalN(nets[k]);
        hist[k].push(l);
        acc[k] = a;
        field(k);
      }
      cu = 0;
      work = null;
      msg =
        'Two networks with identical starting weights' +
        (S.w > 1.5
          ? ', ' + f2(S.w) + ' times larger than the careful choice'
          : S.w < 0.7
            ? ', ' + f2(S.w) + ' times the careful size, so too small'
            : '') +
        '. Press Run.';
      eqShow(eq, ['No batch yet. Press One pass or Run.'], 0);
    };
    const pass = () => {
      if (fin) return;
      const n = D.length,
        idx = D.map((_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rng_() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      for (let s = 0; s + 1 < n; s += S.b) {
        const bi = idx.slice(s, s + S.b);
        if (bi.length < 2) continue;
        const X = bi.map((i) => D[i]),
          y = bi.map((i) => Y[i]);
        if (s === 0) {
          const z = X.map(
              (a) => nets[1].W[0][0][0] * a[0] + nets[1].W[0][0][1] * a[1] + nets[1].B[0][0],
            ).slice(0, 5),
            mu = z.reduce((p, v) => p + v, 0) / z.length,
            sd = Math.sqrt(z.reduce((p, v) => p + (v - mu) ** 2, 0) / z.length + 1e-5);
          work = { z, mu, sd, g: nets[1].G[0][0], b: nets[1].Be[0][0] };
        }
        for (const N of nets) train(N, X, y);
      }
      ep++;
      for (const k of [0, 1]) {
        const [l, a] = evalN(nets[k]);
        hist[k].push(l);
        acc[k] = a;
        if (ep % 2 === 0 || ep < 6) field(k);
      }
      cu = ep % 5;
      const h = hist[1];
      msg =
        'Pass ' +
        ep +
        ': loss ' +
        f3(hist[0][ep]) +
        ' for the plain network and ' +
        f3(h[ep]) +
        ' with batch normalization.';
      if (work)
        eqShow(
          eq,
          [
            'First ' +
              work.z.length +
              ' values of one unit in this batch: ' +
              work.z.map(f2).join(', '),
            'batch mean μ = <b>' +
              f3(work.mu) +
              '</b>, batch standard deviation σ = <b>' +
              f3(work.sd) +
              '</b>',
            'normalized: (z − μ) ÷ σ = ' +
              work.z.map((v) => f2((v - work.mu) / work.sd)).join(', '),
            'learned scale γ = ' + f3(work.g) + ', learned shift β = ' + f3(work.b),
            'into tanh: γ × normalized + β = <b>' +
              work.z.map((v) => f2((work.g * (v - work.mu)) / work.sd + work.b)).join(', ') +
              '</b>',
          ],
          0,
        );
      if ((ep >= 40 && Math.abs(h[ep] - h[ep - 25]) < 0.002) || ep >= 300) {
        fin = true;
        for (const k of [0, 1]) field(k);
        msg =
          'Stopped after ' +
          ep +
          ' passes: the normalized network’s loss has levelled off. Plain network: loss ' +
          f3(hist[0][ep]) +
          ', accuracy ' +
          pct(acc[0]) +
          '. With batch normalization: loss ' +
          f3(h[ep]) +
          ', accuracy ' +
          pct(acc[1]) +
          '.';
      }
    };
    const q90 = (a) => {
      const s = Float64Array.from(a).sort();
      return [s[Math.floor(s.length * 0.05)], s[Math.floor(s.length * 0.95)]];
    };
    const draw = () => {
      const t = T();
      {
        const { g, w, h } = fit(c),
          pw = (w - 14) / 2,
          sz = Math.min(pw, h - 26);
        for (const k of [0, 1]) {
          const x0 = k * (pw + 14) + (pw - sz) / 2,
            y0 = 22,
            X = (x) => x0 + ((x + 3) / 6) * sz,
            Yc = (y) => y0 + ((3 - y) / 6) * sz;
          for (let r = 0; r < GH; r++)
            for (let q = 0; q < GW; q++) {
              const v = fld[k][r * GW + q];
              g.fillStyle =
                v > 0.5 ? mix(t.bg, t.gold, (v - 0.5) * 0.9) : mix(t.bg, t.accent, (0.5 - v) * 0.9);
              const xa = Math.round(x0 + (q * sz) / GW),
                xb = Math.round(x0 + ((q + 1) * sz) / GW),
                ya = Math.round(y0 + (r * sz) / GH),
                yb = Math.round(y0 + ((r + 1) * sz) / GH);
              g.fillRect(xa, ya, xb - xa, yb - ya);
            }
          D.forEach((p, i) => {
            g.lineWidth = 1;
            dotAt(g, X(p[0]), Yc(p[1]), 2.6, Y[i] ? t.gold : t.accent, t.bg);
          });
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText(k ? 'with batch normalization' : 'plain network', x0 + sz / 2, 14);
        }
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.46,
          ym = Math.max(0.75, ...hist[0].slice(0, 3), ...hist[1].slice(0, 3)) * 1.05,
          A = uLine(g, t, 0, 0, w, h1, [], {
            title: 'training loss: plain (amber), batch normalization (green)',
            n: Math.max(40, ep + 1),
            ym,
            y1: 0,
          });
        g.save();
        g.beginPath();
        g.rect(0, 16, w, h1 - 34);
        g.clip();
        [
          [0, t.cut],
          [1, t.accent],
        ].forEach(([k, col]) => {
          g.strokeStyle = col;
          g.lineWidth = 2.2;
          g.beginPath();
          hist[k].forEach((v, i) =>
            i
              ? g.lineTo(A.X(i), A.Y(Math.min(v, ym * 2)))
              : g.moveTo(A.X(i), A.Y(Math.min(v, ym * 2))),
          );
          g.stroke();
        });
        g.restore();
        const y0 = h1 + 6,
          H2 = h - y0,
          m = { l: 86, r: 14, t: 24, b: 8 },
          rh = (H2 - m.t - m.b) / S.L,
          R = 8,
          X = (v) => m.l + ((clamp(v, -R, R) + R) / (2 * R)) * (w - m.l - m.r);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('inputs to tanh at each hidden layer (−8 to 8)', 8, y0 + 12);
        g.strokeStyle = t.muted;
        g.setLineDash([4, 4]);
        g.lineWidth = 1;
        for (const v of [-2, 2]) {
          g.beginPath();
          g.moveTo(X(v), y0 + m.t);
          g.lineTo(X(v), h - m.b);
          g.stroke();
        }
        g.setLineDash([]);
        for (let l = 0; l < S.L; l++) {
          const yy = y0 + m.t + l * rh;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('layer ' + (l + 1), m.l - 8, yy + rh / 2 + 4);
          [
            [0, t.cut],
            [1, t.accent],
          ].forEach(([k, col]) => {
            const pr = nets[k].pre[l];
            if (!pr) return;
            const [a, b] = q90(pr);
            g.fillStyle = col;
            g.fillRect(X(a), yy + 3 + k * (rh / 2 - 2), Math.max(2, X(b) - X(a)), rh / 2 - 5);
          });
        }
      }
      const sat = (k) => {
        const pr = nets[k].pre[S.L - 1];
        if (!pr) return NaN;
        let s = 0;
        for (const v of pr) s += v * v;
        return Math.sqrt(s / pr.length);
      };
      live.textContent = msg;
      uCur(root, 'bn', cu);
      setR(root, 'ep', ep + (fin ? ' (stopped)' : ''));
      setR(root, 'lp', f3(hist[0][ep]));
      setR(root, 'lb', f3(hist[1][ep]));
      setR(root, 'ap', pct(acc[0]));
      setR(root, 'ab', pct(acc[1]));
      setR(root, 'sat', f3(sat(0)) + ' / ' + f3(sat(1)));
    };
    gen();
    reset();
    const anim = Anim(() => {
      if (fin) {
        pb.stop();
        return;
      }
      pass();
      draw();
      chk();
    });
    pb = playBtn($('#bn-play', root), anim);
    chk = runCtl(pb, root, 'bn', {
      done: () => fin,
      restart: reset,
      count: () => ep,
      what: 'passes',
    });
    const go = () => {
        pb.stop();
        reset();
        draw();
        pb.start();
      },
      ch = () => {
        pb.stop();
        reset();
        draw();
      };
    $('#bn-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else pass();
      draw();
    };
    $('#bn-reset', root).onclick = ch;
    $('#bn-new', root).onclick = () => {
      seed++;
      go();
    };
    bind(
      root,
      'bn-w',
      (v) => {
        S.w = v;
        ch();
      },
      (v) => (+v).toFixed(1) + '×',
    );
    bind(
      root,
      'bn-lr',
      (v) => {
        S.lr = v;
      },
      f3,
    );
    bind(
      root,
      'bn-b',
      (v) => {
        S.b = v;
      },
      (v) => String(+v),
    );
    bind(
      root,
      'bn-L',
      (v) => {
        S.L = v;
        ch();
      },
      (v) => String(+v),
    );
    segBind(root, 'bn-d', S.d, (v) => {
      S.d = v;
      gen();
      go();
    });
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
