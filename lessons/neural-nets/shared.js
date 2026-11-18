/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { clamp, mix, randnFrom, rng } from '../../js/core/shared.js';
import { sig } from '../classification/shared.js';

const ACT = {
  tanh: [
    Math.tanh,
    (z) => {
      const a = Math.tanh(z);
      return 1 - a * a;
    },
  ],
  relu: [(z) => (z > 0 ? z : 0), (z) => (z > 0 ? 1 : 0)],
  sigmoid: [
    sig,
    (z) => {
      const a = sig(z);
      return a * (1 - a);
    },
  ],
  leaky: [(z) => (z > 0 ? z : 0.1 * z), (z) => (z > 0 ? 1 : 0.1)],
};

const ACTN = { tanh: 'tanh', relu: 'ReLU', sigmoid: 'Sigmoid', leaky: 'Leaky ReLU' };

function NNet(sizes, o = {}) {
  const act = o.act || 'tanh',
    out = o.out || 'sig',
    r = randnFrom(rng(o.seed || 1)),
    L = sizes.length - 1,
    W = [],
    B = [],
    mW = [],
    vW = [],
    mB = [],
    vB = [],
    f = ACT[act][0],
    df = ACT[act][1];
  let ts = 0;
  for (let l = 0; l < L; l++) {
    const n = sizes[l],
      m = sizes[l + 1],
      sd = (o.scale || 1) * Math.sqrt((act === 'relu' || act === 'leaky' ? 2 : 1) / n);
    W.push(Array.from({ length: m }, () => Float64Array.from({ length: n }, () => sd * r())));
    B.push(new Float64Array(m));
    mW.push(W[l].map(() => new Float64Array(n)));
    vW.push(W[l].map(() => new Float64Array(n)));
    mB.push(new Float64Array(m));
    vB.push(new Float64Array(m));
  }
  const fwd = (x, mask) => {
    const A = [x],
      Z = [];
    for (let l = 0; l < L; l++) {
      const m = sizes[l + 1],
        z = new Float64Array(m),
        a = new Float64Array(m),
        last = l === L - 1,
        p = A[l];
      for (let j = 0; j < m; j++) {
        let s = B[l][j];
        const w = W[l][j];
        for (let i = 0; i < w.length; i++) s += w[i] * p[i];
        z[j] = s;
        a[j] = last ? (out === 'sig' ? sig(s) : s) : f(s) * (mask ? mask[l][j] : 1);
      }
      Z.push(z);
      A.push(a);
    }
    return { A, Z };
  };
  const train = (X, Y, lr, q = {}) => {
    const gW = W.map((wl) => wl.map((w) => new Float64Array(w.length))),
      gB = B.map((b) => new Float64Array(b.length)),
      p = q.drop || 0;
    let loss = 0,
      lastMask = null;
    for (let n = 0; n < X.length; n++) {
      let mask = null;
      if (p > 0) {
        mask = sizes
          .slice(1, L)
          .map((m) =>
            Float64Array.from({ length: m }, () => (Math.random() < p ? 0 : 1 / (1 - p))),
          );
        lastMask = mask;
      }
      const { A, Z } = fwd(X[n], mask),
        y = Y[n];
      let d;
      if (y.length) {
        const O = A[L];
        d = new Float64Array(O.length);
        for (let k = 0; k < O.length; k++) {
          d[k] = O[k] - y[k];
          loss +=
            out === 'sig'
              ? -(y[k] * Math.log(O[k] + 1e-9) + (1 - y[k]) * Math.log(1 - O[k] + 1e-9))
              : 0.5 * d[k] * d[k];
        }
      } else {
        const ov = A[L][0];
        loss +=
          out === 'sig'
            ? -(y * Math.log(ov + 1e-9) + (1 - y) * Math.log(1 - ov + 1e-9))
            : 0.5 * (ov - y) ** 2;
        d = Float64Array.of(ov - y);
      }
      for (let l = L - 1; l >= 0; l--) {
        const pa = A[l];
        for (let j = 0; j < d.length; j++) {
          gB[l][j] += d[j];
          const g = gW[l][j],
            dj = d[j];
          for (let i = 0; i < pa.length; i++) g[i] += dj * pa[i];
        }
        if (l > 0) {
          const nd = new Float64Array(sizes[l]);
          for (let i = 0; i < nd.length; i++) {
            let s = 0;
            for (let j = 0; j < d.length; j++) s += W[l][j][i] * d[j];
            nd[i] = s * df(Z[l - 1][i]) * (mask ? mask[l - 1][i] : 1);
          }
          d = nd;
        }
      }
    }
    ts++;
    const N = X.length,
      c1 = 1 - Math.pow(0.9, ts),
      c2 = 1 - Math.pow(0.999, ts),
      up = (par, g, m, v) => {
        for (let i = 0; i < par.length; i++) {
          const gi = g[i] / N;
          if (q.sgd) par[i] -= lr * gi;
          else {
            m[i] = 0.9 * m[i] + 0.1 * gi;
            v[i] = 0.999 * v[i] + 0.001 * gi * gi;
            par[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
          }
        }
      };
    for (let l = q.freeze || 0; l < L; l++) {
      for (let j = 0; j < W[l].length; j++) up(W[l][j], gW[l][j], mW[l][j], vW[l][j]);
      up(B[l], gB[l], mB[l], vB[l]);
    }
    return { loss: loss / N, mask: lastMask };
  };
  return {
    W,
    B,
    sizes,
    L,
    fwd,
    train,
    predictScalar: (x) => fwd(x).A[L][0],
    predictVector: (x) => fwd(x).A[L],
  };
}

function nnData(kind, n, nz, seed) {
  const r = rng(seed * 19 + 3),
    rn = randnFrom(r),
    P = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    let x, y;
    if (kind === 'circles') {
      const a = r() * 2 * Math.PI,
        rad = c ? 2 + 0.25 * rn() * (1 + 2 * nz) : 0.7 * Math.sqrt(r());
      x = rad * Math.cos(a) + nz * 0.5 * rn();
      y = rad * Math.sin(a) + nz * 0.5 * rn();
    } else if (kind === 'moons') {
      const a = r() * Math.PI;
      x = c ? 0.7 - 1.6 * Math.cos(a) : -0.7 + 1.6 * Math.cos(a);
      y = c ? 0.4 - 1.5 * Math.sin(a) : -0.4 + 1.5 * Math.sin(a);
      x += nz * rn();
      y += nz * rn();
    } else if (kind === 'xor') {
      x = (r() * 2.4 + 0.2) * (r() < 0.5 ? -1 : 1);
      y = (r() * 2.4 + 0.2) * (r() < 0.5 ? -1 : 1);
      x += nz * 0.6 * rn();
      y += nz * 0.6 * rn();
      P.push({
        x: clamp(x, -2.9, 2.9),
        y: clamp(y, -2.9, 2.9),
        c: x * y > 0 ? 1 : 0,
        test: i % 4 === 3,
      });
      continue;
    } else if (kind === 'spiral') {
      const u = r(),
        a = u * 3.6 + c * Math.PI,
        rad = 0.3 + 2.3 * u;
      x = rad * Math.cos(a) + nz * 0.35 * rn();
      y = rad * Math.sin(a) + nz * 0.35 * rn();
    } else {
      x = (c ? 1.2 : -1.2) + (0.6 + nz) * rn();
      y = (c ? 0.8 : -0.8) + (0.6 + nz) * rn();
    }
    P.push({ x: clamp(x, -2.9, 2.9), y: clamp(y, -2.9, 2.9), c, test: i % 4 === 3 });
  }
  return P;
}

function netPos(sizes, x0, y0, w, h) {
  return sizes.map((m, l) =>
    Array.from({ length: m }, (_, j) => [
      x0 + (sizes.length > 1 ? l / (sizes.length - 1) : 0.5) * w,
      y0 + h / 2 + (j - (m - 1) / 2) * Math.min(52, h / Math.max(m, 1)),
    ]),
  );
}

function drawEdges(g, W, pos, t, ph, col) {
  let wm = 1e-9;
  W.forEach((wl) =>
    wl.forEach((w) =>
      w.forEach((v) => {
        wm = Math.max(wm, Math.abs(v));
      }),
    ),
  );
  W.forEach((wl, l) =>
    wl.forEach((w, j) => {
      if (!pos[l + 1][j]) return;
      w.forEach((v, i) => {
        if (!pos[l][i]) return;
        const a = pos[l][i],
          b = pos[l + 1][j],
          al = Math.min(1, Math.abs(v) / wm),
          cl = v >= 0 ? t.accent : t.ok;
        g.strokeStyle = mix(cl, cl, 0, 0.12 + 0.78 * al);
        g.lineWidth = 0.6 + 3.2 * al;
        g.beginPath();
        g.moveTo(a[0], a[1]);
        g.lineTo(b[0], b[1]);
        g.stroke();
        if (ph != null && al > 0.12) {
          const u = (((ph + i * 0.17 + j * 0.31 + l * 0.4) % 1) + 1) % 1;
          g.fillStyle = col || t.gold;
          g.beginPath();
          g.arc(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, 1.6 + 2 * al, 0, 7);
          g.fill();
        }
      });
    }),
  );
}

function sigFlow(act, depth, base, seed) {
  const n = 40,
    bt = 50,
    r = randnFrom(rng(seed)),
    f = ACT[act][0],
    df = ACT[act][1],
    W = [],
    acts = [],
    Zs = [];
  let A = [];
  for (let b = 0; b < bt; b++) A.push(Float64Array.from({ length: n }, () => r()));
  for (let l = 0; l < depth; l++) {
    const sd = Math.sqrt(base / n),
      w = Array.from({ length: n }, () => Float64Array.from({ length: n }, () => sd * r()));
    W.push(w);
    const Z = [],
      NA = [];
    for (let b = 0; b < bt; b++) {
      const z = new Float64Array(n),
        a = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += w[j][i] * A[b][i];
        z[j] = s;
        a[j] = f(s);
      }
      Z.push(z);
      NA.push(a);
    }
    Zs.push(Z);
    A = NA;
    const fl = new Float32Array(n * bt);
    A.forEach((a, b) => fl.set(a, b * n));
    acts.push(fl);
  }
  const gradientRms = new Array(depth).fill(0);
  let D = [];
  for (let b = 0; b < bt; b++) D.push(Float64Array.from({ length: n }, () => r()));
  for (let l = depth - 1; l >= 0; l--) {
    let s = 0;
    const ND = [];
    for (let b = 0; b < bt; b++) {
      const dz = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        dz[j] = D[b][j] * df(Zs[l][b][j]);
        s += dz[j] * dz[j];
      }
      const da = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let q = 0;
        for (let j = 0; j < n; j++) q += W[l][j][i] * dz[j];
        da[i] = q;
      }
      ND.push(da);
    }
    gradientRms[l] = Math.sqrt(s / (n * bt));
    D = ND;
  }
  const sd = acts.map((a) => {
    let m = 0,
      s = 0;
    for (const v of a) m += v;
    m /= a.length;
    for (const v of a) s += (v - m) ** 2;
    return Math.sqrt(s / a.length);
  });
  return { acts, gradientRms, sd };
}

const sci = (v) =>
  v === 0
    ? '0'
    : Math.abs(v) >= 0.01 && Math.abs(v) < 1000
      ? (+v.toPrecision(3)).toString()
      : v.toExponential(1).replace('e', '×10^');

export { ACT, ACTN, NNet, nnData, netPos, drawEdges, sigFlow, sci };
