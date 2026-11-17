/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { randnFrom, rng } from '../core/shared.js';

const laChol = (A) => {
  const n = A.length,
    L = A.map((r) => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 1e-12) s = 1e-12;
        L[i][i] = Math.sqrt(s);
      } else L[i][j] = s / L[j][j];
    }
  return L;
};

const laFwd = (L, b) => {
  const n = L.length,
    y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  return y;
};

const laBck = (L, y) => {
  const n = L.length,
    x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
};

const laInv = (A) => {
  const n = A.length,
    M = A.map((r, i) => [...r, ...r.map((_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++)
      if (r !== c) {
        const f = M[r][c];
        if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
      }
  }
  return M.map((r) => r.slice(n));
};

const laMul = (A, B) =>
  A.map((r) =>
    B[0].map((_, j) => {
      let s = 0;
      for (let k = 0; k < B.length; k++) s += r[k] * B[k][j];
      return s;
    }),
  );

const laT = (A) => A[0].map((_, j) => A.map((r) => r[j]));

const laTop = (A, k, seed = 3) => {
  const n = A.length,
    r = randnFrom(rng(seed)),
    vals = [],
    vecs = [];
  for (let c = 0; c < k; c++) {
    let v = Array.from({ length: n }, r);
    for (let it = 0; it < 300; it++) {
      const w = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        let s = 0;
        const Ai = A[i];
        for (let j = 0; j < n; j++) s += Ai[j] * v[j];
        w[i] = s;
      }
      for (const u of vecs) {
        let d = 0;
        for (let i = 0; i < n; i++) d += w[i] * u[i];
        for (let i = 0; i < n; i++) w[i] -= d * u[i];
      }
      let nm = Math.hypot(...w) || 1e-12,
        df = 0;
      for (let i = 0; i < n; i++) {
        w[i] /= nm;
        df += Math.abs(w[i] - v[i]);
      }
      v = w;
      if (df < 1e-9 && it > 5) break;
    }
    let lam = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += A[i][j] * v[j];
      lam += v[i] * s;
    }
    vals.push(lam);
    vecs.push(v);
  }
  return { vals, vecs };
};

export { laChol, laFwd, laBck, laInv, laMul, laT, laTop };
