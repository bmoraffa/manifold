/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { Anim, clamp, lgam } from '../../js/core/shared.js';

function betacf(a, b, x) {
  const FP = 1e-30,
    qab = a + b,
    qap = a + 1,
    qam = a - 1;
  let c = 1,
    d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FP) d = FP;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FP) d = FP;
    c = 1 + aa / c;
    if (Math.abs(c) < FP) c = FP;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FP) d = FP;
    c = 1 + aa / c;
    if (Math.abs(c) < FP) c = FP;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 3e-10) break;
  }
  return h;
}

function betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgam(a + b) - lgam(a) - lgam(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? (bt * betacf(a, b, x)) / a
    : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

const twoSidedTPValue = (t, df) => betai(df / 2, 0.5, df / (df + t * t));

function tcrit(df) {
  let lo = 1,
    hi = 60;
  for (let i = 0; i < 50; i++) {
    const m = (lo + hi) / 2;
    twoSidedTPValue(m, df) > 0.05 ? (lo = m) : (hi = m);
  }
  return (lo + hi) / 2;
}

const fTestPValue = (F, d1, d2) => (F <= 0 ? 1 : betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)));

const formatPValue = (p) => (p < 1e-4 ? '< 0.0001' : p.toFixed(4));

function solve(A, b) {
  const n = b.length,
    M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    [M[i], M[p]] = [M[p], M[i]];
    const d = Math.abs(M[i][i]) > 1e-14 ? M[i][i] : 1e-14;
    for (let j = i; j <= n; j++) M[i][j] /= d;
    for (let r = 0; r < n; r++)
      if (r !== i) {
        const f = M[r][i];
        if (f) for (let j = i; j <= n; j++) M[r][j] -= f * M[i][j];
      }
  }
  return M.map((r) => r[n]);
}

function lsq(X, y, ridge = 0) {
  const p = X[0].length,
    A = Array.from({ length: p }, () => new Array(p).fill(0)),
    b = new Array(p).fill(0);
  for (let i = 0; i < X.length; i++) {
    const r = X[i];
    for (let j = 0; j < p; j++) {
      b[j] += r[j] * y[i];
      for (let k = j; k < p; k++) A[j][k] += r[j] * r[k];
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) A[j][k] = A[k][j];
    A[j][j] += Array.isArray(ridge) ? ridge[j] : ridge;
  }
  return solve(A, b);
}

function legendre(x, d) {
  const P = [1, x];
  for (let k = 2; k <= d; k++) P.push(((2 * k - 1) * x * P[k - 1] - (k - 1) * P[k - 2]) / k);
  return P.slice(0, d + 1);
}

function splineBasis(kind, x, K) {
  const kn = [];
  for (let j = 1; j <= K; j++) kn.push(-1 + (2 * j) / (K + 1));
  const p = (v) => (v > 0 ? v : 0);
  if (kind === 'step') {
    const o = new Array(K + 1).fill(0);
    let i = 0;
    while (i < K && x >= kn[i]) i++;
    o[i] = 1;
    return o;
  }
  if (kind === 'linear') return [1, x, ...kn.map((k) => p(x - k))];
  if (kind === 'natural') {
    const all = [-1, ...kn, 1],
      m = all.length,
      d = (j) => (p(x - all[j]) ** 3 - p(x - all[m - 1]) ** 3) / (all[m - 1] - all[j]),
      dl = d(m - 2),
      o = [1, x];
    for (let j = 0; j < m - 2; j++) o.push(d(j) - dl);
    return o;
  }
  return [1, x, x * x, x * x * x, ...kn.map((k) => p(x - k) ** 3)];
}

const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

function parseXY(text, limit = 2000) {
  try {
    const rows = String(text)
        .split(/\r?\n/)
        .map((l) => l.split(/[,;\t]/).map((v) => v.trim())),
      nc = rows.reduce((n, r) => Math.max(n, r.length), 0),
      cols = [];
    for (let j = 0; j < nc && cols.length < 2; j++) {
      let ok = 0,
        tot = 0;
      for (const r of rows) {
        if (r[j] === undefined || r[j] === '') continue;
        tot++;
        if (isFinite(Number(r[j]))) ok++;
      }
      if (tot >= 5 && ok / tot > 0.8) cols.push(j);
    }
    if (cols.length < 2)
      return 'This file needs at least two numeric columns with five or more rows. Check the file and upload again.';
    const p = [];
    for (const r of rows) {
      if (cols.some((j) => r[j] === undefined || r[j] === '')) continue;
      const a = Number(r[cols[0]]),
        b = Number(r[cols[1]]);
      if (isFinite(a) && isFinite(b)) p.push([a, b]);
      if (p.length >= limit) break;
    }
    return p.length < 5
      ? 'Fewer than five complete rows were found. Add more rows and upload again.'
      : p;
  } catch {
    return 'That file could not be parsed. Try a smaller plain CSV file.';
  }
}

function Orbit(c, redraw, ctl, V0 = { az: -0.7, el: 0.42, zoom: 1 }) {
  const V = { ...V0 };
  let d = null,
    spinning = false;
  const spin = Anim(() => {
    V.az += 0.004;
    redraw();
  });
  const setSpin = (on) => {
    spinning = on;
    on ? spin.start() : spin.stop();
    const b = ctl && ctl.querySelector('[data-v="spin"]');
    if (b) b.textContent = on ? 'Stop rotating' : 'Rotate for me';
  };
  c.tabIndex = 0;
  c.addEventListener('pointerdown', (e) => {
    setSpin(false);
    d = { x: e.clientX, y: e.clientY, az: V.az, el: V.el };
    c.setPointerCapture(e.pointerId);
  });
  c.addEventListener('pointermove', (e) => {
    if (!d) return;
    V.az = d.az + (e.clientX - d.x) * 0.01;
    V.el = clamp(d.el + (e.clientY - d.y) * 0.008, 0.04, 1.5);
    redraw();
  });
  ['pointerup', 'pointercancel'].forEach((n) =>
    c.addEventListener(n, () => {
      d = null;
    }),
  );
  c.addEventListener('keydown', (e) => {
    const k = e.key;
    if (k === 'ArrowLeft') V.az -= 0.1;
    else if (k === 'ArrowRight') V.az += 0.1;
    else if (k === 'ArrowUp') V.el = clamp(V.el + 0.08, 0.04, 1.5);
    else if (k === 'ArrowDown') V.el = clamp(V.el - 0.08, 0.04, 1.5);
    else return;
    e.preventDefault();
    setSpin(false);
    redraw();
  });
  if (ctl)
    ctl.addEventListener('click', (e) => {
      const a = e.target.dataset && e.target.dataset.v;
      if (!a) return;
      if (a === 'spin') {
        setSpin(!spinning);
        return;
      }
      if (a === 'in') V.zoom = Math.min(2.2, V.zoom * 1.2);
      if (a === 'out') V.zoom = Math.max(0.6, V.zoom / 1.2);
      if (a === 'reset') {
        setSpin(false);
        Object.assign(V, V0);
      }
      redraw();
    });
  return {
    V,
    spin: setSpin,
    cam(w, h, div) {
      const s = (Math.min(w, h) / div) * V.zoom,
        cx = w / 2,
        cy = h * 0.56,
        ca = Math.cos(V.az),
        sa = Math.sin(V.az),
        ce = Math.cos(V.el),
        se = Math.sin(V.el);
      return (x, y, z) => {
        const xr = x * ca - y * sa,
          yr = x * sa + y * ca;
        return [cx + s * xr, cy - s * (z * ce + yr * se), yr * ce - z * se];
      };
    },
  };
}

export {
  betacf,
  betai,
  twoSidedTPValue,
  tcrit,
  fTestPValue,
  formatPValue,
  solve,
  lsq,
  legendre,
  splineBasis,
  dot,
  parseXY,
  Orbit,
};
