/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $, MONO, bind, clamp, fit, randnFrom, rng } from '../../js/core/shared.js';
import { parseXY } from '../regression/shared.js';

function xyUp(root, id, fn, fitBox = true) {
  const f = $('#' + id + '-file', root),
    st = $('#' + id + '-st', root);
  if (!f) return;
  f.onchange = () => {
    const file = f.files[0];
    if (!file) return;
    const fr = new FileReader();
    fr.onload = () => {
      const p = parseXY(fr.result);
      if (typeof p === 'string') {
        st.textContent = p;
        return;
      }
      let q = p.slice(0, 400);
      if (fitBox) {
        const mn = [0, 1].map((j) => Math.min(...q.map((v) => v[j]))),
          mx = [0, 1].map((j) => Math.max(...q.map((v) => v[j])));
        q = q.map((v) => [
          ((v[0] - mn[0]) / (mx[0] - mn[0] || 1) - 0.5) * 6.4,
          ((v[1] - mn[1]) / (mx[1] - mn[1] || 1) - 0.5) * 4.4,
        ]);
      }
      st.textContent =
        'Loaded ' +
        q.length +
        ' rows from ' +
        file.name +
        (fitBox ? ', rescaled to fit the picture.' : '.');
      fn(q);
    };
    fr.readAsText(file);
    f.value = '';
  };
}

const dotAt = (g, x, y, r, fill, stroke) => {
  g.beginPath();
  g.arc(x, y, r, 0, 7);
  if (fill) {
    g.fillStyle = fill;
    g.fill();
  }
  if (stroke) {
    g.strokeStyle = stroke;
    g.stroke();
  }
};

const canvasXY = (c, V, e) => {
  const r = c.getBoundingClientRect();
  return [(e.clientX - r.left - V.w / 2) / V.s, -(e.clientY - r.top - V.h / 2) / V.s];
};

const UC = (t) => [t.accent, t.gold, t.ok, '#4a7fb5', '#8a5fb0', '#c9703a', '#3f9c96', '#8c8c8c'];

const UKIND = {
  shop: 'Shop customers (simulated)',
  five: 'Five groups',
  blobs: 'Three round groups',
  uneven: 'Groups of unequal size',
  long: 'Three stretched groups',
  moons: 'Two moons',
  rings: 'Two rings',
  dens: 'Dense and sparse groups',
  none: 'No groups at all',
  h2: 'Heights of children and adults (simulated)',
  h3: 'Three humps',
  hov: 'Two overlapping humps',
  corr: 'Height and arm span (simulated)',
  banana: 'A curved cloud',
  twoc: 'Two separate groups',
  round: 'A round cloud',
};

function uData(kind, n, seed, bg = 0) {
  const r = rng(seed * 13 + 5),
    rn = randnFrom(r),
    P = [];
  const blob = (m, cx, cy, sx, sy, rot = 0) => {
    m = Math.round(m);
    const c = Math.cos(rot),
      s = Math.sin(rot);
    for (let i = 0; i < m; i++) {
      const a = rn() * sx,
        b = rn() * sy;
      P.push({ x: cx + a * c - b * s, y: cy + a * s + b * c });
    }
  };
  if (kind === 'shop') {
    blob(n * 0.32, -2.2, -1.2, 0.42, 0.38);
    blob(n * 0.26, -1.3, 1.3, 0.5, 0.36);
    blob(n * 0.28, 1.1, -0.5, 0.5, 0.45);
    blob(n * 0.14, 2.5, 1.5, 0.3, 0.3);
  } else if (kind === 'five') {
    [
      [-2.6, 1.5],
      [-2.4, -1.4],
      [0, 0.1],
      [2.4, 1.6],
      [2.6, -1.3],
    ].forEach((m) => blob(n / 5, m[0], m[1], 0.36, 0.36));
  } else if (kind === 'blobs') {
    blob(n / 3, -2, -1, 0.5, 0.5);
    blob(n / 3, 2, -1, 0.5, 0.5);
    blob(n / 3, 0, 1.4, 0.5, 0.5);
  } else if (kind === 'uneven') {
    blob(n * 0.7, -1.2, 0, 1.05, 0.95);
    blob(n * 0.15, 2.3, 1.3, 0.22, 0.22);
    blob(n * 0.15, 2.4, -1.2, 0.22, 0.22);
  } else if (kind === 'long') {
    [-1.5, 0, 1.5].forEach((y) => blob(n / 3, 0, y, 1.5, 0.18, 0.32));
  } else if (kind === 'moons') {
    const hf = Math.ceil(n / 2);
    for (let i = 0; i < n; i++) {
      const a = ((Math.floor(i / 2) + r()) / hf) * Math.PI;
      P.push(
        i % 2
          ? {
              x: 0.9 - 1.9 * Math.cos(a) + 0.085 * rn(),
              y: 0.55 - 1.7 * Math.sin(a) + 0.085 * rn(),
            }
          : {
              x: -0.9 + 1.9 * Math.cos(a) + 0.085 * rn(),
              y: -0.55 + 1.7 * Math.sin(a) + 0.085 * rn(),
            },
      );
    }
  } else if (kind === 'rings') {
    const n0 = Math.ceil(n / 4);
    for (let i = 0; i < n; i++) {
      const k = i < n0 ? 0 : 1,
        a = (((k ? i - n0 : i) + r()) / (k ? n - n0 : n0)) * 2 * Math.PI,
        rad = (k ? 2.2 : 0.7) + 0.085 * rn();
      P.push({ x: rad * 1.15 * Math.cos(a), y: rad * Math.sin(a) });
    }
  } else if (kind === 'dens') {
    blob(n * 0.45, -1.9, 0.3, 0.28, 0.28);
    blob(n * 0.45, 1.6, 0, 1, 0.9);
    blob(n * 0.1, -1.6, -2, 0.2, 0.15);
  } else for (let i = 0; i < n; i++) P.push({ x: r() * 6.8 - 3.4, y: r() * 5 - 2.5 });
  for (let i = 0; i < bg; i++) P.push({ x: r() * 7.2 - 3.6, y: r() * 5.4 - 2.7 });
  for (const p of P) {
    p.x = clamp(p.x, -3.8, 3.8);
    p.y = clamp(p.y, -2.85, 2.85);
  }
  return P;
}

function uView(c) {
  const { g, w, h } = fit(c),
    s = Math.min(w / 8, h / 6),
    X = (x) => w / 2 + x * s,
    Y = (y) => h / 2 - y * s;
  return { g, w, h, s, X, Y };
}

const uInv = (c, e) => {
  const r = c.getBoundingClientRect(),
    s = Math.min(r.width / 8, r.height / 6);
  return [(e.clientX - r.left - r.width / 2) / s, -(e.clientY - r.top - r.height / 2) / s];
};

function uFrame(V, t, xl, yl) {
  const { g, w, h, X, Y } = V;
  g.strokeStyle = t.line;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(X(-4), Y(0));
  g.lineTo(X(4), Y(0));
  g.moveTo(X(0), Y(-3));
  g.lineTo(X(0), Y(3));
  g.stroke();
  if (xl) {
    g.font = MONO;
    g.fillStyle = t.muted;
    g.textAlign = 'right';
    g.fillText(xl, w - 6, h - 6);
    g.textAlign = 'left';
    g.fillText(yl, 6, 12);
  }
}

const uNum = (v) =>
  !isFinite(v)
    ? '–'
    : Math.abs(v) >= 100
      ? v.toFixed(0)
      : Math.abs(v) >= 10
        ? v.toFixed(1)
        : v.toFixed(2);

function uLine(g, t, x0, y0, w, h, ys, o = {}) {
  const m = { l: 44, r: 12, t: 24, b: 20 },
    n = Math.max(2, o.n || ys.length),
    fin = ys.filter(isFinite),
    ym = o.ym != null ? o.ym : Math.max(1e-9, ...fin),
    y1 = o.y1 != null ? o.y1 : 0;
  const X = (i) => x0 + m.l + (i / (n - 1)) * (w - m.l - m.r),
    Y = (v) => y0 + h - m.b - ((v - y1) / (ym - y1 || 1)) * (h - m.t - m.b);
  g.strokeStyle = t.muted;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(x0 + m.l, y0 + h - m.b);
  g.lineTo(x0 + w - m.r, y0 + h - m.b);
  g.stroke();
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'left';
  g.fillText(o.title || '', x0 + 8, y0 + 13);
  g.textAlign = 'right';
  g.fillText(uNum(ym), x0 + m.l - 5, Y(ym) + 4);
  g.fillText(uNum(y1), x0 + m.l - 5, Y(y1) + 4);
  if (ys.length) {
    g.strokeStyle = o.col || t.accent;
    g.lineWidth = 2.2;
    g.beginPath();
    ys.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
    g.stroke();
    if (o.dots) {
      g.fillStyle = o.col || t.accent;
      ys.forEach((v, i) => {
        g.beginPath();
        g.arc(X(i), Y(v), 3, 0, 7);
        g.fill();
      });
    }
  }
  return { X, Y, m };
}

function uDataBind(root, id, S, cb, mk) {
  const st = $('#' + id + '-st', root);
  let custom = null;
  const gen = () =>
    custom ? custom.map((p) => ({ ...p })) : mk ? mk() : uData(S.ds, S.n, S.seed, S.bg || 0);
  $('#' + id + '-ds', root).onchange = (e) => {
    S.ds = e.target.value;
    custom = null;
    st.textContent = '';
    cb();
  };
  bind(
    root,
    id + '-n',
    (v) => {
      S.n = v;
      if (!custom) cb();
    },
    (v) => String(v),
  );
  $('#' + id + '-new', root).onclick = () => {
    S.seed++;
    custom = null;
    st.textContent = '';
    cb();
  };
  $('#' + id + '-file', root).onchange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fr = new FileReader();
    fr.onerror = () => {
      st.textContent =
        'That file could not be read. Save it again as plain CSV and upload it once more.';
    };
    fr.onload = () => {
      const p = parseXY(fr.result);
      if (typeof p === 'string') {
        st.textContent = p;
        return;
      }
      const q = p.slice(0, S.cap || 600),
        m = [0, 1].map((j) => q.reduce((s, v) => s + v[j], 0) / q.length),
        sd = [0, 1].map(
          (j) => Math.sqrt(q.reduce((s, v) => s + (v[j] - m[j]) ** 2, 0) / q.length) || 1,
        );
      custom = q.map((v) => ({
        x: clamp(((v[0] - m[0]) / sd[0]) * 1.25, -3.8, 3.8),
        y: clamp(((v[1] - m[1]) / sd[1]) * 0.95, -2.85, 2.85),
      }));
      st.textContent =
        'Loaded ' +
        f.name +
        ': ' +
        q.length +
        ' rows from the first two numeric columns, each rescaled to mean 0 and the same spread.';
      cb();
    };
    fr.readAsText(f);
    e.target.value = '';
  };
  return gen;
}

const segBind = (root, id, val, fn) => {
  const bs = [...root.querySelectorAll('#' + id + ' button')],
    set = (v) =>
      bs.forEach((b) => b.setAttribute('aria-pressed', b.dataset.v === v ? 'true' : 'false'));
  bs.forEach(
    (b) =>
      (b.onclick = () => {
        set(b.dataset.v);
        fn(b.dataset.v);
      }),
  );
  set(val);
};

const uCur = (root, id, i) =>
  root
    .querySelectorAll('#' + id + '-steps li')
    .forEach((e, k) => e.setAttribute('aria-current', k === i ? 'true' : 'false'));

const d2 = (p, c) => (p.x - c.x) ** 2 + (p.y - c.y) ** 2;

function kmSeed(P, K, pp, r) {
  const C = [];
  if (!P.length) return C;
  if (!pp) {
    const idx = P.map((_, i) => i);
    for (let k = 0; k < K && idx.length; k++) {
      const j = Math.floor(r() * idx.length);
      C.push({ x: P[idx[j]].x, y: P[idx[j]].y });
      idx.splice(j, 1);
    }
    return C;
  }
  const f = P[Math.floor(r() * P.length)];
  C.push({ x: f.x, y: f.y });
  const D = P.map((p) => d2(p, f));
  while (C.length < K) {
    let tot = 0;
    for (const v of D) tot += v;
    if (tot <= 0) break;
    let u = r() * tot,
      i = 0;
    while (i < P.length - 1 && (u -= D[i]) > 0) i++;
    const c = { x: P[i].x, y: P[i].y };
    C.push(c);
    for (let j = 0; j < P.length; j++) {
      const d = d2(P[j], c);
      if (d < D[j]) D[j] = d;
    }
  }
  return C;
}

function kmRun(P, C, maxIt = 60) {
  C = C.map((c) => ({ ...c }));
  const n = P.length,
    K = C.length,
    A = new Int16Array(n).fill(-1);
  let it = 0;
  for (; it < maxIt; it++) {
    let ch = 0;
    for (let i = 0; i < n; i++) {
      let b = 0,
        bd = 1e18;
      for (let k = 0; k < K; k++) {
        const d = d2(P[i], C[k]);
        if (d < bd) {
          bd = d;
          b = k;
        }
      }
      if (A[i] !== b) {
        A[i] = b;
        ch++;
      }
    }
    if (!ch) break;
    const S = C.map(() => [0, 0, 0]);
    for (let i = 0; i < n; i++) {
      const s = S[A[i]];
      s[0] += P[i].x;
      s[1] += P[i].y;
      s[2]++;
    }
    C.forEach((c, k) => {
      if (S[k][2]) {
        c.x = S[k][0] / S[k][2];
        c.y = S[k][1] / S[k][2];
      }
    });
  }
  let J = 0;
  for (let i = 0; i < n; i++) J += d2(P[i], C[A[i]]);
  return { C, A, J, it };
}

function jacobi(A) {
  const n = A.length,
    a = A.map((r) => r.slice()),
    V = a.map((_, i) => a.map((_, j) => (i === j ? 1 : 0)));
  for (let sw = 0; sw < 80; sw++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]),
          tt = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1)),
          c = 1 / Math.sqrt(tt * tt + 1),
          s = tt * c;
        for (let k = 0; k < n; k++) {
          const x = a[k][p],
            y = a[k][q];
          a[k][p] = c * x - s * y;
          a[k][q] = s * x + c * y;
        }
        for (let k = 0; k < n; k++) {
          const x = a[p][k],
            y = a[q][k];
          a[p][k] = c * x - s * y;
          a[q][k] = s * x + c * y;
        }
        for (let k = 0; k < n; k++) {
          const x = V[k][p],
            y = V[k][q];
          V[k][p] = c * x - s * y;
          V[k][q] = s * x + c * y;
        }
      }
  }
  const idx = a.map((_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return {
    val: idx.map((i) => a[i][i]),
    vec: idx.map((i) => {
      const v = V.map((r) => r[i]);
      let s = 0;
      for (const x of v) s += x;
      return s < 0 ? v.map((x) => -x) : v;
    }),
  };
}

export {
  xyUp,
  dotAt,
  canvasXY,
  UC,
  UKIND,
  uData,
  uView,
  uInv,
  uFrame,
  uNum,
  uLine,
  uDataBind,
  segBind,
  uCur,
  d2,
  kmSeed,
  kmRun,
  jacobi,
};
