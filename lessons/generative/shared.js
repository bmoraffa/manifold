/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { AD } from '../../js/math/autodiff.js';
import { MONO, clamp, mix, randnFrom, rng } from '../../js/core/shared.js';
import { drawSeriesChart } from '../unsupervised/shared.js';

const G2S = 1.75;

function g2Samp(ds, RN) {
  const R = Math.random;
  if (ds === 'moons') {
    const a = R() * Math.PI;
    return R() < 0.5
      ? [Math.cos(a) - 0.5 + 0.06 * RN(), -0.25 + Math.sin(a) * 0.9 + 0.06 * RN()]
      : [0.5 - Math.cos(a) + 0.06 * RN(), 0.25 - Math.sin(a) * 0.9 + 0.06 * RN()];
  }
  if (ds === 'ring') {
    const a = R() * 2 * Math.PI;
    return [1.1 * Math.cos(a) + 0.06 * RN(), 1.1 * Math.sin(a) + 0.06 * RN()];
  }
  if (ds === 'spiral') {
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
}

function fractionNearData(xs, M, data, r = 0.15) {
  let ok = 0;
  for (let i = 0; i < M; i++) {
    let b = 1e9;
    for (const q of data) {
      const d = (q[0] - xs[2 * i]) ** 2 + (q[1] - xs[2 * i + 1]) ** 2;
      if (d < b) b = d;
    }
    if (b < r * r) ok++;
  }
  return ok / M;
}

const g2Emb = (u) => {
  const o = [];
  for (let q = 0; q < 4; q++) {
    const f = u * Math.PI * 2 ** q * 0.5;
    o.push(Math.sin(f), Math.cos(f));
  }
  return o;
};

function g2Pred(net, P, e) {
  const n = P.length / 2,
    X = new Float64Array(n * 10);
  for (let i = 0; i < n; i++) {
    X[i * 10] = P[2 * i];
    X[i * 10 + 1] = P[2 * i + 1];
    for (let q = 0; q < 8; q++) X[i * 10 + 2 + q] = e[q];
  }
  const o = net.fwd(AD.input(X, [n, 10]));
  AD.clear();
  return o.v;
}

const vectorFieldGrid = (() => {
  const G = [];
  for (let a = 0; a < 10; a++)
    for (let b = 0; b < 14; b++)
      G.push(-2.1 + (4.2 * (b + 0.5)) / 14, 1.55 - (3.1 * (a + 0.5)) / 10);
  return Float64Array.from(G);
})();

function g2Data(g, V, t, data, al = 0.28, r = 2) {
  g.fillStyle = mix(t.ink, t.ink, 0, al);
  for (const p of data) {
    g.beginPath();
    g.arc(V.X(p[0] * G2S), V.Y(p[1] * G2S), r, 0, 7);
    g.fill();
  }
}

function g2Arrows(g, V, e, col, sgn = 1, al = 0.6, k = 7, mx = 15) {
  g.strokeStyle = mix(col, col, 0, al);
  g.fillStyle = col;
  g.lineWidth = 1.2;
  for (let i = 0; i < vectorFieldGrid.length / 2; i++) {
    const x = V.X(vectorFieldGrid[2 * i] * G2S),
      y = V.Y(vectorFieldGrid[2 * i + 1] * G2S),
      dx = sgn * e[2 * i],
      dy = sgn * e[2 * i + 1],
      m = Math.hypot(dx, dy) || 1,
      L = Math.min(mx, k * m);
    g.beginPath();
    g.moveTo(x, y);
    g.lineTo(x + (dx / m) * L, y - (dy / m) * L);
    g.stroke();
    g.beginPath();
    g.arc(x + (dx / m) * L, y - (dy / m) * L, 1.6, 0, 7);
    g.fill();
  }
}

function g2Pts(g, V, t, xs, M, col, r = 3) {
  g.fillStyle = col || t.gold;
  g.strokeStyle = t.bg;
  g.lineWidth = 0.7;
  for (let i = 0; i < M; i++) {
    const x = V.X(xs[2 * i] * G2S),
      y = V.Y(xs[2 * i + 1] * G2S);
    if (x < -5 || y < -5 || x > V.w + 5 || y > V.h + 5) continue;
    g.beginPath();
    g.arc(x, y, r, 0, 7);
    g.fill();
    g.stroke();
  }
}

function g2Trails(g, V, t, tr) {
  g.lineWidth = 1.6;
  g.lineJoin = 'round';
  for (const p of tr) {
    if (p.length < 2) continue;
    g.strokeStyle = mix(t.accent, t.accent, 0, 0.85);
    g.beginPath();
    p.forEach((q, i) =>
      i ? g.lineTo(V.X(q[0] * G2S), V.Y(q[1] * G2S)) : g.moveTo(V.X(q[0] * G2S), V.Y(q[1] * G2S)),
    );
    g.stroke();
    g.fillStyle = t.bg;
    g.beginPath();
    g.arc(V.X(p[0][0] * G2S), V.Y(p[0][1] * G2S), 3.5, 0, 7);
    g.fill();
    g.stroke();
  }
}

const g2Note = (g, t, h, s) => {
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'left';
  g.fillText(s, 8, h - 8);
};

function drawSamplerBenchmark(g, t, x0, y0, w, h, NS, series, cur, title) {
  let L = null;
  series.forEach((s, i) => {
    L = drawSeriesChart(g, t, x0, y0, w, h - 12, s.ys, {
      title: i ? '' : title,
      n: NS.length,
      ym: 1,
      col: s.col,
      dots: true,
    });
  });
  g.font = MONO;
  g.textAlign = 'center';
  NS.forEach((n, i) => {
    g.fillStyle = i === cur ? t.cut : t.muted;
    g.fillText(String(n), L.X(i), y0 + h - 6);
  });
  if (cur >= 0) {
    g.strokeStyle = mix(t.gold, t.gold, 0, 0.7);
    g.lineWidth = 1.2;
    g.setLineDash([4, 4]);
    g.beginPath();
    g.moveTo(L.X(cur), y0 + 26);
    g.lineTo(L.X(cur), y0 + h - 32);
    g.stroke();
    g.setLineDash([]);
  }
  g.textAlign = 'right';
  series.forEach((s, i) => {
    g.fillStyle = s.col;
    g.fillText(s.lab, x0 + w - 14, y0 + h - 40 - 15 * (series.length - 1 - i));
  });
  return L;
}

function g2Levels(g, t, w, h1, data, specs, title) {
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'left';
  g.fillText(title, 8, 13);
  const pw = (w - 20) / specs.length;
  specs.forEach((sp, q) => {
    const x0 = 10 + q * pw,
      cx = x0 + pw / 2,
      cy = 20 + (h1 - 40) / 2,
      s = Math.min(pw, h1 - 40) / 5.6,
      rn = randnFrom(rng(q * 7 + 3));
    g.strokeStyle = t.line;
    g.lineWidth = 1;
    g.strokeRect(x0 + 2, 20, pw - 4, h1 - 40);
    g.fillStyle = mix(t.ink, t.ink, 0, 0.55);
    for (let i = 0; i < 160 && i < data.length; i++) {
      const p = data[i],
        x = sp.a * p[0] + sp.s * rn(),
        y = sp.a * p[1] + sp.s * rn();
      if (Math.abs(x) > 2.7 || Math.abs(y) > 2.6) continue;
      g.beginPath();
      g.arc(cx + x * s, cy - y * s, 1.5, 0, 7);
      g.fill();
    }
    g.fillStyle = sp.hi ? t.cut : t.muted;
    g.textAlign = 'center';
    g.fillText(sp.lab, cx, h1 - 6);
  });
}

function g2Panel(g, t, x0, y0, w, h, title, zoom = 1) {
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'left';
  g.fillText(title, x0 + 8, y0 + 13);
  const s = Math.min((w - 16) / 4.6, (h - 26) / 3.5) * zoom,
    cx = x0 + w / 2,
    cy = y0 + 20 + (h - 24) / 2;
  g.save();
  g.beginPath();
  g.rect(x0 + 2, y0 + 18, w - 4, h - 20);
  g.clip();
  return { X: (v) => cx + (v * s) / G2S, Y: (v) => cy - (v * s) / G2S, w: x0 + w, h: y0 + h, s, g };
}

const g2Uni = (n) =>
  Float64Array.from({ length: 2 * n }, (_, i) => (Math.random() * 2 - 1) * (i % 2 ? 1.65 : 2.2));

function nfMake(seed, K = 8, H = 32) {
  const rn = randnFrom(rng(seed)),
    L = [];
  for (let k = 0; k < K; k++) {
    const a = Math.PI / 3,
      net = AD.mlp([1, H, H, 2], rn, 'relu'),
      w = net.ps[2];
    for (let i = 0; i < w.v.length; i++) w.v[i] *= 0.1;
    L.push({ net, a, R: AD.input([Math.cos(a), Math.sin(a), -Math.sin(a), Math.cos(a)], [2, 2]) });
  }
  return { L, ps: L.flatMap((l) => l.net.ps) };
}

function nfFwd(F, X, m) {
  let h = AD.input(X, [m, 2]);
  const As = [];
  for (const l of F.L) {
    h = AD.mm(h, l.R);
    const a = AD.cols(h, 0, 1),
      b = AD.cols(h, 1, 2),
      st = l.net.fwd(a),
      y = AD.aff(b, AD.cols(st, 0, 1), AD.cols(st, 1, 2), 1.2);
    As.push(y);
    h = AD.hcat(a, y);
  }
  return { z: h, As };
}

function nfLogp(F, X, m) {
  const { z, As } = nfFwd(F, X, m);
  AD.clear();
  const o = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    o[i] = -0.5 * (z.v[2 * i] ** 2 + z.v[2 * i + 1] ** 2) - Math.log(2 * Math.PI);
    for (const A of As) o[i] += A.s[i];
  }
  return o;
}

function nfInv(F, Z, m) {
  const h = Float64Array.from(Z),
    out = [Float64Array.from(h)];
  /* out[0] is z; then, for each layer undone, the cloud after the stretch is undone and after the turn is undone */ for (
    let k = F.L.length - 1;
    k >= 0;
    k--
  ) {
    const l = F.L[k],
      a = new Float64Array(m);
    for (let i = 0; i < m; i++) a[i] = h[2 * i];
    const st = l.net.fwd(AD.input(a, [m, 1]));
    AD.clear();
    const c = Math.cos(l.a),
      s = Math.sin(l.a);
    for (let i = 0; i < m; i++)
      h[2 * i + 1] = (h[2 * i + 1] - st.v[2 * i + 1]) * Math.exp(-1.2 * Math.tanh(st.v[2 * i]));
    out.push(Float64Array.from(h));
    for (let i = 0; i < m; i++) {
      const a0 = h[2 * i],
        b = h[2 * i + 1];
      h[2 * i] = a0 * c + b * s;
      h[2 * i + 1] = -a0 * s + b * c;
    }
    out.push(Float64Array.from(h));
  }
  return out;
}

function ebE(net, P, n) {
  const o = net.fwd(AD.input(P, [n, 2]));
  AD.clear();
  return o.v;
}

function ebGrad(net, P, n) {
  const X = AD.input(P, [n, 2]),
    l = AD.mean(net.fwd(X));
  AD.backward(l);
  for (const p of net.ps) p.g.fill(0);
  const G = X.g;
  for (let i = 0; i < G.length; i++) G[i] *= n;
  return G;
}

function ebLang(net, P, n, k, a, sg, RN) {
  for (let q = 0; q < k; q++) {
    const G = ebGrad(net, P, n);
    for (let i = 0; i < n; i++) {
      P[2 * i] = clamp(P[2 * i] - a * clamp(G[2 * i], -6, 6) + sg * RN(), -2.25, 2.25);
      P[2 * i + 1] = clamp(P[2 * i + 1] - a * clamp(G[2 * i + 1], -6, 6) + sg * RN(), -1.7, 1.7);
    }
  }
}

export {
  G2S,
  g2Samp,
  fractionNearData,
  g2Emb,
  g2Pred,
  vectorFieldGrid,
  g2Data,
  g2Arrows,
  g2Pts,
  g2Trails,
  g2Note,
  drawSamplerBenchmark,
  g2Levels,
  g2Panel,
  g2Uni,
  nfMake,
  nfFwd,
  nfLogp,
  nfInv,
  ebE,
  ebGrad,
  ebLang,
};
