/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { MONO, mix } from '../../js/core/shared.js';

const ACTION_OFFSETS = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
];

const ACTION_NAMES = ['up', 'right', 'down', 'left'];

const ACTION_ARROWS = ['↑', '→', '↓', '←'];

const rlN = (v) =>
  (v < 0 ? '−' : '') +
  (Math.abs(v) >= 100
    ? Math.abs(v).toFixed(0)
    : Math.abs(v) >= 9.995
      ? Math.abs(v).toFixed(1)
      : Math.abs(v).toFixed(2));

function gwMake(rows, o = {}) {
  const R = rows.length,
    C = rows[0].length,
    G = {
      R,
      C,
      rows,
      n: R * C,
      rs: o.rs != null ? o.rs : -1,
      rg: o.rg != null ? o.rg : 10,
      rp: o.rp != null ? o.rp : -10,
      slip: o.slip || 0,
      back: !!o.back,
    };
  G.ch = (s) => rows[Math.floor(s / C)][s % C];
  G.name = (s) => String.fromCharCode(65 + Math.floor(s / C)) + ((s % C) + 1);
  G.wall = (s) => G.ch(s) === '#';
  G.term = (s) => G.ch(s) === 'G' || (G.ch(s) === 'X' && !G.back);
  G.start = rows.join('').indexOf('S');
  G.states = [...Array(R * C).keys()].filter((s) => !G.wall(s));
  G.move = (s, a) => {
    const r = Math.floor(s / C) + ACTION_OFFSETS[a][0],
      c = (s % C) + ACTION_OFFSETS[a][1];
    if (r < 0 || c < 0 || r >= R || c >= C || rows[r][c] === '#') return s;
    return r * C + c;
  };
  G.rew = (s2) => (G.ch(s2) === 'G' ? G.rg : G.ch(s2) === 'X' ? G.rp : G.rs);
  G.land = (s2) => (G.back && G.ch(s2) === 'X' ? G.start : s2);
  G.outcomes = (s, a) => {
    const sl = G.slip;
    return sl
      ? [
          [1 - sl, G.move(s, a)],
          [sl / 2, G.move(s, (a + 1) % 4)],
          [sl / 2, G.move(s, (a + 3) % 4)],
        ]
      : [[1, G.move(s, a)]];
  };
  G.step = (s, a) => {
    let u = Math.random(),
      raw = s,
      slipped = false;
    const O = G.outcomes(s, a);
    for (let i = 0; i < O.length; i++) {
      u -= O[i][0];
      if (u <= 0 || i === O.length - 1) {
        raw = O[i][1];
        slipped = i > 0;
        break;
      }
    }
    return { s2: G.land(raw), raw, r: G.rew(raw), done: G.term(raw), slipped, bump: raw === s };
  };
  return G;
}

function gwQstar(G, gam) {
  const Q = new Float64Array(G.n * 4);
  for (let it = 0; it < 3000; it++) {
    let d = 0;
    for (const s of G.states) {
      if (G.term(s) || G.ch(s) === 'X') continue;
      for (let a = 0; a < 4; a++) {
        let v = 0;
        for (const [p, raw] of G.outcomes(s, a)) {
          const s2 = G.land(raw);
          v +=
            p *
            (G.rew(raw) +
              (G.term(raw)
                ? 0
                : gam * Math.max(Q[s2 * 4], Q[s2 * 4 + 1], Q[s2 * 4 + 2], Q[s2 * 4 + 3])));
        }
        d = Math.max(d, Math.abs(v - Q[s * 4 + a]));
        Q[s * 4 + a] = v;
      }
    }
    if (d < 1e-9) break;
  }
  return Q;
}

const gwBest = (Q, s) => {
  let b = 0;
  for (let a = 1; a < 4; a++) if (Q[s * 4 + a] > Q[s * 4 + b] + 1e-12) b = a;
  return b;
};

const gwMax = (Q, s) => Math.max(Q[s * 4], Q[s * 4 + 1], Q[s * 4 + 2], Q[s * 4 + 3]);

function gwRoute(G, Q, s0, cap = 60) {
  const path = [s0],
    acts = [];
  let s = s0,
    ret = 0;
  const seen = new Set([s0]);
  for (let i = 0; i < cap; i++) {
    if (G.term(s)) return { path, acts, ret, ok: G.ch(s) === 'G' };
    const a = gwBest(Q, s),
      raw = G.move(s, a);
    ret += G.rew(raw);
    acts.push(a);
    if (G.ch(raw) === 'X' && G.back)
      return { path: path.concat([raw]), acts, ret, ok: false, fell: true };
    s = raw;
    path.push(s);
    if (seen.has(s) && !G.term(s)) return { path, acts, ret, ok: false, loop: true };
    seen.add(s);
  }
  return { path, acts, ret, ok: G.ch(s) === 'G' };
}

function gwArrow(g, cx, cy, a, len, col, lw = 2) {
  const dx = ACTION_OFFSETS[a][1],
    dy = ACTION_OFFSETS[a][0],
    x2 = cx + dx * len,
    y2 = cy + dy * len,
    hs = Math.max(3, Math.min(7, len * 0.45));
  g.strokeStyle = col;
  g.fillStyle = col;
  g.lineWidth = lw;
  g.beginPath();
  g.moveTo(cx, cy);
  g.lineTo(x2, y2);
  g.stroke();
  g.beginPath();
  g.moveTo(x2 + dx * hs * 0.6, y2 + dy * hs * 0.6);
  g.lineTo(x2 - dx * hs * 0.5 - dy * hs * 0.6, y2 - dy * hs * 0.5 + dx * hs * 0.6);
  g.lineTo(x2 - dx * hs * 0.5 + dy * hs * 0.6, y2 - dy * hs * 0.5 - dx * hs * 0.6);
  g.closePath();
  g.fill();
}

function gwDraw(g, t, x0, y0, w, h, G, o = {}) {
  const cs = Math.floor(Math.min(w / G.C, h / G.R)),
    ox = Math.round(x0 + (w - cs * G.C) / 2),
    oy = Math.round(y0 + (h - cs * G.R) / 2),
    ctr = (s) => [ox + (s % G.C) * cs + cs / 2, oy + Math.floor(s / G.C) * cs + cs / 2],
    fam = MONO.split('px ')[1],
    fs = Math.max(8, Math.min(13, Math.round(cs * 0.16)));
  let vm = 1e-9;
  if (o.Q)
    for (const s of G.states)
      if (!G.term(s)) for (let a = 0; a < 4; a++) vm = Math.max(vm, Math.abs(o.Q[s * 4 + a]));
  if (o.V) for (const s of G.states) if (!G.term(s)) vm = Math.max(vm, Math.abs(o.V[s]));
  if (o.vm) vm = o.vm;
  for (let s = 0; s < G.n; s++) {
    const x = ox + (s % G.C) * cs,
      y = oy + Math.floor(s / G.C) * cs,
      ch = G.ch(s),
      cx = x + cs / 2,
      cy = y + cs / 2;
    if (ch === '#') {
      g.fillStyle = mix(t.ink, t.ink, 0, 0.78);
      g.fillRect(x, y, cs, cs);
      continue;
    }
    if (ch === 'G' || ch === 'X') {
      g.fillStyle = mix(
        ch === 'G' ? t.gold : t.cut,
        t.bg,
        ch === 'G' ? 0.25 : 0.1,
        ch === 'G' ? 0.75 : 0.55,
      );
      g.fillRect(x, y, cs, cs);
      g.fillStyle = t.ink;
      g.textAlign = 'center';
      g.font = '600 ' + Math.max(11, Math.round(cs * 0.2)) + 'px ' + fam;
      if (ch === 'G' && G.rg <= 0) {
        g.fillText('goal', cx, cy + 4);
      } else {
        g.fillText((ch === 'G' ? '+' : '−') + Math.abs(ch === 'G' ? G.rg : G.rp), cx, cy + 2);
        g.font = Math.max(8, Math.round(cs * 0.12)) + 'px ' + fam;
        g.fillText(ch === 'G' ? 'goal' : o.xname || 'pit', cx, cy + cs * 0.27);
      }
    } else if (o.Q) {
      const b = gwBest(o.Q, s),
        C4 = [
          [x, y],
          [x + cs, y],
          [x + cs, y + cs],
          [x, y + cs],
        ];
      let any = false;
      for (let a = 0; a < 4; a++) {
        const v = o.Q[s * 4 + a];
        if (v !== 0) any = true;
        const col = v >= 0 ? t.accent : t.cut;
        g.fillStyle = mix(col, col, 0, v === 0 ? 0 : 0.07 + 0.6 * Math.min(1, Math.abs(v) / vm));
        g.beginPath();
        g.moveTo(cx, cy);
        g.lineTo(C4[a][0], C4[a][1]);
        g.lineTo(C4[(a + 1) % 4][0], C4[(a + 1) % 4][1]);
        g.closePath();
        g.fill();
      }
      g.strokeStyle = t.line;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x, y);
      g.lineTo(x + cs, y + cs);
      g.moveTo(x + cs, y);
      g.lineTo(x, y + cs);
      g.stroke();
      if (o.hot && o.hot[0] === s) {
        const a = o.hot[1];
        g.strokeStyle = t.gold;
        g.lineWidth = 3;
        g.beginPath();
        g.moveTo(cx, cy);
        g.lineTo(C4[a][0], C4[a][1]);
        g.lineTo(C4[(a + 1) % 4][0], C4[(a + 1) % 4][1]);
        g.closePath();
        g.stroke();
      }
      if (cs >= 54 && !o.nonum) {
        g.textAlign = 'center';
        for (let a = 0; a < 4; a++) {
          g.font = (any && a === b ? '700 ' : '') + fs + 'px ' + fam;
          g.fillStyle = any && a === b ? t.ink : t.muted;
          g.fillText(
            rlN(o.Q[s * 4 + a]),
            cx + ACTION_OFFSETS[a][1] * cs * 0.3,
            cy + ACTION_OFFSETS[a][0] * cs * 0.33 + fs * 0.35,
          );
        }
      } else if (any) gwArrow(g, cx, cy, b, cs * 0.26, t.ink, 1.6);
    } else if (o.V) {
      const v = o.V[s],
        col = v >= 0 ? t.accent : t.cut;
      g.fillStyle = mix(col, col, 0, v === 0 ? 0 : 0.07 + 0.6 * Math.min(1, Math.abs(v) / vm));
      g.fillRect(x, y, cs, cs);
      g.textAlign = 'center';
      g.font = fs + 1 + 'px ' + fam;
      g.fillStyle = t.ink;
      g.fillText(rlN(v), cx, y + cs - 7);
      if (o.pol && o.pol[s] >= 0)
        gwArrow(
          g,
          cx - ACTION_OFFSETS[o.pol[s]][1] * cs * 0.14,
          cy - 4 - ACTION_OFFSETS[o.pol[s]][0] * cs * 0.14,
          o.pol[s],
          cs * 0.28,
          t.ink,
          2,
        );
    } else if (o.probs) {
      for (let a = 0; a < 4; a++) {
        const p = o.probs[s * 4 + a];
        if (p > 0.02)
          gwArrow(
            g,
            cx,
            cy,
            a,
            cs * 0.42 * p,
            mix(t.accent, t.accent, 0, 0.3 + 0.7 * p),
            1 + 3 * p,
          );
      }
      g.fillStyle = t.ink;
      g.beginPath();
      g.arc(cx, cy, 2, 0, 7);
      g.fill();
    }
    if (ch !== 'G' && ch !== 'X' && !o.noname) {
      g.font = Math.max(8, Math.round(cs * 0.11)) + 'px ' + fam;
      g.fillStyle = t.muted;
      g.textAlign = 'left';
      g.fillText(G.name(s) + (ch === 'S' ? ' start' : ''), x + 3, y + Math.max(9, cs * 0.12));
    }
    g.strokeStyle = t.muted;
    g.lineWidth = 1;
    g.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
    if (o.cur === s) {
      g.strokeStyle = t.gold;
      g.lineWidth = 3;
      g.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
    }
  }
  if (o.trail && o.trail.length > 1) {
    g.strokeStyle = mix(t.gold, t.gold, 0, o.thin ? 0.55 : 0.85);
    g.lineWidth = o.thin ? 2 : Math.max(2, cs * 0.05);
    g.lineJoin = 'round';
    g.beginPath();
    o.trail.forEach((s, i) => {
      const p = ctr(s);
      i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]);
    });
    g.stroke();
  }
  if (o.sel != null && o.sel >= 0) {
    const p = ctr(o.sel);
    g.strokeStyle = t.ink;
    g.lineWidth = 2.5;
    g.setLineDash([5, 4]);
    g.strokeRect(p[0] - cs / 2 + 3, p[1] - cs / 2 + 3, cs - 6, cs - 6);
    g.setLineDash([]);
  }
  if (o.agent != null && o.agent >= 0) {
    const p = ctr(o.agent),
      r = Math.max(5, cs * 0.13);
    g.fillStyle = o.acol || t.ink;
    g.strokeStyle = t.bg;
    g.lineWidth = 2.5;
    g.beginPath();
    g.arc(p[0], p[1], r, 0, 7);
    g.fill();
    g.stroke();
  }
  return {
    cs,
    ox,
    oy,
    ctr,
    at: (px, py) => {
      const c = Math.floor((px - ox) / cs),
        r = Math.floor((py - oy) / cs);
      return r < 0 || c < 0 || r >= G.R || c >= G.C ? -1 : r * G.C + c;
    },
  };
}

const gwClick = (c, fn) =>
  c.addEventListener('click', (e) => {
    const r = c.getBoundingClientRect();
    fn(e.clientX - r.left, e.clientY - r.top);
  });

export {
  ACTION_OFFSETS,
  ACTION_NAMES,
  ACTION_ARROWS,
  rlN,
  gwMake,
  gwQstar,
  gwBest,
  gwMax,
  gwRoute,
  gwArrow,
  gwDraw,
  gwClick,
};
