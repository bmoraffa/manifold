/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $, Anim, RM, SITE, T, fit, mix, randnFrom, rng } from '../core/shared.js';

export function createHero() {
  const c = $('#hero-c'),
    GN = 36,
    N = GN * GN;
  let P = null,
    TG = null,
    si = 0,
    fr = 0,
    mouse = null,
    az = 0.6,
    balls = [],
    lineA = 0;
  const land = (x, y) =>
    0.3 * Math.sin(2.6 * x + 0.5) * Math.cos(2.2 * y) +
    0.2 * (x * x + y * y) -
    0.14 * Math.exp(-((x - 0.45) ** 2 + (y + 0.35) ** 2) / 0.08);
  const SC = [
    ['noise', 80],
    ['name', 340],
    ['gauss', 170],
    ['clusters', 170],
    ['moons', 170],
    ['spiral', 170],
    ['name', 150],
    ['land', 620],
  ];
  function build() {
    const r = rng(9),
      rn = randnFrom(r),
      o = document.createElement('canvas');
    o.width = 1100;
    o.height = 280;
    const g = o.getContext('2d');
    g.fillStyle = '#000';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    let fs = 220;
    g.font = '600 ' + fs + 'px "Cormorant Garamond",Georgia,serif';
    const mw = g.measureText(SITE).width;
    if (mw > 1000) {
      fs = Math.floor((fs * 1000) / mw);
      g.font = '600 ' + fs + 'px "Cormorant Garamond",Georgia,serif';
    }
    g.fillText(SITE, 550, 145);
    const d = g.getImageData(0, 0, 1100, 280).data,
      px = [];
    for (let y = 0; y < 280; y += 3)
      for (let x = 0; x < 1100; x += 3)
        if (d[(y * 1100 + x) * 4 + 3] > 128) px.push([(x - 550) / 520, -(y - 140) / 520]);
    const mk = (fn) => {
      const a = [];
      for (let i = 0; i < N; i++) a.push(fn(i));
      return a;
    };
    TG = {
      name: mk(() => {
        const p = px.length ? px[Math.floor(r() * px.length)] : [0, 0];
        return [p[0] + 0.005 * (r() - 0.5), p[1] + 0.005 * (r() - 0.5), 0];
      }),
      noise: mk(() => [r() * 2.4 - 1.2, r() * 1.5 - 0.75, 0]),
      gauss: mk(() => {
        const a = rn(),
          b = rn();
        return [0.42 * a, 0.21 * (0.75 * a + 0.66 * b), 0];
      }),
      clusters: mk((i) => {
        const k = i % 3,
          m = [
            [-0.6, -0.22],
            [0.6, -0.18],
            [0, 0.4],
          ][k];
        return [m[0] + 0.13 * rn(), m[1] + 0.13 * rn(), k];
      }),
      moons: mk((i) => {
        const k = i % 2,
          a = r() * Math.PI;
        return k
          ? [0.24 - 0.55 * Math.cos(a) + 0.04 * rn(), 0.14 - 0.45 * Math.sin(a) + 0.04 * rn(), 1]
          : [-0.24 + 0.55 * Math.cos(a) + 0.04 * rn(), -0.14 + 0.45 * Math.sin(a) + 0.04 * rn(), 0];
      }),
      spiral: mk((i) => {
        const k = i % 2,
          u = r(),
          a = u * 4.3 + k * Math.PI,
          rad = 0.08 + 0.66 * u;
        return [rad * Math.cos(a) + 0.018 * rn(), rad * Math.sin(a) + 0.018 * rn(), k];
      }),
    };
    if (!P) P = TG.noise.map((p) => ({ x: p[0], y: p[1], c: 0 }));
  }
  const proj = (x, y, z) => {
    const ca = Math.cos(az),
      sa = Math.sin(az),
      xr = x * ca - y * sa,
      yr = x * sa + y * ca;
    return [xr * 0.8, z * 0.55 + yr * 0.37 - 0.03];
  };
  const newBall = () => ({
    x: Math.random() * 1.7 - 0.85,
    y: Math.random() * 1.7 - 0.85,
    vx: 0,
    vy: 0,
    tr: [],
    age: 0,
  });
  function draw() {
    if (!TG) build();
    const t = T(),
      { g, w, h } = fit(c),
      s = Math.min(w / 2.5, h / 1.6),
      X = (x) => w / 2 + x * s,
      Y = (y) => h / 2 - y * s,
      kind = SC[si][0],
      isL = kind === 'land';
    if (isL) {
      az += 0.0028;
      lineA = Math.min(1, lineA + 0.02);
    } else lineA = Math.max(0, lineA - 0.05);
    const ease = isL ? 0.16 : kind === 'noise' ? 0.03 : 0.055,
      cols = [t.ink, t.gold, t.accent];
    for (let i = 0; i < N; i++) {
      const p = P[i];
      let tx,
        ty,
        cl = 0;
      if (isL) {
        const a = i % GN,
          b = (i - a) / GN,
          x = -1 + (2 * a) / (GN - 1),
          y = -1 + (2 * b) / (GN - 1),
          q = proj(x, y, land(x, y));
        tx = q[0];
        ty = q[1];
      } else {
        const q = TG[kind][i];
        tx = q[0];
        ty = q[1];
        cl = q[2];
      }
      if (RM) {
        p.x = tx;
        p.y = ty;
      } else {
        p.x += (tx - p.x) * ease;
        p.y += (ty - p.y) * ease;
        if (mouse && !isL) {
          const dx = p.x - mouse[0],
            dy = p.y - mouse[1],
            d = Math.hypot(dx, dy);
          if (d < 0.22 && d > 1e-4) {
            const f = ((0.22 - d) * 0.35) / d;
            p.x += dx * f;
            p.y += dy * f;
          }
        }
      }
      p.c = cl;
    }
    if (lineA > 0) {
      g.strokeStyle = mix(t.ink, t.ink, 0, 0.2 * lineA);
      g.lineWidth = 1;
      g.beginPath();
      for (let b = 0; b < GN; b++)
        for (let a = 0; a < GN; a++) {
          const i = b * GN + a,
            p = P[i];
          if (a < GN - 1) {
            const q = P[i + 1];
            g.moveTo(X(p.x), Y(p.y));
            g.lineTo(X(q.x), Y(q.y));
          }
          if (b < GN - 1) {
            const q = P[i + GN];
            g.moveTo(X(p.x), Y(p.y));
            g.lineTo(X(q.x), Y(q.y));
          }
        }
      g.stroke();
    }
    const rad = isL ? 1.3 : 2.1;
    for (let k = 0; k < 3; k++) {
      g.fillStyle = cols[k];
      g.beginPath();
      for (let i = 0; i < N; i++) {
        const p = P[i];
        if (p.c !== k) continue;
        const x = X(p.x),
          y = Y(p.y);
        g.moveTo(x + rad, y);
        g.arc(x, y, rad, 0, 7);
      }
      g.fill();
    }
    if (isL && !RM) {
      while (balls.length < 7) balls.push(newBall());
      const e = 1e-3;
      for (const b of balls) {
        const gx = (land(b.x + e, b.y) - land(b.x - e, b.y)) / (2 * e),
          gy = (land(b.x, b.y + e) - land(b.x, b.y - e)) / (2 * e);
        b.vx = 0.9 * b.vx - 0.004 * gx;
        b.vy = 0.9 * b.vy - 0.004 * gy;
        b.x += b.vx;
        b.y += b.vy;
        b.age++;
        b.tr.push([b.x, b.y]);
        if (b.tr.length > 70) b.tr.shift();
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.75 * lineA);
        g.lineWidth = 1.6;
        g.beginPath();
        b.tr.forEach((q, j) => {
          const p = proj(q[0], q[1], land(q[0], q[1]));
          j ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]));
        });
        g.stroke();
        const p = proj(b.x, b.y, land(b.x, b.y));
        g.fillStyle = mix(t.gold, t.gold, 0, lineA);
        g.beginPath();
        g.arc(X(p[0]), Y(p[1]), 5.5, 0, 7);
        g.fill();
        if (b.age > 230 || Math.abs(b.x) > 1 || Math.abs(b.y) > 1) Object.assign(b, newBall());
      }
    } else if (!isL) balls = [];
  }
  const anim = Anim(() => {
    if (++fr >= SC[si][1]) {
      fr = 0;
      si = (si + 1) % SC.length;
    }
    draw();
  });
  const toU = (e) => {
    const r = c.getBoundingClientRect(),
      s = Math.min(r.width / 2.5, r.height / 1.6);
    return [(e.clientX - r.left - r.width / 2) / s, -(e.clientY - r.top - r.height / 2) / s];
  };
  c.addEventListener('pointermove', (e) => {
    mouse = toU(e);
  });
  c.addEventListener('pointerleave', () => {
    mouse = null;
  });
  c.addEventListener('click', () => {
    if (SC[si][0] === 'land') {
      balls.push(newBall(), newBall());
    } else {
      fr = 0;
      si = (si + 1) % SC.length;
    }
  });
  return {
    draw,
    rebuild() {
      TG = null;
      if (RM) {
        si = 1;
        P = null;
        build();
        P = TG.name.map((p) => ({ x: p[0], y: p[1], c: 0 }));
      }
    },
    start() {
      if (RM) {
        si = 1;
        if (!TG) build();
        P = TG.name.map((p) => ({ x: p[0], y: p[1], c: 0 }));
        draw();
      } else anim.start();
    },
    stop() {
      anim.stop();
    },
  };
}
