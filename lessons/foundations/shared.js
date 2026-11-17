/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  area,
  bind,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  plot,
  ramp,
  randn,
  runCtl,
  setR,
} from '../../js/core/shared.js';
import { Orbit } from '../regression/shared.js';
import { renderFamily } from '../../js/app/lesson-view.js';

const SURF = {
  bowl: { t: 'Round bowl', f: (x, y) => 0.5 * (x * x + y * y), one: (x) => 0.5 * x * x },
  ravine: {
    t: 'Narrow valley',
    f: (x, y) => 0.5 * (0.12 * x * x + 2.4 * y * y),
    one: (x) => 1.2 * x * x,
  },
  banana: {
    t: 'Curved valley',
    f: (x, y) => 0.08 * (1.5 - x) ** 2 + 0.6 * (y - 0.35 * x * x) ** 2,
    one: (x) => 0.08 * (1.5 - x) ** 2 + 0.6 * (0.6 - 0.35 * x * x) ** 2,
  },
  two: {
    t: 'Two valleys',
    f: (x, y) => 0.12 * (x * x - 2.2) ** 2 + 0.5 * y * y + 0.15 * x,
    one: (x) => 0.12 * (x * x - 2.2) ** 2 + 0.15 * x,
  },
  saddle: {
    t: 'Saddle between two valleys',
    f: (x, y) => 0.25 * x * x - 0.3 * y * y + 0.035 * y ** 4 + 0.7,
    one: (x) => -0.3 * x * x + 0.035 * x ** 4 + 0.7,
  },
  bumpy: {
    t: 'Bumpy bowl',
    f: (x, y) => 0.25 * (x * x + y * y) + 0.9 * Math.sin(1.9 * x) * Math.sin(1.7 * y) + 1,
    one: (x) => 0.25 * x * x + 0.8 * Math.sin(2.6 * x) + 1,
  },
  plateau: {
    t: 'Flat plateau with one pit',
    f: (x, y) => 1.2 - 1.2 * Math.exp(-(x * x + y * y) / 3.5),
    one: (x) => 1.2 - 1.2 * Math.exp((-x * x) / 1.2),
  },
};

const HEAT = {};

function heatmap(key, w, h, t, fn, R, nc = 80) {
  const k = key + '|' + Math.round(w) + '|' + Math.round(h) + '|' + t.bg;
  if (HEAT[k]) return HEAT[k];
  if (Object.keys(HEAT).length > 14) for (const q in HEAT) delete HEAT[q];
  const o = document.createElement('canvas');
  o.width = Math.round(w);
  o.height = Math.round(h);
  const g = o.getContext('2d'),
    V = [];
  let mn = 1 / 0,
    mx = -1 / 0;
  for (let i = 0; i < nc; i++)
    for (let j = 0; j < nc; j++) {
      const v = fn(-R + (2 * R * (i + 0.5)) / nc, -R + (2 * R * (j + 0.5)) / nc);
      V.push(v);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  const cw = w / nc,
    ch = h / nc;
  for (let i = 0; i < nc; i++)
    for (let j = 0; j < nc; j++) {
      const v = (V[i * nc + j] - mn) / (mx - mn || 1);
      g.fillStyle = ramp(t, Math.pow(1 - Math.sqrt(v), 1.5));
      g.fillRect(i * cw - 0.4, h - (j + 1) * ch - 0.4, cw + 0.8, ch + 0.8);
    }
  g.strokeStyle = mix(t.ink, t.ink, 0, 0.3);
  g.lineWidth = 1;
  g.beginPath();
  const px = (i) => (i + 0.5) * cw,
    py = (j) => h - (j + 0.5) * ch;
  for (let q = 1; q <= 11; q++) {
    const L = mn + (mx - mn) * Math.pow(q / 12, 2);
    for (let i = 0; i < nc - 1; i++)
      for (let j = 0; j < nc - 1; j++) {
        const a = V[i * nc + j],
          b = V[(i + 1) * nc + j],
          c2 = V[(i + 1) * nc + j + 1],
          d = V[i * nc + j + 1];
        if (a < L === b < L && b < L === c2 < L && c2 < L === d < L) continue;
        const ps = [];
        const E = (v1, v2, x1, y1, x2, y2) => {
          if (v1 < L !== v2 < L) {
            const u = (L - v1) / (v2 - v1);
            ps.push(x1 + (x2 - x1) * u, y1 + (y2 - y1) * u);
          }
        };
        E(a, b, px(i), py(j), px(i + 1), py(j));
        E(b, c2, px(i + 1), py(j), px(i + 1), py(j + 1));
        E(c2, d, px(i + 1), py(j + 1), px(i), py(j + 1));
        E(d, a, px(i), py(j + 1), px(i), py(j));
        if (ps.length >= 4) {
          g.moveTo(ps[0], ps[1]);
          g.lineTo(ps[2], ps[3]);
          if (ps.length === 8) {
            g.moveTo(ps[4], ps[5]);
            g.lineTo(ps[6], ps[7]);
          }
        }
      }
  }
  g.stroke();
  o.mn = mn;
  o.mx = mx;
  return (HEAT[k] = o);
}

const OPT = {
  gd: { n: 'Gradient descent', c: 'gold' },
  newton: { n: 'Newton', c: 'ok' },
  sgd: { n: 'SGD', c: 'ink' },
  mom: { n: 'Momentum', c: 'accent' },
  rms: { n: 'RMSProp', c: 'ok' },
  adam: { n: 'Adam', c: 'gold' },
};

const CASES = {
  big: { t: 'Step Too Large', surf: 'ravine', p: [-2.4, 1.2], p1: 1.6, lr: 0.9, no: 0 },
  saddle: { t: 'Balanced on a Saddle', surf: 'saddle', p: [-2.6, 0], p1: 0, lr: 0.15, no: 0 },
  plateau: { t: 'Lost on a Plateau', surf: 'plateau', p: [2.8, 2.7], p1: 2.9, lr: 0.03, no: 0 },
  valley: { t: 'The Wrong Valley', surf: 'two', p: [2.6, 1.5], p1: 2.7, lr: 0.1, no: 0 },
  noise: {
    t: 'Noise That Never Settles',
    surf: 'bowl',
    p: [-2.4, 2.2],
    p1: -2.4,
    lr: 0.3,
    no: 1.5,
  },
};

function gdLesson(o) {
  const id = o.id;
  const M = o.mode;
  const K0 =
    M === 'newton' ? ['gd', 'newton'] : M === 'opt' ? ['sgd', 'mom', 'rms', 'adam'] : ['gd'];
  return {
    ...o,
    renderStage(parts, context) {
      return renderFamily(new URL('./shared.html', import.meta.url), id, parts, context);
    },
    init(root) {
      const c = $('#' + id + '-c', root),
        cl = $('#' + id + '-l', root),
        note = $('#' + id + '-note', root),
        fc = $('#' + id + '-fc', root),
        R = 3,
        TMAX = M === 'sched' ? 300 : 400,
        S = {
          surf: o.surf || 'bowl',
          dim: '3',
          lr: M === 'sched' ? 0.35 : M === 'scale' ? 0.25 : M === 'gd' ? 0.3 : 0.1,
          lr2: 0.08,
          mo: 0,
          no: M === 'sched' ? 0.5 : M === 'opt' ? 0.3 : 0,
          da: 1,
          sc: 'cos',
          s: 4,
          std: false,
          on: { sgd: true, mom: true, rms: true, adam: true },
          hold: false,
        };
      let runs = [],
        k = 0,
        wait = 0,
        fr = 0,
        pb,
        GV = null,
        start = [-2.4, 2.2],
        draw,
        ST = null;
      const orb = Orbit(
          c,
          () => {
            if (S.dim === '3') draw();
          },
          null,
          { az: -0.6, el: 0.62, zoom: 1 },
        ),
        ref = {};
      const kinds = () => (M === 'opt' ? K0.filter((q) => S.on[q]) : K0),
        ratio = () => (S.std ? 1 : S.s);
      const f = (x, y) =>
        M === 'scale'
          ? 0.3 * ((x * x) / ratio() + ratio() * y * y)
          : S.dim === '1'
            ? SURF[S.surf].one(x)
            : SURF[S.surf].f(x, y);
      const stats = () => {
        const key = (M === 'scale' ? 'sc' + ratio().toFixed(1) : S.surf) + S.dim;
        if (ST && ST.key === key) return ST;
        let mn = 1 / 0,
          mx = -1 / 0;
        for (let i = 0; i <= 90; i++)
          for (let j = 0; j <= (S.dim === '1' ? 0 : 90); j++) {
            const v = f(-R + (2 * R * i) / 90, S.dim === '1' ? 0 : -R + (2 * R * j) / 90);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
        return (ST = { key, mn, mx });
      };
      const grad = (x, y) => {
        const e = 1e-4;
        return [
          (f(x + e, y) - f(x - e, y)) / (2 * e),
          S.dim === '1' ? 0 : (f(x, y + e) - f(x, y - e)) / (2 * e),
        ];
      };
      const sch = (i) =>
        S.sc === 'const'
          ? 1
          : S.sc === 'step'
            ? Math.pow(0.5, Math.floor(i / 75))
            : S.sc === 'exp'
              ? Math.exp(-i / 90)
              : S.sc === 'inv'
                ? 1 / (1 + i / 30)
                : S.sc === 'cos'
                  ? 0.5 * (1 + Math.cos((Math.PI * i) / TMAX))
                  : i < 25
                    ? (i + 1) / 25
                    : 0.5 * (1 + Math.cos((Math.PI * (i - 25)) / (TMAX - 25)));
      const lrNow = (i) => (M === 'sched' ? S.lr * sch(i) : S.lr);
      const begin = (p) => {
        if (p) start = p;
        const y0 = S.dim === '1' ? 0 : start[1];
        runs = kinds().map((kd) => ({
          kd,
          x: start[0],
          y: y0,
          vx: 0,
          vy: 0,
          sx: 0,
          sy: 0,
          mx: 0,
          my: 0,
          gn: 1,
          path: [[start[0], y0]],
          loss: [f(start[0], y0)],
          dead: false,
        }));
        k = 0;
        wait = 0;
        note.innerHTML = '';
      };
      const verdict = () => {
        const mn = stats().mn;
        note.innerHTML = runs
          .map((r) => {
            const L = r.loss[r.loss.length - 1],
              nm = `<b style="color:var(--${OPT[r.kd].c === 'gold' ? 'cut' : OPT[r.kd].c})">${OPT[r.kd].n}</b>: `;
            if (r.dead)
              return (
                nm +
                (r.kd === 'newton'
                  ? 'could not take a step here (the curvature is flat).'
                  : 'diverged. Every step overshot further than the last: the step is too large for the steepest direction.')
              );
            if (r.at)
              return (
                nm +
                (S.no
                  ? 'got near the bottom in ' +
                    r.at +
                    ' steps, but the noise keeps knocking it about. It will not settle at this step size.'
                  : 'reached the lowest point in ' + r.at + ' steps.')
              );
            if (r.gn < 3e-3 && !S.no)
              return (
                nm +
                (L - mn < 0.03
                  ? 'reached the lowest point in ' + (r.path.length - 1) + ' steps.'
                  : 'stopped at loss ' +
                    f2(L) +
                    ', where the ground is flat but it is not the lowest point (lowest is ' +
                    f2(mn) +
                    '). With zero slope there is nothing to follow.')
              );
            return (
              nm +
              (L - mn < 0.08
                ? S.no
                  ? 'near the bottom, but the noise keeps knocking it about. It will not settle at this step size.'
                  : 'close to the bottom after ' + TMAX + ' steps.'
                : 'still at loss ' +
                  f2(L) +
                  ' after ' +
                  (r.path.length - 1) +
                  ' steps (lowest is ' +
                  f2(mn) +
                  '). ' +
                  (r.gn < 0.05
                    ? 'The slope here is almost zero, so each step is tiny.'
                    : 'It has not got there in the time allowed.'))
            );
          })
          .join('<br>');
      };
      const step = () => {
        let live = false;
        k++;
        for (const r of runs) {
          if (r.dead) continue;
          let g = grad(r.x, r.y);
          r.gn = Math.hypot(g[0], g[1]);
          if (r.kd === 'newton') {
            const e = 1e-3,
              hxx = (f(r.x + e, r.y) - 2 * f(r.x, r.y) + f(r.x - e, r.y)) / (e * e);
            let dx,
              dy = 0,
              pd;
            if (S.dim === '1') {
              if (Math.abs(hxx) < 1e-7) {
                r.dead = true;
                continue;
              }
              dx = -g[0] / hxx;
              pd = hxx > 0;
            } else {
              const hyy = (f(r.x, r.y + e) - 2 * f(r.x, r.y) + f(r.x, r.y - e)) / (e * e),
                hxy =
                  (f(r.x + e, r.y + e) -
                    f(r.x + e, r.y - e) -
                    f(r.x - e, r.y + e) +
                    f(r.x - e, r.y - e)) /
                  (4 * e * e),
                det = hxx * hyy - hxy * hxy;
              if (Math.abs(det) < 1e-7) {
                r.dead = true;
                continue;
              }
              dx = -(hyy * g[0] - hxy * g[1]) / det;
              dy = -(hxx * g[1] - hxy * g[0]) / det;
              pd = det > 0 && hxx > 0;
            }
            const n = Math.hypot(dx, dy);
            if (n > 2.5) {
              dx *= 2.5 / n;
              dy *= 2.5 / n;
            }
            r.x += S.da * dx;
            r.y += S.da * dy;
            if (!pd) r.bad = true;
          } else {
            if (S.no) g = [g[0] + S.no * randn(), S.dim === '1' ? 0 : g[1] + S.no * randn()];
            const kd = r.kd;
            if (kd === 'gd') {
              const lr = lrNow(k - 1);
              r.vx = S.mo * r.vx - lr * g[0];
              r.vy = S.mo * r.vy - lr * g[1];
              r.x += r.vx;
              r.y += r.vy;
            } else if (kd === 'sgd') {
              r.x -= S.lr * g[0];
              r.y -= S.lr * g[1];
            } else if (kd === 'mom') {
              r.vx = 0.9 * r.vx - S.lr * g[0];
              r.vy = 0.9 * r.vy - S.lr * g[1];
              r.x += r.vx;
              r.y += r.vy;
            } else if (kd === 'rms') {
              r.sx = 0.9 * r.sx + 0.1 * g[0] * g[0];
              r.sy = 0.9 * r.sy + 0.1 * g[1] * g[1];
              r.x -= (S.lr2 * g[0]) / (Math.sqrt(r.sx) + 1e-8);
              r.y -= (S.lr2 * g[1]) / (Math.sqrt(r.sy) + 1e-8);
            } else {
              r.mx = 0.9 * r.mx + 0.1 * g[0];
              r.my = 0.9 * r.my + 0.1 * g[1];
              r.sx = 0.999 * r.sx + 0.001 * g[0] * g[0];
              r.sy = 0.999 * r.sy + 0.001 * g[1] * g[1];
              const b1 = 1 - Math.pow(0.9, k),
                b2 = 1 - Math.pow(0.999, k);
              r.x -= (S.lr2 * (r.mx / b1)) / (Math.sqrt(r.sx / b2) + 1e-8);
              r.y -= (S.lr2 * (r.my / b1)) / (Math.sqrt(r.sy / b2) + 1e-8);
            }
          }
          if (!isFinite(r.x) || !isFinite(r.y) || Math.abs(r.x) > 30 || Math.abs(r.y) > 30) {
            r.dead = true;
            r.x = clamp(r.x || 0, -30, 30);
            r.y = clamp(r.y || 0, -30, 30);
          }
          r.path.push([r.x, r.y]);
          r.loss.push(f(r.x, r.y));
          if (!r.at && !r.dead && r.loss[r.loss.length - 1] - stats().mn < 0.03) r.at = k;
          if (!r.dead && (r.gn > 3e-3 || S.no > 0 || Math.hypot(r.vx, r.vy) > 1e-3)) live = true;
        }
        const more = live && k < TMAX;
        if (!more) verdict();
        return more;
      };
      begin();
      const cv = (t, kd) => t[OPT[kd].c],
        hgt = (v) => {
          const st = stats();
          return 2.6 * Math.pow(clamp((v - st.mn) / (st.mx - st.mn || 1), 0, 1), 0.62);
        };
      draw = () => {
        const t = T(),
          st = stats(),
          key = M === 'scale' ? 'sc' + ratio().toFixed(1) : S.surf;
        fc.textContent =
          S.dim === '1'
            ? 'The loss along one weight (click to start there)'
            : S.dim === '2'
              ? 'The loss over two weights, from above (click to start there)'
              : 'The loss over two weights, as a landscape (drag to rotate)';
        const { g, w, h } = fit(c);
        GV = { w, h };
        if (S.dim === '1') {
          let top = st.mn;
          for (let i = 0; i <= 60; i++) top = Math.max(top, f(-R + (2 * R * i) / 60, 0));
          top = st.mn + (top - st.mn) * 1.08;
          const P = plot(g, w, h, [-R, R], [st.mn - (top - st.mn) * 0.06, top], t, {
            xlabel: 'weight w',
          });
          GV.P = P;
          clipPlot(g, P);
          area(g, P, (x) => f(x, 0), -R, R, 200);
          g.save();
          g.clip();
          g.fillStyle = mix(t.accent, t.accent, 0, 0.1);
          g.fillRect(0, 0, w, h);
          g.restore();
          curve(g, P, (x) => f(x, 0), -R, R, 300);
          g.strokeStyle = t.accent;
          g.lineWidth = 2.8;
          g.stroke();
          for (const r of runs) {
            const col = cv(t, r.kd);
            g.fillStyle = col;
            r.path.slice(-70).forEach((q, i, a) => {
              g.globalAlpha = 0.15 + (0.6 * i) / a.length;
              g.beginPath();
              g.arc(P.X(q[0]), P.Y(f(q[0], 0)), 3, 0, 7);
              g.fill();
            });
            g.globalAlpha = 1;
            const x = clamp(r.x, -3.3, 3.3),
              y = f(x, 0),
              s = grad(x, 0)[0],
              dx = 0.55 / Math.sqrt(1 + s * s * 0.3);
            g.strokeStyle = col;
            g.lineWidth = 1.5;
            g.beginPath();
            g.moveTo(P.X(x - dx), P.Y(y - s * dx));
            g.lineTo(P.X(x + dx), P.Y(y + s * dx));
            g.stroke();
            g.strokeStyle = t.bg;
            g.lineWidth = 2;
            g.beginPath();
            g.arc(P.X(x), P.Y(y), 8, 0, 7);
            g.fill();
            g.stroke();
          }
          g.restore();
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('short line = slope where it stands', P.m.l + 8, P.m.t + 8);
        } else if (S.dim === '2') {
          const hm = heatmap(key, w, h, t, f, R),
            X = (v) => w / 2 + (v * w) / (2 * R),
            Y = (v) => h / 2 - (v * h) / (2 * R);
          g.drawImage(hm, 0, 0, w, h);
          for (const r of runs) {
            const ln = (lw) => {
              g.lineWidth = lw;
              g.beginPath();
              r.path.forEach((q, i) =>
                i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1])),
              );
              g.stroke();
            };
            g.strokeStyle = mix(t.bg, t.bg, 0, 0.75);
            ln(4.5);
            g.strokeStyle = cv(t, r.kd);
            ln(2.2);
            g.fillStyle = cv(t, r.kd);
            g.strokeStyle = t.bg;
            g.lineWidth = 2.2;
            g.beginPath();
            g.arc(X(clamp(r.x, -3.2, 3.2)), Y(clamp(r.y, -3.2, 3.2)), 7.5, 0, 7);
            g.fill();
            g.stroke();
          }
          g.font = MONO;
          g.fillStyle = t.ink;
          g.textAlign = 'left';
          g.fillText(M === 'scale' ? 'weight on feature 1 →' : 'w₁ →', 8, h - 8);
          g.save();
          g.translate(12, h - 30);
          g.rotate(-Math.PI / 2);
          g.fillText(M === 'scale' ? 'weight on feature 2 →' : 'w₂ →', 0, 0);
          g.restore();
        } else {
          const pr = orb.cam(w, h, 9.2),
            N = 34,
            Pp = [],
            Z = [];
          for (let i = 0; i <= N; i++) {
            Pp.push([]);
            Z.push([]);
            for (let j = 0; j <= N; j++) {
              const x = -R + (2 * R * i) / N,
                y = -R + (2 * R * j) / N,
                v = f(x, y);
              Z[i].push(v);
              Pp[i].push(pr(x, y, hgt(v) - 1));
            }
          }
          const Q = [];
          for (let i = 0; i < N; i++)
            for (let j = 0; j < N; j++)
              Q.push({
                i,
                j,
                v: (Z[i][j] + Z[i + 1][j] + Z[i][j + 1] + Z[i + 1][j + 1]) / 4,
                d: (Pp[i][j][2] + Pp[i + 1][j + 1][2]) / 2,
              });
          Q.sort((a, b) => b.d - a.d);
          g.lineWidth = 0.6;
          for (const q of Q) {
            g.beginPath();
            g.moveTo(Pp[q.i][q.j][0], Pp[q.i][q.j][1]);
            g.lineTo(Pp[q.i + 1][q.j][0], Pp[q.i + 1][q.j][1]);
            g.lineTo(Pp[q.i + 1][q.j + 1][0], Pp[q.i + 1][q.j + 1][1]);
            g.lineTo(Pp[q.i][q.j + 1][0], Pp[q.i][q.j + 1][1]);
            g.closePath();
            g.fillStyle = ramp(
              t,
              Math.pow(1 - Math.sqrt(clamp((q.v - st.mn) / (st.mx - st.mn || 1), 0, 1)), 1.5),
            );
            g.fill();
            g.strokeStyle = mix(t.ink, t.ink, 0, 0.2);
            g.stroke();
          }
          for (const r of runs) {
            const pp = (q) => {
                const x = clamp(q[0], -R, R),
                  y = clamp(q[1], -R, R);
                return pr(x, y, hgt(f(x, y)) - 1 + 0.05);
              },
              ln = (lw) => {
                g.lineWidth = lw;
                g.beginPath();
                r.path.forEach((q, i) => {
                  const p = pp(q);
                  i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]);
                });
                g.stroke();
              };
            g.strokeStyle = mix(t.bg, t.bg, 0, 0.8);
            ln(4.5);
            g.strokeStyle = cv(t, r.kd);
            ln(2.2);
            const p = pp([r.x, r.y]);
            g.fillStyle = cv(t, r.kd);
            g.strokeStyle = t.bg;
            g.lineWidth = 2.2;
            g.beginPath();
            g.arc(p[0], p[1], 7.5, 0, 7);
            g.fill();
            g.stroke();
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          let a = pr(0, -R * 1.12, -1);
          g.fillText('w₁', a[0], a[1]);
          a = pr(-R * 1.12, 0, -1);
          g.fillText('w₂', a[0], a[1]);
        }
        if (runs.length > 1) {
          g.font = MONO;
          g.textAlign = 'left';
          let ly = 14;
          for (const r of runs) {
            g.fillStyle = cv(t, r.kd);
            g.fillText('● ' + OPT[r.kd].n, 8, ly);
            ly += 14;
          }
        }
        {
          const { g, w, h } = fit(cl),
            l0 = runs.length ? runs[0].loss[0] : 1,
            lo = st.mn,
            P = plot(g, w, h, [0, Math.max(40, k)], [lo, Math.max(lo + 0.1, l0 * 1.08)], t, {
              xstep: k > 200 ? 100 : k > 80 ? 50 : 10,
              xlabel: 'step',
            });
          clipPlot(g, P);
          if (M === 'sched') {
            g.setLineDash([4, 4]);
            g.strokeStyle = t.accent;
            g.lineWidth = 1.6;
            g.beginPath();
            const n = Math.max(40, k);
            for (let i = 0; i <= n; i++) {
              const y = P.Y(lo) + (P.m.t + 16 - P.Y(lo)) * sch(i);
              i ? g.lineTo(P.X(i), y) : g.moveTo(P.X(i), y);
            }
            g.stroke();
            g.setLineDash([]);
          }
          for (const r of runs) {
            g.strokeStyle = cv(t, r.kd);
            g.lineWidth = 2.4;
            g.beginPath();
            r.loss.forEach((v, i) =>
              i ? g.lineTo(P.X(i), P.Y(Math.min(v, l0 * 3))) : g.moveTo(P.X(i), P.Y(v)),
            );
            g.stroke();
          }
          g.restore();
          g.font = MONO;
          g.textAlign = 'right';
          if (M === 'sched') {
            g.fillStyle = t.accent;
            g.fillText('dashed = learning rate', w - P.m.r, P.m.t + 8);
          }
        }
        setR(root, 'k', String(k));
        K0.forEach((q) => {
          const r = runs.find((x) => x.kd === q);
          setR(root, 'L' + q, r ? (r.dead ? 'diverged' : f3(r.loss[r.loss.length - 1])) : 'off');
        });
        setR(root, 'lo', f3(st.mn));
        setR(root, 'lr', f3(lrNow(k)));
        setR(root, 'kap', (ratio() * ratio()).toFixed(1));
        setR(root, 'mx', f2(2 / (0.6 * ratio())));
        root
          .querySelectorAll('#' + id + '-dim button')
          .forEach((b) => b.setAttribute('aria-pressed', b.dataset.d === S.dim ? 'true' : 'false'));
      };
      const rnd = () => [Math.random() * 5 - 2.5, Math.random() * 5 - 2.5];
      let fin = false,
        chk = () => false;
      const anim = Anim(() => {
        fr++;
        if (fr % 2) return;
        if (!runs.length) {
          pb.stop();
          return;
        }
        if (!step()) {
          fin = true;
          draw();
          pb.stop();
          return;
        }
        draw();
        chk();
      });
      pb = playBtn($('#' + id + '-play', root), anim);
      chk = runCtl(pb, root, id, {
        done: () => fin && k > 0,
        restart: () => {
          fin = false;
          S.hold ? begin() : begin(rnd());
        },
        count: () => k,
      });
      $('#' + id + '-play', root).addEventListener('click', () => {
        if (anim.on && (k >= TMAX || note.innerHTML)) begin();
      });
      $('#' + id + '-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#' + id + '-again', root).onclick = () => {
        S.hold = false;
        begin(rnd());
        draw();
      };
      const rs = () => {
          S.hold = false;
          ST = null;
          begin();
          draw();
        },
        B = (sfx, key, fm) => {
          if ($('#' + id + '-' + sfx, root))
            ref[key] = bind(
              root,
              id + '-' + sfx,
              (v) => {
                S[key] = v;
                rs();
              },
              fm,
            );
        };
      B('lr', 'lr', f3);
      B('lr2', 'lr2', f3);
      B('mo', 'mo');
      B('no', 'no');
      B('da', 'da');
      B('s', 's', (v) => (+v).toFixed(1));
      const fs = $('#' + id + '-f', root);
      if (fs)
        fs.onchange = (e) => {
          S.surf = e.target.value;
          rs();
        };
      const ss = $('#' + id + '-sc', root);
      if (ss)
        ss.onchange = (e) => {
          S.sc = e.target.value;
          rs();
        };
      const sd = $('#' + id + '-std', root);
      if (sd)
        sd.onchange = (e) => {
          S.std = e.target.checked;
          rs();
        };
      K0.forEach((q) => {
        const cb = $('#' + id + '-on-' + q, root);
        if (cb)
          cb.onchange = (e) => {
            S.on[q] = e.target.checked;
            rs();
          };
      });
      $('#' + id + '-dim', root).addEventListener('click', (e) => {
        const d = e.target.dataset && e.target.dataset.d;
        if (!d) return;
        S.dim = d;
        ST = null;
        orb.spin(false);
        begin();
        draw();
      });
      const cs = $('#' + id + '-cases', root);
      if (cs)
        cs.addEventListener('click', (e) => {
          const q = CASES[e.target.dataset && e.target.dataset.case];
          if (!q) return;
          S.surf = q.surf;
          fs.value = q.surf;
          S.lr = q.lr;
          ref.lr.set(q.lr);
          S.no = q.no;
          ref.no.set(q.no);
          if (ref.mo) {
            S.mo = 0;
            ref.mo.set(0);
          }
          ST = null;
          S.hold = true;
          begin(S.dim === '1' ? [q.p1, 0] : q.p.slice());
          draw();
          pb.start();
        });
      c.addEventListener('click', (e) => {
        if (!GV || S.dim === '3') return;
        const r = c.getBoundingClientRect();
        S.hold = false;
        if (S.dim === '1') {
          const P = GV.P;
          begin([
            clamp(-R + ((e.clientX - r.left - P.m.l) / (P.w - P.m.l - P.m.r)) * 2 * R, -2.95, 2.95),
            0,
          ]);
        } else
          begin([
            clamp((e.clientX - r.left - GV.w / 2) / (GV.w / (2 * R)), -2.9, 2.9),
            clamp(-(e.clientY - r.top - GV.h / 2) / (GV.h / (2 * R)), -2.9, 2.9),
          ]);
        draw();
        pb.start();
      });
      return {
        draw,
        stop() {
          pb.stop();
          orb.spin(false);
        },
        auto: () => pb.start(),
      };
    },
  };
}

export { SURF, HEAT, heatmap, OPT, CASES, gdLesson };
