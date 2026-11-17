/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  area,
  bind,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  plot,
  ramp,
  randn,
  attachRunLimit,
  setReadout,
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
  big: { t: 'Step Too Large', surfaceId: 'ravine', p: [-2.4, 1.2], p1: 1.6, lr: 0.9, no: 0 },
  saddle: { t: 'Balanced on a Saddle', surfaceId: 'saddle', p: [-2.6, 0], p1: 0, lr: 0.15, no: 0 },
  plateau: {
    t: 'Lost on a Plateau',
    surfaceId: 'plateau',
    p: [2.8, 2.7],
    p1: 2.9,
    lr: 0.03,
    no: 0,
  },
  valley: { t: 'The Wrong Valley', surfaceId: 'two', p: [2.6, 1.5], p1: 2.7, lr: 0.1, no: 0 },
  noise: {
    t: 'Noise That Never Settles',
    surfaceId: 'bowl',
    p: [-2.4, 2.2],
    p1: -2.4,
    lr: 0.3,
    no: 1.5,
  },
};

function gdLesson(options) {
  const id = options.id;
  const mode = options.mode;
  const optimizerIds =
    mode === 'newton' ? ['gd', 'newton'] : mode === 'opt' ? ['sgd', 'mom', 'rms', 'adam'] : ['gd'];
  return {
    ...options,
    renderStage(parts, context) {
      return renderFamily(new URL('./shared.html', import.meta.url), id, parts, context);
    },
    init(root) {
      const c = $('#' + id + '-c', root),
        cl = $('#' + id + '-l', root),
        note = $('#' + id + '-note', root),
        fc = $('#' + id + '-fc', root),
        R = 3,
        TMAX = mode === 'sched' ? 300 : 400,
        settings = {
          surfaceId: options.surfaceId || 'bowl',
          dim: '3',
          lr: mode === 'sched' ? 0.35 : mode === 'scale' ? 0.25 : mode === 'gd' ? 0.3 : 0.1,
          adaptiveLearningRate: 0.08,
          momentum: 0,
          no: mode === 'sched' ? 0.5 : mode === 'opt' ? 0.3 : 0,
          newtonDamping: 1,
          scheduleKind: 'cos',
          featureScaleRatio: 4,
          standardizeFeatures: false,
          on: { sgd: true, mom: true, rms: true, adam: true },
          hold: false,
        };
      let runs = [],
        k = 0,
        wait = 0,
        fr = 0,
        pb,
        plotGeometry = null,
        start = [-2.4, 2.2],
        draw,
        surfaceStatsCache = null;
      const orb = Orbit(
          c,
          () => {
            if (settings.dim === '3') draw();
          },
          null,
          { az: -0.6, el: 0.62, zoom: 1 },
        ),
        ref = {};
      const kinds = () =>
          mode === 'opt' ? optimizerIds.filter((q) => settings.on[q]) : optimizerIds,
        ratio = () => (settings.standardizeFeatures ? 1 : settings.featureScaleRatio);
      const f = (x, y) =>
        mode === 'scale'
          ? 0.3 * ((x * x) / ratio() + ratio() * y * y)
          : settings.dim === '1'
            ? SURF[settings.surfaceId].one(x)
            : SURF[settings.surfaceId].f(x, y);
      const stats = () => {
        const key =
          (mode === 'scale' ? 'sc' + ratio().toFixed(1) : settings.surfaceId) + settings.dim;
        if (surfaceStatsCache && surfaceStatsCache.key === key) return surfaceStatsCache;
        let mn = 1 / 0,
          mx = -1 / 0;
        for (let i = 0; i <= 90; i++)
          for (let j = 0; j <= (settings.dim === '1' ? 0 : 90); j++) {
            const v = f(-R + (2 * R * i) / 90, settings.dim === '1' ? 0 : -R + (2 * R * j) / 90);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
        return (surfaceStatsCache = { key, mn, mx });
      };
      const grad = (x, y) => {
        const e = 1e-4;
        return [
          (f(x + e, y) - f(x - e, y)) / (2 * e),
          settings.dim === '1' ? 0 : (f(x, y + e) - f(x, y - e)) / (2 * e),
        ];
      };
      const learningRateMultiplier = (i) =>
        settings.scheduleKind === 'const'
          ? 1
          : settings.scheduleKind === 'step'
            ? Math.pow(0.5, Math.floor(i / 75))
            : settings.scheduleKind === 'exp'
              ? Math.exp(-i / 90)
              : settings.scheduleKind === 'inv'
                ? 1 / (1 + i / 30)
                : settings.scheduleKind === 'cos'
                  ? 0.5 * (1 + Math.cos((Math.PI * i) / TMAX))
                  : i < 25
                    ? (i + 1) / 25
                    : 0.5 * (1 + Math.cos((Math.PI * (i - 25)) / (TMAX - 25)));
      const lrNow = (i) =>
        mode === 'sched' ? settings.lr * learningRateMultiplier(i) : settings.lr;
      const begin = (p) => {
        if (p) start = p;
        const y0 = settings.dim === '1' ? 0 : start[1];
        runs = kinds().map((optimizerId) => ({
          optimizerId,
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
              nm = `<b style="color:var(--${OPT[r.optimizerId].c === 'gold' ? 'cut' : OPT[r.optimizerId].c})">${OPT[r.optimizerId].n}</b>: `;
            if (r.dead)
              return (
                nm +
                (r.optimizerId === 'newton'
                  ? 'could not take a step here (the curvature is flat).'
                  : 'diverged. Every step overshot further than the last: the step is too large for the steepest direction.')
              );
            if (r.nearMinimumStep)
              return (
                nm +
                (settings.no
                  ? 'got near the bottom in ' +
                    r.nearMinimumStep +
                    ' steps, but the noise keeps knocking it about. It will not settle at this step size.'
                  : 'reached the lowest point in ' + r.nearMinimumStep + ' steps.')
              );
            if (r.gn < 3e-3 && !settings.no)
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
                ? settings.no
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
          if (r.optimizerId === 'newton') {
            const e = 1e-3,
              hxx = (f(r.x + e, r.y) - 2 * f(r.x, r.y) + f(r.x - e, r.y)) / (e * e);
            let dx,
              dy = 0,
              positiveCurvature;
            if (settings.dim === '1') {
              if (Math.abs(hxx) < 1e-7) {
                r.dead = true;
                continue;
              }
              dx = -g[0] / hxx;
              positiveCurvature = hxx > 0;
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
              positiveCurvature = det > 0 && hxx > 0;
            }
            const n = Math.hypot(dx, dy);
            if (n > 2.5) {
              dx *= 2.5 / n;
              dy *= 2.5 / n;
            }
            r.x += settings.newtonDamping * dx;
            r.y += settings.newtonDamping * dy;
            if (!positiveCurvature) r.bad = true;
          } else {
            if (settings.no)
              g = [
                g[0] + settings.no * randn(),
                settings.dim === '1' ? 0 : g[1] + settings.no * randn(),
              ];
            const optimizerId = r.optimizerId;
            if (optimizerId === 'gd') {
              const lr = lrNow(k - 1);
              r.vx = settings.momentum * r.vx - lr * g[0];
              r.vy = settings.momentum * r.vy - lr * g[1];
              r.x += r.vx;
              r.y += r.vy;
            } else if (optimizerId === 'sgd') {
              r.x -= settings.lr * g[0];
              r.y -= settings.lr * g[1];
            } else if (optimizerId === 'mom') {
              r.vx = 0.9 * r.vx - settings.lr * g[0];
              r.vy = 0.9 * r.vy - settings.lr * g[1];
              r.x += r.vx;
              r.y += r.vy;
            } else if (optimizerId === 'rms') {
              r.sx = 0.9 * r.sx + 0.1 * g[0] * g[0];
              r.sy = 0.9 * r.sy + 0.1 * g[1] * g[1];
              r.x -= (settings.adaptiveLearningRate * g[0]) / (Math.sqrt(r.sx) + 1e-8);
              r.y -= (settings.adaptiveLearningRate * g[1]) / (Math.sqrt(r.sy) + 1e-8);
            } else {
              r.mx = 0.9 * r.mx + 0.1 * g[0];
              r.my = 0.9 * r.my + 0.1 * g[1];
              r.sx = 0.999 * r.sx + 0.001 * g[0] * g[0];
              r.sy = 0.999 * r.sy + 0.001 * g[1] * g[1];
              const b1 = 1 - Math.pow(0.9, k),
                b2 = 1 - Math.pow(0.999, k);
              r.x -= (settings.adaptiveLearningRate * (r.mx / b1)) / (Math.sqrt(r.sx / b2) + 1e-8);
              r.y -= (settings.adaptiveLearningRate * (r.my / b1)) / (Math.sqrt(r.sy / b2) + 1e-8);
            }
          }
          if (!isFinite(r.x) || !isFinite(r.y) || Math.abs(r.x) > 30 || Math.abs(r.y) > 30) {
            r.dead = true;
            r.x = clamp(r.x || 0, -30, 30);
            r.y = clamp(r.y || 0, -30, 30);
          }
          r.path.push([r.x, r.y]);
          r.loss.push(f(r.x, r.y));
          if (!r.nearMinimumStep && !r.dead && r.loss[r.loss.length - 1] - stats().mn < 0.03)
            r.nearMinimumStep = k;
          if (!r.dead && (r.gn > 3e-3 || settings.no > 0 || Math.hypot(r.vx, r.vy) > 1e-3))
            live = true;
        }
        const more = live && k < TMAX;
        if (!more) verdict();
        return more;
      };
      begin();
      const cv = (t, optimizerId) => t[OPT[optimizerId].c],
        hgt = (v) => {
          const st = stats();
          return 2.6 * Math.pow(clamp((v - st.mn) / (st.mx - st.mn || 1), 0, 1), 0.62);
        };
      draw = () => {
        const colors = readThemeColors(),
          st = stats(),
          key = mode === 'scale' ? 'sc' + ratio().toFixed(1) : settings.surfaceId;
        fc.textContent =
          settings.dim === '1'
            ? 'The loss along one weight (click to start there)'
            : settings.dim === '2'
              ? 'The loss over two weights, from above (click to start there)'
              : 'The loss over two weights, as a landscape (drag to rotate)';
        const { g: ctx, w, h } = prepareCanvas(c);
        plotGeometry = { w, h };
        if (settings.dim === '1') {
          let top = st.mn;
          for (let i = 0; i <= 60; i++) top = Math.max(top, f(-R + (2 * R * i) / 60, 0));
          top = st.mn + (top - st.mn) * 1.08;
          const P = plot(ctx, w, h, [-R, R], [st.mn - (top - st.mn) * 0.06, top], colors, {
            xlabel: 'weight w',
          });
          plotGeometry.P = P;
          clipPlot(ctx, P);
          area(ctx, P, (x) => f(x, 0), -R, R, 200);
          ctx.save();
          ctx.clip();
          ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.1);
          ctx.fillRect(0, 0, w, h);
          ctx.restore();
          curve(ctx, P, (x) => f(x, 0), -R, R, 300);
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 2.8;
          ctx.stroke();
          for (const r of runs) {
            const col = cv(colors, r.optimizerId);
            ctx.fillStyle = col;
            r.path.slice(-70).forEach((q, i, a) => {
              ctx.globalAlpha = 0.15 + (0.6 * i) / a.length;
              ctx.beginPath();
              ctx.arc(P.X(q[0]), P.Y(f(q[0], 0)), 3, 0, 7);
              ctx.fill();
            });
            ctx.globalAlpha = 1;
            const x = clamp(r.x, -3.3, 3.3),
              y = f(x, 0),
              s = grad(x, 0)[0],
              dx = 0.55 / Math.sqrt(1 + s * s * 0.3);
            ctx.strokeStyle = col;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(P.X(x - dx), P.Y(y - s * dx));
            ctx.lineTo(P.X(x + dx), P.Y(y + s * dx));
            ctx.stroke();
            ctx.strokeStyle = colors.bg;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(P.X(x), P.Y(y), 8, 0, 7);
            ctx.fill();
            ctx.stroke();
          }
          ctx.restore();
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText('short line = slope where it stands', P.m.l + 8, P.m.t + 8);
        } else if (settings.dim === '2') {
          const hm = heatmap(key, w, h, colors, f, R),
            X = (v) => w / 2 + (v * w) / (2 * R),
            Y = (v) => h / 2 - (v * h) / (2 * R);
          ctx.drawImage(hm, 0, 0, w, h);
          for (const r of runs) {
            const ln = (lw) => {
              ctx.lineWidth = lw;
              ctx.beginPath();
              r.path.forEach((q, i) =>
                i ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1])),
              );
              ctx.stroke();
            };
            ctx.strokeStyle = mix(colors.bg, colors.bg, 0, 0.75);
            ln(4.5);
            ctx.strokeStyle = cv(colors, r.optimizerId);
            ln(2.2);
            ctx.fillStyle = cv(colors, r.optimizerId);
            ctx.strokeStyle = colors.bg;
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            ctx.arc(X(clamp(r.x, -3.2, 3.2)), Y(clamp(r.y, -3.2, 3.2)), 7.5, 0, 7);
            ctx.fill();
            ctx.stroke();
          }
          ctx.font = MONO;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'left';
          ctx.fillText(mode === 'scale' ? 'weight on feature 1 →' : 'w₁ →', 8, h - 8);
          ctx.save();
          ctx.translate(12, h - 30);
          ctx.rotate(-Math.PI / 2);
          ctx.fillText(mode === 'scale' ? 'weight on feature 2 →' : 'w₂ →', 0, 0);
          ctx.restore();
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
          ctx.lineWidth = 0.6;
          for (const q of Q) {
            ctx.beginPath();
            ctx.moveTo(Pp[q.i][q.j][0], Pp[q.i][q.j][1]);
            ctx.lineTo(Pp[q.i + 1][q.j][0], Pp[q.i + 1][q.j][1]);
            ctx.lineTo(Pp[q.i + 1][q.j + 1][0], Pp[q.i + 1][q.j + 1][1]);
            ctx.lineTo(Pp[q.i][q.j + 1][0], Pp[q.i][q.j + 1][1]);
            ctx.closePath();
            ctx.fillStyle = ramp(
              colors,
              Math.pow(1 - Math.sqrt(clamp((q.v - st.mn) / (st.mx - st.mn || 1), 0, 1)), 1.5),
            );
            ctx.fill();
            ctx.strokeStyle = mix(colors.ink, colors.ink, 0, 0.2);
            ctx.stroke();
          }
          for (const r of runs) {
            const pp = (q) => {
                const x = clamp(q[0], -R, R),
                  y = clamp(q[1], -R, R);
                return pr(x, y, hgt(f(x, y)) - 1 + 0.05);
              },
              ln = (lw) => {
                ctx.lineWidth = lw;
                ctx.beginPath();
                r.path.forEach((q, i) => {
                  const p = pp(q);
                  i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
                });
                ctx.stroke();
              };
            ctx.strokeStyle = mix(colors.bg, colors.bg, 0, 0.8);
            ln(4.5);
            ctx.strokeStyle = cv(colors, r.optimizerId);
            ln(2.2);
            const p = pp([r.x, r.y]);
            ctx.fillStyle = cv(colors, r.optimizerId);
            ctx.strokeStyle = colors.bg;
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            ctx.arc(p[0], p[1], 7.5, 0, 7);
            ctx.fill();
            ctx.stroke();
          }
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          let a = pr(0, -R * 1.12, -1);
          ctx.fillText('w₁', a[0], a[1]);
          a = pr(-R * 1.12, 0, -1);
          ctx.fillText('w₂', a[0], a[1]);
        }
        if (runs.length > 1) {
          ctx.font = MONO;
          ctx.textAlign = 'left';
          let ly = 14;
          for (const r of runs) {
            ctx.fillStyle = cv(colors, r.optimizerId);
            ctx.fillText('● ' + OPT[r.optimizerId].n, 8, ly);
            ly += 14;
          }
        }
        {
          const { g: ctx, w, h } = prepareCanvas(cl),
            l0 = runs.length ? runs[0].loss[0] : 1,
            lo = st.mn,
            P = plot(ctx, w, h, [0, Math.max(40, k)], [lo, Math.max(lo + 0.1, l0 * 1.08)], colors, {
              xstep: k > 200 ? 100 : k > 80 ? 50 : 10,
              xlabel: 'step',
            });
          clipPlot(ctx, P);
          if (mode === 'sched') {
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = colors.accent;
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            const n = Math.max(40, k);
            for (let i = 0; i <= n; i++) {
              const y = P.Y(lo) + (P.m.t + 16 - P.Y(lo)) * learningRateMultiplier(i);
              i ? ctx.lineTo(P.X(i), y) : ctx.moveTo(P.X(i), y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
          }
          for (const r of runs) {
            ctx.strokeStyle = cv(colors, r.optimizerId);
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            r.loss.forEach((v, i) =>
              i ? ctx.lineTo(P.X(i), P.Y(Math.min(v, l0 * 3))) : ctx.moveTo(P.X(i), P.Y(v)),
            );
            ctx.stroke();
          }
          ctx.restore();
          ctx.font = MONO;
          ctx.textAlign = 'right';
          if (mode === 'sched') {
            ctx.fillStyle = colors.accent;
            ctx.fillText('dashed = learning rate', w - P.m.r, P.m.t + 8);
          }
        }
        setReadout(root, 'k', String(k));
        optimizerIds.forEach((q) => {
          const r = runs.find((run) => run.optimizerId === q);
          setReadout(
            root,
            'L' + q,
            r ? (r.dead ? 'diverged' : f3(r.loss[r.loss.length - 1])) : 'off',
          );
        });
        setReadout(root, 'lo', f3(st.mn));
        setReadout(root, 'lr', f3(lrNow(k)));
        setReadout(root, 'kap', (ratio() * ratio()).toFixed(1));
        setReadout(root, 'mx', f2(2 / (0.6 * ratio())));
        root
          .querySelectorAll('#' + id + '-dim button')
          .forEach((b) =>
            b.setAttribute('aria-pressed', b.dataset.d === settings.dim ? 'true' : 'false'),
          );
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
      chk = attachRunLimit(pb, root, id, {
        done: () => fin && k > 0,
        restart: () => {
          fin = false;
          settings.hold ? begin() : begin(rnd());
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
        settings.hold = false;
        begin(rnd());
        draw();
      };
      const rs = () => {
          settings.hold = false;
          surfaceStatsCache = null;
          begin();
          draw();
        },
        B = (sfx, key, fm) => {
          if ($('#' + id + '-' + sfx, root))
            ref[key] = bind(
              root,
              id + '-' + sfx,
              (v) => {
                settings[key] = v;
                rs();
              },
              fm,
            );
        };
      B('lr', 'lr', f3);
      B('lr2', 'adaptiveLearningRate', f3);
      B('mo', 'momentum');
      B('no', 'no');
      B('da', 'newtonDamping');
      B('s', 'featureScaleRatio', (v) => (+v).toFixed(1));
      const fs = $('#' + id + '-f', root);
      if (fs)
        fs.onchange = (e) => {
          settings.surfaceId = e.target.value;
          rs();
        };
      const ss = $('#' + id + '-sc', root);
      if (ss)
        ss.onchange = (e) => {
          settings.scheduleKind = e.target.value;
          rs();
        };
      const sd = $('#' + id + '-std', root);
      if (sd)
        sd.onchange = (e) => {
          settings.standardizeFeatures = e.target.checked;
          rs();
        };
      optimizerIds.forEach((q) => {
        const cb = $('#' + id + '-on-' + q, root);
        if (cb)
          cb.onchange = (e) => {
            settings.on[q] = e.target.checked;
            rs();
          };
      });
      $('#' + id + '-dim', root).addEventListener('click', (e) => {
        const d = e.target.dataset && e.target.dataset.d;
        if (!d) return;
        settings.dim = d;
        surfaceStatsCache = null;
        orb.spin(false);
        begin();
        draw();
      });
      const cs = $('#' + id + '-cases', root);
      if (cs)
        cs.addEventListener('click', (e) => {
          const q = CASES[e.target.dataset && e.target.dataset.case];
          if (!q) return;
          settings.surfaceId = q.surfaceId;
          fs.value = q.surfaceId;
          settings.lr = q.lr;
          ref.lr.set(q.lr);
          settings.no = q.no;
          ref.no.set(q.no);
          if (ref.momentum) {
            settings.momentum = 0;
            ref.momentum.set(0);
          }
          surfaceStatsCache = null;
          settings.hold = true;
          begin(settings.dim === '1' ? [q.p1, 0] : q.p.slice());
          draw();
          pb.start();
        });
      c.addEventListener('click', (e) => {
        if (!plotGeometry || settings.dim === '3') return;
        const r = c.getBoundingClientRect();
        settings.hold = false;
        if (settings.dim === '1') {
          const P = plotGeometry.P;
          begin([
            clamp(-R + ((e.clientX - r.left - P.m.l) / (P.w - P.m.l - P.m.r)) * 2 * R, -2.95, 2.95),
            0,
          ]);
        } else
          begin([
            clamp(
              (e.clientX - r.left - plotGeometry.w / 2) / (plotGeometry.w / (2 * R)),
              -2.9,
              2.9,
            ),
            clamp(
              -(e.clientY - r.top - plotGeometry.h / 2) / (plotGeometry.h / (2 * R)),
              -2.9,
              2.9,
            ),
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
