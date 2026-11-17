/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  MONO,
  SANS,
  readThemeColors,
  bind,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  prepareCanvas,
  grab,
  markTP,
  mix,
  playBtn,
  plot,
  ramp,
  randnFrom,
  rng,
  setReadout,
  sweeper,
  testPoint,
} from '../../../js/core/shared.js';
import { dot, legendre, lsq } from '../shared.js';

export default {
  init(root) {
    const mainCanvas = $('#rg-c', root),
      cp = $('#rg-p', root),
      cb = $('#rg-b', root),
      cg = $('#rg-g', root),
      c1 = $('#rg-s1', root),
      ca = $('#rg-ap', root),
      unpenalizedOptimum = [1.7, 0.75],
      TA = [1, 0.55, 1.3],
      settings = { li: 20, a: 0, noise: 0.3, n: 26 },
      Fq = (x) => Math.sin(2.8 * x),
      D = 10,
      NL = 60,
      logLambdaAt = (i) => -4.2 + (5.4 * i) / (NL - 1);
    let seed = 12,
      pts = [],
      mu = [],
      sd = [],
      ym = 0,
      W = [],
      etr = [],
      ete = [],
      playback;
    const feat = (x) => {
        const P = legendre(x, D),
          o = [];
        for (let j = 1; j <= D; j++) o.push((P[j] - mu[j - 1]) / sd[j - 1]);
        return o;
      },
      pred = (x, w) => ym + dot(w, feat(x));
    const data = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < settings.n; i++) {
        const x = r() * 2 - 1;
        pts.push({ x, y: Fq(x) + settings.noise * rn(), test: r() < 0.3 });
      }
    };
    const solveAll = () => {
      const tr = pts.filter((p) => !p.test),
        te = pts.filter((p) => p.test),
        n = tr.length,
        R = tr.map((p) => legendre(p.x, D));
      mu = [];
      sd = [];
      for (let j = 1; j <= D; j++) {
        let m = 0;
        for (const r of R) m += r[j];
        m /= n;
        let v = 0;
        for (const r of R) v += (r[j] - m) ** 2;
        mu.push(m);
        sd.push(Math.sqrt(v / n) || 1);
      }
      ym = tr.reduce((s, p) => s + p.y, 0) / n;
      const Xs = tr.map((p) => feat(p.x)),
        yc = tr.map((p) => p.y - ym);
      let w = new Array(D).fill(0);
      W = [];
      etr = [];
      ete = [];
      for (let i = NL - 1; i >= 0; i--) {
        const lam = Math.pow(10, logLambdaAt(i)),
          res = yc.map((v, k) => v - dot(w, Xs[k]));
        for (let it = 0; it < 260; it++) {
          let mxd = 0;
          for (let j = 0; j < D; j++) {
            let rho = 0;
            for (let k = 0; k < n; k++) rho += Xs[k][j] * (res[k] + Xs[k][j] * w[j]);
            rho /= n;
            const th = lam * settings.a,
              nw = (rho > th ? rho - th : rho < -th ? rho + th : 0) / (1 + lam * (1 - settings.a)),
              dl = nw - w[j];
            if (dl) {
              for (let k = 0; k < n; k++) res[k] -= Xs[k][j] * dl;
              w[j] = nw;
              mxd = Math.max(mxd, Math.abs(dl));
            }
          }
          if (mxd < 1e-7) break;
        }
        W[i] = w.slice();
        const e = (P) =>
          P.length ? P.reduce((s, p) => s + (p.y - pred(p.x, w)) ** 2, 0) / P.length : NaN;
        etr[i] = e(tr);
        ete[i] = e(te);
      }
    };
    data();
    solveAll();
    let geometryViewport = null;
    const tp = testPoint(root, 'rg', ['Your x'], [0.4], [[-1, 1]]);
    tp.on = () => draw();
    const AP = (() => {
      const names = [
          'Hours studied',
          'Classes attended',
          'Hours of sleep',
          'Shoe size',
          'Height',
          'Phone battery',
          'Day of birth',
          'House number',
          'Cups of coffee',
          'Desk row',
          'Pets owned',
          'Commute minutes',
        ],
        bt = [1, 0.7, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        p = 12,
        n = 30,
        r = rng(31),
        rn = randnFrom(r);
      const mk = (m) => {
          const X = [],
            y = [];
          for (let i = 0; i < m; i++) {
            const row = [];
            for (let j = 0; j < p; j++) row.push(rn());
            X.push(row);
            y.push(dot(row, bt) + 1.0 * rn());
          }
          return { X, y };
        },
        tr = mk(n),
        te = mk(400);
      const mse = (w) => te.y.reduce((s, v, i) => s + (v - dot(w, te.X[i])) ** 2, 0) / te.y.length,
        ols = lsq(tr.X, tr.y, 1e-9);
      const lasso = (lam) => {
        const w = new Array(p).fill(0),
          res = tr.y.slice();
        for (let it = 0; it < 300; it++) {
          let mxd = 0;
          for (let j = 0; j < p; j++) {
            let rho = 0,
              xx = 0;
            for (let k = 0; k < n; k++) {
              rho += tr.X[k][j] * (res[k] + tr.X[k][j] * w[j]);
              xx += tr.X[k][j] ** 2;
            }
            rho /= n;
            xx /= n;
            const nw = (rho > lam ? rho - lam : rho < -lam ? rho + lam : 0) / xx,
              dl = nw - w[j];
            if (dl) {
              for (let k = 0; k < n; k++) res[k] -= tr.X[k][j] * dl;
              w[j] = nw;
              mxd = Math.max(mxd, Math.abs(dl));
            }
          }
          if (mxd < 1e-7) break;
        }
        return w;
      };
      return {
        names,
        bt,
        p,
        ols,
        eo: mse(ols),
        ridge: (lam) => lsq(tr.X, tr.y, n * lam),
        lasso,
        mse,
      };
    })();
    const toy = (lam) => {
      let w = [unpenalizedOptimum[0], unpenalizedOptimum[1]];
      const sft = (r, th) => (r > th ? r - th : r < -th ? r + th : 0);
      for (let it = 0; it < 120; it++) {
        w[0] =
          sft(
            TA[0] * unpenalizedOptimum[0] - TA[1] * (w[1] - unpenalizedOptimum[1]),
            lam * settings.a,
          ) /
          (TA[0] + lam * (1 - settings.a));
        w[1] =
          sft(
            TA[2] * unpenalizedOptimum[1] - TA[1] * (w[0] - unpenalizedOptimum[0]),
            lam * settings.a,
          ) /
          (TA[2] + lam * (1 - settings.a));
      }
      return w;
    };
    const tlam = (i) => Math.pow(10, -2.2 + (3.3 * i) / (NL - 1));
    const draw = () => {
      const colors = readThemeColors(),
        w = W[settings.li];
      {
        const { g: ctx, w: cw, h } = prepareCanvas(c1),
          lam = Math.pow(10, -1.5 + (1.9 * settings.li) / (NL - 1)),
          m = { l: 46, r: 12, t: 12, b: 40 },
          R = 3,
          X = (v) => m.l + ((v + R) / (2 * R)) * (cw - m.l - m.r),
          Y = (v) => h - m.b - ((v + R) / (2 * R)) * (h - m.t - m.b),
          sf = (v) => (v > lam ? v - lam : v < -lam ? v + lam : 0);
        ctx.fillStyle = mix(colors.ok, colors.ok, 0, 0.1);
        ctx.fillRect(
          X(-Math.min(lam, R)),
          m.t,
          X(Math.min(lam, R)) - X(-Math.min(lam, R)),
          h - m.t - m.b,
        );
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        for (let v = -3; v <= 3; v++) {
          ctx.beginPath();
          ctx.moveTo(X(v), m.t);
          ctx.lineTo(X(v), h - m.b);
          ctx.moveTo(m.l, Y(v));
          ctx.lineTo(cw - m.r, Y(v));
          ctx.stroke();
          ctx.textAlign = 'center';
          ctx.fillText(String(v), X(v), h - m.b + 14);
          ctx.textAlign = 'right';
          ctx.fillText(String(v), m.l - 6, Y(v) + 4);
        }
        ctx.strokeStyle = colors.muted;
        ctx.beginPath();
        ctx.moveTo(X(0), m.t);
        ctx.lineTo(X(0), h - m.b);
        ctx.moveTo(m.l, Y(0));
        ctx.lineTo(cw - m.r, Y(0));
        ctx.stroke();
        ctx.textAlign = 'center';
        ctx.fillText('coefficient with no penalty', cw / 2 + m.l / 2, h - 8);
        ctx.save();
        ctx.translate(12, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('coefficient after the penalty', 0, 0);
        ctx.restore();
        ctx.save();
        ctx.beginPath();
        ctx.rect(m.l, m.t, cw - m.l - m.r, h - m.t - m.b);
        ctx.clip();
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(X(-R), Y(-R));
        ctx.lineTo(X(R), Y(R));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 3.2;
        ctx.beginPath();
        ctx.moveTo(X(-R), Y(-R / (1 + lam)));
        ctx.lineTo(X(R), Y(R / (1 + lam)));
        ctx.stroke();
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 3.2;
        ctx.beginPath();
        ctx.moveTo(X(-R), Y(sf(-R)));
        ctx.lineTo(X(-Math.min(lam, R)), Y(0));
        ctx.lineTo(X(Math.min(lam, R)), Y(0));
        ctx.lineTo(X(R), Y(sf(R)));
        ctx.stroke();
        [-2.4, -1.5, -0.6, 0.4, 1.1, 2.1].forEach((v) => {
          ctx.strokeStyle = colors.line;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(X(v), Y(v));
          ctx.lineTo(X(v), Y(Math.abs(sf(v)) < Math.abs(v / (1 + lam)) ? sf(v) : v / (1 + lam)));
          ctx.stroke();
          ctx.fillStyle = colors.muted;
          ctx.beginPath();
          ctx.arc(X(v), Y(v), 3.5, 0, 7);
          ctx.fill();
          ctx.fillStyle = colors.accent;
          ctx.beginPath();
          ctx.arc(X(v), Y(v / (1 + lam)), 5, 0, 7);
          ctx.fill();
          ctx.fillStyle = colors.ok;
          ctx.beginPath();
          ctx.arc(X(v), Y(sf(v)), 5, 0, 7);
          ctx.fill();
        });
        ctx.restore();
        ctx.font = SANS;
        ctx.textAlign = 'left';
        let ly = m.t + 16;
        [
          [colors.muted, 'No penalty: unchanged'],
          [colors.accent, 'Ridge: every one scaled down, none reach zero'],
          [colors.ok, 'Lasso: every one moved toward zero by λ;'],
          [colors.ok, 'anything inside the shaded band becomes exactly 0'],
        ].forEach(([col, sx]) => {
          ctx.fillStyle = col;
          ctx.fillText(sx, m.l + 10, ly);
          ly += 17;
        });
      }
      {
        const { g: ctx, w: cw, h } = prepareCanvas(ca),
          lam = Math.pow(10, logLambdaAt(settings.li)),
          wr = AP.ridge(lam),
          wl = AP.lasso(lam),
          m = { l: 128, r: 14, t: 66, b: 30 },
          VR = [-0.7, 1.5],
          X = (v) => m.l + ((v - VR[0]) / (VR[1] - VR[0])) * (cw - m.l - m.r),
          rh = (h - m.t - m.b) / AP.p,
          bh = Math.min(5, rh / 5.2);
        ctx.font = MONO;
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        [-0.5, 0, 0.5, 1, 1.5].forEach((v) => {
          ctx.strokeStyle = v === 0 ? colors.muted : colors.line;
          ctx.beginPath();
          ctx.moveTo(X(v), m.t - 4);
          ctx.lineTo(X(v), h - m.b);
          ctx.stroke();
          ctx.fillText(String(v), X(v), h - m.b + 14);
        });
        ctx.fillText('coefficient', X(0.4), h - 4);
        const sets = [
          [AP.bt, colors.ink],
          [AP.ols, colors.muted],
          [wr, colors.accent],
          [wl, colors.ok],
        ];
        for (let j = 0; j < AP.p; j++) {
          const y0 = m.t + j * rh;
          if (j === 3) {
            ctx.setLineDash([3, 4]);
            ctx.strokeStyle = colors.muted;
            ctx.beginPath();
            ctx.moveTo(8, y0 - 1);
            ctx.lineTo(cw - m.r, y0 - 1);
            ctx.stroke();
            ctx.setLineDash([]);
          }
          ctx.font = j < 3 ? SANS : MONO;
          ctx.fillStyle = j < 3 ? colors.ink : colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText(AP.names[j], m.l - 10, y0 + rh / 2 + 4);
          sets.forEach(([wv, col], k) => {
            const v = clamp(wv[j], VR[0], VR[1]),
              yy = y0 + rh / 2 + (k - 1.5) * (bh + 1.5) - bh / 2;
            if (Math.abs(wv[j]) < 1e-7) {
              if (k === 3) {
                ctx.strokeStyle = col;
                ctx.lineWidth = 1.8;
                ctx.beginPath();
                ctx.arc(X(0), yy + bh / 2, 3.6, 0, 7);
                ctx.stroke();
              }
              return;
            }
            ctx.fillStyle = col;
            ctx.fillRect(Math.min(X(0), X(v)), yy, Math.abs(X(v) - X(0)), bh);
          });
        }
        const nz = wl.filter((v) => Math.abs(v) > 1e-7).length;
        ctx.font = MONO;
        ctx.textAlign = 'left';
        let ly = 12;
        [
          [colors.ink, 'truth: 3 matter, 9 are noise'],
          [colors.muted, 'least squares, no penalty      test error ' + f2(AP.eo)],
          [
            colors.accent,
            'ridge  λ = ' +
              (lam < 0.01 ? lam.toExponential(1) : lam.toFixed(2)) +
              ', keeps all 12       test error ' +
              f2(AP.mse(wr)),
          ],
          [
            colors.ok,
            'lasso  same λ, keeps ' + nz + ' of 12 (○ = 0)  test error ' + f2(AP.mse(wl)),
          ],
        ].forEach(([col, sx]) => {
          ctx.fillStyle = col;
          ctx.fillRect(8, ly - 7, 10, 7);
          ctx.fillText(sx, 24, ly);
          ly += 14;
        });
      }
      {
        const { g: ctx, w: cw, h } = prepareCanvas(cb);
        let mx = 0.2;
        for (let i = 14; i < NL; i++) for (const v of W[i]) mx = Math.max(mx, Math.abs(v));
        mx *= 1.15;
        const P = plot(ctx, cw, h, [0.3, D + 0.7], [-mx, mx], colors, { xlabel: 'coefficient' }),
          bw = (P.X(2) - P.X(1)) * 0.6;
        clipPlot(ctx, P);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(P.X(0.3), P.Y(0));
        ctx.lineTo(P.X(D + 0.7), P.Y(0));
        ctx.stroke();
        for (let j = 0; j < D; j++) {
          const x = P.X(j + 1),
            v0 = clamp(W[0][j], -mx * 1.5, mx * 1.5),
            v = w[j];
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1.2;
          ctx.strokeRect(x - bw / 2, Math.min(P.Y(0), P.Y(v0)), bw, Math.abs(P.Y(v0) - P.Y(0)));
          ctx.setLineDash([]);
          if (Math.abs(v) > 1e-6) {
            ctx.fillStyle = ramp(colors, 0.15 + (0.85 * j) / (D - 1));
            ctx.fillRect(x - bw / 2, Math.min(P.Y(0), P.Y(v)), bw, Math.abs(P.Y(v) - P.Y(0)));
          } else {
            ctx.strokeStyle = colors.ok;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            ctx.arc(x, P.Y(0), 6, 0, 7);
            ctx.stroke();
          }
        }
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.ok;
        ctx.fillText('○ exactly zero', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g: ctx, w: cw, h } = prepareCanvas(cg),
          sc = Math.min(cw, h) / 5.4,
          X = (v) => cw / 2 + v * sc,
          Y = (v) => h / 2 - v * sc;
        geometryViewport = { cw, h, sc };
        const lam = tlam(settings.li),
          ws = toy(lam),
          pen = (v) =>
            settings.a * (Math.abs(v[0]) + Math.abs(v[1])) +
            ((1 - settings.a) / 2) * (v[0] * v[0] + v[1] * v[1]),
          Pv = pen(ws);
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        for (let k = -2; k <= 2; k++) {
          ctx.beginPath();
          ctx.moveTo(X(k), 0);
          ctx.lineTo(X(k), h);
          ctx.moveTo(0, Y(k));
          ctx.lineTo(cw, Y(k));
          ctx.stroke();
        }
        ctx.strokeStyle = colors.muted;
        ctx.beginPath();
        ctx.moveTo(X(0), 0);
        ctx.lineTo(X(0), h);
        ctx.moveTo(0, Y(0));
        ctx.lineTo(cw, Y(0));
        ctx.stroke();
        if (Pv > 1e-9) {
          ctx.beginPath();
          for (let k = 0; k <= 180; k++) {
            const th = (k / 180) * 2 * Math.PI,
              cs = Math.cos(th),
              sn = Math.sin(th),
              a = (1 - settings.a) / 2,
              b = settings.a * (Math.abs(cs) + Math.abs(sn)),
              r = a > 1e-9 ? (-b + Math.sqrt(b * b + 4 * a * Pv)) / (2 * a) : Pv / b;
            k ? ctx.lineTo(X(r * cs), Y(r * sn)) : ctx.moveTo(X(r * cs), Y(r * sn));
          }
          ctx.closePath();
          ctx.fillStyle = mix(colors.gold, colors.gold, 0, 0.3);
          ctx.fill();
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 2.2;
          ctx.stroke();
        }
        const l22 = Math.sqrt(TA[2] - (TA[1] * TA[1]) / TA[0]),
          l21 = TA[1] / Math.sqrt(TA[0]),
          l11 = Math.sqrt(TA[0]),
          ell = (q) => {
            ctx.beginPath();
            for (let k = 0; k <= 90; k++) {
              const th = (k / 90) * 2 * Math.PI,
                u1 = Math.cos(th),
                u2 = Math.sin(th),
                v2 = u2 / l22,
                v1 = (u1 - l21 * v2) / l11,
                r = Math.sqrt(q);
              k
                ? ctx.lineTo(X(unpenalizedOptimum[0] + r * v1), Y(unpenalizedOptimum[1] + r * v2))
                : ctx.moveTo(X(unpenalizedOptimum[0] + r * v1), Y(unpenalizedOptimum[1] + r * v2));
            }
            ctx.closePath();
          };
        ctx.strokeStyle = mix(colors.accent, colors.accent, 0, 0.4);
        ctx.lineWidth = 1;
        [0.12, 0.45, 1, 1.8, 2.9, 4.3, 6].forEach((q) => {
          ell(q);
          ctx.stroke();
        });
        const d0 = ws[0] - unpenalizedOptimum[0],
          d1 = ws[1] - unpenalizedOptimum[1],
          qs = TA[0] * d0 * d0 + 2 * TA[1] * d0 * d1 + TA[2] * d1 * d1;
        if (qs > 1e-6) {
          ell(qs);
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 2.4;
          ctx.stroke();
        }
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        for (let i = 0; i < NL; i++) {
          const q = toy(tlam(i));
          i ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1]));
        }
        ctx.stroke();
        const tx = X(unpenalizedOptimum[0]),
          ty = Y(unpenalizedOptimum[1]);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(tx - 8, ty);
        ctx.lineTo(tx + 8, ty);
        ctx.moveTo(tx, ty - 8);
        ctx.lineTo(tx, ty + 8);
        ctx.stroke();
        ctx.fillStyle = colors.ok;
        ctx.strokeStyle = colors.bg;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(X(ws[0]), Y(ws[1]), 7.5, 0, 7);
        ctx.fill();
        ctx.stroke();
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('w₁', cw - 6, Y(0) - 6);
        ctx.textAlign = 'left';
        ctx.fillText('w₂', X(0) + 6, 12);
        let ly = h - 40;
        [
          [colors.ink, '+ best fit with no penalty'],
          [
            colors.ok,
            '● penalised answer: (' +
              f2(ws[0]) +
              ', ' +
              f2(ws[1]) +
              ')' +
              (Math.abs(ws[0]) < 1e-9 || Math.abs(ws[1]) < 1e-9 ? '  one is exactly 0' : ''),
          ],
          [
            colors.cut,
            'shaded = norm ball, ' +
              (settings.a < 0.01 ? 'L2 circle' : settings.a > 0.99 ? 'L1 diamond' : 'elastic net'),
          ],
        ].forEach(([col, sx]) => {
          ctx.fillStyle = col;
          ctx.fillText(sx, 8, ly);
          ly += 15;
        });
      }
      {
        const { g: ctx, w: cw, h } = prepareCanvas(mainCanvas),
          P = plot(ctx, cw, h, [-1.05, 1.05], [-2.2, 2.2], colors, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: 'x',
          });
        clipPlot(ctx, P);
        ctx.setLineDash([6, 5]);
        curve(ctx, P, Fq, -1, 1);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.6;
        ctx.stroke();
        ctx.setLineDash([]);
        curve(ctx, P, (x) => clamp(pred(x, W[0]), -9, 9), -1, 1, 300);
        ctx.strokeStyle = mix(colors.muted, colors.muted, 0, 0.45);
        ctx.lineWidth = 1.2;
        ctx.stroke();
        curve(ctx, P, (x) => clamp(pred(x, w), -9, 9), -1, 1, 300);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 3.2;
        ctx.stroke();
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 1.5;
        for (const p of pts)
          if (p.test) {
            ctx.beginPath();
            ctx.arc(P.X(p.x), P.Y(p.y), 4, 0, 7);
            ctx.stroke();
          }
        ctx.fillStyle = colors.accent;
        for (const p of pts)
          if (!p.test) {
            ctx.beginPath();
            ctx.arc(P.X(p.x), P.Y(p.y), 4.2, 0, 7);
            ctx.fill();
          }
        markTP(ctx, P.X(tp.v[0]), P.Y(clamp(pred(tp.v[0], w), -9, 9)), colors);
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.muted;
        ctx.fillText('thin grey = no penalty, dashed = truth', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g: ctx, w: cw, h } = prepareCanvas(cp);
        let mx = 0.2;
        for (let i = 14; i < NL; i++) for (const v of W[i]) mx = Math.max(mx, Math.abs(v));
        mx *= 1.15;
        const P = plot(ctx, cw, h, [logLambdaAt(0), logLambdaAt(NL - 1)], [-mx, mx], colors, {
          xlabel: 'log₁₀ λ',
        });
        clipPlot(ctx, P);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(P.X(logLambdaAt(0)), P.Y(0));
        ctx.lineTo(P.X(logLambdaAt(NL - 1)), P.Y(0));
        ctx.stroke();
        ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.5);
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.moveTo(P.X(logLambdaAt(settings.li)), P.Y(-mx));
        ctx.lineTo(P.X(logLambdaAt(settings.li)), P.Y(mx));
        ctx.stroke();
        for (let j = 0; j < D; j++) {
          ctx.strokeStyle = ramp(colors, 0.15 + (0.85 * j) / (D - 1));
          ctx.lineWidth = 2;
          ctx.beginPath();
          for (let i = 0; i < NL; i++)
            i
              ? ctx.lineTo(P.X(logLambdaAt(i)), P.Y(W[i][j]))
              : ctx.moveTo(P.X(logLambdaAt(i)), P.Y(W[i][j]));
          ctx.stroke();
          ctx.fillStyle = ctx.strokeStyle;
          ctx.beginPath();
          ctx.arc(P.X(logLambdaAt(settings.li)), P.Y(clamp(W[settings.li][j], -mx, mx)), 4, 0, 7);
          ctx.fill();
        }
        ctx.restore();
      }
      let bi = 0;
      for (let i = 1; i < NL; i++) if (ete[i] < ete[bi]) bi = i;
      const lam = Math.pow(10, logLambdaAt(settings.li));
      setReadout(root, 'l', lam < 0.01 ? lam.toExponential(1) : lam.toFixed(3));
      setReadout(root, 'tr', f3(etr[settings.li]));
      setReadout(root, 'te', isFinite(ete[settings.li]) ? f3(ete[settings.li]) : '–');
      setReadout(root, 'nz', String(w.filter((v) => Math.abs(v) > 1e-6).length));
      tp.out.textContent =
        'At x = ' +
        f2(tp.v[0]) +
        ' the penalised fit predicts y = ' +
        f2(pred(tp.v[0], w)) +
        '. With no penalty it would predict ' +
        f2(pred(tp.v[0], W[0])) +
        '. The true curve there is ' +
        f2(Fq(tp.v[0])) +
        '.';
      setReadout(root, 'nm', f2(Math.sqrt(dot(w, w))));
      setReadout(root, 'n1', f2(w.reduce((s, v) => s + Math.abs(v), 0)));
      const bl = Math.pow(10, logLambdaAt(bi));
      setReadout(root, 'bl', bl < 0.01 ? bl.toExponential(1) : bl.toFixed(3));
      root
        .querySelectorAll('#rg-kind button')
        .forEach((b) =>
          b.setAttribute(
            'aria-pressed',
            Math.abs(+b.dataset.a - settings.a) < 0.01 ? 'true' : 'false',
          ),
        );
    };
    const ls = bind(
        root,
        'rg-l',
        (v) => {
          settings.li = v;
          draw();
        },
        (v) => logLambdaAt(v).toFixed(1),
      ),
      as = bind(root, 'rg-a', (v) => {
        settings.a = v;
        solveAll();
        draw();
      });
    const anim = sweeper(ls.el, 0, NL - 1, 10);
    playback = playBtn($('#rg-play', root), anim);
    grab(ls.el, playback);
    $('#rg-kind', root).addEventListener('click', (e) => {
      const a = e.target.dataset && e.target.dataset.a;
      if (a == null) return;
      settings.a = +a;
      as.set(settings.a);
      solveAll();
      draw();
    });
    let gd = false;
    const gm = (e) => {
      if (!geometryViewport) return;
      const r = cg.getBoundingClientRect();
      unpenalizedOptimum[0] = clamp(
        (e.clientX - r.left - geometryViewport.cw / 2) / geometryViewport.sc,
        -2.4,
        2.4,
      );
      unpenalizedOptimum[1] = clamp(
        -(e.clientY - r.top - geometryViewport.h / 2) / geometryViewport.sc,
        -2.4,
        2.4,
      );
      draw();
    };
    cg.addEventListener('pointerdown', (e) => {
      gd = true;
      cg.setPointerCapture(e.pointerId);
      gm(e);
    });
    cg.addEventListener('pointermove', (e) => {
      if (gd) gm(e);
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      cg.addEventListener(n, () => {
        gd = false;
      }),
    );
    const re = () => {
      data();
      solveAll();
      draw();
    };
    bind(root, 'rg-s', (v) => {
      settings.noise = v;
      re();
    });
    bind(
      root,
      'rg-n',
      (v) => {
        settings.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#rg-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
