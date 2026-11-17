/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  MONO,
  SANS,
  T,
  bind,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  fit,
  grab,
  markTP,
  mix,
  playBtn,
  plot,
  ramp,
  randnFrom,
  rng,
  setR,
  sweeper,
  testPoint,
} from '../../../js/core/shared.js';
import { dot, legendre, lsq } from '../shared.js';

export default {
  init(root) {
    const c = $('#rg-c', root),
      cp = $('#rg-p', root),
      cb = $('#rg-b', root),
      cg = $('#rg-g', root),
      c1 = $('#rg-s1', root),
      ca = $('#rg-ap', root),
      TC2 = [1.7, 0.75],
      TA = [1, 0.55, 1.3],
      S = { li: 20, a: 0, noise: 0.3, n: 26 },
      Fq = (x) => Math.sin(2.8 * x),
      D = 10,
      NL = 60,
      LL = (i) => -4.2 + (5.4 * i) / (NL - 1);
    let seed = 12,
      pts = [],
      mu = [],
      sd = [],
      ym = 0,
      W = [],
      etr = [],
      ete = [],
      pb;
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
      for (let i = 0; i < S.n; i++) {
        const x = r() * 2 - 1;
        pts.push({ x, y: Fq(x) + S.noise * rn(), test: r() < 0.3 });
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
        const lam = Math.pow(10, LL(i)),
          res = yc.map((v, k) => v - dot(w, Xs[k]));
        for (let it = 0; it < 260; it++) {
          let mxd = 0;
          for (let j = 0; j < D; j++) {
            let rho = 0;
            for (let k = 0; k < n; k++) rho += Xs[k][j] * (res[k] + Xs[k][j] * w[j]);
            rho /= n;
            const th = lam * S.a,
              nw = (rho > th ? rho - th : rho < -th ? rho + th : 0) / (1 + lam * (1 - S.a)),
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
    let GV = null;
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
      let w = [TC2[0], TC2[1]];
      const sft = (r, th) => (r > th ? r - th : r < -th ? r + th : 0);
      for (let it = 0; it < 120; it++) {
        w[0] = sft(TA[0] * TC2[0] - TA[1] * (w[1] - TC2[1]), lam * S.a) / (TA[0] + lam * (1 - S.a));
        w[1] = sft(TA[2] * TC2[1] - TA[1] * (w[0] - TC2[0]), lam * S.a) / (TA[2] + lam * (1 - S.a));
      }
      return w;
    };
    const tlam = (i) => Math.pow(10, -2.2 + (3.3 * i) / (NL - 1));
    const draw = () => {
      const t = T(),
        w = W[S.li];
      {
        const { g, w: cw, h } = fit(c1),
          lam = Math.pow(10, -1.5 + (1.9 * S.li) / (NL - 1)),
          m = { l: 46, r: 12, t: 12, b: 40 },
          R = 3,
          X = (v) => m.l + ((v + R) / (2 * R)) * (cw - m.l - m.r),
          Y = (v) => h - m.b - ((v + R) / (2 * R)) * (h - m.t - m.b),
          sf = (v) => (v > lam ? v - lam : v < -lam ? v + lam : 0);
        g.fillStyle = mix(t.ok, t.ok, 0, 0.1);
        g.fillRect(
          X(-Math.min(lam, R)),
          m.t,
          X(Math.min(lam, R)) - X(-Math.min(lam, R)),
          h - m.t - m.b,
        );
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.font = MONO;
        g.fillStyle = t.muted;
        for (let v = -3; v <= 3; v++) {
          g.beginPath();
          g.moveTo(X(v), m.t);
          g.lineTo(X(v), h - m.b);
          g.moveTo(m.l, Y(v));
          g.lineTo(cw - m.r, Y(v));
          g.stroke();
          g.textAlign = 'center';
          g.fillText(String(v), X(v), h - m.b + 14);
          g.textAlign = 'right';
          g.fillText(String(v), m.l - 6, Y(v) + 4);
        }
        g.strokeStyle = t.muted;
        g.beginPath();
        g.moveTo(X(0), m.t);
        g.lineTo(X(0), h - m.b);
        g.moveTo(m.l, Y(0));
        g.lineTo(cw - m.r, Y(0));
        g.stroke();
        g.textAlign = 'center';
        g.fillText('coefficient with no penalty', cw / 2 + m.l / 2, h - 8);
        g.save();
        g.translate(12, h / 2);
        g.rotate(-Math.PI / 2);
        g.fillText('coefficient after the penalty', 0, 0);
        g.restore();
        g.save();
        g.beginPath();
        g.rect(m.l, m.t, cw - m.l - m.r, h - m.t - m.b);
        g.clip();
        g.setLineDash([5, 5]);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.5;
        g.beginPath();
        g.moveTo(X(-R), Y(-R));
        g.lineTo(X(R), Y(R));
        g.stroke();
        g.setLineDash([]);
        g.strokeStyle = t.accent;
        g.lineWidth = 3.2;
        g.beginPath();
        g.moveTo(X(-R), Y(-R / (1 + lam)));
        g.lineTo(X(R), Y(R / (1 + lam)));
        g.stroke();
        g.strokeStyle = t.ok;
        g.lineWidth = 3.2;
        g.beginPath();
        g.moveTo(X(-R), Y(sf(-R)));
        g.lineTo(X(-Math.min(lam, R)), Y(0));
        g.lineTo(X(Math.min(lam, R)), Y(0));
        g.lineTo(X(R), Y(sf(R)));
        g.stroke();
        [-2.4, -1.5, -0.6, 0.4, 1.1, 2.1].forEach((v) => {
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(X(v), Y(v));
          g.lineTo(X(v), Y(Math.abs(sf(v)) < Math.abs(v / (1 + lam)) ? sf(v) : v / (1 + lam)));
          g.stroke();
          g.fillStyle = t.muted;
          g.beginPath();
          g.arc(X(v), Y(v), 3.5, 0, 7);
          g.fill();
          g.fillStyle = t.accent;
          g.beginPath();
          g.arc(X(v), Y(v / (1 + lam)), 5, 0, 7);
          g.fill();
          g.fillStyle = t.ok;
          g.beginPath();
          g.arc(X(v), Y(sf(v)), 5, 0, 7);
          g.fill();
        });
        g.restore();
        g.font = SANS;
        g.textAlign = 'left';
        let ly = m.t + 16;
        [
          [t.muted, 'No penalty: unchanged'],
          [t.accent, 'Ridge: every one scaled down, none reach zero'],
          [t.ok, 'Lasso: every one moved toward zero by λ;'],
          [t.ok, 'anything inside the shaded band becomes exactly 0'],
        ].forEach(([col, sx]) => {
          g.fillStyle = col;
          g.fillText(sx, m.l + 10, ly);
          ly += 17;
        });
      }
      {
        const { g, w: cw, h } = fit(ca),
          lam = Math.pow(10, LL(S.li)),
          wr = AP.ridge(lam),
          wl = AP.lasso(lam),
          m = { l: 128, r: 14, t: 66, b: 30 },
          VR = [-0.7, 1.5],
          X = (v) => m.l + ((v - VR[0]) / (VR[1] - VR[0])) * (cw - m.l - m.r),
          rh = (h - m.t - m.b) / AP.p,
          bh = Math.min(5, rh / 5.2);
        g.font = MONO;
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        [-0.5, 0, 0.5, 1, 1.5].forEach((v) => {
          g.strokeStyle = v === 0 ? t.muted : t.line;
          g.beginPath();
          g.moveTo(X(v), m.t - 4);
          g.lineTo(X(v), h - m.b);
          g.stroke();
          g.fillText(String(v), X(v), h - m.b + 14);
        });
        g.fillText('coefficient', X(0.4), h - 4);
        const sets = [
          [AP.bt, t.ink],
          [AP.ols, t.muted],
          [wr, t.accent],
          [wl, t.ok],
        ];
        for (let j = 0; j < AP.p; j++) {
          const y0 = m.t + j * rh;
          if (j === 3) {
            g.setLineDash([3, 4]);
            g.strokeStyle = t.muted;
            g.beginPath();
            g.moveTo(8, y0 - 1);
            g.lineTo(cw - m.r, y0 - 1);
            g.stroke();
            g.setLineDash([]);
          }
          g.font = j < 3 ? SANS : MONO;
          g.fillStyle = j < 3 ? t.ink : t.muted;
          g.textAlign = 'right';
          g.fillText(AP.names[j], m.l - 10, y0 + rh / 2 + 4);
          sets.forEach(([wv, col], k) => {
            const v = clamp(wv[j], VR[0], VR[1]),
              yy = y0 + rh / 2 + (k - 1.5) * (bh + 1.5) - bh / 2;
            if (Math.abs(wv[j]) < 1e-7) {
              if (k === 3) {
                g.strokeStyle = col;
                g.lineWidth = 1.8;
                g.beginPath();
                g.arc(X(0), yy + bh / 2, 3.6, 0, 7);
                g.stroke();
              }
              return;
            }
            g.fillStyle = col;
            g.fillRect(Math.min(X(0), X(v)), yy, Math.abs(X(v) - X(0)), bh);
          });
        }
        const nz = wl.filter((v) => Math.abs(v) > 1e-7).length;
        g.font = MONO;
        g.textAlign = 'left';
        let ly = 12;
        [
          [t.ink, 'truth: 3 matter, 9 are noise'],
          [t.muted, 'least squares, no penalty      test error ' + f2(AP.eo)],
          [
            t.accent,
            'ridge  λ = ' +
              (lam < 0.01 ? lam.toExponential(1) : lam.toFixed(2)) +
              ', keeps all 12       test error ' +
              f2(AP.mse(wr)),
          ],
          [t.ok, 'lasso  same λ, keeps ' + nz + ' of 12 (○ = 0)  test error ' + f2(AP.mse(wl))],
        ].forEach(([col, sx]) => {
          g.fillStyle = col;
          g.fillRect(8, ly - 7, 10, 7);
          g.fillText(sx, 24, ly);
          ly += 14;
        });
      }
      {
        const { g, w: cw, h } = fit(cb);
        let mx = 0.2;
        for (let i = 14; i < NL; i++) for (const v of W[i]) mx = Math.max(mx, Math.abs(v));
        mx *= 1.15;
        const P = plot(g, cw, h, [0.3, D + 0.7], [-mx, mx], t, { xlabel: 'coefficient' }),
          bw = (P.X(2) - P.X(1)) * 0.6;
        clipPlot(g, P);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(P.X(0.3), P.Y(0));
        g.lineTo(P.X(D + 0.7), P.Y(0));
        g.stroke();
        for (let j = 0; j < D; j++) {
          const x = P.X(j + 1),
            v0 = clamp(W[0][j], -mx * 1.5, mx * 1.5),
            v = w[j];
          g.setLineDash([3, 3]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.2;
          g.strokeRect(x - bw / 2, Math.min(P.Y(0), P.Y(v0)), bw, Math.abs(P.Y(v0) - P.Y(0)));
          g.setLineDash([]);
          if (Math.abs(v) > 1e-6) {
            g.fillStyle = ramp(t, 0.15 + (0.85 * j) / (D - 1));
            g.fillRect(x - bw / 2, Math.min(P.Y(0), P.Y(v)), bw, Math.abs(P.Y(v) - P.Y(0)));
          } else {
            g.strokeStyle = t.ok;
            g.lineWidth = 2.4;
            g.beginPath();
            g.arc(x, P.Y(0), 6, 0, 7);
            g.stroke();
          }
        }
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.ok;
        g.fillText('○ exactly zero', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g, w: cw, h } = fit(cg),
          sc = Math.min(cw, h) / 5.4,
          X = (v) => cw / 2 + v * sc,
          Y = (v) => h / 2 - v * sc;
        GV = { cw, h, sc };
        const lam = tlam(S.li),
          ws = toy(lam),
          pen = (v) =>
            S.a * (Math.abs(v[0]) + Math.abs(v[1])) + ((1 - S.a) / 2) * (v[0] * v[0] + v[1] * v[1]),
          Pv = pen(ws);
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        for (let k = -2; k <= 2; k++) {
          g.beginPath();
          g.moveTo(X(k), 0);
          g.lineTo(X(k), h);
          g.moveTo(0, Y(k));
          g.lineTo(cw, Y(k));
          g.stroke();
        }
        g.strokeStyle = t.muted;
        g.beginPath();
        g.moveTo(X(0), 0);
        g.lineTo(X(0), h);
        g.moveTo(0, Y(0));
        g.lineTo(cw, Y(0));
        g.stroke();
        if (Pv > 1e-9) {
          g.beginPath();
          for (let k = 0; k <= 180; k++) {
            const th = (k / 180) * 2 * Math.PI,
              cs = Math.cos(th),
              sn = Math.sin(th),
              a = (1 - S.a) / 2,
              b = S.a * (Math.abs(cs) + Math.abs(sn)),
              r = a > 1e-9 ? (-b + Math.sqrt(b * b + 4 * a * Pv)) / (2 * a) : Pv / b;
            k ? g.lineTo(X(r * cs), Y(r * sn)) : g.moveTo(X(r * cs), Y(r * sn));
          }
          g.closePath();
          g.fillStyle = mix(t.gold, t.gold, 0, 0.3);
          g.fill();
          g.strokeStyle = t.gold;
          g.lineWidth = 2.2;
          g.stroke();
        }
        const l22 = Math.sqrt(TA[2] - (TA[1] * TA[1]) / TA[0]),
          l21 = TA[1] / Math.sqrt(TA[0]),
          l11 = Math.sqrt(TA[0]),
          ell = (q) => {
            g.beginPath();
            for (let k = 0; k <= 90; k++) {
              const th = (k / 90) * 2 * Math.PI,
                u1 = Math.cos(th),
                u2 = Math.sin(th),
                v2 = u2 / l22,
                v1 = (u1 - l21 * v2) / l11,
                r = Math.sqrt(q);
              k
                ? g.lineTo(X(TC2[0] + r * v1), Y(TC2[1] + r * v2))
                : g.moveTo(X(TC2[0] + r * v1), Y(TC2[1] + r * v2));
            }
            g.closePath();
          };
        g.strokeStyle = mix(t.accent, t.accent, 0, 0.4);
        g.lineWidth = 1;
        [0.12, 0.45, 1, 1.8, 2.9, 4.3, 6].forEach((q) => {
          ell(q);
          g.stroke();
        });
        const d0 = ws[0] - TC2[0],
          d1 = ws[1] - TC2[1],
          qs = TA[0] * d0 * d0 + 2 * TA[1] * d0 * d1 + TA[2] * d1 * d1;
        if (qs > 1e-6) {
          ell(qs);
          g.strokeStyle = t.accent;
          g.lineWidth = 2.4;
          g.stroke();
        }
        g.strokeStyle = t.ink;
        g.lineWidth = 1.3;
        g.beginPath();
        for (let i = 0; i < NL; i++) {
          const q = toy(tlam(i));
          i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1]));
        }
        g.stroke();
        const tx = X(TC2[0]),
          ty = Y(TC2[1]);
        g.strokeStyle = t.ink;
        g.lineWidth = 2.4;
        g.beginPath();
        g.moveTo(tx - 8, ty);
        g.lineTo(tx + 8, ty);
        g.moveTo(tx, ty - 8);
        g.lineTo(tx, ty + 8);
        g.stroke();
        g.fillStyle = t.ok;
        g.strokeStyle = t.bg;
        g.lineWidth = 2;
        g.beginPath();
        g.arc(X(ws[0]), Y(ws[1]), 7.5, 0, 7);
        g.fill();
        g.stroke();
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('w₁', cw - 6, Y(0) - 6);
        g.textAlign = 'left';
        g.fillText('w₂', X(0) + 6, 12);
        let ly = h - 40;
        [
          [t.ink, '+ best fit with no penalty'],
          [
            t.ok,
            '● penalised answer: (' +
              f2(ws[0]) +
              ', ' +
              f2(ws[1]) +
              ')' +
              (Math.abs(ws[0]) < 1e-9 || Math.abs(ws[1]) < 1e-9 ? '  one is exactly 0' : ''),
          ],
          [
            t.cut,
            'shaded = norm ball, ' +
              (S.a < 0.01 ? 'L2 circle' : S.a > 0.99 ? 'L1 diamond' : 'elastic net'),
          ],
        ].forEach(([col, sx]) => {
          g.fillStyle = col;
          g.fillText(sx, 8, ly);
          ly += 15;
        });
      }
      {
        const { g, w: cw, h } = fit(c),
          P = plot(g, cw, h, [-1.05, 1.05], [-2.2, 2.2], t, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: 'x',
          });
        clipPlot(g, P);
        g.setLineDash([6, 5]);
        curve(g, P, Fq, -1, 1);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.6;
        g.stroke();
        g.setLineDash([]);
        curve(g, P, (x) => clamp(pred(x, W[0]), -9, 9), -1, 1, 300);
        g.strokeStyle = mix(t.muted, t.muted, 0, 0.45);
        g.lineWidth = 1.2;
        g.stroke();
        curve(g, P, (x) => clamp(pred(x, w), -9, 9), -1, 1, 300);
        g.strokeStyle = t.gold;
        g.lineWidth = 3.2;
        g.stroke();
        g.strokeStyle = t.ok;
        g.lineWidth = 1.5;
        for (const p of pts)
          if (p.test) {
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 4, 0, 7);
            g.stroke();
          }
        g.fillStyle = t.accent;
        for (const p of pts)
          if (!p.test) {
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 4.2, 0, 7);
            g.fill();
          }
        markTP(g, P.X(tp.v[0]), P.Y(clamp(pred(tp.v[0], w), -9, 9)), t);
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('thin grey = no penalty, dashed = truth', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g, w: cw, h } = fit(cp);
        let mx = 0.2;
        for (let i = 14; i < NL; i++) for (const v of W[i]) mx = Math.max(mx, Math.abs(v));
        mx *= 1.15;
        const P = plot(g, cw, h, [LL(0), LL(NL - 1)], [-mx, mx], t, { xlabel: 'log₁₀ λ' });
        clipPlot(g, P);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(P.X(LL(0)), P.Y(0));
        g.lineTo(P.X(LL(NL - 1)), P.Y(0));
        g.stroke();
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
        g.lineWidth = 8;
        g.beginPath();
        g.moveTo(P.X(LL(S.li)), P.Y(-mx));
        g.lineTo(P.X(LL(S.li)), P.Y(mx));
        g.stroke();
        for (let j = 0; j < D; j++) {
          g.strokeStyle = ramp(t, 0.15 + (0.85 * j) / (D - 1));
          g.lineWidth = 2;
          g.beginPath();
          for (let i = 0; i < NL; i++)
            i ? g.lineTo(P.X(LL(i)), P.Y(W[i][j])) : g.moveTo(P.X(LL(i)), P.Y(W[i][j]));
          g.stroke();
          g.fillStyle = g.strokeStyle;
          g.beginPath();
          g.arc(P.X(LL(S.li)), P.Y(clamp(W[S.li][j], -mx, mx)), 4, 0, 7);
          g.fill();
        }
        g.restore();
      }
      let bi = 0;
      for (let i = 1; i < NL; i++) if (ete[i] < ete[bi]) bi = i;
      const lam = Math.pow(10, LL(S.li));
      setR(root, 'l', lam < 0.01 ? lam.toExponential(1) : lam.toFixed(3));
      setR(root, 'tr', f3(etr[S.li]));
      setR(root, 'te', isFinite(ete[S.li]) ? f3(ete[S.li]) : '–');
      setR(root, 'nz', String(w.filter((v) => Math.abs(v) > 1e-6).length));
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
      setR(root, 'nm', f2(Math.sqrt(dot(w, w))));
      setR(root, 'n1', f2(w.reduce((s, v) => s + Math.abs(v), 0)));
      const bl = Math.pow(10, LL(bi));
      setR(root, 'bl', bl < 0.01 ? bl.toExponential(1) : bl.toFixed(3));
      root
        .querySelectorAll('#rg-kind button')
        .forEach((b) =>
          b.setAttribute('aria-pressed', Math.abs(+b.dataset.a - S.a) < 0.01 ? 'true' : 'false'),
        );
    };
    const ls = bind(
        root,
        'rg-l',
        (v) => {
          S.li = v;
          draw();
        },
        (v) => LL(v).toFixed(1),
      ),
      as = bind(root, 'rg-a', (v) => {
        S.a = v;
        solveAll();
        draw();
      });
    const anim = sweeper(ls.el, 0, NL - 1, 10);
    pb = playBtn($('#rg-play', root), anim);
    grab(ls.el, pb);
    $('#rg-kind', root).addEventListener('click', (e) => {
      const a = e.target.dataset && e.target.dataset.a;
      if (a == null) return;
      S.a = +a;
      as.set(S.a);
      solveAll();
      draw();
    });
    let gd = false;
    const gm = (e) => {
      if (!GV) return;
      const r = cg.getBoundingClientRect();
      TC2[0] = clamp((e.clientX - r.left - GV.cw / 2) / GV.sc, -2.4, 2.4);
      TC2[1] = clamp(-(e.clientY - r.top - GV.h / 2) / GV.sc, -2.4, 2.4);
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
      S.noise = v;
      re();
    });
    bind(
      root,
      'rg-n',
      (v) => {
        S.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#rg-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
