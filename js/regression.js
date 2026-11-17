/* Manifold — Regression.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ regression: shared numerics ============ */
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
const tp2 = (t, df) => betai(df / 2, 0.5, df / (df + t * t));
function tcrit(df) {
  let lo = 1,
    hi = 60;
  for (let i = 0; i < 50; i++) {
    const m = (lo + hi) / 2;
    tp2(m, df) > 0.05 ? (lo = m) : (hi = m);
  }
  return (lo + hi) / 2;
}
const fpval = (F, d1, d2) => (F <= 0 ? 1 : betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)));
const pfmt = (p) => (p < 1e-4 ? '< 0.0001' : p.toFixed(4));
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

const REG = [
  {
    id: 'simple',
    tab: 'Simple linear regression',
    title: 'A line through a cloud',
    lede: 'Every possible line has a score: the average squared miss. Gradient descent walks downhill on that score until the line fits. Drag any point, or click the loss map to start from a different line.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The data and the current line (drag points, click to add)</figcaption><canvas class="short" id="sr-c" style="touch-action:none;cursor:crosshair" role="img" aria-label="Scatter plot with a fitted line and residuals"></canvas></figure>
 <figure class="fig"><figcaption>The loss for every possible line (click to start there)</figcaption><canvas class="short" id="sr-l" style="cursor:crosshair" role="img" aria-label="Map of the loss over slope and intercept with the gradient descent path"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="sr-play" type="button">Run gradient descent</button><button class="btn" id="sr-step" type="button">One step</button><button class="btn" id="sr-jump" type="button">Jump to the best fit</button></div>${stopBox('sr')}${slider('sr-lr', 'Learning rate', 0.01, 0.45, 0.01, 0.08)}<label class="check" for="sr-res"><input type="checkbox" id="sr-res" checked>Show the misses (residuals)</label><label class="check" for="sr-sq"><input type="checkbox" id="sr-sq">Show them as squares</label><p class="status" id="sr-note" role="status"></p>`)}
 ${grp('The true relationship', slider('sr-m', 'True slope', -2, 2, 0.05, 0.8) + slider('sr-b', 'True intercept', -2, 2, 0.05, 0.5) + slider('sr-s', 'Noise σ', 0, 3, 0.05, 1))}
 ${grp('The data', slider('sr-n', 'Number of points', 8, 120, 1, 40) + `<div class="row"><button class="btn" id="sr-new" type="button">New random sample</button><label class="btn file">Upload a CSV<input type="file" id="sr-file" accept=".csv,.txt,text/csv"></label></div><p class="status" id="sr-status" role="status"></p>`)}</div>
 <dl class="read">${rd('k', 'Steps taken')}${rd('cur', 'Current line: intercept, slope', 'hl')}${rd('tr', 'Train error (MSE)', 'hl3')}${rd('te', 'Test error (MSE)', 'hl2')}</dl>
 <dl class="read">${rd('best', 'Best fit: intercept, slope')}${rd('r2', 'R²')}${rd('se', 'Standard error of the slope')}${rd('t', 't statistic')}${rd('p', 'p-value (slope = 0?)', 'hl2')}${rd('ci', '95% interval for the slope', 'hl')}</dl>`,
    tryit:
      'Raise the learning rate to 0.4 and run again: what happens to the path on the loss map? Then set the noise to 3 and the true slope to 0.2, and press “New random sample” several times while watching the p-value and the interval.',
    math: String.raw`<h4>Intuition</h4><p>A line is two numbers, an intercept and a slope. For any choice of the two, measure how far each point is above or below the line, square those misses and average them. That average is a single number, the loss, and it is small only when the line passes close to the points. The right-hand picture shows the loss for every line at once: it is a bowl, and fitting means finding its lowest point.</p>
 <h4>The model and the loss</h4>$$\hat y_i=\beta_0+\beta_1x_i,\qquad L(\beta_0,\beta_1)=\frac1n\sum_{i=1}^n\big(y_i-\beta_0-\beta_1x_i\big)^2$$<p>Squaring makes the loss smooth, punishes large misses much more than small ones, and gives a bowl with exactly one minimum. The contours of the bowl are ellipses, the same shape as the Gaussian contours, and that is no accident: least squares is maximum likelihood when the noise is Gaussian.</p>
 <h4>Gradient descent</h4>$$\frac{\partial L}{\partial\beta_0}=-\frac2n\sum_i r_i,\qquad \frac{\partial L}{\partial\beta_1}=-\frac2n\sum_i r_i\,x_i,\qquad \boldsymbol\beta\leftarrow\boldsymbol\beta-\eta\,\nabla L$$<p>where \(r_i=y_i-\hat y_i\) is the residual. The gradient points uphill, so each step goes the other way, with length set by the learning rate \(\eta\). Too small and it crawls. Too large and each step overshoots the valley and the loss grows: the path zig-zags and flies off the map.</p>
 <h4>The exact answer</h4><p>Setting both derivatives to zero gives the minimum in closed form:</p>$$\hat\beta_1=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2}=\rho\,\frac{\sigma_y}{\sigma_x},\qquad \hat\beta_0=\bar y-\hat\beta_1\bar x.$$<p>The slope is the one you met in the Gaussian module when slicing the hill.</p>
 <h4>How good is the fit</h4>$$R^2=1-\frac{\sum_i r_i^2}{\sum_i(y_i-\bar y)^2}$$<p>the share of the variance of \(y\) that the line explains. Test error is the same loss measured on points the fit never saw.</p>
 <h4>Is the slope real?</h4><p>A different sample would give a different slope. Its standard error estimates how much it would wobble:</p>$$\operatorname{SE}(\hat\beta_1)=\sqrt{\frac{s^2}{\sum_i(x_i-\bar x)^2}},\qquad s^2=\frac{\sum_i r_i^2}{n-2},\qquad t=\frac{\hat\beta_1}{\operatorname{SE}(\hat\beta_1)}\sim t_{n-2}\ \text{ if the true slope is }0.$$<p>The p-value is the chance of a \(|t|\) this large when there is no relationship at all. The 95% interval is \(\hat\beta_1\pm t^{*}_{n-2}\operatorname{SE}\). More points, less noise, or a wider spread of \(x\) all shrink it.</p>
 <p>An uploaded CSV uses its first two numeric columns, rescaled to fit the plot. \(R^2\), \(t\) and the p-value are unaffected by the rescaling.</p>`,
    q: 'The p-value is 0.30. Does that prove x has no effect on y?',
    a: 'No. It means this sample cannot tell a slope of that size apart from zero. The interval for the slope will be wide and include both zero and values far from it. More data or less noise might reveal an effect.',
    init(root) {
      const c = $('#sr-c', root),
        cl = $('#sr-l', root),
        st = $('#sr-status', root),
        note = $('#sr-note', root);
      const S = { m: 0.8, b: 0.5, noise: 1, n: 40, lr: 0.08, res: true, sq: false },
        XR = [-3.5, 3.5],
        YR = [-7, 7],
        BR = [-3, 3],
        AR = [-5, 5];
      let seed = 3,
        pts = [],
        b0 = -3.5,
        b1 = -2,
        path = [],
        wait = 0,
        fr = 0,
        pb,
        PS = null,
        LM = null;
      const gen = () => {
        const r = rng(seed),
          rn = randnFrom(r);
        pts = [];
        for (let i = 0; i < S.n; i++) {
          const x = r() * 6 - 3;
          pts.push({ x, y: S.b + S.m * x + S.noise * rn(), test: r() < 0.3 });
        }
        if (pts.filter((p) => !p.test).length < 3)
          pts.forEach((p, i) => {
            if (i < 3) p.test = false;
          });
      };
      const mom = (P) => {
        let n = P.length,
          sx = 0,
          sy = 0,
          sxx = 0,
          sxy = 0,
          syy = 0;
        for (const p of P) {
          sx += p.x;
          sy += p.y;
          sxx += p.x * p.x;
          sxy += p.x * p.y;
          syy += p.y * p.y;
        }
        n = Math.max(1, n);
        return { n: P.length, mx: sx / n, my: sy / n, mxx: sxx / n, mxy: sxy / n, myy: syy / n };
      };
      const mse = (M, a, b) =>
        a * a + 2 * a * b * M.mx + b * b * M.mxx - 2 * a * M.my - 2 * b * M.mxy + M.myy;
      const ols = (M) => {
        const v = M.mxx - M.mx * M.mx,
          s = v > 1e-12 ? (M.mxy - M.mx * M.my) / v : 0;
        return [M.my - s * M.mx, s];
      };
      const begin = (a, b) => {
        b0 = a;
        b1 = b;
        path = [[a, b]];
        wait = 0;
        note.textContent = '';
      };
      const step = () => {
        const M = mom(pts.filter((p) => !p.test)),
          g0 = 2 * (b0 + b1 * M.mx - M.my),
          g1 = 2 * (b0 * M.mx + b1 * M.mxx - M.mxy);
        b0 -= S.lr * g0;
        b1 -= S.lr * g1;
        if (!isFinite(b0) || Math.abs(b0) > 40 || Math.abs(b1) > 40) {
          note.textContent =
            'The learning rate is too large: every step overshoots the valley and the loss grows. Lower it and run again.';
          b0 = clamp(b0 || 0, -40, 40);
          b1 = clamp(b1 || 0, -40, 40);
          path.push([b0, b1]);
          return 'div';
        }
        path.push([b0, b1]);
        if (path.length > 600) path.shift();
        return Math.hypot(g0, g1) < 2e-3 ? 'done' : 'go';
      };
      gen();
      begin(b0, b1);
      const tp = testPoint(root, 'sr', ['Your x'], [1.5], [[-3.5, 3.5]]);
      tp.on = () => draw();
      const draw = () => {
        const t = T(),
          trn = pts.filter((p) => !p.test),
          tst = pts.filter((p) => p.test),
          M = mom(trn),
          Mt = mom(tst),
          best = ols(M);
        {
          const { g, w, h } = fit(c),
            P = plot(g, w, h, XR, YR, t, { yticks: [-6, -3, 0, 3, 6], xlabel: 'x' });
          PS = P;
          clipPlot(g, P);
          if (S.res)
            for (const p of trn) {
              const yh = b0 + b1 * p.x;
              if (S.sq) {
                const d = Math.abs(P.Y(yh) - P.Y(p.y));
                g.fillStyle = mix(t.ok, t.ok, 0, 0.13);
                g.fillRect(P.X(p.x), Math.min(P.Y(yh), P.Y(p.y)), d, d);
              }
              g.strokeStyle = mix(t.ok, t.ok, 0, 0.75);
              g.lineWidth = 1.3;
              g.beginPath();
              g.moveTo(P.X(p.x), P.Y(p.y));
              g.lineTo(P.X(p.x), P.Y(yh));
              g.stroke();
            }
          g.strokeStyle = t.gold;
          g.lineWidth = 3.2;
          g.beginPath();
          g.moveTo(P.X(XR[0]), P.Y(b0 + b1 * XR[0]));
          g.lineTo(P.X(XR[1]), P.Y(b0 + b1 * XR[1]));
          g.stroke();
          g.strokeStyle = t.muted;
          g.lineWidth = 1.3;
          for (const p of tst) {
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 4, 0, 7);
            g.stroke();
          }
          g.fillStyle = t.accent;
          for (const p of trn) {
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 4.2, 0, 7);
            g.fill();
          }
          markTP(g, P.X(tp.v[0]), P.Y(b0 + b1 * tp.v[0]), t);
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.muted;
          g.fillText('filled = train, hollow = test', P.m.l + 8, P.m.t + 8);
        }
        {
          const { g, w, h } = fit(cl),
            m = { l: 44, r: 10, t: 10, b: 30 },
            X = (v) => m.l + ((v - BR[0]) / (BR[1] - BR[0])) * (w - m.l - m.r),
            Y = (v) => h - m.b - ((v - AR[0]) / (AR[1] - AR[0])) * (h - m.t - m.b);
          LM = { m, w, h };
          const lmin = mse(M, best[0], best[1]),
            lmax = Math.max(
              mse(M, AR[0], BR[0]),
              mse(M, AR[0], BR[1]),
              mse(M, AR[1], BR[0]),
              mse(M, AR[1], BR[1]),
            ),
            NC = 44,
            cw = (w - m.l - m.r) / NC,
            ch = (h - m.t - m.b) / NC;
          for (let i = 0; i < NC; i++)
            for (let j = 0; j < NC; j++) {
              const s = BR[0] + ((i + 0.5) / NC) * (BR[1] - BR[0]),
                a = AR[0] + ((j + 0.5) / NC) * (AR[1] - AR[0]),
                v = clamp((mse(M, a, s) - lmin) / (lmax - lmin || 1), 0, 1);
              g.fillStyle = ramp(t, Math.pow(1 - Math.sqrt(v), 1.6));
              g.fillRect(m.l + i * cw - 0.3, h - m.b - (j + 1) * ch - 0.3, cw + 0.6, ch + 0.6);
            }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          for (let v = -3; v <= 3; v++) g.fillText(String(v), X(v), h - m.b + 14);
          g.fillText('slope', w / 2 + m.l / 2, h - 3);
          g.textAlign = 'right';
          for (let v = -4; v <= 4; v += 2) g.fillText(String(v), m.l - 6, Y(v) + 4);
          g.save();
          g.translate(11, h / 2);
          g.rotate(-Math.PI / 2);
          g.textAlign = 'center';
          g.fillText('intercept', 0, 0);
          g.restore();
          g.save();
          g.beginPath();
          g.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
          g.clip();
          g.strokeStyle = t.ink;
          g.lineWidth = 1.6;
          g.beginPath();
          path.forEach((q, i) => (i ? g.lineTo(X(q[1]), Y(q[0])) : g.moveTo(X(q[1]), Y(q[0]))));
          g.stroke();
          g.fillStyle = t.ink;
          for (const q of path.slice(-60)) {
            g.beginPath();
            g.arc(X(q[1]), Y(q[0]), 1.8, 0, 7);
            g.fill();
          }
          const bx = X(best[1]),
            by = Y(best[0]);
          g.strokeStyle = t.ink;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(bx - 7, by);
          g.lineTo(bx + 7, by);
          g.moveTo(bx, by - 7);
          g.lineTo(bx, by + 7);
          g.stroke();
          g.fillStyle = t.ok;
          g.strokeStyle = t.bg;
          g.lineWidth = 2;
          g.beginPath();
          g.arc(X(b1), Y(b0), 7, 0, 7);
          g.fill();
          g.stroke();
          g.restore();
          g.fillStyle = t.ink;
          g.textAlign = 'left';
          g.fillText(
            '+ lowest loss',
            clamp(bx + 10, m.l + 4, w - 110),
            clamp(by - 9, m.t + 12, h - m.b - 6),
          );
        }
        const n = M.n,
          sxx = n * (M.mxx - M.mx * M.mx),
          rss = n * mse(M, best[0], best[1]),
          tss = n * (M.myy - M.my * M.my);
        setR(root, 'k', String(path.length - 1));
        setR(root, 'cur', f2(b0) + ', ' + f2(b1));
        setR(root, 'tr', f3(mse(M, b0, b1)));
        setR(root, 'te', Mt.n ? f3(mse(Mt, b0, b1)) : '–');
        setR(root, 'best', f2(best[0]) + ', ' + f2(best[1]));
        setR(root, 'r2', tss > 1e-12 ? f3(1 - rss / tss) : '–');
        if (n > 2 && sxx > 1e-9) {
          const se = Math.sqrt(rss / (n - 2) / sxx),
            tv = se > 1e-12 ? best[1] / se : Infinity,
            tc = tcrit(n - 2);
          setR(root, 'se', f3(se));
          setR(root, 't', isFinite(tv) ? f2(tv) : '∞');
          setR(root, 'p', isFinite(tv) ? pfmt(tp2(tv, n - 2)) : '< 0.0001');
          setR(root, 'ci', '[' + f2(best[1] - tc * se) + ', ' + f2(best[1] + tc * se) + ']');
          const xq = tp.v[0],
            yq = best[0] + best[1] * xq,
            hw = tc * Math.sqrt((rss / (n - 2)) * (1 + 1 / n + (xq - M.mx) ** 2 / sxx));
          tp.out.textContent =
            'At x = ' +
            f2(xq) +
            ' the current line predicts y = ' +
            f2(b0 + b1 * xq) +
            '. The best-fit line predicts ' +
            f2(yq) +
            ', and a new observation there should fall between ' +
            f2(yq - hw) +
            ' and ' +
            f2(yq + hw) +
            ' (95% prediction interval).';
        } else ['se', 't', 'p', 'ci'].forEach((k) => setR(root, k, '–'));
      };
      let fin = false,
        nst = 0,
        chk = () => false;
      const anim = Anim(() => {
        fr++;
        if (fr % 3) return;
        const r = step();
        nst++;
        if (r === 'done') {
          fin = true;
          draw();
          pb.stop();
          note.textContent =
            'Converged after ' +
            nst +
            ' steps: the slope of the loss is essentially zero, so the line has stopped moving.';
          return;
        }
        if (r === 'div') {
          fin = true;
          draw();
          pb.stop();
          return;
        }
        draw();
        chk();
      });
      pb = playBtn($('#sr-play', root), anim);
      chk = runCtl(pb, root, 'sr', {
        done: () => fin,
        restart: () => {
          fin = false;
          nst = 0;
          note.textContent = '';
          begin(Math.random() * 8 - 4, Math.random() * 5 - 2.5);
        },
        count: () => nst,
      });
      $('#sr-play', root).addEventListener('click', () => {
        if (anim.on && (Math.abs(b0) >= 40 || Math.abs(b1) >= 40)) begin(-3.5, -2);
      });
      $('#sr-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#sr-jump', root).onclick = () => {
        pb.stop();
        const o = ols(mom(pts.filter((p) => !p.test)));
        b0 = o[0];
        b1 = o[1];
        path.push([b0, b1]);
        note.textContent = '';
        draw();
      };
      bind(root, 'sr-lr', (v) => {
        S.lr = v;
      });
      const regen = () => {
        st.textContent = '';
        gen();
        draw();
      };
      bind(root, 'sr-m', (v) => {
        S.m = v;
        regen();
      });
      bind(root, 'sr-b', (v) => {
        S.b = v;
        regen();
      });
      bind(root, 'sr-s', (v) => {
        S.noise = v;
        regen();
      });
      bind(
        root,
        'sr-n',
        (v) => {
          S.n = v;
          regen();
        },
        (v) => String(v),
      );
      $('#sr-new', root).onclick = () => {
        seed++;
        regen();
      };
      $('#sr-res', root).onchange = (e) => {
        S.res = e.target.checked;
        draw();
      };
      $('#sr-sq', root).onchange = (e) => {
        S.sq = e.target.checked;
        draw();
      };
      $('#sr-file', root).onchange = (e) => {
        const f = e.target.files[0];
        if (!f) return;
        const fr2 = new FileReader();
        fr2.onerror = () => {
          st.textContent = 'That file could not be read. Try saving it again as plain CSV.';
        };
        fr2.onload = () => {
          const p = parseXY(fr2.result);
          if (typeof p === 'string') {
            st.textContent = p;
            return;
          }
          let x0 = 1 / 0,
            x1 = -1 / 0,
            y0 = 1 / 0,
            y1 = -1 / 0;
          for (const q of p) {
            x0 = Math.min(x0, q[0]);
            x1 = Math.max(x1, q[0]);
            y0 = Math.min(y0, q[1]);
            y1 = Math.max(y1, q[1]);
          }
          const r = rng(4);
          pts = p.map((q) => ({
            x: x1 > x0 ? -3 + (6 * (q[0] - x0)) / (x1 - x0) : 0,
            y: y1 > y0 ? -5 + (10 * (q[1] - y0)) / (y1 - y0) : 0,
            test: r() < 0.3,
          }));
          st.textContent =
            'Showing ' +
            pts.length +
            ' rows from ' +
            f.name.slice(0, 30) +
            ', rescaled to fit the plot.';
          begin(-3.5, -2);
          draw();
          pb.start();
        };
        fr2.readAsText(f);
        e.target.value = '';
      };
      let drag = -1;
      const at = (e) => {
        const r = c.getBoundingClientRect(),
          px = e.clientX - r.left,
          py = e.clientY - r.top;
        return {
          px,
          py,
          x: XR[0] + ((px - PS.m.l) / (PS.w - PS.m.l - PS.m.r)) * (XR[1] - XR[0]),
          y: YR[0] + ((PS.h - PS.m.b - py) / (PS.h - PS.m.t - PS.m.b)) * (YR[1] - YR[0]),
        };
      };
      c.addEventListener('pointerdown', (e) => {
        if (!PS) return;
        const a = at(e);
        let bi = -1,
          bd = 18;
        pts.forEach((p, i) => {
          const d = Math.hypot(PS.X(p.x) - a.px, PS.Y(p.y) - a.py);
          if (d < bd) {
            bd = d;
            bi = i;
          }
        });
        if (bi < 0) {
          pts.push({ x: clamp(a.x, -3.4, 3.4), y: clamp(a.y, -6.8, 6.8), test: false });
          bi = pts.length - 1;
        }
        drag = bi;
        c.setPointerCapture(e.pointerId);
        draw();
      });
      c.addEventListener('pointermove', (e) => {
        if (drag < 0) return;
        const a = at(e);
        pts[drag].x = clamp(a.x, -3.4, 3.4);
        pts[drag].y = clamp(a.y, -6.8, 6.8);
        draw();
      });
      ['pointerup', 'pointercancel'].forEach((n) =>
        c.addEventListener(n, () => {
          drag = -1;
        }),
      );
      cl.addEventListener('click', (e) => {
        if (!LM) return;
        const r = cl.getBoundingClientRect(),
          m = LM.m,
          s = BR[0] + ((e.clientX - r.left - m.l) / (LM.w - m.l - m.r)) * (BR[1] - BR[0]),
          a = AR[0] + ((LM.h - m.b - (e.clientY - r.top)) / (LM.h - m.t - m.b)) * (AR[1] - AR[0]);
        begin(clamp(a, AR[0], AR[1]), clamp(s, BR[0], BR[1]));
        draw();
        pb.start();
      });
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'multiple',
    tab: 'Multiple linear regression',
    title: 'Two inputs make a plane',
    lede: 'With two inputs the line becomes a plane. Rotate it and watch it tilt into place. Then make the two inputs correlated and see the estimates lose their footing.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The data and the fitted plane</figcaption><canvas class="d3" id="mr-c"></canvas>${viewbar}</figure>
 <figure class="fig"><figcaption>The two slopes, re-estimated on 200 fresh samples</figcaption><canvas class="sq" id="mr-e" role="img" aria-label="Cloud of slope estimates across repeated samples"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="mr-play" type="button">Fit the plane</button><button class="btn" id="mr-new" type="button">New random sample</button></div><label class="check" for="mr-res"><input type="checkbox" id="mr-res" checked>Show the misses (residuals)</label>`)}
 ${grp('The true relationship', slider('mr-b1', 'True slope for x₁', -2, 2, 0.05, 1.2) + slider('mr-b2', 'True slope for x₂', -2, 2, 0.05, -0.8) + slider('mr-s', 'Noise σ', 0, 2.5, 0.05, 0.8))}
 ${grp('The inputs', slider('mr-r', 'Correlation between x₁ and x₂', -0.98, 0.98, 0.01, 0) + slider('mr-n', 'Number of points', 10, 200, 1, 60))}</div>
 <dl class="read">${rd('b', 'Estimated intercept, slope₁, slope₂', 'hl')}${rd('r2', 'R²')}${rd('ar2', 'Adjusted R²')}${rd('f', 'F statistic')}${rd('p', 'p-value (all slopes = 0?)', 'hl2')}${rd('vif', 'Variance inflation factor', 'hl3')}</dl>`,
    tryit:
      'Slide the correlation between the inputs from 0 to 0.95. The plane hardly changes, but watch the cloud of estimates on the right. Which combinations of the two slopes can the data no longer tell apart?',
    math: String.raw`<h4>Intuition</h4><p>Each input gets its own slope, and each slope answers a specific question: if this input goes up by one <em>while the other stays fixed</em>, how much does the prediction change? When the two inputs always move together, the data never shows one changing without the other, so it cannot say how to split the credit. The plane is still well pinned down above the data, but it can pivot freely about the line where the data lies.</p>
 <h4>The model</h4>$$\hat y=\beta_0+\beta_1x_1+\beta_2x_2\qquad\text{or, stacking all points,}\qquad \hat{\mathbf y}=X\boldsymbol\beta$$<p>where each row of \(X\) is \((1,x_{i1},x_{i2})\).</p>
 <h4>The normal equations</h4>$$L(\boldsymbol\beta)=\frac1n\lVert\mathbf y-X\boldsymbol\beta\rVert^2,\qquad \nabla L=0\ \Longrightarrow\ X^{\top}X\,\hat{\boldsymbol\beta}=X^{\top}\mathbf y,\qquad \hat{\boldsymbol\beta}=(X^{\top}X)^{-1}X^{\top}\mathbf y.$$<p>Geometrically, \(X\hat{\boldsymbol\beta}\) is the projection of \(\mathbf y\) onto the space spanned by the columns of \(X\); the residual vector is perpendicular to every column.</p>
 <h4>How much the estimates wobble</h4>$$\operatorname{Cov}(\hat{\boldsymbol\beta})=\sigma^2\,(X^{\top}X)^{-1}$$<p>The right-hand cloud is this covariance, drawn by refitting on fresh noise. With uncorrelated inputs it is round. As the correlation \(r\) between \(x_1\) and \(x_2\) grows, \(X^{\top}X\) becomes nearly singular and the cloud stretches into a long thin ellipse: \(\hat\beta_1+\hat\beta_2\) is still known well, \(\hat\beta_1-\hat\beta_2\) is not. The variance of each slope is multiplied by</p>$$\mathrm{VIF}=\frac{1}{1-r^2}.$$
 <h4>Does the model explain anything?</h4>$$R^2=1-\frac{\mathrm{RSS}}{\mathrm{TSS}},\qquad \bar R^2=1-(1-R^2)\frac{n-1}{n-p-1},\qquad F=\frac{(\mathrm{TSS}-\mathrm{RSS})/p}{\mathrm{RSS}/(n-p-1)}\sim F_{p,\,n-p-1}$$<p>with \(p=2\) inputs. \(R^2\) never goes down when you add an input, even a useless one; the adjusted version charges for each extra input. The F test asks whether all slopes could be zero at once. Note that with strongly correlated inputs the F test can be decisive while neither slope is individually significant.</p>`,
    q: 'With correlation 0.95 the fitted plane still predicts well. So why does multicollinearity matter?',
    a: 'It matters when you interpret the slopes or predict away from the data. The individual slopes are unreliable and can even flip sign between samples, and predictions for an x₁ and x₂ that do not follow the usual pattern can be badly wrong.',
    init(root) {
      const c = $('#mr-c', root),
        ce = $('#mr-e', root),
        S = { b: [0.5, 1.2, -0.8], noise: 0.8, r: 0, n: 60, res: true };
      let seed = 5,
        X = [],
        y = [],
        beta = [0, 0, 0],
        ym = 0,
        u = 0,
        wait = 0,
        sim = [],
        pb,
        draw;
      const orb = Orbit(c, () => draw(), $('.viewbar', root));
      const gen = () => {
        const r = rng(seed),
          rn = randnFrom(r),
          q = Math.sqrt(1 - S.r * S.r);
        X = [];
        y = [];
        for (let i = 0; i < S.n; i++) {
          const a = clamp(rn(), -2.4, 2.4),
            b = clamp(S.r * a + q * rn(), -2.4, 2.4);
          X.push([1, a, b]);
          y.push(S.b[0] + S.b[1] * a + S.b[2] * b + S.noise * rn());
        }
        beta = lsq(X, y, 1e-9);
        ym = y.reduce((s, v) => s + v, 0) / y.length;
        const r2 = rng(77),
          rn2 = randnFrom(r2);
        sim = [];
        for (let k = 0; k < 200; k++) {
          const yy = X.map((row) => S.b[0] + S.b[1] * row[1] + S.b[2] * row[2] + S.noise * rn2());
          const bb = lsq(X, yy, 1e-9);
          sim.push([bb[1], bb[2]]);
        }
      };
      gen();
      const tp = testPoint(
        root,
        'mr',
        ['Your x₁', 'Your x₂'],
        [1, -1],
        [
          [-2.5, 2.5],
          [-2.5, 2.5],
        ],
      );
      tp.on = () => draw();
      draw = () => {
        const t = T(),
          e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u),
          cb = [ym + (beta[0] - ym) * e, beta[1] * e, beta[2] * e],
          ZS = 0.42;
        {
          const { g, w, h } = fit(c),
            pr = orb.cam(w, h, 5.6),
            R = 2.5,
            pl = (a, b) => clamp((cb[0] + cb[1] * a + cb[2] * b) * ZS, -3.2, 3.2),
            path = (ps) => {
              g.beginPath();
              ps.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
            };
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          for (let k = -2; k <= 2; k++) {
            path([pr(k * 1.25, -R, -2), pr(k * 1.25, R, -2)]);
            g.stroke();
            path([pr(-R, k * 1.25, -2), pr(R, k * 1.25, -2)]);
            g.stroke();
          }
          const N = 10,
            Q = [];
          for (let i = 0; i < N; i++)
            for (let j = 0; j < N; j++) {
              const a0 = -R + (2 * R * i) / N,
                a1 = -R + (2 * R * (i + 1)) / N,
                b0 = -R + (2 * R * j) / N,
                b1 = -R + (2 * R * (j + 1)) / N,
                ps = [
                  pr(a0, b0, pl(a0, b0)),
                  pr(a1, b0, pl(a1, b0)),
                  pr(a1, b1, pl(a1, b1)),
                  pr(a0, b1, pl(a0, b1)),
                ];
              Q.push({ ps, d: (ps[0][2] + ps[2][2]) / 2, z: (pl(a0, b0) + pl(a1, b1)) / 2 });
            }
          Q.sort((a, b) => b.d - a.d);
          for (const o of Q) {
            path(o.ps);
            g.closePath();
            g.fillStyle = ramp(t, clamp(0.5 + o.z / 5, 0, 1), 0.42);
            g.fill();
            g.strokeStyle = mix(t.ink, t.ink, 0, 0.14);
            g.lineWidth = 0.7;
            g.stroke();
          }
          const D = X.map((row, i) => {
            const z = clamp(y[i] * ZS, -3.4, 3.4),
              zp = pl(row[1], row[2]),
              p = pr(row[1], row[2], z);
            return { p, q: pr(row[1], row[2], zp), up: z >= zp };
          });
          D.sort((a, b) => b.p[2] - a.p[2]);
          for (const o of D) {
            if (S.res) {
              g.strokeStyle = mix(t.ok, t.ok, 0, 0.7);
              g.lineWidth = 1.2;
              path([o.p, o.q]);
              g.stroke();
            }
            g.fillStyle = o.up ? t.accent : mix(t.accent, t.bg, 0.45);
            g.beginPath();
            g.arc(o.p[0], o.p[1], 3.8, 0, 7);
            g.fill();
          }
          {
            const q0 = pr(tp.v[0], tp.v[1], -2),
              q1 = pr(tp.v[0], tp.v[1], pl(tp.v[0], tp.v[1]));
            g.strokeStyle = t.ink;
            g.lineWidth = 1.6;
            g.setLineDash([3, 3]);
            path([q0, q1]);
            g.stroke();
            g.setLineDash([]);
            markTP(g, q1[0], q1[1], t);
          }
          g.font = SANS;
          g.fillStyle = t.ink;
          g.textAlign = 'center';
          let a = pr(0, -R * 1.18, -2);
          g.fillText('x₁', a[0], a[1]);
          a = pr(-R * 1.18, 0, -2);
          g.fillText('x₂', a[0], a[1]);
          a = pr(-R, -R, 2.4);
          g.fillText('y', a[0], a[1]);
        }
        {
          const { g, w, h } = fit(ce),
            m = { l: 40, r: 10, t: 10, b: 30 },
            LR = [-3, 3],
            Xf = (v) => m.l + ((v - LR[0]) / 6) * (w - m.l - m.r),
            Yf = (v) => h - m.b - ((v - LR[0]) / 6) * (h - m.t - m.b);
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.font = MONO;
          g.fillStyle = t.muted;
          for (let v = -3; v <= 3; v++) {
            g.beginPath();
            g.moveTo(Xf(v), m.t);
            g.lineTo(Xf(v), h - m.b);
            g.moveTo(m.l, Yf(v));
            g.lineTo(w - m.r, Yf(v));
            g.stroke();
            g.textAlign = 'center';
            g.fillText(String(v), Xf(v), h - m.b + 14);
            g.textAlign = 'right';
            g.fillText(String(v), m.l - 6, Yf(v) + 4);
          }
          g.textAlign = 'center';
          g.fillText('estimated slope for x₁', w / 2 + m.l / 2, h - 3);
          g.save();
          g.translate(11, h / 2);
          g.rotate(-Math.PI / 2);
          g.fillText('estimated slope for x₂', 0, 0);
          g.restore();
          g.save();
          g.beginPath();
          g.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
          g.clip();
          g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
          for (const s of sim) {
            g.beginPath();
            g.arc(Xf(s[0]), Yf(s[1]), 2.4, 0, 7);
            g.fill();
          }
          const tx = Xf(S.b[1]),
            ty = Yf(S.b[2]);
          g.strokeStyle = t.ink;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(tx - 8, ty);
          g.lineTo(tx + 8, ty);
          g.moveTo(tx, ty - 8);
          g.lineTo(tx, ty + 8);
          g.stroke();
          g.fillStyle = t.gold;
          g.strokeStyle = t.bg;
          g.beginPath();
          g.arc(Xf(beta[1]), Yf(beta[2]), 7, 0, 7);
          g.fill();
          g.stroke();
          g.restore();
          g.textAlign = 'left';
          g.fillStyle = t.ink;
          g.fillText('+ truth', m.l + 8, m.t + 12);
          g.fillStyle = t.cut;
          g.fillText('● this sample', m.l + 8, m.t + 27);
        }
        const n = y.length,
          p = 2;
        let rss = 0,
          tss = 0;
        for (let i = 0; i < n; i++) {
          rss += (y[i] - dot(beta, X[i])) ** 2;
          tss += (y[i] - ym) ** 2;
        }
        const r2 = tss > 1e-12 ? 1 - rss / tss : 0,
          F = rss > 1e-12 ? (tss - rss) / p / (rss / (n - p - 1)) : Infinity;
        let s1 = 0,
          s2 = 0,
          s11 = 0,
          s22 = 0,
          s12 = 0;
        for (const row of X) {
          s1 += row[1];
          s2 += row[2];
          s11 += row[1] * row[1];
          s22 += row[2] * row[2];
          s12 += row[1] * row[2];
        }
        const rr = (s12 - (s1 * s2) / n) / Math.sqrt((s11 - (s1 * s1) / n) * (s22 - (s2 * s2) / n));
        tp.out.textContent =
          'At x₁ = ' +
          f2(tp.v[0]) +
          ', x₂ = ' +
          f2(tp.v[1]) +
          ' the fitted plane predicts y = ' +
          f2(beta[0]) +
          ' + (' +
          f2(beta[1]) +
          ' × ' +
          f2(tp.v[0]) +
          ') + (' +
          f2(beta[2]) +
          ' × ' +
          f2(tp.v[1]) +
          ') = ' +
          f2(dot(beta, [1, tp.v[0], tp.v[1]])) +
          '.';
        setR(root, 'b', f2(beta[0]) + ', ' + f2(beta[1]) + ', ' + f2(beta[2]));
        setR(root, 'r2', f3(r2));
        setR(root, 'ar2', f3(1 - ((1 - r2) * (n - 1)) / (n - p - 1)));
        setR(root, 'f', isFinite(F) ? F.toFixed(1) : '∞');
        setR(root, 'p', isFinite(F) ? pfmt(fpval(F, p, n - p - 1)) : '< 0.0001');
        setR(root, 'vif', (1 / (1 - rr * rr)).toFixed(2));
      };
      const anim = Anim(() => {
        if (wait > 0) {
          wait--;
          if (!wait) u = 0;
          return;
        }
        u = Math.min(1, u + 0.012);
        if (u >= 1) wait = 200;
        draw();
      });
      pb = playBtn($('#mr-play', root), anim);
      $('#mr-play', root).addEventListener('click', () => {
        if (anim.on && u >= 1) {
          u = 0;
          wait = 0;
        }
      });
      const re = () => {
        gen();
        draw();
      };
      bind(root, 'mr-b1', (v) => {
        S.b[1] = v;
        re();
      });
      bind(root, 'mr-b2', (v) => {
        S.b[2] = v;
        re();
      });
      bind(root, 'mr-s', (v) => {
        S.noise = v;
        re();
      });
      bind(root, 'mr-r', (v) => {
        S.r = v;
        re();
      });
      bind(
        root,
        'mr-n',
        (v) => {
          S.n = v;
          re();
        },
        (v) => String(v),
      );
      $('#mr-new', root).onclick = () => {
        seed++;
        u = 0;
        wait = 0;
        re();
        pb.start();
      };
      $('#mr-res', root).onchange = (e) => {
        S.res = e.target.checked;
        draw();
      };
      return {
        draw,
        stop() {
          pb.stop();
          orb.spin(false);
        },
        auto() {
          pb.start();
          orb.spin(true);
        },
      };
    },
  },

  {
    id: 'poly',
    tab: 'Polynomial regression',
    title: 'Bending the line',
    lede: 'Raise the degree and the curve bends to follow the points. A little bending finds the pattern. A lot of bending memorises the noise: training error keeps falling while test error turns back up.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The data and the fitted curve</figcaption><canvas class="short" id="pr-c" role="img" aria-label="Scatter plot with a polynomial fit"></canvas></figure>
 <figure class="fig"><figcaption>Error against degree (log scale)</figcaption><canvas class="short" id="pr-e" role="img" aria-label="Training and test error for each polynomial degree"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="pr-play" type="button">Sweep the degree</button><button class="btn" id="pr-new" type="button">New random sample</button></div>`)}
 ${grp('The model', slider('pr-d', 'Polynomial degree', 1, 14, 1, 3))}
 ${grp('The data', `<label class="ctl" for="pr-f"><span>True curve</span><select id="pr-f"><option value="sine">Sine wave</option><option value="cubic">Cubic</option><option value="bump">Single bump</option><option value="line">Straight line</option></select></label>` + slider('pr-s', 'Noise σ', 0, 0.8, 0.01, 0.25) + slider('pr-n', 'Number of points', 10, 150, 1, 30))}</div>
 <dl class="read">${rd('d', 'Degree, and number of coefficients')}${rd('tr', 'Train error (MSE)', 'hl3')}${rd('te', 'Test error (MSE)', 'hl2')}${rd('best', 'Degree with the lowest test error', 'hl')}</dl>`,
    tryit:
      'Watch the two error curves as the degree climbs. Where do they part ways? Then raise the number of points to 150: does degree 14 still overfit? Try the straight line as the truth with degree 10.',
    math: String.raw`<h4>Intuition</h4><p>A straight line can only tilt. Give the model powers of \(x\) and it can bend, once for each extra degree. With enough bends it can pass through every training point exactly, and that is the trap: the points contain noise, and a curve that chases the noise is wrong everywhere between the points. The fit that looks best on the data it has seen is no longer the fit that predicts best.</p>
 <h4>Still a linear model</h4>$$\hat y=\beta_0+\beta_1x+\beta_2x^2+\dots+\beta_dx^d=\boldsymbol\beta^{\top}\boldsymbol\phi(x),\qquad \boldsymbol\phi(x)=(1,x,x^2,\dots,x^d)$$<p>The curve is not linear in \(x\), but it is linear in the coefficients. So nothing new is needed to fit it: build a matrix \(\Phi\) whose columns are the powers of \(x\) and solve the same normal equations as before, \(\hat{\boldsymbol\beta}=(\Phi^{\top}\Phi)^{-1}\Phi^{\top}\mathbf y\). Swapping raw inputs for features \(\boldsymbol\phi(x)\) is the idea behind kernels and, later, behind the hidden layers of a neural network.</p>
 <h4>Bias and variance</h4><p>Average over many possible training sets, and the expected test error at a point splits into three parts:</p>$$\mathbb E\big[(y-\hat f(x))^2\big]=\underbrace{\big(\mathbb E[\hat f(x)]-f(x)\big)^2}_{\text{bias}^2}+\underbrace{\operatorname{Var}\big(\hat f(x)\big)}_{\text{variance}}+\underbrace{\sigma^2}_{\text{noise}}$$<p>Low degree: the model cannot take the right shape, so bias is high (underfitting). High degree: the model changes wildly from sample to sample, so variance is high (overfitting). Press “New random sample” at degree 2 and at degree 12 to see the two cases. Test error is lowest where the sum of the two is smallest.</p>
 <h4>Why training error cannot be trusted</h4><p>Every polynomial of degree \(d\) is also a polynomial of degree \(d+1\) with the last coefficient zero. So the best degree-\((d+1)\) fit is never worse on the training data: training error can only go down. With \(d=n-1\) it reaches zero. Only held-out data can tell you when to stop, which is what validation sets and cross-validation are for.</p>
 <p>The fit here uses Legendre polynomials in place of raw powers. They span the same curves but keep the arithmetic stable at high degree.</p>`,
    q: 'More data or a lower degree: which fixes overfitting?',
    a: 'Both. A lower degree removes the flexibility that chases noise. More data pins the flexible curve down so it has less freedom to wiggle. With 150 points a degree-14 curve behaves far better than with 20.',
    init(root) {
      const c = $('#pr-c', root),
        ce = $('#pr-e', root),
        S = { d: 3, noise: 0.25, n: 30, f: 'sine' },
        F = {
          sine: (x) => Math.sin(2.8 * x),
          cubic: (x) => 2.4 * x * x * x - 1.3 * x,
          bump: (x) => 1.6 * Math.exp(-9 * x * x) - 0.6,
          line: (x) => 0.9 * x,
        },
        DM = 14;
      let seed = 8,
        pts = [],
        fits = [],
        etr = [],
        ete = [],
        pb;
      const gen = () => {
        const r = rng(seed),
          rn = randnFrom(r);
        pts = [];
        for (let i = 0; i < S.n; i++) {
          const x = r() * 2 - 1;
          pts.push({ x, y: F[S.f](x) + S.noise * rn(), test: r() < 0.3 });
        }
        const tr = pts.filter((p) => !p.test),
          te = pts.filter((p) => p.test);
        fits = [];
        etr = [];
        ete = [];
        for (let d = 1; d <= DM; d++) {
          const b = lsq(
            tr.map((p) => legendre(p.x, d)),
            tr.map((p) => p.y),
            1e-9,
          );
          fits[d] = b;
          const e = (P) =>
            P.length
              ? P.reduce((s, p) => s + (p.y - dot(b, legendre(p.x, d))) ** 2, 0) / P.length
              : NaN;
          etr[d] = e(tr);
          ete[d] = e(te);
        }
      };
      gen();
      const tp = testPoint(root, 'pr', ['Your x'], [0.4], [[-1, 1]]);
      tp.on = () => draw();
      const draw = () => {
        const t = T(),
          b = fits[S.d];
        {
          const { g, w, h } = fit(c),
            P = plot(g, w, h, [-1.05, 1.05], [-2.2, 2.2], t, {
              yticks: [-2, -1, 0, 1, 2],
              xstep: 0.5,
              xlabel: 'x',
            });
          clipPlot(g, P);
          g.setLineDash([6, 5]);
          curve(g, P, F[S.f], -1, 1);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.6;
          g.stroke();
          g.setLineDash([]);
          curve(g, P, (x) => clamp(dot(b, legendre(x, S.d)), -9, 9), -1, 1, 400);
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
          markTP(g, P.X(tp.v[0]), P.Y(clamp(dot(b, legendre(tp.v[0], S.d)), -9, 9)), t);
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.muted;
          g.fillText('dashed = true curve, hollow = test', P.m.l + 8, P.m.t + 8);
        }
        {
          const { g, w, h } = fit(ce),
            LY = [-2.6, 1.2],
            P = plot(g, w, h, [0.5, DM + 0.5], LY, t, { xlabel: 'degree' }),
            lg = (v) => clamp(Math.log10(Math.max(v, 1e-9)), LY[0], LY[1]);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          [-2, -1, 0, 1].forEach((v) => {
            g.strokeStyle = t.line;
            g.beginPath();
            g.moveTo(P.m.l, P.Y(v));
            g.lineTo(w - P.m.r, P.Y(v));
            g.stroke();
            g.fillText(
              v === 0 ? '1' : v === 1 ? '10' : v === -1 ? '0.1' : '0.01',
              P.m.l - 6,
              P.Y(v) + 4,
            );
          });
          g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
          g.lineWidth = 10;
          g.beginPath();
          g.moveTo(P.X(S.d), P.Y(LY[0]));
          g.lineTo(P.X(S.d), P.Y(LY[1]));
          g.stroke();
          [
            [etr, t.accent],
            [ete, t.ok],
          ].forEach(([E, col]) => {
            g.strokeStyle = col;
            g.fillStyle = col;
            g.lineWidth = 2.4;
            g.beginPath();
            for (let d = 1; d <= DM; d++)
              d > 1 ? g.lineTo(P.X(d), P.Y(lg(E[d]))) : g.moveTo(P.X(d), P.Y(lg(E[d])));
            g.stroke();
            for (let d = 1; d <= DM; d++) {
              g.beginPath();
              g.arc(P.X(d), P.Y(lg(E[d])), d === S.d ? 5.5 : 3, 0, 7);
              g.fill();
            }
          });
          g.textAlign = 'left';
          g.fillStyle = t.accent;
          g.fillText('train', P.m.l + 8, P.m.t + 8);
          g.fillStyle = t.ok;
          g.fillText('test', P.m.l + 8, P.m.t + 23);
        }
        let bd = 1;
        for (let d = 2; d <= DM; d++) if (ete[d] < ete[bd]) bd = d;
        tp.out.textContent =
          'At x = ' +
          f2(tp.v[0]) +
          ' the degree-' +
          S.d +
          ' curve predicts y = ' +
          f2(dot(b, legendre(tp.v[0], S.d))) +
          '. The true curve there is ' +
          f2(F[S.f](tp.v[0])) +
          '.';
        setR(root, 'd', S.d + ', ' + (S.d + 1));
        setR(root, 'tr', f3(etr[S.d]));
        setR(
          root,
          'te',
          isFinite(ete[S.d]) ? (ete[S.d] > 999 ? ete[S.d].toExponential(1) : f3(ete[S.d])) : '–',
        );
        setR(root, 'best', String(bd));
      };
      const ds = bind(
        root,
        'pr-d',
        (v) => {
          S.d = v;
          draw();
        },
        (v) => String(v),
      );
      let fr = 0,
        dir = 1;
      const anim = Anim(() => {
        if (++fr % 34) return;
        let d = S.d + dir;
        if (d > DM) {
          dir = -1;
          d = DM - 1;
        }
        if (d < 1) {
          dir = 1;
          d = 2;
        }
        S.d = d;
        ds.set(d);
        draw();
      });
      pb = playBtn($('#pr-play', root), anim);
      grab(ds.el, pb);
      const re = () => {
        gen();
        draw();
      };
      $('#pr-f', root).onchange = (e) => {
        S.f = e.target.value;
        re();
      };
      bind(root, 'pr-s', (v) => {
        S.noise = v;
        re();
      });
      bind(
        root,
        'pr-n',
        (v) => {
          S.n = v;
          re();
        },
        (v) => String(v),
      );
      $('#pr-new', root).onclick = () => {
        seed++;
        re();
      };
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'splines',
    tab: 'Splines',
    title: 'Many small curves, joined smoothly',
    lede: 'One high-degree polynomial wiggles everywhere. A spline uses a low-degree piece between each pair of knots and joins the pieces so the seams do not show. Change the kind of join and watch what happens at the knots and beyond the ends of the data.',
    stage: `<div class="row seg" id="sp-kind" role="group" aria-label="Kind of spline"><button class="btn" type="button" data-k="step">Steps</button><button class="btn" type="button" data-k="linear">Linear Spline</button><button class="btn" type="button" data-k="cubic">Cubic Spline</button><button class="btn" type="button" data-k="natural">Natural Cubic Spline</button><button class="btn" type="button" data-k="pen">Smoothing Spline</button></div>
 <p class="cap" id="sp-cap" role="status"></p>
 <div class="stage"><figure class="fig"><figcaption>The fit, with its knots (shaded = outside the data)</figcaption><canvas class="short" id="sp-c" role="img" aria-label="Scatter plot with a spline fit and knot positions"></canvas></figure>
 <figure class="fig"><figcaption>Curvature of the fit: its second derivative</figcaption><canvas class="short" id="sp-d" role="img" aria-label="Second derivative of the fitted curve"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="sp-play" type="button">Sweep</button><button class="btn" id="sp-new" type="button">New random sample</button></div>`)}
 ${grp('The model', slider('sp-k', 'Number of interior knots', 0, 12, 1, 4) + slider('sp-l', 'Smoothing λ (smoothing spline only)', 0, 40, 1, 18))}
 ${grp('The data', `<label class="ctl" for="sp-f"><span>True curve</span><select id="sp-f"><option value="sine">Sine wave</option><option value="cubic">Cubic</option><option value="bump">Single bump</option><option value="line">Straight line</option></select></label>` + slider('sp-s', 'Noise σ', 0, 0.8, 0.01, 0.25) + slider('sp-n', 'Number of points', 12, 200, 1, 40))}</div>
 <dl class="read">${rd('df', 'Coefficients fitted')}${rd('tr', 'Train error (MSE)', 'hl3')}${rd('te', 'Test error (MSE)', 'hl2')}${rd('end', 'Curvature at the two ends', 'hl')}${rd('best', 'Knots with the lowest test error')}</dl>`,
    tryit:
      'Step through the five kinds with 4 knots and watch the right-hand picture: where is the curvature broken, where is it continuous, and for which kind is it exactly zero at both ends? Then look at the shaded zones: which fit would you trust outside the data?',
    math: String.raw`<h4>Intuition</h4><p>A draughtsman's spline is a thin flexible strip of wood, pinned at a few points and left to settle. It bends smoothly, as little as it can, and beyond the last pin it runs straight. A mathematical spline copies this: cut the \(x\)-axis at a few knots, use a simple polynomial on each stretch, and tie the pieces together so the joins are smooth. Flexibility is local: moving one point bends the curve nearby and leaves the far end alone, which is exactly what a single high-degree polynomial cannot do.</p>
 <h4>Piecewise polynomials and how smoothly they join</h4><p>With knots \(\xi_1<\dots<\xi_K\), fit a polynomial of degree \(M\) on each interval and require the pieces to agree at each knot in value and in their first \(M-1\) derivatives. <b>Steps</b> (\(M=0\)) do not even agree in value. A <b>linear spline</b> (\(M=1\)) is continuous but has corners. A <b>cubic spline</b> (\(M=3\)) matches value, slope and curvature at every knot; only the third derivative jumps, and the eye cannot see that.</p>
 <h4>The cubic spline as a linear model</h4>$$f(x)=\beta_0+\beta_1x+\beta_2x^2+\beta_3x^3+\sum_{k=1}^{K}\theta_k\,(x-\xi_k)_+^3,\qquad (u)_+=\max(u,0)$$<p>Each term \((x-\xi_k)_+^3\) is zero to the left of its knot and switches on smoothly to the right, with value, slope and curvature all zero at the knot itself. So adding it changes the cubic from that knot onward without breaking smoothness. The model has \(K+4\) coefficients and is fitted by ordinary least squares, exactly like polynomial regression with different columns.</p>
 <h4>The natural cubic spline</h4><p>A cubic spline is poorly controlled near the ends, where there are few points, and outside the data it shoots off like a cubic. The natural spline adds two conditions at each boundary knot \(a\) and \(b\):</p>$$f''(a)=0,\qquad f''(b)=0,\qquad\text{and }f\text{ is linear for }x<a\text{ and }x>b.$$<p>Zero curvature at the ends means the curve leaves the data as a straight line. These four constraints remove four coefficients, leaving \(K+2\) when the boundary knots are counted, so it is less flexible exactly where the data is thinnest. Its basis can be written</p>$$N_1=1,\quad N_2=x,\quad N_{k+2}(x)=d_k(x)-d_{K-1}(x),\qquad d_k(x)=\frac{(x-\xi_k)_+^3-(x-\xi_K)_+^3}{\xi_K-\xi_k}.$$
 <h4>The smoothing spline</h4><p>Choosing knots is awkward. The smoothing spline avoids it: use many knots and penalise curvature instead.</p>$$\min_f\ \sum_{i=1}^n\big(y_i-f(x_i)\big)^2+\lambda\int f''(x)^2\,dx$$<p>The first term wants to pass through the points; the second charges for bending. At \(\lambda=0\) the curve interpolates the data, and as \(\lambda\to\infty\) bending becomes unaffordable and the answer is the least-squares straight line. The remarkable fact is that the exact minimiser over <em>all</em> smooth functions is a natural cubic spline with a knot at every data point. The picture uses a close relative, a penalised spline: 14 fixed knots with a ridge penalty on the coefficients \(\theta_k\), since each \(\theta_k\) is the jump in the third derivative at its knot.</p>
 <h4>Counting flexibility</h4><p>More knots, like a higher degree, lower the bias and raise the variance. The number of fitted coefficients (the degrees of freedom) is the dial, and test error picks its setting.</p>`,
    q: 'Why is a natural cubic spline usually preferred to a plain cubic spline with the same knots?',
    a: 'Near and beyond the ends of the data a cubic spline has freedom the data cannot pin down, so it swings wildly there. The natural spline gives up that freedom by forcing zero curvature at the ends. It has four fewer coefficients, lower variance at the boundaries, and it extrapolates as a straight line.',
    init(root) {
      const c = $('#sp-c', root),
        cd = $('#sp-d', root),
        cap = $('#sp-cap', root),
        S = { kind: 'natural', K: 4, li: 18, noise: 0.25, n: 40, f: 'sine' },
        F = {
          sine: (x) => Math.sin(2.8 * x),
          cubic: (x) => 2.4 * x * x * x - 1.3 * x,
          bump: (x) => 1.6 * Math.exp(-9 * x * x) - 0.6,
          line: (x) => 0.9 * x,
        },
        LAM = (i) => Math.pow(10, -7 + (8 * i) / 40);
      let seed = 8,
        pts = [],
        b = [],
        etr = 0,
        ete = 0,
        bestK = 0,
        pb;
      const kx = () => (S.kind === 'pen' ? 14 : S.K),
        bas = (x) => splineBasis(S.kind, x, kx());
      const fitK = (kind, K, lam) => {
        const tr = pts.filter((p) => !p.test),
          X = tr.map((p) => splineBasis(kind, p.x, K)),
          y = tr.map((p) => p.y);
        let rg = 1e-9;
        if (kind === 'pen') rg = X[0].map((_, j) => (j < 4 ? 1e-9 : lam));
        return lsq(X, y, rg);
      };
      const err = (P, kind, K, bb) =>
        P.length
          ? P.reduce((s, p) => s + (p.y - dot(bb, splineBasis(kind, p.x, K))) ** 2, 0) / P.length
          : NaN;
      const gen = () => {
        const r = rng(seed),
          rn = randnFrom(r);
        pts = [];
        for (let i = 0; i < S.n; i++) {
          const x = r() * 2 - 1;
          pts.push({ x, y: F[S.f](x) + S.noise * rn(), test: r() < 0.3 });
        }
      };
      const refit = () => {
        b = fitK(S.kind, kx(), LAM(S.li));
        const tr = pts.filter((p) => !p.test),
          te = pts.filter((p) => p.test);
        etr = err(tr, S.kind, kx(), b);
        ete = err(te, S.kind, kx(), b);
        bestK = -1;
        if (S.kind !== 'pen') {
          let be = 1 / 0;
          for (let k = 0; k <= 12; k++) {
            const bb = fitK(S.kind, k),
              e = err(te, S.kind, k, bb);
            if (e < be) {
              be = e;
              bestK = k;
            }
          }
        }
      };
      gen();
      refit();
      const tp = testPoint(root, 'sp', ['Your x (try beyond ±1)'], [0.4], [[-1.4, 1.4]]);
      tp.on = () => draw();
      const CAP = {
        step: 'Steps: a separate constant on each stretch. The pieces do not even meet, so the fit jumps at every knot.',
        linear:
          'Linear spline: a straight piece on each stretch, joined end to end. Continuous, but with a corner at every knot, so the curvature is a spike there and zero elsewhere.',
        cubic:
          'Cubic spline: a cubic on each stretch, matching value, slope and curvature at every knot. The curvature on the right is one unbroken zig-zag line. Outside the data the fit runs away.',
        natural:
          'Natural cubic spline: a cubic spline whose second derivative is forced to zero at both ends. Look right: the curvature starts and finishes at exactly zero, and outside the data the fit continues as a straight line.',
        pen: 'Smoothing spline: many knots, with a price λ on curvature. Raise λ and the curvature on the right is pressed toward zero until only a straight line is left.',
      };
      const f = (x) => dot(b, bas(x)),
        d2 = (x) => {
          const e = 0.004;
          return (f(x + e) - 2 * f(x) + f(x - e)) / (e * e);
        };
      const draw = () => {
        const t = T(),
          K = kx(),
          kn = [];
        for (let j = 1; j <= K; j++) kn.push(-1 + (2 * j) / (K + 1));
        const XR = [-1.4, 1.4];
        const deco = (g, P, yr) => {
          g.fillStyle = mix(t.muted, t.muted, 0, 0.1);
          g.fillRect(P.X(XR[0]), P.Y(yr[1]), P.X(-1) - P.X(XR[0]), P.Y(yr[0]) - P.Y(yr[1]));
          g.fillRect(P.X(1), P.Y(yr[1]), P.X(XR[1]) - P.X(1), P.Y(yr[0]) - P.Y(yr[1]));
          g.setLineDash([2, 4]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          for (const k of kn) {
            g.beginPath();
            g.moveTo(P.X(k), P.Y(yr[0]));
            g.lineTo(P.X(k), P.Y(yr[1]));
            g.stroke();
          }
          g.setLineDash([]);
        };
        {
          const { g, w, h } = fit(c),
            YR = [-2.4, 2.4],
            P = plot(g, w, h, XR, YR, t, { yticks: [-2, -1, 0, 1, 2], xstep: 0.5, xlabel: 'x' });
          clipPlot(g, P);
          deco(g, P, YR);
          g.setLineDash([6, 5]);
          curve(g, P, F[S.f], XR[0], XR[1]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.5;
          g.stroke();
          g.setLineDash([]);
          curve(g, P, (x) => clamp(f(x), -9, 9), XR[0], XR[1], 560);
          g.strokeStyle = t.gold;
          g.lineWidth = 3.2;
          g.stroke();
          g.fillStyle = t.gold;
          for (const k of kn) {
            g.beginPath();
            g.arc(P.X(k), P.Y(clamp(f(k), -9, 9)), 3.4, 0, 7);
            g.fill();
          }
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
          markTP(g, P.X(tp.v[0]), P.Y(clamp(f(tp.v[0]), -9, 9)), t);
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.muted;
          g.fillText('dotted lines = knots, dashed = truth', P.m.l + 8, P.m.t + 8);
        }
        {
          const { g, w, h } = fit(cd);
          let mx = 1;
          for (let i = 0; i <= 200; i++) {
            const v = Math.abs(d2(-1 + (2 * i) / 200));
            if (isFinite(v)) mx = Math.max(mx, v);
          }
          mx = Math.min(mx * 1.15, 400);
          const YR = [-mx, mx],
            P = plot(g, w, h, XR, YR, t, { xstep: 0.5, xlabel: 'x' });
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          [-mx / 1.15, 0, mx / 1.15].forEach((v) =>
            g.fillText(v.toFixed(0), P.m.l - 6, P.Y(v) + 4),
          );
          clipPlot(g, P);
          deco(g, P, YR);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(P.X(XR[0]), P.Y(0));
          g.lineTo(P.X(XR[1]), P.Y(0));
          g.stroke();
          curve(g, P, (x) => clamp(d2(x), -mx * 2, mx * 2), XR[0], XR[1], 700);
          g.strokeStyle = t.ok;
          g.lineWidth = 2.6;
          g.stroke();
          if (S.kind === 'natural' || S.kind === 'pen' || S.kind === 'cubic') {
            g.fillStyle = t.gold;
            g.strokeStyle = t.bg;
            g.lineWidth = 2;
            [-1, 1].forEach((x) => {
              g.beginPath();
              g.arc(P.X(x), P.Y(clamp(2 * d2(x * 0.99) - d2(x * 0.98), -mx, mx)), 6, 0, 7);
              g.fill();
              g.stroke();
            });
          }
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.ok;
          g.fillText("f''(x)", P.m.l + 8, P.m.t + 8);
          g.fillStyle = t.cut;
          g.fillText('● the two ends of the data', P.m.l + 8, P.m.t + 23);
        }
        cap.textContent = CAP[S.kind];
        const e1 = 2 * d2(-0.99) - d2(-0.98),
          e2 = 2 * d2(0.99) - d2(0.98),
          z = (v) => (Math.abs(v) < 0.05 ? '0.00' : v.toFixed(1));
        tp.out.textContent =
          'At x = ' +
          f2(tp.v[0]) +
          ' this spline predicts y = ' +
          f2(f(tp.v[0])) +
          '. The true curve there is ' +
          f2(F[S.f](tp.v[0])) +
          '.' +
          (Math.abs(tp.v[0]) > 1
            ? ' This point is outside the data, so the prediction is an extrapolation.'
            : '');
        setR(root, 'df', String(b.length) + (S.kind === 'pen' ? ' (penalised)' : ''));
        setR(root, 'tr', f3(etr));
        setR(root, 'te', isFinite(ete) ? (ete > 999 ? ete.toExponential(1) : f3(ete)) : '–');
        setR(
          root,
          'end',
          S.kind === 'step' || S.kind === 'linear'
            ? 'not defined at knots'
            : z(e1) + ' and ' + z(e2),
        );
        setR(root, 'best', bestK < 0 ? 'set by λ here' : String(bestK));
        root
          .querySelectorAll('#sp-kind button')
          .forEach((x) =>
            x.setAttribute('aria-pressed', x.dataset.k === S.kind ? 'true' : 'false'),
          );
      };
      const ks = bind(
          root,
          'sp-k',
          (v) => {
            S.K = v;
            refit();
            draw();
          },
          (v) => String(v),
        ),
        ls = bind(
          root,
          'sp-l',
          (v) => {
            S.li = v;
            refit();
            draw();
          },
          (v) => (-7 + (8 * v) / 40).toFixed(1),
        );
      let fr = 0,
        dir = 1;
      const anim = Anim(() => {
        if (S.kind === 'pen') {
          if (++fr % 4) return;
          let v = S.li + dir;
          if (v > 40) {
            dir = -1;
            v = 39;
          }
          if (v < 0) {
            dir = 1;
            v = 1;
          }
          S.li = v;
          ls.set(v);
        } else {
          if (++fr % 40) return;
          let v = S.K + dir;
          if (v > 12) {
            dir = -1;
            v = 11;
          }
          if (v < 0) {
            dir = 1;
            v = 1;
          }
          S.K = v;
          ks.set(v);
        }
        refit();
        draw();
      });
      pb = playBtn($('#sp-play', root), anim);
      grab(ks.el, pb);
      grab(ls.el, pb);
      $('#sp-kind', root).addEventListener('click', (e) => {
        const k = e.target.dataset && e.target.dataset.k;
        if (!k) return;
        S.kind = k;
        refit();
        draw();
      });
      const re = () => {
        gen();
        refit();
        draw();
      };
      $('#sp-f', root).onchange = (e) => {
        S.f = e.target.value;
        re();
      };
      bind(root, 'sp-s', (v) => {
        S.noise = v;
        re();
      });
      bind(
        root,
        'sp-n',
        (v) => {
          S.n = v;
          re();
        },
        (v) => String(v),
      );
      $('#sp-new', root).onclick = () => {
        seed++;
        re();
      };
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'additive',
    tab: 'Additive models',
    title: 'One smooth curve per input',
    lede: 'With several inputs, give each its own smooth curve and add them up. The curves are found by taking turns: hold one fixed, smooth what is left over against the other, and repeat. Watch the two curves settle.',
    stage: `<p class="cap" id="am-cap" role="status"></p>
 <div class="stage" style="grid-template-columns:minmax(0,1fr) minmax(0,1fr)"><figure class="fig"><figcaption>What is left for x₁ to explain, and its curve f₁</figcaption><canvas class="short" id="am-1" role="img" aria-label="Partial residuals against the first input with its fitted curve"></canvas></figure>
 <figure class="fig"><figcaption>What is left for x₂ to explain, and its curve f₂</figcaption><canvas class="short" id="am-2" role="img" aria-label="Partial residuals against the second input with its fitted curve"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="am-play" type="button">Run backfitting</button><button class="btn" id="am-step" type="button">One turn</button><button class="btn" id="am-reset" type="button">Start over</button></div>${stopBox('am', 'turns')}`)}
 ${grp('The model', slider('am-k', 'Flexibility of each curve (knots)', 0, 10, 1, 4))}
 ${grp('The data', `<label class="ctl" for="am-f"><span>True relationship</span><select id="am-f"><option value="a">Wave in x₁ plus bowl in x₂</option><option value="b">V shape plus cubic</option><option value="c">A product, x₁ × x₂ (not additive)</option></select></label>` + slider('am-s', 'Noise σ', 0, 1, 0.01, 0.3) + slider('am-r', 'Correlation between inputs', -0.9, 0.9, 0.01, 0) + slider('am-n', 'Number of points', 40, 400, 1, 160) + `<div class="row"><button class="btn" id="am-new" type="button">New random sample</button></div>`)}</div>
 <dl class="read">${rd('it', 'Turns taken')}${rd('tr', 'Train error (MSE)', 'hl3')}${rd('te', 'Test error (MSE)', 'hl2')}${rd('lin', 'Test error of a flat plane, for comparison')}${rd('s2', 'Noise level σ² (best possible)', 'hl')}</dl>`,
    tryit:
      'Press “Start over” and take one turn at a time. After the first turn f₁ is rough because f₂ is still flat. How many turns until nothing changes? Then choose the product x₁ × x₂ and see what an additive model cannot do.',
    math: String.raw`<h4>Intuition</h4><p>Splines and local regression bend a curve in one input. With many inputs, fitting a fully flexible surface needs an enormous amount of data, because the points thin out fast as dimensions are added. An additive model takes a middle road: each input gets its own free-form curve, but the curves only ever add. You lose interactions between inputs, and in exchange you get a model that needs far less data and can be read one picture at a time.</p>
 <h4>The model</h4>$$y=\alpha+f_1(x_1)+f_2(x_2)+\dots+f_p(x_p)+\varepsilon$$<p>Linear regression is the special case \(f_j(x_j)=\beta_jx_j\). Each \(f_j\) here is a natural cubic spline, though any smoother from the previous lessons would do. Since a constant can be moved between the curves freely, each \(f_j\) is centred to average zero and the level is carried by \(\alpha=\bar y\).</p>
 <h4>Backfitting</h4><p>If \(f_2\) were known, finding \(f_1\) would be a one-dimensional smoothing problem: subtract what \(f_2\) explains and smooth the remainder against \(x_1\). Neither is known, so take turns:</p>$$r_{i}^{(1)}=y_i-\alpha-f_2(x_{i2}),\qquad f_1\leftarrow\operatorname{smooth}\big(r^{(1)}\ \text{against}\ x_1\big)$$$$r_{i}^{(2)}=y_i-\alpha-f_1(x_{i1}),\qquad f_2\leftarrow\operatorname{smooth}\big(r^{(2)}\ \text{against}\ x_2\big)$$<p>and repeat until the curves stop moving. The dots in each picture are these <b>partial residuals</b>: the part of \(y\) not yet explained by the other input. This is coordinate descent, the same idea used for the lasso, with whole functions in place of single coefficients.</p>
 <h4>Why it takes several turns</h4><p>If the inputs are unrelated, one turn each is nearly enough. When they are correlated, part of the effect of \(x_2\) leaks into \(f_1\) on the first turn and has to be handed back on later turns. Raise the correlation and count the turns.</p>
 <h4>What it cannot do</h4><p>A product such as \(y=x_1x_2\) has no additive part at all: averaged over \(x_2\), the effect of \(x_1\) is zero. The curves come out flat and the error stays high. Interactions must be added by hand as extra terms \(f_{12}(x_1,x_2)\), or left to models that find them on their own, such as trees and neural networks.</p>`,
    q: 'An additive model fits well and f₁ is a clear U shape. What can you say that you could not say from a neural network with the same accuracy?',
    a: 'You can state the effect of x₁ directly: holding the other inputs fixed, the prediction falls and then rises as x₁ increases, by the amounts shown on the curve. That reading is exact for an additive model. A neural network has no such single curve, because the effect of x₁ may differ for every value of the other inputs.',
    init(root) {
      const c1 = $('#am-1', root),
        c2 = $('#am-2', root),
        cap = $('#am-cap', root),
        S = { K: 4, noise: 0.3, r: 0, n: 160, f: 'a' };
      const TF = {
          a: [(x) => Math.sin(3 * x), (x) => 2 * x * x - 2 / 3],
          b: [(x) => 2 * Math.abs(x) - 1, (x) => 2 * x * x * x],
          c: [(x) => 0, (x) => 0],
        },
        gen2 = (a, b) => (S.f === 'c' ? 2.2 * a * b : TF[S.f][0](a) + TF[S.f][1](b));
      let seed = 6,
        pts = [],
        al = 0,
        B = [null, null],
        off = [0, 0],
        G = [new Array(61).fill(0), new Array(61).fill(0)],
        Gp = [G[0].slice(), G[1].slice()],
        u = 1,
        turn = 0,
        wait = 0,
        fr = 0,
        lin = 0,
        delta = 1,
        pb;
      const fj = (j, x) => (B[j] ? dot(B[j], splineBasis('natural', x, S.K)) - off[j] : 0);
      const gen = () => {
        const r = rng(seed),
          rn = randnFrom(r),
          q = Math.sqrt(1 - S.r * S.r);
        pts = [];
        for (let i = 0; i < S.n; i++) {
          const a = r() * 2 - 1,
            b = clamp(S.r * a + q * (r() * 2 - 1), -1, 1);
          pts.push({ x: [a, b], y: gen2(a, b) + S.noise * rn(), test: r() < 0.3 });
        }
        const tr = pts.filter((p) => !p.test),
          te = pts.filter((p) => p.test);
        al = tr.reduce((s, p) => s + p.y, 0) / tr.length;
        const bl = lsq(
          tr.map((p) => [1, p.x[0], p.x[1]]),
          tr.map((p) => p.y),
          1e-9,
        );
        lin = te.length
          ? te.reduce((s, p) => s + (p.y - bl[0] - bl[1] * p.x[0] - bl[2] * p.x[1]) ** 2, 0) /
            te.length
          : NaN;
      };
      const reset = () => {
        B = [null, null];
        off = [0, 0];
        G = [new Array(61).fill(0), new Array(61).fill(0)];
        Gp = [G[0].slice(), G[1].slice()];
        u = 1;
        turn = 0;
        wait = 0;
        delta = 1;
      };
      const takeTurn = () => {
        const j = turn % 2,
          o = 1 - j,
          tr = pts.filter((p) => !p.test),
          res = tr.map((p) => p.y - al - fj(o, p.x[o]));
        B[j] = lsq(
          tr.map((p) => splineBasis('natural', p.x[j], S.K)),
          res,
          1e-8,
        );
        off[j] = 0;
        off[j] = tr.reduce((s, p) => s + fj(j, p.x[j]), 0) / tr.length;
        Gp[j] = G[j].slice();
        G[j] = G[j].map((_, i) => fj(j, -1 + i / 30));
        delta = Math.max(...G[j].map((v, i) => Math.abs(v - Gp[j][i])));
        turn++;
        u = 0;
      };
      gen();
      const tp = testPoint(
        root,
        'am',
        ['Your x₁', 'Your x₂'],
        [0.5, -0.5],
        [
          [-1, 1],
          [-1, 1],
        ],
      );
      tp.on = () => draw();
      const draw = () => {
        const t = T(),
          e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u),
          act = turn ? (turn - 1) % 2 : -1;
        [c1, c2].forEach((c, j) => {
          const o = 1 - j,
            { g, w, h } = fit(c),
            YR = [-2.6, 2.6],
            P = plot(g, w, h, [-1.05, 1.05], YR, t, {
              yticks: [-2, -1, 0, 1, 2],
              xstep: 0.5,
              xlabel: j ? 'x₂' : 'x₁',
            });
          clipPlot(g, P);
          g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
          for (const p of pts)
            if (!p.test) {
              g.beginPath();
              g.arc(P.X(p.x[j]), P.Y(p.y - al - fj(o, p.x[o])), 2.8, 0, 7);
              g.fill();
            }
          if (S.f !== 'c') {
            g.setLineDash([6, 5]);
            curve(g, P, TF[S.f][j], -1, 1);
            g.strokeStyle = t.muted;
            g.lineWidth = 1.5;
            g.stroke();
            g.setLineDash([]);
          }
          g.beginPath();
          for (let i = 0; i <= 60; i++) {
            const v = j === act ? Gp[j][i] + (G[j][i] - Gp[j][i]) * e : G[j][i],
              x = P.X(-1 + i / 30),
              y = P.Y(clamp(v, -9, 9));
            i ? g.lineTo(x, y) : g.moveTo(x, y);
          }
          g.strokeStyle = j === act ? t.gold : t.ok;
          g.lineWidth = j === act ? 3.6 : 2.6;
          g.stroke();
          markTP(g, P.X(tp.v[j]), P.Y(clamp(fj(j, tp.v[j]), -9, 9)), t, false);
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = j === act ? t.cut : t.muted;
          g.fillText(j === act ? 'being updated' : 'held fixed', P.m.l + 8, P.m.t + 8);
        });
        const tr = pts.filter((p) => !p.test),
          te = pts.filter((p) => p.test),
          er = (P) =>
            P.length
              ? P.reduce((s, p) => s + (p.y - al - fj(0, p.x[0]) - fj(1, p.x[1])) ** 2, 0) /
                P.length
              : NaN;
        cap.textContent = !turn
          ? 'Both curves start flat. The dots are the data with only the overall average removed.'
          : 'Turn ' +
            turn +
            ': f' +
            (act ? '₂' : '₁') +
            ' was just refitted to what f' +
            (act ? '₁' : '₂') +
            ' leaves unexplained.' +
            (delta < 0.004 ? ' The curves have stopped moving: backfitting has converged.' : '');
        {
          const a1 = fj(0, tp.v[0]),
            a2 = fj(1, tp.v[1]);
          tp.out.textContent =
            'At x₁ = ' +
            f2(tp.v[0]) +
            ', x₂ = ' +
            f2(tp.v[1]) +
            ' the model predicts y = average + f₁ + f₂ = ' +
            f2(al) +
            ' + (' +
            f2(a1) +
            ') + (' +
            f2(a2) +
            ') = ' +
            f2(al + a1 + a2) +
            '. The true value is ' +
            f2(gen2(tp.v[0], tp.v[1])) +
            '.';
        }
        setR(root, 'it', String(turn));
        setR(root, 'tr', f3(er(tr)));
        setR(root, 'te', f3(er(te)));
        setR(root, 'lin', f3(lin));
        setR(root, 's2', f3(S.noise * S.noise));
      };
      const anim = Anim(() => {
        fr++;
        if (u < 1) {
          u = Math.min(1, u + 0.03);
          draw();
          return;
        }
        if (wait > 0) {
          wait--;
          if (!wait) {
            reset();
            draw();
          }
          return;
        }
        if (fr % 50) return;
        if ((turn >= 2 && delta < 0.004) || turn >= 30) {
          fin = true;
          pb.stop();
          return;
        }
        takeTurn();
        draw();
        chk();
      });
      let fin = false,
        chk = () => false;
      pb = playBtn($('#am-play', root), anim);
      chk = runCtl(pb, root, 'am', {
        done: () => fin,
        restart: () => {
          fin = false;
          reset();
        },
        count: () => turn,
        what: 'turns',
      });
      $('#am-step', root).onclick = () => {
        pb.stop();
        takeTurn();
        u = 1;
        draw();
      };
      $('#am-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      const re = () => {
        gen();
        reset();
        draw();
      };
      bind(
        root,
        'am-k',
        (v) => {
          S.K = v;
          reset();
          draw();
        },
        (v) => String(v),
      );
      $('#am-f', root).onchange = (e) => {
        S.f = e.target.value;
        re();
      };
      bind(root, 'am-s', (v) => {
        S.noise = v;
        re();
      });
      bind(root, 'am-r', (v) => {
        S.r = v;
        re();
      });
      bind(
        root,
        'am-n',
        (v) => {
          S.n = v;
          re();
        },
        (v) => String(v),
      );
      $('#am-new', root).onclick = () => {
        seed++;
        re();
      };
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'regularize',
    tab: 'Ridge, Lasso and Elastic Net',
    title: 'Keeping the coefficients small',
    lede: 'Start with a degree-10 polynomial that overfits. Now charge a price for large coefficients. As the price λ rises the curve calms down: ridge shrinks every coefficient, lasso switches some off entirely.',
    stage: `<div class="stage" style="grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);margin-bottom:30px"><figure class="fig"><figcaption>What the penalty does to one coefficient</figcaption><canvas class="sq" id="rg-s1" role="img" aria-label="Coefficient after the penalty against coefficient before, for ridge and lasso"></canvas></figure>
 <figure class="fig"><figcaption>On a problem: predicting an exam score from 12 measurements, of which only 3 matter (simulated)</figcaption><canvas class="sq" id="rg-ap" role="img" aria-label="True, least-squares, ridge and lasso coefficients for twelve named features"></canvas></figure></div>
 <div class="stage"><figure class="fig"><figcaption>The data and the degree-10 fit</figcaption><canvas class="short" id="rg-c" role="img" aria-label="Scatter plot with a regularized polynomial fit"></canvas></figure>
 <figure class="fig"><figcaption>Each coefficient as the price λ changes</figcaption><canvas class="short" id="rg-p" role="img" aria-label="Coefficient paths against the regularization strength"></canvas></figure></div>
 <div class="stage" style="margin-top:26px"><figure class="fig"><figcaption>The ten coefficients now (outline = with no penalty)</figcaption><canvas class="short" id="rg-b" role="img" aria-label="Bar chart of the coefficients, shrinking or dropping to zero"></canvas></figure>
 <figure class="fig"><figcaption>Why: the norm ball meets the loss (two coefficients; drag the cross)</figcaption><canvas class="sq" id="rg-g" style="touch-action:none;cursor:move" role="img" aria-label="Loss contours touching the L1 or L2 norm ball"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="rg-play" type="button">Sweep λ</button><button class="btn" id="rg-new" type="button">New random sample</button></div>`)}
 ${grp('The penalty', slider('rg-l', 'Price λ (log scale)', 0, 59, 1, 20) + slider('rg-a', 'Mix: 0 = ridge, 1 = lasso', 0, 1, 0.05, 0) + `<div class="row seg" id="rg-kind"><button class="btn" type="button" data-a="0">Ridge</button><button class="btn" type="button" data-a="1">Lasso</button><button class="btn" type="button" data-a="0.5">Elastic Net</button></div>`)}
 ${grp('The data', slider('rg-s', 'Noise σ', 0, 0.8, 0.01, 0.3) + slider('rg-n', 'Number of points', 14, 120, 1, 26))}</div>
 <dl class="read">${rd('l', 'Price λ')}${rd('tr', 'Train error (MSE)', 'hl3')}${rd('te', 'Test error (MSE)', 'hl2')}${rd('nz', 'Coefficients still in use, of 10', 'hl')}${rd('n1', 'L1 norm ‖w‖₁ = Σ|wⱼ|')}${rd('nm', 'L2 norm ‖w‖₂ = √Σwⱼ²')}${rd('bl', 'λ with the lowest test error')}</dl>`,
    tryit:
      'Sweep λ under Ridge and watch the bars: they all shrink, but does any reach exactly zero? Switch to Lasso and sweep again. In the last picture, drag the cross and find where the lasso answer lands on an axis.',
    math: String.raw`<h4>Intuition</h4><p>An overfitted curve gets its wild swings from huge coefficients of opposite sign that almost cancel. So instead of forbidding high degrees, make large coefficients expensive. The model can still use them, but only where the data pays for it. One dial, \(\lambda\), sets the price: at zero you get the wild least-squares fit, at a very high price you get a flat line.</p>
 <h4>The penalised loss</h4>$$\min_{\mathbf w}\ \frac{1}{2n}\lVert\mathbf y-X\mathbf w\rVert^2+\lambda\Big(\alpha\lVert\mathbf w\rVert_1+\frac{1-\alpha}{2}\lVert\mathbf w\rVert_2^2\Big)$$<p>\(\alpha=0\) is <b>ridge</b> (squared penalty), \(\alpha=1\) is <b>lasso</b> (absolute-value penalty), anything between is <b>elastic net</b>. The features are standardised first so that the price is the same for every coefficient, and the intercept is not penalised.</p>
 <h4>What happens to a single coefficient</h4><p>When the features are uncorrelated and standardised, each coefficient can be handled on its own, and both penalties have a one-line effect on the least-squares value \(\hat w\) (the first picture):</p>$$w_{\text{ridge}}=\frac{\hat w}{1+\lambda},\qquad w_{\text{lasso}}=\operatorname{sign}(\hat w)\,\max\big(|\hat w|-\lambda,\ 0\big).$$<p>Ridge multiplies every coefficient by the same factor below one: large ones lose the most, and none ever arrives at zero. Lasso subtracts the same amount \(\lambda\) from every coefficient's size: large ones barely notice, and any coefficient smaller than \(\lambda\) is wiped out. The second picture shows the consequence on a problem with 12 measurements of which 9 are pure noise. Least squares gives every one of them a non-zero weight. Ridge makes all the weights smaller. Lasso removes most of the noise features altogether and reports a shorter model.</p>
 <h4>The two norms</h4>$$\lVert\mathbf w\rVert_1=\sum_j|w_j|,\qquad \lVert\mathbf w\rVert_2=\sqrt{\sum_jw_j^2}$$<p>Both measure the size of the coefficient vector. Points at the same \(L_2\) norm form a circle; points at the same \(L_1\) norm form a diamond with corners on the axes. Paying a price \(\lambda\) per unit of norm is the same as fitting as well as you can while staying inside a ball of some fixed size, where a larger \(\lambda\) means a smaller ball:</p>$$\min_{\mathbf w}\ \lVert\mathbf y-X\mathbf w\rVert^2\quad\text{subject to}\quad\lVert\mathbf w\rVert\le s.$$<p>The last picture shows this for two coefficients. The thin ellipses are the loss, lowest at the cross. The answer is where the smallest ellipse first touches the shaded ball. A circle has no corners, so the touching point is almost never on an axis: both coefficients shrink and neither is zero. A diamond sticks out furthest at its corners, so the ellipse very often touches a corner first, and at a corner one coefficient is exactly zero. Drag the cross around to see how large the set of positions is for which the diamond is hit at a corner.</p>
 <h4>Ridge shrinks</h4>$$\hat{\mathbf w}_{\text{ridge}}=(X^{\top}X+n\lambda I)^{-1}X^{\top}\mathbf y$$<p>Adding \(\lambda I\) makes the matrix safely invertible and pulls every coefficient toward zero in proportion to its size. Large ones are pulled hard, small ones barely, so none arrives at exactly zero. It is the MAP estimate under a Gaussian prior on the weights, the same pull toward a prior you saw in the Gaussian module.</p>
 <h4>Lasso selects</h4><p>For one coefficient with the others held fixed, the lasso solution is a soft threshold:</p>$$w_j\leftarrow\frac{S(\rho_j,\ \lambda\alpha)}{1+\lambda(1-\alpha)},\qquad S(\rho,\tau)=\operatorname{sign}(\rho)\,\max\big(|\rho|-\tau,\ 0\big),\qquad \rho_j=\tfrac1n\mathbf x_j^{\top}\big(\mathbf y-X_{-j}\mathbf w_{-j}\big)$$<p>The absolute-value penalty pulls with the same force however small the coefficient is, so once a feature's correlation with the residual \(|\rho_j|\) drops below \(\lambda\alpha\) the coefficient is set to exactly zero. Lasso does feature selection as a side effect. It is the MAP estimate under a Laplace prior. Cycling this update over the coefficients is coordinate descent, which is how the picture is computed.</p>
 <h4>Choosing λ</h4><p>Training error always rises with \(\lambda\). Test error falls first, as variance is removed, then rises as the model becomes too stiff. The best \(\lambda\) sits at the bottom of that curve and is found on held-out data.</p>`,
    q: 'Two features are almost identical copies of each other. What do ridge and lasso each do with them?',
    a: 'Ridge splits the weight evenly between the two. Lasso tends to keep one and set the other to zero, and which one it keeps can change between samples. Elastic net was designed for this case: it selects like lasso but keeps correlated features together.',
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
        const mse = (w) =>
            te.y.reduce((s, v, i) => s + (v - dot(w, te.X[i])) ** 2, 0) / te.y.length,
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
          w[0] =
            sft(TA[0] * TC2[0] - TA[1] * (w[1] - TC2[1]), lam * S.a) / (TA[0] + lam * (1 - S.a));
          w[1] =
            sft(TA[2] * TC2[1] - TA[1] * (w[0] - TC2[0]), lam * S.a) / (TA[2] + lam * (1 - S.a));
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
              S.a * (Math.abs(v[0]) + Math.abs(v[1])) +
              ((1 - S.a) / 2) * (v[0] * v[0] + v[1] * v[1]),
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
  },

  {
    id: 'local',
    tab: 'Kernel and local regression',
    title: 'Let the neighbours vote',
    lede: 'No formula for the whole curve. To predict at one spot, look only at the points near it and weight them by closeness. Slide the spot along and the predictions trace out a curve. The bandwidth decides who counts as a neighbour.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Predicting at x₀ from its neighbours</figcaption><canvas class="short" id="lr-c" role="img" aria-label="Scatter plot with kernel weights and a local fit"></canvas></figure>
 <figure class="fig"><figcaption>Error against bandwidth</figcaption><canvas class="short" id="lr-e" role="img" aria-label="Training and leave-one-out error for each bandwidth"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="lr-play" type="button">Slide x₀</button><button class="btn" id="lr-new" type="button">New random sample</button></div>${slider('lr-x', 'Position x₀', 0, 10, 0.02, 2)}`)}
 ${grp('The method', slider('lr-h', 'Bandwidth h', 0.12, 3.5, 0.01, 0.7) + `<label class="ctl" for="lr-k"><span>Weighting kernel</span><select id="lr-k"><option value="gauss">Gaussian (smooth)</option><option value="epan">Epanechnikov (dome)</option><option value="box">Box (in or out)</option></select></label><label class="ctl" for="lr-m"><span>Fit near x₀</span><select id="lr-m"><option value="const">A weighted average (kernel regression)</option><option value="lin">A weighted line (LOESS)</option></select></label>`)}
 ${grp('The data', slider('lr-s', 'Noise σ', 0, 1.2, 0.01, 0.4) + slider('lr-n', 'Number of points', 15, 200, 1, 70))}</div>
 <dl class="read">${rd('y', 'Prediction at x₀', 'hl')}${rd('eff', 'Effective number of neighbours', 'hl3')}${rd('tr', 'Train error (MSE)', 'hl3')}${rd('cv', 'Leave-one-out error (MSE)', 'hl2')}${rd('bh', 'Bandwidth with the lowest leave-one-out error')}</dl>`,
    tryit:
      'Shrink the bandwidth to 0.15: the curve chases every point. Stretch it to 3: the curve goes nearly flat. Then switch from a weighted average to a weighted line and watch what changes at the two ends of the data.',
    math: String.raw`<h4>Intuition</h4><p>Every method so far chose a shape in advance, a line or a polynomial, and then fitted it. Here the data chooses the shape. To predict at \(x_0\), ask the training points nearby what their \(y\) values were and average them, listening more to the closest ones. There are no coefficients to learn: the training data itself is the model.</p>
 <h4>Kernel regression (Nadaraya–Watson)</h4>$$\hat f(x_0)=\frac{\sum_{i}K\!\big(\tfrac{x_i-x_0}{h}\big)\,y_i}{\sum_{i}K\!\big(\tfrac{x_i-x_0}{h}\big)}$$<p>\(K\) is a bump centred at zero, drawn along the bottom of the picture; each point's weight is the height of the bump above it. Common choices are the Gaussian \(K(u)=e^{-u^2/2}\), the Epanechnikov dome \(K(u)=\max(1-u^2,0)\) and the box \(K(u)=\mathbb 1[|u|\le1]\). The kernel matters little. The bandwidth \(h\) matters a great deal.</p>
 <h4>Bandwidth is the bias–variance dial</h4><p>A small \(h\) uses few neighbours: the estimate follows the data closely (low bias) but is noisy (high variance). A large \(h\) averages over many points: steady, but it flattens real bumps (high bias). In the limit \(h\to\infty\) every prediction is the overall mean. Roughly, variance scales like \(1/(nh)\) and squared bias like \(h^4\), so the best \(h\) shrinks slowly as data grows.</p>
 <h4>Local linear regression (LOESS)</h4>$$\big(\hat\beta_0,\hat\beta_1\big)=\arg\min_{\beta_0,\beta_1}\sum_iK\!\big(\tfrac{x_i-x_0}{h}\big)\big(y_i-\beta_0-\beta_1(x_i-x_0)\big)^2,\qquad \hat f(x_0)=\hat\beta_0$$<p>Fit a small weighted line near \(x_0\) and read off its height there. A weighted average is biased wherever the neighbours are lopsided, which always happens at the ends of the data, where all neighbours lie on one side. A local line corrects for the slope and removes that bias.</p>
 <h4>Choosing h without a test set</h4>$$\mathrm{CV}(h)=\frac1n\sum_{i=1}^n\big(y_i-\hat f_{-i}(x_i)\big)^2$$<p>Predict each point from all the <em>other</em> points and average the squared misses. Training error is useless for this choice, because it falls to zero as \(h\to0\) when each point simply predicts itself. Leave-one-out error has a real minimum, shown on the right.</p>
 <h4>Where this leads</h4><p>The same weights \(K\) reappear as the kernel in kernel ridge regression, Gaussian processes and support vector machines, and the weighted average is the ancestor of attention in transformers, where the weights are learned.</p>`,
    q: 'Why does training error go to zero as the bandwidth shrinks?',
    a: 'With a tiny bandwidth the only point with any weight near xᵢ is the point itself, so the prediction there is exactly yᵢ. That is memorising, and it says nothing about how the method does on new points. Leave-one-out error removes each point before predicting it, so it cannot cheat this way.',
    init(root) {
      const c = $('#lr-c', root),
        ce = $('#lr-e', root),
        S = { x0: 2, h: 0.7, k: 'gauss', m: 'const', noise: 0.4, n: 70 },
        Fq = (x) => Math.sin(x) + 0.5 * Math.sin(2.3 * x) + 0.12 * x,
        K = {
          gauss: (u) => Math.exp((-u * u) / 2),
          epan: (u) => Math.max(0, 1 - u * u),
          box: (u) => (Math.abs(u) <= 1 ? 1 : 0),
        },
        HS = [];
      for (let i = 0; i < 36; i++) HS.push(0.12 * Math.pow(3.5 / 0.12, i / 35));
      let seed = 4,
        pts = [],
        et = [],
        ec = [],
        pb;
      const loc = (x0, h, skip) => {
        let sw = 0,
          swx = 0,
          swy = 0,
          swxx = 0,
          swxy = 0,
          sw2 = 0;
        for (let i = 0; i < pts.length; i++) {
          if (i === skip) continue;
          const p = pts[i],
            w = K[S.k]((p.x - x0) / h);
          if (!w) continue;
          const d = p.x - x0;
          sw += w;
          swx += w * d;
          swy += w * p.y;
          swxx += w * d * d;
          swxy += w * d * p.y;
          sw2 += w * w;
        }
        if (sw < 1e-9) return null;
        let a = swy / sw,
          b = 0;
        if (S.m === 'lin') {
          const det = sw * swxx - swx * swx;
          if (det > 1e-9 * sw * sw) {
            b = (sw * swxy - swx * swy) / det;
            a = (swy - b * swx) / sw;
          }
        }
        return { a, b, eff: (sw * sw) / sw2 };
      };
      const gen = () => {
        const r = rng(seed),
          rn = randnFrom(r);
        pts = [];
        for (let i = 0; i < S.n; i++) {
          const x = r() * 10;
          pts.push({ x, y: Fq(x) + S.noise * rn() });
        }
      };
      const errs = () => {
        et = [];
        ec = [];
        for (const h of HS) {
          let a = 0,
            b = 0,
            nb = 0;
          pts.forEach((p, i) => {
            const f = loc(p.x, h, -1);
            a += (p.y - f.a) ** 2;
            const g = loc(p.x, h, i);
            if (g) {
              b += (p.y - g.a) ** 2;
              nb++;
            }
          });
          et.push(a / pts.length);
          ec.push(nb > pts.length * 0.8 ? b / nb : NaN);
        }
      };
      gen();
      errs();
      const draw = () => {
        const t = T(),
          f0 = loc(S.x0, S.h, -1);
        {
          const { g, w, h } = fit(c),
            YR = [-2.6, 3.6],
            P = plot(g, w, h, [0, 10], YR, t, { yticks: [-2, 0, 2], xlabel: 'x' });
          clipPlot(g, P);
          const base = P.Y(YR[0]);
          g.beginPath();
          g.moveTo(P.X(0), base);
          for (let i = 0; i <= 200; i++) {
            const x = i / 20;
            g.lineTo(P.X(x), base - K[S.k]((x - S.x0) / S.h) * (h * 0.16));
          }
          g.lineTo(P.X(10), base);
          g.closePath();
          g.fillStyle = mix(t.gold, t.gold, 0, 0.3);
          g.fill();
          g.setLineDash([6, 5]);
          curve(g, P, Fq, 0, 10);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.4;
          g.stroke();
          g.setLineDash([]);
          const tracer = (a, b, al, lw) => {
            g.beginPath();
            let on = false;
            for (let i = 0; i <= 260; i++) {
              const x = a + ((b - a) * i) / 260,
                f = loc(x, S.h, -1);
              if (!f) {
                on = false;
                continue;
              }
              on ? g.lineTo(P.X(x), P.Y(f.a)) : g.moveTo(P.X(x), P.Y(f.a));
              on = true;
            }
            g.strokeStyle = mix(t.ok, t.ok, 0, al);
            g.lineWidth = lw;
            g.stroke();
          };
          tracer(0, 10, 0.3, 1.6);
          tracer(0, S.x0, 1, 3);
          for (const p of pts) {
            const wv = K[S.k]((p.x - S.x0) / S.h);
            g.fillStyle = mix(t.accent, t.accent, 0, 0.22 + 0.78 * wv);
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 2.6 + 3.6 * wv, 0, 7);
            g.fill();
          }
          g.strokeStyle = mix(t.gold, t.gold, 0, 0.8);
          g.lineWidth = 1.3;
          g.beginPath();
          g.moveTo(P.X(S.x0), P.Y(YR[0]));
          g.lineTo(P.X(S.x0), P.Y(YR[1]));
          g.stroke();
          if (f0) {
            const hw = S.k === 'gauss' ? 1.6 * S.h : S.h;
            g.strokeStyle = t.gold;
            g.lineWidth = 3.4;
            g.beginPath();
            g.moveTo(P.X(S.x0 - hw), P.Y(f0.a - f0.b * hw));
            g.lineTo(P.X(S.x0 + hw), P.Y(f0.a + f0.b * hw));
            g.stroke();
            g.fillStyle = t.gold;
            g.strokeStyle = t.bg;
            g.lineWidth = 2;
            g.beginPath();
            g.arc(P.X(S.x0), P.Y(f0.a), 7, 0, 7);
            g.fill();
            g.stroke();
          }
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.muted;
          g.fillText('bigger dots = more weight, dashed = truth', P.m.l + 8, P.m.t + 8);
        }
        {
          const { g, w, h } = fit(ce),
            LX = [Math.log10(0.12), Math.log10(3.5)];
          let ym = 0;
          for (const v of ec) if (isFinite(v)) ym = Math.max(ym, v);
          for (const v of et) ym = Math.max(ym, v);
          ym = Math.min(ym * 1.1, 3) || 1;
          const P = plot(g, w, h, LX, [0, ym], t, { xstep: 100, xlabel: 'bandwidth h' });
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          [0.15, 0.3, 0.6, 1.2, 2.4].forEach((v) =>
            g.fillText(String(v), P.X(Math.log10(v)), h - P.m.b + 17),
          );
          clipPlot(g, P);
          g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
          g.lineWidth = 8;
          g.beginPath();
          g.moveTo(P.X(Math.log10(S.h)), P.Y(0));
          g.lineTo(P.X(Math.log10(S.h)), P.Y(ym));
          g.stroke();
          [
            [et, t.accent],
            [ec, t.ok],
          ].forEach(([E, col]) => {
            g.strokeStyle = col;
            g.lineWidth = 2.4;
            g.beginPath();
            let on = false;
            E.forEach((v, i) => {
              if (!isFinite(v)) {
                on = false;
                return;
              }
              const x = P.X(Math.log10(HS[i])),
                y = P.Y(Math.min(v, ym));
              on ? g.lineTo(x, y) : g.moveTo(x, y);
              on = true;
            });
            g.stroke();
          });
          g.restore();
          g.textAlign = 'left';
          g.fillStyle = t.accent;
          g.fillText('train', P.m.l + 8, P.m.t + 8);
          g.fillStyle = t.ok;
          g.fillText('leave-one-out', P.m.l + 8, P.m.t + 23);
        }
        let bi = -1;
        ec.forEach((v, i) => {
          if (isFinite(v) && (bi < 0 || v < ec[bi])) bi = i;
        });
        let a = 0,
          b = 0,
          nb = 0;
        pts.forEach((p, i) => {
          const f = loc(p.x, S.h, -1);
          a += (p.y - f.a) ** 2;
          const q = loc(p.x, S.h, i);
          if (q) {
            b += (p.y - q.a) ** 2;
            nb++;
          }
        });
        if (typeof tp !== 'undefined')
          tp.out.textContent =
            'At x = ' +
            f2(S.x0) +
            ' the neighbours predict y = ' +
            (f0 ? f2(f0.a) : 'nothing: no training point is close enough') +
            (f0
              ? ', using the equivalent of ' +
                f0.eff.toFixed(1) +
                ' points. The true curve there is ' +
                f2(Fq(S.x0)) +
                '.'
              : '.');
        setR(root, 'y', f0 ? f2(f0.a) : '–');
        setR(root, 'eff', f0 ? f0.eff.toFixed(1) : '0');
        setR(root, 'tr', f3(a / pts.length));
        setR(root, 'cv', nb > pts.length * 0.8 ? f3(b / nb) : 'too few neighbours');
        setR(root, 'bh', bi >= 0 ? f2(HS[bi]) : '–');
      };
      const xs = bind(root, 'lr-x', (v) => {
        S.x0 = v;
        draw();
      });
      const tp = testPoint(root, 'lr', ['Your x (moves x₀)'], [2], [[0, 10]]);
      tp.on = () => {
        pb.stop();
        S.x0 = tp.v[0];
        xs.set(S.x0);
        draw();
      };
      pb = playBtn($('#lr-play', root), sweeper(xs.el, 0, 10, 9));
      grab(xs.el, pb);
      const re = (d) => {
        if (d) gen();
        errs();
        draw();
      };
      bind(root, 'lr-h', (v) => {
        S.h = v;
        draw();
      });
      $('#lr-k', root).onchange = (e) => {
        S.k = e.target.value;
        re();
      };
      $('#lr-m', root).onchange = (e) => {
        S.m = e.target.value;
        re();
      };
      bind(root, 'lr-s', (v) => {
        S.noise = v;
        re(1);
      });
      bind(
        root,
        'lr-n',
        (v) => {
          S.n = v;
          re(1);
        },
        (v) => String(v),
      );
      $('#lr-new', root).onclick = () => {
        seed++;
        re(1);
      };
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },
];
