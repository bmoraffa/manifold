/* Manifold — Foundations.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ foundations ============ */
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
  const id = o.id,
    M = o.mode,
    sel = `<label class="ctl" for="${id}-f"><span>Loss surface</span><select id="${id}-f">${Object.keys(
      SURF,
    )
      .map((k) => `<option value="${k}"${k === o.surf ? ' selected' : ''}>${SURF[k].t}</option>`)
      .join('')}</select></label>`;
  const cases = grp(
    'Cases that go wrong',
    `<div class="row" id="${id}-cases">${Object.keys(CASES)
      .map((k) => `<button class="btn" type="button" data-case="${k}">${CASES[k].t}</button>`)
      .join('')}</div>`,
  );
  const ctl =
    M === 'gd'
      ? grp('The surface', sel) +
        grp(
          'The optimiser',
          slider(id + '-lr', 'Learning rate η', 0.005, 0.9, 0.005, 0.3) +
            slider(id + '-mo', 'Momentum β', 0, 0.98, 0.01, 0) +
            slider(id + '-no', 'Noise in each gradient (as in SGD)', 0, 2, 0.05, 0),
        ) +
        cases
      : M === 'opt'
        ? grp(
            'The surface',
            sel + slider(id + '-no', 'Noise in each gradient (mini-batches)', 0, 2, 0.05, 0.3),
          ) +
          grp(
            'The optimisers',
            ['sgd', 'mom', 'rms', 'adam']
              .map(
                (k) =>
                  `<label class="check" for="${id}-on-${k}"><input type="checkbox" id="${id}-on-${k}" checked>${OPT[k].n}</label>`,
              )
              .join('') +
              slider(id + '-lr', 'Learning rate: SGD and Momentum', 0.005, 0.9, 0.005, 0.1) +
              slider(id + '-lr2', 'Learning rate: RMSProp and Adam', 0.005, 0.6, 0.005, 0.08),
          ) +
          cases
        : M === 'newton'
          ? grp('The surface', sel) +
            grp(
              'The two optimisers',
              slider(id + '-lr', 'Gradient descent: learning rate η', 0.005, 0.9, 0.005, 0.1) +
                slider(id + '-da', 'Newton: fraction of the full step', 0.05, 1, 0.05, 1),
            )
          : M === 'sched'
            ? grp(
                'The schedule',
                `<label class="ctl" for="${id}-sc"><span>Learning rate over time</span><select id="${id}-sc"><option value="const">Constant</option><option value="step">Step decay</option><option value="exp">Exponential decay</option><option value="inv">1 / t decay</option><option value="cos" selected>Cosine</option><option value="warm">Warm-up, then cosine</option></select></label>` +
                  slider(id + '-lr', 'Starting learning rate η₀', 0.01, 0.9, 0.005, 0.35),
              ) +
              grp(
                'The problem',
                sel + slider(id + '-no', 'Noise in each gradient (as in SGD)', 0, 2, 0.05, 0.5),
              )
            : grp(
                'The features',
                slider(id + '-s', 'How different the two feature scales are', 1, 6, 0.1, 4) +
                  `<label class="check" for="${id}-std"><input type="checkbox" id="${id}-std">Standardise the features first</label>`,
              ) +
              grp('The optimiser', slider(id + '-lr', 'Learning rate η', 0.005, 0.9, 0.005, 0.25));
  const K0 =
    M === 'newton' ? ['gd', 'newton'] : M === 'opt' ? ['sgd', 'mom', 'rms', 'adam'] : ['gd'];
  return {
    ...o,
    stage: `<div class="row seg" id="${id}-dim" role="group" aria-label="How to draw the loss">${M === 'scale' ? '' : '<button class="btn" type="button" data-d="1">1D: A Curve</button>'}<button class="btn" type="button" data-d="2">2D: Contour Map</button><button class="btn" type="button" data-d="3">3D: Surface</button></div>
 <div class="stage" style="margin-top:12px"><figure class="fig"><figcaption id="${id}-fc">The loss</figcaption><canvas class="sq" id="${id}-c" style="cursor:crosshair;touch-action:none" role="img" aria-label="The loss with optimiser paths"></canvas></figure>
 <figure class="fig"><figcaption>${M === 'sched' ? 'Loss, and the learning rate, at each step' : 'Loss at each step'}</figcaption><canvas class="short" id="${id}-l" role="img" aria-label="Loss against step"></canvas><p class="status" id="${id}-note" role="status" style="color:var(--ink);margin-top:8px"></p></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="${id}-play" type="button">Run</button><button class="btn" id="${id}-step" type="button">One step</button><button class="btn" id="${id}-again" type="button">New start</button></div>${stopBox(id)}`)}${ctl}</div>
 <dl class="read">${rd('k', 'Steps taken')}${K0.map((k) => rd('L' + k, 'Loss: ' + OPT[k].n, OPT[k].c === 'gold' ? 'hl' : OPT[k].c === 'ok' ? 'hl2' : OPT[k].c === 'accent' ? 'hl3' : '')).join('')}${M === 'sched' ? rd('lr', 'Learning rate now') : ''}${M === 'scale' ? rd('kap', 'Condition number κ', 'hl3') + rd('mx', 'Largest stable learning rate', 'hl2') : ''}${rd('lo', 'Lowest possible loss')}</dl>`,
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

const FND = [
  gdLesson({
    id: 'gd',
    mode: 'gd',
    surf: 'banana',
    tab: 'Gradient descent',
    title: 'Walking downhill in the fog',
    lede: 'Training a model means finding the lowest point of a loss surface you cannot see all at once. Gradient descent feels the slope under its feet and takes a step downhill, again and again. Choose a surface, a step size and a starting point.',
    tryit:
      'Switch between the 1D, 2D and 3D pictures of the same descent. On the narrow valley, raise the learning rate until the path zig-zags, then until it diverges. Then press each of the five “cases that go wrong” and read what happened under the loss chart.',
    math: String.raw`<h4>Intuition</h4><p>You are on a hillside in thick fog and want to reach the valley floor. You cannot see the valley, but you can feel which way the ground slopes where you stand. So you take a step in the steepest downhill direction, feel again, and repeat. That is the whole algorithm. Everything else, step size, momentum, noise, is about how to take those steps well.</p>
 <h4>The update</h4>$$\mathbf w_{t+1}=\mathbf w_t-\eta\,\nabla L(\mathbf w_t)$$<p>The gradient \(\nabla L\) is the vector of slopes in each direction; it points uphill, so the step goes against it. The learning rate \(\eta\) sets the step length.</p>
 <h4>Why the learning rate has a limit</h4><p>Near a minimum the surface is a bowl with curvature \(\lambda\) in each direction. One step multiplies the distance to the bottom in that direction by \((1-\eta\lambda)\). That shrinks only if</p>$$|1-\eta\lambda|<1\quad\Longleftrightarrow\quad \eta<\frac{2}{\lambda_{\max}}.$$<p>The steepest direction sets the limit. In a narrow valley the walls are steep and the floor is nearly flat: a rate small enough for the walls is far too small for the floor, so progress along the valley is painfully slow. Just below the limit the path bounces from wall to wall.</p>
 <h4>Momentum</h4>$$\mathbf v_{t+1}=\beta\,\mathbf v_t-\eta\,\nabla L(\mathbf w_t),\qquad \mathbf w_{t+1}=\mathbf w_t+\mathbf v_{t+1}$$<p>Keep a running velocity, like a heavy ball rolling. Across the valley the gradient keeps flipping sign and cancels; along it the gradient keeps the same sign and builds up. With \(\beta=0.9\) the effective step along a steady direction is about ten times larger.</p>
 <h4>Noise: stochastic gradient descent</h4><p>Real training estimates the gradient from a small batch of data, so each step uses the true gradient plus noise. The path jitters and never settles exactly, but each step is far cheaper, and the noise can shake it out of shallow dips.</p>
 <h4>What it cannot promise</h4><p>Gradient descent only knows the local slope. On a surface with several valleys it ends in whichever one the starting point drains into.</p>`,
    q: 'The loss is going up at every step. What is the most likely cause?',
    a: 'The learning rate is above the limit 2 / λmax for the steepest direction, so each step overshoots the valley floor by more than it started with. Lowering the learning rate fixes it.',
  }),
  gdLesson({
    id: 'optim',
    mode: 'opt',
    surf: 'ravine',
    tab: 'SGD, Momentum, RMSProp and Adam',
    title: 'Four ways to take a step',
    lede: 'All four optimisers follow the same noisy gradient. They differ in what they remember. SGD remembers nothing. Momentum remembers its direction. RMSProp remembers how steep each weight has been. Adam remembers both. Race them on the same surface, in one, two or three dimensions.',
    tryit:
      'On the narrow valley, which optimiser gets along the valley floor fastest? Press “Lost on a Plateau”: which ones escape, and why can RMSProp and Adam move when the slope is nearly zero? Then press “Balanced on a Saddle”, and after they all stall, add a little noise and run again.',
    math: String.raw`<h4>Intuition</h4><p>Plain gradient descent has two weaknesses. In a narrow valley it bounces between the walls instead of running along the floor. And it takes steps in proportion to the slope, so where the ground is nearly flat it barely moves, and where it is steep it leaps. Each optimiser below fixes one or both. <b>Momentum</b> smooths the direction: a heavy ball that keeps rolling. <b>RMSProp</b> evens out the step size: it divides each weight's step by how large that weight's gradients have recently been, so every weight moves at a similar pace whatever its slope. <b>Adam</b> does both.</p>
 <h4>SGD</h4>$$\mathbf g_t=\nabla L_{\text{batch}}(\mathbf w_t),\qquad \mathbf w_{t+1}=\mathbf w_t-\eta\,\mathbf g_t$$<p>The gradient is estimated from a small random batch of data, so \(\mathbf g_t\) is the true gradient plus noise. Each step is cheap and the average direction is right. With a fixed \(\eta\) the noise never averages away, so SGD hovers around a minimum without settling.</p>
 <h4>Momentum</h4>$$\mathbf v_{t+1}=\beta\,\mathbf v_t-\eta\,\mathbf g_t,\qquad \mathbf w_{t+1}=\mathbf w_t+\mathbf v_{t+1},\qquad \beta=0.9$$<p>The velocity is a running sum of past gradients. Components that keep flipping sign cancel; components that agree add up, to as much as \(1/(1-\beta)=10\) times a single step. It also averages out noise. The price: it can overshoot and swing back.</p>
 <h4>RMSProp</h4>$$\mathbf s_{t+1}=\rho\,\mathbf s_t+(1-\rho)\,\mathbf g_t^{2},\qquad \mathbf w_{t+1}=\mathbf w_t-\frac{\eta}{\sqrt{\mathbf s_{t+1}}+\epsilon}\,\mathbf g_t,\qquad \rho=0.9$$<p>All operations are per weight. \(\mathbf s\) tracks the recent average of each weight's squared gradient. Dividing by its square root gives every weight its own learning rate: small for weights with steep, jumpy gradients, large for weights with gentle ones. On a plateau the gradient is tiny but so is \(\sqrt{\mathbf s}\), and the ratio is close to \(\pm1\): it keeps moving at a steady pace of about \(\eta\) per step.</p>
 <h4>Adam</h4>$$\mathbf m_{t+1}=\beta_1\mathbf m_t+(1-\beta_1)\mathbf g_t,\qquad \mathbf s_{t+1}=\beta_2\mathbf s_t+(1-\beta_2)\mathbf g_t^{2}$$$$\hat{\mathbf m}=\frac{\mathbf m_{t+1}}{1-\beta_1^{\,t+1}},\qquad \hat{\mathbf s}=\frac{\mathbf s_{t+1}}{1-\beta_2^{\,t+1}},\qquad \mathbf w_{t+1}=\mathbf w_t-\eta\,\frac{\hat{\mathbf m}}{\sqrt{\hat{\mathbf s}}+\epsilon}$$<p>with \(\beta_1=0.9\), \(\beta_2=0.999\). Momentum's running mean on top, RMSProp's scaling underneath. Both averages start at zero, so early values are too small; dividing by \(1-\beta^{t}\) corrects that. Each step has size about \(\eta\) whatever the scale of the loss, which is why one learning rate works across very different problems, and why Adam is the default for deep networks.</p>
 <h4>When they fail</h4><p><b>Too large a step:</b> SGD and momentum diverge when \(\eta\) exceeds \(2/\lambda_{\max}\) (momentum allows a little more). RMSProp and Adam cannot blow up this way, since their step is capped near \(\eta\), but with a large \(\eta\) they circle the minimum without landing. <b>Saddles and flat spots:</b> where the gradient is exactly zero, every one of them stops; only noise gets them moving again, which is one benefit of mini-batches. <b>Plateaus:</b> SGD and momentum crawl; the adaptive pair walk across. <b>Several valleys:</b> none of them knows about a deeper valley elsewhere. <b>Noise:</b> with a constant learning rate none of them settles exactly; that is what schedules are for.</p>`,
    q: 'Adam’s steps are about the same size whether the gradient is large or small. Why is that useful, and when is it a problem?',
    a: 'It is useful because the learning rate no longer has to be matched to the scale of the loss or to each layer, and flat regions are crossed at a steady pace. It is a problem near the minimum: the gradient shrinks but the step does not shrink with it until the running averages catch up, so Adam can keep circling. Lowering the learning rate over time fixes this.',
  }),
  gdLesson({
    id: 'newton',
    mode: 'newton',
    surf: 'ravine',
    tab: 'Newton’s method',
    title: 'Using the curvature as well as the slope',
    lede: 'Gradient descent knows only which way is down. Newton’s method also measures how the slope is changing, fits a bowl to the surface where it stands, and jumps straight to the bottom of that bowl. Watch the two race.',
    tryit:
      'On the narrow valley Newton arrives in one step while gradient descent crawls. Why? Now choose the saddle and start near the middle: where does Newton go, and is that a good place?',
    math: String.raw`<h4>Intuition</h4><p>Gradient descent takes the same length of step whether the valley floor is one stride away or a mile away, because the slope alone cannot tell the difference. Curvature can. If the slope is flattening quickly, the bottom is near, so take a short step. If the slope is barely changing, the bottom is far, so take a long one. Newton's method rescales the step in every direction by the curvature in that direction.</p>
 <h4>The step</h4><p>Approximate the loss near \(\mathbf w\) by its second-order Taylor expansion, a bowl:</p>$$L(\mathbf w+\mathbf d)\approx L(\mathbf w)+\nabla L^{\top}\mathbf d+\tfrac12\,\mathbf d^{\top}H\,\mathbf d,\qquad H_{ij}=\frac{\partial^2L}{\partial w_i\,\partial w_j}.$$<p>The bottom of that bowl is where its gradient vanishes:</p>$$\mathbf d=-H^{-1}\nabla L,\qquad \mathbf w_{t+1}=\mathbf w_t-H^{-1}\nabla L(\mathbf w_t).$$<p>Compare with gradient descent: the single number \(\eta\) has been replaced by the matrix \(H^{-1}\), a separate, automatically chosen learning rate for every direction.</p>
 <h4>What it buys</h4><p>If the loss really is a bowl (a quadratic), the approximation is exact and Newton lands on the minimum in one step, however stretched the bowl is. Near any minimum it converges quadratically: the number of correct digits roughly doubles each step.</p>
 <h4>What it costs</h4><p>With \(d\) parameters, \(H\) has \(d^2\) entries and inverting it costs about \(d^3\) operations. For a network with a million weights that is out of the question. Second, Newton seeks any point where the gradient is zero. Where the surface curves downward in some direction, \(H\) is not positive definite and the step heads for a saddle or a maximum. Practical methods (damped Newton, quasi-Newton such as L-BFGS, and Adam's per-weight scaling) keep the idea of rescaling by curvature and avoid both problems.</p>`,
    q: 'Why is plain Newton’s method not used to train neural networks?',
    a: 'Two reasons. The Hessian of a network with millions of weights is far too large to store or invert. And the loss surfaces of networks are full of saddle points, which Newton is attracted to. Methods that approximate curvature cheaply, per weight, are used instead.',
  }),
  gdLesson({
    id: 'sched',
    mode: 'sched',
    surf: 'banana',
    tab: 'Learning-rate schedules',
    title: 'Big steps first, small steps later',
    lede: 'With noisy gradients a large learning rate makes fast progress but never settles, and a small one settles but takes forever. A schedule starts large and shrinks. Compare the schedules on the same noisy problem.',
    tryit:
      'Run the constant schedule with noise 0.5 and watch the loss hover without settling. Switch to cosine. Then try exponential decay with a small starting rate: does it reach the bottom before the rate runs out?',
    math: String.raw`<h4>Intuition</h4><p>Think of parking a car. You drive fast to reach the street, slow down to find the space, and creep the last metre. One speed for all three would either take all day or put you through the wall. With noisy gradients the same holds: far from the minimum the signal is large compared with the noise, so big steps are safe; close to it the true gradient is nearly zero and each step is mostly noise, so the steps must shrink for the noise to average out.</p>
 <h4>Why a constant rate cannot settle</h4><p>With gradient noise of variance \(\sigma^2\), a constant rate \(\eta\) leaves the weights bouncing around the minimum with a spread proportional to \(\eta\sigma^2\). The loss levels off at a floor above the true minimum, and halving \(\eta\) halves the floor but doubles the time to get there.</p>
 <h4>Common schedules</h4>$$\text{step: }\ \eta_t=\eta_0\,\gamma^{\lfloor t/s\rfloor}\qquad \text{exponential: }\ \eta_t=\eta_0\,e^{-t/\tau}\qquad \text{inverse: }\ \eta_t=\frac{\eta_0}{1+t/\tau}$$$$\text{cosine: }\ \eta_t=\tfrac12\,\eta_0\Big(1+\cos\frac{\pi t}{T}\Big)\qquad \text{warm-up: }\ \eta_t=\eta_0\,\frac{t}{t_w}\ \text{ for }t<t_w$$<p>Cosine stays high for longer than exponential, then falls smoothly to zero at the end of the budget \(T\); it is the usual default for deep networks. Warm-up starts near zero for the first few steps, when the weights are random and gradients are unreliable, and is standard for transformers.</p>
 <h4>The theory</h4><p>For stochastic gradient descent to converge to the minimum exactly, the rates must satisfy</p>$$\sum_t\eta_t=\infty\qquad\text{and}\qquad\sum_t\eta_t^2<\infty .$$<p>The first says the steps never run out before you arrive; the second says the noise they inject adds up to a finite amount. \(\eta_t\propto1/t\) satisfies both. Exponential decay fails the first: it can stop short.</p>`,
    q: 'Training loss has stopped falling and jitters around a fixed level. What does that suggest about the learning rate?',
    a: 'It is probably too large for this stage. The weights are bouncing around the minimum with a spread set by the learning rate and the gradient noise. Lowering the rate, which is what a schedule does automatically, lets the loss drop to a lower floor.',
  }),
  gdLesson({
    id: 'scaling',
    mode: 'scale',
    tab: 'Feature scaling',
    title: 'Why units matter',
    lede: 'Measure one feature in millimetres and another in kilometres, and the loss surface becomes a long thin trench that gradient descent can barely get along. Rescale the features and the trench becomes a round bowl.',
    tryit:
      'Set the scale difference to 6 and run. Count the steps. Now tick “Standardise the features first” and run again with the same learning rate. Then, without standardising, raise the learning rate to the “largest stable” value and just past it.',
    math: String.raw`<h4>Intuition</h4><p>Suppose a house-price model uses floor area in square metres (values around 100) and number of bedrooms (values around 3). A small change in the weight on area changes the predictions thirty times more than the same change in the weight on bedrooms. The loss is therefore extremely sensitive to one weight and sluggish in the other: a trench. The learning rate must be tiny to survive the steep walls, and at that rate the walk along the trench floor takes forever. The model has not changed, only the units, yet training is many times slower.</p>
 <h4>The condition number</h4><p>For a linear model the curvature of the loss in each weight direction is set by the size of that feature: the Hessian is \(H=\tfrac2nX^{\top}X\). Its largest and smallest eigenvalues give the condition number</p>$$\kappa=\frac{\lambda_{\max}}{\lambda_{\min}}.$$<p>If one feature is \(r\) times larger than another, \(\kappa\approx r^2\). Gradient descent needs \(\eta<2/\lambda_{\max}\), and with the best fixed rate the error shrinks per step by only</p>$$\frac{\kappa-1}{\kappa+1},$$<p>so the number of steps grows in proportion to \(\kappa\). A scale ratio of 6 means roughly 36 times more steps.</p>
 <h4>Standardising</h4>$$z_j=\frac{x_j-\mu_j}{\sigma_j}$$<p>Subtract each feature's mean and divide by its standard deviation, using values computed on the training set only. Every feature then has mean 0 and spread 1, the bowl is round (\(\kappa\) near 1 when features are uncorrelated), and one learning rate suits every direction.</p>
 <h4>Where else it matters</h4><p>Any method that measures distance (k-nearest neighbours, k-means, kernels, SVMs) is dominated by the feature with the largest numbers unless features are scaled. Ridge and lasso penalise all coefficients equally, which is only fair on a common scale. Trees are the main exception: they only compare values within one feature, so units do not matter.</p>`,
    q: 'Should the test set be standardised with its own mean and standard deviation?',
    a: 'No. Use the mean and standard deviation computed on the training set. The scaling is part of the model, and anything computed from the test set leaks information about it. It also keeps a test point’s features on exactly the scale the model was trained on.',
  }),

  {
    id: 'biasvar',
    tab: 'Bias and variance',
    title: 'Two ways to be wrong, and one you cannot fix',
    lede: 'Every prediction error is made of three parts. Bias: the model is too stiff to follow the truth. Variance: the model is so flexible that it changes with every new batch of data. Noise: the world itself is not perfectly predictable, and no model can remove that part. Making a model more flexible trades the first for the second. The third stays put.',
    stage: `<div class="row seg" id="bv-pre" role="group" aria-label="Situations to look at"><button class="btn" type="button" data-v="simple">Too Stiff</button><button class="btn" type="button" data-v="best">Just Right</button><button class="btn" type="button" data-v="flex">Too Flexible</button><button class="btn" type="button" data-v="quiet">Low Noise</button><button class="btn" type="button" data-v="loud">High Noise</button></div>
 <p class="cap" id="bv-cap"></p>
 <div class="stage"><figure class="fig"><figcaption>Each thin line is the same model fitted to a fresh batch of data. Gold is the average of all such fits. Dashed is the truth</figcaption><canvas class="short" id="bv-c" role="img" aria-label="Truth, noise band, repeated fits and their average"></canvas></figure>
 <figure class="fig"><figcaption>Top: the expected error on new data at this flexibility, split into its three parts. Bottom: the three parts at every flexibility</figcaption><canvas class="sq" id="bv-r" role="img" aria-label="Error decomposition bar and the bias-variance curves against flexibility"></canvas></figure></div>
 <div class="howto"><div><h3>How to read it, step by step</h3><ol id="bv-steps"><li>The world: a true curve, plus noise on every measurement. The grey band is where measurements land.</li><li>Collect one batch of data and fit the model to it. That is one thin line.</li><li>Collect a fresh batch and fit again. The line comes out different.</li><li>Bias is the gap between the average line (gold) and the truth. Variance is how far the thin lines scatter around the gold one.</li><li>A new measurement also carries its own noise. Expected error = bias² + variance + noise.</li></ol></div><div><h3>What just happened</h3><p class="live" id="bv-live" role="status"></p></div></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="bv-play" type="button">Draw new batches</button><button class="btn" id="bv-step" type="button">One batch</button><button class="btn" id="bv-reset" type="button">Start over</button></div>${stopBox('bv', 'batches')}<div class="row"><button class="btn" id="bv-sweep" type="button">Sweep the flexibility for me</button></div>`)}
 ${grp('The model', slider('bv-d', 'Flexibility (degree of the polynomial)', 0, 12, 1, 5))}
 ${grp('The world', `<label class="ctl" for="bv-f"><span>True curve</span><select id="bv-f"><option value="wave">A wave</option><option value="bump">A single bump</option><option value="line">A straight line</option></select></label>` + slider('bv-s', 'Irreducible noise σ', 0, 1.2, 0.01, 0.3) + slider('bv-n', 'Points in each batch', 15, 80, 1, 25))}</div>
 <dl class="read">${rd('b', 'Bias²', 'hl2')}${rd('v', 'Variance', 'hl3')}${rd('s', 'Noise σ² (cannot be reduced)')}${rd('t', 'Expected error on new data', 'hl')}${rd('best', 'Best flexibility for this world')}${rd('k', 'Batches drawn')}</dl>`,
    tryit:
      'Press Too Stiff, then Too Flexible, and each time read the bar: which colour takes over? Press Sweep the Flexibility and watch the red and green parts trade places while the grey part never moves. Then press High Noise. Three things happen at once: the grey floor rises, the green curve gets steeper, and the best flexibility moves left. Why does more noise call for a simpler model?',
    math: String.raw`<h4>Intuition</h4><p>Picture an archer shooting many arrows at a target. If the arrows land in a tight group but off to one side, the archer has <b>bias</b>: a consistent error that more arrows will not fix. If they are centred on the bullseye but spread wide, that is <b>variance</b>: each shot depends too much on the moment. Now put the target on a boat that rocks. Even a perfect archer misses a little, because the target itself moves. That is <b>irreducible noise</b>. In machine learning each arrow is a model trained on one batch of data, the bullseye is the truth, and the rocking boat is everything about the outcome that the inputs do not explain.</p>
 <h4>The decomposition</h4><p>Suppose the world produces \(y=f(x)+\varepsilon\), with noise of mean 0 and variance \(\sigma^2\). Train a model on a random batch and call its prediction \(\hat f(x)\). Averaging over batches and over the noise in a new measurement at \(x\),</p>$$\mathbb E\big[(y-\hat f(x))^2\big]=\underbrace{\big(\mathbb E[\hat f(x)]-f(x)\big)^2}_{\text{bias}^2}+\underbrace{\mathbb E\Big[\big(\hat f(x)-\mathbb E[\hat f(x)]\big)^2\Big]}_{\text{variance}}+\underbrace{\sigma^2}_{\text{noise}} .$$
 <p>To see why, write \(y-\hat f=\varepsilon+(f-\bar f)+(\bar f-\hat f)\) with \(\bar f=\mathbb E[\hat f]\). The three pieces are uncorrelated: the new noise is independent of the training batch, and \(\bar f-\hat f\) has mean zero. So the squares add and the cross terms vanish.</p>
 <h4>Why this page can show it exactly</h4><p>The model here is a polynomial fitted by least squares, which is linear in the data: the prediction at any \(x\) is a fixed weighted sum of the batch's \(y\) values, \(\hat f(x)=\mathbf h(x)^{\top}\mathbf y\). Each batch uses the same \(x\) positions with fresh noise. Then nothing has to be estimated:</p>$$\mathbb E[\hat f(x)]=\mathbf h(x)^{\top}\mathbf f,\qquad\text{bias}(x)=\mathbf h(x)^{\top}\mathbf f-f(x),\qquad\text{variance}(x)=\sigma^2\,\lVert\mathbf h(x)\rVert^2 .$$<p>The curves in the lower picture are these formulas averaged over \(x\). The thin lines are simulations, and the second bar shows what you would measure from only the batches drawn so far. It closes in on the exact bar as batches accumulate, and the animation stops once the two agree.</p>
 <h4>The trade-off</h4><p>A polynomial of degree \(d\) has \(p=d+1\) coefficients. Averaged over the training positions, its variance is exactly</p>$$\overline{\text{variance}}=\sigma^2\,\frac{p}{n} .$$<p>Every extra coefficient adds \(\sigma^2/n\) of variance, whether or not it helps. Meanwhile bias² can only fall as \(d\) grows, quickly at first and then hardly at all once the model can follow the truth. The total is therefore U-shaped. Its lowest point is where one more coefficient would remove less bias than the \(\sigma^2/n\) of variance it costs.</p>
 <h4>What irreducible noise does</h4><p>Noise enters the total twice. First directly, as the floor \(\sigma^2\): the best possible model, the truth itself, still has this error, so no amount of data or tuning takes the test error below it. Second through the variance, which is proportional to \(\sigma^2\): noisier data make every fitted line wobble more. Bias does not depend on \(\sigma\) at all. So when noise is high, the variance curve rises more steeply while the bias curve stays where it was, and the bottom of the U moves toward simpler models. With very noisy data a straight line can beat the "correct" flexible model.</p>
 <h4>What more data does</h4><p>Variance is proportional to \(1/n\), so more data lowers the green curve and moves the best flexibility to the right. Bias is unaffected: a stiff model stays wrong however much data it sees. The noise floor is unaffected too. The only ways to lower \(\sigma^2\) are better measurements or new input variables that explain what is now treated as noise.</p>
 <h4>How this guides practice</h4><p>High training error and similar test error: bias dominates, so use a more flexible model or better features. Low training error and much higher test error: variance dominates, so get more data, simplify, regularise, or average many models (bagging). Both errors stuck at a level that no model improves on: you have probably reached the noise floor.</p>`,
    q: 'A team has tuned every model they can think of and the test error will not go below 4.0. They collect ten times more data and it stays at 4.0. What is the most likely explanation, and what could actually help?',
    a: 'They have reached the irreducible noise: with the inputs they have, outcomes that look identical to the model differ by about that much. More data only reduces variance, and tuning only trades bias against variance. The remaining options are to measure the target more accurately, or to find new input variables that explain part of what currently looks like noise.',
    init(root) {
      const c = $('#bv-c', root),
        cr = $('#bv-r', root),
        live = $('#bv-live', root),
        cap = $('#bv-cap', root),
        S = { d: 5, s: 0.3, n: 25, f: 'wave' },
        FN = {
          wave: (x) => Math.sin(4.6 * x) + 0.4 * x,
          bump: (x) => 1.7 * Math.exp(-9 * x * x) - 0.6,
          line: (x) => 0.9 * x,
        },
        NG = 81,
        DM = 12;
      let xs = [],
        fx = [],
        B2 = [],
        VA = [],
        Hg = [],
        avg = [],
        sd = [],
        Mc = null,
        fits = [],
        sum = null,
        sq = null,
        k = 0,
        last = null,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        sweep = null,
        best = 0;
      const leg = (x, p) => {
        const o = [1];
        if (p > 1) o.push(x);
        for (let j = 1; j < p - 1; j++) o.push(((2 * j + 1) * x * o[j] - j * o[j - 1]) / (j + 1));
        return o;
      };
      const inv = (A) => {
        const n = A.length,
          M = A.map((r, i) => r.concat(r.map((_, j) => (i === j ? 1 : 0))));
        for (let i = 0; i < n; i++) {
          let p = i;
          for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
          [M[i], M[p]] = [M[p], M[i]];
          const d = M[i][i] || 1e-12;
          for (let j = 0; j < 2 * n; j++) M[i][j] /= d;
          for (let r = 0; r < n; r++)
            if (r !== i) {
              const f = M[r][i];
              if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[i][j];
            }
        }
        return M.map((r) => r.slice(n));
      };
      const gx = (g) => -1 + (2 * g) / (NG - 1);
      const smoother = (d) => {
        const p = d + 1,
          n = xs.length,
          Bm = xs.map((x) => leg(x, p)),
          A = [];
        for (let a = 0; a < p; a++) {
          A.push([]);
          for (let b = 0; b < p; b++) {
            let s = 0;
            for (let i = 0; i < n; i++) s += Bm[i][a] * Bm[i][b];
            A[a].push(s + (a === b ? 1e-10 : 0));
          }
        }
        const Ai = inv(A),
          M = [];
        for (let a = 0; a < p; a++) {
          M.push(new Float64Array(n));
          for (let i = 0; i < n; i++) {
            let s = 0;
            for (let b = 0; b < p; b++) s += Ai[a][b] * Bm[i][b];
            M[a][i] = s;
          }
        }
        const H = [];
        for (let g = 0; g < NG; g++) {
          const ph = leg(gx(g), p),
            h = new Float64Array(n);
          for (let a = 0; a < p; a++) for (let i = 0; i < n; i++) h[i] += ph[a] * M[a][i];
          H.push(h);
        }
        return { M, H, p };
      };
      const design = () => {
        const r = rng(31 + S.n);
        xs = [];
        for (let i = 0; i < S.n; i++)
          xs.push(clamp(-1 + (2 * (i + 0.5)) / S.n + ((r() - 0.5) * 1.2) / S.n, -1, 1));
      };
      const comp = () => {
        const f = FN[S.f];
        fx = xs.map(f);
        B2 = [];
        VA = [];
        for (let d = 0; d <= DM; d++) {
          const sm = smoother(d);
          let b = 0,
            v = 0;
          for (let g = 0; g < NG; g++) {
            let m = 0,
              h2 = 0;
            const h = sm.H[g];
            for (let i = 0; i < xs.length; i++) {
              m += h[i] * fx[i];
              h2 += h[i] * h[i];
            }
            b += (m - f(gx(g))) ** 2;
            v += h2;
          }
          B2.push(b / NG);
          VA.push((S.s * S.s * v) / NG);
          if (d === S.d) {
            Mc = sm;
            avg = sm.H.map((h) => {
              let m = 0;
              for (let i = 0; i < xs.length; i++) m += h[i] * fx[i];
              return m;
            });
            sd = sm.H.map((h) => {
              let h2 = 0;
              for (let i = 0; i < xs.length; i++) h2 += h[i] * h[i];
              return S.s * Math.sqrt(h2);
            });
          }
        }
        best = 0;
        for (let d = 1; d <= DM; d++) if (B2[d] + VA[d] < B2[best] + VA[best]) best = d;
      };
      const tot = (d) => B2[d] + VA[d] + S.s * S.s;
      const verdict = () => {
        const b = B2[S.d],
          v = VA[S.d],
          n2 = S.s * S.s,
          T = b + v + n2;
        return (
          (S.d < best
            ? 'This model is too stiff for this world: bias² is ' +
              ((100 * b) / T).toFixed(0) +
              '% of the error. The gold average misses the truth however many batches you draw.'
            : S.d > best
              ? 'This model is too flexible for this much data and noise: variance is ' +
                ((100 * v) / T).toFixed(0) +
                '% of the error. The thin lines chase the noise in each batch.'
              : 'This is the best flexibility for this world: one step stiffer adds more bias than it saves in variance, and one step more flexible does the opposite.') +
          ' Noise accounts for ' +
          f3(n2) +
          ' of the ' +
          f3(T) +
          ', which is ' +
          ((100 * n2) / T).toFixed(0) +
          '%, and no model can remove it.'
        );
      };
      const reset = () => {
        fits = [];
        sum = new Float64Array(NG);
        sq = new Float64Array(NG);
        k = 0;
        last = null;
        cu = 0;
        msg = verdict() + ' Press Draw new batches to see it happen.';
        cap.textContent = verdict();
      };
      const emp = () => {
        if (k < 2) return null;
        let b = 0,
          v = 0;
        const f = FN[S.f];
        for (let g = 0; g < NG; g++) {
          const m = sum[g] / k;
          b += (m - f(gx(g))) ** 2;
          v += (sq[g] - k * m * m) / (k - 1);
        }
        b = Math.max(0, b / NG - v / NG / k);
        return { b, v: v / NG };
      };
      const add = () => {
        const n = xs.length,
          y = xs.map((x, i) => fx[i] + S.s * randn()),
          p = Mc.p,
          co = [];
        for (let a = 0; a < p; a++) {
          let s = 0;
          for (let i = 0; i < n; i++) s += Mc.M[a][i] * y[i];
          co.push(s);
        }
        const ft = new Float32Array(NG);
        let dev = 0;
        for (let g = 0; g < NG; g++) {
          const ph = leg(gx(g), p);
          let s = 0;
          for (let a = 0; a < p; a++) s += ph[a] * co[a];
          ft[g] = s;
          sum[g] += s;
          sq[g] += s * s;
          dev += (s - avg[g]) ** 2;
        }
        fits.push(ft);
        if (fits.length > 45) fits.shift();
        k++;
        last = { y, ft };
        cu = k < 2 ? 1 : k < 4 ? 2 : k < 8 ? 3 : 4;
        const e = emp();
        msg =
          'Batch ' +
          k +
          ': ' +
          n +
          ' new measurements, same model. This fit is on average ' +
          f2(Math.sqrt(dev / NG)) +
          ' away from the gold average.' +
          (e
            ? ' Measured from the ' +
              k +
              ' batches so far: bias² ' +
              f3(e.b) +
              ', variance ' +
              f3(e.v) +
              '. Exact values: ' +
              f3(B2[S.d]) +
              ' and ' +
              f3(VA[S.d]) +
              '.'
            : '');
      };
      const conv = () => {
        const e = emp();
        if (k >= 400) return true;
        if (!e || k < 40) return false;
        const T = B2[S.d] + VA[S.d];
        return (
          Math.abs(e.b + e.v - T) < 0.04 * T + 2e-4 &&
          Math.abs(e.v - VA[S.d]) < 0.06 * VA[S.d] + 2e-4
        );
      };
      const tp = testPoint(root, 'bv', ['Your x'], [0.4], [[-1, 1]]);
      const draw = () => {
        const t = T(),
          f = FN[S.f];
        {
          const { g, w, h } = fit(c),
            P = plot(g, w, h, [-1, 1], [-2.7, 2.7], t, {
              xstep: 0.5,
              yticks: [-2, 0, 2],
              xlabel: 'x',
            });
          clipPlot(g, P);
          g.beginPath();
          for (let i = 0; i < NG; i++) {
            const x = gx(i);
            i ? g.lineTo(P.X(x), P.Y(f(x) + S.s)) : g.moveTo(P.X(x), P.Y(f(x) + S.s));
          }
          for (let i = NG - 1; i >= 0; i--) {
            const x = gx(i);
            g.lineTo(P.X(x), P.Y(f(x) - S.s));
          }
          g.closePath();
          g.fillStyle = mix(t.ink, t.ink, 0, 0.1);
          g.fill();
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i
              ? g.lineTo(P.X(gx(i)), P.Y(avg[i] + sd[i]))
              : g.moveTo(P.X(gx(i)), P.Y(avg[i] + sd[i]));
          for (let i = NG - 1; i >= 0; i--) g.lineTo(P.X(gx(i)), P.Y(avg[i] - sd[i]));
          g.closePath();
          g.fillStyle = mix(t.accent, t.accent, 0, 0.16);
          g.fill();
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i ? g.lineTo(P.X(gx(i)), P.Y(avg[i])) : g.moveTo(P.X(gx(i)), P.Y(avg[i]));
          for (let i = NG - 1; i >= 0; i--) g.lineTo(P.X(gx(i)), P.Y(f(gx(i))));
          g.closePath();
          g.fillStyle = mix(t.ok, t.ok, 0, 0.3);
          g.fill();
          g.lineWidth = 1;
          fits.forEach((ft, q) => {
            g.strokeStyle = mix(t.accent, t.accent, 0, q === fits.length - 1 ? 0 : 0.3);
            g.beginPath();
            for (let i = 0; i < NG; i++)
              i ? g.lineTo(P.X(gx(i)), P.Y(ft[i])) : g.moveTo(P.X(gx(i)), P.Y(ft[i]));
            g.stroke();
          });
          g.setLineDash([7, 5]);
          g.strokeStyle = t.ink;
          g.lineWidth = 1.8;
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i ? g.lineTo(P.X(gx(i)), P.Y(f(gx(i)))) : g.moveTo(P.X(gx(i)), P.Y(f(gx(i))));
          g.stroke();
          g.setLineDash([]);
          g.strokeStyle = t.gold;
          g.lineWidth = 3.4;
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i ? g.lineTo(P.X(gx(i)), P.Y(avg[i])) : g.moveTo(P.X(gx(i)), P.Y(avg[i]));
          g.stroke();
          if (last) {
            g.strokeStyle = t.accent;
            g.lineWidth = 2.4;
            g.beginPath();
            for (let i = 0; i < NG; i++)
              i ? g.lineTo(P.X(gx(i)), P.Y(last.ft[i])) : g.moveTo(P.X(gx(i)), P.Y(last.ft[i]));
            g.stroke();
            g.fillStyle = t.ink;
            xs.forEach((x, i) => {
              g.beginPath();
              g.arc(P.X(x), P.Y(last.y[i]), 3, 0, 7);
              g.fill();
            });
          }
          {
            const x = tp.v[0],
              gi = Math.round(((x + 1) / 2) * (NG - 1)),
              px = P.X(x);
            g.strokeStyle = t.ink;
            g.lineWidth = 1.2;
            g.beginPath();
            g.moveTo(px, P.Y(-2.7));
            g.lineTo(px, P.Y(2.7));
            g.stroke();
            markTP(g, px, P.Y(avg[gi]), t, false);
          }
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          let ly = P.m.t + 8;
          [
            [mix(t.ink, t.ink, 0, 0.45), 'grey band: truth ± σ, the irreducible noise'],
            [t.ok, 'red area: bias, the gap between gold and truth'],
            [t.accent, 'green band and thin lines: variance'],
            [t.cut, 'gold: the average of all possible fits'],
          ].forEach((q) => {
            g.fillStyle = q[0];
            g.fillText(q[1], P.m.l + 8, ly);
            ly += 14;
          });
        }
        {
          const { g, w, h } = fit(cr),
            b = B2[S.d],
            v = VA[S.d],
            n2 = S.s * S.s,
            Tt = b + v + n2,
            e = emp(),
            x0 = 12,
            bw = w - 24,
            mx = Math.max(Tt, e ? e.b + e.v + n2 : 0) * 1.02 || 1;
          g.font = MONO;
          const bar = (y, hh, parts, lab) => {
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText(lab, x0, y - 5);
            let x = x0;
            parts.forEach((q) => {
              const ww = (q[0] / mx) * bw;
              g.fillStyle = q[1];
              g.fillRect(x, y, ww, hh);
              if (ww > 58) {
                g.fillStyle = t.bg;
                g.font = SANS;
                g.textAlign = 'center';
                g.fillText(q[2] + ' ' + f3(q[0]), x + ww / 2, y + hh / 2 + 5);
                g.font = MONO;
              }
              x += ww;
            });
          };
          bar(
            20,
            34,
            [
              [b, t.ok, 'bias²'],
              [v, t.accent, 'variance'],
              [n2, mix(t.ink, t.ink, 0, 0.5), 'noise'],
            ],
            'exact: bias² + variance + noise = ' + f3(Tt),
          );
          if (e)
            bar(
              78,
              16,
              [
                [e.b, t.ok, ''],
                [e.v, t.accent, ''],
                [n2, mix(t.ink, t.ink, 0, 0.5), ''],
              ],
              'measured from the ' + k + ' batches drawn so far = ' + f3(e.b + e.v + n2),
            );
          else {
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText('draw batches to measure it yourself', x0, 88);
          }
          const y0 = 112,
            m = { l: 40, r: 12, t: 14, b: 26 },
            ym = Math.max(n2 * 2.6, (B2[0] + n2) * 1.15, tot(best) * 2.2),
            X = (d) => m.l + (d / DM) * (w - m.l - m.r),
            Y = (u) => h - m.b - (Math.min(u, ym * 1.2) / ym) * (h - y0 - m.t - m.b);
          g.fillStyle = mix(t.ink, t.ink, 0, 0.1);
          g.fillRect(m.l, Y(n2), w - m.l - m.r, Y(0) - Y(n2));
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h - m.b);
          g.lineTo(w - m.r, h - m.b);
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          for (let d = 0; d <= DM; d += 2) g.fillText(String(d), X(d), h - m.b + 14);
          g.textAlign = 'right';
          g.fillText('more flexible →', w - m.r, h - 3);
          g.fillText(f2(ym), m.l - 4, Y(ym) + 4);
          g.fillText('0', m.l - 4, Y(0) + 4);
          g.save();
          g.beginPath();
          g.rect(m.l - 2, y0, w - m.l - m.r + 4, h - y0 - m.b + 2);
          g.clip();
          [
            [(d) => B2[d], t.ok, 2.4],
            [(d) => VA[d], t.accent, 2.4],
            [(d) => tot(d), t.ink, 3.2],
          ].forEach((q) => {
            g.strokeStyle = q[1];
            g.lineWidth = q[2];
            g.beginPath();
            for (let d = 0; d <= DM; d++)
              d ? g.lineTo(X(d), Y(q[0](d))) : g.moveTo(X(d), Y(q[0](d)));
            g.stroke();
          });
          g.setLineDash([5, 4]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.4;
          g.beginPath();
          g.moveTo(m.l, Y(n2));
          g.lineTo(w - m.r, Y(n2));
          g.stroke();
          g.setLineDash([]);
          g.restore();
          g.strokeStyle = t.gold;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(X(S.d), y0 + m.t);
          g.lineTo(X(S.d), h - m.b);
          g.stroke();
          g.fillStyle = t.gold;
          g.beginPath();
          g.arc(X(S.d), Y(tot(S.d)), 6, 0, 7);
          g.fill();
          g.strokeStyle = t.ink;
          g.lineWidth = 1.6;
          g.beginPath();
          g.arc(X(best), Y(tot(best)), 10, 0, 7);
          g.stroke();
          g.textAlign = 'left';
          g.fillStyle = t.muted;
          g.fillText('noise floor σ² = ' + f3(n2), m.l + 6, Math.min(h - m.b - 4, Y(n2) + 13));
          g.fillStyle = t.ink;
          g.fillText('black: total', X(4.3), y0 + m.t + 2);
          g.fillStyle = t.ok;
          g.fillText('red: bias²', X(4.3), y0 + m.t + 16);
          g.fillStyle = t.accent;
          g.fillText('green: variance', X(4.3), y0 + m.t + 30);
          g.fillStyle = t.ink;
          g.fillText('ring: lowest total', X(4.3), y0 + m.t + 44);
        }
        live.textContent = msg;
        root
          .querySelectorAll('#bv-steps li')
          .forEach((e, i) => e.setAttribute('aria-current', i === cu ? 'true' : 'false'));
        {
          const x = tp.v[0],
            gi = Math.round(((x + 1) / 2) * (NG - 1)),
            bb = avg[gi] - f(x);
          tp.out.textContent =
            'At x = ' +
            f2(x) +
            ' the truth is ' +
            f2(f(x)) +
            ' and the average fit is ' +
            f2(avg[gi]) +
            ', so the bias is ' +
            f2(bb) +
            '. Fits from different batches vary around that by ±' +
            f2(sd[gi]) +
            ', and a new measurement there varies by ±' +
            f2(S.s) +
            ' whatever the model. Expected squared error: ' +
            f3(bb * bb) +
            ' + ' +
            f3(sd[gi] * sd[gi]) +
            ' + ' +
            f3(S.s * S.s) +
            ' = ' +
            f3(bb * bb + sd[gi] * sd[gi] + S.s * S.s) +
            '.';
        }
        setR(root, 'b', f3(B2[S.d]));
        setR(root, 'v', f3(VA[S.d]));
        setR(root, 's', f3(S.s * S.s));
        setR(root, 't', f3(tot(S.d)));
        setR(root, 'best', 'degree ' + best);
        setR(root, 'k', String(k));
      };
      tp.on = () => draw();
      const redo = () => {
        comp();
        reset();
        draw();
      };
      design();
      comp();
      reset();
      for (let i = 0; i < 12; i++) add();
      const anim = Anim(() => {
        fr++;
        if (sweep) {
          if (fr % 34) return;
          if (sweep.d > DM) {
            sweep = null;
            pb.stop();
            S.d = best;
            db.set(best);
            comp();
            reset();
            msg = 'Sweep finished. The lowest total is at degree ' + best + ': ' + verdict();
            draw();
            return;
          }
          S.d = sweep.d++;
          db.set(S.d);
          comp();
          chk.rebase(0);
          reset();
          for (let i = 0; i < 10 && chk.remaining() > 0; i++) add();
          msg =
            'Degree ' +
            S.d +
            ': bias² ' +
            f3(B2[S.d]) +
            ', variance ' +
            f3(VA[S.d]) +
            ', noise ' +
            f3(S.s * S.s) +
            ', total ' +
            f3(tot(S.d)) +
            '.';
          draw();
          chk();
          return;
        }
        if (fr % (k < 6 ? 40 : k < 20 ? 12 : 4)) return;
        add();
        draw();
        if (conv()) {
          pb.stop();
          msg +=
            ' The measured parts now agree with the exact ones, so there is nothing more to learn from further batches. Stopped after ' +
            k +
            ' batches.';
          draw();
          return;
        }
        chk();
      });
      pb = playBtn($('#bv-play', root), anim);
      chk = runCtl(pb, root, 'bv', {
        done: () => !sweep && conv(),
        restart: () => reset(),
        count: () => k,
        what: 'batches',
      });
      $('#bv-step', root).onclick = () => {
        pb.stop();
        sweep = null;
        add();
        draw();
      };
      $('#bv-reset', root).onclick = () => {
        pb.stop();
        sweep = null;
        reset();
        draw();
      };
      $('#bv-sweep', root).onclick = () => {
        pb.stop();
        sweep = { d: 0 };
        pb.start();
      };
      const db = bind(
          root,
          'bv-d',
          (v) => {
            S.d = v;
            pb.stop();
            sweep = null;
            redo();
          },
          (v) => String(v),
        ),
        sb = bind(root, 'bv-s', (v) => {
          S.s = v;
          pb.stop();
          sweep = null;
          redo();
        });
      bind(
        root,
        'bv-n',
        (v) => {
          S.n = v;
          pb.stop();
          sweep = null;
          design();
          redo();
        },
        (v) => String(v),
      );
      $('#bv-f', root).onchange = (e) => {
        S.f = e.target.value;
        pb.stop();
        sweep = null;
        redo();
      };
      $('#bv-pre', root).addEventListener('click', (e) => {
        const v = e.target.dataset && e.target.dataset.v;
        if (!v) return;
        pb.stop();
        sweep = null;
        const was = best;
        if (v === 'quiet') {
          S.s = 0.12;
          sb.set(S.s);
        }
        if (v === 'loud') {
          S.s = 1;
          sb.set(S.s);
        }
        comp();
        if (v === 'simple') S.d = Math.min(1, Math.max(0, best - 2));
        if (v === 'flex') S.d = 11;
        if (v === 'best' || v === 'quiet' || v === 'loud') S.d = best;
        db.set(S.d);
        comp();
        reset();
        for (let i = 0; i < 12; i++) add();
        if (v === 'loud' || v === 'quiet')
          msg =
            'Noise set to σ = ' +
            f2(S.s) +
            '. The floor is now σ² = ' +
            f3(S.s * S.s) +
            ', and the variance curve scaled with it. The best flexibility ' +
            (best === was ? 'stayed at' : 'moved from degree ' + was + ' to') +
            ' degree ' +
            best +
            '. ' +
            verdict();
        cap.textContent = msg.length > 40 && (v === 'loud' || v === 'quiet') ? msg : verdict();
        draw();
      });
      return {
        draw,
        stop: () => {
          pb.stop();
          sweep = null;
        },
        auto: () => {
          sweep = null;
          reset();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'cv',
    tab: 'Validation and cross-validation',
    title: 'Testing on data the model has not seen',
    lede: 'A model always looks good on the data it was fitted to. To judge it honestly, hold some data back. Cross-validation does this several times, with each block of data taking a turn as the held-out part.',
    stage: `<div class="stage"><figure class="fig"><figcaption>This turn: fit on the filled points, score on the ringed ones</figcaption><canvas class="short" id="cv-c" role="img" aria-label="Data split into folds with the model fitted on the training folds"></canvas></figure>
 <figure class="fig"><figcaption>Top: error on each held-out block. Bottom: the estimate for every degree</figcaption><canvas class="short" id="cv-e" role="img" aria-label="Validation error per fold and cross-validation error against degree"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="cv-play" type="button">Rotate the folds</button><button class="btn" id="cv-new" type="button">New random sample</button></div>`)}
 ${grp('Cross-validation', slider('cv-k', 'Number of folds k', 2, 10, 1, 5) + slider('cv-d', 'Polynomial degree being judged', 1, 12, 1, 3))}${grp('The data', slider('cv-n', 'Number of points', 12, 150, 1, 40) + slider('cv-s', 'Noise σ', 0, 0.8, 0.01, 0.3))}</div>
 <dl class="read">${rd('cv', 'Cross-validation estimate', 'hl2')}${rd('sd', 'Spread across folds')}${rd('tr', 'Training error, for comparison', 'hl3')}${rd('tt', 'True test error (10,000 fresh points)', 'hl')}${rd('pick', 'Degree cross-validation would pick')}</dl>`,
    tryit:
      'Compare the cross-validation estimate with the true test error, and both with the training error. Which one is the liar? Then set k to 2 and to 10 and press “New random sample” a few times: which k gives the steadier estimate?',
    math: String.raw`<h4>Intuition</h4><p>A student who has seen the exam paper in advance will score well, and the score tells you nothing. To learn what they know, set questions they have not seen. A model is the same: its error on the training data is an exam it has already memorised. So hide part of the data during fitting and score the model on that. Cross-validation is the frugal version: every point gets used for fitting most of the time and for scoring exactly once.</p>
 <h4>Three sets, three jobs</h4><p><b>Training set:</b> fit the parameters. <b>Validation set:</b> compare models and choose settings such as the degree or \(\lambda\). <b>Test set:</b> used once, at the very end, to report how the chosen model will do. The third set is needed because choosing the best of many models on the validation set is itself a form of fitting, so the winner's validation score is optimistic.</p>
 <h4>k-fold cross-validation</h4><p>Split the data into \(k\) blocks. For each block in turn, fit on the other \(k-1\) and measure the error on that block:</p>$$\mathrm{CV}_k=\frac1k\sum_{j=1}^{k}\ \frac{1}{|B_j|}\sum_{i\in B_j}\big(y_i-\hat f^{(-j)}(x_i)\big)^2$$<p>where \(\hat f^{(-j)}\) was fitted without block \(j\). With \(k=n\) each block is a single point: leave-one-out.</p>
 <h4>Choosing k</h4><p>Small \(k\) trains on much less data than you really have, so the estimate is pessimistic (biased upward), and it depends heavily on how the split happened to fall. Large \(k\) trains on nearly everything, so bias is small, but it costs \(k\) fits. \(k=5\) or \(10\) is the usual compromise.</p>
 <h4>Using it</h4><p>Compute \(\mathrm{CV}_k\) for each candidate model and pick the lowest, as in the bottom chart. Then refit the winner on all the data. Any step that looks at the targets, including feature selection and scaling, must be done inside each fold, not before the split.</p>`,
    q: 'You try 200 models and report the best cross-validation score as the expected accuracy. What is wrong with that?',
    a: 'The best of 200 scores is partly luck: some model will fit the quirks of the validation folds by chance. The reported number is optimistic. Hold out a separate test set that plays no part in the selection and report the winner’s score on that.',
    init(root) {
      const c = $('#cv-c', root),
        ce = $('#cv-e', root),
        S = { k: 5, d: 3, n: 40, noise: 0.3 },
        Fq = (x) => Math.sin(2.8 * x);
      let seed = 14,
        pts = [],
        fold = 0,
        FE = [],
        CVD = [],
        TTD = [],
        TRD = [],
        bcur = null,
        pb;
      const fitOn = (P, d) =>
          lsq(
            P.map((p) => legendre(p.x, d)),
            P.map((p) => p.y),
            1e-8,
          ),
        er = (P, b, d) =>
          P.reduce((s, p) => s + (p.y - clamp(dot(b, legendre(p.x, d)), -20, 20)) ** 2, 0) /
          P.length;
      const cvFor = (d) => {
        const fe = [];
        for (let j = 0; j < S.k; j++) {
          const tr = pts.filter((p) => p.f !== j),
            va = pts.filter((p) => p.f === j);
          fe.push(va.length && tr.length > d ? er(va, fitOn(tr, d), d) : NaN);
        }
        return fe;
      };
      const gen = () => {
        const r = rng(seed),
          rn = randnFrom(r);
        pts = [];
        for (let i = 0; i < S.n; i++) {
          const x = r() * 2 - 1;
          pts.push({ x, y: Fq(x) + S.noise * rn() });
        }
        const ix = pts.map((_, i) => i);
        for (let i = ix.length - 1; i > 0; i--) {
          const j = Math.floor(r() * (i + 1));
          [ix[i], ix[j]] = [ix[j], ix[i]];
        }
        ix.forEach((q, i) => (pts[q].f = i % S.k));
        const big = [];
        for (let i = 0; i < 3000; i++) {
          const x = r() * 2 - 1;
          big.push({ x, y: Fq(x) + S.noise * rn() });
        }
        CVD = [];
        TTD = [];
        TRD = [];
        for (let d = 1; d <= 12; d++) {
          const fe = cvFor(d).filter(isFinite);
          CVD[d] = fe.length ? fe.reduce((a, b) => a + b, 0) / fe.length : NaN;
          const b = fitOn(pts, d);
          TTD[d] = er(big, b, d);
          TRD[d] = er(pts, b, d);
        }
      };
      const turn = () => {
        FE = cvFor(S.d);
        bcur = fitOn(
          pts.filter((p) => p.f !== fold),
          S.d,
        );
      };
      gen();
      turn();
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c),
            P = plot(g, w, h, [-1.05, 1.05], [-2.2, 2.6], t, {
              yticks: [-2, -1, 0, 1, 2],
              xstep: 0.5,
              xlabel: 'x',
            }),
            bw = (w - P.m.l - P.m.r) / S.k;
          for (let j = 0; j < S.k; j++) {
            g.fillStyle = j === fold ? t.ok : mix(t.accent, t.accent, 0, 0.55);
            g.fillRect(P.m.l + j * bw + 1, P.m.t - 4, bw - 2, 9);
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('blocks: burgundy is held out this turn', P.m.l, P.m.t + 19);
          clipPlot(g, P);
          g.setLineDash([6, 5]);
          curve(g, P, Fq, -1, 1);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.4;
          g.stroke();
          g.setLineDash([]);
          curve(g, P, (x) => clamp(dot(bcur, legendre(x, S.d)), -9, 9), -1, 1, 300);
          g.strokeStyle = t.gold;
          g.lineWidth = 3.2;
          g.stroke();
          for (const p of pts) {
            if (p.f === fold) {
              const yh = clamp(dot(bcur, legendre(p.x, S.d)), -9, 9);
              g.strokeStyle = mix(t.ok, t.ok, 0, 0.6);
              g.lineWidth = 1.2;
              g.beginPath();
              g.moveTo(P.X(p.x), P.Y(p.y));
              g.lineTo(P.X(p.x), P.Y(yh));
              g.stroke();
              g.strokeStyle = t.ok;
              g.lineWidth = 2.2;
              g.beginPath();
              g.arc(P.X(p.x), P.Y(p.y), 5, 0, 7);
              g.stroke();
            } else {
              g.fillStyle = mix(t.accent, t.accent, 0, 0.85);
              g.beginPath();
              g.arc(P.X(p.x), P.Y(p.y), 3.8, 0, 7);
              g.fill();
            }
          }
          g.restore();
        }
        {
          const { g, w, h } = fit(ce),
            h1 = h * 0.42,
            fe = FE.filter(isFinite),
            mean = fe.length ? fe.reduce((a, b) => a + b, 0) / fe.length : 0,
            mx = Math.max(0.05, ...fe) * 1.15,
            m = { l: 38, r: 12 };
          g.font = MONO;
          const bw = (w - m.l - m.r) / S.k,
            Yb = (v) => h1 - 12 - (Math.min(v, mx) / mx) * (h1 - 30);
          for (let j = 0; j < S.k; j++) {
            if (!isFinite(FE[j])) continue;
            g.fillStyle = j === fold ? t.ok : mix(t.ok, t.ok, 0, 0.35);
            g.fillRect(m.l + j * bw + 3, Yb(FE[j]), bw - 6, h1 - 12 - Yb(FE[j]));
          }
          g.strokeStyle = t.ink;
          g.lineWidth = 1.6;
          g.setLineDash([5, 4]);
          g.beginPath();
          g.moveTo(m.l, Yb(mean));
          g.lineTo(w - m.r, Yb(mean));
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.ink;
          g.textAlign = 'right';
          g.fillText('average = CV estimate ' + f3(mean), w - m.r, Math.max(10, Yb(mean) - 5));
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h1 - 12);
          g.lineTo(w - m.r, h1 - 12);
          g.stroke();
          g.save();
          g.translate(0, h1);
          const hh = h - h1,
            LY = [-2.3, 0.8],
            P = plot(g, w, hh, [0.5, 12.5], LY, t, { xlabel: 'degree' }),
            lg = (v) => clamp(Math.log10(Math.max(v, 1e-9)), LY[0], LY[1]);
          clipPlot(g, P);
          g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
          g.lineWidth = 9;
          g.beginPath();
          g.moveTo(P.X(S.d), P.Y(LY[0]));
          g.lineTo(P.X(S.d), P.Y(LY[1]));
          g.stroke();
          [
            [TRD, t.accent],
            [CVD, t.ok],
            [TTD, t.ink],
          ].forEach(([E, cl]) => {
            g.strokeStyle = cl;
            g.lineWidth = 2.2;
            g.beginPath();
            let on = false;
            for (let d = 1; d <= 12; d++) {
              if (!isFinite(E[d])) {
                on = false;
                continue;
              }
              on ? g.lineTo(P.X(d), P.Y(lg(E[d]))) : g.moveTo(P.X(d), P.Y(lg(E[d])));
              on = true;
            }
            g.stroke();
          });
          g.restore();
          g.textAlign = 'left';
          let lx = P.m.l + 8;
          [
            [t.accent, 'training'],
            [t.ok, 'cross-validation'],
            [t.ink, 'true test'],
          ].forEach(([cl, sx]) => {
            g.fillStyle = cl;
            g.fillText(sx, lx, P.m.t + 6);
            lx += g.measureText(sx).width + 16;
          });
          g.restore();
        }
        const fe = FE.filter(isFinite),
          mean = fe.reduce((a, b) => a + b, 0) / (fe.length || 1),
          sd = Math.sqrt(fe.reduce((a, b) => a + (b - mean) ** 2, 0) / (fe.length || 1));
        let pk = 1;
        for (let d = 2; d <= 12; d++) if (CVD[d] < CVD[pk]) pk = d;
        setR(root, 'cv', f3(mean));
        setR(root, 'sd', f3(sd));
        setR(root, 'tr', f3(TRD[S.d]));
        setR(root, 'tt', f3(TTD[S.d]));
        setR(root, 'pick', String(pk));
      };
      let fr = 0;
      const anim = Anim(() => {
        if (++fr % 75) return;
        fold = (fold + 1) % S.k;
        turn();
        draw();
      });
      pb = playBtn($('#cv-play', root), anim);
      const re = () => {
        fold = 0;
        gen();
        turn();
        draw();
      };
      bind(
        root,
        'cv-k',
        (v) => {
          S.k = v;
          re();
        },
        (v) => String(v),
      );
      bind(
        root,
        'cv-d',
        (v) => {
          S.d = v;
          turn();
          draw();
        },
        (v) => String(v),
      );
      bind(
        root,
        'cv-n',
        (v) => {
          S.n = v;
          re();
        },
        (v) => String(v),
      );
      bind(root, 'cv-s', (v) => {
        S.noise = v;
        re();
      });
      $('#cv-new', root).onclick = () => {
        seed++;
        re();
      };
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'metrics',
    tab: 'Classification metrics',
    title: 'One threshold, four outcomes',
    lede: 'A classifier gives every case a score. A threshold turns scores into yes or no, and every case lands in one of four boxes. Slide the threshold and watch what you gain and what you pay.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Scores of the two classes, and the threshold</figcaption><canvas class="short" id="mt-c" role="img" aria-label="Score distributions of positives and negatives with a threshold"></canvas></figure>
 <figure class="fig"><figcaption>Every threshold at once: ROC curve (left) and precision–recall curve (right)</figcaption><canvas class="short" id="mt-r" role="img" aria-label="ROC curve and precision recall curve with the current threshold marked"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="mt-play" type="button">Sweep the threshold</button></div>` + slider('mt-t', 'Threshold', -4, 4, 0.02, 0))}
 ${grp('The classifier', slider('mt-sep', 'How well it separates the classes', 0, 4, 0.05, 1.8))}${grp('The data', slider('mt-pi', 'Share of cases that are truly positive', 0.01, 0.99, 0.01, 0.3))}</div>
 <dl class="read">${rd('tp', 'True positives (hits)', 'hl')}${rd('fn', 'False negatives (misses)')}${rd('fp', 'False positives (false alarms)', 'hl2')}${rd('tn', 'True negatives')}</dl>
 <dl class="read">${rd('acc', 'Accuracy')}${rd('pre', 'Precision', 'hl2')}${rd('rec', 'Recall (true positive rate)', 'hl')}${rd('fpr', 'False positive rate')}${rd('f1', 'F1 score')}${rd('auc', 'Area under the ROC curve', 'hl3')}</dl>`,
    tryit:
      'Set the share of positives to 0.02, as for a rare disease, and put the threshold far to the right so the classifier says “no” to everyone. What is the accuracy? What is the recall? Then watch which of the two curves on the right changes when you change the share of positives.',
    math: String.raw`<h4>Intuition</h4><p>A smoke alarm can be wrong in two ways: it can stay silent in a fire, or go off when you make toast. Turning up its sensitivity trades one mistake for the other, and no setting removes both unless the alarm can tell smoke from toast perfectly. Every classifier has this dial. The right setting depends on what each mistake costs, which is a question about the world, not about the model.</p>
 <h4>The confusion matrix</h4><p>Out of every 1,000 cases the readouts count: <b>TP</b> positives correctly flagged, <b>FN</b> positives missed, <b>FP</b> negatives wrongly flagged, <b>TN</b> negatives correctly passed.</p>
 <h4>The rates</h4>$$\text{accuracy}=\frac{TP+TN}{\text{all}},\qquad \text{precision}=\frac{TP}{TP+FP},\qquad \text{recall}=\frac{TP}{TP+FN},\qquad \text{FPR}=\frac{FP}{FP+TN}$$$$F_1=\frac{2\cdot\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$$<p><b>Recall:</b> of the real positives, how many did I catch? <b>Precision:</b> of the cases I flagged, how many were real? \(F_1\) is their harmonic mean, which is high only when both are.</p>
 <h4>Why accuracy misleads</h4><p>If 2% of cases are positive, answering "no" every time is 98% accurate and catches nothing. With unbalanced classes look at precision and recall instead.</p>
 <h4>The ROC curve</h4><p>Plot recall against the false positive rate for every threshold. A useless classifier follows the diagonal; a perfect one hugs the top-left corner. The area under it has a direct meaning:</p>$$\mathrm{AUC}=P\big(\text{score of a random positive}>\text{score of a random negative}\big).$$<p>Both axes are rates within one class, so the ROC curve does not move when the share of positives changes. That makes it good for comparing classifiers and blind to class imbalance.</p>
 <h4>The precision–recall curve</h4><p>Precision mixes the two classes, so it does depend on the share of positives. When positives are rare, even a small false positive rate produces more false alarms than hits and precision collapses. The precision–recall curve shows this; the ROC curve hides it.</p>`,
    q: 'A screening test for a rare disease has 99% recall and a 5% false positive rate. A patient tests positive. Are they probably ill?',
    a: 'Probably not. If 1 in 1,000 people has the disease, then of 1,000 tested about 1 is a true positive and about 50 are false positives. Precision is about 2%. Set the share of positives to 0.01 and look at the precision readout.',
    init(root) {
      const c = $('#mt-c', root),
        cr = $('#mt-r', root),
        S = { t: 0, sep: 1.8, pi: 0.3 },
        Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2)),
        tpr = (th) => 1 - Phi(th - S.sep / 2),
        fpr = (th) => 1 - Phi(th + S.sep / 2);
      let pb;
      const draw = () => {
        const t = T(),
          TP = S.pi * tpr(S.t),
          FN = S.pi - TP,
          FP = (1 - S.pi) * fpr(S.t),
          TN = 1 - S.pi - FP,
          pre = TP + FP > 1e-9 ? TP / (TP + FP) : 1,
          rec = tpr(S.t);
        {
          const { g, w, h } = fit(c),
            pn = (x) => (1 - S.pi) * npdf(x, -S.sep / 2, 1),
            pp = (x) => S.pi * npdf(x, S.sep / 2, 1),
            ym = Math.max(pn(-S.sep / 2), pp(S.sep / 2)) * 1.25,
            P = plot(g, w, h, [-5, 5], [0, ym], t, { xlabel: 'score' });
          clipPlot(g, P);
          area(g, P, pn, S.t, 5);
          g.fillStyle = mix(t.ok, t.ok, 0, 0.5);
          g.fill();
          area(g, P, pp, -5, S.t);
          g.fillStyle = mix(t.muted, t.muted, 0, 0.4);
          g.fill();
          area(g, P, pp, S.t, 5);
          g.fillStyle = mix(t.gold, t.gold, 0, 0.45);
          g.fill();
          curve(g, P, pn, -5, 5);
          g.strokeStyle = t.accent;
          g.lineWidth = 2.4;
          g.stroke();
          curve(g, P, pp, -5, 5);
          g.strokeStyle = t.gold;
          g.lineWidth = 2.4;
          g.stroke();
          g.strokeStyle = t.ink;
          g.lineWidth = 2.4;
          g.beginPath();
          g.moveTo(P.X(S.t), P.Y(0));
          g.lineTo(P.X(S.t), P.Y(ym));
          g.stroke();
          g.restore();
          g.font = SANS;
          g.fillStyle = t.ink;
          g.textAlign = 'right';
          g.fillText('← say no', P.X(S.t) - 8, P.m.t + 12);
          g.textAlign = 'left';
          g.fillText('say yes →', P.X(S.t) + 8, P.m.t + 12);
          g.font = MONO;
          let ly = P.m.t + 34;
          [
            [t.accent, 'negatives'],
            [t.cut, 'positives'],
            [t.cut, '■ hits'],
            [t.ok, '■ false alarms'],
            [t.muted, '■ misses'],
          ].forEach(([cl, sx]) => {
            g.fillStyle = cl;
            g.fillText(sx, P.m.l + 6, ly);
            ly += 14;
          });
        }
        {
          const { g, w, h } = fit(cr),
            gap = 26,
            pw = (w - gap) / 2,
            sub = (ox, xl, yl, fx, fy, px, py, base) => {
              const m = { l: 42, r: 6, t: 14, b: 30 },
                X = (v) => ox + m.l + v * (pw - m.l - m.r),
                Y = (v) => h - m.b - v * (h - m.t - m.b);
              g.strokeStyle = t.line;
              g.lineWidth = 1;
              g.font = MONO;
              g.fillStyle = t.muted;
              [0, 0.5, 1].forEach((v) => {
                g.beginPath();
                g.moveTo(X(v), Y(0));
                g.lineTo(X(v), Y(1));
                g.moveTo(X(0), Y(v));
                g.lineTo(X(1), Y(v));
                g.stroke();
                g.textAlign = 'center';
                g.fillText(String(v), X(v), h - m.b + 13);
                g.textAlign = 'right';
                g.fillText(String(v), X(0) - 5, Y(v) + 4);
              });
              g.textAlign = 'center';
              g.fillText(xl, X(0.5), h - 4);
              g.save();
              g.translate(ox + 9, Y(0.5));
              g.rotate(-Math.PI / 2);
              g.fillText(yl, 0, 0);
              g.restore();
              g.setLineDash([4, 4]);
              g.strokeStyle = t.muted;
              g.beginPath();
              base(X, Y);
              g.stroke();
              g.setLineDash([]);
              g.strokeStyle = t.accent;
              g.lineWidth = 2.6;
              g.beginPath();
              for (let i = 0; i <= 240; i++) {
                const th = 6 - (12 * i) / 240,
                  x = X(fx(th)),
                  y = Y(fy(th));
                i ? g.lineTo(x, y) : g.moveTo(x, y);
              }
              g.stroke();
              g.fillStyle = t.gold;
              g.strokeStyle = t.bg;
              g.lineWidth = 2;
              g.beginPath();
              g.arc(X(px), Y(py), 7, 0, 7);
              g.fill();
              g.stroke();
            };
          const prec = (th) => {
            const a = S.pi * tpr(th),
              b = (1 - S.pi) * fpr(th);
            return a + b > 1e-12 ? a / (a + b) : 1;
          };
          sub(0, 'false positive rate', 'recall', fpr, tpr, fpr(S.t), rec, (X, Y) => {
            g.moveTo(X(0), Y(0));
            g.lineTo(X(1), Y(1));
          });
          sub(pw + gap, 'recall', 'precision', tpr, prec, rec, pre, (X, Y) => {
            g.moveTo(X(0), Y(S.pi));
            g.lineTo(X(1), Y(S.pi));
          });
        }
        const N = 1000;
        setR(root, 'tp', Math.round(TP * N) + ' of 1,000');
        setR(root, 'fn', String(Math.round(FN * N)));
        setR(root, 'fp', String(Math.round(FP * N)));
        setR(root, 'tn', String(Math.round(TN * N)));
        setR(root, 'acc', (100 * (TP + TN)).toFixed(1) + '%');
        setR(root, 'pre', (100 * pre).toFixed(1) + '%');
        setR(root, 'rec', (100 * rec).toFixed(1) + '%');
        setR(root, 'fpr', (100 * fpr(S.t)).toFixed(1) + '%');
        setR(root, 'f1', pre + rec > 1e-9 ? f3((2 * pre * rec) / (pre + rec)) : '0');
        setR(root, 'auc', f3(Phi(S.sep / Math.SQRT2)));
      };
      const ts = bind(root, 'mt-t', (v) => {
        S.t = v;
        draw();
      });
      pb = playBtn($('#mt-play', root), sweeper(ts.el, -3.5, 3.5, 9));
      grab(ts.el, pb);
      bind(root, 'mt-sep', (v) => {
        S.sep = v;
        draw();
      });
      bind(root, 'mt-pi', (v) => {
        S.pi = v;
        draw();
      });
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'curse',
    tab: 'Curse of dimensionality',
    title: 'Where everything is far from everything',
    lede: 'In two or three dimensions, nearby points are common and space feels crowded. Add dimensions and the room grows so fast that every point ends up alone, about equally far from all the others. Raise the dimension and watch distance lose its meaning.',
    stage: `<div class="stage one" style="margin-bottom:28px"><figure class="fig"><figcaption>The same 100 points with one, two and three coordinates. The gold box always holds 10% of the space. Right: how wide that box must be in any dimension</figcaption><canvas class="wide" id="cd-v" style="aspect-ratio:16/5.4" role="img" aria-label="One hundred points on a line, in a square and in a cube, with a box holding ten percent of the space, and a gauge for higher dimensions"></canvas></figure></div>
 <div class="stage"><figure class="fig"><figcaption>Distances between 200 random points (each divided by the average)</figcaption><canvas class="short" id="cd-c" role="img" aria-label="Histogram of pairwise distances at the chosen dimension"></canvas></figure>
 <figure class="fig"><figcaption>Three symptoms as the dimension grows</figcaption><canvas class="short" id="cd-e" role="img" aria-label="Three curves against dimension"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="cd-play" type="button">Raise the dimension</button></div>`)}${grp('The space', slider('cd-d', 'Number of dimensions', 0, 10, 1, 3))}</div>
 <dl class="read">${rd('d', 'Dimensions')}${rd('nf', 'Nearest neighbour ÷ farthest neighbour', 'hl')}${rd('ball', 'Share of the cube inside its largest ball', 'hl2')}${rd('shell', 'Share of the cube within 5% of its surface', 'hl3')}${rd('need', 'Points needed to keep neighbours 0.1 apart')}</dl>`,
    tryit:
      'Look at the gold box in the top row: it always holds 10% of the space, but how wide does it have to be? Move the slider to 10, then 100. Then look at the histogram below: at 2 dimensions some pairs are close and some far. What has happened to “near” and “far” at 500?',
    math: String.raw`<h4>Intuition</h4><p>Scatter 100 people along a 100-metre line and neighbours are a metre apart. Scatter them over a 100 × 100 metre field and each has about 10 metres to the next. In a 100-metre cube they are over 20 metres apart. Each new dimension multiplies the room, while the number of points stays the same. Carry on to hundreds of dimensions and the data is a few specks in a vast emptiness, with no specks near any other. Methods that work by finding similar examples then have nothing to work with.</p>
 <h4>The picture: a "local" box stops being local</h4><p>Suppose a method predicts from the 10% of the data nearest to a query. If the points are spread evenly, that neighbourhood must cover 10% of the volume. A box with a fraction \(r\) of the volume of the unit cube has side</p>$$e_d(r)=r^{1/d},\qquad e_1(0.1)=0.10,\quad e_2(0.1)=0.32,\quad e_3(0.1)=0.46,\quad e_{10}(0.1)=0.79,\quad e_{100}(0.1)=0.98.$$<p>In ten dimensions the "neighbourhood" already spans 79% of the range of every single feature. It is not a neighbourhood in any useful sense. And if you insist on a small box, it is empty: watch how the distance from the centre to the nearest of the same 100 points grows from the line to the square to the cube.</p>
 <h4>Symptom 1: you need exponentially many points</h4><p>To cover the unit cube with points spaced \(\epsilon\) apart takes about \((1/\epsilon)^d\) points: 10 on a line, 100 in a square, \(10^{100}\) in a hundred dimensions.</p>
 <h4>Symptom 2: distances concentrate</h4><p>The squared distance between two random points is a sum of \(d\) independent terms. Its mean grows like \(d\) but its spread only like \(\sqrt d\), so relative to the mean the spread shrinks like \(1/\sqrt d\):</p>$$\frac{\operatorname{sd}(\text{distance})}{\text{mean distance}}\ \sim\ \frac{1}{\sqrt d},\qquad\text{so}\qquad \frac{\text{nearest}}{\text{farthest}}\ \to\ 1.$$<p>This is the central limit theorem at work, and it is why the histogram narrows to a spike. When the nearest neighbour is barely nearer than the farthest, "nearest" carries little information.</p>
 <h4>Symptom 3: the volume moves to the edges</h4><p>The ball that just fits inside the unit cube has volume</p>$$V_d=\frac{\pi^{d/2}}{\Gamma\!\big(\tfrac d2+1\big)}\Big(\frac12\Big)^{d},$$<p>which is 79% of the square, 52% of the cube, and essentially zero by \(d=20\): almost all of a high-dimensional cube is in its corners. Likewise the share of the cube lying within 5% of its surface is \(1-0.9^{d}\), which passes 99% by \(d=44\). Nearly every point is near a boundary, so nearly every prediction is an extrapolation.</p>
 <h4>Why learning is still possible</h4><p>Real data does not fill its space. Images, speech and text lie close to surfaces of far lower dimension inside the huge space of all possible inputs. This is the manifold hypothesis, and it is what PCA, autoencoders and deep networks rely on: find the few directions that matter and the curse loses most of its force.</p>`,
    q: 'k-nearest neighbours works well with 5 features and badly with 500, on the same amount of data. Why?',
    a: 'With 500 features the nearest neighbours are almost as far away as every other point, so they are no more similar to the query than a random point is. The method averages over points that are not really neighbours. Reducing the dimension first, or using far more data, is needed.',
    init(root) {
      const c = $('#cd-c', root),
        ce = $('#cd-e', root),
        cv = $('#cd-v', root),
        DS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
        S = { i: 3 },
        H = {},
        NFD = {};
      let pb,
        rot = 0.6;
      const PT = (() => {
        const r = rng(42),
          a = [];
        for (let i = 0; i < 100; i++) a.push([r(), r(), r()]);
        return a;
      })();
      const drawTop = () => {
        const t = T(),
          { g, w, h } = fit(cv),
          pw = w / 4,
          sz = Math.min(pw * 0.78, h - 64),
          top = 30,
          d = DS[S.i];
        g.font = SANS;
        g.textAlign = 'center';
        const head = (k, a, b) => {
            const cx = pw * k + pw / 2;
            g.fillStyle = t.ink;
            g.font = SANS;
            g.fillText(a, cx, 16);
            g.font = MONO;
            g.fillStyle = t.cut;
            g.fillText(b, cx, top + sz + 22);
            return cx;
          },
          inb = (p, n, s) => {
            for (let q = 0; q < n; q++) if (Math.abs(p[q] - 0.5) > s / 2) return false;
            return true;
          },
          nn = (n) => {
            let b = 1 / 0;
            for (const p of PT) {
              let q = 0;
              for (let j = 0; j < n; j++) q += (p[j] - 0.5) ** 2;
              b = Math.min(b, Math.sqrt(q));
            }
            return b;
          };
        {
          const s1 = 0.1,
            cx = head(
              0,
              '1 dimension',
              'box side 0.10 · ' + PT.filter((p) => inb(p, 1, s1)).length + ' points inside',
            ),
            x0 = cx - sz / 2,
            y = top + sz / 2;
          g.strokeStyle = t.muted;
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(x0, y);
          g.lineTo(x0 + sz, y);
          g.stroke();
          g.fillStyle = mix(t.gold, t.gold, 0, 0.35);
          g.fillRect(x0 + sz * (0.5 - s1 / 2), y - 12, sz * s1, 24);
          g.strokeStyle = t.gold;
          g.lineWidth = 2;
          g.strokeRect(x0 + sz * (0.5 - s1 / 2), y - 12, sz * s1, 24);
          for (const p of PT) {
            g.fillStyle = inb(p, 1, s1) ? t.cut : mix(t.accent, t.accent, 0, 0.75);
            g.beginPath();
            g.arc(x0 + p[0] * sz, y, 3, 0, 7);
            g.fill();
          }
          g.fillStyle = t.muted;
          g.fillText('nearest point to the centre: ' + nn(1).toFixed(3), cx, top + sz + 38);
        }
        {
          const s2 = Math.sqrt(0.1),
            cx = head(
              1,
              '2 dimensions',
              'box side 0.32 · ' + PT.filter((p) => inb(p, 2, s2)).length + ' points inside',
            ),
            x0 = cx - sz / 2;
          g.strokeStyle = t.muted;
          g.lineWidth = 1.5;
          g.strokeRect(x0, top, sz, sz);
          g.fillStyle = mix(t.gold, t.gold, 0, 0.3);
          g.fillRect(x0 + sz * (0.5 - s2 / 2), top + sz * (0.5 - s2 / 2), sz * s2, sz * s2);
          g.strokeStyle = t.gold;
          g.lineWidth = 2;
          g.strokeRect(x0 + sz * (0.5 - s2 / 2), top + sz * (0.5 - s2 / 2), sz * s2, sz * s2);
          for (const p of PT) {
            g.fillStyle = inb(p, 2, s2) ? t.cut : mix(t.accent, t.accent, 0, 0.75);
            g.beginPath();
            g.arc(x0 + p[0] * sz, top + (1 - p[1]) * sz, 3, 0, 7);
            g.fill();
          }
          g.fillStyle = t.muted;
          g.fillText('nearest point to the centre: ' + nn(2).toFixed(3), cx, top + sz + 38);
        }
        {
          const s3 = Math.cbrt(0.1),
            cx = head(
              2,
              '3 dimensions',
              'box side 0.46 · ' + PT.filter((p) => inb(p, 3, s3)).length + ' points inside',
            ),
            cy = top + sz / 2,
            ca = Math.cos(rot),
            sa = Math.sin(rot),
            sc = sz * 0.56,
            pr = (x, y, z) => {
              x -= 0.5;
              y -= 0.5;
              z -= 0.5;
              const xr = x * ca - y * sa,
                yr = x * sa + y * ca;
              return [cx + sc * xr, cy - sc * (z * 0.92 + yr * 0.38), yr];
            };
          const box = (s, col, lw) => {
            const a = 0.5 - s / 2,
              b = 0.5 + s / 2,
              V = [
                [a, a, a],
                [b, a, a],
                [b, b, a],
                [a, b, a],
                [a, a, b],
                [b, a, b],
                [b, b, b],
                [a, b, b],
              ].map((v) => pr(v[0], v[1], v[2]));
            g.strokeStyle = col;
            g.lineWidth = lw;
            g.beginPath();
            [
              [0, 1],
              [1, 2],
              [2, 3],
              [3, 0],
              [4, 5],
              [5, 6],
              [6, 7],
              [7, 4],
              [0, 4],
              [1, 5],
              [2, 6],
              [3, 7],
            ].forEach(([i, j]) => {
              g.moveTo(V[i][0], V[i][1]);
              g.lineTo(V[j][0], V[j][1]);
            });
            g.stroke();
            return V;
          };
          box(1, t.muted, 1.2);
          const V = box(s3, t.gold, 2);
          g.fillStyle = mix(t.gold, t.gold, 0, 0.16);
          [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
          ].forEach((F) => {
            g.beginPath();
            F.forEach((i, k) => (k ? g.lineTo(V[i][0], V[i][1]) : g.moveTo(V[i][0], V[i][1])));
            g.closePath();
            g.fill();
          });
          PT.map((p) => ({ q: pr(p[0], p[1], p[2]), in: inb(p, 3, s3) }))
            .sort((a, b) => b.q[2] - a.q[2])
            .forEach((o) => {
              g.fillStyle = o.in ? t.cut : mix(t.accent, t.accent, 0, 0.75);
              g.beginPath();
              g.arc(o.q[0], o.q[1], 3, 0, 7);
              g.fill();
            });
          g.fillStyle = t.muted;
          g.fillText('nearest point to the centre: ' + nn(3).toFixed(3), cx, top + sz + 38);
        }
        {
          const cx = head(
              3,
              'Any number of dimensions',
              'box side ' + Math.pow(0.1, 1 / d).toFixed(3) + ' at ' + d + ' dimensions',
            ),
            x0 = cx - sz / 2,
            rh = sz / DS.length;
          g.textAlign = 'left';
          DS.forEach((q, k) => {
            const sd = Math.pow(0.1, 1 / q),
              y = top + k * rh,
              on = q === d,
              bx = x0 + 44,
              bwid = sz - 44 - 38;
            g.font = on ? SANS : MONO;
            g.fillStyle = on ? t.ink : t.muted;
            g.textAlign = 'right';
            g.fillText(String(q), x0 + 36, y + rh / 2 + 4);
            g.fillStyle = mix(t.muted, t.muted, 0, 0.14);
            g.fillRect(bx, y + rh * 0.22, bwid, rh * 0.56);
            g.fillStyle = on ? t.gold : mix(t.gold, t.gold, 0, 0.4);
            g.fillRect(bx, y + rh * 0.22, bwid * sd, rh * 0.56);
            g.textAlign = 'left';
            g.fillStyle = on ? t.ink : t.muted;
            g.fillText(sd.toFixed(2), bx + bwid + 6, y + rh / 2 + 4);
          });
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('gold = box width, grey = full range', cx, top + sz + 38);
        }
      };
      const sim = (d) => {
        if (H[d]) return H[d];
        const r = rng(100 + d),
          N = 200,
          P = [];
        for (let i = 0; i < N; i++) {
          const p = new Float32Array(d);
          for (let j = 0; j < d; j++) p[j] = r();
          P.push(p);
        }
        const ds = [];
        let nf = 0;
        for (let i = 0; i < N; i++) {
          let mn = 1 / 0,
            mx = 0;
          for (let j = 0; j < N; j++) {
            if (i === j) continue;
            let s = 0;
            const a = P[i],
              b = P[j];
            for (let q = 0; q < d; q++) {
              const u = a[q] - b[q];
              s += u * u;
            }
            s = Math.sqrt(s);
            if (s < mn) mn = s;
            if (s > mx) mx = s;
            if (j > i && (i * 7 + j) % 5 === 0) ds.push(s);
          }
          nf += mn / mx;
        }
        const mean = ds.reduce((a, b) => a + b, 0) / ds.length,
          bins = new Array(50).fill(0);
        for (const v of ds) {
          const b = Math.floor((v / mean / 2) * 50);
          if (b >= 0 && b < 50) bins[b]++;
        }
        NFD[d] = nf / N;
        return (H[d] = { bins, n: ds.length });
      };
      DS.forEach(sim);
      const ballS = (d) =>
          Math.exp((d / 2) * Math.log(Math.PI) - lgam(d / 2 + 1) - d * Math.log(2)),
        shell = (d) => 1 - Math.pow(0.9, d);
      const draw = () => {
        const t = T(),
          d = DS[S.i],
          Hd = sim(d);
        drawTop();
        {
          const { g, w, h } = fit(c);
          let mx = 0;
          for (const b of Hd.bins) mx = Math.max(mx, b / Hd.n / 0.04);
          const P = plot(g, w, h, [0, 2], [0, Math.max(4, mx * 1.1)], t, {
            xstep: 0.5,
            xlabel: 'distance ÷ average distance',
          });
          clipPlot(g, P);
          Hd.bins.forEach((b, i) => {
            if (!b) return;
            const v = b / Hd.n / 0.04;
            g.fillStyle = ramp(t, clamp(0.25 + (0.75 * v) / mx, 0, 1));
            g.fillRect(P.X(i * 0.04) + 0.5, P.Y(v), P.X(0.04) - P.X(0) - 1, P.Y(0) - P.Y(v));
          });
          g.restore();
          g.font = SANS;
          g.fillStyle = t.ink;
          g.textAlign = 'left';
          g.fillText(d + (d === 1 ? ' dimension' : ' dimensions'), P.m.l + 10, P.m.t + 14);
        }
        {
          const { g, w, h } = fit(ce),
            P = plot(g, w, h, [0, 3], [0, 1.05], t, {
              yticks: [0.25, 0.5, 0.75, 1],
              xstep: 100,
              xlabel: 'dimensions',
            });
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          [1, 10, 100, 1000].forEach((v) =>
            g.fillText(String(v), P.X(Math.log10(v)), h - P.m.b + 17),
          );
          clipPlot(g, P);
          g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
          g.lineWidth = 9;
          g.beginPath();
          g.moveTo(P.X(Math.log10(d)), P.Y(0));
          g.lineTo(P.X(Math.log10(d)), P.Y(1.05));
          g.stroke();
          [
            [(x) => ballS(x), t.ok],
            [(x) => shell(x), t.accent],
          ].forEach(([fn, cl]) => {
            g.strokeStyle = cl;
            g.lineWidth = 2.4;
            g.beginPath();
            for (let i = 0; i <= 150; i++) {
              const lx = (3 * i) / 150,
                x = Math.pow(10, lx);
              i ? g.lineTo(P.X(lx), P.Y(fn(x))) : g.moveTo(P.X(lx), P.Y(fn(x)));
            }
            g.stroke();
          });
          g.strokeStyle = t.gold;
          g.fillStyle = t.gold;
          g.lineWidth = 2.4;
          g.beginPath();
          DS.forEach((q, i) =>
            i
              ? g.lineTo(P.X(Math.log10(q)), P.Y(NFD[q]))
              : g.moveTo(P.X(Math.log10(q)), P.Y(NFD[q])),
          );
          g.stroke();
          DS.forEach((q) => {
            g.beginPath();
            g.arc(P.X(Math.log10(q)), P.Y(NFD[q]), 3, 0, 7);
            g.fill();
          });
          g.restore();
          g.textAlign = 'left';
          let ly = P.m.t + 34;
          [
            [t.cut, 'nearest ÷ farthest neighbour'],
            [t.ok, 'share of cube inside its ball'],
            [t.accent, 'share of cube near its surface'],
          ].forEach(([cl, sx]) => {
            g.fillStyle = cl;
            g.fillText(sx, P.X(1.2), ly);
            ly += 14;
          });
        }
        setR(root, 'd', String(d));
        setR(root, 'nf', f3(NFD[d]));
        const bs = ballS(d);
        setR(
          root,
          'ball',
          bs < 1e-300
            ? 'effectively zero'
            : bs < 1e-4
              ? bs.toExponential(1)
              : (100 * bs).toFixed(1) + '%',
        );
        setR(root, 'shell', (100 * shell(d)).toFixed(1) + '%');
        setR(root, 'need', d <= 6 ? Math.pow(10, d).toLocaleString() : '10^' + d);
      };
      const ds = bind(
        root,
        'cd-d',
        (v) => {
          S.i = v;
          draw();
        },
        (v) => String(DS[v]),
      );
      let fr = 0,
        dir = 1;
      const anim = Anim(() => {
        rot += 0.008;
        if (++fr % 80) {
          drawTop();
          return;
        }
        let v = S.i + dir;
        if (v > 10) {
          dir = -1;
          v = 9;
        }
        if (v < 0) {
          dir = 1;
          v = 1;
        }
        S.i = v;
        ds.set(v);
        draw();
      });
      pb = playBtn($('#cd-play', root), anim);
      grab(ds.el, pb);
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'search',
    tab: 'Hyperparameter search',
    title: 'Where to look for good settings',
    lede: 'Some settings are not learned from data: you have to try values and see. A grid looks tidy, but if only one setting really matters, a grid wastes most of its tries. Random search spends the same budget and usually finds better.',
    stage: `<div class="stage" style="grid-template-columns:minmax(0,1.5fr) minmax(0,1fr)"><figure class="fig"><figcaption>Validation score over two settings: grid search (left) and random search (right). Ticks along the top = distinct values tried</figcaption><canvas class="wide" id="hs-c" role="img" aria-label="Two score maps with the trials of grid search and random search"></canvas></figure>
 <figure class="fig"><figcaption>Best score found so far</figcaption><canvas class="short" id="hs-e" role="img" aria-label="Best score so far against number of trials"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="hs-play" type="button">Run the search</button><button class="btn" id="hs-new" type="button">New problem</button></div>`)}
 ${grp('The budget', slider('hs-b', 'Number of models you can afford to train', 0, 6, 1, 3))}${grp('The problem', slider('hs-imp', 'How much the second setting matters', 0, 1, 0.05, 0.15) + slider('hs-w', 'Width of the good region', 0.04, 0.4, 0.01, 0.09))}</div>
 <dl class="read">${rd('g', 'Best score: grid', 'hl3')}${rd('r', 'Best score: random', 'hl')}${rd('dg', 'Values of setting 1 tried: grid')}${rd('dr', 'Values of setting 1 tried: random')}${rd('win', 'Random beats grid, over 400 problems', 'hl2')}</dl>`,
    tryit:
      'With 16 trials, count the ticks along the top of each map. How many different values of the important setting did each method try? Then raise “how much the second setting matters” to 1 and see whether random search keeps its advantage.',
    math: String.raw`<h4>Intuition</h4><p>You are tuning a learning rate and a batch size, and you can afford 16 training runs. A 4 × 4 grid tries 4 learning rates, each with 4 batch sizes. If the batch size turns out to make almost no difference, twelve of those sixteen runs told you nothing new: you learned about only 4 learning rates. Sixteen random points try 16 different learning rates. Same cost, four times the information about the setting that counts. Since you rarely know in advance which setting matters, random is the safer bet.</p>
 <h4>Parameters and hyperparameters</h4><p>Weights are learned by minimising the training loss. Hyperparameters, such as the learning rate, the polynomial degree, \(\lambda\), or the number of layers, shape that minimisation from outside and must be chosen by comparing validation scores.</p>
 <h4>Counting distinct values</h4><p>With a budget of \(B\) trials and \(d\) hyperparameters, a grid tests only</p>$$B^{1/d}\ \text{ distinct values of each setting,}$$<p>while random search tests \(B\) distinct values of every setting. With \(d=6\) and \(B=729\) a grid tries just 3 values per setting. If only a few settings matter (and in practice this is usual), the grid's projection onto those few is very coarse and random search's is fine.</p>
 <h4>How likely is random search to succeed?</h4><p>If the good region covers a fraction \(p\) of the search space, each random trial lands in it with probability \(p\), independently:</p>$$P(\text{at least one hit in }B\text{ trials})=1-(1-p)^{B}.$$<p>For the best 5% of settings, 60 trials give \(1-0.95^{60}\approx95\%\), whatever the number of hyperparameters.</p>
 <h4>Doing better than random</h4><p>Random search ignores what earlier trials found. Bayesian optimisation fits a model, often a Gaussian process, to the scores seen so far and picks the next trial where the model predicts a good score or is very uncertain. Successive halving and Hyperband start many runs and stop the poor ones early. All of them should search learning rates and penalties on a log scale.</p>`,
    q: 'When would grid search be the better choice?',
    a: 'When there are only one or two hyperparameters and all of them matter, or when you need results on exactly the same values to compare across experiments. With three or more hyperparameters, or when you do not know which ones matter, random search or a smarter method is better.',
    init(root) {
      const c = $('#hs-c', root),
        ce = $('#hs-e', root),
        BS = [4, 9, 16, 25, 36, 49, 64],
        S = { bi: 3, imp: 0.15, w: 0.09 };
      let seed = 3,
        pk = [0.7, 0.4],
        G = [],
        Rn = [],
        shown = 0,
        win = 0,
        pb;
      const sc = (a, b, p = pk) =>
        Math.exp(-(((a - p[0]) / S.w) ** 2)) *
        (1 - S.imp + S.imp * Math.exp(-(((b - p[1]) / (S.w * 2.2)) ** 2)));
      const plan = (r, B) => {
        const m = Math.round(Math.sqrt(B)),
          g = [];
        for (let i = 0; i < m; i++)
          for (let j = 0; j < m; j++) g.push([(i + 0.5) / m, (j + 0.5) / m]);
        for (let i = g.length - 1; i > 0; i--) {
          const j = Math.floor(r() * (i + 1));
          [g[i], g[j]] = [g[j], g[i]];
        }
        const q = [];
        for (let i = 0; i < B; i++) q.push([r(), r()]);
        return [g, q];
      };
      const gen = () => {
        const r = rng(seed);
        pk = [0.1 + 0.8 * r(), 0.1 + 0.8 * r()];
        [G, Rn] = plan(r, BS[S.bi]);
        shown = 0;
        const r2 = rng(999);
        let wn = 0,
          tot = 0;
        for (let k = 0; k < 400; k++) {
          const p = [0.1 + 0.8 * r2(), 0.1 + 0.8 * r2()],
            [g, q] = plan(r2, BS[S.bi]),
            bg = Math.max(...g.map((v) => sc(v[0], v[1], p))),
            br = Math.max(...q.map((v) => sc(v[0], v[1], p)));
          if (br > bg) wn++;
          tot++;
        }
        win = wn / tot;
      };
      gen();
      const draw = () => {
        const t = T(),
          B = BS[S.bi];
        {
          const { g, w, h } = fit(c),
            gap = 22,
            pw = (w - gap) / 2,
            ph = h - 30,
            key = 'hs' + seed + '|' + S.imp + '|' + S.w;
          [
            [G, 0, 'Grid', t.accent],
            [Rn, pw + gap, 'Random', t.cut],
          ].forEach(([L, ox, nm, cl]) => {
            const hm = heatmap(key, pw, ph - 18, t, (x, y) => -sc((x + 3) / 6, (y + 3) / 6), 3, 60);
            g.drawImage(hm, ox, 18, pw, ph - 18);
            const X = (v) => ox + v * pw,
              Y = (v) => 18 + (1 - v) * (ph - 18);
            for (let i = 0; i < Math.min(shown, L.length); i++) {
              const q = L[i];
              g.strokeStyle = t.ink;
              g.lineWidth = 1.6;
              g.beginPath();
              g.moveTo(X(q[0]), 4);
              g.lineTo(X(q[0]), 15);
              g.stroke();
              g.fillStyle = t.ink;
              g.strokeStyle = t.bg;
              g.lineWidth = 1.5;
              g.beginPath();
              g.arc(X(q[0]), Y(q[1]), 5, 0, 7);
              g.fill();
              g.stroke();
            }
            let bi = -1,
              bv = -1;
            for (let i = 0; i < Math.min(shown, L.length); i++) {
              const v = sc(L[i][0], L[i][1]);
              if (v > bv) {
                bv = v;
                bi = i;
              }
            }
            if (bi >= 0) {
              g.strokeStyle = t.ok;
              g.lineWidth = 3;
              g.beginPath();
              g.arc(X(L[bi][0]), Y(L[bi][1]), 10, 0, 7);
              g.stroke();
            }
            g.font = SANS;
            g.fillStyle = t.ink;
            g.textAlign = 'left';
            g.fillText(nm, ox, h - 8);
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText('setting 1 (matters) →', ox + pw, h - 8);
          });
        }
        {
          const { g, w, h } = fit(ce),
            P = plot(g, w, h, [0, B], [0, 1.05], t, {
              yticks: [0.25, 0.5, 0.75, 1],
              xstep: B > 30 ? 10 : B > 12 ? 5 : 2,
              xlabel: 'models trained',
            });
          clipPlot(g, P);
          [
            [G, t.accent],
            [Rn, t.gold],
          ].forEach(([L, cl]) => {
            g.strokeStyle = cl;
            g.lineWidth = 2.8;
            g.beginPath();
            let b = 0;
            g.moveTo(P.X(0), P.Y(0));
            for (let i = 0; i < Math.min(shown, L.length); i++) {
              g.lineTo(P.X(i + 1), P.Y(b));
              b = Math.max(b, sc(L[i][0], L[i][1]));
              g.lineTo(P.X(i + 1), P.Y(b));
            }
            g.stroke();
          });
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.accent;
          g.fillText('grid', P.m.l + 8, P.m.t + 8);
          g.fillStyle = t.cut;
          g.fillText('random', P.m.l + 8, P.m.t + 23);
        }
        const best = (L) => {
            let b = 0;
            for (let i = 0; i < Math.min(shown, L.length); i++)
              b = Math.max(b, sc(L[i][0], L[i][1]));
            return b;
          },
          dist = (L) => new Set(L.slice(0, shown).map((q) => q[0].toFixed(4))).size;
        setR(root, 'g', f3(best(G)));
        setR(root, 'r', f3(best(Rn)));
        setR(root, 'dg', String(dist(G)));
        setR(root, 'dr', String(dist(Rn)));
        setR(root, 'win', (100 * win).toFixed(0) + '%');
      };
      let fr = 0,
        wait = 0;
      const anim = Anim(() => {
        fr++;
        if (wait > 0) {
          wait--;
          if (!wait) {
            seed++;
            gen();
            draw();
          }
          return;
        }
        if (fr % (BS[S.bi] > 30 ? 5 : 10)) return;
        if (shown < BS[S.bi]) {
          shown++;
          draw();
        } else pb.stop();
      });
      pb = playBtn($('#hs-play', root), anim);
      runCtl(pb, root, 'hs', {
        done: () => shown >= BS[S.bi],
        restart: () => {
          seed++;
          gen();
        },
      });
      const re = () => {
        gen();
        draw();
      };
      bind(
        root,
        'hs-b',
        (v) => {
          S.bi = v;
          re();
        },
        (v) => String(BS[v]),
      );
      bind(root, 'hs-imp', (v) => {
        S.imp = v;
        re();
      });
      bind(root, 'hs-w', (v) => {
        S.w = v;
        re();
      });
      $('#hs-new', root).onclick = () => {
        seed++;
        re();
      };
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },
];
