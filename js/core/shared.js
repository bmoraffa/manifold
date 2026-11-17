/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { controlsFor } from '../ui/controls.js';

const SITE = 'Manifold';

const toTitleCase = (s) => {
  const small = new Set([
    'a',
    'an',
    'the',
    'and',
    'or',
    'of',
    'to',
    'in',
    'on',
    'for',
    'with',
    'vs.',
    'vs',
    'at',
    'by',
    'as',
    'from',
    'its',
  ]);
  let first = true;
  return s
    .split(' ')
    .map((w) => {
      const core = w.replace(/^[“"(]+/, ''),
        pre = w.slice(0, w.length - core.length);
      let out;
      if (/[A-Z]/.test(core.slice(1)) || /^\d/.test(core)) out = core;
      else if (!first && small.has(core.toLowerCase().replace(/[,:;]$/, '')))
        out = core.toLowerCase();
      else
        out = core
          .split('-')
          .map((p, i) =>
            i === 0 && p.length === 1 && core.includes('-') && p === p.toLowerCase()
              ? p
              : p.charAt(0).toUpperCase() + p.slice(1),
          )
          .join('-');
      first = /[:.]$/.test(core);
      return pre + out;
    })
    .join(' ');
};

const $ = (s, r = document) => r.querySelector(s);

const escapeHTML = (value) =>
  value.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const f2 = (x) => (+x).toFixed(2);

const f3 = (x) => (+x).toFixed(3);

let prefersReducedMotion =
  typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

function lgam(x) {
  const cf = [
    76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
    0.1208650973866179e-2, -0.5395239384953e-5,
  ];
  let y = x,
    t = x + 5.5;
  t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015;
  for (let j = 0; j < 6; j++) s += cf[j] / ++y;
  return -t + Math.log((2.5066282746310005 * s) / x);
}

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randnFrom(r) {
  return () => {
    let u = 0;
    while (u < 1e-12) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  };
}

const randn = randnFrom(Math.random);

function gammaS(k) {
  const d = k - 1 / 3,
    c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      x = randn();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v)))
      return d * v;
  }
}

function erf(x) {
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return (
    s *
    (1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
        t *
        Math.exp(-x * x))
  );
}

function readThemeColors() {
  const computedStyle = getComputedStyle(document.documentElement),
    readColor = (n) => computedStyle.getPropertyValue(n).trim();
  return {
    bg: readColor('--bg'),
    surface: readColor('--bg'),
    ink: readColor('--ink'),
    muted: readColor('--muted'),
    line: readColor('--line'),
    accent: readColor('--accent'),
    cut: readColor('--cut'),
    ok: readColor('--ok'),
    gold: readColor('--gold'),
    r0: readColor('--ramp0'),
    r2: readColor('--ramp2'),
  };
}

function hex(h) {
  h = h.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(h.substr(i, 2), 16));
}

function mix(a, b, t, al = 1) {
  const A = hex(a),
    B = hex(b);
  return `rgba(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(',')},${al})`;
}

const ramp = (t, z, al = 1) =>
  z < 0.5 ? mix(t.r0, t.accent, z * 2, al) : mix(t.accent, t.r2, (z - 0.5) * 2, al);

function prepareCanvas(canvas) {
  const bounds = canvas.getBoundingClientRect(),
    pixelRatio = window.devicePixelRatio || 1,
    w = Math.max(60, bounds.width),
    h = Math.max(60, bounds.height);
  const W = Math.round(w * pixelRatio),
    H = Math.round(h * pixelRatio);
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }
  const g = canvas.getContext('2d');
  g.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  g.clearRect(0, 0, w, h);
  g.lineJoin = 'round';
  g.lineCap = 'round';
  return { g, w, h };
}

const MONO = '11px "JetBrains Mono",ui-monospace,monospace';

const SANS = '600 13px "Hanken Grotesk",sans-serif';

function plot(g, w, h, xr, yr, t, o = {}) {
  const m = { l: 38, r: 12, t: 12, b: 26 };
  const X = (x) => m.l + ((x - xr[0]) / (xr[1] - xr[0])) * (w - m.l - m.r),
    Y = (y) => h - m.b - ((y - yr[0]) / (yr[1] - yr[0])) * (h - m.t - m.b);
  g.lineWidth = 1;
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'center';
  g.textBaseline = 'alphabetic';
  const st = o.xstep || 1;
  for (let x = Math.ceil(xr[0] / st) * st; x <= xr[1] + 1e-9; x += st) {
    g.strokeStyle = t.line;
    g.beginPath();
    g.moveTo(X(x), h - m.b);
    g.lineTo(X(x), h - m.b + 4);
    g.stroke();
    g.fillText(String(Math.round(x * 100) / 100), X(x), h - m.b + 17);
  }
  g.textAlign = 'right';
  (o.yticks || []).forEach((y) => {
    if (y > yr[1]) return;
    g.strokeStyle = t.line;
    g.beginPath();
    g.moveTo(m.l, Y(y));
    g.lineTo(w - m.r, Y(y));
    g.stroke();
    g.fillText(String(y), m.l - 6, Y(y) + 4);
  });
  g.strokeStyle = t.muted;
  g.beginPath();
  g.moveTo(m.l, h - m.b);
  g.lineTo(w - m.r, h - m.b);
  g.stroke();
  if (o.xlabel) {
    g.textAlign = 'right';
    g.fillText(o.xlabel, w - m.r, h - m.b - 6);
  }
  return { X, Y, m, w, h };
}

function curve(g, P, f, a, b, n = 260) {
  g.beginPath();
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    i ? g.lineTo(P.X(x), P.Y(f(x))) : g.moveTo(P.X(x), P.Y(f(x)));
  }
}

function area(g, P, f, a, b, n = 120) {
  g.beginPath();
  g.moveTo(P.X(a), P.Y(0));
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    g.lineTo(P.X(x), P.Y(f(x)));
  }
  g.lineTo(P.X(b), P.Y(0));
  g.closePath();
}

function clipPlot(g, P) {
  g.save();
  g.beginPath();
  g.rect(P.m.l, P.m.t - 6, P.w - P.m.l - P.m.r, P.h - P.m.t - P.m.b + 6);
  g.clip();
}

function ellPath(g, X, Y, mx, my, a, b, d, r) {
  const sx = Math.sqrt(a),
    sy = Math.sqrt(d),
    rho = clamp(b / (sx * sy), -0.9999, 0.9999),
    q = Math.sqrt(1 - rho * rho);
  g.beginPath();
  for (let k = 0; k <= 72; k++) {
    const th = (k / 72) * 2 * Math.PI,
      u = r * Math.cos(th),
      v = r * Math.sin(th),
      px = X(mx + sx * u),
      py = Y(my + sy * (rho * u + q * v));
    k ? g.lineTo(px, py) : g.moveTo(px, py);
  }
  g.closePath();
}

const npdf = (x, m, s) =>
  Math.exp((-(x - m) * (x - m)) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));

function Anim(tick, frames = globalThis) {
  let id = null;
  const loop = () => {
    id = frames.requestAnimationFrame(loop);
    tick();
  };
  return {
    get on() {
      return id !== null;
    },
    start() {
      if (id === null) id = frames.requestAnimationFrame(loop);
    },
    stop() {
      if (id !== null) {
        frames.cancelAnimationFrame(id);
        id = null;
      }
    },
  };
}

function playBtn(btn, anim) {
  const lab = btn.textContent,
    o = {
      get on() {
        return anim.on;
      },
      stop() {
        anim.stop();
        btn.textContent = lab;
        btn.classList.remove('on');
      },
      start() {
        anim.start();
        btn.textContent = 'Pause';
        btn.classList.add('on');
      },
    };
  btn.addEventListener('click', () => (anim.on ? o.stop() : o.start()));
  return o;
}

function bind(root, id, fn, fmt = f2) {
  const input = $('#' + id, root),
    output = $('#' + id + '-o', root);
  const updateOutput = () => {
    if (output) output.textContent = fmt(input.value);
  };
  input.addEventListener('input', () => {
    updateOutput();
    fn(+input.value);
  });
  updateOutput();
  return {
    el: input,
    set(v) {
      input.value = v;
      updateOutput();
    },
  };
}

function sweeper(inp, lo, hi, sec = 7) {
  let ph = null;
  const a = Anim(() => {
    if (ph === null) {
      const u = clamp((+inp.value - lo) / (hi - lo), 0, 1);
      ph = Math.acos(1 - 2 * u) / (2 * Math.PI);
    }
    ph += 1 / 60 / sec;
    inp.value = lo + ((hi - lo) * (1 - Math.cos(ph * 2 * Math.PI))) / 2;
    inp.dispatchEvent(new Event('input'));
  });
  const st = a.stop;
  a.stop = () => {
    ph = null;
    st();
  };
  return a;
}

const grab = (inp, pb) =>
  ['pointerdown', 'keydown'].forEach((n) => inp.addEventListener(n, () => pb.stop()));

function testPoint(root, id, labels, defaults, ranges) {
  const { create } = controlsFor(root);
  const children = root.ownerDocument.createDocumentFragment();
  labels.forEach((label, i) => {
    const [min, max] = ranges[i];
    children.append(create('number', { id: id + '-tp' + i, label, min, max, value: defaults[i] }));
  });
  children.append(create('point-status', { id: id + '-tpo' }));
  root
    .querySelector('.controls')
    .append(create('group', { label: 'Predict for your own point' }, children));
  const ins = labels.map((_, i) => $('#' + id + '-tp' + i, root)),
    tp = {
      v: defaults.slice(),
      out: $('#' + id + '-tpo', root),
      on: null,
      set(v) {
        tp.v = v.slice();
        ins.forEach((e, i) => {
          if (document.activeElement !== e) e.value = (+v[i]).toFixed(2);
        });
      },
    };
  ins.forEach((e, i) =>
    e.addEventListener('input', () => {
      const x = parseFloat(e.value);
      if (isFinite(x)) {
        tp.v[i] = clamp(x, ranges[i][0], ranges[i][1]);
        tp.on && tp.on();
      }
    }),
  );
  return tp;
}

function markTP(g, x, y, t, lab = true) {
  g.save();
  g.translate(x, y);
  g.rotate(Math.PI / 4);
  g.fillStyle = t.ink;
  g.strokeStyle = t.bg;
  g.lineWidth = 2.5;
  g.beginPath();
  g.rect(-7, -7, 14, 14);
  g.fill();
  g.stroke();
  g.restore();
  if (lab) {
    g.font = SANS;
    g.fillStyle = t.ink;
    g.textAlign = 'left';
    g.fillText('your point', x + 13, y - 9);
  }
}

function eqShow(el, lines, gap = 420) {
  clearTimeout(el._t);
  el.innerHTML = lines.map((l) => '<div>' + l + '</div>').join('');
  const ds = [...el.children];
  let i = 0;
  const nx = () => {
    if (i < ds.length) {
      ds[i++].classList.add('on');
      if (gap > 0 && !prefersReducedMotion) el._t = setTimeout(nx, gap);
      else nx();
    }
  };
  nx();
}

const maxOf = (root, id) => {
  const e = $('#' + id + '-max', root),
    v = e ? parseInt(e.value, 10) : NaN;
  return v > 0 ? v : Infinity;
};

function attachRunLimit(pb, root, id, o) {
  const st = pb.start,
    limitNotice = $('#' + id + '-maxnote', root),
    stepUnit = o.what || 'steps';
  let startingCount = 0,
    carried = 0;
  const consumed = () => carried + (o.count ? o.count() - startingCount : 0);
  pb.start = () => {
    if (limitNotice) limitNotice.textContent = '';
    if (o.done()) o.restart();
    startingCount = o.count ? o.count() : 0;
    carried = 0;
    st();
  };
  const check = () => {
    if (o.done()) {
      pb.stop();
      return true;
    }
    const m = maxOf(root, id);
    if (pb.on && o.count && consumed() >= m) {
      const used = consumed();
      pb.stop();
      if (limitNotice)
        limitNotice.textContent =
          'Stopped after ' +
          used +
          ' more ' +
          (used === 1 ? stepUnit.replace(/e?s$/, '') : stepUnit) +
          ', as you asked. Press the button again for ' +
          m +
          ' more, or empty the box to run to completion.';
      return true;
    }
    return false;
  };
  check.remaining = () => (pb.on && o.count ? Math.max(0, maxOf(root, id) - consumed()) : Infinity);
  check.rebase = (nextCount) => {
    carried = consumed();
    startingCount = nextCount;
  };
  return check;
}

function letterData(seed, flip) {
  const GL = {
      A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
      E: ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
      H: ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
      O: ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
      T: ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
      X: ['#...#', '#...#', '.#.#.', '..#..', '.#.#.', '#...#', '#...#'],
    },
    CH = Object.keys(GL),
    r = rng(seed * 17 + 1),
    rn = randnFrom(r),
    X = [],
    lab = [];
  CH.forEach((ch, k) => {
    const bits = GL[ch].join('');
    for (let i = 0; i < 25; i++) {
      X.push(
        [...bits].map((b) => {
          let v = b === '#' ? 1 : 0;
          if (r() < flip) v = 1 - v;
          return v + 0.04 * rn();
        }),
      );
      lab.push(k);
    }
  });
  return { X, lab, CH };
}

const setReadout = (root, k, v) => {
  const e = $(`[data-r="${k}"]`, root);
  if (e) e.textContent = v;
};

function shapeBind(root, lessonId, shape, onShapeChange) {
  const sigmaXSlider = bind(root, lessonId + '-sx', (v) => {
      shape.sx = v;
      onShapeChange();
    }),
    sigmaYSlider = bind(root, lessonId + '-sy', (v) => {
      shape.sy = v;
      onShapeChange();
    }),
    rhoSlider = bind(root, lessonId + '-rho', (v) => {
      shape.rho = v;
      onShapeChange();
    });
  return {
    sync() {
      sigmaXSlider.set(shape.sx);
      sigmaYSlider.set(shape.sy);
      rhoSlider.set(shape.rho);
    },
    els: [sigmaXSlider.el, sigmaYSlider.el, rhoSlider.el],
  };
}

const sharedGaussianShape = { sx: 1.4, sy: 0.9, rho: 0.6 };

function eig(G) {
  const a = G.sx * G.sx,
    d = G.sy * G.sy,
    b = G.rho * G.sx * G.sy,
    m = (a + d) / 2,
    r = Math.hypot((a - d) / 2, b);
  return { l1: m + r, l2: m - r, th: 0.5 * Math.atan2(2 * b, a - d), a, b, d };
}

const dens2 = (G, x, y) =>
  Math.exp(
    (-0.5 / (1 - G.rho * G.rho)) *
      ((x * x) / (G.sx * G.sx) - (2 * G.rho * x * y) / (G.sx * G.sy) + (y * y) / (G.sy * G.sy)),
  );

function Hill(c, redraw, viewControls, opt = {}) {
  const defaultView = { az: -0.65, el: 0.5, zoom: 1 },
    view = { ...defaultView };
  let dragStart = null,
    spinning = false;
  const spin = Anim(() => {
    view.az += 0.0035;
    redraw();
  });
  const setSpin = (on) => {
    spinning = on;
    on ? spin.start() : spin.stop();
    const b = viewControls && viewControls.querySelector('[data-v="spin"]');
    if (b) b.textContent = on ? 'Stop rotating' : 'Rotate for me';
  };
  c.tabIndex = 0;
  c.setAttribute('role', 'img');
  c.setAttribute('aria-label', '3D surface. Drag or use arrow keys to rotate.');
  c.addEventListener('pointerdown', (e) => {
    setSpin(false);
    dragStart = { x: e.clientX, y: e.clientY, az: view.az, el: view.el };
    c.setPointerCapture(e.pointerId);
  });
  c.addEventListener('pointermove', (e) => {
    if (!dragStart) return;
    view.az = dragStart.az + (e.clientX - dragStart.x) * 0.01;
    view.el = clamp(dragStart.el + (e.clientY - dragStart.y) * 0.008, 0.06, 1.5);
    redraw();
  });
  ['pointerup', 'pointercancel'].forEach((n) =>
    c.addEventListener(n, () => {
      dragStart = null;
    }),
  );
  c.addEventListener('keydown', (e) => {
    const k = e.key;
    if (k === 'ArrowLeft') view.az -= 0.1;
    else if (k === 'ArrowRight') view.az += 0.1;
    else if (k === 'ArrowUp') view.el = clamp(view.el + 0.08, 0.06, 1.5);
    else if (k === 'ArrowDown') view.el = clamp(view.el - 0.08, 0.06, 1.5);
    else return;
    e.preventDefault();
    setSpin(false);
    redraw();
  });
  if (viewControls)
    viewControls.addEventListener('click', (e) => {
      const a = e.target.dataset && e.target.dataset.v;
      if (!a) return;
      if (a === 'spin') {
        setSpin(!spinning);
        return;
      }
      if (a === 'in') view.zoom = Math.min(2.2, view.zoom * 1.2);
      if (a === 'out') view.zoom = Math.max(0.6, view.zoom / 1.2);
      if (a === 'reset') {
        setSpin(false);
        Object.assign(view, defaultView);
      }
      redraw();
    });
  function draw(t, G, cut = {}) {
    const { g: ctx, w, h } = prepareCanvas(c),
      R = 4,
      N = 40,
      ZH = 3,
      s = Math.min(w / 11.8, h / 8.3) * view.zoom,
      cx = w / 2,
      cy = h * 0.66;
    const ca = Math.cos(view.az),
      sa = Math.sin(view.az),
      ce = Math.cos(view.el),
      se = Math.sin(view.el);
    const pr = (x, y, z) => {
      const xr = x * ca - y * sa,
        yr = x * sa + y * ca;
      return [cx + s * xr, cy - s * (z * ce + yr * se), yr * ce - z * se];
    };
    const path = (ps) => {
      ctx.beginPath();
      ps.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    };
    const q = Math.sqrt(1 - G.rho * G.rho),
      f = (x, y) => dens2(G, x, y);
    const yF = ca >= 0 ? R : -R,
      xF = sa >= 0 ? R : -R;
    ctx.strokeStyle = t.line;
    ctx.lineWidth = 1;
    for (let k = -R; k <= R; k++) {
      path([pr(k, -R, 0), pr(k, R, 0)]);
      ctx.stroke();
      path([pr(-R, k, 0), pr(R, k, 0)]);
      ctx.stroke();
    }
    ctx.strokeStyle = t.muted;
    path([pr(xF, yF, 0), pr(xF, yF, ZH)]);
    ctx.stroke();
    const top = pr(xF, yF, ZH);
    ctx.font = MONO;
    ctx.fillStyle = t.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(
      'peak ' + f3(1 / (2 * Math.PI * G.sx * G.sy * q)),
      clamp(top[0], 40, w - 40),
      Math.max(12, top[1] - 6),
    );
    if (opt.marg) {
      const MH = ZH * 0.62;
      ctx.lineWidth = 2;
      let ps = [];
      for (let i = 0; i <= 80; i++) {
        const x = -R + (2 * R * i) / 80;
        ps.push(pr(x, yF, MH * Math.exp((-x * x) / (2 * G.sx * G.sx))));
      }
      path([pr(-R, yF, 0), ...ps, pr(R, yF, 0)]);
      ctx.closePath();
      ctx.fillStyle = mix(t.accent, t.accent, 0, 0.14);
      ctx.fill();
      path(ps);
      ctx.strokeStyle = t.accent;
      ctx.stroke();
      let a = pr(0, yF, MH + 0.35);
      ctx.fillStyle = t.accent;
      ctx.fillText('p(x)', a[0], a[1]);
      ps = [];
      for (let i = 0; i <= 80; i++) {
        const y = -R + (2 * R * i) / 80;
        ps.push(pr(xF, y, MH * Math.exp((-y * y) / (2 * G.sy * G.sy))));
      }
      path([pr(xF, -R, 0), ...ps, pr(xF, R, 0)]);
      ctx.closePath();
      ctx.fillStyle = mix(t.ok, t.ok, 0, 0.14);
      ctx.fill();
      path(ps);
      ctx.strokeStyle = t.ok;
      ctx.stroke();
      a = pr(xF, 0, MH + 0.35);
      ctx.fillStyle = t.ok;
      ctx.fillText('p(y)', a[0], a[1]);
    }
    const projectedGrid = [],
      densityGrid = [];
    for (let i = 0; i <= N; i++) {
      projectedGrid.push([]);
      densityGrid.push([]);
      for (let j = 0; j <= N; j++) {
        const x = -R + (2 * R * i) / N,
          y = -R + (2 * R * j) / N,
          z = f(x, y);
        densityGrid[i].push(z);
        projectedGrid[i].push(pr(x, y, ZH * z));
      }
    }
    const surfaceCells = [];
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const z =
            (densityGrid[i][j] +
              densityGrid[i + 1][j] +
              densityGrid[i][j + 1] +
              densityGrid[i + 1][j + 1]) /
            4,
          xc = -R + (2 * R * (i + 0.5)) / N;
        if (z < 0.012) continue;
        let ghost = false;
        if (cut.h != null && z > cut.h) ghost = true;
        if (cut.v != null && (xc - cut.v) * xF < 0) ghost = true;
        surfaceCells.push({
          i,
          j,
          z,
          ghost,
          d:
            (projectedGrid[i][j][2] +
              projectedGrid[i + 1][j][2] +
              projectedGrid[i][j + 1][2] +
              projectedGrid[i + 1][j + 1][2]) /
            4,
        });
      }
    surfaceCells.sort((a, b) => b.d - a.d);
    ctx.lineWidth = 0.6;
    for (const o of surfaceCells) {
      path([
        projectedGrid[o.i][o.j],
        projectedGrid[o.i + 1][o.j],
        projectedGrid[o.i + 1][o.j + 1],
        projectedGrid[o.i][o.j + 1],
      ]);
      ctx.closePath();
      const zz = Math.pow(o.z, 0.7);
      if (o.ghost) {
        ctx.fillStyle = ramp(t, zz, 0.07);
        ctx.fill();
        ctx.strokeStyle = mix(t.ink, t.ink, 0, 0.06);
        ctx.stroke();
      } else {
        ctx.fillStyle = ramp(t, zz);
        ctx.fill();
        ctx.strokeStyle = mix(t.ink, t.ink, 0, 0.16);
        ctx.stroke();
      }
    }
    if (cut.h != null) {
      const r = Math.sqrt(-2 * Math.log(cut.h)),
        ring = (z) => {
          const ps = [];
          for (let k = 0; k <= 72; k++) {
            const th = (k / 72) * 2 * Math.PI,
              u = r * Math.cos(th),
              v = r * Math.sin(th);
            ps.push(pr(G.sx * u, G.sy * (G.rho * u + q * v), z));
          }
          return ps;
        };
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = mix(t.ok, t.ok, 0, 0.7);
      path(ring(0));
      ctx.stroke();
      ctx.setLineDash([]);
      const z = ZH * cut.h;
      path([pr(-R, -R, z), pr(R, -R, z), pr(R, R, z), pr(-R, R, z)]);
      ctx.closePath();
      ctx.fillStyle = mix(t.ok, t.ok, 0, 0.06);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = mix(t.ok, t.ok, 0, 0.5);
      ctx.stroke();
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = t.ok;
      path(ring(z));
      ctx.stroke();
    }
    if (cut.v != null) {
      const ps = [];
      for (let k = 0; k <= 90; k++) {
        const y = -R + (2 * R * k) / 90;
        ps.push(pr(cut.v, y, ZH * f(cut.v, y)));
      }
      path([pr(cut.v, -R, 0), ...ps, pr(cut.v, R, 0)]);
      ctx.closePath();
      ctx.fillStyle = mix(t.ok, t.ok, 0, 0.3);
      ctx.fill();
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = t.ok;
      path(ps);
      ctx.stroke();
    }
    ctx.font = SANS;
    ctx.fillStyle = t.ink;
    let a = pr(0, -yF * 1.13, 0);
    ctx.fillText('x', a[0], a[1] + 4);
    a = pr(-xF * 1.13, 0, 0);
    ctx.fillText('y', a[0], a[1] + 4);
  }
  return { draw, spin: setSpin };
}

function drawMap(c, t, G, o = {}) {
  const { g: ctx, w, h } = prepareCanvas(c),
    R = 4,
    s = Math.min(w, h) / 9.2,
    tr = o.tr || ((x, y) => [x, y]);
  const X = (x, y) => {
    const p = tr(x, y);
    return [w / 2 + s * p[0], h / 2 - s * p[1]];
  };
  const q = Math.sqrt(1 - G.rho * G.rho);
  const ring = (r) => {
    ctx.beginPath();
    for (let k = 0; k <= 72; k++) {
      const th = (k / 72) * 2 * Math.PI,
        u = r * Math.cos(th),
        v = r * Math.sin(th),
        p = X(G.sx * u, G.sy * (G.rho * u + q * v));
      k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
    }
    ctx.closePath();
  };
  const seg = (a, b) => {
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  };
  ctx.lineWidth = 1;
  for (let k = -8; k <= 8; k++) {
    ctx.strokeStyle = k === 0 ? t.muted : t.line;
    seg(X(k, -9), X(k, 9));
    seg(X(-9, k), X(9, k));
  }
  [3, 2.5, 2, 1.5, 1, 0.5].forEach((r) => {
    ring(r);
    ctx.fillStyle = ramp(t, Math.pow(Math.exp((-r * r) / 2), 0.55), 0.9);
    ctx.fill();
  });
  if (o.pts) {
    ctx.fillStyle = mix(t.ink, t.ink, 0, 0.6);
    for (const z of o.pts) {
      const p = X(G.sx * z[0], G.sy * (G.rho * z[0] + q * z[1]));
      ctx.beginPath();
      ctx.arc(p[0], p[1], 1.7, 0, 7);
      ctx.fill();
    }
  }
  if (o.rings) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = mix(t.ok, t.ok, 0, 0.6);
    o.rings.forEach((r) => {
      ring(r);
      ctx.stroke();
    });
  }
  if (o.ring != null) {
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = t.ok;
    ring(o.ring);
    ctx.stroke();
  }
  if (o.eig) {
    const e = eig(G),
      c1 = Math.cos(e.th),
      s1 = Math.sin(e.th),
      a = 2 * Math.sqrt(e.l1),
      b = 2 * Math.sqrt(e.l2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = t.ok;
    seg(X(-a * c1, -a * s1), X(a * c1, a * s1));
    ctx.strokeStyle = t.ink;
    seg(X(b * s1, -b * c1), X(-b * s1, b * c1));
    ctx.font = SANS;
    ctx.textAlign = 'left';
    ctx.fillStyle = t.ok;
    let p = X(a * c1, a * s1);
    ctx.fillText('λ₁', clamp(p[0] + 6, 4, w - 22), clamp(p[1], 14, h - 6));
    ctx.fillStyle = t.ink;
    p = X(-b * s1, b * c1);
    ctx.fillText('λ₂', clamp(p[0] + 6, 4, w - 22), clamp(p[1], 14, h - 6));
  }
  ctx.font = SANS;
  ctx.fillStyle = t.muted;
  ctx.textAlign = 'right';
  ctx.fillText('x', w - 6, h / 2 - 6);
  ctx.textAlign = 'left';
  ctx.fillText('y', w / 2 + 6, 14);
  return { g: ctx, X, ring, seg, s, w, h };
}

function observeReducedMotion(media) {
  const update = () => {
    prefersReducedMotion = media.matches;
  };
  update();
  media.addEventListener?.('change', update);
  return () => media.removeEventListener?.('change', update);
}

export {
  SITE,
  toTitleCase,
  $,
  escapeHTML,
  clamp,
  f2,
  f3,
  prefersReducedMotion,
  lgam,
  rng,
  randnFrom,
  randn,
  gammaS,
  erf,
  readThemeColors,
  hex,
  mix,
  ramp,
  prepareCanvas,
  MONO,
  SANS,
  plot,
  curve,
  area,
  clipPlot,
  ellPath,
  npdf,
  Anim,
  playBtn,
  bind,
  sweeper,
  grab,
  testPoint,
  markTP,
  eqShow,
  maxOf,
  attachRunLimit,
  letterData,
  setReadout,
  shapeBind,
  sharedGaussianShape,
  eig,
  dens2,
  Hill,
  drawMap,
  observeReducedMotion,
};
