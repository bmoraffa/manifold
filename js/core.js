/* Manifold — Shared helpers: drawing, controls, animation, small maths.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
const SITE = 'Manifold';
const TC = (s) => {
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
/* ============ helpers ============ */
const $ = (s, r = document) => r.querySelector(s);
const escapeHTML = (value) =>
  value.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const f2 = (x) => (+x).toFixed(2),
  f3 = (x) => (+x).toFixed(3);
const RM = (() => {
  try {
    return matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch (e) {
    return false;
  }
})();
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
function T() {
  const s = getComputedStyle(document.documentElement),
    v = (n) => s.getPropertyValue(n).trim();
  return {
    bg: v('--bg'),
    surface: v('--bg'),
    ink: v('--ink'),
    muted: v('--muted'),
    line: v('--line'),
    accent: v('--accent'),
    cut: v('--cut'),
    ok: v('--ok'),
    gold: v('--gold'),
    r0: v('--ramp0'),
    r2: v('--ramp2'),
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
function fit(c) {
  const r = c.getBoundingClientRect(),
    d = window.devicePixelRatio || 1,
    w = Math.max(60, r.width),
    h = Math.max(60, r.height);
  const W = Math.round(w * d),
    H = Math.round(h * d);
  if (c.width !== W || c.height !== H) {
    c.width = W;
    c.height = H;
  }
  const g = c.getContext('2d');
  g.setTransform(d, 0, 0, d, 0, 0);
  g.clearRect(0, 0, w, h);
  g.lineJoin = 'round';
  g.lineCap = 'round';
  return { g, w, h };
}
const MONO = '11px "JetBrains Mono",ui-monospace,monospace',
  SANS = '600 13px "Hanken Grotesk",sans-serif';
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
function Anim(tick) {
  let id = null;
  const loop = () => {
    id = requestAnimationFrame(loop);
    tick();
  };
  return {
    get on() {
      return id !== null;
    },
    start() {
      if (id === null) id = requestAnimationFrame(loop);
    },
    stop() {
      if (id !== null) {
        cancelAnimationFrame(id);
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
  const i = $('#' + id, root),
    o = $('#' + id + '-o', root);
  const up = () => {
    if (o) o.textContent = fmt(i.value);
  };
  i.addEventListener('input', () => {
    up();
    fn(+i.value);
  });
  up();
  return {
    el: i,
    set(v) {
      i.value = v;
      up();
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
const slider = (id, label, min, max, step, val) =>
  `<label class="ctl" for="${id}"><span>${label}</span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"><output id="${id}-o" for="${id}"></output></label>`;
function testPoint(root, id, labels, defaults, ranges) {
  root
    .querySelector('.controls')
    .insertAdjacentHTML(
      'beforeend',
      grp(
        'Predict for your own point',
        labels
          .map(
            (l, i) =>
              `<label class="ctl" for="${id}-tp${i}"><span>${l}</span><input type="number" class="num" id="${id}-tp${i}" step="0.1" min="${ranges[i][0]}" max="${ranges[i][1]}" value="${defaults[i]}"></label>`,
          )
          .join('') + `<p class="tpout" id="${id}-tpo" role="status"></p>`,
      ),
    );
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
const stopBox = (id, what = 'steps') =>
  `<label class="ctl" for="${id}-max"><span>Stop after this many ${what} (empty: run until it converges)</span><input type="number" class="num" id="${id}-max" min="1" step="1" placeholder="no limit" aria-describedby="${id}-maxnote"></label><p class="status" id="${id}-maxnote" role="status"></p>`;
function eqShow(el, lines, gap = 420) {
  clearTimeout(el._t);
  el.innerHTML = lines.map((l) => '<div>' + l + '</div>').join('');
  const ds = [...el.children];
  let i = 0;
  const nx = () => {
    if (i < ds.length) {
      ds[i++].classList.add('on');
      if (gap > 0 && !RM) el._t = setTimeout(nx, gap);
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
function runCtl(pb, root, id, o) {
  const st = pb.start,
    nt = $('#' + id + '-maxnote', root),
    wh = o.what || 'steps';
  let c0 = 0,
    carried = 0;
  const consumed = () => carried + (o.count ? o.count() - c0 : 0);
  pb.start = () => {
    if (nt) nt.textContent = '';
    if (o.done()) o.restart();
    c0 = o.count ? o.count() : 0;
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
      if (nt)
        nt.textContent =
          'Stopped after ' +
          used +
          ' more ' +
          (used === 1 ? wh.replace(/e?s$/, '') : wh) +
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
    c0 = nextCount;
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
const grp = (label, inner) => `<fieldset class="grp"><legend>${label}</legend>${inner}</fieldset>`;
const rd = (k, label, cls = '') =>
  `<div class="${cls}"><dt>${label}</dt><dd data-r="${k}">–</dd></div>`;
const setR = (root, k, v) => {
  const e = $(`[data-r="${k}"]`, root);
  if (e) e.textContent = v;
};
const viewbar = `<div class="viewbar"><span>Drag to rotate</span><button class="link" data-v="spin" type="button">Rotate for me</button><button class="link" data-v="in" type="button">Zoom in</button><button class="link" data-v="out" type="button">Zoom out</button><button class="link" data-v="reset" type="button">Reset view</button></div>`;
const shapeHTML = (p, g) =>
  slider(p + '-sx', 'Spread of x, σₓ', 0.45, 2, 0.01, g.sx) +
  slider(p + '-sy', 'Spread of y, σᵧ', 0.45, 2, 0.01, g.sy) +
  slider(p + '-rho', 'Correlation ρ', -0.97, 0.97, 0.01, g.rho);
function shapeBind(root, p, Gx, cb) {
  const a = bind(root, p + '-sx', (v) => {
      Gx.sx = v;
      cb();
    }),
    b = bind(root, p + '-sy', (v) => {
      Gx.sy = v;
      cb();
    }),
    r = bind(root, p + '-rho', (v) => {
      Gx.rho = v;
      cb();
    });
  return {
    sync() {
      a.set(Gx.sx);
      b.set(Gx.sy);
      r.set(Gx.rho);
    },
    els: [a.el, b.el, r.el],
  };
}

/* ============ shared 2D Gaussian (mean 0) ============ */
const G = { sx: 1.4, sy: 0.9, rho: 0.6 };
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

/* ============ 3D surface viewer ============ */
function Hill(c, redraw, ctl, opt = {}) {
  const V0 = { az: -0.65, el: 0.5, zoom: 1 },
    V = { ...V0 };
  let d = null,
    spinning = false;
  const spin = Anim(() => {
    V.az += 0.0035;
    redraw();
  });
  const setSpin = (on) => {
    spinning = on;
    on ? spin.start() : spin.stop();
    const b = ctl && ctl.querySelector('[data-v="spin"]');
    if (b) b.textContent = on ? 'Stop rotating' : 'Rotate for me';
  };
  c.tabIndex = 0;
  c.setAttribute('role', 'img');
  c.setAttribute('aria-label', '3D surface. Drag or use arrow keys to rotate.');
  c.addEventListener('pointerdown', (e) => {
    setSpin(false);
    d = { x: e.clientX, y: e.clientY, az: V.az, el: V.el };
    c.setPointerCapture(e.pointerId);
  });
  c.addEventListener('pointermove', (e) => {
    if (!d) return;
    V.az = d.az + (e.clientX - d.x) * 0.01;
    V.el = clamp(d.el + (e.clientY - d.y) * 0.008, 0.06, 1.5);
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
    else if (k === 'ArrowUp') V.el = clamp(V.el + 0.08, 0.06, 1.5);
    else if (k === 'ArrowDown') V.el = clamp(V.el - 0.08, 0.06, 1.5);
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
  function draw(t, G, cut = {}) {
    const { g, w, h } = fit(c),
      R = 4,
      N = 40,
      ZH = 3,
      s = Math.min(w / 11.8, h / 8.3) * V.zoom,
      cx = w / 2,
      cy = h * 0.66;
    const ca = Math.cos(V.az),
      sa = Math.sin(V.az),
      ce = Math.cos(V.el),
      se = Math.sin(V.el);
    const pr = (x, y, z) => {
      const xr = x * ca - y * sa,
        yr = x * sa + y * ca;
      return [cx + s * xr, cy - s * (z * ce + yr * se), yr * ce - z * se];
    };
    const path = (ps) => {
      g.beginPath();
      ps.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
    };
    const q = Math.sqrt(1 - G.rho * G.rho),
      f = (x, y) => dens2(G, x, y);
    const yF = ca >= 0 ? R : -R,
      xF = sa >= 0 ? R : -R;
    g.strokeStyle = t.line;
    g.lineWidth = 1;
    for (let k = -R; k <= R; k++) {
      path([pr(k, -R, 0), pr(k, R, 0)]);
      g.stroke();
      path([pr(-R, k, 0), pr(R, k, 0)]);
      g.stroke();
    }
    g.strokeStyle = t.muted;
    path([pr(xF, yF, 0), pr(xF, yF, ZH)]);
    g.stroke();
    const top = pr(xF, yF, ZH);
    g.font = MONO;
    g.fillStyle = t.muted;
    g.textAlign = 'center';
    g.textBaseline = 'alphabetic';
    g.fillText(
      'peak ' + f3(1 / (2 * Math.PI * G.sx * G.sy * q)),
      clamp(top[0], 40, w - 40),
      Math.max(12, top[1] - 6),
    );
    if (opt.marg) {
      const MH = ZH * 0.62;
      g.lineWidth = 2;
      let ps = [];
      for (let i = 0; i <= 80; i++) {
        const x = -R + (2 * R * i) / 80;
        ps.push(pr(x, yF, MH * Math.exp((-x * x) / (2 * G.sx * G.sx))));
      }
      path([pr(-R, yF, 0), ...ps, pr(R, yF, 0)]);
      g.closePath();
      g.fillStyle = mix(t.accent, t.accent, 0, 0.14);
      g.fill();
      path(ps);
      g.strokeStyle = t.accent;
      g.stroke();
      let a = pr(0, yF, MH + 0.35);
      g.fillStyle = t.accent;
      g.fillText('p(x)', a[0], a[1]);
      ps = [];
      for (let i = 0; i <= 80; i++) {
        const y = -R + (2 * R * i) / 80;
        ps.push(pr(xF, y, MH * Math.exp((-y * y) / (2 * G.sy * G.sy))));
      }
      path([pr(xF, -R, 0), ...ps, pr(xF, R, 0)]);
      g.closePath();
      g.fillStyle = mix(t.ok, t.ok, 0, 0.14);
      g.fill();
      path(ps);
      g.strokeStyle = t.ok;
      g.stroke();
      a = pr(xF, 0, MH + 0.35);
      g.fillStyle = t.ok;
      g.fillText('p(y)', a[0], a[1]);
    }
    const P = [],
      F = [];
    for (let i = 0; i <= N; i++) {
      P.push([]);
      F.push([]);
      for (let j = 0; j <= N; j++) {
        const x = -R + (2 * R * i) / N,
          y = -R + (2 * R * j) / N,
          z = f(x, y);
        F[i].push(z);
        P[i].push(pr(x, y, ZH * z));
      }
    }
    const Q = [];
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const z = (F[i][j] + F[i + 1][j] + F[i][j + 1] + F[i + 1][j + 1]) / 4,
          xc = -R + (2 * R * (i + 0.5)) / N;
        if (z < 0.012) continue;
        let ghost = false;
        if (cut.h != null && z > cut.h) ghost = true;
        if (cut.v != null && (xc - cut.v) * xF < 0) ghost = true;
        Q.push({
          i,
          j,
          z,
          ghost,
          d: (P[i][j][2] + P[i + 1][j][2] + P[i][j + 1][2] + P[i + 1][j + 1][2]) / 4,
        });
      }
    Q.sort((a, b) => b.d - a.d);
    g.lineWidth = 0.6;
    for (const o of Q) {
      path([P[o.i][o.j], P[o.i + 1][o.j], P[o.i + 1][o.j + 1], P[o.i][o.j + 1]]);
      g.closePath();
      const zz = Math.pow(o.z, 0.7);
      if (o.ghost) {
        g.fillStyle = ramp(t, zz, 0.07);
        g.fill();
        g.strokeStyle = mix(t.ink, t.ink, 0, 0.06);
        g.stroke();
      } else {
        g.fillStyle = ramp(t, zz);
        g.fill();
        g.strokeStyle = mix(t.ink, t.ink, 0, 0.16);
        g.stroke();
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
      g.setLineDash([5, 5]);
      g.lineWidth = 1.5;
      g.strokeStyle = mix(t.ok, t.ok, 0, 0.7);
      path(ring(0));
      g.stroke();
      g.setLineDash([]);
      const z = ZH * cut.h;
      path([pr(-R, -R, z), pr(R, -R, z), pr(R, R, z), pr(-R, R, z)]);
      g.closePath();
      g.fillStyle = mix(t.ok, t.ok, 0, 0.06);
      g.fill();
      g.lineWidth = 1;
      g.strokeStyle = mix(t.ok, t.ok, 0, 0.5);
      g.stroke();
      g.lineWidth = 3.5;
      g.strokeStyle = t.ok;
      path(ring(z));
      g.stroke();
    }
    if (cut.v != null) {
      const ps = [];
      for (let k = 0; k <= 90; k++) {
        const y = -R + (2 * R * k) / 90;
        ps.push(pr(cut.v, y, ZH * f(cut.v, y)));
      }
      path([pr(cut.v, -R, 0), ...ps, pr(cut.v, R, 0)]);
      g.closePath();
      g.fillStyle = mix(t.ok, t.ok, 0, 0.3);
      g.fill();
      g.lineWidth = 3.5;
      g.strokeStyle = t.ok;
      path(ps);
      g.stroke();
    }
    g.font = SANS;
    g.fillStyle = t.ink;
    let a = pr(0, -yF * 1.13, 0);
    g.fillText('x', a[0], a[1] + 4);
    a = pr(-xF * 1.13, 0, 0);
    g.fillText('y', a[0], a[1] + 4);
  }
  return { draw, spin: setSpin };
}

/* ============ top-down contour map ============ */
function drawMap(c, t, G, o = {}) {
  const { g, w, h } = fit(c),
    R = 4,
    s = Math.min(w, h) / 9.2,
    tr = o.tr || ((x, y) => [x, y]);
  const X = (x, y) => {
    const p = tr(x, y);
    return [w / 2 + s * p[0], h / 2 - s * p[1]];
  };
  const q = Math.sqrt(1 - G.rho * G.rho);
  const ring = (r) => {
    g.beginPath();
    for (let k = 0; k <= 72; k++) {
      const th = (k / 72) * 2 * Math.PI,
        u = r * Math.cos(th),
        v = r * Math.sin(th),
        p = X(G.sx * u, G.sy * (G.rho * u + q * v));
      k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]);
    }
    g.closePath();
  };
  const seg = (a, b) => {
    g.beginPath();
    g.moveTo(a[0], a[1]);
    g.lineTo(b[0], b[1]);
    g.stroke();
  };
  g.lineWidth = 1;
  for (let k = -8; k <= 8; k++) {
    g.strokeStyle = k === 0 ? t.muted : t.line;
    seg(X(k, -9), X(k, 9));
    seg(X(-9, k), X(9, k));
  }
  [3, 2.5, 2, 1.5, 1, 0.5].forEach((r) => {
    ring(r);
    g.fillStyle = ramp(t, Math.pow(Math.exp((-r * r) / 2), 0.55), 0.9);
    g.fill();
  });
  if (o.pts) {
    g.fillStyle = mix(t.ink, t.ink, 0, 0.6);
    for (const z of o.pts) {
      const p = X(G.sx * z[0], G.sy * (G.rho * z[0] + q * z[1]));
      g.beginPath();
      g.arc(p[0], p[1], 1.7, 0, 7);
      g.fill();
    }
  }
  if (o.rings) {
    g.lineWidth = 1.5;
    g.strokeStyle = mix(t.ok, t.ok, 0, 0.6);
    o.rings.forEach((r) => {
      ring(r);
      g.stroke();
    });
  }
  if (o.ring != null) {
    g.lineWidth = 3.5;
    g.strokeStyle = t.ok;
    ring(o.ring);
    g.stroke();
  }
  if (o.eig) {
    const e = eig(G),
      c1 = Math.cos(e.th),
      s1 = Math.sin(e.th),
      a = 2 * Math.sqrt(e.l1),
      b = 2 * Math.sqrt(e.l2);
    g.lineWidth = 3;
    g.strokeStyle = t.ok;
    seg(X(-a * c1, -a * s1), X(a * c1, a * s1));
    g.strokeStyle = t.ink;
    seg(X(b * s1, -b * c1), X(-b * s1, b * c1));
    g.font = SANS;
    g.textAlign = 'left';
    g.fillStyle = t.ok;
    let p = X(a * c1, a * s1);
    g.fillText('λ₁', clamp(p[0] + 6, 4, w - 22), clamp(p[1], 14, h - 6));
    g.fillStyle = t.ink;
    p = X(-b * s1, b * c1);
    g.fillText('λ₂', clamp(p[0] + 6, 4, w - 22), clamp(p[1], 14, h - 6));
  }
  g.font = SANS;
  g.fillStyle = t.muted;
  g.textAlign = 'right';
  g.fillText('x', w - 6, h / 2 - 6);
  g.textAlign = 'left';
  g.fillText('y', w / 2 + 6, 14);
  return { g, X, ring, seg, s, w, h };
}
