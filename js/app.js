/* Manifold — Topic list, home-page animation, menu and routing.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ topics ============ */
const TOPICS = [
  { id: 'gaussian', t: 'The Gaussian', lessons: LESSONS },
  { id: 'foundations', t: 'Foundations', lessons: FND },
  { id: 'regression', t: 'Regression', lessons: REG },
  { id: 'classification', t: 'Classification', lessons: CLF },
  { id: 'ensembles', t: 'Ensembles', lessons: ENS },
  { id: 'unsupervised', t: 'Unsupervised learning', lessons: UNS },
  { id: 'neural-nets', t: 'Neural networks', lessons: NNL },
  { id: 'deep-learning', t: 'Deep learning', lessons: DLL },
  { id: 'generative', t: 'Generative AI', lessons: GEN },
  { id: 'language-models', t: 'Language models', lessons: LML },
  { id: 'reinforcement', t: 'Reinforcement learning', lessons: RLL },
  { id: 'probabilistic', t: 'Probabilistic methods', lessons: PML },
];

/* ============ home: one set of particles, several pictures ============ */
const Hero = (() => {
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
})();

TOPICS.push({ id: 'about', t: 'About Manifold', lessons: ABOUT, about: true });
/* ============ shell: home menu, module pages with lesson tabs, routing ============ */
const side = $('#side'),
  homeEl = $('#home'),
  lessonEl = $('#lesson'),
  modbar = $('#modbar'),
  ltabs = $('#ltabs'),
  modName = $('#mod-name');
let cur = null,
  curTopic = null;
const ROOTS = {};
const entries = (tp) =>
  (tp.lessons || [])
    .map((L) => ({ id: L.id, t: TC(L.tab), L }))
    .concat((tp.items || []).map((t, i) => ({ id: tp.id + '-' + (i + 1), t: TC(t) })));
side.innerHTML =
  `<button class="brand" type="button" data-go="home">${SITE}</button><p class="brand-sub">Machine learning you can see move</p>` +
  TOPICS.filter((tp) => !tp.about)
    .map((tp) => `<button class="topic-b" type="button" data-go="${tp.id}">${TC(tp.t)}</button>`)
    .join('') +
  `<div class="side-about">` +
  ABOUT.map(
    (L) => `<button class="topic-b about-b" type="button" data-go="${L.id}">${L.tab}</button>`,
  ).join('') +
  `</div>`;
$('#menu-b').onclick = (e) => {
  const on = document.body.classList.toggle('menu-open');
  e.currentTarget.setAttribute('aria-expanded', on);
};
document.addEventListener('click', (e) => {
  const t = e.target.closest('[data-go]');
  if (t) {
    document.body.classList.remove('menu-open');
    go(t.dataset.go);
  }
});
function mathify(el, tries = 0) {
  if (el.dataset.mj) return;
  if (!(window.MathJax && MathJax.typesetPromise)) {
    if (tries < 60) setTimeout(() => mathify(el, tries + 1), 400);
    return;
  }
  el.dataset.mj = '1';
  MathJax.typesetPromise([el]).catch(() => {
    delete el.dataset.mj;
  });
}
function lessonRoot(L, i, list) {
  if (ROOTS[L.id]) return ROOTS[L.id];
  const s = document.createElement('div');
  if (L.page) {
    s.className = 'about about-' + L.id;
    s.innerHTML = `<p class="eyebrow">About Manifold</p><h1>${L.title}</h1><div class="rule"></div><p class="lede">${L.lede}</p><div class="about-body">${L.html}</div>`;
    ROOTS[L.id] = s;
    L.root = s;
    return s;
  }
  let intro = '',
    rest = L.math;
  const k = L.math.indexOf('<h4>', 4);
  if (k > 0) {
    intro = L.math.slice(0, k).replace(/<h4>.*?<\/h4>/, '');
    rest = L.math.slice(k);
  }
  s.innerHTML = `<h1>${TC(L.title)}</h1><p class="lede">${L.lede}</p>${intro ? `<div class="prose intro">${intro}</div>` : ''}${L.stage}
  <p class="try"><b>Try this.</b> ${L.tryit}</p>
  <div class="prose after">${rest}</div>
  <details><summary>Check yourself</summary><div class="prose"><p><b>${L.q}</b></p><p>${L.a}</p></div></details>
  <div class="pager">${i < list.length - 1 ? `<button type="button" data-go="${list[i + 1].id}">Next: ${TC(list[i + 1].tab)} →</button>` : ''}</div>`;
  ROOTS[L.id] = s;
  L.root = s;
  return s;
}
function enter(tp, id) {
  document.body.classList.add('in-module');
  homeEl.hidden = true;
  lessonEl.hidden = false;
  modbar.hidden = false;
  if (curTopic !== tp) {
    curTopic = tp;
    modName.textContent = TC(tp.t);
    ltabs.innerHTML = entries(tp)
      .map((e) => `<button type="button" role="tab" data-go="${e.id}">${e.t}</button>`)
      .join('');
  }
  ltabs.querySelectorAll('button').forEach((b) => {
    const on = b.dataset.go === id;
    b.setAttribute('aria-selected', on ? 'true' : 'false');
    if (on && b.scrollIntoView) b.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  });
}
function show(id) {
  if (cur && cur.api && cur.api.stop) cur.api.stop();
  Hero.stop();
  let tp = TOPICS.find((t) => t.id === id);
  if (tp) id = entries(tp)[0].id;
  else tp = TOPICS.find((t) => entries(t).some((e) => e.id === id));
  if (!tp) {
    cur = null;
    curTopic = null;
    document.body.classList.remove('in-module');
    modbar.hidden = true;
    lessonEl.hidden = true;
    homeEl.hidden = false;
    requestAnimationFrame(() => Hero.start());
    return id && id !== 'home' ? 'home' : id;
  }
  enter(tp, id);
  const list = tp.lessons || [],
    i = list.findIndex((l) => l.id === id);
  if (i >= 0) {
    const L = list[i],
      root = lessonRoot(L, i, list);
    lessonEl.replaceChildren(root);
    if (!L.api) L.api = L.init(root);
    if (L.api.sync) L.api.sync();
    cur = L;
    root.querySelectorAll('.prose').forEach((p) => mathify(p));
    requestAnimationFrame(() => {
      L.api.draw();
      if (!RM && L.api.auto && cur === L && !lessonEl.hidden) L.api.auto();
    });
  } else {
    cur = null;
    const e = entries(tp).find((x) => x.id === id);
    lessonEl.innerHTML = `<h1>${e.t}</h1><p class="lede">This one is not built yet. Topics are being added one at a time, each with its own animation, controls and math.</p>`;
  }
  return id;
}
function go(id) {
  const real = show(id) || 'home';
  try {
    if (location.hash !== '#' + real) history.replaceState(null, '', '#' + real);
  } catch (e) {}
  window.scrollTo({ top: 0 });
}
const redraw = () => {
  if (!homeEl.hidden) Hero.draw();
  else if (cur) cur.api.draw();
};
window.addEventListener('resize', redraw);
window.addEventListener('hashchange', () => show(location.hash.slice(1)));
try {
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
} catch (e) {}
new MutationObserver(redraw).observe(document.documentElement, {
  attributes: true,
  attributeFilter: ['data-theme'],
});
if (document.fonts && document.fonts.ready)
  document.fonts.ready.then(() => {
    Hero.rebuild();
    redraw();
  });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    if (cur && cur.api && cur.api.stop) cur.api.stop();
    Hero.stop();
  } else if (!homeEl.hidden) Hero.start();
});
show(location.hash.slice(1) || 'home');
