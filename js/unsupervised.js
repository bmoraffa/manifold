/* Manifold — Unsupervised learning.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ unsupervised learning ============ */
const UC = (t) => [t.accent, t.gold, t.ok, '#4a7fb5', '#8a5fb0', '#c9703a', '#3f9c96', '#8c8c8c'];
const UKIND = {
  shop: 'Shop customers (simulated)',
  five: 'Five groups',
  blobs: 'Three round groups',
  uneven: 'Groups of unequal size',
  long: 'Three stretched groups',
  moons: 'Two moons',
  rings: 'Two rings',
  dens: 'Dense and sparse groups',
  none: 'No groups at all',
  h2: 'Heights of children and adults (simulated)',
  h3: 'Three humps',
  hov: 'Two overlapping humps',
  corr: 'Height and arm span (simulated)',
  banana: 'A curved cloud',
  twoc: 'Two separate groups',
  round: 'A round cloud',
};
function uData(kind, n, seed, bg = 0) {
  const r = rng(seed * 13 + 5),
    rn = randnFrom(r),
    P = [];
  const blob = (m, cx, cy, sx, sy, rot = 0) => {
    m = Math.round(m);
    const c = Math.cos(rot),
      s = Math.sin(rot);
    for (let i = 0; i < m; i++) {
      const a = rn() * sx,
        b = rn() * sy;
      P.push({ x: cx + a * c - b * s, y: cy + a * s + b * c });
    }
  };
  if (kind === 'shop') {
    blob(n * 0.32, -2.2, -1.2, 0.42, 0.38);
    blob(n * 0.26, -1.3, 1.3, 0.5, 0.36);
    blob(n * 0.28, 1.1, -0.5, 0.5, 0.45);
    blob(n * 0.14, 2.5, 1.5, 0.3, 0.3);
  } else if (kind === 'five') {
    [
      [-2.6, 1.5],
      [-2.4, -1.4],
      [0, 0.1],
      [2.4, 1.6],
      [2.6, -1.3],
    ].forEach((m) => blob(n / 5, m[0], m[1], 0.36, 0.36));
  } else if (kind === 'blobs') {
    blob(n / 3, -2, -1, 0.5, 0.5);
    blob(n / 3, 2, -1, 0.5, 0.5);
    blob(n / 3, 0, 1.4, 0.5, 0.5);
  } else if (kind === 'uneven') {
    blob(n * 0.7, -1.2, 0, 1.05, 0.95);
    blob(n * 0.15, 2.3, 1.3, 0.22, 0.22);
    blob(n * 0.15, 2.4, -1.2, 0.22, 0.22);
  } else if (kind === 'long') {
    [-1.5, 0, 1.5].forEach((y) => blob(n / 3, 0, y, 1.5, 0.18, 0.32));
  } else if (kind === 'moons') {
    const hf = Math.ceil(n / 2);
    for (let i = 0; i < n; i++) {
      const a = ((Math.floor(i / 2) + r()) / hf) * Math.PI;
      P.push(
        i % 2
          ? {
              x: 0.9 - 1.9 * Math.cos(a) + 0.085 * rn(),
              y: 0.55 - 1.7 * Math.sin(a) + 0.085 * rn(),
            }
          : {
              x: -0.9 + 1.9 * Math.cos(a) + 0.085 * rn(),
              y: -0.55 + 1.7 * Math.sin(a) + 0.085 * rn(),
            },
      );
    }
  } else if (kind === 'rings') {
    const n0 = Math.ceil(n / 4);
    for (let i = 0; i < n; i++) {
      const k = i < n0 ? 0 : 1,
        a = (((k ? i - n0 : i) + r()) / (k ? n - n0 : n0)) * 2 * Math.PI,
        rad = (k ? 2.2 : 0.7) + 0.085 * rn();
      P.push({ x: rad * 1.15 * Math.cos(a), y: rad * Math.sin(a) });
    }
  } else if (kind === 'dens') {
    blob(n * 0.45, -1.9, 0.3, 0.28, 0.28);
    blob(n * 0.45, 1.6, 0, 1, 0.9);
    blob(n * 0.1, -1.6, -2, 0.2, 0.15);
  } else for (let i = 0; i < n; i++) P.push({ x: r() * 6.8 - 3.4, y: r() * 5 - 2.5 });
  for (let i = 0; i < bg; i++) P.push({ x: r() * 7.2 - 3.6, y: r() * 5.4 - 2.7 });
  for (const p of P) {
    p.x = clamp(p.x, -3.8, 3.8);
    p.y = clamp(p.y, -2.85, 2.85);
  }
  return P;
}
function uView(c) {
  const { g, w, h } = fit(c),
    s = Math.min(w / 8, h / 6),
    X = (x) => w / 2 + x * s,
    Y = (y) => h / 2 - y * s;
  return { g, w, h, s, X, Y };
}
const uInv = (c, e) => {
  const r = c.getBoundingClientRect(),
    s = Math.min(r.width / 8, r.height / 6);
  return [(e.clientX - r.left - r.width / 2) / s, -(e.clientY - r.top - r.height / 2) / s];
};
function uFrame(V, t, xl, yl) {
  const { g, w, h, X, Y } = V;
  g.strokeStyle = t.line;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(X(-4), Y(0));
  g.lineTo(X(4), Y(0));
  g.moveTo(X(0), Y(-3));
  g.lineTo(X(0), Y(3));
  g.stroke();
  if (xl) {
    g.font = MONO;
    g.fillStyle = t.muted;
    g.textAlign = 'right';
    g.fillText(xl, w - 6, h - 6);
    g.textAlign = 'left';
    g.fillText(yl, 6, 12);
  }
}
const uNum = (v) =>
  !isFinite(v)
    ? '–'
    : Math.abs(v) >= 100
      ? v.toFixed(0)
      : Math.abs(v) >= 10
        ? v.toFixed(1)
        : v.toFixed(2);
function uLine(g, t, x0, y0, w, h, ys, o = {}) {
  const m = { l: 44, r: 12, t: 24, b: 20 },
    n = Math.max(2, o.n || ys.length),
    fin = ys.filter(isFinite),
    ym = o.ym != null ? o.ym : Math.max(1e-9, ...fin),
    y1 = o.y1 != null ? o.y1 : 0;
  const X = (i) => x0 + m.l + (i / (n - 1)) * (w - m.l - m.r),
    Y = (v) => y0 + h - m.b - ((v - y1) / (ym - y1 || 1)) * (h - m.t - m.b);
  g.strokeStyle = t.muted;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(x0 + m.l, y0 + h - m.b);
  g.lineTo(x0 + w - m.r, y0 + h - m.b);
  g.stroke();
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'left';
  g.fillText(o.title || '', x0 + 8, y0 + 13);
  g.textAlign = 'right';
  g.fillText(uNum(ym), x0 + m.l - 5, Y(ym) + 4);
  g.fillText(uNum(y1), x0 + m.l - 5, Y(y1) + 4);
  if (ys.length) {
    g.strokeStyle = o.col || t.accent;
    g.lineWidth = 2.2;
    g.beginPath();
    ys.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
    g.stroke();
    if (o.dots) {
      g.fillStyle = o.col || t.accent;
      ys.forEach((v, i) => {
        g.beginPath();
        g.arc(X(i), Y(v), 3, 0, 7);
        g.fill();
      });
    }
  }
  return { X, Y, m };
}
const uDataHTML = (id, kinds, def, nmin, nmax, n, extra = '') =>
  grp(
    'The data',
    `<label class="ctl" for="${id}-ds"><span>Data set</span><select id="${id}-ds">${kinds.map((k) => `<option value="${k}"${k === def ? ' selected' : ''}>${UKIND[k]}</option>`).join('')}</select></label>` +
      slider(id + '-n', 'Number of points', nmin, nmax, 1, n) +
      extra +
      `<div class="row"><button class="btn" id="${id}-new" type="button">New random sample</button><label class="btn file">Upload a CSV<input type="file" id="${id}-file" accept=".csv,.txt,text/csv"></label></div><p class="status" id="${id}-st" role="status"></p>`,
  );
function uDataBind(root, id, S, cb, mk) {
  const st = $('#' + id + '-st', root);
  let custom = null;
  const gen = () =>
    custom ? custom.map((p) => ({ ...p })) : mk ? mk() : uData(S.ds, S.n, S.seed, S.bg || 0);
  $('#' + id + '-ds', root).onchange = (e) => {
    S.ds = e.target.value;
    custom = null;
    st.textContent = '';
    cb();
  };
  bind(
    root,
    id + '-n',
    (v) => {
      S.n = v;
      if (!custom) cb();
    },
    (v) => String(v),
  );
  $('#' + id + '-new', root).onclick = () => {
    S.seed++;
    custom = null;
    st.textContent = '';
    cb();
  };
  $('#' + id + '-file', root).onchange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fr = new FileReader();
    fr.onerror = () => {
      st.textContent =
        'That file could not be read. Save it again as plain CSV and upload it once more.';
    };
    fr.onload = () => {
      const p = parseXY(fr.result);
      if (typeof p === 'string') {
        st.textContent = p;
        return;
      }
      const q = p.slice(0, S.cap || 600),
        m = [0, 1].map((j) => q.reduce((s, v) => s + v[j], 0) / q.length),
        sd = [0, 1].map(
          (j) => Math.sqrt(q.reduce((s, v) => s + (v[j] - m[j]) ** 2, 0) / q.length) || 1,
        );
      custom = q.map((v) => ({
        x: clamp(((v[0] - m[0]) / sd[0]) * 1.25, -3.8, 3.8),
        y: clamp(((v[1] - m[1]) / sd[1]) * 0.95, -2.85, 2.85),
      }));
      st.textContent =
        'Loaded ' +
        f.name +
        ': ' +
        q.length +
        ' rows from the first two numeric columns, each rescaled to mean 0 and the same spread.';
      cb();
    };
    fr.readAsText(f);
    e.target.value = '';
  };
  return gen;
}
const segBind = (root, id, val, fn) => {
  const bs = [...root.querySelectorAll('#' + id + ' button')],
    set = (v) =>
      bs.forEach((b) => b.setAttribute('aria-pressed', b.dataset.v === v ? 'true' : 'false'));
  bs.forEach(
    (b) =>
      (b.onclick = () => {
        set(b.dataset.v);
        fn(b.dataset.v);
      }),
  );
  set(val);
};
const uCur = (root, id, i) =>
  root
    .querySelectorAll('#' + id + '-steps li')
    .forEach((e, k) => e.setAttribute('aria-current', k === i ? 'true' : 'false'));
const uHow = (id, steps) =>
  `<div class="howto"><div><h3>How it works, step by step</h3><ol id="${id}-steps">${steps.map((x) => `<li>${x}</li>`).join('')}</ol></div><div><h3>What just happened</h3><p class="live" id="${id}-live" role="status"></p></div></div>`;
const uBtns = (id, a, b, what) =>
  grp(
    'Animation',
    `<div class="row"><button class="btn go" id="${id}-play" type="button">${a}</button><button class="btn" id="${id}-step" type="button">${b}</button><button class="btn" id="${id}-reset" type="button">Start over</button></div>${what ? stopBox(id, what) : ''}`,
  );
const d2 = (p, c) => (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
function kmSeed(P, K, pp, r) {
  const C = [];
  if (!P.length) return C;
  if (!pp) {
    const idx = P.map((_, i) => i);
    for (let k = 0; k < K && idx.length; k++) {
      const j = Math.floor(r() * idx.length);
      C.push({ x: P[idx[j]].x, y: P[idx[j]].y });
      idx.splice(j, 1);
    }
    return C;
  }
  const f = P[Math.floor(r() * P.length)];
  C.push({ x: f.x, y: f.y });
  const D = P.map((p) => d2(p, f));
  while (C.length < K) {
    let tot = 0;
    for (const v of D) tot += v;
    if (tot <= 0) break;
    let u = r() * tot,
      i = 0;
    while (i < P.length - 1 && (u -= D[i]) > 0) i++;
    const c = { x: P[i].x, y: P[i].y };
    C.push(c);
    for (let j = 0; j < P.length; j++) {
      const d = d2(P[j], c);
      if (d < D[j]) D[j] = d;
    }
  }
  return C;
}
function kmRun(P, C, maxIt = 60) {
  C = C.map((c) => ({ ...c }));
  const n = P.length,
    K = C.length,
    A = new Int16Array(n).fill(-1);
  let it = 0;
  for (; it < maxIt; it++) {
    let ch = 0;
    for (let i = 0; i < n; i++) {
      let b = 0,
        bd = 1e18;
      for (let k = 0; k < K; k++) {
        const d = d2(P[i], C[k]);
        if (d < bd) {
          bd = d;
          b = k;
        }
      }
      if (A[i] !== b) {
        A[i] = b;
        ch++;
      }
    }
    if (!ch) break;
    const S = C.map(() => [0, 0, 0]);
    for (let i = 0; i < n; i++) {
      const s = S[A[i]];
      s[0] += P[i].x;
      s[1] += P[i].y;
      s[2]++;
    }
    C.forEach((c, k) => {
      if (S[k][2]) {
        c.x = S[k][0] / S[k][2];
        c.y = S[k][1] / S[k][2];
      }
    });
  }
  let J = 0;
  for (let i = 0; i < n; i++) J += d2(P[i], C[A[i]]);
  return { C, A, J, it };
}
function jacobi(A) {
  const n = A.length,
    a = A.map((r) => r.slice()),
    V = a.map((_, i) => a.map((_, j) => (i === j ? 1 : 0)));
  for (let sw = 0; sw < 80; sw++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]),
          tt = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1)),
          c = 1 / Math.sqrt(tt * tt + 1),
          s = tt * c;
        for (let k = 0; k < n; k++) {
          const x = a[k][p],
            y = a[k][q];
          a[k][p] = c * x - s * y;
          a[k][q] = s * x + c * y;
        }
        for (let k = 0; k < n; k++) {
          const x = a[p][k],
            y = a[q][k];
          a[p][k] = c * x - s * y;
          a[q][k] = s * x + c * y;
        }
        for (let k = 0; k < n; k++) {
          const x = V[k][p],
            y = V[k][q];
          V[k][p] = c * x - s * y;
          V[k][q] = s * x + c * y;
        }
      }
  }
  const idx = a.map((_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return {
    val: idx.map((i) => a[i][i]),
    vec: idx.map((i) => {
      const v = V.map((r) => r[i]);
      let s = 0;
      for (const x of v) s += x;
      return s < 0 ? v.map((x) => -x) : v;
    }),
  };
}

const UNS = [
  {
    id: 'kmeans',
    tab: 'k-means',
    title: 'k-means: assign, move, repeat',
    lede: 'Give the computer a cloud of points and a number K, and it finds K groups with no labels to learn from. The whole algorithm is two steps repeated. Where it starts decides where it ends, which is why the starting rule called k-means++ matters so much.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The points, the centres (squares) and each centre’s territory. The lines trace where each centre has been. Drag a centre to move it</figcaption><canvas class="short" id="km-c" role="img" aria-label="Points coloured by cluster with the cluster centres"></canvas></figure>
 <figure class="fig"><figcaption>Top: this run, step by step. Middle: 150 runs from each kind of start. Bottom: the best result for each K</figcaption><canvas class="sq" id="km-r" role="img" aria-label="Sum of squares by step, a comparison of random and k-means++ starts, and the elbow curve"></canvas></figure></div>
 <figure class="fig" style="margin-top:20px"><figcaption>Where every centroid is after each round, and how far it moved</figcaption><div class="ctab" id="km-tab"></div></figure>
 ${uHow('km', ['Choose K. Place K starting centres: all at random, or one at a time with k-means++.', 'Assign: colour every point by its nearest centre.', 'Move: shift each centre to the average of the points that carry its colour.', 'Repeat steps 2 and 3 until no point changes colour.'])}
 <div class="controls">${uBtns('km', 'Run', 'One step', 'rounds')}
 ${grp('The model', slider('km-K', 'Number of clusters K', 1, 9, 1, 5) + `<div class="row seg" id="km-init" role="group" aria-label="Starting rule"><button class="btn" type="button" data-v="rand">Random Start</button><button class="btn" type="button" data-v="pp">k-means++ Start</button><button class="btn" type="button" data-v="own">Your Own Start</button></div>`)}
 ${uDataHTML('km', ['five', 'shop', 'blobs', 'uneven', 'long', 'moons', 'none'], 'five', 30, 600, 250)}</div>
 <dl class="read">${rd('ph', 'Rounds so far')}${rd('J', 'Sum of squared distances', 'hl')}${rd('best', 'Best known for this K')}${rd('ra', 'Random start: runs that reach the best', 'hl2')}${rd('rp', 'k-means++: runs that reach the best', 'hl3')}${rd('ia', 'Rounds needed, random / k-means++')}</dl>`,
    tryit:
      'Read the table under the pictures as you step: each row is one round, and the moves shrink toward zero. Choose Your Own Start and click all the centres into one group: can the algorithm recover? With Five Groups and Random Start, press Start over a few times and step through. How often do two centres begin in the same group, and can they ever leave it? Switch to k-means++ and step through the seeding: watch which dots grow before each pick. Then compare the two rows of bars in the middle picture. Finally choose Two Moons with K = 2: no starting rule can rescue that one. Why?',
    math: String.raw`<h4>Intuition</h4><p>Imagine a city that wants to open K post offices so that people walk as little as possible. If the offices were already built, each household would simply use its nearest one. If the households were already split into K districts, the best place for each office would be the centre of its district. k-means does not know either answer, so it guesses the offices and then alternates: send everyone to the nearest office, move each office to the middle of the people it serves, and again. Each of the two moves can only shorten the total walk, so the process must settle. What it settles on depends on the first guess.</p>
 <h4>What it minimises</h4><p>Let \(\mu_1,\dots,\mu_K\) be the centres and \(c(i)\) the cluster of point \(x_i\). k-means minimises the within-cluster sum of squares (also called inertia or distortion):</p>$$J=\sum_{i=1}^{n}\lVert x_i-\mu_{c(i)}\rVert^2 .$$
 <p><b>Assign</b> holds the centres fixed and picks the best \(c(i)\) for each point: \(c(i)=\arg\min_k\lVert x_i-\mu_k\rVert^2\). <b>Move</b> holds the assignments fixed and picks the best centres. Setting \(\partial J/\partial\mu_k=0\) gives</p>$$\mu_k=\frac{1}{n_k}\sum_{i:\,c(i)=k}x_i ,$$<p>the plain average, which is where the name comes from. Neither step can raise \(J\), and there are only finitely many ways to split \(n\) points into \(K\) groups, so the algorithm stops after a finite number of rounds. The top-right picture shows \(J\) falling at every step.</p>
 <h4>Why the start matters</h4><p>\(J\) is not convex. It has many local minima, and the two steps only ever go downhill, so k-means lands in whichever valley the start belongs to. The classic failure: two starting centres fall inside one true group and none in another. The two centres split their group between them, a single centre elsewhere is left covering two groups, and no assign or move step can carry a centre across the gap, because every step is locally an improvement.</p>
 <h4>k-means++</h4><p>k-means++ changes only how the K starting centres are chosen:</p><ol><li>Pick the first centre uniformly at random from the data points.</li><li>For every point compute \(D(x)^2\), its squared distance to the nearest centre chosen so far.</li><li>Pick the next centre from the data points with probability \(\dfrac{D(x)^2}{\sum_{x'}D(x')^2}\).</li><li>Repeat steps 2 and 3 until there are K centres, then run ordinary k-means.</li></ol>
 <p>A point sitting next to an existing centre has \(D^2\approx 0\) and is almost never picked. A whole group with no centre yet holds most of the total \(D^2\), so the next centre very probably lands there. The choice is random rather than "always the farthest point" on purpose: the farthest point is often an outlier, while sampling in proportion to \(D^2\) favours far regions that contain many points.</p>
 <p>Arthur and Vassilvitskii (2007) proved that this start alone, before any assign or move step, already gives \(\mathbb{E}[J]\le 8(\ln K+2)\,J_{\text{opt}}\). Random starts carry no such guarantee and can be arbitrarily bad. The cost is K passes over the data, repaid by fewer rounds afterwards. In practice both are also restarted several times and the run with the lowest \(J\) is kept.</p>
 <h4>Choosing K</h4><p>\(J\) always falls when K grows, reaching zero at \(K=n\), so the lowest \(J\) cannot choose K. The usual tools are the <b>elbow</b>, the K after which the curve in the bottom picture flattens, and the <b>silhouette</b>, which compares each point's distance to its own cluster with its distance to the next nearest one. Neither is a proof. With the No Groups data set the elbow curve is smooth, which is the honest answer.</p>
 <h4>What it assumes</h4><p>Because \(J\) uses plain squared distance, k-means prefers round groups of similar size and spread. It cuts long thin groups across the middle, splits a large group to tidy up two small ones, and cannot follow curved shapes at all. Rescaling one column changes the answer, so standardise features first. Gaussian mixtures relax the "round and equal" assumption, and DBSCAN drops the idea of a centre altogether.</p>`,
    q: 'k-means++ picks centres far from the existing ones. Why not simply always pick the single farthest point?',
    a: 'The farthest point from the current centres is very often an outlier: one stray point at the edge of the data. A centre placed on it is wasted. Sampling with probability proportional to squared distance means a far region with two hundred points is two hundred times more likely to receive the centre than one lonely point at the same distance. It is also what makes the proof of the 8(ln K + 2) guarantee work.',
    init(root) {
      const c = $('#km-c', root),
        cr = $('#km-r', root),
        live = $('#km-live', root),
        S = { K: 5, init: 'rand', ds: 'five', n: 250, seed: 3 },
        NR = 150;
      let P = [],
        C = [],
        prev = null,
        A = [],
        ph = 'seed',
        it = 0,
        hist = [],
        msg = '',
        run = 1,
        R = rng(1),
        mv = 0,
        wait = 0,
        fr = 0,
        pb,
        race = null,
        elbow = [],
        cu = -1,
        drag = -1,
        CH = [],
        tv = 0,
        tvd = -1;
      const near = (p) => {
        let b = 0,
          bd = 1e18;
        for (let k = 0; k < C.length; k++) {
          const d = d2(p, C[k]);
          if (d < bd) {
            bd = d;
            b = k;
          }
        }
        return [b, bd];
      };
      const J = () => {
        let s = 0;
        P.forEach((p, i) => {
          if (A[i] >= 0 && C[A[i]]) s += d2(p, C[A[i]]);
        });
        return s;
      };
      const snap = (lab) => {
        const Cs = C.map((q) => ({ x: q.x, y: q.y }));
        let j = 0;
        for (const p of P) {
          let bd = 1e18;
          for (const q of Cs) bd = Math.min(bd, d2(p, q));
          j += bd;
        }
        CH.push({ lab, C: Cs, J: j });
        tv++;
      };
      const doRace = () => {
        if (P.length < S.K) {
          race = null;
          return;
        }
        const r = rng(77 + S.seed * 3 + S.K),
          go = (pp) => {
            const o = [];
            for (let i = 0; i < NR; i++) o.push(kmRun(P, kmSeed(P, S.K, pp, r)));
            return o;
          },
          a = go(false),
          b = go(true);
        const best = Math.min(...a.map((q) => q.J), ...b.map((q) => q.J)),
          sum = (o) => ({
            J: o.map((q) => q.J),
            ok: o.filter((q) => q.J <= best * 1.02 + 1e-9).length / o.length,
            it: o.reduce((s, q) => s + q.it, 0) / o.length,
          });
        race = {
          best,
          a: sum(a),
          b: sum(b),
          max: Math.max(...a.map((q) => q.J), ...b.map((q) => q.J)),
        };
      };
      const doElbow = () => {
        const r = rng(5 + S.seed);
        elbow = [];
        for (let k = 1; k <= 9; k++) {
          let m = 1e99;
          if (P.length >= k)
            for (let i = 0; i < 5; i++) m = Math.min(m, kmRun(P, kmSeed(P, k, true, r)).J);
          elbow.push(m < 1e98 ? m : NaN);
        }
      };
      const reset = () => {
        R = rng(1000 + run * 7 + S.seed);
        C = [];
        prev = null;
        A = new Array(P.length).fill(-1);
        ph = 'seed';
        it = 0;
        hist = [];
        mv = 0;
        wait = 0;
        cu = -1;
        CH = [];
        tv++;
        msg =
          S.init === 'own'
            ? 'No centres yet. Click on the left picture ' +
              S.K +
              ' times to place the starting centres yourself.'
            : 'No centres yet. ' +
              (S.init === 'pp'
                ? 'k-means++ will place them one at a time.'
                : 'A random start picks ' + S.K + ' data points all at once.') +
              ' Press Run or One step.';
      };
      const step = () => {
        if (P.length < S.K) return;
        if (ph === 'done') {
          run++;
          reset();
          return;
        }
        if (ph === 'seed') {
          cu = 0;
          if (S.init === 'own') {
            msg =
              'Your own start: click on the left picture to place centre ' +
              (C.length + 1) +
              ' of ' +
              S.K +
              '.';
            return;
          }
          if (S.init === 'rand') {
            C = kmSeed(P, S.K, false, R);
            snap('start');
            ph = 'assign';
            msg =
              'Random start: ' +
              C.length +
              ' data points were picked at random to be the first centres. Nothing stops two of them from landing in the same group.';
          } else {
            if (!C.length) {
              const i = Math.floor(R() * P.length);
              C.push({ x: P[i].x, y: P[i].y });
              msg =
                'k-means++: the first centre is a data point picked at random. Every other point is now drawn in proportion to its squared distance from that centre: the bigger the dot, the more likely it is to be picked next.';
            } else {
              const D = P.map((p) => near(p)[1]);
              let tot = 0;
              for (const v of D) tot += v;
              if (tot <= 0) {
                ph = 'assign';
                return;
              }
              let u = R() * tot,
                i = 0;
              while (i < P.length - 1 && (u -= D[i]) > 0) i++;
              const srt = D.slice().sort((a, b) => a - b);
              let lo = 0;
              for (let k = 0; k < srt.length / 2; k++) lo += srt[k];
              C.push({ x: P[i].x, y: P[i].y });
              msg =
                'k-means++: centre ' +
                C.length +
                ' of ' +
                S.K +
                ' was drawn with probability proportional to squared distance from the nearest existing centre. The point picked was ' +
                f2(Math.sqrt(D[i])) +
                ' away. The half of the points closest to an existing centre shared only ' +
                ((100 * lo) / tot).toFixed(1) +
                '% of the chance.';
            }
            if (C.length >= S.K) {
              ph = 'assign';
              snap('start');
              msg += ' All ' + S.K + ' centres are placed.';
            }
          }
          return;
        }
        if (ph === 'assign') {
          cu = 1;
          let ch = 0;
          P.forEach((p, i) => {
            const b = near(p)[0];
            if (A[i] !== b) {
              A[i] = b;
              ch++;
            }
          });
          const j = J();
          hist.push(j);
          if (!ch && it > 0) {
            ph = 'done';
            cu = 3;
            msg =
              'No point changed cluster, so the centres will not move again. Converged after ' +
              it +
              ' round' +
              (it === 1 ? '' : 's') +
              ' with sum of squares ' +
              f2(j) +
              '.' +
              (race
                ? j <= race.best * 1.02 + 1e-9
                  ? ' This matches the best arrangement found in ' + 2 * NR + ' other runs.'
                  : ' The best known arrangement has ' +
                    f2(race.best) +
                    ': this start was trapped in a worse local minimum.'
                : '');
          } else {
            ph = 'move';
            msg =
              'Assign: every point takes the colour of its nearest centre. ' +
              ch +
              ' point' +
              (ch === 1 ? '' : 's') +
              ' changed cluster. Sum of squared distances: ' +
              f2(j) +
              '.';
          }
          return;
        }
        if (ph === 'move') {
          cu = 2;
          prev = C.map((q) => ({ ...q }));
          const Sx = C.map(() => [0, 0, 0]);
          P.forEach((p, i) => {
            const s = Sx[A[i]];
            s[0] += p.x;
            s[1] += p.y;
            s[2]++;
          });
          let far = 0,
            emp = 0;
          C.forEach((q, k) => {
            if (Sx[k][2]) {
              const nx = Sx[k][0] / Sx[k][2],
                ny = Sx[k][1] / Sx[k][2];
              far = Math.max(far, Math.hypot(nx - q.x, ny - q.y));
              q.x = nx;
              q.y = ny;
            } else emp++;
          });
          it++;
          const j = J();
          hist.push(j);
          mv = RM ? 0 : 18;
          ph = 'assign';
          snap('after round ' + it);
          msg =
            'Move: each centre jumps to the average of its own points. The largest move was ' +
            f2(far) +
            '. Sum of squared distances: ' +
            f2(j) +
            '.' +
            (emp
              ? ' ' +
                emp +
                ' centre' +
                (emp === 1 ? ' has' : 's have') +
                ' no points and stay' +
                (emp === 1 ? 's' : '') +
                ' put.'
              : '');
        }
      };
      const tp = testPoint(
        root,
        'km',
        ['Your x', 'Your y'],
        [0.5, 0.5],
        [
          [-3.8, 3.8],
          [-2.85, 2.85],
        ],
      );
      const draw = () => {
        const t = T(),
          col = UC(t);
        {
          const V = uView(c),
            { g, w, h, X, Y, s } = V;
          if (C.length === S.K && ph !== 'seed') {
            const cs = 9;
            for (let py = 0; py < h; py += cs)
              for (let px = 0; px < w; px += cs) {
                const b = near({ x: (px + cs / 2 - w / 2) / s, y: -(py + cs / 2 - h / 2) / s })[0];
                g.fillStyle = mix(col[b % 8], col[b % 8], 0, 0.11);
                g.fillRect(px, py, cs, cs);
              }
          }
          uFrame(
            V,
            t,
            S.ds === 'shop' ? 'visits per month (standardised)' : 'x',
            S.ds === 'shop' ? 'average basket (standardised)' : 'y',
          );
          const sh = ph === 'seed' && S.init === 'pp' && C.length > 0;
          let D = null,
            dm = 1;
          if (sh) {
            D = P.map((p) => near(p)[1]);
            dm = Math.max(1e-9, ...D);
          }
          P.forEach((p, i) => {
            g.beginPath();
            if (sh) {
              g.arc(X(p.x), Y(p.y), 1.4 + 8.5 * Math.sqrt(D[i] / dm), 0, 7);
              g.fillStyle = mix(t.ink, t.ink, 0, 0.34);
            } else {
              g.arc(X(p.x), Y(p.y), 3.4, 0, 7);
              g.fillStyle = A[i] >= 0 ? col[A[i] % 8] : mix(t.ink, t.ink, 0, 0.42);
            }
            g.fill();
          });
          if (CH.length > 1)
            C.forEach((q, k) => {
              g.strokeStyle = col[k % 8];
              g.lineWidth = 2;
              g.beginPath();
              CH.forEach((sn, i) => {
                const z = sn.C[k];
                if (!z) return;
                i ? g.lineTo(X(z.x), Y(z.y)) : g.moveTo(X(z.x), Y(z.y));
              });
              g.stroke();
              CH.forEach((sn, i) => {
                const z = sn.C[k];
                if (!z) return;
                g.fillStyle = t.bg;
                g.beginPath();
                g.arc(X(z.x), Y(z.y), i ? 3 : 5, 0, 7);
                g.fill();
                g.stroke();
              });
            });
          const u = prev && mv > 0 ? 1 - mv / 18 : 1;
          C.forEach((q, k) => {
            const o = prev && prev[k] ? prev[k] : q,
              x = X(o.x + (q.x - o.x) * u),
              y = Y(o.y + (q.y - o.y) * u);
            if (prev && prev[k]) {
              g.strokeStyle = t.ink;
              g.lineWidth = 1.4;
              g.setLineDash([4, 4]);
              g.beginPath();
              g.moveTo(X(o.x), Y(o.y));
              g.lineTo(x, y);
              g.stroke();
              g.setLineDash([]);
            }
            g.fillStyle = col[k % 8];
            g.strokeStyle = t.bg;
            g.lineWidth = 5;
            g.beginPath();
            g.rect(x - 8, y - 8, 16, 16);
            g.stroke();
            g.fill();
            g.strokeStyle = t.ink;
            g.lineWidth = 1.6;
            g.strokeRect(x - 8, y - 8, 16, 16);
          });
          if (C.length === S.K) markTP(g, X(tp.v[0]), Y(tp.v[1]), t);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            sh
              ? 'bigger dot = more likely to be the next centre'
              : 'run ' + run + ', ' + (S.init === 'pp' ? 'k-means++ start' : 'random start'),
            8,
            h - 8,
          );
        }
        {
          const { g, w, h } = fit(cr),
            bh = h / 3;
          const L = uLine(g, t, 0, 0, w, bh, hist, {
            title: 'sum of squared distances after each step of this run',
            dots: true,
            n: Math.max(8, hist.length),
            ym: Math.max(race ? race.best * 1.6 : 1, ...hist),
          });
          if (race) {
            g.setLineDash([5, 4]);
            g.strokeStyle = t.gold;
            g.lineWidth = 1.4;
            g.beginPath();
            g.moveTo(L.m.l, L.Y(race.best));
            g.lineTo(w - L.m.r, L.Y(race.best));
            g.stroke();
            g.setLineDash([]);
            g.fillStyle = t.cut;
            g.textAlign = 'right';
            g.fillText('best known', w - L.m.r, L.Y(race.best) - 5);
            const y0 = bh,
              m = { l: 44, r: 12, t: 40, b: 20 },
              nb = 16,
              lo = race.best,
              hi = Math.max(race.max, lo * 1.0001 + 1e-9),
              bw = (w - m.l - m.r) / nb,
              H = bh - m.t - m.b,
              cnt = (o) => {
                const q = new Array(nb).fill(0);
                for (const v of o.J) q[Math.min(nb - 1, Math.floor(((v - lo) / (hi - lo)) * nb))]++;
                return q;
              },
              ca = cnt(race.a),
              cb = cnt(race.b),
              cm = Math.max(1, ...ca, ...cb);
            for (let i = 0; i < nb; i++) {
              g.fillStyle = t.ok;
              g.fillRect(
                m.l + i * bw + 1,
                y0 + bh - m.b - (ca[i] / cm) * H,
                bw / 2 - 1.5,
                (ca[i] / cm) * H,
              );
              g.fillStyle = t.accent;
              g.fillRect(
                m.l + i * bw + bw / 2,
                y0 + bh - m.b - (cb[i] / cm) * H,
                bw / 2 - 1.5,
                (cb[i] / cm) * H,
              );
            }
            g.strokeStyle = t.muted;
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(m.l, y0 + bh - m.b);
            g.lineTo(w - m.r, y0 + bh - m.b);
            g.stroke();
            g.font = MONO;
            g.textAlign = 'left';
            g.fillStyle = t.ok;
            g.fillText(
              'random start: ' +
                (100 * race.a.ok).toFixed(0) +
                '% of ' +
                NR +
                ' runs reach the best',
              8,
              y0 + 13,
            );
            g.fillStyle = t.accent;
            g.fillText(
              'k-means++ start: ' +
                (100 * race.b.ok).toFixed(0) +
                '% of ' +
                NR +
                ' runs reach the best',
              8,
              y0 + 27,
            );
            g.fillStyle = t.muted;
            g.fillText('best', m.l, y0 + bh - 5);
            g.textAlign = 'right';
            g.fillText('final sum of squares: worse →', w - m.r, y0 + bh - 5);
          }
          const E = uLine(g, t, 0, 2 * bh, w, bh, elbow, {
            title: 'lowest sum of squares found for K = 1 … 9: look for the bend',
            dots: true,
            col: t.muted,
          });
          g.font = MONO;
          g.textAlign = 'center';
          elbow.forEach((v, i) => {
            g.fillStyle = t.muted;
            g.fillText(String(i + 1), E.X(i), h - 5);
            if (i + 1 === S.K && isFinite(v)) {
              g.fillStyle = t.gold;
              g.beginPath();
              g.arc(E.X(i), E.Y(v), 6, 0, 7);
              g.fill();
            }
          });
        }
        live.textContent = msg;
        uCur(root, 'km', cu);
        if (tvd !== tv) {
          tvd = tv;
          const tb = $('#km-tab', root);
          tb.innerHTML = !CH.length
            ? '<p class="status">The centroids are not all placed yet.</p>'
            : '<table><thead><tr><th>when</th>' +
              CH[0].C.map(
                (_, k) =>
                  '<th><i style="background:' + col[k % 8] + '"></i>centroid ' + (k + 1) + '</th>',
              ).join('') +
              '<th>largest move</th><th>sum of squares</th></tr></thead><tbody>' +
              CH.map(
                (sn, i) =>
                  '<tr><td>' +
                  sn.lab +
                  '</td>' +
                  sn.C.map((z) => '<td>(' + f2(z.x) + ', ' + f2(z.y) + ')</td>').join('') +
                  '<td>' +
                  (i
                    ? f2(
                        Math.max(
                          ...sn.C.map((z, k) =>
                            Math.hypot(z.x - CH[i - 1].C[k].x, z.y - CH[i - 1].C[k].y),
                          ),
                        ),
                      )
                    : '–') +
                  '</td><td>' +
                  f2(sn.J) +
                  '</td></tr>',
              ).join('') +
              '</tbody></table>';
          tb.scrollTop = tb.scrollHeight;
        }
        if (C.length === S.K) {
          const [b, bd] = near({ x: tp.v[0], y: tp.v[1] });
          tp.out.textContent =
            'Your point is ' +
            f2(Math.sqrt(bd)) +
            ' from centre ' +
            (b + 1) +
            ', which is its nearest, so k-means puts it in cluster ' +
            (b + 1) +
            '.';
        } else tp.out.textContent = 'Place the centres first: press One step.';
        setR(root, 'ph', String(it));
        setR(root, 'J', hist.length ? f2(hist[hist.length - 1]) : '–');
        setR(root, 'best', race ? f2(race.best) : '–');
        setR(root, 'ra', race ? (100 * race.a.ok).toFixed(0) + '%' : '–');
        setR(root, 'rp', race ? (100 * race.b.ok).toFixed(0) + '%' : '–');
        setR(root, 'ia', race ? race.a.it.toFixed(1) + ' / ' + race.b.it.toFixed(1) : '–');
      };
      tp.on = () => draw();
      const gen = uDataBind(root, 'km', S, () => {
        P = gen();
        doRace();
        doElbow();
        run = 1;
        reset();
        draw();
      });
      P = gen();
      doRace();
      doElbow();
      reset();
      const anim = Anim(() => {
        fr++;
        if (mv > 0) {
          mv--;
          draw();
          return;
        }
        if (wait > 0) {
          wait--;
          if (!wait) {
            run++;
            reset();
            draw();
          }
          return;
        }
        if (fr % (ph === 'seed' ? 44 : 30)) return;
        step();
        draw();
        chk();
      });
      pb = playBtn($('#km-play', root), anim);
      const chk = runCtl(pb, root, 'km', {
        done: () => ph === 'done',
        restart: () => {
          run++;
          reset();
        },
        count: () => it,
        what: 'rounds',
      });
      $('#km-step', root).onclick = () => {
        pb.stop();
        mv = 0;
        step();
        draw();
      };
      $('#km-reset', root).onclick = () => {
        pb.stop();
        run++;
        reset();
        draw();
      };
      bind(
        root,
        'km-K',
        (v) => {
          S.K = v;
          pb.stop();
          doRace();
          reset();
          draw();
        },
        (v) => String(v),
      );
      segBind(root, 'km-init', S.init, (v) => {
        S.init = v;
        pb.stop();
        reset();
        draw();
      });
      c.style.touchAction = 'none';
      c.addEventListener('pointerdown', (e) => {
        const r = c.getBoundingClientRect(),
          s = Math.min(r.width / 8, r.height / 6);
        drag = -1;
        C.forEach((q, k) => {
          if (
            Math.hypot(
              r.width / 2 + q.x * s - (e.clientX - r.left),
              r.height / 2 - q.y * s - (e.clientY - r.top),
            ) < 16
          )
            drag = k;
        });
        if (drag >= 0) {
          pb.stop();
          mv = 0;
          prev = null;
          c.setPointerCapture(e.pointerId);
        } else if (S.init === 'own' && ph === 'seed' && C.length < S.K) {
          const q = uInv(c, e);
          C.push({ x: clamp(q[0], -3.9, 3.9), y: clamp(q[1], -2.9, 2.9) });
          cu = 0;
          if (C.length >= S.K) {
            ph = 'assign';
            snap('your start');
            msg = 'All ' + S.K + ' centres are placed where you chose. Press Run or One step.';
          } else
            msg =
              'Centre ' +
              C.length +
              ' placed at (' +
              f2(q[0]) +
              ', ' +
              f2(q[1]) +
              '). Click to place centre ' +
              (C.length + 1) +
              ' of ' +
              S.K +
              '.';
          draw();
        }
      });
      c.addEventListener('pointermove', (e) => {
        if (drag < 0) return;
        const q = uInv(c, e);
        C[drag].x = clamp(q[0], -3.9, 3.9);
        C[drag].y = clamp(q[1], -2.9, 2.9);
        if (C.length === S.K) {
          P.forEach((p, i) => {
            A[i] = near(p)[0];
          });
          ph = 'move';
          cu = 1;
          msg =
            'You moved centre ' +
            (drag + 1) +
            '. The points have been reassigned to their nearest centres: sum of squared distances ' +
            f2(J()) +
            '. Press One step to let the centres move.';
        }
        draw();
      });
      ['pointerup', 'pointercancel'].forEach((n) =>
        c.addEventListener(n, () => {
          if (drag >= 0 && C.length === S.K) {
            snap('you moved centroid ' + (drag + 1));
            draw();
          }
          drag = -1;
        }),
      );
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'kmeans-picture',
    tab: 'k-means on a picture',
    title: 'A real job for k-means: fewer colours',
    lede: 'A photograph can contain tens of thousands of different colours. Treat every pixel as a point in red-green-blue space, run k-means, and repaint each pixel with the colour of its centre. The picture survives on K colours. Use the built-in picture or upload your own.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The original. Click a pixel to see where it goes</figcaption><canvas class="short" id="kp-o" role="img" aria-label="Original picture"></canvas></figure>
 <figure class="fig"><figcaption>The same picture repainted with K colours, and the K colours</figcaption><canvas class="short" id="kp-q" role="img" aria-label="Picture repainted with K colours"></canvas></figure></div>
 <div class="stage" style="margin-top:22px"><figure class="fig"><figcaption>Every dot is a pixel, placed by how much red, green and blue it has. Rings are the K centres</figcaption><canvas class="short" id="kp-c" role="img" aria-label="Pixels as points in colour space"></canvas></figure>
 <figure class="fig"><figcaption>Average colour error after each step</figcaption><canvas class="short" id="kp-e" role="img" aria-label="Colour error by step"></canvas></figure></div>
 ${uHow('kp', ['Read every pixel as a point with three coordinates: red, green, blue.', 'Pick K starting colours from the pixels (k-means++).', 'Assign: give every pixel the nearest of the K colours.', 'Move: change each of the K colours to the average of the pixels assigned to it.', 'Repeat steps 3 and 4 until the picture stops changing.'])}
 <div class="controls">${uBtns('kp', 'Run', 'One step', 'rounds')}
 ${grp('The model', slider('kp-K', 'Number of colours K', 2, 24, 1, 6) + slider('kp-az', 'Turn the colour cube', 0, 360, 1, 35))}
 ${grp('The picture', `<div class="row"><button class="btn" id="kp-built" type="button">Built-in Picture</button><label class="btn file">Upload a Picture<input type="file" id="kp-file" accept="image/*"></label></div><p class="status" id="kp-st" role="status"></p><p class="tpout" id="kp-tpo" role="status"></p>`)}</div>
 <dl class="read">${rd('n', 'Pixels')}${rd('uc', 'Different colours in the original')}${rd('k', 'Colours after k-means', 'hl')}${rd('bits', 'Bits per pixel, before → after')}${rd('err', 'Average colour error (0 to 255 scale)', 'hl2')}${rd('it', 'Rounds so far')}</dl>`,
    tryit:
      'Step through with K = 2, then 4, 8 and 16. At which K do you stop noticing the difference? Watch the colour cube while it runs: the rings slide into the dense strands of pixels. Then upload a photo of your own. Faces and skies need more colours than you expect, because smooth gradients break into bands.',
    math: String.raw`<h4>Intuition</h4><p>A box of crayons with only six crayons can still draw a recognisable landscape, as long as the six are well chosen: a sky blue, a leaf green, something for the roofs. Choosing those six is exactly a clustering problem. Pixels with similar colour should share one crayon, and each crayon should be the average of the pixels that use it, so that no pixel is repainted with a colour far from its own.</p>
 <h4>The picture as data</h4><p>A picture with \(n\) pixels becomes \(n\) points \(x_i=(r_i,g_i,b_i)\in[0,255]^3\). Where a pixel sits in the picture is ignored: only its colour matters. k-means with K centres \(\mu_1,\dots,\mu_K\) minimises</p>$$J=\sum_{i=1}^{n}\lVert x_i-\mu_{c(i)}\rVert^2 ,$$<p>the total squared colour error of the repainted picture. Dividing by \(n\) and taking the square root gives the average colour error shown under the pictures. Nothing about the algorithm changes from the two-dimensional lesson: three coordinates instead of two.</p>
 <h4>Why this compresses</h4><p>Storing a full-colour pixel takes 24 bits, eight for each of red, green and blue. After k-means each pixel only needs to say which of the K colours it uses, which takes \(\lceil\log_2 K\rceil\) bits, plus one small table of K colours for the whole picture. With \(K=16\) that is 4 bits per pixel, a sixth of the original. This is called <b>vector quantisation</b>, and the table of colours is the <b>codebook</b> or palette. The GIF image format works this way with up to 256 colours.</p>
 <h4>What to notice</h4><p>The centres do not spread evenly through the colour cube. They go where the pixels are: a picture that is mostly sky spends several of its colours on slightly different blues, because that lowers \(J\) most. Small bright details, such as a red roof, get a colour of their own only when K is large enough that splitting a big region again would gain less than fixing the small one. Smooth gradients turn into visible bands, because nearby pixels with nearly equal colours fall on opposite sides of a boundary between two centres.</p>
 <h4>The same idea elsewhere</h4><p>Replacing each item by the index of its nearest centre is used well beyond pictures: compressing audio, building visual "words" from image patches, speeding up nearest-neighbour search in large collections of embeddings (product quantisation), and the discrete codes of VQ-VAE models that you will meet in the generative part of this site.</p>`,
    q: 'Two pictures have the same number of pixels. One is a photo of a clear blue sky, the other a photo of a flower market. For the same K, which has the smaller colour error, and why?',
    a: 'The sky. Its pixels form one thin strand in colour space, so a few centres placed along that strand are close to every pixel. The flower market fills much more of the colour cube, so the same number of centres must each cover a wider range of colours and the average error is larger. How well k-means compresses depends on how concentrated the data is, not on how many points there are.',
    init(root) {
      const co = $('#kp-o', root),
        cq = $('#kp-q', root),
        cc = $('#kp-c', root),
        ce = $('#kp-e', root),
        live = $('#kp-live', root),
        st = $('#kp-st', root),
        tpo = $('#kp-tpo', root),
        S = { K: 6, az: 35 };
      let src = null,
        out = null,
        W = 0,
        H = 0,
        N = 0,
        px = null,
        C = [],
        A = null,
        ph = 'seed',
        it = 0,
        hist = [],
        msg = '',
        cu = 0,
        wait = 0,
        fr = 0,
        pb,
        run = 1,
        samp = [],
        uniq = 0,
        lastCh = 0,
        pick = -1,
        spin = false;
      const builtin = () => {
        const w = 160,
          h = 120,
          o = document.createElement('canvas');
        o.width = w;
        o.height = h;
        const g = o.getContext('2d'),
          r = rng(5);
        let gr = g.createLinearGradient(0, 0, 0, h * 0.62);
        gr.addColorStop(0, '#2a5b9c');
        gr.addColorStop(0.6, '#8fc0e8');
        gr.addColorStop(1, '#f6d9a8');
        g.fillStyle = gr;
        g.fillRect(0, 0, w, h);
        gr = g.createRadialGradient(w * 0.76, h * 0.24, 2, w * 0.76, h * 0.24, h * 0.24);
        gr.addColorStop(0, '#fff6c8');
        gr.addColorStop(0.35, '#ffd45e');
        gr.addColorStop(1, 'rgba(255,200,80,0)');
        g.fillStyle = gr;
        g.fillRect(0, 0, w, h);
        g.fillStyle = 'rgba(255,255,255,.85)';
        [
          [30, 22, 13],
          [44, 19, 10],
          [55, 24, 9],
          [110, 36, 8],
          [120, 33, 10],
        ].forEach((q) => {
          g.beginPath();
          g.ellipse(q[0], q[1], q[2], q[2] * 0.5, 0, 0, 7);
          g.fill();
        });
        g.fillStyle = '#6f8fa6';
        g.beginPath();
        g.moveTo(0, h);
        for (let x = 0; x <= w; x += 2)
          g.lineTo(
            x,
            h * 0.56 - h * 0.09 * Math.sin((x / w) * 5 + 1) - h * 0.03 * Math.sin((x / w) * 13),
          );
        g.lineTo(w, h);
        g.fill();
        gr = g.createLinearGradient(0, h * 0.58, 0, h);
        gr.addColorStop(0, '#6aa84f');
        gr.addColorStop(1, '#1f4a22');
        g.fillStyle = gr;
        g.beginPath();
        g.moveTo(0, h);
        for (let x = 0; x <= w; x += 2)
          g.lineTo(x, h * 0.69 - h * 0.06 * Math.sin((x / w) * 3.2 + 2.2));
        g.lineTo(w, h);
        g.fill();
        g.fillStyle = '#efe6d4';
        g.fillRect(34, 62, 36, 26);
        g.fillStyle = '#b5452f';
        g.beginPath();
        g.moveTo(29, 63);
        g.lineTo(52, 44);
        g.lineTo(75, 63);
        g.fill();
        g.fillStyle = '#5a3a22';
        g.fillRect(48, 72, 9, 16);
        g.fillStyle = '#ffd97a';
        g.fillRect(38, 68, 7, 7);
        g.fillRect(60, 68, 7, 7);
        g.fillStyle = '#4a3020';
        g.fillRect(112, 66, 5, 26);
        [
          ['#2f7d32', 114, 58, 15],
          ['#3f9a3d', 106, 64, 10],
          ['#286b2c', 123, 64, 10],
        ].forEach((q) => {
          g.fillStyle = q[0];
          g.beginPath();
          g.arc(q[1], q[2], q[3], 0, 7);
          g.fill();
        });
        for (let i = 0; i < 70; i++) {
          g.fillStyle = ['#e8443a', '#f2c230', '#f5f0e6', '#d65aa8'][i % 4];
          g.beginPath();
          g.arc(r() * w, h * 0.8 + r() * h * 0.19, 1.3, 0, 7);
          g.fill();
        }
        return o;
      };
      const load = (o) => {
        src = o;
        W = o.width;
        H = o.height;
        N = W * H;
        const d = o.getContext('2d').getImageData(0, 0, W, H).data;
        px = new Float32Array(N * 3);
        const seen = new Set();
        for (let i = 0; i < N; i++) {
          px[i * 3] = d[i * 4];
          px[i * 3 + 1] = d[i * 4 + 1];
          px[i * 3 + 2] = d[i * 4 + 2];
          seen.add((d[i * 4] << 16) | (d[i * 4 + 1] << 8) | d[i * 4 + 2]);
        }
        uniq = seen.size;
        out = document.createElement('canvas');
        out.width = W;
        out.height = H;
        const r = rng(3);
        samp = [];
        for (let i = 0; i < 1500; i++) samp.push(Math.floor(r() * N));
        pick = -1;
        tpo.textContent = '';
        run = 1;
        reset();
      };
      const dist = (i, q) =>
        (px[i * 3] - q[0]) ** 2 + (px[i * 3 + 1] - q[1]) ** 2 + (px[i * 3 + 2] - q[2]) ** 2;
      const nearC = (i) => {
        let b = 0,
          bd = 1e18;
        for (let k = 0; k < C.length; k++) {
          const d = dist(i, C[k]);
          if (d < bd) {
            bd = d;
            b = k;
          }
        }
        return [b, bd];
      };
      const paint = () => {
        const g = out.getContext('2d'),
          im = g.createImageData(W, H),
          d = im.data;
        for (let i = 0; i < N; i++) {
          const q = A && A[i] < C.length ? C[A[i]] : null;
          d[i * 4] = q ? q[0] : px[i * 3];
          d[i * 4 + 1] = q ? q[1] : px[i * 3 + 1];
          d[i * 4 + 2] = q ? q[2] : px[i * 3 + 2];
          d[i * 4 + 3] = 255;
        }
        g.putImageData(im, 0, 0);
      };
      const reset = () => {
        C = [];
        A = null;
        ph = 'seed';
        it = 0;
        hist = [];
        cu = 0;
        wait = 0;
        lastCh = 0;
        msg =
          'The picture has been read as ' +
          N.toLocaleString('en-US') +
          ' points in colour space. Press Run or One step to pick ' +
          S.K +
          ' starting colours.';
        if (out) out.getContext('2d').drawImage(src, 0, 0);
      };
      const step = () => {
        if (!px) return;
        if (ph === 'done') {
          run++;
          reset();
          return;
        }
        if (ph === 'seed') {
          const r = rng(40 + run * 11 + S.K),
            f = Math.floor(r() * N);
          C = [[px[f * 3], px[f * 3 + 1], px[f * 3 + 2]]];
          const D = new Float64Array(N);
          for (let i = 0; i < N; i++) D[i] = dist(i, C[0]);
          while (C.length < S.K) {
            let tot = 0;
            for (let i = 0; i < N; i++) tot += D[i];
            if (tot <= 0) break;
            let u = r() * tot,
              i = 0;
            while (i < N - 1 && (u -= D[i]) > 0) i++;
            const q = [px[i * 3], px[i * 3 + 1], px[i * 3 + 2]];
            C.push(q);
            for (let j = 0; j < N; j++) {
              const d = dist(j, q);
              if (d < D[j]) D[j] = d;
            }
          }
          A = null;
          ph = 'assign';
          cu = 1;
          msg =
            'k-means++ picked ' +
            C.length +
            ' starting colours from the pixels, each one likely to be far in colour from those already chosen. They are the rings in the colour cube.';
          return;
        }
        if (ph === 'assign') {
          cu = 2;
          if (!A) A = new Uint8Array(N).fill(255);
          let ch = 0,
            J = 0;
          for (let i = 0; i < N; i++) {
            const [b, bd] = nearC(i);
            if (A[i] !== b) {
              A[i] = b;
              ch++;
            }
            J += bd;
          }
          lastCh = ch;
          hist.push(Math.sqrt(J / N));
          paint();
          if ((ch === 0 && it > 0) || it >= 120) {
            ph = 'done';
            cu = 4;
            msg =
              (ch === 0
                ? 'Converged: no pixel changed colour, so the centres will not move again. It took ' +
                  it +
                  ' rounds.'
                : 'Reached the limit of 120 rounds with ' + ch + ' pixels still changing.') +
              ' Average colour error ' +
              uNum(Math.sqrt(J / N)) +
              ' on a 0 to 255 scale, using ' +
              C.length +
              ' colours in place of ' +
              uniq.toLocaleString('en-US') +
              '.';
          } else {
            ph = 'move';
            msg =
              'Assign: every pixel was repainted with the nearest of the ' +
              C.length +
              ' colours. ' +
              ch.toLocaleString('en-US') +
              ' pixels changed. Average colour error ' +
              uNum(Math.sqrt(J / N)) +
              '.';
          }
          return;
        }
        if (ph === 'move') {
          cu = 3;
          const Sx = C.map(() => [0, 0, 0, 0]);
          for (let i = 0; i < N; i++) {
            const s = Sx[A[i]];
            s[0] += px[i * 3];
            s[1] += px[i * 3 + 1];
            s[2] += px[i * 3 + 2];
            s[3]++;
          }
          let far = 0;
          C.forEach((q, k) => {
            const s = Sx[k];
            if (s[3]) {
              const nq = [s[0] / s[3], s[1] / s[3], s[2] / s[3]];
              far = Math.max(far, Math.hypot(nq[0] - q[0], nq[1] - q[1], nq[2] - q[2]));
              C[k] = nq;
            }
          });
          let J = 0;
          for (let i = 0; i < N; i++) J += dist(i, C[A[i]]);
          hist.push(Math.sqrt(J / N));
          it++;
          paint();
          ph = 'assign';
          msg =
            'Move: each of the ' +
            C.length +
            ' colours became the average of the pixels using it. The largest shift was ' +
            uNum(far) +
            ' colour units. Average colour error ' +
            uNum(Math.sqrt(J / N)) +
            '.';
        }
      };
      const rect = (w, h, fh) => {
        const s = Math.min(w / W, (h * fh) / H);
        return [(w - W * s) / 2, (h * fh - H * s) / 2, W * s, H * s];
      };
      const draw = () => {
        if (!px) return;
        const t = T(),
          hx = (q) =>
            'rgb(' + Math.round(q[0]) + ',' + Math.round(q[1]) + ',' + Math.round(q[2]) + ')';
        {
          const { g, w, h } = fit(co),
            R = rect(w, h, 0.88);
          g.imageSmoothingEnabled = false;
          g.drawImage(src, R[0], R[1], R[2], R[3]);
          if (pick >= 0) {
            const x = R[0] + (((pick % W) + 0.5) * R[2]) / W,
              y = R[1] + ((Math.floor(pick / W) + 0.5) * R[3]) / H;
            g.strokeStyle = '#fff';
            g.lineWidth = 3;
            g.beginPath();
            g.arc(x, y, 8, 0, 7);
            g.stroke();
            g.strokeStyle = '#000';
            g.lineWidth = 1.4;
            g.stroke();
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            W + ' × ' + H + ' pixels, ' + uniq.toLocaleString('en-US') + ' different colours',
            R[0],
            h * 0.88 + 16,
          );
        }
        {
          const { g, w, h } = fit(cq),
            R = rect(w, h, 0.88);
          g.imageSmoothingEnabled = false;
          g.drawImage(out, R[0], R[1], R[2], R[3]);
          const K = C.length,
            sw = Math.min(34, R[2] / Math.max(1, K));
          C.forEach((q, k) => {
            g.fillStyle = hx(q);
            g.fillRect(R[0] + k * sw, h * 0.88 + 5, sw - 2, h * 0.12 - 8);
          });
          if (!K) {
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText('no colours chosen yet', R[0], h * 0.88 + 16);
          }
        }
        {
          const { g, w, h } = fit(cc),
            a = (S.az * Math.PI) / 180,
            ca = Math.cos(a),
            sa = Math.sin(a),
            sc = Math.min(w * 0.62, h) * 0.5,
            pr = (r, gg, b) => {
              const x = r / 255 - 0.5,
                y = b / 255 - 0.5,
                z = gg / 255 - 0.5,
                xr = x * ca - y * sa,
                yr = x * sa + y * ca;
              return [w / 2 + sc * xr, h / 2 - sc * (z * 0.92 + yr * 0.36)];
            };
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          const cs = [0, 255];
          for (const i of cs)
            for (const j of cs) {
              [
                [
                  [0, i, j],
                  [255, i, j],
                ],
                [
                  [i, 0, j],
                  [i, 255, j],
                ],
                [
                  [i, j, 0],
                  [i, j, 255],
                ],
              ].forEach((e) => {
                const p = pr(...e[0]),
                  q = pr(...e[1]);
                g.beginPath();
                g.moveTo(p[0], p[1]);
                g.lineTo(q[0], q[1]);
                g.stroke();
              });
            }
          g.font = MONO;
          g.textAlign = 'center';
          [
            ['black', [0, 0, 0]],
            ['red', [255, 0, 0]],
            ['green', [0, 255, 0]],
            ['blue', [0, 0, 255]],
            ['white', [255, 255, 255]],
          ].forEach((q) => {
            const p = pr(...q[1]);
            g.fillStyle = t.muted;
            g.fillText(q[0], p[0], p[1] + (q[1][1] ? -6 : 14));
          });
          for (const i of samp) {
            const q = A && A[i] < C.length ? C[A[i]] : [px[i * 3], px[i * 3 + 1], px[i * 3 + 2]],
              p = pr(px[i * 3], px[i * 3 + 1], px[i * 3 + 2]);
            g.fillStyle = hx(q);
            g.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3);
          }
          C.forEach((q) => {
            const p = pr(...q);
            g.fillStyle = hx(q);
            g.strokeStyle = t.bg;
            g.lineWidth = 5;
            g.beginPath();
            g.arc(p[0], p[1], 8, 0, 7);
            g.stroke();
            g.fill();
            g.strokeStyle = t.ink;
            g.lineWidth = 1.8;
            g.stroke();
          });
          if (pick >= 0) {
            const p = pr(px[pick * 3], px[pick * 3 + 1], px[pick * 3 + 2]);
            markTP(g, p[0], p[1], t, false);
          }
        }
        {
          const { g, w, h } = fit(ce);
          const L = uLine(g, t, 0, 0, w, h, hist, {
            title: 'average colour error after each assign and each move',
            dots: true,
            n: Math.max(8, hist.length),
            col: t.ok,
          });
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('step', w - L.m.r, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'kp', cu);
        if (pick >= 0) {
          const q = [px[pick * 3], px[pick * 3 + 1], px[pick * 3 + 2]];
          tpo.textContent =
            'The pixel you clicked is ' +
            hx(q) +
            '.' +
            (C.length
              ? (() => {
                  const [b, bd] = nearC(pick);
                  return (
                    ' Its nearest palette colour is number ' +
                    (b + 1) +
                    ', ' +
                    hx(C[b]) +
                    ', ' +
                    uNum(Math.sqrt(bd)) +
                    ' colour units away, so it is repainted with that.'
                  );
                })()
              : ' Press One step to choose the palette.');
        }
        const bb = Math.max(1, Math.ceil(Math.log2(Math.max(2, S.K))));
        setR(root, 'n', N.toLocaleString('en-US'));
        setR(root, 'uc', uniq.toLocaleString('en-US'));
        setR(root, 'k', C.length ? String(C.length) : '–');
        setR(root, 'bits', '24 → ' + bb + ' (' + ((100 * bb) / 24).toFixed(0) + '% of the size)');
        setR(root, 'err', hist.length ? uNum(hist[hist.length - 1]) : '–');
        setR(root, 'it', String(it));
      };
      load(builtin());
      for (let i = 0; i < 260 && ph !== 'done'; i++) step();
      const anim = Anim(() => {
        fr++;
        if (spin) {
          S.az = (S.az + 0.35) % 360;
          azb.set(S.az);
        }
        if (wait > 0) {
          wait--;
          if (!wait) {
            run++;
            reset();
          }
          draw();
          return;
        }
        if (fr % (it < 3 ? 34 : 9) === 0) {
          step();
          draw();
          chk();
          return;
        }
        draw();
      });
      pb = playBtn($('#kp-play', root), anim);
      const ps = pb.stop,
        pst = pb.start;
      pb.stop = () => {
        spin = false;
        ps();
      };
      pb.start = () => {
        spin = true;
        pst();
      };
      const chk = runCtl(pb, root, 'kp', {
        done: () => ph === 'done',
        restart: () => {
          run++;
          reset();
        },
        count: () => it,
        what: 'rounds',
      });
      $('#kp-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#kp-reset', root).onclick = () => {
        pb.stop();
        run++;
        reset();
        draw();
      };
      bind(
        root,
        'kp-K',
        (v) => {
          S.K = v;
          pb.stop();
          reset();
          for (let i = 0; i < 260 && ph !== 'done'; i++) step();
          draw();
        },
        (v) => String(v),
      );
      const azb = bind(
        root,
        'kp-az',
        (v) => {
          S.az = v;
          draw();
        },
        (v) => Math.round(v) + '°',
      );
      $('#kp-built', root).onclick = () => {
        pb.stop();
        st.textContent = '';
        load(builtin());
        draw();
      };
      $('#kp-file', root).onchange = (e) => {
        const f = e.target.files[0];
        e.target.value = '';
        if (!f) return;
        const use = (im) => {
            const iw = im.width,
              ih = im.height;
            if (!iw || !ih) {
              st.textContent = 'That picture could not be read. Try a JPEG or PNG file.';
              return;
            }
            const s = Math.min(1, 160 / Math.max(iw, ih)),
              o = document.createElement('canvas');
            o.width = Math.max(8, Math.round(iw * s));
            o.height = Math.max(8, Math.round(ih * s));
            o.getContext('2d').drawImage(im, 0, 0, o.width, o.height);
            pb.stop();
            load(o);
            st.textContent =
              'Loaded ' +
              f.name +
              ', shrunk to ' +
              o.width +
              ' × ' +
              o.height +
              ' pixels so that every step stays fast. The picture stays in your browser.';
            draw();
          },
          bad = () => {
            st.textContent = 'That file could not be opened as a picture. Try a JPEG or PNG file.';
          };
        if (window.createImageBitmap) createImageBitmap(f).then(use, bad);
        else {
          const fr2 = new FileReader();
          fr2.onerror = bad;
          fr2.onload = () => {
            const im = new Image();
            im.onload = () => use(im);
            im.onerror = bad;
            im.src = fr2.result;
          };
          fr2.readAsDataURL(f);
        }
      };
      co.addEventListener('click', (e) => {
        const r = co.getBoundingClientRect(),
          R = rect(r.width, r.height, 0.88),
          x = Math.floor(((e.clientX - r.left - R[0]) / R[2]) * W),
          y = Math.floor(((e.clientY - r.top - R[1]) / R[3]) * H);
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        pick = y * W + x;
        draw();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'gmm',
    tab: 'Gaussian mixtures and EM',
    title: 'Soft clusters: Gaussian mixtures and EM',
    lede: 'k-means gives every point to exactly one cluster. A Gaussian mixture says each cluster is a bell shape, and lets a point belong 70% to one cluster and 30% to another. The EM algorithm fits it with two alternating steps. Start in one dimension, where every part of the idea can be seen, then move to two.',
    stage: `<div class="row seg" id="gm-dim" role="group" aria-label="Number of dimensions"><button class="btn" type="button" data-v="1">One Dimension</button><button class="btn" type="button" data-v="2">Two Dimensions</button><button class="btn" type="button" data-v="3">Two Dimensions, Seen as Hills</button></div>
 <p class="cap" id="gm-cap"></p>
 <div class="stage"><figure class="fig"><figcaption id="gm-fc">The data and the clusters</figcaption><canvas class="short" id="gm-c" role="img" aria-label="Data with the fitted Gaussian clusters"></canvas><div id="gm-vb" style="display:none">${viewbar}</div></figure>
 <figure class="fig"><figcaption>Top: how the clusters share your point. Middle: each cluster’s share of all the points. Bottom: log-likelihood after each step</figcaption><canvas class="short" id="gm-r" role="img" aria-label="Responsibilities of your point, cluster weights and the log-likelihood curve"></canvas></figure></div>
 <figure class="fig" style="margin-top:20px"><figcaption>The numbers behind the picture: every cluster’s weight, centre and spread right now</figcaption><div class="ctab" id="gm-tab"></div></figure>
 ${uHow('gm', ['Start with K bell shapes: a centre, a spread and a weight for each.', 'E-step: for every point, work out what share of it each bell is responsible for. Nothing moves; only the colours change.', 'M-step: move each bell to the weighted average of its points, set its spread to their weighted spread, and set its weight to its share of all the points. Only the bells change.', 'Repeat steps 2 and 3 until the log-likelihood stops rising.'])}
 <div class="controls">${uBtns('gm', 'Run', 'One step', 'rounds')}
 ${grp('The model', slider('gm-K', 'Number of clusters K', 1, 6, 1, 2) + `<label class="ctl" for="gm-cov" id="gm-covl"><span>Shape each bell may take</span><select id="gm-cov"><option value="full">Any ellipse (full covariance)</option><option value="diag">Ellipse along the axes (diagonal)</option><option value="sph">Circle (spherical)</option></select></label>`)}
 ${uDataHTML('gm', ['h2', 'h3', 'hov'], 'h2', 40, 500, 240)}</div>
 <dl class="read">${rd('it', 'Rounds so far')}${rd('ll', 'Log-likelihood', 'hl')}${rd('dl', 'Rise in the last round')}${rd('sh', 'Points shared between clusters (none above 90%)', 'hl2')}${rd('st', 'Status')}</dl>`,
    tryit:
      'In One Dimension, press Start over and use One step. After each E-step look only at the colours of the dots and the band under them; after each M-step look only at how the curves moved. Put your point where two curves cross and read the top-right bar. Then choose Three Humps and set K to 3. In Two Dimensions, pick Three Stretched Groups with K = 3 and watch the ellipses turn; then restrict the shape to Circle and see what is lost. Seen as Hills shows the same fit as a landscape.',
    math: String.raw`<h4>Intuition</h4><p>Suppose you measure the heights of a crowd without recording who is an adult and who is a child. The histogram has two humps. If you knew who belonged to which group, fitting a bell curve to each group would be easy. If you knew the two bell curves, you could say for any height how likely it is to be a child's. You know neither, so you guess the curves, compute everyone's probable group, refit the curves using those probabilities as weights, and repeat. That loop is the EM algorithm, and the pair of bell curves is a Gaussian mixture.</p>
 <h4>The model</h4><p>The data are assumed to come from K Gaussians. Cluster \(k\) has a weight \(\pi_k\) (its share of the data, with \(\sum_k\pi_k=1\)), a mean \(\boldsymbol\mu_k\) and a covariance matrix \(\boldsymbol\Sigma_k\). The density of a point is</p>$$p(\mathbf x)=\sum_{k=1}^{K}\pi_k\,\mathcal N(\mathbf x\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k).$$<p>We want the parameters that maximise the log-likelihood of the data,</p>$$\ell=\sum_{i=1}^{n}\log\sum_{k=1}^{K}\pi_k\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k).$$<p>The sum inside the logarithm prevents a closed-form answer, which is why an iterative method is needed.</p>
 <h4>E-step: responsibilities</h4><p>With the current parameters, Bayes' rule gives the probability that point \(i\) came from cluster \(k\):</p>$$r_{ik}=\frac{\pi_k\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{\sum_{j}\pi_j\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_j,\boldsymbol\Sigma_j)} .$$<p>These are the colours in the picture: a point halfway between two clusters is drawn in a blend of their colours.</p>
 <h4>M-step: weighted fits</h4><p>Treat each \(r_{ik}\) as a fraction of a point. With \(N_k=\sum_i r_{ik}\) the effective number of points in cluster \(k\),</p>$$\pi_k=\frac{N_k}{n},\qquad\boldsymbol\mu_k=\frac{1}{N_k}\sum_i r_{ik}\,\mathbf x_i,\qquad\boldsymbol\Sigma_k=\frac{1}{N_k}\sum_i r_{ik}\,(\mathbf x_i-\boldsymbol\mu_k)(\mathbf x_i-\boldsymbol\mu_k)^{\top}.$$<p>These are the ordinary maximum-likelihood formulas for a Gaussian from the first module, with every point counted according to its responsibility.</p>
 <h4>Why it works</h4><p>Each E-step followed by an M-step can never lower \(\ell\). The E-step builds a lower bound on \(\ell\) that touches it at the current parameters, and the M-step moves to the top of that bound. So the curve in the lower right rises at every step and flattens at a local maximum. Like k-means, EM finds a local optimum that depends on the start, so it is usually started from a k-means++ seeding and restarted a few times.</p>
 <h4>How it relates to k-means</h4><p>Force every covariance to be the same small circle, \(\boldsymbol\Sigma_k=\sigma^2\mathbf I\), and let \(\sigma\to0\). Each responsibility becomes 0 or 1, the E-step becomes "assign to the nearest centre" and the M-step becomes "move to the average". k-means is the hard, round limit of a Gaussian mixture. Choosing Circle in the controls shows the halfway case.</p>
 <h4>Cautions</h4><p>A bell can shrink onto a single point, where its density, and with it the likelihood, grows without bound. Implementations add a small constant to the diagonal of each \(\boldsymbol\Sigma_k\), as this page does. More clusters always raise the likelihood, so K is chosen with a penalty for the number of parameters, such as BIC \(=-2\ell+p\log n\), or by likelihood on held-out data. And a mixture of Gaussians still cannot follow a curved shape such as the two moons with K = 2: it needs several bells per moon.</p>`,
    q: 'After EM converges, a point has responsibilities 0.5 and 0.5 for two clusters. k-means would have put it firmly in one. Which answer is more useful?',
    a: 'The 0.5 and 0.5 is information that k-means throws away: the model is telling you that this point is on the border and that its membership is uncertain. If a decision depends on the cluster, for example which offer to send a customer, knowing that the assignment is a coin flip matters. A hard label can always be recovered from the soft one by taking the larger share, but not the other way round.',
    init(root) {
      const c = $('#gm-c', root),
        cr = $('#gm-r', root),
        live = $('#gm-live', root),
        cap = $('#gm-cap', root),
        fc = $('#gm-fc', root),
        vb = $('#gm-vb', root),
        tab = $('#gm-tab', root),
        dsel = $('#gm-ds', root),
        S = { dim: '1', K: 2, cov: 'full', ds: 'h2', n: 240, seed: 2 };
      let P = [],
        G = [],
        prevG = null,
        Rs = null,
        ph = 'E',
        it = 0,
        LL = [],
        msg = '',
        cu = 0,
        mv = 0,
        fr = 0,
        pb,
        run = 1,
        chk = () => false;
      const one = () => S.dim === '1',
        CAP = {
          1: 'Each dot under the axis is one measurement. Each coloured curve is one cluster and the black curve is their sum, which should follow the grey histogram. The band at the bottom shows, for every position, how the clusters would split a point that fell there.',
          2: 'Each ellipse is one cluster: the inner ring is one standard deviation, the outer ring two. A point’s colour is a blend of the cluster colours, in proportion to how the clusters share it.',
          3: 'The same fit as Two Dimensions, drawn as a landscape. Height is the mixture density, each hill is one cluster, and the dots on the floor are the data. Drag to rotate.',
        };
      const mk = () => {
        if (!one()) return uData(S.ds, S.n, S.seed);
        const r = rng(S.seed * 11 + 2),
          rn = randnFrom(r),
          spec = {
            h2: [
              [0.45, -1.5, 0.55],
              [0.55, 1.2, 0.75],
            ],
            h3: [
              [0.3, -2.3, 0.4],
              [0.4, 0, 0.5],
              [0.3, 2.2, 0.45],
            ],
            hov: [
              [0.5, -0.6, 0.8],
              [0.5, 0.9, 0.8],
            ],
          }[S.ds] || [[1, 0, 1]],
          o = [];
        for (let i = 0; i < S.n; i++) {
          let u = r(),
            k = 0;
          while (k < spec.length - 1 && (u -= spec[k][0]) > 0) k++;
          o.push({ x: clamp(spec[k][1] + spec[k][2] * rn(), -3.8, 3.8), y: 0 });
        }
        return o;
      };
      const d1 = (q, x) => Math.exp((-0.5 * (x - q.mx) ** 2) / q.a) / Math.sqrt(2 * Math.PI * q.a);
      const dG = (q, x, y) => {
        const det = Math.max(1e-12, q.a * q.d - q.b * q.b),
          dx = x - q.mx,
          dy = y - q.my;
        return (
          Math.exp((-0.5 * (q.d * dx * dx - 2 * q.b * dx * dy + q.a * dy * dy)) / det) /
          (2 * Math.PI * Math.sqrt(det))
        );
      };
      const dens = (q, x, y) => (one() ? d1(q, x) : dG(q, x, y));
      const respG = (GG, x, y) => {
          const v = GG.map((q) => q.pi * dens(q, x, y)),
            s = v.reduce((a, b) => a + b, 0);
          return { r: v.map((u) => (s > 0 ? u / s : 1 / GG.length)), s };
        },
        resp = (x, y) => respG(G, x, y);
      const where = (p) => (one() ? 'x = ' + f2(p[0]) : '(' + f2(p[0]) + ', ' + f2(p[1]) + ')');
      const reset = () => {
        const n = P.length || 1,
          mx = P.reduce((s, p) => s + p.x, 0) / n,
          my = P.reduce((s, p) => s + p.y, 0) / n,
          vx = P.reduce((s, p) => s + (p.x - mx) ** 2, 0) / n,
          vy = P.reduce((s, p) => s + (p.y - my) ** 2, 0) / n,
          v = one() ? Math.max(0.08, vx / (S.K * S.K)) : Math.max(0.12, (vx + vy) / 2 / S.K);
        G = kmSeed(P, S.K, true, rng(300 + run * 5 + S.seed)).map((q) => ({
          pi: 1 / S.K,
          mx: q.x,
          my: one() ? 0 : q.y,
          a: v,
          b: 0,
          d: one() ? 1 : v,
        }));
        prevG = null;
        Rs = null;
        ph = 'E';
        it = 0;
        LL = [];
        cu = 0;
        mv = 0;
        msg =
          'Start: ' +
          G.length +
          ' bell' +
          (G.length === 1 ? '' : 's') +
          ' placed on data points chosen by k-means++, all with the same spread and each with weight 1/' +
          G.length +
          '. The points are grey because nothing has been shared out yet. Press Run or One step.';
      };
      const step = () => {
        if (P.length < 2 || !G.length) return;
        if (ph === 'done') {
          run++;
          reset();
          return;
        }
        if (ph === 'E') {
          cu = 1;
          let ll = 0,
            sh = 0;
          Rs = P.map((p) => {
            const q = resp(p.x, p.y);
            ll += Math.log(q.s + 1e-300);
            if (Math.max(...q.r) < 0.9) sh++;
            return q.r;
          });
          const prevL = LL.length ? LL[LL.length - 1] : null;
          LL.push(ll);
          const ex = resp(tp.v[0], one() ? 0 : tp.v[1]).r;
          if ((prevL !== null && Math.abs(ll - prevL) < 1e-4 * P.length) || it >= 300) {
            ph = 'done';
            cu = 3;
            msg =
              (it >= 300
                ? 'Reached the limit of 300 rounds.'
                : 'Converged: the log-likelihood rose by less than a ten-thousandth per point in the last round, so the bells have stopped moving.') +
              ' It took ' +
              it +
              ' rounds and the log-likelihood is ' +
              uNum(ll) +
              '. ' +
              sh +
              ' of ' +
              P.length +
              ' points are still shared between clusters.';
          } else {
            ph = 'M';
            msg =
              'E-step: the bells stayed where they were, and every point was shared out among them in proportion to how well each explains it. Example: your point at ' +
              where(tp.v) +
              ' is ' +
              ex.map((v, k) => (100 * v).toFixed(0) + '% cluster ' + (k + 1)).join(', ') +
              '. ' +
              sh +
              ' of ' +
              P.length +
              ' points are split noticeably. Log-likelihood: ' +
              uNum(ll) +
              (prevL !== null ? ', up from ' + uNum(prevL) : '') +
              '.';
          }
          return;
        }
        if (ph === 'M') {
          cu = 2;
          prevG = G.map((q) => ({ ...q }));
          const n = P.length;
          G.forEach((q, k) => {
            let N = 0,
              sx = 0,
              sy = 0;
            P.forEach((p, i) => {
              const r = Rs[i][k];
              N += r;
              sx += r * p.x;
              sy += r * p.y;
            });
            if (N < 1e-6) return;
            const mx = sx / N,
              my = sy / N;
            let a = 0,
              b = 0,
              d = 0;
            P.forEach((p, i) => {
              const r = Rs[i][k],
                dx = p.x - mx,
                dy = p.y - my;
              a += r * dx * dx;
              b += r * dx * dy;
              d += r * dy * dy;
            });
            a = a / N + 2e-3;
            d = d / N + 2e-3;
            b = b / N;
            if (one()) {
              Object.assign(q, { pi: N / n, mx, my: 0, a, b: 0, d: 1 });
              return;
            }
            if (S.cov === 'diag') b = 0;
            if (S.cov === 'sph') {
              a = d = (a + d) / 2;
              b = 0;
            }
            Object.assign(q, { pi: N / n, mx, my, a, b, d });
          });
          it++;
          mv = RM ? 0 : 18;
          ph = 'E';
          msg =
            'M-step: the shares stayed fixed, and each bell was refitted to the points it claims. Centres moved to ' +
            G.map((q, k) => (one() ? f2(q.mx) : '(' + f2(q.mx) + ', ' + f2(q.my) + ')')).join(
              ', ',
            ) +
            '. Weights became π = ' +
            G.map((q) => f2(q.pi)).join(', ') +
            ': each is that cluster’s share of all the points.';
        }
      };
      const tp = testPoint(
        root,
        'gm',
        ['Your x', 'Your y'],
        [-0.2, 0.7],
        [
          [-3.8, 3.8],
          [-2.85, 2.85],
        ],
      );
      const orb = Orbit(
        c,
        () => {
          if (S.dim === '3') draw();
        },
        vb,
        { az: -0.55, el: 0.62, zoom: 1 },
      );
      const draw = () => {
        const t = T(),
          col = UC(t),
          rgb = col.map(hex),
          bl = (r) => {
            let R = 0,
              Gn = 0,
              B = 0;
            r.forEach((v, k) => {
              R += v * rgb[k % 8][0];
              Gn += v * rgb[k % 8][1];
              B += v * rgb[k % 8][2];
            });
            return [Math.round(R), Math.round(Gn), Math.round(B)];
          },
          css = (q, al = 1) => 'rgba(' + q[0] + ',' + q[1] + ',' + q[2] + ',' + al + ')';
        const u = prevG && mv > 0 ? 1 - mv / 18 : 1,
          GG = G.map((q, k) => {
            const o = prevG && prevG[k] ? prevG[k] : q,
              L = (f) => o[f] + (q[f] - o[f]) * u;
            return { pi: L('pi'), mx: L('mx'), my: L('my'), a: L('a'), b: L('b'), d: L('d') };
          }),
          ty = one() ? 0 : tp.v[1];
        cap.textContent = CAP[S.dim];
        fc.textContent = one()
          ? 'Histogram of the data, the bell of each cluster, and their sum'
          : S.dim === '2'
            ? 'The points and the cluster ellipses'
            : 'The mixture density as a landscape';
        if (one()) {
          const { g, w, h } = fit(c),
            X = (x) => w / 2 + (x * (w - 40)) / 8,
            base = h - 84,
            top = 34,
            nb = 32,
            bw = 8 / nb,
            cnt = new Array(nb).fill(0);
          for (const p of P) cnt[clamp(Math.floor((p.x + 4) / bw), 0, nb - 1)]++;
          const hd = cnt.map((v) => v / (P.length * bw));
          const mixf = (x) => GG.reduce((s, q) => s + q.pi * d1(q, x), 0);
          let ym = Math.max(...hd, 0.05);
          for (let i = 0; i <= 200; i++) ym = Math.max(ym, mixf(-4 + i * 0.04));
          ym *= 1.12;
          const Y = (v) => base - (v / ym) * (base - top);
          g.fillStyle = mix(t.ink, t.ink, 0, 0.11);
          hd.forEach((v, i) => g.fillRect(X(-4 + i * bw) + 1, Y(v), X(bw) - X(0) - 2, base - Y(v)));
          GG.forEach((q, k) => {
            const f = (x) => q.pi * d1(q, x);
            g.beginPath();
            g.moveTo(X(-4), base);
            for (let i = 0; i <= 240; i++) {
              const x = -4 + i / 30;
              g.lineTo(X(x), Y(f(x)));
            }
            g.lineTo(X(4), base);
            g.closePath();
            g.fillStyle = mix(col[k % 8], col[k % 8], 0, 0.2);
            g.fill();
            g.beginPath();
            for (let i = 0; i <= 240; i++) {
              const x = -4 + i / 30;
              i ? g.lineTo(X(x), Y(f(x))) : g.moveTo(X(x), Y(f(x)));
            }
            g.strokeStyle = col[k % 8];
            g.lineWidth = 2.6;
            g.stroke();
            const pk = Y(f(q.mx));
            g.setLineDash([4, 4]);
            g.lineWidth = 1.4;
            g.beginPath();
            g.moveTo(X(q.mx), base);
            g.lineTo(X(q.mx), pk);
            g.stroke();
            g.setLineDash([]);
            g.font = SANS;
            g.fillStyle = col[k % 8];
            g.textAlign = 'center';
            g.fillText('cluster ' + (k + 1), clamp(X(q.mx), 50, w - 50), Math.max(14, pk - 20));
            g.font = MONO;
            g.fillText(
              'centre ' + f2(q.mx) + ', spread ' + f2(Math.sqrt(q.a)),
              clamp(X(q.mx), 80, w - 80),
              Math.max(27, pk - 7),
            );
          });
          g.beginPath();
          for (let i = 0; i <= 240; i++) {
            const x = -4 + i / 30;
            i ? g.lineTo(X(x), Y(mixf(x))) : g.moveTo(X(x), Y(mixf(x)));
          }
          g.strokeStyle = t.ink;
          g.lineWidth = 1.8;
          g.stroke();
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(X(-4), base);
          g.lineTo(X(4), base);
          g.stroke();
          P.forEach((p, i) => {
            g.fillStyle = Rs ? css(bl(Rs[i])) : mix(t.ink, t.ink, 0, 0.4);
            g.beginPath();
            g.arc(X(p.x), base + 8 + (i % 4) * 4.5, 2.5, 0, 7);
            g.fill();
          });
          const by = base + 34,
            bh = 24;
          for (let px = X(-4); px < X(4); px += 3) {
            const r = respG(GG, ((px + 1.5 - w / 2) * 8) / (w - 40), 0).r;
            let y = by;
            r.forEach((v, k) => {
              g.fillStyle = col[k % 8];
              g.fillRect(px, y, 3.2, v * bh + 0.4);
              y += v * bh;
            });
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          [-3, -2, -1, 0, 1, 2, 3].forEach((v) => g.fillText(String(v), X(v), by + bh + 14));
          g.textAlign = 'left';
          g.fillText('band: who owns a point at each position', X(-4), h - 4);
          const x = X(tp.v[0]);
          g.strokeStyle = t.ink;
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(x, top - 8);
          g.lineTo(x, by + bh);
          g.stroke();
          markTP(g, x, top - 8, t);
        } else if (S.dim === '2') {
          const V = uView(c),
            { g, w, h, X, Y } = V;
          uFrame(
            V,
            t,
            S.ds === 'shop' ? 'visits per month (standardised)' : 'x',
            S.ds === 'shop' ? 'average basket (standardised)' : 'y',
          );
          GG.forEach((q, k) => {
            ellPath(g, X, Y, q.mx, q.my, q.a, q.b, q.d, 2);
            g.fillStyle = mix(col[k % 8], col[k % 8], 0, 0.09);
            g.fill();
            g.strokeStyle = mix(col[k % 8], col[k % 8], 0, 0.6);
            g.lineWidth = 1.2;
            g.stroke();
            ellPath(g, X, Y, q.mx, q.my, q.a, q.b, q.d, 1);
            g.strokeStyle = col[k % 8];
            g.lineWidth = 2.4;
            g.stroke();
          });
          P.forEach((p, i) => {
            g.beginPath();
            g.arc(X(p.x), Y(p.y), 3.4, 0, 7);
            g.fillStyle = Rs ? css(bl(Rs[i])) : mix(t.ink, t.ink, 0, 0.42);
            g.fill();
          });
          GG.forEach((q, k) => {
            if (prevG && prevG[k]) {
              g.strokeStyle = t.ink;
              g.lineWidth = 1.4;
              g.setLineDash([4, 4]);
              g.beginPath();
              g.moveTo(X(prevG[k].mx), Y(prevG[k].my));
              g.lineTo(X(q.mx), Y(q.my));
              g.stroke();
              g.setLineDash([]);
            }
            g.fillStyle = col[k % 8];
            g.strokeStyle = t.bg;
            g.lineWidth = 4;
            g.beginPath();
            g.rect(X(q.mx) - 6, Y(q.my) - 6, 12, 12);
            g.stroke();
            g.fill();
            g.strokeStyle = t.ink;
            g.lineWidth = 1.4;
            g.strokeRect(X(q.mx) - 6, Y(q.my) - 6, 12, 12);
            g.font = SANS;
            g.fillStyle = t.ink;
            g.textAlign = 'left';
            g.fillText(String(k + 1), X(q.mx) + 10, Y(q.my) - 8);
          });
          markTP(g, X(tp.v[0]), Y(tp.v[1]), t);
        } else {
          const { g, w, h } = fit(c),
            pr = orb.cam(w, h, 9.6),
            nx = 44,
            ny = 33,
            Z = [],
            Cc = [];
          let zm = 1e-9;
          for (let i = 0; i <= nx; i++) {
            Z.push([]);
            for (let j = 0; j <= ny; j++) {
              const q = respG(GG, -4 + (8 * i) / nx, -3 + (6 * j) / ny);
              Z[i].push(q.s);
              zm = Math.max(zm, q.s);
            }
          }
          const zs = 2.7 / zm;
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.beginPath();
          [
            [-4, -3],
            [4, -3],
            [4, 3],
            [-4, 3],
            [-4, -3],
          ].forEach((q, i) => {
            const p = pr(q[0], q[1], 0);
            i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]);
          });
          g.stroke();
          P.forEach((p, i) => {
            const q = pr(p.x, p.y, 0);
            g.fillStyle = Rs ? css(bl(Rs[i]), 0.9) : mix(t.ink, t.ink, 0, 0.42);
            g.beginPath();
            g.arc(q[0], q[1], 2.2, 0, 7);
            g.fill();
          });
          const Q = [];
          for (let i = 0; i < nx; i++)
            for (let j = 0; j < ny; j++) {
              const z = (Z[i][j] + Z[i + 1][j] + Z[i][j + 1] + Z[i + 1][j + 1]) / 4;
              if (z < zm * 0.03) continue;
              const x0 = -4 + (8 * i) / nx,
                x1 = -4 + (8 * (i + 1)) / nx,
                y0 = -3 + (6 * j) / ny,
                y1 = -3 + (6 * (j + 1)) / ny,
                a = pr(x0, y0, zs * Z[i][j]),
                b = pr(x1, y0, zs * Z[i + 1][j]),
                cc = pr(x1, y1, zs * Z[i + 1][j + 1]),
                d = pr(x0, y1, zs * Z[i][j + 1]);
              Q.push({
                a,
                b,
                c: cc,
                d,
                z,
                dp: (a[2] + b[2] + cc[2] + d[2]) / 4,
                r: respG(GG, (x0 + x1) / 2, (y0 + y1) / 2).r,
              });
            }
          Q.sort((p, q) => q.dp - p.dp);
          g.lineWidth = 0.5;
          for (const o of Q) {
            g.beginPath();
            g.moveTo(o.a[0], o.a[1]);
            g.lineTo(o.b[0], o.b[1]);
            g.lineTo(o.c[0], o.c[1]);
            g.lineTo(o.d[0], o.d[1]);
            g.closePath();
            g.fillStyle = css(bl(o.r), 0.35 + 0.6 * Math.pow(o.z / zm, 0.6));
            g.fill();
            g.strokeStyle = mix(t.ink, t.ink, 0, 0.13);
            g.stroke();
          }
          GG.forEach((q, k) => {
            const p = pr(q.mx, q.my, zs * respG(GG, q.mx, q.my).s);
            g.font = SANS;
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.fillText('cluster ' + (k + 1), p[0], p[1] - 8);
          });
          const a = pr(tp.v[0], tp.v[1], 0),
            b = pr(tp.v[0], tp.v[1], zs * respG(GG, tp.v[0], tp.v[1]).s);
          g.strokeStyle = t.ink;
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(a[0], a[1]);
          g.lineTo(b[0], b[1]);
          g.stroke();
          markTP(g, b[0], b[1], t);
        }
        {
          const { g, w, h } = fit(cr),
            q = respG(GG, tp.v[0], ty),
            ha = h * 0.24,
            hb = h * 0.3;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('your point at ' + where(tp.v) + ' is shared like this', 8, 13);
          let x = 10;
          const bwid = w - 20;
          q.r.forEach((v, k) => {
            g.fillStyle = col[k % 8];
            g.fillRect(x, 22, v * bwid, 30);
            if (v * bwid > 46) {
              g.fillStyle = t.bg;
              g.font = SANS;
              g.textAlign = 'center';
              g.fillText((100 * v).toFixed(0) + '%', x + (v * bwid) / 2, 42);
            }
            x += v * bwid;
          });
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.strokeRect(10, 22, bwid, 30);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            'weight π of each cluster = its share of all ' + P.length + ' points',
            8,
            ha + 13,
          );
          const rh = Math.min(24, (hb - 22) / Math.max(1, GG.length));
          GG.forEach((z, k) => {
            const y = ha + 20 + k * rh;
            g.fillStyle = t.ink;
            g.textAlign = 'left';
            g.fillText('cluster ' + (k + 1), 10, y + rh * 0.65);
            g.fillStyle = mix(t.ink, t.ink, 0, 0.08);
            g.fillRect(84, y + 2, w - 190, rh - 5);
            g.fillStyle = col[k % 8];
            g.fillRect(84, y + 2, (w - 190) * z.pi, rh - 5);
            g.fillStyle = t.muted;
            g.fillText(
              'π = ' + f2(z.pi) + ', ≈ ' + Math.round(z.pi * P.length) + ' pts',
              w - 100,
              y + rh * 0.65,
            );
          });
          const lo = LL.length ? Math.min(...LL) : 0,
            hi = LL.length ? Math.max(...LL) : 1,
            pad = Math.max(1e-6, (hi - lo) * 0.08);
          uLine(g, t, 0, ha + hb, w, h - ha - hb, LL, {
            title: 'log-likelihood after each E-step: it can only rise',
            dots: true,
            n: Math.max(8, LL.length),
            y1: lo - pad,
            ym: hi + pad,
            col: t.gold,
          });
        }
        if (!mv)
          tab.innerHTML =
            '<table><thead><tr><th>cluster</th><th>weight π</th><th>about how many points</th><th>centre</th>' +
            (one()
              ? '<th>spread σ</th>'
              : '<th>spread along x</th><th>spread along y</th><th>tilt (correlation)</th>') +
            '</tr></thead><tbody>' +
            G.map(
              (q, k) =>
                '<tr><td><i style="background:' +
                col[k % 8] +
                '"></i>' +
                (k + 1) +
                '</td><td>' +
                f2(q.pi) +
                '</td><td>' +
                (q.pi * P.length).toFixed(1) +
                '</td><td>' +
                (one() ? f2(q.mx) : '(' + f2(q.mx) + ', ' + f2(q.my) + ')') +
                '</td>' +
                (one()
                  ? '<td>' + f2(Math.sqrt(q.a)) + '</td>'
                  : '<td>' +
                    f2(Math.sqrt(q.a)) +
                    '</td><td>' +
                    f2(Math.sqrt(q.d)) +
                    '</td><td>' +
                    f2(q.b / Math.sqrt(q.a * q.d)) +
                    '</td>') +
                '</tr>',
            ).join('') +
            '</tbody></table>';
        live.textContent = msg;
        uCur(root, 'gm', cu);
        {
          const q = resp(tp.v[0], ty),
            b = q.r.indexOf(Math.max(...q.r));
          tp.out.textContent =
            'Your point at ' +
            where(tp.v) +
            ' is shared as ' +
            q.r.map((v, k) => (100 * v).toFixed(0) + '% cluster ' + (k + 1)).join(', ') +
            '. A hard answer would be cluster ' +
            (b + 1) +
            '. The mixture density there is ' +
            f3(q.s) +
            '.';
        }
        const L = LL.length;
        setR(root, 'it', String(it));
        setR(root, 'll', L ? uNum(LL[L - 1]) : '–');
        setR(root, 'dl', L > 1 ? (LL[L - 1] - LL[L - 2]).toFixed(4) : '–');
        setR(
          root,
          'sh',
          Rs ? Rs.filter((r) => Math.max(...r) < 0.9).length + ' of ' + P.length : '–',
        );
        setR(root, 'st', ph === 'done' ? 'Converged' : it ? 'Running' : 'Not started');
      };
      tp.on = () => draw();
      const load = () => {
          P = gen();
          if (one())
            P.forEach((p) => {
              p.y = 0;
            });
        },
        full = () => {
          for (let i = 0; i < 700 && ph !== 'done'; i++) step();
          mv = 0;
        };
      const gen = uDataBind(
        root,
        'gm',
        S,
        () => {
          pb.stop();
          load();
          run = 1;
          reset();
          full();
          draw();
        },
        mk,
      );
      load();
      reset();
      full();
      const anim = Anim(() => {
        fr++;
        if (mv > 0) {
          mv--;
          draw();
          return;
        }
        if (fr % (it < 4 ? 44 : 14)) return;
        step();
        draw();
        chk();
      });
      pb = playBtn($('#gm-play', root), anim);
      chk = runCtl(pb, root, 'gm', {
        done: () => ph === 'done',
        restart: () => {
          run++;
          reset();
        },
        count: () => it,
        what: 'rounds',
      });
      $('#gm-step', root).onclick = () => {
        pb.stop();
        mv = 0;
        step();
        draw();
      };
      $('#gm-reset', root).onclick = () => {
        pb.stop();
        run++;
        reset();
        draw();
      };
      const kb = bind(
        root,
        'gm-K',
        (v) => {
          S.K = v;
          pb.stop();
          reset();
          full();
          draw();
        },
        (v) => String(v),
      );
      $('#gm-cov', root).onchange = (e) => {
        S.cov = e.target.value;
        pb.stop();
        reset();
        full();
        draw();
      };
      const lay = () => {
        $('#gm-covl', root).style.display = one() ? 'none' : '';
        $('#gm-tp1', root).closest('label').style.display = one() ? 'none' : '';
        vb.style.display = S.dim === '3' ? '' : 'none';
        c.style.cursor = S.dim === '3' ? 'grab' : '';
        c.style.touchAction = S.dim === '3' ? 'none' : '';
      };
      segBind(root, 'gm-dim', S.dim, (v) => {
        const was = one();
        S.dim = v;
        pb.stop();
        orb.spin(false);
        if (was !== one()) {
          const ks = one() ? ['h2', 'h3', 'hov'] : ['long', 'blobs', 'uneven', 'shop', 'moons'];
          dsel.innerHTML = ks
            .map((k) => '<option value="' + k + '">' + UKIND[k] + '</option>')
            .join('');
          S.ds = ks[0];
          S.K = one() ? 2 : 3;
          kb.set(S.K);
          dsel.dispatchEvent(new Event('change'));
        }
        lay();
        draw();
      });
      lay();
      return {
        draw,
        stop: () => {
          pb.stop();
          orb.spin(false);
        },
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'hierarchical',
    tab: 'Hierarchical clustering',
    title: 'Build the family tree of the data',
    lede: 'There is no need to choose the number of clusters in advance. Start with every point alone, join the two closest clusters, and repeat until one remains. The record of who joined whom, and at what distance, is a tree called a dendrogram. Cutting it at any height gives a clustering.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The points. Lines join each point to the centre of its cluster. The thick line is the latest merge</figcaption><canvas class="short" id="hc-c" role="img" aria-label="Points joined into clusters"></canvas></figure>
 <figure class="fig"><figcaption>The dendrogram: each join is drawn at the distance where it happened</figcaption><canvas class="short" id="hc-r" role="img" aria-label="Dendrogram"></canvas></figure></div>
 ${uHow('hc', ['Start with every point as its own cluster.', 'Measure the distance between every pair of clusters, using the linkage rule.', 'Merge the closest pair, and draw the join on the tree at that distance.', 'Repeat steps 2 and 3 until a single cluster is left.', 'Cut the tree at any height: the branches below the cut are the clusters.'])}
 <div class="controls">${uBtns('hc', 'Merge', 'One merge')}
 ${grp('The model', `<label class="ctl" for="hc-link"><span>Distance between two clusters (linkage)</span><select id="hc-link"><option value="single">Single: their two closest points</option><option value="complete">Complete: their two farthest points</option><option value="average" selected>Average: mean over all pairs</option><option value="ward">Ward: growth in within-cluster spread</option></select></label>` + slider('hc-K', 'Cut the finished tree into this many clusters', 1, 10, 1, 3) + `<div class="row"><button class="btn" id="hc-fin" type="button">Finish the Tree</button></div>`)}
 ${uDataHTML('hc', ['blobs', 'moons', 'long', 'uneven', 'shop', 'none'], 'blobs', 8, 70, 36)}</div>
 <dl class="read">${rd('cl', 'Clusters right now', 'hl')}${rd('mg', 'Merges done')}${rd('ht', 'Distance of the latest merge')}${rd('gap', 'Largest jump in merge distance suggests', 'hl2')}</dl>`,
    tryit:
      'Step through the first ten merges on Three Round Groups: the early joins are tiny, and the last two are huge. That jump is how the tree tells you there are three groups. Then choose Two Moons with 40 or more points, finish the tree and cut it into 2, first with Single linkage and then with Complete. Only one of them follows the moons. Which, and why?',
    math: String.raw`<h4>Intuition</h4><p>Biologists do not sort animals into one fixed set of groups. They build a tree: house cats and lions join early, then cats with dogs, then all mammals, then mammals with birds. The tree holds every level of grouping at once, and you choose the level that suits your question. Hierarchical clustering builds the same kind of tree for any data, using nothing but the distances between points.</p>
 <h4>The agglomerative algorithm</h4><p>Begin with \(n\) clusters, each holding one point. At every step find the two clusters \(A\) and \(B\) with the smallest distance \(d(A,B)\), merge them, and record the merge at height \(d(A,B)\). After \(n-1\) merges everything is in one cluster. Unlike k-means there is no random start: the same data and the same linkage always give the same tree.</p>
 <h4>Linkage: the distance between two clusters</h4><p>Distance between two points is clear. Distance between two groups of points needs a rule, and the choice changes the result:</p>
 $$d_{\text{single}}(A,B)=\min_{a\in A,\,b\in B}\lVert a-b\rVert,\qquad d_{\text{complete}}(A,B)=\max_{a\in A,\,b\in B}\lVert a-b\rVert,$$$$d_{\text{average}}(A,B)=\frac{1}{|A|\,|B|}\sum_{a\in A}\sum_{b\in B}\lVert a-b\rVert,\qquad d_{\text{Ward}}(A,B)=\frac{|A|\,|B|}{|A|+|B|}\,\lVert\bar a-\bar b\rVert^2 .$$
 <p><b>Single</b> linkage joins clusters that touch anywhere, so it can follow long curved chains, and for the same reason one stray point between two groups can bridge them. <b>Complete</b> linkage keeps every cluster compact and tends to break large groups. <b>Average</b> sits between the two. <b>Ward</b> merges the pair whose union adds least to the within-cluster sum of squares, the same quantity k-means minimises, and so gives round, even clusters.</p>
 <h4>Reading the dendrogram</h4><p>Leaves along the bottom are the points. Each ∩-shaped join is one merge, and its height is the distance at which it happened. Short joins near the bottom connect near-identical points. A tall stretch of the tree with no joins means the clusters below it are far apart compared with their own size. Cutting there, with a horizontal line, gives a natural number of clusters: the number of vertical lines the cut crosses. The order of leaves left to right is partly arbitrary, since any join can be flipped, so only the heights carry meaning.</p>
 <h4>Cost</h4><p>The method needs all \(\tfrac12 n(n-1)\) pairwise distances, so memory grows as \(n^2\) and a plain implementation takes about \(n^3\) time. It suits hundreds or a few thousand points, not millions. A merge can never be undone: an early mistake stays in the tree.</p>
 <h4>When to use it</h4><p>When the number of clusters is unknown, when you want groups at several levels at once, or when you only have distances and no coordinates (for example edit distances between gene sequences). The dendrogram is itself a useful picture of the data.</p>`,
    q: 'On the two moons, single linkage recovers the two moons but average linkage does not. Explain in one sentence each.',
    a: 'Single linkage measures the gap between the nearest members of two clusters, and along each moon neighbouring points are closer to each other than to anything in the other moon, so each moon is chained together before the two ever join. Average linkage looks at all pairs of points, and the two halves of one moon are far apart on average, farther than half of one moon is from the nearby half of the other, so it merges pieces across moons.',
    init(root) {
      const c = $('#hc-c', root),
        cr = $('#hc-r', root),
        live = $('#hc-live', root),
        S = { link: 'average', K: 3, ds: 'blobs', n: 36, seed: 4, cap: 70 };
      let P = [],
        M = [],
        mem = {},
        pos = [],
        xp = {},
        hh = {},
        shown = 0,
        cut = false,
        msg = '',
        cu = 0,
        wait = 0,
        fr = 0,
        pb;
      const build = () => {
        const n = P.length,
          W = S.link === 'ward',
          N2 = 2 * n,
          D = [];
        for (let i = 0; i < N2; i++) D.push(new Float64Array(N2));
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++) {
            const e = d2(P[i], P[j]);
            D[i][j] = W ? e : Math.sqrt(e);
          }
        let act = [];
        mem = {};
        const sz = {};
        for (let i = 0; i < n; i++) {
          act.push(i);
          mem[i] = [i];
          sz[i] = 1;
        }
        M = [];
        for (let m = 0; m < n - 1; m++) {
          let bi = -1,
            bj = -1,
            bd = 1e18;
          for (let a = 0; a < act.length; a++)
            for (let b = a + 1; b < act.length; b++) {
              const d = D[act[a]][act[b]];
              if (d < bd) {
                bd = d;
                bi = act[a];
                bj = act[b];
              }
            }
          const id = n + m,
            si = sz[bi],
            sj = sz[bj];
          for (const k of act) {
            if (k === bi || k === bj) continue;
            const a = D[k][bi],
              b = D[k][bj],
              sk = sz[k];
            let v;
            if (S.link === 'single') v = Math.min(a, b);
            else if (S.link === 'complete') v = Math.max(a, b);
            else if (S.link === 'average') v = (si * a + sj * b) / (si + sj);
            else v = ((si + sk) * a + (sj + sk) * b - sk * bd) / (si + sj + sk);
            D[k][id] = D[id][k] = v;
          }
          sz[id] = si + sj;
          mem[id] = mem[bi].concat(mem[bj]);
          act = act.filter((k) => k !== bi && k !== bj);
          act.push(id);
          M.push({ a: bi, b: bj, id, h: W ? Math.sqrt(Math.max(0, bd)) : bd, na: si, nb: sj });
        }
        const ord = (id) => {
          if (id < n) return [id];
          const q = M[id - n];
          return ord(q.a).concat(ord(q.b));
        };
        const o = n > 1 ? ord(2 * n - 2) : [0];
        pos = new Array(n);
        o.forEach((l, i) => {
          pos[l] = i;
        });
        xp = {};
        hh = {};
        for (let i = 0; i < n; i++) {
          xp[i] = pos[i];
          hh[i] = 0;
        }
        M.forEach((q) => {
          xp[q.id] = (xp[q.a] + xp[q.b]) / 2;
          hh[q.id] = q.h;
        });
      };
      const labels = (m) => {
        const n = P.length,
          L = new Array(n);
        for (let i = 0; i < n; i++) L[i] = i;
        for (let k = 0; k < m; k++) for (const l of mem[M[k].id]) L[l] = M[k].id;
        return L;
      };
      const colOf = (id) => {
        let m = 1e9;
        for (const l of mem[id]) m = Math.min(m, pos[l]);
        return m % 8;
      };
      const reset = () => {
        shown = 0;
        cut = false;
        wait = 0;
        cu = 0;
        msg =
          'Every one of the ' + P.length + ' points is its own cluster. Press Merge or One merge.';
      };
      const names = {
        single: 'single linkage (their two closest points)',
        complete: 'complete linkage (their two farthest points)',
        average: 'average linkage (the mean over all pairs)',
        ward: 'Ward linkage (the growth in spread, shown as a distance)',
      };
      const step = () => {
        if (cut || shown >= M.length) {
          if (shown >= M.length && !cut) {
            cut = true;
            cu = 4;
            msg = cutMsg();
            return;
          }
          reset();
          return;
        }
        const q = M[shown];
        shown++;
        cu = 2;
        msg =
          'Merge ' +
          shown +
          ' of ' +
          M.length +
          ': joined a cluster of ' +
          q.na +
          ' point' +
          (q.na === 1 ? '' : 's') +
          ' with a cluster of ' +
          q.nb +
          '. Their distance under ' +
          names[S.link] +
          ' was ' +
          f2(q.h) +
          ', the smallest of all pairs. ' +
          (P.length - shown) +
          ' cluster' +
          (P.length - shown === 1 ? ' remains.' : 's remain.');
        if (shown >= M.length) {
          cu = 3;
        }
      };
      const lvl = () => (cut ? Math.max(0, P.length - Math.min(S.K, P.length)) : shown);
      const cutMsg = () => {
        const k = Math.min(S.K, P.length),
          m = P.length - k,
          lo = m > 0 ? M[m - 1].h : 0,
          hi = m < M.length ? M[m].h : lo;
        return (
          'The tree is finished. A cut anywhere between height ' +
          f2(lo) +
          ' and ' +
          f2(hi) +
          ' crosses ' +
          k +
          ' vertical line' +
          (k === 1 ? '' : 's') +
          ', so it gives ' +
          k +
          ' cluster' +
          (k === 1 ? '' : 's') +
          '. ' +
          (hi - lo > 0.3 * hi
            ? 'That is a wide gap: these clusters are well separated.'
            : 'That gap is narrow: this number of clusters is not strongly supported by the tree.')
        );
      };
      const sugg = () => {
        let b = 2,
          bg = -1;
        for (let k = 2; k <= Math.min(9, P.length - 1); k++) {
          const m = P.length - k,
            g = M[m].h - M[m - 1].h;
          if (g > bg) {
            bg = g;
            b = k;
          }
        }
        return b;
      };
      const draw = () => {
        const t = T(),
          col = UC(t),
          n = P.length,
          L = labels(lvl());
        {
          const V = uView(c),
            { g, w, h, X, Y } = V;
          uFrame(V, t, 'x', 'y');
          const cen = {};
          for (let i = 0; i < n; i++) {
            const k = L[i];
            if (k < n) continue;
            const q = cen[k] || (cen[k] = [0, 0, 0]);
            q[0] += P[i].x;
            q[1] += P[i].y;
            q[2]++;
          }
          for (let i = 0; i < n; i++) {
            const k = L[i];
            if (k < n) continue;
            const q = cen[k];
            g.strokeStyle = mix(col[colOf(k)], col[colOf(k)], 0, 0.45);
            g.lineWidth = 1.2;
            g.beginPath();
            g.moveTo(X(P[i].x), Y(P[i].y));
            g.lineTo(X(q[0] / q[2]), Y(q[1] / q[2]));
            g.stroke();
          }
          if (!cut && shown > 0) {
            const q = M[shown - 1],
              A = mem[q.a],
              B = mem[q.b];
            let pa, pb2;
            if (S.link === 'single' || S.link === 'complete') {
              let bd = S.link === 'single' ? 1e18 : -1;
              for (const i of A)
                for (const j of B) {
                  const d = d2(P[i], P[j]);
                  if (S.link === 'single' ? d < bd : d > bd) {
                    bd = d;
                    pa = P[i];
                    pb2 = P[j];
                  }
                }
            } else {
              const cn = (Q) => ({
                x: Q.reduce((s, i) => s + P[i].x, 0) / Q.length,
                y: Q.reduce((s, i) => s + P[i].y, 0) / Q.length,
              });
              pa = cn(A);
              pb2 = cn(B);
            }
            g.strokeStyle = t.ink;
            g.lineWidth = 3.2;
            g.beginPath();
            g.moveTo(X(pa.x), Y(pa.y));
            g.lineTo(X(pb2.x), Y(pb2.y));
            g.stroke();
          }
          for (let i = 0; i < n; i++) {
            const k = L[i];
            g.beginPath();
            g.arc(X(P[i].x), Y(P[i].y), 4.4, 0, 7);
            g.fillStyle = k < n ? mix(t.ink, t.ink, 0, 0.42) : col[colOf(k)];
            g.fill();
            g.strokeStyle = t.bg;
            g.lineWidth = 1.2;
            g.stroke();
          }
        }
        {
          const { g, w, h } = fit(cr),
            m = { l: 44, r: 14, t: 16, b: 22 },
            hm = Math.max(1e-9, M.length ? M[M.length - 1].h : 1) * 1.06,
            X = (v) => m.l + (n > 1 ? v / (n - 1) : 0.5) * (w - m.l - m.r),
            Y = (v) => h - m.b - (v / hm) * (h - m.t - m.b),
            lv = lvl();
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l - 8, m.t);
          g.lineTo(m.l - 8, h - m.b);
          g.stroke();
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText(f2(hm / 1.06), m.l - 12, Y(hm / 1.06) + 4);
          g.fillText('0', m.l - 12, Y(0) + 4);
          g.save();
          g.translate(11, h / 2);
          g.rotate(-Math.PI / 2);
          g.textAlign = 'center';
          g.fillText('merge distance', 0, 0);
          g.restore();
          for (let k = 0; k < shown; k++) {
            const q = M[k];
            g.strokeStyle = k < lv ? col[colOf(L[mem[q.id][0]])] : t.ink;
            g.lineWidth = k === shown - 1 && !cut ? 3 : 1.7;
            g.beginPath();
            g.moveTo(X(xp[q.a]), Y(hh[q.a]));
            g.lineTo(X(xp[q.a]), Y(q.h));
            g.lineTo(X(xp[q.b]), Y(q.h));
            g.lineTo(X(xp[q.b]), Y(hh[q.b]));
            g.stroke();
          }
          for (let i = 0; i < n; i++) {
            g.fillStyle = L[i] < n ? mix(t.ink, t.ink, 0, 0.42) : col[colOf(L[i])];
            g.beginPath();
            g.arc(X(pos[i]), h - m.b + 7, 3, 0, 7);
            g.fill();
          }
          if (cut && M.length) {
            const k = Math.min(S.K, n),
              mm = n - k,
              lo = mm > 0 ? M[mm - 1].h : 0,
              hi = mm < M.length ? M[mm].h : lo * 1.05,
              y = Y((lo + hi) / 2);
            g.setLineDash([7, 5]);
            g.strokeStyle = t.gold;
            g.lineWidth = 2;
            g.beginPath();
            g.moveTo(m.l - 8, y);
            g.lineTo(w - m.r, y);
            g.stroke();
            g.setLineDash([]);
            g.fillStyle = t.cut;
            g.textAlign = 'right';
            g.fillText('cut here: ' + k + ' cluster' + (k === 1 ? '' : 's'), w - m.r, y - 6);
          }
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('one dot per point', w / 2, h - 2);
        }
        live.textContent = msg;
        uCur(root, 'hc', cu);
        setR(root, 'cl', String(n - lvl()));
        setR(root, 'mg', shown + ' of ' + M.length);
        setR(root, 'ht', shown ? f2(M[shown - 1].h) : '–');
        setR(root, 'gap', M.length > 2 ? sugg() + ' clusters' : '–');
      };
      const fin = () => {
        shown = M.length;
        cut = true;
        cu = 4;
        msg = cutMsg();
      };
      const gen = uDataBind(
        root,
        'hc',
        S,
        () => {
          P = gen();
          build();
          reset();
          fin();
          draw();
        },
        () => {
          if (S.ds !== 'moons') return uData(S.ds, S.n, S.seed);
          const r = rng(S.seed * 13 + 5),
            rn = randnFrom(r),
            o = [],
            hf = Math.ceil(S.n / 2);
          for (let i = 0; i < S.n; i++) {
            const a = ((Math.floor(i / 2) + 0.5 + 0.3 * (r() - 0.5)) / hf) * Math.PI;
            o.push(
              i % 2
                ? {
                    x: 0.9 - 1.9 * Math.cos(a) + 0.05 * rn(),
                    y: 0.55 - 1.7 * Math.sin(a) + 0.05 * rn(),
                  }
                : {
                    x: -0.9 + 1.9 * Math.cos(a) + 0.05 * rn(),
                    y: -0.55 + 1.7 * Math.sin(a) + 0.05 * rn(),
                  },
            );
          }
          return o;
        },
      );
      P = gen();
      build();
      fin();
      const anim = Anim(() => {
        fr++;
        if (wait > 0) {
          wait--;
          if (!wait) {
            reset();
            draw();
          }
          return;
        }
        if (fr % (shown < 6 ? 30 : shown > M.length - 6 ? 26 : 7)) return;
        step();
        draw();
        chk();
      });
      pb = playBtn($('#hc-play', root), anim);
      const chk = runCtl(pb, root, 'hc', { done: () => cut, restart: () => reset() });
      $('#hc-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#hc-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      $('#hc-fin', root).onclick = () => {
        pb.stop();
        fin();
        draw();
      };
      $('#hc-link', root).onchange = (e) => {
        S.link = e.target.value;
        pb.stop();
        build();
        fin();
        draw();
      };
      bind(
        root,
        'hc-K',
        (v) => {
          S.K = v;
          pb.stop();
          fin();
          draw();
        },
        (v) => String(v),
      );
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'dbscan',
    tab: 'DBSCAN',
    title: 'Clusters are crowded places',
    lede: 'DBSCAN has no centres and needs no K. It says a cluster is a region where points are packed densely, and it grows each cluster outward from point to point for as long as the crowd continues. Points in thin air are labelled noise. It can follow any shape.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Filled dots are core points, rings are border points, crosses are noise. The circle has radius ε around the point being examined</figcaption><canvas class="short" id="db-c" role="img" aria-label="Points labelled as core, border or noise"></canvas></figure>
 <figure class="fig"><figcaption>Choosing ε: every point’s distance to its minPts-th nearest neighbour, sorted</figcaption><canvas class="short" id="db-r" role="img" aria-label="Sorted k-distance plot with the chosen epsilon"></canvas></figure></div>
 ${uHow('db', ['Pick a point that has not been visited and count the points within distance ε of it, itself included.', 'Fewer than minPts: mark it as noise for now and go back to step 1.', 'At least minPts: it is a core point. Start a new cluster and put its neighbours in a queue.', 'Take the next point from the queue and add it to the cluster. If it is a core point too, its neighbours join the queue. If not, it is a border point and the cluster does not spread through it.', 'When the queue is empty the cluster is complete. Go back to step 1 until every point has been visited.'])}
 <div class="controls">${uBtns('db', 'Run', 'One point')}
 ${grp('The model', slider('db-eps', 'Radius ε', 0.1, 1.3, 0.01, 0.3) + slider('db-m', 'minPts: neighbours needed to be a core point', 2, 14, 1, 5) + `<div class="row"><button class="btn" id="db-fin" type="button">Run to the End</button></div>`)}
 ${uDataHTML('db', ['moons', 'rings', 'dens', 'blobs', 'long', 'none'], 'moons', 40, 500, 220, slider('db-bg', 'Stray background points', 0, 60, 1, 14))}</div>
 <dl class="read">${rd('v', 'Points visited')}${rd('cl', 'Clusters found', 'hl')}${rd('co', 'Core points', 'hl3')}${rd('bo', 'Border points')}${rd('no', 'Noise points', 'hl2')}</dl>`,
    tryit:
      'Step through the first cluster on Two Moons and watch the queue (gold rings) run along the moon without ever jumping the gap. Then drag ε slowly up: at what value do the two moons fuse into one cluster, and what does the right-hand plot show at that value? On Dense and Sparse Groups, try to find one ε that captures all three groups. You will not find one. That is the main weakness of DBSCAN.',
    math: String.raw`<h4>Intuition</h4><p>Look at a satellite photo of a country at night. Cities are where the lights are packed together. You do not need to know how many cities there are, or what shape they have: you follow the lights, and a city ends where the lights thin out. A lone farmhouse is not a city. DBSCAN (density-based spatial clustering of applications with noise) turns that into an algorithm with two numbers: how far to look, and how many lights count as "packed".</p>
 <h4>The definitions</h4><p>Fix a radius \(\varepsilon\) and a count \(\text{minPts}\). The \(\varepsilon\)-neighbourhood of a point is</p>$$N_\varepsilon(p)=\{q:\lVert p-q\rVert\le\varepsilon\},$$<p>which includes \(p\) itself. Every point is then one of three kinds:</p><ul><li><b>Core point:</b> \(|N_\varepsilon(p)|\ge\text{minPts}\). It sits inside a crowd.</li><li><b>Border point:</b> not core, but inside the neighbourhood of some core point. It sits on the edge of a crowd.</li><li><b>Noise:</b> neither. It belongs to no cluster.</li></ul>
 <p>A point \(q\) is <b>density-reachable</b> from a core point \(p\) if there is a chain \(p=p_1,p_2,\dots,p_m=q\) in which each point lies within \(\varepsilon\) of the one before and all of \(p_1,\dots,p_{m-1}\) are core points. A cluster is a core point together with everything density-reachable from it. The chain is what lets a cluster bend: each link is short, yet the two ends can be far apart.</p>
 <h4>The algorithm</h4><p>The step list above is the whole algorithm. Each point's neighbourhood is computed once. With a spatial index that costs about \(n\log n\) in total; without one, \(n^2\). A border point within reach of two clusters goes to whichever reaches it first, which is the only way the result can depend on the order of the data.</p>
 <h4>Choosing the two numbers</h4><p><b>minPts</b> is usually set from the dimension \(d\) of the data: at least \(d+1\), and \(2d\) is a common default, larger for noisy data. Then for every point compute the distance to its minPts-th nearest neighbour and sort those distances, as in the right-hand picture. Points inside clusters have small values and stray points have large ones. A good <b>ε</b> sits at the bend between the long flat part and the steep rise. Points whose value lies above ε cannot be core points.</p>
 <h4>Strengths and limits</h4><p>DBSCAN finds clusters of any shape, decides the number of clusters itself, and labels outliers instead of forcing them into a cluster, which also makes it a simple anomaly detector. Its limit is the single global ε: if one cluster is much denser than another, an ε small enough to separate the dense ones marks the sparse one as noise, and an ε large enough for the sparse one merges the dense ones. HDBSCAN and OPTICS address this by considering all values of ε at once. Distances also become less informative in very high dimensions, where every point is roughly as far from every other.</p>`,
    q: 'A point has only 3 neighbours within ε and minPts is 5, yet DBSCAN puts it in a cluster. How can that be?',
    a: 'It is a border point. It is not dense enough to be a core point itself, but it lies within ε of some core point, so it is density-reachable from that core point and joins its cluster. What it cannot do is extend the cluster: its own neighbours are not added to the queue. This is why a thin bridge of sparse points does not merge two clusters.',
    init(root) {
      const c = $('#db-c', root),
        cr = $('#db-r', root),
        live = $('#db-live', root),
        S = { eps: 0.3, m: 5, ds: 'moons', n: 220, seed: 2, bg: 14 };
      let P = [],
        lab = [],
        typ = [],
        inQ = [],
        queue = [],
        order = [],
        scan = 0,
        nc = 0,
        curC = -1,
        cur = -1,
        nb = [],
        done = false,
        vis = 0,
        msg = '',
        cu = 0,
        wait = 0,
        fr = 0,
        pb,
        kd = [],
        run = 1;
      const nbr = (x, y) => {
        const e2 = S.eps * S.eps,
          o = [];
        for (let j = 0; j < P.length; j++)
          if ((P[j].x - x) ** 2 + (P[j].y - y) ** 2 <= e2) o.push(j);
        return o;
      };
      const kdist = () => {
        const k = Math.min(S.m, P.length) - 1;
        kd = P.map((p) => {
          const d = P.map((q) => d2(p, q)).sort((a, b) => a - b);
          return Math.sqrt(d[Math.max(0, k)]);
        }).sort((a, b) => b - a);
      };
      const reset = () => {
        const n = P.length,
          r = rng(60 + run * 3);
        lab = new Array(n).fill(-2);
        typ = new Array(n).fill(0);
        inQ = new Array(n).fill(0);
        queue = [];
        order = P.map((_, i) => i);
        for (let i = n - 1; i > 0; i--) {
          const j = Math.floor(r() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
        scan = 0;
        nc = 0;
        curC = -1;
        cur = -1;
        nb = [];
        done = false;
        vis = 0;
        cu = -1;
        wait = 0;
        msg = 'No point has been visited yet. Press Run or One point.';
      };
      const step = () => {
        if (done) {
          run++;
          reset();
          return;
        }
        while (queue.length) {
          const q = queue.shift();
          if (lab[q] >= 0) continue;
          cur = q;
          nb = nbr(P[q].x, P[q].y);
          cu = 3;
          if (lab[q] === -1) {
            lab[q] = curC;
            typ[q] = 2;
            msg =
              'This point was marked as noise earlier, but it lies within ε of a core point of cluster ' +
              (curC + 1) +
              '. It becomes a border point of that cluster.' +
              (queue.length
                ? ''
                : ' The queue is now empty: cluster ' + (curC + 1) + ' is complete.');
            if (!queue.length) cu = 4;
            return;
          }
          lab[q] = curC;
          vis++;
          if (nb.length >= S.m) {
            typ[q] = 1;
            let add = 0;
            for (const j of nb)
              if ((lab[j] === -2 || lab[j] === -1) && !inQ[j]) {
                inQ[j] = 1;
                queue.push(j);
                add++;
              }
            msg =
              'Growing cluster ' +
              (curC + 1) +
              ': this point has ' +
              nb.length +
              ' points within ε, at least minPts = ' +
              S.m +
              ', so it is a core point. ' +
              add +
              ' new neighbour' +
              (add === 1 ? ' joins' : 's join') +
              ' the queue, which now holds ' +
              queue.length +
              '.';
          } else {
            typ[q] = 2;
            msg =
              'Growing cluster ' +
              (curC + 1) +
              ': this point has only ' +
              nb.length +
              ' points within ε, fewer than minPts = ' +
              S.m +
              '. It joins the cluster as a border point, and the cluster does not spread through it. ' +
              queue.length +
              ' left in the queue.';
          }
          if (!queue.length) {
            msg += ' The queue is empty: cluster ' + (curC + 1) + ' is complete.';
            cu = 4;
          }
          return;
        }
        while (scan < order.length && lab[order[scan]] !== -2) scan++;
        if (scan >= order.length) {
          done = true;
          cur = -1;
          nb = [];
          cu = 4;
          const no = lab.filter((v) => v === -1).length;
          msg =
            'Every point has been visited. DBSCAN found ' +
            nc +
            ' cluster' +
            (nc === 1 ? '' : 's') +
            ' and left ' +
            no +
            ' point' +
            (no === 1 ? '' : 's') +
            ' as noise, without being told how many clusters to look for.';
          return;
        }
        const p = order[scan];
        cur = p;
        nb = nbr(P[p].x, P[p].y);
        vis++;
        if (nb.length < S.m) {
          lab[p] = -1;
          typ[p] = 3;
          cu = 1;
          msg =
            'New point: only ' +
            nb.length +
            ' point' +
            (nb.length === 1 ? '' : 's') +
            ' within ε (itself included), fewer than minPts = ' +
            S.m +
            '. Marked as noise for now. A cluster that grows this far may still claim it as a border point.';
        } else {
          curC = nc++;
          lab[p] = curC;
          typ[p] = 1;
          for (const j of nb)
            if ((lab[j] === -2 || lab[j] === -1) && !inQ[j]) {
              inQ[j] = 1;
              queue.push(j);
            }
          cu = 2;
          msg =
            'New point: ' +
            nb.length +
            ' points within ε, at least minPts = ' +
            S.m +
            ', so it is a core point. Cluster ' +
            (curC + 1) +
            ' starts here, and its ' +
            queue.length +
            ' unclaimed neighbours go into the queue.';
        }
      };
      const fin = () => {
        let k = 0;
        while (!done && k++ < 1e5) step();
        cur = -1;
        nb = [];
      };
      const tp = testPoint(
        root,
        'db',
        ['Your x', 'Your y'],
        [0, 0],
        [
          [-3.8, 3.8],
          [-2.85, 2.85],
        ],
      );
      const draw = () => {
        const t = T(),
          col = UC(t);
        {
          const V = uView(c),
            { g, w, h, X, Y, s } = V;
          uFrame(V, t, 'x', 'y');
          if (cur >= 0) {
            g.beginPath();
            g.arc(X(P[cur].x), Y(P[cur].y), S.eps * s, 0, 7);
            g.fillStyle = mix(t.ink, t.ink, 0, 0.07);
            g.fill();
            g.strokeStyle = t.ink;
            g.lineWidth = 1.6;
            g.stroke();
          }
          P.forEach((p, i) => {
            const x = X(p.x),
              y = Y(p.y),
              k = lab[i];
            if (typ[i] === 3 && k === -1) {
              g.strokeStyle = t.muted;
              g.lineWidth = 1.6;
              g.beginPath();
              g.moveTo(x - 3.5, y - 3.5);
              g.lineTo(x + 3.5, y + 3.5);
              g.moveTo(x + 3.5, y - 3.5);
              g.lineTo(x - 3.5, y + 3.5);
              g.stroke();
            } else if (k >= 0 && typ[i] === 1) {
              g.fillStyle = col[k % 8];
              g.beginPath();
              g.arc(x, y, 4.2, 0, 7);
              g.fill();
            } else if (k >= 0) {
              g.strokeStyle = col[k % 8];
              g.lineWidth = 2;
              g.fillStyle = t.bg;
              g.beginPath();
              g.arc(x, y, 3.6, 0, 7);
              g.fill();
              g.stroke();
            } else {
              g.fillStyle = mix(t.ink, t.ink, 0, 0.32);
              g.beginPath();
              g.arc(x, y, 2.8, 0, 7);
              g.fill();
            }
            if (inQ[i] && lab[i] < 0 && !done) {
              g.strokeStyle = t.gold;
              g.lineWidth = 1.8;
              g.beginPath();
              g.arc(x, y, 6.5, 0, 7);
              g.stroke();
            }
          });
          if (cur >= 0) {
            g.strokeStyle = t.ink;
            g.lineWidth = 2.4;
            g.beginPath();
            g.arc(X(P[cur].x), Y(P[cur].y), 7.5, 0, 7);
            g.stroke();
          }
          {
            const x = X(tp.v[0]),
              y = Y(tp.v[1]);
            g.setLineDash([4, 4]);
            g.strokeStyle = t.muted;
            g.lineWidth = 1.2;
            g.beginPath();
            g.arc(x, y, S.eps * s, 0, 7);
            g.stroke();
            g.setLineDash([]);
            markTP(g, x, y, t);
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('gold ring = waiting in the queue', 8, h - 8);
        }
        {
          const { g, w, h } = fit(cr),
            ym = Math.max(1.3, kd.length ? kd[0] : 1) * 1.05,
            L = uLine(g, t, 0, 0, w, h, kd, {
              title:
                'distance to the ' +
                S.m +
                (S.m === 2 ? 'nd' : S.m === 3 ? 'rd' : 'th') +
                ' nearest point (itself counted), largest first',
              ym,
              col: t.accent,
            }),
            y = L.Y(S.eps);
          g.setLineDash([7, 5]);
          g.strokeStyle = t.gold;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(L.m.l, y);
          g.lineTo(w - L.m.r, y);
          g.stroke();
          g.setLineDash([]);
          const above = kd.filter((v) => v > S.eps).length;
          g.font = MONO;
          g.fillStyle = t.cut;
          g.textAlign = 'right';
          g.fillText(
            'ε = ' +
              f2(S.eps) +
              ': ' +
              above +
              ' of ' +
              kd.length +
              ' points lie above and cannot be core',
            w - L.m.r,
            y - 6,
          );
          g.fillStyle = t.muted;
          g.fillText('points, most isolated first →', w - L.m.r, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'db', cu);
        {
          const q = nbr(tp.v[0], tp.v[1]),
            cores = q.filter((j) => typ[j] === 1),
            cnt = q.length + 1;
          let s =
            'Your point would have ' +
            cnt +
            ' point' +
            (cnt === 1 ? '' : 's') +
            ' within ε, counting itself. ';
          if (!done) s += 'Run to the end to see which cluster claims it.';
          else if (cnt >= S.m && cores.length)
            s +=
              'That reaches minPts, so it would be a core point of cluster ' +
              (lab[cores[0]] + 1) +
              '.';
          else if (cnt >= S.m)
            s +=
              'That reaches minPts, so it would be a core point and start a cluster of its own with its neighbours.';
          else if (cores.length)
            s +=
              'That is below minPts, but a core point of cluster ' +
              (lab[cores[0]] + 1) +
              ' is within reach, so it would be a border point of that cluster.';
          else s += 'That is below minPts and no core point is within reach: it would be noise.';
          tp.out.textContent = s;
        }
        setR(root, 'v', vis + ' of ' + P.length);
        setR(root, 'cl', String(nc));
        setR(root, 'co', String(typ.filter((v) => v === 1).length));
        setR(root, 'bo', String(typ.filter((v) => v === 2).length));
        setR(root, 'no', String(lab.filter((v) => v === -1).length));
      };
      tp.on = () => draw();
      const redo = () => {
        pb.stop();
        kdist();
        reset();
        fin();
        draw();
      };
      const gen = uDataBind(root, 'db', S, () => {
        P = gen();
        redo();
      });
      P = gen();
      kdist();
      reset();
      fin();
      const anim = Anim(() => {
        fr++;
        if (wait > 0) {
          wait--;
          if (!wait) {
            run++;
            reset();
            draw();
          }
          return;
        }
        if (fr % (vis < 12 ? 26 : 4)) return;
        step();
        draw();
        chk();
      });
      pb = playBtn($('#db-play', root), anim);
      const chk = runCtl(pb, root, 'db', {
        done: () => done,
        restart: () => {
          run++;
          reset();
        },
      });
      $('#db-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#db-reset', root).onclick = () => {
        pb.stop();
        run++;
        reset();
        draw();
      };
      $('#db-fin', root).onclick = () => {
        pb.stop();
        if (done) reset();
        fin();
        draw();
      };
      bind(root, 'db-eps', (v) => {
        S.eps = v;
        redo();
      });
      bind(
        root,
        'db-m',
        (v) => {
          S.m = v;
          redo();
        },
        (v) => String(v),
      );
      bind(
        root,
        'db-bg',
        (v) => {
          S.bg = v;
          P = gen();
          redo();
        },
        (v) => String(v),
      );
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'pca',
    tab: 'PCA',
    title: 'PCA: find the direction that matters most',
    lede: 'Two measurements that rise and fall together are partly saying the same thing twice. Principal component analysis turns the axes to line up with the data, so that the first new axis carries as much of the spread as one number can, and the rest can be dropped with the least possible loss.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The cloud, a line through its middle, and each point’s shadow on the line</figcaption><canvas class="short" id="pc-c" role="img" aria-label="Point cloud with a direction and the projections onto it"></canvas></figure>
 <figure class="fig"><figcaption>Top: spread kept and spread lost for every direction. Bottom: the data seen along the current direction</figcaption><canvas class="sq" id="pc-r" role="img" aria-label="Variance against angle and the one-dimensional projection"></canvas></figure></div>
 ${uHow('pc', ['Centre: subtract the mean of each column, so the cloud sits on the origin.', 'Compute the covariance matrix: the spread of each column and how they move together.', 'Find the direction with the most spread. Here: start with any direction, multiply it by the covariance matrix, rescale, and repeat until it stops turning.', 'The second component is the direction of most spread at right angles to the first.', 'Project: keep each point’s position along the first component and drop the rest.'])}
 <div class="controls">${uBtns('pc', 'Run', 'One step')}
 ${grp('Try a direction yourself', slider('pc-th', 'Angle of the line θ', 0, 180, 1, 150) + `<p class="status" id="pc-hint">Move the slider and watch the gold curve: can you find its peak by hand?</p>`)}
 ${uDataHTML('pc', ['corr', 'banana', 'twoc', 'round'], 'corr', 20, 400, 160, slider('pc-rho', 'Correlation between the two columns', -0.97, 0.97, 0.01, 0.8))}</div>
 <dl class="read">${rd('th', 'Angle of the line')}${rd('va', 'Spread along the line (variance)', 'hl')}${rd('sh', 'Share of the total spread kept', 'hl')}${rd('er', 'Spread lost (mean squared distance to the line)', 'hl2')}${rd('l1', 'Eigenvalues λ₁, λ₂')}${rd('a1', 'Angle of the first component')}</dl>`,
    tryit:
      'Before pressing anything, drag the angle slider and try to make the shadows as spread out as possible. Note your angle, then press Start over and step through: how close were you? Set the correlation to 0 and look at the two curves on the right. Why is there nothing for PCA to find? Finally choose A Curved Cloud: PCA still gives an answer. What does it miss?',
    math: String.raw`<h4>Intuition</h4><p>Hold a pencil under a lamp and look at its shadow on the table. Pointed at the lamp, the shadow is a dot and tells you nothing. Held sideways, the shadow shows the whole length. A shadow is a projection: it flattens the object and loses one dimension, and how much it loses depends on the direction. PCA asks for the direction of the lamp that keeps the longest shadow of the data. For a cloud of points, "longest" means the largest variance of the shadows.</p>
 <h4>Setting it up</h4><p>Let \(\mathbf x_1,\dots,\mathbf x_n\) be the points with their mean subtracted. The covariance matrix is</p>$$\mathbf S=\frac{1}{n-1}\sum_{i=1}^{n}\mathbf x_i\mathbf x_i^{\top}.$$<p>Take a direction, a unit vector \(\mathbf w\). The shadow of \(\mathbf x_i\) on that direction is the number \(z_i=\mathbf w^{\top}\mathbf x_i\), and the variance of the shadows is</p>$$\operatorname{Var}(z)=\frac{1}{n-1}\sum_i(\mathbf w^{\top}\mathbf x_i)^2=\mathbf w^{\top}\mathbf S\,\mathbf w .$$<p>This is the gold curve on the right, drawn for every angle.</p>
 <h4>The first component</h4><p>We want to maximise \(\mathbf w^{\top}\mathbf S\mathbf w\) subject to \(\mathbf w^{\top}\mathbf w=1\). With a Lagrange multiplier \(\lambda\), setting the gradient of \(\mathbf w^{\top}\mathbf S\mathbf w-\lambda(\mathbf w^{\top}\mathbf w-1)\) to zero gives</p>$$\mathbf S\,\mathbf w=\lambda\,\mathbf w .$$<p>So the best direction is an eigenvector of the covariance matrix, and the variance it captures is \(\mathbf w^{\top}\mathbf S\mathbf w=\lambda\), its eigenvalue. The first principal component is the eigenvector with the largest eigenvalue \(\lambda_1\). The second is the eigenvector with the next largest, and because \(\mathbf S\) is symmetric it is automatically at right angles to the first. In the Gaussian module these were the long and short axes of the covariance ellipse.</p>
 <h4>The same answer from the other side</h4><p>By Pythagoras, for each centred point, (length)² = (shadow)² + (distance to the line)². The total of the left side is fixed by the data. So the direction that maximises the spread of the shadows is exactly the direction that minimises the total squared distance from the points to the line: the red curve is the gold curve turned upside down. PCA is at once "keep the most variance" and "the best-fitting line", where distance is measured at right angles to the line, not vertically as in regression.</p>
 <h4>How the direction is found</h4><p>The animation uses <b>power iteration</b>. Start from any unit vector \(\mathbf w_0\) and repeat</p>$$\mathbf w_{t+1}=\frac{\mathbf S\,\mathbf w_t}{\lVert\mathbf S\,\mathbf w_t\rVert}.$$<p>Write \(\mathbf w_0\) as a mix of the eigenvectors. Each multiplication by \(\mathbf S\) stretches the part along the first eigenvector by \(\lambda_1\) and the rest by less, so after a few rounds the first eigenvector dominates. The error shrinks by a factor \(\lambda_2/\lambda_1\) every step: quickly for a long thin cloud, slowly for a nearly round one. Libraries use the singular value decomposition of the data matrix instead, which gives all components at once.</p>
 <h4>How much to keep</h4><p>The total variance is \(\lambda_1+\lambda_2+\dots+\lambda_d\). Keeping the first \(m\) components keeps the share</p>$$\frac{\lambda_1+\dots+\lambda_m}{\lambda_1+\dots+\lambda_d},$$<p>and the mean squared reconstruction error is the sum of the eigenvalues that were dropped. The rebuilt point is \(\hat{\mathbf x}=\boldsymbol\mu+\sum_{k\le m}z_k\mathbf w_k\).</p>
 <h4>What PCA cannot see</h4><p>PCA only looks at variances and covariances, so it only finds straight directions. On the curved cloud the data really has one degree of freedom, position along the curve, but no straight line captures it. PCA also knows nothing about labels: the direction of most spread need not be the direction that separates two classes. And it depends on units, since a column measured in millimetres has a thousand times the variance of the same column in metres. The next tab deals with that.</p>`,
    q: 'Regression of y on x and the first principal component both draw a line through the cloud. Are they the same line?',
    a: 'No. Regression minimises the vertical distances from the points to the line, because it treats x as given and y as the thing to predict. PCA minimises the perpendicular distances, treating both columns alike. The PCA line is steeper than the regression line of y on x whenever the points do not lie exactly on a line, and it lies between the regression of y on x and the regression of x on y.',
    init(root) {
      const c = $('#pc-c', root),
        cr = $('#pc-r', root),
        live = $('#pc-live', root),
        S = { ds: 'corr', n: 160, seed: 1, rho: 0.8 };
      let P = [],
        mu = { x: 0, y: 0 },
        Q = [],
        cv = { a: 1, b: 0, d: 1 },
        e1 = { l1: 1, l2: 1, th: 0 },
        v = null,
        ph = 'raw',
        cen = 0,
        pj = 0,
        pit = 0,
        msg = '',
        cu = -1,
        wait = 0,
        fr = 0,
        pb,
        twId = null,
        man = false;
      const mk = () => {
        const r = rng(S.seed * 7 + 3),
          rn = randnFrom(r),
          o = [];
        for (let i = 0; i < S.n; i++) {
          const u = rn(),
            w = rn();
          let x, y;
          if (S.ds === 'corr') {
            x = 1.05 * u;
            y = 0.8 * (S.rho * u + Math.sqrt(1 - S.rho * S.rho) * w);
          } else if (S.ds === 'banana') {
            x = 1.05 * u;
            y = 0.42 * x * x - 0.7 + 0.2 * w;
          } else if (S.ds === 'twoc') {
            const k = i % 2 ? 1 : -1;
            x = k * 1.3 + 0.42 * u;
            y = k * 0.75 + 0.42 * w;
          } else {
            x = 0.85 * u;
            y = 0.85 * w;
          }
          o.push({ x: clamp(x + 0.8, -3.8, 3.8), y: clamp(y + 0.55, -2.85, 2.85) });
        }
        return o;
      };
      const comp = () => {
        const n = P.length || 1;
        mu = { x: P.reduce((s, p) => s + p.x, 0) / n, y: P.reduce((s, p) => s + p.y, 0) / n };
        Q = P.map((p) => ({ x: p.x - mu.x, y: p.y - mu.y }));
        let a = 0,
          b = 0,
          d = 0;
        for (const q of Q) {
          a += q.x * q.x;
          b += q.x * q.y;
          d += q.y * q.y;
        }
        const m = Math.max(1, n - 1);
        cv = { a: a / m, b: b / m, d: d / m };
        const mm = (cv.a + cv.d) / 2,
          rr = Math.hypot((cv.a - cv.d) / 2, cv.b);
        let th = 0.5 * Math.atan2(2 * cv.b, cv.a - cv.d);
        if (th < 0) th += Math.PI;
        e1 = { l1: mm + rr, l2: Math.max(0, mm - rr), th };
      };
      const va = (th) => {
          const cc = Math.cos(th),
            ss = Math.sin(th);
          return cv.a * cc * cc + 2 * cv.b * cc * ss + cv.d * ss * ss;
        },
        ang = () => {
          let a = Math.atan2(v.y, v.x);
          if (a < 0) a += Math.PI;
          if (a >= Math.PI) a -= Math.PI;
          return a;
        },
        deg = (a) => ((a * 180) / Math.PI).toFixed(0) + '°';
      const tween = (set, to) => {
        cancelAnimationFrame(twId);
        if (RM) {
          set(to);
          return;
        }
        let k = 0;
        const f = () => {
          k++;
          set(to * (k / 22));
          draw();
          if (k < 22) twId = requestAnimationFrame(f);
        };
        twId = requestAnimationFrame(f);
      };
      const reset = () => {
        cancelAnimationFrame(twId);
        v = null;
        ph = 'raw';
        cen = 0;
        pj = 0;
        pit = 0;
        cu = -1;
        wait = 0;
        man = false;
        msg =
          'The raw data: ' +
          P.length +
          ' points with mean (' +
          f2(mu.x) +
          ', ' +
          f2(mu.y) +
          '), marked by the cross. Press Run or One step.';
      };
      const step = () => {
        if (P.length < 3) return;
        man = false;
        if (ph === 'raw') {
          ph = 'centre';
          cu = 0;
          tween((u) => {
            cen = u;
          }, 1);
          msg =
            'Centre: the mean (' +
            f2(mu.x) +
            ', ' +
            f2(mu.y) +
            ') was subtracted from every point, so the cloud now sits on the origin. PCA describes spread around the mean, not where the mean is.';
          return;
        }
        if (ph === 'centre') {
          cen = 1;
          ph = 'cov';
          cu = 1;
          msg =
            'Covariance matrix: the variance of x is ' +
            f2(cv.a) +
            ', the variance of y is ' +
            f2(cv.d) +
            ', and their covariance is ' +
            f2(cv.b) +
            '. The total spread is ' +
            f2(cv.a) +
            ' + ' +
            f2(cv.d) +
            ' = ' +
            f2(cv.a + cv.d) +
            '. The ellipse is drawn from these three numbers.';
          return;
        }
        if (ph === 'cov') {
          ph = 'power';
          cu = 2;
          pit = 0;
          const a = e1.th + 1.4;
          v = { x: Math.cos(a), y: Math.sin(a) };
          msg =
            'Start with an arbitrary direction, at ' +
            deg(ang()) +
            '. The spread of the shadows along it is ' +
            f2(va(ang())) +
            ', which is ' +
            ((100 * va(ang())) / (cv.a + cv.d)).toFixed(0) +
            '% of the total. Now improve it.';
          return;
        }
        if (ph === 'power') {
          const o = ang(),
            wx = cv.a * v.x + cv.b * v.y,
            wy = cv.b * v.x + cv.d * v.y,
            nn = Math.hypot(wx, wy) || 1;
          v = { x: wx / nn, y: wy / nn };
          pit++;
          let dl = Math.abs(ang() - o);
          if (dl > Math.PI / 2) dl = Math.PI - dl;
          msg =
            'Power iteration ' +
            pit +
            ': the direction was multiplied by the covariance matrix and rescaled to length 1. It turned by ' +
            ((dl * 180) / Math.PI).toFixed(1) +
            '° toward the long axis of the cloud, to ' +
            deg(ang()) +
            '. Spread along it: ' +
            f2(va(ang())) +
            ' (' +
            ((100 * va(ang())) / (cv.a + cv.d)).toFixed(0) +
            '% of the total).';
          if (dl < 0.004 || pit >= 40) {
            ph = 'pc1';
            msg +=
              ' It has stopped turning. This is the first principal component, and the spread along it is the largest eigenvalue, λ₁ = ' +
              f2(e1.l1) +
              '.';
          }
          return;
        }
        if (ph === 'pc1') {
          ph = 'pc2';
          cu = 3;
          msg =
            'The second component is at right angles to the first. In two dimensions there is only one such direction. The spread along it is λ₂ = ' +
            f2(e1.l2) +
            ', the smallest of any direction, and λ₁ + λ₂ = ' +
            f2(e1.l1 + e1.l2) +
            ' is the total.';
          return;
        }
        if (ph === 'pc2') {
          ph = 'proj';
          cu = 4;
          tween((u) => {
            pj = u;
          }, 1);
          msg =
            'Project: every point slid along its dashed line onto the first component and now needs one number instead of two. That keeps ' +
            ((100 * e1.l1) / (e1.l1 + e1.l2)).toFixed(1) +
            '% of the spread. What was thrown away is the mean squared distance the points moved: λ₂ = ' +
            f2(e1.l2) +
            '.';
          return;
        }
        reset();
      };
      const tp = testPoint(
        root,
        'pc',
        ['Your x', 'Your y'],
        [2, 0.4],
        [
          [-3.8, 3.8],
          [-2.85, 2.85],
        ],
      );
      const draw = () => {
        const t = T(),
          tot = cv.a + cv.d || 1,
          hasV = !!v,
          know = man || ['pc1', 'pc2', 'proj'].includes(ph);
        {
          const V = uView(c),
            { g, w, h, X, Y } = V;
          uFrame(
            V,
            t,
            S.ds === 'corr' ? 'height (standardised)' : 'x',
            S.ds === 'corr' ? 'arm span (standardised)' : 'y',
          );
          const ox = mu.x * (1 - cen),
            oy = mu.y * (1 - cen);
          if (ph !== 'raw' && ph !== 'centre') {
            ellPath(g, X, Y, ox, oy, cv.a, cv.b, cv.d, 2);
            g.strokeStyle = t.muted;
            g.lineWidth = 1.2;
            g.setLineDash([3, 4]);
            g.stroke();
            g.setLineDash([]);
          }
          if (hasV) {
            const L = 6;
            g.strokeStyle = know && !man ? t.gold : t.ink;
            g.lineWidth = 2.6;
            g.beginPath();
            g.moveTo(X(ox - L * v.x), Y(oy - L * v.y));
            g.lineTo(X(ox + L * v.x), Y(oy + L * v.y));
            g.stroke();
            if (ph === 'pc2' || ph === 'proj') {
              g.strokeStyle = t.muted;
              g.lineWidth = 1.6;
              g.beginPath();
              g.moveTo(X(ox + 1.6 * v.y), Y(oy - 1.6 * v.x));
              g.lineTo(X(ox - 1.6 * v.y), Y(oy + 1.6 * v.x));
              g.stroke();
              g.font = SANS;
              g.fillStyle = t.muted;
              g.textAlign = 'left';
              g.fillText('second component', X(ox - 1.6 * v.y) + 8, Y(oy + 1.6 * v.x));
            }
            if (know && !man) {
              g.font = SANS;
              g.fillStyle = t.cut;
              g.textAlign = 'left';
              const s = v.x < 0 ? -1 : 1;
              g.fillText(
                'first component',
                clamp(X(ox + s * 2.7 * v.x) + 8, 10, w - 120),
                clamp(Y(oy + s * 2.7 * v.y) - 8, 16, h - 10),
              );
            }
            g.strokeStyle = mix(t.ok, t.ok, 0, 0.4);
            g.lineWidth = 1;
            g.beginPath();
            for (const q of Q) {
              const z = q.x * v.x + q.y * v.y;
              g.moveTo(X(ox + q.x), Y(oy + q.y));
              g.lineTo(X(ox + z * v.x), Y(oy + z * v.y));
            }
            g.stroke();
          }
          g.fillStyle = mix(t.accent, t.accent, 0, 0.85);
          for (const q of Q) {
            let x = ox + q.x,
              y = oy + q.y;
            if (hasV && pj > 0) {
              const z = q.x * v.x + q.y * v.y;
              x += (ox + z * v.x - x) * pj;
              y += (oy + z * v.y - y) * pj;
            }
            g.beginPath();
            g.arc(X(x), Y(y), 3.3, 0, 7);
            g.fill();
          }
          g.strokeStyle = t.ink;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(X(ox) - 7, Y(oy));
          g.lineTo(X(ox) + 7, Y(oy));
          g.moveTo(X(ox), Y(oy) - 7);
          g.lineTo(X(ox), Y(oy) + 7);
          g.stroke();
          {
            const dx = tp.v[0] - mu.x,
              dy = tp.v[1] - mu.y,
              x = X(ox + dx),
              y = Y(oy + dy);
            if (hasV && know) {
              const z = dx * v.x + dy * v.y;
              g.strokeStyle = t.ink;
              g.lineWidth = 1.4;
              g.setLineDash([4, 3]);
              g.beginPath();
              g.moveTo(x, y);
              g.lineTo(X(ox + z * v.x), Y(oy + z * v.y));
              g.stroke();
              g.setLineDash([]);
              g.fillStyle = t.bg;
              g.beginPath();
              g.arc(X(ox + z * v.x), Y(oy + z * v.y), 5, 0, 7);
              g.fill();
              g.stroke();
            }
            markTP(g, x, y, t);
          }
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.56,
            m = { l: 44, r: 12, t: 26, b: 22 },
            Xa = (a) => m.l + (a / Math.PI) * (w - m.l - m.r),
            ym = Math.max(1e-9, e1.l1) * 1.12,
            Ya = (u) => h1 - m.b - (u / ym) * (h1 - m.t - m.b);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h1 - m.b);
          g.lineTo(w - m.r, h1 - m.b);
          g.stroke();
          g.font = MONO;
          g.textAlign = 'center';
          g.fillStyle = t.muted;
          [0, 45, 90, 135, 180].forEach((a) =>
            g.fillText(a + '°', Xa((a * Math.PI) / 180), h1 - m.b + 14),
          );
          g.textAlign = 'right';
          g.fillText(uNum(ym / 1.12), m.l - 5, Ya(ym / 1.12) + 4);
          g.fillText('0', m.l - 5, Ya(0) + 4);
          g.textAlign = 'left';
          g.fillStyle = t.cut;
          g.fillText('gold: spread kept along the line', 8, 12);
          g.fillStyle = t.ok;
          g.fillText('red: spread lost (they always add up to ' + f2(tot) + ')', 8, 25);
          [
            [(a) => va(a), t.gold],
            [(a) => tot - va(a), t.ok],
          ].forEach(([f, cl]) => {
            g.strokeStyle = cl;
            g.lineWidth = 2.4;
            g.beginPath();
            for (let i = 0; i <= 180; i++) {
              const a = (i * Math.PI) / 180;
              i ? g.lineTo(Xa(a), Ya(f(a))) : g.moveTo(Xa(a), Ya(f(a)));
            }
            g.stroke();
          });
          if ((know && !man) || ph === 'proj') {
            g.setLineDash([4, 4]);
            g.strokeStyle = t.muted;
            g.lineWidth = 1.2;
            g.beginPath();
            g.moveTo(Xa(e1.th), Ya(0));
            g.lineTo(Xa(e1.th), Ya(e1.l1));
            g.stroke();
            g.setLineDash([]);
          }
          if (hasV) {
            const a = ang();
            g.strokeStyle = t.ink;
            g.lineWidth = 1.6;
            g.beginPath();
            g.moveTo(Xa(a), Ya(0));
            g.lineTo(Xa(a), m.t + 4);
            g.stroke();
            g.fillStyle = t.gold;
            g.beginPath();
            g.arc(Xa(a), Ya(va(a)), 6, 0, 7);
            g.fill();
            g.fillStyle = t.ok;
            g.beginPath();
            g.arc(Xa(a), Ya(tot - va(a)), 5, 0, 7);
            g.fill();
          }
          const y0 = h1 + 18,
            hb = h - y0 - 24,
            Xs = (z) => w / 2 + (z * (w - 56)) / 8.4;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            hasV ? 'each point’s position along the line (its shadow)' : 'no direction chosen yet',
            8,
            y0,
          );
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(28, h - 22);
          g.lineTo(w - 28, h - 22);
          g.stroke();
          g.textAlign = 'center';
          [-4, -2, 0, 2, 4].forEach((z) => g.fillText(String(z), Xs(z), h - 7));
          if (hasV) {
            const bins = {};
            g.fillStyle = mix(t.accent, t.accent, 0, 0.85);
            const rad = Math.max(2, Math.min(3.4, hb / 26));
            for (const q of Q) {
              const z = q.x * v.x + q.y * v.y,
                b = Math.round(z / 0.14),
                k = (bins[b] = (bins[b] || 0) + 1);
              g.beginPath();
              g.arc(Xs(b * 0.14), h - 22 - rad - 1 - (k - 1) * rad * 2, rad, 0, 7);
              g.fill();
            }
            const sdv = Math.sqrt(va(ang()));
            g.strokeStyle = t.gold;
            g.lineWidth = 3;
            g.beginPath();
            g.moveTo(Xs(-sdv), h - 22);
            g.lineTo(Xs(sdv), h - 22);
            g.stroke();
          }
        }
        live.textContent = msg;
        uCur(root, 'pc', cu);
        if (hasV) {
          const a = ang();
          setR(root, 'th', deg(a));
          setR(root, 'va', f2(va(a)));
          setR(root, 'sh', ((100 * va(a)) / tot).toFixed(1) + '%');
          setR(root, 'er', f2(tot - va(a)));
        } else ['th', 'va', 'sh', 'er'].forEach((k) => setR(root, k, '–'));
        const sh = ph !== 'raw' && ph !== 'centre';
        setR(root, 'l1', sh ? f2(e1.l1) + ', ' + f2(e1.l2) : '–');
        setR(root, 'a1', (know && !man) || ph === 'proj' ? deg(e1.th) : '–');
        if (hasV && know) {
          const dx = tp.v[0] - mu.x,
            dy = tp.v[1] - mu.y,
            z = dx * v.x + dy * v.y,
            rx = mu.x + z * v.x,
            ry = mu.y + z * v.y;
          tp.out.textContent =
            'Your point minus the mean is (' +
            f2(dx) +
            ', ' +
            f2(dy) +
            '). Its position along the line is z = ' +
            f2(z) +
            '. Rebuilt from that one number it becomes (' +
            f2(rx) +
            ', ' +
            f2(ry) +
            '), which is ' +
            f2(Math.hypot(tp.v[0] - rx, tp.v[1] - ry)) +
            ' from where it started.';
        } else
          tp.out.textContent = 'Find a direction first: press One step, or move the angle slider.';
        if (hasV && !man) thb.set((ang() * 180) / Math.PI);
      };
      tp.on = () => draw();
      const gen = uDataBind(
        root,
        'pc',
        S,
        () => {
          pb.stop();
          P = gen();
          comp();
          reset();
          draw();
        },
        mk,
      );
      P = gen();
      comp();
      reset();
      const thb = bind(
        root,
        'pc-th',
        (d) => {
          pb.stop();
          cancelAnimationFrame(twId);
          const a = (d * Math.PI) / 180;
          v = { x: Math.cos(a), y: Math.sin(a) };
          man = true;
          ph = 'manual';
          cen = 1;
          pj = 0;
          cu = 2;
          const b = (e1.th * 180) / Math.PI;
          msg =
            'Your direction, at ' +
            d.toFixed(0) +
            '°, keeps ' +
            ((100 * va(a)) / (cv.a + cv.d)).toFixed(1) +
            '% of the spread. The most any direction can keep is ' +
            ((100 * e1.l1) / (cv.a + cv.d)).toFixed(1) +
            '%. ' +
            (Math.min(Math.abs(d - b), 180 - Math.abs(d - b)) < 3
              ? 'You have found it.'
              : 'Keep turning.');
          draw();
        },
        (d) => Math.round(d) + '°',
      );
      bind(root, 'pc-rho', (r) => {
        S.rho = r;
        pb.stop();
        if (S.ds === 'corr') {
          P = gen();
          comp();
          const k = man;
          reset();
          if (k) {
            thb.el.dispatchEvent(new Event('input'));
            return;
          }
        }
        draw();
      });
      const anim = Anim(() => {
        fr++;
        if (wait > 0) {
          wait--;
          if (!wait) {
            reset();
            draw();
          }
          return;
        }
        if (fr % (ph === 'power' ? 30 : 70)) return;
        if (ph === 'manual') reset();
        step();
        draw();
        chk();
      });
      pb = playBtn($('#pc-play', root), anim);
      const chk = runCtl(pb, root, 'pc', { done: () => ph === 'proj', restart: () => reset() });
      $('#pc-step', root).onclick = () => {
        pb.stop();
        if (ph === 'manual') reset();
        step();
        draw();
      };
      $('#pc-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      return {
        draw,
        stop: () => {
          pb.stop();
          cancelAnimationFrame(twId);
        },
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'pca-table',
    tab: 'PCA on a table',
    title: 'PCA on a real table: many columns into two',
    lede: 'With two columns you can look at the cloud. With six or twelve you cannot, and that is where PCA earns its place: it finds the few combinations of columns that carry most of the variation, so a wide table can be drawn on one page. Use the built-in tables or upload your own.',
    stage: `<div class="stage"><figure class="fig"><figcaption id="pt-hcap">The table</figcaption><canvas class="short" id="pt-h" role="img" aria-label="First rows of the table, then the correlation matrix"></canvas></figure>
 <figure class="fig"><figcaption>Scree plot: share of the total spread carried by each component, and the running total</figcaption><canvas class="short" id="pt-s" role="img" aria-label="Scree plot"></canvas></figure></div>
 <div class="stage" style="margin-top:22px"><figure class="fig"><figcaption>Every row as a point on the first two components. Arrows show which way each column pulls</figcaption><canvas class="short" id="pt-b" role="img" aria-label="Rows plotted on the first two components with column arrows"></canvas></figure>
 <figure class="fig"><figcaption id="pt-lcap">Weights of the component</figcaption><canvas class="short" id="pt-l" role="img" aria-label="Weights of one component on each column"></canvas></figure></div>
 ${uHow('pt', ['Centre each column, and divide by its standard deviation so that units do not matter.', 'Compute the correlation matrix: one number for every pair of columns.', 'Find its eigenvectors, largest eigenvalue first. Each is a component: a recipe that mixes the columns.', 'Read the scree plot and decide how many components to keep.', 'Project: replace every row by its scores on the components you kept.'])}
 <div class="controls">${uBtns('pt', 'Run', 'One step')}
 ${grp('The model', `<label class="check"><input type="checkbox" id="pt-std" checked> Standardise the columns first</label>` + slider('pt-k', 'Component to inspect', 1, 12, 1, 1))}
 ${grp('The data', `<label class="ctl" for="pt-ds"><span>Table</span><select id="pt-ds"><option value="exam">Exam marks in six subjects (simulated)</option><option value="body">Body measurements (simulated)</option></select></label><div class="row"><label class="btn file">Upload a CSV<input type="file" id="pt-file" accept=".csv,.txt,text/csv"></label></div><p class="status" id="pt-st" role="status"></p>`)}</div>
 <dl class="read">${rd('n', 'Rows')}${rd('d', 'Columns')}${rd('p1', 'Spread carried by component 1', 'hl')}${rd('p2', 'By components 1 and 2 together', 'hl')}${rd('k90', 'Components needed for 90%', 'hl2')}</dl>`,
    tryit:
      'On the exam marks, step through and read the weights of component 1 and then component 2. One is "good at everything" and the other is "numbers against words": the table of six subjects is mostly two ideas. Now switch to Body Measurements and untick Standardise. Which column takes over component 1, and why? Then upload a table of your own with a header row.',
    math: String.raw`<h4>Intuition</h4><p>A school report lists six marks for each student. If you had to describe a student with only two numbers, you would not pick two of the six subjects and discard the rest. You would say something like "overall, strong" and "better at sciences than at languages". Those are not columns of the table. They are combinations of columns, and they summarise the table well because the subjects are correlated. PCA finds such combinations automatically and tells you how much each one explains.</p>
 <h4>From the table to the components</h4><p>Let \(\mathbf X\) be the table with \(n\) rows and \(d\) columns. Standardise each column: \(z_{ij}=(x_{ij}-\bar x_j)/s_j\). The correlation matrix is</p>$$\mathbf R=\frac{1}{n-1}\mathbf Z^{\top}\mathbf Z,$$<p>a \(d\times d\) table whose entry \((j,k)\) is the correlation between columns \(j\) and \(k\). Its eigenvectors \(\mathbf w_1,\dots,\mathbf w_d\), ordered by eigenvalue \(\lambda_1\ge\dots\ge\lambda_d\), are the principal components. The entries of \(\mathbf w_k\) are called <b>weights</b> or <b>loadings</b>: how much of each column goes into component \(k\).</p>
 <h4>Scores</h4><p>The score of row \(i\) on component \(k\) is \(t_{ik}=\mathbf w_k^{\top}\mathbf z_i\): multiply each standardised entry by its weight and add. The lower-left picture plots \((t_{i1},t_{i2})\) for every row. Two rows that are close in that picture are similar across all \(d\) columns, to the extent that two components capture the table.</p>
 <h4>The scree plot</h4><p>With standardised columns each has variance 1, so the total is \(d\) and \(\sum_k\lambda_k=d\). Component \(k\) carries the share \(\lambda_k/d\). Plotting these shares in order gives the scree plot. Common rules for how many to keep: enough to reach 80 or 90% of the total; the components before the curve flattens; or those with \(\lambda_k&gt;1\), meaning they carry more than one original column's worth. None is a law, and the right number depends on what the components are for.</p>
 <h4>Why standardise</h4><p>Without it PCA uses the covariance matrix, and a column with large numbers dominates purely because of its units. Weight in kilograms with a standard deviation of 12 has a variance of 144; shoe size with a standard deviation of 2 has a variance of 4. Component 1 would be almost entirely weight. Standardise unless all columns share one unit and their differences in spread are meaningful.</p>
 <h4>Reading the arrows</h4><p>In the lower-left picture each column is an arrow from the origin to its pair of weights on components 1 and 2. Arrows pointing the same way belong to columns that rise together. Arrows at right angles are unrelated, and opposite arrows are columns that trade off. A row lying far out along an arrow scores high on that column.</p>
 <h4>Cautions</h4><p>Components are defined only up to sign: flipping all the weights of one component changes nothing. They are forced to be at right angles, which real underlying causes need not be, so the interpretation of a component is a suggestion and not a discovery. And a component with a small eigenvalue is not necessarily unimportant for a later prediction task: PCA never looked at any target.</p>`,
    q: 'A table has 10 columns and the first two components carry 95% of the spread. What does that tell you about the 10 columns?',
    a: 'That they are highly redundant. The rows lie very close to a flat two-dimensional sheet inside the ten-dimensional space, so the ten measurements are close to being ten different mixtures of only two underlying quantities plus a little noise. You could store two numbers per row and rebuild all ten columns with an error of 5% of the total variance.',
    init(root) {
      const chh = $('#pt-h', root),
        cs = $('#pt-s', root),
        cb = $('#pt-b', root),
        cl = $('#pt-l', root),
        live = $('#pt-live', root),
        st = $('#pt-st', root),
        S = { ds: 'exam', std: true, k: 1 };
      let names = [],
        X = [],
        n = 0,
        d = 0,
        mu = [],
        sd = [],
        Z = [],
        Cm = [],
        E = null,
        tot = 1,
        sc = [],
        ph = 0,
        msg = '',
        cu = -1,
        wait = 0,
        fr = 0,
        pb,
        custom = null;
      const sim = (kind) => {
        const r = rng(kind === 'exam' ? 21 : 34),
          rn = randnFrom(r),
          rows = [];
        for (let i = 0; i < 220; i++) {
          if (kind === 'exam') {
            const gq = rn(),
              q = rn(),
              w = rn(),
              cl2 = (x) => clamp(Math.round(x), 0, 100);
            rows.push([
              cl2(62 + 9 * (0.6 * gq + 0.7 * q) + 5 * rn()),
              cl2(58 + 11 * (0.6 * gq + 0.75 * q) + 5 * rn()),
              cl2(60 + 10 * (0.55 * gq + 0.65 * q) + 6 * rn()),
              cl2(65 + 8 * (0.6 * gq + 0.7 * w) + 5 * rn()),
              cl2(63 + 9 * (0.55 * gq + 0.7 * w) + 6 * rn()),
              cl2(66 + 8 * (0.5 * gq + 0.75 * w) + 5 * rn()),
            ]);
          } else {
            const s = rn(),
              b = rn(),
              r1 = (x) => Math.round(x * 10) / 10;
            rows.push([
              r1(170 + 9 * s + 2 * rn()),
              r1(171 + 9.5 * s + 2.5 * rn()),
              r1(80 + 5 * s + 1.5 * rn()),
              r1(70 + 7 * s + 9 * b + 3 * rn()),
              r1(82 + 3 * s + 9 * b + 3 * rn()),
              r1(42 + 1.8 * s + 0.6 * rn()),
            ]);
          }
        }
        return {
          names:
            kind === 'exam'
              ? ['Algebra', 'Calculus', 'Physics', 'Literature', 'History', 'Languages']
              : ['Height cm', 'Arm span cm', 'Leg cm', 'Weight kg', 'Waist cm', 'Shoe size'],
          X: rows,
        };
      };
      const parse = (text) => {
        const lines = String(text)
          .split(/\r?\n/)
          .filter((l) => l.trim())
          .map((l) => l.split(/[,;\t]/).map((v) => v.trim().replace(/^"|"$/g, '')));
        if (lines.length < 11)
          return 'This file has fewer than 10 data rows. PCA needs more rows than that to say anything. Add rows and upload again.';
        const isNum = (v) => v !== '' && isFinite(+v),
          head = lines[0].some((v) => !isNum(v)),
          body = head ? lines.slice(1) : lines,
          nc = Math.max(...lines.map((l) => l.length)),
          cols = [];
        for (let j = 0; j < nc && cols.length < 12; j++) {
          let ok = 0;
          for (const r of body) if (isNum(r[j] || '')) ok++;
          if (ok / body.length > 0.9) cols.push(j);
        }
        if (cols.length < 3)
          return 'Fewer than three numeric columns were found. PCA on a table needs at least three. Check that the file is comma-separated and upload again.';
        const rows = [];
        for (const r of body) {
          const v = cols.map((j) => +(r[j] || NaN));
          if (v.every(isFinite)) rows.push(v);
          if (rows.length >= 2000) break;
        }
        if (rows.length < 10)
          return 'Fewer than 10 complete rows were found after dropping rows with gaps. Fill the gaps and upload again.';
        return {
          names: cols.map((j, i) =>
            head && lines[0][j] ? lines[0][j].slice(0, 18) : 'column ' + (i + 1),
          ),
          X: rows,
          skipped: nc - cols.length,
        };
      };
      const comp = () => {
        const D = custom || sim(S.ds);
        names = D.names;
        X = D.X;
        n = X.length;
        d = names.length;
        mu = names.map((_, j) => X.reduce((s, r) => s + r[j], 0) / n);
        sd = names.map(
          (_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / (n - 1)) || 1,
        );
        Z = X.map((r) => r.map((v, j) => (v - mu[j]) / (S.std ? sd[j] : 1)));
        Cm = names.map((_, j) =>
          names.map((_, k) => {
            let s = 0;
            for (const r of Z) s += r[j] * r[k];
            return s / (n - 1);
          }),
        );
        E = jacobi(Cm);
        tot = E.val.reduce((a, b) => a + Math.max(0, b), 0) || 1;
        sc = Z.map((r) => [0, 1].map((k) => r.reduce((s, v, j) => s + v * E.vec[k][j], 0)));
        S.k = clamp(S.k, 1, d);
        kb.el.max = d;
        kb.set(S.k);
      };
      const nrev = () => clamp(ph - 2, 0, d),
        end = () => 3 + d,
        cum = (k) => E.val.slice(0, k).reduce((a, b) => a + Math.max(0, b), 0) / tot;
      const reset = () => {
        ph = 0;
        cu = -1;
        wait = 0;
        msg =
          'The table has ' +
          n +
          ' rows and ' +
          d +
          ' columns: ' +
          names.join(', ') +
          '. No column is singled out as the thing to predict. Press Run or One step.';
      };
      const top = (v, m) =>
        v
          .map((x, j) => [x, j])
          .sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]))
          .slice(0, m)
          .map(
            (q) => names[q[1]] + ' (' + (q[0] >= 0 ? '+' : '−') + Math.abs(q[0]).toFixed(2) + ')',
          )
          .join(', ');
      const say = () => {
        if (ph === 1) {
          cu = 0;
          const hi = sd.indexOf(Math.max(...sd)),
            lo = sd.indexOf(Math.min(...sd));
          msg = S.std
            ? 'Standardise: “' +
              names[hi] +
              '” had the largest standard deviation (' +
              uNum(sd[hi]) +
              ') and “' +
              names[lo] +
              '” the smallest (' +
              uNum(sd[lo]) +
              '). Each column has now had its mean subtracted and been divided by its standard deviation, so all of them have mean 0 and spread 1 and none dominates because of its units.'
            : 'Centre only: each column had its mean subtracted but keeps its own units. “' +
              names[hi] +
              '” has the largest standard deviation (' +
              uNum(sd[hi]) +
              ') against ' +
              uNum(sd[lo]) +
              ' for “' +
              names[lo] +
              '”, so it will dominate the first component.';
        } else if (ph === 2) {
          cu = 1;
          let bj = 0,
            bk = 1,
            bv = -1;
          for (let j = 0; j < d; j++)
            for (let k = j + 1; k < d; k++) {
              const r = Cm[j][k] / Math.sqrt(Cm[j][j] * Cm[k][k]);
              if (Math.abs(r) > bv) {
                bv = Math.abs(r);
                bj = j;
                bk = k;
              }
            }
          msg =
            (S.std ? 'Correlation' : 'Covariance') +
            ' matrix: one number for each of the ' +
            (d * (d - 1)) / 2 +
            ' pairs of columns. The most strongly related pair is “' +
            names[bj] +
            '” and “' +
            names[bk] +
            '”, with correlation ' +
            f2(Cm[bj][bk] / Math.sqrt(Cm[bj][bj] * Cm[bk][bk])) +
            '. Blocks of strong colour are groups of columns that move together.';
        } else if (ph < end()) {
          const k = ph - 2;
          cu = k === d ? 3 : 2;
          msg =
            'Component ' +
            k +
            ': the direction of greatest spread' +
            (k > 1 ? ' at right angles to the ' + (k - 1) + ' before it' : '') +
            '. Its largest weights are on ' +
            top(E.vec[k - 1], 3) +
            '. It carries ' +
            ((100 * Math.max(0, E.val[k - 1])) / tot).toFixed(1) +
            '% of the total spread, and the first ' +
            k +
            ' together carry ' +
            (100 * cum(k)).toFixed(1) +
            '%.';
          if (k <= d) {
            S.k = k;
            kb.set(k);
          }
        } else {
          cu = 4;
          msg =
            'Project: every row is replaced by its scores on the first two components. Two numbers per row now keep ' +
            (100 * cum(2)).toFixed(1) +
            '% of the spread of all ' +
            d +
            ' columns. Each dot on the lower left is one row of the table.';
        }
      };
      const step = () => {
        if (ph >= end()) {
          reset();
          return;
        }
        ph++;
        say();
      };
      const fin = () => {
        ph = end();
        say();
      };
      const draw = () => {
        const t = T(),
          short = (s) => (s.length > 10 ? s.slice(0, 9) + '…' : s);
        {
          const { g, w, h } = fit(chh);
          $('#pt-hcap', root).textContent =
            ph < 2
              ? ph === 0
                ? 'The first rows of the table, as given'
                : S.std
                  ? 'The same rows after standardising'
                  : 'The same rows after centring'
              : S.std
                ? 'Correlation matrix: green pairs rise together, red pairs trade off'
                : 'Covariance matrix: green pairs rise together, red pairs trade off';
          g.font = MONO;
          if (ph < 2) {
            const nc = Math.min(d, 7),
              cw = (w - 16) / nc,
              rows = Math.min(n, Math.floor((h - 40) / 20));
            g.textAlign = 'right';
            g.fillStyle = t.ink;
            for (let j = 0; j < nc; j++) g.fillText(short(names[j]), 8 + (j + 1) * cw - 6, 18);
            g.strokeStyle = t.line;
            g.beginPath();
            g.moveTo(8, 26);
            g.lineTo(w - 8, 26);
            g.stroke();
            g.fillStyle = t.muted;
            for (let i = 0; i < rows; i++)
              for (let j = 0; j < nc; j++) {
                const v = ph ? Z[i][j] : X[i][j];
                g.fillStyle = ph ? (v >= 0 ? t.accent : t.ok) : t.muted;
                g.fillText(
                  ph ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2) : uNum(v),
                  8 + (j + 1) * cw - 6,
                  44 + i * 20,
                );
              }
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText(
              '… ' + (n - rows) + ' more rows' + (d > nc ? ', ' + (d - nc) + ' more columns' : ''),
              8,
              h - 6,
            );
          } else {
            const lm = Math.min(86, w * 0.2),
              tm = 20,
              cz = Math.min((w - lm - 8) / d, (h - tm - 8) / d);
            for (let j = 0; j < d; j++) {
              g.fillStyle = t.muted;
              g.textAlign = 'right';
              g.fillText(short(names[j]), lm - 6, tm + (j + 0.5) * cz + 4);
              g.textAlign = 'center';
              g.fillText(names[j].slice(0, cz > 52 ? 7 : 3), lm + (j + 0.5) * cz, tm - 6);
              for (let k = 0; k < d; k++) {
                const r = Cm[j][k] / Math.sqrt(Cm[j][j] * Cm[k][k] || 1),
                  cl2 = r >= 0 ? t.accent : t.ok;
                g.fillStyle = mix(cl2, cl2, 0, Math.min(1, Math.abs(r)) * 0.9);
                g.fillRect(lm + k * cz + 1, tm + j * cz + 1, cz - 2, cz - 2);
                if (cz > 30) {
                  g.fillStyle = Math.abs(r) > 0.55 ? t.bg : t.ink;
                  g.fillText(r.toFixed(2), lm + (k + 0.5) * cz, tm + (j + 0.5) * cz + 4);
                }
              }
            }
          }
        }
        {
          const { g, w, h } = fit(cs),
            m = { l: 44, r: 14, t: 16, b: 26 },
            bw = (w - m.l - m.r) / d,
            Y = (v) => h - m.b - v * (h - m.t - m.b),
            nr = nrev();
          g.font = MONO;
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h - m.b);
          g.lineTo(w - m.r, h - m.b);
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          [0, 0.5, 1].forEach((v) => g.fillText(100 * v + '%', m.l - 5, Y(v) + 4));
          g.setLineDash([3, 4]);
          g.strokeStyle = t.line;
          g.beginPath();
          g.moveTo(m.l, Y(0.9));
          g.lineTo(w - m.r, Y(0.9));
          g.stroke();
          g.setLineDash([]);
          g.textAlign = 'right';
          g.fillText('90%', w - m.r, Y(0.9) - 4);
          for (let k = 0; k < d; k++) {
            const v = Math.max(0, E.val[k]) / tot,
              x = m.l + k * bw + bw * 0.18,
              bwid = bw * 0.64;
            g.textAlign = 'center';
            g.fillStyle = t.muted;
            g.fillText(String(k + 1), x + bwid / 2, h - m.b + 15);
            if (k < nr) {
              g.fillStyle = k + 1 === S.k ? t.gold : mix(t.gold, t.gold, 0, 0.55);
              g.fillRect(x, Y(v), bwid, Y(0) - Y(v));
              const tall = Y(0) - Y(v) > 26;
              g.fillStyle = tall ? t.bg : t.ink;
              g.fillText((100 * v).toFixed(0) + '%', x + bwid / 2, tall ? Y(v) + 15 : Y(v) - 5);
            } else {
              g.strokeStyle = t.line;
              g.setLineDash([3, 3]);
              g.strokeRect(x, Y(0.02), bwid, Y(0) - Y(0.02));
              g.setLineDash([]);
            }
          }
          if (nr) {
            g.strokeStyle = t.accent;
            g.lineWidth = 2.2;
            g.beginPath();
            for (let k = 0; k < nr; k++) {
              const x = m.l + (k + 0.5) * bw,
                y = Y(cum(k + 1));
              k ? g.lineTo(x, y) : g.moveTo(x, y);
            }
            g.stroke();
            g.fillStyle = t.accent;
            for (let k = 0; k < nr; k++) {
              g.beginPath();
              g.arc(m.l + (k + 0.5) * bw, Y(cum(k + 1)), 3.4, 0, 7);
              g.fill();
            }
          } else {
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText('no components found yet', w / 2, h / 2);
          }
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('component', w - m.r, h - 3);
        }
        {
          const { g, w, h } = fit(cb);
          g.font = MONO;
          if (nrev() < 2) {
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText('this picture needs the first two components', w / 2, h / 2);
          } else {
            let mx = 1e-9,
              my = 1e-9;
            for (const q of sc) {
              mx = Math.max(mx, Math.abs(q[0]));
              my = Math.max(my, Math.abs(q[1]));
            }
            const X2 = (v) => w / 2 + (v / mx) * (w / 2 - 30),
              Y2 = (v) => h / 2 - (v / my) * (h / 2 - 26);
            g.strokeStyle = t.line;
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(8, h / 2);
            g.lineTo(w - 8, h / 2);
            g.moveTo(w / 2, 8);
            g.lineTo(w / 2, h - 8);
            g.stroke();
            g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
            for (const q of sc) {
              g.beginPath();
              g.arc(X2(q[0]), Y2(q[1]), 2.6, 0, 7);
              g.fill();
            }
            let lm = 1e-9;
            const put = [];
            for (let j = 0; j < d; j++) lm = Math.max(lm, Math.hypot(E.vec[0][j], E.vec[1][j]));
            for (let j = 0; j < d; j++) {
              const ax = (E.vec[0][j] / lm) * 0.82,
                ay = (E.vec[1][j] / lm) * 0.82,
                x = w / 2 + ax * (w / 2 - 30),
                y = h / 2 - ay * (h / 2 - 26);
              g.strokeStyle = t.ok;
              g.lineWidth = 2;
              g.beginPath();
              g.moveTo(w / 2, h / 2);
              g.lineTo(x, y);
              g.stroke();
              g.fillStyle = t.ok;
              g.beginPath();
              g.arc(x, y, 3.5, 0, 7);
              g.fill();
              g.font = SANS;
              g.fillStyle = t.ink;
              g.textAlign = ax >= 0 ? 'left' : 'right';
              const lx = clamp(x + (ax >= 0 ? 7 : -7), 50, w - 50);
              let ly = clamp(y + (ay >= 0 ? -6 : 14), 14, h - 6);
              for (let q = 0; q < 8; q++) {
                if (put.some((p) => Math.abs(p[0] - lx) < 80 && Math.abs(p[1] - ly) < 14))
                  ly += ay >= 0 ? -14 : 14;
                else break;
              }
              put.push([lx, ly]);
              g.fillText(short(names[j]), lx, ly);
            }
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText('component 1 →', w - 8, h / 2 - 6);
            g.textAlign = 'left';
            g.fillText('↑ component 2', w / 2 + 6, 14);
          }
        }
        {
          const { g, w, h } = fit(cl),
            k = Math.min(S.k, Math.max(1, nrev()));
          $('#pt-lcap', root).textContent = 'Weights of component ' + k + ' on each column';
          g.font = MONO;
          if (!nrev()) {
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText('no components found yet', w / 2, h / 2);
          } else {
            const lm = Math.min(110, w * 0.26),
              rh = Math.min(34, (h - 30) / d),
              x0 = lm + (w - lm - 50) / 2,
              sx = (w - lm - 50) / 2;
            g.strokeStyle = t.muted;
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(x0, 8);
            g.lineTo(x0, 14 + d * rh);
            g.stroke();
            for (let j = 0; j < d; j++) {
              const v = E.vec[k - 1][j],
                y = 14 + j * rh;
              g.fillStyle = t.ink;
              g.textAlign = 'right';
              g.fillText(names[j].slice(0, 14), lm - 8, y + rh / 2 + 4);
              g.fillStyle = v >= 0 ? t.accent : t.ok;
              g.fillRect(v >= 0 ? x0 : x0 + v * sx, y + rh * 0.2, Math.abs(v) * sx, rh * 0.6);
              g.fillStyle = t.muted;
              g.textAlign = v >= 0 ? 'left' : 'right';
              g.fillText(
                (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2),
                x0 + v * sx + (v >= 0 ? 5 : -5),
                y + rh / 2 + 4,
              );
            }
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(
              'score = sum of weight × ' + (S.std ? 'standardised' : 'centred') + ' value',
              w / 2,
              h - 6,
            );
          }
        }
        live.textContent = msg;
        uCur(root, 'pt', cu);
        const nr = nrev();
        setR(root, 'n', String(n));
        setR(root, 'd', String(d));
        setR(root, 'p1', nr >= 1 ? (100 * cum(1)).toFixed(1) + '%' : '–');
        setR(root, 'p2', nr >= 2 ? (100 * cum(2)).toFixed(1) + '%' : '–');
        let k90 = 1;
        while (k90 < d && cum(k90) < 0.9) k90++;
        setR(root, 'k90', nr >= d ? k90 + ' of ' + d : '–');
      };
      const kb = bind(
        root,
        'pt-k',
        (v) => {
          S.k = clamp(v, 1, d || 1);
          pb && pb.stop();
          draw();
        },
        (v) => String(v),
      );
      comp();
      fin();
      const anim = Anim(() => {
        fr++;
        if (wait > 0) {
          wait--;
          if (!wait) {
            reset();
            draw();
          }
          return;
        }
        if (fr % 64) return;
        step();
        draw();
        chk();
      });
      pb = playBtn($('#pt-play', root), anim);
      const chk = runCtl(pb, root, 'pt', { done: () => ph >= end(), restart: () => reset() });
      $('#pt-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#pt-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      $('#pt-std', root).onchange = (e) => {
        S.std = e.target.checked;
        pb.stop();
        comp();
        fin();
        draw();
      };
      $('#pt-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        custom = null;
        st.textContent = '';
        pb.stop();
        comp();
        fin();
        draw();
      };
      $('#pt-file', root).onchange = (e) => {
        const f = e.target.files[0];
        e.target.value = '';
        if (!f) return;
        const fr2 = new FileReader();
        fr2.onerror = () => {
          st.textContent =
            'That file could not be read. Save it again as plain CSV and upload it once more.';
        };
        fr2.onload = () => {
          const r = parse(fr2.result);
          if (typeof r === 'string') {
            st.textContent = r;
            return;
          }
          custom = r;
          pb.stop();
          comp();
          fin();
          st.textContent =
            'Loaded ' +
            f.name +
            ': ' +
            r.X.length +
            ' rows and ' +
            r.names.length +
            ' numeric columns.' +
            (r.skipped > 0
              ? ' ' +
                r.skipped +
                ' column' +
                (r.skipped === 1 ? ' was' : 's were') +
                ' skipped (text, mostly empty, or beyond the 12-column limit).'
              : '');
          draw();
        };
        fr2.readAsText(f);
      };
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'tsne',
    tab: 't-SNE',
    title: 't-SNE: a map that keeps neighbours together',
    lede: 'PCA can only cast a flat shadow. t-SNE gives up on preserving distances altogether and asks for less: points that are neighbours in the original space should be neighbours on the page. It starts from a random scatter and moves the points, a little at a time, until that is true.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The t-SNE map taking shape. Each mark is one tiny picture of a letter. Point at one to see it</figcaption><canvas class="short" id="ts-c" role="img" aria-label="t-SNE embedding of the letter pictures"></canvas></figure>
 <figure class="fig"><figcaption>Top: PCA of the same pictures. Bottom: mismatch between the map and the original neighbourhoods</figcaption><canvas class="sq" id="ts-r" role="img" aria-label="PCA projection for comparison and the KL divergence curve"></canvas></figure></div>
 ${uHow('ts', ['In the original space, turn distances into probabilities: for each point, which others are its neighbours? Perplexity sets about how many.', 'Drop all the points at random positions on the page.', 'On the page, compute the same kind of neighbour probabilities, using a curve with heavier tails.', 'Move every point: toward those it should be near but is not, away from those it sits near but should not.', 'Repeat steps 3 and 4 a few hundred times.'])}
 <div class="controls">${uBtns('ts', 'Run', 'Ten iterations', 'iterations')}
 ${grp('The model', slider('ts-perp', 'Perplexity (roughly: neighbours per point)', 2, 45, 1, 15) + slider('ts-lr', 'Learning rate', 5, 150, 1, 40) + `<label class="check"><input type="checkbox" id="ts-col" checked> Show the true letter (the algorithm never sees it)</label>`)}
 ${grp('The data', slider('ts-flip', 'Noise: chance that a pixel is flipped', 0, 0.25, 0.01, 0.06) + `<div class="row"><button class="btn" id="ts-new" type="button">New random sample</button></div>`)}</div>
 <dl class="read">${rd('it', 'Iterations')}${rd('kl', 'Mismatch (KL divergence)', 'hl')}${rd('nt', 'Neighbours kept by t-SNE', 'hl3')}${rd('np', 'Neighbours kept by PCA', 'hl2')}${rd('ph', 'Phase')}</dl>`,
    tryit:
      'Untick Show the True Letter, press Start over and run: the algorithm sees only grey dots, yet groups appear. Tick it again to check them. Then set perplexity to 2 and to 45 and compare the maps. Raise the pixel noise to 0.20: at what point do the groups dissolve? Run the same settings twice: the groups survive, but where they sit on the page does not. What does that say about reading distances between groups?',
    math: String.raw`<h4>Intuition</h4><p>Imagine seating guests at a wedding. You cannot reproduce on a floor plan every relationship among 150 people, but you can try to seat friends at the same table. You do not care how far table 3 is from table 9, only that each person sits near the people they are close to. t-SNE (t-distributed stochastic neighbour embedding) draws data this way. Here each "guest" is a 5 by 7 picture of a letter, a point with 35 coordinates, and the page has only two.</p>
 <h4>Step 1: neighbours in the original space</h4><p>For each point \(i\), put a Gaussian on it and ask how much of its probability falls on each other point \(j\):</p>$$p_{j\mid i}=\frac{\exp\!\big(-\lVert\mathbf x_i-\mathbf x_j\rVert^2/2\sigma_i^2\big)}{\sum_{k\ne i}\exp\!\big(-\lVert\mathbf x_i-\mathbf x_k\rVert^2/2\sigma_i^2\big)},\qquad p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n}.$$<p>Each point gets its own width \(\sigma_i\), chosen so that the distribution \(p_{\cdot\mid i}\) has a fixed <b>perplexity</b> \(2^{H(p_{\cdot\mid i})}\), where \(H\) is the entropy in bits. Perplexity is the effective number of neighbours: in a dense region \(\sigma_i\) comes out small, in a sparse one large, so every point ends up with about the same number of neighbours.</p>
 <h4>Step 3: neighbours on the page</h4><p>For the positions \(\mathbf y_i\) on the page the same idea is used with a Student-t curve with one degree of freedom in place of the Gaussian:</p>$$q_{ij}=\frac{\big(1+\lVert\mathbf y_i-\mathbf y_j\rVert^2\big)^{-1}}{\sum_{k\ne l}\big(1+\lVert\mathbf y_k-\mathbf y_l\rVert^2\big)^{-1}} .$$<p>The heavy tails are the important design choice. In 35 dimensions a point can have many neighbours all at a moderate distance. On a flat page there is not enough room for that. With heavy tails, a moderate distance in the original space is matched by a much larger distance on the page, which gives the groups room to move apart instead of piling up in the middle (the "crowding problem").</p>
 <h4>Step 4: what is minimised</h4><p>The positions are chosen to make \(Q\) resemble \(P\), measured by the Kullback–Leibler divergence</p>$$C=\mathrm{KL}(P\,\Vert\,Q)=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}},\qquad\frac{\partial C}{\partial\mathbf y_i}=4\sum_{j}(p_{ij}-q_{ij})\,\frac{\mathbf y_i-\mathbf y_j}{1+\lVert\mathbf y_i-\mathbf y_j\rVert^2}.$$<p>Read the gradient as forces. If \(p_{ij}&gt;q_{ij}\) the two points are neighbours that sit too far apart, and they attract. If \(p_{ij}&lt;q_{ij}\) they sit closer than they should, and they repel. Because the cost weights each term by \(p_{ij}\), putting true neighbours far apart is punished heavily, while putting strangers close costs little. t-SNE therefore protects local structure first.</p>
 <h4>Early exaggeration</h4><p>For the first iterations every \(p_{ij}\) is multiplied by a constant (4 on this page). The attraction between neighbours is then far stronger than the repulsion, so tight clumps form while the map is still small and the clumps can slide past each other into a good arrangement. Then the factor is removed and the map relaxes.</p>
 <h4>How to read a t-SNE map</h4><ul><li><b>Trust:</b> which points are grouped together.</li><li><b>Do not trust:</b> the distance between groups, the size of a group, or its position. These change from run to run and with perplexity.</li><li><b>Perplexity</b> between 5 and 50 is usual. Too small and the map shatters into fragments of a few points. Too large and distinct groups blur together. It must be smaller than the number of points.</li><li>The cost is not convex, so different random starts give different maps. Run it more than once.</li><li>There is no formula that places a new point on an existing map, unlike PCA.</li></ul>
 <h4>UMAP</h4><p>UMAP follows the same plan with different ingredients: a nearest-neighbour graph in place of the \(p_{ij}\), a different curve on the page, and a cost that is cheaper to optimise. It is much faster on large data and tends to keep somewhat more of the large-scale arrangement, but the same cautions about distances between groups apply.</p>`,
    q: 'On a t-SNE map, cluster A is twice as far from cluster B as from cluster C. Can you conclude that A is more similar to C than to B?',
    a: 'No. t-SNE is built to keep neighbours together and makes almost no attempt to preserve distances between points that are not neighbours. The heavy-tailed curve on the page deliberately stretches moderate distances by an amount that depends on local density, and the arrangement of groups changes between runs. To compare groups, go back to the original distances or use a method that preserves them, such as PCA.',
    init(root) {
      const c = $('#ts-c', root),
        cr = $('#ts-r', root),
        live = $('#ts-live', root),
        S = { perp: 15, lr: 40, flip: 0.06, col: true },
        EX = 90,
        MAXIT = 1000,
        GL = {
          A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
          E: ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
          H: ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
          O: ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
          T: ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
          X: ['#...#', '#...#', '.#.#.', '..#..', '.#.#.', '#...#', '#...#'],
        },
        CH = Object.keys(GL);
      let Xh = [],
        lab = [],
        N = 0,
        Pm = null,
        Y = null,
        vel = null,
        gn = null,
        it = 0,
        KL = [],
        pY = [],
        pShare = 0,
        nbH = [],
        seed = 1,
        run = 1,
        scl = 1,
        hover = -1,
        msg = '',
        cu = 1,
        wait = 0,
        fr = 0,
        pb,
        conv = false;
      const fin = () => conv || it >= MAXIT;
      const knn = (dist, k) => {
        const o = [];
        for (let i = 0; i < N; i++) {
          const a = [];
          for (let j = 0; j < N; j++) if (j !== i) a.push([dist(i, j), j]);
          a.sort((u, w) => u[0] - w[0]);
          o.push(a.slice(0, k).map((q) => q[1]));
        }
        return o;
      };
      const keep = (pos) => {
        const nb = knn((i, j) => (pos(i)[0] - pos(j)[0]) ** 2 + (pos(i)[1] - pos(j)[1]) ** 2, 10);
        let s = 0;
        for (let i = 0; i < N; i++) {
          const h = new Set(nbH[i]);
          for (const j of nb[i]) if (h.has(j)) s++;
        }
        return s / (N * 10);
      };
      const gen = () => {
        const r = rng(seed * 17 + 1),
          rn = randnFrom(r);
        Xh = [];
        lab = [];
        CH.forEach((ch, k) => {
          const bits = GL[ch].join('');
          for (let i = 0; i < 25; i++) {
            Xh.push(
              [...bits].map((b) => {
                let v = b === '#' ? 1 : 0;
                if (r() < S.flip) v = 1 - v;
                return v + 0.04 * rn();
              }),
            );
            lab.push(k);
          }
        });
        N = Xh.length;
        const d = 35,
          mu = new Array(d).fill(0);
        for (const x of Xh) for (let j = 0; j < d; j++) mu[j] += x[j] / N;
        const Cv = [];
        for (let j = 0; j < d; j++) {
          Cv.push(new Array(d).fill(0));
        }
        for (const x of Xh)
          for (let j = 0; j < d; j++) {
            const a = x[j] - mu[j];
            for (let k = j; k < d; k++) Cv[j][k] += (a * (x[k] - mu[k])) / (N - 1);
          }
        for (let j = 0; j < d; j++) for (let k = 0; k < j; k++) Cv[j][k] = Cv[k][j];
        const E = jacobi(Cv);
        pY = Xh.map((x) =>
          [0, 1].map((k) => x.reduce((s, v, j) => s + (v - mu[j]) * E.vec[k][j], 0)),
        );
        pShare = (E.val[0] + E.val[1]) / E.val.reduce((a, b) => a + Math.max(0, b), 0);
        nbH = knn((i, j) => {
          let s = 0;
          for (let k = 0; k < d; k++) s += (Xh[i][k] - Xh[j][k]) ** 2;
          return s;
        }, 10);
      };
      const calcP = () => {
        const n = N,
          D = new Float64Array(n * n);
        for (let i = 0; i < n; i++)
          for (let j = i + 1; j < n; j++) {
            let s = 0;
            for (let k = 0; k < 35; k++) s += (Xh[i][k] - Xh[j][k]) ** 2;
            D[i * n + j] = D[j * n + i] = s;
          }
        const C = new Float64Array(n * n),
          H = Math.log(Math.min(S.perp, (n - 1) / 3));
        for (let i = 0; i < n; i++) {
          let lo = 0,
            hi = Infinity,
            beta = 1,
            sum = 1;
          for (let tr = 0; tr < 60; tr++) {
            sum = 0;
            let sp = 0;
            for (let j = 0; j < n; j++) {
              if (j === i) continue;
              const p = Math.exp(-D[i * n + j] * beta);
              C[i * n + j] = p;
              sum += p;
              sp += D[i * n + j] * p;
            }
            sum = sum || 1e-300;
            const Hc = Math.log(sum) + (beta * sp) / sum;
            if (Math.abs(Hc - H) < 1e-5) break;
            if (Hc > H) {
              lo = beta;
              beta = hi === Infinity ? beta * 2 : (beta + hi) / 2;
            } else {
              hi = beta;
              beta = (beta + lo) / 2;
            }
          }
          for (let j = 0; j < n; j++) C[i * n + j] = j === i ? 0 : C[i * n + j] / sum;
        }
        Pm = new Float64Array(n * n);
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++)
            Pm[i * n + j] = Math.max((C[i * n + j] + C[j * n + i]) / (2 * n), 1e-12);
      };
      const reset = () => {
        const r = randnFrom(rng(900 + seed * 3 + run * 7));
        Y = new Float64Array(2 * N);
        for (let i = 0; i < 2 * N; i++) Y[i] = r() * 0.01;
        vel = new Float64Array(2 * N);
        gn = new Float64Array(2 * N).fill(1);
        it = 0;
        KL = [];
        scl = 0.03;
        cu = 1;
        wait = 0;
        conv = false;
        msg =
          'Steps 1 and 2 are done. Every picture now has about ' +
          S.perp +
          ' neighbours in the original 35-dimensional space, and all ' +
          N +
          ' points have been dropped in a tiny random blob. Press Run or Ten iterations.';
      };
      const num = new Float64Array(150 * 150);
      const iter = () => {
        const n = N,
          ex = it < EX ? 4 : 1,
          mom = it < EX ? 0.5 : 0.8;
        let Zs = 0;
        for (let i = 0; i < n; i++)
          for (let j = i + 1; j < n; j++) {
            const dx = Y[2 * i] - Y[2 * j],
              dy = Y[2 * i + 1] - Y[2 * j + 1],
              q = 1 / (1 + dx * dx + dy * dy);
            num[i * n + j] = num[j * n + i] = q;
            Zs += 2 * q;
          }
        let kl = 0;
        const G = new Float64Array(2 * n);
        for (let i = 0; i < n; i++) {
          let gx = 0,
            gy = 0;
          for (let j = 0; j < n; j++) {
            if (j === i) continue;
            const q = num[i * n + j],
              Qv = Math.max(q / Zs, 1e-12),
              p = Pm[i * n + j],
              mlt = 4 * (ex * p - Qv) * q;
            gx += mlt * (Y[2 * i] - Y[2 * j]);
            gy += mlt * (Y[2 * i + 1] - Y[2 * j + 1]);
            kl += p * Math.log(p / Qv);
          }
          G[2 * i] = gx;
          G[2 * i + 1] = gy;
        }
        let mx = 0,
          my = 0;
        for (let k = 0; k < 2 * n; k++) {
          gn[k] = G[k] > 0 !== vel[k] > 0 ? gn[k] + 0.2 : Math.max(0.01, gn[k] * 0.8);
          vel[k] = mom * vel[k] - S.lr * gn[k] * G[k];
          Y[k] += vel[k];
          if (k % 2) my += Y[k];
          else mx += Y[k];
        }
        mx /= n;
        my /= n;
        for (let i = 0; i < n; i++) {
          Y[2 * i] -= mx;
          Y[2 * i + 1] -= my;
        }
        it++;
        KL.push(kl);
        if (it > EX + 80 && Math.abs(KL[it - 1] - KL[it - 41]) < 2e-3 * Math.max(1e-9, KL[it - 1]))
          conv = true;
      };
      const say = () => {
        cu = fin() ? 4 : 3;
        msg =
          'Iteration ' +
          it +
          '. ' +
          (it <= EX
            ? 'Early exaggeration: the pull between neighbours is multiplied by 4, so tight clumps form quickly while the map is still small enough for them to slide past one another.'
            : !fin()
              ? 'Fine-tuning: each point is pulled toward its original neighbours and pushed away from anything that sits too close on the page without being a neighbour.'
              : (conv
                  ? 'Converged after ' +
                    it +
                    ' iterations: the mismatch has stopped falling, so the points have stopped moving.'
                  : 'Reached the limit of ' + MAXIT + ' iterations.') +
                ' The groups are real, but their positions and the gaps between them are not meaningful.') +
          ' Mismatch (KL divergence): ' +
          f3(KL[KL.length - 1]) +
          '.';
      };
      const step = (k) => {
        if (fin()) {
          run++;
          reset();
          return;
        }
        for (let i = 0; i < k && !fin(); i++) iter();
        say();
      };
      const mark = (g, x, y, i, t, col, big) => {
        if (S.col) {
          g.font = (big ? '600 13px' : '600 11px') + ' "Hanken Grotesk",sans-serif';
          g.fillStyle = col[lab[i] % 8];
          g.textAlign = 'center';
          g.fillText(CH[lab[i]], x, y + 4);
        } else {
          g.fillStyle = mix(t.ink, t.ink, 0, 0.6);
          g.beginPath();
          g.arc(x, y, big ? 3.2 : 2.4, 0, 7);
          g.fill();
        }
        if (i === hover) {
          g.strokeStyle = t.ink;
          g.lineWidth = 1.8;
          g.beginPath();
          g.arc(x, y, 9, 0, 7);
          g.stroke();
        }
      };
      let PX = [];
      const draw = () => {
        const t = T(),
          col = UC(t);
        {
          const { g, w, h } = fit(c);
          let m = 1e-9;
          for (let k = 0; k < 2 * N; k++) m = Math.max(m, Math.abs(Y[k]));
          scl += (m * 1.12 - scl) * (pb && pb.on ? 0.25 : 1);
          const s = (Math.min(w, h) / 2 / scl) * 0.94;
          PX = [];
          for (let i = 0; i < N; i++) {
            const x = w / 2 + Y[2 * i] * s,
              y = h / 2 - Y[2 * i + 1] * s;
            PX.push([x, y]);
            mark(g, x, y, i, t, col, true);
          }
          if (hover >= 0) {
            const cz = 8,
              x0 = 10,
              y0 = 10;
            g.fillStyle = t.bg;
            g.fillRect(x0 - 4, y0 - 4, 5 * cz + 8, 7 * cz + 8);
            g.strokeStyle = t.muted;
            g.lineWidth = 1;
            g.strokeRect(x0 - 4, y0 - 4, 5 * cz + 8, 7 * cz + 8);
            for (let k = 0; k < 35; k++) {
              const v = clamp(Xh[hover][k], 0, 1);
              g.fillStyle = mix(t.bg, t.ink, v);
              g.fillRect(x0 + (k % 5) * cz, y0 + Math.floor(k / 5) * cz, cz - 1, cz - 1);
            }
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText('the picture behind this point', x0 + 5 * cz + 12, y0 + 12);
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('iteration ' + it, w - 8, h - 8);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.56;
          let mx = 1e-9,
            my = 1e-9;
          for (const q of pY) {
            mx = Math.max(mx, Math.abs(q[0]));
            my = Math.max(my, Math.abs(q[1]));
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            'PCA: the best flat shadow keeps ' + (100 * pShare).toFixed(0) + '% of the spread',
            8,
            13,
          );
          for (let i = 0; i < N; i++)
            mark(
              g,
              w / 2 + (pY[i][0] / mx) * (w / 2 - 18),
              22 + (h1 - 30) / 2 - (pY[i][1] / my) * ((h1 - 30) / 2 - 8),
              i,
              t,
              col,
              false,
            );
          uLine(g, t, 0, h1, w, h - h1, KL, {
            title: 'mismatch between map and original neighbourhoods, by iteration',
            n: MAXIT,
            ym: Math.max(0.5, ...KL),
            col: t.ok,
          });
        }
        live.textContent = msg;
        uCur(root, 'ts', cu);
        setR(root, 'it', String(it));
        setR(root, 'kl', KL.length ? f3(KL[KL.length - 1]) : '–');
        setR(root, 'nt', it ? pct(keep((i) => [Y[2 * i], Y[2 * i + 1]])) : '–');
        setR(root, 'np', pct(keep((i) => pY[i])));
        setR(
          root,
          'ph',
          it === 0
            ? 'Random start'
            : it <= EX
              ? 'Early exaggeration'
              : !fin()
                ? 'Fine-tuning'
                : conv
                  ? 'Converged'
                  : 'Stopped at the limit',
        );
      };
      gen();
      calcP();
      reset();
      while (!fin()) iter();
      say();
      scl = 1e-9;
      const anim = Anim(() => {
        fr++;
        if (it < 12 && fr % 6) return;
        for (let i = 0; i < (it < 40 ? 1 : 3) && !fin(); i++) iter();
        say();
        draw();
        chk();
      });
      pb = playBtn($('#ts-play', root), anim);
      const chk = runCtl(pb, root, 'ts', {
        done: fin,
        restart: () => {
          run++;
          reset();
        },
        count: () => it,
        what: 'iterations',
      });
      $('#ts-step', root).onclick = () => {
        pb.stop();
        step(10);
        draw();
      };
      $('#ts-reset', root).onclick = () => {
        pb.stop();
        run++;
        reset();
        draw();
      };
      bind(
        root,
        'ts-perp',
        (v) => {
          S.perp = v;
          pb.stop();
          calcP();
          reset();
          draw();
        },
        (v) => String(v),
      );
      bind(
        root,
        'ts-lr',
        (v) => {
          S.lr = v;
        },
        (v) => String(v),
      );
      $('#ts-col', root).onchange = (e) => {
        S.col = e.target.checked;
        draw();
      };
      bind(root, 'ts-flip', (v) => {
        S.flip = v;
        pb.stop();
        gen();
        calcP();
        reset();
        draw();
      });
      $('#ts-new', root).onclick = () => {
        seed++;
        pb.stop();
        gen();
        calcP();
        reset();
        draw();
      };
      c.addEventListener('pointermove', (e) => {
        const r = c.getBoundingClientRect(),
          x = e.clientX - r.left,
          y = e.clientY - r.top;
        let b = -1,
          bd = 196;
        PX.forEach((q, i) => {
          const d = (q[0] - x) ** 2 + (q[1] - y) ** 2;
          if (d < bd) {
            bd = d;
            b = i;
          }
        });
        if (b !== hover) {
          hover = b;
          if (!pb.on) draw();
        }
      });
      c.addEventListener('pointerleave', () => {
        if (hover >= 0) {
          hover = -1;
          if (!pb.on) draw();
        }
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },
  {
    id: 'umap',
    tab: 'UMAP',
    title: 'UMAP: draw the neighbour graph, then lay it out',
    lede: 'UMAP has the same aim as t-SNE: a flat map on which neighbours stay neighbours. It gets there by a different route. First it joins every point to its nearest neighbours to make a graph. Then it lays that graph out on the page with springs: joined points pull together, and random pairs push apart.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The UMAP map taking shape. Each dot is one tiny picture of a letter, coloured by which letter it is. The faint lines are the strongest links of the neighbour graph. Point at a dot to see its picture</figcaption><canvas class="short" id="um-c" role="img" aria-label="UMAP embedding of the letter pictures"></canvas></figure>
 <figure class="fig"><figcaption>Top: how similar two points count as, by their distance on the page. Bottom: share of true neighbours kept, by epoch</figcaption><canvas class="sq" id="um-r" role="img" aria-label="Similarity curve on the page and neighbours kept by epoch"></canvas></figure></div>
 ${uHow('um', ['For each point, find its k nearest neighbours in the original space.', 'Turn that into a weighted graph: the nearest neighbour gets weight 1, farther ones less, and each point uses its own scale.', 'Drop all the points on the page.', 'One epoch: pull together the two ends of each link, and for each pull push a few randomly chosen points away.', 'Repeat step 4 with a step size that shrinks to zero. When it reaches zero the layout stops.'])}
 <div class="controls">${uBtns('um', 'Run', 'Ten epochs', 'epochs')}
 ${grp('The model', slider('um-k', 'Neighbours per point (n_neighbors)', 3, 40, 1, 15) + slider('um-md', 'Minimum distance on the page (min_dist)', 0, 0.9, 0.01, 0.6) + `<label class="check"><input type="checkbox" id="um-col" checked> Show the true letter (the algorithm never sees it)</label><label class="check"><input type="checkbox" id="um-ed" checked> Show the neighbour graph</label>`)}
 ${grp('The data', slider('um-flip', 'Noise: chance that a pixel is flipped', 0, 0.25, 0.01, 0.06) + `<div class="row"><button class="btn" id="um-new" type="button">New random sample</button></div>`)}</div>
 <dl class="read">${rd('ep', 'Epochs')}${rd('lr', 'Step size now')}${rd('ed', 'Links in the neighbour graph')}${rd('nk', 'Neighbours kept', 'hl')}${rd('ab', 'Curve constants a, b')}${rd('ph', 'Status')}</dl>`,
    tryit:
      'Press Start over and use Ten epochs a few times: the graph is fixed from the start, and only the layout changes. Set the neighbours to 3 and run again: the map breaks into small pieces. Set it to 40: the groups hold together and start to merge. Then move the minimum distance from 0 to 0.9 and watch how tightly each group is packed. Notice how small the groups are next to the gaps between them: UMAP packs neighbours closely and leaves wide empty space, and the width of that space means nothing. Compare the finished map with the t-SNE tab for the same letters.',
    math: String.raw`<h4>Intuition</h4><p>Think of a class of students and who is friends with whom. You can draw that as a network: a line between every pair of friends. To make the drawing readable you would put friends near each other and keep people who are not friends from landing on top of one another. UMAP (uniform manifold approximation and projection) does this with data. The "friendships" are nearest neighbours in the original space, here the 35 pixels of each letter picture, and the drawing is the map.</p>
 <h4>Step 1 and 2: the weighted neighbour graph</h4><p>For each point \(i\) take its \(k\) nearest neighbours. Let \(\rho_i\) be the distance to the very nearest one. The link from \(i\) to a neighbour \(j\) gets the weight</p>$$w_{j\mid i}=\exp\!\Big(-\frac{\max(0,\;d(\mathbf x_i,\mathbf x_j)-\rho_i)}{\sigma_i}\Big),$$<p>so the nearest neighbour always has weight 1, and \(\sigma_i\) is chosen for each point so that \(\sum_j w_{j\mid i}=\log_2 k\). A point in a sparse region gets a large \(\sigma_i\) and a point in a crowd a small one, so every point is connected about equally strongly whatever the local density. The two directions are then combined into one undirected weight,</p>$$w_{ij}=w_{j\mid i}+w_{i\mid j}-w_{j\mid i}\,w_{i\mid j},$$<p>the probability that at least one of the two directed links is present.</p>
 <h4>Step 4: similarity on the page</h4><p>On the page, two points at distance \(d\) are given the similarity</p>$$q_{ij}=\frac{1}{1+a\,d^{\,2b}} .$$<p>The constants \(a\) and \(b\) are fitted so that this curve stays close to 1 up to the distance <b>min_dist</b> and then falls away, as in the top-right picture. With \(a=b=1\) it is exactly the heavy-tailed curve of t-SNE.</p>
 <h4>What is minimised</h4><p>UMAP makes the page similarities match the graph weights by minimising a cross-entropy over all pairs:</p>$$C=\sum_{i\ne j}\Big[w_{ij}\log\frac{w_{ij}}{q_{ij}}+(1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}\Big].$$<p>The first term pulls linked points together. The second pushes unlinked points apart; t-SNE has no separate term of this kind. Computing the second term for all pairs would be slow, so UMAP uses <b>negative sampling</b>: each time it pulls a link, it also picks a few points at random and pushes them away. A random point is almost never a neighbour, so this approximates the second term at a small fraction of the cost.</p>
 <h4>Step 5: when does it stop?</h4><p>The layout is found by stochastic gradient descent with a step size that falls in a straight line from its starting value to zero over a fixed number of epochs (300 here). UMAP does not test for convergence: it stops when the step size reaches zero. That is why the Status readout says "finished" and not "converged".</p>
 <h4>The two dials</h4><p><b>n_neighbors</b> sets the scale of structure that is kept. Small values look only at each point's closest few neighbours and give many small fragments. Large values tie distant points into the graph and show the broader arrangement, at the cost of fine detail. <b>min_dist</b> affects only the look of the map: near 0 the points of a group are packed into a tight clump, and larger values spread them out.</p>
 <h4>UMAP and t-SNE side by side</h4><ul><li>Both keep local neighbourhoods and both distort distances between far-apart groups. The cautions on the t-SNE tab apply here too.</li><li>UMAP works on a sparse graph of \(n\times k\) links instead of all \(n^2\) pairs, so it is much faster on large data.</li><li>UMAP usually starts from a spectral layout of the graph instead of a random scatter, which makes runs more alike. This page starts from a random scatter so that you can watch the map form.</li><li>A fitted UMAP can place new points on an existing map. Standard t-SNE cannot.</li></ul>`,
    q: 'With n_neighbors = 3 the map of the letters breaks into more than six pieces. The data did not change. What did?',
    a: 'The graph. With only three neighbours per point, small groups of nearly identical pictures are linked among themselves and have no links to the rest of their own letter. Nothing pulls those fragments together, and the random pushes drive them apart. The map is a faithful drawing of a graph that is too sparse to show the six letters as six connected groups.',
    init(root) {
      const c = $('#um-c', root),
        cr = $('#um-r', root),
        live = $('#um-live', root),
        S = { k: 15, md: 0.6, flip: 0.06, col: true, ed: true },
        NE = 300;
      let X = [],
        lab = [],
        LT = [],
        N = 0,
        D = null,
        nbH = [],
        E = [],
        Y = null,
        ep = 0,
        ab = [1.58, 0.9],
        hist = [],
        kept = NaN,
        seed = 1,
        run = 1,
        scl = 1,
        hover = -1,
        msg = '',
        cu = 2,
        fr = 0,
        pb,
        chk = () => false,
        PX = [];
      const gen = () => {
        const d = letterData(seed, S.flip);
        X = d.X;
        lab = d.lab;
        LT = d.CH;
        N = X.length;
        D = new Float64Array(N * N);
        for (let i = 0; i < N; i++)
          for (let j = i + 1; j < N; j++) {
            let s = 0;
            for (let k = 0; k < 35; k++) s += (X[i][k] - X[j][k]) ** 2;
            D[i * N + j] = D[j * N + i] = Math.sqrt(s);
          }
      };
      const graph = () => {
        const W = new Float64Array(N * N),
          tg = Math.log2(S.k);
        nbH = [];
        for (let i = 0; i < N; i++) {
          const a = [];
          for (let j = 0; j < N; j++) if (j !== i) a.push([D[i * N + j], j]);
          a.sort((u, v) => u[0] - v[0]);
          nbH.push(a.slice(0, 10).map((q) => q[1]));
          const nb = a.slice(0, S.k),
            rho = nb[0][0];
          let lo = 0,
            hi = Infinity,
            sg = 1;
          for (let tr = 0; tr < 60; tr++) {
            let s = 0;
            for (const q of nb) s += Math.exp(-Math.max(0, q[0] - rho) / sg);
            if (Math.abs(s - tg) < 1e-4) break;
            if (s > tg) {
              hi = sg;
              sg = (lo + hi) / 2;
            } else {
              lo = sg;
              sg = hi === Infinity ? sg * 2 : (lo + hi) / 2;
            }
          }
          for (const q of nb) W[i * N + q[1]] = Math.exp(-Math.max(0, q[0] - rho) / sg);
        }
        E = [];
        for (let i = 0; i < N; i++)
          for (let j = i + 1; j < N; j++) {
            const a = W[i * N + j],
              b = W[j * N + i],
              w = a + b - a * b;
            if (w > 1e-3) E.push([i, j, w]);
          }
      };
      const fitAB = () => {
        let best = 1e99;
        for (let a = 0.2; a <= 3.2; a += 0.04)
          for (let b = 0.5; b <= 2; b += 0.03) {
            let e = 0;
            for (let q = 1; q <= 60; q++) {
              const d = q * 0.05,
                tgt = d <= S.md ? 1 : Math.exp(-(d - S.md)),
                v = 1 / (1 + a * Math.pow(d, 2 * b));
              e += (v - tgt) ** 2;
            }
            if (e < best) {
              best = e;
              ab = [a, b];
            }
          }
      };
      const keep = () => {
        let s = 0;
        for (let i = 0; i < N; i++) {
          const a = [];
          for (let j = 0; j < N; j++)
            if (j !== i)
              a.push([(Y[2 * i] - Y[2 * j]) ** 2 + (Y[2 * i + 1] - Y[2 * j + 1]) ** 2, j]);
          a.sort((u, v) => u[0] - v[0]);
          const h = new Set(nbH[i]);
          for (let k = 0; k < 10; k++) if (h.has(a[k][1])) s++;
        }
        return s / (N * 10);
      };
      const reset = () => {
        const r = rng(700 + seed * 3 + run * 7);
        Y = new Float64Array(2 * N);
        for (let i = 0; i < 2 * N; i++) Y[i] = (r() - 0.5) * 8;
        ep = 0;
        hist = [];
        kept = keep();
        scl = 5;
        cu = 2;
        msg =
          'Steps 1 to 3 are done: the graph has ' +
          E.length +
          ' links among ' +
          N +
          ' points, and the points have been dropped at random. The links are fixed from here on. Press Run or Ten epochs.';
      };
      const cl = (v) => Math.max(-4, Math.min(4, v));
      const epoch = () => {
        const a = ab[0],
          b = ab[1],
          lr = 1 - ep / NE;
        for (const e of E) {
          if (Math.random() > e[2]) continue;
          for (let s = 0; s < 2; s++) {
            const i = s ? e[1] : e[0],
              j = s ? e[0] : e[1];
            let dx = Y[2 * i] - Y[2 * j],
              dy = Y[2 * i + 1] - Y[2 * j + 1],
              q = dx * dx + dy * dy;
            if (q > 1e-12) {
              const gc = (-2 * a * b * Math.pow(q, b - 1)) / (1 + a * Math.pow(q, b));
              Y[2 * i] += lr * cl(gc * dx);
              Y[2 * i + 1] += lr * cl(gc * dy);
            }
            for (let m = 0; m < 3; m++) {
              const k = Math.floor(Math.random() * N);
              if (k === i) continue;
              dx = Y[2 * i] - Y[2 * k];
              dy = Y[2 * i + 1] - Y[2 * k + 1];
              q = dx * dx + dy * dy;
              const gr = (2 * b) / ((0.001 + q) * (1 + a * Math.pow(q, b)));
              Y[2 * i] += lr * cl(gr * dx);
              Y[2 * i + 1] += lr * cl(gr * dy);
            }
          }
        }
        ep++;
        if (ep % 5 === 0 || ep >= NE) {
          kept = keep();
          hist.push(kept);
        }
      };
      const fin = () => ep >= NE;
      const say = () => {
        cu = fin() ? 4 : 3;
        msg =
          'Epoch ' +
          ep +
          ' of ' +
          NE +
          '. ' +
          (fin()
            ? 'The step size has reached zero, so the layout has stopped. UMAP has no convergence test: it stops because its steps are scheduled to shrink to nothing.'
            : 'Each link pulled its two ends together, and for every pull three randomly chosen points were pushed away. The step size is now ' +
              f2(1 - ep / NE) +
              ' of where it began.') +
          ' ' +
          pct(kept) +
          ' of each point’s ten true nearest neighbours are also its neighbours on the page.';
      };
      const step = (k) => {
        if (fin()) {
          run++;
          reset();
          return;
        }
        for (let i = 0; i < k && !fin(); i++) epoch();
        say();
      };
      const draw = () => {
        const t = T(),
          col = UC(t);
        {
          const { g, w, h } = fit(c);
          let m = 1e-9;
          for (let k = 0; k < 2 * N; k++) m = Math.max(m, Math.abs(Y[k]));
          scl += (m * 1.1 - scl) * (pb && pb.on ? 0.2 : 1);
          const s = (Math.min(w, h) / 2 / scl) * 0.94;
          PX = [];
          for (let i = 0; i < N; i++) PX.push([w / 2 + Y[2 * i] * s, h / 2 - Y[2 * i + 1] * s]);
          if (S.ed) {
            g.strokeStyle = mix(t.ink, t.ink, 0, 0.13);
            g.lineWidth = 1;
            g.beginPath();
            for (const e of E) {
              if (e[2] < 0.5) continue;
              g.moveTo(PX[e[0]][0], PX[e[0]][1]);
              g.lineTo(PX[e[1]][0], PX[e[1]][1]);
            }
            g.stroke();
          }
          for (let i = 0; i < N; i++) {
            const x = PX[i][0],
              y = PX[i][1];
            g.fillStyle = S.col ? col[lab[i] % 8] : mix(t.ink, t.ink, 0, 0.6);
            g.beginPath();
            g.arc(x, y, 3.4, 0, 7);
            g.fill();
            g.strokeStyle = t.bg;
            g.lineWidth = 0.8;
            g.stroke();
            if (i === hover) {
              g.strokeStyle = t.ink;
              g.lineWidth = 1.8;
              g.beginPath();
              g.arc(x, y, 9, 0, 7);
              g.stroke();
            }
          }
          if (S.col)
            LT.forEach((ch, k) => {
              let mx = 0,
                my = 1e9,
                n = 0;
              for (let i = 0; i < N; i++)
                if (lab[i] === k) {
                  mx += PX[i][0];
                  my = Math.min(my, PX[i][1]);
                  n++;
                }
              if (!n) return;
              g.font = '600 17px "Hanken Grotesk",sans-serif';
              g.fillStyle = col[k % 8];
              g.textAlign = 'center';
              g.fillText(ch, mx / n, Math.max(16, my - 9));
            });
          if (hover >= 0) {
            const cz = 8,
              x0 = 10,
              y0 = 10;
            g.fillStyle = t.bg;
            g.fillRect(x0 - 4, y0 - 4, 5 * cz + 8, 7 * cz + 8);
            g.strokeStyle = t.muted;
            g.lineWidth = 1;
            g.strokeRect(x0 - 4, y0 - 4, 5 * cz + 8, 7 * cz + 8);
            for (let k = 0; k < 35; k++) {
              g.fillStyle = mix(t.bg, t.ink, clamp(X[hover][k], 0, 1));
              g.fillRect(x0 + (k % 5) * cz, y0 + Math.floor(k / 5) * cz, cz - 1, cz - 1);
            }
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText('the picture behind this point', x0 + 5 * cz + 12, y0 + 12);
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('epoch ' + ep + ' of ' + NE, w - 8, h - 8);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.48,
            m = { l: 44, r: 12, t: 26, b: 22 },
            Xd = (d) => m.l + (d / 3) * (w - m.l - m.r),
            Yv = (v) => h1 - m.b - v * (h1 - m.t - m.b);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('similarity on the page against distance on the page', 8, 13);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h1 - m.b);
          g.lineTo(w - m.r, h1 - m.b);
          g.stroke();
          g.textAlign = 'right';
          g.fillText('1', m.l - 5, Yv(1) + 4);
          g.fillText('0', m.l - 5, Yv(0) + 4);
          g.textAlign = 'center';
          [0, 1, 2, 3].forEach((d) => g.fillText(String(d), Xd(d), h1 - m.b + 14));
          g.setLineDash([4, 4]);
          g.strokeStyle = t.muted;
          g.beginPath();
          for (let i = 0; i <= 120; i++) {
            const d = i / 40,
              v = d <= S.md ? 1 : Math.exp(-(d - S.md));
            i ? g.lineTo(Xd(d), Yv(v)) : g.moveTo(Xd(d), Yv(v));
          }
          g.stroke();
          g.setLineDash([]);
          g.strokeStyle = t.gold;
          g.lineWidth = 2.6;
          g.beginPath();
          for (let i = 0; i <= 120; i++) {
            const d = i / 40,
              v = 1 / (1 + ab[0] * Math.pow(d, 2 * ab[1]));
            i ? g.lineTo(Xd(d), Yv(v)) : g.moveTo(Xd(d), Yv(v));
          }
          g.stroke();
          g.strokeStyle = t.ok;
          g.lineWidth = 1.4;
          g.beginPath();
          g.moveTo(Xd(S.md), Yv(0));
          g.lineTo(Xd(S.md), Yv(1));
          g.stroke();
          g.fillStyle = t.ok;
          g.textAlign = 'left';
          g.fillText('min_dist = ' + f2(S.md), Xd(S.md) + 6, Yv(0.08));
          g.fillStyle = t.cut;
          g.textAlign = 'right';
          g.fillText('gold: the curve used; dashed: the target it was fitted to', w - m.r, m.t + 2);
          uLine(g, t, 0, h1, w, h - h1, hist, {
            title: 'share of true neighbours kept, measured every 5 epochs',
            n: NE / 5,
            ym: 1,
            col: t.accent,
            dots: hist.length < 30,
          });
        }
        live.textContent = msg;
        uCur(root, 'um', cu);
        setR(root, 'ep', ep + ' of ' + NE);
        setR(root, 'lr', f2(Math.max(0, 1 - ep / NE)));
        setR(root, 'ed', String(E.length));
        setR(root, 'nk', pct(kept));
        setR(root, 'ab', f2(ab[0]) + ', ' + f2(ab[1]));
        setR(
          root,
          'ph',
          ep === 0 ? 'Random start' : fin() ? 'Finished (step size is zero)' : 'Running',
        );
      };
      const all = () => {
        while (!fin()) epoch();
        say();
        scl = 1e-9;
      };
      gen();
      graph();
      fitAB();
      reset();
      all();
      const anim = Anim(() => {
        fr++;
        if (fr % (ep < 10 ? 8 : 2)) return;
        epoch();
        say();
        draw();
        chk();
      });
      pb = playBtn($('#um-play', root), anim);
      chk = runCtl(pb, root, 'um', {
        done: fin,
        restart: () => {
          run++;
          reset();
        },
        count: () => ep,
        what: 'epochs',
      });
      $('#um-step', root).onclick = () => {
        pb.stop();
        step(10);
        draw();
      };
      $('#um-reset', root).onclick = () => {
        pb.stop();
        run++;
        reset();
        draw();
      };
      bind(
        root,
        'um-k',
        (v) => {
          S.k = v;
          pb.stop();
          graph();
          reset();
          all();
          draw();
        },
        (v) => String(v),
      );
      bind(root, 'um-md', (v) => {
        S.md = v;
        pb.stop();
        fitAB();
        reset();
        all();
        draw();
      });
      $('#um-col', root).onchange = (e) => {
        S.col = e.target.checked;
        draw();
      };
      $('#um-ed', root).onchange = (e) => {
        S.ed = e.target.checked;
        draw();
      };
      bind(root, 'um-flip', (v) => {
        S.flip = v;
        pb.stop();
        gen();
        graph();
        reset();
        all();
        draw();
      });
      $('#um-new', root).onclick = () => {
        seed++;
        pb.stop();
        gen();
        graph();
        reset();
        all();
        draw();
      };
      c.addEventListener('pointermove', (e) => {
        const r = c.getBoundingClientRect(),
          x = e.clientX - r.left,
          y = e.clientY - r.top;
        let b = -1,
          bd = 196;
        PX.forEach((q, i) => {
          const d = (q[0] - x) ** 2 + (q[1] - y) ** 2;
          if (d < bd) {
            bd = d;
            b = i;
          }
        });
        if (b !== hover) {
          hover = b;
          if (!pb.on) draw();
        }
      });
      c.addEventListener('pointerleave', () => {
        if (hover >= 0) {
          hover = -1;
          if (!pb.on) draw();
        }
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          reset();
          draw();
          pb.start();
        },
      };
    },
  },
];
