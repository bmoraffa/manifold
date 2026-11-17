/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { MONO, clamp, clipPlot, prepareCanvas, hex, plot } from '../../js/core/shared.js';
import { clfLesson, sig } from '../classification/shared.js';
import { dot } from '../regression/shared.js';

function parseXgbCSV(text) {
  const lines = String(text)
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => l.split(/[,;\t]/).map((v) => v.trim().replace(/^"|"$/g, '').trim()));
  if (lines.length < 30)
    return 'This file has fewer than 30 data rows. XGBoost needs more than that to hold some out. Add rows and upload again.';
  const hasHead = lines[0].some((v) => v !== '' && !isFinite(Number(v))),
    head = hasHead ? lines[0] : lines[0].map((_, j) => 'column ' + (j + 1)),
    body = (hasHead ? lines.slice(1) : lines).slice(0, 4000),
    nc = head.length,
    keep = [];
  for (let j = 0; j < nc; j++) {
    let ok = 0;
    for (const rw of body) if (rw[j] !== undefined && rw[j] !== '' && isFinite(Number(rw[j]))) ok++;
    if (ok / body.length >= 0.9) keep.push(j);
  }
  if (keep.length < 2)
    return 'This file needs at least two numeric columns: one to predict and one or more to predict it from. Text columns are skipped.';
  const use = keep.slice(-13),
    rows = [];
  for (const rw of body) {
    if (use.some((j) => rw[j] === undefined || rw[j] === '')) continue;
    const v = use.map((j) => Number(rw[j]));
    if (v.every((x) => rw.length && isFinite(x) && x === x)) rows.push(v);
  }
  if (rows.length < 30)
    return 'Fewer than 30 rows have a number in every numeric column. Fill the gaps and upload again.';

  return { names: use.map((j) => head[j] || 'column ' + (j + 1)), rows, skipped: nc - use.length };
}

function xgbBuildTree(X, order, g, h, rows, { depth, lam, gam }) {
  const n = X.length,
    where = new Int32Array(n).fill(-1),
    all = [],
    stat = (nd) => {
      let G = 0,
        H = 0;
      for (const q of nd.rows) {
        G += g[q];
        H += h[q];
      }
      nd.G = G;
      nd.H = H;
      nd.n = nd.rows.length;
      nd.w = -G / (H + lam);
    },
    rt = { id: 0, rows };
  all.push(rt);
  for (const q of rows) where[q] = 0;
  stat(rt);
  let front = [rt];
  for (let d = 0; d < depth; d++) {
    const next = [];
    for (const nd of front) {
      if (nd.n < 4) continue;
      let best = null;
      const b0 = (nd.G * nd.G) / (nd.H + lam);
      for (let f = 0; f < X[0].length; f++) {
        let GL = 0,
          HL = 0,
          cnt = 0,
          prev = 0;
        const ord = order[f];
        for (let q = 0; q < ord.length; q++) {
          const r = ord[q];
          if (where[r] !== nd.id) continue;
          const v = X[r][f];
          if (cnt >= 2 && nd.n - cnt >= 2 && v !== prev) {
            const gain =
              0.5 * ((GL * GL) / (HL + lam) + ((nd.G - GL) * (nd.G - GL)) / (nd.H - HL + lam) - b0);
            if (!best || gain > best.gain) {
              const mid = (v + prev) / 2;
              // Keep adjacent values separated if the midpoint rounds up or overflows.
              best = { f, thr: mid >= prev && mid < v ? mid : prev, gain };
            }
          }
          GL += g[r];
          HL += h[r];
          cnt++;
          prev = v;
        }
      }
      if (!best) continue;
      const L = { id: all.length, rows: [] },
        R = { id: all.length + 1, rows: [] };
      all.push(L, R);
      for (const q of nd.rows) {
        if (X[q][best.f] <= best.thr) {
          L.rows.push(q);
          where[q] = L.id;
        } else {
          R.rows.push(q);
          where[q] = R.id;
        }
      }
      stat(L);
      stat(R);
      nd.sp = { f: best.f, thr: best.thr, gain: best.gain, L, R };
      next.push(L, R);
    }
    front = next;
  }
  let np = 0;
  const pr = (nd) => {
    if (!nd.sp) return;
    pr(nd.sp.L);
    pr(nd.sp.R);
    if (!nd.sp.L.sp && !nd.sp.R.sp && nd.sp.gain < gam) {
      nd.cut = nd.sp;
      nd.sp = null;
      np++;
    }
  };
  pr(rt);
  for (const nd of all) nd.rows = null;
  return { root: rt, np };
}

function growTree(P, o = {}) {
  const D = o.depth || 1,
    ML = o.minleaf || 1,
    W = (p) => (p.w == null ? 1 : p.w),
    gi = (n, n1) => (n > 0 ? 2 * (n1 / n) * (1 - n1 / n) : 0);
  const mk = (Q, dp) => {
    let n = 0,
      n1 = 0;
    for (const p of Q) {
      const w = W(p);
      n += w;
      if (p.c) n1 += w;
    }
    const nd = { n, n1, pb: n > 0 ? n1 / n : 0.5, sp: null };
    if (dp >= D || n1 <= 1e-12 || n - n1 <= 1e-12 || Q.length < 2 * ML) return nd;
    const I = gi(n, n1),
      fs = o.mtry === 1 ? [Math.random() < 0.5 ? 'x' : 'y'] : ['x', 'y'];
    let best = null;
    for (const f of fs) {
      const s = Q.slice().sort((a, b) => a[f] - b[f]);
      let ln = 0,
        l1 = 0;
      for (let i = 1; i < s.length; i++) {
        const w = W(s[i - 1]);
        ln += w;
        if (s[i - 1].c) l1 += w;
        if (s[i][f] === s[i - 1][f] || i < ML || s.length - i < ML) continue;
        const gn = I - (ln / n) * gi(ln, l1) - ((n - ln) / n) * gi(n - ln, n1 - l1);
        if (!best || gn > best.gn) best = { gn, f, thr: (s[i][f] + s[i - 1][f]) / 2, i, s };
      }
    }
    if (!best || (best.gn <= 1e-12 && !o.force)) return nd;
    if (o.imp) o.imp[best.f] += best.gn * n;
    nd.sp = {
      f: best.f,
      thr: best.thr,
      L: mk(best.s.slice(0, best.i), dp + 1),
      R: mk(best.s.slice(best.i), dp + 1),
    };
    return nd;
  };
  return mk(P, 0);
}

const treeP = (nd, x, y) => {
  while (nd.sp) nd = (nd.sp.f === 'x' ? x : y) <= nd.sp.thr ? nd.sp.L : nd.sp.R;
  return nd.pb;
};

function fitLogit(F, T, iters = 700) {
  const v = F(0, 0).map(() => 0);
  for (let k = 0; k < iters; k++) {
    const g = v.map(() => 0);
    for (const p of T) {
      const f = F(p.x, p.y),
        e = sig(dot(v, f)) - p.c;
      for (let j = 0; j < v.length; j++) g[j] += e * f[j];
    }
    for (let j = 0; j < v.length; j++) v[j] -= 0.6 * (g[j] / T.length + (j ? 0.002 * v[j] : 0));
  }
  return v;
}

function miniField(g, x0, y0, sz, pf, t, holder, NC = 26) {
  if (!holder.img || holder.key !== t.bg) {
    const o = document.createElement('canvas');
    o.width = o.height = NC;
    const fg = o.getContext('2d'),
      im = fg.createImageData(NC, NC),
      rgb = [hex(t.accent), hex(t.gold)];
    for (let i = 0; i < NC; i++)
      for (let j = 0; j < NC; j++) {
        const v = pf(-3 + (6 * (i + 0.5)) / NC, -3 + (6 * (j + 0.5)) / NC),
          k = v >= 0.5 ? 1 : 0,
          q = ((NC - 1 - j) * NC + i) * 4;
        im.data[q] = rgb[k][0];
        im.data[q + 1] = rgb[k][1];
        im.data[q + 2] = rgb[k][2];
        im.data[q + 3] = Math.round(255 * (0.18 + 0.62 * Math.abs(v - 0.5) * 2));
      }
    fg.putImageData(im, 0, 0);
    holder.img = o;
    holder.key = t.bg;
  }
  g.imageSmoothingEnabled = false;
  g.drawImage(holder.img, x0, y0, sz, sz);
  g.imageSmoothingEnabled = true;
  g.strokeStyle = t.muted;
  g.lineWidth = 1;
  g.strokeRect(x0, y0, sz, sz);
}

function errPlot(g, w, h, y0, t, n, series, xl) {
  g.save();
  g.translate(0, y0);
  let ym = 0.08;
  for (const s of series) for (const v of s[0]) if (isFinite(v)) ym = Math.max(ym, v);
  const P = plot(g, w, h - y0, [0, Math.max(8, n)], [0, ym * 1.15], t, {
    xstep: n > 60 ? 20 : n > 30 ? 10 : n > 12 ? 5 : 2,
    xlabel: xl,
  });
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'right';
  g.fillText((100 * ym).toFixed(0) + '%', P.m.l - 4, P.Y(ym) + 4);
  g.fillText('0', P.m.l - 4, P.Y(0) + 4);
  clipPlot(g, P);
  for (const s of series) {
    g.strokeStyle = s[1];
    g.lineWidth = 2.2;
    g.beginPath();
    let on = false;
    s[0].forEach((v, i) => {
      if (!isFinite(v)) {
        on = false;
        return;
      }
      on ? g.lineTo(P.X(i + 1), P.Y(v)) : g.moveTo(P.X(i + 1), P.Y(v));
      on = true;
    });
    g.stroke();
  }
  g.restore();
  g.textAlign = 'left';
  let lx = P.m.l + 8;
  for (const s of series) {
    g.fillStyle = s[1];
    g.fillText(s[2], lx, P.m.t + 6);
    lx += g.measureText(s[2]).width + 14;
  }
  g.restore();
}

const pct = (v) => (isFinite(v) ? (100 * v).toFixed(1) + '%' : '–');

function bagLesson(o) {
  const randomForest = !!o.randomForest;
  return clfLesson({
    ...o,
    fromZero: true,
    ds: ['moons', 'overlap', 'blobs', 'circles', 'xor'],
    stepBtn: true,
    hard: false,
    S: { B: 40, depth: 6, mtry: randomForest ? 1 : 2 },
    n: 140,
    nc: 46,
    setup(api) {
      const S = api.S;
      let trees = [],
        testVoteSums,
        oobVoteSums,
        oobVoteCounts,
        eT,
        eO,
        individualTreeAccuracies,
        imp,
        wait = 0,
        fr = 0,
        msg = '',
        fresh = true,
        treeAgreementRate = NaN;
      const reset = () => {
        trees = [];
        testVoteSums = api.test.map(() => 0);
        oobVoteSums = api.train.map(() => 0);
        oobVoteCounts = api.train.map(() => 0);
        eT = [];
        eO = [];
        individualTreeAccuracies = [];
        imp = { x: 0, y: 0 };
        wait = 0;
        treeAgreementRate = NaN;
        msg = 'No trees yet. Press Grow or One step.';
      };
      const add = () => {
        const tr = api.train,
          te = api.test,
          n = tr.length;
        if (n < 4) return;
        const inb = new Uint8Array(n),
          smp = [];
        for (let i = 0; i < n; i++) {
          const j = Math.floor(Math.random() * n);
          inb[j] = 1;
          smp.push(tr[j]);
        }
        const tree = growTree(smp, { depth: S.depth, minleaf: 1, mtry: S.mtry, imp });
        tree.h = {};
        trees.push(tree);
        let ok1 = 0;
        te.forEach((p, i) => {
          const v = treeP(tree, p.x, p.y);
          testVoteSums[i] += v;
          if ((v >= 0.5 ? 1 : 0) === p.c) ok1++;
        });
        individualTreeAccuracies.push(te.length ? ok1 / te.length : NaN);
        let d = 0;
        tr.forEach((p, i) => {
          if (!inb[i]) {
            oobVoteSums[i] += treeP(tree, p.x, p.y);
            oobVoteCounts[i]++;
          } else d++;
        });
        const k = trees.length;
        let ok = 0;
        te.forEach((p, i) => {
          if ((testVoteSums[i] / k >= 0.5 ? 1 : 0) === p.c) ok++;
        });
        eT.push(te.length ? 1 - ok / te.length : NaN);
        let oo = 0,
          oc = 0;
        tr.forEach((p, i) => {
          if (oobVoteCounts[i]) {
            oc++;
            if ((oobVoteSums[i] / oobVoteCounts[i] >= 0.5 ? 1 : 0) === p.c) oo++;
          }
        });
        eO.push(oc ? 1 - oo / oc : NaN);
        if (k > 1) {
          let a = 0,
            c = 0;
          const G = [];
          for (let i = 0; i < 60; i++) G.push([Math.random() * 5 - 2.5, Math.random() * 5 - 2.5]);
          const V = trees
            .slice(-12)
            .map((tq) => G.map((q) => (treeP(tq, q[0], q[1]) >= 0.5 ? 1 : 0)));
          for (let i = 0; i < V.length; i++)
            for (let j = i + 1; j < V.length; j++) {
              for (let q = 0; q < 60; q++) if (V[i][q] === V[j][q]) a++;
              c += 60;
            }
          treeAgreementRate = a / c;
        }
        msg =
          'Tree ' +
          k +
          ': trained on ' +
          n +
          ' points drawn with replacement from the ' +
          n +
          ' training points. ' +
          d +
          ' different points were drawn and ' +
          (n - d) +
          ' were left out. On its own this tree scores ' +
          pct(individualTreeAccuracies[k - 1]) +
          ' on the test set. The vote of all ' +
          k +
          (k === 1 ? ' tree' : ' trees') +
          ' scores ' +
          pct(1 - eT[k - 1]) +
          '.';
      };
      const rebuild = () => {
        reset();
        for (let i = 0; i < S.B; i++) add();
        api.redraw();
      };
      api.bind('B', 'B', (v) => String(v), rebuild);
      api.bind('depth', 'depth', (v) => String(v), rebuild);
      if (randomForest) api.bind('mtry', 'mtry', (v) => String(v), rebuild);
      return {
        reset,
        fit() {
          const k = fresh ? S.B : trees.length;
          fresh = false;
          reset();
          for (let i = 0; i < k; i++) add();
        },
        p: (x, y) => {
          if (!trees.length) return 0.5;
          let s = 0;
          for (const tq of trees) s += treeP(tq, x, y);
          return clamp(s / trees.length, 0.001, 0.999);
        },
        stepOnce() {
          if (trees.length >= S.B) reset();
          add();
        },
        done: () => wait > 0,
        tick() {
          fr++;
          if (wait > 0) return false;
          if (fr % (trees.length < 6 ? 55 : 20)) return false;
          if (trees.length < S.B) add();
          else wait = 240;
          return true;
        },
        status: () => msg,
        cur: () => (!trees.length ? 0 : trees.length < S.B ? 1 : 4),
        right(t) {
          const { g: ctx, w, h } = prepareCanvas(api.cr),
            cols = 4,
            sz = Math.min((w - 10) / cols - 10, (h * 0.6 - 34) / 2),
            k = trees.length,
            from = Math.max(0, k - 8);
          ctx.font = MONO;
          ctx.textAlign = 'left';
          for (let q = from; q < k; q++) {
            const i = q - from,
              x = 6 + (i % cols) * (sz + 10),
              y = 4 + Math.floor(i / cols) * (sz + 17);
            miniField(ctx, x, y, sz, (a, b) => treeP(trees[q], a, b), t, trees[q].h);
            ctx.fillStyle = t.muted;
            ctx.fillText(
              'tree ' + (q + 1) + ' · ' + pct(individualTreeAccuracies[q]),
              x,
              y + sz + 12,
            );
          }
          errPlot(
            ctx,
            w,
            h,
            h * 0.6,
            t,
            S.B,
            [
              [eT, t.ok, 'test error of the vote'],
              [eO, t.accent, 'out-of-bag error'],
            ],
            'trees',
          );
        },
        read(set) {
          const k = trees.length,
            av = k ? individualTreeAccuracies.reduce((a, b) => a + b, 0) / k : NaN;
          set('k', k + ' of ' + S.B);
          set('one', pct(av));
          set('oob', k ? pct(1 - eO[k - 1]) : '–');
          set('agree', pct(treeAgreementRate));
          const tot = imp.x + imp.y;
          if (randomForest)
            set(
              'imp',
              tot
                ? 'x ' +
                    ((100 * imp.x) / tot).toFixed(0) +
                    '%, y ' +
                    ((100 * imp.y) / tot).toFixed(0) +
                    '%'
                : '–',
            );
        },
      };
    },
  });
}

export { parseXgbCSV, xgbBuildTree, growTree, treeP, fitLogit, miniField, errPlot, pct, bagLesson };
