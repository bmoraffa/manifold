/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { canvasXY, dotAt, segBind, uCur, uLine, uView, xyUp } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { controlsFor } from '../../../js/ui/controls.js';

export default {
  init(root) {
    const c = $('#an-c', root),
      cr = $('#an-r', root),
      live = $('#an-live', root),
      S = { d: 'blob', m: 'iso', q: 6, k: 5 },
      GW = 48,
      GH = 36,
      PSI = 64,
      cF = (n) => (n <= 1 ? 0 : 2 * (Math.log(n - 1) + 0.5772) - (2 * (n - 1)) / n),
      CP = cF(PSI);
    let D = [],
      pl = [],
      trees = [],
      sumD = null,
      sumG = null,
      sc = [],
      fld = null,
      flag = [],
      tp = [2.6, 1.8],
      tpPaths = [],
      lastCuts = [],
      it = 0,
      fin = false,
      same = 0,
      prev = '',
      seed = 1,
      msg = '',
      cu = 0,
      pb,
      chk = () => false,
      V = null,
      mu = null,
      Si = null,
      auc = NaN,
      hasPl = true;
    const gen = () => {
      const r = rng(seed * 29 + S.d.length),
        rn = randnFrom(r);
      D = [];
      pl = [];
      hasPl = true;
      const n = 150;
      for (let i = 0; i < n; i++) {
        let p;
        if (S.d === 'blob') p = [0.9 * rn() + 0.55 * rn(), 0.7 * rn()];
        else if (S.d === 'two')
          p =
            i % 2 ? [-1.9 + 0.5 * rn(), -0.6 + 0.5 * rn()] : [1.7 + 0.6 * rn(), 0.7 + 0.45 * rn()];
        else {
          const a = Math.PI * (0.05 + 0.9 * r());
          p = [2.6 * Math.cos(a) + 0.17 * rn(), 2.6 * Math.sin(a) - 1.6 + 0.17 * rn()];
        }
        D.push(p);
        pl.push(0);
      }
      const A = {
        blob: [
          [3, 1.9],
          [-2.9, 1.6],
          [2.6, -2],
          [-3, -1.7],
          [0.2, 2.5],
          [-0.4, -2.4],
          [3.4, 0.2],
        ],
        two: [
          [0, 0.1],
          [-0.2, 1.9],
          [0.4, -2],
          [3.5, -1.6],
          [-3.5, 1.5],
          [-0.1, -0.7],
          [1.7, 2.6],
        ],
        arc: [
          [0, -0.6],
          [0.5, 0.1],
          [-0.7, -0.2],
          [3.4, 1.9],
          [-3.3, 2],
          [0, 2.6],
          [0.1, -1.4],
        ],
      }[S.d];
      for (const a of A) {
        D.push([a[0] + 0.1 * rn(), a[1] + 0.1 * rn()]);
        pl.push(1);
      }
    };
    const build = (idx, depth, r) => {
      if (idx.length <= 1 || depth >= 7) return { n: idx.length };
      const d = r() < 0.5 ? 0 : 1;
      let lo = 1e9,
        hi = -1e9;
      for (const i of idx) {
        lo = Math.min(lo, D[i][d]);
        hi = Math.max(hi, D[i][d]);
      }
      if (hi - lo < 1e-9) return { n: idx.length };
      const sp = lo + (hi - lo) * r();
      return {
        d,
        sp,
        l: build(
          idx.filter((i) => D[i][d] < sp),
          depth + 1,
          r,
        ),
        r: build(
          idx.filter((i) => D[i][d] >= sp),
          depth + 1,
          r,
        ),
      };
    };
    const path = (T_, p, cuts) => {
      let nd = T_,
        h = 0,
        bx = [-4, 4, -3, 3];
      while (nd.d !== undefined) {
        if (cuts) cuts.push({ d: nd.d, sp: nd.sp, b: bx.slice() });
        if (p[nd.d] < nd.sp) {
          bx[nd.d * 2 + 1] = nd.sp;
          nd = nd.l;
        } else {
          bx[nd.d * 2] = nd.sp;
          nd = nd.r;
        }
        h++;
      }
      return h + cF(nd.n);
    };
    const prep = () => {
      const n = D.length;
      mu = [0, 1].map((j) => D.reduce((s, p) => s + p[j], 0) / n);
      let a = 0,
        b = 0,
        d = 0;
      for (const p of D) {
        a += (p[0] - mu[0]) ** 2 / n;
        b += ((p[0] - mu[0]) * (p[1] - mu[1])) / n;
        d += (p[1] - mu[1]) ** 2 / n;
      }
      const dt = a * d - b * b;
      Si = [d / dt, -b / dt, a / dt];
    };
    const sGauss = (p) => {
      const x = p[0] - mu[0],
        y = p[1] - mu[1];
      return Math.sqrt(Si[0] * x * x + 2 * Si[1] * x * y + Si[2] * y * y);
    };
    const sKnn = (p, self) => {
      const ds = [];
      for (let i = 0; i < D.length; i++) {
        if (i === self) continue;
        ds.push(Math.hypot(p[0] - D[i][0], p[1] - D[i][1]));
      }
      ds.sort((x, y) => x - y);
      return ds[Math.min(S.k, ds.length) - 1];
    };
    const gridP = (k) => [
      -4 + (8 * ((k % GW) + 0.5)) / GW,
      3 - (6 * (Math.floor(k / GW) + 0.5)) / GH,
    ];
    const score = () => {
      const n = D.length;
      if (S.m === 'iso') {
        const T_ = Math.max(1, trees.length);
        sc = Array.from(sumD, (v) => (trees.length ? 2 ** (-(v / T_) / CP) : 0.5));
        fld = Array.from(sumG, (v) => (trees.length ? 2 ** (-(v / T_) / CP) : 0.5));
      } else {
        const f = S.m === 'gauss' ? (p) => sGauss(p) : (p, i) => sKnn(p, i);
        sc = D.map((p, i) => f(p, i));
        fld = Array.from({ length: GW * GH }, (_, k) => f(gridP(k), -1));
      }
      const ord = sc.map((_, i) => i).sort((a, b) => sc[b] - sc[a]),
        nf = Math.max(1, Math.round((n * S.q) / 100));
      flag = new Array(n).fill(false);
      ord.slice(0, nf).forEach((i) => (flag[i] = true));
      if (hasPl) {
        let pairs = 0,
          win = 0;
        for (let i = 0; i < n; i++)
          if (pl[i])
            for (let j = 0; j < n; j++)
              if (!pl[j]) {
                pairs++;
                win += sc[i] > sc[j] ? 1 : sc[i] === sc[j] ? 0.5 : 0;
              }
        auc = win / pairs;
      }
      return ord
        .slice(0, nf)
        .sort((a, b) => a - b)
        .join(',');
    };
    const reset = () => {
      prep();
      trees = [];
      sumD = new Float64Array(D.length);
      sumG = new Float64Array(GW * GH);
      tpPaths = [];
      lastCuts = [];
      it = 0;
      fin = S.m !== 'iso';
      same = 0;
      prev = score();
      cu = 0;
      msg =
        S.m === 'iso'
          ? 'An empty forest: every place scores the same. Press Add one tree, or Run.'
          : (S.m === 'knn'
              ? 'Each point is scored by the distance to its ' +
                S.k +
                (S.k === 1 ? 'st' : S.k === 2 ? 'nd' : S.k === 3 ? 'rd' : 'th') +
                ' nearest neighbour.'
              : 'One Gaussian is fitted to all the data; each point is scored by how many standard deviations it lies from the centre.') +
            ' This method needs no iterations, so there is nothing to run. ' +
            summary();
    };
    const summary = () => {
      if (!hasPl) return flag.filter(Boolean).length + ' points flagged.';
      const hit = flag.filter((f, i) => f && pl[i]).length,
        np = pl.reduce((s, v) => s + v, 0);
      return (
        'Caught ' +
        hit +
        ' of ' +
        np +
        ' planted anomalies, with ' +
        (flag.filter(Boolean).length - hit) +
        ' false alarms.'
      );
    };
    const step = () => {
      if (fin || S.m !== 'iso') return;
      const r = rng(seed * 1000 + trees.length * 7 + 3),
        idx = [];
      const all = D.map((_, i) => i);
      for (let i = 0; i < Math.min(PSI, all.length); i++) {
        const j = i + Math.floor(r() * (all.length - i));
        [all[i], all[j]] = [all[j], all[i]];
        idx.push(all[i]);
      }
      const T_ = build(idx, 0, r);
      trees.push(T_);
      for (let i = 0; i < D.length; i++) sumD[i] += path(T_, D[i]);
      for (let k = 0; k < GW * GH; k++) sumG[k] += path(T_, gridP(k));
      lastCuts = [];
      const h = path(T_, tp, lastCuts);
      tpPaths.push(lastCuts.length);
      it++;
      const key = score();
      same = key === prev ? same + 1 : 0;
      prev = key;
      cu = it < 3 ? it : 4;
      msg =
        'Tree ' +
        it +
        ': your point at (' +
        f2(tp[0]) +
        ', ' +
        f2(tp[1]) +
        ') was isolated after ' +
        lastCuts.length +
        ' cut' +
        (lastCuts.length === 1 ? '' : 's') +
        '. A typical point in a tree of ' +
        PSI +
        ' needs about ' +
        CP.toFixed(1) +
        '.';
      if ((it >= 40 && same >= 25) || it >= 200) {
        fin = true;
        cu = 5;
        msg =
          'Stopped at ' +
          it +
          ' trees: the flagged set has not changed for ' +
          same +
          ' trees. ' +
          summary();
      }
    };
    const draw = () => {
      const t = T();
      V = uView(c);
      {
        const { g, w, h, X, Y } = V,
          lo = Math.min(...fld),
          hi = Math.max(...fld);
        for (let k = 0; k < GW * GH; k++) {
          const q = k % GW,
            r = Math.floor(k / GW),
            v = (fld[k] - lo) / (hi - lo || 1);
          g.fillStyle = mix(t.bg, t.cut, v * 0.6);
          const xa = Math.round(X(-4 + (8 * q) / GW)),
            xb = Math.round(X(-4 + (8 * (q + 1)) / GW)),
            ya = Math.round(Y(3 - (6 * r) / GH)),
            yb = Math.round(Y(3 - (6 * (r + 1)) / GH));
          g.fillRect(xa, ya, xb - xa, yb - ya);
        }
        if (S.m === 'iso' && lastCuts.length) {
          g.strokeStyle = t.ink;
          g.lineWidth = 1.3;
          lastCuts.forEach((q, i) => {
            g.globalAlpha = 1 - 0.1 * i;
            g.beginPath();
            if (q.d === 0) {
              g.moveTo(X(q.sp), Y(q.b[2]));
              g.lineTo(X(q.sp), Y(q.b[3]));
            } else {
              g.moveTo(X(q.b[0]), Y(q.sp));
              g.lineTo(X(q.b[1]), Y(q.sp));
            }
            g.stroke();
          });
          g.globalAlpha = 1;
        }
        D.forEach((p, i) => {
          const x = X(p[0]),
            y = Y(p[1]);
          if (pl[i] && hasPl) {
            g.strokeStyle = t.ink;
            g.lineWidth = 2;
            g.beginPath();
            g.moveTo(x - 4, y - 4);
            g.lineTo(x + 4, y + 4);
            g.moveTo(x + 4, y - 4);
            g.lineTo(x - 4, y + 4);
            g.stroke();
          } else dotAt(g, x, y, 3, mix(t.accent, t.accent, 0, 0.85));
          if (flag[i]) {
            g.lineWidth = 2;
            dotAt(g, x, y, 8, null, t.cut);
          }
        });
        markTP(g, X(tp[0]), Y(tp[1]), t);
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.55,
          n = D.length,
          ord = sc.map((_, i) => i).sort((a, b) => sc[b] - sc[a]),
          lo = Math.min(...sc),
          hi = Math.max(...sc),
          A = uLine(g, t, 0, 0, w, h1, [], {
            title: 'scores of all ' + n + ' points, most unusual first',
            n: n + 1,
            ym: hi + (hi - lo) * 0.05,
            y1: lo - (hi - lo) * 0.02,
          }),
          bw = A.X(1) - A.X(0);
        ord.forEach((i, k) => {
          g.fillStyle = pl[i] && hasPl ? t.cut : mix(t.accent, t.accent, 0, 0.55);
          g.fillRect(
            A.X(k),
            A.Y(sc[i]),
            Math.max(1, bw - 0.6),
            A.Y(lo - (hi - lo) * 0.02) - A.Y(sc[i]),
          );
        });
        const nf = flag.filter(Boolean).length;
        g.strokeStyle = t.ink;
        g.lineWidth = 1.5;
        g.setLineDash([5, 4]);
        g.beginPath();
        g.moveTo(A.X(nf), A.Y(hi));
        g.lineTo(A.X(nf), A.Y(lo));
        g.stroke();
        g.setLineDash([]);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('← flagged', A.X(nf) + 5, A.Y(hi) + 10);
        if (S.m === 'iso') {
          const B = uLine(g, t, 0, h1, w, h - h1, tpPaths, {
            title: 'cuts needed to isolate your point, tree by tree',
            n: Math.max(30, tpPaths.length),
            ym: 8,
            y1: 0,
          });
          g.strokeStyle = t.muted;
          g.setLineDash([4, 4]);
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(B.X(0), B.Y(CP));
          g.lineTo(B.X(Math.max(30, tpPaths.length) - 1), B.Y(CP));
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('typical point', w - 14, B.Y(CP) - 4);
          tpPaths.forEach((v, i) => dotAt(g, B.X(i), B.Y(Math.min(8, v)), 2.5, t.gold));
        } else {
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('this method computes every score in one pass: no trees', 8, h1 + 16);
        }
      }
      const myS =
          S.m === 'iso'
            ? trees.length
              ? 2 ** (-(trees.reduce((s, T_) => s + path(T_, tp), 0) / trees.length) / CP)
              : 0.5
            : S.m === 'gauss'
              ? sGauss(tp)
              : sKnn(tp, -1),
        above = sc.filter((v) => v < myS).length;
      tpo.textContent =
        'Your point (' +
        f2(tp[0]) +
        ', ' +
        f2(tp[1]) +
        ') scores ' +
        f2(myS) +
        ': more unusual than ' +
        pct(above / sc.length) +
        ' of the data.' +
        (S.m === 'iso' && !trees.length ? ' (Add trees first.)' : '');
      const hit = flag.filter((f, i) => f && pl[i]).length,
        np = pl.reduce((s, v) => s + v, 0),
        nf = flag.filter(Boolean).length;
      live.textContent = msg;
      uCur(root, 'an', cu);
      setR(root, 'tr', S.m === 'iso' ? it + (fin ? ' (stopped)' : '') : 'not used');
      setR(root, 'fl', nf + ' of ' + D.length);
      setR(root, 'hit', hasPl ? hit + ' of ' + np : '–');
      setR(root, 'fa', hasPl ? String(nf - hit) : '–');
      setR(root, 'auc', hasPl && (S.m !== 'iso' || trees.length) ? f3(auc) : '–');
    };
    const { create } = controlsFor(root);
    root
      .querySelector('.controls')
      .append(create('group', { label: 'Your point' }, create('point-status', { id: 'an-tpo' })));
    const tpo = $('#an-tpo', root);
    gen();
    reset();
    if (RM) {
      while (!fin) step();
    }
    let fr = 0;
    const anim = Anim(() => {
      if (fin) {
        pb.stop();
        return;
      }
      if (it < 6 && ++fr % 20) return;
      step();
      draw();
      chk();
    });
    pb = playBtn($('#an-play', root), anim);
    chk = runCtl(pb, root, 'an', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'trees',
    });
    const go = () => {
      pb.stop();
      reset();
      draw();
      if (S.m === 'iso') {
        fr = 0;
        pb.start();
      }
    };
    $('#an-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#an-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    $('#an-new', root).onclick = () => {
      seed++;
      gen();
      go();
    };
    bind(
      root,
      'an-q',
      (v) => {
        S.q = v;
        prev = score();
        same = 0;
        draw();
      },
      (v) => +v + '%',
    );
    bind(
      root,
      'an-k',
      (v) => {
        S.k = v;
        if (S.m === 'knn') {
          reset();
          draw();
        }
      },
      (v) => String(+v),
    );
    segBind(root, 'an-d', S.d, (v) => {
      S.d = v;
      gen();
      go();
    });
    segBind(root, 'an-m', S.m, (v) => {
      S.m = v;
      go();
    });
    xyUp(root, 'an', (q) => {
      D = q.slice(0, 300);
      pl = D.map(() => 0);
      hasPl = false;
      go();
    });
    c.style.cursor = 'crosshair';
    c.addEventListener('click', (e) => {
      if (!V) return;
      const [x, y] = canvasXY(c, V, e);
      tp = [clamp(x, -4, 4), clamp(y, -3, 3)];
      tpPaths = trees.map((T_) => {
        const cs = [];
        path(T_, tp, cs);
        return cs.length;
      });
      lastCuts = [];
      if (trees.length) path(trees[trees.length - 1], tp, lastCuts);
      draw();
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        fr = 0;
        pb.start();
      },
    };
  },
};
