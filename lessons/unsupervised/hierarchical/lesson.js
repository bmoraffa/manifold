/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  f2,
  fit,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { UC, d2, uCur, uData, uDataBind, uFrame, uView } from '../shared.js';

export default {
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
};
