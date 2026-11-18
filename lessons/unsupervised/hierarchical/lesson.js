/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  bind,
  f2,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { UC, d2, highlightLessonStep, uData, uDataBind, uFrame, uView } from '../shared.js';

export default {
  init(root) {
    const c = $('#hc-c', root),
      cr = $('#hc-r', root),
      live = $('#hc-live', root),
      settings = { link: 'average', K: 3, ds: 'blobs', n: 36, seed: 4, cap: 70 };
    let P = [],
      merges = [],
      mem = {},
      pos = [],
      nodeXPositions = {},
      nodeHeights = {},
      shown = 0,
      cut = false,
      msg = '',
      activeStepIndex = 0,
      wait = 0,
      frameCount = 0,
      playback;
    const build = () => {
      const n = P.length,
        W = settings.link === 'ward',
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
      merges = [];
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
          if (settings.link === 'single') v = Math.min(a, b);
          else if (settings.link === 'complete') v = Math.max(a, b);
          else if (settings.link === 'average') v = (si * a + sj * b) / (si + sj);
          else v = ((si + sk) * a + (sj + sk) * b - sk * bd) / (si + sj + sk);
          D[k][id] = D[id][k] = v;
        }
        sz[id] = si + sj;
        mem[id] = mem[bi].concat(mem[bj]);
        act = act.filter((k) => k !== bi && k !== bj);
        act.push(id);
        merges.push({ a: bi, b: bj, id, h: W ? Math.sqrt(Math.max(0, bd)) : bd, na: si, nb: sj });
      }
      const ord = (id) => {
        if (id < n) return [id];
        const q = merges[id - n];
        return ord(q.a).concat(ord(q.b));
      };
      const o = n > 1 ? ord(2 * n - 2) : [0];
      pos = new Array(n);
      o.forEach((l, i) => {
        pos[l] = i;
      });
      nodeXPositions = {};
      nodeHeights = {};
      for (let i = 0; i < n; i++) {
        nodeXPositions[i] = pos[i];
        nodeHeights[i] = 0;
      }
      merges.forEach((q) => {
        nodeXPositions[q.id] = (nodeXPositions[q.a] + nodeXPositions[q.b]) / 2;
        nodeHeights[q.id] = q.h;
      });
    };
    const labels = (m) => {
      const n = P.length,
        L = new Array(n);
      for (let i = 0; i < n; i++) L[i] = i;
      for (let k = 0; k < m; k++) for (const l of mem[merges[k].id]) L[l] = merges[k].id;
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
      activeStepIndex = 0;
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
      if (cut || shown >= merges.length) {
        if (shown >= merges.length && !cut) {
          cut = true;
          activeStepIndex = 4;
          msg = cutMsg();
          return;
        }
        reset();
        return;
      }
      const q = merges[shown];
      shown++;
      activeStepIndex = 2;
      msg =
        'Merge ' +
        shown +
        ' of ' +
        merges.length +
        ': joined a cluster of ' +
        q.na +
        ' point' +
        (q.na === 1 ? '' : 's') +
        ' with a cluster of ' +
        q.nb +
        '. Their distance under ' +
        names[settings.link] +
        ' was ' +
        f2(q.h) +
        ', the smallest of all pairs. ' +
        (P.length - shown) +
        ' cluster' +
        (P.length - shown === 1 ? ' remains.' : 's remain.');
      if (shown >= merges.length) {
        activeStepIndex = 3;
      }
    };
    const lvl = () => (cut ? Math.max(0, P.length - Math.min(settings.K, P.length)) : shown);
    const cutMsg = () => {
      const k = Math.min(settings.K, P.length),
        m = P.length - k,
        lo = m > 0 ? merges[m - 1].h : 0,
        hi = m < merges.length ? merges[m].h : lo;
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
          g = merges[m].h - merges[m - 1].h;
        if (g > bg) {
          bg = g;
          b = k;
        }
      }
      return b;
    };
    const draw = () => {
      const colors = readThemeColors(),
        col = UC(colors),
        n = P.length,
        L = labels(lvl());
      {
        const V = uView(c),
          { g, w, h, X, Y } = V;
        uFrame(V, colors, 'x', 'y');
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
          const q = merges[shown - 1],
            A = mem[q.a],
            B = mem[q.b];
          let pa, pb2;
          if (settings.link === 'single' || settings.link === 'complete') {
            let bd = settings.link === 'single' ? 1e18 : -1;
            for (const i of A)
              for (const j of B) {
                const d = d2(P[i], P[j]);
                if (settings.link === 'single' ? d < bd : d > bd) {
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
          g.strokeStyle = colors.ink;
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
          g.fillStyle = k < n ? mix(colors.ink, colors.ink, 0, 0.42) : col[colOf(k)];
          g.fill();
          g.strokeStyle = colors.bg;
          g.lineWidth = 1.2;
          g.stroke();
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          m = { l: 44, r: 14, t: 16, b: 22 },
          hm = Math.max(1e-9, merges.length ? merges[merges.length - 1].h : 1) * 1.06,
          X = (v) => m.l + (n > 1 ? v / (n - 1) : 0.5) * (w - m.l - m.r),
          Y = (v) => h - m.b - (v / hm) * (h - m.t - m.b),
          lv = lvl();
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l - 8, m.t);
        ctx.lineTo(m.l - 8, h - m.b);
        ctx.stroke();
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText(f2(hm / 1.06), m.l - 12, Y(hm / 1.06) + 4);
        ctx.fillText('0', m.l - 12, Y(0) + 4);
        ctx.save();
        ctx.translate(11, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('merge distance', 0, 0);
        ctx.restore();
        for (let k = 0; k < shown; k++) {
          const q = merges[k];
          ctx.strokeStyle = k < lv ? col[colOf(L[mem[q.id][0]])] : colors.ink;
          ctx.lineWidth = k === shown - 1 && !cut ? 3 : 1.7;
          ctx.beginPath();
          ctx.moveTo(X(nodeXPositions[q.a]), Y(nodeHeights[q.a]));
          ctx.lineTo(X(nodeXPositions[q.a]), Y(q.h));
          ctx.lineTo(X(nodeXPositions[q.b]), Y(q.h));
          ctx.lineTo(X(nodeXPositions[q.b]), Y(nodeHeights[q.b]));
          ctx.stroke();
        }
        for (let i = 0; i < n; i++) {
          ctx.fillStyle = L[i] < n ? mix(colors.ink, colors.ink, 0, 0.42) : col[colOf(L[i])];
          ctx.beginPath();
          ctx.arc(X(pos[i]), h - m.b + 7, 3, 0, 7);
          ctx.fill();
        }
        if (cut && merges.length) {
          const k = Math.min(settings.K, n),
            mm = n - k,
            lo = mm > 0 ? merges[mm - 1].h : 0,
            hi = mm < merges.length ? merges[mm].h : lo * 1.05,
            y = Y((lo + hi) / 2);
          ctx.setLineDash([7, 5]);
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(m.l - 8, y);
          ctx.lineTo(w - m.r, y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = colors.cut;
          ctx.textAlign = 'right';
          ctx.fillText('cut here: ' + k + ' cluster' + (k === 1 ? '' : 's'), w - m.r, y - 6);
        }
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        ctx.fillText('one dot per point', w / 2, h - 2);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'hc', activeStepIndex);
      setReadout(root, 'cl', String(n - lvl()));
      setReadout(root, 'mg', shown + ' of ' + merges.length);
      setReadout(root, 'ht', shown ? f2(merges[shown - 1].h) : '–');
      setReadout(root, 'gap', merges.length > 2 ? sugg() + ' clusters' : '–');
    };
    const fin = () => {
      shown = merges.length;
      cut = true;
      activeStepIndex = 4;
      msg = cutMsg();
    };
    const gen = uDataBind(
      root,
      'hc',
      settings,
      () => {
        P = gen();
        build();
        reset();
        fin();
        draw();
      },
      () => {
        if (settings.ds !== 'moons') return uData(settings.ds, settings.n, settings.seed);
        const r = rng(settings.seed * 13 + 5),
          rn = randnFrom(r),
          o = [],
          hf = Math.ceil(settings.n / 2);
        for (let i = 0; i < settings.n; i++) {
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
      frameCount++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          reset();
          draw();
        }
        return;
      }
      if (frameCount % (shown < 6 ? 30 : shown > merges.length - 6 ? 26 : 7)) return;
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#hc-play', root), anim);
    const checkRunStop = attachRunLimit(playback, root, 'hc', {
      done: () => cut,
      restart: () => reset(),
    });
    $('#hc-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#hc-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    $('#hc-fin', root).onclick = () => {
      playback.stop();
      fin();
      draw();
    };
    $('#hc-link', root).onchange = (e) => {
      settings.link = e.target.value;
      playback.stop();
      build();
      fin();
      draw();
    };
    bind(
      root,
      'hc-K',
      (v) => {
        settings.K = v;
        playback.stop();
        fin();
        draw();
      },
      (v) => String(v),
    );
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        reset();
        draw();
        playback.start();
      },
    };
  },
};
