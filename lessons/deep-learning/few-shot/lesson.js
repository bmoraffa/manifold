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
import { UC, dotAt, uCur, uLine } from '../../unsupervised/shared.js';
import { laTop } from '../../../js/math/linear-algebra.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#fs-c', root),
      cr = $('#fs-r', root),
      cd = $('#fs-d', root),
      live = $('#fs-live', root),
      out = $('#fs-out', root),
      S = { k: 1, w: 5, z: 1 },
      G = 7,
      P = 49,
      DM = 8,
      H = 48,
      NQ = 5;
    const mkC = (seed) => {
      const r = rng(seed * 7919 + 13),
        st = [];
      let p = [1 + Math.floor(r() * 5), 1 + Math.floor(r() * 5)];
      for (let k = 0; k < 3; k++) {
        let q;
        do {
          q = [Math.floor(r() * G), Math.floor(r() * G)];
        } while (Math.hypot(q[0] - p[0], q[1] - p[1]) < 3);
        st.push([p, q]);
        p = r() < 0.6 ? q : [1 + Math.floor(r() * 5), 1 + Math.floor(r() * 5)];
      }
      return st;
    };
    const render = (st, r, z) => {
      const img = new Float64Array(P),
        dx = Math.round((r() - 0.5) * 2 * z),
        dy = Math.round((r() - 0.5) * 2 * z);
      for (const [a, b] of st) {
        const j = [0, 0, 0, 0].map(() => (r() - 0.5) * 1.2 * z),
          n = 14;
        for (let i = 0; i <= n; i++) {
          const x = Math.round(a[0] + j[0] + ((b[0] + j[2] - a[0] - j[0]) * i) / n) + dx,
            y = Math.round(a[1] + j[1] + ((b[1] + j[3] - a[1] - j[1]) * i) / n) + dy;
          if (x >= 0 && x < G && y >= 0 && y < G) img[y * G + x] = 1;
        }
      }
      for (let i = 0; i < P; i++) if (r() < 0.03 * z) img[i] = 1 - img[i];
      return img;
    };
    const BASE = Array.from({ length: 200 }, (_, i) => mkC(i + 1)),
      NOV = Array.from({ length: 400 }, (_, i) => mkC(1000 + i));
    let N = null,
      ep = 0,
      fin = false,
      hist = [[], []],
      pxA = [NaN, NaN],
      E = null,
      tseed = 1,
      mine = new Float64Array(P),
      msg = '',
      cu = 0,
      pb,
      chk = () => false,
      trng = null,
      best = 0,
      since = 0;
    const mkNet = () => {
      const r = randnFrom(rng(3));
      N = {
        W1: Array.from({ length: H }, () => Float64Array.from({ length: P }, () => r() / 7)),
        b1: new Float64Array(H),
        W2: Array.from({ length: DM }, () =>
          Float64Array.from({ length: H }, () => r() / Math.sqrt(H)),
        ),
        b2: new Float64Array(DM),
        t: 0,
        m: {},
        v: {},
      };
    };
    const fwd = (x) => {
      const h = new Float64Array(H);
      for (let j = 0; j < H; j++) {
        let s = N.b1[j];
        const w = N.W1[j];
        for (let i = 0; i < P; i++) if (x[i]) s += w[i];
        h[j] = Math.tanh(s);
      }
      const e = new Float64Array(DM);
      for (let k = 0; k < DM; k++) {
        let s = N.b2[k];
        const w = N.W2[k];
        for (let j = 0; j < H; j++) s += w[j] * h[j];
        e[k] = s;
      }
      return { h, e };
    };
    const pick = (pool, r, way) => {
      const cs = [];
      while (cs.length < way) {
        const q = pool[Math.floor(r() * pool.length)];
        if (!cs.includes(q)) cs.push(q);
      }
      return cs;
    };
    const protos = (fs, shot) =>
      fs.map((a) => {
        const m = new Float64Array(DM);
        for (const f of a) for (let k = 0; k < DM; k++) m[k] += f.e[k] / shot;
        return m;
      });
    const probs = (e, pro) => {
      const ds = pro.map((q) => {
          let s = 0;
          for (let k = 0; k < DM; k++) s += (e[k] - q[k]) ** 2;
          return s;
        }),
        mn = Math.min(...ds),
        ex = ds.map((v) => Math.exp(-(v - mn))),
        Z = ex.reduce((p, v) => p + v, 0);
      return ex.map((v) => v / Z);
    };
    const episode = (pool, r, way, shot, z, lr) => {
      const cs = pick(pool, r, way),
        sup = cs.map((q) => Array.from({ length: shot }, () => render(q, r, z))),
        qry = cs.map((q) => Array.from({ length: NQ }, () => render(q, r, z))),
        fs = sup.map((a) => a.map(fwd)),
        fq = qry.map((a) => a.map(fwd)),
        pro = protos(fs, shot);
      let ok = 0;
      const tot = way * NQ,
        gS = fs.map((a) => a.map(() => new Float64Array(DM))),
        gQ = fq.map((a) => a.map(() => new Float64Array(DM)));
      fq.forEach((a, y) =>
        a.forEach((f, qi) => {
          const p = probs(f.e, pro);
          if (p.indexOf(Math.max(...p)) === y) ok++;
          if (lr)
            for (let q = 0; q < way; q++) {
              const co = (q === y ? 1 : 0) - p[q];
              for (let k = 0; k < DM; k++) {
                const df = 2 * co * (f.e[k] - pro[q][k]);
                gQ[y][qi][k] += df;
                for (let s = 0; s < shot; s++) gS[q][s][k] -= df / shot;
              }
            }
        }),
      );
      if (lr) {
        const gW1 = N.W1.map(() => new Float64Array(P)),
          gb1 = new Float64Array(H),
          gW2 = N.W2.map(() => new Float64Array(H)),
          gb2 = new Float64Array(DM),
          acc = (x, f, g) => {
            const dh = new Float64Array(H);
            for (let k = 0; k < DM; k++) {
              gb2[k] += g[k];
              for (let j = 0; j < H; j++) {
                gW2[k][j] += g[k] * f.h[j];
                dh[j] += g[k] * N.W2[k][j];
              }
            }
            for (let j = 0; j < H; j++) {
              const tt = dh[j] * (1 - f.h[j] ** 2);
              gb1[j] += tt;
              for (let i = 0; i < P; i++) if (x[i]) gW1[j][i] += tt;
            }
          };
        sup.forEach((a, q) => a.forEach((x, s) => acc(x, fs[q][s], gS[q][s])));
        qry.forEach((a, q) => a.forEach((x, s) => acc(x, fq[q][s], gQ[q][s])));
        N.t++;
        const up = (nm, w, g) => {
          const m = N.m[nm] || (N.m[nm] = new Float64Array(w.length)),
            v = N.v[nm] || (N.v[nm] = new Float64Array(w.length));
          for (let i = 0; i < w.length; i++) {
            const gi = g[i] / tot;
            m[i] = 0.9 * m[i] + 0.1 * gi;
            v[i] = 0.999 * v[i] + 0.001 * gi * gi;
            w[i] -=
              (lr * (m[i] / (1 - 0.9 ** N.t))) / (Math.sqrt(v[i] / (1 - 0.999 ** N.t)) + 1e-8);
          }
        };
        N.W1.forEach((w, j) => up('a' + j, w, gW1[j]));
        up('b1', N.b1, gb1);
        N.W2.forEach((w, k) => up('c' + k, w, gW2[k]));
        up('b2', N.b2, gb2);
      }
      return ok / tot;
    };
    const evalN = (shot) => {
      const r = rng(99 + shot);
      let s = 0;
      for (let e = 0; e < 100; e++) s += episode(NOV, r, S.w, shot, S.z, 0);
      return s / 100;
    };
    const evalPx = (shot) => {
      const r = rng(99 + shot);
      let s = 0;
      for (let e = 0; e < 100; e++) {
        const cs = pick(NOV, r, S.w),
          sup = cs.map((q) => Array.from({ length: shot }, () => render(q, r, S.z))),
          qry = cs.map((q) => Array.from({ length: NQ }, () => render(q, r, S.z))),
          pro = sup.map((a) => {
            const m = new Float64Array(P);
            for (const x of a) for (let i = 0; i < P; i++) m[i] += x[i] / shot;
            return m;
          });
        let ok = 0;
        qry.forEach((a, y) =>
          a.forEach((x) => {
            let bi = 0,
              bd = 1e9;
            pro.forEach((m, i) => {
              let d = 0;
              for (let k = 0; k < P; k++) d += (x[k] - m[k]) ** 2;
              if (d < bd) {
                bd = d;
                bi = i;
              }
            });
            if (bi === y) ok++;
          }),
        );
        s += ok / (S.w * NQ);
      }
      return s / 100;
    };
    const measure = () => {
      hist[0].push(evalN(1));
      hist[1].push(S.k === 1 ? hist[0][hist[0].length - 1] : evalN(S.k));
    };
    const task = () => {
      const r = rng(tseed * 101 + 7),
        cs = pick(NOV, r, S.w);
      E = {
        cs,
        sup: cs.map((q) => Array.from({ length: S.k }, () => render(q, r, S.z))),
        qry: cs.map((q) => Array.from({ length: NQ }, () => render(q, r, S.z))),
      };
    };
    const solve = () => {
      const fs = E.sup.map((a) => a.map(fwd)),
        fq = E.qry.map((a) => a.map(fwd));
      E.fs = fs;
      E.fq = fq;
      E.pro = protos(fs, S.k);
      E.pred = fq.map((a) =>
        a.map((f) => {
          const p = probs(f.e, E.pro);
          return p.indexOf(Math.max(...p));
        }),
      );
      E.ok = E.pred.reduce((s, a, y) => s + a.filter((v) => v === y).length, 0);
    };
    const reset = () => {
      mkNet();
      ep = 0;
      fin = false;
      hist = [[], []];
      trng = rng(7);
      best = 0;
      since = 0;
      pxA = [evalPx(1), S.k === 1 ? NaN : evalPx(S.k)];
      if (S.k === 1) pxA[1] = pxA[0];
      measure();
      solve();
      cu = 0;
      msg =
        'An untrained network: its map of the drawings is arbitrary. On unseen symbols with one example each it scores ' +
        pct(hist[0][0]) +
        '; comparing raw pixels scores ' +
        pct(pxA[0]) +
        '. Chance is ' +
        pct(1 / S.w) +
        '. Press Run.';
    };
    const train = (n) => {
      if (fin) return;
      for (let i = 0; i < n; i++) {
        episode(BASE, trng, 5, 3, S.z, 0.004);
        ep++;
        if (ep % 100 === 0) {
          measure();
          const a = hist[0][hist[0].length - 1];
          if (a > best + 0.007) {
            best = a;
            since = 0;
          } else since++;
          if ((ep >= 1500 && since >= 8) || ep >= 6000) {
            fin = true;
            break;
          }
        }
      }
      solve();
      cu = 1 + (Math.floor(ep / 20) % 4);
      const a1 = hist[0][hist[0].length - 1];
      msg =
        'After ' + ep + ' training tasks: ' + pct(a1) + ' on unseen symbols with one example each.';
      if (fin) {
        cu = 5;
        msg =
          'Stopped after ' +
          ep +
          ' training tasks: accuracy on unseen symbols has levelled off. One example each: ' +
          pct(a1) +
          '. With ' +
          S.k +
          ' example' +
          (S.k === 1 ? '' : 's') +
          ' each: ' +
          pct(hist[1][hist[1].length - 1]) +
          '. Comparing raw pixels: ' +
          pct(pxA[1]) +
          '. Chance: ' +
          pct(1 / S.w) +
          '.';
      }
    };
    const glyph = (g, img, x, y, s, t, bord) => {
      const cs = s / G;
      g.fillStyle = mix(t.bg, t.ink, 0.05);
      g.fillRect(x, y, s, s);
      g.fillStyle = t.ink;
      for (let i = 0; i < P; i++)
        if (img[i])
          g.fillRect(x + (i % G) * cs, y + Math.floor(i / G) * cs, Math.ceil(cs), Math.ceil(cs));
      if (bord) {
        g.strokeStyle = bord;
        g.lineWidth = 2.5;
        g.strokeRect(x - 2, y - 2, s + 4, s + 4);
      }
    };
    const drawMine = () => {
      const t = T(),
        g = cd.getContext('2d'),
        s = 126;
      g.clearRect(0, 0, s, s);
      glyph(g, mine, 0, 0, s, t, null);
      g.strokeStyle = t.line;
      g.lineWidth = 1;
      for (let i = 1; i < G; i++) {
        g.beginPath();
        g.moveTo(i * 18, 0);
        g.lineTo(i * 18, s);
        g.moveTo(0, i * 18);
        g.lineTo(s, i * 18);
        g.stroke();
      }
      if (!mine.some((v) => v)) {
        out.textContent = 'Click squares to draw a symbol, or start from one in the task.';
        return;
      }
      const p = probs(fwd(mine).e, E.pro),
        bi = p.indexOf(Math.max(...p));
      out.textContent =
        'Your drawing is closest to row ' +
        (bi + 1) +
        ' (' +
        pct(p[bi]) +
        '). All rows: ' +
        p.map((v, i) => i + 1 + ': ' + Math.round(100 * v) + '%').join(', ') +
        '.';
    };
    const draw = () => {
      const t = T(),
        cols = UC(t);
      {
        const { g, w, h } = fit(c),
          rows = S.w,
          nc = S.k + NQ,
          gs = Math.min((h - 28) / rows - 8, (w - 70) / (nc + 0.6) - 8),
          y0 = 24;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('given', 34, 14);
        g.fillText('to classify', 34 + S.k * (gs + 8) + 18, 14);
        for (let r = 0; r < rows; r++) {
          const y = y0 + r * (gs + 8);
          g.lineWidth = 1;
          dotAt(g, 14, y + gs / 2, 7, cols[r % cols.length], t.bg);
          g.fillStyle = t.bg;
          g.font = MONO;
          g.textAlign = 'center';
          g.fillText(String(r + 1), 14, y + gs / 2 + 4);
          E.sup[r].forEach((im, k) =>
            glyph(g, im, 34 + k * (gs + 8), y, gs, t, cols[r % cols.length]),
          );
          E.qry[r].forEach((im, k) => {
            const okk = E.pred[r][k] === r,
              x = 34 + S.k * (gs + 8) + 18 + k * (gs + 8);
            glyph(g, im, x, y, gs, t, okk ? mix(t.ink, t.ink, 0, 0.55) : t.cut);
            if (!okk) {
              g.fillStyle = t.cut;
              g.font = MONO;
              g.textAlign = 'right';
              g.fillText('→' + (E.pred[r][k] + 1), x + gs - 2, y + gs - 3);
            }
          });
        }
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.beginPath();
        const lx = 34 + S.k * (gs + 8) + 5;
        g.moveTo(lx, y0 - 4);
        g.lineTo(lx, y0 + rows * (gs + 8));
        g.stroke();
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.56,
          pts = [];
        E.fs.forEach((a, y) => a.forEach((f) => pts.push([f.e, y, 0])));
        E.fq.forEach((a, y) => a.forEach((f, k) => pts.push([f.e, y, 1, E.pred[y][k] === y])));
        const mu = new Float64Array(DM);
        pts.forEach((p) => {
          for (let k = 0; k < DM; k++) mu[k] += p[0][k] / pts.length;
        });
        const C = Array.from({ length: DM }, (_, a) =>
            Array.from(
              { length: DM },
              (_, b) =>
                pts.reduce((s, p) => s + (p[0][a] - mu[a]) * (p[0][b] - mu[b]), 0) / pts.length,
            ),
          ),
          ev = laTop(C, 2, 5).vecs,
          pr = (e) =>
            ev.map((v) => {
              let s = 0;
              for (let k = 0; k < DM; k++) s += v[k] * (e[k] - mu[k]);
              return s;
            }),
          Q = pts.map((p) => pr(p[0])),
          mx = Math.max(1e-6, ...Q.map((q) => Math.abs(q[0]))),
          my = Math.max(1e-6, ...Q.map((q) => Math.abs(q[1]))),
          X = (v) => w / 2 + (v / mx) * (w / 2 - 22),
          Yc = (v) => 16 + (h1 - 22) / 2 - (v / my) * ((h1 - 30) / 2);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('this task in the network’s space', 8, 13);
        E.pro.forEach((q, y) => {
          const z = pr(q);
          g.lineWidth = 2;
          dotAt(
            g,
            X(z[0]),
            Yc(z[1]),
            10,
            mix(cols[y % cols.length], cols[y % cols.length], 0, 0.35),
            cols[y % cols.length],
          );
          g.fillStyle = t.ink;
          g.textAlign = 'center';
          g.fillText(String(y + 1), X(z[0]), Yc(z[1]) + 4);
        });
        pts.forEach((p, i) => {
          if (!p[2]) return;
          g.lineWidth = 1.5;
          dotAt(g, X(Q[i][0]), Yc(Q[i][1]), 3.6, cols[p[1] % cols.length], p[3] ? null : t.cut);
        });
        const n = Math.max(16, hist[0].length),
          A = uLine(g, t, 0, h1, w, h - h1, hist[0], {
            title:
              'accuracy on unseen symbols: 1 example (green)' +
              (S.k > 1 ? ', ' + S.k + ' examples (gold)' : '') +
              ', pixels (dashed)',
            n,
            y1: 0,
            ym: 1,
          });
        g.strokeStyle = t.cut;
        g.setLineDash([6, 4]);
        g.lineWidth = 1.5;
        g.beginPath();
        g.moveTo(A.X(0), A.Y(pxA[1]));
        g.lineTo(A.X(n - 1), A.Y(pxA[1]));
        g.stroke();
        g.setLineDash([]);
        if (S.k > 1) {
          g.strokeStyle = t.gold;
          g.lineWidth = 2;
          g.beginPath();
          hist[1].forEach((v, i) => (i ? g.lineTo(A.X(i), A.Y(v)) : g.moveTo(A.X(i), A.Y(v))));
          g.stroke();
        }
        g.strokeStyle = t.accent;
        g.lineWidth = 2.2;
        g.beginPath();
        hist[0].forEach((v, i) => (i ? g.lineTo(A.X(i), A.Y(v)) : g.moveTo(A.X(i), A.Y(v))));
        g.stroke();
        if (hist[0].length === 1) dotAt(g, A.X(0), A.Y(hist[0][0]), 4, t.accent);
      }
      drawMine();
      live.textContent = msg;
      uCur(root, 'fs', cu);
      setR(root, 'ep', ep + (fin ? ' (stopped)' : ''));
      setR(root, 'a1', pct(hist[0][hist[0].length - 1]));
      setR(root, 'ak', pct(hist[1][hist[1].length - 1]) + ' (' + S.k + ' each)');
      setR(root, 'px', pct(pxA[1]));
      setR(root, 'this', E.ok + ' of ' + S.w * NQ + ' right');
    };
    task();
    reset();
    const anim = Anim(() => {
      if (fin) {
        pb.stop();
        return;
      }
      train(Math.min(ep < 200 ? 4 : 10, chk.remaining()));
      draw();
      chk();
    });
    pb = playBtn($('#fs-play', root), anim);
    chk = runCtl(pb, root, 'fs', {
      done: () => fin,
      restart: reset,
      count: () => ep,
      what: 'tasks',
    });
    $('#fs-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else train(20);
      draw();
    };
    $('#fs-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    $('#fs-new', root).onclick = () => {
      tseed++;
      task();
      solve();
      draw();
    };
    const re = () => {
      task();
      pxA = [evalPx(1), S.k === 1 ? NaN : evalPx(S.k)];
      if (S.k === 1) pxA[1] = pxA[0];
      hist[0] = [evalN(1)];
      hist[1] = [S.k === 1 ? hist[0][0] : evalN(S.k)];
      solve();
      msg =
        'Task changed. Accuracy on unseen symbols is now measured on ' +
        S.w +
        '-way tasks with handwriting sloppiness ' +
        f2(S.z) +
        '. The network keeps what it has learned.';
      draw();
    };
    bind(
      root,
      'fs-k',
      (v) => {
        S.k = v;
        re();
      },
      (v) => String(+v),
    );
    bind(
      root,
      'fs-w',
      (v) => {
        S.w = v;
        re();
      },
      (v) => String(+v),
    );
    bind(
      root,
      'fs-z',
      (v) => {
        S.z = v;
        re();
      },
      f2,
    );
    const tog = (e) => {
      const r = cd.getBoundingClientRect(),
        x = Math.floor(((e.clientX - r.left) / r.width) * G),
        y = Math.floor(((e.clientY - r.top) / r.height) * G);
      if (x >= 0 && x < G && y >= 0 && y < G) {
        mine[y * G + x] = 1 - mine[y * G + x];
        drawMine();
      }
    };
    cd.addEventListener('pointerdown', tog);
    $('#fs-copy', root).onclick = () => {
      const r = Math.floor(Math.random() * S.w);
      mine = Float64Array.from(E.qry[r][Math.floor(Math.random() * NQ)]);
      drawMine();
    };
    $('#fs-clr', root).onclick = () => {
      mine = new Float64Array(P);
      drawMine();
    };
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        draw();
        pb.start();
      },
    };
  },
};
