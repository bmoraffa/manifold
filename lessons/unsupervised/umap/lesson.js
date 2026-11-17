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
  fit,
  letterData,
  mix,
  playBtn,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { UC, uCur, uLine } from '../shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
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
          if (j !== i) a.push([(Y[2 * i] - Y[2 * j]) ** 2 + (Y[2 * i + 1] - Y[2 * j + 1]) ** 2, j]);
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
    if (RM) all();
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
};
