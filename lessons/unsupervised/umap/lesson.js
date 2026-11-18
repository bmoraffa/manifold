/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  f2,
  prepareCanvas,
  letterData,
  mix,
  playBtn,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { UC, highlightLessonStep, drawSeriesChart } from '../shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#um-c', root),
      cr = $('#um-r', root),
      live = $('#um-live', root),
      settings = { k: 15, md: 0.6, flip: 0.06, col: true, ed: true },
      NE = 300;
    let X = [],
      lab = [],
      letterNames = [],
      N = 0,
      D = null,
      inputNeighbors = [],
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
      activeStepIndex = 2,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      screenPositions = [];
    const gen = () => {
      const d = letterData(seed, settings.flip);
      X = d.X;
      lab = d.lab;
      letterNames = d.CH;
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
        tg = Math.log2(settings.k);
      inputNeighbors = [];
      for (let i = 0; i < N; i++) {
        const a = [];
        for (let j = 0; j < N; j++) if (j !== i) a.push([D[i * N + j], j]);
        a.sort((u, v) => u[0] - v[0]);
        inputNeighbors.push(a.slice(0, 10).map((q) => q[1]));
        const nb = a.slice(0, settings.k),
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
              tgt = d <= settings.md ? 1 : Math.exp(-(d - settings.md)),
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
        const h = new Set(inputNeighbors[i]);
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
      activeStepIndex = 2;
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
      activeStepIndex = fin() ? 4 : 3;
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
      const colors = readThemeColors(),
        col = UC(colors);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas);
        let m = 1e-9;
        for (let k = 0; k < 2 * N; k++) m = Math.max(m, Math.abs(Y[k]));
        scl += (m * 1.1 - scl) * (playback && playback.on ? 0.2 : 1);
        const s = (Math.min(w, h) / 2 / scl) * 0.94;
        screenPositions = [];
        for (let i = 0; i < N; i++)
          screenPositions.push([w / 2 + Y[2 * i] * s, h / 2 - Y[2 * i + 1] * s]);
        if (settings.ed) {
          ctx.strokeStyle = mix(colors.ink, colors.ink, 0, 0.13);
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (const e of E) {
            if (e[2] < 0.5) continue;
            ctx.moveTo(screenPositions[e[0]][0], screenPositions[e[0]][1]);
            ctx.lineTo(screenPositions[e[1]][0], screenPositions[e[1]][1]);
          }
          ctx.stroke();
        }
        for (let i = 0; i < N; i++) {
          const x = screenPositions[i][0],
            y = screenPositions[i][1];
          ctx.fillStyle = settings.col ? col[lab[i] % 8] : mix(colors.ink, colors.ink, 0, 0.6);
          ctx.beginPath();
          ctx.arc(x, y, 3.4, 0, 7);
          ctx.fill();
          ctx.strokeStyle = colors.bg;
          ctx.lineWidth = 0.8;
          ctx.stroke();
          if (i === hover) {
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.arc(x, y, 9, 0, 7);
            ctx.stroke();
          }
        }
        if (settings.col)
          letterNames.forEach((ch, k) => {
            let mx = 0,
              my = 1e9,
              n = 0;
            for (let i = 0; i < N; i++)
              if (lab[i] === k) {
                mx += screenPositions[i][0];
                my = Math.min(my, screenPositions[i][1]);
                n++;
              }
            if (!n) return;
            ctx.font = '600 17px "Hanken Grotesk",sans-serif';
            ctx.fillStyle = col[k % 8];
            ctx.textAlign = 'center';
            ctx.fillText(ch, mx / n, Math.max(16, my - 9));
          });
        if (hover >= 0) {
          const cz = 8,
            x0 = 10,
            y0 = 10;
          ctx.fillStyle = colors.bg;
          ctx.fillRect(x0 - 4, y0 - 4, 5 * cz + 8, 7 * cz + 8);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1;
          ctx.strokeRect(x0 - 4, y0 - 4, 5 * cz + 8, 7 * cz + 8);
          for (let k = 0; k < 35; k++) {
            ctx.fillStyle = mix(colors.bg, colors.ink, clamp(X[hover][k], 0, 1));
            ctx.fillRect(x0 + (k % 5) * cz, y0 + Math.floor(k / 5) * cz, cz - 1, cz - 1);
          }
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText('the picture behind this point', x0 + 5 * cz + 12, y0 + 12);
        }
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('epoch ' + ep + ' of ' + NE, w - 8, h - 8);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.48,
          m = { l: 44, r: 12, t: 26, b: 22 },
          Xd = (d) => m.l + (d / 3) * (w - m.l - m.r),
          Yv = (v) => h1 - m.b - v * (h1 - m.t - m.b);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('similarity on the page against distance on the page', 8, 13);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, h1 - m.b);
        ctx.lineTo(w - m.r, h1 - m.b);
        ctx.stroke();
        ctx.textAlign = 'right';
        ctx.fillText('1', m.l - 5, Yv(1) + 4);
        ctx.fillText('0', m.l - 5, Yv(0) + 4);
        ctx.textAlign = 'center';
        [0, 1, 2, 3].forEach((d) => ctx.fillText(String(d), Xd(d), h1 - m.b + 14));
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = colors.muted;
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const d = i / 40,
            v = d <= settings.md ? 1 : Math.exp(-(d - settings.md));
          i ? ctx.lineTo(Xd(d), Yv(v)) : ctx.moveTo(Xd(d), Yv(v));
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const d = i / 40,
            v = 1 / (1 + ab[0] * Math.pow(d, 2 * ab[1]));
          i ? ctx.lineTo(Xd(d), Yv(v)) : ctx.moveTo(Xd(d), Yv(v));
        }
        ctx.stroke();
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(Xd(settings.md), Yv(0));
        ctx.lineTo(Xd(settings.md), Yv(1));
        ctx.stroke();
        ctx.fillStyle = colors.ok;
        ctx.textAlign = 'left';
        ctx.fillText('min_dist = ' + f2(settings.md), Xd(settings.md) + 6, Yv(0.08));
        ctx.fillStyle = colors.cut;
        ctx.textAlign = 'right';
        ctx.fillText('gold: the curve used; dashed: the target it was fitted to', w - m.r, m.t + 2);
        drawSeriesChart(ctx, colors, 0, h1, w, h - h1, hist, {
          title: 'share of true neighbours kept, measured every 5 epochs',
          n: NE / 5,
          ym: 1,
          col: colors.accent,
          dots: hist.length < 30,
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'um', activeStepIndex);
      setReadout(root, 'ep', ep + ' of ' + NE);
      setReadout(root, 'lr', f2(Math.max(0, 1 - ep / NE)));
      setReadout(root, 'ed', String(E.length));
      setReadout(root, 'nk', pct(kept));
      setReadout(root, 'ab', f2(ab[0]) + ', ' + f2(ab[1]));
      setReadout(
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
    if (prefersReducedMotion) all();
    const anim = Anim(() => {
      frameCount++;
      if (frameCount % (ep < 10 ? 8 : 2)) return;
      epoch();
      say();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#um-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'um', {
      done: fin,
      restart: () => {
        run++;
        reset();
      },
      count: () => ep,
      what: 'epochs',
    });
    $('#um-step', root).onclick = () => {
      playback.stop();
      step(10);
      draw();
    };
    $('#um-reset', root).onclick = () => {
      playback.stop();
      run++;
      reset();
      draw();
    };
    bind(
      root,
      'um-k',
      (v) => {
        settings.k = v;
        playback.stop();
        graph();
        reset();
        all();
        draw();
      },
      (v) => String(v),
    );
    bind(root, 'um-md', (v) => {
      settings.md = v;
      playback.stop();
      fitAB();
      reset();
      all();
      draw();
    });
    $('#um-col', root).onchange = (e) => {
      settings.col = e.target.checked;
      draw();
    };
    $('#um-ed', root).onchange = (e) => {
      settings.ed = e.target.checked;
      draw();
    };
    bind(root, 'um-flip', (v) => {
      settings.flip = v;
      playback.stop();
      gen();
      graph();
      reset();
      all();
      draw();
    });
    $('#um-new', root).onclick = () => {
      seed++;
      playback.stop();
      gen();
      graph();
      reset();
      all();
      draw();
    };
    mainCanvas.addEventListener('pointermove', (e) => {
      const r = mainCanvas.getBoundingClientRect(),
        x = e.clientX - r.left,
        y = e.clientY - r.top;
      let b = -1,
        bd = 196;
      screenPositions.forEach((q, i) => {
        const d = (q[0] - x) ** 2 + (q[1] - y) ** 2;
        if (d < bd) {
          bd = d;
          b = i;
        }
      });
      if (b !== hover) {
        hover = b;
        if (!playback.on) draw();
      }
    });
    mainCanvas.addEventListener('pointerleave', () => {
      if (hover >= 0) {
        hover = -1;
        if (!playback.on) draw();
      }
    });
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
