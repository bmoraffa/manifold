/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { UC, jacobi, highlightLessonStep, drawSeriesChart } from '../shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#ts-c', root),
      cr = $('#ts-r', root),
      live = $('#ts-live', root),
      settings = { perp: 15, lr: 40, flip: 0.06, col: true },
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
      adaptiveGains = null,
      it = 0,
      KL = [],
      pcaEmbedding = [],
      pcaVarianceShare = 0,
      nbH = [],
      seed = 1,
      run = 1,
      scl = 1,
      hover = -1,
      msg = '',
      activeStepIndex = 1,
      wait = 0,
      frameCount = 0,
      playback,
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
              if (r() < settings.flip) v = 1 - v;
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
      pcaEmbedding = Xh.map((x) =>
        [0, 1].map((k) => x.reduce((s, v, j) => s + (v - mu[j]) * E.vec[k][j], 0)),
      );
      pcaVarianceShare = (E.val[0] + E.val[1]) / E.val.reduce((a, b) => a + Math.max(0, b), 0);
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
        H = Math.log(Math.min(settings.perp, (n - 1) / 3));
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
      adaptiveGains = new Float64Array(2 * N).fill(1);
      it = 0;
      KL = [];
      scl = 0.03;
      activeStepIndex = 1;
      wait = 0;
      conv = false;
      msg =
        'Steps 1 and 2 are done. Every picture now has about ' +
        settings.perp +
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
        adaptiveGains[k] =
          G[k] > 0 !== vel[k] > 0 ? adaptiveGains[k] + 0.2 : Math.max(0.01, adaptiveGains[k] * 0.8);
        vel[k] = mom * vel[k] - settings.lr * adaptiveGains[k] * G[k];
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
      activeStepIndex = fin() ? 4 : 3;
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
      if (settings.col) {
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
      const colors = readThemeColors(),
        col = UC(colors);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas);
        let m = 1e-9;
        for (let k = 0; k < 2 * N; k++) m = Math.max(m, Math.abs(Y[k]));
        scl += (m * 1.12 - scl) * (playback && playback.on ? 0.25 : 1);
        const s = (Math.min(w, h) / 2 / scl) * 0.94;
        PX = [];
        for (let i = 0; i < N; i++) {
          const x = w / 2 + Y[2 * i] * s,
            y = h / 2 - Y[2 * i + 1] * s;
          PX.push([x, y]);
          mark(ctx, x, y, i, colors, col, true);
        }
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
            const v = clamp(Xh[hover][k], 0, 1);
            ctx.fillStyle = mix(colors.bg, colors.ink, v);
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
        ctx.fillText('iteration ' + it, w - 8, h - 8);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.56;
        let mx = 1e-9,
          my = 1e-9;
        for (const q of pcaEmbedding) {
          mx = Math.max(mx, Math.abs(q[0]));
          my = Math.max(my, Math.abs(q[1]));
        }
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          'PCA: the best flat shadow keeps ' +
            (100 * pcaVarianceShare).toFixed(0) +
            '% of the spread',
          8,
          13,
        );
        for (let i = 0; i < N; i++)
          mark(
            ctx,
            w / 2 + (pcaEmbedding[i][0] / mx) * (w / 2 - 18),
            22 + (h1 - 30) / 2 - (pcaEmbedding[i][1] / my) * ((h1 - 30) / 2 - 8),
            i,
            colors,
            col,
            false,
          );
        drawSeriesChart(ctx, colors, 0, h1, w, h - h1, KL, {
          title: 'mismatch between map and original neighbourhoods, by iteration',
          n: MAXIT,
          ym: Math.max(0.5, ...KL),
          col: colors.ok,
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'ts', activeStepIndex);
      setReadout(root, 'it', String(it));
      setReadout(root, 'kl', KL.length ? f3(KL[KL.length - 1]) : '–');
      setReadout(root, 'nt', it ? pct(keep((i) => [Y[2 * i], Y[2 * i + 1]])) : '–');
      setReadout(root, 'np', pct(keep((i) => pcaEmbedding[i])));
      setReadout(
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
    if (prefersReducedMotion) while (!fin()) iter();
    say();
    scl = 1e-9;
    const anim = Anim(() => {
      frameCount++;
      if (it < 12 && frameCount % 6) return;
      for (let i = 0; i < (it < 40 ? 1 : 3) && !fin() && checkRunStop.remaining() > 0; i++) iter();
      say();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#ts-play', root), anim);
    const checkRunStop = attachRunLimit(playback, root, 'ts', {
      done: fin,
      restart: () => {
        run++;
        reset();
      },
      count: () => it,
      what: 'iterations',
    });
    $('#ts-step', root).onclick = () => {
      playback.stop();
      step(10);
      draw();
    };
    $('#ts-reset', root).onclick = () => {
      playback.stop();
      run++;
      reset();
      draw();
    };
    bind(
      root,
      'ts-perp',
      (v) => {
        settings.perp = v;
        playback.stop();
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
        settings.lr = v;
      },
      (v) => String(v),
    );
    $('#ts-col', root).onchange = (e) => {
      settings.col = e.target.checked;
      draw();
    };
    bind(root, 'ts-flip', (v) => {
      settings.flip = v;
      playback.stop();
      gen();
      calcP();
      reset();
      draw();
    });
    $('#ts-new', root).onclick = () => {
      seed++;
      playback.stop();
      gen();
      calcP();
      reset();
      draw();
    };
    mainCanvas.addEventListener('pointermove', (e) => {
      const r = mainCanvas.getBoundingClientRect(),
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
