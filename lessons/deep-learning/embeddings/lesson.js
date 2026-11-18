/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  SANS,
  readThemeColors,
  bind,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { UC, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';
import { sig } from '../../classification/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#we-c', root),
      cr = $('#we-r', root),
      live = $('#we-live', root),
      settings = { col: true, neg: 2 },
      GR = [
        ['king', 'queen', 'prince', 'princess'],
        ['cat', 'dog', 'horse', 'bird'],
        ['apple', 'bread', 'cheese', 'rice'],
        ['paris', 'london', 'tokyo', 'rome'],
      ],
      CX = [
        ['crown', 'throne', 'palace'],
        ['fur', 'tail', 'paw'],
        ['eat', 'cook', 'tasty'],
        ['city', 'travel', 'map'],
      ];
    let W = [],
      G = [],
      positivePairs = [],
      U,
      Vc,
      ep = 0,
      LL = [],
      conv = false,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      sel = 1,
      lastTrainingExample = null,
      PX = [];
    GR.forEach((g, k) => {
      g.forEach((w) => {
        W.push(w);
        G.push(k);
      });
      CX[k].forEach((w) => {
        W.push(w);
        G.push(k);
      });
    });
    GR.forEach((g, k) =>
      g.forEach((w) =>
        CX[k].forEach((x) => {
          positivePairs.push([W.indexOf(w), W.indexOf(x)]);
          positivePairs.push([W.indexOf(x), W.indexOf(w)]);
        }),
      ),
    );
    const N = W.length;
    const build = () => {
      const r = rng(run * 13 + 1);
      U = W.map(() => [(r() - 0.5) * 1.6, (r() - 0.5) * 1.6]);
      Vc = W.map(() => [(r() - 0.5) * 0.4, (r() - 0.5) * 0.4]);
      ep = 0;
      LL = [];
      conv = false;
      activeStepIndex = 0;
      lastTrainingExample = null;
      msg =
        'Every word has a random position. The model knows nothing about any of them. Press Train or One pass.';
    };
    const upd = (a, b, y) => {
      const u = U[a],
        v = Vc[b],
        p = sig(u[0] * v[0] + u[1] * v[1]),
        g = (p - y) * 0.06,
        u0 = u[0],
        u1 = u[1];
      u[0] -= g * v[0];
      u[1] -= g * v[1];
      v[0] -= g * u0;
      v[1] -= g * u1;
      return -Math.log((y ? p : 1 - p) + 1e-9);
    };
    const one = () => {
      const pr = positivePairs[Math.floor(Math.random() * positivePairs.length)];
      let l = upd(pr[0], pr[1], 1),
        ng = -1;
      for (let k = 0; k < settings.neg; k++) {
        let nw;
        do {
          nw = Math.floor(Math.random() * N);
        } while (G[nw] === G[pr[0]]);
        l += upd(pr[0], nw, 0);
        ng = nw;
      }
      lastTrainingExample = [pr[0], pr[1], ng];
      return l;
    };
    const cosv = (a, b) =>
        (U[a][0] * U[b][0] + U[a][1] * U[b][1]) / (Math.hypot(...U[a]) * Math.hypot(...U[b]) || 1),
      near = (a) =>
        W.map((_, i) => i)
          .filter((i) => i !== a)
          .sort((i, j) => cosv(a, j) - cosv(a, i));
    const nearestByEuclideanDistance = (a) => {
      let b = -1,
        bd = 1e9;
      for (let i = 0; i < N; i++) {
        if (i === a) continue;
        const d = Math.hypot(U[a][0] - U[i][0], U[a][1] - U[i][1]);
        if (d < bd) {
          bd = d;
          b = i;
        }
      }
      return b;
    };
    const pass = () => {
      let l = 0;
      for (let i = 0; i < positivePairs.length; i++) l += one();
      ep++;
      LL.push(l / positivePairs.length);
      if ((ep > 60 && Math.abs(LL[ep - 1] - LL[ep - 41]) < 0.01 * LL[ep - 1]) || ep >= 600)
        conv = true;
      activeStepIndex = conv ? 4 : 2 + (ep % 2);
      msg =
        'Pass ' +
        ep +
        ': all ' +
        positivePairs.length +
        ' pairs from the text were pulled together once, each with ' +
        settings.neg +
        ' random pair' +
        (settings.neg === 1 ? '' : 's') +
        ' pushed apart. Last pair pulled: “' +
        W[lastTrainingExample[0]] +
        '” and “' +
        W[lastTrainingExample[1]] +
        '”. Error ' +
        f3(LL[ep - 1]) +
        '.' +
        (conv
          ? ep >= 600
            ? ' Reached the limit of 600 passes.'
            : ' Converged: the error has stopped falling, so the map has settled.'
          : '');
    };
    const draw = () => {
      const colors = readThemeColors(),
        col = UC(colors);
      {
        const { g, w, h } = prepareCanvas(mainCanvas);
        let m = 1e-9;
        U.forEach((u) => {
          m = Math.max(m, Math.abs(u[0]), Math.abs(u[1]));
        });
        const X = (v) => w / 2 + (v / m) * (w / 2 - 46),
          Y = (v) => h / 2 - (v / m) * (h / 2 - 22);
        PX = U.map((u) => [X(u[0]), Y(u[1])]);
        if (lastTrainingExample && playback && playback.on) {
          g.lineWidth = 2.2;
          g.strokeStyle = colors.gold;
          g.beginPath();
          g.moveTo(...PX[lastTrainingExample[0]]);
          g.lineTo(...PX[lastTrainingExample[1]]);
          g.stroke();
          if (lastTrainingExample[2] >= 0) {
            g.strokeStyle = colors.ok;
            g.setLineDash([5, 4]);
            g.beginPath();
            g.moveTo(...PX[lastTrainingExample[0]]);
            g.lineTo(...PX[lastTrainingExample[2]]);
            g.stroke();
            g.setLineDash([]);
          }
        }
        W.forEach((wd, i) => {
          const ctx = i % 7 > 3;
          g.font =
            (i === sel ? '700 15px' : ctx ? '500 12px' : '600 13px') +
            ' "Hanken Grotesk",sans-serif';
          g.fillStyle = settings.col ? col[G[i] % 8] : colors.ink;
          g.globalAlpha = ctx && i !== sel ? 0.7 : 1;
          g.textAlign = 'center';
          g.fillText(wd, PX[i][0], PX[i][1] + 4);
          g.globalAlpha = 1;
          if (i === sel) {
            const tw = g.measureText(wd).width;
            g.strokeStyle = colors.ink;
            g.lineWidth = 1.5;
            g.strokeRect(PX[i][0] - tw / 2 - 5, PX[i][1] - 11, tw + 10, 21);
          }
        });
      }
      {
        const { g, w, h } = prepareCanvas(cr),
          h1 = h * 0.56,
          nb = near(sel).slice(0, 9),
          rh = (h1 - 26) / 9;
        g.font = MONO;
        g.fillStyle = colors.muted;
        g.textAlign = 'left';
        g.fillText('words most similar to “' + W[sel] + '” (cosine similarity)', 8, 13);
        nb.forEach((i, k) => {
          const y = 22 + k * rh,
            v = cosv(sel, i);
          g.font = SANS;
          g.fillStyle = settings.col ? col[G[i] % 8] : colors.ink;
          g.textAlign = 'right';
          g.fillText(W[i], 88, y + rh * 0.7);
          g.fillStyle = mix(colors.ink, colors.ink, 0, 0.08);
          g.fillRect(96, y + 3, w - 150, rh - 6);
          g.fillStyle = G[i] === G[sel] ? colors.gold : colors.muted;
          g.fillRect(96, y + 3, (w - 150) * Math.max(0, v), rh - 6);
          g.font = MONO;
          g.fillStyle = colors.muted;
          g.textAlign = 'left';
          g.fillText(f2(v), w - 48, y + rh * 0.7);
        });
        drawSeriesChart(g, colors, 0, h1, w, h - h1, LL, {
          title: 'error, by pass over the text',
          n: Math.max(60, LL.length),
          col: colors.ok,
          ym: Math.max(1.5, LL[0] || 1.5),
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'we', activeStepIndex);
      let ok = 0;
      for (let i = 0; i < N; i++) if (G[nearestByEuclideanDistance(i)] === G[i]) ok++;
      setReadout(root, 'ep', String(ep));
      setReadout(root, 'L', ep ? f3(LL[ep - 1]) : '–');
      setReadout(root, 'sel', '“' + W[sel] + '”');
      setReadout(
        root,
        'nn',
        near(sel)
          .slice(0, 3)
          .map((i) => W[i])
          .join(', '),
      );
      setReadout(root, 'grp', ok + ' of ' + N);
    };
    build();
    if (prefersReducedMotion) while (!conv) pass();
    const anim = Anim(() => {
      frameCount++;
      if (frameCount % (ep < 6 ? 14 : 5) === 0) pass();
      else if (frameCount % 2 === 0) one();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#we-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'we', {
      done: () => conv,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'passes',
    });
    $('#we-step', root).onclick = () => {
      playback.stop();
      if (conv) {
        run++;
        build();
      } else pass();
      draw();
    };
    $('#we-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    $('#we-col', root).onchange = (e) => {
      settings.col = e.target.checked;
      draw();
    };
    bind(
      root,
      'we-neg',
      (v) => {
        settings.neg = v;
        playback.stop();
        run++;
        build();
        draw();
      },
      (v) => String(v),
    );
    mainCanvas.addEventListener('click', (e) => {
      const r = mainCanvas.getBoundingClientRect(),
        x = e.clientX - r.left,
        y = e.clientY - r.top;
      let b = -1,
        bd = 900;
      PX.forEach((p, i) => {
        const d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
        if (d < bd) {
          bd = d;
          b = i;
        }
      });
      if (b >= 0) {
        sel = b;
        if (!playback.on) draw();
      }
    });
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        run++;
        build();
        draw();
        playback.start();
      },
    };
  },
};
