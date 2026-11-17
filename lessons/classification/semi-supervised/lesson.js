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
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import {
  canvasXY,
  dotAt,
  segBind,
  highlightLessonStep,
  drawSeriesChart,
  uView,
} from '../../unsupervised/shared.js';
import { g2Note } from '../../generative/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#ss-c', root),
      cr = $('#ss-r', root),
      live = $('#ss-live', root),
      ol = $('#ss-steps', root),
      settings = { d: 'moons', m: 'lp', n: 3, k: 7 };
    let D = [],
      Y = [],
      L = [],
      F = [],
      neighborGraph = [],
      baselinePredictions = [],
      hist = [],
      it = 0,
      fin = false,
      seed = 1,
      msg = '',
      activeStepIndex = 0,
      playback,
      checkRunStop = () => false,
      V = null,
      baselineAccuracy = 0,
      pseudo = [];
    const HOW = {
      lp: [
        'Connect every point to its nearest neighbours.',
        'Give each labelled point its label, with full confidence. Unlabelled points start undecided.',
        'Every point takes the average belief of its neighbours.',
        'Reset the labelled points to their true labels.',
        'Repeat: beliefs spread outwards from the labelled points along the connections.',
        'Stop when the beliefs no longer change.',
      ],
      self: [
        'Start with only the labelled points.',
        'For every unlabelled point, find the nearest point that has a label. That is its prediction.',
        'Find the five unlabelled points that are closest to a labelled one. These are the most confident predictions.',
        'Adopt those predictions as if they were real labels.',
        'Repeat with the larger labelled set.',
        'Stop when every point has a label.',
      ],
    };
    const gen = () => {
      const r = rng(seed * 37 + settings.d.length),
        rn = randnFrom(r);
      D = [];
      Y = [];
      for (let i = 0; i < 180; i++) {
        const k = i % 2;
        let p;
        if (settings.d === 'moons') {
          const a = Math.PI * r();
          p = k
            ? [1.7 * Math.cos(a) - 0.85 + 0.1 * rn(), 1.7 * Math.sin(a) - 0.6 + 0.1 * rn()]
            : [0.85 - 1.7 * Math.cos(a) + 0.1 * rn(), 0.6 - 1.7 * Math.sin(a) + 0.1 * rn()];
        } else if (settings.d === 'mixed')
          p = [(k ? 0.75 : -0.75) + 0.85 * rn(), (k ? 0.3 : -0.3) + 0.8 * rn()];
        else p = [(k ? 1.9 : -1.9) + 0.55 * rn(), (k ? 0.5 : -0.5) + 0.6 * rn()];
        D.push(p);
        Y.push(k);
      }
      if (settings.d === 'bridge')
        for (let i = 0; i < 14; i++) {
          const u = (i + 0.5) / 14;
          D.push([-1.3 + 2.6 * u + 0.05 * rn(), -0.35 + 0.7 * u + 0.05 * rn()]);
          Y.push(u > 0.5 ? 1 : 0);
        }
      L = D.map(() => false);
      for (const k of [0, 1]) {
        const idx = D.map((_, i) => i).filter((i) => Y[i] === k && i < 180);
        for (let q = 0; q < settings.n; q++) {
          const j = Math.floor(r() * idx.length);
          L[idx.splice(j, 1)[0]] = true;
        }
      }
    };
    const graph = () => {
      neighborGraph = D.map((p, i) => {
        const ds = D.map((q, j) => [Math.hypot(p[0] - q[0], p[1] - q[1]), j])
            .filter((e) => e[1] !== i)
            .sort((a, b) => a[0] - b[0])
            .slice(0, settings.k),
          sg = ds[ds.length - 1][0] || 1;
        return ds.map(([d, j]) => [j, Math.exp((-d * d) / (sg * sg))]);
      });
      const sym = D.map(() => new Map());
      neighborGraph.forEach((a, i) =>
        a.forEach(([j, w]) => {
          sym[i].set(j, Math.max(w, sym[i].get(j) || 0));
          sym[j].set(i, Math.max(w, sym[j].get(i) || 0));
        }),
      );
      neighborGraph = sym.map((m) => {
        const a = [...m.entries()],
          s = a.reduce((p, e) => p + e[1], 0);
        return a.map(([j, w]) => [j, w / s]);
      });
    };
    const nearLab = (p, lab) => {
      let bd = 1e9,
        bi = -1;
      for (let j = 0; j < D.length; j++)
        if (lab[j] >= 0) {
          const d = Math.hypot(p[0] - D[j][0], p[1] - D[j][1]);
          if (d < bd) {
            bd = d;
            bi = j;
          }
        }
      return bi;
    };
    const accOf = (pred) => {
      let ok = 0,
        n = 0;
      for (let i = 0; i < D.length; i++)
        if (!L[i]) {
          n++;
          if (pred[i] === Y[i]) ok++;
        }
      return n ? ok / n : 1;
    };
    const thr = () => {
      if (settings.m !== 'lp') return [0.5, 0.5];
      const u = F.filter((_, i) => !L[i]).sort((a, b) => a - b);
      if (!u.length) return [0.5, 0.5];
      const n1 = L.reduce((s, v, i) => s + (v && Y[i] ? 1 : 0), 0),
        nl = L.filter(Boolean).length,
        q = clamp(1 - n1 / Math.max(1, nl), 0.05, 0.95),
        th = u[Math.min(u.length - 1, Math.floor(q * u.length))];
      return [th, Math.max(1e-6, Math.min(u[u.length - 1] - th, th - u[0]))];
    };
    const cur = () => {
      const [th, sp] = thr();
      return F.map((f) => (Math.abs(f - th) <= 0.03 * sp ? -1 : f > th ? 1 : 0));
    };
    const reset = () => {
      ol.innerHTML = HOW[settings.m].map((s) => '<li>' + s + '</li>').join('');
      graph();
      const lab = D.map((_, i) => (L[i] ? Y[i] : -1));
      baselinePredictions = D.map((p, i) => (L[i] ? Y[i] : Y[nearLab(p, lab)]));
      baselineAccuracy = accOf(baselinePredictions);
      F = D.map((_, i) => (L[i] ? Y[i] : 0.5));
      pseudo = lab.slice();
      it = 0;
      fin = false;
      hist = [accOf(cur())];
      activeStepIndex = 1;
      msg =
        L.filter(Boolean).length +
        ' points carry a label; ' +
        (D.length - L.filter(Boolean).length) +
        ' do not. Using the labels alone, ' +
        pct(baselineAccuracy) +
        ' of the unlabelled points would be classified correctly. Press Run.';
    };
    const step = () => {
      if (fin) return;
      if (settings.m === 'lp') {
        const N = F.slice();
        let mx = 0;
        for (let i = 0; i < D.length; i++) {
          if (L[i]) continue;
          let s = 0;
          for (const [j, w] of neighborGraph[i]) s += w * F[j];
          N[i] = s;
          mx = Math.max(mx, Math.abs(s - F[i]));
        }
        F = N;
        it++;
        activeStepIndex = it % 2 ? 2 : 3;
        const und = cur().filter((v, i) => v < 0 && !L[i]).length;
        msg =
          'Step ' +
          it +
          ': the largest change in any belief was ' +
          (mx < 0.01 ? mx.toExponential(1) : f3(mx)) +
          '. ' +
          und +
          ' points are still undecided.';
        if ((mx < 1e-4 && it >= 5) || it >= 6000) {
          fin = true;
        }
      } else {
        const cand = [];
        for (let i = 0; i < D.length; i++) {
          if (pseudo[i] >= 0) continue;
          const ds = [];
          for (let j = 0; j < D.length; j++)
            if (pseudo[j] >= 0)
              ds.push([Math.hypot(D[i][0] - D[j][0], D[i][1] - D[j][1]), pseudo[j]]);
          let bd = 1e9,
            bl = 0;
          for (const e of ds)
            if (e[0] < bd) {
              bd = e[0];
              bl = e[1];
            }
          cand.push([i, bl, 0, bd]);
        }
        cand.sort((a, b) => a[3] - b[3]);
        const take = cand.slice(0, 5);
        for (const [i, y] of take) {
          pseudo[i] = y;
          F[i] = y;
        }
        it++;
        activeStepIndex = 3;
        const left = pseudo.filter((v) => v < 0).length;
        msg =
          'Step ' +
          it +
          ': adopted ' +
          take.length +
          ' confident predictions as labels. ' +
          left +
          ' points remain unlabelled.';
        if (!left) fin = true;
      }
      hist.push(accOf(cur()));
      if (fin) {
        activeStepIndex = 5;
        const a = hist[hist.length - 1];
        msg =
          'Stopped after ' +
          it +
          ' steps: ' +
          (settings.m === 'lp'
            ? 'the beliefs have stopped changing.'
            : 'every point has a label.') +
          ' Accuracy on the unlabelled points: ' +
          pct(a) +
          ', against ' +
          pct(baselineAccuracy) +
          ' from the labels alone. ' +
          (a > baselineAccuracy + 0.03
            ? 'The unlabelled data helped.'
            : a < baselineAccuracy - 0.03
              ? 'The unlabelled data made things worse here: their shape pointed the wrong way.'
              : 'The unlabelled data made little difference here.');
      }
    };
    const tp = testPoint(
      root,
      'ss',
      ['Your x', 'Your y'],
      [0.2, 1.4],
      [
        [-4, 4],
        [-3, 3],
      ],
    );
    const draw = () => {
      const colors = readThemeColors(),
        cs = [colors.accent, colors.gold],
        [TH, SP] = thr(),
        col = (f) =>
          f > TH
            ? mix(colors.muted, cs[1], clamp(((f - TH) / SP) * 1.5, 0, 1))
            : mix(colors.muted, cs[0], clamp(((TH - f) / SP) * 1.5, 0, 1));
      V = uView(c);
      {
        const { g, w, h, X, Y: PY } = V;
        if (settings.m === 'lp' && D.length < 260) {
          g.strokeStyle = mix(colors.ink, colors.ink, 0, 0.07);
          g.lineWidth = 1;
          g.beginPath();
          neighborGraph.forEach((a, i) =>
            a.forEach(([j]) => {
              if (j > i) {
                g.moveTo(X(D[i][0]), PY(D[i][1]));
                g.lineTo(X(D[j][0]), PY(D[j][1]));
              }
            }),
          );
          g.stroke();
        }
        D.forEach((p, i) => {
          if (L[i]) return;
          const wrong = fin && cur()[i] >= 0 && cur()[i] !== Y[i];
          g.lineWidth = 1.5;
          dotAt(g, X(p[0]), PY(p[1]), 4.2, col(F[i]), wrong ? colors.ink : null);
        });
        D.forEach((p, i) => {
          if (!L[i]) return;
          g.lineWidth = 2.5;
          dotAt(g, X(p[0]), PY(p[1]), 7, cs[Y[i]], colors.ink);
        });
        const ds = D.map((p, j) => [Math.hypot(tp.v[0] - p[0], tp.v[1] - p[1]), j])
            .sort((a, b) => a[0] - b[0])
            .slice(0, settings.k),
          f = 0.5 + clamp((ds.reduce((s, e) => s + F[e[1]], 0) / ds.length - TH) / SP, -1, 1) / 2,
          lab = D.map((_, i) => (L[i] ? Y[i] : -1)),
          bl = Y[nearLab(tp.v, lab)];
        markTP(g, X(tp.v[0]), PY(tp.v[1]), colors);
        tp.out.textContent =
          'Your point: the ' +
          settings.k +
          ' nearest data points believe "' +
          (f > 0.5 ? 'gold' : 'green') +
          '" with strength ' +
          pct(Math.abs(f - 0.5) * 2) +
          '. The nearest labelled point alone says "' +
          (bl ? 'gold' : 'green') +
          '".';
        g2Note(g, colors, h, fin ? 'black outline: classified wrongly' : 'ringed: labelled');
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.56,
          sc = Math.min(w / 8.4, (h1 - 20) / 6),
          X = (x) => w / 2 + x * sc,
          Yc = (y) => 10 + (h1 - 10) / 2 - y * sc;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('labels alone: nearest labelled point decides', 8, 13);
        D.forEach((p, i) => {
          ctx.lineWidth = 1.5;
          if (L[i]) dotAt(ctx, X(p[0]), Yc(p[1]), 5, cs[Y[i]], colors.ink);
          else
            dotAt(
              ctx,
              X(p[0]),
              Yc(p[1]),
              2.8,
              cs[baselinePredictions[i]],
              baselinePredictions[i] !== Y[i] ? colors.ink : null,
            );
        });
        const A = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, hist, {
          title: 'accuracy on the unlabelled points',
          n: Math.max(20, hist.length),
          y1: 0.4,
          ym: 1,
        });
        ctx.strokeStyle = colors.cut;
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(A.X(0), A.Y(Math.max(0.4, baselineAccuracy)));
        ctx.lineTo(A.X(Math.max(20, hist.length) - 1), A.Y(Math.max(0.4, baselineAccuracy)));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        hist.forEach((v, i) =>
          i ? ctx.lineTo(A.X(i), A.Y(Math.max(0.4, v))) : ctx.moveTo(A.X(i), A.Y(Math.max(0.4, v))),
        );
        ctx.stroke();
      }
      const nl = L.filter(Boolean).length;
      live.textContent = msg;
      highlightLessonStep(root, 'ss', activeStepIndex);
      setReadout(root, 'lab', nl + ' / ' + (D.length - nl));
      setReadout(root, 'it', it + (fin ? ' (stopped)' : ''));
      setReadout(root, 'base', pct(baselineAccuracy));
      setReadout(root, 'acc', pct(hist[hist.length - 1]));
      setReadout(root, 'und', String(cur().filter((v, i) => v < 0 && !L[i]).length));
    };
    tp.on = () => draw();
    gen();
    reset();
    if (prefersReducedMotion) {
      while (!fin) step();
    }
    let frameCount = 0;
    const anim = Anim(() => {
      if (fin) {
        playback.stop();
        return;
      }
      frameCount++;
      if (settings.m === 'self') {
        if (frameCount % 6) return;
        step();
      } else {
        const k = it < 30 ? 1 : it < 120 ? 3 : it < 400 ? 10 : 30;
        if (it < 30 && frameCount % 3) return;
        for (let i = 0; i < k && !fin && checkRunStop.remaining() > 0; i++) step();
      }
      draw();
      checkRunStop();
    });
    playback = playBtn($('#ss-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'ss', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'steps',
    });
    const go = () => {
      playback.stop();
      reset();
      draw();
      frameCount = 0;
      playback.start();
    };
    $('#ss-step', root).onclick = () => {
      playback.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#ss-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    $('#ss-new', root).onclick = () => {
      seed++;
      gen();
      go();
    };
    bind(
      root,
      'ss-n',
      (v) => {
        settings.n = v;
        gen();
        go();
      },
      (v) => String(+v),
    );
    bind(
      root,
      'ss-k',
      (v) => {
        settings.k = v;
        go();
      },
      (v) => String(+v),
    );
    segBind(root, 'ss-d', settings.d, (v) => {
      settings.d = v;
      gen();
      go();
    });
    segBind(root, 'ss-m', settings.m, (v) => {
      settings.m = v;
      go();
    });
    c.style.cursor = 'pointer';
    c.addEventListener('click', (e) => {
      if (!V) return;
      const [x, y] = canvasXY(c, V, e);
      let bi = -1,
        bd = 14;
      D.forEach((p, i) => {
        const d = Math.hypot((p[0] - x) * V.s, (p[1] - y) * V.s);
        if (d < bd) {
          bd = d;
          bi = i;
        }
      });
      if (bi >= 0 && !L[bi]) {
        L[bi] = true;
        go();
      } else if (bi < 0) {
        tp.set([clamp(x, -4, 4), clamp(y, -3, 3)]);
        draw();
      }
    });
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        reset();
        draw();
        frameCount = 0;
        playback.start();
      },
    };
  },
};
