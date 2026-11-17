/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { canvasXY, dotAt, segBind, uCur, uLine, uView } from '../../unsupervised/shared.js';
import { g2Note } from '../../generative/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#ss-c', root),
      cr = $('#ss-r', root),
      live = $('#ss-live', root),
      ol = $('#ss-steps', root),
      S = { d: 'moons', m: 'lp', n: 3, k: 7 };
    let D = [],
      Y = [],
      L = [],
      F = [],
      nb = [],
      base = [],
      hist = [],
      it = 0,
      fin = false,
      seed = 1,
      msg = '',
      cu = 0,
      pb,
      chk = () => false,
      V = null,
      bacc = 0,
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
      const r = rng(seed * 37 + S.d.length),
        rn = randnFrom(r);
      D = [];
      Y = [];
      for (let i = 0; i < 180; i++) {
        const k = i % 2;
        let p;
        if (S.d === 'moons') {
          const a = Math.PI * r();
          p = k
            ? [1.7 * Math.cos(a) - 0.85 + 0.1 * rn(), 1.7 * Math.sin(a) - 0.6 + 0.1 * rn()]
            : [0.85 - 1.7 * Math.cos(a) + 0.1 * rn(), 0.6 - 1.7 * Math.sin(a) + 0.1 * rn()];
        } else if (S.d === 'mixed')
          p = [(k ? 0.75 : -0.75) + 0.85 * rn(), (k ? 0.3 : -0.3) + 0.8 * rn()];
        else p = [(k ? 1.9 : -1.9) + 0.55 * rn(), (k ? 0.5 : -0.5) + 0.6 * rn()];
        D.push(p);
        Y.push(k);
      }
      if (S.d === 'bridge')
        for (let i = 0; i < 14; i++) {
          const u = (i + 0.5) / 14;
          D.push([-1.3 + 2.6 * u + 0.05 * rn(), -0.35 + 0.7 * u + 0.05 * rn()]);
          Y.push(u > 0.5 ? 1 : 0);
        }
      L = D.map(() => false);
      for (const k of [0, 1]) {
        const idx = D.map((_, i) => i).filter((i) => Y[i] === k && i < 180);
        for (let q = 0; q < S.n; q++) {
          const j = Math.floor(r() * idx.length);
          L[idx.splice(j, 1)[0]] = true;
        }
      }
    };
    const graph = () => {
      nb = D.map((p, i) => {
        const ds = D.map((q, j) => [Math.hypot(p[0] - q[0], p[1] - q[1]), j])
            .filter((e) => e[1] !== i)
            .sort((a, b) => a[0] - b[0])
            .slice(0, S.k),
          sg = ds[ds.length - 1][0] || 1;
        return ds.map(([d, j]) => [j, Math.exp((-d * d) / (sg * sg))]);
      });
      const sym = D.map(() => new Map());
      nb.forEach((a, i) =>
        a.forEach(([j, w]) => {
          sym[i].set(j, Math.max(w, sym[i].get(j) || 0));
          sym[j].set(i, Math.max(w, sym[j].get(i) || 0));
        }),
      );
      nb = sym.map((m) => {
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
      if (S.m !== 'lp') return [0.5, 0.5];
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
      ol.innerHTML = HOW[S.m].map((s) => '<li>' + s + '</li>').join('');
      graph();
      const lab = D.map((_, i) => (L[i] ? Y[i] : -1));
      base = D.map((p, i) => (L[i] ? Y[i] : Y[nearLab(p, lab)]));
      bacc = accOf(base);
      F = D.map((_, i) => (L[i] ? Y[i] : 0.5));
      pseudo = lab.slice();
      it = 0;
      fin = false;
      hist = [accOf(cur())];
      cu = 1;
      msg =
        L.filter(Boolean).length +
        ' points carry a label; ' +
        (D.length - L.filter(Boolean).length) +
        ' do not. Using the labels alone, ' +
        pct(bacc) +
        ' of the unlabelled points would be classified correctly. Press Run.';
    };
    const step = () => {
      if (fin) return;
      if (S.m === 'lp') {
        const N = F.slice();
        let mx = 0;
        for (let i = 0; i < D.length; i++) {
          if (L[i]) continue;
          let s = 0;
          for (const [j, w] of nb[i]) s += w * F[j];
          N[i] = s;
          mx = Math.max(mx, Math.abs(s - F[i]));
        }
        F = N;
        it++;
        cu = it % 2 ? 2 : 3;
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
        cu = 3;
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
        cu = 5;
        const a = hist[hist.length - 1];
        msg =
          'Stopped after ' +
          it +
          ' steps: ' +
          (S.m === 'lp' ? 'the beliefs have stopped changing.' : 'every point has a label.') +
          ' Accuracy on the unlabelled points: ' +
          pct(a) +
          ', against ' +
          pct(bacc) +
          ' from the labels alone. ' +
          (a > bacc + 0.03
            ? 'The unlabelled data helped.'
            : a < bacc - 0.03
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
      const t = T(),
        cs = [t.accent, t.gold],
        [TH, SP] = thr(),
        col = (f) =>
          f > TH
            ? mix(t.muted, cs[1], clamp(((f - TH) / SP) * 1.5, 0, 1))
            : mix(t.muted, cs[0], clamp(((TH - f) / SP) * 1.5, 0, 1));
      V = uView(c);
      {
        const { g, w, h, X, Y: PY } = V;
        if (S.m === 'lp' && D.length < 260) {
          g.strokeStyle = mix(t.ink, t.ink, 0, 0.07);
          g.lineWidth = 1;
          g.beginPath();
          nb.forEach((a, i) =>
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
          dotAt(g, X(p[0]), PY(p[1]), 4.2, col(F[i]), wrong ? t.ink : null);
        });
        D.forEach((p, i) => {
          if (!L[i]) return;
          g.lineWidth = 2.5;
          dotAt(g, X(p[0]), PY(p[1]), 7, cs[Y[i]], t.ink);
        });
        const ds = D.map((p, j) => [Math.hypot(tp.v[0] - p[0], tp.v[1] - p[1]), j])
            .sort((a, b) => a[0] - b[0])
            .slice(0, S.k),
          f = 0.5 + clamp((ds.reduce((s, e) => s + F[e[1]], 0) / ds.length - TH) / SP, -1, 1) / 2,
          lab = D.map((_, i) => (L[i] ? Y[i] : -1)),
          bl = Y[nearLab(tp.v, lab)];
        markTP(g, X(tp.v[0]), PY(tp.v[1]), t);
        tp.out.textContent =
          'Your point: the ' +
          S.k +
          ' nearest data points believe "' +
          (f > 0.5 ? 'gold' : 'green') +
          '" with strength ' +
          pct(Math.abs(f - 0.5) * 2) +
          '. The nearest labelled point alone says "' +
          (bl ? 'gold' : 'green') +
          '".';
        g2Note(g, t, h, fin ? 'black outline: classified wrongly' : 'ringed: labelled');
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.56,
          sc = Math.min(w / 8.4, (h1 - 20) / 6),
          X = (x) => w / 2 + x * sc,
          Yc = (y) => 10 + (h1 - 10) / 2 - y * sc;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('labels alone: nearest labelled point decides', 8, 13);
        D.forEach((p, i) => {
          g.lineWidth = 1.5;
          if (L[i]) dotAt(g, X(p[0]), Yc(p[1]), 5, cs[Y[i]], t.ink);
          else dotAt(g, X(p[0]), Yc(p[1]), 2.8, cs[base[i]], base[i] !== Y[i] ? t.ink : null);
        });
        const A = uLine(g, t, 0, h1, w, h - h1, hist, {
          title: 'accuracy on the unlabelled points',
          n: Math.max(20, hist.length),
          y1: 0.4,
          ym: 1,
        });
        g.strokeStyle = t.cut;
        g.setLineDash([6, 4]);
        g.lineWidth = 1.6;
        g.beginPath();
        g.moveTo(A.X(0), A.Y(Math.max(0.4, bacc)));
        g.lineTo(A.X(Math.max(20, hist.length) - 1), A.Y(Math.max(0.4, bacc)));
        g.stroke();
        g.setLineDash([]);
        g.strokeStyle = t.accent;
        g.lineWidth = 2.2;
        g.beginPath();
        hist.forEach((v, i) =>
          i ? g.lineTo(A.X(i), A.Y(Math.max(0.4, v))) : g.moveTo(A.X(i), A.Y(Math.max(0.4, v))),
        );
        g.stroke();
      }
      const nl = L.filter(Boolean).length;
      live.textContent = msg;
      uCur(root, 'ss', cu);
      setR(root, 'lab', nl + ' / ' + (D.length - nl));
      setR(root, 'it', it + (fin ? ' (stopped)' : ''));
      setR(root, 'base', pct(bacc));
      setR(root, 'acc', pct(hist[hist.length - 1]));
      setR(root, 'und', String(cur().filter((v, i) => v < 0 && !L[i]).length));
    };
    tp.on = () => draw();
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
      fr++;
      if (S.m === 'self') {
        if (fr % 6) return;
        step();
      } else {
        const k = it < 30 ? 1 : it < 120 ? 3 : it < 400 ? 10 : 30;
        if (it < 30 && fr % 3) return;
        for (let i = 0; i < k && !fin && chk.remaining() > 0; i++) step();
      }
      draw();
      chk();
    });
    pb = playBtn($('#ss-play', root), anim);
    chk = runCtl(pb, root, 'ss', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'steps',
    });
    const go = () => {
      pb.stop();
      reset();
      draw();
      fr = 0;
      pb.start();
    };
    $('#ss-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#ss-reset', root).onclick = () => {
      pb.stop();
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
        S.n = v;
        gen();
        go();
      },
      (v) => String(+v),
    );
    bind(
      root,
      'ss-k',
      (v) => {
        S.k = v;
        go();
      },
      (v) => String(+v),
    );
    segBind(root, 'ss-d', S.d, (v) => {
      S.d = v;
      gen();
      go();
    });
    segBind(root, 'ss-m', S.m, (v) => {
      S.m = v;
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
