/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { dotAt, segBind, uCur, uLine, xyUp } from '../../unsupervised/shared.js';
import { laInv } from '../../../js/math/linear-algebra.js';

export default {
  init(root) {
    const c = $('#rs-c', root),
      cr = $('#rs-r', root),
      live = $('#rs-live', root),
      S = { d: 'good', m: 1, log: false, drop: false };
    let D = [],
      co = [0, 0, 0],
      E = [],
      F = [],
      u = 0,
      dir = 1,
      seed = 1,
      msg = '',
      cu = 0,
      pb,
      st = {},
      xr = [0, 10],
      dropI = -1,
      up = false;
    const gen = () => {
      const r = rng(seed * 7 + S.d.length * 3),
        rn = randnFrom(r);
      D = [];
      up = false;
      for (let i = 0; i < 60; i++) {
        const x = 0.4 + 9.2 * r();
        D.push([
          x,
          S.d === 'good'
            ? 6 + 2.4 * x + 2.6 * rn()
            : S.d === 'curve'
              ? 5 + 0.4 * x + 0.26 * x * x + 1.6 * rn()
              : S.d === 'funnel'
                ? Math.exp(0.9 + 0.28 * x + 0.38 * rn())
                : 6 + 2.4 * x + 2 * rn(),
        ]);
      }
      if (S.d === 'out') {
        D[0] = [9.5, 3];
      }
    };
    const fitM = () => {
      const ty = (v) => (S.log ? Math.log(Math.max(v, 1e-6)) : v);
      let mx = 0,
        k = -1;
      if (S.drop && E.length === D.length) {
        E.forEach((e, i) => {
          if (Math.abs(e) > mx) {
            mx = Math.abs(e);
            k = i;
          }
        });
      }
      dropI = S.drop ? (dropI >= 0 ? dropI : k) : -1;
      const p = S.m + 1,
        A = Array.from({ length: p }, () => new Array(p).fill(0)),
        bv = new Array(p).fill(0);
      D.forEach(([x, y], i) => {
        if (i === dropI) return;
        const ph = [1, x, x * x].slice(0, p);
        for (let a = 0; a < p; a++) {
          bv[a] += ph[a] * ty(y);
          for (let b = 0; b < p; b++) A[a][b] += ph[a] * ph[b];
        }
      });
      const iv = laInv(A);
      co = [0, 0, 0];
      for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) co[a] += iv[a][b] * bv[b];
      F = D.map(([x]) => co[0] + co[1] * x + co[2] * x * x);
      E = D.map(([x, y], i) => ty(y) - F[i]);
      const n = D.length,
        keep = E.filter((_, i) => i !== dropI),
        ys = D.filter((_, i) => i !== dropI).map((d) => ty(d[1])),
        my = ys.reduce((s, v) => s + v, 0) / ys.length,
        sse = keep.reduce((s, v) => s + v * v, 0),
        sst = ys.reduce((s, v) => s + (v - my) ** 2, 0),
        mf = F.reduce((s, v) => s + v, 0) / n,
        cor = (a, b) => {
          const ma = a.reduce((s, v) => s + v, 0) / a.length,
            mb = b.reduce((s, v) => s + v, 0) / b.length;
          let sab = 0,
            sa = 0,
            sb = 0;
          a.forEach((v, i) => {
            sab += (v - ma) * (b[i] - mb);
            sa += (v - ma) ** 2;
            sb += (b[i] - mb) ** 2;
          });
          return sab / Math.sqrt(sa * sb || 1);
        };
      const srt = E.map(Math.abs).sort((a, b) => a - b),
        mad = srt[Math.floor(n / 2)] / 0.6745 || 1e-9,
        idx = E.map((_, i) => i).filter((i) => i !== dropI && Math.abs(E[i]) / mad < 3.5),
        e2 = idx.map((i) => E[i]),
        f2_ = idx.map((i) => F[i]);
      st = {
        r2: 1 - sse / sst,
        bend: cor(
          e2,
          f2_.map((v) => (v - mf) ** 2),
        ),
        fun: cor(e2.map(Math.abs), f2_),
        big: Math.max(...E.map((e, i) => (i === dropI ? 0 : Math.abs(e)))) / mad,
        mad,
        sd: Math.sqrt(sse / keep.length),
      };
      const pr = [];
      if (Math.abs(st.bend) > 0.35) pr.push('a bend');
      if (st.fun > 0.35) pr.push('a funnel');
      if (st.big > 3.5) pr.push('a straggler');
      st.ver = pr.length
        ? 'The residuals show ' + pr.join(' and ') + '.'
        : 'No pattern: the leftovers look like noise.';
    };
    const refit = () => {
      dropI = -1;
      if (S.drop) {
        const sv = S.drop;
        S.drop = false;
        fitM();
        S.drop = sv;
      }
      fitM();
      u = 0;
      dir = 1;
      cu = 0;
      msg =
        'Fitted. ' + st.ver + ' Press Run to flatten the line and see the residuals on their own.';
    };
    const tp = testPoint(root, 'rs', ['Your x'], [5], [[0, 12]]);
    const draw = () => {
      const t = T(),
        ty = (v) => (S.log ? Math.log(Math.max(v, 1e-6)) : v),
        ys = D.map((d) => ty(d[1])),
        lo = Math.min(...ys, ...F),
        hi = Math.max(...ys, ...F),
        em = Math.max(...E.map(Math.abs)) * 1.15,
        s = u * u * (3 - 2 * u),
        ya = lo - (hi - lo) * 0.06 + (-em - (lo - (hi - lo) * 0.06)) * s,
        yb = hi + (hi - lo) * 0.06 + (em - (hi + (hi - lo) * 0.06)) * s;
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, xr, [ya, yb], t, {
            xstep: 2,
            xlabel: s > 0.5 ? 'x (the fitted model is now the flat line)' : 'x',
          });
        clipPlot(g, P);
        g.strokeStyle = t.accent;
        g.lineWidth = 2.6;
        curve(g, P, (x) => (co[0] + co[1] * x + co[2] * x * x) * (1 - s), xr[0], xr[1], 60);
        g.stroke();
        D.forEach(([x, y], i) => {
          const f = F[i] * (1 - s),
            v = ty(y) - F[i] * s;
          g.strokeStyle = mix(t.cut, t.cut, 0, 0.55);
          g.lineWidth = 1.2;
          g.beginPath();
          g.moveTo(P.X(x), P.Y(f));
          g.lineTo(P.X(x), P.Y(v));
          g.stroke();
          g.lineWidth = 1.5;
          dotAt(
            g,
            P.X(x),
            P.Y(v),
            i === dropI ? 5 : 3.6,
            i === dropI ? null : mix(t.ink, t.ink, 0, 0.7),
            i === dropI ? t.cut : null,
          );
        });
        const xv = clamp(tp.v[0], xr[0], xr[1]),
          pm = co[0] + co[1] * xv + co[2] * xv * xv;
        if (s < 0.5) markTP(g, P.X(xv), P.Y(pm * (1 - s)), t, false);
        g.restore();
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          s > 0.5 ? 'residual: observed − predicted' : S.log ? 'log of y' : 'y',
          P.m.l + 6,
          P.m.t + 8,
        );
        const bk = (v) => (S.log ? Math.exp(v) : v),
          warn = Math.abs(st.bend) > 0.35 || st.fun > 0.35;
        tp.out.textContent =
          'At x = ' +
          f2(xv) +
          ' the model predicts ' +
          f2(bk(pm)) +
          ', give or take about ' +
          (S.log ? 'a factor of ' + f2(Math.exp(2 * st.sd)) : f2(2 * st.sd)) +
          '.' +
          (warn
            ? ' Do not trust that margin: the residuals show the model is missing something, so its errors are not the same everywhere.'
            : '');
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5,
          NB = 13,
          hs = new Array(NB).fill(0),
          R = 3.5 * st.sd || 1;
        E.forEach((e) => {
          const k = Math.floor(((e + R) / (2 * R)) * NB);
          if (k >= 0 && k < NB) hs[k]++;
        });
        const hm = Math.max(...hs, 1),
          A = uLine(g, t, 0, 0, w, h1, [], {
            title: 'residuals: how many of each size (bars) and a bell curve',
            n: NB + 1,
            ym: hm * 1.1,
            y1: 0,
          }),
          bw = A.X(1) - A.X(0);
        hs.forEach((v, k) => {
          g.fillStyle = mix(t.accent, t.accent, 0, 0.6);
          g.fillRect(A.X(k) + 1, A.Y(v), bw - 2, A.Y(0) - A.Y(v));
        });
        g.strokeStyle = t.cut;
        g.lineWidth = 2;
        g.beginPath();
        for (let i = 0; i <= 60; i++) {
          const e = -R + (2 * R * i) / 60,
            v =
              (D.length * ((2 * R) / NB) * Math.exp((-e * e) / (2 * st.sd * st.sd))) /
              (st.sd * Math.sqrt(2 * Math.PI));
          i ? g.lineTo(A.X((i / 60) * NB), A.Y(v)) : g.moveTo(A.X(0), A.Y(v));
        }
        g.stroke();
        const fl = Math.min(...F),
          fh = Math.max(...F),
          am = Math.max(...E.map(Math.abs)) * 1.1,
          B = uLine(g, t, 0, h1, w, h - h1, [], {
            title: 'size of residual against the prediction (small → large)',
            n: 101,
            ym: am,
            y1: 0,
          });
        D.forEach((_, i) =>
          dotAt(
            g,
            B.X(((F[i] - fl) / (fh - fl || 1)) * 100),
            B.Y(Math.abs(E[i])),
            3,
            mix(t.gold, t.gold, 0, 0.75),
          ),
        );
        let sx = 0,
          sy = 0,
          sxx = 0,
          sxy = 0;
        const n = D.length;
        D.forEach((_, i) => {
          const x = ((F[i] - fl) / (fh - fl || 1)) * 100,
            y = Math.abs(E[i]);
          sx += x;
          sy += y;
          sxx += x * x;
          sxy += x * y;
        });
        const b = (n * sxy - sx * sy) / (n * sxx - sx * sx),
          a = (sy - b * sx) / n;
        g.strokeStyle = t.cut;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(B.X(0), B.Y(Math.max(0, a)));
        g.lineTo(B.X(100), B.Y(Math.max(0, a + 100 * b)));
        g.stroke();
      }
      live.textContent = msg;
      uCur(root, 'rs', cu);
      setR(root, 'r2', f3(st.r2));
      setR(root, 'bend', f2(st.bend) + (Math.abs(st.bend) > 0.35 ? ' (flagged)' : ''));
      setR(root, 'fun', f2(st.fun) + (st.fun > 0.35 ? ' (flagged)' : ''));
      setR(root, 'big', f2(st.big) + (st.big > 3.5 ? ' (flagged)' : ''));
      setR(root, 'ver', st.ver);
    };
    tp.on = () => draw();
    gen();
    refit();
    if (RM) u = 1;
    const anim = Anim(() => {
      u = clamp(u + dir * 0.012, 0, 1);
      if (u >= 1 || u <= 0) {
        pb.stop();
        cu = u >= 1 ? (Math.abs(st.bend) > 0.35 ? 3 : st.fun > 0.35 ? 4 : st.big > 3.5 ? 5 : 2) : 0;
        msg =
          u >= 1
            ? 'Flattened. This is the residual plot. ' + st.ver
            : 'Back to the data and the fitted model.';
      }
      draw();
    });
    pb = playBtn($('#rs-play', root), anim);
    {
      const s0 = pb.start;
      pb.start = () => {
        if (u >= 1) {
          u = 0;
        }
        dir = 1;
        s0();
      };
    }
    $('#rs-step', root).onclick = () => {
      pb.stop();
      u = u < 0.5 ? 1 : 0;
      cu = u ? 2 : 0;
      msg = u ? 'The residual plot. ' + st.ver : 'The data and the fitted model.';
      draw();
    };
    $('#rs-reset', root).onclick = () => {
      pb.stop();
      refit();
      draw();
    };
    $('#rs-new', root).onclick = () => {
      pb.stop();
      seed++;
      gen();
      refit();
      draw();
      pb.start();
    };
    const re = () => {
      pb.stop();
      refit();
      draw();
      pb.start();
    };
    segBind(root, 'rs-d', S.d, (v) => {
      S.d = v;
      gen();
      re();
    });
    segBind(root, 'rs-m', '1', (v) => {
      S.m = +v;
      re();
    });
    $('#rs-log', root).onchange = (e) => {
      S.log = e.target.checked;
      re();
    };
    $('#rs-drop', root).onchange = (e) => {
      S.drop = e.target.checked;
      re();
    };
    xyUp(
      root,
      'rs',
      (q) => {
        const x0 = Math.min(...q.map((d) => d[0])),
          x1 = Math.max(...q.map((d) => d[0]));
        D = q.map((d) => [0.4 + (9.2 * (d[0] - x0)) / (x1 - x0 || 1), d[1]]);
        if (S.log && D.some((d) => d[1] <= 0)) {
          S.log = false;
          $('#rs-log', root).checked = false;
        }
        re();
      },
      false,
    );
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        u = 0;
        draw();
        pb.start();
      },
    };
  },
};
