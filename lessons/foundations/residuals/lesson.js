/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  clamp,
  clipPlot,
  curve,
  f2,
  f3,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import {
  dotAt,
  segBind,
  highlightLessonStep,
  drawSeriesChart,
  xyUp,
} from '../../unsupervised/shared.js';
import { laInv } from '../../../js/math/linear-algebra.js';

export default {
  init(root) {
    const mainCanvas = $('#rs-c', root),
      cr = $('#rs-r', root),
      live = $('#rs-live', root),
      settings = { d: 'good', m: 1, log: false, drop: false };
    let D = [],
      co = [0, 0, 0],
      residuals = [],
      fittedValues = [],
      u = 0,
      dir = 1,
      seed = 1,
      msg = '',
      activeStepIndex = 0,
      playback,
      residualDiagnostics = {},
      xr = [0, 10],
      dropI = -1,
      up = false;
    const gen = () => {
      const r = rng(seed * 7 + settings.d.length * 3),
        rn = randnFrom(r);
      D = [];
      up = false;
      for (let i = 0; i < 60; i++) {
        const x = 0.4 + 9.2 * r();
        D.push([
          x,
          settings.d === 'good'
            ? 6 + 2.4 * x + 2.6 * rn()
            : settings.d === 'curve'
              ? 5 + 0.4 * x + 0.26 * x * x + 1.6 * rn()
              : settings.d === 'funnel'
                ? Math.exp(0.9 + 0.28 * x + 0.38 * rn())
                : 6 + 2.4 * x + 2 * rn(),
        ]);
      }
      if (settings.d === 'out') {
        D[0] = [9.5, 3];
      }
    };
    const fitM = () => {
      const ty = (v) => (settings.log ? Math.log(Math.max(v, 1e-6)) : v);
      let mx = 0,
        k = -1;
      if (settings.drop && residuals.length === D.length) {
        residuals.forEach((e, i) => {
          if (Math.abs(e) > mx) {
            mx = Math.abs(e);
            k = i;
          }
        });
      }
      dropI = settings.drop ? (dropI >= 0 ? dropI : k) : -1;
      const p = settings.m + 1,
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
      fittedValues = D.map(([x]) => co[0] + co[1] * x + co[2] * x * x);
      residuals = D.map(([x, y], i) => ty(y) - fittedValues[i]);
      const n = D.length,
        keep = residuals.filter((_, i) => i !== dropI),
        ys = D.filter((_, i) => i !== dropI).map((d) => ty(d[1])),
        my = ys.reduce((s, v) => s + v, 0) / ys.length,
        sse = keep.reduce((s, v) => s + v * v, 0),
        sst = ys.reduce((s, v) => s + (v - my) ** 2, 0),
        mf = fittedValues.reduce((s, v) => s + v, 0) / n,
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
      const srt = residuals.map(Math.abs).sort((a, b) => a - b),
        mad = srt[Math.floor(n / 2)] / 0.6745 || 1e-9,
        idx = residuals
          .map((_, i) => i)
          .filter((i) => i !== dropI && Math.abs(residuals[i]) / mad < 3.5),
        e2 = idx.map((i) => residuals[i]),
        f2_ = idx.map((i) => fittedValues[i]);
      residualDiagnostics = {
        r2: 1 - sse / sst,
        bend: cor(
          e2,
          f2_.map((v) => (v - mf) ** 2),
        ),
        fun: cor(e2.map(Math.abs), f2_),
        big: Math.max(...residuals.map((e, i) => (i === dropI ? 0 : Math.abs(e)))) / mad,
        mad,
        sd: Math.sqrt(sse / keep.length),
      };
      const pr = [];
      if (Math.abs(residualDiagnostics.bend) > 0.35) pr.push('a bend');
      if (residualDiagnostics.fun > 0.35) pr.push('a funnel');
      if (residualDiagnostics.big > 3.5) pr.push('a straggler');
      residualDiagnostics.ver = pr.length
        ? 'The residuals show ' + pr.join(' and ') + '.'
        : 'No pattern: the leftovers look like noise.';
    };
    const refit = () => {
      dropI = -1;
      if (settings.drop) {
        const sv = settings.drop;
        settings.drop = false;
        fitM();
        settings.drop = sv;
      }
      fitM();
      u = 0;
      dir = 1;
      activeStepIndex = 0;
      msg =
        'Fitted. ' +
        residualDiagnostics.ver +
        ' Press Run to flatten the line and see the residuals on their own.';
    };
    const tp = testPoint(root, 'rs', ['Your x'], [5], [[0, 12]]);
    const draw = () => {
      const colors = readThemeColors(),
        ty = (v) => (settings.log ? Math.log(Math.max(v, 1e-6)) : v),
        ys = D.map((d) => ty(d[1])),
        lo = Math.min(...ys, ...fittedValues),
        hi = Math.max(...ys, ...fittedValues),
        em = Math.max(...residuals.map(Math.abs)) * 1.15,
        s = u * u * (3 - 2 * u),
        ya = lo - (hi - lo) * 0.06 + (-em - (lo - (hi - lo) * 0.06)) * s,
        yb = hi + (hi - lo) * 0.06 + (em - (hi + (hi - lo) * 0.06)) * s;
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = plot(ctx, w, h, xr, [ya, yb], colors, {
            xstep: 2,
            xlabel: s > 0.5 ? 'x (the fitted model is now the flat line)' : 'x',
          });
        clipPlot(ctx, P);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.6;
        curve(ctx, P, (x) => (co[0] + co[1] * x + co[2] * x * x) * (1 - s), xr[0], xr[1], 60);
        ctx.stroke();
        D.forEach(([x, y], i) => {
          const f = fittedValues[i] * (1 - s),
            v = ty(y) - fittedValues[i] * s;
          ctx.strokeStyle = mix(colors.cut, colors.cut, 0, 0.55);
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(P.X(x), P.Y(f));
          ctx.lineTo(P.X(x), P.Y(v));
          ctx.stroke();
          ctx.lineWidth = 1.5;
          dotAt(
            ctx,
            P.X(x),
            P.Y(v),
            i === dropI ? 5 : 3.6,
            i === dropI ? null : mix(colors.ink, colors.ink, 0, 0.7),
            i === dropI ? colors.cut : null,
          );
        });
        const xv = clamp(tp.v[0], xr[0], xr[1]),
          pm = co[0] + co[1] * xv + co[2] * xv * xv;
        if (s < 0.5) markTP(ctx, P.X(xv), P.Y(pm * (1 - s)), colors, false);
        ctx.restore();
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          s > 0.5 ? 'residual: observed − predicted' : settings.log ? 'log of y' : 'y',
          P.m.l + 6,
          P.m.t + 8,
        );
        const bk = (v) => (settings.log ? Math.exp(v) : v),
          warn = Math.abs(residualDiagnostics.bend) > 0.35 || residualDiagnostics.fun > 0.35;
        tp.out.textContent =
          'At x = ' +
          f2(xv) +
          ' the model predicts ' +
          f2(bk(pm)) +
          ', give or take about ' +
          (settings.log
            ? 'a factor of ' + f2(Math.exp(2 * residualDiagnostics.sd))
            : f2(2 * residualDiagnostics.sd)) +
          '.' +
          (warn
            ? ' Do not trust that margin: the residuals show the model is missing something, so its errors are not the same everywhere.'
            : '');
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          NB = 13,
          hs = new Array(NB).fill(0),
          R = 3.5 * residualDiagnostics.sd || 1;
        residuals.forEach((e) => {
          const k = Math.floor(((e + R) / (2 * R)) * NB);
          if (k >= 0 && k < NB) hs[k]++;
        });
        const hm = Math.max(...hs, 1),
          A = drawSeriesChart(ctx, colors, 0, 0, w, h1, [], {
            title: 'residuals: how many of each size (bars) and a bell curve',
            n: NB + 1,
            ym: hm * 1.1,
            y1: 0,
          }),
          bw = A.X(1) - A.X(0);
        hs.forEach((v, k) => {
          ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.6);
          ctx.fillRect(A.X(k) + 1, A.Y(v), bw - 2, A.Y(0) - A.Y(v));
        });
        ctx.strokeStyle = colors.cut;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i <= 60; i++) {
          const e = -R + (2 * R * i) / 60,
            v =
              (D.length *
                ((2 * R) / NB) *
                Math.exp((-e * e) / (2 * residualDiagnostics.sd * residualDiagnostics.sd))) /
              (residualDiagnostics.sd * Math.sqrt(2 * Math.PI));
          i ? ctx.lineTo(A.X((i / 60) * NB), A.Y(v)) : ctx.moveTo(A.X(0), A.Y(v));
        }
        ctx.stroke();
        const fl = Math.min(...fittedValues),
          fh = Math.max(...fittedValues),
          am = Math.max(...residuals.map(Math.abs)) * 1.1,
          B = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, [], {
            title: 'size of residual against the prediction (small → large)',
            n: 101,
            ym: am,
            y1: 0,
          });
        D.forEach((_, i) =>
          dotAt(
            ctx,
            B.X(((fittedValues[i] - fl) / (fh - fl || 1)) * 100),
            B.Y(Math.abs(residuals[i])),
            3,
            mix(colors.gold, colors.gold, 0, 0.75),
          ),
        );
        let sx = 0,
          sy = 0,
          sxx = 0,
          sxy = 0;
        const n = D.length;
        D.forEach((_, i) => {
          const x = ((fittedValues[i] - fl) / (fh - fl || 1)) * 100,
            y = Math.abs(residuals[i]);
          sx += x;
          sy += y;
          sxx += x * x;
          sxy += x * y;
        });
        const b = (n * sxy - sx * sy) / (n * sxx - sx * sx),
          a = (sy - b * sx) / n;
        ctx.strokeStyle = colors.cut;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(B.X(0), B.Y(Math.max(0, a)));
        ctx.lineTo(B.X(100), B.Y(Math.max(0, a + 100 * b)));
        ctx.stroke();
      }
      live.textContent = msg;
      highlightLessonStep(root, 'rs', activeStepIndex);
      setReadout(root, 'r2', f3(residualDiagnostics.r2));
      setReadout(
        root,
        'bend',
        f2(residualDiagnostics.bend) +
          (Math.abs(residualDiagnostics.bend) > 0.35 ? ' (flagged)' : ''),
      );
      setReadout(
        root,
        'fun',
        f2(residualDiagnostics.fun) + (residualDiagnostics.fun > 0.35 ? ' (flagged)' : ''),
      );
      setReadout(
        root,
        'big',
        f2(residualDiagnostics.big) + (residualDiagnostics.big > 3.5 ? ' (flagged)' : ''),
      );
      setReadout(root, 'ver', residualDiagnostics.ver);
    };
    tp.on = () => draw();
    gen();
    refit();
    if (prefersReducedMotion) u = 1;
    const anim = Anim(() => {
      u = clamp(u + dir * 0.012, 0, 1);
      if (u >= 1 || u <= 0) {
        playback.stop();
        activeStepIndex =
          u >= 1
            ? Math.abs(residualDiagnostics.bend) > 0.35
              ? 3
              : residualDiagnostics.fun > 0.35
                ? 4
                : residualDiagnostics.big > 3.5
                  ? 5
                  : 2
            : 0;
        msg =
          u >= 1
            ? 'Flattened. This is the residual plot. ' + residualDiagnostics.ver
            : 'Back to the data and the fitted model.';
      }
      draw();
    });
    playback = playBtn($('#rs-play', root), anim);
    {
      const s0 = playback.start;
      playback.start = () => {
        if (u >= 1) {
          u = 0;
        }
        dir = 1;
        s0();
      };
    }
    $('#rs-step', root).onclick = () => {
      playback.stop();
      u = u < 0.5 ? 1 : 0;
      activeStepIndex = u ? 2 : 0;
      msg = u ? 'The residual plot. ' + residualDiagnostics.ver : 'The data and the fitted model.';
      draw();
    };
    $('#rs-reset', root).onclick = () => {
      playback.stop();
      refit();
      draw();
    };
    $('#rs-new', root).onclick = () => {
      playback.stop();
      seed++;
      gen();
      refit();
      draw();
      playback.start();
    };
    const re = () => {
      playback.stop();
      refit();
      draw();
      playback.start();
    };
    segBind(root, 'rs-d', settings.d, (v) => {
      settings.d = v;
      gen();
      re();
    });
    segBind(root, 'rs-m', '1', (v) => {
      settings.m = +v;
      re();
    });
    $('#rs-log', root).onchange = (e) => {
      settings.log = e.target.checked;
      re();
    };
    $('#rs-drop', root).onchange = (e) => {
      settings.drop = e.target.checked;
      re();
    };
    xyUp(
      root,
      'rs',
      (q) => {
        const x0 = Math.min(...q.map((d) => d[0])),
          x1 = Math.max(...q.map((d) => d[0]));
        D = q.map((d) => [0.4 + (9.2 * (d[0] - x0)) / (x1 - x0 || 1), d[1]]);
        if (settings.log && D.some((d) => d[1] <= 0)) {
          settings.log = false;
          $('#rs-log', root).checked = false;
        }
        re();
      },
      false,
    );
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        u = 0;
        draw();
        playback.start();
      },
    };
  },
};
