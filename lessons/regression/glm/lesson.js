/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  clipPlot,
  curve,
  eqShow,
  f2,
  f3,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { dotAt, segBind, highlightLessonStep, xyUp } from '../../unsupervised/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#gl-c', root),
      cr = $('#gl-r', root),
      live = $('#gl-live', root),
      eq = $('#gl-eq', root),
      settings = { f: 'pois', n: 40 },
      CFG = {
        pois: {
          xr: [8, 36],
          yr: [0, 60],
          xl: 'temperature, °C',
          yl: 'ice creams sold',
          xs: 4,
          tp: 28,
          inv: (e) => Math.exp(e),
          lnk: (m) => Math.log(m),
        },
        bern: {
          xr: [0, 10],
          yr: [-0.15, 1.15],
          xl: 'hours studied',
          yl: 'passed (1) or failed (0)',
          xs: 2,
          tp: 6,
          inv: (e) => 1 / (1 + Math.exp(-e)),
          lnk: (m) => Math.log(m / (1 - m)),
        },
        gaus: {
          xr: [0, 10],
          yr: [0, 60],
          xl: 'fertilizer, grams',
          yl: 'height, cm',
          xs: 2,
          tp: 5,
          inv: (e) => e,
          lnk: (m) => m,
        },
      };
    let D = [],
      a = 0,
      b = 0,
      it = 0,
      fin = false,
      dev = NaN,
      nullDeviance = NaN,
      msg = '',
      activeStepIndex = 0,
      seed = 1,
      playback,
      checkRunStop = () => false,
      ols = [0, 0],
      Z = [],
      W = [],
      up = false;
    const pois = (lam, r) => {
      let L = Math.exp(-lam),
        k = 0,
        p = 1;
      do {
        k++;
        p *= r();
      } while (p > L);
      return k - 1;
    };
    const gen = () => {
      const r = rng(seed * 31 + 7),
        rn = randnFrom(r),
        C = CFG[settings.f];
      D = [];
      up = false;
      for (let i = 0; i < settings.n; i++) {
        const x = C.xr[0] + (C.xr[1] - C.xr[0]) * (0.03 + 0.94 * r());
        D.push([
          x,
          settings.f === 'pois'
            ? pois(Math.exp(-1.1 + 0.135 * x), r)
            : settings.f === 'bern'
              ? r() < 1 / (1 + Math.exp(-(-4.2 + 0.85 * x)))
                ? 1
                : 0
              : Math.max(0, 8 + 4.2 * x + 5 * rn()),
        ]);
      }
    };
    const devOf = (a, b) => {
      const C = CFG[settings.f];
      let d = 0;
      for (const [x, y] of D) {
        const m = C.inv(a + b * x);
        d +=
          settings.f === 'pois'
            ? 2 * ((y > 0 ? y * Math.log(y / m) : 0) - (y - m))
            : settings.f === 'bern'
              ? -2 * (y * Math.log(Math.max(m, 1e-12)) + (1 - y) * Math.log(Math.max(1 - m, 1e-12)))
              : (y - m) ** 2;
      }
      return d;
    };
    const updateWorkingData = () => {
      Z = [];
      W = [];
      const C = CFG[settings.f];
      for (const [x, y] of D) {
        const e = a + b * x,
          m = C.inv(e),
          dm = settings.f === 'pois' ? m : settings.f === 'bern' ? Math.max(m * (1 - m), 1e-9) : 1;
        Z.push(e + (y - m) / dm);
        W.push(dm);
      }
    };
    const reset = () => {
      const C = CFG[settings.f],
        n = D.length;
      let my = 0,
        mx = 0;
      for (const d of D) {
        mx += d[0] / n;
        my += d[1] / n;
      }
      my = settings.f === 'bern' ? clamp(my, 0.02, 0.98) : Math.max(my, 0.05);
      a = C.lnk(my);
      b = 0;
      it = 0;
      fin = false;
      dev = nullDeviance = devOf(a, b);
      let sxx = 0,
        sxy = 0;
      for (const d of D) {
        sxx += (d[0] - mx) ** 2;
        sxy += (d[0] - mx) * (d[1] - my);
      }
      ols = [my - (sxy / sxx) * mx, sxy / sxx];
      updateWorkingData();
      activeStepIndex = 0;
      msg =
        'Start: a flat model that predicts the average, ' +
        f2(my) +
        ', everywhere. Press One refit.';
      eqShow(eq, ['No refits yet. The model is flat at the average of the data.'], 0);
    };
    const step = () => {
      if (fin) return;
      updateWorkingData();
      let sw = 0,
        sx = 0,
        sz = 0,
        sxx = 0,
        sxz = 0;
      D.forEach(([x], i) => {
        const w = W[i];
        sw += w;
        sx += w * x;
        sz += w * Z[i];
        sxx += w * x * x;
        sxz += w * x * Z[i];
      });
      const den = sw * sxx - sx * sx,
        nb = (sw * sxz - sx * sz) / den,
        na = (sz - nb * sx) / sw,
        d1 = devOf(na, nb),
        k = D.reduce((m, d, i) => (W[i] > W[m] ? i : m), 0);
      it++;
      eqShow(
        eq,
        [
          'Refit ' + it + '. Current line on the link scale: η = ' + f3(a) + ' + ' + f3(b) + ' × x',
          'Example, the point with the most weight: x = ' +
            f2(D[k][0]) +
            ', y = ' +
            f2(D[k][1]) +
            ', fitted mean ' +
            f2(CFG[settings.f].inv(a + b * D[k][0])) +
            ' → working value z = ' +
            f3(Z[k]) +
            ', weight ' +
            f3(W[k]),
          'Weighted straight line through all ' + D.length + ' working values:',
          'new intercept = <b>' + f3(na) + '</b>, new slope = <b>' + f3(nb) + '</b>',
          'deviance: ' + f3(dev) + ' → <b>' + f3(d1) + '</b>',
        ],
        0,
      );
      const gain = dev - d1;
      a = na;
      b = nb;
      dev = d1;
      updateWorkingData();
      activeStepIndex = 3;
      msg =
        'Refit ' +
        it +
        ': the deviance fell by ' +
        (gain < 1e-3 ? gain.toExponential(1) : f3(gain)) +
        ' to ' +
        f3(dev) +
        '.';
      if (Math.abs(gain) < 1e-8 || it >= 50) {
        fin = true;
        activeStepIndex = 5;
        msg =
          'Stopped after ' +
          it +
          ' refit' +
          (it === 1 ? '' : 's') +
          ': the fit has stopped changing. ' +
          (settings.f === 'gaus'
            ? 'With Gaussian data the first refit is already ordinary least squares, and the second only confirms it.'
            : 'Deviance ' + f2(dev) + ', against ' + f2(nullDeviance) + ' for the flat model.');
      }
    };
    const tp = testPoint(root, 'gl', ['Your x'], [CFG.pois.tp], [[0, 40]]);
    const draw = () => {
      const colors = readThemeColors(),
        C = CFG[settings.f];
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = plot(ctx, w, h, C.xr, C.yr, colors, {
            xstep: C.xs,
            xlabel: C.xl,
            yticks: settings.f === 'bern' ? [0, 0.5, 1] : [0, 20, 40, 60],
          });
        clipPlot(ctx, P);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 5]);
        curve(ctx, P, (x) => ols[0] + ols[1] * x, C.xr[0], C.xr[1]);
        ctx.stroke();
        ctx.setLineDash([]);
        if (settings.f === 'pois') {
          ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.13);
          ctx.beginPath();
          for (let i = 0; i <= 80; i++) {
            const x = C.xr[0] + ((C.xr[1] - C.xr[0]) * i) / 80,
              m = C.inv(a + b * x);
            ctx.lineTo(P.X(x), P.Y(m + 2 * Math.sqrt(m)));
          }
          for (let i = 80; i >= 0; i--) {
            const x = C.xr[0] + ((C.xr[1] - C.xr[0]) * i) / 80,
              m = C.inv(a + b * x);
            ctx.lineTo(P.X(x), P.Y(Math.max(0, m - 2 * Math.sqrt(m))));
          }
          ctx.fill();
        }
        for (const [x, y] of D)
          dotAt(ctx, P.X(x), P.Y(y), 3.5, mix(colors.ink, colors.ink, 0, 0.6));
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.8;
        curve(ctx, P, (x) => C.inv(a + b * x), C.xr[0], C.xr[1]);
        ctx.stroke();
        const xv = clamp(tp.v[0], C.xr[0], C.xr[1]),
          pm = C.inv(a + b * xv);
        markTP(ctx, P.X(xv), P.Y(pm), colors, false);
        ctx.restore();
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(C.yl, P.m.l + 6, P.m.t + 8);
        tp.out.textContent =
          'At x = ' +
          f2(xv) +
          ' the model predicts ' +
          (settings.f === 'bern'
            ? 'a ' + pct(pm) + ' chance of passing'
            : settings.f === 'pois'
              ? f2(pm) +
                ' ice creams on average, usually between ' +
                Math.max(0, Math.round(pm - 2 * Math.sqrt(pm))) +
                ' and ' +
                Math.round(pm + 2 * Math.sqrt(pm))
              : f2(pm) + ' cm') +
          '. The dashed straight line says ' +
          f2(ols[0] + ols[1] * xv) +
          '.';
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          zs = Z.filter(isFinite),
          lo = Math.min(...zs, C.lnk(settings.f === 'bern' ? 0.02 : 0.5)),
          hi = Math.max(...zs),
          pad = 0.08 * (hi - lo + 1e-9),
          P = plot(ctx, w, h, C.xr, [lo - pad, hi + pad], colors, { xstep: C.xs, xlabel: C.xl });
        clipPlot(ctx, P);
        const wm = Math.max(...W);
        D.forEach(([x], i) =>
          dotAt(
            ctx,
            P.X(x),
            P.Y(Z[i]),
            1.5 + 5 * Math.sqrt(W[i] / wm),
            mix(colors.gold, colors.gold, 0, 0.6),
          ),
        );
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.6;
        curve(ctx, P, (x) => a + b * x, C.xr[0], C.xr[1], 2);
        ctx.stroke();
        ctx.restore();
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          settings.f === 'pois'
            ? 'log of the expected count'
            : settings.f === 'bern'
              ? 'log-odds of passing'
              : 'height (the link changes nothing)',
          P.m.l + 6,
          P.m.t + 8,
        );
      }
      let bad = 0;
      for (const [x] of D) {
        const v = ols[0] + ols[1] * x;
        if ((settings.f === 'pois' && v < 0) || (settings.f === 'bern' && (v < 0 || v > 1))) bad++;
      }
      const lo = ols[0] + ols[1] * C.xr[0],
        hi = ols[0] + ols[1] * C.xr[1];
      live.textContent = msg;
      highlightLessonStep(root, 'gl', activeStepIndex);
      setReadout(root, 'it', it + (fin ? ' (stopped)' : ''));
      setReadout(root, 'ab', f3(a) + ' and ' + f3(b));
      setReadout(root, 'dev', f2(dev));
      setReadout(root, 'dev0', f2(nullDeviance));
      setReadout(
        root,
        'mean',
        settings.f === 'pois'
          ? 'multiplies the expected count by ' + f3(Math.exp(b))
          : settings.f === 'bern'
            ? 'multiplies the odds of passing by ' + f2(Math.exp(b))
            : 'adds ' + f2(b) + ' cm',
      );
      setReadout(
        root,
        'bad',
        settings.f === 'gaus'
          ? 'none: a straight line is the right model here'
          : settings.f === 'pois'
            ? lo < 0
              ? 'below zero at the cold end (' + f2(lo) + ' at ' + C.xr[0] + ' °C)'
              : 'none on this range'
            : lo < 0 || hi > 1
              ? 'outside 0 to 1 (' + f2(lo) + ' at 0 hours, ' + f2(hi) + ' at 10 hours)'
              : 'none on this range',
      );
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
      if (++frameCount % 45) return;
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#gl-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'gl', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'refits',
    });
    $('#gl-step', root).onclick = () => {
      playback.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#gl-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    $('#gl-new', root).onclick = () => {
      playback.stop();
      seed++;
      gen();
      reset();
      draw();
      frameCount = 20;
      playback.start();
    };
    bind(
      root,
      'gl-n',
      (v) => {
        settings.n = v;
        playback.stop();
        gen();
        reset();
        draw();
      },
      (v) => String(+v),
    );
    segBind(root, 'gl-f', settings.f, (v) => {
      settings.f = v;
      playback.stop();
      tp.set([CFG[v].tp]);
      gen();
      reset();
      draw();
      frameCount = 20;
      playback.start();
    });
    xyUp(
      root,
      'gl',
      (q) => {
        const st = $('#gl-st', root),
          C = CFG[settings.f];
        if (settings.f === 'bern' && q.some((d) => d[1] !== 0 && d[1] !== 1)) {
          st.textContent =
            'For yes-or-no data the second column must contain only 0 and 1. Choose another kind of data or another file.';
          return;
        }
        if (settings.f === 'pois' && q.some((d) => d[1] < 0 || d[1] !== Math.round(d[1]))) {
          st.textContent =
            'For counts the second column must contain whole numbers of zero or more. Choose another kind of data or another file.';
          return;
        }
        const x0 = Math.min(...q.map((d) => d[0])),
          x1 = Math.max(...q.map((d) => d[0])),
          y1 = Math.max(...q.map((d) => d[1]));
        D = q.map((d) => [
          C.xr[0] + ((d[0] - x0) / (x1 - x0 || 1)) * (C.xr[1] - C.xr[0]),
          settings.f === 'bern'
            ? d[1]
            : settings.f === 'pois' && y1 <= 60
              ? d[1]
              : (d[1] / (y1 || 1)) * 55,
        ]);
        if (settings.f === 'pois') D = D.map((d) => [d[0], Math.round(d[1])]);
        st.textContent += ' The x column was rescaled to the axis shown.';
        playback.stop();
        reset();
        draw();
      },
      false,
    );
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        reset();
        draw();
        frameCount = 20;
        playback.start();
      },
    };
  },
};
