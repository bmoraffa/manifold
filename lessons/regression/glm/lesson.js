/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  clipPlot,
  curve,
  eqShow,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { dotAt, segBind, uCur, xyUp } from '../../unsupervised/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#gl-c', root),
      cr = $('#gl-r', root),
      live = $('#gl-live', root),
      eq = $('#gl-eq', root),
      S = { f: 'pois', n: 40 },
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
      dev0 = NaN,
      msg = '',
      cu = 0,
      seed = 1,
      pb,
      chk = () => false,
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
        C = CFG[S.f];
      D = [];
      up = false;
      for (let i = 0; i < S.n; i++) {
        const x = C.xr[0] + (C.xr[1] - C.xr[0]) * (0.03 + 0.94 * r());
        D.push([
          x,
          S.f === 'pois'
            ? pois(Math.exp(-1.1 + 0.135 * x), r)
            : S.f === 'bern'
              ? r() < 1 / (1 + Math.exp(-(-4.2 + 0.85 * x)))
                ? 1
                : 0
              : Math.max(0, 8 + 4.2 * x + 5 * rn()),
        ]);
      }
    };
    const devOf = (a, b) => {
      const C = CFG[S.f];
      let d = 0;
      for (const [x, y] of D) {
        const m = C.inv(a + b * x);
        d +=
          S.f === 'pois'
            ? 2 * ((y > 0 ? y * Math.log(y / m) : 0) - (y - m))
            : S.f === 'bern'
              ? -2 * (y * Math.log(Math.max(m, 1e-12)) + (1 - y) * Math.log(Math.max(1 - m, 1e-12)))
              : (y - m) ** 2;
      }
      return d;
    };
    const work = () => {
      Z = [];
      W = [];
      const C = CFG[S.f];
      for (const [x, y] of D) {
        const e = a + b * x,
          m = C.inv(e),
          dm = S.f === 'pois' ? m : S.f === 'bern' ? Math.max(m * (1 - m), 1e-9) : 1;
        Z.push(e + (y - m) / dm);
        W.push(dm);
      }
    };
    const reset = () => {
      const C = CFG[S.f],
        n = D.length;
      let my = 0,
        mx = 0;
      for (const d of D) {
        mx += d[0] / n;
        my += d[1] / n;
      }
      my = S.f === 'bern' ? clamp(my, 0.02, 0.98) : Math.max(my, 0.05);
      a = C.lnk(my);
      b = 0;
      it = 0;
      fin = false;
      dev = dev0 = devOf(a, b);
      let sxx = 0,
        sxy = 0;
      for (const d of D) {
        sxx += (d[0] - mx) ** 2;
        sxy += (d[0] - mx) * (d[1] - my);
      }
      ols = [my - (sxy / sxx) * mx, sxy / sxx];
      work();
      cu = 0;
      msg =
        'Start: a flat model that predicts the average, ' +
        f2(my) +
        ', everywhere. Press One refit.';
      eqShow(eq, ['No refits yet. The model is flat at the average of the data.'], 0);
    };
    const step = () => {
      if (fin) return;
      work();
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
            f2(CFG[S.f].inv(a + b * D[k][0])) +
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
      work();
      cu = 3;
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
        cu = 5;
        msg =
          'Stopped after ' +
          it +
          ' refit' +
          (it === 1 ? '' : 's') +
          ': the fit has stopped changing. ' +
          (S.f === 'gaus'
            ? 'With Gaussian data the first refit is already ordinary least squares, and the second only confirms it.'
            : 'Deviance ' + f2(dev) + ', against ' + f2(dev0) + ' for the flat model.');
      }
    };
    const tp = testPoint(root, 'gl', ['Your x'], [CFG.pois.tp], [[0, 40]]);
    const draw = () => {
      const t = T(),
        C = CFG[S.f];
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, C.xr, C.yr, t, {
            xstep: C.xs,
            xlabel: C.xl,
            yticks: S.f === 'bern' ? [0, 0.5, 1] : [0, 20, 40, 60],
          });
        clipPlot(g, P);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.5;
        g.setLineDash([6, 5]);
        curve(g, P, (x) => ols[0] + ols[1] * x, C.xr[0], C.xr[1]);
        g.stroke();
        g.setLineDash([]);
        if (S.f === 'pois') {
          g.fillStyle = mix(t.accent, t.accent, 0, 0.13);
          g.beginPath();
          for (let i = 0; i <= 80; i++) {
            const x = C.xr[0] + ((C.xr[1] - C.xr[0]) * i) / 80,
              m = C.inv(a + b * x);
            g.lineTo(P.X(x), P.Y(m + 2 * Math.sqrt(m)));
          }
          for (let i = 80; i >= 0; i--) {
            const x = C.xr[0] + ((C.xr[1] - C.xr[0]) * i) / 80,
              m = C.inv(a + b * x);
            g.lineTo(P.X(x), P.Y(Math.max(0, m - 2 * Math.sqrt(m))));
          }
          g.fill();
        }
        for (const [x, y] of D) dotAt(g, P.X(x), P.Y(y), 3.5, mix(t.ink, t.ink, 0, 0.6));
        g.strokeStyle = t.accent;
        g.lineWidth = 2.8;
        curve(g, P, (x) => C.inv(a + b * x), C.xr[0], C.xr[1]);
        g.stroke();
        const xv = clamp(tp.v[0], C.xr[0], C.xr[1]),
          pm = C.inv(a + b * xv);
        markTP(g, P.X(xv), P.Y(pm), t, false);
        g.restore();
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(C.yl, P.m.l + 6, P.m.t + 8);
        tp.out.textContent =
          'At x = ' +
          f2(xv) +
          ' the model predicts ' +
          (S.f === 'bern'
            ? 'a ' + pct(pm) + ' chance of passing'
            : S.f === 'pois'
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
        const { g, w, h } = fit(cr),
          zs = Z.filter(isFinite),
          lo = Math.min(...zs, C.lnk(S.f === 'bern' ? 0.02 : 0.5)),
          hi = Math.max(...zs),
          pad = 0.08 * (hi - lo + 1e-9),
          P = plot(g, w, h, C.xr, [lo - pad, hi + pad], t, { xstep: C.xs, xlabel: C.xl });
        clipPlot(g, P);
        const wm = Math.max(...W);
        D.forEach(([x], i) =>
          dotAt(g, P.X(x), P.Y(Z[i]), 1.5 + 5 * Math.sqrt(W[i] / wm), mix(t.gold, t.gold, 0, 0.6)),
        );
        g.strokeStyle = t.accent;
        g.lineWidth = 2.6;
        curve(g, P, (x) => a + b * x, C.xr[0], C.xr[1], 2);
        g.stroke();
        g.restore();
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          S.f === 'pois'
            ? 'log of the expected count'
            : S.f === 'bern'
              ? 'log-odds of passing'
              : 'height (the link changes nothing)',
          P.m.l + 6,
          P.m.t + 8,
        );
      }
      let bad = 0;
      for (const [x] of D) {
        const v = ols[0] + ols[1] * x;
        if ((S.f === 'pois' && v < 0) || (S.f === 'bern' && (v < 0 || v > 1))) bad++;
      }
      const lo = ols[0] + ols[1] * C.xr[0],
        hi = ols[0] + ols[1] * C.xr[1];
      live.textContent = msg;
      uCur(root, 'gl', cu);
      setR(root, 'it', it + (fin ? ' (stopped)' : ''));
      setR(root, 'ab', f3(a) + ' and ' + f3(b));
      setR(root, 'dev', f2(dev));
      setR(root, 'dev0', f2(dev0));
      setR(
        root,
        'mean',
        S.f === 'pois'
          ? 'multiplies the expected count by ' + f3(Math.exp(b))
          : S.f === 'bern'
            ? 'multiplies the odds of passing by ' + f2(Math.exp(b))
            : 'adds ' + f2(b) + ' cm',
      );
      setR(
        root,
        'bad',
        S.f === 'gaus'
          ? 'none: a straight line is the right model here'
          : S.f === 'pois'
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
    if (RM) {
      while (!fin) step();
    }
    let fr = 0;
    const anim = Anim(() => {
      if (fin) {
        pb.stop();
        return;
      }
      if (++fr % 45) return;
      step();
      draw();
      chk();
    });
    pb = playBtn($('#gl-play', root), anim);
    chk = runCtl(pb, root, 'gl', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'refits',
    });
    $('#gl-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#gl-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    $('#gl-new', root).onclick = () => {
      pb.stop();
      seed++;
      gen();
      reset();
      draw();
      fr = 20;
      pb.start();
    };
    bind(
      root,
      'gl-n',
      (v) => {
        S.n = v;
        pb.stop();
        gen();
        reset();
        draw();
      },
      (v) => String(+v),
    );
    segBind(root, 'gl-f', S.f, (v) => {
      S.f = v;
      pb.stop();
      tp.set([CFG[v].tp]);
      gen();
      reset();
      draw();
      fr = 20;
      pb.start();
    });
    xyUp(
      root,
      'gl',
      (q) => {
        const st = $('#gl-st', root),
          C = CFG[S.f];
        if (S.f === 'bern' && q.some((d) => d[1] !== 0 && d[1] !== 1)) {
          st.textContent =
            'For yes-or-no data the second column must contain only 0 and 1. Choose another kind of data or another file.';
          return;
        }
        if (S.f === 'pois' && q.some((d) => d[1] < 0 || d[1] !== Math.round(d[1]))) {
          st.textContent =
            'For counts the second column must contain whole numbers of zero or more. Choose another kind of data or another file.';
          return;
        }
        const x0 = Math.min(...q.map((d) => d[0])),
          x1 = Math.max(...q.map((d) => d[0])),
          y1 = Math.max(...q.map((d) => d[1]));
        D = q.map((d) => [
          C.xr[0] + ((d[0] - x0) / (x1 - x0 || 1)) * (C.xr[1] - C.xr[0]),
          S.f === 'bern' ? d[1] : S.f === 'pois' && y1 <= 60 ? d[1] : (d[1] / (y1 || 1)) * 55,
        ]);
        if (S.f === 'pois') D = D.map((d) => [d[0], Math.round(d[1])]);
        st.textContent += ' The x column was rescaled to the axis shown.';
        pb.stop();
        reset();
        draw();
      },
      false,
    );
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        fr = 20;
        pb.start();
      },
    };
  },
};
