/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import {
  canvasXY,
  dotAt,
  segBind,
  highlightLessonStep,
  uFrame,
  drawSeriesChart,
  uView,
  xyUp,
} from '../../unsupervised/shared.js';
import { g2Note } from '../../generative/shared.js';

export default {
  init(root) {
    const c = $('#rr-c', root),
      cr = $('#rr-r', root),
      live = $('#rr-live', root),
      settings = { d: 'one', m: 'huber', k: 1.35, s: 'ols' },
      trueCoefficients = [0.3, 0.55];
    let D = [],
      W = [],
      a = 0,
      b = 0,
      ols = [0, 0],
      residualScale = 1,
      it = 0,
      fin = false,
      msg = '',
      activeStepIndex = 0,
      seed = 1,
      playback,
      checkRunStop = () => false,
      V = null,
      draggedPointIndex = -1,
      kB;
    const gen = () => {
      const r = rng(seed * 13 + settings.d.length),
        rn = randnFrom(r);
      D = [];
      for (let i = 0; i < 30; i++) {
        const x = -3.4 + (6.8 * (i + 0.5)) / 30 + 0.1 * rn();
        D.push([x, trueCoefficients[0] + trueCoefficients[1] * x + 0.3 * rn()]);
      }
      if (settings.d === 'one') D[27][1] = -2.6;
      else if (settings.d === 'few')
        for (const i of [3, 12, 24, 28]) D[i][1] += (i > 15 ? -3 : 3) * (0.7 + 0.5 * r());
      else if (settings.d === 'many') for (let i = 20; i < 30; i++) D[i][1] = -1.6 + 0.25 * rn();
    };
    const wls = (w) => {
      let sw = 0,
        sx = 0,
        sy = 0,
        sxx = 0,
        sxy = 0;
      D.forEach(([x, y], i) => {
        sw += w[i];
        sx += w[i] * x;
        sy += w[i] * y;
        sxx += w[i] * x * x;
        sxy += w[i] * x * y;
      });
      const den = sw * sxx - sx * sx || 1e-12,
        bb = (sw * sxy - sx * sy) / den;
      return [(sy - bb * sx) / sw, bb];
    };
    const med = (v) => {
      const s = v.slice().sort((p, q) => p - q),
        n = s.length;
      return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    };
    const scale = () => {
      const r = D.map((d) => d[1] - a - b * d[0]),
        m = med(r);
      return Math.max(1e-6, med(r.map((v) => Math.abs(v - m))) / 0.6745);
    };
    const wOf = (u) => {
      const au = Math.abs(u);
      return settings.m === 'huber'
        ? Math.min(1, settings.k / Math.max(au, 1e-9))
        : settings.m === 'tukey'
          ? au < settings.k
            ? (1 - (u * u) / (settings.k * settings.k)) ** 2
            : 0
          : 1 / Math.max(au, 0.05);
    };
    const lmsfit = () => {
      let best = wls(D.map(() => 1)),
        bm = 1e9;
      const n = Math.min(D.length, 60);
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(D[j][0] - D[i][0]) < 1e-6) continue;
          const q = (D[j][1] - D[i][1]) / (D[j][0] - D[i][0]),
            p = D[i][1] - q * D[i][0],
            m = med(D.map((d) => Math.abs(d[1] - p - q * d[0])));
          if (m < bm) {
            bm = m;
            best = [p, q];
          }
        }
      return best;
    };
    const weights = () => {
      residualScale = scale();
      W = D.map((d) => wOf((d[1] - a - b * d[0]) / residualScale));
      if (settings.m === 'l1') {
        const m = Math.max(...W);
        W = W.map((v) => v / m);
      }
    };
    const reset = () => {
      ols = wls(D.map(() => 1));
      [a, b] = settings.s === 'ols' ? ols : lmsfit();
      it = 0;
      fin = false;
      weights();
      activeStepIndex = 0;
      msg =
        'First line: ' +
        (settings.s === 'ols'
          ? 'least squares'
          : 'the line through two data points that leaves the smallest median miss') +
        '. Every point has been given a weight from the size of its miss. Press One refit.';
    };
    const step = () => {
      if (fin) return;
      weights();
      if (W.reduce((s, v) => s + v, 0) < 1e-9) {
        fin = true;
        msg = 'Every point is beyond the cut-off, so nothing is left to fit. Raise the cut-off.';
        return;
      }
      const [na, nb] = wls(W),
        da = Math.abs(na - a) + Math.abs(nb - b);
      a = na;
      b = nb;
      it++;
      weights();
      activeStepIndex = 3;
      const low = W.filter((v) => v < 0.5).length;
      msg =
        'Refit ' +
        it +
        ': the line moved by ' +
        (da < 1e-3 ? da.toExponential(1) : f3(da)) +
        '.' +
        (settings.m === 'l1'
          ? ''
          : ' ' + low + ' point' + (low === 1 ? ' has' : 's have') + ' weight below one half.');
      if (da < 1e-5 || it >= 200) {
        fin = true;
        activeStepIndex = 5;
        msg =
          'Stopped after ' +
          it +
          ' refits: the line has stopped moving. Robust slope ' +
          f2(b) +
          ', least-squares slope ' +
          f2(ols[1]) +
          ', true slope ' +
          f2(trueCoefficients[1]) +
          '. ' +
          (Math.abs(b - trueCoefficients[1]) < Math.abs(ols[1] - trueCoefficients[1]) - 0.02
            ? 'The robust line is closer to the truth.'
            : Math.abs(b - trueCoefficients[1]) > Math.abs(ols[1] - trueCoefficients[1]) + 0.02
              ? 'Here the robust line is further from the truth than least squares.'
              : 'The two lines are about equally close to the truth.');
      }
    };
    const rho = (u) => {
      const au = Math.abs(u);
      return settings.m === 'huber'
        ? au <= settings.k
          ? 0.5 * u * u
          : settings.k * au - 0.5 * settings.k * settings.k
        : settings.m === 'tukey'
          ? au <= settings.k
            ? ((settings.k * settings.k) / 6) * (1 - (1 - (u * u) / (settings.k * settings.k)) ** 3)
            : (settings.k * settings.k) / 6
          : au;
    };
    const draw = () => {
      const colors = readThemeColors();
      V = uView(c);
      const { g, w, h, X, Y } = V;
      uFrame(V, colors, 'x', 'y');
      const ln = (p, q) => {
        g.beginPath();
        g.moveTo(X(-4), Y(p - 4 * q));
        g.lineTo(X(4), Y(p + 4 * q));
        g.stroke();
      };
      g.strokeStyle = colors.muted;
      g.lineWidth = 1;
      g.globalAlpha = 0.6;
      ln(trueCoefficients[0], trueCoefficients[1]);
      g.globalAlpha = 1;
      g.strokeStyle = colors.cut;
      g.lineWidth = 2;
      g.setLineDash([7, 5]);
      ln(ols[0], ols[1]);
      g.setLineDash([]);
      g.strokeStyle = colors.accent;
      g.lineWidth = 2.8;
      ln(a, b);
      D.forEach((d, i) => {
        const wv = W[i] || 0;
        g.strokeStyle = mix(colors.ink, colors.ink, 0, 0.25);
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X(d[0]), Y(d[1]));
        g.lineTo(X(d[0]), Y(a + b * d[0]));
        g.stroke();
        g.lineWidth = 1.5;
        dotAt(
          g,
          X(d[0]),
          Y(d[1]),
          2 + 5 * Math.sqrt(wv),
          mix(colors.ink, colors.cut, 1 - wv, 0.85),
          wv < 0.05 ? colors.cut : null,
        );
      });
      g2Note(g, colors, h, 'big dark dots are trusted; small amber dots have been turned down');
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          U = 6,
          n = 121,
          us = Array.from({ length: n }, (_, i) => -U + (2 * U * i) / (n - 1)),
          rs = us.map(rho),
          ym = Math.min(8, Math.max(...rs) * 1.1),
          A = drawSeriesChart(ctx, colors, 0, 0, w, h1, [], {
            title: 'penalty against the size of the miss (in typical-miss units, −6 to 6)',
            n,
            ym,
            y1: 0,
          });
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, w, h1 - 18);
        ctx.clip();
        ctx.strokeStyle = colors.cut;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        us.forEach((u, i) =>
          i ? ctx.lineTo(A.X(i), A.Y(0.5 * u * u)) : ctx.moveTo(A.X(i), A.Y(0.5 * u * u)),
        );
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        rs.forEach((v, i) => (i ? ctx.lineTo(A.X(i), A.Y(v)) : ctx.moveTo(A.X(i), A.Y(v))));
        ctx.stroke();
        ctx.restore();
        const ws = us.map((u) =>
            settings.m === 'l1' ? Math.min(1, 0.25 / Math.max(Math.abs(u), 0.01)) : wOf(u),
          ),
          B = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, [], {
            title: 'weight against the size of the miss',
            n,
            ym: 1.08,
            y1: 0,
          });
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ws.forEach((v, i) => (i ? ctx.lineTo(B.X(i), B.Y(v)) : ctx.moveTo(B.X(i), B.Y(v))));
        ctx.stroke();
        D.forEach((d, i) => {
          const u = clamp((d[1] - a - b * d[0]) / residualScale, -U, U),
            xi = ((u + U) / (2 * U)) * (n - 1);
          ctx.lineWidth = 1;
          dotAt(
            ctx,
            B.X(xi),
            B.Y(settings.m === 'l1' ? Math.min(1, 0.25 / Math.max(Math.abs(u), 0.01)) : wOf(u)),
            3.5,
            mix(colors.gold, colors.gold, 0, 0.8),
            colors.bg,
          );
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'rr', activeStepIndex);
      setReadout(root, 'it', it + (fin ? ' (stopped)' : ''));
      setReadout(root, 'tr', f2(trueCoefficients[0]) + ', ' + f2(trueCoefficients[1]));
      setReadout(root, 'ls', f2(ols[0]) + ', ' + f2(ols[1]));
      setReadout(root, 'rb', f2(a) + ', ' + f2(b));
      setReadout(root, 'sc', f2(residualScale));
      setReadout(
        root,
        'dn',
        settings.m === 'l1'
          ? 'not meaningful for this penalty'
          : W.filter((v) => v < 0.5).length + ' of ' + D.length,
      );
    };
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
      if (++frameCount % 14) return;
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#rr-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'rr', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'refits',
    });
    const go = () => {
      playback.stop();
      reset();
      draw();
      frameCount = 0;
      playback.start();
    };
    $('#rr-step', root).onclick = () => {
      playback.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#rr-reset', root).onclick = () => {
      playback.stop();
      gen();
      reset();
      draw();
    };
    $('#rr-new', root).onclick = () => {
      seed++;
      gen();
      go();
    };
    kB = bind(root, 'rr-k', (v) => {
      settings.k = v;
      playback.stop();
      reset();
      draw();
    });
    segBind(root, 'rr-d', settings.d, (v) => {
      settings.d = v;
      gen();
      go();
    });
    segBind(root, 'rr-m', settings.m, (v) => {
      settings.m = v;
      settings.k = v === 'tukey' ? 4.69 : 1.35;
      kB.set(settings.k);
      kB.el.disabled = v === 'l1';
      go();
    });
    segBind(root, 'rr-s', settings.s, (v) => {
      settings.s = v;
      go();
    });
    xyUp(root, 'rr', (q) => {
      D = q.slice(0, 120);
      trueCoefficients[0] = NaN;
      trueCoefficients[1] = NaN;
      go();
    });
    c.style.cursor = 'grab';
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', (e) => {
      if (!V) return;
      const [x, y] = canvasXY(c, V, e);
      draggedPointIndex = D.findIndex((d) => Math.hypot((d[0] - x) * V.s, (d[1] - y) * V.s) < 12);
      if (draggedPointIndex < 0 && D.length < 120) {
        D.push([x, y]);
        draggedPointIndex = D.length - 1;
      }
      c.setPointerCapture(e.pointerId);
      playback.stop();
      reset();
      draw();
    });
    c.addEventListener('pointermove', (e) => {
      if (draggedPointIndex < 0 || !V) return;
      const [x, y] = canvasXY(c, V, e);
      D[draggedPointIndex] = [clamp(x, -4, 4), clamp(y, -3, 3)];
      reset();
      while (!fin) step();
      draw();
    });
    const end = () => {
      draggedPointIndex = -1;
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
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
