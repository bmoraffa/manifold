/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  f2,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { dotAt, segBind, highlightLessonStep, uFrame, drawSeriesChart, uView } from '../shared.js';
import { g2Note } from '../../generative/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#ic-c', root),
      cr = $('#ic-r', root),
      live = $('#ic-live', root),
      settings = { d: 'wave', a: 0.6, b: 0.45 },
      N = 500;
    let Sr = [],
      X = [],
      Zw = [],
      th = 0,
      u = 0,
      it = 0,
      fin = false,
      hist = [],
      contrastByAngle = [],
      seed = 1,
      msg = '',
      activeStepIndex = 0,
      playback,
      checkRunStop = () => false,
      ad = null,
      whitenedMatch = 0,
      mixedMatch = 0,
      anim = null;
    const gen = () => {
      const r = rng(seed * 23 + settings.d.length),
        rn = randnFrom(r);
      Sr = [[], []];
      let burst = 0;
      const p1 = 6 * r(),
        p2 = r(),
        f1 = 0.09 + 0.04 * r(),
        f2_ = 0.026 + 0.012 * r();
      for (let t = 0; t < N; t++) {
        if (settings.d === 'wave') {
          Sr[0].push(Math.sin(t * f1 + p1));
          Sr[1].push(2 * ((t * f2_ + p2) % 1) - 1);
        } else if (settings.d === 'talk') {
          if (r() < 0.04) burst = 1 + 3 * r();
          burst *= 0.94;
          Sr[0].push(burst * Math.sin(t * 0.9) * (0.6 + 0.4 * rn()));
          Sr[1].push(Math.sin(t * f2_ * 1.6 + p1) > 0 ? 1 : -1);
        } else {
          Sr[0].push(rn());
          Sr[1].push(rn());
        }
      }
      Sr = Sr.map((s) => {
        const m = s.reduce((a, v) => a + v, 0) / N,
          sd = Math.sqrt(s.reduce((a, v) => a + (v - m) ** 2, 0) / N);
        return s.map((v) => (v - m) / sd);
      });
    };
    const corr = (a, b) => {
      let s = 0;
      for (let i = 0; i < N; i++) s += a[i] * b[i];
      const na = Math.sqrt(a.reduce((q, v) => q + v * v, 0)),
        nb = Math.sqrt(b.reduce((q, v) => q + v * v, 0));
      return s / (na * nb);
    };
    const match = (Y) => {
      const c00 = Math.abs(corr(Y[0], Sr[0])),
        c11 = Math.abs(corr(Y[1], Sr[1])),
        c01 = Math.abs(corr(Y[0], Sr[1])),
        c10 = Math.abs(corr(Y[1], Sr[0]));
      return Math.max((c00 + c11) / 2, (c01 + c10) / 2);
    };
    const rot = (t) => {
      const cs = Math.cos(t),
        sn = Math.sin(t);
      return [
        Zw[0].map((v, i) => cs * v + sn * Zw[1][i]),
        Zw[0].map((v, i) => -sn * v + cs * Zw[1][i]),
      ];
    };
    const kurt = (y) => {
      let s = 0;
      for (const v of y) s += v ** 4;
      return s / N - 3;
    };
    const J = (t) => {
      const Y = rot(t);
      return kurt(Y[0]) ** 2 + kurt(Y[1]) ** 2;
    };
    const mix_ = () => {
      X = [
        Sr[0].map((v, i) => v + settings.a * Sr[1][i]),
        Sr[1].map((v, i) => v + settings.b * Sr[0][i]),
      ];
      X = X.map((s) => {
        const m = s.reduce((a, v) => a + v, 0) / N;
        return s.map((v) => v - m);
      });
      let a = 0,
        b = 0,
        d = 0;
      for (let i = 0; i < N; i++) {
        a += X[0][i] ** 2 / N;
        b += (X[0][i] * X[1][i]) / N;
        d += X[1][i] ** 2 / N;
      }
      const tr = a + d,
        dt = a * d - b * b,
        l1 = tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - dt)),
        l2 = Math.max(1e-9, tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - dt))),
        ph = Math.abs(b) < 1e-12 ? 0 : Math.atan2(l1 - a, b),
        e1 = [Math.cos(ph), Math.sin(ph)],
        e2 = [-Math.sin(ph), Math.cos(ph)];
      const z1 = X[0].map((v, i) => (e1[0] * v + e1[1] * X[1][i]) / Math.sqrt(l1)),
        z2 = X[0].map((v, i) => (e2[0] * v + e2[1] * X[1][i]) / Math.sqrt(l2));
      Zw = [z1, z2];
      whitenedMatch = match(Zw);
      mixedMatch = match(X);
      contrastByAngle = Array.from({ length: 91 }, (_, k) => J((k * Math.PI) / 180));
    };
    const reset = () => {
      th = 0;
      u = 0;
      it = 0;
      fin = false;
      hist = [];
      ad = null;
      activeStepIndex = 0;
      msg =
        'Two microphones, each a blend of both sources. The cloud is slanted because the recordings are correlated. Press Run or One step.';
    };
    const step = () => {
      if (fin) return;
      if (u < 1) {
        u = Math.min(1, u + (anim && anim.on ? 0.03 : 1));
        activeStepIndex = 1;
        msg =
          u >= 1
            ? 'Whitened: the cloud is round and the two signals are uncorrelated. They match the sources ' +
              pct(whitenedMatch) +
              ', which is not yet a separation. Now the turning begins.'
            : 'Whitening: rotating and rescaling the cloud until it is round.';
        return;
      }
      if (!ad) ad = { m: 0, v: 0, t: 0 };
      const gr = (J(th + 1e-3) - J(th - 1e-3)) / 2e-3;
      ad.t++;
      ad.m = 0.7 * ad.m + 0.3 * gr;
      ad.v = 0.99 * ad.v + 0.01 * gr * gr;
      const d = (0.03 * (ad.m / (1 - 0.7 ** ad.t))) / (Math.sqrt(ad.v / (1 - 0.99 ** ad.t)) + 1e-9);
      th += d;
      it++;
      hist.push(Math.abs(d));
      activeStepIndex = 4;
      const Y = rot(th);
      msg =
        'Step ' +
        it +
        ': turned by ' +
        ((d * 180) / Math.PI).toFixed(2) +
        '° to ' +
        ((th * 180) / Math.PI).toFixed(1) +
        '°. Match with the true sources: ' +
        pct(match(Y)) +
        '.';
      if ((it >= 8 && Math.max(...hist.slice(-4)) < 1e-4) || it >= 400) {
        fin = true;
        activeStepIndex = 5;
        const m = match(Y),
          flat = Math.max(...contrastByAngle) < 0.3;
        msg =
          'Stopped after ' +
          it +
          ' steps at ' +
          ((th * 180) / Math.PI).toFixed(1) +
          '°. ' +
          (flat
            ? 'The contrast curve is almost flat, so this angle is no better than any other: Gaussian sources cannot be separated. Match ' +
              pct(m) +
              '.'
            : 'The recovered signals match the true sources ' +
              pct(m) +
              ', against ' +
              pct(whitenedMatch) +
              ' after whitening alone and ' +
              pct(mixedMatch) +
              ' for the raw microphones.');
      }
    };
    const draw = () => {
      const colors = readThemeColors(),
        s = u * u * (3 - 2 * u),
        Y = rot(th);
      {
        const V = uView(c),
          { g, w, h, X: PX, Y: PY } = V;
        uFrame(
          V,
          colors,
          s < 0.5 ? 'microphone 1' : 'output 1',
          s < 0.5 ? 'microphone 2' : 'output 2',
        );
        const k = 1.15;
        for (let i = 0; i < N; i++) {
          const x = X[0][i] * (1 - s) + Y[0][i] * s,
            y = X[1][i] * (1 - s) + Y[1][i] * s;
          dotAt(g, PX(x * k), PY(y * k), 2, mix(colors.accent, colors.accent, 0, 0.55));
        }
        if (u >= 1) {
          g.strokeStyle = mix(colors.muted, colors.muted, 0, 0.7);
          g.lineWidth = 1;
          g.setLineDash([4, 4]);
          for (const a of [-th, -th + Math.PI / 2]) {
            g.beginPath();
            g.moveTo(PX(-2.8 * Math.cos(a)), PY(-2.8 * Math.sin(a)));
            g.lineTo(PX(2.8 * Math.cos(a)), PY(2.8 * Math.sin(a)));
            g.stroke();
          }
          g.setLineDash([]);
          g.strokeStyle = colors.gold;
          g.lineWidth = 2.5;
          for (const a of [0, Math.PI / 2]) {
            g.beginPath();
            g.moveTo(PX(0), PY(0));
            g.lineTo(PX(2.6 * Math.cos(a)), PY(2.6 * Math.sin(a)));
            g.stroke();
          }
        }
        g2Note(
          g,
          colors,
          h,
          u < 1
            ? 'the slanted cloud of raw recordings'
            : 'dashed: where the axes were after whitening; the cloud has turned ' +
                ((th * 180) / Math.PI).toFixed(1) +
                '°',
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.74,
          rows = [
            [Sr[0], 'true source 1', colors.muted],
            [Sr[1], 'true source 2', colors.muted],
            [X[0], 'microphone 1', colors.cut],
            [X[1], 'microphone 2', colors.cut],
            [
              u < 1 ? X[0].map((v, i) => v * (1 - s) + Zw[0][i] * s) : Y[0],
              'recovered 1',
              colors.accent,
            ],
            [
              u < 1 ? X[1].map((v, i) => v * (1 - s) + Zw[1][i] * s) : Y[1],
              'recovered 2',
              colors.accent,
            ],
          ],
          rh = h1 / 6,
          M = 220;
        ctx.font = MONO;
        rows.forEach(([y, nm, col], r) => {
          const mx = Math.max(...y.slice(0, M).map(Math.abs)) || 1;
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          for (let i = 0; i < M; i++) {
            const px = 98 + ((w - 106) * i) / (M - 1),
              py = r * rh + rh / 2 - (y[i] / mx) * (rh / 2 - 3);
            i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
          }
          ctx.stroke();
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(nm, 4, r * rh + rh / 2 + 4);
        });
        const jm = Math.max(...contrastByAngle, 1e-6),
          A = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, contrastByAngle, {
            title: 'how un-Gaussian the outputs are, at turning angles 0° to 90°',
            n: 91,
            ym: Math.max(jm * 1.1, 0.3),
            y1: 0,
          });
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2;
        ctx.beginPath();
        contrastByAngle.forEach((v, i) =>
          i ? ctx.lineTo(A.X(i), A.Y(v)) : ctx.moveTo(A.X(i), A.Y(v)),
        );
        ctx.stroke();
        const ad_ = ((((th * 180) / Math.PI) % 90) + 90) % 90;
        dotAt(ctx, A.X(ad_), A.Y(J(th)), 5, colors.accent, colors.bg);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'ic', activeStepIndex);
      setReadout(root, 'it', it + (fin ? ' (stopped)' : ''));
      setReadout(root, 'ang', ((th * 180) / Math.PI).toFixed(1) + '°');
      setReadout(root, 'kur', u >= 1 ? f2(kurt(Y[0])) + ' and ' + f2(kurt(Y[1])) : '–');
      setReadout(root, 'm', u >= 1 ? pct(match(Y)) : '–');
      setReadout(root, 'mw', pct(whitenedMatch));
      setReadout(root, 'mm', pct(mixedMatch));
    };
    gen();
    mix_();
    reset();
    if (prefersReducedMotion) {
      while (!fin) step();
    }
    let frameCount = 0;
    anim = Anim(() => {
      if (fin) {
        playback.stop();
        return;
      }
      if (u >= 1 && ++frameCount % 4) return;
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#ic-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'ic', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'steps',
    });
    const go = () => {
      playback.stop();
      mix_();
      reset();
      draw();
      playback.start();
    };
    $('#ic-step', root).onclick = () => {
      playback.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#ic-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    $('#ic-new', root).onclick = () => {
      seed++;
      gen();
      go();
    };
    bind(root, 'ic-a', (v) => {
      settings.a = v;
      playback.stop();
      mix_();
      reset();
      draw();
    });
    bind(root, 'ic-b', (v) => {
      settings.b = v;
      playback.stop();
      mix_();
      reset();
      draw();
    });
    segBind(root, 'ic-d', settings.d, (v) => {
      settings.d = v;
      gen();
      go();
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
