/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  eqShow,
  f2,
  f3,
  prepareCanvas,
  lgam,
  markTP,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#by-c', root),
      cr = $('#by-r', root),
      live = $('#by-live', root),
      eq = $('#by-eq', root),
      settings = { p: 0.3, m: 0.5, n0: 2, pr: 'flat' },
      PR = { flat: [1, 1], mild: [5, 5], stub: [48, 12] },
      MAXN = 1000,
      NG = 400;
    let a0 = 1,
      b0 = 1,
      D = [],
      k = 0,
      posteriorMeanHistory = [],
      HL = [],
      HU = [],
      meanHistoryByPrior = [[], [], []],
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false,
      bm,
      bn;
    const pdf = (a, b) => {
      const o = new Float64Array(NG + 1),
        lb = lgam(a) + lgam(b) - lgam(a + b);
      for (let i = 0; i <= NG; i++) {
        const x = clamp(i / NG, 1e-6, 1 - 1e-6);
        o[i] = Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lb);
      }
      return o;
    };
    const credibleInterval95 = (a, b) => {
      const d = pdf(a, b);
      let tot = 0;
      for (let i = 0; i <= NG; i++) tot += d[i];
      let s = 0,
        lo = 0,
        hi = 1,
        gl = false;
      for (let i = 0; i <= NG; i++) {
        s += d[i] / tot;
        if (!gl && s >= 0.025) {
          lo = i / NG;
          gl = true;
        }
        if (s >= 0.975) {
          hi = i / NG;
          break;
        }
      }
      return [lo, hi];
    };
    const push = () => {
      const n = D.length,
        a = a0 + k,
        b = b0 + n - k,
        [lo, hi] = credibleInterval95(a, b);
      posteriorMeanHistory.push(a / (a + b));
      HL.push(lo);
      HU.push(hi);
      Object.values(PR).forEach((q, i) =>
        meanHistoryByPrior[i].push((q[0] + k) / (q[0] + q[1] + n)),
      );
    };
    const build = () => {
      D = [];
      k = 0;
      posteriorMeanHistory = [];
      HL = [];
      HU = [];
      meanHistoryByPrior = [[], [], []];
      fin = false;
      activeStepIndex = 0;
      push();
      msg =
        'The prior is Beta(' +
        f2(a0) +
        ', ' +
        f2(b0) +
        '): the owner’s best guess is ' +
        pct(a0 / (a0 + b0)) +
        ', held as firmly as if ' +
        f2(a0 + b0) +
        ' customers had already been asked. No real customer has answered yet.';
      eqShow(
        eq,
        ['No answers yet. The belief is the prior, Beta(' + f2(a0) + ', ' + f2(b0) + ').'],
        0,
      );
    };
    const add = (y) => {
      if (fin || D.length >= MAXN) return;
      const n0 = D.length,
        a = a0 + k,
        b = b0 + n0 - k;
      D.push(y);
      k += y;
      const a2 = a0 + k,
        b2 = b0 + D.length - k;
      push();
      const lo = HL[HL.length - 1],
        hi = HU[HU.length - 1];
      activeStepIndex = y ? 2 : 2;
      eqShow(
        eq,
        [
          'Customer ' + D.length + ' says <b>' + (y ? 'yes' : 'no') + '</b>.',
          'belief before: Beta(' +
            f2(a) +
            ', ' +
            f2(b) +
            ')     best estimate ' +
            f2(a) +
            ' ÷ (' +
            f2(a) +
            ' + ' +
            f2(b) +
            ') = ' +
            f3(a / (a + b)),
          (y ? 'a yes adds 1 to the first number' : 'a no adds 1 to the second number') +
            ':   belief after: <b>Beta(' +
            f2(a2) +
            ', ' +
            f2(b2) +
            ')</b>',
          'best estimate ' +
            f2(a2) +
            ' ÷ (' +
            f2(a2) +
            ' + ' +
            f2(b2) +
            ') = <b>' +
            f3(a2 / (a2 + b2)) +
            '</b>     95% of the belief between ' +
            f2(lo) +
            ' and ' +
            f2(hi) +
            ' (width ' +
            f2(hi - lo) +
            ')',
          'for comparison, the plain frequency: ' + k + ' ÷ ' + D.length + ' = ' + f3(k / D.length),
        ],
        0,
      );
      msg =
        'Customer ' +
        D.length +
        ' said ' +
        (y ? 'yes' : 'no') +
        '. The best estimate moved from ' +
        pct(a / (a + b)) +
        ' to ' +
        pct(a2 / (a2 + b2)) +
        ', and 95% of the belief now lies between ' +
        pct(lo) +
        ' and ' +
        pct(hi) +
        '.';
      if (hi - lo < 0.1 || D.length >= MAXN) {
        fin = true;
        activeStepIndex = 5;
        msg =
          (hi - lo < 0.1
            ? 'Stopped after ' +
              D.length +
              ' customers: 95% of the belief now fits in a range narrower than 0.1, from ' +
              pct(lo) +
              ' to ' +
              pct(hi) +
              '.'
            : 'Stopped at the limit of ' + MAXN + ' customers.') +
          ' Best estimate ' +
          pct(a2 / (a2 + b2)) +
          '. The true share is ' +
          pct(settings.p) +
          ', which is ' +
          (settings.p >= lo && settings.p <= hi ? 'inside' : 'outside') +
          ' that range.';
      }
    };
    const tp = testPoint(root, 'by', ['A share to ask about'], [0.5], [[0, 1]]);
    const draw = () => {
      const colors = readThemeColors(),
        n = D.length,
        a = a0 + k,
        b = b0 + n - k;
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          m = { l: 14, r: 14, t: 40, b: 26 },
          X = (v) => m.l + v * (w - m.l - m.r),
          pr = pdf(a0, b0),
          po = pdf(a, b),
          lk = n ? pdf(k + 1, n - k + 1) : null,
          ym = Math.max(...po, ...pr.map((v) => Math.min(v, 6))) * 1.05,
          Y = (v) => h - m.b - (Math.min(v, ym) / ym) * (h - m.t - m.b),
          lo = HL[HL.length - 1],
          hi = HU[HU.length - 1];
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.55);
        ctx.beginPath();
        ctx.moveTo(X(lo), Y(0));
        for (let i = 0; i <= NG; i++) {
          const x = i / NG;
          if (x >= lo && x <= hi) ctx.lineTo(X(x), Y(po[i]));
        }
        ctx.lineTo(X(hi), Y(0));
        ctx.closePath();
        ctx.fill();
        const line = (d, col, lw, dash) => {
          ctx.strokeStyle = col;
          ctx.lineWidth = lw;
          ctx.setLineDash(dash || []);
          ctx.beginPath();
          for (let i = 0; i <= NG; i++)
            i ? ctx.lineTo(X(i / NG), Y(d[i])) : ctx.moveTo(X(0), Y(d[i]));
          ctx.stroke();
          ctx.setLineDash([]);
        };
        if (lk) line(lk, colors.ok, 1.3);
        line(pr, colors.muted, 1.8, [6, 5]);
        line(po, colors.accent, 2.6);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, h - m.b);
        ctx.lineTo(w - m.r, h - m.b);
        ctx.stroke();
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) ctx.fillText(100 * v + '%', X(v), h - 8);
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(X(settings.p), m.t - 6);
        ctx.lineTo(X(settings.p), h - m.b);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colors.cut;
        ctx.textAlign = settings.p > 0.8 ? 'right' : 'left';
        ctx.fillText('true share', X(settings.p) + (settings.p > 0.8 ? -5 : 5), m.t + 4);
        markTP(ctx, X(tp.v[0]), h - m.b, colors, false);
        const show = D.slice(-Math.floor((w - 28) / 9)),
          x0 = 14;
        show.forEach((y, i) => {
          ctx.beginPath();
          ctx.arc(x0 + i * 9 + 4, 14, 3.4, 0, 7);
          if (y) {
            ctx.fillStyle = colors.accent;
            ctx.fill();
          } else {
            ctx.strokeStyle = colors.muted;
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }
        });
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          n
            ? 'answers, oldest to newest: filled = yes, open = no' +
                (n > show.length ? ' (the last ' + show.length + ')' : '')
            : 'no answers yet',
          14,
          31,
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          nn = Math.max(30, posteriorMeanHistory.length),
          A = drawSeriesChart(ctx, colors, 0, 0, w, h / 2, posteriorMeanHistory, {
            title: 'this owner: best estimate and 95% range, by customers asked',
            n: nn,
            ym: 1,
            col: colors.accent,
          });
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.2);
        ctx.beginPath();
        HU.forEach((v, i) => (i ? ctx.lineTo(A.X(i), A.Y(v)) : ctx.moveTo(A.X(i), A.Y(v))));
        for (let i = HL.length - 1; i >= 0; i--) ctx.lineTo(A.X(i), A.Y(HL[i]));
        ctx.closePath();
        ctx.fill();
        const tl = (L, y0) => {
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(L.m.l, L.Y(settings.p));
          ctx.lineTo(w - L.m.r, L.Y(settings.p));
          ctx.stroke();
          ctx.setLineDash([]);
        };
        tl(A);
        const col = [colors.accent, colors.ok, colors.cut],
          nm = ['no opinion', 'mild hunch', 'stubborn'];
        let B = null;
        meanHistoryByPrior.forEach((d, i) => {
          B = drawSeriesChart(ctx, colors, 0, h / 2, w, h / 2, d, {
            title: i ? '' : 'three owners, same answers: their best estimates',
            n: nn,
            ym: 1,
            col: col[i],
          });
        });
        tl(B);
        ctx.font = MONO;
        meanHistoryByPrior.forEach((d, i) => {
          ctx.fillStyle = col[i];
          ctx.textAlign = 'right';
          ctx.fillText(nm[i], w - 14, h / 2 + 38 + i * 13);
        });
        ctx.fillStyle = colors.muted;
        ctx.fillText('customers asked →', w - 12, h - 5);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'by', activeStepIndex);
      const po = pdf(a, b);
      let tot = 0,
        ab = 0;
      for (let i = 0; i <= NG; i++) {
        tot += po[i];
        if (i / NG > tp.v[0]) ab += po[i];
      }
      tp.out.textContent =
        'Given everything so far, the probability that the true share is above ' +
        pct(tp.v[0]) +
        ' is ' +
        pct(ab / tot) +
        '. The probability that the next customer says yes is ' +
        pct(a / (a + b)) +
        '.';
      setReadout(root, 'n', n + (fin ? ' (stopped)' : ''));
      setReadout(root, 'k', String(k));
      setReadout(root, 'mean', pct(a / (a + b)));
      setReadout(root, 'ci', pct(HL[HL.length - 1]) + ' and ' + pct(HU[HU.length - 1]));
      setReadout(root, 'fr', n ? pct(k / n) : '–');
      setReadout(root, 'pb', 'Beta(' + f2(a) + ', ' + f2(b) + ')');
    };
    tp.on = () => draw();
    build();
    if (prefersReducedMotion) {
      while (!fin) add(Math.random() < settings.p ? 1 : 0);
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (frameCount % (D.length < 12 ? 26 : D.length < 60 ? 7 : 2)) return;
      add(Math.random() < settings.p ? 1 : 0);
      draw();
      checkRunStop();
    });
    playback = playBtn($('#by-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'by', {
      done: () => fin,
      restart: () => build(),
      count: () => D.length,
      what: 'customers',
    });
    $('#by-step', root).onclick = () => {
      playback.stop();
      if (fin) build();
      else add(Math.random() < settings.p ? 1 : 0);
      draw();
    };
    $('#by-reset', root).onclick = () => {
      playback.stop();
      build();
      draw();
    };
    $('#by-yes', root).onclick = () => {
      playback.stop();
      add(1);
      draw();
    };
    $('#by-no', root).onclick = () => {
      playback.stop();
      add(0);
      draw();
    };
    const setP = () => {
      a0 = settings.m * settings.n0;
      b0 = (1 - settings.m) * settings.n0;
      if (playback) playback.stop();
      build();
      draw();
    };
    bm = bind(
      root,
      'by-m',
      (v) => {
        settings.m = +v;
        setP();
      },
      f2,
    );
    bn = bind(
      root,
      'by-n',
      (v) => {
        settings.n0 = +v;
        setP();
      },
      (v) => String(v),
    );
    bind(
      root,
      'by-p',
      (v) => {
        settings.p = +v;
        if (playback) {
          playback.stop();
          build();
          draw();
        }
      },
      f2,
    );
    segBind(root, 'by-pr', settings.pr, (v) => {
      const q = PR[v];
      settings.m = q[0] / (q[0] + q[1]);
      settings.n0 = q[0] + q[1];
      bm.set(settings.m);
      bn.set(settings.n0);
      a0 = q[0];
      b0 = q[1];
      playback.stop();
      build();
      draw();
      playback.start();
    });
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        build();
        draw();
        playback.start();
      },
    };
  },
};
