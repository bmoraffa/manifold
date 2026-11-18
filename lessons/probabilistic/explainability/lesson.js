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
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#xp-c', root),
      cr = $('#xp-r', root),
      live = $('#xp-live', root),
      eq = $('#xp-eq', root),
      settings = { x: [150, 12, 4, 4, 1], ft: 0 },
      NF = 5,
      FN = ['floor area', 'age', 'distance', 'bedrooms', 'garden'],
      UN = [' m²', ' years', ' km', '', ''],
      RG = [
        [40, 220],
        [0, 90],
        [0, 30],
        [1, 6],
        [0, 1],
      ],
      MAXN = 1500,
      NB = 60;
    let net,
      MU,
      SD,
      BG = [],
      base = 0,
      pr = 0,
      exactShapleyValues = [],
      SUM,
      n = 0,
      estimatedShapleyValues,
      HIST = [],
      last = null,
      GI = [],
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false,
      coalitionValueCache = null;
    const truth = (a, r) => {
      const [s, ag, d, b, gd] = a,
        near = 1 - d / 34;
      return (
        45 +
        1.5 * s +
        1.1 * s * near -
        0.9 * ag +
        0.004 * ag * ag * 2 -
        3.2 * d +
        9 * b +
        gd * (8 + 0.12 * s * near) +
        (r ? 9 * r() : 0)
      );
    };
    const sampleH = (r) => {
      const s = 45 + 170 * r(),
        ag = 85 * r() * r(),
        d = 30 * r(),
        b = clamp(Math.round(1 + s / 45 + 0.8 * (r() - 0.5) * 2), 1, 6),
        gd = r() < 0.3 + 0.4 * (d / 30) ? 1 : 0;
      return [s, ag, d, b, gd];
    };
    const fz = (rows) => {
      const B = rows.length,
        X = new Float64Array(B * NF);
      rows.forEach((q, i) => {
        for (let j = 0; j < NF; j++) X[i * NF + j] = (q[j] - MU[j]) / SD[j];
      });
      const o = net.fwd(AD.input(X, [B, NF]));
      AD.clear();
      return Array.from(o.v, (v) => v * 100);
    };
    const val = (mask) => {
      const key = mask;
      if (coalitionValueCache.has(key)) return coalitionValueCache.get(key);
      const rows = BG.map((b) => b.map((v, j) => ((mask >> j) & 1 ? settings.x[j] : v))),
        o = fz(rows),
        m = o.reduce((a, b) => a + b, 0) / o.length;
      coalitionValueCache.set(key, m);
      return m;
    };
    const train = () => {
      const r = rng(31),
        rn = randnFrom(r),
        D = Array.from({ length: 700 }, () => sampleH(r)),
        Yv = D.map((q) => truth(q, rn) / 100);
      MU = [0, 1, 2, 3, 4].map((j) => D.reduce((s, q) => s + q[j], 0) / D.length);
      SD = [0, 1, 2, 3, 4].map(
        (j) => Math.sqrt(D.reduce((s, q) => s + (q[j] - MU[j]) ** 2, 0) / D.length) || 1,
      );
      net = AD.mlp([NF, 24, 24, 1], randnFrom(rng(7)), 'tanh');
      const Xs = D.map((q) => q.map((v, j) => (v - MU[j]) / SD[j]));
      for (let ts = 1; ts <= 900; ts++) {
        const B = 64,
          X = new Float64Array(B * NF),
          Yb = new Float64Array(B);
        for (let i = 0; i < B; i++) {
          const k = Math.floor(r() * D.length);
          X.set(Xs[k], i * NF);
          Yb[i] = Yv[k];
        }
        AD.backward(AD.mse(net.fwd(AD.input(X, [B, NF])), Yb));
        AD.adam(net.ps, 0.01 * Math.pow(0.15, ts / 900), ts, 1);
      }
      BG = D.slice(0, NB);
      const keep = settings.x;
      GI = new Array(NF).fill(0);
      const H = D.slice(100, 180);
      for (const hh of H) {
        settings.x = hh;
        coalitionValueCache = new Map();
        const e = exact();
        for (let j = 0; j < NF; j++) GI[j] += Math.abs(e[j]) / H.length;
      }
      settings.x = keep;
    };
    const exact = () => {
      const fact = [1, 1, 2, 6, 24, 120],
        o = new Array(NF).fill(0);
      for (let j = 0; j < NF; j++)
        for (let m = 0; m < 32; m++) {
          if ((m >> j) & 1) continue;
          let sz = 0;
          for (let q = 0; q < NF; q++) if ((m >> q) & 1) sz++;
          o[j] += ((fact[sz] * fact[NF - sz - 1]) / 120) * (val(m | (1 << j)) - val(m));
        }
      return o;
    };
    const build = () => {
      coalitionValueCache = new Map();
      base = val(0);
      pr = val(31);
      exactShapleyValues = exact();
      SUM = new Float64Array(NF);
      estimatedShapleyValues = new Float64Array(NF);
      n = 0;
      HIST = [];
      last = null;
      fin = false;
      activeStepIndex = 0;
      msg =
        'The model estimates this house at ' +
        f2(pr) +
        ' thousand. For an average house it says ' +
        f2(base) +
        '. The difference of ' +
        f2(pr - base) +
        ' has to be shared out among the five features. Press One order or Run.';
      eqShow(eq, ['No orders tried yet.'], 0);
    };
    const order = () => {
      if (fin) return;
      const o = [0, 1, 2, 3, 4]
        .map((j) => [Math.random(), j])
        .sort((a, b) => a[0] - b[0])
        .map((q) => q[1]);
      let m = 0,
        v = base;
      const st = [];
      for (const j of o) {
        m |= 1 << j;
        const v2 = val(m);
        st.push({ j, from: v, to: v2 });
        SUM[j] += v2 - v;
        v = v2;
      }
      n++;
      const prev = Float64Array.from(estimatedShapleyValues);
      for (let j = 0; j < NF; j++) estimatedShapleyValues[j] = SUM[j] / n;
      HIST.push(
        Math.max(...[0, 1, 2, 3, 4].map((j) => Math.abs(estimatedShapleyValues[j] - prev[j]))),
      );
      last = st;
      eqShow(
        eq,
        [
          'Order ' + n + ': ' + o.map((j) => FN[j]).join('  →  '),
          'start from the average house: ' + f2(base),
        ].concat(
          st.map(
            (q) =>
              'reveal ' +
              FN[q.j] +
              ' = ' +
              settings.x[q.j] +
              UN[q.j] +
              ':   ' +
              f2(q.from) +
              ' → ' +
              f2(q.to) +
              '   (<b>' +
              (q.to >= q.from ? '+' : '−') +
              f2(Math.abs(q.to - q.from)) +
              '</b>)   average for ' +
              FN[q.j] +
              ' over ' +
              n +
              ' order' +
              (n > 1 ? 's' : '') +
              ': ' +
              f2(estimatedShapleyValues[q.j]),
          ),
        ),
        0,
      );
      activeStepIndex = n < 2 ? 2 : 4;
      const gp = Math.max(
        ...[0, 1, 2, 3, 4].map((j) => Math.abs(estimatedShapleyValues[j] - exactShapleyValues[j])),
      );
      msg =
        'Order ' +
        n +
        ': ' +
        o.map((j) => FN[j]).join(', ') +
        '. The running averages are within ' +
        f2(gp) +
        ' thousand of the exact Shapley values.';
      if ((n >= 80 && Math.max(...HIST.slice(-60)) < 0.3) || n >= MAXN) {
        fin = true;
        activeStepIndex = 5;
        const idx = [0, 1, 2, 3, 4].sort(
          (a, b) => estimatedShapleyValues[b] - estimatedShapleyValues[a],
        );
        msg =
          (n >= MAXN
            ? 'Stopped at the limit of ' + MAXN + ' orders.'
            : 'Stopped after ' +
              n +
              ' orders: no average has moved by more than 0.3 over the last 60.') +
          ' Check: ' +
          f2(base) +
          ' (average house) ' +
          [0, 1, 2, 3, 4]
            .map(
              (j) =>
                (estimatedShapleyValues[j] >= 0 ? '+ ' : '− ') +
                f2(Math.abs(estimatedShapleyValues[j])),
            )
            .join(' ') +
          ' = ' +
          f2(base + estimatedShapleyValues.reduce((a, b) => a + b, 0)) +
          ', and the model’s estimate is ' +
          f2(pr) +
          '. The largest share is ' +
          FN[idx[0]] +
          '.';
      }
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          h1 = h * 0.44,
          m = { l: 92, r: 16 },
          all = [base, pr].concat(last ? last.flatMap((q) => [q.from, q.to]) : []),
          lo = Math.min(...all) - 12,
          hi = Math.max(...all) + 12,
          X = (v) => m.l + ((v - lo) / (hi - lo)) * (w - m.l - m.r),
          rh = (h1 - 30) / 6;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          last
            ? 'order ' + n + ': from the average house to this house, one feature at a time'
            : 'the first order appears here',
          8,
          13,
        );
        ctx.textAlign = 'right';
        ctx.fillStyle = colors.muted;
        ctx.fillText('average house', m.l - 8, 24 + rh * 0.5 + 4);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(X(base), 20);
        ctx.lineTo(X(base), 20 + rh);
        ctx.stroke();
        ctx.textAlign = 'left';
        ctx.fillText(f2(base), X(base) + 5, 24 + rh * 0.5 + 4);
        if (last)
          last.forEach((q, i) => {
            const y = 20 + (i + 1) * rh,
              up = q.to >= q.from;
            ctx.fillStyle = up ? colors.accent : colors.cut;
            ctx.fillRect(
              Math.min(X(q.from), X(q.to)),
              y + 3,
              Math.max(2, Math.abs(X(q.to) - X(q.from))),
              rh - 6,
            );
            ctx.fillStyle = colors.ink;
            ctx.textAlign = 'right';
            ctx.fillText(FN[q.j], m.l - 8, y + rh / 2 + 4);
            ctx.textAlign = up ? 'left' : 'right';
            ctx.fillStyle = colors.muted;
            ctx.fillText(
              (up ? '+' : '−') + f2(Math.abs(q.to - q.from)),
              X(q.to) + (up ? 5 : -5),
              y + rh / 2 + 4,
            );
            ctx.strokeStyle = colors.line;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(X(q.to), y + rh - 3);
            ctx.lineTo(X(q.to), y + rh + 3);
            ctx.stroke();
          });
        const y0 = h1 + 22,
          H = h - y0 - 8,
          r2 = H / NF,
          mx =
            Math.max(
              10,
              ...exactShapleyValues.map(Math.abs),
              ...Array.from(estimatedShapleyValues).map(Math.abs),
            ) * 1.25,
          X2 = (v) => m.l + (w - m.l - m.r) / 2 + ((v / mx) * (w - m.l - m.r)) / 2;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          'each feature’s share of the price, in thousands: average over ' +
            n +
            ' order' +
            (n === 1 ? '' : 's'),
          8,
          h1 + 12,
        );
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(X2(0), y0);
        ctx.lineTo(X2(0), y0 + H);
        ctx.stroke();
        for (let j = 0; j < NF; j++) {
          const y = y0 + j * r2,
            v = estimatedShapleyValues[j];
          ctx.fillStyle = v >= 0 ? colors.accent : colors.cut;
          if (n) ctx.fillRect(Math.min(X2(0), X2(v)), y + 4, Math.abs(X2(v) - X2(0)), r2 - 8);
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(X2(exactShapleyValues[j]), y + 1);
          ctx.lineTo(X2(exactShapleyValues[j]), y + r2 - 1);
          ctx.stroke();
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'right';
          ctx.fillText(FN[j], m.l - 8, y + r2 / 2 - 2);
          ctx.fillStyle = colors.muted;
          ctx.fillText(settings.x[j] + UN[j], m.l - 8, y + r2 / 2 + 11);
          ctx.textAlign = v >= 0 ? 'left' : 'right';
          ctx.fillText(
            n ? (v >= 0 ? '+' : '−') + f2(Math.abs(v)) : '',
            X2(v) + (v >= 0 ? 6 : -6),
            y + r2 / 2 + 4,
          );
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.4,
          m = { l: 92, r: 40 },
          rh = (h1 - 26) / NF,
          mx = Math.max(...GI) * 1.1;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('average size of each feature’s share, over 80 houses (thousands)', 8, 13);
        for (let j = 0; j < NF; j++) {
          const y = 20 + j * rh;
          ctx.fillStyle =
            j === settings.ft ? colors.gold : mix(colors.accent, colors.accent, 0, 0.75);
          ctx.fillRect(m.l, y + 3, ((w - m.l - m.r) * GI[j]) / mx, rh - 6);
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'right';
          ctx.fillText(FN[j], m.l - 8, y + rh / 2 + 4);
          ctx.textAlign = 'left';
          ctx.fillStyle = colors.muted;
          ctx.fillText(f2(GI[j]), m.l + ((w - m.l - m.r) * GI[j]) / mx + 5, y + rh / 2 + 4);
        }
        const j = settings.ft,
          NP = j >= 3 ? (j === 3 ? 6 : 2) : 30,
          xs = Array.from(
            { length: NP },
            (_, i) => RG[j][0] + ((RG[j][1] - RG[j][0]) * i) / (NP - 1),
          ),
          ice = fz(xs.map((v) => settings.x.map((q, k) => (k === j ? v : q)))),
          pdp = xs.map((v) => {
            const o = fz(BG.map((b) => b.map((q, k) => (k === j ? v : q))));
            return o.reduce((a, b) => a + b, 0) / o.length;
          }),
          lo = Math.min(...ice, ...pdp),
          hi = Math.max(...ice, ...pdp),
          L = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, ice, {
            title: 'estimate as ' + FN[j] + ' changes, everything else held fixed',
            n: NP,
            y1: lo - 5,
            ym: hi + 5,
            col: colors.accent,
            dots: NP < 8,
          });
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        pdp.forEach((v, i) => (i ? ctx.lineTo(L.X(i), L.Y(v)) : ctx.moveTo(L.X(i), L.Y(v))));
        ctx.stroke();
        ctx.setLineDash([]);
        const xi = ((settings.x[j] - RG[j][0]) / (RG[j][1] - RG[j][0])) * (NP - 1);
        ctx.fillStyle = colors.gold;
        ctx.beginPath();
        ctx.arc(L.X(xi), L.Y(pr), 6, 0, 7);
        ctx.fill();
        ctx.textAlign = 'right';
        ctx.fillStyle = colors.accent;
        ctx.fillText('this house', w - 14, h - 48);
        ctx.fillStyle = colors.muted;
        ctx.fillText('dashed: on average over typical houses', w - 14, h - 34);
        ctx.textAlign = 'left';
        ctx.fillText(RG[j][0] + UN[j], L.m.l, h - 5);
        ctx.textAlign = 'right';
        ctx.fillText(RG[j][1] + UN[j], w - L.m.r, h - 5);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'xp', activeStepIndex);
      const src = n ? Array.from(estimatedShapleyValues) : exactShapleyValues,
        idx = [0, 1, 2, 3, 4].sort((a, b) => src[b] - src[a]);
      setReadout(root, 'pr', f2(pr) + ' thousand');
      setReadout(root, 'ba', f2(base) + ' thousand');
      setReadout(root, 'n', n + (fin ? ' (stopped)' : ''));
      setReadout(root, 'top', n ? FN[idx[0]] + ' (+' + f2(src[idx[0]]) + ')' : '–');
      setReadout(
        root,
        'bot',
        n
          ? src[idx[4]] < 0
            ? FN[idx[4]] + ' (−' + f2(-src[idx[4]]) + ')'
            : 'none takes away'
          : '–',
      );
      setReadout(
        root,
        'gap',
        n
          ? f2(
              Math.max(
                ...[0, 1, 2, 3, 4].map((j) =>
                  Math.abs(estimatedShapleyValues[j] - exactShapleyValues[j]),
                ),
              ),
            )
          : '–',
      );
    };
    train();
    build();
    if (prefersReducedMotion) {
      while (!fin) order();
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (frameCount % (n < 6 ? 42 : n < 30 ? 8 : 2)) return;
      order();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#xp-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'xp', {
      done: () => fin,
      restart: () => build(),
      count: () => n,
      what: 'orders',
    });
    $('#xp-step', root).onclick = () => {
      playback.stop();
      if (fin) build();
      else order();
      draw();
    };
    $('#xp-reset', root).onclick = () => {
      playback.stop();
      build();
      draw();
    };
    for (let j = 0; j < NF; j++)
      bind(
        root,
        'xp-f' + j,
        (v) => {
          settings.x[j] = +v;
          if (playback) {
            playback.stop();
            build();
            draw();
          }
        },
        (v) => (j === 4 ? (+v ? 'yes' : 'no') : String(v)),
      );
    $('#xp-ft', root).onchange = (e) => {
      settings.ft = +e.target.value;
      draw();
    };
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
