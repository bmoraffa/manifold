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
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#ts-c', root),
      cr = $('#ts-r', root),
      live = $('#ts-live', root),
      eq = $('#ts-eq', root),
      st = $('#ts-st', root),
      settings = { m: 2, s: 12, a: 0.3, p: 13, d: 'ice' },
      NM = ['Last Value', 'Same Season Last Cycle', 'Holt-Winters', 'Autoregression'],
      HZ = 24;
    let Yd = [],
      t0 = 0,
      t = 0,
      backtestForecasts = [[], [], [], []],
      runningMaeByModel = [[0], [0], [0], [0]],
      absoluteErrorsByModel = [[], [], [], []],
      hw = null,
      phi = null,
      ACF = [],
      FUT = null,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      fin = false,
      custom = null,
      unit = '';
    const mk = (d) => {
      const r = rng(d === 'ice' ? 11 : d === 'temp' ? 23 : d === 'walk' ? 37 : 51),
        rn = randnFrom(r),
        o = [];
      if (d === 'ice')
        for (let i = 0; i < 96; i++)
          o.push(
            40 +
              0.45 * i +
              18 * Math.sin((((i % 12) - 3) / 12) * 2 * Math.PI) +
              6 * Math.max(0, Math.sin((((i % 12) - 3) / 12) * 2 * Math.PI)) +
              3.2 * rn(),
          );
      else if (d === 'temp')
        for (let i = 0; i < 96; i++)
          o.push(12 + 9 * Math.sin((((i % 12) - 3.5) / 12) * 2 * Math.PI) + 1.6 * rn());
      else if (d === 'walk') {
        let v = 100;
        for (let i = 0; i < 120; i++) {
          v += 2.2 * rn();
          o.push(v);
        }
      } else
        for (let i = 0; i < 112; i++)
          o.push(500 + 0.8 * i + [60, 80, 70, 65, 40, -150, -170][i % 7] + 25 * rn());
      return o;
    };
    const acf = (y) => {
      const n = y.length,
        m = y.reduce((a, b) => a + b, 0) / n;
      let v = 0;
      for (const q of y) v += (q - m) ** 2;
      const o = [];
      for (let k = 1; k <= 24; k++) {
        let s = 0;
        for (let i = k; i < n; i++) s += (y[i] - m) * (y[i - k] - m);
        o.push(v ? s / v : 0);
      }
      return o;
    };
    const hwInit = () => {
      const m = settings.s,
        y = Yd,
        n = Math.min(t0, 2 * m),
        L = y.slice(0, m).reduce((a, b) => a + b, 0) / m,
        L2 = n >= 2 * m ? y.slice(m, 2 * m).reduce((a, b) => a + b, 0) / m : L;
      hw = {
        l: L,
        b: n >= 2 * m ? (L2 - L) / m : 0,
        s: Array.from({ length: m }, (_, i) => y[i] - L),
      };
      for (let i = 0; i < t0; i++) hwUp(y[i], i);
    };
    const hwUp = (y, i) => {
      const m = settings.s,
        a = settings.a,
        so = hw.s[i % m],
        l0 = hw.l;
      hw.l = a * (y - so) + (1 - a) * (hw.l + hw.b);
      hw.b = 0.05 * (hw.l - l0) + 0.95 * hw.b;
      hw.s[i % m] = 0.25 * (y - hw.l) + 0.75 * so;
    };
    const arFit = (n) => {
      const p = Math.min(settings.p, Math.floor(n / 3)),
        k = p + 1,
        A = Array.from({ length: k }, () => new Float64Array(k)),
        b = new Float64Array(k);
      for (let i = p; i < n; i++) {
        const x = [1];
        for (let j = 1; j <= p; j++) x.push(Yd[i - j]);
        for (let r = 0; r < k; r++) {
          b[r] += x[r] * Yd[i];
          for (let q = 0; q < k; q++) A[r][q] += x[r] * x[q];
        }
      }
      for (let r = 1; r < k; r++) A[r][r] += 1e-6 * (A[r][r] + 1);
      for (let i = 0; i < k; i++) {
        let pv = i;
        for (let r = i + 1; r < k; r++) if (Math.abs(A[r][i]) > Math.abs(A[pv][i])) pv = r;
        [A[i], A[pv]] = [A[pv], A[i]];
        [b[i], b[pv]] = [b[pv], b[i]];
        const d = A[i][i] || 1e-12;
        for (let r = i + 1; r < k; r++) {
          const f = A[r][i] / d;
          for (let q = i; q < k; q++) A[r][q] -= f * A[i][q];
          b[r] -= f * b[i];
        }
      }
      const w = new Float64Array(k);
      for (let i = k - 1; i >= 0; i--) {
        let s = b[i];
        for (let q = i + 1; q < k; q++) s -= A[i][q] * w[q];
        w[i] = s / (A[i][i] || 1e-12);
      }
      return w;
    };
    const fc = (n) => {
      const m = settings.s,
        p = phi.length - 1;
      let ar = phi[0];
      for (let j = 1; j <= p; j++) ar += phi[j] * Yd[n - j];
      return [Yd[n - 1], n - m >= 0 ? Yd[n - m] : Yd[n - 1], hw.l + hw.b + hw.s[n % m], ar];
    };
    const build = () => {
      Yd = custom || mk(settings.d);
      const n = Yd.length;
      t0 = clamp(Math.max(Math.round(n * 0.4), 3 * settings.s), 8, n - 6);
      t = t0;
      backtestForecasts = [[], [], [], []];
      runningMaeByModel = [[], [], [], []];
      absoluteErrorsByModel = [[], [], [], []];
      FUT = null;
      fin = false;
      activeStepIndex = 0;
      ACF = acf(Yd);
      hwInit();
      phi = arFit(t);
      msg =
        'The first ' +
        t0 +
        ' of ' +
        n +
        ' points are revealed. The rest are hidden. Each forecaster will now predict the next point before it is shown. Press One step or Run.';
      eqShow(eq, ['The walk has not started.'], 0);
    };
    const explain = (n, f) => {
      const m = settings.s,
        w = settings.m,
        L = [
          'Predicting point ' +
            (n + 1) +
            ' from the first ' +
            n +
            '. Watching: <b>' +
            NM[w] +
            '</b>.',
        ];
      if (w === 0) L.push('forecast = the last value seen = <b>' + f2(f[0]) + '</b>');
      else if (w === 1)
        L.push(
          'forecast = the value ' +
            m +
            ' steps ago (point ' +
            (n + 1 - m) +
            ') = <b>' +
            f2(f[1]) +
            '</b>',
        );
      else if (w === 2)
        L.push(
          'level ' +
            f2(hw.l) +
            '  +  slope ' +
            f3(hw.b) +
            '  +  offset for this place in the cycle ' +
            f2(hw.s[n % m]) +
            '  =  <b>' +
            f2(f[2]) +
            '</b>',
        );
      else {
        const p = phi.length - 1,
          top = [...Array(p).keys()]
            .map((j) => [Math.abs(phi[j + 1] * Yd[n - 1 - j]), j + 1])
            .sort((a, b) => b[0] - a[0])
            .slice(0, 3)
            .sort((a, b) => a[1] - b[1]);
        L.push(
          'constant ' +
            f2(phi[0]) +
            top
              .map(
                (q) =>
                  '  +  ' +
                  f3(phi[q[1]]) +
                  ' × value ' +
                  q[1] +
                  ' step' +
                  (q[1] > 1 ? 's' : '') +
                  ' back (' +
                  f2(Yd[n - q[1]]) +
                  ')',
              )
              .join('') +
            (p > 3 ? '  +  ' + (p - 3) + ' smaller terms' : '') +
            '  =  <b>' +
            f2(f[3]) +
            '</b>',
        );
        L.push(
          'largest weights: ' +
            [...Array(p).keys()]
              .map((j) => [Math.abs(phi[j + 1]), j + 1])
              .sort((a, b) => b[0] - a[0])
              .slice(0, 3)
              .map((q) => 'lag ' + q[1] + ': ' + f2(phi[q[1]]))
              .join(',  '),
        );
      }
      return L;
    };
    const step = () => {
      if (fin) return;
      const n = t,
        f = fc(n),
        y = Yd[n],
        L = explain(n, f);
      for (let k = 0; k < 4; k++) {
        backtestForecasts[k].push(f[k]);
        absoluteErrorsByModel[k].push(Math.abs(y - f[k]));
        runningMaeByModel[k].push(
          absoluteErrorsByModel[k].reduce((a, b) => a + b, 0) / absoluteErrorsByModel[k].length,
        );
      }
      L.push(
        'the true value is revealed: ' +
          f2(y) +
          '.  This forecaster missed by <b>' +
          f2(Math.abs(y - f[settings.m])) +
          '</b>',
      );
      eqShow(eq, L, 0);
      hwUp(y, n);
      t++;
      phi = arFit(t);
      activeStepIndex = 2;
      const bestk = [0, 1, 2, 3].sort(
        (a, b) =>
          runningMaeByModel[a][runningMaeByModel[a].length - 1] -
          runningMaeByModel[b][runningMaeByModel[b].length - 1],
      )[0];
      msg =
        'Point ' +
        t +
        ' revealed: ' +
        f2(y) +
        '. ' +
        NM[settings.m] +
        ' had forecast ' +
        f2(f[settings.m]) +
        '. Smallest average miss so far: ' +
        NM[bestk] +
        ', ' +
        f2(runningMaeByModel[bestk][runningMaeByModel[bestk].length - 1]) +
        '.';
      if (t >= Yd.length) {
        fin = true;
        activeStepIndex = 5;
        future();
        msg =
          'The walk is finished. Average miss: ' +
          [0, 1, 2, 3]
            .map((k) => NM[k] + ' ' + f2(runningMaeByModel[k][runningMaeByModel[k].length - 1]))
            .join(', ') +
          '. The forecast for the next ' +
          HZ +
          ' points by ' +
          NM[settings.m] +
          ' is drawn on the right of the series.';
      }
    };
    const future = () => {
      const n = Yd.length,
        m = settings.s,
        ext = Yd.slice(),
        p = phi.length - 1,
        o = [];
      for (let h = 1; h <= HZ; h++) {
        let v;
        if (settings.m === 0) v = Yd[n - 1];
        else if (settings.m === 1) v = ext[n + h - 1 - m];
        else if (settings.m === 2) v = hw.l + h * hw.b + hw.s[(n + h - 1) % m];
        else {
          v = phi[0];
          for (let j = 1; j <= p; j++) v += phi[j] * ext[n + h - 1 - j];
        }
        ext.push(v);
        o.push(v);
      }
      const e = absoluteErrorsByModel[settings.m],
        sd = Math.sqrt(e.reduce((a, b) => a + b * b, 0) / Math.max(1, e.length));
      FUT = { v: o, sd };
    };
    const tp = testPoint(root, 'ts', ['Steps ahead'], [6], [[1, HZ]]);
    const draw = () => {
      const tt = readThemeColors(),
        n = Yd.length,
        tot = n + HZ,
        nf = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : f2(v));
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          m = { l: 44, r: 10, t: 12, b: 22 },
          all = Yd.concat(FUT ? FUT.v : []),
          lo = Math.min(...all),
          hi = Math.max(...all),
          pd = (hi - lo) * 0.3 || 1,
          X = (i) => m.l + (i / (tot - 1)) * (w - m.l - m.r),
          Yv = (v) => h - m.b - ((v - lo + pd) / (hi - lo + 2 * pd)) * (h - m.t - m.b);
        ctx.font = MONO;
        ctx.strokeStyle = tt.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, h - m.b);
        ctx.lineTo(w - m.r, h - m.b);
        ctx.stroke();
        ctx.fillStyle = tt.muted;
        ctx.textAlign = 'right';
        ctx.fillText(nf(lo), m.l - 4, Yv(lo) + 4);
        ctx.fillText(nf(hi), m.l - 4, Yv(hi) + 4);
        ctx.save();
        ctx.beginPath();
        ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
        ctx.clip();
        ctx.fillStyle = mix(tt.ink, tt.ink, 0, 0.06);
        ctx.fillRect(X(0), m.t, X(t0 - 1) - X(0), h - m.t - m.b);
        ctx.strokeStyle = mix(tt.ink, tt.ink, 0, 0.2);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let i = Math.max(0, t - 1); i < n; i++)
          i === Math.max(0, t - 1) ? ctx.moveTo(X(i), Yv(Yd[i])) : ctx.lineTo(X(i), Yv(Yd[i]));
        ctx.stroke();
        ctx.strokeStyle = tt.ink;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let i = 0; i < t; i++) i ? ctx.lineTo(X(i), Yv(Yd[i])) : ctx.moveTo(X(i), Yv(Yd[i]));
        ctx.stroke();
        const f = backtestForecasts[settings.m];
        ctx.strokeStyle = mix(tt.gold, tt.gold, 0, 0.8);
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        f.forEach((v, i) => (i ? ctx.lineTo(X(t0 + i), Yv(v)) : ctx.moveTo(X(t0 + i), Yv(v))));
        ctx.stroke();
        ctx.fillStyle = tt.gold;
        f.forEach((v, i) => {
          ctx.beginPath();
          ctx.arc(X(t0 + i), Yv(v), 2.6, 0, 7);
          ctx.fill();
        });
        if (!fin && t < n) {
          const nx = fc(t)[settings.m];
          ctx.strokeStyle = tt.cut;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(X(t), Yv(nx), 6, 0, 7);
          ctx.stroke();
        }
        if (FUT) {
          ctx.fillStyle = mix(tt.accent, tt.accent, 0, 0.2);
          ctx.beginPath();
          FUT.v.forEach((v, i) => {
            const y = Yv(v + 2 * FUT.sd * Math.sqrt(i + 1));
            i ? ctx.lineTo(X(n + i), y) : ctx.moveTo(X(n + i), y);
          });
          for (let i = HZ - 1; i >= 0; i--)
            ctx.lineTo(X(n + i), Yv(FUT.v[i] - 2 * FUT.sd * Math.sqrt(i + 1)));
          ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = tt.accent;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          ctx.moveTo(X(n - 1), Yv(Yd[n - 1]));
          FUT.v.forEach((v, i) => ctx.lineTo(X(n + i), Yv(v)));
          ctx.stroke();
          const hh = clamp(Math.round(tp.v[0]), 1, HZ);
          markTP(ctx, X(n + hh - 1), Yv(FUT.v[hh - 1]), tt, false);
        }
        ctx.restore();
        ctx.fillStyle = tt.muted;
        ctx.textAlign = 'left';
        ctx.fillText('history given at the start', X(0) + 4, m.t + 10);
        ctx.textAlign = 'right';
        ctx.fillText(fin ? 'the future →' : '', w - m.r, m.t + 10);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.42,
          m = { l: 44, r: 12, t: 26, b: 20 },
          bw = (w - m.l - m.r) / 24,
          Y0 = m.t + (h1 - m.t - m.b) / 2,
          sc = (h1 - m.t - m.b) / 2;
        ctx.font = MONO;
        ctx.fillStyle = tt.muted;
        ctx.textAlign = 'left';
        ctx.fillText('resemblance to the value k steps earlier (autocorrelation)', 8, 13);
        ACF.forEach((v, i) => {
          ctx.fillStyle = i + 1 === settings.s ? tt.gold : mix(tt.accent, tt.accent, 0, 0.7);
          ctx.fillRect(m.l + i * bw + 1, v >= 0 ? Y0 - v * sc : Y0, bw - 2, Math.abs(v) * sc);
          if ((i + 1) % 2 === 0 || i === 0) {
            ctx.fillStyle = i + 1 === settings.s ? tt.cut : tt.muted;
            ctx.textAlign = 'center';
            ctx.fillText(String(i + 1), m.l + (i + 0.5) * bw, h1 - 4);
          }
        });
        ctx.strokeStyle = tt.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, Y0);
        ctx.lineTo(w - m.r, Y0);
        ctx.stroke();
        ctx.textAlign = 'right';
        ctx.fillStyle = tt.muted;
        ctx.fillText('1', m.l - 5, Y0 - sc + 4);
        ctx.fillText('−1', m.l - 5, Y0 + sc + 4);
        const col = [tt.muted, tt.ok, tt.accent, tt.cut],
          nn = Math.max(10, n - t0 + 1),
          ym =
            Math.max(
              1e-6,
              ...runningMaeByModel.map((a) =>
                a.length ? Math.max(...a.slice(Math.min(a.length - 1, 4))) : 0,
              ),
            ) * 1.1;
        let L = null;
        runningMaeByModel.forEach((a, k) => {
          L = drawSeriesChart(
            ctx,
            tt,
            0,
            h1,
            w - 150,
            h - h1,
            a.map((v) => Math.min(v, ym)),
            { title: k ? '' : 'average miss so far (lower is better)', n: nn, ym, col: col[k] },
          );
        });
        const SH = ['Last Value', 'Same Season', 'Holt-Winters', 'Autoregression'];
        runningMaeByModel.forEach((a, k) => {
          ctx.fillStyle = col[k];
          ctx.textAlign = 'left';
          ctx.font = (k === settings.m ? '700 ' : '') + '11px ' + MONO.split('px ')[1];
          ctx.fillText(SH[k], w - 146, h1 + 40 + k * 30);
          ctx.fillText(a.length ? nf(a[a.length - 1]) : '–', w - 146, h1 + 53 + k * 30);
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'ts', activeStepIndex);
      if (FUT) {
        const hh = clamp(Math.round(tp.v[0]), 1, HZ),
          v = FUT.v[hh - 1],
          r = 2 * FUT.sd * Math.sqrt(hh);
        tp.out.textContent =
          hh +
          ' step' +
          (hh > 1 ? 's' : '') +
          ' ahead, ' +
          NM[settings.m] +
          ' forecasts ' +
          f2(v) +
          '. A rough range for what will actually happen: ' +
          f2(v - r) +
          ' to ' +
          f2(v + r) +
          '.';
      } else
        tp.out.textContent =
          'Finish the walk first: the forecast for the future appears at the end.';
      setReadout(root, 't', t + ' of ' + n);
      setReadout(root, 'f', t < n ? f2(fc(t)[settings.m]) : FUT ? f2(FUT.v[0]) : '–');
      for (let k = 0; k < 4; k++)
        setReadout(
          root,
          'e' + k,
          runningMaeByModel[k].length
            ? f2(runningMaeByModel[k][runningMaeByModel[k].length - 1])
            : '–',
        );
    };
    tp.on = () => draw();
    build();
    if (prefersReducedMotion) {
      while (!fin) step();
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (frameCount % (t - t0 < 5 ? 30 : 5)) return;
      step();
      draw();
      if (fin) playback.stop();
    });
    playback = playBtn($('#ts-play', root), anim);
    attachRunLimit(playback, root, 'ts', { done: () => fin, restart: () => build() });
    const re = (go) => {
      if (!playback) return;
      playback.stop();
      build();
      draw();
      if (go) playback.start();
    };
    $('#ts-step', root).onclick = () => {
      playback.stop();
      if (fin) build();
      else step();
      draw();
    };
    $('#ts-reset', root).onclick = () => re(false);
    const bs = bind(
      root,
      'ts-s',
      (v) => {
        settings.s = +v;
        re(false);
      },
      (v) => String(v),
    );
    bind(
      root,
      'ts-a',
      (v) => {
        settings.a = +v;
        re(false);
      },
      f2,
    );
    bind(
      root,
      'ts-p',
      (v) => {
        settings.p = +v;
        re(false);
      },
      (v) => String(v),
    );
    segBind(root, 'ts-m', String(settings.m), (v) => {
      settings.m = +v;
      if (fin) future();
      draw();
    });
    $('#ts-d', root).onchange = (e) => {
      settings.d = e.target.value;
      custom = null;
      st.textContent = '';
      const m = settings.d === 'traffic' ? 7 : 12;
      settings.s = m;
      bs.set(m);
      re(true);
    };
    $('#ts-file', root).onchange = (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (!f) return;
      const r = new FileReader();
      r.onerror = () => {
        st.textContent = 'That file could not be read. Save it as plain CSV and try again.';
      };
      r.onload = () => {
        const rows = String(r.result)
          .split(/\r?\n/)
          .map((l) => l.split(/[,;\t]/).map((x) => parseFloat(x)));
        let best = [];
        for (let cI = 0; cI < 6; cI++) {
          const col = rows.map((q) => q[cI]).filter((v) => isFinite(v));
          if (col.length > best.length) best = col;
        }
        if (best.length < 30) {
          st.textContent =
            'At least 30 numbers are needed, one per line or in one column. This file has ' +
            best.length +
            '.';
          return;
        }
        custom = best.slice(0, 400);
        st.textContent =
          'Loaded ' +
          f.name +
          ': ' +
          custom.length +
          ' values from its longest numeric column. Set the cycle length to match your data.';
        re(true);
      };
      r.readAsText(f);
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
