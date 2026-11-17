/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  eqShow,
  f2,
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
import { segBind, uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#ts-c', root),
      cr = $('#ts-r', root),
      live = $('#ts-live', root),
      eq = $('#ts-eq', root),
      st = $('#ts-st', root),
      S = { m: 2, s: 12, a: 0.3, p: 13, d: 'ice' },
      NM = ['Last Value', 'Same Season Last Cycle', 'Holt-Winters', 'Autoregression'],
      HZ = 24;
    let Yd = [],
      t0 = 0,
      t = 0,
      F = [[], [], [], []],
      AE = [[0], [0], [0], [0]],
      ER = [[], [], [], []],
      hw = null,
      phi = null,
      ACF = [],
      FUT = null,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
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
      const m = S.s,
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
      const m = S.s,
        a = S.a,
        so = hw.s[i % m],
        l0 = hw.l;
      hw.l = a * (y - so) + (1 - a) * (hw.l + hw.b);
      hw.b = 0.05 * (hw.l - l0) + 0.95 * hw.b;
      hw.s[i % m] = 0.25 * (y - hw.l) + 0.75 * so;
    };
    const arFit = (n) => {
      const p = Math.min(S.p, Math.floor(n / 3)),
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
      const m = S.s,
        p = phi.length - 1;
      let ar = phi[0];
      for (let j = 1; j <= p; j++) ar += phi[j] * Yd[n - j];
      return [Yd[n - 1], n - m >= 0 ? Yd[n - m] : Yd[n - 1], hw.l + hw.b + hw.s[n % m], ar];
    };
    const build = () => {
      Yd = custom || mk(S.d);
      const n = Yd.length;
      t0 = clamp(Math.max(Math.round(n * 0.4), 3 * S.s), 8, n - 6);
      t = t0;
      F = [[], [], [], []];
      AE = [[], [], [], []];
      ER = [[], [], [], []];
      FUT = null;
      fin = false;
      cu = 0;
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
      const m = S.s,
        w = S.m,
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
        F[k].push(f[k]);
        ER[k].push(Math.abs(y - f[k]));
        AE[k].push(ER[k].reduce((a, b) => a + b, 0) / ER[k].length);
      }
      L.push(
        'the true value is revealed: ' +
          f2(y) +
          '.  This forecaster missed by <b>' +
          f2(Math.abs(y - f[S.m])) +
          '</b>',
      );
      eqShow(eq, L, 0);
      hwUp(y, n);
      t++;
      phi = arFit(t);
      cu = 2;
      const bestk = [0, 1, 2, 3].sort(
        (a, b) => AE[a][AE[a].length - 1] - AE[b][AE[b].length - 1],
      )[0];
      msg =
        'Point ' +
        t +
        ' revealed: ' +
        f2(y) +
        '. ' +
        NM[S.m] +
        ' had forecast ' +
        f2(f[S.m]) +
        '. Smallest average miss so far: ' +
        NM[bestk] +
        ', ' +
        f2(AE[bestk][AE[bestk].length - 1]) +
        '.';
      if (t >= Yd.length) {
        fin = true;
        cu = 5;
        future();
        msg =
          'The walk is finished. Average miss: ' +
          [0, 1, 2, 3].map((k) => NM[k] + ' ' + f2(AE[k][AE[k].length - 1])).join(', ') +
          '. The forecast for the next ' +
          HZ +
          ' points by ' +
          NM[S.m] +
          ' is drawn on the right of the series.';
      }
    };
    const future = () => {
      const n = Yd.length,
        m = S.s,
        ext = Yd.slice(),
        p = phi.length - 1,
        o = [];
      for (let h = 1; h <= HZ; h++) {
        let v;
        if (S.m === 0) v = Yd[n - 1];
        else if (S.m === 1) v = ext[n + h - 1 - m];
        else if (S.m === 2) v = hw.l + h * hw.b + hw.s[(n + h - 1) % m];
        else {
          v = phi[0];
          for (let j = 1; j <= p; j++) v += phi[j] * ext[n + h - 1 - j];
        }
        ext.push(v);
        o.push(v);
      }
      const e = ER[S.m],
        sd = Math.sqrt(e.reduce((a, b) => a + b * b, 0) / Math.max(1, e.length));
      FUT = { v: o, sd };
    };
    const tp = testPoint(root, 'ts', ['Steps ahead'], [6], [[1, HZ]]);
    const draw = () => {
      const tt = T(),
        n = Yd.length,
        tot = n + HZ,
        nf = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : f2(v));
      {
        const { g, w, h } = fit(c),
          m = { l: 44, r: 10, t: 12, b: 22 },
          all = Yd.concat(FUT ? FUT.v : []),
          lo = Math.min(...all),
          hi = Math.max(...all),
          pd = (hi - lo) * 0.3 || 1,
          X = (i) => m.l + (i / (tot - 1)) * (w - m.l - m.r),
          Yv = (v) => h - m.b - ((v - lo + pd) / (hi - lo + 2 * pd)) * (h - m.t - m.b);
        g.font = MONO;
        g.strokeStyle = tt.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h - m.b);
        g.lineTo(w - m.r, h - m.b);
        g.stroke();
        g.fillStyle = tt.muted;
        g.textAlign = 'right';
        g.fillText(nf(lo), m.l - 4, Yv(lo) + 4);
        g.fillText(nf(hi), m.l - 4, Yv(hi) + 4);
        g.save();
        g.beginPath();
        g.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
        g.clip();
        g.fillStyle = mix(tt.ink, tt.ink, 0, 0.06);
        g.fillRect(X(0), m.t, X(t0 - 1) - X(0), h - m.t - m.b);
        g.strokeStyle = mix(tt.ink, tt.ink, 0, 0.2);
        g.lineWidth = 1.2;
        g.beginPath();
        for (let i = Math.max(0, t - 1); i < n; i++)
          i === Math.max(0, t - 1) ? g.moveTo(X(i), Yv(Yd[i])) : g.lineTo(X(i), Yv(Yd[i]));
        g.stroke();
        g.strokeStyle = tt.ink;
        g.lineWidth = 1.8;
        g.beginPath();
        for (let i = 0; i < t; i++) i ? g.lineTo(X(i), Yv(Yd[i])) : g.moveTo(X(i), Yv(Yd[i]));
        g.stroke();
        const f = F[S.m];
        g.strokeStyle = mix(tt.gold, tt.gold, 0, 0.8);
        g.lineWidth = 1.4;
        g.beginPath();
        f.forEach((v, i) => (i ? g.lineTo(X(t0 + i), Yv(v)) : g.moveTo(X(t0 + i), Yv(v))));
        g.stroke();
        g.fillStyle = tt.gold;
        f.forEach((v, i) => {
          g.beginPath();
          g.arc(X(t0 + i), Yv(v), 2.6, 0, 7);
          g.fill();
        });
        if (!fin && t < n) {
          const nx = fc(t)[S.m];
          g.strokeStyle = tt.cut;
          g.lineWidth = 2;
          g.beginPath();
          g.arc(X(t), Yv(nx), 6, 0, 7);
          g.stroke();
        }
        if (FUT) {
          g.fillStyle = mix(tt.accent, tt.accent, 0, 0.2);
          g.beginPath();
          FUT.v.forEach((v, i) => {
            const y = Yv(v + 2 * FUT.sd * Math.sqrt(i + 1));
            i ? g.lineTo(X(n + i), y) : g.moveTo(X(n + i), y);
          });
          for (let i = HZ - 1; i >= 0; i--)
            g.lineTo(X(n + i), Yv(FUT.v[i] - 2 * FUT.sd * Math.sqrt(i + 1)));
          g.closePath();
          g.fill();
          g.strokeStyle = tt.accent;
          g.lineWidth = 2.4;
          g.beginPath();
          g.moveTo(X(n - 1), Yv(Yd[n - 1]));
          FUT.v.forEach((v, i) => g.lineTo(X(n + i), Yv(v)));
          g.stroke();
          const hh = clamp(Math.round(tp.v[0]), 1, HZ);
          markTP(g, X(n + hh - 1), Yv(FUT.v[hh - 1]), tt, false);
        }
        g.restore();
        g.fillStyle = tt.muted;
        g.textAlign = 'left';
        g.fillText('history given at the start', X(0) + 4, m.t + 10);
        g.textAlign = 'right';
        g.fillText(fin ? 'the future →' : '', w - m.r, m.t + 10);
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.42,
          m = { l: 44, r: 12, t: 26, b: 20 },
          bw = (w - m.l - m.r) / 24,
          Y0 = m.t + (h1 - m.t - m.b) / 2,
          sc = (h1 - m.t - m.b) / 2;
        g.font = MONO;
        g.fillStyle = tt.muted;
        g.textAlign = 'left';
        g.fillText('resemblance to the value k steps earlier (autocorrelation)', 8, 13);
        ACF.forEach((v, i) => {
          g.fillStyle = i + 1 === S.s ? tt.gold : mix(tt.accent, tt.accent, 0, 0.7);
          g.fillRect(m.l + i * bw + 1, v >= 0 ? Y0 - v * sc : Y0, bw - 2, Math.abs(v) * sc);
          if ((i + 1) % 2 === 0 || i === 0) {
            g.fillStyle = i + 1 === S.s ? tt.cut : tt.muted;
            g.textAlign = 'center';
            g.fillText(String(i + 1), m.l + (i + 0.5) * bw, h1 - 4);
          }
        });
        g.strokeStyle = tt.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, Y0);
        g.lineTo(w - m.r, Y0);
        g.stroke();
        g.textAlign = 'right';
        g.fillStyle = tt.muted;
        g.fillText('1', m.l - 5, Y0 - sc + 4);
        g.fillText('−1', m.l - 5, Y0 + sc + 4);
        const col = [tt.muted, tt.ok, tt.accent, tt.cut],
          nn = Math.max(10, n - t0 + 1),
          ym =
            Math.max(
              1e-6,
              ...AE.map((a) => (a.length ? Math.max(...a.slice(Math.min(a.length - 1, 4))) : 0)),
            ) * 1.1;
        let L = null;
        AE.forEach((a, k) => {
          L = uLine(
            g,
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
        AE.forEach((a, k) => {
          g.fillStyle = col[k];
          g.textAlign = 'left';
          g.font = (k === S.m ? '700 ' : '') + '11px ' + MONO.split('px ')[1];
          g.fillText(SH[k], w - 146, h1 + 40 + k * 30);
          g.fillText(a.length ? nf(a[a.length - 1]) : '–', w - 146, h1 + 53 + k * 30);
        });
      }
      live.textContent = msg;
      uCur(root, 'ts', cu);
      if (FUT) {
        const hh = clamp(Math.round(tp.v[0]), 1, HZ),
          v = FUT.v[hh - 1],
          r = 2 * FUT.sd * Math.sqrt(hh);
        tp.out.textContent =
          hh +
          ' step' +
          (hh > 1 ? 's' : '') +
          ' ahead, ' +
          NM[S.m] +
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
      setR(root, 't', t + ' of ' + n);
      setR(root, 'f', t < n ? f2(fc(t)[S.m]) : FUT ? f2(FUT.v[0]) : '–');
      for (let k = 0; k < 4; k++)
        setR(root, 'e' + k, AE[k].length ? f2(AE[k][AE[k].length - 1]) : '–');
    };
    tp.on = () => draw();
    build();
    if (RM) {
      while (!fin) step();
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      if (fr % (t - t0 < 5 ? 30 : 5)) return;
      step();
      draw();
      if (fin) pb.stop();
    });
    pb = playBtn($('#ts-play', root), anim);
    runCtl(pb, root, 'ts', { done: () => fin, restart: () => build() });
    const re = (go) => {
      if (!pb) return;
      pb.stop();
      build();
      draw();
      if (go) pb.start();
    };
    $('#ts-step', root).onclick = () => {
      pb.stop();
      if (fin) build();
      else step();
      draw();
    };
    $('#ts-reset', root).onclick = () => re(false);
    const bs = bind(
      root,
      'ts-s',
      (v) => {
        S.s = +v;
        re(false);
      },
      (v) => String(v),
    );
    bind(
      root,
      'ts-a',
      (v) => {
        S.a = +v;
        re(false);
      },
      f2,
    );
    bind(
      root,
      'ts-p',
      (v) => {
        S.p = +v;
        re(false);
      },
      (v) => String(v),
    );
    segBind(root, 'ts-m', String(S.m), (v) => {
      S.m = +v;
      if (fin) future();
      draw();
    });
    $('#ts-d', root).onchange = (e) => {
      S.d = e.target.value;
      custom = null;
      st.textContent = '';
      const m = S.d === 'traffic' ? 7 : 12;
      S.s = m;
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
      stop: () => pb.stop(),
      auto: () => {
        build();
        draw();
        pb.start();
      },
    };
  },
};
