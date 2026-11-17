/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  f3,
  fit,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { laChol, laInv, laMul, laT, laTop } from '../../../js/math/linear-algebra.js';
import { segBind, uCur, uLine } from '../shared.js';

export default {
  init(root) {
    const c = $('#fa-c', root),
      cr = $('#fa-r', root),
      live = $('#fa-live', root),
      S = { d: 'two', k: 2, rot: true },
      NM = ['Algebra', 'Geometry', 'Statistics', 'Reading', 'Writing', 'Vocabulary'],
      p = 6;
    let Sm = null,
      L = null,
      Ps = null,
      it = 0,
      ll = NaN,
      hist = [],
      fin = false,
      seed = 1,
      msg = '',
      cu = 0,
      pb,
      chk = () => false,
      pc1 = 0;
    const gen = () => {
      const r = rng(seed * 19 + S.d.length),
        rn = randnFrom(r),
        n = 300,
        X = [];
      for (let i = 0; i < n; i++) {
        const m = rn(),
          v = S.d === 'one' ? m : 0.35 * m + Math.sqrt(1 - 0.35 * 0.35) * rn(),
          ld = S.d === 'one' ? [0.9, 0.8, 0.85, 0.8, 0.85, 0.75] : [0.9, 0.85, 0.8, 0.85, 0.9, 0.8],
          ns = [0.5, 0.6, S.d === 'noisy' ? 2.2 : 0.6, 0.55, 0.5, 0.65];
        X.push(ld.map((l, j) => l * (j < 3 ? m : v) + ns[j] * rn()));
      }
      const mu = Array.from({ length: p }, (_, j) => X.reduce((s, x) => s + x[j], 0) / n);
      Sm = Array.from({ length: p }, (_, a) =>
        Array.from(
          { length: p },
          (_, b) => X.reduce((s, x) => s + (x[a] - mu[a]) * (x[b] - mu[b]), 0) / n,
        ),
      );
      const e = laTop(Sm, 1).vecs[0];
      pc1 = e.reduce((m, v, i) => (Math.abs(v) > Math.abs(e[m]) ? i : m), 0);
    };
    const sig = () =>
      L.map((r, a) =>
        L.map((q, b) => r.reduce((s, v, j) => s + v * q[j], 0) + (a === b ? Ps[a] : 0)),
      );
    const loglik = () => {
      const Sg = sig(),
        C = laChol(Sg),
        iv = laInv(Sg);
      let ld = 0,
        tr = 0;
      for (let i = 0; i < p; i++) {
        ld += 2 * Math.log(C[i][i]);
        for (let j = 0; j < p; j++) tr += iv[i][j] * Sm[j][i];
      }
      return -0.5 * (p * Math.log(2 * Math.PI) + ld + tr);
    };
    const reset = () => {
      const r = randnFrom(rng(seed * 5 + S.k));
      L = Array.from({ length: p }, (_, a) =>
        Array.from({ length: S.k }, () => 0.4 * Math.sqrt(Sm[a][a]) * r()),
      );
      Ps = Sm.map((r, a) => r[a]);
      it = 0;
      fin = false;
      ll = loglik();
      hist = [ll];
      cu = 0;
      msg =
        'A random first guess for the loadings; every test’s own noise starts at its full variance. Press One step or Run.';
    };
    const step = () => {
      if (fin) return;
      const k = S.k,
        iv = laInv(sig()),
        B = laMul(laT(L), iv),
        BL = laMul(B, L),
        BSB = laMul(laMul(B, Sm), laT(B)),
        C = BL.map((r, a) => r.map((v, b) => (a === b ? 1 : 0) - v + BSB[a][b])),
        SB = laMul(Sm, laT(B));
      L = laMul(SB, laInv(C));
      const LBS = laMul(laMul(L, B), Sm);
      Ps = Sm.map((r, a) => Math.max(1e-4, r[a] - LBS[a][a]));
      it++;
      const nl = loglik(),
        gain = nl - ll;
      ll = nl;
      hist.push(ll);
      cu = it % 2 ? 1 : 2;
      msg =
        'Step ' +
        it +
        ': log-likelihood per student ' +
        ll.toFixed(4) +
        ' (+' +
        (gain < 1e-3 ? gain.toExponential(1) : f3(gain)) +
        ').';
      if (gain < 1e-7 || it >= 1500) {
        fin = true;
        cu = 5;
        const R = view(),
          top = (m) => {
            const col = R.map((r) => Math.abs(r[m]));
            return NM.filter((_, i) => col[i] > 0.5 * Math.max(...col)).join(', ');
          };
        msg =
          'Stopped after ' +
          it +
          ' steps: the log-likelihood has stopped rising. ' +
          (S.k === 1
            ? 'The single factor drives ' + top(0) + '.'
            : 'Factor 1 drives ' +
              top(0) +
              '; factor 2 drives ' +
              top(1) +
              (S.k === 3 ? '; factor 3 drives ' + top(2) : '') +
              '.') +
          ' Largest mismatch between the grids: ' +
          f3(gap()) +
          '.';
      }
    };
    const gap = () => {
      const Sg = sig();
      let m = 0;
      for (let a = 0; a < p; a++)
        for (let b = 0; b < p; b++) m = Math.max(m, Math.abs(Sg[a][b] - Sm[a][b]));
      return m;
    };
    const view = () => {
      let R = L.map((r) => r.slice());
      const k = S.k;
      if (S.rot && k > 1) {
        for (let sweep = 0; sweep < 25; sweep++)
          for (let a = 0; a < k; a++)
            for (let b = a + 1; b < k; b++) {
              let A = 0,
                B = 0,
                C = 0,
                Dd = 0;
              for (let i = 0; i < p; i++) {
                const x = R[i][a],
                  y = R[i][b],
                  uu = x * x - y * y,
                  vv = 2 * x * y;
                A += uu;
                B += vv;
                C += uu * uu - vv * vv;
                Dd += 2 * uu * vv;
              }
              const ph = 0.25 * Math.atan2(Dd - (2 * A * B) / p, C - (A * A - B * B) / p),
                cs = Math.cos(ph),
                sn = Math.sin(ph);
              for (let i = 0; i < p; i++) {
                const x = R[i][a],
                  y = R[i][b];
                R[i][a] = cs * x + sn * y;
                R[i][b] = -sn * x + cs * y;
              }
            }
      }
      const ord = Array.from({ length: k }, (_, m) => m).sort(
        (a, b) => R.reduce((s, r) => s + r[b] * r[b], 0) - R.reduce((s, r) => s + r[a] * r[a], 0),
      );
      R = R.map((r) => ord.map((m) => r[m]));
      for (let m = 0; m < k; m++) {
        if (R.reduce((s, r) => s + r[m], 0) < 0) R.forEach((r) => (r[m] = -r[m]));
      }
      return R;
    };
    const grid = (g, t, M, x0, y0, sz, title, mx) => {
      const cs = sz / p;
      g.font = MONO;
      g.fillStyle = t.muted;
      g.textAlign = 'center';
      g.fillText(title, x0 + sz / 2, y0 - 8);
      for (let a = 0; a < p; a++)
        for (let b = 0; b < p; b++) {
          const v = M[a][b];
          g.fillStyle =
            v >= 0
              ? mix(t.bg, t.accent, clamp(v / mx, 0, 1) * 0.9)
              : mix(t.bg, t.cut, clamp(-v / mx, 0, 1) * 0.9);
          g.fillRect(x0 + b * cs + 1, y0 + a * cs + 1, cs - 2, cs - 2);
          if (cs > 30) {
            g.fillStyle = Math.abs(v) / mx > 0.55 ? t.bg : t.ink;
            g.font = '10px "JetBrains Mono",monospace';
            g.fillText(v.toFixed(2), x0 + b * cs + cs / 2, y0 + a * cs + cs / 2 + 3);
          }
        }
    };
    const draw = () => {
      const t = T(),
        Sg = sig(),
        R = view();
      {
        const { g, w, h } = fit(c),
          lm = 64,
          sz = Math.min((w - lm - 24) / 2, h - 52),
          y0 = 30,
          mx =
            Math.max(
              ...Sm.map((r, a) => Math.max(...r.map((v, b) => (a === b ? 0 : Math.abs(v))))),
            ) * 1.15;
        grid(g, t, Sm, lm, y0, sz, 'data', mx);
        grid(g, t, Sg, lm + sz + 18, y0, sz, 'model', mx);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        NM.forEach((nm, a) => g.fillText(nm.slice(0, 7), lm - 5, y0 + ((a + 0.5) * sz) / p + 4));
        g.textAlign = 'left';
        g.fillText('colour is capped at the largest off-diagonal value', lm, y0 + sz + 16);
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.6,
          m = { l: 74, r: 10, t: 22, b: 8 },
          rh = (h1 - m.t - m.b) / p,
          cols = [t.accent, t.gold, t.ok],
          mx = Math.max(1, ...R.flat().map(Math.abs), ...Ps.map(Math.sqrt)),
          x0 = m.l + (w - m.l - m.r) * 0.38,
          sc = ((w - m.l - m.r) * 0.6) / mx;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          'loadings (coloured, one per factor) and own noise (grey, as a standard deviation)',
          8,
          13,
        );
        g.strokeStyle = t.line;
        g.beginPath();
        g.moveTo(x0, m.t);
        g.lineTo(x0, h1 - m.b);
        g.stroke();
        for (let a = 0; a < p; a++) {
          const y = m.t + a * rh,
            bh = (rh - 6) / (S.k + 1);
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText(NM[a], m.l - 6, y + rh / 2 + 4);
          for (let q = 0; q < S.k; q++) {
            g.fillStyle = cols[q];
            const v = R[a][q] * sc;
            g.fillRect(v >= 0 ? x0 : x0 + v, y + 3 + q * bh, Math.abs(v), bh - 1.5);
          }
          g.fillStyle = mix(t.muted, t.muted, 0, 0.55);
          g.fillRect(x0, y + 3 + S.k * bh, Math.sqrt(Ps[a]) * sc, bh - 1.5);
        }
        const lo = Math.min(...hist.slice(Math.min(3, hist.length - 1))),
          hi = Math.max(...hist),
          B = uLine(g, t, 0, h1, w, h - h1, hist, {
            title: 'log-likelihood per student',
            n: Math.max(30, hist.length),
            y1: lo - 0.02 * (hi - lo + 1e-9),
            ym: hi + 0.05 * (hi - lo + 1e-9),
          });
        g.save();
        g.beginPath();
        g.rect(0, h1 + 16, w, h - h1 - 34);
        g.clip();
        g.strokeStyle = t.accent;
        g.lineWidth = 2;
        g.beginPath();
        hist.forEach((v, i) => (i ? g.lineTo(B.X(i), B.Y(v)) : g.moveTo(B.X(i), B.Y(v))));
        g.stroke();
        g.restore();
      }
      const f1 = R.reduce((m, r, i) => (Math.abs(r[0]) > Math.abs(R[m][0]) ? i : m), 0);
      live.textContent = msg;
      uCur(root, 'fa', cu);
      setR(root, 'it', it + (fin ? ' (stopped)' : ''));
      setR(root, 'll', ll.toFixed(4));
      setR(root, 'gap', f3(gap()));
      setR(root, 'pc', NM[pc1]);
      setR(root, 'f1', it ? NM[f1] : '–');
    };
    gen();
    reset();
    if (RM) {
      while (!fin) step();
    }
    const anim = Anim(() => {
      if (fin) {
        pb.stop();
        return;
      }
      const k = it < 12 ? 1 : it < 60 ? 3 : 8;
      if (it < 12 && ++fr % 8) return;
      for (let i = 0; i < k && !fin && chk.remaining() > 0; i++) step();
      draw();
      chk();
    });
    let fr = 0;
    pb = playBtn($('#fa-play', root), anim);
    chk = runCtl(pb, root, 'fa', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'steps',
    });
    const go = () => {
      pb.stop();
      reset();
      draw();
      pb.start();
    };
    $('#fa-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#fa-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    $('#fa-new', root).onclick = () => {
      seed++;
      gen();
      go();
    };
    const kB = bind(
      root,
      'fa-k',
      (v) => {
        S.k = v;
        go();
      },
      (v) => String(+v),
    );
    segBind(root, 'fa-d', S.d, (v) => {
      S.d = v;
      S.k = v === 'one' ? 1 : 2;
      kB.set(S.k);
      gen();
      go();
    });
    $('#fa-rot', root).onchange = (e) => {
      S.rot = e.target.checked;
      draw();
    };
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        pb.start();
      },
    };
  },
};
