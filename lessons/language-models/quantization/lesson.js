/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  eqShow,
  f2,
  fit,
  letterData,
  mix,
  playBtn,
  randnFrom,
  rng,
  setR,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { g2Bench } from '../../generative/shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#qz-c', root),
      cr = $('#qz-r', root),
      ct = $('#qz-t', root),
      live = $('#qz-live', root),
      eq = $('#qz-eq', root),
      BITS = [1, 2, 3, 4, 5, 6, 8, 16, 32],
      S = { bi: 8, clip: 'full', let: 0 },
      CH = ['A', 'E', 'H', 'O', 'T', 'X'];
    let net,
      Wf,
      ACC = { full: [], clip: [] },
      acc0 = NaN,
      Wq = null,
      info = null,
      test,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      run = 1,
      pic = null,
      wi = 0,
      sweep = false;
    const quant = (w, b, clip) => {
      const n = w.length,
        o = new Float64Array(n);
      if (b >= 32)
        return {
          o: Float64Array.from(w),
          d: 0,
          m: Math.max(...Array.from(w).map(Math.abs)),
          Q: Infinity,
          lev: [],
        };
      const ab = Array.from(w).map(Math.abs);
      if (b === 1) {
        const mu = ab.reduce((s, v) => s + v, 0) / n;
        for (let i = 0; i < n; i++) o[i] = w[i] < 0 ? -mu : mu;
        return { o, d: 2 * mu, m: mu, Q: 1, lev: [-mu, mu], one: true };
      }
      const m = clip
          ? ab.slice().sort((x, y) => x - y)[Math.floor(0.99 * (n - 1))]
          : Math.max(...ab),
        Q = Math.pow(2, b - 1) - 1,
        d = m / Q;
      for (let i = 0; i < n; i++) o[i] = clamp(Math.round(w[i] / d), -Q, Q) * d;
      const lev = [];
      if (Q <= 64) for (let q = -Q; q <= Q; q++) lev.push(q * d);
      return { o, d, m, Q, lev };
    };
    const setW = (a, b2) => {
      net.ps[0].v.set(a);
      net.ps[1].v.set(b2);
    };
    const probs = (X, n) => {
      const o = net.fwd(AD.input(X, [n, 35]));
      AD.clear();
      const P = new Float64Array(n * 6);
      for (let i = 0; i < n; i++) {
        let mx = -1e30;
        for (let j = 0; j < 6; j++) mx = Math.max(mx, o.v[i * 6 + j]);
        let z = 0;
        for (let j = 0; j < 6; j++) {
          P[i * 6 + j] = Math.exp(o.v[i * 6 + j] - mx);
          z += P[i * 6 + j];
        }
        for (let j = 0; j < 6; j++) P[i * 6 + j] /= z;
      }
      return P;
    };
    const accOf = () => {
      const n = test.lab.length,
        P = probs(Float64Array.from(test.X.flat()), n);
      let ok = 0;
      for (let i = 0; i < n; i++) {
        let b = 0;
        for (let j = 1; j < 6; j++) if (P[i * 6 + j] > P[i * 6 + b]) b = j;
        if (b === test.lab[i]) ok++;
      }
      return ok / n;
    };
    const build = () => {
      net = AD.mlp([35, 24, 6], randnFrom(rng(run * 17 + 5)), 'relu');
      const tr = letterData(run, 0.12),
        t1 = letterData(run + 50, 0.12),
        t2 = letterData(run + 90, 0.12);
      test = { X: t1.X.concat(t2.X), lab: t1.lab.concat(t2.lab) };
      for (let ts = 1; ts <= 500; ts++) {
        const B = 32,
          X = new Float64Array(B * 35),
          tg = new Int32Array(B);
        for (let i = 0; i < B; i++) {
          const j = Math.floor(Math.random() * 150);
          X.set(tr.X[j], i * 35);
          tg[i] = tr.lab[j];
        }
        const l = AD.ce(net.fwd(AD.input(X, [B, 35])), tg, 0);
        AD.backward(l);
        AD.adam(net.ps, 0.01, ts, 1);
      }
      Wf = [Float64Array.from(net.ps[0].v), Float64Array.from(net.ps[1].v)];
      acc0 = accOf();
      for (const m of ['full', 'clip'])
        ACC[m] = BITS.map((b) => {
          setW(quant(Wf[0], b, m === 'clip').o, quant(Wf[1], b, m === 'clip').o);
          return accOf();
        });
      wi = Math.floor(Math.random() * 840);
      newPic();
      apply();
      cu = 0;
      msg =
        'The network has been trained with ordinary 32-bit weights and reads ' +
        pct(acc0) +
        ' of 300 test pictures correctly. Press Run to round its weights more and more coarsely, or move the slider.';
    };
    const apply = () => {
      const b = BITS[S.bi],
        q0 = quant(Wf[0], b, S.clip === 'clip'),
        q1 = quant(Wf[1], b, S.clip === 'clip');
      Wq = [q0.o, q1.o];
      info = q0;
      let e = 0;
      for (let i = 0; i < 840; i++) e += Math.abs(Wf[0][i] - q0.o[i]);
      info.err = e / 840 / (2 * Math.max(...Array.from(Wf[0]).map(Math.abs)));
      const w = Wf[0][wi],
        L = [];
      if (b >= 32)
        L.push(
          'weight number ' + (wi + 1) + ' of the first layer = ' + w.toFixed(5),
          '32 bits: stored as it is. Nothing is rounded.',
        );
      else if (b === 1)
        L.push(
          'weight number ' + (wi + 1) + ' of the first layer = ' + w.toFixed(5),
          '1 bit: only the sign is kept. The sign is ' +
            (w < 0 ? 'minus' : 'plus') +
            ', stored as <b>' +
            (w < 0 ? '0' : '1') +
            '</b>',
          'every weight gets the same size, the average size in the layer = ' + q0.m.toFixed(5),
          'read back as <b>' + q0.o[wi].toFixed(5) + '</b>    error ' + (q0.o[wi] - w).toFixed(5),
        );
      else {
        const qi = clamp(Math.round(w / q0.d), -q0.Q, q0.Q),
          bin = ((qi + q0.Q) >>> 0).toString(2).padStart(b, '0');
        L.push(
          'weight number ' + (wi + 1) + ' of the first layer = ' + w.toFixed(5),
          'largest size allowed m = ' +
            q0.m.toFixed(5) +
            '    whole numbers from −' +
            q0.Q +
            ' to ' +
            q0.Q +
            '    step Δ = m ÷ ' +
            q0.Q +
            ' = ' +
            q0.d.toFixed(5),
          'q = round(' +
            w.toFixed(5) +
            ' ÷ ' +
            q0.d.toFixed(5) +
            ') = round(' +
            (w / q0.d).toFixed(2) +
            ') = <b>' +
            qi +
            '</b>' +
            (Math.abs(Math.round(w / q0.d)) > q0.Q ? '  (clipped to the largest allowed)' : ''),
          'stored in ' +
            b +
            ' bits as <b>' +
            bin +
            '</b>' +
            (b > 12 ? '' : '  (position ' + (qi + q0.Q) + ' on the list of allowed values)'),
          'read back as ' +
            qi +
            ' × ' +
            q0.d.toFixed(5) +
            ' = <b>' +
            (qi * q0.d).toFixed(5) +
            '</b>    error ' +
            (qi * q0.d - w).toFixed(5),
        );
      }
      eqShow(eq, L, sweep ? 0 : 260);
    };
    const newPic = () => {
      const d = letterData(Math.floor(Math.random() * 1e6), 0.12);
      pic = d.X[S.let * 25 + Math.floor(Math.random() * 25)];
    };
    const say = () => {
      const b = BITS[S.bi],
        a = ACC[S.clip][S.bi];
      cu = b >= 32 ? 0 : 4;
      msg =
        b >= 32
          ? '32 bits per weight: the network as it was trained. The 984 weights take ' +
            984 * 4 +
            ' bytes and the accuracy is ' +
            pct(acc0) +
            '.'
          : 'At ' +
            b +
            ' bit' +
            (b > 1 ? 's' : '') +
            ' each weight may take one of ' +
            (b === 1
              ? 2
              : b > 20
                ? 'about ' + Math.pow(2, b).toExponential(1)
                : Math.pow(2, b) - 1) +
            ' values' +
            (b === 1 ? ': plus or minus one fixed size' : '') +
            '. Every weight was rounded to the nearest one. The weights now take ' +
            Math.ceil((984 * b) / 8) +
            ' bytes, ' +
            f2(32 / b) +
            ' times less than before, and the accuracy is ' +
            pct(a) +
            ' (it was ' +
            pct(acc0) +
            ').';
    };
    const draw = () => {
      const t = T(),
        b = BITS[S.bi];
      {
        const { g, w, h } = fit(c),
          h1 = h * 0.44,
          m = { l: 14, r: 14, t: 22, b: 18 },
          W = Wf[0],
          mx = Math.max(...Array.from(W).map(Math.abs)) * 1.04,
          NB = 48,
          cnt = new Array(NB).fill(0);
        for (const v of W) cnt[clamp(Math.floor(((v + mx) / (2 * mx)) * NB), 0, NB - 1)]++;
        const cm = Math.max(...cnt),
          X = (v) => m.l + ((v + mx) / (2 * mx)) * (w - m.l - m.r);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          'the 840 weights of the first layer, and the values they may be rounded to',
          8,
          13,
        );
        cnt.forEach((n, i) => {
          const bh = (n / cm) * (h1 - m.t - m.b);
          g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
          g.fillRect(
            m.l + (i * (w - m.l - m.r)) / NB + 0.5,
            h1 - m.b - bh,
            (w - m.l - m.r) / NB - 1,
            bh,
          );
        });
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h1 - m.b);
        g.lineTo(w - m.r, h1 - m.b);
        g.stroke();
        if (info.lev.length) {
          g.strokeStyle = t.gold;
          g.lineWidth = info.lev.length > 40 ? 0.7 : 1.6;
          for (const v of info.lev) {
            g.beginPath();
            g.moveTo(X(v), m.t);
            g.lineTo(X(v), h1 - m.b + 5);
            g.stroke();
          }
        } else {
          g.fillStyle = t.cut;
          g.textAlign = 'center';
          g.fillText(
            b >= 32
              ? '32 bits: every value is allowed'
              : 'too many allowed values to draw: ' + (Math.pow(2, b) - 1),
            w / 2,
            m.t + 12,
          );
        }
        g.fillStyle = t.ink;
        g.beginPath();
        g.arc(X(Wf[0][wi]), h1 - m.b + 9, 3.5, 0, 7);
        g.fill();
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('0', X(0) - 3, h1 - 2);
        g.textAlign = 'right';
        g.fillText('dot: the weight followed below', w - m.r, h1 - 2);
        const cols = 12,
          s = Math.floor(
            Math.min((w - 20) / (cols * 5 + (cols - 1) * 0.8), (h - h1 - 30) / (2 * 7 + 1)),
          );
        const ox = (w - (cols * 5 * s + (cols - 1) * 0.8 * s)) / 2,
          oy = h1 + 20,
          q = Wq[0],
          qm = Math.max(1e-9, ...Array.from(q).map(Math.abs));
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('incoming weights of each hidden neuron, after rounding', 8, h1 + 13);
        for (let u = 0; u < 24; u++) {
          const x0 = ox + (u % cols) * 5.8 * s,
            y0 = oy + Math.floor(u / cols) * 8 * s;
          for (let i = 0; i < 35; i++) {
            const v = q[i * 24 + u] / qm,
              col = v > 0 ? t.accent : t.cut;
            g.fillStyle = mix(col, col, 0, Math.min(1, Math.abs(v) * 1.5));
            g.fillRect(x0 + (i % 5) * s, y0 + Math.floor(i / 5) * s, s, s);
          }
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.strokeRect(x0 - 0.5, y0 - 0.5, 5 * s + 1, 7 * s + 1);
        }
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.46,
          L = g2Bench(
            g,
            t,
            0,
            0,
            w,
            h1,
            BITS,
            [
              { ys: ACC.clip, col: t.cut, lab: 'largest 1% clipped' },
              { ys: ACC.full, col: t.accent, lab: 'full range' },
            ],
            S.bi,
            'accuracy on 300 test pictures, by bits per weight',
          );
        const m = { l: 44, r: 14, t: 26, b: 20 },
          y0 = h1,
          H = h - h1,
          mx = Math.max(...Array.from(Wf[0]).map(Math.abs)) * 1.04,
          X = (v) => m.l + ((v + mx) / (2 * mx)) * (w - m.l - m.r),
          Y = (v) => y0 + H - m.b - ((v + mx) / (2 * mx)) * (H - m.t - m.b);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('each weight: original value (across) and rounded value (up)', 8, y0 + 13);
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X(-mx), Y(-mx));
        g.lineTo(X(mx), Y(mx));
        g.stroke();
        g.fillStyle = mix(t.accent, t.accent, 0, 0.6);
        for (let i = 0; i < 840; i += 2) {
          g.beginPath();
          g.arc(X(Wf[0][i]), Y(Wq[0][i]), 1.6, 0, 7);
          g.fill();
        }
        g.fillStyle = t.gold;
        g.beginPath();
        g.arc(X(Wf[0][wi]), Y(Wq[0][wi]), 5, 0, 7);
        g.fill();
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('thin line: no rounding', w - m.r, y0 + H - 5);
      }
      {
        const { g, w, h } = fit(ct),
          s = Math.floor(Math.min(h - 30, 180) / 7.4),
          x0 = 18,
          y0 = (h - 7 * s) / 2;
        for (let i = 0; i < 35; i++) {
          g.fillStyle = pic[i] > 0.5 ? t.ink : t.bg;
          g.fillRect(x0 + (i % 5) * s, y0 + Math.floor(i / 5) * s, s - 1, s - 1);
          if (pic[i] <= 0.5) {
            g.strokeStyle = t.line;
            g.lineWidth = 1;
            g.strokeRect(x0 + (i % 5) * s + 0.5, y0 + Math.floor(i / 5) * s + 0.5, s - 2, s - 2);
          }
        }
        setW(Wf[0], Wf[1]);
        const P0 = probs(Float64Array.from(pic), 1);
        setW(Wq[0], Wq[1]);
        const P1 = probs(Float64Array.from(pic), 1);
        const bx = x0 + 5 * s + 46,
          bw = (w - bx - 16) / 6,
          H = h - 56;
        g.font = MONO;
        for (let j = 0; j < 6; j++) {
          const x = bx + j * bw;
          g.fillStyle = t.accent;
          g.fillRect(x + 4, 20 + H - P0[j] * H, bw * 0.36, P0[j] * H);
          g.fillStyle = t.gold;
          g.fillRect(x + 4 + bw * 0.4, 20 + H - P1[j] * H, bw * 0.36, P1[j] * H);
          g.fillStyle = j === S.let ? t.cut : t.muted;
          g.textAlign = 'center';
          g.fillText(CH[j] + (j === S.let ? ' (true)' : ''), x + bw * 0.42, h - 18);
        }
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(bx, 20 + H);
        g.lineTo(w - 12, 20 + H);
        g.stroke();
        const a0 = P0.indexOf(Math.max(...P0)),
          a1 = P1.indexOf(Math.max(...P1));
        $('#qz-tpo', root).textContent =
          'This picture is a noisy ' +
          CH[S.let] +
          '. The full-precision network says ' +
          CH[a0] +
          ' (' +
          pct(P0[a0]) +
          ' sure). At ' +
          b +
          ' bit' +
          (b > 1 ? 's' : '') +
          ' the rounded network says ' +
          CH[a1] +
          ' (' +
          pct(P1[a1]) +
          ' sure).' +
          (a0 === S.let && a1 !== S.let
            ? ' Rounding changed a right answer into a wrong one.'
            : a0 !== S.let && a1 === S.let
              ? ' Here rounding happened to fix a wrong answer: luck, not a rule.'
              : '');
      }
      live.textContent = msg;
      uCur(root, 'qz', cu);
      setR(root, 'b', String(b));
      setR(
        root,
        'lv',
        b >= 32 ? 'about 4 billion' : b === 1 ? '2' : b > 20 ? '–' : String(Math.pow(2, b) - 1),
      );
      setR(
        root,
        'sz',
        Math.ceil((984 * b) / 8) + ' bytes' + (b < 32 ? ' (' + f2(32 / b) + '× smaller)' : ''),
      );
      setR(root, 'acc', pct(ACC[S.clip][S.bi]));
      setR(root, 'acc0', pct(acc0));
      setR(root, 'err', b >= 32 ? '0' : pct(info.err));
    };
    $('#qz-let', root).innerHTML = CH.map(
      (ch, i) =>
        '<button class="btn" type="button" data-v="' + i + '">Test a Noisy ' + ch + '</button>',
    ).join('');
    build();
    const sl = bind(
      root,
      'qz-b',
      (v) => {
        S.bi = +v;
        if (!pb) return;
        pb.stop();
        apply();
        say();
        draw();
      },
      (v) => String(BITS[+v]),
    );
    const anim = Anim(() => {
      fr++;
      if (fr % 55) return;
      if (S.bi <= 0) {
        pb.stop();
        sweep = false;
        return;
      }
      S.bi--;
      sl.set(S.bi);
      apply();
      say();
      draw();
      if (S.bi <= 0) {
        pb.stop();
        sweep = false;
      }
    });
    pb = playBtn($('#qz-play', root), anim);
    const st0 = pb.start;
    pb.start = () => {
      sweep = true;
      if (S.bi <= 0) {
        S.bi = 8;
        sl.set(8);
        apply();
        say();
        draw();
      }
      st0();
    };
    $('#qz-step', root).onclick = () => {
      pb.stop();
      sweep = false;
      S.bi = S.bi <= 0 ? 8 : S.bi - 1;
      sl.set(S.bi);
      apply();
      say();
      draw();
    };
    $('#qz-reset', root).onclick = () => {
      pb.stop();
      sweep = false;
      run++;
      S.bi = 8;
      sl.set(8);
      build();
      draw();
    };
    segBind(root, 'qz-clip', S.clip, (v) => {
      S.clip = v;
      apply();
      say();
      draw();
    });
    segBind(root, 'qz-let', '0', (v) => {
      S.let = +v;
      newPic();
      draw();
    });
    $('#qz-noise', root).onclick = () => {
      newPic();
      draw();
    };
    $('#qz-w', root).onclick = () => {
      wi = Math.floor(Math.random() * 840);
      apply();
      draw();
    };
    return {
      draw,
      stop: () => {
        pb.stop();
        sweep = false;
      },
      auto: () => {
        S.bi = 8;
        sl.set(8);
        apply();
        say();
        draw();
        pb.start();
      },
    };
  },
};
