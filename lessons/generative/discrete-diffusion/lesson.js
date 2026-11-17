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
  letterData,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { g2Bench } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#dd-c', root),
      cr = $('#dd-r', root),
      live = $('#dd-live', root),
      S = { kind: 'pic', ni: 5, ord: 'conf' },
      G = 8,
      KD = {
        pic: {
          Ln: 35,
          V: 2,
          NS: [1, 2, 3, 5, 7, 12, 18, 35],
          def: 5,
          what: 'a real letter',
          unit: 'squares',
        },
        word: { Ln: 5, V: 26, NS: [1, 2, 3, 4, 5], def: 4, what: 'a real word', unit: 'letters' },
      },
      MAXS = 1500,
      WORDS =
        'apple bread chair dance eagle flame grape house juice knife lemon mouse night ocean piano queen river snake table uncle voice water zebra beach cloud dream earth field green heart light money music paper plant radio sheep smile stone sugar tiger train truck watch world young brain bring candy clean clock drink floor fruit glass happy horse lunch magic party pizza robot shirt storm'.split(
          ' ',
        );
    let K,
      net,
      ts = 0,
      LL = [],
      sm = [],
      phase = 'train',
      seqs = [],
      clean = [],
      CHN = [],
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      X = null,
      PR = null,
      si = 0,
      rev = 0,
      done = false,
      Q = [[], []],
      bi = 0,
      okV = NaN,
      just = [],
      ex = null,
      exO = null,
      res = [];
    const W = () => K.Ln * (K.V + 1);
    const enc = (rows) => {
      const B = rows.length,
        w = W(),
        A = new Float64Array(B * w);
      rows.forEach((r, b) => {
        for (let i = 0; i < K.Ln; i++) A[b * w + i * (K.V + 1) + (r[i] < 0 ? K.V : r[i])] = 1;
      });
      return A;
    };
    const probs = (rows) => {
      const B = rows.length,
        o = net.fwd(AD.input(enc(rows), [B, W()]));
      AD.clear();
      const V = K.V,
        P = new Float64Array(o.v.length);
      for (let i = 0; i < B * K.Ln; i++) {
        let mx = -1e30;
        for (let j = 0; j < V; j++) mx = Math.max(mx, o.v[i * V + j]);
        let z = 0;
        for (let j = 0; j < V; j++) {
          P[i * V + j] = Math.exp(o.v[i * V + j] - mx);
          z += P[i * V + j];
        }
        for (let j = 0; j < V; j++) P[i * V + j] /= z;
      }
      return P;
    };
    const match = (r) => {
      if (S.kind === 'word')
        return WORDS.indexOf(r.map((v) => String.fromCharCode(97 + v)).join(''));
      let b = -1,
        bd = 3;
      clean.forEach((g, k) => {
        let d = 0;
        for (let i = 0; i < 35; i++) if (g[i] !== r[i]) d++;
        if (d < bd) {
          bd = d;
          b = k;
        }
      });
      return b;
    };
    const build = () => {
      K = KD[S.kind];
      if (S.kind === 'pic') {
        const d = letterData(run, 0.02);
        seqs = d.X.map((x) => x.map((v) => (v > 0.5 ? 1 : 0)));
        CHN = d.CH;
        const q = letterData(1, 0);
        clean = d.CH.map((_, k) => q.X[k * 25].map((v) => (v > 0.5 ? 1 : 0)));
      } else {
        seqs = WORDS.map((w) => [...w].map((ch) => ch.charCodeAt(0) - 97));
        clean = [];
      }
      net = AD.mlp([W(), 96, 96, K.Ln * K.V], randnFrom(rng(run * 9 + 2)), 'relu');
      ts = 0;
      LL = [];
      sm = [];
      phase = 'train';
      X = null;
      PR = null;
      done = false;
      cu = 0;
      Q = [K.NS.map(() => NaN), K.NS.map(() => NaN)];
      bi = 0;
      okV = NaN;
      just = [];
      res = [];
      si = 0;
      rev = 0;
      ex = seqs[Math.floor(Math.random() * seqs.length)];
      exO = Array.from({ length: K.Ln }, (_, i) => i).sort(() => Math.random() - 0.5);
      msg =
        'A network with random weights. It has ' +
        K.Ln +
        ' positions to fill, each with one of ' +
        K.V +
        ' tokens. Press Run to train it.';
    };
    const train = (k) => {
      const B = 32,
        Ln = K.Ln;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const rows = [],
          tg = new Int32Array(B * Ln),
          mk = new Uint8Array(B * Ln);
        for (let b = 0; b < B; b++) {
          const s = seqs[Math.floor(Math.random() * seqs.length)],
            u = Math.random(),
            r = s.slice();
          let any = 0;
          for (let i = 0; i < Ln; i++) {
            tg[b * Ln + i] = s[i];
            if (Math.random() < u) {
              r[i] = -1;
              mk[b * Ln + i] = 1;
              any = 1;
            }
          }
          if (!any) {
            const i = Math.floor(Math.random() * Ln);
            r[i] = -1;
            mk[b * Ln + i] = 1;
          }
          rows.push(r);
        }
        const l = AD.mce(net.fwd(AD.input(enc(rows), [B, W()])), K.V, tg, mk);
        LL.push(l.v[0]);
        AD.backward(l);
        AD.adam(net.ps, 0.003, ++ts, 1);
        if (ts % 20 === 0) {
          let s = 0;
          for (let j = ts - 20; j < ts; j++) s += LL[j];
          sm.push(s / 20);
        }
      }
      cu = Math.floor(ts / 100) % 2;
      msg =
        'Training step ' +
        ts +
        ' of ' +
        MAXS +
        ': 32 real examples each had a random share of their ' +
        K.unit +
        ' hidden, and the network guessed what was under every blank. Average error over the last 20 steps: ' +
        f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
        ' (guessing blindly would score ' +
        f3(Math.log(K.V)) +
        ').';
      if (ts >= MAXS) {
        phase = 'bench';
        bi = 0;
        cu = 2;
        msg =
          'Training is finished. Next the network generates with each number of reveal steps, in both orders.';
      }
    };
    const reveal = (rows, P, cnt, ord) => {
      const out = [],
        V = K.V,
        Ln = K.Ln;
      rows.forEach((r, b) => {
        const hid = [];
        for (let i = 0; i < Ln; i++)
          if (r[i] < 0) {
            let m = 0;
            for (let j = 0; j < V; j++) m = Math.max(m, P[(b * Ln + i) * V + j]);
            hid.push([i, ord === 'conf' ? m + 1e-6 * Math.random() : Math.random()]);
          }
        hid.sort((a, b2) => b2[1] - a[1]);
        for (let q = 0; q < cnt && q < hid.length; q++) {
          const i = hid[q][0];
          let u = Math.random(),
            tk = 0;
          for (; tk < V - 1; tk++) {
            u -= P[(b * Ln + i) * V + tk];
            if (u <= 0) break;
          }
          r[i] = tk;
          out.push(b * Ln + i);
        }
      });
      return out;
    };
    const sampleAll = (N, ord, B) => {
      const rows = Array.from({ length: B }, () => new Array(K.Ln).fill(-1));
      let rv = 0;
      for (let j = 1; j <= N; j++) {
        const tgt = Math.round((K.Ln * j) / N);
        reveal(rows, probs(rows), tgt - rv, ord);
        rv = tgt;
      }
      let ok = 0;
      for (const r of rows) if (match(r) >= 0) ok++;
      return ok / B;
    };
    const startGen = () => {
      X = Array.from({ length: G }, () => new Array(K.Ln).fill(-1));
      PR = probs(X);
      si = 0;
      rev = 0;
      done = false;
      just = [];
      okV = NaN;
      res = [];
    };
    const bench = () => {
      const j = bi >> 1,
        e = bi & 1,
        N = K.NS[j];
      Q[e][j] = sampleAll(N, e ? 'conf' : 'rand', 64);
      bi++;
      cu = 2;
      msg =
        'Measuring: ' +
        N +
        ' reveal step' +
        (N > 1 ? 's' : '') +
        ', ' +
        (e ? 'most confident first' : 'random order') +
        ': ' +
        pct(Q[e][j]) +
        ' of 64 samples came out as ' +
        K.what +
        '.';
      if (bi >= 2 * K.NS.length) {
        phase = 'gen';
        startGen();
        cu = 3;
        msg =
          'All settings are measured: see the chart at the bottom right. Now watch one run. Eight samples, everything hidden. The tint shows what the network believes before anything is revealed, and it is the same for all eight.';
      }
    };
    const gstep = () => {
      if (done || !X) return;
      const N = K.NS[S.ni],
        tgt = Math.round((K.Ln * (si + 1)) / N),
        cnt = tgt - rev;
      just = reveal(X, PR, cnt, S.ord);
      rev = tgt;
      si++;
      cu = 4;
      msg =
        'Reveal step ' +
        si +
        ' of ' +
        N +
        ': the network was run once on each sample. ' +
        cnt +
        ' hidden ' +
        (cnt === 1 ? K.unit.slice(0, -1) + ' was' : K.unit + ' were') +
        ' chosen in each (' +
        (S.ord === 'conf' ? 'where the network was most sure' : 'at random') +
        ') and filled by drawing from its beliefs' +
        (cnt > 1 ? ', each drawn without knowing the others.' : '.') +
        ' ' +
        (K.Ln - rev) +
        ' remain hidden.';
      if (si >= N) {
        done = true;
        cu = 5;
        res = X.map(match);
        let ok = 0;
        for (const v of res) if (v >= 0) ok++;
        okV = ok / G;
        msg += ' Finished: ' + ok + ' of ' + G + ' samples are ' + K.what + '.';
      } else {
        PR = probs(X);
        cu = 5;
      }
    };
    const step = () => {
      if (phase === 'train') train(50);
      else if (phase === 'bench') bench();
      else if (done) {
        startGen();
        cu = 3;
        msg = 'Eight fresh samples with everything hidden. Press Run or One step.';
      } else gstep();
    };
    const cell = (g, t, x, y, s, v, p, hot) => {
      if (v < 0) {
        g.fillStyle = mix(t.accent, t.accent, 0, 0.08 + 0.6 * p);
        g.fillRect(x, y, s - 1, s - 1);
      } else {
        g.fillStyle = v ? t.ink : t.bg;
        g.fillRect(x, y, s - 1, s - 1);
        if (!v) {
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.strokeRect(x + 0.5, y + 0.5, s - 2, s - 2);
        }
      }
      if (hot) {
        g.strokeStyle = t.gold;
        g.lineWidth = 2.5;
        g.strokeRect(x + 0.5, y + 0.5, s - 2, s - 2);
      }
    };
    const tile = (g, t, x, y, s, v, P0, hot) => {
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      if (v < 0) {
        let b = 0;
        for (let j = 1; j < 26; j++) if (P0[j] > P0[b]) b = j;
        g.fillStyle = mix(t.accent, t.accent, 0, 0.08 + 0.3 * P0[b]);
        g.fillRect(x, y, s - 3, s - 3);
        g.fillStyle = mix(t.accent, t.accent, 0, 0.35 + 0.65 * P0[b]);
        g.font = '500 ' + Math.round(s * 0.42) + 'px ' + MONO.split('px ')[1];
        g.fillText(String.fromCharCode(97 + b), x + (s - 3) / 2, y + (s - 3) / 2 - s * 0.08);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.fillText((100 * P0[b]).toFixed(0) + '%', x + (s - 3) / 2, y + s * 0.74);
      } else {
        g.fillStyle = t.bg;
        g.fillRect(x, y, s - 3, s - 3);
        g.strokeStyle = t.ink;
        g.lineWidth = 1.2;
        g.strokeRect(x + 0.5, y + 0.5, s - 4, s - 4);
        g.fillStyle = t.ink;
        g.font = '600 ' + Math.round(s * 0.55) + 'px ' + MONO.split('px ')[1];
        g.fillText(String.fromCharCode(97 + v), x + (s - 3) / 2, y + (s - 3) / 2);
      }
      if (hot) {
        g.strokeStyle = t.gold;
        g.lineWidth = 3;
        g.strokeRect(x + 0.5, y + 0.5, s - 4, s - 4);
      }
      g.textBaseline = 'alphabetic';
    };
    const draw = () => {
      const t = T(),
        Ln = K.Ln,
        V = K.V,
        N = K.NS[S.ni],
        hot = new Set(just);
      {
        const { g, w, h } = fit(c);
        g.font = MONO;
        if (!X) {
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText(
            phase === 'bench'
              ? 'measuring each number of reveal steps'
              : 'the eight samples appear here once the network is trained',
            w / 2,
            h / 2,
          );
        } else if (S.kind === 'pic') {
          const s = Math.floor(Math.min((w - 50) / (4 * 5 + 3 * 1.2), (h - 70) / (2 * 7 + 1.6)));
          const ox = (w - (4 * 5 * s + 3 * 1.2 * s)) / 2,
            oy = (h - (2 * 7 * s + 1.6 * s)) / 2 - 6;
          for (let b = 0; b < G; b++) {
            const x0 = ox + (b % 4) * 6.2 * s,
              y0 = oy + Math.floor(b / 4) * 8.6 * s;
            for (let i = 0; i < 35; i++)
              cell(
                g,
                t,
                x0 + (i % 5) * s,
                y0 + Math.floor(i / 5) * s,
                s,
                X[b][i],
                PR ? PR[(b * Ln + i) * V + 1] : 0,
                hot.has(b * Ln + i),
              );
            if (done) {
              g.font = MONO;
              g.textAlign = 'center';
              g.fillStyle = res[b] >= 0 ? t.ok : t.cut;
              g.fillText(
                res[b] >= 0 ? 'reads as ' + CHN[res[b]] : 'not a letter',
                x0 + 2.5 * s,
                y0 + 7 * s + 15,
              );
            }
          }
        } else {
          const s = Math.floor(Math.min((w - 70) / (2 * 5 + 1.4), (h - 60) / (4 * 1.5)));
          const ox = (w - (2 * 5 * s + 1.4 * s)) / 2,
            oy = (h - 4 * 1.5 * s) / 2;
          for (let b = 0; b < G; b++) {
            const x0 = ox + (b % 2) * 6.4 * s,
              y0 = oy + Math.floor(b / 2) * 1.5 * s;
            for (let i = 0; i < 5; i++)
              tile(
                g,
                t,
                x0 + i * s,
                y0,
                s,
                X[b][i],
                PR ? PR.subarray((b * Ln + i) * V, (b * Ln + i + 1) * V) : null,
                hot.has(b * Ln + i),
              );
            if (done) {
              g.font = MONO;
              g.textAlign = 'left';
              g.fillStyle = res[b] >= 0 ? t.ok : t.cut;
              g.fillText(res[b] >= 0 ? 'a real word' : 'not a word', x0, y0 + s + 9);
            }
          }
        }
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.44,
          FR = [0, 0.25, 0.5, 0.75, 1],
          pw = (w - 20) / 5;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('one training example, hidden more and more: what training sees', 8, 13);
        FR.forEach((f, q) => {
          const nh = Math.round(f * Ln),
            hid = new Set(exO.slice(0, nh)),
            cx = 10 + q * pw + pw / 2;
          if (S.kind === 'pic') {
            const s = Math.floor(Math.min((pw - 14) / 5, (h1 - 48) / 7)),
              x0 = cx - 2.5 * s,
              y0 = 22;
            for (let i = 0; i < 35; i++) {
              const x = x0 + (i % 5) * s,
                y = y0 + Math.floor(i / 5) * s;
              if (hid.has(i)) {
                g.fillStyle = mix(t.accent, t.accent, 0, 0.22);
                g.fillRect(x, y, s - 1, s - 1);
              } else {
                g.fillStyle = ex[i] ? t.ink : t.bg;
                g.fillRect(x, y, s - 1, s - 1);
                if (!ex[i]) {
                  g.strokeStyle = t.line;
                  g.lineWidth = 1;
                  g.strokeRect(x + 0.5, y + 0.5, s - 2, s - 2);
                }
              }
            }
          } else {
            const rh = (h1 - 24) / 5,
              s = Math.floor(Math.min(rh - 4, 30)),
              x0 = w / 2 - 2.5 * s - 50,
              y0 = 20 + q * rh;
            for (let i = 0; i < 5; i++) {
              const x = x0 + i * s;
              g.textAlign = 'center';
              if (hid.has(i)) {
                g.fillStyle = mix(t.accent, t.accent, 0, 0.22);
                g.fillRect(x, y0, s - 2, s - 2);
              } else {
                g.strokeStyle = t.ink;
                g.lineWidth = 1;
                g.strokeRect(x + 0.5, y0 + 0.5, s - 3, s - 3);
                g.fillStyle = t.ink;
                g.font = '600 ' + Math.round(s * 0.6) + 'px ' + MONO.split('px ')[1];
                g.fillText(String.fromCharCode(97 + ex[i]), x + (s - 2) / 2, y0 + s * 0.68);
              }
            }
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText(Math.round(f * 100) + '% hidden', x0 + 5 * s + 12, y0 + s * 0.62);
            return;
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText(Math.round(f * 100) + '% hidden', cx, h1 - 6);
        });
        if (phase === 'train')
          uLine(g, t, 0, h1, w, h - h1, sm, {
            title: 'error at guessing hidden tokens (average of every 20 steps)',
            n: MAXS / 20,
            col: t.ok,
            ym: Math.max(Math.log(V) * 1.05, sm[0] || 0),
          });
        else
          g2Bench(
            g,
            t,
            0,
            h1,
            w,
            h - h1,
            K.NS,
            [
              { ys: Q[0], col: t.cut, lab: 'random order' },
              { ys: Q[1], col: t.accent, lab: 'most confident first' },
            ],
            phase === 'gen' ? S.ni : -1,
            'share of samples that are ' + K.what + ', by number of reveal steps',
          );
      }
      live.textContent = msg;
      uCur(root, 'dd', cu);
      setR(
        root,
        'ph',
        phase === 'train'
          ? ts
            ? 'Training'
            : 'Not started'
          : phase === 'bench'
            ? 'Measuring'
            : done
              ? 'Finished'
              : 'Generating',
      );
      setR(root, 'ts', ts + ' of ' + MAXS);
      setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
      setR(root, 'rs', phase === 'gen' ? si + ' of ' + N : '–');
      setR(root, 'hid', phase === 'gen' ? Ln - rev + ' of ' + Ln : '–');
      setR(root, 'ok', isFinite(okV) ? Math.round(okV * G) + ' of ' + G : '–');
    };
    const sl = $('#dd-n', root),
      slo = $('#dd-n-o', root),
      setSl = () => {
        sl.max = K.NS.length - 1;
        S.ni = K.def;
        sl.value = K.def;
        if (slo) slo.textContent = String(K.NS[S.ni]);
      };
    build();
    setSl();
    if (RM) {
      train(MAXS);
      while (phase === 'bench') bench();
      while (!done) gstep();
    }
    const anim = Anim(() => {
      fr++;
      if (phase === 'train') {
        train(Math.min(ts < 60 ? 1 : 5, chk.remaining()));
        draw();
        chk();
        return;
      }
      if (phase === 'bench') {
        if (fr % 5 === 0) {
          bench();
          draw();
        }
        return;
      }
      if (fr % clamp(Math.round(260 / K.NS[S.ni]), 14, 70) === 0 && !done) gstep();
      draw();
      if (done) pb.stop();
    });
    pb = playBtn($('#dd-play', root), anim);
    chk = runCtl(pb, root, 'dd', {
      done: () => phase === 'gen' && done,
      restart: () => startGen(),
      count: () => ts,
      what: 'training steps',
    });
    const regen = () => {
      if (phase !== 'gen') {
        msg = 'Samples can be generated once training and measuring are finished. Press Run.';
        draw();
        return;
      }
      pb.stop();
      startGen();
      draw();
      pb.start();
    };
    $('#dd-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#dd-reset', root).onclick = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    $('#dd-gen', root).onclick = regen;
    bind(
      root,
      'dd-n',
      (v) => {
        if (!pb) return;
        S.ni = clamp(+v, 0, K.NS.length - 1);
        if (phase === 'gen') regen();
        else draw();
      },
      (v) => String((K || KD.pic).NS[Math.min(+v, (K || KD.pic).NS.length - 1)]),
    );
    segBind(root, 'dd-ord', S.ord, (v) => {
      S.ord = v;
      if (phase === 'gen') regen();
    });
    segBind(root, 'dd-kind', S.kind, (v) => {
      S.kind = v;
      pb.stop();
      run++;
      build();
      setSl();
      draw();
      pb.start();
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        run++;
        build();
        draw();
        pb.start();
      },
    };
  },
};
