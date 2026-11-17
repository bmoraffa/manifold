/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
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
import { UC, segBind, uCur, uLine } from '../../unsupervised/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#me-c', root),
      cr = $('#me-r', root),
      live = $('#me-live', root),
      S = { K: 3, rt: 'soft', ds: 'three', seed: 1 },
      N = 160,
      MAXS = 4000;
    let P = [],
      th,
      M1,
      M2,
      ts = 0,
      LL = [],
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      fin = false,
      e1 = NaN,
      run = 1;
    const F = {
      three: (x) => (x < -1 ? x + 2.5 : x < 1 ? -1.2 * x + 0.3 : 0.4 * x - 1.3),
      step: (x) => (x < 0 ? -1 : 1 + 0.5 * x),
      zig: (x) => {
        const u = (((x + 3) % 2) + 2) % 2;
        return 1.6 * Math.abs(u - 1) - 0.8;
      },
      wave: (x) => 1.3 * Math.sin(1.5 * x),
    };
    const gen = () => {
      const r = rng(S.seed * 31 + 7),
        rn = randnFrom(r);
      P = Array.from({ length: N }, () => {
        const x = -3 + 6 * r();
        return [x, F[S.ds](x) + 0.12 * rn()];
      });
      let sx = 0,
        sy = 0,
        sxx = 0,
        sxy = 0;
      for (const p of P) {
        sx += p[0];
        sy += p[1];
        sxx += p[0] * p[0];
        sxy += p[0] * p[1];
      }
      const a = (N * sxy - sx * sy) / (N * sxx - sx * sx),
        b = (sy - a * sx) / N;
      e1 = P.reduce((s, p) => s + (a * p[0] + b - p[1]) ** 2, 0) / N;
    };
    const init0 = () => {
      const K = S.K,
        rn = randnFrom(rng(run * 19 + K));
      th = new Float64Array(4 * K);
      for (let k = 0; k < K; k++) {
        th[4 * k] = 0.4 * rn();
        th[4 * k + 1] = 0.4 * rn();
        th[4 * k + 2] = 0.6 * rn();
        th[4 * k + 3] = 0.1 * rn();
      }
      M1 = new Float64Array(4 * K);
      M2 = new Float64Array(4 * K);
      ts = 0;
      LL = [];
      fin = false;
      cu = 0;
      msg =
        K +
        ' expert' +
        (K > 1
          ? 's, each a random line, and a router that has not yet decided anything'
          : ', a single line, so there is nothing to route') +
        '. Press Run.';
    };
    const gate = (x, g) => {
      const K = S.K;
      let mx = -1e30;
      for (let k = 0; k < K; k++) {
        g[k] = th[4 * k + 2] * x + th[4 * k + 3];
        if (g[k] > mx) mx = g[k];
      }
      let z = 0;
      for (let k = 0; k < K; k++) {
        g[k] = Math.exp(g[k] - mx);
        z += g[k];
      }
      for (let k = 0; k < K; k++) g[k] /= z;
      return g;
    };
    const out = (x, hard) => {
      const K = S.K,
        g = gate(x, new Float64Array(K));
      if (hard) {
        let b = 0;
        for (let k = 1; k < K; k++) if (g[k] > g[b]) b = k;
        return th[4 * b] * x + th[4 * b + 1];
      }
      let y = 0;
      for (let k = 0; k < K; k++) y += g[k] * (th[4 * k] * x + th[4 * k + 1]);
      return y;
    };
    const err = (hard) => P.reduce((s, p) => s + (out(p[0], hard) - p[1]) ** 2, 0) / N;
    const load = () => {
      const K = S.K,
        L = new Float64Array(K),
        g = new Float64Array(K);
      for (const p of P) {
        gate(p[0], g);
        for (let k = 0; k < K; k++) L[k] += g[k] / N;
      }
      return L;
    };
    const train = (n) => {
      const K = S.K,
        G = new Float64Array(4 * K),
        g = new Float64Array(K);
      for (let q = 0; q < n && !fin; q++) {
        G.fill(0);
        let L = 0;
        for (const p of P) {
          const x = p[0];
          gate(x, g);
          let y = 0;
          for (let k = 0; k < K; k++) y += g[k] * (th[4 * k] * x + th[4 * k + 1]);
          const d = y - p[1];
          L += (d * d) / N;
          for (let k = 0; k < K; k++) {
            const f = th[4 * k] * x + th[4 * k + 1],
              gf = (2 * d * g[k]) / N,
              gu = (2 * d * g[k] * (f - y)) / N;
            G[4 * k] += gf * x;
            G[4 * k + 1] += gf;
            G[4 * k + 2] += gu * x;
            G[4 * k + 3] += gu;
          }
        }
        ts++;
        LL.push(L);
        const c1 = 1 - Math.pow(0.9, ts),
          c2 = 1 - Math.pow(0.999, ts);
        for (let i = 0; i < 4 * K; i++) {
          M1[i] = 0.9 * M1[i] + 0.1 * G[i];
          M2[i] = 0.999 * M2[i] + 0.001 * G[i] * G[i];
          th[i] -= (0.03 * (M1[i] / c1)) / (Math.sqrt(M2[i] / c2) + 1e-8);
        }
        if (
          ts >= MAXS ||
          (ts > 150 && Math.abs(LL[ts - 1] - LL[ts - 101]) < 1e-3 * Math.max(1e-3, LL[ts - 1]))
        )
          fin = true;
      }
      const Ld = load(),
        top = Array.from(Ld)
          .map((v, k) => [v, k])
          .sort((a, b) => b[0] - a[0]);
      cu = fin ? 5 : 1 + (Math.floor(ts / 40) % 4);
      msg =
        'Step ' +
        ts +
        ': the mixture’s error is ' +
        f3(LL[ts - 1]) +
        ' (one straight line alone: ' +
        f3(e1) +
        '). ' +
        (K > 1
          ? 'Expert ' +
            (top[0][1] + 1) +
            ' currently handles the most data, ' +
            pct(top[0][0]) +
            '; expert ' +
            (top[K - 1][1] + 1) +
            ' the least, ' +
            pct(top[K - 1][0]) +
            '.'
          : '');
      if (fin) {
        const act = top.filter((v) => v[0] >= 0.05).length;
        msg =
          (ts >= MAXS
            ? 'Stopped at the limit of ' + MAXS + ' steps.'
            : 'Converged after ' + ts + ' steps: the error stopped changing.') +
          ' Blended error ' +
          f3(err(false)) +
          ', top-expert-only error ' +
          f3(err(true)) +
          ', one line alone ' +
          f3(e1) +
          '. ' +
          act +
          ' of ' +
          K +
          ' expert' +
          (K > 1 ? 's handle' : ' handles') +
          ' at least 5% of the data.';
      }
    };
    const tp = testPoint(root, 'me', ['Your x'], [0.5], [[-3, 3]]);
    const draw = () => {
      const t = T(),
        col = UC(t),
        K = S.K,
        g = new Float64Array(K),
        hard = S.rt === 'hard';
      {
        const { g: G, w, h } = fit(c),
          X = (x) => 30 + ((x + 3) / 6) * (w - 50),
          Y = (y) => h / 2 - (y * (h - 50)) / 4.6;
        G.strokeStyle = t.line;
        G.lineWidth = 1;
        G.beginPath();
        G.moveTo(X(-3), Y(0));
        G.lineTo(X(3), Y(0));
        G.stroke();
        G.fillStyle = mix(t.ink, t.ink, 0, 0.4);
        for (const p of P) {
          G.beginPath();
          G.arc(X(p[0]), Y(p[1]), 2.6, 0, 7);
          G.fill();
        }
        const NS = 120;
        for (let k = 0; k < K; k++) {
          for (let i = 0; i < NS; i++) {
            const xa = -3 + (6 * i) / NS,
              xb = -3 + (6 * (i + 1)) / NS;
            gate((xa + xb) / 2, g);
            G.strokeStyle = mix(col[k % 8], col[k % 8], 0, 0.1 + 0.9 * g[k]);
            G.lineWidth = 1 + 3 * g[k];
            G.beginPath();
            G.moveTo(X(xa), Y(th[4 * k] * xa + th[4 * k + 1]));
            G.lineTo(X(xb), Y(th[4 * k] * xb + th[4 * k + 1]));
            G.stroke();
          }
        }
        G.strokeStyle = t.ink;
        G.lineWidth = 2;
        G.setLineDash(hard ? [] : []);
        G.beginPath();
        for (let i = 0; i <= 240; i++) {
          const x = -3 + (6 * i) / 240,
            y = out(x, hard);
          i ? G.lineTo(X(x), Y(y)) : G.moveTo(X(x), Y(y));
        }
        G.stroke();
        markTP(G, X(tp.v[0]), Y(out(tp.v[0], hard)), t);
        G.font = MONO;
        G.textAlign = 'left';
        for (let k = 0; k < K; k++) {
          G.fillStyle = col[k % 8];
          G.fillText('expert ' + (k + 1), 10 + k * 74, 14);
        }
        G.fillStyle = t.muted;
        G.fillText(
          'thick line: ' +
            (hard ? 'the top expert’s answer at each x' : 'all experts blended by their shares'),
          8,
          h - 8,
        );
      }
      {
        const { g: G, w, h } = fit(cr),
          h1 = h * 0.34,
          h2 = h * 0.24,
          m = { l: 44, r: 12 },
          X = (x) => m.l + ((x + 3) / 6) * (w - m.l - m.r);
        G.font = MONO;
        G.fillStyle = t.muted;
        G.textAlign = 'left';
        G.fillText('the router’s shares at every input x: who is in charge where', 8, 13);
        const NS = 140,
          y0 = 22,
          H = h1 - 36;
        for (let i = 0; i < NS; i++) {
          const x = -3 + (6 * (i + 0.5)) / NS;
          gate(x, g);
          let acc = 0;
          const xa = Math.round(X(-3 + (6 * i) / NS)),
            xb = Math.round(X(-3 + (6 * (i + 1)) / NS));
          for (let k = 0; k < K; k++) {
            G.fillStyle = mix(col[k % 8], col[k % 8], 0, 0.85);
            const ya = Math.round(y0 + acc * H),
              yb = Math.round(y0 + (acc + g[k]) * H);
            G.fillRect(xa, ya, xb - xa, yb - ya);
            acc += g[k];
          }
        }
        G.fillStyle = t.muted;
        G.textAlign = 'left';
        G.fillText('x = −3', m.l, y0 + H + 12);
        G.textAlign = 'right';
        G.fillText('x = 3', w - m.r, y0 + H + 12);
        const tx = X(tp.v[0]);
        G.strokeStyle = t.ink;
        G.lineWidth = 1.5;
        G.beginPath();
        G.moveTo(tx, y0 - 3);
        G.lineTo(tx, y0 + H + 3);
        G.stroke();
        const Ld = load(),
          yb0 = h1 + 20,
          bh = Math.min(16, (h2 - 24) / K);
        G.fillStyle = t.muted;
        G.textAlign = 'left';
        G.fillText('share of the data each expert handles', 8, h1 + 13);
        for (let k = 0; k < K; k++) {
          G.fillStyle = col[k % 8];
          G.fillRect(m.l, yb0 + k * bh + 1, Math.max(1, (w - m.l - 70) * Ld[k]), bh - 3);
          G.fillStyle = t.muted;
          G.textAlign = 'right';
          G.fillText(String(k + 1), m.l - 6, yb0 + k * bh + bh - 4);
          G.textAlign = 'left';
          G.fillText(
            (100 * Ld[k]).toFixed(0) + '%',
            m.l + (w - m.l - 70) * Ld[k] + 5,
            yb0 + k * bh + bh - 4,
          );
        }
        const Lp = uLine(
          G,
          t,
          0,
          h1 + h2,
          w,
          h - h1 - h2,
          LL.map((v) => Math.min(v, e1 * 1.25)),
          {
            title: 'error of the mixture (mean squared)',
            n: Math.max(300, LL.length),
            col: t.ok,
            ym: e1 * 1.25,
          },
        );
        G.setLineDash([5, 4]);
        G.strokeStyle = t.gold;
        G.lineWidth = 1.3;
        G.beginPath();
        G.moveTo(Lp.m.l, Lp.Y(e1));
        G.lineTo(w - Lp.m.r, Lp.Y(e1));
        G.stroke();
        G.setLineDash([]);
        G.fillStyle = t.cut;
        G.textAlign = 'right';
        G.fillText('one straight line alone', w - Lp.m.r, Lp.Y(e1) + 13);
      }
      live.textContent = msg;
      uCur(root, 'me', cu);
      gate(tp.v[0], g);
      let b = 0;
      for (let k = 1; k < K; k++) if (g[k] > g[b]) b = k;
      tp.out.textContent =
        'At x = ' +
        f2(tp.v[0]) +
        ' the router gives expert ' +
        (b + 1) +
        ' the largest share, ' +
        pct(g[b]) +
        '. That expert alone answers ' +
        f2(th[4 * b] * tp.v[0] + th[4 * b + 1]) +
        '. All experts blended answer ' +
        f2(out(tp.v[0], false)) +
        '. The curve the data was drawn from is at ' +
        f2(F[S.ds](tp.v[0])) +
        '.';
      const Ld = load();
      setR(root, 'ts', String(ts) + (fin ? ' (stopped)' : ''));
      setR(root, 'e', ts ? f3(err(false)) : '–');
      setR(root, 'eh', ts ? f3(err(true)) : '–');
      setR(root, 'e1', f3(e1));
      setR(root, 'act', Array.from(Ld).filter((v) => v >= 0.05).length + ' of ' + K);
      setR(root, 'fr', '1 of ' + K + ' (' + pct(1 / K) + ')');
    };
    tp.on = () => draw();
    gen();
    init0();
    if (RM) {
      while (!fin) train(50);
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      train(Math.min(ts < 120 ? 1 : ts < 600 ? 4 : 12, chk.remaining()));
      draw();
      chk();
    });
    pb = playBtn($('#me-play', root), anim);
    chk = runCtl(pb, root, 'me', {
      done: () => fin,
      restart: () => {
        run++;
        init0();
      },
      count: () => ts,
      what: 'training steps',
    });
    $('#me-step', root).onclick = () => {
      pb.stop();
      if (fin) {
        run++;
        init0();
      } else train(20);
      draw();
    };
    $('#me-reset', root).onclick = () => {
      pb.stop();
      run++;
      init0();
      draw();
    };
    bind(
      root,
      'me-K',
      (v) => {
        S.K = +v;
        if (!pb) return;
        pb.stop();
        run++;
        init0();
        draw();
        pb.start();
      },
      (v) => String(v),
    );
    segBind(root, 'me-rt', S.rt, (v) => {
      S.rt = v;
      draw();
    });
    $('#me-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      pb.stop();
      gen();
      run++;
      init0();
      draw();
      pb.start();
    };
    $('#me-new', root).onclick = () => {
      pb.stop();
      S.seed++;
      gen();
      run++;
      init0();
      draw();
      pb.start();
    };
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        run++;
        init0();
        draw();
        pb.start();
      },
    };
  },
};
