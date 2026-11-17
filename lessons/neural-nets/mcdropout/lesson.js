/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  clipPlot,
  f2,
  fit,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { NNet } from '../shared.js';
import { uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#mc-c', root),
      cr = $('#mc-r', root),
      live = $('#mc-live', root),
      S = { p: 0.1, hw: 48, gap: 2.2, nz: 0.12, seed: 2 },
      NG = 110,
      EPM = 1200,
      f = (x) => Math.sin(1.6 * x) + 0.25 * x;
    let D = [],
      net,
      ep = 0,
      passes = [],
      sum,
      sq,
      k = 0,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      det = null,
      last = 0,
      stable = 0,
      done = false,
      tph = [];
    const gx = (i) => -6 + (12 * i) / (NG - 1),
      gen = () => {
        const r = rng(S.seed * 9 + 4),
          rn = randnFrom(r);
        D = [];
        for (let i = 0; i < 46; i++) {
          const left = i % 2 === 0,
            x = left
              ? -2.6 + r() * (2.6 - S.gap / 2 - 0.15)
              : S.gap / 2 + 0.15 + r() * (2.6 - S.gap / 2 - 0.15);
          D.push({ x, y: f(x) + S.nz * rn() });
        }
      };
    const build = () => {
      net = NNet([1, S.hw, S.hw, 1], { act: 'relu', out: 'lin', seed: S.seed * 5 + run * 11 });
      ep = 0;
      passes = [];
      sum = new Float64Array(NG);
      sq = new Float64Array(NG);
      k = 0;
      det = null;
      done = false;
      stable = 0;
      tph = [];
      cu = 0;
      msg =
        'Stage 1: training a network with dropout (p = ' +
        f2(S.p) +
        ') on ' +
        D.length +
        ' measurements. Nothing lies in the gap between x = ' +
        f2(-S.gap / 2) +
        ' and ' +
        f2(S.gap / 2) +
        '.';
    };
    const sdAt = (i) =>
        k > 1 ? Math.sqrt(Math.max(0, (sq[i] - (sum[i] * sum[i]) / k) / (k - 1))) : 0,
      avg = (a, b) => {
        let s = 0,
          n = 0;
        for (let i = 0; i < NG; i++) {
          const x = gx(i);
          if (x >= a && x <= b) {
            s += sdAt(i);
            n++;
          }
        }
        return n ? s / n : 0;
      };
    const pass = () => {
      const mask = [0, 1].map(() =>
          Float64Array.from({ length: S.hw }, () => (Math.random() < S.p ? 0 : 1 / (1 - S.p))),
        ),
        ft = new Float32Array(NG);
      for (let i = 0; i < NG; i++) {
        const v = net.fwd([gx(i)], mask).A[3][0];
        ft[i] = v;
        sum[i] += v;
        sq[i] += v * v;
      }
      passes.push(ft);
      if (passes.length > 50) passes.shift();
      k++;
      tph.push(net.fwd([tp.v[0]], mask).A[3][0]);
      cu = k < 2 ? 2 : k < 5 ? 3 : 4;
      const tot = avg(-6, 6);
      if (k > 12 && Math.abs(tot - last) < 0.012 * tot) stable++;
      else stable = 0;
      last = tot;
      if (stable >= 14 || k >= 400) done = true;
      msg =
        'Pass ' +
        k +
        ': a random ' +
        mask.reduce((s, m) => s + m.filter((q) => !q).length, 0) +
        ' of the ' +
        2 * S.hw +
        ' hidden neurons were switched off and the rest drew this curve. So far the answers disagree by ±' +
        f2(avg(-2.5, -S.gap / 2 - 0.2)) +
        ' over the data and by ±' +
        f2(avg(-S.gap / 2 + 0.1, S.gap / 2 - 0.1) || 0) +
        ' in the gap.' +
        (done ? ' More passes no longer change the band, so it stops after ' + k + ' passes.' : '');
    };
    const tr = (n) => {
      for (let i = 0; i < n && ep < EPM; i++) {
        net.train(
          D.map((q) => [q.x]),
          D.map((q) => q.y),
          0.01,
          { drop: S.p },
        );
        ep++;
      }
      if (ep >= EPM && !det) {
        det = new Float32Array(NG);
        for (let i = 0; i < NG; i++) det[i] = net.p([gx(i)]);
        cu = 1;
        msg =
          'Stage 2: training is finished. The black line is the ordinary prediction, with every neuron on: one confident answer for every x, including across the gap. Now dropout is left on.';
      } else if (ep < EPM) msg = 'Stage 1: training with dropout, epoch ' + ep + ' of ' + EPM + '.';
    };
    const step = () => {
      if (done) {
        run++;
        build();
        return;
      }
      if (ep < EPM) tr(EPM);
      else pass();
    };
    const tp = testPoint(root, 'mc', ['Your x'], [0], [[-6, 6]]);
    const draw = () => {
      const t = T();
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, [-6, 6], [-2.6, 2.6], t, {
            xstep: 2,
            yticks: [-2, 0, 2],
            xlabel: 'x',
          });
        clipPlot(g, P);
        g.fillStyle = mix(t.ok, t.ok, 0, 0.07);
        g.fillRect(
          P.X(-S.gap / 2),
          P.Y(2.6),
          P.X(S.gap / 2) - P.X(-S.gap / 2),
          P.Y(-2.6) - P.Y(2.6),
        );
        g.fillRect(P.X(-6), P.Y(2.6), P.X(-2.6) - P.X(-6), P.Y(-2.6) - P.Y(2.6));
        g.fillRect(P.X(2.6), P.Y(2.6), P.X(6) - P.X(2.6), P.Y(-2.6) - P.Y(2.6));
        if (k > 1) {
          g.beginPath();
          for (let i = 0; i < NG; i++) {
            const y = sum[i] / k + 1.96 * sdAt(i);
            i ? g.lineTo(P.X(gx(i)), P.Y(y)) : g.moveTo(P.X(gx(i)), P.Y(y));
          }
          for (let i = NG - 1; i >= 0; i--) g.lineTo(P.X(gx(i)), P.Y(sum[i] / k - 1.96 * sdAt(i)));
          g.closePath();
          g.fillStyle = mix(t.accent, t.accent, 0, 0.2);
          g.fill();
        }
        passes.forEach((ft, q) => {
          const lastOne = q === passes.length - 1;
          g.strokeStyle = mix(t.accent, t.accent, 0, lastOne ? 1 : 0.22);
          g.lineWidth = lastOne ? 2 : 1;
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i ? g.lineTo(P.X(gx(i)), P.Y(ft[i])) : g.moveTo(P.X(gx(i)), P.Y(ft[i]));
          g.stroke();
        });
        if (ep < EPM) {
          g.strokeStyle = t.ink;
          g.lineWidth = 2;
          g.beginPath();
          for (let i = 0; i < NG; i++) {
            const y = net.p([gx(i)]);
            i ? g.lineTo(P.X(gx(i)), P.Y(y)) : g.moveTo(P.X(gx(i)), P.Y(y));
          }
          g.stroke();
        }
        if (det) {
          g.strokeStyle = t.ink;
          g.lineWidth = 2;
          g.setLineDash(k ? [6, 4] : []);
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i ? g.lineTo(P.X(gx(i)), P.Y(det[i])) : g.moveTo(P.X(gx(i)), P.Y(det[i]));
          g.stroke();
          g.setLineDash([]);
        }
        if (k > 1) {
          g.strokeStyle = t.gold;
          g.lineWidth = 3;
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i ? g.lineTo(P.X(gx(i)), P.Y(sum[i] / k)) : g.moveTo(P.X(gx(i)), P.Y(sum[i] / k));
          g.stroke();
        }
        g.fillStyle = t.ink;
        for (const q of D) {
          g.beginPath();
          g.arc(P.X(q.x), P.Y(q.y), 3.2, 0, 7);
          g.fill();
        }
        const px = P.X(tp.v[0]);
        g.strokeStyle = t.ink;
        g.lineWidth = 1.2;
        g.beginPath();
        g.moveTo(px, P.Y(-2.6));
        g.lineTo(px, P.Y(2.6));
        g.stroke();
        g.restore();
        g.font = MONO;
        g.fillStyle = t.ok;
        g.textAlign = 'center';
        g.fillText('no data here', P.X(0), P.m.t + 10);
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5,
          m = { l: 40, r: 12, t: 24, b: 20 };
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          'answers given at x = ' + f2(tp.v[0]) + ' by the ' + tph.length + ' passes',
          8,
          13,
        );
        const X = (v) => m.l + ((v + 2.6) / 5.2) * (w - m.l - m.r),
          nb = 40,
          cnt = new Array(nb).fill(0);
        for (const v of tph) cnt[clamp(Math.floor(((v + 2.6) / 5.2) * nb), 0, nb - 1)]++;
        const cm = Math.max(4, ...cnt);
        g.fillStyle = t.accent;
        cnt.forEach((q, i) =>
          g.fillRect(
            m.l + (i * (w - m.l - m.r)) / nb,
            h1 - m.b - (q / cm) * (h1 - m.t - m.b),
            (w - m.l - m.r) / nb - 1,
            (q / cm) * (h1 - m.t - m.b),
          ),
        );
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h1 - m.b);
        g.lineTo(w - m.r, h1 - m.b);
        g.stroke();
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        [-2, -1, 0, 1, 2].forEach((v) => g.fillText(String(v), X(v), h1 - m.b + 13));
        if (det) {
          const gi = Math.round(((tp.v[0] + 6) / 12) * (NG - 1));
          g.strokeStyle = t.ink;
          g.lineWidth = 2;
          g.setLineDash([6, 4]);
          g.beginPath();
          g.moveTo(X(det[gi]), m.t);
          g.lineTo(X(det[gi]), h1 - m.b);
          g.stroke();
          g.setLineDash([]);
          g.strokeStyle = t.ok;
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(X(f(tp.v[0])), m.t);
          g.lineTo(X(f(tp.v[0])), h1 - m.b);
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('dashed: ordinary answer', w - m.r, m.t + 12);
          g.fillText('red: the truth', w - m.r, m.t + 25);
        }
        const sds = Array.from({ length: NG }, (_, i) => sdAt(i)),
          U = uLine(g, t, 0, h1, w, h - h1, sds, {
            title: 'disagreement between passes (±), at every x',
            n: NG,
            ym: Math.max(0.3, ...sds) * 1.1,
            col: t.accent,
          });
        g.fillStyle = t.ink;
        for (const q of D) {
          g.fillRect(U.X(((q.x + 6) / 12) * (NG - 1)) - 1, h - 17, 2, 7);
        }
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('ticks: where the training data is', w - 12, h - 3);
      }
      live.textContent = msg;
      uCur(root, 'mc', cu);
      const gi = Math.round(((tp.v[0] + 6) / 12) * (NG - 1));
      tp.out.textContent = !det
        ? 'The network is still training.'
        : k < 2
          ? 'Ordinary prediction at x = ' +
            f2(tp.v[0]) +
            ': ' +
            f2(det[gi]) +
            ', with nothing to say how sure it is.'
          : 'At x = ' +
            f2(tp.v[0]) +
            ' the ordinary network says ' +
            f2(det[gi]) +
            '. The ' +
            k +
            ' dropout passes average ' +
            f2(sum[gi] / k) +
            ' and disagree by ±' +
            f2(sdAt(gi)) +
            ', so a 95% range is ' +
            f2(sum[gi] / k - 1.96 * sdAt(gi)) +
            ' to ' +
            f2(sum[gi] / k + 1.96 * sdAt(gi)) +
            '. The truth is ' +
            f2(f(tp.v[0])) +
            '.';
      let ins = 0;
      if (k > 1)
        for (const q of D) {
          const i = Math.round(((q.x + 6) / 12) * (NG - 1));
          if (Math.abs(q.y - sum[i] / k) <= 1.96 * sdAt(i) + 1e-9) ins++;
        }
      setR(
        root,
        'ph',
        ep < EPM
          ? 'Training (' + ep + ' of ' + EPM + ' epochs)'
          : done
            ? 'Finished'
            : k
              ? 'Polling with dropout on'
              : 'Trained',
      );
      setR(root, 'k', String(k));
      setR(
        root,
        'sd',
        k > 1 ? f2((avg(-2.5, -S.gap / 2 - 0.2) + avg(S.gap / 2 + 0.2, 2.5)) / 2) : '–',
      );
      setR(root, 'sg', k > 1 && S.gap > 0.3 ? f2(avg(-S.gap / 2 + 0.1, S.gap / 2 - 0.1)) : '–');
      setR(root, 'so', k > 1 ? f2((avg(-6, -4.5) + avg(4.5, 6)) / 2) : '–');
      setR(root, 'ins', k > 1 ? ins + ' of ' + D.length : '–');
    };
    tp.on = () => {
      tph = [];
      if (det && k) {
        /* re-poll this x with fresh masks */ for (let q = 0; q < Math.min(k, 200); q++) {
          const mask = [0, 1].map(() =>
            Float64Array.from({ length: S.hw }, () => (Math.random() < S.p ? 0 : 1 / (1 - S.p))),
          );
          tph.push(net.fwd([tp.v[0]], mask).A[3][0]);
        }
      }
      draw();
    };
    gen();
    build();
    if (RM) {
      tr(EPM);
      for (let i = 0; i < 80; i++) pass();
    }
    const redo = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      fr++;
      if (ep < EPM) {
        tr(12);
        draw();
        return;
      }
      if (fr % (k < 8 ? 34 : 5)) return;
      pass();
      draw();
      chk();
    });
    pb = playBtn($('#mc-play', root), anim);
    chk = runCtl(pb, root, 'mc', {
      done: () => done,
      restart: () => {
        run++;
        build();
      },
      count: () => k,
      what: 'passes',
    });
    $('#mc-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#mc-reset', root).onclick = redo;
    bind(root, 'mc-p', (v) => {
      S.p = v;
      redo();
    });
    bind(
      root,
      'mc-hw',
      (v) => {
        S.hw = v;
        redo();
      },
      (v) => String(v),
    );
    bind(root, 'mc-gap', (v) => {
      S.gap = v;
      gen();
      redo();
    });
    bind(root, 'mc-nz', (v) => {
      S.nz = v;
      gen();
      redo();
    });
    $('#mc-new', root).onclick = () => {
      S.seed++;
      gen();
      redo();
    };
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
