/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  T,
  bind,
  clamp,
  clipPlot,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  plot,
  randn,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#bv-c', root),
      cr = $('#bv-r', root),
      live = $('#bv-live', root),
      cap = $('#bv-cap', root),
      S = { d: 5, s: 0.3, n: 25, f: 'wave' },
      FN = {
        wave: (x) => Math.sin(4.6 * x) + 0.4 * x,
        bump: (x) => 1.7 * Math.exp(-9 * x * x) - 0.6,
        line: (x) => 0.9 * x,
      },
      NG = 81,
      DM = 12;
    let xs = [],
      fx = [],
      B2 = [],
      VA = [],
      Hg = [],
      avg = [],
      sd = [],
      Mc = null,
      fits = [],
      sum = null,
      sq = null,
      k = 0,
      last = null,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      sweep = null,
      best = 0;
    const leg = (x, p) => {
      const o = [1];
      if (p > 1) o.push(x);
      for (let j = 1; j < p - 1; j++) o.push(((2 * j + 1) * x * o[j] - j * o[j - 1]) / (j + 1));
      return o;
    };
    const inv = (A) => {
      const n = A.length,
        M = A.map((r, i) => r.concat(r.map((_, j) => (i === j ? 1 : 0))));
      for (let i = 0; i < n; i++) {
        let p = i;
        for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
        [M[i], M[p]] = [M[p], M[i]];
        const d = M[i][i] || 1e-12;
        for (let j = 0; j < 2 * n; j++) M[i][j] /= d;
        for (let r = 0; r < n; r++)
          if (r !== i) {
            const f = M[r][i];
            if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[i][j];
          }
      }
      return M.map((r) => r.slice(n));
    };
    const gx = (g) => -1 + (2 * g) / (NG - 1);
    const smoother = (d) => {
      const p = d + 1,
        n = xs.length,
        Bm = xs.map((x) => leg(x, p)),
        A = [];
      for (let a = 0; a < p; a++) {
        A.push([]);
        for (let b = 0; b < p; b++) {
          let s = 0;
          for (let i = 0; i < n; i++) s += Bm[i][a] * Bm[i][b];
          A[a].push(s + (a === b ? 1e-10 : 0));
        }
      }
      const Ai = inv(A),
        M = [];
      for (let a = 0; a < p; a++) {
        M.push(new Float64Array(n));
        for (let i = 0; i < n; i++) {
          let s = 0;
          for (let b = 0; b < p; b++) s += Ai[a][b] * Bm[i][b];
          M[a][i] = s;
        }
      }
      const H = [];
      for (let g = 0; g < NG; g++) {
        const ph = leg(gx(g), p),
          h = new Float64Array(n);
        for (let a = 0; a < p; a++) for (let i = 0; i < n; i++) h[i] += ph[a] * M[a][i];
        H.push(h);
      }
      return { M, H, p };
    };
    const design = () => {
      const r = rng(31 + S.n);
      xs = [];
      for (let i = 0; i < S.n; i++)
        xs.push(clamp(-1 + (2 * (i + 0.5)) / S.n + ((r() - 0.5) * 1.2) / S.n, -1, 1));
    };
    const comp = () => {
      const f = FN[S.f];
      fx = xs.map(f);
      B2 = [];
      VA = [];
      for (let d = 0; d <= DM; d++) {
        const sm = smoother(d);
        let b = 0,
          v = 0;
        for (let g = 0; g < NG; g++) {
          let m = 0,
            h2 = 0;
          const h = sm.H[g];
          for (let i = 0; i < xs.length; i++) {
            m += h[i] * fx[i];
            h2 += h[i] * h[i];
          }
          b += (m - f(gx(g))) ** 2;
          v += h2;
        }
        B2.push(b / NG);
        VA.push((S.s * S.s * v) / NG);
        if (d === S.d) {
          Mc = sm;
          avg = sm.H.map((h) => {
            let m = 0;
            for (let i = 0; i < xs.length; i++) m += h[i] * fx[i];
            return m;
          });
          sd = sm.H.map((h) => {
            let h2 = 0;
            for (let i = 0; i < xs.length; i++) h2 += h[i] * h[i];
            return S.s * Math.sqrt(h2);
          });
        }
      }
      best = 0;
      for (let d = 1; d <= DM; d++) if (B2[d] + VA[d] < B2[best] + VA[best]) best = d;
    };
    const tot = (d) => B2[d] + VA[d] + S.s * S.s;
    const verdict = () => {
      const b = B2[S.d],
        v = VA[S.d],
        n2 = S.s * S.s,
        T = b + v + n2;
      return (
        (S.d < best
          ? 'This model is too stiff for this world: bias² is ' +
            ((100 * b) / T).toFixed(0) +
            '% of the error. The gold average misses the truth however many batches you draw.'
          : S.d > best
            ? 'This model is too flexible for this much data and noise: variance is ' +
              ((100 * v) / T).toFixed(0) +
              '% of the error. The thin lines chase the noise in each batch.'
            : 'This is the best flexibility for this world: one step stiffer adds more bias than it saves in variance, and one step more flexible does the opposite.') +
        ' Noise accounts for ' +
        f3(n2) +
        ' of the ' +
        f3(T) +
        ', which is ' +
        ((100 * n2) / T).toFixed(0) +
        '%, and no model can remove it.'
      );
    };
    const reset = () => {
      fits = [];
      sum = new Float64Array(NG);
      sq = new Float64Array(NG);
      k = 0;
      last = null;
      cu = 0;
      msg = verdict() + ' Press Draw new batches to see it happen.';
      cap.textContent = verdict();
    };
    const emp = () => {
      if (k < 2) return null;
      let b = 0,
        v = 0;
      const f = FN[S.f];
      for (let g = 0; g < NG; g++) {
        const m = sum[g] / k;
        b += (m - f(gx(g))) ** 2;
        v += (sq[g] - k * m * m) / (k - 1);
      }
      b = Math.max(0, b / NG - v / NG / k);
      return { b, v: v / NG };
    };
    const add = () => {
      const n = xs.length,
        y = xs.map((x, i) => fx[i] + S.s * randn()),
        p = Mc.p,
        co = [];
      for (let a = 0; a < p; a++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += Mc.M[a][i] * y[i];
        co.push(s);
      }
      const ft = new Float32Array(NG);
      let dev = 0;
      for (let g = 0; g < NG; g++) {
        const ph = leg(gx(g), p);
        let s = 0;
        for (let a = 0; a < p; a++) s += ph[a] * co[a];
        ft[g] = s;
        sum[g] += s;
        sq[g] += s * s;
        dev += (s - avg[g]) ** 2;
      }
      fits.push(ft);
      if (fits.length > 45) fits.shift();
      k++;
      last = { y, ft };
      cu = k < 2 ? 1 : k < 4 ? 2 : k < 8 ? 3 : 4;
      const e = emp();
      msg =
        'Batch ' +
        k +
        ': ' +
        n +
        ' new measurements, same model. This fit is on average ' +
        f2(Math.sqrt(dev / NG)) +
        ' away from the gold average.' +
        (e
          ? ' Measured from the ' +
            k +
            ' batches so far: bias² ' +
            f3(e.b) +
            ', variance ' +
            f3(e.v) +
            '. Exact values: ' +
            f3(B2[S.d]) +
            ' and ' +
            f3(VA[S.d]) +
            '.'
          : '');
    };
    const conv = () => {
      const e = emp();
      if (k >= 400) return true;
      if (!e || k < 40) return false;
      const T = B2[S.d] + VA[S.d];
      return (
        Math.abs(e.b + e.v - T) < 0.04 * T + 2e-4 && Math.abs(e.v - VA[S.d]) < 0.06 * VA[S.d] + 2e-4
      );
    };
    const tp = testPoint(root, 'bv', ['Your x'], [0.4], [[-1, 1]]);
    const draw = () => {
      const t = T(),
        f = FN[S.f];
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, [-1, 1], [-2.7, 2.7], t, {
            xstep: 0.5,
            yticks: [-2, 0, 2],
            xlabel: 'x',
          });
        clipPlot(g, P);
        g.beginPath();
        for (let i = 0; i < NG; i++) {
          const x = gx(i);
          i ? g.lineTo(P.X(x), P.Y(f(x) + S.s)) : g.moveTo(P.X(x), P.Y(f(x) + S.s));
        }
        for (let i = NG - 1; i >= 0; i--) {
          const x = gx(i);
          g.lineTo(P.X(x), P.Y(f(x) - S.s));
        }
        g.closePath();
        g.fillStyle = mix(t.ink, t.ink, 0, 0.1);
        g.fill();
        g.beginPath();
        for (let i = 0; i < NG; i++)
          i ? g.lineTo(P.X(gx(i)), P.Y(avg[i] + sd[i])) : g.moveTo(P.X(gx(i)), P.Y(avg[i] + sd[i]));
        for (let i = NG - 1; i >= 0; i--) g.lineTo(P.X(gx(i)), P.Y(avg[i] - sd[i]));
        g.closePath();
        g.fillStyle = mix(t.accent, t.accent, 0, 0.16);
        g.fill();
        g.beginPath();
        for (let i = 0; i < NG; i++)
          i ? g.lineTo(P.X(gx(i)), P.Y(avg[i])) : g.moveTo(P.X(gx(i)), P.Y(avg[i]));
        for (let i = NG - 1; i >= 0; i--) g.lineTo(P.X(gx(i)), P.Y(f(gx(i))));
        g.closePath();
        g.fillStyle = mix(t.ok, t.ok, 0, 0.3);
        g.fill();
        g.lineWidth = 1;
        fits.forEach((ft, q) => {
          g.strokeStyle = mix(t.accent, t.accent, 0, q === fits.length - 1 ? 0 : 0.3);
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i ? g.lineTo(P.X(gx(i)), P.Y(ft[i])) : g.moveTo(P.X(gx(i)), P.Y(ft[i]));
          g.stroke();
        });
        g.setLineDash([7, 5]);
        g.strokeStyle = t.ink;
        g.lineWidth = 1.8;
        g.beginPath();
        for (let i = 0; i < NG; i++)
          i ? g.lineTo(P.X(gx(i)), P.Y(f(gx(i)))) : g.moveTo(P.X(gx(i)), P.Y(f(gx(i))));
        g.stroke();
        g.setLineDash([]);
        g.strokeStyle = t.gold;
        g.lineWidth = 3.4;
        g.beginPath();
        for (let i = 0; i < NG; i++)
          i ? g.lineTo(P.X(gx(i)), P.Y(avg[i])) : g.moveTo(P.X(gx(i)), P.Y(avg[i]));
        g.stroke();
        if (last) {
          g.strokeStyle = t.accent;
          g.lineWidth = 2.4;
          g.beginPath();
          for (let i = 0; i < NG; i++)
            i ? g.lineTo(P.X(gx(i)), P.Y(last.ft[i])) : g.moveTo(P.X(gx(i)), P.Y(last.ft[i]));
          g.stroke();
          g.fillStyle = t.ink;
          xs.forEach((x, i) => {
            g.beginPath();
            g.arc(P.X(x), P.Y(last.y[i]), 3, 0, 7);
            g.fill();
          });
        }
        {
          const x = tp.v[0],
            gi = Math.round(((x + 1) / 2) * (NG - 1)),
            px = P.X(x);
          g.strokeStyle = t.ink;
          g.lineWidth = 1.2;
          g.beginPath();
          g.moveTo(px, P.Y(-2.7));
          g.lineTo(px, P.Y(2.7));
          g.stroke();
          markTP(g, px, P.Y(avg[gi]), t, false);
        }
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        let ly = P.m.t + 8;
        [
          [mix(t.ink, t.ink, 0, 0.45), 'grey band: truth ± σ, the irreducible noise'],
          [t.ok, 'red area: bias, the gap between gold and truth'],
          [t.accent, 'green band and thin lines: variance'],
          [t.cut, 'gold: the average of all possible fits'],
        ].forEach((q) => {
          g.fillStyle = q[0];
          g.fillText(q[1], P.m.l + 8, ly);
          ly += 14;
        });
      }
      {
        const { g, w, h } = fit(cr),
          b = B2[S.d],
          v = VA[S.d],
          n2 = S.s * S.s,
          Tt = b + v + n2,
          e = emp(),
          x0 = 12,
          bw = w - 24,
          mx = Math.max(Tt, e ? e.b + e.v + n2 : 0) * 1.02 || 1;
        g.font = MONO;
        const bar = (y, hh, parts, lab) => {
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(lab, x0, y - 5);
          let x = x0;
          parts.forEach((q) => {
            const ww = (q[0] / mx) * bw;
            g.fillStyle = q[1];
            g.fillRect(x, y, ww, hh);
            if (ww > 58) {
              g.fillStyle = t.bg;
              g.font = SANS;
              g.textAlign = 'center';
              g.fillText(q[2] + ' ' + f3(q[0]), x + ww / 2, y + hh / 2 + 5);
              g.font = MONO;
            }
            x += ww;
          });
        };
        bar(
          20,
          34,
          [
            [b, t.ok, 'bias²'],
            [v, t.accent, 'variance'],
            [n2, mix(t.ink, t.ink, 0, 0.5), 'noise'],
          ],
          'exact: bias² + variance + noise = ' + f3(Tt),
        );
        if (e)
          bar(
            78,
            16,
            [
              [e.b, t.ok, ''],
              [e.v, t.accent, ''],
              [n2, mix(t.ink, t.ink, 0, 0.5), ''],
            ],
            'measured from the ' + k + ' batches drawn so far = ' + f3(e.b + e.v + n2),
          );
        else {
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('draw batches to measure it yourself', x0, 88);
        }
        const y0 = 112,
          m = { l: 40, r: 12, t: 14, b: 26 },
          ym = Math.max(n2 * 2.6, (B2[0] + n2) * 1.15, tot(best) * 2.2),
          X = (d) => m.l + (d / DM) * (w - m.l - m.r),
          Y = (u) => h - m.b - (Math.min(u, ym * 1.2) / ym) * (h - y0 - m.t - m.b);
        g.fillStyle = mix(t.ink, t.ink, 0, 0.1);
        g.fillRect(m.l, Y(n2), w - m.l - m.r, Y(0) - Y(n2));
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h - m.b);
        g.lineTo(w - m.r, h - m.b);
        g.stroke();
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        for (let d = 0; d <= DM; d += 2) g.fillText(String(d), X(d), h - m.b + 14);
        g.textAlign = 'right';
        g.fillText('more flexible →', w - m.r, h - 3);
        g.fillText(f2(ym), m.l - 4, Y(ym) + 4);
        g.fillText('0', m.l - 4, Y(0) + 4);
        g.save();
        g.beginPath();
        g.rect(m.l - 2, y0, w - m.l - m.r + 4, h - y0 - m.b + 2);
        g.clip();
        [
          [(d) => B2[d], t.ok, 2.4],
          [(d) => VA[d], t.accent, 2.4],
          [(d) => tot(d), t.ink, 3.2],
        ].forEach((q) => {
          g.strokeStyle = q[1];
          g.lineWidth = q[2];
          g.beginPath();
          for (let d = 0; d <= DM; d++) d ? g.lineTo(X(d), Y(q[0](d))) : g.moveTo(X(d), Y(q[0](d)));
          g.stroke();
        });
        g.setLineDash([5, 4]);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.4;
        g.beginPath();
        g.moveTo(m.l, Y(n2));
        g.lineTo(w - m.r, Y(n2));
        g.stroke();
        g.setLineDash([]);
        g.restore();
        g.strokeStyle = t.gold;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(X(S.d), y0 + m.t);
        g.lineTo(X(S.d), h - m.b);
        g.stroke();
        g.fillStyle = t.gold;
        g.beginPath();
        g.arc(X(S.d), Y(tot(S.d)), 6, 0, 7);
        g.fill();
        g.strokeStyle = t.ink;
        g.lineWidth = 1.6;
        g.beginPath();
        g.arc(X(best), Y(tot(best)), 10, 0, 7);
        g.stroke();
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('noise floor σ² = ' + f3(n2), m.l + 6, Math.min(h - m.b - 4, Y(n2) + 13));
        g.fillStyle = t.ink;
        g.fillText('black: total', X(4.3), y0 + m.t + 2);
        g.fillStyle = t.ok;
        g.fillText('red: bias²', X(4.3), y0 + m.t + 16);
        g.fillStyle = t.accent;
        g.fillText('green: variance', X(4.3), y0 + m.t + 30);
        g.fillStyle = t.ink;
        g.fillText('ring: lowest total', X(4.3), y0 + m.t + 44);
      }
      live.textContent = msg;
      root
        .querySelectorAll('#bv-steps li')
        .forEach((e, i) => e.setAttribute('aria-current', i === cu ? 'true' : 'false'));
      {
        const x = tp.v[0],
          gi = Math.round(((x + 1) / 2) * (NG - 1)),
          bb = avg[gi] - f(x);
        tp.out.textContent =
          'At x = ' +
          f2(x) +
          ' the truth is ' +
          f2(f(x)) +
          ' and the average fit is ' +
          f2(avg[gi]) +
          ', so the bias is ' +
          f2(bb) +
          '. Fits from different batches vary around that by ±' +
          f2(sd[gi]) +
          ', and a new measurement there varies by ±' +
          f2(S.s) +
          ' whatever the model. Expected squared error: ' +
          f3(bb * bb) +
          ' + ' +
          f3(sd[gi] * sd[gi]) +
          ' + ' +
          f3(S.s * S.s) +
          ' = ' +
          f3(bb * bb + sd[gi] * sd[gi] + S.s * S.s) +
          '.';
      }
      setR(root, 'b', f3(B2[S.d]));
      setR(root, 'v', f3(VA[S.d]));
      setR(root, 's', f3(S.s * S.s));
      setR(root, 't', f3(tot(S.d)));
      setR(root, 'best', 'degree ' + best);
      setR(root, 'k', String(k));
    };
    tp.on = () => draw();
    const redo = () => {
      comp();
      reset();
      draw();
    };
    design();
    comp();
    reset();
    for (let i = 0; i < 12; i++) add();
    const anim = Anim(() => {
      fr++;
      if (sweep) {
        if (fr % 34) return;
        if (sweep.d > DM) {
          sweep = null;
          pb.stop();
          S.d = best;
          db.set(best);
          comp();
          reset();
          msg = 'Sweep finished. The lowest total is at degree ' + best + ': ' + verdict();
          draw();
          return;
        }
        S.d = sweep.d++;
        db.set(S.d);
        comp();
        chk.rebase(0);
        reset();
        for (let i = 0; i < 10 && chk.remaining() > 0; i++) add();
        msg =
          'Degree ' +
          S.d +
          ': bias² ' +
          f3(B2[S.d]) +
          ', variance ' +
          f3(VA[S.d]) +
          ', noise ' +
          f3(S.s * S.s) +
          ', total ' +
          f3(tot(S.d)) +
          '.';
        draw();
        chk();
        return;
      }
      if (fr % (k < 6 ? 40 : k < 20 ? 12 : 4)) return;
      add();
      draw();
      if (conv()) {
        pb.stop();
        msg +=
          ' The measured parts now agree with the exact ones, so there is nothing more to learn from further batches. Stopped after ' +
          k +
          ' batches.';
        draw();
        return;
      }
      chk();
    });
    pb = playBtn($('#bv-play', root), anim);
    chk = runCtl(pb, root, 'bv', {
      done: () => !sweep && conv(),
      restart: () => reset(),
      count: () => k,
      what: 'batches',
    });
    $('#bv-step', root).onclick = () => {
      pb.stop();
      sweep = null;
      add();
      draw();
    };
    $('#bv-reset', root).onclick = () => {
      pb.stop();
      sweep = null;
      reset();
      draw();
    };
    $('#bv-sweep', root).onclick = () => {
      pb.stop();
      sweep = { d: 0 };
      pb.start();
    };
    const db = bind(
        root,
        'bv-d',
        (v) => {
          S.d = v;
          pb.stop();
          sweep = null;
          redo();
        },
        (v) => String(v),
      ),
      sb = bind(root, 'bv-s', (v) => {
        S.s = v;
        pb.stop();
        sweep = null;
        redo();
      });
    bind(
      root,
      'bv-n',
      (v) => {
        S.n = v;
        pb.stop();
        sweep = null;
        design();
        redo();
      },
      (v) => String(v),
    );
    $('#bv-f', root).onchange = (e) => {
      S.f = e.target.value;
      pb.stop();
      sweep = null;
      redo();
    };
    $('#bv-pre', root).addEventListener('click', (e) => {
      const v = e.target.dataset && e.target.dataset.v;
      if (!v) return;
      pb.stop();
      sweep = null;
      const was = best;
      if (v === 'quiet') {
        S.s = 0.12;
        sb.set(S.s);
      }
      if (v === 'loud') {
        S.s = 1;
        sb.set(S.s);
      }
      comp();
      if (v === 'simple') S.d = Math.min(1, Math.max(0, best - 2));
      if (v === 'flex') S.d = 11;
      if (v === 'best' || v === 'quiet' || v === 'loud') S.d = best;
      db.set(S.d);
      comp();
      reset();
      for (let i = 0; i < 12; i++) add();
      if (v === 'loud' || v === 'quiet')
        msg =
          'Noise set to σ = ' +
          f2(S.s) +
          '. The floor is now σ² = ' +
          f3(S.s * S.s) +
          ', and the variance curve scaled with it. The best flexibility ' +
          (best === was ? 'stayed at' : 'moved from degree ' + was + ' to') +
          ' degree ' +
          best +
          '. ' +
          verdict();
      cap.textContent = msg.length > 40 && (v === 'loud' || v === 'quiet') ? msg : verdict();
      draw();
    });
    return {
      draw,
      stop: () => {
        pb.stop();
        sweep = null;
      },
      auto: () => {
        sweep = null;
        reset();
        draw();
        pb.start();
      },
    };
  },
};
