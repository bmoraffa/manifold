/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  readThemeColors,
  bind,
  clamp,
  clipPlot,
  f2,
  f3,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  plot,
  randn,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#bv-c', root),
      cr = $('#bv-r', root),
      live = $('#bv-live', root),
      cap = $('#bv-cap', root),
      settings = { d: 5, s: 0.3, n: 25, f: 'wave' },
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
      varianceByDegree = [],
      Hg = [],
      avg = [],
      sd = [],
      selectedSmoother = null,
      fits = [],
      sum = null,
      sq = null,
      k = 0,
      last = null,
      msg = '',
      cu = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
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
      const r = rng(31 + settings.n);
      xs = [];
      for (let i = 0; i < settings.n; i++)
        xs.push(clamp(-1 + (2 * (i + 0.5)) / settings.n + ((r() - 0.5) * 1.2) / settings.n, -1, 1));
    };
    const computeBiasVariance = () => {
      const f = FN[settings.f];
      fx = xs.map(f);
      B2 = [];
      varianceByDegree = [];
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
        varianceByDegree.push((settings.s * settings.s * v) / NG);
        if (d === settings.d) {
          selectedSmoother = sm;
          avg = sm.H.map((h) => {
            let m = 0;
            for (let i = 0; i < xs.length; i++) m += h[i] * fx[i];
            return m;
          });
          sd = sm.H.map((h) => {
            let h2 = 0;
            for (let i = 0; i < xs.length; i++) h2 += h[i] * h[i];
            return settings.s * Math.sqrt(h2);
          });
        }
      }
      best = 0;
      for (let d = 1; d <= DM; d++)
        if (B2[d] + varianceByDegree[d] < B2[best] + varianceByDegree[best]) best = d;
    };
    const tot = (d) => B2[d] + varianceByDegree[d] + settings.s * settings.s;
    const verdict = () => {
      const b = B2[settings.d],
        v = varianceByDegree[settings.d],
        n2 = settings.s * settings.s,
        T = b + v + n2;
      return (
        (settings.d < best
          ? 'This model is too stiff for this world: bias² is ' +
            ((100 * b) / T).toFixed(0) +
            '% of the error. The gold average misses the truth however many batches you draw.'
          : settings.d > best
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
      const f = FN[settings.f];
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
        y = xs.map((x, i) => fx[i] + settings.s * randn()),
        p = selectedSmoother.p,
        co = [];
      for (let a = 0; a < p; a++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += selectedSmoother.M[a][i] * y[i];
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
            f3(B2[settings.d]) +
            ' and ' +
            f3(varianceByDegree[settings.d]) +
            '.'
          : '');
    };
    const conv = () => {
      const e = emp();
      if (k >= 400) return true;
      if (!e || k < 40) return false;
      const T = B2[settings.d] + varianceByDegree[settings.d];
      return (
        Math.abs(e.b + e.v - T) < 0.04 * T + 2e-4 &&
        Math.abs(e.v - varianceByDegree[settings.d]) < 0.06 * varianceByDegree[settings.d] + 2e-4
      );
    };
    const tp = testPoint(root, 'bv', ['Your x'], [0.4], [[-1, 1]]);
    const draw = () => {
      const colors = readThemeColors(),
        f = FN[settings.f];
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = plot(ctx, w, h, [-1, 1], [-2.7, 2.7], colors, {
            xstep: 0.5,
            yticks: [-2, 0, 2],
            xlabel: 'x',
          });
        clipPlot(ctx, P);
        ctx.beginPath();
        for (let i = 0; i < NG; i++) {
          const x = gx(i);
          i
            ? ctx.lineTo(P.X(x), P.Y(f(x) + settings.s))
            : ctx.moveTo(P.X(x), P.Y(f(x) + settings.s));
        }
        for (let i = NG - 1; i >= 0; i--) {
          const x = gx(i);
          ctx.lineTo(P.X(x), P.Y(f(x) - settings.s));
        }
        ctx.closePath();
        ctx.fillStyle = mix(colors.ink, colors.ink, 0, 0.1);
        ctx.fill();
        ctx.beginPath();
        for (let i = 0; i < NG; i++)
          i
            ? ctx.lineTo(P.X(gx(i)), P.Y(avg[i] + sd[i]))
            : ctx.moveTo(P.X(gx(i)), P.Y(avg[i] + sd[i]));
        for (let i = NG - 1; i >= 0; i--) ctx.lineTo(P.X(gx(i)), P.Y(avg[i] - sd[i]));
        ctx.closePath();
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.16);
        ctx.fill();
        ctx.beginPath();
        for (let i = 0; i < NG; i++)
          i ? ctx.lineTo(P.X(gx(i)), P.Y(avg[i])) : ctx.moveTo(P.X(gx(i)), P.Y(avg[i]));
        for (let i = NG - 1; i >= 0; i--) ctx.lineTo(P.X(gx(i)), P.Y(f(gx(i))));
        ctx.closePath();
        ctx.fillStyle = mix(colors.ok, colors.ok, 0, 0.3);
        ctx.fill();
        ctx.lineWidth = 1;
        fits.forEach((ft, q) => {
          ctx.strokeStyle = mix(colors.accent, colors.accent, 0, q === fits.length - 1 ? 0 : 0.3);
          ctx.beginPath();
          for (let i = 0; i < NG; i++)
            i ? ctx.lineTo(P.X(gx(i)), P.Y(ft[i])) : ctx.moveTo(P.X(gx(i)), P.Y(ft[i]));
          ctx.stroke();
        });
        ctx.setLineDash([7, 5]);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let i = 0; i < NG; i++)
          i ? ctx.lineTo(P.X(gx(i)), P.Y(f(gx(i)))) : ctx.moveTo(P.X(gx(i)), P.Y(f(gx(i))));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 3.4;
        ctx.beginPath();
        for (let i = 0; i < NG; i++)
          i ? ctx.lineTo(P.X(gx(i)), P.Y(avg[i])) : ctx.moveTo(P.X(gx(i)), P.Y(avg[i]));
        ctx.stroke();
        if (last) {
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          for (let i = 0; i < NG; i++)
            i ? ctx.lineTo(P.X(gx(i)), P.Y(last.ft[i])) : ctx.moveTo(P.X(gx(i)), P.Y(last.ft[i]));
          ctx.stroke();
          ctx.fillStyle = colors.ink;
          xs.forEach((x, i) => {
            ctx.beginPath();
            ctx.arc(P.X(x), P.Y(last.y[i]), 3, 0, 7);
            ctx.fill();
          });
        }
        {
          const x = tp.v[0],
            gi = Math.round(((x + 1) / 2) * (NG - 1)),
            px = P.X(x);
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(px, P.Y(-2.7));
          ctx.lineTo(px, P.Y(2.7));
          ctx.stroke();
          markTP(ctx, px, P.Y(avg[gi]), colors, false);
        }
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        let ly = P.m.t + 8;
        [
          [mix(colors.ink, colors.ink, 0, 0.45), 'grey band: truth ± σ, the irreducible noise'],
          [colors.ok, 'red area: bias, the gap between gold and truth'],
          [colors.accent, 'green band and thin lines: variance'],
          [colors.cut, 'gold: the average of all possible fits'],
        ].forEach((q) => {
          ctx.fillStyle = q[0];
          ctx.fillText(q[1], P.m.l + 8, ly);
          ly += 14;
        });
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          b = B2[settings.d],
          v = varianceByDegree[settings.d],
          n2 = settings.s * settings.s,
          Tt = b + v + n2,
          e = emp(),
          x0 = 12,
          bw = w - 24,
          mx = Math.max(Tt, e ? e.b + e.v + n2 : 0) * 1.02 || 1;
        ctx.font = MONO;
        const bar = (y, hh, parts, lab) => {
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(lab, x0, y - 5);
          let x = x0;
          parts.forEach((q) => {
            const ww = (q[0] / mx) * bw;
            ctx.fillStyle = q[1];
            ctx.fillRect(x, y, ww, hh);
            if (ww > 58) {
              ctx.fillStyle = colors.bg;
              ctx.font = SANS;
              ctx.textAlign = 'center';
              ctx.fillText(q[2] + ' ' + f3(q[0]), x + ww / 2, y + hh / 2 + 5);
              ctx.font = MONO;
            }
            x += ww;
          });
        };
        bar(
          20,
          34,
          [
            [b, colors.ok, 'bias²'],
            [v, colors.accent, 'variance'],
            [n2, mix(colors.ink, colors.ink, 0, 0.5), 'noise'],
          ],
          'exact: bias² + variance + noise = ' + f3(Tt),
        );
        if (e)
          bar(
            78,
            16,
            [
              [e.b, colors.ok, ''],
              [e.v, colors.accent, ''],
              [n2, mix(colors.ink, colors.ink, 0, 0.5), ''],
            ],
            'measured from the ' + k + ' batches drawn so far = ' + f3(e.b + e.v + n2),
          );
        else {
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText('draw batches to measure it yourself', x0, 88);
        }
        const y0 = 112,
          m = { l: 40, r: 12, t: 14, b: 26 },
          ym = Math.max(n2 * 2.6, (B2[0] + n2) * 1.15, tot(best) * 2.2),
          X = (d) => m.l + (d / DM) * (w - m.l - m.r),
          Y = (u) => h - m.b - (Math.min(u, ym * 1.2) / ym) * (h - y0 - m.t - m.b);
        ctx.fillStyle = mix(colors.ink, colors.ink, 0, 0.1);
        ctx.fillRect(m.l, Y(n2), w - m.l - m.r, Y(0) - Y(n2));
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, h - m.b);
        ctx.lineTo(w - m.r, h - m.b);
        ctx.stroke();
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        for (let d = 0; d <= DM; d += 2) ctx.fillText(String(d), X(d), h - m.b + 14);
        ctx.textAlign = 'right';
        ctx.fillText('more flexible →', w - m.r, h - 3);
        ctx.fillText(f2(ym), m.l - 4, Y(ym) + 4);
        ctx.fillText('0', m.l - 4, Y(0) + 4);
        ctx.save();
        ctx.beginPath();
        ctx.rect(m.l - 2, y0, w - m.l - m.r + 4, h - y0 - m.b + 2);
        ctx.clip();
        [
          [(d) => B2[d], colors.ok, 2.4],
          [(d) => varianceByDegree[d], colors.accent, 2.4],
          [(d) => tot(d), colors.ink, 3.2],
        ].forEach((q) => {
          ctx.strokeStyle = q[1];
          ctx.lineWidth = q[2];
          ctx.beginPath();
          for (let d = 0; d <= DM; d++)
            d ? ctx.lineTo(X(d), Y(q[0](d))) : ctx.moveTo(X(d), Y(q[0](d)));
          ctx.stroke();
        });
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(m.l, Y(n2));
        ctx.lineTo(w - m.r, Y(n2));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(X(settings.d), y0 + m.t);
        ctx.lineTo(X(settings.d), h - m.b);
        ctx.stroke();
        ctx.fillStyle = colors.gold;
        ctx.beginPath();
        ctx.arc(X(settings.d), Y(tot(settings.d)), 6, 0, 7);
        ctx.fill();
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(X(best), Y(tot(best)), 10, 0, 7);
        ctx.stroke();
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.muted;
        ctx.fillText('noise floor σ² = ' + f3(n2), m.l + 6, Math.min(h - m.b - 4, Y(n2) + 13));
        ctx.fillStyle = colors.ink;
        ctx.fillText('black: total', X(4.3), y0 + m.t + 2);
        ctx.fillStyle = colors.ok;
        ctx.fillText('red: bias²', X(4.3), y0 + m.t + 16);
        ctx.fillStyle = colors.accent;
        ctx.fillText('green: variance', X(4.3), y0 + m.t + 30);
        ctx.fillStyle = colors.ink;
        ctx.fillText('ring: lowest total', X(4.3), y0 + m.t + 44);
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
          f2(settings.s) +
          ' whatever the model. Expected squared error: ' +
          f3(bb * bb) +
          ' + ' +
          f3(sd[gi] * sd[gi]) +
          ' + ' +
          f3(settings.s * settings.s) +
          ' = ' +
          f3(bb * bb + sd[gi] * sd[gi] + settings.s * settings.s) +
          '.';
      }
      setReadout(root, 'b', f3(B2[settings.d]));
      setReadout(root, 'v', f3(varianceByDegree[settings.d]));
      setReadout(root, 's', f3(settings.s * settings.s));
      setReadout(root, 't', f3(tot(settings.d)));
      setReadout(root, 'best', 'degree ' + best);
      setReadout(root, 'k', String(k));
    };
    tp.on = () => draw();
    const redo = () => {
      computeBiasVariance();
      reset();
      draw();
    };
    design();
    computeBiasVariance();
    reset();
    for (let i = 0; i < 12; i++) add();
    const anim = Anim(() => {
      frameCount++;
      if (sweep) {
        if (frameCount % 34) return;
        if (sweep.d > DM) {
          sweep = null;
          playback.stop();
          settings.d = best;
          db.set(best);
          computeBiasVariance();
          reset();
          msg = 'Sweep finished. The lowest total is at degree ' + best + ': ' + verdict();
          draw();
          return;
        }
        settings.d = sweep.d++;
        db.set(settings.d);
        computeBiasVariance();
        checkRunStop.rebase(0);
        reset();
        for (let i = 0; i < 10 && checkRunStop.remaining() > 0; i++) add();
        msg =
          'Degree ' +
          settings.d +
          ': bias² ' +
          f3(B2[settings.d]) +
          ', variance ' +
          f3(varianceByDegree[settings.d]) +
          ', noise ' +
          f3(settings.s * settings.s) +
          ', total ' +
          f3(tot(settings.d)) +
          '.';
        draw();
        checkRunStop();
        return;
      }
      if (frameCount % (k < 6 ? 40 : k < 20 ? 12 : 4)) return;
      add();
      draw();
      if (conv()) {
        playback.stop();
        msg +=
          ' The measured parts now agree with the exact ones, so there is nothing more to learn from further batches. Stopped after ' +
          k +
          ' batches.';
        draw();
        return;
      }
      checkRunStop();
    });
    playback = playBtn($('#bv-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'bv', {
      done: () => !sweep && conv(),
      restart: () => reset(),
      count: () => k,
      what: 'batches',
    });
    $('#bv-step', root).onclick = () => {
      playback.stop();
      sweep = null;
      add();
      draw();
    };
    $('#bv-reset', root).onclick = () => {
      playback.stop();
      sweep = null;
      reset();
      draw();
    };
    $('#bv-sweep', root).onclick = () => {
      playback.stop();
      sweep = { d: 0 };
      playback.start();
    };
    const db = bind(
        root,
        'bv-d',
        (v) => {
          settings.d = v;
          playback.stop();
          sweep = null;
          redo();
        },
        (v) => String(v),
      ),
      sb = bind(root, 'bv-s', (v) => {
        settings.s = v;
        playback.stop();
        sweep = null;
        redo();
      });
    bind(
      root,
      'bv-n',
      (v) => {
        settings.n = v;
        playback.stop();
        sweep = null;
        design();
        redo();
      },
      (v) => String(v),
    );
    $('#bv-f', root).onchange = (e) => {
      settings.f = e.target.value;
      playback.stop();
      sweep = null;
      redo();
    };
    $('#bv-pre', root).addEventListener('click', (e) => {
      const v = e.target.dataset && e.target.dataset.v;
      if (!v) return;
      playback.stop();
      sweep = null;
      const was = best;
      if (v === 'quiet') {
        settings.s = 0.12;
        sb.set(settings.s);
      }
      if (v === 'loud') {
        settings.s = 1;
        sb.set(settings.s);
      }
      computeBiasVariance();
      if (v === 'simple') settings.d = Math.min(1, Math.max(0, best - 2));
      if (v === 'flex') settings.d = 11;
      if (v === 'best' || v === 'quiet' || v === 'loud') settings.d = best;
      db.set(settings.d);
      computeBiasVariance();
      reset();
      for (let i = 0; i < 12; i++) add();
      if (v === 'loud' || v === 'quiet')
        msg =
          'Noise set to σ = ' +
          f2(settings.s) +
          '. The floor is now σ² = ' +
          f3(settings.s * settings.s) +
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
        playback.stop();
        sweep = null;
      },
      auto: () => {
        sweep = null;
        reset();
        draw();
        playback.start();
      },
    };
  },
};
