/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import {
  jacobi,
  segBind,
  highlightLessonStep,
  drawSeriesChart,
} from '../../unsupervised/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#lo-c', root),
      cr = $('#lo-r', root),
      live = $('#lo-live', root),
      settings = { m: 'full', r: 2, task: 'spiral' },
      H = 32,
      PRE = 500,
      MAXA = 1200,
      RN = randnFrom(Math.random);
    let W,
      b,
      W0,
      b0,
      A,
      Bm,
      ps,
      ts = 0,
      as = 0,
      phase = 'pre',
      sourceData,
      targetData,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      ACN = [],
      sourceAccuracyHistory = [],
      PL = [],
      accPre = NaN,
      best = 0,
      bestAt = 0,
      fin = false,
      dW = null,
      SV = [],
      gridA = null,
      gridB = null,
      lastG = -99,
      ntr = 0;
    const mkA = (n) => {
      const X = [],
        y = [];
      for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI,
          k = i % 2;
        X.push(
          k
            ? [1.25 * Math.cos(a) - 0.6 + 0.1 * RN(), -0.3 + 1.2 * Math.sin(a) + 0.1 * RN()]
            : [0.6 - 1.25 * Math.cos(a) + 0.1 * RN(), 0.3 - 1.2 * Math.sin(a) + 0.1 * RN()],
        );
        y.push(k);
      }
      return { X, y };
    };
    const TB = {
        spiral: (n) => {
          const X = [],
            y = [];
          for (let i = 0; i < n; i++) {
            const k = i % 2,
              u = Math.random(),
              a = u * 3.2 * Math.PI + k * Math.PI,
              r = 0.25 + 1.45 * u;
            X.push([r * Math.cos(a) + 0.05 * RN(), 0.8 * r * Math.sin(a) + 0.05 * RN()]);
            y.push(k);
          }
          return { X, y };
        },
        stripes: (n) => {
          const X = [],
            y = [];
          for (let i = 0; i < n; i++) {
            const x = (Math.random() * 2 - 1) * 2.1,
              yy = (Math.random() * 2 - 1) * 1.6;
            X.push([x, yy]);
            y.push(Math.floor((x + yy * 0.6 + 3.2) / 0.9) % 2);
          }
          return { X, y };
        },
        ring: (n) => {
          const X = [],
            y = [];
          for (let i = 0; i < n; i++) {
            const k = i % 2,
              a = Math.random() * 2 * Math.PI;
            X.push(
              k
                ? [1.45 * Math.cos(a) + 0.12 * RN(), 1.3 * Math.sin(a) + 0.12 * RN()]
                : [0.33 * RN(), 0.33 * RN()],
            );
            y.push(k);
          }
          return { X, y };
        },
      },
      TBN = {
        spiral: 'tell the two arms of a spiral apart',
        stripes: 'tell diagonal stripes apart',
        ring: 'tell a central blob from the ring around it',
      },
      mkB = (n) => TB[settings.task](n);
    const fwd = (X, n, lora) => {
      let h = AD.input(X, [n, 2]);
      for (let l = 0; l < 3; l++) {
        let z = AD.mm(h, W[l]);
        if (lora) z = AD.add(z, AD.mm(AD.mm(h, A[l]), Bm[l]));
        z = AD.addb(z, b[l]);
        h = l < 2 ? AD.relu(z) : z;
      }
      return h;
    };
    const useL = () => phase !== 'pre' && settings.m === 'lora';
    const predict = (P, n) => {
      const o = fwd(P, n, useL());
      AD.clear();
      return o.v;
    };
    const acc = (D) => {
      const n = D.y.length,
        o = predict(Float64Array.from(D.X.flat()), n);
      let ok = 0;
      for (let i = 0; i < n; i++) if ((o[2 * i + 1] > o[2 * i] ? 1 : 0) === D.y[i]) ok++;
      return ok / n;
    };
    const accWith = (D, Wx, bx) => {
      const sw = W,
        sb = b,
        sp = phase;
      W = Wx;
      b = bx;
      phase = 'pre';
      const a = acc(D);
      W = sw;
      b = sb;
      phase = sp;
      return a;
    };
    const cp = (a) =>
      a.map((p) => {
        const q = AD.param(p.sh, 0, RN);
        q.v.set(p.v);
        return q;
      });
    const build = () => {
      const rn = randnFrom(rng(run * 11 + 3));
      W = [
        AD.param([2, H], 1, rn),
        AD.param([H, H], Math.sqrt(2 / H), rn),
        AD.param([H, 2], Math.sqrt(1 / H), rn),
      ];
      b = [AD.param([H], 0, rn), AD.param([H], 0, rn), AD.param([2], 0, rn)];
      sourceData = mkA(300);
      targetData = mkB(400);
      ts = 0;
      as = 0;
      phase = 'pre';
      PL = [];
      ACN = [];
      sourceAccuracyHistory = [];
      accPre = NaN;
      fin = false;
      dW = null;
      SV = [];
      activeStepIndex = 0;
      lastG = -99;
      ps = W.concat(b);
      ntr = 1218;
      msg =
        'A network with random weights. First it learns the old task, two interleaved moons, from scratch. Press Run.';
    };
    const startAdapt = () => {
      W = cp(W0);
      b = cp(b0);
      const rn = randnFrom(rng(run * 7 + settings.r));
      A = [
        [2, settings.r],
        [H, settings.r],
        [H, settings.r],
      ].map((s) => AD.param(s, 1 / Math.sqrt(s[0]), rn));
      Bm = [
        [settings.r, H],
        [settings.r, H],
        [settings.r, 2],
      ].map((s) => AD.param(s, 0, rn));
      ps =
        settings.m === 'full' ? W.concat(b) : settings.m === 'lora' ? A.concat(Bm) : [W[2], b[2]];
      ntr = settings.m === 'full' ? 1218 : settings.m === 'lora' ? 132 * settings.r : 66;
      phase = 'adapt';
      as = 0;
      ACN = [acc(targetData)];
      sourceAccuracyHistory = [acc(sourceData)];
      best = 0;
      bestAt = 0;
      fin = false;
      activeStepIndex = 1;
      lastG = -99;
      delta();
      msg =
        'Pretraining is finished: ' +
        pct(accPre) +
        ' on the old task. A copy of those weights is kept. Now the new task: ' +
        TBN[settings.task] +
        '. ' +
        (settings.m === 'full'
          ? 'All 1,218 weights are free to change.'
          : settings.m === 'lora'
            ? 'The 1,218 weights are frozen. Only two thin matrices per layer are trained: ' +
              ntr +
              ' numbers.'
            : 'Only the last layer’s 66 weights are free to change.');
    };
    const delta = () => {
      dW = new Float64Array(H * H);
      if (phase === 'pre') return;
      if (settings.m === 'lora') {
        const a = A[1].v,
          bb = Bm[1].v,
          r = settings.r;
        for (let i = 0; i < H; i++)
          for (let j = 0; j < H; j++) {
            let s = 0;
            for (let k = 0; k < r; k++) s += a[i * r + k] * bb[k * H + j];
            dW[i * H + j] = s;
          }
      } else for (let i = 0; i < H * H; i++) dW[i] = W[1].v[i] - W0[1].v[i];
      const G = [];
      for (let i = 0; i < H; i++) {
        G.push(new Array(H).fill(0));
      }
      for (let i = 0; i < H; i++)
        for (let j = i; j < H; j++) {
          let s = 0;
          for (let k = 0; k < H; k++) s += dW[k * H + i] * dW[k * H + j];
          G[i][j] = G[j][i] = s;
        }
      SV = jacobi(G).val.map((v) => Math.sqrt(Math.max(0, v)));
    };
    const step1 = (D, lora) => {
      const Bn = 64,
        X = new Float64Array(2 * Bn),
        tg = new Int32Array(Bn);
      for (let i = 0; i < Bn; i++) {
        const j = Math.floor(Math.random() * D.y.length);
        X[2 * i] = D.X[j][0];
        X[2 * i + 1] = D.X[j][1];
        tg[i] = D.y[j];
      }
      const l = AD.ce(fwd(X, Bn, lora), tg, 0);
      AD.backward(l);
      return l.v[0];
    };
    const zero = () => {
      for (const p of W.concat(b)) if (!ps.includes(p)) p.g.fill(0);
    };
    const train = (k) => {
      if (phase === 'pre') {
        for (let q = 0; q < k && ts < PRE; q++) {
          PL.push(step1(sourceData, false));
          AD.adam(ps, 0.01, ++ts, 1);
        }
        activeStepIndex = 0;
        const a = acc(sourceData);
        sourceAccuracyHistory = [a];
        msg =
          'Pretraining step ' +
          ts +
          ' of ' +
          PRE +
          ' on the old task: accuracy ' +
          pct(a) +
          '. All 1,218 weights are being trained.';
        if (ts >= PRE) {
          accPre = a;
          W0 = cp(W);
          b0 = cp(b);
          startAdapt();
        }
        return;
      }
      if (fin) return;
      for (let q = 0; q < k && as < MAXA && !fin; q++) {
        step1(targetData, settings.m === 'lora');
        zero();
        AD.adam(ps, settings.m === 'lora' ? 0.01 : 0.005, ++as, 1);
        if (as % 5 === 0) {
          const a = acc(targetData);
          ACN.push(a);
          sourceAccuracyHistory.push(acc(sourceData));
          if (a > best + 0.004) {
            best = a;
            bestAt = as;
          }
          if (as - bestAt >= 200 || as >= MAXA) fin = true;
        }
      }
      delta();
      activeStepIndex = fin ? 4 : 3;
      const an = ACN[ACN.length - 1],
        ao = sourceAccuracyHistory[sourceAccuracyHistory.length - 1];
      msg =
        'Adaptation step ' +
        as +
        ': ' +
        ntr +
        ' numbers were adjusted on 64 examples of the new task. Accuracy is ' +
        pct(an) +
        ' on the new task and ' +
        pct(ao) +
        ' on the old one.';
      if (fin) {
        const e2 = SV.length
          ? (SV[0] ** 2 + SV[1] ** 2) / (SV.reduce((s, v) => s + v * v, 0) || 1)
          : NaN;
        msg =
          'Finished after ' +
          as +
          ' steps: the accuracy on the new task stopped improving at ' +
          pct(an) +
          '. ' +
          ntr +
          ' numbers were trained (' +
          pct(ntr / 1218) +
          ' of the network). Accuracy on the old task fell from ' +
          pct(accPre) +
          ' to ' +
          pct(ao) +
          '. ' +
          (settings.m === 'full'
            ? 'The original weights have been overwritten: getting the old skill back needs the saved copy, a whole second network.'
            : settings.m === 'lora'
              ? 'The original weights were never touched: take the correction away and the old accuracy of ' +
                pct(accWith(sourceData, W0, b0)) +
                ' is back.'
              : 'Only the last layer changed, so restoring the old skill needs just the 66 old numbers.') +
          (isFinite(e2) && settings.m !== 'last'
            ? ' The two strongest directions hold ' +
              pct(e2) +
              ' of the change to the middle matrix.'
            : '');
      }
    };
    const gridFor = (D) => {
      const GW = 40,
        GH = 30,
        P = new Float64Array(2 * GW * GH);
      for (let r = 0; r < GH; r++)
        for (let q = 0; q < GW; q++) {
          P[2 * (r * GW + q)] = (-4 + (8 * (q + 0.5)) / GW) / 1.6;
          P[2 * (r * GW + q) + 1] = (3 - (6 * (r + 0.5)) / GH) / 1.6;
        }
      const o = predict(P, GW * GH),
        g = new Float32Array(GW * GH);
      for (let i = 0; i < GW * GH; i++) g[i] = 1 / (1 + Math.exp(o[2 * i] - o[2 * i + 1]));
      return g;
    };
    const panel = (g, t, x0, y0, w, h, D, G, lab) => {
      const GW = 40,
        GH = 30,
        s = Math.min(w / 8, h / 6),
        cx = x0 + w / 2,
        cy = y0 + h / 2,
        X = (v) => cx + v * 1.6 * s,
        Y = (v) => cy - v * 1.6 * s;
      g.save();
      g.beginPath();
      g.rect(x0, y0, w, h);
      g.clip();
      if (G)
        for (let r = 0; r < GH; r++)
          for (let q = 0; q < GW; q++) {
            const p = G[r * GW + q],
              col = p > 0.5 ? t.accent : t.cut,
              xa = Math.round(cx + (-4 + (8 * q) / GW) * s),
              xb = Math.round(cx + (-4 + (8 * (q + 1)) / GW) * s),
              ya = Math.round(cy - (3 - (6 * r) / GH) * s),
              yb = Math.round(cy - (3 - (6 * (r + 1)) / GH) * s);
            g.fillStyle = mix(col, col, 0, 0.08 + 0.3 * Math.abs(p - 0.5) * 2);
            g.fillRect(xa, ya, xb - xa, yb - ya);
          }
      const r = w > 300 ? 3 : 1.6;
      D.X.forEach((p, i) => {
        if (w < 300 && i % 2 === 0 && i > 150) return;
        g.beginPath();
        g.arc(X(p[0]), Y(p[1]), r, 0, 7);
        g.fillStyle = D.y[i] ? t.accent : t.cut;
        g.fill();
        if (w > 300) {
          g.strokeStyle = t.bg;
          g.lineWidth = 0.6;
          g.stroke();
        }
      });
      g.restore();
      if (lab) {
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(lab, x0 + 4, y0 + h + 13);
      }
    };
    const draw = () => {
      const colors = readThemeColors();
      if (Math.abs((phase === 'pre' ? ts : as) - lastG) >= 3 || fin) {
        gridA = gridFor(sourceData);
        gridB = phase === 'pre' ? null : gridFor(targetData);
        lastG = phase === 'pre' ? ts : as;
      }
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas);
        panel(
          ctx,
          colors,
          0,
          0,
          w,
          h - 4,
          phase === 'pre' ? sourceData : targetData,
          phase === 'pre' ? gridA : gridB,
          '',
        );
        ctx.font = MONO;
        ctx.fillStyle = colors.ink;
        ctx.textAlign = 'left';
        ctx.fillText(
          phase === 'pre'
            ? 'OLD TASK: which moon? (pretraining)'
            : 'NEW TASK: ' + TBN[settings.task],
          8,
          14,
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.46,
          pw = (w - 16) / 3,
          ph2 = h1 - 34;
        ctx.font = MONO;
        panel(
          ctx,
          colors,
          4,
          16,
          pw - 8,
          ph2,
          sourceData,
          gridA,
          'old task: ' +
            (sourceAccuracyHistory.length
              ? (100 * sourceAccuracyHistory[sourceAccuracyHistory.length - 1]).toFixed(0) + '%'
              : '–'),
        );
        {
          const x0 = 8 + pw,
            s = Math.floor(Math.min(pw - 10, ph2) / H),
            ox = x0 + (pw - s * H) / 2,
            oy = 16 + (ph2 - s * H) / 2,
            mx = dW ? Math.max(1e-9, ...Array.from(dW).map(Math.abs)) : 1;
          ctx.strokeStyle = colors.line;
          ctx.lineWidth = 1;
          ctx.strokeRect(ox - 0.5, oy - 0.5, s * H + 1, s * H + 1);
          if (dW && phase !== 'pre')
            for (let i = 0; i < H; i++)
              for (let j = 0; j < H; j++) {
                const v = dW[i * H + j] / mx;
                if (Math.abs(v) < 0.04) continue;
                const col = v > 0 ? colors.accent : colors.cut;
                ctx.fillStyle = mix(col, col, 0, Math.min(1, Math.abs(v) * 1.2));
                ctx.fillRect(ox + j * s, oy + i * s, s, s);
              }
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(
            phase === 'pre'
              ? 'change to W: none yet'
              : settings.m === 'last'
                ? 'middle W: unchanged'
                : 'change to middle W',
            x0 + 2,
            16 + ph2 + 13,
          );
        }
        {
          const x0 = 12 + 2 * pw,
            n = 12,
            bw = (pw - 16) / n,
            mx = Math.max(1e-9, ...SV);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x0, 16 + ph2);
          ctx.lineTo(x0 + pw - 12, 16 + ph2);
          ctx.stroke();
          if (phase !== 'pre')
            SV.slice(0, n).forEach((v, i) => {
              const bh = (v / mx) * (ph2 - 6);
              ctx.fillStyle =
                i < (settings.m === 'lora' ? settings.r : 2)
                  ? colors.gold
                  : mix(colors.accent, colors.accent, 0, 0.7);
              ctx.fillRect(x0 + i * bw + 1, 16 + ph2 - bh, bw - 2, bh);
            });
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText('its directions', x0, 16 + ph2 + 13);
        }
        const n = MAXA / 5 + 1,
          L1 = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, phase === 'pre' ? [] : ACN, {
            title: 'accuracy while adapting',
            n,
            ym: 1,
            y1: 0,
            col: colors.accent,
          });
        if (phase !== 'pre') {
          ctx.strokeStyle = colors.cut;
          ctx.lineWidth = 2;
          ctx.beginPath();
          sourceAccuracyHistory.forEach((v, i) =>
            i ? ctx.lineTo(L1.X(i), L1.Y(v)) : ctx.moveTo(L1.X(i), L1.Y(v)),
          );
          ctx.stroke();
          const k = ACN.length - 1,
            lx = Math.max(L1.X(k), L1.m.l + 70);
          ctx.textAlign = 'right';
          ctx.fillStyle = colors.accent;
          ctx.fillText('new task', lx, L1.Y(ACN[k]) + (ACN[k] > 0.8 ? 16 : -8));
          ctx.fillStyle = colors.cut;
          ctx.fillText('old task', lx, L1.Y(sourceAccuracyHistory[k]) - 9);
        } else {
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText('the curves start when the new task begins', w / 2, h1 + (h - h1) / 2);
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'lo', activeStepIndex);
      const e2 =
        SV.length && phase !== 'pre' && settings.m !== 'last'
          ? (SV[0] ** 2 + SV[1] ** 2) / (SV.reduce((s, v) => s + v * v, 0) || 1)
          : NaN;
      setReadout(
        root,
        'ph',
        phase === 'pre'
          ? ts
            ? 'Pretraining on the old task'
            : 'Not started'
          : fin
            ? 'Finished'
            : 'Adapting to the new task',
      );
      setReadout(root, 'st', phase === 'pre' ? '–' : String(as));
      setReadout(root, 'tp', ntr + ' of 1218 (' + pct(ntr / 1218) + ')');
      setReadout(root, 'an', ACN.length ? pct(ACN[ACN.length - 1]) : '–');
      setReadout(
        root,
        'ao',
        phase !== 'pre' && sourceAccuracyHistory.length
          ? pct(sourceAccuracyHistory[sourceAccuracyHistory.length - 1])
          : '–',
      );
      setReadout(root, 'au', isFinite(accPre) ? pct(accPre) : '–');
      setReadout(root, 'sv', isFinite(e2) ? pct(e2) : '–');
    };
    build();
    if (prefersReducedMotion) {
      train(PRE);
      while (!fin) train(20);
    }
    const anim = Anim(() => {
      frameCount++;
      train(phase === 'pre' ? 5 : Math.min(as < 40 ? 1 : 4, checkRunStop.remaining()));
      draw();
      checkRunStop();
    });
    playback = playBtn($('#lo-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'lo', {
      done: () => phase === 'adapt' && fin,
      restart: () => startAdapt(),
      count: () => as,
      what: 'adaptation steps',
    });
    const redo = () => {
      if (phase === 'pre') {
        draw();
        return;
      }
      playback.stop();
      startAdapt();
      draw();
      playback.start();
    };
    $('#lo-step', root).onclick = () => {
      playback.stop();
      if (phase === 'adapt' && fin) startAdapt();
      else train(25);
      draw();
    };
    $('#lo-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    segBind(root, 'lo-m', settings.m, (v) => {
      settings.m = v;
      redo();
    });
    $('#lo-task', root).onchange = (e) => {
      settings.task = e.target.value;
      targetData = mkB(400);
      redo();
    };
    bind(
      root,
      'lo-rank',
      (v) => {
        settings.r = +v;
        if (settings.m === 'lora') redo();
        else if (playback) draw();
      },
      (v) => String(v),
    );
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        run++;
        build();
        draw();
        playback.start();
      },
    };
  },
};
