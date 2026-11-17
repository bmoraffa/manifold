/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  SANS,
  readThemeColors,
  bind,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#tf-c', root),
      cr = $('#tf-r', root),
      cs = $('#tf-s', root),
      live = $('#tf-live', root),
      st = $('#tf-st', root),
      settings = { task: 'rev', text: 'cafeb', pos: 1 },
      n = 5,
      V = 8,
      d = 16,
      T2 = 2 * n,
      LT = 'abcdefgh';
    let P,
      ps,
      ts = 0,
      LL = [],
      AC = [],
      conv = false,
      good = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      demo = null;
    const build = () => {
      const q = randnFrom(rng(run * 11 + 5)),
        pm = (a, b, sd) => AD.param([a, b], sd, q);
      P = {
        E: pm(V + 1, d, 0.3),
        Pos: pm(T2, d, 0.3),
        Wq: pm(d, d, 0.3),
        Wk: pm(d, d, 0.3),
        Wv: pm(d, d, 0.3),
        W1: pm(d, 2 * d, 0.25),
        W2: pm(2 * d, d, 0.25),
        Wu: pm(d, V + 1, 0.3),
      };
      ps = Object.values(P);
      ts = 0;
      LL = [];
      AC = [];
      conv = false;
      good = 0;
      activeStepIndex = 0;
      refreshDemoPrediction();
      msg =
        'A Transformer with random weights. It has never seen the task. Its answer for “' +
        settings.text +
        '” is “' +
        demo.out +
        '”. Press Train.';
    };
    const fwd = (tok, tg) => {
      const X = AD.add(
          AD.gather(P.E, tok),
          AD.gather(
            P.Pos,
            tok.map((_, i) => i),
          ),
        ),
        A = AD.attn(AD.mmT(AD.mm(X, P.Wq), AD.mm(X, P.Wk)), 1 / Math.sqrt(d), true),
        Z = AD.mm(A, AD.mm(X, P.Wv)),
        R1 = AD.add(X, Z),
        R2 = AD.add(R1, AD.mm(AD.relu(AD.mm(R1, P.W1)), P.W2)),
        L = AD.mm(R2, P.Wu);
      return { X, A, Z, R2, L, loss: tg ? AD.ce(L, tg, n) : null };
    };
    const ans = (x) => (settings.task === 'rev' ? x.slice().reverse() : x.slice()),
      makeTrainingExample = () => {
        const x = Array.from({ length: n }, () => Math.floor(Math.random() * V)),
          y = ans(x),
          tg = new Array(T2).fill(0);
        for (let i = 0; i < n; i++) tg[n + i] = y[i];
        return { tok: x.concat([V], y.slice(0, n - 1)), tg };
      };
    const gen = (x) => {
      const tok = x.concat([V]);
      let o;
      for (let i = 0; i < n; i++) {
        o = fwd(tok);
        AD.clear();
        const r = (tok.length - 1) * (V + 1);
        let b = 0;
        for (let j = 0; j < V; j++) if (o.L.v[r + j] > o.L.v[r + b]) b = j;
        if (i < n - 1) tok.push(b);
        else tok.out = tok.slice(n + 1).concat([b]);
      }
      return { tok, o, out: tok.out };
    };
    const refreshDemoPrediction = () => {
      const x = [...settings.text].map((ch) => Math.max(0, LT.indexOf(ch)));
      while (x.length < n) x.push(0);
      const gq = gen(x),
        outputTokenProbs = [];
      for (let t = n; t < T2; t++) {
        const row = [];
        let mx = -1e30,
          z = 0;
        for (let j = 0; j <= V; j++) mx = Math.max(mx, gq.o.L.v[t * (V + 1) + j]);
        for (let j = 0; j <= V; j++) {
          const e = Math.exp(gq.o.L.v[t * (V + 1) + j] - mx);
          row.push(e);
          z += e;
        }
        outputTokenProbs.push(row.map((v) => v / z));
      }
      demo = {
        x,
        tok: gq.tok,
        A: gq.o.A.v,
        X: gq.o.X.v,
        Z: gq.o.Z.v,
        R2: gq.o.R2.v,
        pr: outputTokenProbs,
        outI: gq.out,
        out: gq.out.map((i) => LT[i]).join(''),
        want: ans(x)
          .map((i) => LT[i])
          .join(''),
      };
    };
    const evalAcc = () => {
      let ok = 0;
      for (let k = 0; k < 40; k++) {
        const x = Array.from({ length: n }, () => Math.floor(Math.random() * V));
        if (gen(x).out.join() === ans(x).join()) ok++;
      }
      return ok / 40;
    };
    const train = (k) => {
      for (let q = 0; q < k && !conv; q++) {
        let l = 0;
        for (let b = 0; b < 12; b++) {
          const s = makeTrainingExample(),
            o = fwd(s.tok, s.tg);
          l += o.loss.v[0];
          AD.backward(o.loss);
        }
        AD.adam(ps, 0.004, ++ts, 12);
        LL.push(l / 12);
        if (ts % 10 === 0) {
          const a = evalAcc();
          AC.push(a);
          good = a >= 0.99 ? good + 1 : 0;
          if (good >= 3 || ts >= 4000) conv = true;
        }
      }
      refreshDemoPrediction();
      activeStepIndex = conv ? 4 : 1 + (ts % 3);
      msg =
        'Step ' +
        ts +
        ': 12 fresh sequences went forward, the error was ' +
        f3(LL[ts - 1]) +
        ', and backpropagation adjusted all ' +
        ps.reduce((a, p) => a + p.v.length, 0) +
        ' weights. For “' +
        settings.text +
        '” it now writes “' +
        demo.out +
        '” (the right answer is “' +
        demo.want +
        '”).' +
        (conv
          ? ts >= 4000
            ? ' Reached the limit of 4000 steps.'
            : ' Converged: it gets every fresh sequence right, so training stops. Type your own letters.'
          : '');
    };
    const draw = () => {
      const colors = readThemeColors(),
        D = demo,
        pi = n + settings.pos - 1,
        lab = (i) => (i < n ? LT[D.x[i]] : i === n ? '▸' : LT[D.tok[i]]),
        u = 1;
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          bw = Math.min(46, (w - 20) / 11.6),
          x0 = (w - bw * 11.4) / 2,
          X = (i) => x0 + (i < n ? i : i === n ? i + 0.2 : i + 0.4) * bw + bw / 2,
          yw = h * 0.2,
          cell = Math.min((w - 140) / T2, (h * 0.62 - 30) / n),
          mx = (w - cell * T2) / 2 + 20,
          my = h * 0.4 + 6;
        ctx.font = SANS;
        for (let j = 0; j <= pi; j++) {
          const wt = D.A[pi * T2 + j];
          if (j === pi || wt < 0.02) continue;
          ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.2 + 0.8 * wt);
          ctx.lineWidth = 0.5 + 10 * wt;
          const xo = pi + 1 < T2 ? X(pi + 1) : X(T2 - 1) + bw;
          ctx.beginPath();
          ctx.moveTo(xo, yw - 15);
          ctx.quadraticCurveTo(
            (xo + X(j)) / 2,
            yw - 15 - Math.min(h * 0.14, Math.abs(xo - X(j)) * 0.3),
            X(j),
            yw - 15,
          );
          ctx.stroke();
        }
        for (let i = 0; i < T2 + 1; i++) {
          const isOut = i > n,
            x = i < T2 ? X(i) : X(T2 - 1) + bw,
            ch = i <= n ? lab(i) : LT[D.outI[i - n - 1]],
            ok = isOut && ch === D.want[i - n - 1],
            hi = i === pi + 1 || (i === pi && false);
          if (i === T2) void 0;
          ctx.fillStyle =
            i === pi + 1
              ? colors.gold
              : isOut
                ? mix(ok ? colors.accent : colors.ok, colors.bg, 0, 0.18)
                : colors.bg;
          ctx.strokeStyle = isOut ? (ok ? colors.accent : colors.ok) : colors.ink;
          ctx.lineWidth = i === pi + 1 ? 2.6 : 1.3;
          ctx.beginPath();
          ctx.rect(x - bw * 0.42, yw - 14, bw * 0.84, 28);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = i === pi + 1 ? colors.bg : colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText(ch, x, yw + 5);
        }
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        ctx.fillText('input', X(2), yw + 30);
        ctx.fillText('its output', X(n + 3), yw + 30);
        for (let r = 0; r < n; r++) {
          const ti = n + r;
          ctx.fillStyle = r === settings.pos - 1 ? colors.ink : colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText('output ' + (r + 1), mx - 6, my + (r + 0.65) * cell);
          for (let j = 0; j < T2; j++) {
            const v = j <= ti ? D.A[ti * T2 + j] : -1;
            ctx.fillStyle =
              v < 0
                ? mix(colors.ink, colors.ink, 0, 0.13)
                : mix(colors.bg, colors.gold, Math.pow(v, 0.6));
            ctx.fillRect(mx + j * cell, my + r * cell, cell - 1.5, cell - 1.5);
          }
          if (r === settings.pos - 1) {
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.6;
            ctx.strokeRect(mx - 1, my + r * cell - 1, cell * T2, cell);
          }
        }
        for (let j = 0; j < T2; j++) {
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText(lab(j), mx + (j + 0.5) * cell, my + n * cell + 13);
        }
        ctx.textAlign = 'left';
        ctx.fillText('looks at →   (grey: not yet written, hidden by the mask)', mx, my - 7);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          ha = h * 0.34,
          pr = D.pr[settings.pos - 1],
          bw = (w - 30) / (V + 1);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          'output ' +
            settings.pos +
            ': confidence in each letter (should be “' +
            D.want[settings.pos - 1] +
            '”)',
          8,
          13,
        );
        for (let j = 0; j <= V; j++) {
          const x = 15 + j * bw,
            p = pr[j],
            top = ha - 22 - p * (ha - 48);
          ctx.fillStyle = j < V && LT[j] === D.want[settings.pos - 1] ? colors.accent : colors.gold;
          ctx.fillRect(x + bw * 0.15, top, bw * 0.7, ha - 22 - top);
          ctx.font = SANS;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText(j < V ? LT[j] : '▸', x + bw / 2, ha - 6);
          ctx.font = MONO;
          if (p > 0.04) {
            ctx.fillStyle = colors.muted;
            ctx.fillText((100 * p).toFixed(0), x + bw / 2, top - 4);
          }
        }
        const hb = (h - ha) / 2;
        drawSeriesChart(ctx, colors, 0, ha, w, hb, LL, {
          title: 'error, by training step',
          n: Math.max(200, LL.length),
          col: colors.ok,
          ym: Math.max(2.2, LL[0] || 2.2),
        });
        drawSeriesChart(ctx, colors, 0, ha + hb, w, hb, AC, {
          title: 'fresh sequences written entirely correctly (checked every 10 steps)',
          n: Math.max(20, AC.length),
          col: colors.accent,
          ym: 1,
          dots: AC.length < 40,
        });
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cs),
          lm = 64,
          tw = (w - lm - 10) / 3,
          cw = Math.min((tw * 0.8) / d, 22),
          chh = (h - 48) / T2,
          y0 = 30,
          TB = [
            [D.X, 'letter + position vectors, X'],
            [D.Z, 'what attention adds, Z'],
            [D.R2, 'leaving the block, R₂'],
          ];
        ctx.font = MONO;
        for (let i = 0; i < T2; i++) {
          ctx.fillStyle = i === pi ? colors.ink : colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText(
            (i < n ? 'in ' : i === n ? '' : 'out ') + lab(i),
            lm - 8,
            y0 + (i + 0.7) * chh,
          );
        }
        TB.forEach((tb, q) => {
          const x0 = lm + q * tw;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(tb[1], x0, y0 - 10);
          let m = 1e-9;
          for (const v of tb[0]) m = Math.max(m, Math.abs(v));
          for (let i = 0; i < T2; i++)
            for (let j = 0; j < d; j++) {
              const v = tb[0][i * d + j] / m;
              ctx.fillStyle = mix(
                colors.bg,
                v >= 0 ? colors.gold : colors.accent,
                Math.min(1, Math.abs(v) * 1.3),
              );
              ctx.fillRect(x0 + j * cw, y0 + i * chh, cw - 0.8, chh - 0.8);
            }
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 1.4;
          ctx.strokeRect(x0 - 1, y0 + pi * chh - 1, cw * d + 1, chh + 1);
          if (q < 2) {
            const xa = x0 + cw * d + 4,
              xb = x0 + tw - 6,
              ym = y0 + (chh * T2) / 2;
            ctx.strokeStyle = colors.muted;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(xa, ym);
            ctx.lineTo(xb, ym);
            ctx.lineTo(xb - 5, ym - 4);
            ctx.moveTo(xb, ym);
            ctx.lineTo(xb - 5, ym + 4);
            ctx.stroke();
            ctx.fillStyle = colors.muted;
            ctx.textAlign = 'center';
            ctx.fillText(q ? '+ X, feed-forward, +' : 'attention', (xa + xb) / 2, ym - 8);
          }
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'tf', activeStepIndex);
      let b = 0;
      for (let j = 0; j <= pi; j++) if (D.A[pi * T2 + j] > D.A[pi * T2 + b]) b = j;
      setReadout(root, 'step', String(ts));
      setReadout(root, 'L', ts ? f3(LL[ts - 1]) : '–');
      setReadout(root, 'acc', AC.length ? pct(AC[AC.length - 1]) : '–');
      setReadout(
        root,
        'out',
        '“' + D.out + '”' + (D.out === D.want ? ' (correct)' : ' (should be “' + D.want + '”)'),
      );
      setReadout(
        root,
        'look',
        (b < n
          ? 'input letter ' + (b + 1) + ', “' + LT[D.x[b]] + '”'
          : b === n
            ? 'the ▸ sign'
            : 'its own earlier output “' + lab(b) + '”') +
          ' (' +
          (100 * D.A[pi * T2 + b]).toFixed(0) +
          '%)',
      );
      setReadout(root, 'np', String(ps.reduce((a, p) => a + p.v.length, 0)));
    };
    build();
    if (prefersReducedMotion) while (!conv) train(20);
    const anim = Anim(() => {
      frameCount++;
      if (frameCount % (ts < 30 ? 5 : ts < 150 ? 2 : 1)) return;
      train(Math.min(ts < 150 ? 1 : 3, checkRunStop.remaining()));
      draw();
      checkRunStop();
    });
    playback = playBtn($('#tf-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'tf', {
      done: () => conv,
      restart: () => {
        run++;
        build();
      },
      count: () => ts,
      what: 'steps',
    });
    $('#tf-step', root).onclick = () => {
      playback.stop();
      if (conv) {
        run++;
        build();
      } else train(25);
      draw();
    };
    $('#tf-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    segBind(root, 'tf-task', settings.task, (v) => {
      settings.task = v;
      playback.stop();
      run++;
      build();
      draw();
      playback.start();
    });
    bind(
      root,
      'tf-pos',
      (v) => {
        settings.pos = v;
        draw();
      },
      (v) => String(v),
    );
    $('#tf-text', root).addEventListener('input', (e) => {
      const v = e.target.value.toLowerCase().replace(/[^a-h]/g, '');
      st.textContent = v.length < 5 ? 'Type five letters, using only a to h.' : '';
      if (v.length === 5) {
        settings.text = v;
        refreshDemoPrediction();
        msg =
          'For “' +
          settings.text +
          '” it writes “' +
          demo.out +
          '”' +
          (demo.out === demo.want
            ? ', which is correct' +
              (conv ? '. It was never trained on this particular sequence.' : '.')
            : '. The right answer is “' + demo.want + '”.');
        if (!playback.on) draw();
      }
    });
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
