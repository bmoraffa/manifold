/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  f2,
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
    const mainCanvas = $('#ns-c', root),
      cr = $('#ns-r', root),
      live = $('#ns-live', root),
      outEl = $('#ns-out', root),
      settings = { t: 1, k: 40, p: 1 },
      CTX = 3,
      ED = 12,
      MAXS = 700,
      MAXW = 40,
      SENT = [
        'the cat sat on the mat',
        'the dog sat on the rug',
        'the cat chased the mouse',
        'the dog chased the cat',
        'the mouse ate the cheese',
        'the cat drank the milk',
        'the dog ate the bone',
        'the bird sang in the tree',
        'the cat slept in the sun',
        'the dog slept on the rug',
        'the mouse hid under the mat',
        'the bird flew over the tree',
        'a small cat sat in the sun',
        'a big dog sat under the tree',
        'the small mouse ate the old cheese',
        'the big dog chased the small cat',
        'the old cat slept on the warm mat',
        'the bird sang and the cat slept',
        'the dog barked and the cat ran',
        'the cat ran up the tree',
        'the mouse ran under the rug',
        'the bird ate the bread',
        'the girl fed the cat',
        'the boy fed the dog',
        'the girl saw a small bird',
        'the boy saw a big dog',
        'the girl sat in the sun',
        'the boy ran after the dog',
        'the cat looked at the bird',
        'the dog looked at the bone',
        'the sun was warm and the cat was happy',
        'the rain fell and the dog was sad',
        'the girl was happy and the boy was happy',
        'the mouse was small and the cheese was old',
        'a small bird sat on the old tree',
        'the big dog drank the milk',
        'the girl saw the mouse and the mouse ran',
        'the boy sat on the warm rug',
        'the old dog slept in the sun',
        'the cat hid under the rug',
      ];
    const VOC = ['.', ...new Set(SENT.join(' ').split(' '))],
      V = VOC.length,
      ix = (w) => VOC.indexOf(w);
    let E,
      W1,
      b1,
      W2,
      b2,
      ps,
      ts = 0,
      LL = [],
      sm = [],
      phase = 'train',
      PX = [],
      PY = [],
      text = [],
      np0 = 0,
      dist = null,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      SUR = [],
      EFF = [],
      lastP = NaN,
      lastAl = NaN,
      pickI = -1;
    const fwd = (ctxs) => {
      const B = ctxs.length / CTX,
        x = AD.gather(E, Array.from(ctxs));
      x.sh = [B, CTX * ED];
      return AD.addb(AD.mm(AD.tanh(AD.addb(AD.mm(x, W1), b1)), W2), b2);
    };
    const build = () => {
      const rn = randnFrom(rng(run * 13 + 7)),
        r = rng(run * 5 + 1);
      E = AD.param([V, ED], 0.5, rn);
      W1 = AD.param([CTX * ED, 48], Math.sqrt(1 / (CTX * ED)), rn);
      b1 = AD.param([48], 0, rn);
      W2 = AD.param([48, V], Math.sqrt(1 / 48), rn);
      b2 = AD.param([V], 0, rn);
      ps = [E, W1, b1, W2, b2];
      PX = [];
      PY = [];
      for (let e = 0; e < 6; e++) {
        const o = SENT.map((s, i) => [r(), i])
            .sort((a, b) => a[0] - b[0])
            .map((q) => SENT[q[1]]),
          st = ['.', '.', '.'].concat(o.join(' . ').split(' '), ['.']).map(ix);
        for (let i = CTX; i < st.length; i++) {
          PX.push(st[i - 3], st[i - 2], st[i - 1]);
          PY.push(st[i]);
        }
      }
      ts = 0;
      LL = [];
      sm = [];
      phase = 'train';
      activeStepIndex = 0;
      setPrompt();
      msg =
        'The network has ' +
        V +
        ' words in its vocabulary and random weights. Every word is equally likely to it. Press Run: it trains for a few seconds, then starts writing.';
    };
    const setPrompt = () => {
      text = ['.', '.', '.'].concat($('#ns-pr', root).value.split(' ')).map(ix);
      np0 = text.length;
      SUR = [];
      EFF = [];
      dist = null;
      lastP = NaN;
      lastAl = NaN;
      pickI = -1;
    };
    const train = (k) => {
      const B = 64,
        N = PY.length;
      for (let q = 0; q < k && ts < MAXS; q++) {
        const cx = new Int32Array(B * CTX),
          tg = new Int32Array(B);
        for (let i = 0; i < B; i++) {
          const j = Math.floor(Math.random() * N);
          cx[3 * i] = PX[3 * j];
          cx[3 * i + 1] = PX[3 * j + 1];
          cx[3 * i + 2] = PX[3 * j + 2];
          tg[i] = PY[j];
        }
        const l = AD.ce(fwd(cx), tg, 0);
        LL.push(l.v[0]);
        AD.backward(l);
        AD.adam(ps, 0.01 * Math.pow(0.2, ts / MAXS), ++ts, 1);
        if (ts % 10 === 0) {
          let s = 0;
          for (let j = ts - 10; j < ts; j++) s += LL[j];
          sm.push(s / 10);
        }
      }
      activeStepIndex = 0;
      calc();
      msg =
        'Training step ' +
        ts +
        ' of ' +
        MAXS +
        ': the network was shown 64 groups of three words and asked for the next word each time. Its average surprise is ' +
        f2((sm.length ? sm[sm.length - 1] : LL[ts - 1]) / Math.LN2) +
        ' bits per word (a blind guess among ' +
        V +
        ' words costs ' +
        f2(Math.log2(V)) +
        ' bits). The bars show what it currently expects after “' +
        text
          .slice(-3)
          .map((i) => VOC[i])
          .join(' ')
          .replace(/^[. ]+/, '') +
        '”.';
      if (ts >= MAXS) {
        phase = 'gen';
        activeStepIndex = 1;
        msg =
          'Training is finished. The bars show the network’s probabilities for the word after “' +
          text
            .slice(np0 - 2)
            .map((i) => VOC[i])
            .join(' ') +
          '”, with your settings applied. Press Run or One word.';
      }
    };
    const calc = () => {
      const z = fwd(Int32Array.from(text.slice(-3)));
      AD.clear();
      const n = V,
        baseProbs = new Float64Array(n),
        temperatureProbs = new Float64Array(n);
      let mx = -1e30;
      for (let i = 0; i < n; i++) mx = Math.max(mx, z.v[i]);
      let s1 = 0,
        st = 0;
      for (let i = 0; i < n; i++) {
        baseProbs[i] = Math.exp(z.v[i] - mx);
        s1 += baseProbs[i];
        temperatureProbs[i] = Math.exp((z.v[i] - mx) / settings.t);
        st += temperatureProbs[i];
      }
      let H = 0;
      for (let i = 0; i < n; i++) {
        baseProbs[i] /= s1;
        temperatureProbs[i] /= st;
        if (temperatureProbs[i] > 1e-12) H -= temperatureProbs[i] * Math.log2(temperatureProbs[i]);
      }
      const ord = Array.from({ length: n }, (_, i) => i).sort(
          (a, b) => temperatureProbs[b] - temperatureProbs[a],
        ),
        keep = new Uint8Array(n);
      let cum = 0,
        m = 0;
      for (const i of ord) {
        if (m >= settings.k) break;
        keep[i] = 1;
        m++;
        cum += temperatureProbs[i];
        if (cum >= settings.p - 1e-9) break;
      }
      const samplingProbs = new Float64Array(n);
      for (let i = 0; i < n; i++) if (keep[i]) samplingProbs[i] = temperatureProbs[i] / cum;
      dist = { baseProbs, temperatureProbs, samplingProbs, ord, keep, m, eff: Math.pow(2, H) };
    };
    const gstep = () => {
      if (phase !== 'gen' || text.length - np0 >= MAXW) return;
      calc();
      let u = Math.random(),
        pick = dist.ord[0];
      for (const i of dist.ord) {
        if (!dist.keep[i]) continue;
        u -= dist.samplingProbs[i];
        pick = i;
        if (u <= 0) break;
      }
      const prev = text
        .slice(-3)
        .map((i) => VOC[i])
        .join(' ')
        .replace(/^[. ]+/, '');
      text.push(pick);
      SUR.push(-Math.log2(Math.max(1e-9, dist.baseProbs[pick])));
      EFF.push(dist.eff);
      lastP = dist.samplingProbs[pick];
      lastAl = dist.m;
      pickI = pick;
      activeStepIndex = 4;
      const rk = dist.ord.indexOf(pick) + 1;
      msg =
        'After “' +
        prev +
        '” the network gave probabilities to all ' +
        V +
        ' words. With temperature ' +
        f2(settings.t) +
        ' that amounts to about ' +
        f2(dist.eff) +
        ' real choices. The cuts left ' +
        dist.m +
        ' word' +
        (dist.m > 1 ? 's' : '') +
        '. The word drawn was “' +
        VOC[pick] +
        '”, the network’s choice number ' +
        rk +
        ', with probability ' +
        pct(lastP) +
        ' after rescaling.' +
        (text.length - np0 >= MAXW ? ' That is ' + MAXW + ' words: finished.' : '');
    };
    const reps = () => {
      const w = text.slice(3),
        C = new Map();
      let r = 0;
      for (let i = 0; i + 3 <= w.length; i++) {
        const k = w.slice(i, i + 3).join(',');
        C.set(k, (C.get(k) || 0) + 1);
      }
      for (const v of C.values()) if (v > 1) r++;
      return r;
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas);
        ctx.font = MONO;
        if (!dist) {
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText('press Run', w / 2, h / 2);
        } else {
          const NB = 14,
            rh = Math.min(28, (h - 30) / NB),
            x0 = 86,
            mx = Math.max(
              0.05,
              dist.temperatureProbs[dist.ord[0]],
              ...Array.from(dist.samplingProbs),
            ),
            W = w - x0 - 64;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(
            pickI >= 0 && phase === 'gen'
              ? 'the choice just made: the 14 most likely words, of ' + V
              : 'the 14 most likely next words, of ' + V,
            8,
            13,
          );
          for (let r = 0; r < NB; r++) {
            const i = dist.ord[r],
              y = 22 + r * rh,
              kept = dist.keep[i],
              pk = i === pickI && phase === 'gen';
            ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.25);
            ctx.fillRect(x0, y + 3, Math.max(1, (W * dist.temperatureProbs[i]) / mx), rh - 7);
            if (kept) {
              ctx.fillStyle = pk ? colors.gold : colors.accent;
              ctx.fillRect(
                x0,
                y + rh * 0.3,
                Math.max(1, (W * dist.samplingProbs[i]) / mx),
                rh * 0.4,
              );
            }
            ctx.font = '12px ' + MONO.split('px ')[1];
            ctx.textAlign = 'right';
            ctx.fillStyle = kept ? (pk ? colors.cut : colors.ink) : colors.muted;
            const lab = VOC[i] === '.' ? '. (end)' : VOC[i];
            ctx.fillText(lab, x0 - 8, y + rh / 2 + 4);
            if (!kept) {
              const tw = ctx.measureText(lab).width;
              ctx.strokeStyle = colors.muted;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(x0 - 8 - tw, y + rh / 2);
              ctx.lineTo(x0 - 8, y + rh / 2);
              ctx.stroke();
            }
            ctx.font = MONO;
            ctx.textAlign = 'left';
            ctx.fillStyle = colors.muted;
            ctx.fillText(
              kept ? (100 * dist.samplingProbs[i]).toFixed(1) + '%' : 'cut',
              x0 +
                Math.max(
                  (W * dist.temperatureProbs[i]) / mx,
                  kept ? (W * dist.samplingProbs[i]) / mx : 0,
                ) +
                6,
              y + rh / 2 + 4,
            );
          }
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr);
        if (phase === 'train')
          drawSeriesChart(
            ctx,
            colors,
            0,
            0,
            w,
            h,
            sm.map((v) => v / Math.LN2),
            {
              title: 'training: average surprise at the true next word, in bits',
              n: MAXS / 10,
              col: colors.ok,
              ym: Math.log2(V) * 1.05,
            },
          );
        else {
          const n = Math.max(MAXW, SUR.length),
            A = drawSeriesChart(ctx, colors, 0, 0, w, h / 2, [], {
              title: 'surprise of each chosen word, in bits (by the network’s own odds)',
              n,
              ym: Math.max(6, ...SUR),
            }),
            bw = (w - A.m.l - A.m.r) / n;
          SUR.forEach((v, i) => {
            ctx.fillStyle =
              i === SUR.length - 1 ? colors.gold : mix(colors.accent, colors.accent, 0, 0.75);
            ctx.fillRect(A.m.l + i * bw + 1, A.Y(v), Math.max(1, bw - 2), A.Y(0) - A.Y(v));
          });
          drawSeriesChart(ctx, colors, 0, h / 2, w, h / 2, EFF, {
            title: 'effective number of choices at each step, before the cuts',
            n,
            col: colors.ok,
            ym: Math.max(8, ...EFF),
            dots: EFF.length < 45,
          });
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText('words written →', w - 12, h - 5);
        }
      }
      const wd = text.slice(3).map((i) => VOC[i]),
        pr = wd.slice(0, np0 - 3).join(' '),
        rest = wd.slice(np0 - 3);
      let s = '';
      rest.forEach((x) => {
        s += x === '.' ? '.' : ' ' + x;
      });
      outEl.innerHTML =
        '<b>' +
        pr +
        '</b>' +
        s.replace(/\. /g, '. ') +
        (phase === 'gen' && rest.length < MAXW ? ' ▌' : '');
      live.textContent = msg;
      highlightLessonStep(root, 'ns', activeStepIndex);
      setReadout(
        root,
        'ph',
        phase === 'train'
          ? ts
            ? 'Training, step ' + ts + ' of ' + MAXS
            : 'Not started'
          : rest.length >= MAXW
            ? 'Finished'
            : 'Writing',
      );
      setReadout(root, 'n', rest.length + ' of ' + MAXW);
      setReadout(root, 'pk', isFinite(lastP) ? pct(lastP) : '–');
      setReadout(root, 'al', dist ? dist.m + ' of ' + V : '–');
      setReadout(root, 'ef', dist ? f2(dist.eff) : '–');
      setReadout(root, 'sp', SUR.length ? f2(SUR.reduce((a, b) => a + b, 0) / SUR.length) : '–');
      setReadout(root, 'rp', rest.length ? String(reps()) : '–');
    };
    build();
    calc();
    if (prefersReducedMotion) {
      train(MAXS);
      while (text.length - np0 < MAXW) gstep();
    }
    const anim = Anim(() => {
      frameCount++;
      if (phase === 'train') {
        train(ts < 40 ? 1 : 4);
        draw();
        checkRunStop();
        return;
      }
      if (frameCount % 22 === 0) {
        gstep();
        draw();
        checkRunStop();
      }
    });
    playback = playBtn($('#ns-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'ns', {
      done: () => phase === 'gen' && text.length - np0 >= MAXW,
      restart: () => {
        setPrompt();
        calc();
      },
      count: () => (phase === 'train' ? 0 : text.length - np0),
      what: 'words',
    });
    const re = () => {
      if (phase !== 'gen') return;
      calc();
      pickI = -1;
      draw();
    };
    $('#ns-step', root).onclick = () => {
      playback.stop();
      if (phase === 'train') train(50);
      else if (text.length - np0 >= MAXW) {
        setPrompt();
        calc();
        msg = 'The text is back to the prompt. Press Run or One word.';
      } else gstep();
      draw();
    };
    $('#ns-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      calc();
      draw();
    };
    const bt = bind(
        root,
        'ns-t',
        (v) => {
          settings.t = +v;
          re();
        },
        f2,
      ),
      bk = bind(
        root,
        'ns-k',
        (v) => {
          settings.k = +v;
          re();
        },
        (v) => (+v >= 40 ? 'all' : String(v)),
      ),
      bp = bind(
        root,
        'ns-p',
        (v) => {
          settings.p = +v;
          re();
        },
        f2,
      );
    const PRE = {
      greedy: [0.1, 1, 1],
      pure: [1, 40, 1],
      topk: [1, 5, 1],
      topp: [1, 40, 0.9],
      hot: [2.2, 40, 1],
    };
    segBind(root, 'ns-pre', 'pure', (v) => {
      const q = PRE[v];
      settings.t = q[0];
      settings.k = q[1];
      settings.p = q[2];
      bt.set(q[0]);
      bk.set(q[1]);
      bp.set(q[2]);
      if (phase === 'gen') {
        playback.stop();
        setPrompt();
        calc();
        draw();
        playback.start();
      }
    });
    $('#ns-pr', root).onchange = () => {
      playback.stop();
      setPrompt();
      if (phase === 'gen') {
        calc();
        msg = 'New prompt. The bars show what the network expects next.';
      } else calc();
      draw();
    };
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        run++;
        build();
        calc();
        draw();
        playback.start();
      },
    };
  },
};
