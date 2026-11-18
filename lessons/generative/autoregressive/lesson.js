/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  readThemeColors,
  bind,
  f2,
  prepareCanvas,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { highlightLessonStep } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#ar-c', root),
      cr = $('#ar-r', root),
      live = $('#ar-live', root),
      outEl = $('#ar-out', root),
      trainingTextInput = $('#ar-text', root),
      st = $('#ar-st', root),
      settings = { k: 3, temp: 1 },
      MAXL = 320,
      DEF =
        'the cat sat on the mat and looked at the rain. the rain fell on the roof and on the road and on the old stone wall. a dog ran past the wall and the cat ran after the dog. then the sun came out and the cat sat in the sun and the dog sat in the shade. the rain had stopped and the road was wet and the wall was warm.';
    let text = '',
      nextCharacterCounts = {},
      out = '',
      last = null,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      perplexityByOrder = [];
    const learn = () => {
      text = (trainingTextInput.value || DEF).toLowerCase().replace(/\s+/g, ' ').slice(0, 6000);
      if (text.length < 40) {
        st.textContent =
          'That text is very short. Paste at least a sentence or two; using the built-in text for now.';
        text = DEF;
      } else st.textContent = '';
      nextCharacterCounts = {};
      for (let k = 0; k <= 6; k++) {
        const m = (nextCharacterCounts[k] = {});
        for (let i = k; i < text.length; i++) {
          const cx = text.slice(i - k, i),
            t = m[cx] || (m[cx] = {});
          t[text[i]] = (t[text[i]] || 0) + 1;
        }
      }
      perplexityByOrder = [];
      for (let k = 0; k <= 6; k++) {
        let s = 0,
          n = 0;
        for (let i = k; i < text.length; i++) {
          const t = nextCharacterCounts[k][text.slice(i - k, i)];
          let tot = 0;
          for (const q in t) tot += t[q];
          s -= Math.log(t[text[i]] / tot);
          n++;
        }
        perplexityByOrder.push(Math.exp(s / n));
      }
    };
    const dist = () => {
      let k = Math.min(settings.k, out.length),
        t;
      for (; ; k--) {
        t = nextCharacterCounts[k][out.slice(out.length - k)];
        if (t || k === 0) break;
      }
      const ent = Object.entries(t || { ' ': 1 });
      let z = 0;
      const pr = ent.map(([ch, n]) => {
        const p = Math.pow(n, 1 / settings.temp);
        z += p;
        return [ch, p, n];
      });
      pr.forEach((q) => {
        q[1] /= z;
      });
      pr.sort((a, b) => b[1] - a[1]);
      return { pr, k, ctx: out.slice(out.length - k) };
    };
    const reset = () => {
      out = text.slice(0, Math.max(1, settings.k));
      last = null;
      activeStepIndex = 0;
      msg =
        'The model has read ' +
        text.length +
        ' letters and counted what follows every run of ' +
        settings.k +
        '. It starts from “' +
        out +
        '”. Press Write or One letter.';
      outEl.textContent = out;
    };
    const step = () => {
      if (out.length >= MAXL) {
        reset();
        return;
      }
      const d = dist();
      let u = Math.random(),
        pick = d.pr[d.pr.length - 1];
      for (const q of d.pr) {
        if ((u -= q[1]) <= 0) {
          pick = q;
          break;
        }
      }
      last = { d, pick };
      out += pick[0];
      activeStepIndex = 2 + (out.length % 3);
      outEl.textContent = out;
      msg =
        'Context “' +
        d.ctx +
        '”' +
        (d.k < settings.k
          ? ' (shortened, because the longer context never appeared in the text)'
          : '') +
        ': in the text it was followed by ' +
        d.pr.length +
        ' different letter' +
        (d.pr.length === 1 ? '' : 's') +
        '. The model picked “' +
        (pick[0] === ' ' ? 'space' : pick[0]) +
        '”, which had probability ' +
        (100 * pick[1]).toFixed(0) +
        '%.' +
        (out.length >= MAXL ? ' It has written ' + MAXL + ' letters, so it stops.' : '');
    };
    const draw = () => {
      const colors = readThemeColors(),
        d = last ? last.d : dist(),
        sh = d.pr.slice(0, 14);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          bw = (w - 30) / 14;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('after', 10, 16);
        ctx.font = SANS;
        const cx = '“' + d.ctx.replace(/ /g, '␣') + '”',
          tw = ctx.measureText(cx).width + 16;
        ctx.fillStyle = colors.gold;
        ctx.fillRect(52, 2, tw, 22);
        ctx.fillStyle = colors.bg;
        ctx.fillText(cx, 60, 18);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.fillText('the next letter is…', 60 + tw + 4, 16);
        sh.forEach((q, i) => {
          const x = 15 + i * bw,
            top = h - 30 - q[1] * (h - 90),
            pk = last && q === last.pick;
          ctx.fillStyle = pk ? colors.gold : colors.accent;
          ctx.fillRect(x + bw * 0.14, top, bw * 0.72, h - 30 - top);
          ctx.font = SANS;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText(q[0] === ' ' ? '␣' : q[0], x + bw / 2, h - 12);
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.fillText((100 * q[1]).toFixed(0) + '%', x + bw / 2, top - 5);
        });
        if (d.pr.length > 14) {
          ctx.textAlign = 'right';
          ctx.fillText('+' + (d.pr.length - 14) + ' more', w - 8, 44);
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.48;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          'the counts behind those bars: after “' + d.ctx.replace(/ /g, '␣') + '” the text had',
          8,
          13,
        );
        const tot = d.pr.reduce((s, q) => s + q[2], 0);
        sh.slice(0, 7).forEach((q, i) => {
          const y = 24 + i * ((h1 - 30) / 7);
          ctx.fillStyle = colors.ink;
          ctx.fillText(
            '“' +
              (q[0] === ' ' ? '␣' : q[0]) +
              '”  ' +
              q[2] +
              ' time' +
              (q[2] === 1 ? '' : 's') +
              ' of ' +
              tot,
            14,
            y + 11,
          );
        });
        const m = { l: 44, r: 12, t: 22, b: 22 },
          bw = (w - m.l - m.r) / 7,
          ym = Math.max(...perplexityByOrder) * 1.1,
          Y = (v) => h - m.b - (v / ym) * (h - h1 - m.t - m.b);
        ctx.fillStyle = colors.muted;
        ctx.fillText(
          'perplexity on the training text (lower = less surprised), by memory',
          8,
          h1 + 12,
        );
        perplexityByOrder.forEach((v, k) => {
          const x = m.l + k * bw + bw * 0.15;
          ctx.fillStyle = k === settings.k ? colors.gold : colors.ok;
          ctx.fillRect(x, Y(v), bw * 0.7, h - m.b - Y(v));
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText(String(k), x + bw * 0.35, h - m.b + 14);
          ctx.fillText(v.toFixed(1), x + bw * 0.35, Y(v) - 4);
        });
        ctx.textAlign = 'right';
        ctx.fillText('letters of memory', w - m.r, h - 3);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'ar', activeStepIndex);
      setReadout(root, 'ctx', '“' + d.ctx.replace(/ /g, '␣') + '”');
      setReadout(root, 'opt', String(d.pr.length));
      setReadout(
        root,
        'pick',
        last
          ? '“' +
              (last.pick[0] === ' ' ? '␣' : last.pick[0]) +
              '” (' +
              (100 * last.pick[1]).toFixed(0) +
              '%)'
          : '–',
      );
      setReadout(root, 'n', String(text.length));
      setReadout(root, 'pp', perplexityByOrder[settings.k].toFixed(2));
    };
    trainingTextInput.value = DEF;
    learn();
    reset();
    for (let i = 0; i < 120; i++) step();
    const anim = Anim(() => {
      frameCount++;
      if (frameCount % (out.length < settings.k + 8 ? 40 : 5)) return;
      step();
      draw();
      if (out.length >= MAXL) playback.stop();
    });
    playback = playBtn($('#ar-play', root), anim);
    attachRunLimit(playback, root, 'ar', {
      done: () => out.length >= MAXL,
      restart: () => reset(),
    });
    $('#ar-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#ar-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    bind(
      root,
      'ar-k',
      (v) => {
        settings.k = v;
        playback.stop();
        reset();
        draw();
        playback.start();
      },
      (v) => String(v),
    );
    bind(
      root,
      'ar-temp',
      (v) => {
        settings.temp = v;
        last = null;
        if (!playback.on) draw();
      },
      f2,
    );
    trainingTextInput.addEventListener('change', () => {
      playback.stop();
      learn();
      reset();
      draw();
      playback.start();
    });
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        reset();
        draw();
        playback.start();
      },
    };
  },
};
