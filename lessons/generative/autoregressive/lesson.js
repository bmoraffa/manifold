/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  T,
  bind,
  f2,
  fit,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#ar-c', root),
      cr = $('#ar-r', root),
      live = $('#ar-live', root),
      outEl = $('#ar-out', root),
      ta = $('#ar-text', root),
      st = $('#ar-st', root),
      S = { k: 3, temp: 1 },
      MAXL = 320,
      DEF =
        'the cat sat on the mat and looked at the rain. the rain fell on the roof and on the road and on the old stone wall. a dog ran past the wall and the cat ran after the dog. then the sun came out and the cat sat in the sun and the dog sat in the shade. the rain had stopped and the road was wet and the wall was warm.';
    let text = '',
      M = {},
      out = '',
      last = null,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      pps = [];
    const learn = () => {
      text = (ta.value || DEF).toLowerCase().replace(/\s+/g, ' ').slice(0, 6000);
      if (text.length < 40) {
        st.textContent =
          'That text is very short. Paste at least a sentence or two; using the built-in text for now.';
        text = DEF;
      } else st.textContent = '';
      M = {};
      for (let k = 0; k <= 6; k++) {
        const m = (M[k] = {});
        for (let i = k; i < text.length; i++) {
          const cx = text.slice(i - k, i),
            t = m[cx] || (m[cx] = {});
          t[text[i]] = (t[text[i]] || 0) + 1;
        }
      }
      pps = [];
      for (let k = 0; k <= 6; k++) {
        let s = 0,
          n = 0;
        for (let i = k; i < text.length; i++) {
          const t = M[k][text.slice(i - k, i)];
          let tot = 0;
          for (const q in t) tot += t[q];
          s -= Math.log(t[text[i]] / tot);
          n++;
        }
        pps.push(Math.exp(s / n));
      }
    };
    const dist = () => {
      let k = Math.min(S.k, out.length),
        t;
      for (; ; k--) {
        t = M[k][out.slice(out.length - k)];
        if (t || k === 0) break;
      }
      const ent = Object.entries(t || { ' ': 1 });
      let z = 0;
      const pr = ent.map(([ch, n]) => {
        const p = Math.pow(n, 1 / S.temp);
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
      out = text.slice(0, Math.max(1, S.k));
      last = null;
      cu = 0;
      msg =
        'The model has read ' +
        text.length +
        ' letters and counted what follows every run of ' +
        S.k +
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
      cu = 2 + (out.length % 3);
      outEl.textContent = out;
      msg =
        'Context “' +
        d.ctx +
        '”' +
        (d.k < S.k ? ' (shortened, because the longer context never appeared in the text)' : '') +
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
      const t = T(),
        d = last ? last.d : dist(),
        sh = d.pr.slice(0, 14);
      {
        const { g, w, h } = fit(c),
          bw = (w - 30) / 14;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('after', 10, 16);
        g.font = SANS;
        const cx = '“' + d.ctx.replace(/ /g, '␣') + '”',
          tw = g.measureText(cx).width + 16;
        g.fillStyle = t.gold;
        g.fillRect(52, 2, tw, 22);
        g.fillStyle = t.bg;
        g.fillText(cx, 60, 18);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.fillText('the next letter is…', 60 + tw + 4, 16);
        sh.forEach((q, i) => {
          const x = 15 + i * bw,
            top = h - 30 - q[1] * (h - 90),
            pk = last && q === last.pick;
          g.fillStyle = pk ? t.gold : t.accent;
          g.fillRect(x + bw * 0.14, top, bw * 0.72, h - 30 - top);
          g.font = SANS;
          g.fillStyle = t.ink;
          g.textAlign = 'center';
          g.fillText(q[0] === ' ' ? '␣' : q[0], x + bw / 2, h - 12);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.fillText((100 * q[1]).toFixed(0) + '%', x + bw / 2, top - 5);
        });
        if (d.pr.length > 14) {
          g.textAlign = 'right';
          g.fillText('+' + (d.pr.length - 14) + ' more', w - 8, 44);
        }
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.48;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          'the counts behind those bars: after “' + d.ctx.replace(/ /g, '␣') + '” the text had',
          8,
          13,
        );
        const tot = d.pr.reduce((s, q) => s + q[2], 0);
        sh.slice(0, 7).forEach((q, i) => {
          const y = 24 + i * ((h1 - 30) / 7);
          g.fillStyle = t.ink;
          g.fillText(
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
          ym = Math.max(...pps) * 1.1,
          Y = (v) => h - m.b - (v / ym) * (h - h1 - m.t - m.b);
        g.fillStyle = t.muted;
        g.fillText(
          'perplexity on the training text (lower = less surprised), by memory',
          8,
          h1 + 12,
        );
        pps.forEach((v, k) => {
          const x = m.l + k * bw + bw * 0.15;
          g.fillStyle = k === S.k ? t.gold : t.ok;
          g.fillRect(x, Y(v), bw * 0.7, h - m.b - Y(v));
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText(String(k), x + bw * 0.35, h - m.b + 14);
          g.fillText(v.toFixed(1), x + bw * 0.35, Y(v) - 4);
        });
        g.textAlign = 'right';
        g.fillText('letters of memory', w - m.r, h - 3);
      }
      live.textContent = msg;
      uCur(root, 'ar', cu);
      setR(root, 'ctx', '“' + d.ctx.replace(/ /g, '␣') + '”');
      setR(root, 'opt', String(d.pr.length));
      setR(
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
      setR(root, 'n', String(text.length));
      setR(root, 'pp', pps[S.k].toFixed(2));
    };
    ta.value = DEF;
    learn();
    reset();
    for (let i = 0; i < 120; i++) step();
    const anim = Anim(() => {
      fr++;
      if (fr % (out.length < S.k + 8 ? 40 : 5)) return;
      step();
      draw();
      if (out.length >= MAXL) pb.stop();
    });
    pb = playBtn($('#ar-play', root), anim);
    runCtl(pb, root, 'ar', { done: () => out.length >= MAXL, restart: () => reset() });
    $('#ar-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#ar-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    bind(
      root,
      'ar-k',
      (v) => {
        S.k = v;
        pb.stop();
        reset();
        draw();
        pb.start();
      },
      (v) => String(v),
    );
    bind(
      root,
      'ar-temp',
      (v) => {
        S.temp = v;
        last = null;
        if (!pb.on) draw();
      },
      f2,
    );
    ta.addEventListener('change', () => {
      pb.stop();
      learn();
      reset();
      draw();
      pb.start();
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        pb.start();
      },
    };
  },
};
