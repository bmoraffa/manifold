/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  f2,
  prepareCanvas,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { UC, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#bp-c', root),
      cr = $('#bp-r', root),
      live = $('#bp-live', root),
      ta = $('#bp-text', root),
      st = $('#bp-st', root),
      chips = $('#bp-chips', root),
      DEF =
        'the low lower lowest. the new newer newest. the wide wider widest. she walks, she walked, she is walking. he talks, he talked, he is talking. they jump, they jumped, they are jumping. the walker and the talker and the jumper. slow slower slowest, fast faster fastest, old older oldest. walking slowly, talking loudly, jumping quickly. lower and lower, newer and newer, faster and faster. the lowest walker walked the slowest and the newest jumper jumped the fastest.';
    let words = [],
      order = [],
      base = [],
      merges = [],
      vocab = [],
      tokenCountHistory = [],
      vocabSizeHistory = [],
      pairs = [],
      nch = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false,
      tv = 0,
      tvd = -1,
      flash = '';
    const pretokenize = (txt) => {
      const o = [];
      const re = /([a-zà-ÿ0-9']+)|([^\sa-zà-ÿ0-9'])/g;
      let m;
      const s = txt.toLowerCase();
      while ((m = re.exec(s))) {
        const sp = m.index === 0 || /\s/.test(s[m.index - 1]);
        o.push((sp ? '_' : '') + m[0]);
      }
      return o;
    };
    const total = () => words.reduce((s, w) => s + w.sym.length * w.n, 0);
    const count = () => {
      const C = new Map();
      for (const w of words)
        for (let i = 0; i < w.sym.length - 1; i++) {
          const k = w.sym[i] + '\u0001' + w.sym[i + 1],
            e = C.get(k);
          if (e) e.n += w.n;
          else C.set(k, { a: w.sym[i], b: w.sym[i + 1], n: w.n });
        }
      return [...C.values()].sort((x, y) => y.n - x.n);
    };
    const apply = (sym, a, b) => {
      const o = [];
      for (let i = 0; i < sym.length; i++) {
        if (i < sym.length - 1 && sym[i] === a && sym[i + 1] === b) {
          o.push(a + b);
          i++;
        } else o.push(sym[i]);
      }
      return o;
    };
    const load = (txt) => {
      const P = pretokenize(txt);
      if (P.length < 8)
        return 'That text is too short to learn from. Use at least a few sentences.';
      const M = new Map();
      order = P.slice(0, 70);
      for (const p of P) {
        const e = M.get(p);
        if (e) e.n++;
        else M.set(p, { w: p, sym: [...p], n: 1 });
      }
      words = [...M.values()];
      base = [...new Set(words.flatMap((w) => w.sym))].sort();
      vocab = base.slice();
      merges = [];
      nch = total();
      tokenCountHistory = [nch];
      vocabSizeHistory = [vocab.length];
      pairs = count();
      fin = false;
      activeStepIndex = 0;
      flash = '';
      tv++;
      msg =
        'The text has ' +
        P.length +
        ' words, split into ' +
        nch +
        ' single characters. There are ' +
        base.length +
        ' different characters, and they are the whole vocabulary so far. Press Run or One merge.';
      return null;
    };
    const step = () => {
      if (fin) return;
      pairs = count();
      const p = pairs[0];
      if (!p || p.n < 2) {
        fin = true;
        activeStepIndex = 4;
        msg =
          'No pair of neighbouring tokens occurs more than once any more, so merging further would only memorise single words. Finished with ' +
          merges.length +
          ' merges: ' +
          nch +
          ' characters are now written with ' +
          total() +
          ' tokens.';
        return;
      }
      for (const w of words) w.sym = apply(w.sym, p.a, p.b);
      merges.push(p);
      vocab.push(p.a + p.b);
      tokenCountHistory.push(total());
      vocabSizeHistory.push(vocab.length);
      flash = p.a + p.b;
      activeStepIndex = 2;
      tv++;
      const ex = words
        .filter((w) => w.sym.includes(flash))
        .sort((x, y) => y.n - x.n)
        .slice(0, 4)
        .map((w) => w.w.replace('_', ''));
      msg =
        'Merge ' +
        merges.length +
        ': “' +
        p.a +
        '” followed by “' +
        p.b +
        '” occurred ' +
        p.n +
        ' times, more than any other pair, so they are now one token, “' +
        flash +
        '”. It appears in: ' +
        ex.join(', ') +
        '. The text went from ' +
        tokenCountHistory[tokenCountHistory.length - 2] +
        ' tokens to ' +
        tokenCountHistory[tokenCountHistory.length - 1] +
        '.';
    };
    const encode = (txt) =>
      pretokenize(txt).map((w) => {
        let s = [...w];
        for (const m of merges) s = apply(s, m.a, m.b);
        return s;
      });
    const chip = (tk, t, col) => {
      const i = vocab.indexOf(tk),
        unk = i < 0,
        cl = unk ? t.cut : col[i % 8];
      return (
        '<span style="background:' +
        mix(cl, cl, 0, unk ? 0.25 : 0.16) +
        ';' +
        (tk === flash ? 'outline:2px solid ' + t.gold + ';' : '') +
        (unk ? 'text-decoration:underline wavy;' : '') +
        '">' +
        tk.replace(/&/g, '&amp;').replace(/</g, '&lt;') +
        '</span>'
      );
    };
    const tryIt = () => {
      const colors = readThemeColors(),
        col = UC(colors),
        v = $('#bp-try', root).value,
        E = encode(v),
        flat = E.flat(),
        ids = flat.map((k) => vocab.indexOf(k)),
        unk = ids.filter((i) => i < 0).length;
      $('#bp-tryout', root).innerHTML = flat.length
        ? E.map((s) => s.map((k) => chip(k, colors, col)).join('')).join(
            '<span style="padding:0 3px"></span>',
          )
        : '';
      $('#bp-tryst', root).textContent = flat.length
        ? v.replace(/\s/g, '').length +
          ' characters became ' +
          flat.length +
          ' token' +
          (flat.length > 1 ? 's' : '') +
          '. As numbers: ' +
          ids.map((i) => (i < 0 ? '?' : i)).join(' ') +
          '.' +
          (unk
            ? ' ' +
              unk +
              ' character' +
              (unk > 1 ? 's were' : ' was') +
              ' never seen in the training text, so ' +
              (unk > 1 ? 'they have' : 'it has') +
              ' no number (shown with a wavy line). Real tokenizers avoid this by starting from all 256 bytes.'
            : '')
        : 'Type a sentence above.';
    };
    const draw = () => {
      const colors = readThemeColors(),
        col = UC(colors);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          P = pairs.slice(0, 12),
          mx = Math.max(1, ...P.map((p) => p.n)),
          rh = Math.min(30, (h - 34) / 12),
          x0 = 118;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          fin
            ? 'the most frequent pairs that remain: none occurs twice'
            : 'pair of neighbouring tokens, and how many times it occurs',
          8,
          13,
        );
        P.forEach((p, i) => {
          const y = 24 + i * rh,
            bw = ((w - x0 - 46) * p.n) / mx,
            win = i === 0 && !fin && p.n > 1;
          ctx.fillStyle = win ? colors.gold : mix(colors.accent, colors.accent, 0, 0.7);
          ctx.fillRect(x0, y + 3, Math.max(1, bw), rh - 7);
          ctx.fillStyle = win ? colors.cut : colors.ink;
          ctx.textAlign = 'right';
          ctx.font = '12px ' + MONO.split('px ')[1];
          ctx.fillText(p.a + ' + ' + p.b, x0 - 8, y + rh / 2 + 4);
          ctx.textAlign = 'left';
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.fillText(String(p.n), x0 + bw + 6, y + rh / 2 + 4);
        });
        if (!P.length) {
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText('every word is a single token: there are no pairs left', w / 2, h / 2);
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          n = Math.max(12, tokenCountHistory.length);
        drawSeriesChart(ctx, colors, 0, 0, w, h / 2, tokenCountHistory, {
          title: 'tokens needed to write the whole training text',
          n,
          col: colors.accent,
          ym: nch,
          dots: tokenCountHistory.length < 70,
        });
        drawSeriesChart(ctx, colors, 0, h / 2, w, h / 2, vocabSizeHistory, {
          title: 'different tokens in the vocabulary',
          n,
          col: colors.ok,
          ym: Math.max(base.length + 12, vocabSizeHistory[vocabSizeHistory.length - 1]),
          dots: vocabSizeHistory.length < 70,
        });
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('merges →', w - 12, h - 5);
      }
      if (tvd !== tv) {
        tvd = tv;
        const cur = new Map(words.map((w) => [w.w, w.sym]));
        chips.innerHTML = order
          .map((w) => (cur.get(w) || [...w]).map((k) => chip(k, colors, col)).join(''))
          .join('<span style="padding:0 3px"></span>');
        const tb = $('#bp-tab', root);
        tb.innerHTML = !merges.length
          ? '<p class="status">No merges yet. The vocabulary is the ' +
            base.length +
            ' single characters: ' +
            base.map((b) => (b === '_' ? '_' : b)).join(' ') +
            '</p>'
          : '<table><thead><tr><th>merge</th><th>left piece</th><th>right piece</th><th>new token</th><th>number</th><th>times seen</th></tr></thead><tbody>' +
            merges
              .map(
                (m, i) =>
                  '<tr><td>' +
                  (i + 1) +
                  '</td><td>' +
                  m.a +
                  '</td><td>' +
                  m.b +
                  '</td><td><b>' +
                  m.a +
                  m.b +
                  '</b></td><td>' +
                  (base.length + i) +
                  '</td><td>' +
                  m.n +
                  '</td></tr>',
              )
              .join('') +
            '</tbody></table>';
        tb.scrollTop = tb.scrollHeight;
        tryIt();
      }
      live.textContent = msg;
      highlightLessonStep(root, 'bp', activeStepIndex);
      const tot = tokenCountHistory[tokenCountHistory.length - 1];
      setReadout(root, 'm', String(merges.length));
      setReadout(root, 'v', String(vocab.length));
      setReadout(root, 'n', tot + ' (from ' + nch + ')');
      setReadout(root, 'cpt', f2(nch / tot));
      setReadout(root, 'last', merges.length ? flash : '–');
      setReadout(
        root,
        'long',
        vocab.reduce((a, b) => (b.length > a.length ? b : a), ''),
      );
    };
    ta.value = DEF;
    load(DEF);
    const anim = Anim(() => {
      frameCount++;
      if (frameCount % (merges.length < 8 ? 40 : 14)) return;
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#bp-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'bp', {
      done: () => fin,
      restart: () => {
        if (load(ta.value)) {
          ta.value = DEF;
          load(DEF);
        }
      },
      count: () => merges.length,
      what: 'merges',
    });
    $('#bp-step', root).onclick = () => {
      playback.stop();
      if (fin) load(ta.value);
      else step();
      draw();
    };
    $('#bp-reset', root).onclick = () => {
      playback.stop();
      load(ta.value);
      draw();
    };
    $('#bp-use', root).onclick = () => {
      playback.stop();
      const e = load(ta.value);
      if (e) {
        st.textContent = e;
        load(DEF);
        ta.value = DEF;
      } else st.textContent = 'Loaded your text. Press Run.';
      draw();
    };
    $('#bp-try', root).addEventListener('input', tryIt);
    if (prefersReducedMotion) {
      while (!fin) step();
    }
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        load(ta.value);
        draw();
        playback.start();
      },
    };
  },
};
