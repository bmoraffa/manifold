/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  f2,
  fit,
  markTP,
  mix,
  playBtn,
  setR,
} from '../../../js/core/shared.js';
import { jacobi, segBind, uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#rg-c', root),
      cr = $('#rg-r', root),
      live = $('#rg-live', root),
      inp = $('#rg-in', root),
      st = $('#rg-st', root),
      S = { k: 2, mode: 'on' },
      TH = 0.12,
      BASE = [
        'The library is open from 8 am to 10 pm on weekdays and from 10 am to 6 pm on weekends.',
        'Students may borrow up to twelve books at a time. Each loan lasts three weeks and can be renewed twice.',
        'The machine learning course is taught by Professor Okafor on Tuesday and Thursday mornings in room 204.',
        'The final project for the machine learning course is due on December 12. It counts for forty percent of the grade.',
        'Homework may be submitted up to two days late. The penalty is ten percent per day.',
        'The computing cluster has sixteen graphics cards. Each student may reserve one card for up to eight hours a day.',
        'To reserve the computing cluster, book a slot on the lab calendar at least one day in advance.',
        'The cafeteria serves lunch from 11:30 am to 2 pm. A vegetarian meal costs six dollars.',
        'Parking permits cost forty dollars per semester. They are sold at the security office in the Hale building.',
        'The tutoring centre offers free help with statistics and programming every afternoon from 3 pm to 5 pm.',
        'Exam results are published on the student portal within ten working days of the exam.',
        'A student who misses an exam because of illness must send a doctor’s note to the registrar within five days.',
        'The campus shuttle runs every twenty minutes between the main gate and the train station until 9 pm.',
        'Printing costs five cents per page in black and white and twenty cents per page in colour.',
      ],
      QS = [
        'When is the final project due?',
        'How many books can I borrow?',
        'How much does a parking permit cost?',
        'Who teaches machine learning?',
        'Can I hand in homework late?',
        'What is the wifi password?',
      ],
      STOP = new Set(
        'a an the is are was were be been of to in on at by for from with and or it its this that these those i me my we you your he she they them do does did can could may might must will would should what when where who whom which how why as up per each every than then there here not no so if into about over under out any some much many have has had get'.split(
          ' ',
        ),
      );
    let docs = BASE.slice(),
      voc = [],
      idf = [],
      DV = [],
      PC = [],
      ax = null,
      mean = null,
      q = '',
      qv = null,
      qxy = [0, 0],
      sims = [],
      rank = [],
      ph = 0,
      msg = '',
      fr = 0,
      pb,
      qtop = [],
      flyU = 1;
    const stem = (w) => {
      w = w.replace(/[’']s$/, '');
      if (w.length > 4 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
      if (w.length > 4 && /(ches|shes|sses|xes)$/.test(w)) return w.slice(0, -2);
      if (w.length > 3 && w.endsWith('s') && !/(ss|us|is)$/.test(w)) w = w.slice(0, -1);
      return w;
    };
    const toks = (s) =>
      (s.toLowerCase().match(/[a-zà-ÿ0-9:’']+/g) || []).map(stem).filter((w) => w && !STOP.has(w));
    const vec = (s) => {
      const v = new Float64Array(voc.length);
      for (const w of toks(s)) {
        const i = voc.indexOf(w);
        if (i >= 0) v[i] += idf[i];
      }
      return v;
    };
    const nrm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)),
      cosv = (a, b) => {
        const na = nrm(a),
          nb = nrm(b);
        if (!na || !nb) return 0;
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s / (na * nb);
      };
    const index = () => {
      const T2 = docs.map(toks);
      voc = [...new Set(T2.flat())];
      const N = docs.length;
      idf = voc.map((w) => Math.log((N + 1) / (T2.filter((t) => t.includes(w)).length + 1)));
      DV = docs.map(vec).map((v) => {
        const n = nrm(v) || 1;
        return v.map((x) => x / n);
      });
      mean = new Float64Array(voc.length);
      DV.forEach((v) => v.forEach((x, i) => (mean[i] += x / N)));
      const Xc = DV.map((v) => v.map((x, i) => x - mean[i])),
        G = Xc.map((a) =>
          Xc.map((b) => {
            let s = 0;
            for (let i = 0; i < a.length; i++) s += a[i] * b[i];
            return s;
          }),
        ),
        J = jacobi(G);
      ax = [0, 1].map((j) => {
        const u = new Float64Array(voc.length),
          l = Math.sqrt(Math.max(1e-12, J.val[j]));
        Xc.forEach((x, i) => {
          for (let d = 0; d < u.length; d++) u[d] += (x[d] * J.vec[j][i]) / l;
        });
        return u;
      });
      PC = Xc.map((x) =>
        ax.map((u) => {
          let s = 0;
          for (let d = 0; d < u.length; d++) s += x[d] * u[d];
          return s;
        }),
      );
    };
    const ask = () => {
      q = inp.value.trim();
      const v = vec(q),
        n = nrm(v) || 1;
      qv = v.map((x) => x / n);
      qxy = nrm(v)
        ? ax.map((u) => {
            let s = 0;
            for (let d = 0; d < u.length; d++) s += (qv[d] - mean[d]) * u[d];
            return s;
          })
        : [0, 0];
      sims = DV.map((d) => cosv(qv, d));
      rank = sims.map((s, i) => i).sort((a, b) => sims[b] - sims[a]);
      qtop = [...new Set(toks(q))]
        .map((w) => [w, voc.indexOf(w)])
        .map(([w, i]) => [w, i < 0 ? 0 : idf[i]])
        .sort((a, b) => b[1] - a[1]);
    };
    const sentBest = () => {
      const ql = q.toLowerCase(),
        kind = /^who\b/.test(ql)
          ? /Professor|Dr\.|Ms\.|Mr\./
          : /^when\b|what time|what day/.test(ql)
            ? /\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|june|july|august|september|october|november|december/i
            : /^how (much|many|long|often)/.test(ql)
              ? /\d|\b(one|two|three|four|five|six|eight|ten|twelve|sixteen|twenty|forty)\b/i
              : null;
      let best = '',
        bs = -1,
        bd = -1;
      for (const i of rank.slice(0, S.k))
        for (const s of docs[i].match(/[^.!?]+(?:[.!?](?=\s|$)|$)/g) || [docs[i]]) {
          const sc = cosv(qv, vec(s)) + 0.15 * sims[i] + (kind && kind.test(s) ? 0.25 : 0);
          if (sc > bs) {
            bs = sc;
            best = s.trim();
            bd = i;
          }
        }
      return { best, bd };
    };
    const prompt = () =>
      S.mode === 'off'
        ? 'Question: ' + q + '\n\nAnswer:'
        : 'Answer the question using only the passages below. If the answer is not in them, say that you do not know.\n\n' +
          rank
            .slice(0, S.k)
            .map((i, j) => 'Passage ' + (j + 1) + ': ' + docs[i])
            .join('\n') +
          '\n\nQuestion: ' +
          q +
          '\n\nAnswer:';
    const MSG = () => {
      const known = qtop.filter((x) => x[1] > 0),
        b = rank[0];
      return [
        'The collection holds ' +
          docs.length +
          ' passages. Each has been turned into a list of ' +
          voc.length +
          ' numbers, one for every different word, and stored. Press Run or One step to send the question through.',
        'The question was turned into numbers in the same way. ' +
          (known.length
            ? 'The words that carry weight are: ' +
              known
                .slice(0, 5)
                .map((x) => x[0] + ' (' + f2(x[1]) + ')')
                .join(', ') +
              '.'
            : 'None of its words appears anywhere in the collection, so its embedding is empty.') +
          (qtop.length > known.length
            ? ' Not in any passage, so ignored: ' +
              qtop
                .filter((x) => x[1] <= 0)
                .map((x) => x[0])
                .join(', ') +
              '.'
            : ''),
        'The question was compared with all ' +
          docs.length +
          ' passages. The closest scores ' +
          f2(sims[b]) +
          ': “' +
          docs[b].slice(0, 70) +
          (docs[b].length > 70 ? '…' : '') +
          '”. ' +
          (sims[b] < TH
            ? 'That is a very weak match.'
            : 'The next best scores ' + f2(sims[rank[1]]) + '.'),
        'The ' +
          S.k +
          ' closest passage' +
          (S.k > 1 ? 's are' : ' is') +
          ' kept: number' +
          (S.k > 1 ? 's ' : ' ') +
          rank
            .slice(0, S.k)
            .map((i) => i + 1)
            .join(', ') +
          ' in the collection. The other ' +
          (docs.length - S.k) +
          ' are never shown to the model.',
        S.mode === 'off'
          ? 'Without retrieval the prompt is only the question, ' +
            prompt().split(/\s+/).length +
            ' words. The model has nothing to read.'
          : 'The prompt has been written: an instruction, the ' +
            S.k +
            ' retrieved passage' +
            (S.k > 1 ? 's' : '') +
            ', then the question. It is ' +
            prompt().split(/\s+/).length +
            ' words long. This text is everything the model will see.',
        S.mode === 'off'
          ? 'The model answered from memory alone, and this handbook was never in its memory.'
          : sims[b] < TH
            ? 'The best match was below ' +
              TH +
              ', so the reader declined to answer instead of guessing.'
            : 'The reader found its answer in passage ' +
              (sentBest().bd + 1) +
              ' and copied the sentence that fits the question best.',
      ][ph];
    };
    const answer = () => {
      const el = $('#rg-ans', root);
      if (ph < 5) {
        el.textContent = '…';
        el.style.opacity = 0.4;
        return;
      }
      el.style.opacity = 1;
      if (S.mode === 'off') {
        el.innerHTML =
          '<b>No passages were supplied.</b> A model that has never read this handbook cannot know the answer. A careful one says so. A careless one produces a fluent guess, and nothing in the guess would tell you it was invented.';
        return;
      }
      if (!sims.length || sims[rank[0]] < TH) {
        el.innerHTML =
          '<b>I do not know.</b> Nothing in the collection is close enough to this question (best similarity ' +
          f2(sims[rank[0]] || 0) +
          ').';
        return;
      }
      const a = sentBest();
      el.innerHTML =
        '<b>' +
        a.best.replace(/</g, '&lt;') +
        '</b><br><span style="color:var(--muted);font-size:.86rem">Source: passage ' +
        (a.bd + 1) +
        ' of the collection, similarity ' +
        f2(sims[a.bd]) +
        '. Copied by a simple reader that stands in for the language model.</span>';
    };
    const draw = () => {
      const t = T(),
        top = new Set(ph >= 3 && S.mode === 'on' ? rank.slice(0, S.k) : []);
      {
        const { g, w, h } = fit(c),
          xs = PC.map((p) => p[0]).concat(ph >= 1 ? [qxy[0]] : []),
          ys = PC.map((p) => p[1]).concat(ph >= 1 ? [qxy[1]] : []),
          x0 = Math.min(...xs),
          x1 = Math.max(...xs),
          y0 = Math.min(...ys),
          y1 = Math.max(...ys),
          X = (v) => 50 + ((v - x0) / (x1 - x0 || 1)) * (w - 150),
          Y = (v) => h - 34 - ((v - y0) / (y1 - y0 || 1)) * (h - 70);
        const qx = X(qxy[0]),
          qy = Y(qxy[1]);
        if (ph >= 3 && S.mode === 'on')
          rank.slice(0, S.k).forEach((i) => {
            g.strokeStyle = t.gold;
            g.lineWidth = 1 + 5 * sims[i];
            g.beginPath();
            g.moveTo(qx, qy);
            g.lineTo(qx + (X(PC[i][0]) - qx) * flyU, qy + (Y(PC[i][1]) - qy) * flyU);
            g.stroke();
          });
        PC.forEach((p, i) => {
          const hot = top.has(i),
            s = ph >= 2 ? sims[i] : 0;
          g.beginPath();
          g.arc(X(p[0]), Y(p[1]), hot ? 8 : 5 + 6 * s, 0, 7);
          g.fillStyle = hot ? t.gold : mix(t.accent, t.accent, 0, 0.35 + 0.6 * Math.min(1, s * 2));
          g.fill();
          g.strokeStyle = t.bg;
          g.lineWidth = 1.5;
          g.stroke();
          g.font = MONO;
          g.fillStyle = hot ? t.cut : t.muted;
          g.textAlign = 'left';
          const lab =
            i +
            1 +
            ' ' +
            toks(docs[i])
              .sort((a, b) => idf[voc.indexOf(b)] - idf[voc.indexOf(a)])
              .slice(0, 2)
              .join(' ');
          g.fillText(lab, X(p[0]) + 10, Y(p[1]) + 4);
        });
        if (ph >= 1) markTP(g, qx, qy, t, false);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('each label: passage number and its two most distinctive words', 8, h - 8);
      }
      {
        const { g, w, h } = fit(cr),
          n = docs.length,
          rh = Math.min(24, (h - 30) / n),
          x0 = 88,
          W = w - x0 - 50;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          ph >= 2 ? 'cosine similarity to the question, best first' : 'the scores appear at step 3',
          8,
          13,
        );
        if (ph >= 2) {
          rank.forEach((i, r) => {
            const y = 22 + r * rh,
              hot = top.has(i);
            g.fillStyle = hot ? t.gold : mix(t.accent, t.accent, 0, 0.6);
            g.fillRect(x0, y + 3, Math.max(1, W * sims[i]), rh - 6);
            g.fillStyle = hot ? t.cut : t.muted;
            g.textAlign = 'right';
            g.fillText('passage ' + (i + 1), x0 - 6, y + rh / 2 + 4);
            g.textAlign = 'left';
            g.fillText(f2(sims[i]), x0 + W * sims[i] + 6, y + rh / 2 + 4);
          });
          g.setLineDash([4, 4]);
          g.strokeStyle = t.cut;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(x0 + W * TH, 20);
          g.lineTo(x0 + W * TH, 22 + n * rh);
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.cut;
          g.textAlign = 'left';
          g.fillText('below this line: too weak to trust', x0 + W * TH + 5, 22 + n * rh + 12);
        }
      }
      const pe = $('#rg-prompt', root);
      pe.textContent = ph >= 4 ? prompt() : 'The prompt is written at step 5.';
      pe.style.opacity = ph >= 4 ? 1 : 0.4;
      answer();
      live.textContent = MSG();
      uCur(root, 'rg', ph);
      setR(root, 'nd', String(docs.length));
      setR(root, 'nv', String(voc.length));
      setR(
        root,
        'qw',
        ph >= 1
          ? qtop
              .filter((x) => x[1] > 0)
              .map((x) => x[0])
              .join(', ') || 'none'
          : '–',
      );
      setR(root, 'best', ph >= 2 ? f2(sims[rank[0]]) : '–');
      setR(root, 'sent', ph >= 3 ? (S.mode === 'on' ? String(S.k) : '0') : '–');
      setR(root, 'pl', ph >= 4 ? String(prompt().split(/\s+/).length) : '–');
    };
    const go2 = (p) => {
      ph = p;
      if (p === 3) flyU = RM ? 1 : 0;
    };
    $('#rg-q', root).innerHTML = QS.map(
      (s, i) => '<button class="btn" type="button" data-v="' + i + '">' + s + '</button>',
    ).join('');
    inp.value = QS[0];
    index();
    ask();
    if (RM) ph = 5;
    const anim = Anim(() => {
      fr++;
      if (flyU < 1) {
        flyU = Math.min(1, flyU + 0.06);
        draw();
        return;
      }
      if (fr % 70) return;
      if (ph >= 5) {
        pb.stop();
        return;
      }
      go2(ph + 1);
      draw();
      if (ph >= 5) pb.stop();
    });
    pb = playBtn($('#rg-play', root), anim);
    const st0 = pb.start;
    pb.start = () => {
      if (ph >= 5) {
        ph = 0;
        draw();
      }
      st0();
    };
    const again = () => {
      pb.stop();
      ask();
      ph = 0;
      flyU = 1;
      draw();
      pb.start();
    };
    $('#rg-step', root).onclick = () => {
      pb.stop();
      go2(ph >= 5 ? 0 : ph + 1);
      flyU = 1;
      draw();
    };
    $('#rg-reset', root).onclick = () => {
      pb.stop();
      ask();
      ph = 0;
      draw();
    };
    segBind(root, 'rg-q', '0', (v) => {
      inp.value = QS[+v];
      again();
    });
    inp.addEventListener('change', again);
    inp.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        again();
      }
    });
    bind(
      root,
      'rg-k',
      (v) => {
        S.k = Math.min(+v, docs.length);
        if (ph >= 3) {
          flyU = 1;
          draw();
        }
      },
      (v) => String(v),
    );
    segBind(root, 'rg-mode', S.mode, (v) => {
      S.mode = v;
      draw();
    });
    $('#rg-addb', root).onclick = () => {
      const v = $('#rg-add', root).value.trim().replace(/\s+/g, ' ');
      if (v.split(' ').length < 4) {
        st.textContent = 'Write a full sentence or two, then press the button.';
        return;
      }
      if (docs.length >= 22) {
        st.textContent =
          'The collection is full at 22 passages. Go back to the original collection to add more.';
        return;
      }
      docs.push(v);
      $('#rg-add', root).value = '';
      index();
      st.textContent =
        'Added as passage ' + docs.length + ' and embedded. Now ask a question about it.';
      again();
    };
    $('#rg-resetc', root).onclick = () => {
      docs = BASE.slice();
      index();
      st.textContent = 'Back to the original 14 passages.';
      again();
    };
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        ask();
        ph = 0;
        draw();
        pb.start();
      },
    };
  },
};
