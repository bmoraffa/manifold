/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  SANS,
  readThemeColors,
  bind,
  clamp,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { UC, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#at-c', root),
      cr = $('#at-r', root),
      live = $('#at-live', root),
      eq = $('#at-eq', root),
      settings = { a: 0, p: 0, j: 0, o: 0, tau: 1 },
      AN = ['cat', 'dog', 'horse', 'bird', 'fox'],
      PL = ['mat', 'sofa', 'rug', 'bed', 'floor'],
      JA = ['tired', 'hungry', 'sleepy', 'happy'],
      JP = ['soft', 'warm', 'flat', 'dirty'],
      ADJ = JA.concat(JP),
      FW = ['the', 'sat', 'on', 'because', 'it', 'was'],
      WD = AN.concat(PL, FW, ADJ),
      wi = (w) => WD.indexOf(w);
    let Kt,
      Qt,
      TRN = [],
      TST = [],
      ep = 0,
      LL = [],
      AC = [],
      conv = false,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      t0 = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      explanationStep = 99,
      seen = new Set();
    const makeSentenceExample = (a, p, j, o) => {
      const w = o
        ? ['on', 'the', PL[p], 'sat', 'the', AN[a], 'because', 'it', 'was', ADJ[j]]
        : ['the', AN[a], 'sat', 'on', 'the', PL[p], 'because', 'it', 'was', ADJ[j]];
      return { w, ix: w.map(wi), antecedentIndex: j < 4 ? (o ? 5 : 1) : o ? 2 : 5, j };
    };
    const fill = (id, arr) => {
      $('#' + id, root).innerHTML = arr
        .map((w, i) => '<option value="' + i + '">' + w + '</option>')
        .join('');
    };
    fill('at-a', AN);
    fill('at-p', PL);
    fill('at-j', ADJ);
    const build = () => {
      const r = rng(run * 17 + 3),
        rn = randnFrom(r);
      Kt = WD.map(() => [0.25 * rn(), 0.25 * rn()]);
      Qt = ADJ.map(() => [0.25 * rn(), 0.25 * rn()]);
      TRN = [];
      TST = [];
      seen = new Set();
      for (let a = 0; a < 5; a++)
        for (let p = 0; p < 5; p++)
          for (let j = 0; j < 8; j++)
            for (let o = 0; o < 2; o++) {
              const key = a + '-' + p + '-' + j + '-' + o;
              if (r() < 0.4) {
                TRN.push(makeSentenceExample(a, p, j, o));
                seen.add(key);
              } else TST.push(makeSentenceExample(a, p, j, o));
            }
      ep = 0;
      LL = [];
      AC = [];
      conv = false;
      activeStepIndex = 0;
      explanationStep = 99;
      eq.innerHTML = '';
      msg =
        'Random keys and questions. ' +
        TRN.length +
        ' sentences will be used for training and ' +
        TST.length +
        ' are held back. Right now “it” listens to every word about equally. Press Train.';
    };
    const att = (s, tau = 1) => {
      const q = Qt[s.j],
        sc = s.ix.map((i) => (q[0] * Kt[i][0] + q[1] * Kt[i][1]) * tau),
        m = Math.max(...sc),
        e = sc.map((v) => Math.exp(v - m)),
        z = e.reduce((x, y) => x + y, 0);
      return { sc, al: e.map((v) => v / z) };
    };
    const accOn = (D) =>
      D.filter((s) => {
        const al = att(s).al;
        return al.indexOf(Math.max(...al)) === s.antecedentIndex;
      }).length / D.length;
    const pass = () => {
      let l = 0;
      const ord = TRN.map((_, i) => i);
      for (let i = ord.length - 1; i > 0; i--) {
        const k = Math.floor(Math.random() * (i + 1));
        [ord[i], ord[k]] = [ord[k], ord[i]];
      }
      for (const oi of ord) {
        const s = TRN[oi],
          { al } = att(s),
          q = Qt[s.j];
        l -= Math.log(al[s.antecedentIndex] + 1e-12);
        const dq = [0, 0];
        s.ix.forEach((w, pz) => {
          const d = al[pz] - (pz === s.antecedentIndex ? 1 : 0);
          dq[0] += d * Kt[w][0];
          dq[1] += d * Kt[w][1];
        });
        s.ix.forEach((w, pz) => {
          const d = al[pz] - (pz === s.antecedentIndex ? 1 : 0);
          Kt[w][0] -= 0.012 * d * q[0];
          Kt[w][1] -= 0.012 * d * q[1];
        });
        q[0] -= 0.012 * dq[0];
        q[1] -= 0.012 * dq[1];
      }
      ep++;
      LL.push(l / TRN.length);
      AC.push(accOn(TST));
      if (
        LL[ep - 1] < 0.06 ||
        ep >= 400 ||
        (ep > 60 && Math.abs(LL[ep - 1] - LL[ep - 31]) < 0.01 * LL[ep - 1])
      )
        conv = true;
      activeStepIndex = conv ? 4 : 1 + (ep % 3);
      msg =
        'Pass ' +
        ep +
        ': all ' +
        TRN.length +
        ' training sentences were shown once, and after each the keys and questions were nudged. Training error ' +
        f3(LL[ep - 1]) +
        '. On the ' +
        TST.length +
        ' sentences it has never seen, it picks the right word in ' +
        pct(AC[ep - 1]) +
        '.' +
        (conv ? ' Converged: training stops. Change the words above to test it.' : '');
    };
    const cur = () => makeSentenceExample(settings.a, settings.p, settings.j, settings.o);
    const explain = () => {
      const s = cur(),
        { sc, al } = att(s, settings.tau),
        q = Qt[s.j],
        b = al.indexOf(Math.max(...al)),
        nf = (v) => (v < 0 ? '(−' + Math.abs(v).toFixed(2) + ')' : v.toFixed(2));
      if (explanationStep <= 10) {
        const j = explanationStep - 1,
          k = Kt[s.ix[j]];
        activeStepIndex = 1;
        msg =
          'The question of “' +
          s.w[9] +
          '” is compared with the key of “' +
          s.w[j] +
          '”: score ' +
          f2(sc[j]) +
          '.';
        eq.innerHTML =
          '<div class="on">score(“' +
          s.w[j] +
          '”) = question · key × sharpness</div><div class="on">   = (' +
          nf(q[0]) +
          '·' +
          nf(k[0]) +
          ' + ' +
          nf(q[1]) +
          '·' +
          nf(k[1]) +
          ') × ' +
          f2(settings.tau) +
          ' = <b>' +
          f2(sc[j]) +
          '</b></div>';
      } else if (explanationStep === 11) {
        activeStepIndex = 2;
        msg =
          'Softmax turns the ten scores into shares that add up to 1. “' +
          s.w[b] +
          '” gets ' +
          (100 * al[b]).toFixed(0) +
          '%.';
        eq.innerHTML =
          '<div class="on">share(“' +
          s.w[b] +
          '”) = e^' +
          f2(sc[b]) +
          ' / ( ' +
          sc
            .map((v) => 'e^' + f2(v))
            .slice(0, 4)
            .join(' + ') +
          ' + … ) = <b>' +
          f2(al[b]) +
          '</b></div>';
      } else {
        activeStepIndex = 4;
        msg =
          'So in this sentence “it” means “' +
          s.w[b] +
          '”' +
          (b === s.antecedentIndex
            ? ', which is right.'
            : ', which is wrong: it should be “' + s.w[s.antecedentIndex] + '”.') +
          ' The layer was never given a rule; it placed the keys and questions so that the scores come out this way.';
        eq.innerHTML =
          '<div class="on">“it” takes ' +
          (100 * al[b]).toFixed(0) +
          '% of its new meaning from <b>“' +
          s.w[b] +
          '”</b></div>';
      }
    };
    const draw = () => {
      const colors = readThemeColors(),
        col = UC(colors),
        s = cur(),
        { sc, al } = att(s, settings.tau),
        n = 10,
        u = prefersReducedMotion ? 1 : clamp((frameCount - t0) / 22, 0, 1),
        ex = explanationStep <= 12,
        ns = ex ? Math.min(explanationStep, 10) : 10,
        shares = !ex || explanationStep > 10;
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          yw = h * 0.4,
          bw = (w - 20) / n,
          X = (j) => 10 + j * bw + bw / 2,
          sm = Math.max(0.6, ...sc.map(Math.abs)),
          by = h - 30,
          bh = h * 0.26;
        ctx.font = SANS;
        if (shares)
          for (let j = 0; j < n; j++) {
            if (j === 7) continue;
            const wt = al[j] * (explanationStep === 11 ? u : 1);
            ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.2 + 0.8 * wt);
            ctx.lineWidth = 0.5 + 13 * wt;
            ctx.beginPath();
            ctx.moveTo(X(7), yw - 16);
            ctx.quadraticCurveTo(
              (X(7) + X(j)) / 2,
              yw - 16 - Math.min(h * 0.32, Math.abs(X(j) - X(7)) * 0.45),
              X(j),
              yw - 16,
            );
            ctx.stroke();
          }
        for (let j = 0; j < n; j++) {
          const x = X(j),
            me = j === 7,
            tgt = j === s.antecedentIndex;
          ctx.font = SANS;
          const tw = Math.max(34, ctx.measureText(s.w[j]).width + 12);
          ctx.fillStyle = me ? colors.ink : colors.bg;
          ctx.strokeStyle = tgt ? colors.accent : colors.ink;
          ctx.lineWidth = tgt ? 2.6 : 1.3;
          ctx.beginPath();
          ctx.rect(x - tw / 2, yw - 14, tw, 28);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = me ? colors.bg : colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText(s.w[j], x, yw + 5);
          ctx.font = MONO;
          if (!shares) {
            if (j < ns) {
              const v = sc[j] * (j === ns - 1 ? u : 1),
                hh = (Math.abs(v) / sm) * bh * 0.5;
              ctx.fillStyle = v >= 0 ? colors.accent : colors.ok;
              ctx.fillRect(
                x - bw * 0.28,
                v >= 0 ? by - bh * 0.5 - hh : by - bh * 0.5,
                bw * 0.56,
                Math.max(1, hh),
              );
              ctx.fillStyle = colors.muted;
              ctx.fillText(f2(sc[j]), x, by + 14);
            }
          } else {
            const v = al[j] * (explanationStep === 11 ? u : 1),
              hh = v * bh;
            ctx.fillStyle = colors.gold;
            ctx.fillRect(x - bw * 0.28, by - hh, bw * 0.56, Math.max(1, hh));
            ctx.fillStyle = colors.muted;
            ctx.fillText((100 * al[j]).toFixed(0) + '%', x, by + 14);
          }
        }
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          shares
            ? 'share of attention that “it” gives each word'
            : 'match score of each word’s key with the question',
          10,
          by - bh - 8,
        );
        ctx.fillStyle = colors.accent;
        ctx.textAlign = 'right';
        ctx.fillText('green frame: the correct answer', w - 10, yw + 34);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.58,
          ox = w / 2,
          oy = h1 / 2 + 6;
        let m = 1e-9;
        Kt.forEach((k) => {
          m = Math.max(m, Math.abs(k[0]), Math.abs(k[1]));
        });
        Qt.forEach((k) => {
          m = Math.max(m, Math.abs(k[0]), Math.abs(k[1]));
        });
        const sx = (w / 2 - 50) / m,
          sy = (h1 / 2 - 20) / m,
          P = (v) => [ox + v[0] * sx, oy - v[1] * sy];
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(8, oy);
        ctx.lineTo(w - 8, oy);
        ctx.moveTo(ox, 8);
        ctx.lineTo(ox, h1 - 4);
        ctx.stroke();
        ADJ.forEach((a, j) => {
          const p = P(Qt[j]),
            me = j === settings.j;
          ctx.strokeStyle = me
            ? colors.gold
            : mix(j < 4 ? colors.accent : colors.cut, colors.bg, 0, 0.55);
          ctx.lineWidth = me ? 3.2 : 1.3;
          ctx.beginPath();
          ctx.moveTo(ox, oy);
          ctx.lineTo(p[0], p[1]);
          ctx.stroke();
          ctx.font = MONO;
          ctx.fillStyle = me ? colors.cut : colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText(a + '?', p[0], p[1] + (p[1] < oy ? -5 : 12));
        });
        WD.forEach((wd, i) => {
          if (i >= 16) return;
          const p = P(Kt[i]),
            inS = s.ix.includes(i);
          ctx.fillStyle = i < 5 ? colors.accent : i < 10 ? colors.gold : colors.muted;
          ctx.beginPath();
          ctx.arc(p[0], p[1], inS ? 5.5 : 3.5, 0, 7);
          ctx.fill();
          if (inS) {
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
          ctx.font = inS ? SANS : MONO;
          ctx.fillStyle = inS ? colors.ink : colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(wd, p[0] + 7, p[1] + 4);
        });
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.accent;
        ctx.fillText('● animals', 8, 14);
        ctx.fillStyle = colors.cut;
        ctx.fillText('● places', 84, 14);
        ctx.fillStyle = colors.muted;
        ctx.fillText('● other words     lines: each adjective’s question', 150, 14);
        const U = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, LL, {
          title: 'training error (red) and accuracy on unseen sentences (green, 0 to 100%)',
          n: Math.max(40, LL.length),
          col: colors.ok,
          ym: Math.max(2.4, LL[0] || 2.4),
        });
        if (AC.length) {
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          AC.forEach((v, i) => {
            const y = U.Y(v * Math.max(2.4, LL[0] || 2.4));
            i ? ctx.lineTo(U.X(i), y) : ctx.moveTo(U.X(i), y);
          });
          ctx.stroke();
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'at', activeStepIndex);
      const b = al.indexOf(Math.max(...al));
      setReadout(root, 'ep', String(ep));
      setReadout(root, 'L', ep ? f3(LL[ep - 1]) : '–');
      setReadout(root, 'te', ep ? pct(AC[ep - 1]) : pct(accOn(TST)));
      setReadout(
        root,
        'ans',
        '“' +
          s.w[b] +
          '” (' +
          (100 * al[b]).toFixed(0) +
          '%)' +
          (b === s.antecedentIndex ? ', correct' : ', wrong'),
      );
      setReadout(
        root,
        'seen',
        seen.has(settings.a + '-' + settings.p + '-' + settings.j + '-' + settings.o)
          ? 'Yes'
          : 'No: never seen',
      );
    };
    build();
    if (prefersReducedMotion) while (!conv) pass();
    const anim = Anim(() => {
      frameCount++;
      if (!conv) {
        if (frameCount % (ep < 8 ? 18 : 5) === 0) {
          pass();
          t0 = frameCount;
          if (conv) {
            explanationStep = 1;
            explain();
          }
          checkRunStop();
        }
        draw();
        return;
      }
      if (explanationStep <= 12) {
        if (frameCount - t0 > (explanationStep <= 10 ? 14 : 110)) {
          explanationStep++;
          t0 = frameCount;
          if (explanationStep > 12) {
            draw();
            playback.stop();
            return;
          }
          explain();
        }
        draw();
        return;
      }
      draw();
      playback.stop();
    });
    playback = playBtn($('#at-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'at', {
      done: () => conv && explanationStep > 12,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'passes',
    });
    $('#at-step', root).onclick = () => {
      playback.stop();
      if (conv) {
        run++;
        build();
      } else for (let i = 0; i < 5 && !conv; i++) pass();
      draw();
    };
    $('#at-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    $('#at-exp', root).onclick = () => {
      playback.stop();
      if (!conv) {
        msg = 'Train first: the keys and questions are still random.';
        draw();
        return;
      }
      explanationStep = 1;
      t0 = frameCount;
      explain();
      const st = playback.start;
      anim.start();
      $('#at-play', root).textContent = 'Pause';
      $('#at-play', root).classList.add('on');
    };
    const chg = () => {
      explanationStep = 99;
      eq.innerHTML = '';
      const s = cur(),
        al = att(s, settings.tau).al,
        b = al.indexOf(Math.max(...al));
      if (conv)
        msg =
          '“' +
          s.w.join(' ') +
          '”: “it” gives ' +
          (100 * al[b]).toFixed(0) +
          '% of its attention to “' +
          s.w[b] +
          '”' +
          (b === s.antecedentIndex ? ', the right word.' : ', which is wrong.') +
          (seen.has(settings.a + '-' + settings.p + '-' + settings.j + '-' + settings.o)
            ? ''
            : ' This exact sentence was never in the training set.');
      if (!playback.on) draw();
    };
    [
      ['at-a', 'a'],
      ['at-p', 'p'],
      ['at-j', 'j'],
      ['at-o', 'o'],
    ].forEach(([id, k]) => {
      $('#' + id, root).onchange = (e) => {
        settings[k] = +e.target.value;
        chg();
      };
    });
    bind(
      root,
      'at-tau',
      (v) => {
        settings.tau = v;
        chg();
      },
      f2,
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
