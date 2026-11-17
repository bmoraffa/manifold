/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  eqShow,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { gwClick } from '../../reinforcement/shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#hm-c', root),
      cr = $('#hm-r', root),
      live = $('#hm-live', root),
      eq = $('#hm-eq', root),
      settings = { v: 'fwd', st: 0.8, em: 1, show: false },
      TT = 14,
      ON = ['walk', 'shop', 'clean'],
      SN = ['sunny', 'rainy'];
    let Z = [],
      X = [],
      AL = [],
      predictedStateProbs = [],
      viterbiScores = [],
      BP = [],
      smoothedStateProbs = [],
      VP = [],
      t = 0,
      ll = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      fin = false,
      P = null;
    const Bm = () => {
        const e = settings.em,
          u = 1 / 3;
        return [
          [u + e * (0.6 - u), u + e * (0.3 - u), u + e * (0.1 - u)],
          [u + e * (0.1 - u), u + e * (0.3 - u), u + e * (0.6 - u)],
        ];
      },
      A = () => [
        [settings.st, 1 - settings.st],
        [1 - settings.st, settings.st],
      ];
    const gen = () => {
      const a = A(),
        b = Bm();
      Z = [];
      X = [];
      let z = Math.random() < 0.5 ? 0 : 1;
      for (let i = 0; i < TT; i++) {
        if (i) z = Math.random() < a[z][z] ? z : 1 - z;
        Z.push(z);
        const u = Math.random();
        X.push(u < b[z][0] ? 0 : u < b[z][0] + b[z][1] ? 1 : 2);
      }
    };
    const smooth = () => {
      const a = A(),
        b = Bm(),
        al = [],
        cs = [];
      let p = [0.5, 0.5];
      for (let i = 0; i < TT; i++) {
        const pr = i
            ? [p[0] * a[0][0] + p[1] * a[1][0], p[0] * a[0][1] + p[1] * a[1][1]]
            : [0.5, 0.5],
          u = [pr[0] * b[0][X[i]], pr[1] * b[1][X[i]]],
          s = u[0] + u[1];
        p = [u[0] / s, u[1] / s];
        al.push(p);
        cs.push(s);
      }
      let be = [1, 1];
      const out = new Array(TT);
      for (let i = TT - 1; i >= 0; i--) {
        const g = [al[i][0] * be[0], al[i][1] * be[1]],
          s = g[0] + g[1];
        out[i] = [g[0] / s, g[1] / s];
        if (i) {
          const nb = [0, 1].map((q) => a[q][0] * b[0][X[i]] * be[0] + a[q][1] * b[1][X[i]] * be[1]);
          const m = nb[0] + nb[1];
          be = [nb[0] / m, nb[1] / m];
        }
      }
      return out;
    };
    const reset = () => {
      AL = [];
      predictedStateProbs = [];
      viterbiScores = [];
      BP = [];
      VP = [];
      t = 0;
      ll = 0;
      fin = false;
      activeStepIndex = 0;
      smoothedStateProbs = smooth();
      msg =
        'Fourteen evenings of reports are in the top row. Nothing has been worked out yet. Press One day or Run.';
      eqShow(eq, ['No days processed yet.'], 0);
    };
    const step = () => {
      if (fin) return;
      const a = A(),
        b = Bm(),
        x = X[t],
        d = t + 1,
        L = [
          'Day ' +
            d +
            ': she went to <b>' +
            (x === 0 ? 'walk' : x === 1 ? 'the shops' : 'clean the house') +
            '</b>.',
        ];
      {
        const p = t ? AL[t - 1] : [0.5, 0.5],
          pr = t ? [p[0] * a[0][0] + p[1] * a[1][0], p[0] * a[0][1] + p[1] * a[1][1]] : [0.5, 0.5],
          u = [pr[0] * b[0][x], pr[1] * b[1][x]],
          s = u[0] + u[1];
        predictedStateProbs.push(pr);
        AL.push([u[0] / s, u[1] / s]);
        ll += Math.log(s);
        if (settings.v === 'fwd') {
          L.push(
            t
              ? 'predict: P(sunny today) = ' +
                  f3(p[0]) +
                  ' × ' +
                  f2(a[0][0]) +
                  ' + ' +
                  f3(p[1]) +
                  ' × ' +
                  f2(a[1][0]) +
                  ' = <b>' +
                  f3(pr[0]) +
                  '</b>,  so P(rainy) = ' +
                  f3(pr[1])
              : 'before hearing from her: sunny 0.500, rainy 0.500',
            'weigh by her activity:  sunny ' +
              f3(pr[0]) +
              ' × P(' +
              ON[x] +
              ' | sunny) ' +
              f2(b[0][x]) +
              ' = ' +
              f3(u[0]) +
              '     rainy ' +
              f3(pr[1]) +
              ' × P(' +
              ON[x] +
              ' | rainy) ' +
              f2(b[1][x]) +
              ' = ' +
              f3(u[1]),
            'rescale to add up to 1:  sunny ' +
              f3(u[0]) +
              ' ÷ ' +
              f3(s) +
              ' = <b>' +
              f3(u[0] / s) +
              '</b>     rainy <b>' +
              f3(u[1] / s) +
              '</b>',
          );
          activeStepIndex = t ? 2 : 0;
          msg =
            'Day ' +
            d +
            ' (' +
            ON[x] +
            '): before her report the model expected sun with probability ' +
            pct(pr[0]) +
            '. After it, ' +
            pct(u[0] / s) +
            '.' +
            (Math.abs(b[0][x] - b[1][x]) < 0.02
              ? ' This activity is equally likely in any weather, so the report changed nothing: only the prediction step moved the belief.'
              : '');
        }
      }
      {
        const p = t ? viterbiScores[t - 1] : null,
          de = [0, 0],
          bp = [0, 0];
        for (let j = 0; j < 2; j++) {
          if (!t) {
            de[j] = 0.5 * b[j][x];
            bp[j] = -1;
          } else {
            const c0 = p[0] * a[0][j],
              c1 = p[1] * a[1][j];
            bp[j] = c0 >= c1 ? 0 : 1;
            de[j] = Math.max(c0, c1) * b[j][x];
          }
        }
        const s = de[0] + de[1];
        viterbiScores.push([de[0] / s, de[1] / s]);
        BP.push(bp);
        if (settings.v === 'vit') {
          if (t)
            for (let j = 0; j < 2; j++)
              L.push(
                'best story ending ' +
                  SN[j] +
                  ' today:  max( from sunny ' +
                  f3(p[0]) +
                  ' × ' +
                  f2(a[0][j]) +
                  ',  from rainy ' +
                  f3(p[1]) +
                  ' × ' +
                  f2(a[1][j]) +
                  ' ) × P(' +
                  ON[x] +
                  ' | ' +
                  SN[j] +
                  ') ' +
                  f2(b[j][x]) +
                  '   → came from <b>' +
                  SN[bp[j]] +
                  '</b>',
              );
          else
            L.push(
              'day 1 has no yesterday:  sunny 0.5 × ' +
                f2(b[0][x]) +
                ',  rainy 0.5 × ' +
                f2(b[1][x]),
            );
          L.push('scores, rescaled:  sunny ' + f3(de[0] / s) + '   rainy ' + f3(de[1] / s));
          activeStepIndex = 4;
          msg =
            'Day ' +
            d +
            ' (' +
            ON[x] +
            '): for each weather, the best story ending in it today was extended by one day and the link back was remembered.';
        }
      }
      t++;
      if (t >= TT) {
        fin = true;
        let j = viterbiScores[TT - 1][0] >= viterbiScores[TT - 1][1] ? 0 : 1;
        VP = new Array(TT);
        for (let i = TT - 1; i >= 0; i--) {
          VP[i] = j;
          j = BP[i][j];
        }
        const ok = acc();
        activeStepIndex = settings.v === 'vit' ? 5 : 3;
        if (settings.v === 'vit')
          L.push(
            'Last day reached. The better ending is ' +
              SN[VP[TT - 1]] +
              '. Following the remembered links back gives the gold line.',
          );
        msg +=
          ' The fortnight is finished. ' +
          (settings.v === 'vit'
            ? 'The single best story gets ' + ok[2] + ' of ' + TT + ' days right.'
            : 'Judging each day as it came, the model gets ' +
              ok[0] +
              ' of ' +
              TT +
              ' days right. With hindsight it gets ' +
              ok[1] +
              '.') +
          ' Press Show the True Weather to compare.';
      }
      eqShow(eq, L, 0);
    };
    const acc = () => {
      let f = 0,
        s = 0,
        v = 0;
      for (let i = 0; i < TT; i++) {
        if (AL[i] && (AL[i][1] > 0.5 ? 1 : 0) === Z[i]) f++;
        if ((smoothedStateProbs[i][1] > 0.5 ? 1 : 0) === Z[i]) s++;
        if (VP.length && VP[i] === Z[i]) v++;
      }
      return [f, s, v];
    };
    const draw = () => {
      const tt = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          m = 58,
          cw = (w - m - 10) / TT,
          yo = 34,
          ys = [h * 0.42, h * 0.66],
          R = Math.min(cw * 0.4, 26),
          fam = MONO.split('px ')[1];
        P = { m, cw, yo };
        ctx.font = MONO;
        ctx.textAlign = 'right';
        ctx.fillStyle = tt.muted;
        ctx.fillText('she did', m - 8, yo + 4);
        ctx.fillText('sunny', m - 8, ys[0] + 4);
        ctx.fillText('rainy', m - 8, ys[1] + 4);
        if (settings.show) ctx.fillText('truth', m - 8, h * 0.86 + 4);
        for (let i = 0; i < TT; i++) {
          const cx = m + (i + 0.5) * cw;
          if (i === t - 1) {
            ctx.fillStyle = mix(tt.gold, tt.gold, 0, 0.18);
            ctx.fillRect(cx - cw / 2, 8, cw, h - 30);
          }
          ctx.textAlign = 'center';
          ctx.fillStyle = tt.ink;
          ctx.font = '600 ' + Math.max(9, Math.min(12, cw * 0.26)) + 'px ' + fam;
          ctx.fillText(ON[X[i]], cx, yo + 4);
          ctx.strokeStyle = tt.line;
          ctx.lineWidth = 1;
          ctx.strokeRect(cx - cw / 2 + 2, yo - 14, cw - 4, 26);
          ctx.font = MONO;
          ctx.fillStyle = tt.muted;
          ctx.fillText(String(i + 1), cx, h - 8);
          for (let j = 0; j < 2; j++) {
            const pr = i < t ? (settings.v === 'fwd' ? AL[i][j] : viterbiScores[i][j]) : 0;
            ctx.beginPath();
            ctx.arc(cx, ys[j], R, 0, 7);
            ctx.strokeStyle = tt.muted;
            ctx.lineWidth = 1;
            ctx.stroke();
            if (i < t) {
              ctx.beginPath();
              ctx.arc(cx, ys[j], R * Math.sqrt(pr), 0, 7);
              ctx.fillStyle = j ? mix(tt.accent, tt.accent, 0, 0.85) : mix(tt.cut, tt.cut, 0, 0.85);
              ctx.fill();
              if (R > 15) {
                ctx.fillStyle = tt.ink;
                ctx.fillText((100 * pr).toFixed(0), cx, ys[j] + R + 12);
              }
            }
          }
          if (settings.v === 'vit' && i < t && i > 0)
            for (let j = 0; j < 2; j++) {
              const cx0 = m + (i - 0.5) * cw;
              ctx.strokeStyle = mix(tt.ink, tt.ink, 0, 0.35);
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(cx0 + R, ys[BP[i][j]]);
              ctx.lineTo(cx - R, ys[j]);
              ctx.stroke();
            }
          if (settings.show) {
            ctx.fillStyle = Z[i] ? tt.accent : tt.cut;
            ctx.fillRect(cx - cw / 2 + 3, h * 0.86 - 9, cw - 6, 18);
            ctx.fillStyle = tt.bg;
            ctx.fillText(Z[i] ? 'rain' : 'sun', cx, h * 0.86 + 4);
          }
        }
        const path = settings.v === 'vit' ? (fin ? VP : []) : AL.map((p) => (p[1] > 0.5 ? 1 : 0));
        if (path.length) {
          ctx.strokeStyle = tt.gold;
          ctx.lineWidth = 3;
          ctx.lineJoin = 'round';
          ctx.beginPath();
          path.forEach((j, i) => {
            const cx = m + (i + 0.5) * cw;
            i ? ctx.lineTo(cx, ys[j]) : ctx.moveTo(cx, ys[j]);
          });
          ctx.stroke();
        }
        ctx.textAlign = 'left';
        ctx.fillStyle = tt.muted;
        ctx.fillText('day', 8, h - 8);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          L = drawSeriesChart(
            ctx,
            tt,
            0,
            0,
            w,
            h1,
            AL.map((p) => p[1]),
            {
              title: 'probability of rain on each day',
              n: TT,
              ym: 1,
              col: tt.accent,
              dots: true,
            },
          );
        if (fin) {
          ctx.strokeStyle = tt.ok;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          smoothedStateProbs.forEach((p, i) =>
            i ? ctx.lineTo(L.X(i), L.Y(p[1])) : ctx.moveTo(L.X(i), L.Y(p[1])),
          );
          ctx.stroke();
        }
        if (settings.show)
          Z.forEach((z, i) => {
            if (z) {
              ctx.fillStyle = mix(tt.accent, tt.accent, 0, 0.14);
              ctx.fillRect(
                L.X(i) - (w - L.m.l - L.m.r) / (TT - 1) / 2,
                L.Y(1),
                (w - L.m.l - L.m.r) / (TT - 1),
                L.Y(0) - L.Y(1),
              );
            }
          });
        ctx.font = MONO;
        ctx.textAlign = 'right';
        ctx.fillStyle = tt.ok;
        ctx.fillText(fin ? 'with hindsight' : '(hindsight: at the end)', w - 12, 13);
        ctx.fillStyle = tt.accent;
        ctx.fillText('on the day', w - 12 - (fin ? 104 : 160), 13);
        const a = A(),
          b = Bm(),
          y0 = h1 + 18,
          fam = MONO.split('px ')[1];
        ctx.textAlign = 'left';
        ctx.fillStyle = tt.muted;
        ctx.fillText('how the weather changes overnight', 8, y0);
        ctx.fillText('what she does in each weather', w / 2 + 4, y0);
        ctx.font = '12px ' + fam;
        ctx.fillStyle = tt.ink;
        [
          ['', '→ sunny', '→ rainy'],
          ['sunny', f2(a[0][0]), f2(a[0][1])],
          ['rainy', f2(a[1][0]), f2(a[1][1])],
        ].forEach((r, i) =>
          r.forEach((v, j) => ctx.fillText(v, 12 + j * ((w / 2 - 20) / 3), y0 + 22 + i * 20)),
        );
        [
          ['', 'walk', 'shop', 'clean'],
          ['sunny', ...b[0].map(f2)],
          ['rainy', ...b[1].map(f2)],
        ].forEach((r, i) =>
          r.forEach((v, j) =>
            ctx.fillText(v, w / 2 + 8 + j * ((w / 2 - 20) / 4), y0 + 22 + i * 20),
          ),
        );
      }
      live.textContent = msg;
      highlightLessonStep(root, 'hm', activeStepIndex);
      const ok = acc(),
        cur = t ? (settings.v === 'fwd' ? AL[t - 1] : viterbiScores[t - 1]) : null;
      setReadout(root, 'd', t + ' of ' + TT);
      setReadout(root, 'b', cur ? pct(cur[0]) + ' / ' + pct(cur[1]) : '–');
      setReadout(
        root,
        'af',
        fin && settings.show ? ok[0] + ' of ' + TT : fin ? 'press Show the True Weather' : '–',
      );
      setReadout(
        root,
        'as',
        fin && settings.show ? ok[1] + ' of ' + TT : fin ? 'press Show the True Weather' : '–',
      );
      setReadout(
        root,
        'av',
        fin && settings.show ? ok[2] + ' of ' + TT : fin ? 'press Show the True Weather' : '–',
      );
      setReadout(root, 'lk', t ? Math.exp(ll).toExponential(2) : '–');
    };
    gen();
    reset();
    if (prefersReducedMotion) {
      while (!fin) step();
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (frameCount % 46) return;
      step();
      draw();
      if (fin) playback.stop();
    });
    playback = playBtn($('#hm-play', root), anim);
    attachRunLimit(playback, root, 'hm', { done: () => fin, restart: () => reset() });
    $('#hm-step', root).onclick = () => {
      playback.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#hm-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    $('#hm-new', root).onclick = () => {
      playback.stop();
      gen();
      reset();
      draw();
      playback.start();
    };
    $('#hm-tr', root).onclick = (e) => {
      settings.show = !settings.show;
      e.target.textContent = settings.show ? 'Hide the True Weather' : 'Show the True Weather';
      draw();
    };
    segBind(root, 'hm-v', settings.v, (v) => {
      settings.v = v;
      playback.stop();
      reset();
      draw();
      playback.start();
    });
    bind(
      root,
      'hm-st',
      (v) => {
        settings.st = +v;
        if (playback) {
          playback.stop();
          reset();
          draw();
        }
      },
      f2,
    );
    bind(
      root,
      'hm-em',
      (v) => {
        settings.em = +v;
        if (playback) {
          playback.stop();
          reset();
          draw();
        }
      },
      f2,
    );
    gwClick(mainCanvas, (px, py) => {
      if (!P || py > P.yo + 16 || py < P.yo - 18) return;
      const i = Math.floor((px - P.m) / P.cw);
      if (i < 0 || i >= TT) return;
      playback.stop();
      X[i] = (X[i] + 1) % 3;
      reset();
      msg =
        'Day ' +
        (i + 1) +
        ' was changed to “' +
        ON[X[i]] +
        '”. The true weather was not changed. Press Run to work through the fortnight again.';
      draw();
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
