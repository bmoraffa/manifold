/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  eqShow,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  randnFrom,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { UC, segBind, uCur, uLine } from '../../unsupervised/shared.js';
import { gwClick } from '../shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#mb-c', root),
      cr = $('#mb-r', root),
      live = $('#mb-live', root),
      eq = $('#mb-eq', root),
      S = { w: 2, eps: 0.1, sp: 2, rev: false },
      K = 5,
      TMAX = 1500,
      NM = ['Greedy', 'ε-greedy', 'UCB', 'Thompson sampling', 'You'];
    let MU,
      best,
      PL,
      t = 0,
      ty = 0,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      fin = false,
      P = null;
    const rnorm = randnFrom(Math.random),
      rgam = (k) => {
        const d = k - 1 / 3,
          cc = 1 / Math.sqrt(9 * d);
        for (;;) {
          let x, v;
          do {
            x = rnorm();
            v = 1 + cc * x;
          } while (v <= 0);
          v = v * v * v;
          const u = Math.random();
          if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
        }
      },
      rbeta = (a, b) => {
        const x = rgam(a),
          y = rgam(b);
        return x / (x + y);
      };
    const build = () => {
      MU = [0.2, 0.35, 0.5, 0.6, 0.8]
        .map((v) => [Math.random(), v])
        .sort((a, b) => a[0] - b[0])
        .map((v) => v[1]);
      best = MU.indexOf(Math.max(...MU));
      PL = NM.map(() => ({
        n: new Int32Array(K),
        w: new Int32Array(K),
        reg: [0],
        bs: 0,
        good: 0,
        last: -1,
        lw: 0,
        info: null,
      }));
      t = 0;
      ty = 0;
      fin = false;
      cu = 0;
      S.rev = false;
      msg =
        'Five trucks, five hidden chances of a good lunch. Press Run to let the four strategies play, or choose Play Yourself and click a truck.';
      eqShow(eq, ['No lunches yet.'], 0);
    };
    const est = (p, k) => (p.n[k] ? p.w[k] / p.n[k] : 0);
    const choose = (i, p, day) => {
      const un = [...Array(K).keys()].filter((k) => !p.n[k]);
      if (i === 0) {
        if (un.length)
          return {
            k: un[0],
            why: 'It has not tried truck ' + (un[0] + 1) + ' yet, so it tries it once.',
          };
        let b = 0;
        for (let k = 1; k < K; k++) if (est(p, k) > est(p, b)) b = k;
        return {
          k: b,
          why:
            'Best record so far: truck ' +
            (b + 1) +
            ' with ' +
            pct(est(p, b)) +
            '. Greedy goes there and nowhere else.',
        };
      }
      if (i === 1) {
        if (Math.random() < S.eps) {
          const k = Math.floor(Math.random() * K);
          return {
            k,
            why:
              'It rolled a number below ε = ' +
              f2(S.eps) +
              ', so it <b>explores</b>: a truck chosen at random, number ' +
              (k + 1) +
              '.',
          };
        }
        let b = 0;
        for (let k = 1; k < K; k++) if (est(p, k) > est(p, b)) b = k;
        return {
          k: b,
          why:
            'It <b>exploits</b>: the best record so far is truck ' +
            (b + 1) +
            ' with ' +
            pct(est(p, b)) +
            '.',
        };
      }
      if (i === 2) {
        if (un.length)
          return {
            k: un[0],
            why:
              'It has not tried truck ' +
              (un[0] + 1) +
              ' yet. An untried truck has an unlimited bonus, so it goes there.',
          };
        const sc = [...Array(K).keys()].map(
          (k) => est(p, k) + Math.sqrt(Math.log(day) / (2 * p.n[k])),
        );
        let b = 0;
        for (let k = 1; k < K; k++) if (sc[k] > sc[b]) b = k;
        return { k: b, sc, es: [...Array(K).keys()].map((k) => est(p, k)), why: null };
      }
      const dr = [...Array(K).keys()].map((k) => rbeta(1 + p.w[k], 1 + p.n[k] - p.w[k]));
      let b = 0;
      for (let k = 1; k < K; k++) if (dr[k] > dr[b]) b = k;
      return { k: b, dr, why: null };
    };
    const pull = (i, k, info) => {
      const p = PL[i],
        win = Math.random() < MU[k] ? 1 : 0;
      p.n[k]++;
      p.w[k] += win;
      p.good += win;
      p.last = k;
      p.lw = win;
      p.info = info;
      p.reg.push(p.reg[p.reg.length - 1] + MU[best] - MU[k]);
      if (k === best) p.bs++;
    };
    const explain = (i) => {
      const p = PL[i],
        o = p.info,
        k = p.last,
        day = i === 4 ? ty : t,
        L = ['Day ' + day + ', ' + NM[i] + '.'];
      if (i === 4) L.push('You chose truck ' + (k + 1) + '.');
      else if (o.why) L.push(o.why);
      else if (o.sc) {
        for (let j = 0; j < K; j++)
          L.push(
            'truck ' +
              (j + 1) +
              ':  record ' +
              f3(o.es[j]) +
              '  +  bonus √(ln ' +
              day +
              ' ÷ (2 × ' +
              (p.n[j] - (j === k ? 1 : 0)) +
              '))  =  ' +
              f3(o.sc[j]) +
              (j === k ? '   <b>← highest score</b>' : ''),
          );
      } else {
        for (let j = 0; j < K; j++)
          L.push(
            'truck ' +
              (j + 1) +
              ':  ' +
              (p.w[j] - (j === k ? p.lw : 0)) +
              ' good, ' +
              (p.n[j] - p.w[j] - (j === k ? 1 - p.lw : 0)) +
              ' bad so far.  A plausible success rate drawn from that: ' +
              f3(o.dr[j]) +
              (j === k ? '   <b>← highest draw</b>' : ''),
          );
      }
      L.push(
        'Lunch at truck ' +
          (k + 1) +
          ': <b>' +
          (p.lw ? 'good' : 'bad') +
          '</b>. Its record there is now ' +
          p.w[k] +
          ' good out of ' +
          p.n[k] +
          ' = ' +
          pct(est(p, k)) +
          '.',
      );
      return L;
    };
    const day = (quiet) => {
      if (fin) return;
      t++;
      for (let i = 0; i < 4; i++) {
        const o = choose(i, PL[i], t);
        if (i === 2 && o.sc) {
          /* scores use counts before the pull */
        }
        pull(i, o.k, o);
      }
      if (!quiet && S.w < 4) {
        eqShow(eq, explain(S.w), 0);
        const p = PL[S.w];
        cu = S.w < 2 ? 3 : 4;
        msg =
          'Day ' +
          t +
          ': ' +
          NM[S.w] +
          ' went to truck ' +
          (p.last + 1) +
          ' and had a ' +
          (p.lw ? 'good' : 'bad') +
          ' lunch. Its regret so far is ' +
          f2(p.reg[p.reg.length - 1]) +
          ' good lunches.';
      }
      if (t >= TMAX) {
        fin = true;
        cu = 5;
        const o = [0, 1, 2, 3].sort((a, b) => PL[a].reg[t] - PL[b].reg[t]);
        msg =
          'Finished: 1,500 days. Good lunches missed, least first: ' +
          o.map((i) => NM[i] + ' ' + f2(PL[i].reg[t])).join(', ') +
          '. The best truck was number ' +
          (best + 1) +
          ', where ' +
          pct(MU[best]) +
          ' of lunches are good.' +
          (ty ? ' You missed ' + f2(PL[4].reg[ty]) + ' in ' + ty + ' days.' : '');
        S.rev = true;
      }
    };
    const draw = () => {
      const T0 = T(),
        col = UC(T0),
        w0 = S.w,
        p = PL[w0];
      {
        const { g, w, h } = fit(c),
          m = { l: 36, r: 10, t: 26, b: 54 },
          cw = (w - m.l - m.r) / K,
          Y = (v) => h - m.b - v * (h - m.t - m.b);
        g.font = MONO;
        g.fillStyle = T0.muted;
        g.textAlign = 'left';
        g.fillText(
          NM[w0] +
            (w0 === 4 ? ': click a truck to have lunch there' : '’s view of the five trucks'),
          8,
          14,
        );
        g.strokeStyle = T0.line;
        g.lineWidth = 1;
        for (const v of [0, 0.25, 0.5, 0.75, 1]) {
          g.beginPath();
          g.moveTo(m.l, Y(v));
          g.lineTo(w - m.r, Y(v));
          g.stroke();
          g.textAlign = 'right';
          g.fillText(100 * v + '%', m.l - 5, Y(v) + 4);
        }
        for (let k = 0; k < K; k++) {
          const x = m.l + k * cw,
            cx = x + cw / 2,
            e = est(p, k),
            bw = cw * 0.36;
          g.fillStyle = mix(T0.accent, T0.accent, 0, p.n[k] ? 0.8 : 0.15);
          g.fillRect(cx - bw / 2, Y(e), bw, Y(0) - Y(e));
          if (w0 === 2 && p.n[k] && t > 0) {
            const u = Math.min(1.25, e + Math.sqrt(Math.log(Math.max(2, t)) / (2 * p.n[k])));
            g.strokeStyle = T0.cut;
            g.lineWidth = 2;
            g.beginPath();
            g.moveTo(cx, Y(e));
            g.lineTo(cx, Y(Math.min(1.12, u)));
            g.moveTo(cx - 8, Y(Math.min(1.12, u)));
            g.lineTo(cx + 8, Y(Math.min(1.12, u)));
            g.stroke();
          }
          if (w0 === 3 || w0 === 4) {
            const a = 1 + p.w[k],
              b = 1 + p.n[k] - p.w[k];
            let mx = 0;
            const D = [];
            for (let i = 0; i <= 60; i++) {
              const x2 = (i + 0.5) / 61,
                d = Math.exp((a - 1) * Math.log(x2) + (b - 1) * Math.log(1 - x2));
              D.push(d);
              mx = Math.max(mx, d);
            }
            g.fillStyle = mix(T0.cut, T0.cut, 0, 0.45);
            g.beginPath();
            g.moveTo(cx + bw / 2 + 3, Y(0));
            D.forEach((d, i) =>
              g.lineTo(cx + bw / 2 + 3 + (d / mx) * cw * 0.26, Y((i + 0.5) / 61)),
            );
            g.lineTo(cx + bw / 2 + 3, Y(1));
            g.closePath();
            g.fill();
            if (w0 === 3 && p.info && p.info.dr) {
              g.fillStyle = T0.ink;
              g.beginPath();
              g.arc(cx + bw / 2 + 3, Y(p.info.dr[k]), 3.5, 0, 7);
              g.fill();
            }
          }
          if (S.rev) {
            g.setLineDash([5, 4]);
            g.strokeStyle = T0.gold;
            g.lineWidth = 2;
            g.beginPath();
            g.moveTo(x + 6, Y(MU[k]));
            g.lineTo(x + cw - 6, Y(MU[k]));
            g.stroke();
            g.setLineDash([]);
          }
          if (p.last === k) {
            g.strokeStyle = T0.gold;
            g.lineWidth = 3;
            g.strokeRect(x + 4, m.t - 4, cw - 8, h - m.t - 12);
            g.fillStyle = p.lw ? T0.ok : T0.muted;
            g.textAlign = 'center';
            g.fillText(p.lw ? 'good lunch' : 'bad lunch', cx, m.t + 10);
          }
          g.textAlign = 'center';
          g.fillStyle = T0.ink;
          g.font = '600 12px ' + MONO.split('px ')[1];
          g.fillText('truck ' + (k + 1), cx, h - m.b + 16);
          g.font = MONO;
          g.fillStyle = T0.muted;
          g.fillText(p.n[k] ? p.w[k] + ' good of ' + p.n[k] : 'not tried', cx, h - m.b + 30);
          g.fillText(p.n[k] ? pct(e) : '', cx, h - m.b + 43);
        }
        P = { m, cw };
        g.textAlign = 'left';
        g.fillStyle = T0.muted;
        if (w0 === 2) g.fillText('whisker: record plus UCB bonus', m.l + 4, m.t + 10 + 0 * 12);
        if (w0 >= 3)
          g.fillText(
            'shape: belief about the true chance' + (w0 === 3 ? '; dot: today’s draw' : ''),
            m.l + 4,
            h - 6,
          );
        if (S.rev) {
          g.fillStyle = T0.cut;
          g.textAlign = 'right';
          g.fillText('dashed: the true chance', w - m.r, 14);
        }
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.62,
          n = TMAX + 1,
          ym = Math.max(12, ...PL.map((q) => q.reg[q.reg.length - 1])),
          ids = [0, 1, 2, 3].concat(ty ? [4] : []);
        let L = null;
        ids.forEach((i, j) => {
          L = uLine(g, T0, 0, 0, w, h1, PL[i].reg, {
            title: j ? '' : 'regret: good lunches missed so far',
            n,
            ym,
            col: col[i],
          });
        });
        g.font = MONO;
        ids.forEach((i) => {
          const q = PL[i],
            k = q.reg.length - 1;
          g.fillStyle = col[i];
          g.textAlign = 'left';
          g.fillText(NM[i], Math.min(L.X(k) + 5, w - 128), L.Y(q.reg[k]) - 6);
        });
        const y0 = h1 + 18,
          bh = Math.min(17, (h - h1 - 26) / 5);
        g.fillStyle = T0.muted;
        g.textAlign = 'left';
        g.fillText('share of days spent at the best truck', 8, h1 + 12);
        ids.forEach((i, j) => {
          const q = PL[i],
            d = i === 4 ? ty : t,
            sh = d ? q.bs / d : 0,
            x0 = 120;
          g.fillStyle = col[i];
          g.fillRect(x0, y0 + j * bh + 1, Math.max(1, (w - x0 - 60) * sh), bh - 3);
          g.fillStyle = T0.muted;
          g.textAlign = 'right';
          g.fillText(NM[i].replace(' sampling', ''), x0 - 6, y0 + j * bh + bh - 5);
          g.textAlign = 'left';
          g.fillText(d ? pct(sh) : '–', x0 + (w - x0 - 60) * sh + 5, y0 + j * bh + bh - 5);
        });
      }
      live.textContent = msg;
      uCur(root, 'mb', cu);
      const d = w0 === 4 ? ty : t;
      let fv = -1;
      for (let k = 0; k < K; k++) if (p.n[k] && (fv < 0 || est(p, k) > est(p, fv))) fv = k;
      setR(root, 'd', t + ' of ' + TMAX + (ty ? ' (you: ' + ty + ')' : ''));
      setR(root, 'w', NM[w0]);
      setR(root, 'g', p.good + ' of ' + d);
      setR(root, 'rg', f2(p.reg[p.reg.length - 1]));
      setR(root, 'bs', d ? (S.rev ? pct(p.bs / d) : 'hidden until you reveal') : '–');
      setR(root, 'fav', fv < 0 ? '–' : 'truck ' + (fv + 1) + ' (' + pct(est(p, fv)) + ')');
    };
    build();
    if (RM) {
      while (!fin) day(true);
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      if (S.sp === 1) {
        if (fr % 40) return;
        day(false);
      } else if (S.sp === 2) {
        for (let i = 0; i < 3 && !fin && chk.remaining() > 0; i++) day(i < 2);
      } else {
        for (let i = 0; i < 15 && !fin && chk.remaining() > 0; i++) day(i < 14);
      }
      draw();
      chk();
    });
    pb = playBtn($('#mb-play', root), anim);
    chk = runCtl(pb, root, 'mb', {
      done: () => fin,
      restart: () => build(),
      count: () => t,
      what: 'days',
    });
    $('#mb-step', root).onclick = () => {
      pb.stop();
      if (fin) build();
      else day(false);
      draw();
    };
    $('#mb-reset', root).onclick = () => {
      pb.stop();
      build();
      draw();
    };
    $('#mb-rev', root).onclick = () => {
      S.rev = !S.rev;
      draw();
    };
    segBind(root, 'mb-s', String(S.w), (v) => {
      S.w = +v;
      if (S.w === 4) {
        pb.stop();
        msg =
          'Your turn: click a truck in the left picture to have lunch there. Your regret is drawn with the others.';
      }
      if (PL[S.w].info || (S.w === 4 && ty))
        eqShow(eq, PL[S.w].last >= 0 ? explain(S.w) : ['No lunches yet.'], 0);
      draw();
    });
    bind(
      root,
      'mb-e',
      (v) => {
        S.eps = +v;
      },
      f2,
    );
    bind(
      root,
      'mb-sp',
      (v) => {
        S.sp = +v;
      },
      (v) => ['', 'slow', 'steady', 'fast'][+v],
    );
    gwClick(c, (x, y) => {
      if (S.w !== 4 || !P || ty >= TMAX) return;
      const k = Math.floor((x - P.m.l) / P.cw);
      if (k < 0 || k >= K) return;
      ty++;
      pull(4, k, {});
      eqShow(eq, explain(4), 0);
      msg =
        'Day ' +
        ty +
        ' for you: truck ' +
        (k + 1) +
        ', a ' +
        (PL[4].lw ? 'good' : 'bad') +
        ' lunch. You have missed ' +
        f2(PL[4].reg[ty]) +
        ' good lunches so far compared with always choosing the best truck.';
      draw();
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        build();
        draw();
        pb.start();
      },
    };
  },
};
