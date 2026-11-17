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
  fit,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { GWS, gwDraw, gwMake, gwMax, gwRoute, rlN } from '../shared.js';
import { uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#sa-c', root),
      cr = $('#sa-r', root),
      live = $('#sa-live', root),
      eq = $('#sa-eq', root),
      S = { eps: 0.1, al: 0.5, sp: 2 },
      ROWS = ['..........', '..........', '..........', 'SXXXXXXXXG'],
      MAXE = 500,
      CAP = 400;
    let G,
      A = [],
      ep = 0,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      fin = false;
    const pick = (Q, s) => {
      if (Math.random() < S.eps) return { a: Math.floor(Math.random() * 4), ex: true };
      const row = [0, 1, 2, 3].map((a) => Q[s * 4 + a]),
        mx = Math.max(...row),
        ties = [0, 1, 2, 3].filter((k) => row[k] >= mx - 1e-12);
      return { a: ties[Math.floor(Math.random() * ties.length)], ex: false };
    };
    const mk = (sarsa) => {
      const o = {
        sarsa,
        Q: new Float64Array(G.n * 4),
        s: G.start,
        R: 0,
        n: 0,
        RET: [],
        FALL: [0],
        falls: 0,
        trail: [G.start],
        done: false,
        last: null,
      };
      o.nx = pick(o.Q, o.s);
      return o;
    };
    const build = () => {
      G = gwMake(ROWS, { rp: -100, rg: -1, back: true });
      A = [mk(false), mk(true)];
      ep = 0;
      fin = false;
      cu = 0;
      msg = 'Two tables of zeros, two robots in the bottom-left corner. Press Run.';
      eqShow(
        eq,
        ['Nothing has been updated yet. Press <b>One step</b> to see both rules applied.'],
        0,
      );
    };
    const stepA = (o) => {
      if (o.done) return;
      const s = o.s,
        a = o.nx.a,
        ex = o.nx.ex,
        r = G.step(s, a),
        s2 = r.s2,
        old = o.Q[s * 4 + a];
      let nxt = null,
        tv;
      if (r.done) tv = 0;
      else if (o.sarsa) {
        nxt = pick(o.Q, s2);
        tv = o.Q[s2 * 4 + nxt.a];
      } else tv = gwMax(o.Q, s2);
      const tg = r.r - 0 + tv,
        nw = old + S.al * (tg - old);
      o.Q[s * 4 + a] = nw;
      o.R += r.r;
      o.n++;
      if (G.ch(r.raw) === 'X') {
        o.falls++;
        o.trail = [G.start];
      } else o.trail.push(s2);
      o.last = {
        s,
        a,
        ex,
        r: r.r,
        s2,
        old,
        nw,
        tv,
        tg,
        nxt,
        done: r.done,
        fell: G.ch(r.raw) === 'X',
      };
      o.s = s2;
      if (!o.sarsa && !r.done) nxt = pick(o.Q, s2);
      o.nx = nxt;
      if (r.done || o.n >= CAP) {
        o.done = true;
      }
    };
    const endEp = () => {
      for (const o of A) {
        o.RET.push(o.R);
        o.FALL.push(o.falls);
        o.R = 0;
        o.n = 0;
        o.s = G.start;
        o.trail = [G.start];
        o.done = false;
        o.nx = pick(o.Q, o.s);
      }
      ep++;
      if (ep >= MAXE) {
        fin = true;
        cu = 5;
        const rq = gwRoute(G, A[0].Q, G.start),
          rs = gwRoute(G, A[1].Q, G.start),
          av = (o) => o.RET.slice(-50).reduce((x, y) => x + y, 0) / 50;
        msg =
          'Finished: 500 episodes each. Q-learning recommends a route of ' +
          (rq.ok ? rq.acts.length + ' steps' : 'no complete route') +
          ' and averaged ' +
          f2(av(A[0])) +
          ' per episode over the last 50. SARSA recommends ' +
          (rs.ok ? rs.acts.length + ' steps' : 'no complete route') +
          ' and averaged ' +
          f2(av(A[1])) +
          '. Falls: ' +
          A[0].falls +
          ' and ' +
          A[1].falls +
          '.';
      }
    };
    const line = (o) => {
      const l = o.last;
      if (!l) return '';
      const nm = o.sarsa ? 'SARSA' : 'Q-learning';
      return [
        '<b>' +
          nm +
          '</b>: in ' +
          G.name(l.s) +
          ' it ' +
          (l.ex ? 'explored' : 'took its best move') +
          ', ' +
          GWS[l.a] +
          '. ' +
          (l.fell
            ? 'It stepped off the cliff: reward −100, back to the start.'
            : 'Reward ' + rlN(l.r).replace('.00', '') + ', now in ' + G.name(l.s2) + '.'),
        l.done
          ? '   the goal ends the episode: target = ' + rlN(l.r)
          : o.sarsa
            ? '   its next move will be ' +
              GWS[l.nxt.a] +
              (l.nxt.ex ? ' (a random, exploring move)' : ' (its best move)') +
              ', worth ' +
              rlN(l.tv) +
              ':  target = ' +
              rlN(l.r) +
              ' + ' +
              rlN(l.tv) +
              ' = <b>' +
              rlN(l.tg) +
              '</b>'
            : '   the best move from there is worth ' +
              rlN(l.tv) +
              ':  target = ' +
              rlN(l.r) +
              ' + ' +
              rlN(l.tv) +
              ' = <b>' +
              rlN(l.tg) +
              '</b>',
        '   Q(' +
          G.name(l.s) +
          ', ' +
          GWS[l.a] +
          ') = ' +
          rlN(l.old) +
          ' + ' +
          f2(S.al) +
          ' × (' +
          rlN(l.tg) +
          ' − ' +
          rlN(l.old) +
          ') = <b>' +
          rlN(l.nw) +
          '</b>',
      ];
    };
    const stepBoth = (quiet) => {
      if (fin) return;
      for (const o of A) stepA(o);
      if (!quiet) {
        eqShow(eq, line(A[0]).concat(line(A[1])), 0);
        const d = A[1].last && A[1].last.nxt && A[1].last.nxt.ex;
        cu = d ? 3 : 4;
        msg =
          'Episode ' +
          (ep + 1) +
          '. ' +
          (A.some((o) => o.last && o.last.fell) ? 'A robot fell off the cliff. ' : '') +
          (d
            ? 'SARSA’s next move is an exploring one, so its target uses the value of that random move. Q-learning would have used the best move here.'
            : 'Each robot moved one square and updated one entry of its table.');
      }
      if (A.every((o) => o.done)) endEp();
    };
    const draw = () => {
      const t = T();
      {
        const { g, w, h } = fit(c),
          hh = (h - 34) / 2;
        g.font = MONO;
        A.forEach((o, i) => {
          const y0 = 16 + i * (hh + 18),
            R = gwRoute(G, o.Q, G.start);
          g.font = MONO;
          g.fillStyle = t.ink;
          g.textAlign = 'left';
          g.fillText(
            (i ? 'SARSA' : 'Q-LEARNING') +
              ': ' +
              (R.ok ? 'recommended route, ' + R.acts.length + ' steps' : 'no complete route yet'),
            6,
            y0 - 4,
          );
          gwDraw(g, t, 4, y0, w - 8, hh, G, {
            Q: o.Q,
            agent: o.s,
            trail: R.ok ? R.path : null,
            noname: true,
            nonum: true,
            xname: 'cliff',
            vm: 30,
            acol: i ? t.ok : t.accent,
          });
        });
      }
      {
        const { g, w, h } = fit(cr),
          sm = (o) =>
            o.RET.map((_, i) => {
              const a = o.RET.slice(Math.max(0, i - 19), i + 1);
              return Math.max(-120, a.reduce((x, y) => x + y, 0) / a.length);
            }),
          n = MAXE,
          L = uLine(g, t, 0, 0, w, h * 0.56, sm(A[0]), {
            title: 'reward per episode while learning (average of the last 20)',
            n,
            y1: -120,
            ym: 0,
            col: t.accent,
          }),
          s1 = sm(A[1]);
        g.strokeStyle = t.ok;
        g.lineWidth = 2.2;
        g.beginPath();
        s1.forEach((v, i) => (i ? g.lineTo(L.X(i), L.Y(v)) : g.moveTo(L.X(i), L.Y(v))));
        g.stroke();
        g.font = MONO;
        g.textAlign = 'right';
        g.fillStyle = t.accent;
        g.fillText('Q-learning', w - 14, h * 0.56 - 40);
        g.fillStyle = t.ok;
        g.fillText('SARSA', w - 14, h * 0.56 - 26);
        const fm = Math.max(10, A[0].falls, A[1].falls),
          L2 = uLine(g, t, 0, h * 0.56, w, h * 0.44, A[0].FALL, {
            title: 'falls off the cliff so far',
            n: n + 1,
            ym: fm,
            col: t.accent,
          });
        g.strokeStyle = t.ok;
        g.lineWidth = 2.2;
        g.beginPath();
        A[1].FALL.forEach((v, i) => (i ? g.lineTo(L2.X(i), L2.Y(v)) : g.moveTo(L2.X(i), L2.Y(v))));
        g.stroke();
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('episodes →', w - 12, h - 5);
      }
      live.textContent = msg;
      uCur(root, 'sa', cu);
      const av = (o) =>
          o.RET.length
            ? f2(o.RET.slice(-50).reduce((x, y) => x + y, 0) / Math.min(50, o.RET.length))
            : '–',
        rt = (o) => {
          const R = gwRoute(G, o.Q, G.start);
          if (!R.ok) return 'not found yet';
          const edge = R.path.filter((q) => Math.floor(q / G.C) === 2).length;
          return R.acts.length + ' steps, ' + edge + ' of them along the cliff edge';
        };
      setR(root, 'ep', ep + (fin ? ' (stopped)' : ''));
      setR(root, 'rq', av(A[0]));
      setR(root, 'rs', av(A[1]));
      setR(root, 'pq', rt(A[0]));
      setR(root, 'ps', rt(A[1]));
      setR(root, 'fq', A[0].falls + ' / ' + A[1].falls);
    };
    build();
    if (RM) {
      while (!fin) stepBoth(true);
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      if (S.sp === 1) {
        if (fr % 20) return;
        stepBoth(false);
      } else if (S.sp === 2) {
        if (ep < 1) {
          for (let i = 0; i < 3 && ep < 1 && chk.remaining() > 0; i++) stepBoth(false);
        } else {
          const e0 = ep;
          let k = 0;
          while (ep === e0 && !fin && k++ < 2 * CAP && chk.remaining() > 0) stepBoth(true);
        }
      } else {
        const e0 = ep;
        let k = 0;
        while (ep < e0 + 5 && !fin && k++ < 12 * CAP && chk.remaining() > 0) stepBoth(true);
      }
      draw();
      chk();
    });
    pb = playBtn($('#sa-play', root), anim);
    chk = runCtl(pb, root, 'sa', {
      done: () => fin,
      restart: () => build(),
      count: () => ep,
      what: 'episodes',
    });
    $('#sa-step', root).onclick = () => {
      pb.stop();
      if (fin) build();
      else stepBoth(false);
      draw();
    };
    $('#sa-reset', root).onclick = () => {
      pb.stop();
      build();
      draw();
    };
    bind(
      root,
      'sa-ep2',
      (v) => {
        S.eps = +v;
      },
      f2,
    );
    bind(
      root,
      'sa-al',
      (v) => {
        S.al = +v;
      },
      f2,
    );
    bind(
      root,
      'sa-sp',
      (v) => {
        S.sp = +v;
      },
      (v) => ['', 'one step at a time', 'one episode per frame', 'five episodes per frame'][+v],
    );
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
