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
  prepareCanvas,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { ACTION_ARROWS, gwDraw, gwMake, gwMax, gwRoute, rlN } from '../shared.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#sa-c', root),
      cr = $('#sa-r', root),
      live = $('#sa-live', root),
      eq = $('#sa-eq', root),
      settings = { eps: 0.1, al: 0.5, sp: 2 },
      ROWS = ['..........', '..........', '..........', 'SXXXXXXXXG'],
      MAXE = 500,
      CAP = 400;
    let G,
      agents = [],
      ep = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false;
    const pick = (Q, s) => {
      if (Math.random() < settings.eps) return { a: Math.floor(Math.random() * 4), ex: true };
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
      o.nextActionChoice = pick(o.Q, o.s);
      return o;
    };
    const build = () => {
      G = gwMake(ROWS, { rp: -100, rg: -1, back: true });
      agents = [mk(false), mk(true)];
      ep = 0;
      fin = false;
      activeStepIndex = 0;
      msg = 'Two tables of zeros, two robots in the bottom-left corner. Press Run.';
      eqShow(
        eq,
        ['Nothing has been updated yet. Press <b>One step</b> to see both rules applied.'],
        0,
      );
    };
    const stepAgent = (o) => {
      if (o.done) return;
      const s = o.s,
        a = o.nextActionChoice.a,
        ex = o.nextActionChoice.ex,
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
        nw = old + settings.al * (tg - old);
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
      o.nextActionChoice = nxt;
      if (r.done || o.n >= CAP) {
        o.done = true;
      }
    };
    const endEp = () => {
      for (const o of agents) {
        o.RET.push(o.R);
        o.FALL.push(o.falls);
        o.R = 0;
        o.n = 0;
        o.s = G.start;
        o.trail = [G.start];
        o.done = false;
        o.nextActionChoice = pick(o.Q, o.s);
      }
      ep++;
      if (ep >= MAXE) {
        fin = true;
        activeStepIndex = 5;
        const rq = gwRoute(G, agents[0].Q, G.start),
          rs = gwRoute(G, agents[1].Q, G.start),
          av = (o) => o.RET.slice(-50).reduce((x, y) => x + y, 0) / 50;
        msg =
          'Finished: 500 episodes each. Q-learning recommends a route of ' +
          (rq.ok ? rq.acts.length + ' steps' : 'no complete route') +
          ' and averaged ' +
          f2(av(agents[0])) +
          ' per episode over the last 50. SARSA recommends ' +
          (rs.ok ? rs.acts.length + ' steps' : 'no complete route') +
          ' and averaged ' +
          f2(av(agents[1])) +
          '. Falls: ' +
          agents[0].falls +
          ' and ' +
          agents[1].falls +
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
          ACTION_ARROWS[l.a] +
          '. ' +
          (l.fell
            ? 'It stepped off the cliff: reward −100, back to the start.'
            : 'Reward ' + rlN(l.r).replace('.00', '') + ', now in ' + G.name(l.s2) + '.'),
        l.done
          ? '   the goal ends the episode: target = ' + rlN(l.r)
          : o.sarsa
            ? '   its next move will be ' +
              ACTION_ARROWS[l.nxt.a] +
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
          ACTION_ARROWS[l.a] +
          ') = ' +
          rlN(l.old) +
          ' + ' +
          f2(settings.al) +
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
      for (const o of agents) stepAgent(o);
      if (!quiet) {
        eqShow(eq, line(agents[0]).concat(line(agents[1])), 0);
        const d = agents[1].last && agents[1].last.nxt && agents[1].last.nxt.ex;
        activeStepIndex = d ? 3 : 4;
        msg =
          'Episode ' +
          (ep + 1) +
          '. ' +
          (agents.some((o) => o.last && o.last.fell) ? 'A robot fell off the cliff. ' : '') +
          (d
            ? 'SARSA’s next move is an exploring one, so its target uses the value of that random move. Q-learning would have used the best move here.'
            : 'Each robot moved one square and updated one entry of its table.');
      }
      if (agents.every((o) => o.done)) endEp();
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          hh = (h - 34) / 2;
        ctx.font = MONO;
        agents.forEach((o, i) => {
          const y0 = 16 + i * (hh + 18),
            R = gwRoute(G, o.Q, G.start);
          ctx.font = MONO;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'left';
          ctx.fillText(
            (i ? 'SARSA' : 'Q-LEARNING') +
              ': ' +
              (R.ok ? 'recommended route, ' + R.acts.length + ' steps' : 'no complete route yet'),
            6,
            y0 - 4,
          );
          gwDraw(ctx, colors, 4, y0, w - 8, hh, G, {
            Q: o.Q,
            agent: o.s,
            trail: R.ok ? R.path : null,
            noname: true,
            nonum: true,
            xname: 'cliff',
            vm: 30,
            acol: i ? colors.ok : colors.accent,
          });
        });
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          sm = (o) =>
            o.RET.map((_, i) => {
              const a = o.RET.slice(Math.max(0, i - 19), i + 1);
              return Math.max(-120, a.reduce((x, y) => x + y, 0) / a.length);
            }),
          n = MAXE,
          L = drawSeriesChart(ctx, colors, 0, 0, w, h * 0.56, sm(agents[0]), {
            title: 'reward per episode while learning (average of the last 20)',
            n,
            y1: -120,
            ym: 0,
            col: colors.accent,
          }),
          s1 = sm(agents[1]);
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        s1.forEach((v, i) => (i ? ctx.lineTo(L.X(i), L.Y(v)) : ctx.moveTo(L.X(i), L.Y(v))));
        ctx.stroke();
        ctx.font = MONO;
        ctx.textAlign = 'right';
        ctx.fillStyle = colors.accent;
        ctx.fillText('Q-learning', w - 14, h * 0.56 - 40);
        ctx.fillStyle = colors.ok;
        ctx.fillText('SARSA', w - 14, h * 0.56 - 26);
        const fm = Math.max(10, agents[0].falls, agents[1].falls),
          L2 = drawSeriesChart(ctx, colors, 0, h * 0.56, w, h * 0.44, agents[0].FALL, {
            title: 'falls off the cliff so far',
            n: n + 1,
            ym: fm,
            col: colors.accent,
          });
        ctx.strokeStyle = colors.ok;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        agents[1].FALL.forEach((v, i) =>
          i ? ctx.lineTo(L2.X(i), L2.Y(v)) : ctx.moveTo(L2.X(i), L2.Y(v)),
        );
        ctx.stroke();
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('episodes →', w - 12, h - 5);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'sa', activeStepIndex);
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
      setReadout(root, 'ep', ep + (fin ? ' (stopped)' : ''));
      setReadout(root, 'rq', av(agents[0]));
      setReadout(root, 'rs', av(agents[1]));
      setReadout(root, 'pq', rt(agents[0]));
      setReadout(root, 'ps', rt(agents[1]));
      setReadout(root, 'fq', agents[0].falls + ' / ' + agents[1].falls);
    };
    build();
    if (prefersReducedMotion) {
      while (!fin) stepBoth(true);
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (settings.sp === 1) {
        if (frameCount % 20) return;
        stepBoth(false);
      } else if (settings.sp === 2) {
        if (ep < 1) {
          for (let i = 0; i < 3 && ep < 1 && checkRunStop.remaining() > 0; i++) stepBoth(false);
        } else {
          const e0 = ep;
          let k = 0;
          while (ep === e0 && !fin && k++ < 2 * CAP && checkRunStop.remaining() > 0) stepBoth(true);
        }
      } else {
        const e0 = ep;
        let k = 0;
        while (ep < e0 + 5 && !fin && k++ < 12 * CAP && checkRunStop.remaining() > 0)
          stepBoth(true);
      }
      draw();
      checkRunStop();
    });
    playback = playBtn($('#sa-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'sa', {
      done: () => fin,
      restart: () => build(),
      count: () => ep,
      what: 'episodes',
    });
    $('#sa-step', root).onclick = () => {
      playback.stop();
      if (fin) build();
      else stepBoth(false);
      draw();
    };
    $('#sa-reset', root).onclick = () => {
      playback.stop();
      build();
      draw();
    };
    bind(
      root,
      'sa-ep2',
      (v) => {
        settings.eps = +v;
      },
      f2,
    );
    bind(
      root,
      'sa-al',
      (v) => {
        settings.al = +v;
      },
      f2,
    );
    bind(
      root,
      'sa-sp',
      (v) => {
        settings.sp = +v;
      },
      (v) => ['', 'one step at a time', 'one episode per frame', 'five episodes per frame'][+v],
    );
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        build();
        draw();
        playback.start();
      },
    };
  },
};
