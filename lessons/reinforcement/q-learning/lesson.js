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
import {
  ACTION_NAMES,
  ACTION_ARROWS,
  gwBest,
  gwClick,
  gwDraw,
  gwMake,
  gwMax,
  gwQstar,
  gwRoute,
  rlN,
} from '../shared.js';
import { segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#ql-c', root),
      cr = $('#ql-r', root),
      live = $('#ql-live', root),
      eq = $('#ql-eq', root),
      tab = $('#ql-tab', root),
      settings = { w: 'room', al: 0.5, ga: 0.9, eps: 0.2, slip: 0, sp: 2 },
      W = {
        hall: ['S...G'],
        room: ['...G', '.#.X', 'S...'],
        maze: ['S..#..', '.#...#', '.#.#..', '...#.X', '.#...G'],
      },
      MAXE = 400,
      CAP = 200;
    let G,
      Q,
      Qs,
      s,
      ep = 0,
      stp = 0,
      epR = 0,
      RET = [],
      meanQErrorHistory = [],
      maxQChangeHistory = [],
      episodeMaxQChange = 0,
      trail = [],
      last = null,
      sel = -1,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false,
      P = null,
      tv = 0,
      tvd = -1,
      best0 = NaN,
      lastSeq = 0,
      lastL = null;
    const gap = () => {
      let e = 0,
        n = 0;
      for (const q of G.states) {
        if (G.term(q)) continue;
        for (let a = 0; a < 4; a++) {
          e += Math.abs(Q[q * 4 + a] - Qs[q * 4 + a]);
          n++;
        }
      }
      return e / n;
    };
    const okShare = () => {
      let k = 0,
        n = 0;
      for (const q of G.states) {
        if (G.term(q)) continue;
        for (let a = 0; a < 4; a++) {
          if (Math.abs(Q[q * 4 + a] - Qs[q * 4 + a]) <= 0.1) k++;
          n++;
        }
      }
      return k + ' of ' + n;
    };
    const truth = () => {
      Qs = gwQstar(G, settings.ga);
      const R = gwRoute(gwMake(G.rows, { slip: 0 }), Qs, G.start);
      best0 = R.ok ? R.ret : NaN;
    };
    const build = () => {
      G = gwMake(W[settings.w], { slip: settings.slip });
      Q = new Float64Array(G.n * 4);
      truth();
      s = G.start;
      ep = 0;
      stp = 0;
      epR = 0;
      RET = [];
      meanQErrorHistory = [gap()];
      maxQChangeHistory = [];
      episodeMaxQChange = 0;
      trail = [s];
      last = null;
      fin = false;
      activeStepIndex = 0;
      tv++;
      sel = -1;
      msg =
        'The table is all zeros and the robot is on the start square, ' +
        G.name(s) +
        '. It knows nothing about this world. Press One step to follow each update, or Run.';
      eqShow(
        eq,
        [
          'Nothing has been updated yet.',
          'Press <b>One step</b>: the robot makes one move and one entry of the table is recomputed here, line by line.',
        ],
        0,
      );
    };
    const step = (quiet) => {
      if (fin) return;
      const row = [0, 1, 2, 3].map((a) => Q[s * 4 + a]),
        mx = Math.max(...row),
        explore = Math.random() < settings.eps;
      let a,
        ties = [0, 1, 2, 3].filter((k) => row[k] >= mx - 1e-12);
      if (explore) a = Math.floor(Math.random() * 4);
      else a = ties[Math.floor(Math.random() * ties.length)];
      const o = G.step(s, a),
        old = Q[s * 4 + a],
        s2 = o.s2,
        nx = o.done ? 0 : gwMax(Q, s2),
        tg = o.r + settings.ga * nx,
        nw = old + settings.al * (tg - old);
      Q[s * 4 + a] = nw;
      episodeMaxQChange = Math.max(episodeMaxQChange, Math.abs(nw - old));
      stp++;
      epR += o.r;
      trail.push(s2);
      last = { s, a, s2, old, nw, ep: ep + 1, stp };
      tv++;
      {
        const L = [
          'Episode ' +
            (ep + 1) +
            ', step ' +
            stp +
            '. The robot is in <b>' +
            G.name(s) +
            '</b>. Its row of the table: ' +
            row.map((v, k) => ACTION_ARROWS[k] + ' ' + rlN(v)).join('   '),
          explore
            ? 'It rolled a random number below ε = ' +
              f2(settings.eps) +
              ', so it <b>explores</b>: a random move, ' +
              ACTION_ARROWS[a] +
              ' ' +
              ACTION_NAMES[a] +
              '.'
            : 'It <b>exploits</b>: the highest value in the row is ' +
              rlN(mx) +
              (ties.length > 1
                ? ', shared by ' + ties.length + ' moves, so one of them is picked at random: '
                : ', which is ') +
              ACTION_ARROWS[a] +
              ' ' +
              ACTION_NAMES[a] +
              '.',
          (o.slipped ? 'The floor was slippery and it slid sideways. ' : '') +
            (o.bump
              ? 'It bumped into a wall and stayed in ' + G.name(s2)
              : 'It lands in ' + G.name(s2)) +
            '. Reward: <b>' +
            rlN(o.r).replace('.00', '') +
            '</b>.',
          o.done
            ? G.name(s2) + ' ends the episode, so there is no future to add: best next value = 0'
            : 'Best value on offer from ' +
              G.name(s2) +
              ':  max( ' +
              [0, 1, 2, 3]
                .map(
                  (k) =>
                    ACTION_ARROWS[k] +
                    ' ' +
                    rlN(Q[s2 * 4 + k] - (s2 === s && k === a ? nw - old : 0)),
                )
                .join(',  ') +
              ' ) = ' +
              rlN(nx),
          'target = reward + γ × best next = ' +
            rlN(o.r) +
            ' + ' +
            f2(settings.ga) +
            ' × ' +
            rlN(nx) +
            ' = <b>' +
            rlN(tg) +
            '</b>',
          'new Q(' +
            G.name(s) +
            ', ' +
            ACTION_ARROWS[a] +
            ') = old + α × (target − old) = ' +
            rlN(old) +
            ' + ' +
            f2(settings.al) +
            ' × (' +
            rlN(tg) +
            ' − ' +
            rlN(old) +
            ') = <b>' +
            rlN(nw) +
            '</b>',
        ];
        lastL = L;
        if (!quiet) eqShow(eq, L, 0);
        activeStepIndex = o.done ? 5 : 4;
        msg =
          'The entry for ' +
          ACTION_ARROWS[a] +
          ' ' +
          ACTION_NAMES[a] +
          ' in row ' +
          G.name(s) +
          ' changed from ' +
          rlN(old) +
          ' to ' +
          rlN(nw) +
          '.' +
          (o.done
            ? (G.ch(o.raw) === 'G'
                ? ' The robot reached the goal.'
                : ' The robot fell into the pit.') +
              ' Episode ' +
              (ep + 1) +
              ' is over with a total reward of ' +
              rlN(epR).replace('.00', '') +
              '. Back to the start.'
            : '');
      }
      s = s2;
      if (o.done || stp >= CAP) {
        RET.push(epR);
        meanQErrorHistory.push(gap());
        maxQChangeHistory.push(episodeMaxQChange);
        ep++;
        const lastR = epR;
        episodeMaxQChange = 0;
        stp = 0;
        epR = 0;
        s = G.start;
        trail = [s];
        const w = maxQChangeHistory.slice(-25);
        if (ep >= MAXE || (maxQChangeHistory.length >= 25 && Math.max(...w) < 0.01)) {
          fin = true;
          activeStepIndex = 5;
          const R = gwRoute(G, Q, G.start);
          msg =
            (ep >= MAXE
              ? 'Stopped at the limit of ' +
                MAXE +
                ' episodes: the table is still changing by up to ' +
                f2(Math.max(...w)) +
                ' per episode' +
                (settings.slip ? ', which is what a slippery floor and a fixed α produce' : '') +
                '.'
              : 'The table has stopped changing: for 25 episodes in a row no entry moved by more than 0.01. Finished after ' +
                ep +
                ' episodes.') +
            ' ' +
            okShare() +
            ' entries are within 0.1 of their true values' +
            (settings.slip
              ? '. A smaller α would steady them. '
              : ': the ones still off belong to moves the robot rarely makes. ') +
            (R.ok
              ? 'Following the bold moves from the start reaches the goal in ' +
                R.acts.length +
                ' steps.'
              : 'Following the bold moves from the start does not reach the goal.');
        }
      }
    };
    const tableHTML = (t) => {
      let h =
        '<table><thead><tr><th>square</th>' +
        ACTION_ARROWS.map((x, k) => '<th>' + x + ' ' + ACTION_NAMES[k] + '</th>').join('') +
        '<th>best move</th></tr></thead><tbody>';
      for (const q of G.states) {
        if (G.term(q)) {
          h +=
            '<tr><td>' +
            G.name(q) +
            '</td><td colspan="5" style="color:var(--muted)">' +
            (G.ch(q) === 'G'
              ? 'goal: the episode ends, +' + G.rg
              : 'pit: the episode ends, −' + Math.abs(G.rp)) +
            '</td></tr>';
          continue;
        }
        const b = gwBest(Q, q),
          any = [0, 1, 2, 3].some((a) => Q[q * 4 + a] !== 0);
        h +=
          '<tr' +
          (q === sel ? ' style="outline:2px dashed var(--ink)"' : '') +
          '><td>' +
          (q === s ? '● ' : '') +
          G.name(q) +
          '</td>' +
          [0, 1, 2, 3]
            .map((a) => {
              const hot = last && last.s === q && last.a === a && lastSeq === tv;
              return (
                '<td' +
                (hot ? ' style="background:' + mix(t.gold, t.gold, 0, 0.45) + '"' : '') +
                '>' +
                (any && a === b ? '<b>' + rlN(Q[q * 4 + a]) + '</b>' : rlN(Q[q * 4 + a])) +
                '</td>'
              );
            })
            .join('') +
          '<td>' +
          (any ? ACTION_ARROWS[b] + ' ' + ACTION_NAMES[b] : '–') +
          '</td></tr>';
      }
      return h + '</tbody></table>';
    };
    const ask = () => {
      const el = $('#ql-tpo', root);
      if (sel < 0 || G.wall(sel)) {
        el.textContent = 'Click a square in the picture to ask the table about it.';
        return;
      }
      if (G.term(sel)) {
        el.textContent = G.name(sel) + ' ends the episode. Nothing is learned for it.';
        return;
      }
      const b = gwBest(Q, sel),
        any = [0, 1, 2, 3].some((a) => Q[sel * 4 + a] !== 0),
        R = gwRoute(G, Q, sel),
        tb = gwBest(Qs, sel);
      el.textContent = !any
        ? 'Square ' +
          G.name(sel) +
          ': every entry is still 0. The robot has not learned anything here yet.'
        : 'Square ' +
          G.name(sel) +
          ': the table says the best move is ' +
          ACTION_ARROWS[b] +
          ' ' +
          ACTION_NAMES[b] +
          ', worth ' +
          rlN(Q[sel * 4 + b]) +
          '. ' +
          (R.ok
            ? 'Following the bold moves from here reaches the goal in ' +
              R.acts.length +
              ' step' +
              (R.acts.length > 1 ? 's' : '') +
              ' (' +
              R.acts.map((a) => ACTION_ARROWS[a]).join(' ') +
              ') for a total reward of ' +
              rlN(R.ret).replace('.00', '') +
              '.'
            : R.loop
              ? 'Following the bold moves from here goes round in a circle: the table is not good enough yet.'
              : 'Following the bold moves from here ends in the pit.') +
          ' The true value of the best move here is ' +
          rlN(Qs[sel * 4 + tb]) +
          ' (' +
          ACTION_ARROWS[tb] +
          ' ' +
          ACTION_NAMES[tb] +
          ').';
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas);
        P = gwDraw(ctx, colors, 6, 6, w - 12, h - 12, G, {
          Q,
          agent: s,
          trail,
          sel,
          hot: last && lastSeq === tv ? [last.s, last.a] : null,
        });
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          n = Math.max(30, RET.length),
          lo = Math.min(-20, ...RET.slice(-n)),
          A = drawSeriesChart(
            ctx,
            colors,
            0,
            0,
            w / 2,
            h,
            RET.map((v) => Math.max(v, lo)),
            {
              title: 'reward of each episode',
              n,
              y1: lo,
              ym: Math.max(10, isFinite(best0) ? best0 : 10),
              col: colors.accent,
            },
          );
        if (isFinite(best0)) {
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          ctx.moveTo(A.m.l, A.Y(best0));
          ctx.lineTo(w / 2 - A.m.r, A.Y(best0));
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = colors.cut;
          ctx.textAlign = 'right';
          ctx.fillText(
            'dashed: best possible' + (settings.slip ? ' without sliding' : ''),
            w / 2 - A.m.r,
            h - A.m.b - 8,
          );
        }
        drawSeriesChart(ctx, colors, w / 2, 0, w / 2, h, meanQErrorHistory, {
          title: 'average gap: table against true values',
          n: n + 1,
          col: colors.ok,
          ym: Math.max(0.5, meanQErrorHistory[0]),
        });
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText('episodes →', w - 12, h - 5);
      }
      if (tvd !== tv) {
        tvd = tv;
        tab.innerHTML = tableHTML(colors);
        ask();
      }
      live.textContent = msg;
      highlightLessonStep(root, 'ql', activeStepIndex);
      const R = gwRoute(G, Q, G.start),
        w25 = maxQChangeHistory.slice(-25);
      setReadout(root, 'ep', ep + (fin ? ' (stopped)' : ''));
      setReadout(root, 'st', String(stp));
      setReadout(root, 'lr', RET.length ? rlN(RET[RET.length - 1]).replace('.00', '') : '–');
      setReadout(
        root,
        'rt',
        R.ok
          ? R.acts.map((a) => ACTION_ARROWS[a]).join(' ') + ' (' + R.acts.length + ' steps)'
          : 'not found yet',
      );
      setReadout(root, 'ok', okShare());
      setReadout(root, 'dm', w25.length ? f3(Math.max(...w25)) : '–');
    };
    const one = (q) => {
      step(q);
      lastSeq = tv;
    };
    build();
    if (prefersReducedMotion) {
      while (!fin) step(true);
      lastSeq = -1;
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (settings.sp === 1) {
        if (frameCount % 34) return;
        one(false);
      } else if (settings.sp === 2) {
        if (frameCount % 7) return;
        one(false);
      } else if (settings.sp === 3) {
        for (let i = 0; i < 4 && !fin && checkRunStop.remaining() > 0; i++) one(true);
        if (lastL) eqShow(eq, lastL, 0);
      } else {
        const e0 = ep;
        while (ep < e0 + 2 && !fin && checkRunStop.remaining() > 0) one(true);
        if (lastL) eqShow(eq, lastL, 0);
      }
      draw();
      checkRunStop();
    });
    playback = playBtn($('#ql-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'ql', {
      done: () => fin,
      restart: () => build(),
      count: () => ep,
      what: 'episodes',
    });
    $('#ql-step', root).onclick = () => {
      playback.stop();
      if (fin) build();
      else one(false);
      draw();
    };
    $('#ql-reset', root).onclick = () => {
      playback.stop();
      build();
      draw();
    };
    $('#ql-ep', root).onclick = () => {
      playback.stop();
      if (fin) {
        draw();
        return;
      }
      const e0 = ep;
      let k = 0;
      while (ep === e0 && !fin && k++ < CAP + 2) one(false);
      draw();
    };
    segBind(root, 'ql-w', settings.w, (v) => {
      settings.w = v;
      playback.stop();
      build();
      draw();
    });
    segBind(root, 'ql-sl', String(settings.slip), (v) => {
      settings.slip = +v;
      playback.stop();
      build();
      draw();
    });
    bind(
      root,
      'ql-sp',
      (v) => {
        settings.sp = +v;
      },
      (v) => ['', 'slow, to read each update', 'steady', 'fast', 'two episodes at a time'][+v],
    );
    bind(
      root,
      'ql-al',
      (v) => {
        settings.al = +v;
      },
      f2,
    );
    bind(
      root,
      'ql-ep2',
      (v) => {
        settings.eps = +v;
      },
      f2,
    );
    bind(
      root,
      'ql-ga',
      (v) => {
        settings.ga = +v;
        if (!playback) return;
        truth();
        meanQErrorHistory = [gap()];
        RET = [];
        maxQChangeHistory = [];
        ep = 0;
        fin = false;
        tv++;
        msg =
          'γ changed, so the true values changed too. The charts start again. The table keeps what it has learned.';
        draw();
      },
      f2,
    );
    gwClick(mainCanvas, (x, y) => {
      if (!P) return;
      const q = P.at(x, y);
      if (q < 0 || G.wall(q)) return;
      sel = q;
      tv++;
      draw();
    });
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
