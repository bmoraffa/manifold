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
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { ACTION_NAMES, ACTION_ARROWS, gwClick, gwDraw, gwMake, gwRoute, rlN } from '../shared.js';
import { segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#vi-c', root),
      cr = $('#vi-r', root),
      live = $('#vi-live', root),
      eq = $('#vi-eq', root),
      settings = { m: 'vi', w: 'room', sl: 0.2, ga: 0.9, rs: -1 },
      W = {
        room: ['...G', '.#.X', 'S...'],
        maze: ['S..#..', '.#...#', '.#.#..', '...#.X', '.#...G'],
      };
    let G,
      V,
      Vn,
      pol,
      order = [],
      sweepCellIndex = 0,
      sw = 0,
      DL = [],
      policyChangeHistory = [],
      maxValueChange = 0,
      pch = 0,
      cur = -1,
      sel = -1,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false,
      P = null,
      evs = 0;
    const qa = (s, a, Vx) => {
      let v = 0;
      for (const [p, s2] of G.outcomes(s, a))
        v += p * (G.rew(s2) + (G.term(s2) ? 0 : settings.ga * Vx[s2]));
      return v;
    };
    const bestA = (s, Vx) => {
      let b = 0,
        bv = -1e30;
      for (let a = 0; a < 4; a++) {
        const v = qa(s, a, Vx);
        if (v > bv + 1e-10) {
          bv = v;
          b = a;
        }
      }
      return [b, bv];
    };
    const build = () => {
      G = gwMake(W[settings.w], { slip: settings.sl, rs: settings.rs });
      V = new Float64Array(G.n);
      Vn = new Float64Array(G.n);
      pol = new Int8Array(G.n).fill(-1);
      order = G.states.filter((s) => !G.term(s));
      if (settings.m === 'pi') for (const s of order) pol[s] = 0;
      sweepCellIndex = 0;
      sw = 0;
      DL = [];
      policyChangeHistory = [];
      maxValueChange = 0;
      pch = 0;
      cur = -1;
      fin = false;
      activeStepIndex = 0;
      evs = 0;
      msg =
        settings.m === 'vi'
          ? 'Every value is zero and there are no arrows yet. Press One step to recompute one square, or Run.'
          : 'Policy iteration starts from an arbitrary set of arrows: every square says “up”. Press One step for one round of evaluate and improve.';
      eqShow(eq, ['Nothing has been computed yet.'], 0);
    };
    const lines = (s, Vx, head) => {
      const L = [head];
      let bv = -1e30,
        b = 0;
      for (let a = 0; a < 4; a++) {
        const O = G.outcomes(s, a),
          v = qa(s, a, Vx);
        if (v > bv + 1e-10) {
          bv = v;
          b = a;
        }
        L.push(
          ACTION_ARROWS[a] +
            ' ' +
            ACTION_NAMES[a].padEnd(5) +
            ': ' +
            O.map(
              ([p, s2]) =>
                (O.length > 1 ? f2(p) + ' × ' : '') +
                '(' +
                rlN(G.rew(s2)) +
                (G.term(s2) ? '' : ' + ' + f2(settings.ga) + ' × ' + rlN(Vx[s2])) +
                ')' +
                (O.length > 1 ? '' : '   [lands in ' + G.name(s2) + ']'),
            ).join(' + ') +
            ' = ' +
            rlN(v),
        );
      }
      return { L, b, bv };
    };
    const cell = () => {
      if (fin) return;
      const s = order[sweepCellIndex],
        r = lines(
          s,
          V,
          'Sweep ' +
            (sw + 1) +
            ', square <b>' +
            G.name(s) +
            '</b>. Its value before: ' +
            rlN(V[s]) +
            '. Each move is worth reward + γ × value of where it lands:',
        ),
        d = Math.abs(r.bv - V[s]);
      Vn[s] = r.bv;
      if (pol[s] !== r.b) {
        pch++;
        pol[s] = r.b;
      }
      maxValueChange = Math.max(maxValueChange, d);
      cur = s;
      r.L.push(
        'best: ' +
          ACTION_ARROWS[r.b] +
          ' ' +
          ACTION_NAMES[r.b] +
          '.  New value of ' +
          G.name(s) +
          ' = <b>' +
          rlN(r.bv) +
          '</b>   (changed by ' +
          rlN(d) +
          ')',
      );
      eqShow(eq, r.L, 0);
      activeStepIndex = 2;
      msg =
        'Square ' +
        G.name(s) +
        ': the best of its four moves is ' +
        ACTION_NAMES[r.b] +
        ', worth ' +
        rlN(r.bv) +
        '. Its value changed by ' +
        rlN(d) +
        '.';
      sweepCellIndex++;
      if (sweepCellIndex >= order.length) endSweep();
    };
    const endSweep = () => {
      for (const s of order) V[s] = Vn[s];
      sw++;
      DL.push(maxValueChange);
      policyChangeHistory.push(pch);
      const d = maxValueChange,
        pc = pch;
      maxValueChange = 0;
      pch = 0;
      sweepCellIndex = 0;
      activeStepIndex = 3;
      msg =
        'Sweep ' +
        sw +
        ' is complete: the largest change in any value was ' +
        f3(d) +
        ', and ' +
        pc +
        ' arrow' +
        (pc === 1 ? '' : 's') +
        ' changed.';
      if (d < 1e-3) {
        fin = true;
        msg += ' That is below 0.001, so the values have converged after ' + sw + ' sweeps.';
      }
    };
    const sweep = () => {
      if (fin) return;
      while (true) {
        const s = order[sweepCellIndex],
          [b, bv] = bestA(s, V);
        maxValueChange = Math.max(maxValueChange, Math.abs(bv - V[s]));
        Vn[s] = bv;
        if (pol[s] !== b) {
          pch++;
          pol[s] = b;
        }
        sweepCellIndex++;
        if (sweepCellIndex >= order.length) {
          cur = -1;
          endSweep();
          break;
        }
      }
    };
    const round = () => {
      if (fin) return;
      let n = 0;
      for (; n < 2000; n++) {
        let d = 0;
        for (const s of order) {
          const v = qa(s, pol[s], V);
          d = Math.max(d, Math.abs(v - V[s]));
          Vn[s] = v;
        }
        for (const s of order) V[s] = Vn[s];
        if (d < 1e-4) {
          n++;
          break;
        }
      }
      evs += n;
      const ch = [];
      for (const s of order) {
        const [b] = bestA(s, V),
          old = pol[s];
        if (qa(s, b, V) > qa(s, old, V) + 1e-9) {
          pol[s] = b;
          ch.push(G.name(s) + ' ' + ACTION_ARROWS[old] + '→' + ACTION_ARROWS[b]);
        }
      }
      sw++;
      DL.push(n);
      policyChangeHistory.push(ch.length);
      cur = -1;
      activeStepIndex = 4;
      eqShow(
        eq,
        [
          'Round ' + sw + ' of policy iteration.',
          '<b>Evaluate</b>: with the arrows held fixed, the values were swept ' +
            n +
            ' times until they stopped changing.',
          '<b>Improve</b>: every square looked at its four moves under those values. ' +
            ch.length +
            ' square' +
            (ch.length === 1 ? '' : 's') +
            ' found a better move than its arrow.',
          ch.length
            ? 'changed: ' + ch.slice(0, 10).join(',  ') + (ch.length > 10 ? ', …' : '')
            : 'No arrow changed, so these arrows are the best possible.',
        ],
        0,
      );
      msg =
        'Round ' +
        sw +
        ': evaluating the arrows took ' +
        n +
        ' sweeps, then ' +
        ch.length +
        ' arrow' +
        (ch.length === 1 ? '' : 's') +
        ' changed.';
      if (!ch.length) {
        fin = true;
        msg +=
          ' Policy iteration has converged after ' +
          sw +
          ' rounds and ' +
          evs +
          ' evaluation sweeps in total.';
      }
    };
    const show = () => {
      if (sel < 0 || G.wall(sel) || G.term(sel)) return;
      const r = lines(
        sel,
        V,
        'Square <b>' +
          G.name(sel) +
          '</b> with the values as they are now (' +
          rlN(V[sel]) +
          '). Each move is worth reward + γ × value of where it lands:',
      );
      r.L.push(
        'best: ' +
          ACTION_ARROWS[r.b] +
          ' ' +
          ACTION_NAMES[r.b] +
          ', worth <b>' +
          rlN(r.bv) +
          '</b>',
      );
      eqShow(eq, r.L, 0);
    };
    const draw = () => {
      const colors = readThemeColors(),
        D = new Float64Array(V);
      if (settings.m === 'vi') for (let i = 0; i < sweepCellIndex; i++) D[order[i]] = Vn[order[i]];
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas);
        P = gwDraw(ctx, colors, 6, 6, w - 12, h - 12, G, { V: D, pol, cur, sel });
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr);
        if (settings.m === 'vi') {
          drawSeriesChart(ctx, colors, 0, 0, w, h / 2, DL, {
            title: 'largest change in any value, per sweep',
            n: Math.max(12, DL.length),
            col: colors.accent,
            dots: DL.length < 50,
            ym: Math.max(1, ...DL),
          });
          drawSeriesChart(ctx, colors, 0, h / 2, w, h / 2, policyChangeHistory, {
            title: 'arrows that changed, per sweep',
            n: Math.max(12, policyChangeHistory.length),
            col: colors.ok,
            dots: policyChangeHistory.length < 50,
            ym: Math.max(4, ...policyChangeHistory),
          });
        } else {
          drawSeriesChart(ctx, colors, 0, 0, w, h / 2, DL, {
            title: 'evaluation sweeps needed, per round',
            n: Math.max(6, DL.length),
            col: colors.accent,
            dots: true,
            ym: Math.max(10, ...DL),
          });
          drawSeriesChart(ctx, colors, 0, h / 2, w, h / 2, policyChangeHistory, {
            title: 'arrows that changed, per round',
            n: Math.max(6, policyChangeHistory.length),
            col: colors.ok,
            dots: true,
            ym: Math.max(4, ...policyChangeHistory),
          });
        }
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'right';
        ctx.fillText(settings.m === 'vi' ? 'sweeps →' : 'rounds →', w - 12, h - 5);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'vi', activeStepIndex);
      let rt = '–';
      if (pol[G.start] >= 0) {
        const Qx = new Float64Array(G.n * 4);
        for (const s of order) if (pol[s] >= 0) Qx[s * 4 + pol[s]] = 1;
        const R = gwRoute(G, Qx, G.start);
        rt = R.ok
          ? R.acts.map((a) => ACTION_ARROWS[a]).join(' ') + ' (' + R.acts.length + ' steps)'
          : R.loop
            ? 'goes round in a circle so far'
            : 'ends in the pit';
      }
      setReadout(root, 'me', settings.m === 'vi' ? 'Value iteration' : 'Policy iteration');
      setReadout(
        root,
        'sw',
        settings.m === 'vi'
          ? sw + (fin ? ' (converged)' : '')
          : sw + ' rounds, ' + evs + ' evaluation sweeps' + (fin ? ' (converged)' : ''),
      );
      setReadout(root, 'dl', settings.m === 'vi' && DL.length ? f3(DL[DL.length - 1]) : '–');
      setReadout(root, 'vs', rlN(D[G.start]));
      setReadout(root, 'rt', rt);
      setReadout(
        root,
        'pc',
        policyChangeHistory.length
          ? String(policyChangeHistory[policyChangeHistory.length - 1])
          : '–',
      );
    };
    build();
    if (prefersReducedMotion) {
      while (!fin) settings.m === 'vi' ? sweep() : round();
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (settings.m === 'vi') {
        if (sw < 2) {
          if (frameCount % 9) return;
          cell();
        } else {
          if (frameCount % 10) return;
          sweep();
        }
      } else {
        if (frameCount % 70) return;
        round();
      }
      draw();
      checkRunStop();
    });
    playback = playBtn($('#vi-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'vi', {
      done: () => fin,
      restart: () => build(),
      count: () => sw,
      what: 'sweeps',
    });
    $('#vi-step', root).onclick = () => {
      playback.stop();
      if (fin) build();
      else settings.m === 'vi' ? cell() : round();
      draw();
    };
    $('#vi-reset', root).onclick = () => {
      playback.stop();
      build();
      draw();
    };
    const re = () => {
      if (!playback) return;
      playback.stop();
      build();
      draw();
      playback.start();
    };
    segBind(root, 'vi-m', settings.m, (v) => {
      settings.m = v;
      re();
    });
    segBind(root, 'vi-w', settings.w, (v) => {
      settings.w = v;
      sel = -1;
      re();
    });
    bind(
      root,
      'vi-sl',
      (v) => {
        settings.sl = +v;
        re();
      },
      f2,
    );
    bind(
      root,
      'vi-ga',
      (v) => {
        settings.ga = +v;
        re();
      },
      f2,
    );
    bind(
      root,
      'vi-rs',
      (v) => {
        settings.rs = +v;
        re();
      },
      f2,
    );
    gwClick(mainCanvas, (x, y) => {
      if (!P) return;
      const q = P.at(x, y);
      if (q < 0 || G.wall(q) || G.term(q)) return;
      playback.stop();
      sel = q;
      show();
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
