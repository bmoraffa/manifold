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
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { GWN, GWS, gwClick, gwDraw, gwMake, gwRoute, rlN } from '../shared.js';
import { segBind, uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#vi-c', root),
      cr = $('#vi-r', root),
      live = $('#vi-live', root),
      eq = $('#vi-eq', root),
      S = { m: 'vi', w: 'room', sl: 0.2, ga: 0.9, rs: -1 },
      W = {
        room: ['...G', '.#.X', 'S...'],
        maze: ['S..#..', '.#...#', '.#.#..', '...#.X', '.#...G'],
      };
    let G,
      V,
      Vn,
      pol,
      order = [],
      ci = 0,
      sw = 0,
      DL = [],
      PCH = [],
      dcur = 0,
      pch = 0,
      cur = -1,
      sel = -1,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      fin = false,
      P = null,
      evs = 0;
    const qa = (s, a, Vx) => {
      let v = 0;
      for (const [p, s2] of G.outcomes(s, a))
        v += p * (G.rew(s2) + (G.term(s2) ? 0 : S.ga * Vx[s2]));
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
      G = gwMake(W[S.w], { slip: S.sl, rs: S.rs });
      V = new Float64Array(G.n);
      Vn = new Float64Array(G.n);
      pol = new Int8Array(G.n).fill(-1);
      order = G.states.filter((s) => !G.term(s));
      if (S.m === 'pi') for (const s of order) pol[s] = 0;
      ci = 0;
      sw = 0;
      DL = [];
      PCH = [];
      dcur = 0;
      pch = 0;
      cur = -1;
      fin = false;
      cu = 0;
      evs = 0;
      msg =
        S.m === 'vi'
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
          GWS[a] +
            ' ' +
            GWN[a].padEnd(5) +
            ': ' +
            O.map(
              ([p, s2]) =>
                (O.length > 1 ? f2(p) + ' × ' : '') +
                '(' +
                rlN(G.rew(s2)) +
                (G.term(s2) ? '' : ' + ' + f2(S.ga) + ' × ' + rlN(Vx[s2])) +
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
      const s = order[ci],
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
      dcur = Math.max(dcur, d);
      cur = s;
      r.L.push(
        'best: ' +
          GWS[r.b] +
          ' ' +
          GWN[r.b] +
          '.  New value of ' +
          G.name(s) +
          ' = <b>' +
          rlN(r.bv) +
          '</b>   (changed by ' +
          rlN(d) +
          ')',
      );
      eqShow(eq, r.L, 0);
      cu = 2;
      msg =
        'Square ' +
        G.name(s) +
        ': the best of its four moves is ' +
        GWN[r.b] +
        ', worth ' +
        rlN(r.bv) +
        '. Its value changed by ' +
        rlN(d) +
        '.';
      ci++;
      if (ci >= order.length) endSweep();
    };
    const endSweep = () => {
      for (const s of order) V[s] = Vn[s];
      sw++;
      DL.push(dcur);
      PCH.push(pch);
      const d = dcur,
        pc = pch;
      dcur = 0;
      pch = 0;
      ci = 0;
      cu = 3;
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
        const s = order[ci],
          [b, bv] = bestA(s, V);
        dcur = Math.max(dcur, Math.abs(bv - V[s]));
        Vn[s] = bv;
        if (pol[s] !== b) {
          pch++;
          pol[s] = b;
        }
        ci++;
        if (ci >= order.length) {
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
          ch.push(G.name(s) + ' ' + GWS[old] + '→' + GWS[b]);
        }
      }
      sw++;
      DL.push(n);
      PCH.push(ch.length);
      cur = -1;
      cu = 4;
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
      r.L.push('best: ' + GWS[r.b] + ' ' + GWN[r.b] + ', worth <b>' + rlN(r.bv) + '</b>');
      eqShow(eq, r.L, 0);
    };
    const draw = () => {
      const t = T(),
        D = new Float64Array(V);
      if (S.m === 'vi') for (let i = 0; i < ci; i++) D[order[i]] = Vn[order[i]];
      {
        const { g, w, h } = fit(c);
        P = gwDraw(g, t, 6, 6, w - 12, h - 12, G, { V: D, pol, cur, sel });
      }
      {
        const { g, w, h } = fit(cr);
        if (S.m === 'vi') {
          uLine(g, t, 0, 0, w, h / 2, DL, {
            title: 'largest change in any value, per sweep',
            n: Math.max(12, DL.length),
            col: t.accent,
            dots: DL.length < 50,
            ym: Math.max(1, ...DL),
          });
          uLine(g, t, 0, h / 2, w, h / 2, PCH, {
            title: 'arrows that changed, per sweep',
            n: Math.max(12, PCH.length),
            col: t.ok,
            dots: PCH.length < 50,
            ym: Math.max(4, ...PCH),
          });
        } else {
          uLine(g, t, 0, 0, w, h / 2, DL, {
            title: 'evaluation sweeps needed, per round',
            n: Math.max(6, DL.length),
            col: t.accent,
            dots: true,
            ym: Math.max(10, ...DL),
          });
          uLine(g, t, 0, h / 2, w, h / 2, PCH, {
            title: 'arrows that changed, per round',
            n: Math.max(6, PCH.length),
            col: t.ok,
            dots: true,
            ym: Math.max(4, ...PCH),
          });
        }
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText(S.m === 'vi' ? 'sweeps →' : 'rounds →', w - 12, h - 5);
      }
      live.textContent = msg;
      uCur(root, 'vi', cu);
      let rt = '–';
      if (pol[G.start] >= 0) {
        const Qx = new Float64Array(G.n * 4);
        for (const s of order) if (pol[s] >= 0) Qx[s * 4 + pol[s]] = 1;
        const R = gwRoute(G, Qx, G.start);
        rt = R.ok
          ? R.acts.map((a) => GWS[a]).join(' ') + ' (' + R.acts.length + ' steps)'
          : R.loop
            ? 'goes round in a circle so far'
            : 'ends in the pit';
      }
      setR(root, 'me', S.m === 'vi' ? 'Value iteration' : 'Policy iteration');
      setR(
        root,
        'sw',
        S.m === 'vi'
          ? sw + (fin ? ' (converged)' : '')
          : sw + ' rounds, ' + evs + ' evaluation sweeps' + (fin ? ' (converged)' : ''),
      );
      setR(root, 'dl', S.m === 'vi' && DL.length ? f3(DL[DL.length - 1]) : '–');
      setR(root, 'vs', rlN(D[G.start]));
      setR(root, 'rt', rt);
      setR(root, 'pc', PCH.length ? String(PCH[PCH.length - 1]) : '–');
    };
    build();
    if (RM) {
      while (!fin) S.m === 'vi' ? sweep() : round();
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      if (S.m === 'vi') {
        if (sw < 2) {
          if (fr % 9) return;
          cell();
        } else {
          if (fr % 10) return;
          sweep();
        }
      } else {
        if (fr % 70) return;
        round();
      }
      draw();
      chk();
    });
    pb = playBtn($('#vi-play', root), anim);
    chk = runCtl(pb, root, 'vi', {
      done: () => fin,
      restart: () => build(),
      count: () => sw,
      what: 'sweeps',
    });
    $('#vi-step', root).onclick = () => {
      pb.stop();
      if (fin) build();
      else S.m === 'vi' ? cell() : round();
      draw();
    };
    $('#vi-reset', root).onclick = () => {
      pb.stop();
      build();
      draw();
    };
    const re = () => {
      if (!pb) return;
      pb.stop();
      build();
      draw();
      pb.start();
    };
    segBind(root, 'vi-m', S.m, (v) => {
      S.m = v;
      re();
    });
    segBind(root, 'vi-w', S.w, (v) => {
      S.w = v;
      sel = -1;
      re();
    });
    bind(
      root,
      'vi-sl',
      (v) => {
        S.sl = +v;
        re();
      },
      f2,
    );
    bind(
      root,
      'vi-ga',
      (v) => {
        S.ga = +v;
        re();
      },
      f2,
    );
    bind(
      root,
      'vi-rs',
      (v) => {
        S.rs = +v;
        re();
      },
      f2,
    );
    gwClick(c, (x, y) => {
      if (!P) return;
      const q = P.at(x, y);
      if (q < 0 || G.wall(q) || G.term(q)) return;
      pb.stop();
      sel = q;
      show();
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
