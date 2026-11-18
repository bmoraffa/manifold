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
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import {
  ACTION_NAMES,
  ACTION_ARROWS,
  gwClick,
  gwDraw,
  gwMake,
  gwMax,
  gwQstar,
  gwRoute,
  rlN,
} from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#pg-c', root),
      cr = $('#pg-r', root),
      live = $('#pg-live', root),
      eq = $('#pg-eq', root),
      settings = { al: 0.1, base: 1, sp: 2 },
      GA = 0.95,
      CAP = 60,
      MAXE = 1500;
    let G,
      TH,
      B,
      baselineVisitCounts,
      ep = 0,
      RET = [],
      lastPath = [],
      lastN = 0,
      sel,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false,
      P = null,
      optimalRouteReturn = 0,
      bpath = [],
      nearOptimalStreak = 0,
      QS = null;
    const prob = (s) => {
      const z = [0, 1, 2, 3].map((a) => TH[s * 4 + a]),
        mx = Math.max(...z),
        e = z.map((v) => Math.exp(v - mx)),
        sm = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / sm);
    };
    const build = () => {
      G = gwMake(['...G', '.#.X', 'S...']);
      TH = new Float64Array(G.n * 4);
      B = new Float64Array(G.n);
      baselineVisitCounts = new Int32Array(G.n);
      ep = 0;
      RET = [];
      lastPath = [G.start];
      lastN = 0;
      sel = G.start;
      fin = false;
      activeStepIndex = 0;
      nearOptimalStreak = 0;
      const Qs = gwQstar(G, GA),
        R = gwRoute(G, Qs, G.start);
      QS = Qs;
      optimalRouteReturn = R.ret;
      bpath = R.path.slice(0, -1).map((s, i) => [s, R.acts[i]]);
      msg =
        'All preferences are zero: in every square the four moves are equally likely, as the equal arrows show. Press One episode or Run.';
      eqShow(eq, ['No episode has been played yet.'], 0);
    };
    const episode = (quiet) => {
      if (fin) return;
      let s = G.start;
      const tr = [];
      let tot = 0;
      for (let i = 0; i < CAP; i++) {
        const p = prob(s);
        let u = Math.random(),
          a = 3;
        for (let k = 0; k < 4; k++) {
          u -= p[k];
          if (u <= 0) {
            a = k;
            break;
          }
        }
        const o = G.step(s, a);
        tr.push({ s, a, r: o.r, p: p[a] });
        tot += o.r;
        s = o.s2;
        if (o.done) break;
      }
      let Gt = 0;
      for (let i = tr.length - 1; i >= 0; i--) {
        Gt = tr[i].r + GA * Gt;
        tr[i].G = Gt;
      }
      let ex = null;
      const pick = Math.floor(Math.random() * Math.min(tr.length, 6));
      tr.forEach((e, i) => {
        const b = settings.base ? B[e.s] : 0,
          adv = e.G - b,
          p = prob(e.s),
          old = TH[e.s * 4 + e.a];
        for (let k = 0; k < 4; k++)
          TH[e.s * 4 + k] += settings.al * adv * ((k === e.a ? 1 : 0) - p[k]);
        if (i === pick)
          ex = { e, b, adv, p: p[e.a], old, nw: TH[e.s * 4 + e.a], p2: prob(e.s)[e.a], i };
        baselineVisitCounts[e.s]++;
        B[e.s] += (e.G - B[e.s]) / Math.min(baselineVisitCounts[e.s], 20);
      });
      ep++;
      RET.push(tot);
      lastPath = [G.start].concat(tr.map((e, i) => (i + 1 < tr.length ? tr[i + 1].s : s)));
      lastN = tr.length;
      const how =
        G.ch(s) === 'G'
          ? 'reached the goal'
          : G.ch(s) === 'X'
            ? 'fell into the pit'
            : 'ran out of time after ' + CAP + ' steps';
      if (!quiet && ex) {
        const e = ex.e;
        eqShow(
          eq,
          [
            'Episode ' +
              ep +
              ' ' +
              how +
              ' after ' +
              tr.length +
              ' steps, total reward ' +
              rlN(tot).replace('.00', '') +
              '. Take step ' +
              (ex.i + 1) +
              ' as an example:',
            'in <b>' +
              G.name(e.s) +
              '</b> the robot chose ' +
              ACTION_ARROWS[e.a] +
              ' ' +
              ACTION_NAMES[e.a] +
              ', which had probability ' +
              pct(ex.p) +
              '.',
            'return from that step to the end:  G = <b>' +
              rlN(e.G) +
              '</b>' +
              (settings.base
                ? '     usual return from ' + G.name(e.s) + ' (baseline): ' + rlN(ex.b)
                : '     no baseline: compared with 0'),
            'better or worse than usual:  G − baseline = ' +
              rlN(e.G) +
              ' − ' +
              rlN(ex.b) +
              ' = <b>' +
              rlN(ex.adv) +
              '</b>',
            'preference for ' +
              ACTION_ARROWS[e.a] +
              ' in ' +
              G.name(e.s) +
              ':  ' +
              rlN(ex.old) +
              ' + ' +
              f2(settings.al) +
              ' × ' +
              rlN(ex.adv) +
              ' × (1 − ' +
              f2(ex.p) +
              ') = <b>' +
              rlN(ex.nw) +
              '</b>',
            'so the probability of ' +
              ACTION_ARROWS[e.a] +
              ' in ' +
              G.name(e.s) +
              ' went from ' +
              pct(ex.p) +
              ' to <b>' +
              pct(ex.p2) +
              '</b>. The other three moves shifted the opposite way.',
          ],
          0,
        );
      }
      const av = RET.slice(-30).reduce((a, b) => a + b, 0) / Math.min(30, RET.length);
      activeStepIndex = 4;
      msg =
        'Episode ' +
        ep +
        ' ' +
        how +
        ': total reward ' +
        rlN(tot).replace('.00', '') +
        '. Every move made in it was strengthened or weakened according to what followed. Average over the last 30 episodes: ' +
        f2(av) +
        ' (best possible: ' +
        rlN(optimalRouteReturn).replace('.00', '') +
        ').';
      if (RET.length >= 30 && av >= optimalRouteReturn - 1) nearOptimalStreak++;
      else nearOptimalStreak = 0;
      if (nearOptimalStreak >= 30 || ep >= MAXE) {
        fin = true;
        activeStepIndex = 5;
        msg =
          (ep >= MAXE
            ? 'Stopped at the limit of ' +
              MAXE +
              ' episodes with an average reward of ' +
              f2(av) +
              '.'
            : 'Converged after ' +
              ep +
              ' episodes: for 30 episodes in a row the average reward has been within 1 of the best possible (' +
              rlN(optimalRouteReturn).replace('.00', '') +
              '). It is now ' +
              f2(av) +
              '.') +
          ' On the squares of its last episode it chooses a best move ' +
          pct(pbest()) +
          ' of the time.';
      }
    };
    const pbest = () => {
      const sq = [...new Set(lastPath.filter((q) => !G.term(q)))];
      if (!sq.length) return NaN;
      let t2 = 0;
      for (const q of sq) {
        const p = prob(q),
          m = gwMax(QS, q);
        for (let a = 0; a < 4; a++) if (QS[q * 4 + a] >= m - 1e-9) t2 += p[a];
      }
      return t2 / sq.length;
    };
    const draw = () => {
      const colors = readThemeColors(),
        PR = new Float64Array(G.n * 4);
      for (const s of G.states) if (!G.term(s)) prob(s).forEach((p, a) => (PR[s * 4 + a] = p));
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas);
        P = gwDraw(ctx, colors, 6, 6, w - 12, h - 12, G, {
          probs: PR,
          trail: lastPath,
          thin: true,
          sel,
        });
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.56,
          n = Math.max(60, RET.length),
          lo = -70,
          A = drawSeriesChart(ctx, colors, 0, 0, w, h1, [], {
            title: 'total reward of each episode, and its average over 30',
            n,
            y1: lo,
            ym: 10,
          });
        ctx.strokeStyle = mix(colors.accent, colors.accent, 0, 0.3);
        ctx.lineWidth = 1;
        ctx.beginPath();
        RET.forEach((v, i) =>
          i ? ctx.lineTo(A.X(i), A.Y(Math.max(lo, v))) : ctx.moveTo(A.X(i), A.Y(Math.max(lo, v))),
        );
        ctx.stroke();
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        RET.forEach((_, i) => {
          const a = RET.slice(Math.max(0, i - 29), i + 1),
            v = Math.max(lo, a.reduce((x, y) => x + y, 0) / a.length);
          i ? ctx.lineTo(A.X(i), A.Y(v)) : ctx.moveTo(A.X(i), A.Y(v));
        });
        ctx.stroke();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(A.m.l, A.Y(optimalRouteReturn));
        ctx.lineTo(w - A.m.r, A.Y(optimalRouteReturn));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = MONO;
        ctx.fillStyle = colors.cut;
        ctx.textAlign = 'right';
        ctx.fillText('best possible', w - A.m.r, A.Y(optimalRouteReturn) + 13);
        const p = G.term(sel) ? [0, 0, 0, 0] : prob(sel),
          y0 = h1 + 24,
          H = h - h1 - 48,
          bw = (w - 80) / 4;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('square ' + G.name(sel) + ': probability of each move', 8, h1 + 14);
        for (let a = 0; a < 4; a++) {
          const x = 50 + a * bw;
          ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.35 + 0.65 * p[a]);
          ctx.fillRect(x, y0 + H - p[a] * H, bw * 0.6, p[a] * H);
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText(
            ACTION_ARROWS[a] + ' ' + ACTION_NAMES[a] + '  ' + pct(p[a]),
            x + bw * 0.3,
            h - 8,
          );
        }
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(40, y0 + H);
        ctx.lineTo(w - 20, y0 + H);
        ctx.stroke();
      }
      live.textContent = msg;
      highlightLessonStep(root, 'pg', activeStepIndex);
      const av = RET.length
        ? RET.slice(-30).reduce((a, b) => a + b, 0) / Math.min(30, RET.length)
        : NaN;
      setReadout(root, 'ep', ep + (fin ? ' (stopped)' : ''));
      setReadout(root, 'lr', RET.length ? rlN(RET[RET.length - 1]).replace('.00', '') : '–');
      setReadout(root, 'av', isFinite(av) ? f2(av) : '–');
      setReadout(root, 'op', rlN(optimalRouteReturn).replace('.00', ''));
      setReadout(root, 'ps', pct(pbest()));
      setReadout(root, 'ls', lastN ? String(lastN) : '–');
    };
    build();
    if (prefersReducedMotion) {
      while (!fin) episode(true);
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (settings.sp === 1) {
        if (frameCount % 45) return;
        episode(false);
      } else if (settings.sp === 2) {
        if (ep < 5) {
          if (frameCount % 20) return;
          episode(false);
        } else {
          if (frameCount % 2) return;
          episode(false);
        }
      } else {
        for (let i = 0; i < 5 && !fin && checkRunStop.remaining() > 0; i++) episode(i < 4);
      }
      draw();
      checkRunStop();
    });
    playback = playBtn($('#pg-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'pg', {
      done: () => fin,
      restart: () => build(),
      count: () => ep,
      what: 'episodes',
    });
    $('#pg-step', root).onclick = () => {
      playback.stop();
      if (fin) build();
      else episode(false);
      draw();
    };
    $('#pg-reset', root).onclick = () => {
      playback.stop();
      build();
      draw();
    };
    bind(
      root,
      'pg-al',
      (v) => {
        settings.al = +v;
      },
      f2,
    );
    bind(
      root,
      'pg-sp',
      (v) => {
        settings.sp = +v;
      },
      (v) => ['', 'one episode at a time', 'steady', 'fast'][+v],
    );
    segBind(root, 'pg-b', String(settings.base), (v) => {
      settings.base = +v;
      playback.stop();
      build();
      draw();
      playback.start();
    });
    gwClick(mainCanvas, (x, y) => {
      if (!P) return;
      const q = P.at(x, y);
      if (q < 0 || G.wall(q) || G.term(q)) return;
      sel = q;
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
