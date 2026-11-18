/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  eqShow,
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
import { gwClick } from '../../reinforcement/shared.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#rc-c', root),
      cr = $('#rc-r', root),
      live = $('#rc-live', root),
      eq = $('#rc-eq', root),
      settings = { k: 2, l: 0.15 },
      US = ['Ana', 'Ben', 'Chloe', 'Dev', 'Emma', 'Farid', 'Grace', 'Hugo', 'Ines', 'Jun', 'You'],
      FM = [
        'Star Raiders',
        'Steel Horizon',
        'Night Chase',
        'Robot Dawn',
        'Letters from Paris',
        'Summer in Tuscany',
        'The Wedding Guest',
        'Autumn Hearts',
        'Spy in Love',
        'Laugh Track',
      ],
      FT = [
        [1.1, -0.6],
        [1, -0.4],
        [0.9, -0.2],
        [1.2, -0.7],
        [-0.7, 1.1],
        [-0.5, 1],
        [-0.6, 0.9],
        [-0.8, 1.2],
        [0.5, 0.6],
        [0.1, 0.2],
      ],
      NU = 11,
      NI = 10,
      ME = 10,
      MAXE = 400,
      ETA = 0.05;
    let simulatedTrainRatings = [],
      heldOutRatings = [],
      learnerRatings = new Array(NI).fill(0),
      mu = 3,
      bu,
      bi,
      Pu,
      Qi,
      ep = 0,
      TR = [],
      TS = [],
      best = 9,
      bestAt = 0,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false,
      seed = 1,
      PC = null,
      base = NaN;
    const gen = () => {
      const r = rng(seed * 13 + 5),
        rn = randnFrom(r);
      simulatedTrainRatings = [];
      heldOutRatings = [];
      for (let u = 0; u < ME; u++) {
        const a = u % 2 ? [0.2 + 1.1 * r(), 1.1 - 1.3 * r()] : [1.1 - 1.3 * r(), 0.2 + 1.1 * r()],
          gb = 0.5 * rn();
        for (let i = 0; i < NI; i++) {
          if (r() > 0.75) continue;
          const v = clamp(
            Math.round(3 + gb + 1.5 * (a[0] * FT[i][0] + a[1] * FT[i][1]) + 0.45 * rn()),
            1,
            5,
          );
          (r() < 0.25 ? heldOutRatings : simulatedTrainRatings).push([u, i, v]);
        }
      }
    };
    const all = () =>
      simulatedTrainRatings.concat(
        learnerRatings.map((v, i) => (v ? [ME, i, v] : null)).filter((x) => x),
      );
    const pred = (u, i) => {
      let s = mu + bu[u] + bi[i];
      for (let k = 0; k < settings.k; k++) s += Pu[u * 4 + k] * Qi[i * 4 + k];
      return s;
    };
    const rmse = (L) =>
      L.length
        ? Math.sqrt(L.reduce((s, [u, i, v]) => s + (v - pred(u, i)) ** 2, 0) / L.length)
        : NaN;
    const build = () => {
      const rn = randnFrom(rng(seed * 7 + settings.k)),
        A = all();
      mu = A.reduce((s, q) => s + q[2], 0) / A.length;
      bu = new Float64Array(NU);
      bi = new Float64Array(NI);
      Pu = Float64Array.from({ length: NU * 4 }, () => 0.1 * rn());
      Qi = Float64Array.from({ length: NI * 4 }, () => 0.1 * rn());
      ep = 0;
      TR = [rmse(A)];
      TS = [rmse(heldOutRatings)];
      best = 9;
      bestAt = 0;
      fin = false;
      activeStepIndex = 0;
      const sm = new Float64Array(NI),
        ct = new Float64Array(NI);
      for (const [u, i, v] of A) {
        sm[i] += v;
        ct[i]++;
      }
      base = Math.sqrt(
        heldOutRatings.reduce((s, [u, i, v]) => s + (v - (ct[i] ? sm[i] / ct[i] : mu)) ** 2, 0) /
          Math.max(1, heldOutRatings.length),
      );
      msg =
        'The table holds ' +
        A.length +
        ' real ratings for training and ' +
        heldOutRatings.length +
        ' set aside. Every person and film has ' +
        settings.k +
        ' hidden number' +
        (settings.k > 1 ? 's' : '') +
        ', random for now, so every guess is close to the overall average of ' +
        f2(mu) +
        '. Press Run.';
      eqShow(eq, ['No updates yet.'], 0);
    };
    const pass = () => {
      if (fin) return;
      const A0 = all(),
        A = A0.concat(
          A0.filter((q) => q[0] === ME),
          A0.filter((q) => q[0] === ME),
          A0.filter((q) => q[0] === ME),
        )
          .map((q) => [Math.random(), q])
          .sort((a, b) => a[0] - b[0])
          .map((q) => q[1]);
      let ex = null;
      A.forEach(([u, i, v], n) => {
        const p0 = pred(u, i),
          e = v - p0;
        if (n === 0) ex = { u, i, v, p0, e, pu: Pu[u * 4], qi: Qi[i * 4] };
        bu[u] += ETA * (e - settings.l * bu[u]);
        bi[i] += ETA * (e - settings.l * bi[i]);
        for (let k = 0; k < settings.k; k++) {
          const a = Pu[u * 4 + k],
            b = Qi[i * 4 + k];
          Pu[u * 4 + k] += ETA * (e * b - settings.l * a);
          Qi[i * 4 + k] += ETA * (e * a - settings.l * b);
        }
        if (n === 0) {
          ex.pu2 = Pu[u * 4];
          ex.qi2 = Qi[i * 4];
          ex.p1 = pred(u, i);
        }
      });
      ep++;
      const tr = rmse(A0),
        te = rmse(heldOutRatings);
      TR.push(tr);
      TS.push(te);
      if (te < best - 1e-4) {
        best = te;
        bestAt = ep;
      }
      if (ex)
        eqShow(
          eq,
          [
            'Pass ' +
              ep +
              ', first rating visited: <b>' +
              US[ex.u] +
              '</b> gave <b>' +
              FM[ex.i] +
              '</b> a ' +
              ex.v +
              '.',
            'the model predicted ' +
              f2(ex.p0) +
              ', so the error is ' +
              ex.v +
              ' − ' +
              f2(ex.p0) +
              ' = <b>' +
              f2(ex.e) +
              '</b>',
            US[ex.u] +
              '’s first hidden number: ' +
              f3(ex.pu) +
              ' + ' +
              ETA +
              ' × (' +
              f2(ex.e) +
              ' × the film’s ' +
              f3(ex.qi) +
              ' − ' +
              f2(settings.l) +
              ' × ' +
              f3(ex.pu) +
              ') = <b>' +
              f3(ex.pu2) +
              '</b>',
            'the film’s first hidden number: ' +
              f3(ex.qi) +
              ' → <b>' +
              f3(ex.qi2) +
              '</b>, moved in the same way toward ' +
              US[ex.u] +
              '’s',
            'the prediction for this cell is now ' +
              f2(ex.p1) +
              (Math.abs(ex.v - ex.p1) < Math.abs(ex.e) ? ', closer to ' + ex.v : ''),
          ],
          0,
        );
      activeStepIndex = 4;
      msg =
        'Pass ' +
        ep +
        ': all ' +
        A0.length +
        ' training ratings were visited. Error on them: ' +
        f3(tr) +
        ' rating points. Error on the ' +
        heldOutRatings.length +
        ' hidden ratings: ' +
        f3(te) +
        '.';
      if ((ep >= 80 && ep - bestAt >= 40) || ep >= MAXE) {
        fin = true;
        activeStepIndex = 5;
        msg =
          (ep >= MAXE
            ? 'Stopped at the limit of ' + MAXE + ' passes.'
            : 'Stopped after ' +
              ep +
              ' passes: the error on the hidden ratings has not improved for 40 passes. Its best value was ' +
              f3(best) +
              ' at pass ' +
              bestAt +
              '.') +
          ' Final error: ' +
          f3(tr) +
          ' on training ratings, ' +
          f3(te) +
          ' on hidden ones. Always guessing each film’s average would score ' +
          f3(base) +
          '.';
      }
    };
    const recs = () => {
      const el = $('#rc-tpo', root),
        n = learnerRatings.filter((v) => v).length;
      if (!n) {
        el.textContent =
          'Click cells in the bottom row of the table to rate films yourself. Each click raises the rating by one, and a sixth click clears it.';
        return;
      }
      const o = [...Array(NI).keys()]
        .filter((i) => !learnerRatings[i])
        .map((i) => [pred(ME, i), i])
        .sort((a, b) => b[0] - a[0]);
      el.textContent =
        'You have rated ' +
        n +
        ' film' +
        (n > 1 ? 's' : '') +
        '. ' +
        (ep < 5
          ? 'Press Run to train with your ratings included.'
          : o.length
            ? 'For you, the model’s highest guesses among the films you have not rated: ' +
              o
                .slice(0, 3)
                .map((q) => FM[q[1]] + ' (' + f2(clamp(q[0], 1, 5)) + ')')
                .join(', ') +
              '. Its lowest: ' +
              FM[o[o.length - 1][1]] +
              ' (' +
              f2(clamp(o[o.length - 1][0], 1, 5)) +
              ').'
            : 'You have rated everything.');
    };
    const draw = () => {
      const colors = readThemeColors(),
        A = all(),
        known = new Map(A.map((q) => [q[0] * NI + q[1], q[2]])),
        held = new Map(heldOutRatings.map((q) => [q[0] * NI + q[1], q[2]]));
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          lx = 54,
          ty = 122,
          cw = (w - lx - 6) / NI,
          rh = (h - ty - 6) / NU,
          fam = MONO.split('px ')[1];
        PC = { lx, ty, cw, rh };
        ctx.font = MONO;
        for (let i = 0; i < NI; i++) {
          ctx.save();
          ctx.translate(lx + (i + 0.5) * cw + 3, ty - 6);
          ctx.rotate(-Math.PI / 2.5);
          ctx.fillStyle = i < 4 ? colors.cut : i < 8 ? colors.accent : colors.muted;
          ctx.textAlign = 'left';
          ctx.fillText(FM[i], 0, 0);
          ctx.restore();
        }
        for (let u = 0; u < NU; u++) {
          ctx.textAlign = 'right';
          ctx.fillStyle = u === ME ? colors.cut : colors.ink;
          ctx.font = (u === ME ? '600 ' : '') + '12px ' + fam;
          ctx.fillText(US[u], lx - 8, ty + (u + 0.5) * rh + 4);
          for (let i = 0; i < NI; i++) {
            const x = lx + i * cw,
              y = ty + u * rh,
              kv = known.get(u * NI + i),
              hv = held.get(u * NI + i),
              p = clamp(pred(u, i), 1, 5),
              v = kv || p,
              col = v >= 3 ? colors.accent : colors.cut;
            ctx.fillStyle = mix(
              col,
              col,
              0,
              (kv ? 0.2 : 0.07) + (Math.abs(v - 3) / 2) * (kv ? 0.6 : 0.3),
            );
            ctx.fillRect(x + 1, y + 1, cw - 2, rh - 2);
            ctx.textAlign = 'center';
            if (kv) {
              ctx.fillStyle = colors.ink;
              ctx.font = '700 ' + Math.min(15, rh * 0.5) + 'px ' + fam;
              ctx.fillText(String(kv), x + cw / 2, y + rh / 2 + 5);
            } else {
              ctx.fillStyle = colors.muted;
              ctx.font = Math.min(11, rh * 0.38) + 'px ' + fam;
              ctx.fillText(ep ? p.toFixed(1) : '·', x + cw / 2, y + rh / 2 + 4);
              if (hv) {
                ctx.strokeStyle = colors.gold;
                ctx.lineWidth = 2;
                ctx.strokeRect(x + 2.5, y + 2.5, cw - 5, rh - 5);
                ctx.fillStyle = colors.cut;
                ctx.font = '9px ' + fam;
                ctx.fillText('was ' + hv, x + cw / 2, y + rh - 4);
              }
            }
          }
          if (u === ME) {
            ctx.strokeStyle = colors.cut;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.strokeRect(lx + 0.5, ty + u * rh + 0.5, NI * cw - 1, rh - 1);
            ctx.setLineDash([]);
          }
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.56,
          pts = [];
        for (let u = 0; u < NU; u++)
          if (u < ME || learnerRatings.some((v) => v))
            pts.push([Pu[u * 4], settings.k > 1 ? Pu[u * 4 + 1] : 0, US[u], 0, u]);
        for (let i = 0; i < NI; i++)
          pts.push([Qi[i * 4], settings.k > 1 ? Qi[i * 4 + 1] : 0, FM[i], 1, i]);
        const mx = Math.max(0.2, ...pts.map((p) => Math.abs(p[0]))) * 1.25,
          my = Math.max(0.2, ...pts.map((p) => Math.abs(p[1]))) * 1.25,
          X = (v) => w / 2 + (v / mx) * (w / 2 - 70),
          Y = (v) => 18 + (h1 - 30) / 2 - (v / my) * ((h1 - 40) / 2);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          'hidden number 1 (across) and 2 (up): circles are people, squares are films',
          8,
          13,
        );
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(X(-mx), Y(0));
        ctx.lineTo(X(mx), Y(0));
        ctx.moveTo(X(0), Y(-my));
        ctx.lineTo(X(0), Y(my));
        ctx.stroke();
        for (const p of pts) {
          const x = X(p[0]),
            y = Y(p[1]);
          if (p[3]) {
            ctx.fillStyle = p[4] < 4 ? colors.cut : p[4] < 8 ? colors.accent : colors.muted;
            ctx.fillRect(x - 4, y - 4, 8, 8);
          } else {
            ctx.beginPath();
            ctx.arc(x, y, 4.5, 0, 7);
            ctx.fillStyle = p[4] === ME ? colors.gold : colors.bg;
            ctx.fill();
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.6;
            ctx.stroke();
          }
          ctx.fillStyle = p[3] ? colors.muted : colors.ink;
          ctx.textAlign = 'left';
          ctx.fillText(p[2], x + 7, y + 4);
        }
        const n = Math.max(60, TR.length),
          ym = Math.max(1.2, TR[0], TS[0] || 0),
          L = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, TR, {
            title: 'error in rating points, by pass',
            n,
            ym,
            col: colors.accent,
          });
        ctx.strokeStyle = colors.cut;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        TS.forEach((v, i) =>
          i ? ctx.lineTo(L.X(i), L.Y(Math.min(v, ym))) : ctx.moveTo(L.X(i), L.Y(Math.min(v, ym))),
        );
        ctx.stroke();
        ctx.textAlign = 'right';
        ctx.fillStyle = colors.accent;
        ctx.fillText('ratings it trains on', w - 14, h1 + 38);
        ctx.fillStyle = colors.cut;
        ctx.fillText('ratings hidden from it', w - 14, h1 + 52);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'rc', activeStepIndex);
      recs();
      setReadout(root, 'ep', ep + (fin ? ' (stopped)' : ''));
      setReadout(root, 'tr', f3(TR[TR.length - 1]));
      setReadout(root, 'te', f3(TS[TS.length - 1]));
      setReadout(root, 'bl', f3(base));
      setReadout(root, 'nr', A.length + ' / ' + heldOutRatings.length);
      setReadout(root, 'np', String((NU + NI) * (settings.k + 1)));
    };
    gen();
    build();
    if (prefersReducedMotion) {
      while (!fin) pass();
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      if (frameCount % (ep < 6 ? 24 : 3)) return;
      pass();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#rc-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'rc', {
      done: () => fin,
      restart: () => build(),
      count: () => ep,
      what: 'passes',
    });
    $('#rc-step', root).onclick = () => {
      playback.stop();
      if (fin) build();
      else pass();
      draw();
    };
    $('#rc-reset', root).onclick = () => {
      playback.stop();
      build();
      draw();
    };
    $('#rc-new', root).onclick = () => {
      playback.stop();
      seed++;
      gen();
      build();
      draw();
      playback.start();
    };
    $('#rc-clr', root).onclick = () => {
      playback.stop();
      learnerRatings.fill(0);
      build();
      draw();
    };
    bind(
      root,
      'rc-k',
      (v) => {
        settings.k = +v;
        if (playback) {
          playback.stop();
          build();
          draw();
          playback.start();
        }
      },
      (v) => String(v),
    );
    bind(
      root,
      'rc-l',
      (v) => {
        settings.l = +v;
        if (playback) {
          playback.stop();
          build();
          draw();
          playback.start();
        }
      },
      f2,
    );
    gwClick(mainCanvas, (px, py) => {
      if (!PC) return;
      const i = Math.floor((px - PC.lx) / PC.cw),
        u = Math.floor((py - PC.ty) / PC.rh);
      if (u !== ME || i < 0 || i >= NI) return;
      playback.stop();
      learnerRatings[i] = (learnerRatings[i] + 1) % 6;
      build();
      msg =
        'You ' +
        (learnerRatings[i]
          ? 'gave “' + FM[i] + '” a ' + learnerRatings[i]
          : 'cleared your rating of “' + FM[i] + '”') +
        '. The model starts again with your ratings included. Press Run.';
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
