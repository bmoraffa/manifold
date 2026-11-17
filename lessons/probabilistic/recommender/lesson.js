/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  eqShow,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { gwClick } from '../../reinforcement/shared.js';
import { uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#rc-c', root),
      cr = $('#rc-r', root),
      live = $('#rc-live', root),
      eq = $('#rc-eq', root),
      S = { k: 2, l: 0.15 },
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
    let R = [],
      TE = [],
      MY = new Array(NI).fill(0),
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
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      fin = false,
      seed = 1,
      PC = null,
      base = NaN;
    const gen = () => {
      const r = rng(seed * 13 + 5),
        rn = randnFrom(r);
      R = [];
      TE = [];
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
          (r() < 0.25 ? TE : R).push([u, i, v]);
        }
      }
    };
    const all = () => R.concat(MY.map((v, i) => (v ? [ME, i, v] : null)).filter((x) => x));
    const pred = (u, i) => {
      let s = mu + bu[u] + bi[i];
      for (let k = 0; k < S.k; k++) s += Pu[u * 4 + k] * Qi[i * 4 + k];
      return s;
    };
    const rmse = (L) =>
      L.length
        ? Math.sqrt(L.reduce((s, [u, i, v]) => s + (v - pred(u, i)) ** 2, 0) / L.length)
        : NaN;
    const build = () => {
      const rn = randnFrom(rng(seed * 7 + S.k)),
        A = all();
      mu = A.reduce((s, q) => s + q[2], 0) / A.length;
      bu = new Float64Array(NU);
      bi = new Float64Array(NI);
      Pu = Float64Array.from({ length: NU * 4 }, () => 0.1 * rn());
      Qi = Float64Array.from({ length: NI * 4 }, () => 0.1 * rn());
      ep = 0;
      TR = [rmse(A)];
      TS = [rmse(TE)];
      best = 9;
      bestAt = 0;
      fin = false;
      cu = 0;
      const sm = new Float64Array(NI),
        ct = new Float64Array(NI);
      for (const [u, i, v] of A) {
        sm[i] += v;
        ct[i]++;
      }
      base = Math.sqrt(
        TE.reduce((s, [u, i, v]) => s + (v - (ct[i] ? sm[i] / ct[i] : mu)) ** 2, 0) /
          Math.max(1, TE.length),
      );
      msg =
        'The table holds ' +
        A.length +
        ' real ratings for training and ' +
        TE.length +
        ' set aside. Every person and film has ' +
        S.k +
        ' hidden number' +
        (S.k > 1 ? 's' : '') +
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
        bu[u] += ETA * (e - S.l * bu[u]);
        bi[i] += ETA * (e - S.l * bi[i]);
        for (let k = 0; k < S.k; k++) {
          const a = Pu[u * 4 + k],
            b = Qi[i * 4 + k];
          Pu[u * 4 + k] += ETA * (e * b - S.l * a);
          Qi[i * 4 + k] += ETA * (e * a - S.l * b);
        }
        if (n === 0) {
          ex.pu2 = Pu[u * 4];
          ex.qi2 = Qi[i * 4];
          ex.p1 = pred(u, i);
        }
      });
      ep++;
      const tr = rmse(A0),
        te = rmse(TE);
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
              f2(S.l) +
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
      cu = 4;
      msg =
        'Pass ' +
        ep +
        ': all ' +
        A0.length +
        ' training ratings were visited. Error on them: ' +
        f3(tr) +
        ' rating points. Error on the ' +
        TE.length +
        ' hidden ratings: ' +
        f3(te) +
        '.';
      if ((ep >= 80 && ep - bestAt >= 40) || ep >= MAXE) {
        fin = true;
        cu = 5;
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
        n = MY.filter((v) => v).length;
      if (!n) {
        el.textContent =
          'Click cells in the bottom row of the table to rate films yourself. Each click raises the rating by one, and a sixth click clears it.';
        return;
      }
      const o = [...Array(NI).keys()]
        .filter((i) => !MY[i])
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
      const t = T(),
        A = all(),
        known = new Map(A.map((q) => [q[0] * NI + q[1], q[2]])),
        held = new Map(TE.map((q) => [q[0] * NI + q[1], q[2]]));
      {
        const { g, w, h } = fit(c),
          lx = 54,
          ty = 122,
          cw = (w - lx - 6) / NI,
          rh = (h - ty - 6) / NU,
          fam = MONO.split('px ')[1];
        PC = { lx, ty, cw, rh };
        g.font = MONO;
        for (let i = 0; i < NI; i++) {
          g.save();
          g.translate(lx + (i + 0.5) * cw + 3, ty - 6);
          g.rotate(-Math.PI / 2.5);
          g.fillStyle = i < 4 ? t.cut : i < 8 ? t.accent : t.muted;
          g.textAlign = 'left';
          g.fillText(FM[i], 0, 0);
          g.restore();
        }
        for (let u = 0; u < NU; u++) {
          g.textAlign = 'right';
          g.fillStyle = u === ME ? t.cut : t.ink;
          g.font = (u === ME ? '600 ' : '') + '12px ' + fam;
          g.fillText(US[u], lx - 8, ty + (u + 0.5) * rh + 4);
          for (let i = 0; i < NI; i++) {
            const x = lx + i * cw,
              y = ty + u * rh,
              kv = known.get(u * NI + i),
              hv = held.get(u * NI + i),
              p = clamp(pred(u, i), 1, 5),
              v = kv || p,
              col = v >= 3 ? t.accent : t.cut;
            g.fillStyle = mix(
              col,
              col,
              0,
              (kv ? 0.2 : 0.07) + (Math.abs(v - 3) / 2) * (kv ? 0.6 : 0.3),
            );
            g.fillRect(x + 1, y + 1, cw - 2, rh - 2);
            g.textAlign = 'center';
            if (kv) {
              g.fillStyle = t.ink;
              g.font = '700 ' + Math.min(15, rh * 0.5) + 'px ' + fam;
              g.fillText(String(kv), x + cw / 2, y + rh / 2 + 5);
            } else {
              g.fillStyle = t.muted;
              g.font = Math.min(11, rh * 0.38) + 'px ' + fam;
              g.fillText(ep ? p.toFixed(1) : '·', x + cw / 2, y + rh / 2 + 4);
              if (hv) {
                g.strokeStyle = t.gold;
                g.lineWidth = 2;
                g.strokeRect(x + 2.5, y + 2.5, cw - 5, rh - 5);
                g.fillStyle = t.cut;
                g.font = '9px ' + fam;
                g.fillText('was ' + hv, x + cw / 2, y + rh - 4);
              }
            }
          }
          if (u === ME) {
            g.strokeStyle = t.cut;
            g.lineWidth = 1.5;
            g.setLineDash([4, 3]);
            g.strokeRect(lx + 0.5, ty + u * rh + 0.5, NI * cw - 1, rh - 1);
            g.setLineDash([]);
          }
        }
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.56,
          pts = [];
        for (let u = 0; u < NU; u++)
          if (u < ME || MY.some((v) => v))
            pts.push([Pu[u * 4], S.k > 1 ? Pu[u * 4 + 1] : 0, US[u], 0, u]);
        for (let i = 0; i < NI; i++)
          pts.push([Qi[i * 4], S.k > 1 ? Qi[i * 4 + 1] : 0, FM[i], 1, i]);
        const mx = Math.max(0.2, ...pts.map((p) => Math.abs(p[0]))) * 1.25,
          my = Math.max(0.2, ...pts.map((p) => Math.abs(p[1]))) * 1.25,
          X = (v) => w / 2 + (v / mx) * (w / 2 - 70),
          Y = (v) => 18 + (h1 - 30) / 2 - (v / my) * ((h1 - 40) / 2);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          'hidden number 1 (across) and 2 (up): circles are people, squares are films',
          8,
          13,
        );
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X(-mx), Y(0));
        g.lineTo(X(mx), Y(0));
        g.moveTo(X(0), Y(-my));
        g.lineTo(X(0), Y(my));
        g.stroke();
        for (const p of pts) {
          const x = X(p[0]),
            y = Y(p[1]);
          if (p[3]) {
            g.fillStyle = p[4] < 4 ? t.cut : p[4] < 8 ? t.accent : t.muted;
            g.fillRect(x - 4, y - 4, 8, 8);
          } else {
            g.beginPath();
            g.arc(x, y, 4.5, 0, 7);
            g.fillStyle = p[4] === ME ? t.gold : t.bg;
            g.fill();
            g.strokeStyle = t.ink;
            g.lineWidth = 1.6;
            g.stroke();
          }
          g.fillStyle = p[3] ? t.muted : t.ink;
          g.textAlign = 'left';
          g.fillText(p[2], x + 7, y + 4);
        }
        const n = Math.max(60, TR.length),
          ym = Math.max(1.2, TR[0], TS[0] || 0),
          L = uLine(g, t, 0, h1, w, h - h1, TR, {
            title: 'error in rating points, by pass',
            n,
            ym,
            col: t.accent,
          });
        g.strokeStyle = t.cut;
        g.lineWidth = 2.2;
        g.beginPath();
        TS.forEach((v, i) =>
          i ? g.lineTo(L.X(i), L.Y(Math.min(v, ym))) : g.moveTo(L.X(i), L.Y(Math.min(v, ym))),
        );
        g.stroke();
        g.textAlign = 'right';
        g.fillStyle = t.accent;
        g.fillText('ratings it trains on', w - 14, h1 + 38);
        g.fillStyle = t.cut;
        g.fillText('ratings hidden from it', w - 14, h1 + 52);
      }
      live.textContent = msg;
      uCur(root, 'rc', cu);
      recs();
      setR(root, 'ep', ep + (fin ? ' (stopped)' : ''));
      setR(root, 'tr', f3(TR[TR.length - 1]));
      setR(root, 'te', f3(TS[TS.length - 1]));
      setR(root, 'bl', f3(base));
      setR(root, 'nr', A.length + ' / ' + TE.length);
      setR(root, 'np', String((NU + NI) * (S.k + 1)));
    };
    gen();
    build();
    if (RM) {
      while (!fin) pass();
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      if (fr % (ep < 6 ? 24 : 3)) return;
      pass();
      draw();
      chk();
    });
    pb = playBtn($('#rc-play', root), anim);
    chk = runCtl(pb, root, 'rc', {
      done: () => fin,
      restart: () => build(),
      count: () => ep,
      what: 'passes',
    });
    $('#rc-step', root).onclick = () => {
      pb.stop();
      if (fin) build();
      else pass();
      draw();
    };
    $('#rc-reset', root).onclick = () => {
      pb.stop();
      build();
      draw();
    };
    $('#rc-new', root).onclick = () => {
      pb.stop();
      seed++;
      gen();
      build();
      draw();
      pb.start();
    };
    $('#rc-clr', root).onclick = () => {
      pb.stop();
      MY.fill(0);
      build();
      draw();
    };
    bind(
      root,
      'rc-k',
      (v) => {
        S.k = +v;
        if (pb) {
          pb.stop();
          build();
          draw();
          pb.start();
        }
      },
      (v) => String(v),
    );
    bind(
      root,
      'rc-l',
      (v) => {
        S.l = +v;
        if (pb) {
          pb.stop();
          build();
          draw();
          pb.start();
        }
      },
      f2,
    );
    gwClick(c, (px, py) => {
      if (!PC) return;
      const i = Math.floor((px - PC.lx) / PC.cw),
        u = Math.floor((py - PC.ty) / PC.rh);
      if (u !== ME || i < 0 || i >= NI) return;
      pb.stop();
      MY[i] = (MY[i] + 1) % 6;
      build();
      msg =
        'You ' +
        (MY[i] ? 'gave “' + FM[i] + '” a ' + MY[i] : 'cleared your rating of “' + FM[i] + '”') +
        '. The model starts again with your ratings included. Press Run.';
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
