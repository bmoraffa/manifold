/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  RM,
  T,
  bind,
  clamp,
  ellPath,
  eqShow,
  erf,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { canvasXY, dotAt, segBind, uCur, uLine, uView } from '../../unsupervised/shared.js';
import { g2Note } from '../../generative/shared.js';
import { pct } from '../../ensembles/shared.js';

export default {
  init(root) {
    const c = $('#vb-c', root),
      cr = $('#vb-r', root),
      live = $('#vb-live', root),
      eq = $('#vb-eq', root),
      S = { tg: 'corr', f: 'mf', sp: 2 },
      GW = 64,
      GH = 48,
      XR = 4,
      YR = 3,
      NB = 32,
      K = 128;
    const LP = {
      corr: (x, y) =>
        (-(x * x - (2 * 0.85 * x * y) / 0.9 + (y * y) / 0.81) / (2 * (1 - 0.85 * 0.85))) * 0.8,
      banana: (x, y) => -((x * x) / 4.5 + 2.2 * (y + 0.9 - 0.35 * x * x) ** 2),
      two: (x, y) =>
        Math.log(
          Math.exp(-((x + 1.9) ** 2 + (y + 0.2) ** 2) / 0.55) +
            Math.exp(-((x - 1.9) ** 2 + (y - 0.2) ** 2) / 0.55) +
            1e-300,
        ),
    };
    const LAM = (() => {
        const k = 0.8 / (1 - 0.85 * 0.85);
        return [k, (-k * 0.85) / 0.9, k / 0.81];
      })(),
      ZS = (() => {
        const h = (i, b) => {
            let f = 1,
              r = 0;
            while (i > 0) {
              f /= b;
              r += f * (i % b);
              i = Math.floor(i / b);
            }
            return r;
          },
          o = [];
        for (let i = 1; i <= K; i++) {
          const u = h(i, 2),
            v = h(i, 3),
            R = Math.sqrt(-2 * Math.log(u));
          o.push([R * Math.cos(2 * Math.PI * v), R * Math.sin(2 * Math.PI * v)]);
        }
        return o;
      })();
    let lp,
      dens,
      dmax = 1,
      margT,
      lz = 0,
      tm = [0, 0],
      tC = [1, 0, 1],
      rT = 0,
      th = [0, 0, 0, 0, 0],
      path = [],
      hist = [],
      it = 0,
      fin = false,
      msg = '',
      cu = 0,
      pb,
      chk = () => false,
      P = null,
      ad = null,
      small = 0,
      start = [-2.6, 1.6],
      fr = 0;
    const setup = () => {
      lp = LP[S.tg];
      dens = new Float64Array(GW * GH);
      margT = new Float64Array(NB);
      let tot = 0,
        sx = 0,
        sy = 0,
        sxx = 0,
        sxy = 0,
        syy = 0,
        rr = 0;
      const da = ((2 * XR) / GW) * ((2 * YR) / GH);
      for (let r = 0; r < GH; r++)
        for (let q = 0; q < GW; q++) {
          const px = -XR + (2 * XR * (q + 0.5)) / GW,
            py = YR - (2 * YR * (r + 0.5)) / GH,
            d = Math.exp(lp(px, py));
          dens[r * GW + q] = d;
          tot += d;
          sx += d * px;
          sy += d * py;
          sxx += d * px * px;
          sxy += d * px * py;
          syy += d * py * py;
          if (px > 0) rr += d;
          margT[Math.min(NB - 1, Math.floor(((px + XR) / (2 * XR)) * NB))] += d;
        }
      dmax = Math.max(...dens);
      for (let i = 0; i < NB; i++) margT[i] /= tot;
      lz = Math.log(tot * da);
      tm = [sx / tot, sy / tot];
      tC = [sxx / tot - tm[0] ** 2, sxy / tot - tm[0] * tm[1], syy / tot - tm[1] ** 2];
      rT = rr / tot;
    };
    const cov = (t) => {
      const a = Math.exp(t[2]),
        b = t[4],
        cc = Math.exp(t[3]);
      return [a * a, a * b, b * b + cc * cc];
    };
    const elbo = (t) => {
      const a = Math.exp(t[2]),
        b = t[4],
        cc = Math.exp(t[3]);
      let s = 0;
      for (const z of ZS) s += lp(t[0] + a * z[0], t[1] + b * z[0] + cc * z[1]);
      return s / K + t[2] + t[3] + 1 + Math.log(2 * Math.PI);
    };
    const exact = () => S.tg === 'corr' && S.f === 'mf';
    const reset = () => {
      th = [
        start[0],
        start[1],
        Math.log(exact() ? 1 / Math.sqrt(LAM[0]) : 0.6),
        Math.log(exact() ? 1 / Math.sqrt(LAM[2]) : 0.6),
        0,
      ];
      path = [[th[0], th[1]]];
      it = 0;
      fin = false;
      ad = null;
      small = 0;
      cu = 1;
      hist = [elbo(th)];
      if (S.f === 'fwd') {
        const L11 = Math.sqrt(tC[0]),
          L21 = tC[1] / L11,
          L22 = Math.sqrt(Math.max(1e-9, tC[2] - L21 * L21));
        th = [tm[0], tm[1], Math.log(L11), Math.log(L22), L21];
        path = [[th[0], th[1]]];
        hist = [elbo(th)];
        fin = true;
        cu = 5;
        msg =
          'Moment matching needs no iterations: the bell is given the mean and covariance of the true belief, computed here from the grid. It covers everything the truth covers' +
          (S.tg === 'two'
            ? ', including the empty gap between the islands, where it puts its peak.'
            : S.tg === 'banana'
              ? ', and a good deal of empty space inside the curve of the banana.'
              : ', and for a bell-shaped truth that is the exact answer.');
        eqShow(
          eq,
          [
            'mean of the truth: (' + f2(tm[0]) + ', ' + f2(tm[1]) + ')',
            'variances: ' +
              f2(tC[0]) +
              ' left-right, ' +
              f2(tC[2]) +
              ' up-down; covariance ' +
              f2(tC[1]),
            'q is the bell with exactly these moments. This minimizes KL(p ‖ q).',
          ],
          0,
        );
        return;
      }
      msg =
        'The approximation starts at (' +
        f2(th[0]) +
        ', ' +
        f2(th[1]) +
        '). ' +
        (exact()
          ? 'Each step will update one coordinate exactly.'
          : 'Each step is one gradient step on the ELBO.') +
        ' Press One step or Run.';
      eqShow(eq, ['No steps yet.'], 0);
    };
    const step = () => {
      if (fin) return;
      const e0 = hist[hist.length - 1];
      if (exact()) {
        const j = it % 2,
          old = th[j],
          o = th[1 - j],
          nv = (-LAM[1] / (j ? LAM[2] : LAM[0])) * o;
        th[j] = nv;
        eqShow(
          eq,
          [
            'Step ' +
              (it + 1) +
              ': update the ' +
              (j ? 'up-down' : 'left-right') +
              ' coordinate, holding the other at ' +
              f3(o) +
              '.',
            'm' +
              (j + 1) +
              ' ← μ' +
              (j + 1) +
              ' − (Λ₁₂ ÷ Λ' +
              (j + 1) +
              (j + 1) +
              ') × (m' +
              (2 - j) +
              ' − μ' +
              (2 - j) +
              ') = 0 − (' +
              f3(LAM[1]) +
              ' ÷ ' +
              f3(j ? LAM[2] : LAM[0]) +
              ') × ' +
              f3(o) +
              ' = <b>' +
              f3(nv) +
              '</b>',
            'its spread stays at 1 ÷ √Λ' +
              (j + 1) +
              (j + 1) +
              ' = <b>' +
              f3(1 / Math.sqrt(j ? LAM[2] : LAM[0])) +
              '</b> (the true spread is ' +
              f3(Math.sqrt(j ? tC[2] : tC[0])) +
              ')',
            'the centre moved by ' + f3(Math.abs(nv - old)),
          ],
          0,
        );
      } else {
        if (!ad) ad = { m: [0, 0, 0, 0, 0], v: [0, 0, 0, 0, 0], t: 0 };
        const np = S.f === 'full' ? 5 : 4,
          g = [0, 0, 0, 0, 0];
        for (let i = 0; i < np; i++) {
          const a = th.slice(),
            b = th.slice();
          a[i] += 1e-4;
          b[i] -= 1e-4;
          g[i] = (elbo(a) - elbo(b)) / 2e-4;
        }
        ad.t++;
        const lr = 0.06 / (1 + ad.t / 150);
        for (let i = 0; i < np; i++) {
          ad.m[i] = 0.85 * ad.m[i] + 0.15 * g[i];
          ad.v[i] = 0.99 * ad.v[i] + 0.01 * g[i] * g[i];
          th[i] +=
            (lr * (ad.m[i] / (1 - 0.85 ** ad.t))) /
            (Math.sqrt(ad.v[i] / (1 - 0.99 ** ad.t)) + 1e-8);
        }
        th[2] = clamp(th[2], -4, 1.5);
        th[3] = clamp(th[3], -4, 1.5);
        if (it % 3 === 0 || !anim.on)
          eqShow(
            eq,
            [
              'Step ' +
                (it + 1) +
                ': slope of the ELBO with respect to the centre: (' +
                f3(g[0]) +
                ', ' +
                f3(g[1]) +
                ')',
              'slope with respect to the log-spreads: (' +
                f3(g[2]) +
                ', ' +
                f3(g[3]) +
                ')' +
                (S.f === 'full' ? ', and to the tilt: ' + f3(g[4]) : ''),
              'centre → (' +
                f3(th[0]) +
                ', ' +
                f3(th[1]) +
                '), spreads → ' +
                f3(Math.sqrt(cov(th)[0])) +
                ' and ' +
                f3(Math.sqrt(cov(th)[2])),
            ],
            0,
          );
      }
      it++;
      path.push([th[0], th[1]]);
      const e1 = elbo(th);
      hist.push(e1);
      cu = exact() ? 3 : 3;
      const gain = e1 - e0;
      small = Math.abs(gain) < 1e-5 ? small + 1 : 0;
      msg =
        'Step ' +
        it +
        ': ELBO ' +
        f3(e1) +
        ' (' +
        (gain >= 0 ? '+' : '') +
        (Math.abs(gain) < 1e-3 ? gain.toExponential(1) : f3(gain)) +
        '). Gap to the true log evidence: ' +
        f3(Math.max(0, lz - e1)) +
        '.';
      if (small >= (exact() ? 2 : 10) || it >= 2500) {
        fin = true;
        cu = 5;
        const C = cov(th),
          rx = Math.sqrt(C[0] / tC[0]),
          ry = Math.sqrt(C[2] / tC[2]);
        msg =
          'Stopped after ' +
          it +
          ' steps: the ELBO has stopped rising at ' +
          f3(e1) +
          ', ' +
          f3(Math.max(0, lz - e1)) +
          ' below the true log evidence. ' +
          (S.tg === 'two'
            ? 'The approximation has settled on the ' +
              (th[0] < 0 ? 'left' : 'right') +
              ' island and ignores the other, which holds half of the probability.'
            : rx < 0.8 || ry < 0.8
              ? 'Its spread is ' +
                pct(rx) +
                ' of the truth left-to-right and ' +
                pct(ry) +
                ' up-down: too confident.'
              : 'Its spread matches the truth (' + pct(rx) + ' and ' + pct(ry) + ').');
      }
    };
    const qDens = (x, y) => {
      const C = cov(th),
        dt = C[0] * C[2] - C[1] * C[1],
        dx = x - th[0],
        dy = y - th[1];
      return (
        Math.exp((-0.5 * (C[2] * dx * dx - 2 * C[1] * dx * dy + C[0] * dy * dy)) / dt) /
        (2 * Math.PI * Math.sqrt(dt))
      );
    };
    const draw = () => {
      const t = T(),
        C = cov(th);
      {
        const V = uView(c),
          { g, w, h, X, Y } = V;
        P = V;
        for (let r = 0; r < GH; r++)
          for (let q = 0; q < GW; q++) {
            const v = dens[r * GW + q] / dmax;
            if (v < 0.01) continue;
            g.fillStyle = mix(t.accent, t.accent, 0, Math.min(0.85, Math.pow(v, 0.55) * 0.8));
            const xa = Math.round(X(-XR + (2 * XR * q) / GW)),
              xb = Math.round(X(-XR + (2 * XR * (q + 1)) / GW)),
              ya = Math.round(Y(YR - (2 * YR * r) / GH)),
              yb = Math.round(Y(YR - (2 * YR * (r + 1)) / GH));
            g.fillRect(xa, ya, xb - xa, yb - ya);
          }
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.7);
        g.lineWidth = 1.4;
        g.beginPath();
        path.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
        g.stroke();
        g.strokeStyle = t.gold;
        for (const r of [1, 2]) {
          g.lineWidth = r === 1 ? 2.8 : 1.6;
          ellPath(g, X, Y, th[0], th[1], C[0], C[1], C[2], r);
          g.stroke();
        }
        g.lineWidth = 2;
        dotAt(g, X(th[0]), Y(th[1]), 5, t.gold, t.bg);
        g2Note(
          g,
          t,
          h,
          S.f === 'mf'
            ? 'mean-field: the ellipse must stay upright'
            : S.f === 'full'
              ? 'full bell: the ellipse may tilt'
              : 'moment matching: same mean and covariance as the truth',
        );
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5,
          qm = Array.from({ length: NB }, (_, i) => {
            const x = -XR + (2 * XR * (i + 0.5)) / NB;
            return (
              (Math.exp((-0.5 * (x - th[0]) ** 2) / C[0]) / Math.sqrt(2 * Math.PI * C[0])) *
              ((2 * XR) / NB)
            );
          }),
          ym = Math.max(...margT, ...qm) * 1.08,
          A = uLine(g, t, 0, 0, w, h1, [], {
            title: 'left-to-right spread: truth (line), approximation (filled)',
            n: NB,
            ym,
            y1: 0,
          });
        g.fillStyle = mix(t.gold, t.gold, 0, 0.4);
        g.beginPath();
        g.moveTo(A.X(0), A.Y(0));
        qm.forEach((v, i) => g.lineTo(A.X(i), A.Y(v)));
        g.lineTo(A.X(NB - 1), A.Y(0));
        g.fill();
        g.strokeStyle = t.accent;
        g.lineWidth = 2.2;
        g.beginPath();
        for (let i = 0; i < NB; i++)
          i ? g.lineTo(A.X(i), A.Y(margT[i])) : g.moveTo(A.X(i), A.Y(margT[i]));
        g.stroke();
        const lo = Math.max(Math.min(...hist), lz - 6),
          hi = lz + 0.15,
          B = uLine(g, t, 0, h1, w, h - h1, hist, {
            title: 'ELBO (solid) and its ceiling, the true log evidence (dashed)',
            n: Math.max(20, hist.length),
            y1: lo - 0.1,
            ym: hi,
          });
        g.strokeStyle = t.cut;
        g.setLineDash([6, 4]);
        g.lineWidth = 1.6;
        g.beginPath();
        g.moveTo(B.X(0), B.Y(lz));
        g.lineTo(B.X(Math.max(20, hist.length) - 1), B.Y(lz));
        g.stroke();
        g.setLineDash([]);
        g.save();
        g.beginPath();
        g.rect(0, h1 + 16, w, h - h1 - 34);
        g.clip();
        g.strokeStyle = t.accent;
        g.lineWidth = 2.2;
        g.beginPath();
        hist.forEach((v, i) => (i ? g.lineTo(B.X(i), B.Y(v)) : g.moveTo(B.X(i), B.Y(v))));
        g.stroke();
        g.restore();
        if (hist.length === 1) dotAt(g, B.X(0), B.Y(Math.max(lo, hist[0])), 4, t.accent);
      }
      const e = hist[hist.length - 1],
        rq = 0.5 * (1 + erf(th[0] / Math.sqrt(2 * C[0])));
      live.textContent = msg;
      uCur(root, 'vb', cu);
      setR(root, 'it', S.f === 'fwd' ? 'none needed' : it + (fin ? ' (stopped)' : ''));
      setR(root, 'elbo', f3(e));
      setR(root, 'lz', f3(lz));
      setR(
        root,
        'kl',
        S.f === 'fwd'
          ? f3(Math.max(0, lz - e)) + ' (not what this method minimizes)'
          : f3(Math.max(0, lz - e)),
      );
      setR(root, 'sx', f2(Math.sqrt(C[0] / tC[0])) + ', ' + f2(Math.sqrt(C[2] / tC[2])));
      setR(root, 'isl', pct(rq) + ' / ' + pct(rT));
    };
    setup();
    reset();
    if (RM) {
      while (!fin) step();
    }
    const anim = Anim(() => {
      if (fin) {
        pb.stop();
        return;
      }
      fr++;
      if (exact() || S.sp === 1) {
        if (fr % (S.sp === 3 ? 8 : S.sp === 2 ? 22 : 40)) return;
        step();
      } else {
        const k = S.sp === 2 ? 2 : 8;
        for (let i = 0; i < k && !fin && chk.remaining() > 0; i++) step();
      }
      draw();
      chk();
    });
    pb = playBtn($('#vb-play', root), anim);
    chk = runCtl(pb, root, 'vb', {
      done: () => fin,
      restart: reset,
      count: () => it,
      what: 'steps',
    });
    const go = () => {
      pb.stop();
      reset();
      draw();
      if (S.f !== 'fwd') {
        fr = 0;
        pb.start();
      }
    };
    $('#vb-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#vb-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    bind(
      root,
      'vb-sp',
      (v) => {
        S.sp = v;
      },
      (v) => ['', 'slow', 'steady', 'fast'][+v],
    );
    segBind(root, 'vb-tg', S.tg, (v) => {
      S.tg = v;
      start = v === 'two' ? [-1, 0.9] : [-2.6, 1.6];
      setup();
      go();
    });
    segBind(root, 'vb-f', S.f, (v) => {
      S.f = v;
      go();
    });
    c.style.cursor = 'crosshair';
    c.addEventListener('click', (e) => {
      if (!P) return;
      const [x, y] = canvasXY(c, P, e);
      start = [clamp(x, -XR, XR), clamp(y, -YR, YR)];
      go();
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        fr = 0;
        if (S.f !== 'fwd') pb.start();
      },
    };
  },
};
