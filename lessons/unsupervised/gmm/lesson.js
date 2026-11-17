/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  SANS,
  readThemeColors,
  bind,
  clamp,
  ellPath,
  f2,
  f3,
  prepareCanvas,
  hex,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { Orbit } from '../../regression/shared.js';
import {
  UC,
  UKIND,
  kmSeed,
  segBind,
  highlightLessonStep,
  uData,
  uDataBind,
  uFrame,
  drawSeriesChart,
  uNum,
  uView,
} from '../shared.js';

export default {
  init(root) {
    const mainCanvas = $('#gm-c', root),
      cr = $('#gm-r', root),
      live = $('#gm-live', root),
      cap = $('#gm-cap', root),
      fc = $('#gm-fc', root),
      vb = $('#gm-vb', root),
      tab = $('#gm-tab', root),
      dsel = $('#gm-ds', root),
      settings = { dim: '1', K: 2, cov: 'full', ds: 'h2', n: 240, seed: 2 };
    let P = [],
      G = [],
      previousComponents = null,
      responsibilities = null,
      ph = 'E',
      it = 0,
      logLikelihoodHistory = [],
      msg = '',
      activeStepIndex = 0,
      mv = 0,
      frameCount = 0,
      playback,
      run = 1,
      checkRunStop = () => false;
    const one = () => settings.dim === '1',
      CAP = {
        1: 'Each dot under the axis is one measurement. Each coloured curve is one cluster and the black curve is their sum, which should follow the grey histogram. The band at the bottom shows, for every position, how the clusters would split a point that fell there.',
        2: 'Each ellipse is one cluster: the inner ring is one standard deviation, the outer ring two. A point’s colour is a blend of the cluster colours, in proportion to how the clusters share it.',
        3: 'The same fit as Two Dimensions, drawn as a landscape. Height is the mixture density, each hill is one cluster, and the dots on the floor are the data. Drag to rotate.',
      };
    const mk = () => {
      if (!one()) return uData(settings.ds, settings.n, settings.seed);
      const r = rng(settings.seed * 11 + 2),
        rn = randnFrom(r),
        spec = {
          h2: [
            [0.45, -1.5, 0.55],
            [0.55, 1.2, 0.75],
          ],
          h3: [
            [0.3, -2.3, 0.4],
            [0.4, 0, 0.5],
            [0.3, 2.2, 0.45],
          ],
          hov: [
            [0.5, -0.6, 0.8],
            [0.5, 0.9, 0.8],
          ],
        }[settings.ds] || [[1, 0, 1]],
        o = [];
      for (let i = 0; i < settings.n; i++) {
        let u = r(),
          k = 0;
        while (k < spec.length - 1 && (u -= spec[k][0]) > 0) k++;
        o.push({ x: clamp(spec[k][1] + spec[k][2] * rn(), -3.8, 3.8), y: 0 });
      }
      return o;
    };
    const d1 = (q, x) => Math.exp((-0.5 * (x - q.mx) ** 2) / q.a) / Math.sqrt(2 * Math.PI * q.a);
    const dG = (q, x, y) => {
      const det = Math.max(1e-12, q.a * q.d - q.b * q.b),
        dx = x - q.mx,
        dy = y - q.my;
      return (
        Math.exp((-0.5 * (q.d * dx * dx - 2 * q.b * dx * dy + q.a * dy * dy)) / det) /
        (2 * Math.PI * Math.sqrt(det))
      );
    };
    const dens = (q, x, y) => (one() ? d1(q, x) : dG(q, x, y));
    const respG = (GG, x, y) => {
        const v = GG.map((q) => q.pi * dens(q, x, y)),
          s = v.reduce((a, b) => a + b, 0);
        return { r: v.map((u) => (s > 0 ? u / s : 1 / GG.length)), s };
      },
      resp = (x, y) => respG(G, x, y);
    const where = (p) => (one() ? 'x = ' + f2(p[0]) : '(' + f2(p[0]) + ', ' + f2(p[1]) + ')');
    const reset = () => {
      const n = P.length || 1,
        mx = P.reduce((s, p) => s + p.x, 0) / n,
        my = P.reduce((s, p) => s + p.y, 0) / n,
        vx = P.reduce((s, p) => s + (p.x - mx) ** 2, 0) / n,
        vy = P.reduce((s, p) => s + (p.y - my) ** 2, 0) / n,
        v = one()
          ? Math.max(0.08, vx / (settings.K * settings.K))
          : Math.max(0.12, (vx + vy) / 2 / settings.K);
      G = kmSeed(P, settings.K, true, rng(300 + run * 5 + settings.seed)).map((q) => ({
        pi: 1 / settings.K,
        mx: q.x,
        my: one() ? 0 : q.y,
        a: v,
        b: 0,
        d: one() ? 1 : v,
      }));
      previousComponents = null;
      responsibilities = null;
      ph = 'E';
      it = 0;
      logLikelihoodHistory = [];
      activeStepIndex = 0;
      mv = 0;
      msg =
        'Start: ' +
        G.length +
        ' bell' +
        (G.length === 1 ? '' : 's') +
        ' placed on data points chosen by k-means++, all with the same spread and each with weight 1/' +
        G.length +
        '. The points are grey because nothing has been shared out yet. Press Run or One step.';
    };
    const step = () => {
      if (P.length < 2 || !G.length) return;
      if (ph === 'done') {
        run++;
        reset();
        return;
      }
      if (ph === 'E') {
        activeStepIndex = 1;
        let ll = 0,
          sh = 0;
        responsibilities = P.map((p) => {
          const q = resp(p.x, p.y);
          ll += Math.log(q.s + 1e-300);
          if (Math.max(...q.r) < 0.9) sh++;
          return q.r;
        });
        const prevL = logLikelihoodHistory.length
          ? logLikelihoodHistory[logLikelihoodHistory.length - 1]
          : null;
        logLikelihoodHistory.push(ll);
        const ex = resp(tp.v[0], one() ? 0 : tp.v[1]).r;
        if ((prevL !== null && Math.abs(ll - prevL) < 1e-4 * P.length) || it >= 300) {
          ph = 'done';
          activeStepIndex = 3;
          msg =
            (it >= 300
              ? 'Reached the limit of 300 rounds.'
              : 'Converged: the log-likelihood rose by less than a ten-thousandth per point in the last round, so the bells have stopped moving.') +
            ' It took ' +
            it +
            ' rounds and the log-likelihood is ' +
            uNum(ll) +
            '. ' +
            sh +
            ' of ' +
            P.length +
            ' points are still shared between clusters.';
        } else {
          ph = 'M';
          msg =
            'E-step: the bells stayed where they were, and every point was shared out among them in proportion to how well each explains it. Example: your point at ' +
            where(tp.v) +
            ' is ' +
            ex.map((v, k) => (100 * v).toFixed(0) + '% cluster ' + (k + 1)).join(', ') +
            '. ' +
            sh +
            ' of ' +
            P.length +
            ' points are split noticeably. Log-likelihood: ' +
            uNum(ll) +
            (prevL !== null ? ', up from ' + uNum(prevL) : '') +
            '.';
        }
        return;
      }
      if (ph === 'M') {
        activeStepIndex = 2;
        previousComponents = G.map((q) => ({ ...q }));
        const n = P.length;
        G.forEach((q, k) => {
          let N = 0,
            sx = 0,
            sy = 0;
          P.forEach((p, i) => {
            const r = responsibilities[i][k];
            N += r;
            sx += r * p.x;
            sy += r * p.y;
          });
          if (N < 1e-6) return;
          const mx = sx / N,
            my = sy / N;
          let a = 0,
            b = 0,
            d = 0;
          P.forEach((p, i) => {
            const r = responsibilities[i][k],
              dx = p.x - mx,
              dy = p.y - my;
            a += r * dx * dx;
            b += r * dx * dy;
            d += r * dy * dy;
          });
          a = a / N + 2e-3;
          d = d / N + 2e-3;
          b = b / N;
          if (one()) {
            Object.assign(q, { pi: N / n, mx, my: 0, a, b: 0, d: 1 });
            return;
          }
          if (settings.cov === 'diag') b = 0;
          if (settings.cov === 'sph') {
            a = d = (a + d) / 2;
            b = 0;
          }
          Object.assign(q, { pi: N / n, mx, my, a, b, d });
        });
        it++;
        mv = prefersReducedMotion ? 0 : 18;
        ph = 'E';
        msg =
          'M-step: the shares stayed fixed, and each bell was refitted to the points it claims. Centres moved to ' +
          G.map((q, k) => (one() ? f2(q.mx) : '(' + f2(q.mx) + ', ' + f2(q.my) + ')')).join(', ') +
          '. Weights became π = ' +
          G.map((q) => f2(q.pi)).join(', ') +
          ': each is that cluster’s share of all the points.';
      }
    };
    const tp = testPoint(
      root,
      'gm',
      ['Your x', 'Your y'],
      [-0.2, 0.7],
      [
        [-3.8, 3.8],
        [-2.85, 2.85],
      ],
    );
    const orb = Orbit(
      mainCanvas,
      () => {
        if (settings.dim === '3') draw();
      },
      vb,
      { az: -0.55, el: 0.62, zoom: 1 },
    );
    const draw = () => {
      const colors = readThemeColors(),
        col = UC(colors),
        rgb = col.map(hex),
        bl = (r) => {
          let R = 0,
            Gn = 0,
            B = 0;
          r.forEach((v, k) => {
            R += v * rgb[k % 8][0];
            Gn += v * rgb[k % 8][1];
            B += v * rgb[k % 8][2];
          });
          return [Math.round(R), Math.round(Gn), Math.round(B)];
        },
        css = (q, al = 1) => 'rgba(' + q[0] + ',' + q[1] + ',' + q[2] + ',' + al + ')';
      const u = previousComponents && mv > 0 ? 1 - mv / 18 : 1,
        GG = G.map((q, k) => {
          const o = previousComponents && previousComponents[k] ? previousComponents[k] : q,
            L = (f) => o[f] + (q[f] - o[f]) * u;
          return { pi: L('pi'), mx: L('mx'), my: L('my'), a: L('a'), b: L('b'), d: L('d') };
        }),
        ty = one() ? 0 : tp.v[1];
      cap.textContent = CAP[settings.dim];
      fc.textContent = one()
        ? 'Histogram of the data, the bell of each cluster, and their sum'
        : settings.dim === '2'
          ? 'The points and the cluster ellipses'
          : 'The mixture density as a landscape';
      if (one()) {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          X = (x) => w / 2 + (x * (w - 40)) / 8,
          base = h - 84,
          top = 34,
          nb = 32,
          bw = 8 / nb,
          cnt = new Array(nb).fill(0);
        for (const p of P) cnt[clamp(Math.floor((p.x + 4) / bw), 0, nb - 1)]++;
        const hd = cnt.map((v) => v / (P.length * bw));
        const mixf = (x) => GG.reduce((s, q) => s + q.pi * d1(q, x), 0);
        let ym = Math.max(...hd, 0.05);
        for (let i = 0; i <= 200; i++) ym = Math.max(ym, mixf(-4 + i * 0.04));
        ym *= 1.12;
        const Y = (v) => base - (v / ym) * (base - top);
        ctx.fillStyle = mix(colors.ink, colors.ink, 0, 0.11);
        hd.forEach((v, i) => ctx.fillRect(X(-4 + i * bw) + 1, Y(v), X(bw) - X(0) - 2, base - Y(v)));
        GG.forEach((q, k) => {
          const f = (x) => q.pi * d1(q, x);
          ctx.beginPath();
          ctx.moveTo(X(-4), base);
          for (let i = 0; i <= 240; i++) {
            const x = -4 + i / 30;
            ctx.lineTo(X(x), Y(f(x)));
          }
          ctx.lineTo(X(4), base);
          ctx.closePath();
          ctx.fillStyle = mix(col[k % 8], col[k % 8], 0, 0.2);
          ctx.fill();
          ctx.beginPath();
          for (let i = 0; i <= 240; i++) {
            const x = -4 + i / 30;
            i ? ctx.lineTo(X(x), Y(f(x))) : ctx.moveTo(X(x), Y(f(x)));
          }
          ctx.strokeStyle = col[k % 8];
          ctx.lineWidth = 2.6;
          ctx.stroke();
          const pk = Y(f(q.mx));
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(X(q.mx), base);
          ctx.lineTo(X(q.mx), pk);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.font = SANS;
          ctx.fillStyle = col[k % 8];
          ctx.textAlign = 'center';
          ctx.fillText('cluster ' + (k + 1), clamp(X(q.mx), 50, w - 50), Math.max(14, pk - 20));
          ctx.font = MONO;
          ctx.fillText(
            'centre ' + f2(q.mx) + ', spread ' + f2(Math.sqrt(q.a)),
            clamp(X(q.mx), 80, w - 80),
            Math.max(27, pk - 7),
          );
        });
        ctx.beginPath();
        for (let i = 0; i <= 240; i++) {
          const x = -4 + i / 30;
          i ? ctx.lineTo(X(x), Y(mixf(x))) : ctx.moveTo(X(x), Y(mixf(x)));
        }
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.8;
        ctx.stroke();
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(X(-4), base);
        ctx.lineTo(X(4), base);
        ctx.stroke();
        P.forEach((p, i) => {
          ctx.fillStyle = responsibilities
            ? css(bl(responsibilities[i]))
            : mix(colors.ink, colors.ink, 0, 0.4);
          ctx.beginPath();
          ctx.arc(X(p.x), base + 8 + (i % 4) * 4.5, 2.5, 0, 7);
          ctx.fill();
        });
        const by = base + 34,
          bh = 24;
        for (let px = X(-4); px < X(4); px += 3) {
          const r = respG(GG, ((px + 1.5 - w / 2) * 8) / (w - 40), 0).r;
          let y = by;
          r.forEach((v, k) => {
            ctx.fillStyle = col[k % 8];
            ctx.fillRect(px, y, 3.2, v * bh + 0.4);
            y += v * bh;
          });
        }
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        [-3, -2, -1, 0, 1, 2, 3].forEach((v) => ctx.fillText(String(v), X(v), by + bh + 14));
        ctx.textAlign = 'left';
        ctx.fillText('band: who owns a point at each position', X(-4), h - 4);
        const x = X(tp.v[0]);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, top - 8);
        ctx.lineTo(x, by + bh);
        ctx.stroke();
        markTP(ctx, x, top - 8, colors);
      } else if (settings.dim === '2') {
        const V = uView(mainCanvas),
          { g, w, h, X, Y } = V;
        uFrame(
          V,
          colors,
          settings.ds === 'shop' ? 'visits per month (standardised)' : 'x',
          settings.ds === 'shop' ? 'average basket (standardised)' : 'y',
        );
        GG.forEach((q, k) => {
          ellPath(g, X, Y, q.mx, q.my, q.a, q.b, q.d, 2);
          g.fillStyle = mix(col[k % 8], col[k % 8], 0, 0.09);
          g.fill();
          g.strokeStyle = mix(col[k % 8], col[k % 8], 0, 0.6);
          g.lineWidth = 1.2;
          g.stroke();
          ellPath(g, X, Y, q.mx, q.my, q.a, q.b, q.d, 1);
          g.strokeStyle = col[k % 8];
          g.lineWidth = 2.4;
          g.stroke();
        });
        P.forEach((p, i) => {
          g.beginPath();
          g.arc(X(p.x), Y(p.y), 3.4, 0, 7);
          g.fillStyle = responsibilities
            ? css(bl(responsibilities[i]))
            : mix(colors.ink, colors.ink, 0, 0.42);
          g.fill();
        });
        GG.forEach((q, k) => {
          if (previousComponents && previousComponents[k]) {
            g.strokeStyle = colors.ink;
            g.lineWidth = 1.4;
            g.setLineDash([4, 4]);
            g.beginPath();
            g.moveTo(X(previousComponents[k].mx), Y(previousComponents[k].my));
            g.lineTo(X(q.mx), Y(q.my));
            g.stroke();
            g.setLineDash([]);
          }
          g.fillStyle = col[k % 8];
          g.strokeStyle = colors.bg;
          g.lineWidth = 4;
          g.beginPath();
          g.rect(X(q.mx) - 6, Y(q.my) - 6, 12, 12);
          g.stroke();
          g.fill();
          g.strokeStyle = colors.ink;
          g.lineWidth = 1.4;
          g.strokeRect(X(q.mx) - 6, Y(q.my) - 6, 12, 12);
          g.font = SANS;
          g.fillStyle = colors.ink;
          g.textAlign = 'left';
          g.fillText(String(k + 1), X(q.mx) + 10, Y(q.my) - 8);
        });
        markTP(g, X(tp.v[0]), Y(tp.v[1]), colors);
      } else {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          pr = orb.cam(w, h, 9.6),
          nx = 44,
          ny = 33,
          Z = [],
          Cc = [];
        let zm = 1e-9;
        for (let i = 0; i <= nx; i++) {
          Z.push([]);
          for (let j = 0; j <= ny; j++) {
            const q = respG(GG, -4 + (8 * i) / nx, -3 + (6 * j) / ny);
            Z[i].push(q.s);
            zm = Math.max(zm, q.s);
          }
        }
        const zs = 2.7 / zm;
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.beginPath();
        [
          [-4, -3],
          [4, -3],
          [4, 3],
          [-4, 3],
          [-4, -3],
        ].forEach((q, i) => {
          const p = pr(q[0], q[1], 0);
          i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
        });
        ctx.stroke();
        P.forEach((p, i) => {
          const q = pr(p.x, p.y, 0);
          ctx.fillStyle = responsibilities
            ? css(bl(responsibilities[i]), 0.9)
            : mix(colors.ink, colors.ink, 0, 0.42);
          ctx.beginPath();
          ctx.arc(q[0], q[1], 2.2, 0, 7);
          ctx.fill();
        });
        const Q = [];
        for (let i = 0; i < nx; i++)
          for (let j = 0; j < ny; j++) {
            const z = (Z[i][j] + Z[i + 1][j] + Z[i][j + 1] + Z[i + 1][j + 1]) / 4;
            if (z < zm * 0.03) continue;
            const x0 = -4 + (8 * i) / nx,
              x1 = -4 + (8 * (i + 1)) / nx,
              y0 = -3 + (6 * j) / ny,
              y1 = -3 + (6 * (j + 1)) / ny,
              a = pr(x0, y0, zs * Z[i][j]),
              b = pr(x1, y0, zs * Z[i + 1][j]),
              cc = pr(x1, y1, zs * Z[i + 1][j + 1]),
              d = pr(x0, y1, zs * Z[i][j + 1]);
            Q.push({
              a,
              b,
              c: cc,
              d,
              z,
              dp: (a[2] + b[2] + cc[2] + d[2]) / 4,
              r: respG(GG, (x0 + x1) / 2, (y0 + y1) / 2).r,
            });
          }
        Q.sort((p, q) => q.dp - p.dp);
        ctx.lineWidth = 0.5;
        for (const o of Q) {
          ctx.beginPath();
          ctx.moveTo(o.a[0], o.a[1]);
          ctx.lineTo(o.b[0], o.b[1]);
          ctx.lineTo(o.c[0], o.c[1]);
          ctx.lineTo(o.d[0], o.d[1]);
          ctx.closePath();
          ctx.fillStyle = css(bl(o.r), 0.35 + 0.6 * Math.pow(o.z / zm, 0.6));
          ctx.fill();
          ctx.strokeStyle = mix(colors.ink, colors.ink, 0, 0.13);
          ctx.stroke();
        }
        GG.forEach((q, k) => {
          const p = pr(q.mx, q.my, zs * respG(GG, q.mx, q.my).s);
          ctx.font = SANS;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText('cluster ' + (k + 1), p[0], p[1] - 8);
        });
        const a = pr(tp.v[0], tp.v[1], 0),
          b = pr(tp.v[0], tp.v[1], zs * respG(GG, tp.v[0], tp.v[1]).s);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        markTP(ctx, b[0], b[1], colors);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          q = respG(GG, tp.v[0], ty),
          ha = h * 0.24,
          hb = h * 0.3;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('your point at ' + where(tp.v) + ' is shared like this', 8, 13);
        let x = 10;
        const bwid = w - 20;
        q.r.forEach((v, k) => {
          ctx.fillStyle = col[k % 8];
          ctx.fillRect(x, 22, v * bwid, 30);
          if (v * bwid > 46) {
            ctx.fillStyle = colors.bg;
            ctx.font = SANS;
            ctx.textAlign = 'center';
            ctx.fillText((100 * v).toFixed(0) + '%', x + (v * bwid) / 2, 42);
          }
          x += v * bwid;
        });
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.strokeRect(10, 22, bwid, 30);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          'weight π of each cluster = its share of all ' + P.length + ' points',
          8,
          ha + 13,
        );
        const rh = Math.min(24, (hb - 22) / Math.max(1, GG.length));
        GG.forEach((z, k) => {
          const y = ha + 20 + k * rh;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'left';
          ctx.fillText('cluster ' + (k + 1), 10, y + rh * 0.65);
          ctx.fillStyle = mix(colors.ink, colors.ink, 0, 0.08);
          ctx.fillRect(84, y + 2, w - 190, rh - 5);
          ctx.fillStyle = col[k % 8];
          ctx.fillRect(84, y + 2, (w - 190) * z.pi, rh - 5);
          ctx.fillStyle = colors.muted;
          ctx.fillText(
            'π = ' + f2(z.pi) + ', ≈ ' + Math.round(z.pi * P.length) + ' pts',
            w - 100,
            y + rh * 0.65,
          );
        });
        const lo = logLikelihoodHistory.length ? Math.min(...logLikelihoodHistory) : 0,
          hi = logLikelihoodHistory.length ? Math.max(...logLikelihoodHistory) : 1,
          pad = Math.max(1e-6, (hi - lo) * 0.08);
        drawSeriesChart(ctx, colors, 0, ha + hb, w, h - ha - hb, logLikelihoodHistory, {
          title: 'log-likelihood after each E-step: it can only rise',
          dots: true,
          n: Math.max(8, logLikelihoodHistory.length),
          y1: lo - pad,
          ym: hi + pad,
          col: colors.gold,
        });
      }
      if (!mv)
        tab.innerHTML =
          '<table><thead><tr><th>cluster</th><th>weight π</th><th>about how many points</th><th>centre</th>' +
          (one()
            ? '<th>spread σ</th>'
            : '<th>spread along x</th><th>spread along y</th><th>tilt (correlation)</th>') +
          '</tr></thead><tbody>' +
          G.map(
            (q, k) =>
              '<tr><td><i style="background:' +
              col[k % 8] +
              '"></i>' +
              (k + 1) +
              '</td><td>' +
              f2(q.pi) +
              '</td><td>' +
              (q.pi * P.length).toFixed(1) +
              '</td><td>' +
              (one() ? f2(q.mx) : '(' + f2(q.mx) + ', ' + f2(q.my) + ')') +
              '</td>' +
              (one()
                ? '<td>' + f2(Math.sqrt(q.a)) + '</td>'
                : '<td>' +
                  f2(Math.sqrt(q.a)) +
                  '</td><td>' +
                  f2(Math.sqrt(q.d)) +
                  '</td><td>' +
                  f2(q.b / Math.sqrt(q.a * q.d)) +
                  '</td>') +
              '</tr>',
          ).join('') +
          '</tbody></table>';
      live.textContent = msg;
      highlightLessonStep(root, 'gm', activeStepIndex);
      {
        const q = resp(tp.v[0], ty),
          b = q.r.indexOf(Math.max(...q.r));
        tp.out.textContent =
          'Your point at ' +
          where(tp.v) +
          ' is shared as ' +
          q.r.map((v, k) => (100 * v).toFixed(0) + '% cluster ' + (k + 1)).join(', ') +
          '. A hard answer would be cluster ' +
          (b + 1) +
          '. The mixture density there is ' +
          f3(q.s) +
          '.';
      }
      const L = logLikelihoodHistory.length;
      setReadout(root, 'it', String(it));
      setReadout(root, 'll', L ? uNum(logLikelihoodHistory[L - 1]) : '–');
      setReadout(
        root,
        'dl',
        L > 1 ? (logLikelihoodHistory[L - 1] - logLikelihoodHistory[L - 2]).toFixed(4) : '–',
      );
      setReadout(
        root,
        'sh',
        responsibilities
          ? responsibilities.filter((r) => Math.max(...r) < 0.9).length + ' of ' + P.length
          : '–',
      );
      setReadout(root, 'st', ph === 'done' ? 'Converged' : it ? 'Running' : 'Not started');
    };
    tp.on = () => draw();
    const load = () => {
        P = gen();
        if (one())
          P.forEach((p) => {
            p.y = 0;
          });
      },
      full = () => {
        for (let i = 0; i < 700 && ph !== 'done'; i++) step();
        mv = 0;
      };
    const gen = uDataBind(
      root,
      'gm',
      settings,
      () => {
        playback.stop();
        load();
        run = 1;
        reset();
        full();
        draw();
      },
      mk,
    );
    load();
    reset();
    full();
    const anim = Anim(() => {
      frameCount++;
      if (mv > 0) {
        mv--;
        draw();
        return;
      }
      if (frameCount % (it < 4 ? 44 : 14)) return;
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#gm-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'gm', {
      done: () => ph === 'done',
      restart: () => {
        run++;
        reset();
      },
      count: () => it,
      what: 'rounds',
    });
    $('#gm-step', root).onclick = () => {
      playback.stop();
      mv = 0;
      step();
      draw();
    };
    $('#gm-reset', root).onclick = () => {
      playback.stop();
      run++;
      reset();
      draw();
    };
    const kb = bind(
      root,
      'gm-K',
      (v) => {
        settings.K = v;
        playback.stop();
        reset();
        full();
        draw();
      },
      (v) => String(v),
    );
    $('#gm-cov', root).onchange = (e) => {
      settings.cov = e.target.value;
      playback.stop();
      reset();
      full();
      draw();
    };
    const lay = () => {
      $('#gm-covl', root).style.display = one() ? 'none' : '';
      $('#gm-tp1', root).closest('label').style.display = one() ? 'none' : '';
      vb.style.display = settings.dim === '3' ? '' : 'none';
      mainCanvas.style.cursor = settings.dim === '3' ? 'grab' : '';
      mainCanvas.style.touchAction = settings.dim === '3' ? 'none' : '';
    };
    segBind(root, 'gm-dim', settings.dim, (v) => {
      const was = one();
      settings.dim = v;
      playback.stop();
      orb.spin(false);
      if (was !== one()) {
        const ks = one() ? ['h2', 'h3', 'hov'] : ['long', 'blobs', 'uneven', 'shop', 'moons'];
        dsel.innerHTML = ks
          .map((k) => '<option value="' + k + '">' + UKIND[k] + '</option>')
          .join('');
        settings.ds = ks[0];
        settings.K = one() ? 2 : 3;
        kb.set(settings.K);
        dsel.dispatchEvent(new Event('change'));
      }
      lay();
      draw();
    });
    lay();
    return {
      draw,
      stop: () => {
        playback.stop();
        orb.spin(false);
      },
      auto: () => {
        reset();
        draw();
        playback.start();
      },
    };
  },
};
