/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  readThemeColors,
  bind,
  clamp,
  clipPlot,
  f3,
  prepareCanvas,
  grab,
  lgam,
  mix,
  playBtn,
  plot,
  ramp,
  rng,
  setReadout,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#cd-c', root),
      ce = $('#cd-e', root),
      cv = $('#cd-v', root),
      DS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
      S = { i: 3 },
      distanceHistograms = {},
      meanNearFarRatioByDimension = {};
    let playback,
      rot = 0.6;
    const PT = (() => {
      const r = rng(42),
        a = [];
      for (let i = 0; i < 100; i++) a.push([r(), r(), r()]);
      return a;
    })();
    const drawTop = () => {
      const colors = readThemeColors(),
        { g: ctx, w, h } = prepareCanvas(cv),
        pw = w / 4,
        sz = Math.min(pw * 0.78, h - 64),
        top = 30,
        d = DS[S.i];
      ctx.font = SANS;
      ctx.textAlign = 'center';
      const head = (k, a, b) => {
          const cx = pw * k + pw / 2;
          ctx.fillStyle = colors.ink;
          ctx.font = SANS;
          ctx.fillText(a, cx, 16);
          ctx.font = MONO;
          ctx.fillStyle = colors.cut;
          ctx.fillText(b, cx, top + sz + 22);
          return cx;
        },
        inb = (p, n, s) => {
          for (let q = 0; q < n; q++) if (Math.abs(p[q] - 0.5) > s / 2) return false;
          return true;
        },
        nn = (n) => {
          let b = 1 / 0;
          for (const p of PT) {
            let q = 0;
            for (let j = 0; j < n; j++) q += (p[j] - 0.5) ** 2;
            b = Math.min(b, Math.sqrt(q));
          }
          return b;
        };
      {
        const s1 = 0.1,
          cx = head(
            0,
            '1 dimension',
            'box side 0.10 · ' + PT.filter((p) => inb(p, 1, s1)).length + ' points inside',
          ),
          x0 = cx - sz / 2,
          y = top + sz / 2;
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x0, y);
        ctx.lineTo(x0 + sz, y);
        ctx.stroke();
        ctx.fillStyle = mix(colors.gold, colors.gold, 0, 0.35);
        ctx.fillRect(x0 + sz * (0.5 - s1 / 2), y - 12, sz * s1, 24);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2;
        ctx.strokeRect(x0 + sz * (0.5 - s1 / 2), y - 12, sz * s1, 24);
        for (const p of PT) {
          ctx.fillStyle = inb(p, 1, s1) ? colors.cut : mix(colors.accent, colors.accent, 0, 0.75);
          ctx.beginPath();
          ctx.arc(x0 + p[0] * sz, y, 3, 0, 7);
          ctx.fill();
        }
        ctx.fillStyle = colors.muted;
        ctx.fillText('nearest point to the centre: ' + nn(1).toFixed(3), cx, top + sz + 38);
      }
      {
        const s2 = Math.sqrt(0.1),
          cx = head(
            1,
            '2 dimensions',
            'box side 0.32 · ' + PT.filter((p) => inb(p, 2, s2)).length + ' points inside',
          ),
          x0 = cx - sz / 2;
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x0, top, sz, sz);
        ctx.fillStyle = mix(colors.gold, colors.gold, 0, 0.3);
        ctx.fillRect(x0 + sz * (0.5 - s2 / 2), top + sz * (0.5 - s2 / 2), sz * s2, sz * s2);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2;
        ctx.strokeRect(x0 + sz * (0.5 - s2 / 2), top + sz * (0.5 - s2 / 2), sz * s2, sz * s2);
        for (const p of PT) {
          ctx.fillStyle = inb(p, 2, s2) ? colors.cut : mix(colors.accent, colors.accent, 0, 0.75);
          ctx.beginPath();
          ctx.arc(x0 + p[0] * sz, top + (1 - p[1]) * sz, 3, 0, 7);
          ctx.fill();
        }
        ctx.fillStyle = colors.muted;
        ctx.fillText('nearest point to the centre: ' + nn(2).toFixed(3), cx, top + sz + 38);
      }
      {
        const s3 = Math.cbrt(0.1),
          cx = head(
            2,
            '3 dimensions',
            'box side 0.46 · ' + PT.filter((p) => inb(p, 3, s3)).length + ' points inside',
          ),
          cy = top + sz / 2,
          ca = Math.cos(rot),
          sa = Math.sin(rot),
          sc = sz * 0.56,
          pr = (x, y, z) => {
            x -= 0.5;
            y -= 0.5;
            z -= 0.5;
            const xr = x * ca - y * sa,
              yr = x * sa + y * ca;
            return [cx + sc * xr, cy - sc * (z * 0.92 + yr * 0.38), yr];
          };
        const box = (s, col, lw) => {
          const a = 0.5 - s / 2,
            b = 0.5 + s / 2,
            V = [
              [a, a, a],
              [b, a, a],
              [b, b, a],
              [a, b, a],
              [a, a, b],
              [b, a, b],
              [b, b, b],
              [a, b, b],
            ].map((v) => pr(v[0], v[1], v[2]));
          ctx.strokeStyle = col;
          ctx.lineWidth = lw;
          ctx.beginPath();
          [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
          ].forEach(([i, j]) => {
            ctx.moveTo(V[i][0], V[i][1]);
            ctx.lineTo(V[j][0], V[j][1]);
          });
          ctx.stroke();
          return V;
        };
        box(1, colors.muted, 1.2);
        const V = box(s3, colors.gold, 2);
        ctx.fillStyle = mix(colors.gold, colors.gold, 0, 0.16);
        [
          [0, 1, 2, 3],
          [4, 5, 6, 7],
          [0, 1, 5, 4],
          [2, 3, 7, 6],
        ].forEach((F) => {
          ctx.beginPath();
          F.forEach((i, k) => (k ? ctx.lineTo(V[i][0], V[i][1]) : ctx.moveTo(V[i][0], V[i][1])));
          ctx.closePath();
          ctx.fill();
        });
        PT.map((p) => ({ q: pr(p[0], p[1], p[2]), in: inb(p, 3, s3) }))
          .sort((a, b) => b.q[2] - a.q[2])
          .forEach((o) => {
            ctx.fillStyle = o.in ? colors.cut : mix(colors.accent, colors.accent, 0, 0.75);
            ctx.beginPath();
            ctx.arc(o.q[0], o.q[1], 3, 0, 7);
            ctx.fill();
          });
        ctx.fillStyle = colors.muted;
        ctx.fillText('nearest point to the centre: ' + nn(3).toFixed(3), cx, top + sz + 38);
      }
      {
        const cx = head(
            3,
            'Any number of dimensions',
            'box side ' + Math.pow(0.1, 1 / d).toFixed(3) + ' at ' + d + ' dimensions',
          ),
          x0 = cx - sz / 2,
          rh = sz / DS.length;
        ctx.textAlign = 'left';
        DS.forEach((q, k) => {
          const sd = Math.pow(0.1, 1 / q),
            y = top + k * rh,
            on = q === d,
            bx = x0 + 44,
            bwid = sz - 44 - 38;
          ctx.font = on ? SANS : MONO;
          ctx.fillStyle = on ? colors.ink : colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText(String(q), x0 + 36, y + rh / 2 + 4);
          ctx.fillStyle = mix(colors.muted, colors.muted, 0, 0.14);
          ctx.fillRect(bx, y + rh * 0.22, bwid, rh * 0.56);
          ctx.fillStyle = on ? colors.gold : mix(colors.gold, colors.gold, 0, 0.4);
          ctx.fillRect(bx, y + rh * 0.22, bwid * sd, rh * 0.56);
          ctx.textAlign = 'left';
          ctx.fillStyle = on ? colors.ink : colors.muted;
          ctx.fillText(sd.toFixed(2), bx + bwid + 6, y + rh / 2 + 4);
        });
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        ctx.fillText('gold = box width, grey = full range', cx, top + sz + 38);
      }
    };
    const sim = (d) => {
      if (distanceHistograms[d]) return distanceHistograms[d];
      const r = rng(100 + d),
        N = 200,
        P = [];
      for (let i = 0; i < N; i++) {
        const p = new Float32Array(d);
        for (let j = 0; j < d; j++) p[j] = r();
        P.push(p);
      }
      const ds = [];
      let nf = 0;
      for (let i = 0; i < N; i++) {
        let mn = 1 / 0,
          mx = 0;
        for (let j = 0; j < N; j++) {
          if (i === j) continue;
          let s = 0;
          const a = P[i],
            b = P[j];
          for (let q = 0; q < d; q++) {
            const u = a[q] - b[q];
            s += u * u;
          }
          s = Math.sqrt(s);
          if (s < mn) mn = s;
          if (s > mx) mx = s;
          if (j > i && (i * 7 + j) % 5 === 0) ds.push(s);
        }
        nf += mn / mx;
      }
      const mean = ds.reduce((a, b) => a + b, 0) / ds.length,
        bins = new Array(50).fill(0);
      for (const v of ds) {
        const b = Math.floor((v / mean / 2) * 50);
        if (b >= 0 && b < 50) bins[b]++;
      }
      meanNearFarRatioByDimension[d] = nf / N;
      return (distanceHistograms[d] = { bins, n: ds.length });
    };
    DS.forEach(sim);
    const inscribedBallFraction = (d) =>
        Math.exp((d / 2) * Math.log(Math.PI) - lgam(d / 2 + 1) - d * Math.log(2)),
      shell = (d) => 1 - Math.pow(0.9, d);
    const draw = () => {
      const colors = readThemeColors(),
        d = DS[S.i],
        Hd = sim(d);
      drawTop();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas);
        let mx = 0;
        for (const b of Hd.bins) mx = Math.max(mx, b / Hd.n / 0.04);
        const P = plot(ctx, w, h, [0, 2], [0, Math.max(4, mx * 1.1)], colors, {
          xstep: 0.5,
          xlabel: 'distance ÷ average distance',
        });
        clipPlot(ctx, P);
        Hd.bins.forEach((b, i) => {
          if (!b) return;
          const v = b / Hd.n / 0.04;
          ctx.fillStyle = ramp(colors, clamp(0.25 + (0.75 * v) / mx, 0, 1));
          ctx.fillRect(P.X(i * 0.04) + 0.5, P.Y(v), P.X(0.04) - P.X(0) - 1, P.Y(0) - P.Y(v));
        });
        ctx.restore();
        ctx.font = SANS;
        ctx.fillStyle = colors.ink;
        ctx.textAlign = 'left';
        ctx.fillText(d + (d === 1 ? ' dimension' : ' dimensions'), P.m.l + 10, P.m.t + 14);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(ce),
          P = plot(ctx, w, h, [0, 3], [0, 1.05], colors, {
            yticks: [0.25, 0.5, 0.75, 1],
            xstep: 100,
            xlabel: 'dimensions',
          });
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        [1, 10, 100, 1000].forEach((v) =>
          ctx.fillText(String(v), P.X(Math.log10(v)), h - P.m.b + 17),
        );
        clipPlot(ctx, P);
        ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.5);
        ctx.lineWidth = 9;
        ctx.beginPath();
        ctx.moveTo(P.X(Math.log10(d)), P.Y(0));
        ctx.lineTo(P.X(Math.log10(d)), P.Y(1.05));
        ctx.stroke();
        [
          [(x) => inscribedBallFraction(x), colors.ok],
          [(x) => shell(x), colors.accent],
        ].forEach(([fn, cl]) => {
          ctx.strokeStyle = cl;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          for (let i = 0; i <= 150; i++) {
            const lx = (3 * i) / 150,
              x = Math.pow(10, lx);
            i ? ctx.lineTo(P.X(lx), P.Y(fn(x))) : ctx.moveTo(P.X(lx), P.Y(fn(x)));
          }
          ctx.stroke();
        });
        ctx.strokeStyle = colors.gold;
        ctx.fillStyle = colors.gold;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        DS.forEach((q, i) =>
          i
            ? ctx.lineTo(P.X(Math.log10(q)), P.Y(meanNearFarRatioByDimension[q]))
            : ctx.moveTo(P.X(Math.log10(q)), P.Y(meanNearFarRatioByDimension[q])),
        );
        ctx.stroke();
        DS.forEach((q) => {
          ctx.beginPath();
          ctx.arc(P.X(Math.log10(q)), P.Y(meanNearFarRatioByDimension[q]), 3, 0, 7);
          ctx.fill();
        });
        ctx.restore();
        ctx.textAlign = 'left';
        let ly = P.m.t + 34;
        [
          [colors.cut, 'nearest ÷ farthest neighbour'],
          [colors.ok, 'share of cube inside its ball'],
          [colors.accent, 'share of cube near its surface'],
        ].forEach(([cl, sx]) => {
          ctx.fillStyle = cl;
          ctx.fillText(sx, P.X(1.2), ly);
          ly += 14;
        });
      }
      setReadout(root, 'd', String(d));
      setReadout(root, 'nf', f3(meanNearFarRatioByDimension[d]));
      const bs = inscribedBallFraction(d);
      setReadout(
        root,
        'ball',
        bs < 1e-300
          ? 'effectively zero'
          : bs < 1e-4
            ? bs.toExponential(1)
            : (100 * bs).toFixed(1) + '%',
      );
      setReadout(root, 'shell', (100 * shell(d)).toFixed(1) + '%');
      setReadout(root, 'need', d <= 6 ? Math.pow(10, d).toLocaleString() : '10^' + d);
    };
    const ds = bind(
      root,
      'cd-d',
      (v) => {
        S.i = v;
        draw();
      },
      (v) => String(DS[v]),
    );
    let frameCount = 0,
      dir = 1;
    const anim = Anim(() => {
      rot += 0.008;
      if (++frameCount % 80) {
        drawTop();
        return;
      }
      let v = S.i + dir;
      if (v > 10) {
        dir = -1;
        v = 9;
      }
      if (v < 0) {
        dir = 1;
        v = 1;
      }
      S.i = v;
      ds.set(v);
      draw();
    });
    playback = playBtn($('#cd-play', root), anim);
    grab(ds.el, playback);
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
