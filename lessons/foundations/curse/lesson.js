/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  T,
  bind,
  clamp,
  clipPlot,
  f3,
  fit,
  grab,
  lgam,
  mix,
  playBtn,
  plot,
  ramp,
  rng,
  setR,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#cd-c', root),
      ce = $('#cd-e', root),
      cv = $('#cd-v', root),
      DS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
      S = { i: 3 },
      H = {},
      NFD = {};
    let pb,
      rot = 0.6;
    const PT = (() => {
      const r = rng(42),
        a = [];
      for (let i = 0; i < 100; i++) a.push([r(), r(), r()]);
      return a;
    })();
    const drawTop = () => {
      const t = T(),
        { g, w, h } = fit(cv),
        pw = w / 4,
        sz = Math.min(pw * 0.78, h - 64),
        top = 30,
        d = DS[S.i];
      g.font = SANS;
      g.textAlign = 'center';
      const head = (k, a, b) => {
          const cx = pw * k + pw / 2;
          g.fillStyle = t.ink;
          g.font = SANS;
          g.fillText(a, cx, 16);
          g.font = MONO;
          g.fillStyle = t.cut;
          g.fillText(b, cx, top + sz + 22);
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
        g.strokeStyle = t.muted;
        g.lineWidth = 1.5;
        g.beginPath();
        g.moveTo(x0, y);
        g.lineTo(x0 + sz, y);
        g.stroke();
        g.fillStyle = mix(t.gold, t.gold, 0, 0.35);
        g.fillRect(x0 + sz * (0.5 - s1 / 2), y - 12, sz * s1, 24);
        g.strokeStyle = t.gold;
        g.lineWidth = 2;
        g.strokeRect(x0 + sz * (0.5 - s1 / 2), y - 12, sz * s1, 24);
        for (const p of PT) {
          g.fillStyle = inb(p, 1, s1) ? t.cut : mix(t.accent, t.accent, 0, 0.75);
          g.beginPath();
          g.arc(x0 + p[0] * sz, y, 3, 0, 7);
          g.fill();
        }
        g.fillStyle = t.muted;
        g.fillText('nearest point to the centre: ' + nn(1).toFixed(3), cx, top + sz + 38);
      }
      {
        const s2 = Math.sqrt(0.1),
          cx = head(
            1,
            '2 dimensions',
            'box side 0.32 · ' + PT.filter((p) => inb(p, 2, s2)).length + ' points inside',
          ),
          x0 = cx - sz / 2;
        g.strokeStyle = t.muted;
        g.lineWidth = 1.5;
        g.strokeRect(x0, top, sz, sz);
        g.fillStyle = mix(t.gold, t.gold, 0, 0.3);
        g.fillRect(x0 + sz * (0.5 - s2 / 2), top + sz * (0.5 - s2 / 2), sz * s2, sz * s2);
        g.strokeStyle = t.gold;
        g.lineWidth = 2;
        g.strokeRect(x0 + sz * (0.5 - s2 / 2), top + sz * (0.5 - s2 / 2), sz * s2, sz * s2);
        for (const p of PT) {
          g.fillStyle = inb(p, 2, s2) ? t.cut : mix(t.accent, t.accent, 0, 0.75);
          g.beginPath();
          g.arc(x0 + p[0] * sz, top + (1 - p[1]) * sz, 3, 0, 7);
          g.fill();
        }
        g.fillStyle = t.muted;
        g.fillText('nearest point to the centre: ' + nn(2).toFixed(3), cx, top + sz + 38);
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
          g.strokeStyle = col;
          g.lineWidth = lw;
          g.beginPath();
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
            g.moveTo(V[i][0], V[i][1]);
            g.lineTo(V[j][0], V[j][1]);
          });
          g.stroke();
          return V;
        };
        box(1, t.muted, 1.2);
        const V = box(s3, t.gold, 2);
        g.fillStyle = mix(t.gold, t.gold, 0, 0.16);
        [
          [0, 1, 2, 3],
          [4, 5, 6, 7],
          [0, 1, 5, 4],
          [2, 3, 7, 6],
        ].forEach((F) => {
          g.beginPath();
          F.forEach((i, k) => (k ? g.lineTo(V[i][0], V[i][1]) : g.moveTo(V[i][0], V[i][1])));
          g.closePath();
          g.fill();
        });
        PT.map((p) => ({ q: pr(p[0], p[1], p[2]), in: inb(p, 3, s3) }))
          .sort((a, b) => b.q[2] - a.q[2])
          .forEach((o) => {
            g.fillStyle = o.in ? t.cut : mix(t.accent, t.accent, 0, 0.75);
            g.beginPath();
            g.arc(o.q[0], o.q[1], 3, 0, 7);
            g.fill();
          });
        g.fillStyle = t.muted;
        g.fillText('nearest point to the centre: ' + nn(3).toFixed(3), cx, top + sz + 38);
      }
      {
        const cx = head(
            3,
            'Any number of dimensions',
            'box side ' + Math.pow(0.1, 1 / d).toFixed(3) + ' at ' + d + ' dimensions',
          ),
          x0 = cx - sz / 2,
          rh = sz / DS.length;
        g.textAlign = 'left';
        DS.forEach((q, k) => {
          const sd = Math.pow(0.1, 1 / q),
            y = top + k * rh,
            on = q === d,
            bx = x0 + 44,
            bwid = sz - 44 - 38;
          g.font = on ? SANS : MONO;
          g.fillStyle = on ? t.ink : t.muted;
          g.textAlign = 'right';
          g.fillText(String(q), x0 + 36, y + rh / 2 + 4);
          g.fillStyle = mix(t.muted, t.muted, 0, 0.14);
          g.fillRect(bx, y + rh * 0.22, bwid, rh * 0.56);
          g.fillStyle = on ? t.gold : mix(t.gold, t.gold, 0, 0.4);
          g.fillRect(bx, y + rh * 0.22, bwid * sd, rh * 0.56);
          g.textAlign = 'left';
          g.fillStyle = on ? t.ink : t.muted;
          g.fillText(sd.toFixed(2), bx + bwid + 6, y + rh / 2 + 4);
        });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('gold = box width, grey = full range', cx, top + sz + 38);
      }
    };
    const sim = (d) => {
      if (H[d]) return H[d];
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
      NFD[d] = nf / N;
      return (H[d] = { bins, n: ds.length });
    };
    DS.forEach(sim);
    const ballS = (d) => Math.exp((d / 2) * Math.log(Math.PI) - lgam(d / 2 + 1) - d * Math.log(2)),
      shell = (d) => 1 - Math.pow(0.9, d);
    const draw = () => {
      const t = T(),
        d = DS[S.i],
        Hd = sim(d);
      drawTop();
      {
        const { g, w, h } = fit(c);
        let mx = 0;
        for (const b of Hd.bins) mx = Math.max(mx, b / Hd.n / 0.04);
        const P = plot(g, w, h, [0, 2], [0, Math.max(4, mx * 1.1)], t, {
          xstep: 0.5,
          xlabel: 'distance ÷ average distance',
        });
        clipPlot(g, P);
        Hd.bins.forEach((b, i) => {
          if (!b) return;
          const v = b / Hd.n / 0.04;
          g.fillStyle = ramp(t, clamp(0.25 + (0.75 * v) / mx, 0, 1));
          g.fillRect(P.X(i * 0.04) + 0.5, P.Y(v), P.X(0.04) - P.X(0) - 1, P.Y(0) - P.Y(v));
        });
        g.restore();
        g.font = SANS;
        g.fillStyle = t.ink;
        g.textAlign = 'left';
        g.fillText(d + (d === 1 ? ' dimension' : ' dimensions'), P.m.l + 10, P.m.t + 14);
      }
      {
        const { g, w, h } = fit(ce),
          P = plot(g, w, h, [0, 3], [0, 1.05], t, {
            yticks: [0.25, 0.5, 0.75, 1],
            xstep: 100,
            xlabel: 'dimensions',
          });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        [1, 10, 100, 1000].forEach((v) =>
          g.fillText(String(v), P.X(Math.log10(v)), h - P.m.b + 17),
        );
        clipPlot(g, P);
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
        g.lineWidth = 9;
        g.beginPath();
        g.moveTo(P.X(Math.log10(d)), P.Y(0));
        g.lineTo(P.X(Math.log10(d)), P.Y(1.05));
        g.stroke();
        [
          [(x) => ballS(x), t.ok],
          [(x) => shell(x), t.accent],
        ].forEach(([fn, cl]) => {
          g.strokeStyle = cl;
          g.lineWidth = 2.4;
          g.beginPath();
          for (let i = 0; i <= 150; i++) {
            const lx = (3 * i) / 150,
              x = Math.pow(10, lx);
            i ? g.lineTo(P.X(lx), P.Y(fn(x))) : g.moveTo(P.X(lx), P.Y(fn(x)));
          }
          g.stroke();
        });
        g.strokeStyle = t.gold;
        g.fillStyle = t.gold;
        g.lineWidth = 2.4;
        g.beginPath();
        DS.forEach((q, i) =>
          i ? g.lineTo(P.X(Math.log10(q)), P.Y(NFD[q])) : g.moveTo(P.X(Math.log10(q)), P.Y(NFD[q])),
        );
        g.stroke();
        DS.forEach((q) => {
          g.beginPath();
          g.arc(P.X(Math.log10(q)), P.Y(NFD[q]), 3, 0, 7);
          g.fill();
        });
        g.restore();
        g.textAlign = 'left';
        let ly = P.m.t + 34;
        [
          [t.cut, 'nearest ÷ farthest neighbour'],
          [t.ok, 'share of cube inside its ball'],
          [t.accent, 'share of cube near its surface'],
        ].forEach(([cl, sx]) => {
          g.fillStyle = cl;
          g.fillText(sx, P.X(1.2), ly);
          ly += 14;
        });
      }
      setR(root, 'd', String(d));
      setR(root, 'nf', f3(NFD[d]));
      const bs = ballS(d);
      setR(
        root,
        'ball',
        bs < 1e-300
          ? 'effectively zero'
          : bs < 1e-4
            ? bs.toExponential(1)
            : (100 * bs).toFixed(1) + '%',
      );
      setR(root, 'shell', (100 * shell(d)).toFixed(1) + '%');
      setR(root, 'need', d <= 6 ? Math.pow(10, d).toLocaleString() : '10^' + d);
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
    let fr = 0,
      dir = 1;
    const anim = Anim(() => {
      rot += 0.008;
      if (++fr % 80) {
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
    pb = playBtn($('#cd-play', root), anim);
    grab(ds.el, pb);
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
