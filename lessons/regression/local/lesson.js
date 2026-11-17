/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  MONO,
  T,
  bind,
  clipPlot,
  curve,
  f2,
  f3,
  fit,
  grab,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  setR,
  sweeper,
  testPoint,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#lr-c', root),
      ce = $('#lr-e', root),
      S = { x0: 2, h: 0.7, k: 'gauss', m: 'const', noise: 0.4, n: 70 },
      Fq = (x) => Math.sin(x) + 0.5 * Math.sin(2.3 * x) + 0.12 * x,
      K = {
        gauss: (u) => Math.exp((-u * u) / 2),
        epan: (u) => Math.max(0, 1 - u * u),
        box: (u) => (Math.abs(u) <= 1 ? 1 : 0),
      },
      HS = [];
    for (let i = 0; i < 36; i++) HS.push(0.12 * Math.pow(3.5 / 0.12, i / 35));
    let seed = 4,
      pts = [],
      et = [],
      ec = [],
      pb;
    const loc = (x0, h, skip) => {
      let sw = 0,
        swx = 0,
        swy = 0,
        swxx = 0,
        swxy = 0,
        sw2 = 0;
      for (let i = 0; i < pts.length; i++) {
        if (i === skip) continue;
        const p = pts[i],
          w = K[S.k]((p.x - x0) / h);
        if (!w) continue;
        const d = p.x - x0;
        sw += w;
        swx += w * d;
        swy += w * p.y;
        swxx += w * d * d;
        swxy += w * d * p.y;
        sw2 += w * w;
      }
      if (sw < 1e-9) return null;
      let a = swy / sw,
        b = 0;
      if (S.m === 'lin') {
        const det = sw * swxx - swx * swx;
        if (det > 1e-9 * sw * sw) {
          b = (sw * swxy - swx * swy) / det;
          a = (swy - b * swx) / sw;
        }
      }
      return { a, b, eff: (sw * sw) / sw2 };
    };
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < S.n; i++) {
        const x = r() * 10;
        pts.push({ x, y: Fq(x) + S.noise * rn() });
      }
    };
    const errs = () => {
      et = [];
      ec = [];
      for (const h of HS) {
        let a = 0,
          b = 0,
          nb = 0;
        pts.forEach((p, i) => {
          const f = loc(p.x, h, -1);
          a += (p.y - f.a) ** 2;
          const g = loc(p.x, h, i);
          if (g) {
            b += (p.y - g.a) ** 2;
            nb++;
          }
        });
        et.push(a / pts.length);
        ec.push(nb > pts.length * 0.8 ? b / nb : NaN);
      }
    };
    gen();
    errs();
    const draw = () => {
      const t = T(),
        f0 = loc(S.x0, S.h, -1);
      {
        const { g, w, h } = fit(c),
          YR = [-2.6, 3.6],
          P = plot(g, w, h, [0, 10], YR, t, { yticks: [-2, 0, 2], xlabel: 'x' });
        clipPlot(g, P);
        const base = P.Y(YR[0]);
        g.beginPath();
        g.moveTo(P.X(0), base);
        for (let i = 0; i <= 200; i++) {
          const x = i / 20;
          g.lineTo(P.X(x), base - K[S.k]((x - S.x0) / S.h) * (h * 0.16));
        }
        g.lineTo(P.X(10), base);
        g.closePath();
        g.fillStyle = mix(t.gold, t.gold, 0, 0.3);
        g.fill();
        g.setLineDash([6, 5]);
        curve(g, P, Fq, 0, 10);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.4;
        g.stroke();
        g.setLineDash([]);
        const tracer = (a, b, al, lw) => {
          g.beginPath();
          let on = false;
          for (let i = 0; i <= 260; i++) {
            const x = a + ((b - a) * i) / 260,
              f = loc(x, S.h, -1);
            if (!f) {
              on = false;
              continue;
            }
            on ? g.lineTo(P.X(x), P.Y(f.a)) : g.moveTo(P.X(x), P.Y(f.a));
            on = true;
          }
          g.strokeStyle = mix(t.ok, t.ok, 0, al);
          g.lineWidth = lw;
          g.stroke();
        };
        tracer(0, 10, 0.3, 1.6);
        tracer(0, S.x0, 1, 3);
        for (const p of pts) {
          const wv = K[S.k]((p.x - S.x0) / S.h);
          g.fillStyle = mix(t.accent, t.accent, 0, 0.22 + 0.78 * wv);
          g.beginPath();
          g.arc(P.X(p.x), P.Y(p.y), 2.6 + 3.6 * wv, 0, 7);
          g.fill();
        }
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.8);
        g.lineWidth = 1.3;
        g.beginPath();
        g.moveTo(P.X(S.x0), P.Y(YR[0]));
        g.lineTo(P.X(S.x0), P.Y(YR[1]));
        g.stroke();
        if (f0) {
          const hw = S.k === 'gauss' ? 1.6 * S.h : S.h;
          g.strokeStyle = t.gold;
          g.lineWidth = 3.4;
          g.beginPath();
          g.moveTo(P.X(S.x0 - hw), P.Y(f0.a - f0.b * hw));
          g.lineTo(P.X(S.x0 + hw), P.Y(f0.a + f0.b * hw));
          g.stroke();
          g.fillStyle = t.gold;
          g.strokeStyle = t.bg;
          g.lineWidth = 2;
          g.beginPath();
          g.arc(P.X(S.x0), P.Y(f0.a), 7, 0, 7);
          g.fill();
          g.stroke();
        }
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('bigger dots = more weight, dashed = truth', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g, w, h } = fit(ce),
          LX = [Math.log10(0.12), Math.log10(3.5)];
        let ym = 0;
        for (const v of ec) if (isFinite(v)) ym = Math.max(ym, v);
        for (const v of et) ym = Math.max(ym, v);
        ym = Math.min(ym * 1.1, 3) || 1;
        const P = plot(g, w, h, LX, [0, ym], t, { xstep: 100, xlabel: 'bandwidth h' });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        [0.15, 0.3, 0.6, 1.2, 2.4].forEach((v) =>
          g.fillText(String(v), P.X(Math.log10(v)), h - P.m.b + 17),
        );
        clipPlot(g, P);
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
        g.lineWidth = 8;
        g.beginPath();
        g.moveTo(P.X(Math.log10(S.h)), P.Y(0));
        g.lineTo(P.X(Math.log10(S.h)), P.Y(ym));
        g.stroke();
        [
          [et, t.accent],
          [ec, t.ok],
        ].forEach(([E, col]) => {
          g.strokeStyle = col;
          g.lineWidth = 2.4;
          g.beginPath();
          let on = false;
          E.forEach((v, i) => {
            if (!isFinite(v)) {
              on = false;
              return;
            }
            const x = P.X(Math.log10(HS[i])),
              y = P.Y(Math.min(v, ym));
            on ? g.lineTo(x, y) : g.moveTo(x, y);
            on = true;
          });
          g.stroke();
        });
        g.restore();
        g.textAlign = 'left';
        g.fillStyle = t.accent;
        g.fillText('train', P.m.l + 8, P.m.t + 8);
        g.fillStyle = t.ok;
        g.fillText('leave-one-out', P.m.l + 8, P.m.t + 23);
      }
      let bi = -1;
      ec.forEach((v, i) => {
        if (isFinite(v) && (bi < 0 || v < ec[bi])) bi = i;
      });
      let a = 0,
        b = 0,
        nb = 0;
      pts.forEach((p, i) => {
        const f = loc(p.x, S.h, -1);
        a += (p.y - f.a) ** 2;
        const q = loc(p.x, S.h, i);
        if (q) {
          b += (p.y - q.a) ** 2;
          nb++;
        }
      });
      if (typeof tp !== 'undefined')
        tp.out.textContent =
          'At x = ' +
          f2(S.x0) +
          ' the neighbours predict y = ' +
          (f0 ? f2(f0.a) : 'nothing: no training point is close enough') +
          (f0
            ? ', using the equivalent of ' +
              f0.eff.toFixed(1) +
              ' points. The true curve there is ' +
              f2(Fq(S.x0)) +
              '.'
            : '.');
      setR(root, 'y', f0 ? f2(f0.a) : '–');
      setR(root, 'eff', f0 ? f0.eff.toFixed(1) : '0');
      setR(root, 'tr', f3(a / pts.length));
      setR(root, 'cv', nb > pts.length * 0.8 ? f3(b / nb) : 'too few neighbours');
      setR(root, 'bh', bi >= 0 ? f2(HS[bi]) : '–');
    };
    const xs = bind(root, 'lr-x', (v) => {
      S.x0 = v;
      draw();
    });
    const tp = testPoint(root, 'lr', ['Your x (moves x₀)'], [2], [[0, 10]]);
    tp.on = () => {
      pb.stop();
      S.x0 = tp.v[0];
      xs.set(S.x0);
      draw();
    };
    pb = playBtn($('#lr-play', root), sweeper(xs.el, 0, 10, 9));
    grab(xs.el, pb);
    const re = (d) => {
      if (d) gen();
      errs();
      draw();
    };
    bind(root, 'lr-h', (v) => {
      S.h = v;
      draw();
    });
    $('#lr-k', root).onchange = (e) => {
      S.k = e.target.value;
      re();
    };
    $('#lr-m', root).onchange = (e) => {
      S.m = e.target.value;
      re();
    };
    bind(root, 'lr-s', (v) => {
      S.noise = v;
      re(1);
    });
    bind(
      root,
      'lr-n',
      (v) => {
        S.n = v;
        re(1);
      },
      (v) => String(v),
    );
    $('#lr-new', root).onclick = () => {
      seed++;
      re(1);
    };
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
