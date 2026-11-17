/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  MONO,
  readThemeColors,
  bind,
  clipPlot,
  curve,
  f2,
  f3,
  prepareCanvas,
  grab,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  setReadout,
  sweeper,
  testPoint,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#lr-c', root),
      ce = $('#lr-e', root),
      settings = { x0: 2, h: 0.7, k: 'gauss', m: 'const', noise: 0.4, n: 70 },
      Fq = (x) => Math.sin(x) + 0.5 * Math.sin(2.3 * x) + 0.12 * x,
      K = {
        gauss: (u) => Math.exp((-u * u) / 2),
        epan: (u) => Math.max(0, 1 - u * u),
        box: (u) => (Math.abs(u) <= 1 ? 1 : 0),
      },
      bandwidths = [];
    for (let i = 0; i < 36; i++) bandwidths.push(0.12 * Math.pow(3.5 / 0.12, i / 35));
    let seed = 4,
      pts = [],
      trainMseByBandwidth = [],
      looMseByBandwidth = [],
      playback;
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
          w = K[settings.k]((p.x - x0) / h);
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
      if (settings.m === 'lin') {
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
      for (let i = 0; i < settings.n; i++) {
        const x = r() * 10;
        pts.push({ x, y: Fq(x) + settings.noise * rn() });
      }
    };
    const errs = () => {
      trainMseByBandwidth = [];
      looMseByBandwidth = [];
      for (const h of bandwidths) {
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
        trainMseByBandwidth.push(a / pts.length);
        looMseByBandwidth.push(nb > pts.length * 0.8 ? b / nb : NaN);
      }
    };
    gen();
    errs();
    const draw = () => {
      const colors = readThemeColors(),
        f0 = loc(settings.x0, settings.h, -1);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          YR = [-2.6, 3.6],
          P = plot(ctx, w, h, [0, 10], YR, colors, { yticks: [-2, 0, 2], xlabel: 'x' });
        clipPlot(ctx, P);
        const base = P.Y(YR[0]);
        ctx.beginPath();
        ctx.moveTo(P.X(0), base);
        for (let i = 0; i <= 200; i++) {
          const x = i / 20;
          ctx.lineTo(P.X(x), base - K[settings.k]((x - settings.x0) / settings.h) * (h * 0.16));
        }
        ctx.lineTo(P.X(10), base);
        ctx.closePath();
        ctx.fillStyle = mix(colors.gold, colors.gold, 0, 0.3);
        ctx.fill();
        ctx.setLineDash([6, 5]);
        curve(ctx, P, Fq, 0, 10);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.setLineDash([]);
        const tracer = (a, b, al, lw) => {
          ctx.beginPath();
          let on = false;
          for (let i = 0; i <= 260; i++) {
            const x = a + ((b - a) * i) / 260,
              f = loc(x, settings.h, -1);
            if (!f) {
              on = false;
              continue;
            }
            on ? ctx.lineTo(P.X(x), P.Y(f.a)) : ctx.moveTo(P.X(x), P.Y(f.a));
            on = true;
          }
          ctx.strokeStyle = mix(colors.ok, colors.ok, 0, al);
          ctx.lineWidth = lw;
          ctx.stroke();
        };
        tracer(0, 10, 0.3, 1.6);
        tracer(0, settings.x0, 1, 3);
        for (const p of pts) {
          const wv = K[settings.k]((p.x - settings.x0) / settings.h);
          ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.22 + 0.78 * wv);
          ctx.beginPath();
          ctx.arc(P.X(p.x), P.Y(p.y), 2.6 + 3.6 * wv, 0, 7);
          ctx.fill();
        }
        ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.8);
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(P.X(settings.x0), P.Y(YR[0]));
        ctx.lineTo(P.X(settings.x0), P.Y(YR[1]));
        ctx.stroke();
        if (f0) {
          const hw = settings.k === 'gauss' ? 1.6 * settings.h : settings.h;
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 3.4;
          ctx.beginPath();
          ctx.moveTo(P.X(settings.x0 - hw), P.Y(f0.a - f0.b * hw));
          ctx.lineTo(P.X(settings.x0 + hw), P.Y(f0.a + f0.b * hw));
          ctx.stroke();
          ctx.fillStyle = colors.gold;
          ctx.strokeStyle = colors.bg;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(P.X(settings.x0), P.Y(f0.a), 7, 0, 7);
          ctx.fill();
          ctx.stroke();
        }
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.muted;
        ctx.fillText('bigger dots = more weight, dashed = truth', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(ce),
          LX = [Math.log10(0.12), Math.log10(3.5)];
        let ym = 0;
        for (const v of looMseByBandwidth) if (isFinite(v)) ym = Math.max(ym, v);
        for (const v of trainMseByBandwidth) ym = Math.max(ym, v);
        ym = Math.min(ym * 1.1, 3) || 1;
        const P = plot(ctx, w, h, LX, [0, ym], colors, { xstep: 100, xlabel: 'bandwidth h' });
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        [0.15, 0.3, 0.6, 1.2, 2.4].forEach((v) =>
          ctx.fillText(String(v), P.X(Math.log10(v)), h - P.m.b + 17),
        );
        clipPlot(ctx, P);
        ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.5);
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.moveTo(P.X(Math.log10(settings.h)), P.Y(0));
        ctx.lineTo(P.X(Math.log10(settings.h)), P.Y(ym));
        ctx.stroke();
        [
          [trainMseByBandwidth, colors.accent],
          [looMseByBandwidth, colors.ok],
        ].forEach(([E, col]) => {
          ctx.strokeStyle = col;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          let on = false;
          E.forEach((v, i) => {
            if (!isFinite(v)) {
              on = false;
              return;
            }
            const x = P.X(Math.log10(bandwidths[i])),
              y = P.Y(Math.min(v, ym));
            on ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            on = true;
          });
          ctx.stroke();
        });
        ctx.restore();
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.accent;
        ctx.fillText('train', P.m.l + 8, P.m.t + 8);
        ctx.fillStyle = colors.ok;
        ctx.fillText('leave-one-out', P.m.l + 8, P.m.t + 23);
      }
      let bi = -1;
      looMseByBandwidth.forEach((v, i) => {
        if (isFinite(v) && (bi < 0 || v < looMseByBandwidth[bi])) bi = i;
      });
      let a = 0,
        b = 0,
        nb = 0;
      pts.forEach((p, i) => {
        const f = loc(p.x, settings.h, -1);
        a += (p.y - f.a) ** 2;
        const q = loc(p.x, settings.h, i);
        if (q) {
          b += (p.y - q.a) ** 2;
          nb++;
        }
      });
      if (typeof tp !== 'undefined')
        tp.out.textContent =
          'At x = ' +
          f2(settings.x0) +
          ' the neighbours predict y = ' +
          (f0 ? f2(f0.a) : 'nothing: no training point is close enough') +
          (f0
            ? ', using the equivalent of ' +
              f0.eff.toFixed(1) +
              ' points. The true curve there is ' +
              f2(Fq(settings.x0)) +
              '.'
            : '.');
      setReadout(root, 'y', f0 ? f2(f0.a) : '–');
      setReadout(root, 'eff', f0 ? f0.eff.toFixed(1) : '0');
      setReadout(root, 'tr', f3(a / pts.length));
      setReadout(root, 'cv', nb > pts.length * 0.8 ? f3(b / nb) : 'too few neighbours');
      setReadout(root, 'bh', bi >= 0 ? f2(bandwidths[bi]) : '–');
    };
    const xs = bind(root, 'lr-x', (v) => {
      settings.x0 = v;
      draw();
    });
    const tp = testPoint(root, 'lr', ['Your x (moves x₀)'], [2], [[0, 10]]);
    tp.on = () => {
      playback.stop();
      settings.x0 = tp.v[0];
      xs.set(settings.x0);
      draw();
    };
    playback = playBtn($('#lr-play', root), sweeper(xs.el, 0, 10, 9));
    grab(xs.el, playback);
    const re = (d) => {
      if (d) gen();
      errs();
      draw();
    };
    bind(root, 'lr-h', (v) => {
      settings.h = v;
      draw();
    });
    $('#lr-k', root).onchange = (e) => {
      settings.k = e.target.value;
      re();
    };
    $('#lr-m', root).onchange = (e) => {
      settings.m = e.target.value;
      re();
    };
    bind(root, 'lr-s', (v) => {
      settings.noise = v;
      re(1);
    });
    bind(
      root,
      'lr-n',
      (v) => {
        settings.n = v;
        re(1);
      },
      (v) => String(v),
    );
    $('#lr-new', root).onclick = () => {
      seed++;
      re(1);
    };
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
