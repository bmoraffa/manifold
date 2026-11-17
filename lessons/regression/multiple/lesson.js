/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  readThemeColors,
  bind,
  clamp,
  f2,
  f3,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  ramp,
  randnFrom,
  rng,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { Orbit, dot, fTestPValue, lsq, formatPValue } from '../shared.js';

export default {
  init(root) {
    const mainCanvas = $('#mr-c', root),
      coefficientCanvas = $('#mr-e', root),
      settings = { b: [0.5, 1.2, -0.8], noise: 0.8, r: 0, n: 60, res: true };
    let seed = 5,
      X = [],
      y = [],
      beta = [0, 0, 0],
      ym = 0,
      fitAnimationProgress = 0,
      wait = 0,
      simulatedSlopePairs = [],
      playback,
      draw;
    const orb = Orbit(mainCanvas, () => draw(), $('.viewbar', root));
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r),
        q = Math.sqrt(1 - settings.r * settings.r);
      X = [];
      y = [];
      for (let i = 0; i < settings.n; i++) {
        const a = clamp(rn(), -2.4, 2.4),
          b = clamp(settings.r * a + q * rn(), -2.4, 2.4);
        X.push([1, a, b]);
        y.push(settings.b[0] + settings.b[1] * a + settings.b[2] * b + settings.noise * rn());
      }
      beta = lsq(X, y, 1e-9);
      ym = y.reduce((s, v) => s + v, 0) / y.length;
      const r2 = rng(77),
        rn2 = randnFrom(r2);
      simulatedSlopePairs = [];
      for (let k = 0; k < 200; k++) {
        const yy = X.map(
          (row) =>
            settings.b[0] +
            settings.b[1] * row[1] +
            settings.b[2] * row[2] +
            settings.noise * rn2(),
        );
        const bb = lsq(X, yy, 1e-9);
        simulatedSlopePairs.push([bb[1], bb[2]]);
      }
    };
    gen();
    const tp = testPoint(
      root,
      'mr',
      ['Your x₁', 'Your x₂'],
      [1, -1],
      [
        [-2.5, 2.5],
        [-2.5, 2.5],
      ],
    );
    tp.on = () => draw();
    draw = () => {
      const colors = readThemeColors(),
        e =
          fitAnimationProgress < 0.5
            ? 2 * fitAnimationProgress * fitAnimationProgress
            : 1 - 2 * (1 - fitAnimationProgress) * (1 - fitAnimationProgress),
        cb = [ym + (beta[0] - ym) * e, beta[1] * e, beta[2] * e],
        ZS = 0.42;
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          pr = orb.cam(w, h, 5.6),
          R = 2.5,
          pl = (a, b) => clamp((cb[0] + cb[1] * a + cb[2] * b) * ZS, -3.2, 3.2),
          path = (ps) => {
            ctx.beginPath();
            ps.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
          };
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        for (let k = -2; k <= 2; k++) {
          path([pr(k * 1.25, -R, -2), pr(k * 1.25, R, -2)]);
          ctx.stroke();
          path([pr(-R, k * 1.25, -2), pr(R, k * 1.25, -2)]);
          ctx.stroke();
        }
        const N = 10,
          Q = [];
        for (let i = 0; i < N; i++)
          for (let j = 0; j < N; j++) {
            const a0 = -R + (2 * R * i) / N,
              a1 = -R + (2 * R * (i + 1)) / N,
              b0 = -R + (2 * R * j) / N,
              b1 = -R + (2 * R * (j + 1)) / N,
              ps = [
                pr(a0, b0, pl(a0, b0)),
                pr(a1, b0, pl(a1, b0)),
                pr(a1, b1, pl(a1, b1)),
                pr(a0, b1, pl(a0, b1)),
              ];
            Q.push({ ps, d: (ps[0][2] + ps[2][2]) / 2, z: (pl(a0, b0) + pl(a1, b1)) / 2 });
          }
        Q.sort((a, b) => b.d - a.d);
        for (const o of Q) {
          path(o.ps);
          ctx.closePath();
          ctx.fillStyle = ramp(colors, clamp(0.5 + o.z / 5, 0, 1), 0.42);
          ctx.fill();
          ctx.strokeStyle = mix(colors.ink, colors.ink, 0, 0.14);
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
        const D = X.map((row, i) => {
          const z = clamp(y[i] * ZS, -3.4, 3.4),
            zp = pl(row[1], row[2]),
            p = pr(row[1], row[2], z);
          return { p, q: pr(row[1], row[2], zp), up: z >= zp };
        });
        D.sort((a, b) => b.p[2] - a.p[2]);
        for (const o of D) {
          if (settings.res) {
            ctx.strokeStyle = mix(colors.ok, colors.ok, 0, 0.7);
            ctx.lineWidth = 1.2;
            path([o.p, o.q]);
            ctx.stroke();
          }
          ctx.fillStyle = o.up ? colors.accent : mix(colors.accent, colors.bg, 0.45);
          ctx.beginPath();
          ctx.arc(o.p[0], o.p[1], 3.8, 0, 7);
          ctx.fill();
        }
        {
          const q0 = pr(tp.v[0], tp.v[1], -2),
            q1 = pr(tp.v[0], tp.v[1], pl(tp.v[0], tp.v[1]));
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 1.6;
          ctx.setLineDash([3, 3]);
          path([q0, q1]);
          ctx.stroke();
          ctx.setLineDash([]);
          markTP(ctx, q1[0], q1[1], colors);
        }
        ctx.font = SANS;
        ctx.fillStyle = colors.ink;
        ctx.textAlign = 'center';
        let a = pr(0, -R * 1.18, -2);
        ctx.fillText('x₁', a[0], a[1]);
        a = pr(-R * 1.18, 0, -2);
        ctx.fillText('x₂', a[0], a[1]);
        a = pr(-R, -R, 2.4);
        ctx.fillText('y', a[0], a[1]);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(coefficientCanvas),
          m = { l: 40, r: 10, t: 10, b: 30 },
          LR = [-3, 3],
          Xf = (v) => m.l + ((v - LR[0]) / 6) * (w - m.l - m.r),
          Yf = (v) => h - m.b - ((v - LR[0]) / 6) * (h - m.t - m.b);
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        for (let v = -3; v <= 3; v++) {
          ctx.beginPath();
          ctx.moveTo(Xf(v), m.t);
          ctx.lineTo(Xf(v), h - m.b);
          ctx.moveTo(m.l, Yf(v));
          ctx.lineTo(w - m.r, Yf(v));
          ctx.stroke();
          ctx.textAlign = 'center';
          ctx.fillText(String(v), Xf(v), h - m.b + 14);
          ctx.textAlign = 'right';
          ctx.fillText(String(v), m.l - 6, Yf(v) + 4);
        }
        ctx.textAlign = 'center';
        ctx.fillText('estimated slope for x₁', w / 2 + m.l / 2, h - 3);
        ctx.save();
        ctx.translate(11, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('estimated slope for x₂', 0, 0);
        ctx.restore();
        ctx.save();
        ctx.beginPath();
        ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
        ctx.clip();
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.55);
        for (const s of simulatedSlopePairs) {
          ctx.beginPath();
          ctx.arc(Xf(s[0]), Yf(s[1]), 2.4, 0, 7);
          ctx.fill();
        }
        const tx = Xf(settings.b[1]),
          ty = Yf(settings.b[2]);
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(tx - 8, ty);
        ctx.lineTo(tx + 8, ty);
        ctx.moveTo(tx, ty - 8);
        ctx.lineTo(tx, ty + 8);
        ctx.stroke();
        ctx.fillStyle = colors.gold;
        ctx.strokeStyle = colors.bg;
        ctx.beginPath();
        ctx.arc(Xf(beta[1]), Yf(beta[2]), 7, 0, 7);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.ink;
        ctx.fillText('+ truth', m.l + 8, m.t + 12);
        ctx.fillStyle = colors.cut;
        ctx.fillText('● this sample', m.l + 8, m.t + 27);
      }
      const n = y.length,
        p = 2;
      let rss = 0,
        tss = 0;
      for (let i = 0; i < n; i++) {
        rss += (y[i] - dot(beta, X[i])) ** 2;
        tss += (y[i] - ym) ** 2;
      }
      const r2 = tss > 1e-12 ? 1 - rss / tss : 0,
        F = rss > 1e-12 ? (tss - rss) / p / (rss / (n - p - 1)) : Infinity;
      let s1 = 0,
        s2 = 0,
        s11 = 0,
        s22 = 0,
        s12 = 0;
      for (const row of X) {
        s1 += row[1];
        s2 += row[2];
        s11 += row[1] * row[1];
        s22 += row[2] * row[2];
        s12 += row[1] * row[2];
      }
      const rr = (s12 - (s1 * s2) / n) / Math.sqrt((s11 - (s1 * s1) / n) * (s22 - (s2 * s2) / n));
      tp.out.textContent =
        'At x₁ = ' +
        f2(tp.v[0]) +
        ', x₂ = ' +
        f2(tp.v[1]) +
        ' the fitted plane predicts y = ' +
        f2(beta[0]) +
        ' + (' +
        f2(beta[1]) +
        ' × ' +
        f2(tp.v[0]) +
        ') + (' +
        f2(beta[2]) +
        ' × ' +
        f2(tp.v[1]) +
        ') = ' +
        f2(dot(beta, [1, tp.v[0], tp.v[1]])) +
        '.';
      setReadout(root, 'b', f2(beta[0]) + ', ' + f2(beta[1]) + ', ' + f2(beta[2]));
      setReadout(root, 'r2', f3(r2));
      setReadout(root, 'ar2', f3(1 - ((1 - r2) * (n - 1)) / (n - p - 1)));
      setReadout(root, 'f', isFinite(F) ? F.toFixed(1) : '∞');
      setReadout(root, 'p', isFinite(F) ? formatPValue(fTestPValue(F, p, n - p - 1)) : '< 0.0001');
      setReadout(root, 'vif', (1 / (1 - rr * rr)).toFixed(2));
    };
    const anim = Anim(() => {
      if (wait > 0) {
        wait--;
        if (!wait) fitAnimationProgress = 0;
        return;
      }
      fitAnimationProgress = Math.min(1, fitAnimationProgress + 0.012);
      if (fitAnimationProgress >= 1) wait = 200;
      draw();
    });
    playback = playBtn($('#mr-play', root), anim);
    $('#mr-play', root).addEventListener('click', () => {
      if (anim.on && fitAnimationProgress >= 1) {
        fitAnimationProgress = 0;
        wait = 0;
      }
    });
    const re = () => {
      gen();
      draw();
    };
    bind(root, 'mr-b1', (v) => {
      settings.b[1] = v;
      re();
    });
    bind(root, 'mr-b2', (v) => {
      settings.b[2] = v;
      re();
    });
    bind(root, 'mr-s', (v) => {
      settings.noise = v;
      re();
    });
    bind(root, 'mr-r', (v) => {
      settings.r = v;
      re();
    });
    bind(
      root,
      'mr-n',
      (v) => {
        settings.n = v;
        re();
      },
      (v) => String(v),
    );
    $('#mr-new', root).onclick = () => {
      seed++;
      fitAnimationProgress = 0;
      wait = 0;
      re();
      playback.start();
    };
    $('#mr-res', root).onchange = (e) => {
      settings.res = e.target.checked;
      draw();
    };
    return {
      draw,
      stop() {
        playback.stop();
        orb.spin(false);
      },
      auto() {
        playback.start();
        orb.spin(true);
      },
    };
  },
};
