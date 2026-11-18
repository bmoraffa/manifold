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
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import { highlightLessonStep, uDataBind, uFrame, uNum, uView } from '../shared.js';

export default {
  init(root) {
    const c = $('#pc-c', root),
      cr = $('#pc-r', root),
      live = $('#pc-live', root),
      settings = { ds: 'corr', n: 160, seed: 1, rho: 0.8 };
    let P = [],
      mu = { x: 0, y: 0 },
      Q = [],
      cv = { a: 1, b: 0, d: 1 },
      e1 = { l1: 1, l2: 1, th: 0 },
      v = null,
      ph = 'raw',
      centeringProgress = 0,
      projectionProgress = 0,
      powerIterations = 0,
      msg = '',
      activeStepIndex = -1,
      wait = 0,
      frameCount = 0,
      playback,
      twId = null,
      man = false;
    const mk = () => {
      const r = rng(settings.seed * 7 + 3),
        rn = randnFrom(r),
        o = [];
      for (let i = 0; i < settings.n; i++) {
        const u = rn(),
          w = rn();
        let x, y;
        if (settings.ds === 'corr') {
          x = 1.05 * u;
          y = 0.8 * (settings.rho * u + Math.sqrt(1 - settings.rho * settings.rho) * w);
        } else if (settings.ds === 'banana') {
          x = 1.05 * u;
          y = 0.42 * x * x - 0.7 + 0.2 * w;
        } else if (settings.ds === 'twoc') {
          const k = i % 2 ? 1 : -1;
          x = k * 1.3 + 0.42 * u;
          y = k * 0.75 + 0.42 * w;
        } else {
          x = 0.85 * u;
          y = 0.85 * w;
        }
        o.push({ x: clamp(x + 0.8, -3.8, 3.8), y: clamp(y + 0.55, -2.85, 2.85) });
      }
      return o;
    };
    const comp = () => {
      const n = P.length || 1;
      mu = { x: P.reduce((s, p) => s + p.x, 0) / n, y: P.reduce((s, p) => s + p.y, 0) / n };
      Q = P.map((p) => ({ x: p.x - mu.x, y: p.y - mu.y }));
      let a = 0,
        b = 0,
        d = 0;
      for (const q of Q) {
        a += q.x * q.x;
        b += q.x * q.y;
        d += q.y * q.y;
      }
      const m = Math.max(1, n - 1);
      cv = { a: a / m, b: b / m, d: d / m };
      const mm = (cv.a + cv.d) / 2,
        rr = Math.hypot((cv.a - cv.d) / 2, cv.b);
      let th = 0.5 * Math.atan2(2 * cv.b, cv.a - cv.d);
      if (th < 0) th += Math.PI;
      e1 = { l1: mm + rr, l2: Math.max(0, mm - rr), th };
    };
    const va = (th) => {
        const cc = Math.cos(th),
          ss = Math.sin(th);
        return cv.a * cc * cc + 2 * cv.b * cc * ss + cv.d * ss * ss;
      },
      ang = () => {
        let a = Math.atan2(v.y, v.x);
        if (a < 0) a += Math.PI;
        if (a >= Math.PI) a -= Math.PI;
        return a;
      },
      deg = (a) => ((a * 180) / Math.PI).toFixed(0) + '°';
    const tween = (set, to) => {
      cancelAnimationFrame(twId);
      if (prefersReducedMotion) {
        set(to);
        return;
      }
      let k = 0;
      const f = () => {
        k++;
        set(to * (k / 22));
        draw();
        if (k < 22) twId = requestAnimationFrame(f);
      };
      twId = requestAnimationFrame(f);
    };
    const reset = () => {
      cancelAnimationFrame(twId);
      v = null;
      ph = 'raw';
      centeringProgress = 0;
      projectionProgress = 0;
      powerIterations = 0;
      activeStepIndex = -1;
      wait = 0;
      man = false;
      msg =
        'The raw data: ' +
        P.length +
        ' points with mean (' +
        f2(mu.x) +
        ', ' +
        f2(mu.y) +
        '), marked by the cross. Press Run or One step.';
    };
    const step = () => {
      if (P.length < 3) return;
      man = false;
      if (ph === 'raw') {
        ph = 'centre';
        activeStepIndex = 0;
        tween((u) => {
          centeringProgress = u;
        }, 1);
        msg =
          'Centre: the mean (' +
          f2(mu.x) +
          ', ' +
          f2(mu.y) +
          ') was subtracted from every point, so the cloud now sits on the origin. PCA describes spread around the mean, not where the mean is.';
        return;
      }
      if (ph === 'centre') {
        centeringProgress = 1;
        ph = 'cov';
        activeStepIndex = 1;
        msg =
          'Covariance matrix: the variance of x is ' +
          f2(cv.a) +
          ', the variance of y is ' +
          f2(cv.d) +
          ', and their covariance is ' +
          f2(cv.b) +
          '. The total spread is ' +
          f2(cv.a) +
          ' + ' +
          f2(cv.d) +
          ' = ' +
          f2(cv.a + cv.d) +
          '. The ellipse is drawn from these three numbers.';
        return;
      }
      if (ph === 'cov') {
        ph = 'power';
        activeStepIndex = 2;
        powerIterations = 0;
        const a = e1.th + 1.4;
        v = { x: Math.cos(a), y: Math.sin(a) };
        msg =
          'Start with an arbitrary direction, at ' +
          deg(ang()) +
          '. The spread of the shadows along it is ' +
          f2(va(ang())) +
          ', which is ' +
          ((100 * va(ang())) / (cv.a + cv.d)).toFixed(0) +
          '% of the total. Now improve it.';
        return;
      }
      if (ph === 'power') {
        const o = ang(),
          wx = cv.a * v.x + cv.b * v.y,
          wy = cv.b * v.x + cv.d * v.y,
          nn = Math.hypot(wx, wy) || 1;
        v = { x: wx / nn, y: wy / nn };
        powerIterations++;
        let dl = Math.abs(ang() - o);
        if (dl > Math.PI / 2) dl = Math.PI - dl;
        msg =
          'Power iteration ' +
          powerIterations +
          ': the direction was multiplied by the covariance matrix and rescaled to length 1. It turned by ' +
          ((dl * 180) / Math.PI).toFixed(1) +
          '° toward the long axis of the cloud, to ' +
          deg(ang()) +
          '. Spread along it: ' +
          f2(va(ang())) +
          ' (' +
          ((100 * va(ang())) / (cv.a + cv.d)).toFixed(0) +
          '% of the total).';
        if (dl < 0.004 || powerIterations >= 40) {
          ph = 'pc1';
          msg +=
            ' It has stopped turning. This is the first principal component, and the spread along it is the largest eigenvalue, λ₁ = ' +
            f2(e1.l1) +
            '.';
        }
        return;
      }
      if (ph === 'pc1') {
        ph = 'pc2';
        activeStepIndex = 3;
        msg =
          'The second component is at right angles to the first. In two dimensions there is only one such direction. The spread along it is λ₂ = ' +
          f2(e1.l2) +
          ', the smallest of any direction, and λ₁ + λ₂ = ' +
          f2(e1.l1 + e1.l2) +
          ' is the total.';
        return;
      }
      if (ph === 'pc2') {
        ph = 'proj';
        activeStepIndex = 4;
        tween((u) => {
          projectionProgress = u;
        }, 1);
        msg =
          'Project: every point slid along its dashed line onto the first component and now needs one number instead of two. That keeps ' +
          ((100 * e1.l1) / (e1.l1 + e1.l2)).toFixed(1) +
          '% of the spread. What was thrown away is the mean squared distance the points moved: λ₂ = ' +
          f2(e1.l2) +
          '.';
        return;
      }
      reset();
    };
    const tp = testPoint(
      root,
      'pc',
      ['Your x', 'Your y'],
      [2, 0.4],
      [
        [-3.8, 3.8],
        [-2.85, 2.85],
      ],
    );
    const draw = () => {
      const colors = readThemeColors(),
        tot = cv.a + cv.d || 1,
        hasV = !!v,
        know = man || ['pc1', 'pc2', 'proj'].includes(ph);
      {
        const V = uView(c),
          { g, w, h, X, Y } = V;
        uFrame(
          V,
          colors,
          settings.ds === 'corr' ? 'height (standardised)' : 'x',
          settings.ds === 'corr' ? 'arm span (standardised)' : 'y',
        );
        const ox = mu.x * (1 - centeringProgress),
          oy = mu.y * (1 - centeringProgress);
        if (ph !== 'raw' && ph !== 'centre') {
          ellPath(g, X, Y, ox, oy, cv.a, cv.b, cv.d, 2);
          g.strokeStyle = colors.muted;
          g.lineWidth = 1.2;
          g.setLineDash([3, 4]);
          g.stroke();
          g.setLineDash([]);
        }
        if (hasV) {
          const L = 6;
          g.strokeStyle = know && !man ? colors.gold : colors.ink;
          g.lineWidth = 2.6;
          g.beginPath();
          g.moveTo(X(ox - L * v.x), Y(oy - L * v.y));
          g.lineTo(X(ox + L * v.x), Y(oy + L * v.y));
          g.stroke();
          if (ph === 'pc2' || ph === 'proj') {
            g.strokeStyle = colors.muted;
            g.lineWidth = 1.6;
            g.beginPath();
            g.moveTo(X(ox + 1.6 * v.y), Y(oy - 1.6 * v.x));
            g.lineTo(X(ox - 1.6 * v.y), Y(oy + 1.6 * v.x));
            g.stroke();
            g.font = SANS;
            g.fillStyle = colors.muted;
            g.textAlign = 'left';
            g.fillText('second component', X(ox - 1.6 * v.y) + 8, Y(oy + 1.6 * v.x));
          }
          if (know && !man) {
            g.font = SANS;
            g.fillStyle = colors.cut;
            g.textAlign = 'left';
            const s = v.x < 0 ? -1 : 1;
            g.fillText(
              'first component',
              clamp(X(ox + s * 2.7 * v.x) + 8, 10, w - 120),
              clamp(Y(oy + s * 2.7 * v.y) - 8, 16, h - 10),
            );
          }
          g.strokeStyle = mix(colors.ok, colors.ok, 0, 0.4);
          g.lineWidth = 1;
          g.beginPath();
          for (const q of Q) {
            const z = q.x * v.x + q.y * v.y;
            g.moveTo(X(ox + q.x), Y(oy + q.y));
            g.lineTo(X(ox + z * v.x), Y(oy + z * v.y));
          }
          g.stroke();
        }
        g.fillStyle = mix(colors.accent, colors.accent, 0, 0.85);
        for (const q of Q) {
          let x = ox + q.x,
            y = oy + q.y;
          if (hasV && projectionProgress > 0) {
            const z = q.x * v.x + q.y * v.y;
            x += (ox + z * v.x - x) * projectionProgress;
            y += (oy + z * v.y - y) * projectionProgress;
          }
          g.beginPath();
          g.arc(X(x), Y(y), 3.3, 0, 7);
          g.fill();
        }
        g.strokeStyle = colors.ink;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(X(ox) - 7, Y(oy));
        g.lineTo(X(ox) + 7, Y(oy));
        g.moveTo(X(ox), Y(oy) - 7);
        g.lineTo(X(ox), Y(oy) + 7);
        g.stroke();
        {
          const dx = tp.v[0] - mu.x,
            dy = tp.v[1] - mu.y,
            x = X(ox + dx),
            y = Y(oy + dy);
          if (hasV && know) {
            const z = dx * v.x + dy * v.y;
            g.strokeStyle = colors.ink;
            g.lineWidth = 1.4;
            g.setLineDash([4, 3]);
            g.beginPath();
            g.moveTo(x, y);
            g.lineTo(X(ox + z * v.x), Y(oy + z * v.y));
            g.stroke();
            g.setLineDash([]);
            g.fillStyle = colors.bg;
            g.beginPath();
            g.arc(X(ox + z * v.x), Y(oy + z * v.y), 5, 0, 7);
            g.fill();
            g.stroke();
          }
          markTP(g, x, y, colors);
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.56,
          m = { l: 44, r: 12, t: 26, b: 22 },
          Xa = (a) => m.l + (a / Math.PI) * (w - m.l - m.r),
          ym = Math.max(1e-9, e1.l1) * 1.12,
          Ya = (u) => h1 - m.b - (u / ym) * (h1 - m.t - m.b);
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, h1 - m.b);
        ctx.lineTo(w - m.r, h1 - m.b);
        ctx.stroke();
        ctx.font = MONO;
        ctx.textAlign = 'center';
        ctx.fillStyle = colors.muted;
        [0, 45, 90, 135, 180].forEach((a) =>
          ctx.fillText(a + '°', Xa((a * Math.PI) / 180), h1 - m.b + 14),
        );
        ctx.textAlign = 'right';
        ctx.fillText(uNum(ym / 1.12), m.l - 5, Ya(ym / 1.12) + 4);
        ctx.fillText('0', m.l - 5, Ya(0) + 4);
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.cut;
        ctx.fillText('gold: spread kept along the line', 8, 12);
        ctx.fillStyle = colors.ok;
        ctx.fillText('red: spread lost (they always add up to ' + f2(tot) + ')', 8, 25);
        [
          [(a) => va(a), colors.gold],
          [(a) => tot - va(a), colors.ok],
        ].forEach(([f, cl]) => {
          ctx.strokeStyle = cl;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          for (let i = 0; i <= 180; i++) {
            const a = (i * Math.PI) / 180;
            i ? ctx.lineTo(Xa(a), Ya(f(a))) : ctx.moveTo(Xa(a), Ya(f(a)));
          }
          ctx.stroke();
        });
        if ((know && !man) || ph === 'proj') {
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(Xa(e1.th), Ya(0));
          ctx.lineTo(Xa(e1.th), Ya(e1.l1));
          ctx.stroke();
          ctx.setLineDash([]);
        }
        if (hasV) {
          const a = ang();
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(Xa(a), Ya(0));
          ctx.lineTo(Xa(a), m.t + 4);
          ctx.stroke();
          ctx.fillStyle = colors.gold;
          ctx.beginPath();
          ctx.arc(Xa(a), Ya(va(a)), 6, 0, 7);
          ctx.fill();
          ctx.fillStyle = colors.ok;
          ctx.beginPath();
          ctx.arc(Xa(a), Ya(tot - va(a)), 5, 0, 7);
          ctx.fill();
        }
        const y0 = h1 + 18,
          hb = h - y0 - 24,
          Xs = (z) => w / 2 + (z * (w - 56)) / 8.4;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          hasV ? 'each point’s position along the line (its shadow)' : 'no direction chosen yet',
          8,
          y0,
        );
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(28, h - 22);
        ctx.lineTo(w - 28, h - 22);
        ctx.stroke();
        ctx.textAlign = 'center';
        [-4, -2, 0, 2, 4].forEach((z) => ctx.fillText(String(z), Xs(z), h - 7));
        if (hasV) {
          const bins = {};
          ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.85);
          const rad = Math.max(2, Math.min(3.4, hb / 26));
          for (const q of Q) {
            const z = q.x * v.x + q.y * v.y,
              b = Math.round(z / 0.14),
              k = (bins[b] = (bins[b] || 0) + 1);
            ctx.beginPath();
            ctx.arc(Xs(b * 0.14), h - 22 - rad - 1 - (k - 1) * rad * 2, rad, 0, 7);
            ctx.fill();
          }
          const sdv = Math.sqrt(va(ang()));
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(Xs(-sdv), h - 22);
          ctx.lineTo(Xs(sdv), h - 22);
          ctx.stroke();
        }
      }
      live.textContent = msg;
      highlightLessonStep(root, 'pc', activeStepIndex);
      if (hasV) {
        const a = ang();
        setReadout(root, 'th', deg(a));
        setReadout(root, 'va', f2(va(a)));
        setReadout(root, 'sh', ((100 * va(a)) / tot).toFixed(1) + '%');
        setReadout(root, 'er', f2(tot - va(a)));
      } else ['th', 'va', 'sh', 'er'].forEach((k) => setReadout(root, k, '–'));
      const sh = ph !== 'raw' && ph !== 'centre';
      setReadout(root, 'l1', sh ? f2(e1.l1) + ', ' + f2(e1.l2) : '–');
      setReadout(root, 'a1', (know && !man) || ph === 'proj' ? deg(e1.th) : '–');
      if (hasV && know) {
        const dx = tp.v[0] - mu.x,
          dy = tp.v[1] - mu.y,
          z = dx * v.x + dy * v.y,
          rx = mu.x + z * v.x,
          ry = mu.y + z * v.y;
        tp.out.textContent =
          'Your point minus the mean is (' +
          f2(dx) +
          ', ' +
          f2(dy) +
          '). Its position along the line is z = ' +
          f2(z) +
          '. Rebuilt from that one number it becomes (' +
          f2(rx) +
          ', ' +
          f2(ry) +
          '), which is ' +
          f2(Math.hypot(tp.v[0] - rx, tp.v[1] - ry)) +
          ' from where it started.';
      } else
        tp.out.textContent = 'Find a direction first: press One step, or move the angle slider.';
      if (hasV && !man) thb.set((ang() * 180) / Math.PI);
    };
    tp.on = () => draw();
    const gen = uDataBind(
      root,
      'pc',
      settings,
      () => {
        playback.stop();
        P = gen();
        comp();
        reset();
        draw();
      },
      mk,
    );
    P = gen();
    comp();
    reset();
    const thb = bind(
      root,
      'pc-th',
      (d) => {
        playback.stop();
        cancelAnimationFrame(twId);
        const a = (d * Math.PI) / 180;
        v = { x: Math.cos(a), y: Math.sin(a) };
        man = true;
        ph = 'manual';
        centeringProgress = 1;
        projectionProgress = 0;
        activeStepIndex = 2;
        const b = (e1.th * 180) / Math.PI;
        msg =
          'Your direction, at ' +
          d.toFixed(0) +
          '°, keeps ' +
          ((100 * va(a)) / (cv.a + cv.d)).toFixed(1) +
          '% of the spread. The most any direction can keep is ' +
          ((100 * e1.l1) / (cv.a + cv.d)).toFixed(1) +
          '%. ' +
          (Math.min(Math.abs(d - b), 180 - Math.abs(d - b)) < 3
            ? 'You have found it.'
            : 'Keep turning.');
        draw();
      },
      (d) => Math.round(d) + '°',
    );
    bind(root, 'pc-rho', (r) => {
      settings.rho = r;
      playback.stop();
      if (settings.ds === 'corr') {
        P = gen();
        comp();
        const k = man;
        reset();
        if (k) {
          thb.el.dispatchEvent(new Event('input'));
          return;
        }
      }
      draw();
    });
    const anim = Anim(() => {
      frameCount++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          reset();
          draw();
        }
        return;
      }
      if (frameCount % (ph === 'power' ? 30 : 70)) return;
      if (ph === 'manual') reset();
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#pc-play', root), anim);
    const checkRunStop = attachRunLimit(playback, root, 'pc', {
      done: () => ph === 'proj',
      restart: () => reset(),
    });
    $('#pc-step', root).onclick = () => {
      playback.stop();
      if (ph === 'manual') reset();
      step();
      draw();
    };
    $('#pc-reset', root).onclick = () => {
      playback.stop();
      reset();
      draw();
    };
    return {
      draw,
      stop: () => {
        playback.stop();
        cancelAnimationFrame(twId);
      },
      auto: () => {
        reset();
        draw();
        playback.start();
      },
    };
  },
};
