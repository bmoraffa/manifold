/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  readThemeColors,
  area,
  bind,
  clamp,
  clipPlot,
  curve,
  ellPath,
  f2,
  prepareCanvas,
  gammaS,
  lgam,
  mix,
  npdf,
  playBtn,
  plot,
  randn,
  randnFrom,
  rng,
  setReadout,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#by-c', root),
      c2 = $('#by-c2', root),
      cap = $('#by-cap', root),
      TR = { mx: 1.3, my: 0.6, s: 1, rho: 0.6 },
      settings = { n: 0, m0: [-1.8, -0.9], k0: 6, v0: 10, s0: 1.3, mode: 'map' };
    let seed = 21,
      data = [];
    const mk = () => {
      const rn = randnFrom(rng(seed)),
        q = Math.sqrt(1 - TR.rho * TR.rho);
      data = [];
      for (let i = 0; i < 120; i++) {
        const a = rn(),
          b = rn();
        data.push([TR.mx + 1.15 * TR.s * a, TR.my + 0.7 * TR.s * (TR.rho * a + q * b)]);
      }
    };
    mk();
    let samples = [],
      dirty = true,
      view = null,
      acc = 0,
      frameCount = 0,
      playback;
    const tpdf = (x, nu, loc, sc) => {
      const z = (x - loc) / sc;
      return (
        (Math.exp(lgam((nu + 1) / 2) - lgam(nu / 2)) / (Math.sqrt(nu * Math.PI) * sc)) *
        Math.pow(1 + (z * z) / nu, -(nu + 1) / 2)
      );
    };
    const post = () => {
      const n = settings.n;
      let mx = 0,
        my = 0;
      for (let i = 0; i < n; i++) {
        mx += data[i][0];
        my += data[i][1];
      }
      if (n) {
        mx /= n;
        my /= n;
      }
      let a = 0,
        b = 0,
        d = 0;
      for (let i = 0; i < n; i++) {
        const u = data[i][0] - mx,
          v = data[i][1] - my;
        a += u * u;
        b += u * v;
        d += v * v;
      }
      const kn = settings.k0 + n,
        vn = settings.v0 + n,
        l0 = settings.v0 * settings.s0 * settings.s0,
        dx = mx - settings.m0[0],
        dy = my - settings.m0[1],
        sh = n ? (settings.k0 * n) / kn : 0;
      return {
        n,
        mx,
        my,
        a,
        b,
        d,
        kn,
        vn,
        l0,
        mun: n
          ? [
              (settings.k0 * settings.m0[0] + n * mx) / kn,
              (settings.k0 * settings.m0[1] + n * my) / kn,
            ]
          : settings.m0.slice(),
        La: l0 + a + sh * dx * dx,
        Lb: b + sh * dx * dy,
        Ld: l0 + d + sh * dy * dy,
      };
    };
    const sampleNIW = (P) => {
      const det = P.La * P.Ld - P.Lb * P.Lb,
        ia = P.Ld / det,
        ib = -P.Lb / det,
        idd = P.La / det,
        l11 = Math.sqrt(ia),
        l21 = ib / l11,
        l22 = Math.sqrt(Math.max(1e-12, idd - l21 * l21));
      const a11 = Math.sqrt(2 * gammaS(P.vn / 2)),
        a21 = randn(),
        a22 = Math.sqrt(2 * gammaS((P.vn - 1) / 2)),
        m11 = l11 * a11,
        m21 = l21 * a11 + l22 * a21,
        m22 = l22 * a22;
      const wa = m11 * m11,
        wb = m11 * m21,
        wd = m21 * m21 + m22 * m22,
        wdet = wa * wd - wb * wb,
        sa = wd / wdet,
        sb = -wb / wdet,
        sd = wa / wdet;
      const c11 = Math.sqrt(sa / P.kn),
        c21 = sb / P.kn / c11,
        c22 = Math.sqrt(Math.max(1e-12, sd / P.kn - c21 * c21)),
        z1 = randn(),
        z2 = randn();
      return { mx: P.mun[0] + c11 * z1, my: P.mun[1] + c21 * z1 + c22 * z2, a: sa, b: sb, d: sd };
    };
    const draw = () => {
      const colors = readThemeColors(),
        P = post(),
        M = settings.mode,
        showMle = M === 'mle' || M === 'all',
        sP = M !== 'mle',
        showMap = M === 'map' || M === 'all',
        showBayes = M === 'bayes' || M === 'all',
        hasM = P.n >= 3 && P.a * P.d - P.b * P.b > 1e-9;
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          sc = Math.min(w / 9.6, h / 6.6),
          X = (x) => w / 2 + x * sc,
          Y = (y) => h / 2 - y * sc;
        view = { w, h, sc };
        const lab = (s, mx, my, d, col) => {
          ctx.font = SANS;
          ctx.fillStyle = col;
          ctx.textAlign = 'center';
          ctx.fillText(s, clamp(X(mx), 40, w - 40), clamp(Y(my + 2 * Math.sqrt(d)) - 7, 13, h - 6));
        };
        if (dirty) {
          samples = [];
          for (let i = 0; i < 12; i++) samples.push(sampleNIW(P));
          dirty = false;
        }
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 1;
        for (let k = -5; k <= 5; k++) {
          ctx.beginPath();
          ctx.moveTo(X(k), 0);
          ctx.lineTo(X(k), h);
          ctx.moveTo(0, Y(k));
          ctx.lineTo(w, Y(k));
          ctx.stroke();
        }
        const tsx = 1.15 * TR.s,
          tsy = 0.7 * TR.s;
        ctx.setLineDash([7, 5]);
        ctx.strokeStyle = mix(colors.ink, colors.ink, 0, 0.45);
        ctx.lineWidth = 1.3;
        ellPath(ctx, X, Y, TR.mx, TR.my, tsx * tsx, TR.rho * tsx * tsy, tsy * tsy, 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'center';
        ctx.fillText('truth, dashed', X(TR.mx), Math.min(h - 6, Y(TR.my - 2 * tsy) + 15));
        if (showBayes) {
          ctx.lineWidth = 1;
          ctx.strokeStyle = mix(colors.gold, colors.gold, 0, 0.5);
          for (const s of samples) {
            ellPath(ctx, X, Y, s.mx, s.my, s.a, s.b, s.d, 2);
            ctx.stroke();
          }
        }
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.9);
        for (let i = 0; i < settings.n; i++) {
          ctx.beginPath();
          ctx.arc(
            X(data[i][0]),
            Y(data[i][1]),
            i === settings.n - 1 && settings.n < data.length ? 6.5 : 3.2,
            0,
            7,
          );
          ctx.fill();
        }
        const hx = X(settings.m0[0]),
          hy = Y(settings.m0[1]);
        if (sP) {
          ctx.setLineDash([3, 5]);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1.6;
          ellPath(
            ctx,
            X,
            Y,
            settings.m0[0],
            settings.m0[1],
            settings.s0 * settings.s0,
            0,
            settings.s0 * settings.s0,
            2,
          );
          ctx.stroke();
          ctx.setLineDash([]);
          lab(
            'prior guess',
            settings.m0[0],
            settings.m0[1],
            settings.s0 * settings.s0,
            colors.muted,
          );
          if (P.n && showMap) {
            ctx.strokeStyle = colors.muted;
            ctx.lineWidth = 1.2;
            ctx.setLineDash([2, 4]);
            ctx.beginPath();
            ctx.moveTo(hx, hy);
            ctx.lineTo(X(P.mx), Y(P.my));
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
        if (showMle) {
          if (hasM) {
            ctx.strokeStyle = colors.ok;
            ctx.lineWidth = 3;
            ellPath(ctx, X, Y, P.mx, P.my, P.a / P.n, P.b / P.n, P.d / P.n, 2);
            ctx.stroke();
            lab('MLE', P.mx, P.my, P.d / P.n, colors.ok);
          }
          if (P.n) {
            ctx.fillStyle = colors.ok;
            ctx.beginPath();
            ctx.arc(X(P.mx), Y(P.my), 5.5, 0, 7);
            ctx.fill();
          }
        } else if (P.n && showMap) {
          ctx.strokeStyle = colors.ok;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(X(P.mx), Y(P.my), 5, 0, 7);
          ctx.stroke();
          ctx.font = MONO;
          ctx.fillStyle = colors.ok;
          ctx.textAlign = 'left';
          ctx.fillText('data mean', X(P.mx) + 9, Y(P.my) + 4);
        }
        if (showBayes) {
          const dn = P.kn * (P.vn - 1);
          ellPath(ctx, X, Y, P.mun[0], P.mun[1], P.La / dn, P.Lb / dn, P.Ld / dn, 2);
          ctx.fillStyle = mix(colors.gold, colors.gold, 0, 0.4);
          ctx.fill();
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 1.5;
          ctx.stroke();
          if (M === 'bayes') {
            ctx.font = MONO;
            ctx.fillStyle = colors.cut;
            ctx.textAlign = 'left';
            ctx.fillText('where the mean could be', X(P.mun[0]) + 10, Y(P.mun[1]) - 10);
          }
        }
        if (showMap) {
          const dn = P.vn + 4;
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 3.2;
          ellPath(ctx, X, Y, P.mun[0], P.mun[1], P.La / dn, P.Lb / dn, P.Ld / dn, 2);
          ctx.stroke();
          lab('MAP', P.mun[0], P.mun[1], P.Ld / dn, colors.cut);
          ctx.fillStyle = colors.gold;
          ctx.beginPath();
          ctx.arc(X(P.mun[0]), Y(P.mun[1]), 6, 0, 7);
          ctx.fill();
        }
        if (sP) {
          ctx.fillStyle = colors.bg;
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(hx, hy, 7.5, 0, 7);
          ctx.fill();
          ctx.stroke();
        }
      }
      {
        const { g: ctx, w, h } = prepareCanvas(c2),
          share = P.n / (settings.k0 + P.n),
          bx = 38,
          bw = w - 50;
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.muted;
        ctx.fillText('who has the vote', bx, 12);
        ctx.fillStyle = mix(colors.muted, colors.muted, 0, 0.45);
        ctx.fillRect(bx, 18, bw * (1 - share), 12);
        ctx.fillStyle = colors.accent;
        ctx.fillRect(bx + bw * (1 - share), 18, bw * share, 12);
        ctx.fillStyle = colors.muted;
        ctx.fillText('prior κ₀ = ' + (+settings.k0).toFixed(1), bx, 44);
        ctx.textAlign = 'right';
        ctx.fillStyle = colors.accent;
        ctx.fillText('data n = ' + P.n, bx + bw, 44);
        ctx.save();
        ctx.translate(0, 52);
        const hh = h - 52,
          pri = (x) =>
            tpdf(
              x,
              settings.v0 - 1,
              settings.m0[0],
              Math.sqrt(P.l0 / (settings.k0 * (settings.v0 - 1))),
            ),
          pos = (x) => tpdf(x, P.vn - 1, P.mun[0], Math.sqrt(P.La / (P.kn * (P.vn - 1))));
        const sdD = P.n >= 2 ? Math.sqrt(P.a / (P.n - 1) / P.n) : 0,
          dat = (x) => npdf(x, P.mx, sdD);
        let ym = 0.3;
        if (sP) ym = Math.max(ym, pri(settings.m0[0]), pos(P.mun[0]));
        if (sdD > 0 && M !== 'bayes') ym = Math.max(ym, dat(P.mx));
        ym = Math.min(ym * 1.12, 8);
        const Pl = plot(ctx, w, hh, [-4.8, 4.8], [0, ym], colors, { xlabel: 'mean, x-coordinate' });
        clipPlot(ctx, Pl);
        if (sP) {
          ctx.setLineDash([3, 5]);
          curve(ctx, Pl, pri, -4.8, 4.8);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.setLineDash([]);
        }
        if (sdD > 0 && M !== 'bayes') {
          curve(ctx, Pl, dat, -4.8, 4.8, 520);
          ctx.strokeStyle = colors.ok;
          ctx.lineWidth = 2.4;
          ctx.stroke();
        }
        if (sP) {
          area(ctx, Pl, pos, -4.8, 4.8, 520);
          ctx.fillStyle = mix(colors.gold, colors.gold, 0, 0.3);
          ctx.fill();
          curve(ctx, Pl, pos, -4.8, 4.8, 520);
          ctx.strokeStyle = colors.gold;
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        let ly = 70;
        const L = [];
        if (sP) L.push([colors.muted, 'prior, dotted']);
        if (M !== 'bayes')
          L.push([colors.ok, sdD > 0 ? 'data alone (peak = MLE)' : 'data alone (needs 2 points)']);
        if (sP)
          L.push([
            colors.cut,
            M === 'bayes' ? 'posterior (keep all of it)' : 'posterior (peak = MAP)',
          ]);
        L.forEach(([col, s]) => {
          ctx.fillStyle = col;
          ctx.fillText(s, 46, ly);
          ly += 15;
        });
      }
      const pc = Math.round((100 * P.n) / (settings.k0 + P.n));
      cap.textContent =
        M === 'mle'
          ? hasM
            ? 'MLE: the Gaussian that fits these ' +
              P.n +
              ' points best. It ignores any prior belief.'
            : 'MLE has no ellipse yet. It needs at least 3 points and has nothing to say before that.'
          : M === 'map'
            ? P.n
              ? 'MAP: the prior guess, pulled toward the data. With ' +
                P.n +
                ' points the data holds ' +
                pc +
                '% of the vote, so the gold dot sits ' +
                pc +
                '% of the way from the prior to the data mean.'
              : 'MAP: with no data yet, the estimate is the prior guess itself. Add points and watch it move.'
            : M === 'bayes'
              ? 'Bayesian: each thin outline is one Gaussian that could still explain the data. The shaded patch is where the true mean probably lies. Both tighten as points arrive.'
              : 'All three. Burgundy is MLE, gold is MAP, thin gold outlines are the Bayesian bundle. They disagree with little data and merge with a lot.';
      setReadout(root, 'n', String(P.n) + ' of ' + data.length);
      setReadout(root, 'mle', P.n ? '(' + f2(P.mx) + ', ' + f2(P.my) + ')' : '–');
      setReadout(root, 'map', '(' + f2(P.mun[0]) + ', ' + f2(P.mun[1]) + ')');
      setReadout(root, 'w', pc + '%');
      setReadout(root, 'pm', '(' + f2(settings.m0[0]) + ', ' + f2(settings.m0[1]) + ')');
    };
    const chg = () => {
      dirty = true;
      draw();
    };
    const setMode = (m) => {
      settings.mode = m;
      root
        .querySelectorAll('#by-mode button')
        .forEach((b) => b.setAttribute('aria-pressed', b.dataset.m === m ? 'true' : 'false'));
      chg();
    };
    $('#by-mode', root).addEventListener('click', (e) => {
      const m = e.target.dataset && e.target.dataset.m;
      if (m) setMode(m);
    });
    const anim = Anim(() => {
      frameCount++;
      acc += settings.n < 10 ? 0.022 : settings.n < 40 ? 0.1 : 0.4;
      if (acc >= 1) {
        acc = 0;
        if (settings.n < data.length) {
          settings.n++;
          dirty = true;
        }
      }
      if (frameCount % 18 === 0) dirty = true;
      draw();
    });
    playback = playBtn($('#by-play', root), anim);
    $('#by-play', root).addEventListener('click', () => {
      if (anim.on && settings.n >= data.length) {
        settings.n = 0;
        dirty = true;
      }
    });
    $('#by-step', root).onclick = () => {
      playback.stop();
      if (settings.n < data.length) settings.n++;
      chg();
    };
    $('#by-reset', root).onclick = () => {
      playback.stop();
      settings.n = 0;
      acc = 0;
      chg();
    };
    bind(
      root,
      'by-k',
      (v) => {
        settings.k0 = v;
        chg();
      },
      (v) => (+v).toFixed(1),
    );
    bind(
      root,
      'by-v',
      (v) => {
        settings.v0 = v;
        chg();
      },
      (v) => String(v),
    );
    bind(root, 'by-s', (v) => {
      settings.s0 = v;
      chg();
    });
    bind(root, 'by-ds', (v) => {
      TR.s = v;
      mk();
      chg();
    });
    bind(root, 'by-dr', (v) => {
      TR.rho = v;
      mk();
      chg();
    });
    $('#by-new', root).onclick = () => {
      seed++;
      mk();
      settings.n = 0;
      acc = 0;
      chg();
      playback.start();
    };
    let drag = false;
    const mv = (e) => {
      if (!view) return;
      const r = mainCanvas.getBoundingClientRect();
      settings.m0 = [
        clamp((e.clientX - r.left - view.w / 2) / view.sc, -4.4, 4.4),
        clamp(-(e.clientY - r.top - view.h / 2) / view.sc, -2.9, 2.9),
      ];
      chg();
    };
    mainCanvas.addEventListener('pointerdown', (e) => {
      if (settings.mode === 'mle') return;
      drag = true;
      mainCanvas.setPointerCapture(e.pointerId);
      mv(e);
    });
    mainCanvas.addEventListener('pointermove', (e) => {
      if (drag) mv(e);
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      mainCanvas.addEventListener(n, () => {
        drag = false;
      }),
    );
    setMode('map');
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
