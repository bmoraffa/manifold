/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  T,
  area,
  bind,
  clamp,
  clipPlot,
  curve,
  ellPath,
  f2,
  fit,
  gammaS,
  lgam,
  mix,
  npdf,
  playBtn,
  plot,
  randn,
  randnFrom,
  rng,
  setR,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#by-c', root),
      c2 = $('#by-c2', root),
      cap = $('#by-cap', root),
      TR = { mx: 1.3, my: 0.6, s: 1, rho: 0.6 },
      S = { n: 0, m0: [-1.8, -0.9], k0: 6, v0: 10, s0: 1.3, mode: 'map' };
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
      fr = 0,
      pb;
    const tpdf = (x, nu, loc, sc) => {
      const z = (x - loc) / sc;
      return (
        (Math.exp(lgam((nu + 1) / 2) - lgam(nu / 2)) / (Math.sqrt(nu * Math.PI) * sc)) *
        Math.pow(1 + (z * z) / nu, -(nu + 1) / 2)
      );
    };
    const post = () => {
      const n = S.n;
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
      const kn = S.k0 + n,
        vn = S.v0 + n,
        l0 = S.v0 * S.s0 * S.s0,
        dx = mx - S.m0[0],
        dy = my - S.m0[1],
        sh = n ? (S.k0 * n) / kn : 0;
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
        mun: n ? [(S.k0 * S.m0[0] + n * mx) / kn, (S.k0 * S.m0[1] + n * my) / kn] : S.m0.slice(),
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
      const t = T(),
        P = post(),
        M = S.mode,
        sM = M === 'mle' || M === 'all',
        sP = M !== 'mle',
        sA = M === 'map' || M === 'all',
        sB = M === 'bayes' || M === 'all',
        hasM = P.n >= 3 && P.a * P.d - P.b * P.b > 1e-9;
      {
        const { g, w, h } = fit(c),
          sc = Math.min(w / 9.6, h / 6.6),
          X = (x) => w / 2 + x * sc,
          Y = (y) => h / 2 - y * sc;
        view = { w, h, sc };
        const lab = (s, mx, my, d, col) => {
          g.font = SANS;
          g.fillStyle = col;
          g.textAlign = 'center';
          g.fillText(s, clamp(X(mx), 40, w - 40), clamp(Y(my + 2 * Math.sqrt(d)) - 7, 13, h - 6));
        };
        if (dirty) {
          samples = [];
          for (let i = 0; i < 12; i++) samples.push(sampleNIW(P));
          dirty = false;
        }
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        for (let k = -5; k <= 5; k++) {
          g.beginPath();
          g.moveTo(X(k), 0);
          g.lineTo(X(k), h);
          g.moveTo(0, Y(k));
          g.lineTo(w, Y(k));
          g.stroke();
        }
        const tsx = 1.15 * TR.s,
          tsy = 0.7 * TR.s;
        g.setLineDash([7, 5]);
        g.strokeStyle = mix(t.ink, t.ink, 0, 0.45);
        g.lineWidth = 1.3;
        ellPath(g, X, Y, TR.mx, TR.my, tsx * tsx, TR.rho * tsx * tsy, tsy * tsy, 2);
        g.stroke();
        g.setLineDash([]);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('truth, dashed', X(TR.mx), Math.min(h - 6, Y(TR.my - 2 * tsy) + 15));
        if (sB) {
          g.lineWidth = 1;
          g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
          for (const s of samples) {
            ellPath(g, X, Y, s.mx, s.my, s.a, s.b, s.d, 2);
            g.stroke();
          }
        }
        g.fillStyle = mix(t.accent, t.accent, 0, 0.9);
        for (let i = 0; i < S.n; i++) {
          g.beginPath();
          g.arc(X(data[i][0]), Y(data[i][1]), i === S.n - 1 && S.n < data.length ? 6.5 : 3.2, 0, 7);
          g.fill();
        }
        const hx = X(S.m0[0]),
          hy = Y(S.m0[1]);
        if (sP) {
          g.setLineDash([3, 5]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.6;
          ellPath(g, X, Y, S.m0[0], S.m0[1], S.s0 * S.s0, 0, S.s0 * S.s0, 2);
          g.stroke();
          g.setLineDash([]);
          lab('prior guess', S.m0[0], S.m0[1], S.s0 * S.s0, t.muted);
          if (P.n && sA) {
            g.strokeStyle = t.muted;
            g.lineWidth = 1.2;
            g.setLineDash([2, 4]);
            g.beginPath();
            g.moveTo(hx, hy);
            g.lineTo(X(P.mx), Y(P.my));
            g.stroke();
            g.setLineDash([]);
          }
        }
        if (sM) {
          if (hasM) {
            g.strokeStyle = t.ok;
            g.lineWidth = 3;
            ellPath(g, X, Y, P.mx, P.my, P.a / P.n, P.b / P.n, P.d / P.n, 2);
            g.stroke();
            lab('MLE', P.mx, P.my, P.d / P.n, t.ok);
          }
          if (P.n) {
            g.fillStyle = t.ok;
            g.beginPath();
            g.arc(X(P.mx), Y(P.my), 5.5, 0, 7);
            g.fill();
          }
        } else if (P.n && sA) {
          g.strokeStyle = t.ok;
          g.lineWidth = 2;
          g.beginPath();
          g.arc(X(P.mx), Y(P.my), 5, 0, 7);
          g.stroke();
          g.font = MONO;
          g.fillStyle = t.ok;
          g.textAlign = 'left';
          g.fillText('data mean', X(P.mx) + 9, Y(P.my) + 4);
        }
        if (sB) {
          const dn = P.kn * (P.vn - 1);
          ellPath(g, X, Y, P.mun[0], P.mun[1], P.La / dn, P.Lb / dn, P.Ld / dn, 2);
          g.fillStyle = mix(t.gold, t.gold, 0, 0.4);
          g.fill();
          g.strokeStyle = t.gold;
          g.lineWidth = 1.5;
          g.stroke();
          if (M === 'bayes') {
            g.font = MONO;
            g.fillStyle = t.cut;
            g.textAlign = 'left';
            g.fillText('where the mean could be', X(P.mun[0]) + 10, Y(P.mun[1]) - 10);
          }
        }
        if (sA) {
          const dn = P.vn + 4;
          g.strokeStyle = t.gold;
          g.lineWidth = 3.2;
          ellPath(g, X, Y, P.mun[0], P.mun[1], P.La / dn, P.Lb / dn, P.Ld / dn, 2);
          g.stroke();
          lab('MAP', P.mun[0], P.mun[1], P.Ld / dn, t.cut);
          g.fillStyle = t.gold;
          g.beginPath();
          g.arc(X(P.mun[0]), Y(P.mun[1]), 6, 0, 7);
          g.fill();
        }
        if (sP) {
          g.fillStyle = t.bg;
          g.strokeStyle = t.muted;
          g.lineWidth = 2.5;
          g.beginPath();
          g.arc(hx, hy, 7.5, 0, 7);
          g.fill();
          g.stroke();
        }
      }
      {
        const { g, w, h } = fit(c2),
          share = P.n / (S.k0 + P.n),
          bx = 38,
          bw = w - 50;
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('who has the vote', bx, 12);
        g.fillStyle = mix(t.muted, t.muted, 0, 0.45);
        g.fillRect(bx, 18, bw * (1 - share), 12);
        g.fillStyle = t.accent;
        g.fillRect(bx + bw * (1 - share), 18, bw * share, 12);
        g.fillStyle = t.muted;
        g.fillText('prior κ₀ = ' + (+S.k0).toFixed(1), bx, 44);
        g.textAlign = 'right';
        g.fillStyle = t.accent;
        g.fillText('data n = ' + P.n, bx + bw, 44);
        g.save();
        g.translate(0, 52);
        const hh = h - 52,
          pri = (x) => tpdf(x, S.v0 - 1, S.m0[0], Math.sqrt(P.l0 / (S.k0 * (S.v0 - 1)))),
          pos = (x) => tpdf(x, P.vn - 1, P.mun[0], Math.sqrt(P.La / (P.kn * (P.vn - 1))));
        const sdD = P.n >= 2 ? Math.sqrt(P.a / (P.n - 1) / P.n) : 0,
          dat = (x) => npdf(x, P.mx, sdD);
        let ym = 0.3;
        if (sP) ym = Math.max(ym, pri(S.m0[0]), pos(P.mun[0]));
        if (sdD > 0 && M !== 'bayes') ym = Math.max(ym, dat(P.mx));
        ym = Math.min(ym * 1.12, 8);
        const Pl = plot(g, w, hh, [-4.8, 4.8], [0, ym], t, { xlabel: 'mean, x-coordinate' });
        clipPlot(g, Pl);
        if (sP) {
          g.setLineDash([3, 5]);
          curve(g, Pl, pri, -4.8, 4.8);
          g.strokeStyle = t.muted;
          g.lineWidth = 2;
          g.stroke();
          g.setLineDash([]);
        }
        if (sdD > 0 && M !== 'bayes') {
          curve(g, Pl, dat, -4.8, 4.8, 520);
          g.strokeStyle = t.ok;
          g.lineWidth = 2.4;
          g.stroke();
        }
        if (sP) {
          area(g, Pl, pos, -4.8, 4.8, 520);
          g.fillStyle = mix(t.gold, t.gold, 0, 0.3);
          g.fill();
          curve(g, Pl, pos, -4.8, 4.8, 520);
          g.strokeStyle = t.gold;
          g.lineWidth = 3;
          g.stroke();
        }
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        let ly = 70;
        const L = [];
        if (sP) L.push([t.muted, 'prior, dotted']);
        if (M !== 'bayes')
          L.push([t.ok, sdD > 0 ? 'data alone (peak = MLE)' : 'data alone (needs 2 points)']);
        if (sP)
          L.push([t.cut, M === 'bayes' ? 'posterior (keep all of it)' : 'posterior (peak = MAP)']);
        L.forEach(([col, s]) => {
          g.fillStyle = col;
          g.fillText(s, 46, ly);
          ly += 15;
        });
      }
      const pc = Math.round((100 * P.n) / (S.k0 + P.n));
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
      setR(root, 'n', String(P.n) + ' of ' + data.length);
      setR(root, 'mle', P.n ? '(' + f2(P.mx) + ', ' + f2(P.my) + ')' : '–');
      setR(root, 'map', '(' + f2(P.mun[0]) + ', ' + f2(P.mun[1]) + ')');
      setR(root, 'w', pc + '%');
      setR(root, 'pm', '(' + f2(S.m0[0]) + ', ' + f2(S.m0[1]) + ')');
    };
    const chg = () => {
      dirty = true;
      draw();
    };
    const setMode = (m) => {
      S.mode = m;
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
      fr++;
      acc += S.n < 10 ? 0.022 : S.n < 40 ? 0.1 : 0.4;
      if (acc >= 1) {
        acc = 0;
        if (S.n < data.length) {
          S.n++;
          dirty = true;
        }
      }
      if (fr % 18 === 0) dirty = true;
      draw();
    });
    pb = playBtn($('#by-play', root), anim);
    $('#by-play', root).addEventListener('click', () => {
      if (anim.on && S.n >= data.length) {
        S.n = 0;
        dirty = true;
      }
    });
    $('#by-step', root).onclick = () => {
      pb.stop();
      if (S.n < data.length) S.n++;
      chg();
    };
    $('#by-reset', root).onclick = () => {
      pb.stop();
      S.n = 0;
      acc = 0;
      chg();
    };
    bind(
      root,
      'by-k',
      (v) => {
        S.k0 = v;
        chg();
      },
      (v) => (+v).toFixed(1),
    );
    bind(
      root,
      'by-v',
      (v) => {
        S.v0 = v;
        chg();
      },
      (v) => String(v),
    );
    bind(root, 'by-s', (v) => {
      S.s0 = v;
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
      S.n = 0;
      acc = 0;
      chg();
      pb.start();
    };
    let drag = false;
    const mv = (e) => {
      if (!view) return;
      const r = c.getBoundingClientRect();
      S.m0 = [
        clamp((e.clientX - r.left - view.w / 2) / view.sc, -4.4, 4.4),
        clamp(-(e.clientY - r.top - view.h / 2) / view.sc, -2.9, 2.9),
      ];
      chg();
    };
    c.addEventListener('pointerdown', (e) => {
      if (S.mode === 'mle') return;
      drag = true;
      c.setPointerCapture(e.pointerId);
      mv(e);
    });
    c.addEventListener('pointermove', (e) => {
      if (drag) mv(e);
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      c.addEventListener(n, () => {
        drag = false;
      }),
    );
    setMode('map');
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
