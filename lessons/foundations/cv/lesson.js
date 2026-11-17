/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  clipPlot,
  curve,
  f3,
  fit,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  setR,
} from '../../../js/core/shared.js';
import { dot, legendre, lsq } from '../../regression/shared.js';

export default {
  init(root) {
    const c = $('#cv-c', root),
      ce = $('#cv-e', root),
      S = { k: 5, d: 3, n: 40, noise: 0.3 },
      Fq = (x) => Math.sin(2.8 * x);
    let seed = 14,
      pts = [],
      fold = 0,
      FE = [],
      CVD = [],
      TTD = [],
      TRD = [],
      bcur = null,
      pb;
    const fitOn = (P, d) =>
        lsq(
          P.map((p) => legendre(p.x, d)),
          P.map((p) => p.y),
          1e-8,
        ),
      er = (P, b, d) =>
        P.reduce((s, p) => s + (p.y - clamp(dot(b, legendre(p.x, d)), -20, 20)) ** 2, 0) / P.length;
    const cvFor = (d) => {
      const fe = [];
      for (let j = 0; j < S.k; j++) {
        const tr = pts.filter((p) => p.f !== j),
          va = pts.filter((p) => p.f === j);
        fe.push(va.length && tr.length > d ? er(va, fitOn(tr, d), d) : NaN);
      }
      return fe;
    };
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < S.n; i++) {
        const x = r() * 2 - 1;
        pts.push({ x, y: Fq(x) + S.noise * rn() });
      }
      const ix = pts.map((_, i) => i);
      for (let i = ix.length - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [ix[i], ix[j]] = [ix[j], ix[i]];
      }
      ix.forEach((q, i) => (pts[q].f = i % S.k));
      const big = [];
      for (let i = 0; i < 3000; i++) {
        const x = r() * 2 - 1;
        big.push({ x, y: Fq(x) + S.noise * rn() });
      }
      CVD = [];
      TTD = [];
      TRD = [];
      for (let d = 1; d <= 12; d++) {
        const fe = cvFor(d).filter(isFinite);
        CVD[d] = fe.length ? fe.reduce((a, b) => a + b, 0) / fe.length : NaN;
        const b = fitOn(pts, d);
        TTD[d] = er(big, b, d);
        TRD[d] = er(pts, b, d);
      }
    };
    const turn = () => {
      FE = cvFor(S.d);
      bcur = fitOn(
        pts.filter((p) => p.f !== fold),
        S.d,
      );
    };
    gen();
    turn();
    const draw = () => {
      const t = T();
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, [-1.05, 1.05], [-2.2, 2.6], t, {
            yticks: [-2, -1, 0, 1, 2],
            xstep: 0.5,
            xlabel: 'x',
          }),
          bw = (w - P.m.l - P.m.r) / S.k;
        for (let j = 0; j < S.k; j++) {
          g.fillStyle = j === fold ? t.ok : mix(t.accent, t.accent, 0, 0.55);
          g.fillRect(P.m.l + j * bw + 1, P.m.t - 4, bw - 2, 9);
        }
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('blocks: burgundy is held out this turn', P.m.l, P.m.t + 19);
        clipPlot(g, P);
        g.setLineDash([6, 5]);
        curve(g, P, Fq, -1, 1);
        g.strokeStyle = t.muted;
        g.lineWidth = 1.4;
        g.stroke();
        g.setLineDash([]);
        curve(g, P, (x) => clamp(dot(bcur, legendre(x, S.d)), -9, 9), -1, 1, 300);
        g.strokeStyle = t.gold;
        g.lineWidth = 3.2;
        g.stroke();
        for (const p of pts) {
          if (p.f === fold) {
            const yh = clamp(dot(bcur, legendre(p.x, S.d)), -9, 9);
            g.strokeStyle = mix(t.ok, t.ok, 0, 0.6);
            g.lineWidth = 1.2;
            g.beginPath();
            g.moveTo(P.X(p.x), P.Y(p.y));
            g.lineTo(P.X(p.x), P.Y(yh));
            g.stroke();
            g.strokeStyle = t.ok;
            g.lineWidth = 2.2;
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 5, 0, 7);
            g.stroke();
          } else {
            g.fillStyle = mix(t.accent, t.accent, 0, 0.85);
            g.beginPath();
            g.arc(P.X(p.x), P.Y(p.y), 3.8, 0, 7);
            g.fill();
          }
        }
        g.restore();
      }
      {
        const { g, w, h } = fit(ce),
          h1 = h * 0.42,
          fe = FE.filter(isFinite),
          mean = fe.length ? fe.reduce((a, b) => a + b, 0) / fe.length : 0,
          mx = Math.max(0.05, ...fe) * 1.15,
          m = { l: 38, r: 12 };
        g.font = MONO;
        const bw = (w - m.l - m.r) / S.k,
          Yb = (v) => h1 - 12 - (Math.min(v, mx) / mx) * (h1 - 30);
        for (let j = 0; j < S.k; j++) {
          if (!isFinite(FE[j])) continue;
          g.fillStyle = j === fold ? t.ok : mix(t.ok, t.ok, 0, 0.35);
          g.fillRect(m.l + j * bw + 3, Yb(FE[j]), bw - 6, h1 - 12 - Yb(FE[j]));
        }
        g.strokeStyle = t.ink;
        g.lineWidth = 1.6;
        g.setLineDash([5, 4]);
        g.beginPath();
        g.moveTo(m.l, Yb(mean));
        g.lineTo(w - m.r, Yb(mean));
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = t.ink;
        g.textAlign = 'right';
        g.fillText('average = CV estimate ' + f3(mean), w - m.r, Math.max(10, Yb(mean) - 5));
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h1 - 12);
        g.lineTo(w - m.r, h1 - 12);
        g.stroke();
        g.save();
        g.translate(0, h1);
        const hh = h - h1,
          LY = [-2.3, 0.8],
          P = plot(g, w, hh, [0.5, 12.5], LY, t, { xlabel: 'degree' }),
          lg = (v) => clamp(Math.log10(Math.max(v, 1e-9)), LY[0], LY[1]);
        clipPlot(g, P);
        g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
        g.lineWidth = 9;
        g.beginPath();
        g.moveTo(P.X(S.d), P.Y(LY[0]));
        g.lineTo(P.X(S.d), P.Y(LY[1]));
        g.stroke();
        [
          [TRD, t.accent],
          [CVD, t.ok],
          [TTD, t.ink],
        ].forEach(([E, cl]) => {
          g.strokeStyle = cl;
          g.lineWidth = 2.2;
          g.beginPath();
          let on = false;
          for (let d = 1; d <= 12; d++) {
            if (!isFinite(E[d])) {
              on = false;
              continue;
            }
            on ? g.lineTo(P.X(d), P.Y(lg(E[d]))) : g.moveTo(P.X(d), P.Y(lg(E[d])));
            on = true;
          }
          g.stroke();
        });
        g.restore();
        g.textAlign = 'left';
        let lx = P.m.l + 8;
        [
          [t.accent, 'training'],
          [t.ok, 'cross-validation'],
          [t.ink, 'true test'],
        ].forEach(([cl, sx]) => {
          g.fillStyle = cl;
          g.fillText(sx, lx, P.m.t + 6);
          lx += g.measureText(sx).width + 16;
        });
        g.restore();
      }
      const fe = FE.filter(isFinite),
        mean = fe.reduce((a, b) => a + b, 0) / (fe.length || 1),
        sd = Math.sqrt(fe.reduce((a, b) => a + (b - mean) ** 2, 0) / (fe.length || 1));
      let pk = 1;
      for (let d = 2; d <= 12; d++) if (CVD[d] < CVD[pk]) pk = d;
      setR(root, 'cv', f3(mean));
      setR(root, 'sd', f3(sd));
      setR(root, 'tr', f3(TRD[S.d]));
      setR(root, 'tt', f3(TTD[S.d]));
      setR(root, 'pick', String(pk));
    };
    let fr = 0;
    const anim = Anim(() => {
      if (++fr % 75) return;
      fold = (fold + 1) % S.k;
      turn();
      draw();
    });
    pb = playBtn($('#cv-play', root), anim);
    const re = () => {
      fold = 0;
      gen();
      turn();
      draw();
    };
    bind(
      root,
      'cv-k',
      (v) => {
        S.k = v;
        re();
      },
      (v) => String(v),
    );
    bind(
      root,
      'cv-d',
      (v) => {
        S.d = v;
        turn();
        draw();
      },
      (v) => String(v),
    );
    bind(
      root,
      'cv-n',
      (v) => {
        S.n = v;
        re();
      },
      (v) => String(v),
    );
    bind(root, 'cv-s', (v) => {
      S.noise = v;
      re();
    });
    $('#cv-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
