/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  T,
  bind,
  clipPlot,
  f3,
  fit,
  playBtn,
  plot,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { heatmap } from '../shared.js';

export default {
  init(root) {
    const c = $('#hs-c', root),
      ce = $('#hs-e', root),
      BS = [4, 9, 16, 25, 36, 49, 64],
      S = { bi: 3, imp: 0.15, w: 0.09 };
    let seed = 3,
      pk = [0.7, 0.4],
      G = [],
      Rn = [],
      shown = 0,
      win = 0,
      pb;
    const sc = (a, b, p = pk) =>
      Math.exp(-(((a - p[0]) / S.w) ** 2)) *
      (1 - S.imp + S.imp * Math.exp(-(((b - p[1]) / (S.w * 2.2)) ** 2)));
    const plan = (r, B) => {
      const m = Math.round(Math.sqrt(B)),
        g = [];
      for (let i = 0; i < m; i++)
        for (let j = 0; j < m; j++) g.push([(i + 0.5) / m, (j + 0.5) / m]);
      for (let i = g.length - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [g[i], g[j]] = [g[j], g[i]];
      }
      const q = [];
      for (let i = 0; i < B; i++) q.push([r(), r()]);
      return [g, q];
    };
    const gen = () => {
      const r = rng(seed);
      pk = [0.1 + 0.8 * r(), 0.1 + 0.8 * r()];
      [G, Rn] = plan(r, BS[S.bi]);
      shown = 0;
      const r2 = rng(999);
      let wn = 0,
        tot = 0;
      for (let k = 0; k < 400; k++) {
        const p = [0.1 + 0.8 * r2(), 0.1 + 0.8 * r2()],
          [g, q] = plan(r2, BS[S.bi]),
          bg = Math.max(...g.map((v) => sc(v[0], v[1], p))),
          br = Math.max(...q.map((v) => sc(v[0], v[1], p)));
        if (br > bg) wn++;
        tot++;
      }
      win = wn / tot;
    };
    gen();
    const draw = () => {
      const t = T(),
        B = BS[S.bi];
      {
        const { g, w, h } = fit(c),
          gap = 22,
          pw = (w - gap) / 2,
          ph = h - 30,
          key = 'hs' + seed + '|' + S.imp + '|' + S.w;
        [
          [G, 0, 'Grid', t.accent],
          [Rn, pw + gap, 'Random', t.cut],
        ].forEach(([L, ox, nm, cl]) => {
          const hm = heatmap(key, pw, ph - 18, t, (x, y) => -sc((x + 3) / 6, (y + 3) / 6), 3, 60);
          g.drawImage(hm, ox, 18, pw, ph - 18);
          const X = (v) => ox + v * pw,
            Y = (v) => 18 + (1 - v) * (ph - 18);
          for (let i = 0; i < Math.min(shown, L.length); i++) {
            const q = L[i];
            g.strokeStyle = t.ink;
            g.lineWidth = 1.6;
            g.beginPath();
            g.moveTo(X(q[0]), 4);
            g.lineTo(X(q[0]), 15);
            g.stroke();
            g.fillStyle = t.ink;
            g.strokeStyle = t.bg;
            g.lineWidth = 1.5;
            g.beginPath();
            g.arc(X(q[0]), Y(q[1]), 5, 0, 7);
            g.fill();
            g.stroke();
          }
          let bi = -1,
            bv = -1;
          for (let i = 0; i < Math.min(shown, L.length); i++) {
            const v = sc(L[i][0], L[i][1]);
            if (v > bv) {
              bv = v;
              bi = i;
            }
          }
          if (bi >= 0) {
            g.strokeStyle = t.ok;
            g.lineWidth = 3;
            g.beginPath();
            g.arc(X(L[bi][0]), Y(L[bi][1]), 10, 0, 7);
            g.stroke();
          }
          g.font = SANS;
          g.fillStyle = t.ink;
          g.textAlign = 'left';
          g.fillText(nm, ox, h - 8);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('setting 1 (matters) →', ox + pw, h - 8);
        });
      }
      {
        const { g, w, h } = fit(ce),
          P = plot(g, w, h, [0, B], [0, 1.05], t, {
            yticks: [0.25, 0.5, 0.75, 1],
            xstep: B > 30 ? 10 : B > 12 ? 5 : 2,
            xlabel: 'models trained',
          });
        clipPlot(g, P);
        [
          [G, t.accent],
          [Rn, t.gold],
        ].forEach(([L, cl]) => {
          g.strokeStyle = cl;
          g.lineWidth = 2.8;
          g.beginPath();
          let b = 0;
          g.moveTo(P.X(0), P.Y(0));
          for (let i = 0; i < Math.min(shown, L.length); i++) {
            g.lineTo(P.X(i + 1), P.Y(b));
            b = Math.max(b, sc(L[i][0], L[i][1]));
            g.lineTo(P.X(i + 1), P.Y(b));
          }
          g.stroke();
        });
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.accent;
        g.fillText('grid', P.m.l + 8, P.m.t + 8);
        g.fillStyle = t.cut;
        g.fillText('random', P.m.l + 8, P.m.t + 23);
      }
      const best = (L) => {
          let b = 0;
          for (let i = 0; i < Math.min(shown, L.length); i++) b = Math.max(b, sc(L[i][0], L[i][1]));
          return b;
        },
        dist = (L) => new Set(L.slice(0, shown).map((q) => q[0].toFixed(4))).size;
      setR(root, 'g', f3(best(G)));
      setR(root, 'r', f3(best(Rn)));
      setR(root, 'dg', String(dist(G)));
      setR(root, 'dr', String(dist(Rn)));
      setR(root, 'win', (100 * win).toFixed(0) + '%');
    };
    let fr = 0,
      wait = 0;
    const anim = Anim(() => {
      fr++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          seed++;
          gen();
          draw();
        }
        return;
      }
      if (fr % (BS[S.bi] > 30 ? 5 : 10)) return;
      if (shown < BS[S.bi]) {
        shown++;
        draw();
      } else pb.stop();
    });
    pb = playBtn($('#hs-play', root), anim);
    runCtl(pb, root, 'hs', {
      done: () => shown >= BS[S.bi],
      restart: () => {
        seed++;
        gen();
      },
    });
    const re = () => {
      gen();
      draw();
    };
    bind(
      root,
      'hs-b',
      (v) => {
        S.bi = v;
        re();
      },
      (v) => String(BS[v]),
    );
    bind(root, 'hs-imp', (v) => {
      S.imp = v;
      re();
    });
    bind(root, 'hs-w', (v) => {
      S.w = v;
      re();
    });
    $('#hs-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
