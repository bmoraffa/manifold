/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  readThemeColors,
  bind,
  clipPlot,
  f3,
  prepareCanvas,
  playBtn,
  plot,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { heatmap } from '../shared.js';

export default {
  init(root) {
    const mainCanvas = $('#hs-c', root),
      ce = $('#hs-e', root),
      BS = [4, 9, 16, 25, 36, 49, 64],
      settings = { bi: 3, imp: 0.15, w: 0.09 };
    let seed = 3,
      scorePeak = [0.7, 0.4],
      gridCandidates = [],
      randomCandidates = [],
      shown = 0,
      win = 0,
      playback;
    const sc = (a, b, p = scorePeak) =>
      Math.exp(-(((a - p[0]) / settings.w) ** 2)) *
      (1 - settings.imp + settings.imp * Math.exp(-(((b - p[1]) / (settings.w * 2.2)) ** 2)));
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
      scorePeak = [0.1 + 0.8 * r(), 0.1 + 0.8 * r()];
      [gridCandidates, randomCandidates] = plan(r, BS[settings.bi]);
      shown = 0;
      const r2 = rng(999);
      let wn = 0,
        tot = 0;
      for (let k = 0; k < 400; k++) {
        const p = [0.1 + 0.8 * r2(), 0.1 + 0.8 * r2()],
          [g, q] = plan(r2, BS[settings.bi]),
          bg = Math.max(...g.map((v) => sc(v[0], v[1], p))),
          br = Math.max(...q.map((v) => sc(v[0], v[1], p)));
        if (br > bg) wn++;
        tot++;
      }
      win = wn / tot;
    };
    gen();
    const draw = () => {
      const colors = readThemeColors(),
        B = BS[settings.bi];
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          gap = 22,
          pw = (w - gap) / 2,
          ph = h - 30,
          key = 'hs' + seed + '|' + settings.imp + '|' + settings.w;
        [
          [gridCandidates, 0, 'Grid', colors.accent],
          [randomCandidates, pw + gap, 'Random', colors.cut],
        ].forEach(([L, ox, nm, cl]) => {
          const hm = heatmap(
            key,
            pw,
            ph - 18,
            colors,
            (x, y) => -sc((x + 3) / 6, (y + 3) / 6),
            3,
            60,
          );
          ctx.drawImage(hm, ox, 18, pw, ph - 18);
          const X = (v) => ox + v * pw,
            Y = (v) => 18 + (1 - v) * (ph - 18);
          for (let i = 0; i < Math.min(shown, L.length); i++) {
            const q = L[i];
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            ctx.moveTo(X(q[0]), 4);
            ctx.lineTo(X(q[0]), 15);
            ctx.stroke();
            ctx.fillStyle = colors.ink;
            ctx.strokeStyle = colors.bg;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(X(q[0]), Y(q[1]), 5, 0, 7);
            ctx.fill();
            ctx.stroke();
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
            ctx.strokeStyle = colors.ok;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(X(L[bi][0]), Y(L[bi][1]), 10, 0, 7);
            ctx.stroke();
          }
          ctx.font = SANS;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'left';
          ctx.fillText(nm, ox, h - 8);
          ctx.font = MONO;
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'right';
          ctx.fillText('setting 1 (matters) →', ox + pw, h - 8);
        });
      }
      {
        const { g: ctx, w, h } = prepareCanvas(ce),
          P = plot(ctx, w, h, [0, B], [0, 1.05], colors, {
            yticks: [0.25, 0.5, 0.75, 1],
            xstep: B > 30 ? 10 : B > 12 ? 5 : 2,
            xlabel: 'models trained',
          });
        clipPlot(ctx, P);
        [
          [gridCandidates, colors.accent],
          [randomCandidates, colors.gold],
        ].forEach(([L, cl]) => {
          ctx.strokeStyle = cl;
          ctx.lineWidth = 2.8;
          ctx.beginPath();
          let b = 0;
          ctx.moveTo(P.X(0), P.Y(0));
          for (let i = 0; i < Math.min(shown, L.length); i++) {
            ctx.lineTo(P.X(i + 1), P.Y(b));
            b = Math.max(b, sc(L[i][0], L[i][1]));
            ctx.lineTo(P.X(i + 1), P.Y(b));
          }
          ctx.stroke();
        });
        ctx.restore();
        ctx.font = MONO;
        ctx.textAlign = 'left';
        ctx.fillStyle = colors.accent;
        ctx.fillText('grid', P.m.l + 8, P.m.t + 8);
        ctx.fillStyle = colors.cut;
        ctx.fillText('random', P.m.l + 8, P.m.t + 23);
      }
      const best = (L) => {
          let b = 0;
          for (let i = 0; i < Math.min(shown, L.length); i++) b = Math.max(b, sc(L[i][0], L[i][1]));
          return b;
        },
        dist = (L) => new Set(L.slice(0, shown).map((q) => q[0].toFixed(4))).size;
      setReadout(root, 'g', f3(best(gridCandidates)));
      setReadout(root, 'r', f3(best(randomCandidates)));
      setReadout(root, 'dg', String(dist(gridCandidates)));
      setReadout(root, 'dr', String(dist(randomCandidates)));
      setReadout(root, 'win', (100 * win).toFixed(0) + '%');
    };
    let frameCount = 0,
      wait = 0;
    const anim = Anim(() => {
      frameCount++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          seed++;
          gen();
          draw();
        }
        return;
      }
      if (frameCount % (BS[settings.bi] > 30 ? 5 : 10)) return;
      if (shown < BS[settings.bi]) {
        shown++;
        draw();
      } else playback.stop();
    });
    playback = playBtn($('#hs-play', root), anim);
    attachRunLimit(playback, root, 'hs', {
      done: () => shown >= BS[settings.bi],
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
        settings.bi = v;
        re();
      },
      (v) => String(BS[v]),
    );
    bind(root, 'hs-imp', (v) => {
      settings.imp = v;
      re();
    });
    bind(root, 'hs-w', (v) => {
      settings.w = v;
      re();
    });
    $('#hs-new', root).onclick = () => {
      seed++;
      re();
    };
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
