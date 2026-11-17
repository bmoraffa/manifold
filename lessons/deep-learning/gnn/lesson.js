/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  clamp,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { uCur, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#gn-c', root),
      cr = $('#gn-r', root),
      live = $('#gn-live', root),
      S = { fix: true },
      NP = [
        [-2.6, 0.9],
        [-1.9, 1.7],
        [-1.5, 0.5],
        [-2.4, -0.4],
        [-1.3, -0.6],
        [-0.7, 1.2],
        [-0.6, 0.1],
        [-1.8, -1.5],
        [0.6, -0.2],
        [0.7, 1.1],
        [1.4, 0.3],
        [1.5, -1],
        [2.3, 1.2],
        [2.5, -0.2],
        [2.1, -1.6],
        [0.9, -1.7],
      ],
      ED = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 5],
        [2, 4],
        [2, 5],
        [2, 6],
        [3, 4],
        [3, 7],
        [4, 6],
        [4, 7],
        [5, 6],
        [6, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [9, 10],
        [9, 12],
        [10, 12],
        [10, 13],
        [11, 13],
        [11, 14],
        [11, 15],
        [13, 14],
        [12, 13],
        [14, 15],
        [5, 9],
      ],
      TR = NP.map((_, i) => (i < 8 ? 0 : 1)),
      N = 16;
    let L,
      h,
      hist,
      rd = 0,
      ch = 1,
      conv = false,
      msg = '',
      cu = 0,
      fr = 0,
      t0 = -999,
      pb,
      chk = () => false,
      prev,
      PX = [];
    const nb = NP.map(() => []);
    ED.forEach((e) => {
      nb[e[0]].push(e[1]);
      nb[e[1]].push(e[0]);
    });
    const reset = () => {
      h = L.map((v) => (v < 0 ? 0.5 : v));
      prev = h.slice();
      hist = [h.slice()];
      rd = 0;
      ch = 1;
      conv = false;
      cu = 0;
      t0 = -999;
      msg =
        L.filter((v) => v >= 0).length +
        ' people have told us their club. Everyone else starts undecided at one half. Press Pass messages or One round.';
    };
    const round = () => {
      prev = h.slice();
      const nh = h.map((v, i) => {
        let s = v;
        for (const j of nb[i]) s += h[j];
        return s / (1 + nb[i].length);
      });
      if (S.fix)
        L.forEach((v, i) => {
          if (v >= 0) nh[i] = v;
        });
      ch = Math.max(...nh.map((v, i) => Math.abs(v - h[i])));
      h = nh;
      hist.push(h.slice());
      rd++;
      t0 = fr;
      if (ch < 0.004 || rd >= 80) conv = true;
      cu = conv ? 4 : 1 + (rd % 3);
      const und = h.filter((v) => Math.abs(v - 0.5) < 0.02).length;
      msg =
        'Round ' +
        rd +
        ': every node averaged its own belief with those of its friends. The largest change was ' +
        f3(ch) +
        '. ' +
        und +
        ' node' +
        (und === 1 ? ' is' : 's are') +
        ' still undecided.' +
        (conv
          ? (rd >= 80
              ? ' Reached the limit of 80 rounds.'
              : ' Converged: the beliefs have stopped changing.') +
            (Math.max(...h) - Math.min(...h) < 0.05
              ? ' Every node now holds almost the same value: the network has over-smoothed and can no longer tell the two groups apart.'
              : '')
          : '');
    };
    const draw = () => {
      const t = T(),
        u = RM ? 1 : clamp((fr - t0) / 36, 0, 1),
        cv = (v) => mix(t.accent, t.gold, clamp(v, 0, 1));
      {
        const V = uView(c),
          { g, w, h: hh, X, Y } = V;
        PX = NP.map((p) => [X(p[0] * 1.25), Y(p[1] * 1.25)]);
        g.strokeStyle = mix(t.ink, t.ink, 0, 0.3);
        g.lineWidth = 1.4;
        ED.forEach((e) => {
          g.beginPath();
          g.moveTo(...PX[e[0]]);
          g.lineTo(...PX[e[1]]);
          g.stroke();
        });
        if (u < 1 && rd > 0)
          ED.forEach((e) => {
            [
              [e[0], e[1]],
              [e[1], e[0]],
            ].forEach(([a, b]) => {
              g.fillStyle = cv(prev[a]);
              g.strokeStyle = t.bg;
              g.lineWidth = 1;
              g.beginPath();
              g.arc(
                PX[a][0] + (PX[b][0] - PX[a][0]) * u,
                PX[a][1] + (PX[b][1] - PX[a][1]) * u,
                4.5,
                0,
                7,
              );
              g.fill();
              g.stroke();
            });
          });
        for (let i = 0; i < N; i++) {
          const v = rd > 0 ? prev[i] + (h[i] - prev[i]) * (u < 0.8 ? 0 : (u - 0.8) / 0.2) : h[i],
            [x, y] = PX[i];
          g.fillStyle =
            Math.abs(v - 0.5) < 0.02
              ? mix(t.ink, t.ink, 0, 0.25)
              : mix(v < 0.5 ? t.accent : t.gold, t.bg, 0, 0.25 + 0.75 * Math.abs(v - 0.5) * 2);
          g.strokeStyle = t.ink;
          g.lineWidth = L[i] >= 0 ? 3.4 : 1.2;
          g.beginPath();
          g.arc(x, y, 15, 0, 7);
          g.fill();
          g.stroke();
          g.font = MONO;
          g.fillStyle = t.ink;
          g.textAlign = 'center';
          g.fillText(f2(v), x, y + 29);
        }
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('thick ring: club known.  click: unknown → green → gold', 8, hh - 8);
      }
      {
        const { g, w, h: hh } = fit(cr),
          m = { l: 36, r: 12, t: 16, b: 22 },
          nR = Math.max(12, hist.length - 1),
          X2 = (q) => m.l + (q / nR) * (w - m.l - m.r),
          Y2 = (v) => hh - m.b - v * (hh - m.t - m.b);
        g.font = MONO;
        [0, 0.5, 1].forEach((v) => {
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, Y2(v));
          g.lineTo(w - m.r, Y2(v));
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText(String(v), m.l - 5, Y2(v) + 4);
        });
        for (let i = 0; i < N; i++) {
          g.strokeStyle = mix(TR[i] ? t.gold : t.accent, t.bg, 0, 0.85);
          g.lineWidth = L[i] >= 0 ? 3 : 1.5;
          g.beginPath();
          hist.forEach((r, q) => (q ? g.lineTo(X2(q), Y2(r[i])) : g.moveTo(X2(q), Y2(r[i]))));
          g.stroke();
        }
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('line colour: the node’s true club.  thick: labelled', m.l + 4, hh - 5);
      }
      live.textContent = msg;
      uCur(root, 'gn', cu);
      const ok = h.filter(
        (v, i) => Math.abs(v - 0.5) >= 0.02 && (v > 0.5 ? 1 : 0) === TR[i],
      ).length;
      setR(root, 'rd', String(rd));
      setR(root, 'ch', rd ? f3(ch) : '–');
      setR(root, 'acc', ok + ' of ' + N);
      setR(root, 'lab', String(L.filter((v) => v >= 0).length));
      setR(root, 'st', conv ? 'Converged' : rd ? 'Running' : 'Not started');
    };
    L = NP.map(() => -1);
    L[0] = 0;
    L[13] = 1;
    reset();
    if (RM) while (!conv) round();
    const anim = Anim(() => {
      fr++;
      if (fr - t0 > (rd < 4 ? 80 : 44) && !conv) {
        round();
        if (chk()) {
          draw();
          return;
        }
      }
      draw();
      if (conv && fr - t0 > 40) pb.stop();
    });
    pb = playBtn($('#gn-play', root), anim);
    chk = runCtl(pb, root, 'gn', {
      done: () => conv && fr - t0 > 40,
      restart: () => reset(),
      count: () => rd,
      what: 'rounds',
    });
    $('#gn-step', root).onclick = () => {
      pb.stop();
      if (conv) reset();
      else round();
      t0 = fr - 100;
      draw();
    };
    $('#gn-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    $('#gn-fix', root).onchange = (e) => {
      S.fix = e.target.checked;
      pb.stop();
      reset();
      draw();
      pb.start();
    };
    $('#gn-clear', root).onclick = () => {
      pb.stop();
      L = NP.map(() => -1);
      reset();
      msg = 'No labels. Click a node to tell the network which club that person joined.';
      draw();
    };
    c.addEventListener('click', (e) => {
      const r = c.getBoundingClientRect(),
        x = e.clientX - r.left,
        y = e.clientY - r.top;
      PX.forEach((p, i) => {
        if (Math.hypot(p[0] - x, p[1] - y) < 18) {
          pb.stop();
          L[i] = L[i] < 0 ? 0 : L[i] === 0 ? 1 : -1;
          reset();
          draw();
          pb.start();
        }
      });
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        pb.start();
      },
    };
  },
};
