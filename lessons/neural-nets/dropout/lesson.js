/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  SANS,
  T,
  bind,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { NNet, drawEdges, netPos, nnData } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#dr-c', root),
      cr = $('#dr-r', root),
      live = $('#dr-live', root),
      S = { p: 0.5, hw: 24, n: 60, fl: 0.12, seed: 2 },
      EPM = 500;
    let tr = [],
      te = [],
      N0,
      N1,
      ep = 0,
      H = [[], [], [], []],
      mask = null,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1;
    const gen = () => {
      const a = nnData('moons', Math.round((S.n * 4) / 3) + 2, 0.28, S.seed)
          .filter((p) => !p.test)
          .slice(0, S.n),
        r = rng(S.seed * 5 + 1);
      tr = a.map((p) => ({ ...p, bad: r() < S.fl }));
      tr.forEach((p) => {
        if (p.bad) p.c = 1 - p.c;
      });
      te = nnData('moons', 400, 0.28, S.seed + 77);
    };
    const build = () => {
      const sz = [2, S.hw, S.hw, 1],
        sd = S.seed * 3 + run * 17;
      N0 = NNet(sz, { act: 'relu', seed: sd });
      N1 = NNet(sz, { act: 'relu', seed: sd });
      ep = 0;
      H = [[], [], [], []];
      mask = null;
      cu = 0;
      msg =
        'Two networks with the same ' +
        (2 * S.hw + S.hw * S.hw + S.hw + 2 * S.hw + 1) +
        ' starting weights and the same ' +
        tr.length +
        ' training points, ' +
        tr.filter((p) => p.bad).length +
        ' of which carry the wrong label. Press Train both or One epoch.';
    };
    const ll = (N, P) =>
        -P.reduce((s, p) => {
          const o = N.p([p.x, p.y]);
          return s + (p.c ? Math.log(o + 1e-9) : Math.log(1 - o + 1e-9));
        }, 0) / P.length,
      ac = (N, P) => P.filter((p) => (N.p([p.x, p.y]) >= 0.5 ? 1 : 0) === p.c).length / P.length;
    const epoch = () => {
      const X = tr.map((p) => [p.x, p.y]),
        Y = tr.map((p) => p.c),
        a = N0.train(X, Y, 0.02),
        b = N1.train(X, Y, 0.02, { drop: S.p });
      mask = b.mask;
      ep++;
      H[0].push(a.loss);
      H[1].push(b.loss);
      H[2].push(ll(N0, te));
      H[3].push(ll(N1, te));
      cu = ep < 3 ? 2 : 3;
      const off = mask ? mask.reduce((s, m) => s + m.filter((v) => v === 0).length, 0) : 0;
      msg =
        'Epoch ' +
        ep +
        ': on the last example ' +
        off +
        ' of the ' +
        2 * S.hw +
        ' hidden neurons in the dropout network were switched off. Loss on new data: normal ' +
        f3(H[2][ep - 1]) +
        ', dropout ' +
        f3(H[3][ep - 1]) +
        '.' +
        (ep >= EPM
          ? ' Both have had their ' +
            EPM +
            ' epochs, so training stops. ' +
            (H[3][ep - 1] < H[2][ep - 1]
              ? 'Dropout ends with the lower error on new data.'
              : 'Here dropout did not help: there was little overfitting to cure, or p is too large.')
          : '');
      if (ep >= EPM) cu = 4;
    };
    const draw = () => {
      const t = T();
      {
        const { g, w, h } = fit(c),
          hw = w / 2,
          cs = 7;
        [N0, N1].forEach((N, q) => {
          const x0 = q * hw,
            X = (x) => x0 + hw / 2 + (x * hw) / 6.4,
            Y = (y) => h / 2 - (y * h) / 6.4;
          g.save();
          g.beginPath();
          g.rect(x0, 0, hw, h);
          g.clip();
          for (let py = 0; py < h; py += cs)
            for (let px = 0; px < hw; px += cs) {
              const p = N.p([
                  ((px + cs / 2 - hw / 2) * 6.4) / hw,
                  (-(py + cs / 2 - h / 2) * 6.4) / h,
                ]),
                s = Math.abs(p - 0.5) * 2;
              g.fillStyle = mix(p < 0.5 ? t.accent : t.gold, t.bg, 0, 0.12 + 0.42 * s);
              g.fillRect(x0 + px, py, cs, cs);
            }
          for (const p of tr) {
            g.beginPath();
            g.arc(X(p.x), Y(p.y), 3.6, 0, 7);
            g.fillStyle = p.c ? t.gold : t.accent;
            g.fill();
            g.strokeStyle = p.bad ? t.ink : t.bg;
            g.lineWidth = p.bad ? 2 : 0.8;
            g.stroke();
          }
          g.restore();
          g.font = SANS;
          g.fillStyle = t.ink;
          g.textAlign = 'left';
          g.fillText(q ? 'With dropout, p = ' + f2(S.p) : 'Without dropout', x0 + 10, 18);
        });
        g.strokeStyle = t.ink;
        g.lineWidth = 1.5;
        g.beginPath();
        g.moveTo(hw, 0);
        g.lineTo(hw, h);
        g.stroke();
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('black ring = wrong label', w - 8, h - 8);
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5,
          sh = Math.min(10, S.hw),
          pos = netPos([2, sh, sh, 1], 30, 4, w - 60, h1 - 20);
        drawEdges(
          g,
          [
            N1.W[0].slice(0, sh),
            N1.W[1].slice(0, sh).map((q) => q.slice(0, sh)),
            [N1.W[2][0].slice(0, sh)],
          ],
          pos,
          t,
          pb && pb.on ? fr / 45 : null,
        );
        pos.forEach((col, l) =>
          col.forEach((q, j) => {
            const off = mask && l > 0 && l < 3 && mask[l - 1][j] === 0;
            g.fillStyle = off ? t.bg : t.ink;
            g.strokeStyle = off ? t.muted : t.ink;
            g.lineWidth = 1.4;
            g.beginPath();
            g.arc(q[0], q[1], 6.5, 0, 7);
            g.fill();
            g.stroke();
            if (off) {
              g.strokeStyle = t.ok;
              g.lineWidth = 2;
              g.beginPath();
              g.moveTo(q[0] - 6, q[1] - 6);
              g.lineTo(q[0] + 6, q[1] + 6);
              g.moveTo(q[0] + 6, q[1] - 6);
              g.lineTo(q[0] - 6, q[1] + 6);
              g.stroke();
            }
          }),
        );
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText(
          S.hw > sh
            ? 'first ' + sh + ' of ' + S.hw + ' neurons in each hidden layer'
            : 'the dropout network',
          w / 2,
          h1 + 2,
        );
        const ym = 1.6,
          U = uLine(
            g,
            t,
            0,
            h1 + 8,
            w,
            h - h1 - 8,
            H[2].map((v) => Math.min(v, ym)),
            {
              title:
                'loss on new data: normal (red), dropout (green). Dashed: on the training points',
              n: EPM,
              ym,
              col: t.ok,
            },
          );
        g.strokeStyle = t.accent;
        g.lineWidth = 2.4;
        g.beginPath();
        H[3].forEach((v, i) =>
          i ? g.lineTo(U.X(i), U.Y(Math.min(v, ym))) : g.moveTo(U.X(i), U.Y(Math.min(v, ym))),
        );
        g.stroke();
        [
          [H[0], t.ok],
          [H[1], t.accent],
        ].forEach((q) => {
          g.strokeStyle = mix(q[1], q[1], 0, 0.4);
          g.lineWidth = 1;
          g.setLineDash([3, 3]);
          g.beginPath();
          q[0].forEach((v, i) =>
            i ? g.lineTo(U.X(i), U.Y(Math.min(v, ym))) : g.moveTo(U.X(i), U.Y(Math.min(v, ym))),
          );
          g.stroke();
          g.setLineDash([]);
        });
      }
      live.textContent = msg;
      uCur(root, 'dr', cu);
      setR(root, 'ep', ep + ' of ' + EPM);
      setR(root, 'a0', pct(ac(N0, tr)) + ' / ' + pct(ac(N0, te)));
      setR(root, 'a1', pct(ac(N1, tr)) + ' / ' + pct(ac(N1, te)));
      setR(root, 'l0', ep ? f3(H[2][ep - 1]) : '–');
      setR(root, 'l1', ep ? f3(H[3][ep - 1]) : '–');
    };
    gen();
    build();
    if (RM) for (let i = 0; i < EPM; i++) epoch();
    const redo = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      fr++;
      if (ep < 12 && fr % 8) return;
      for (let i = 0; i < (ep < 12 ? 1 : 2) && ep < EPM && chk.remaining() > 0; i++) epoch();
      draw();
      chk();
    });
    pb = playBtn($('#dr-play', root), anim);
    chk = runCtl(pb, root, 'dr', {
      done: () => ep >= EPM,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'epochs',
    });
    $('#dr-step', root).onclick = () => {
      pb.stop();
      if (ep >= EPM) {
        run++;
        build();
      } else epoch();
      draw();
    };
    $('#dr-reset', root).onclick = redo;
    bind(root, 'dr-p', (v) => {
      S.p = v;
      redo();
    });
    bind(
      root,
      'dr-hw',
      (v) => {
        S.hw = v;
        redo();
      },
      (v) => String(v),
    );
    bind(
      root,
      'dr-n',
      (v) => {
        S.n = v;
        gen();
        redo();
      },
      (v) => String(v),
    );
    bind(root, 'dr-fl', (v) => {
      S.fl = v;
      gen();
      redo();
    });
    $('#dr-new', root).onclick = () => {
      S.seed++;
      gen();
      redo();
    };
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        run++;
        build();
        draw();
        pb.start();
      },
    };
  },
};
