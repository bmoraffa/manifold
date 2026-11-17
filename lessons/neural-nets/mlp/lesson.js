/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  f2,
  f3,
  fit,
  hex,
  markTP,
  mix,
  playBtn,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { NNet, drawEdges, netPos, nnData } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#nn-c', root),
      cr = $('#nn-r', root),
      live = $('#nn-live', root),
      S = { hl: 2, hw: 5, act: 'tanh', lr: 0.015, ds: 'circles', nz: 0.2, n: 200, seed: 3 };
    let pts = [],
      net,
      ep = 0,
      LT = [],
      LE = [],
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      conv = false,
      run = 1;
    const tr = () => pts.filter((p) => !p.test),
      te = () => pts.filter((p) => p.test),
      ll = (P) =>
        P.length
          ? -P.reduce((s, p) => {
              const o = net.p([p.x, p.y]);
              return s + (p.c ? Math.log(o + 1e-9) : Math.log(1 - o + 1e-9));
            }, 0) / P.length
          : NaN,
      ac = (P) =>
        P.length
          ? P.filter((p) => (net.p([p.x, p.y]) >= 0.5 ? 1 : 0) === p.c).length / P.length
          : NaN;
    const build = () => {
      net = NNet([2, ...Array(S.hl).fill(S.hw), 1], { act: S.act, seed: S.seed * 7 + run * 13 });
      ep = 0;
      LT = [];
      LE = [];
      conv = false;
      cu = 0;
      msg =
        'Random weights: the network has not seen any data yet, and its answer is a smear. Press Train or One epoch.';
    };
    const epoch = () => {
      const T = tr(),
        l = net.train(
          T.map((p) => [p.x, p.y]),
          T.map((p) => p.c),
          S.lr,
        ).loss;
      ep++;
      LT.push(l);
      LE.push(ll(te()));
      cu = 3;
      if (
        (ep > 120 && Math.abs(LT[ep - 1] - LT[ep - 61]) < 4e-3 * LT[ep - 1]) ||
        l < 0.012 ||
        ep >= 4000
      )
        conv = true;
      msg =
        'Epoch ' +
        ep +
        ': all ' +
        T.length +
        ' training points went forward through the network, the loss came out at ' +
        f3(l) +
        (ep > 1 ? ' (it was ' + f3(LT[ep - 2]) + ')' : '') +
        ', and every weight was nudged to lower it. Training accuracy ' +
        pct(ac(T)) +
        ', test accuracy ' +
        pct(ac(te())) +
        '.' +
        (conv
          ? ep >= 4000
            ? ' Reached the limit of 4000 epochs.'
            : ' Converged: the loss has stopped falling, so training stops here.'
          : '');
      if (conv) cu = 4;
    };
    const tp = testPoint(
      root,
      'nn',
      ['Your x', 'Your y'],
      [0, 0],
      [
        [-3, 3],
        [-3, 3],
      ],
    );
    const draw = () => {
      const t = T(),
        ca = hex(t.accent),
        cg = hex(t.gold);
      {
        const { g, w, h } = fit(c),
          cs = 8,
          X = (x) => w / 2 + (x * w) / 6,
          Y = (y) => h / 2 - (y * h) / 6;
        for (let py = 0; py < h; py += cs)
          for (let px = 0; px < w; px += cs) {
            const p = net.p([((px + cs / 2 - w / 2) * 6) / w, (-(py + cs / 2 - h / 2) * 6) / h]),
              q = Math.abs(p - 0.5) * 2;
            g.fillStyle = mix(p < 0.5 ? t.accent : t.gold, t.bg, 0, 0.1 + 0.42 * q);
            g.fillRect(px, py, cs, cs);
          }
        for (const p of pts) {
          const cl = p.c ? t.gold : t.accent;
          g.beginPath();
          g.arc(X(p.x), Y(p.y), p.test ? 3.4 : 3.6, 0, 7);
          if (p.test) {
            g.fillStyle = t.bg;
            g.fill();
            g.strokeStyle = cl;
            g.lineWidth = 1.8;
            g.stroke();
          } else {
            g.fillStyle = cl;
            g.fill();
            g.strokeStyle = t.bg;
            g.lineWidth = 0.8;
            g.stroke();
          }
        }
        markTP(g, X(tp.v[0]), Y(tp.v[1]), t);
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.62,
          pos = netPos(net.sizes, 34, 8, w - 68, h1 - 16),
          NC = 9,
          act = [];
        drawEdges(g, net.W, pos, t, pb && pb.on ? fr / 50 : null);
        for (let a = 0; a < NC; a++)
          for (let b = 0; b < NC; b++)
            act.push(net.fwd([-3 + (6 * (b + 0.5)) / NC, 3 - (6 * (a + 0.5)) / NC]).A);
        pos.forEach((col, l) =>
          col.forEach((q, j) => {
            const sz = l === 0 ? 22 : 30,
              x0 = q[0] - sz / 2,
              y0 = q[1] - sz / 2,
              cz = sz / NC;
            let lo = 1e9,
              hi = -1e9;
            if (l > 0 && l < net.L)
              for (const A of act) {
                lo = Math.min(lo, A[l][j]);
                hi = Math.max(hi, A[l][j]);
              }
            for (let k = 0; k < NC * NC; k++) {
              let v =
                l === 0
                  ? j === 0
                    ? ((k % NC) + 0.5) / NC
                    : 1 - (Math.floor(k / NC) + 0.5) / NC
                  : l === net.L
                    ? act[k][l][j]
                    : (act[k][l][j] - lo) / (hi - lo || 1);
              g.fillStyle =
                'rgb(' + ca.map((u, i) => Math.round(u + (cg[i] - u) * v)).join(',') + ')';
              g.fillRect(x0 + (k % NC) * cz, y0 + Math.floor(k / NC) * cz, cz + 0.6, cz + 0.6);
            }
            g.strokeStyle = t.ink;
            g.lineWidth = 1.3;
            g.strokeRect(x0, y0, sz, sz);
          }),
        );
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('x, y', pos[0][0][0], h1 - 1);
        g.fillText('answer', pos[net.L][0][0], h1 - 1);
        if (net.L > 1) g.fillText('hidden layers', w / 2, h1 - 1);
        const ym = Math.max(0.72, ...LT.slice(0, 3), ...LE.slice(0, 3)),
          U = uLine(g, t, 0, h1, w, h - h1, LT, {
            title: 'loss: training (green) and test (red)',
            n: Math.max(60, LT.length),
            ym,
            col: t.accent,
          });
        if (LE.length) {
          g.strokeStyle = t.ok;
          g.lineWidth = 2;
          g.beginPath();
          LE.forEach((v, i) =>
            i ? g.lineTo(U.X(i), U.Y(Math.min(v, ym))) : g.moveTo(U.X(i), U.Y(Math.min(v, ym))),
          );
          g.stroke();
        }
      }
      live.textContent = msg;
      uCur(root, 'nn', cu);
      const o = net.p(tp.v);
      tp.out.textContent =
        'At (' +
        f2(tp.v[0]) +
        ', ' +
        f2(tp.v[1]) +
        ') the network outputs ' +
        f3(o) +
        ', so it says ' +
        (o >= 0.5 ? 'gold' : 'green') +
        ' with ' +
        (100 * Math.max(o, 1 - o)).toFixed(1) +
        '% confidence.';
      let np = 0;
      net.W.forEach((wl, l) => {
        np += wl.length * (wl[0].length + 1);
      });
      setR(root, 'ep', String(ep));
      setR(root, 'lt', LT.length ? f3(LT[ep - 1]) : '–');
      setR(root, 'le', LE.length ? f3(LE[ep - 1]) : '–');
      setR(root, 'at', pct(ac(tr())));
      setR(root, 'ae', pct(ac(te())));
      setR(root, 'np', String(np));
      setR(root, 'st', conv ? 'Converged' : ep ? 'Training' : 'Not started');
    };
    tp.on = () => draw();
    const gen = () => {
        pts = nnData(S.ds, S.n, S.nz, S.seed);
      },
      redo = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
    gen();
    build();
    if (RM) for (let i = 0; i < 4000 && !conv; i++) epoch();
    const anim = Anim(() => {
      fr++;
      if (fr % (ep < 20 ? 8 : ep < 90 ? 3 : 1)) {
        draw();
        return;
      }
      for (let i = 0; i < (ep < 400 ? 1 : 3) && !conv && chk.remaining() > 0; i++) epoch();
      draw();
      chk();
    });
    pb = playBtn($('#nn-play', root), anim);
    chk = runCtl(pb, root, 'nn', {
      done: () => conv,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'epochs',
    });
    $('#nn-step', root).onclick = () => {
      pb.stop();
      if (conv) {
        run++;
        build();
      } else epoch();
      draw();
    };
    $('#nn-reset', root).onclick = redo;
    bind(
      root,
      'nn-hl',
      (v) => {
        S.hl = v;
        redo();
      },
      (v) => String(v),
    );
    bind(
      root,
      'nn-hw',
      (v) => {
        S.hw = v;
        redo();
      },
      (v) => String(v),
    );
    $('#nn-act', root).onchange = (e) => {
      S.act = e.target.value;
      redo();
    };
    bind(
      root,
      'nn-lr',
      (v) => {
        S.lr = v;
      },
      f3,
    );
    $('#nn-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      gen();
      redo();
    };
    bind(root, 'nn-nz', (v) => {
      S.nz = v;
      gen();
      redo();
    });
    bind(
      root,
      'nn-n',
      (v) => {
        S.n = v;
        gen();
        redo();
      },
      (v) => String(v),
    );
    $('#nn-new', root).onclick = () => {
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
