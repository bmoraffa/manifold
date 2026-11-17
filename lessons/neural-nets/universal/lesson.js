/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  clipPlot,
  f2,
  fit,
  markTP,
  mix,
  playBtn,
  plot,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { ACT, NNet, sci } from '../shared.js';
import { UC, uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#ua-c', root),
      cr = $('#ua-r', root),
      live = $('#ua-live', root),
      st = $('#ua-st', root),
      S = { H: 5, act: 'tanh', tg: 'wave' },
      NP = 100,
      TG = {
        wave: (x) => 0.8 * Math.sin(4.2 * x) + 0.3 * x,
        hill: (x) => 1.3 * Math.exp(-7 * x * x) - 0.5,
        step: (x) => (x < -0.5 ? -0.8 : x < 0 ? -0.1 : x < 0.5 ? 0.7 : 0.1),
        sky: (x) =>
          [0.2, 0.9, 0.4, -0.3, 0.6, -0.6, 0.1, 0.8, -0.1, 0.5][
            Math.min(9, Math.floor((x + 1) * 5))
          ],
      };
    let Tg = new Float64Array(NP),
      net,
      ep = 0,
      LL = [],
      conv = false,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      grow = false,
      best = {},
      painting = false,
      lastI = -1;
    const xi = (i) => -1 + (2 * i) / (NP - 1),
      setT = () => {
        if (S.tg !== 'own') for (let i = 0; i < NP; i++) Tg[i] = TG[S.tg](xi(i));
      };
    const build = () => {
      net = NNet([1, S.H, 1], {
        act: S.act,
        out: 'lin',
        seed: run * 31 + S.H * 7,
        scale: S.act === 'relu' ? 1.2 : 2.2,
      });
      for (let j = 0; j < S.H; j++) net.B[0][j] = (Math.random() - 0.5) * 3;
      ep = 0;
      LL = [];
      conv = false;
      cu = 0;
      msg =
        S.H +
        ' neuron' +
        (S.H === 1 ? '' : 's') +
        ' with random settings: the pieces are scattered and their sum looks nothing like the target. Press Train.';
    };
    const X = Array.from({ length: NP }, (_, i) => [xi(i) * 3]);
    const train = (k) => {
      for (let q = 0; q < k && !conv; q++) {
        const l = 2 * net.train(X, Tg, 0.02).loss;
        ep++;
        LL.push(l);
        if (
          (ep > 300 && Math.abs(LL[ep - 1] - LL[ep - 201]) < 0.01 * LL[ep - 1] + 1e-7) ||
          ep >= 6000
        )
          conv = true;
      }
      const l = LL[ep - 1];
      best[S.H] = Math.min(best[S.H] == null ? 1e9 : best[S.H], l);
      cu = conv ? 4 : 3;
      msg =
        'Step ' +
        ep +
        ': the ' +
        S.H +
        ' piece' +
        (S.H === 1 ? '' : 's') +
        ' shifted a little. Mean squared error ' +
        sci(l) +
        '.' +
        (conv
          ? (ep >= 6000
              ? ' Reached the limit of 6000 steps.'
              : ' Converged: the error has stopped falling with ' +
                S.H +
                ' neuron' +
                (S.H === 1 ? '' : 's') +
                '.') +
            (l > 0.004
              ? ' The fit is still rough: there are not enough pieces for this curve.'
              : ' The gold line follows the target closely.')
          : '');
    };
    const draw = () => {
      const t = T(),
        col = UC(t);
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, [-1, 1], [-1.6, 1.6], t, {
            xstep: 0.5,
            yticks: [-1, 0, 1],
            xlabel: 'x',
          });
        clipPlot(g, P);
        const f = ACT[S.act][0];
        for (let j = 0; j < S.H; j++) {
          g.strokeStyle = mix(col[j % 8], col[j % 8], 0, 0.75);
          g.lineWidth = 1.3;
          g.beginPath();
          for (let i = 0; i < NP; i++) {
            const v = net.W[1][0][j] * f(net.W[0][j][0] * xi(i) * 3 + net.B[0][j]);
            i ? g.lineTo(P.X(xi(i)), P.Y(v)) : g.moveTo(P.X(xi(i)), P.Y(v));
          }
          g.stroke();
        }
        g.setLineDash([7, 5]);
        g.strokeStyle = t.ink;
        g.lineWidth = 2;
        g.beginPath();
        for (let i = 0; i < NP; i++)
          i ? g.lineTo(P.X(xi(i)), P.Y(Tg[i])) : g.moveTo(P.X(xi(i)), P.Y(Tg[i]));
        g.stroke();
        g.setLineDash([]);
        g.strokeStyle = t.gold;
        g.lineWidth = 3.4;
        g.beginPath();
        for (let i = 0; i < NP; i++) {
          const v = net.p(X[i]);
          i ? g.lineTo(P.X(xi(i)), P.Y(v)) : g.moveTo(P.X(xi(i)), P.Y(v));
        }
        g.stroke();
        {
          const x = tp.v[0],
            v = net.p([x * 3]);
          markTP(g, P.X(x), P.Y(v), t);
        }
        g.restore();
        c._P = P;
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5;
        const lo = LL.length ? Math.min(...LL) : 0;
        uLine(
          g,
          t,
          0,
          0,
          w,
          h1,
          LL.map((v) => Math.log10(v + 1e-9)),
          {
            title: 'error by training step (log scale): 10^…',
            n: Math.max(100, LL.length),
            y1: Math.min(-3, Math.floor(Math.log10(lo + 1e-9))),
            ym: 0.5,
            col: t.ok,
          },
        );
        const m = { l: 44, r: 12, t: 22, b: 22 },
          bw = (w - m.l - m.r) / 24,
          Y = (v) =>
            h - m.b - ((clamp(Math.log10(v + 1e-9), -4, 0) + 4) / 4) * (h - h1 - m.t - m.b);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('best error reached with each number of neurons', 8, h1 + 13);
        g.textAlign = 'right';
        g.fillText('1', m.l - 5, Y(1) + 4);
        g.fillText('10^-4', m.l - 5, Y(1e-4) + 4);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h - m.b);
        g.lineTo(w - m.r, h - m.b);
        g.stroke();
        for (let k = 1; k <= 24; k++) {
          const x = m.l + (k - 1) * bw + bw * 0.15;
          if (best[k] != null) {
            g.fillStyle = k === S.H ? t.gold : t.accent;
            g.fillRect(x, Y(best[k]), bw * 0.7, h - m.b - Y(best[k]));
          }
          if (k % 2 === 1) {
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(String(k), x + bw * 0.35, h - m.b + 14);
          }
        }
      }
      live.textContent = msg;
      uCur(root, 'ua', cu);
      const x = tp.v[0],
        gi = Math.round(((x + 1) / 2) * (NP - 1));
      tp.out.textContent =
        'At x = ' +
        f2(x) +
        ' the network outputs ' +
        f2(net.p([x * 3])) +
        ' and the target is ' +
        f2(Tg[gi]) +
        '.';
      setR(root, 'H', String(S.H));
      setR(root, 'np', String(3 * S.H + 1));
      setR(root, 'ep', String(ep));
      setR(root, 'L', LL.length ? sci(LL[ep - 1]) : '–');
      setR(
        root,
        'st',
        grow ? 'Adding neurons' : conv ? 'Converged' : ep ? 'Training' : 'Not started',
      );
    };
    const tp = testPoint(root, 'ua', ['Your x'], [0.3], [[-1, 1]]);
    tp.on = () => draw();
    setT();
    build();
    if (RM) train(6000);
    const anim = Anim(() => {
      fr++;
      if (ep < 40 && fr % 4) return;
      train(Math.min(ep < 40 ? 2 : ep < 400 ? 4 : 10, chk.remaining()));
      if (conv && grow) {
        if (S.H < 12 && LL[ep - 1] > 3e-4) {
          S.H++;
          hb.set(S.H);
          run++;
          chk.rebase(0);
          build();
          msg = 'Added a neuron: now ' + S.H + '. Training again from a fresh random start.';
        } else {
          grow = false;
          msg += ' Stopped adding neurons at ' + S.H + '.';
        }
      }
      draw();
      chk();
    });
    pb = playBtn($('#ua-play', root), anim);
    chk = runCtl(pb, root, 'ua', {
      done: () => conv && !grow,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'steps',
    });
    const redo = () => {
      pb.stop();
      grow = false;
      run++;
      build();
      draw();
    };
    $('#ua-step', root).onclick = () => {
      pb.stop();
      grow = false;
      if (conv) {
        run++;
        build();
      } else train(20);
      draw();
    };
    $('#ua-reset', root).onclick = redo;
    $('#ua-grow', root).onclick = () => {
      pb.stop();
      best = {};
      S.H = 1;
      hb.set(1);
      run++;
      build();
      grow = true;
      pb.start();
    };
    const hb = bind(
      root,
      'ua-H',
      (v) => {
        S.H = v;
        redo();
      },
      (v) => String(v),
    );
    $('#ua-act', root).onchange = (e) => {
      S.act = e.target.value;
      best = {};
      redo();
    };
    $('#ua-tg', root).onchange = (e) => {
      S.tg = e.target.value;
      best = {};
      if (S.tg === 'own') st.textContent = 'Drag across the left picture to draw the curve.';
      else st.textContent = '';
      setT();
      redo();
    };
    const paint = (e) => {
      const P = c._P;
      if (!P) return;
      const r = c.getBoundingClientRect(),
        x = clamp(((e.clientX - r.left - P.m.l) / (P.w - P.m.l - P.m.r)) * 2 - 1, -1, 1),
        y = clamp(
          -((e.clientY - r.top - (P.h - P.m.b)) / (P.h - P.m.t - P.m.b)) * 3.2 - 1.6,
          -1.5,
          1.5,
        ),
        i = Math.round(((x + 1) / 2) * (NP - 1));
      if (lastI >= 0 && lastI !== i) {
        const a = Math.min(i, lastI),
          b = Math.max(i, lastI),
          ya = Tg[lastI];
        for (let q = a; q <= b; q++)
          Tg[q] = ya + (y - ya) * (i === lastI ? 1 : Math.abs(q - lastI) / Math.abs(i - lastI));
      } else Tg[i] = y;
      lastI = i;
    };
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', (e) => {
      pb.stop();
      grow = false;
      painting = true;
      lastI = -1;
      S.tg = 'own';
      $('#ua-tg', root).value = 'own';
      st.textContent = 'Your own curve. Press Train, or Add Neurons One at a Time.';
      c.setPointerCapture(e.pointerId);
      paint(e);
      draw();
    });
    c.addEventListener('pointermove', (e) => {
      if (!painting) return;
      paint(e);
      draw();
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      c.addEventListener(n, () => {
        if (painting) {
          painting = false;
          best = {};
          run++;
          build();
          draw();
        }
      }),
    );
    return {
      draw,
      stop: () => {
        pb.stop();
        grow = false;
      },
      auto: () => {
        run++;
        build();
        draw();
        pb.start();
      },
    };
  },
};
