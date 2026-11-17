/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  f3,
  fit,
  letterData,
  markTP,
  mix,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { NNet } from '../../neural-nets/shared.js';
import { UC, uCur, uLine } from '../../unsupervised/shared.js';
import { pct } from '../../ensembles/shared.js';
import { sig } from '../../classification/shared.js';

export default {
  init(root) {
    const c = $('#ae-c', root),
      cr = $('#ae-r', root),
      live = $('#ae-live', root),
      S = { hw: 14, flip: 0.04, col: true, seed: 1 };
    let X = [],
      lab = [],
      LT = [],
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
      pr = [0.2, 0.2],
      drag = false,
      codes = [];
    const gen = () => {
        const d = letterData(S.seed, S.flip);
        X = d.X.map((x) => Float64Array.from(x, (v) => clamp(Math.round(v), 0, 1)));
        lab = d.lab;
        LT = d.CH;
      },
      build = () => {
        net = NNet([35, S.hw, 2, S.hw, 35], { act: 'tanh', out: 'sig', seed: run * 9 + S.seed });
        ep = 0;
        LL = [];
        conv = false;
        cu = 0;
        enc();
        msg =
          'Random weights: every picture lands near the middle of the map and comes back as a grey blur. Press Train.';
      };
    const enc = () => {
        codes = X.map((x) => net.fwd(x).A[2]);
      },
      dec = (z) => {
        let a = z;
        for (let l = 2; l < 4; l++) {
          const m = net.sizes[l + 1],
            o = new Float64Array(m);
          for (let j = 0; j < m; j++) {
            let s = net.B[l][j];
            for (let i = 0; i < a.length; i++) s += net.W[l][j][i] * a[i];
            o[j] = l === 3 ? sig(s) : Math.tanh(s);
          }
          a = o;
        }
        return a;
      };
    const acc = () => {
      let ok = 0;
      X.forEach((x) => {
        const o = net.p2(x);
        for (let k = 0; k < 35; k++) if ((o[k] >= 0.5 ? 1 : 0) === x[k]) ok++;
      });
      return ok / (X.length * 35);
    };
    const train = (k) => {
      for (let i = 0; i < k && !conv; i++) {
        const l = net.train(X, X, 0.01).loss / 35;
        ep++;
        LL.push(l);
        if ((ep > 250 && Math.abs(LL[ep - 1] - LL[ep - 151]) < 0.012 * l) || ep >= 3000)
          conv = true;
      }
      enc();
      cu = conv ? 4 : ep < 4 ? 1 : 3;
      msg =
        'Epoch ' +
        ep +
        ': all ' +
        X.length +
        ' pictures were squeezed to two numbers and rebuilt. Error per pixel ' +
        f3(LL[ep - 1]) +
        '; ' +
        pct(acc()) +
        ' of pixels come back right.' +
        (conv
          ? ep >= 3000
            ? ' Reached the limit of 3000 epochs.'
            : ' Converged: the error has stopped falling, so training stops.'
          : '');
    };
    const glyph = (g, v, x, y, cz, t, al = 1) => {
      for (let k = 0; k < 35; k++) {
        g.fillStyle = mix(t.bg, t.ink, clamp(v[k], 0, 1), al);
        g.fillRect(x + (k % 5) * cz, y + Math.floor(k / 5) * cz, cz - 0.4, cz - 0.4);
      }
    };
    const draw = () => {
      const t = T(),
        col = UC(t);
      {
        const { g, w, h } = fit(c),
          X2 = (v) => w / 2 + (v * (w / 2 - 14)) / 1.08,
          Y2 = (v) => h / 2 - (v * (h / 2 - 14)) / 1.08,
          NGd = 8;
        for (let a = 0; a < NGd; a++)
          for (let b = 0; b < NGd; b++) {
            const zx = -1 + (2 * (b + 0.5)) / NGd,
              zy = 1 - (2 * (a + 0.5)) / NGd;
            glyph(g, dec([zx, zy]), X2(zx) - 10, Y2(zy) - 14, 4, t, 0.22);
          }
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.strokeRect(X2(-1), Y2(1), X2(1) - X2(-1), Y2(-1) - Y2(1));
        codes.forEach((z, i) => {
          const x = X2(z[0]),
            y = Y2(z[1]);
          if (S.col) {
            g.font = '600 13px "Hanken Grotesk",sans-serif';
            g.fillStyle = col[lab[i] % 8];
            g.textAlign = 'center';
            g.fillText(LT[lab[i]], x, y + 4);
          } else {
            g.fillStyle = mix(t.ink, t.ink, 0, 0.6);
            g.beginPath();
            g.arc(x, y, 3, 0, 7);
            g.fill();
          }
        });
        markTP(g, X2(pr[0]), Y2(pr[1]), t, false);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('first number of the code →', X2(-1) + 4, Y2(-1) - 5);
        c._g = { X2, Y2, w, h };
      }
      {
        const { g, w, h } = fit(cr),
          cz = Math.min((w - 40) / 6 / 6, (h * 0.21) / 7.4),
          gw = 5 * cz,
          x0 = (w - 6 * gw * 1.25) / 2;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('in', 6, 16 + 3.5 * cz);
        g.fillText('out', 6, 28 + 10.5 * cz);
        for (let k = 0; k < 6; k++) {
          const i = k * 25,
            x = x0 + k * gw * 1.25;
          glyph(g, X[i], x, 10, cz, t);
          glyph(g, net.p2(X[i]), x, 22 + 7 * cz, cz, t);
        }
        const y3 = 38 + 14 * cz,
          pz = Math.min(9, (h - y3 - 8) / 7.2);
        glyph(g, dec(pr), 14, y3 + 10, pz, t);
        g.fillStyle = t.muted;
        g.fillText('the diamond', 14, y3 + 2);
        uLine(g, t, 5 * pz + 50, y3 - 4, w - 5 * pz - 54, h - y3 + 4, LL, {
          title: 'error per pixel',
          n: Math.max(100, LL.length),
          col: t.ok,
          ym: Math.max(0.7, LL[0] || 0.7),
        });
      }
      live.textContent = msg;
      uCur(root, 'ae', cu);
      setR(root, 'ep', String(ep));
      setR(root, 'L', ep ? f3(LL[ep - 1]) : '–');
      setR(root, 'px', pct(acc()));
      setR(root, 'cmp', '35 → 2');
      setR(root, 'st', conv ? 'Converged' : ep ? 'Training' : 'Not started');
    };
    gen();
    build();
    if (RM) train(3000);
    const redo = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      fr++;
      if (ep < 30 && fr % 4) return;
      train(Math.min(ep < 30 ? 1 : ep < 300 ? 3 : 6, chk.remaining()));
      draw();
      chk();
    });
    pb = playBtn($('#ae-play', root), anim);
    chk = runCtl(pb, root, 'ae', {
      done: () => conv,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'epochs',
    });
    $('#ae-step', root).onclick = () => {
      pb.stop();
      if (conv) {
        run++;
        build();
      } else train(10);
      draw();
    };
    $('#ae-reset', root).onclick = redo;
    bind(
      root,
      'ae-hw',
      (v) => {
        S.hw = v;
        redo();
      },
      (v) => String(v),
    );
    bind(root, 'ae-flip', (v) => {
      S.flip = v;
      gen();
      redo();
    });
    $('#ae-col', root).onchange = (e) => {
      S.col = e.target.checked;
      draw();
    };
    $('#ae-new', root).onclick = () => {
      S.seed++;
      gen();
      redo();
    };
    const mv = (e) => {
      const q = c._g;
      if (!q) return;
      const r = c.getBoundingClientRect();
      pr = [
        clamp((e.clientX - r.left - q.w / 2) / ((q.w / 2 - 14) / 1.08), -1, 1),
        clamp(-(e.clientY - r.top - q.h / 2) / ((q.h / 2 - 14) / 1.08), -1, 1),
      ];
      if (!pb.on) draw();
    };
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', (e) => {
      drag = true;
      c.setPointerCapture(e.pointerId);
      mv(e);
    });
    c.addEventListener('pointermove', (e) => {
      if (drag) mv(e);
    });
    ['pointerup', 'pointercancel'].forEach((n2) =>
      c.addEventListener(n2, () => {
        drag = false;
      }),
    );
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
