/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  f2,
  fit,
  letterData,
  markTP,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { UC, uCur, uLine } from '../../unsupervised/shared.js';
import { sig } from '../../classification/shared.js';

export default {
  init(root) {
    const c = $('#va-c', root),
      cr = $('#va-r', root),
      live = $('#va-live', root),
      S = { beta: 1, flip: 0.04, col: true, seed: 1 },
      R = 3.2;
    let X,
      XF,
      lab,
      LT,
      enc,
      dec,
      ps,
      ts = 0,
      RC = [],
      KL = [],
      conv = false,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      MU = [],
      SD = [],
      pr = [0.5, 0.5],
      zs = [],
      drag = false;
    const gen = () => {
        const d = letterData(S.seed, S.flip);
        X = d.X.map((x) => x.map((v) => clamp(Math.round(v), 0, 1)));
        lab = d.lab;
        LT = d.CH;
        XF = new Float64Array(X.length * 35);
        X.forEach((x, i) =>
          x.forEach((v, j) => {
            XF[i * 35 + j] = v;
          }),
        );
      },
      newz = () => {
        const rn = randnFrom(Math.random);
        zs = Array.from({ length: 8 }, () => [rn(), rn()]);
      };
    const build = () => {
      const q = randnFrom(rng(run * 7 + 2));
      enc = AD.mlp([35, 24, 4], q, 'tanh');
      dec = AD.mlp([2, 24, 35], q, 'tanh');
      ps = enc.ps.concat(dec.ps);
      ts = 0;
      RC = [];
      KL = [];
      conv = false;
      cu = 0;
      codes();
      msg =
        'Random weights. Every picture’s cloud sits near the middle and everything decodes to grey. Press Train.';
    };
    const codes = () => {
      const o = enc.fwd(AD.input(XF, [X.length, 35]));
      AD.clear();
      MU = [];
      SD = [];
      for (let i = 0; i < X.length; i++) {
        MU.push([o.v[i * 4], o.v[i * 4 + 1]]);
        SD.push([Math.exp(0.5 * o.v[i * 4 + 2]), Math.exp(0.5 * o.v[i * 4 + 3])]);
      }
    };
    const decode = (z) => {
      const o = dec.fwd(AD.input(z, [1, 2]));
      AD.clear();
      return Float64Array.from(o.v, sig);
    };
    const train = (k) => {
      const n = X.length,
        rn = randnFrom(Math.random);
      for (let q = 0; q < k && !conv; q++) {
        const h = enc.fwd(AD.input(XF, [n, 35])),
          mu = AD.cols(h, 0, 2),
          lv = AD.cols(h, 2, 4),
          z = AD.reparam(
            mu,
            lv,
            Float64Array.from({ length: 2 * n }, () => rn()),
          ),
          rec = AD.bce(dec.fwd(z), XF),
          kl = AD.kl(mu, lv, n),
          loss = AD.wsum(rec, kl, S.beta / 35);
        RC.push(rec.v[0] * 35);
        KL.push(kl.v[0]);
        AD.backward(loss);
        AD.adam(ps, 0.006, ++ts, 1);
        if (
          (ts > 300 &&
            Math.abs(RC[ts - 1] + S.beta * KL[ts - 1] - RC[ts - 201] - S.beta * KL[ts - 201]) <
              0.01 * (RC[ts - 1] + S.beta * KL[ts - 1])) ||
          ts >= 2500
        )
          conv = true;
      }
      codes();
      cu = conv ? 4 : ts % 4;
      msg =
        'Epoch ' +
        ts +
        ': every picture became a cloud, one point was drawn from each and decoded. Reconstruction error ' +
        f2(RC[ts - 1]) +
        ' per picture; distance of the clouds from the centre ' +
        f2(KL[ts - 1]) +
        '.' +
        (conv
          ? ' Converged: the total has stopped falling. The second row on the right shows pictures made from random points.'
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
          sc = (Math.min(w, h) / 2 / R) * 0.96,
          X2 = (v) => w / 2 + v * sc,
          Y2 = (v) => h / 2 - v * sc,
          NG = 9;
        for (let a = 0; a < NG; a++)
          for (let b = 0; b < NG; b++) {
            const zx = -R + (2 * R * (b + 0.5)) / NG,
              zy = R - (2 * R * (a + 0.5)) / NG;
            glyph(g, decode([zx, zy]), X2(zx) - 9, Y2(zy) - 12, 3.6, t, 0.22);
          }
        [1, 2].forEach((r) => {
          g.strokeStyle = mix(t.gold, t.gold, 0, r === 1 ? 0.9 : 0.5);
          g.lineWidth = 1.6;
          g.setLineDash(r === 2 ? [5, 5] : []);
          g.beginPath();
          g.arc(X2(0), Y2(0), r * sc, 0, 7);
          g.stroke();
        });
        g.setLineDash([]);
        MU.forEach((m, i) => {
          const x = X2(m[0]),
            y = Y2(m[1]);
          if (x < 0 || y < 0 || x > w || y > h) return;
          g.fillStyle = mix(col[lab[i] % 8], col[lab[i] % 8], 0, 0.12);
          g.beginPath();
          g.ellipse(x, y, Math.max(1, SD[i][0] * sc), Math.max(1, SD[i][1] * sc), 0, 0, 7);
          g.fill();
          if (S.col) {
            g.font = '600 12px "Hanken Grotesk",sans-serif';
            g.fillStyle = col[lab[i] % 8];
            g.textAlign = 'center';
            g.fillText(LT[lab[i]], x, y + 4);
          } else {
            g.fillStyle = mix(t.ink, t.ink, 0, 0.6);
            g.beginPath();
            g.arc(x, y, 2.5, 0, 7);
            g.fill();
          }
        });
        markTP(g, X2(pr[0]), Y2(pr[1]), t, false);
        g.font = MONO;
        g.fillStyle = t.cut;
        g.textAlign = 'left';
        g.fillText('rings: where random draws come from (1 and 2 standard deviations)', 8, h - 8);
        c._g = { w, h, sc };
      }
      {
        const { g, w, h } = fit(cr),
          cz = Math.min((w - 60) / 8 / 6.2, (h * 0.15) / 7.4),
          gw = cz * 6.2,
          x0 = 50;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('in', 6, 12 + 3.5 * cz);
        g.fillText('out', 6, 18 + 10.5 * cz);
        g.fillStyle = t.cut;
        g.fillText('new', 6, 26 + 17.5 * cz);
        const o = enc.fwd(AD.input(XF, [X.length, 35]));
        AD.clear();
        for (let k = 0; k < 6; k++) {
          const i = k * 25;
          glyph(g, X[i], x0 + k * gw, 6, cz, t);
          glyph(g, decode([o.v[i * 4], o.v[i * 4 + 1]]), x0 + k * gw, 12 + 7 * cz, cz, t);
        }
        glyph(g, decode(pr), x0 + 6.6 * gw, 6, cz * 1.5, t);
        g.fillStyle = t.muted;
        g.fillText('diamond', x0 + 6.6 * gw, 18 + 10.5 * cz + 4);
        zs.forEach((z, k) => {
          glyph(g, decode(z), x0 + k * gw, 20 + 14 * cz, cz, t);
          g.strokeStyle = t.gold;
          g.lineWidth = 1;
          g.strokeRect(x0 + k * gw - 1.5, 18.5 + 14 * cz, 5 * cz + 2, 7 * cz + 2);
        });
        const y3 = 34 + 21 * cz,
          U = uLine(g, t, 0, y3, w, h - y3, RC, {
            title: 'reconstruction error (red) and β × cloud distance (green)',
            n: Math.max(200, RC.length),
            col: t.ok,
            ym: Math.max(24, RC[0] || 24),
          });
        g.strokeStyle = t.accent;
        g.lineWidth = 2;
        g.beginPath();
        KL.forEach((v, i) =>
          i ? g.lineTo(U.X(i), U.Y(S.beta * v)) : g.moveTo(U.X(i), U.Y(S.beta * v)),
        );
        g.stroke();
      }
      live.textContent = msg;
      uCur(root, 'va', cu);
      setR(root, 'ep', String(ts));
      setR(root, 'rec', ts ? f2(RC[ts - 1]) : '–');
      setR(root, 'kl', ts ? f2(KL[ts - 1]) : '–');
      setR(root, 'in', MU.filter((m) => Math.hypot(m[0], m[1]) < 2).length + ' of ' + MU.length);
      setR(root, 'st', conv ? 'Converged' : ts ? 'Training' : 'Not started');
    };
    gen();
    newz();
    build();
    if (RM) while (!conv) train(50);
    const redo = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    const anim = Anim(() => {
      fr++;
      if (ts < 30 && fr % 4) return;
      train(Math.min(ts < 30 ? 1 : ts < 300 ? 2 : 5, chk.remaining()));
      draw();
      chk();
    });
    pb = playBtn($('#va-play', root), anim);
    chk = runCtl(pb, root, 'va', {
      done: () => conv,
      restart: () => {
        run++;
        build();
      },
      count: () => ts,
      what: 'epochs',
    });
    $('#va-step', root).onclick = () => {
      pb.stop();
      if (conv) {
        run++;
        build();
      } else train(10);
      draw();
    };
    $('#va-reset', root).onclick = redo;
    $('#va-samp', root).onclick = () => {
      newz();
      if (!pb.on) draw();
    };
    bind(
      root,
      'va-beta',
      (v) => {
        S.beta = v;
        pb.stop();
        run++;
        build();
        draw();
        pb.start();
      },
      f2,
    );
    bind(root, 'va-flip', (v) => {
      S.flip = v;
      gen();
      redo();
    });
    $('#va-col', root).onchange = (e) => {
      S.col = e.target.checked;
      draw();
    };
    const mv = (e) => {
      const q = c._g;
      if (!q) return;
      const r = c.getBoundingClientRect();
      pr = [
        clamp((e.clientX - r.left - q.w / 2) / q.sc, -R, R),
        clamp(-(e.clientY - r.top - q.h / 2) / q.sc, -R, R),
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
