/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  fit,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { ACT, sci } from '../../neural-nets/shared.js';
import { uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#rs-c', root),
      cr = $('#rs-r', root),
      cd = $('#rs-d', root),
      live = $('#rs-live', root),
      S = { d: 20, v: 0.8, act: 'tanh' };
    let Gp = [],
      Gr = [],
      k = 0,
      msg = '',
      cu = 0,
      fr = 0,
      t0 = 0,
      pb;
    const comp = () => {
      const n = 30,
        bt = 30,
        D = S.d,
        r = randnFrom(rng(5)),
        f = ACT[S.act][0],
        df = ACT[S.act][1],
        W = [],
        sdv = S.v / Math.sqrt(n),
        bta = 0.5;
      for (let l = 0; l < D; l++)
        W.push(Array.from({ length: n }, () => Float64Array.from({ length: n }, () => sdv * r())));
      const flow = (res) => {
        let A = [];
        for (let b = 0; b < bt; b++) A.push(Float64Array.from({ length: n }, () => r()));
        const Zs = [];
        for (let l = 0; l < D; l++) {
          const Z = [],
            NA = [];
          for (let b = 0; b < bt; b++) {
            const z = new Float64Array(n),
              a = new Float64Array(n);
            for (let j = 0; j < n; j++) {
              let s = 0;
              for (let i = 0; i < n; i++) s += W[l][j][i] * A[b][i];
              z[j] = s;
              a[j] = res ? A[b][j] + bta * f(s) : f(s);
            }
            Z.push(z);
            NA.push(a);
          }
          Zs.push(Z);
          A = NA;
        }
        let Dl = [];
        for (let b = 0; b < bt; b++) Dl.push(Float64Array.from({ length: n }, () => r()));
        const G = new Array(D + 1).fill(0),
          rms = (Q) => {
            let s = 0;
            for (const q of Q) for (const u of q) s += u * u;
            return Math.sqrt(s / (n * bt));
          };
        G[D] = rms(Dl);
        for (let l = D - 1; l >= 0; l--) {
          const ND = [];
          for (let b = 0; b < bt; b++) {
            const da = new Float64Array(n);
            for (let i = 0; i < n; i++) {
              let q = 0;
              for (let j = 0; j < n; j++) q += W[l][j][i] * Dl[b][j] * df(Zs[l][b][j]);
              da[i] = res ? Dl[b][i] + bta * q : q;
            }
            ND.push(da);
          }
          Dl = ND;
          G[l] = rms(Dl);
        }
        return G.map((q) => q / G[D]);
      };
      Gp = flow(false);
      Gr = flow(true);
      k = 0;
      cu = 0;
      msg =
        'Two stacks of ' +
        S.d +
        ' layers are ready, built from the same weights. Press Send the signal back or One layer.';
    };
    const step = () => {
      if (k >= S.d) {
        k = 0;
        cu = 0;
        return;
      }
      k++;
      t0 = fr;
      const l = S.d - k;
      cu = k === 1 ? 2 : k >= S.d ? 4 : 3;
      msg =
        'Back to layer ' +
        (l + 1) +
        '. Plain: the signal was multiplied by ' +
        sci(Gp[l] / Gp[l + 1]) +
        ' and is now ' +
        sci(Gp[l]) +
        ' of its starting strength. With shortcuts: multiplied by ' +
        sci(Gr[l] / Gr[l + 1]) +
        ', now ' +
        sci(Gr[l]) +
        '.' +
        (k >= S.d
          ? ' At the first layer the plain stack has ' +
            sci(Gp[0]) +
            ' left and the stack with shortcuts has ' +
            sci(Gr[0]) +
            '.'
          : '');
    };
    const draw = () => {
      const t = T(),
        D = S.d,
        u = RM ? 1 : clamp((fr - t0) / 22, 0, 1);
      {
        const { g, w, h } = fit(c),
          m = 30,
          bw = (w - 2 * m) / D,
          rowY = [h * 0.27, h * 0.72],
          rad = (v) => clamp(13 + 3.4 * Math.log10(v + 1e-300), 1.5, 24);
        g.font = MONO;
        [
          [Gp, 'plain stack', t.ok, false],
          [Gr, 'with shortcuts', t.accent, true],
        ].forEach(([G, lab, cl, res], ri) => {
          const y = rowY[ri];
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(lab, m, y - h * 0.2);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.2;
          g.beginPath();
          g.moveTo(m, y);
          g.lineTo(w - m, y);
          g.stroke();
          for (let l = 0; l < D; l++) {
            const x = m + l * bw + bw * 0.2,
              ww = bw * 0.6;
            g.fillStyle = t.bg;
            g.strokeStyle = t.ink;
            g.lineWidth = 1.2;
            g.beginPath();
            g.rect(x, y - 11, ww, 22);
            g.fill();
            g.stroke();
            if (res) {
              g.strokeStyle = mix(cl, cl, 0, 0.8);
              g.lineWidth = 1.4;
              g.beginPath();
              g.moveTo(m + l * bw + bw * 0.06, y);
              g.quadraticCurveTo(
                m + l * bw + bw * 0.5,
                y - h * 0.17,
                m + (l + 1) * bw - bw * 0.06,
                y,
              );
              g.stroke();
            }
            if (D <= 24 || l % 5 === 0) {
              g.fillStyle = t.muted;
              g.textAlign = 'center';
              g.fillText(String(l + 1), x + ww / 2, y + 28);
            }
          }
          if (k > 0) {
            const pos = D - k + (1 - u),
              li = Math.min(D, Math.round(pos)),
              v = G[D - k] * u + G[Math.min(D, D - k + 1)] * (1 - u),
              x = m + pos * bw;
            g.fillStyle = cl;
            g.beginPath();
            g.arc(x, y, rad(v), 0, 7);
            g.fill();
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.fillText(sci(G[D - k]), clamp(x, 60, w - 60), y + h * 0.19);
          } else {
            g.fillStyle = cl;
            g.beginPath();
            g.arc(w - m, y, rad(1), 0, 7);
            g.fill();
          }
        });
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('← the learning signal travels this way', w - m, h - 6);
      }
      {
        const { g, w, h } = fit(cr),
          m = { l: 54, r: 10, t: 16, b: 22 },
          bw = (w - m.l - m.r) / D;
        let lo = -3,
          hi = 1;
        [Gp, Gr].forEach((G) =>
          G.forEach((v) => {
            const q = Math.log10(v + 1e-300);
            lo = Math.min(lo, Math.floor(q));
            hi = Math.max(hi, Math.ceil(q));
          }),
        );
        lo = Math.max(lo, -16);
        hi = Math.min(hi, 12);
        const Y = (v) =>
          h - m.b - ((clamp(Math.log10(v + 1e-300), lo, hi) - lo) / (hi - lo)) * (h - m.t - m.b);
        g.font = MONO;
        for (let e = lo; e <= hi; e += Math.max(1, Math.round((hi - lo) / 6))) {
          g.strokeStyle = e === 0 ? t.muted : t.line;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, Y(10 ** e));
          g.lineTo(w - m.r, Y(10 ** e));
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText(e === 0 ? '1' : '10^' + e, m.l - 5, Y(10 ** e) + 4);
        }
        for (let l = 0; l < D; l++) {
          const on = l >= D - k,
            x = m.l + l * bw;
          [
            [Gp, t.ok, 0.08],
            [Gr, t.accent, 0.5],
          ].forEach(([G, cl, o]) => {
            const y1 = Y(1),
              y2 = on ? Y(G[l]) : y1;
            g.fillStyle = on ? cl : mix(t.ink, t.ink, 0, 0.07);
            g.fillRect(
              x + bw * o,
              Math.min(y1, y2) - (on ? 0 : 1),
              bw * 0.4,
              Math.max(2, Math.abs(y2 - y1)),
            );
          });
        }
        g.textAlign = 'left';
        g.fillStyle = t.ok;
        g.fillText('red: plain', m.l + 6, m.t + 2);
        g.fillStyle = t.accent;
        g.fillText('green: with shortcuts', m.l + 90, m.t + 2);
        g.fillStyle = t.muted;
        g.fillText('first layer', m.l, h - 5);
        g.textAlign = 'right';
        g.fillText('last layer', w - m.r, h - 5);
      }
      {
        const { g, w, h } = fit(cd),
          bx = (x, y, tx) => {
            g.fillStyle = t.bg;
            g.strokeStyle = t.ink;
            g.lineWidth = 1.4;
            g.beginPath();
            g.rect(x - 58, y - 15, 116, 30);
            g.fill();
            g.stroke();
            g.font = MONO;
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.fillText(tx, x, y + 4);
          },
          ar = (x1, y1, x2, y2) => {
            g.strokeStyle = t.ink;
            g.lineWidth = 1.4;
            g.beginPath();
            g.moveTo(x1, y1);
            g.lineTo(x2, y2);
            g.stroke();
          },
          ph = (fr / 70) % 1;
        g.font = MONO;
        const xa = w * 0.25,
          xb = w * 0.72;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('plain', xa, 14);
        g.fillText('with a shortcut', xb, 14);
        ar(xa, 28, xa, h - 24);
        bx(xa, h * 0.5, 'weights, activation');
        g.fillStyle = t.ink;
        g.fillText('input a', xa, 28);
        g.fillText('output F(a)', xa, h - 10);
        ar(xb, 28, xb, h - 24);
        bx(xb, h * 0.45, 'weights, activation');
        g.strokeStyle = t.accent;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(xb, 40);
        g.bezierCurveTo(xb + 120, 40, xb + 120, h * 0.74, xb + 10, h * 0.74);
        g.stroke();
        g.fillStyle = t.bg;
        g.strokeStyle = t.ink;
        g.lineWidth = 1.4;
        g.beginPath();
        g.arc(xb, h * 0.74, 10, 0, 7);
        g.fill();
        g.stroke();
        g.fillStyle = t.ink;
        g.fillText('+', xb, h * 0.74 + 4);
        g.fillText('input a', xb, 28);
        g.fillText('output a + F(a)', xb, h - 10);
        g.fillStyle = t.accent;
        g.textAlign = 'left';
        g.fillText('shortcut: a, unchanged', Math.min(w - 150, xb + 62), h * 0.3);
        if (pb && pb.on) {
          g.fillStyle = t.gold;
          [xa, xb].forEach((x) => {
            g.beginPath();
            g.arc(x, h - 24 - (h - 52) * ph, 5, 0, 7);
            g.fill();
          });
          const q = 1 - ph,
            bz = (a, b2, cc, d) =>
              q ** 3 * a + 3 * q * q * ph * b2 + 3 * q * ph * ph * cc + ph ** 3 * d;
          g.fillStyle = t.accent;
          g.beginPath();
          g.arc(bz(xb + 10, xb + 120, xb + 120, xb), bz(h * 0.74, h * 0.74, 40, 40), 5, 0, 7);
          g.fill();
        }
      }
      live.textContent = msg;
      uCur(root, 'rs', cu);
      const fin = k >= D;
      setR(root, 'pl', fin ? sci(Gp[0]) : '–');
      setR(root, 'rs', fin ? sci(Gr[0]) : '–');
      setR(root, 'pp', '× ' + sci(Math.pow(Gp[0], 1 / D)));
      setR(root, 'rp', '× ' + sci(Math.pow(Gr[0], 1 / D)));
    };
    comp();
    k = S.d;
    const anim = Anim(() => {
      fr++;
      if (fr - t0 >= (S.d > 30 ? 10 : 24) && k < S.d) step();
      draw();
      if (k >= S.d && fr - t0 > 26) pb.stop();
    });
    pb = playBtn($('#rs-play', root), anim);
    runCtl(pb, root, 'rs', {
      done: () => k >= S.d,
      restart: () => {
        k = 0;
      },
    });
    $('#rs-step', root).onclick = () => {
      pb.stop();
      step();
      t0 = fr - 100;
      draw();
    };
    $('#rs-reset', root).onclick = () => {
      pb.stop();
      comp();
      draw();
    };
    const redo = () => {
      pb.stop();
      comp();
      draw();
      pb.start();
    };
    bind(
      root,
      'rs-depth',
      (v) => {
        S.d = v;
        pb.stop();
        comp();
        k = S.d;
        draw();
      },
      (v) => String(v),
    );
    bind(root, 'rs-v', (v) => {
      S.v = v;
      pb.stop();
      comp();
      k = S.d;
      draw();
    });
    $('#rs-act', root).onchange = (e) => {
      S.act = e.target.value;
      redo();
    };
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        comp();
        draw();
        pb.start();
      },
    };
  },
};
