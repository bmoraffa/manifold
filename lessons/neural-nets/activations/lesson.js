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
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { ACT, ACTN, sci, sigFlow } from '../shared.js';
import { segBind, uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#af-c', root),
      cr = $('#af-r', root),
      live = $('#af-live', root),
      S = { act: 'sigmoid', d: 12, w: 1 };
    let F = null,
      shown = 0,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      ph = 0;
    const comp = () => {
      F = sigFlow(S.act, S.d, S.w * S.w, 11);
      shown = 0;
      cu = 0;
      msg =
        'A stack of ' +
        S.d +
        ' layers with ' +
        ACTN[S.act] +
        ' is ready, and data has been sent forward through it. Press Send the signal back or One layer.';
    };
    const rel = (l) => F.gr[l] / F.gr[S.d - 1];
    const step = () => {
      if (shown >= S.d) {
        shown = 0;
        cu = 0;
        return;
      }
      shown++;
      const l = S.d - shown;
      cu = shown === 1 ? 1 : shown >= S.d ? 4 : shown === 2 ? 2 : 3;
      msg =
        shown === 1
          ? 'The signal starts at layer ' + S.d + ', the last one, with strength 1.'
          : 'Layer ' +
            (l + 1) +
            ': going back one layer multiplied the signal by ' +
            sci(F.gr[l] / F.gr[l + 1]) +
            '. It is now ' +
            sci(rel(l)) +
            ' of what it was at the last layer.';
      if (shown >= S.d)
        msg +=
          ' That is what reaches the first layer after ' +
          S.d +
          ' layers: ' +
          (rel(0) < 0.01
            ? 'almost nothing. The first layers of this network would learn extremely slowly.'
            : rel(0) > 100
              ? 'far too much. This network’s first updates would be wildly large.'
              : 'a healthy share. Every layer of this network can learn.');
    };
    const draw = () => {
      const t = T(),
        f = ACT[S.act][0],
        df = ACT[S.act][1],
        z = 4.6 * Math.sin(ph);
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, [-5, 5], [-1.3, S.act === 'relu' || S.act === 'leaky' ? 3.2 : 1.3], t, {
            xstep: 1,
            yticks: [-1, 0, 1],
            xlabel: 'total input z',
          });
        clipPlot(g, P);
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(P.X(0), P.Y(-2));
        g.lineTo(P.X(0), P.Y(4));
        g.stroke();
        curve(g, P, df, -5, 5, 400);
        g.strokeStyle = t.ok;
        g.lineWidth = 2.2;
        g.stroke();
        curve(g, P, f, -5, 5, 400);
        g.strokeStyle = t.accent;
        g.lineWidth = 3;
        g.stroke();
        const s = df(z);
        g.strokeStyle = t.gold;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(P.X(z - 1.1), P.Y(f(z) - 1.1 * s));
        g.lineTo(P.X(z + 1.1), P.Y(f(z) + 1.1 * s));
        g.stroke();
        g.setLineDash([3, 3]);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(P.X(z), P.Y(f(z)));
        g.lineTo(P.X(z), P.Y(s));
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = t.accent;
        g.beginPath();
        g.arc(P.X(z), P.Y(f(z)), 6.5, 0, 7);
        g.fill();
        g.fillStyle = t.ok;
        g.beginPath();
        g.arc(P.X(z), P.Y(s), 5, 0, 7);
        g.fill();
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.accent;
        g.fillText('green: ' + ACTN[S.act] + '(z)', P.m.l + 8, P.m.t + 8);
        g.fillStyle = t.ok;
        g.fillText('red: its slope = share of the signal let through', P.m.l + 8, P.m.t + 22);
        g.fillStyle = t.cut;
        g.fillText('gold: the tangent at the dot', P.m.l + 8, P.m.t + 36);
      }
      {
        const { g, w, h } = fit(cr),
          m = { l: 54, r: 12, t: 20, b: 36 },
          D = S.d,
          bw = (w - m.l - m.r) / D;
        let lo = -6,
          hi = 1;
        for (let l = 0; l < D; l++) {
          const v = Math.log10(rel(l) + 1e-300);
          lo = Math.min(lo, Math.floor(v));
          hi = Math.max(hi, Math.ceil(v));
        }
        lo = Math.max(lo, -14);
        const Y = (v) =>
          h - m.b - ((clamp(Math.log10(v + 1e-300), lo, hi) - lo) / (hi - lo)) * (h - m.t - m.b);
        g.font = MONO;
        g.textAlign = 'right';
        for (let e = lo; e <= hi; e += Math.max(1, Math.round((hi - lo) / 7))) {
          g.strokeStyle = e === 0 ? t.muted : t.line;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, Y(10 ** e));
          g.lineTo(w - m.r, Y(10 ** e));
          g.stroke();
          g.fillStyle = t.muted;
          g.fillText(e === 0 ? '1' : '10^' + e, m.l - 5, Y(10 ** e) + 4);
        }
        for (let l = 0; l < D; l++) {
          const on = l >= D - shown,
            x = m.l + l * bw + bw * 0.14,
            v = rel(l),
            y1 = Y(1),
            y2 = Y(v);
          g.fillStyle = on
            ? l === D - shown
              ? t.gold
              : v < 0.01
                ? t.ok
                : t.accent
            : mix(t.ink, t.ink, 0, 0.07);
          if (on) g.fillRect(x, Math.min(y1, y2), bw * 0.72, Math.max(3, Math.abs(y2 - y1)));
          else g.fillRect(x, y1 - 2, bw * 0.72, 4);
          if (D <= 16 || l % 2 === 0) {
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(String(l + 1), x + bw * 0.36, h - m.b + 14);
          }
        }
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('first layer', m.l, h - 5);
        g.textAlign = 'right';
        g.fillText('last layer: the signal starts here ←', w - m.r, h - 5);
        g.textAlign = 'left';
        g.fillText('strength compared with the start (each line is ×10)', m.l, 12);
      }
      live.textContent = msg;
      uCur(root, 'af', cu);
      setR(root, 'f', f3(f(z)));
      setR(root, 's', f3(df(z)));
      setR(root, 'mx', S.act === 'sigmoid' ? '0.25' : '1');
      setR(
        root,
        'g1',
        shown >= S.d ? sci(rel(0)) + ' of the start' : 'send the signal back to see',
      );
      setR(root, 'pl', '× ' + sci(Math.pow(rel(0), 1 / Math.max(1, S.d - 1))));
    };
    comp();
    shown = S.d;
    const anim = Anim(() => {
      fr++;
      ph += 0.02;
      if (fr % 34 === 0 && shown < S.d) step();
      draw();
      if (shown >= S.d && fr % 34 === 0) pb.stop();
    });
    pb = playBtn($('#af-play', root), anim);
    runCtl(pb, root, 'af', {
      done: () => shown >= S.d,
      restart: () => {
        shown = 0;
      },
    });
    $('#af-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#af-reset', root).onclick = () => {
      pb.stop();
      comp();
      draw();
    };
    segBind(root, 'af-act', S.act, (v) => {
      S.act = v;
      pb.stop();
      comp();
      draw();
      pb.start();
    });
    bind(
      root,
      'af-d',
      (v) => {
        S.d = v;
        pb.stop();
        comp();
        shown = S.d;
        draw();
      },
      (v) => String(v),
    );
    bind(root, 'af-w', (v) => {
      S.w = v;
      pb.stop();
      comp();
      shown = S.d;
      draw();
    });
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
