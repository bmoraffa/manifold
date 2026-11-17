/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  MONO,
  SANS,
  T,
  area,
  bind,
  clipPlot,
  curve,
  erf,
  f3,
  fit,
  grab,
  mix,
  npdf,
  playBtn,
  plot,
  setR,
  sweeper,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#mt-c', root),
      cr = $('#mt-r', root),
      S = { t: 0, sep: 1.8, pi: 0.3 },
      Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2)),
      tpr = (th) => 1 - Phi(th - S.sep / 2),
      fpr = (th) => 1 - Phi(th + S.sep / 2);
    let pb;
    const draw = () => {
      const t = T(),
        TP = S.pi * tpr(S.t),
        FN = S.pi - TP,
        FP = (1 - S.pi) * fpr(S.t),
        TN = 1 - S.pi - FP,
        pre = TP + FP > 1e-9 ? TP / (TP + FP) : 1,
        rec = tpr(S.t);
      {
        const { g, w, h } = fit(c),
          pn = (x) => (1 - S.pi) * npdf(x, -S.sep / 2, 1),
          pp = (x) => S.pi * npdf(x, S.sep / 2, 1),
          ym = Math.max(pn(-S.sep / 2), pp(S.sep / 2)) * 1.25,
          P = plot(g, w, h, [-5, 5], [0, ym], t, { xlabel: 'score' });
        clipPlot(g, P);
        area(g, P, pn, S.t, 5);
        g.fillStyle = mix(t.ok, t.ok, 0, 0.5);
        g.fill();
        area(g, P, pp, -5, S.t);
        g.fillStyle = mix(t.muted, t.muted, 0, 0.4);
        g.fill();
        area(g, P, pp, S.t, 5);
        g.fillStyle = mix(t.gold, t.gold, 0, 0.45);
        g.fill();
        curve(g, P, pn, -5, 5);
        g.strokeStyle = t.accent;
        g.lineWidth = 2.4;
        g.stroke();
        curve(g, P, pp, -5, 5);
        g.strokeStyle = t.gold;
        g.lineWidth = 2.4;
        g.stroke();
        g.strokeStyle = t.ink;
        g.lineWidth = 2.4;
        g.beginPath();
        g.moveTo(P.X(S.t), P.Y(0));
        g.lineTo(P.X(S.t), P.Y(ym));
        g.stroke();
        g.restore();
        g.font = SANS;
        g.fillStyle = t.ink;
        g.textAlign = 'right';
        g.fillText('← say no', P.X(S.t) - 8, P.m.t + 12);
        g.textAlign = 'left';
        g.fillText('say yes →', P.X(S.t) + 8, P.m.t + 12);
        g.font = MONO;
        let ly = P.m.t + 34;
        [
          [t.accent, 'negatives'],
          [t.cut, 'positives'],
          [t.cut, '■ hits'],
          [t.ok, '■ false alarms'],
          [t.muted, '■ misses'],
        ].forEach(([cl, sx]) => {
          g.fillStyle = cl;
          g.fillText(sx, P.m.l + 6, ly);
          ly += 14;
        });
      }
      {
        const { g, w, h } = fit(cr),
          gap = 26,
          pw = (w - gap) / 2,
          sub = (ox, xl, yl, fx, fy, px, py, base) => {
            const m = { l: 42, r: 6, t: 14, b: 30 },
              X = (v) => ox + m.l + v * (pw - m.l - m.r),
              Y = (v) => h - m.b - v * (h - m.t - m.b);
            g.strokeStyle = t.line;
            g.lineWidth = 1;
            g.font = MONO;
            g.fillStyle = t.muted;
            [0, 0.5, 1].forEach((v) => {
              g.beginPath();
              g.moveTo(X(v), Y(0));
              g.lineTo(X(v), Y(1));
              g.moveTo(X(0), Y(v));
              g.lineTo(X(1), Y(v));
              g.stroke();
              g.textAlign = 'center';
              g.fillText(String(v), X(v), h - m.b + 13);
              g.textAlign = 'right';
              g.fillText(String(v), X(0) - 5, Y(v) + 4);
            });
            g.textAlign = 'center';
            g.fillText(xl, X(0.5), h - 4);
            g.save();
            g.translate(ox + 9, Y(0.5));
            g.rotate(-Math.PI / 2);
            g.fillText(yl, 0, 0);
            g.restore();
            g.setLineDash([4, 4]);
            g.strokeStyle = t.muted;
            g.beginPath();
            base(X, Y);
            g.stroke();
            g.setLineDash([]);
            g.strokeStyle = t.accent;
            g.lineWidth = 2.6;
            g.beginPath();
            for (let i = 0; i <= 240; i++) {
              const th = 6 - (12 * i) / 240,
                x = X(fx(th)),
                y = Y(fy(th));
              i ? g.lineTo(x, y) : g.moveTo(x, y);
            }
            g.stroke();
            g.fillStyle = t.gold;
            g.strokeStyle = t.bg;
            g.lineWidth = 2;
            g.beginPath();
            g.arc(X(px), Y(py), 7, 0, 7);
            g.fill();
            g.stroke();
          };
        const prec = (th) => {
          const a = S.pi * tpr(th),
            b = (1 - S.pi) * fpr(th);
          return a + b > 1e-12 ? a / (a + b) : 1;
        };
        sub(0, 'false positive rate', 'recall', fpr, tpr, fpr(S.t), rec, (X, Y) => {
          g.moveTo(X(0), Y(0));
          g.lineTo(X(1), Y(1));
        });
        sub(pw + gap, 'recall', 'precision', tpr, prec, rec, pre, (X, Y) => {
          g.moveTo(X(0), Y(S.pi));
          g.lineTo(X(1), Y(S.pi));
        });
      }
      const N = 1000;
      setR(root, 'tp', Math.round(TP * N) + ' of 1,000');
      setR(root, 'fn', String(Math.round(FN * N)));
      setR(root, 'fp', String(Math.round(FP * N)));
      setR(root, 'tn', String(Math.round(TN * N)));
      setR(root, 'acc', (100 * (TP + TN)).toFixed(1) + '%');
      setR(root, 'pre', (100 * pre).toFixed(1) + '%');
      setR(root, 'rec', (100 * rec).toFixed(1) + '%');
      setR(root, 'fpr', (100 * fpr(S.t)).toFixed(1) + '%');
      setR(root, 'f1', pre + rec > 1e-9 ? f3((2 * pre * rec) / (pre + rec)) : '0');
      setR(root, 'auc', f3(Phi(S.sep / Math.SQRT2)));
    };
    const ts = bind(root, 'mt-t', (v) => {
      S.t = v;
      draw();
    });
    pb = playBtn($('#mt-play', root), sweeper(ts.el, -3.5, 3.5, 9));
    grab(ts.el, pb);
    bind(root, 'mt-sep', (v) => {
      S.sep = v;
      draw();
    });
    bind(root, 'mt-pi', (v) => {
      S.pi = v;
      draw();
    });
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
