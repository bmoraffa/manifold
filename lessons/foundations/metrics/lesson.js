/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  MONO,
  SANS,
  readThemeColors,
  area,
  bind,
  clipPlot,
  curve,
  erf,
  f3,
  prepareCanvas,
  grab,
  mix,
  npdf,
  playBtn,
  plot,
  setReadout,
  sweeper,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#mt-c', root),
      cr = $('#mt-r', root),
      settings = { t: 0, sep: 1.8, pi: 0.3 },
      Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2)),
      tpr = (th) => 1 - Phi(th - settings.sep / 2),
      fpr = (th) => 1 - Phi(th + settings.sep / 2);
    let playback;
    const draw = () => {
      const colors = readThemeColors(),
        TP = settings.pi * tpr(settings.t),
        FN = settings.pi - TP,
        FP = (1 - settings.pi) * fpr(settings.t),
        TN = 1 - settings.pi - FP,
        pre = TP + FP > 1e-9 ? TP / (TP + FP) : 1,
        rec = tpr(settings.t);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          weightedNegativeDensity = (x) => (1 - settings.pi) * npdf(x, -settings.sep / 2, 1),
          weightedPositiveDensity = (x) => settings.pi * npdf(x, settings.sep / 2, 1),
          ym =
            Math.max(
              weightedNegativeDensity(-settings.sep / 2),
              weightedPositiveDensity(settings.sep / 2),
            ) * 1.25,
          P = plot(ctx, w, h, [-5, 5], [0, ym], colors, { xlabel: 'score' });
        clipPlot(ctx, P);
        area(ctx, P, weightedNegativeDensity, settings.t, 5);
        ctx.fillStyle = mix(colors.ok, colors.ok, 0, 0.5);
        ctx.fill();
        area(ctx, P, weightedPositiveDensity, -5, settings.t);
        ctx.fillStyle = mix(colors.muted, colors.muted, 0, 0.4);
        ctx.fill();
        area(ctx, P, weightedPositiveDensity, settings.t, 5);
        ctx.fillStyle = mix(colors.gold, colors.gold, 0, 0.45);
        ctx.fill();
        curve(ctx, P, weightedNegativeDensity, -5, 5);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.4;
        ctx.stroke();
        curve(ctx, P, weightedPositiveDensity, -5, 5);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2.4;
        ctx.stroke();
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(P.X(settings.t), P.Y(0));
        ctx.lineTo(P.X(settings.t), P.Y(ym));
        ctx.stroke();
        ctx.restore();
        ctx.font = SANS;
        ctx.fillStyle = colors.ink;
        ctx.textAlign = 'right';
        ctx.fillText('← say no', P.X(settings.t) - 8, P.m.t + 12);
        ctx.textAlign = 'left';
        ctx.fillText('say yes →', P.X(settings.t) + 8, P.m.t + 12);
        ctx.font = MONO;
        let ly = P.m.t + 34;
        [
          [colors.accent, 'negatives'],
          [colors.cut, 'positives'],
          [colors.cut, '■ hits'],
          [colors.ok, '■ false alarms'],
          [colors.muted, '■ misses'],
        ].forEach(([cl, sx]) => {
          ctx.fillStyle = cl;
          ctx.fillText(sx, P.m.l + 6, ly);
          ly += 14;
        });
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          gap = 26,
          pw = (w - gap) / 2,
          drawMetricPanel = (ox, xl, yl, fx, fy, px, py, base) => {
            const m = { l: 42, r: 6, t: 14, b: 30 },
              X = (v) => ox + m.l + v * (pw - m.l - m.r),
              Y = (v) => h - m.b - v * (h - m.t - m.b);
            ctx.strokeStyle = colors.line;
            ctx.lineWidth = 1;
            ctx.font = MONO;
            ctx.fillStyle = colors.muted;
            [0, 0.5, 1].forEach((v) => {
              ctx.beginPath();
              ctx.moveTo(X(v), Y(0));
              ctx.lineTo(X(v), Y(1));
              ctx.moveTo(X(0), Y(v));
              ctx.lineTo(X(1), Y(v));
              ctx.stroke();
              ctx.textAlign = 'center';
              ctx.fillText(String(v), X(v), h - m.b + 13);
              ctx.textAlign = 'right';
              ctx.fillText(String(v), X(0) - 5, Y(v) + 4);
            });
            ctx.textAlign = 'center';
            ctx.fillText(xl, X(0.5), h - 4);
            ctx.save();
            ctx.translate(ox + 9, Y(0.5));
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(yl, 0, 0);
            ctx.restore();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = colors.muted;
            ctx.beginPath();
            base(X, Y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.strokeStyle = colors.accent;
            ctx.lineWidth = 2.6;
            ctx.beginPath();
            for (let i = 0; i <= 240; i++) {
              const th = 6 - (12 * i) / 240,
                x = X(fx(th)),
                y = Y(fy(th));
              i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            }
            ctx.stroke();
            ctx.fillStyle = colors.gold;
            ctx.strokeStyle = colors.bg;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(X(px), Y(py), 7, 0, 7);
            ctx.fill();
            ctx.stroke();
          };
        const prec = (th) => {
          const a = settings.pi * tpr(th),
            b = (1 - settings.pi) * fpr(th);
          return a + b > 1e-12 ? a / (a + b) : 1;
        };
        drawMetricPanel(
          0,
          'false positive rate',
          'recall',
          fpr,
          tpr,
          fpr(settings.t),
          rec,
          (X, Y) => {
            ctx.moveTo(X(0), Y(0));
            ctx.lineTo(X(1), Y(1));
          },
        );
        drawMetricPanel(pw + gap, 'recall', 'precision', tpr, prec, rec, pre, (X, Y) => {
          ctx.moveTo(X(0), Y(settings.pi));
          ctx.lineTo(X(1), Y(settings.pi));
        });
      }
      const N = 1000;
      setReadout(root, 'tp', Math.round(TP * N) + ' of 1,000');
      setReadout(root, 'fn', String(Math.round(FN * N)));
      setReadout(root, 'fp', String(Math.round(FP * N)));
      setReadout(root, 'tn', String(Math.round(TN * N)));
      setReadout(root, 'acc', (100 * (TP + TN)).toFixed(1) + '%');
      setReadout(root, 'pre', (100 * pre).toFixed(1) + '%');
      setReadout(root, 'rec', (100 * rec).toFixed(1) + '%');
      setReadout(root, 'fpr', (100 * fpr(settings.t)).toFixed(1) + '%');
      setReadout(root, 'f1', pre + rec > 1e-9 ? f3((2 * pre * rec) / (pre + rec)) : '0');
      setReadout(root, 'auc', f3(Phi(settings.sep / Math.SQRT2)));
    };
    const ts = bind(root, 'mt-t', (v) => {
      settings.t = v;
      draw();
    });
    playback = playBtn($('#mt-play', root), sweeper(ts.el, -3.5, 3.5, 9));
    grab(ts.el, playback);
    bind(root, 'mt-sep', (v) => {
      settings.sep = v;
      draw();
    });
    bind(root, 'mt-pi', (v) => {
      settings.pi = v;
      draw();
    });
    return { draw, stop: () => playback.stop(), auto: () => playback.start() };
  },
};
