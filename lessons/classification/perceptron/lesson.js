/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { MONO, clipPlot, f2, fit, plot } from '../../../js/core/shared.js';
import { clfLesson, sig } from '../shared.js';

export default clfLesson({
  id: 'perceptron',
  maxWhat: 'passes',
  hard: true,

  ds: ['blobs', 'overlap', 'moons', 'xor'],
  stepBtn: true,
  S: { lr: 0.5 },
  n: 60,
  setup(api) {
    const S = api.S;
    let w,
      order,
      i,
      ep,
      mist,
      hist,
      last,
      upd,
      wait = 0,
      fr = 0,
      msg = '',
      cu = 0;
    const newEp = () => {
      order = api.train.slice();
      for (let a = order.length - 1; a > 0; a--) {
        const b = Math.floor(Math.random() * (a + 1));
        [order[a], order[b]] = [order[b], order[a]];
      }
      i = 0;
      mist = 0;
    };
    const reset = () => {
      w = [0, Math.random() - 0.5, Math.random() - 0.5];
      ep = 0;
      hist = [];
      upd = 0;
      last = null;
      wait = 0;
      newEp();
      api.note('');
      msg = 'A random starting line. Press Train or One step.';
      cu = 0;
    };
    const adv = () => {
      while (i < order.length) {
        const p = order[i++],
          y = p.c ? 1 : -1;
        if (y * (w[0] + w[1] * p.x + w[2] * p.y) <= 0) {
          const o = w.slice();
          w[0] += S.lr * y;
          w[1] += S.lr * y * p.x;
          w[2] += S.lr * y * p.y;
          mist++;
          upd++;
          last = p;
          cu = 3;
          msg =
            'Pass ' +
            (ep + 1) +
            ', point ' +
            i +
            ' of ' +
            order.length +
            ': (' +
            f2(p.x) +
            ', ' +
            f2(p.y) +
            ') is class ' +
            (p.c ? 'B' : 'A') +
            ' but lies on the ' +
            (p.c ? 'A' : 'B') +
            ' side. Wrong, so add ' +
            f2(S.lr) +
            ' × (' +
            (y > 0 ? '+1' : '−1') +
            ') × (1, ' +
            f2(p.x) +
            ', ' +
            f2(p.y) +
            ') to the weights: (' +
            o.map(f2).join(', ') +
            ') → (' +
            w.map(f2).join(', ') +
            '). Correction number ' +
            upd +
            '.';
          return 'upd';
        }
      }
      hist.push(mist);
      ep++;
      const done = mist === 0;
      msg =
        'Pass ' +
        ep +
        ' finished with ' +
        mist +
        (mist === 1 ? ' mistake.' : ' mistakes.') +
        (done
          ? ' Every point is on the correct side, so the perceptron stops.'
          : ' Start the next pass.');
      cu = done ? 4 : 1;
      newEp();
      if (done) last = null;
      return done ? 'done' : 'epoch';
    };
    api.bind('lr', 'lr', f2, () => {});
    return {
      reset,
      p: (x, y) => sig((7 * (w[0] + w[1] * x + w[2] * y)) / (Math.hypot(w[1], w[2]) || 1)),
      stepOnce: () => {
        adv();
      },
      count: () => ep,
      done: () => wait > 0,
      tick() {
        fr++;
        if (wait > 0) return false;
        if (fr % 5) return false;
        const r = adv();
        if (r === 'done') {
          api.note(
            'A full pass with no mistakes: every training point is on the right side, so it stops.',
          );
          wait = 220;
        } else if (ep >= 40) {
          api.note(
            '40 passes and still making mistakes. No straight line separates these classes, so the perceptron will never stop.',
          );
          wait = 320;
        }
        return true;
      },
      overlay(g, X, Y, t) {
        if (last) {
          g.strokeStyle = t.ink;
          g.lineWidth = 3;
          g.beginPath();
          g.arc(X(last.x), Y(last.y), 11, 0, 7);
          g.stroke();
        }
        const n = Math.hypot(w[1], w[2]) || 1,
          cx = (-w[0] * w[1]) / (n * n),
          cy = (-w[0] * w[2]) / (n * n);
        if (Math.abs(cx) < 2.8 && Math.abs(cy) < 2.8) {
          g.strokeStyle = t.ink;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(X(cx), Y(cy));
          g.lineTo(X(cx + (0.6 * w[1]) / n), Y(cy + (0.6 * w[2]) / n));
          g.stroke();
          g.font = MONO;
          g.fillStyle = t.ink;
          g.fillText('w', X(cx + (0.75 * w[1]) / n), Y(cy + (0.75 * w[2]) / n));
        }
      },
      right(t) {
        const { g, w: cw, h } = fit(api.cr),
          n = Math.max(12, hist.length),
          mx = Math.max(4, ...hist),
          P = plot(g, cw, h, [0, n + 0.5], [0, mx * 1.1], t, {
            xstep: n > 30 ? 10 : n > 14 ? 5 : 2,
            xlabel: 'pass',
          });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText(String(mx), P.m.l - 6, P.Y(mx) + 4);
        g.fillText('0', P.m.l - 6, P.Y(0) + 4);
        clipPlot(g, P);
        const bw = (P.X(1) - P.X(0)) * 0.7;
        hist.forEach((v, k) => {
          g.fillStyle = v ? t.ok : t.accent;
          g.fillRect(
            P.X(k + 1) - bw / 2,
            P.Y(Math.max(v, mx * 0.012)),
            bw,
            P.Y(0) - P.Y(Math.max(v, mx * 0.012)),
          );
        });
        g.restore();
      },
      status: () => msg,
      cur: () => cu,
      read(set) {
        set('ep', String(ep));
        set('up', String(upd));
        set('ms', hist.length ? String(hist[hist.length - 1]) : '–');
      },
    };
  },
});
