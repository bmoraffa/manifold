/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  f2,
  fit,
  mix,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { sci, sigFlow } from '../shared.js';
import { uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#wi-c', root),
      cf = $('#wi-f', root),
      cb = $('#wi-b', root),
      live = $('#wi-live', root),
      S = { act: 'tanh', v: 1, d: 10 },
      MV = { small: 0.2, xavier: 1, he: 2, large: 6 };
    let F = null,
      k = 0,
      msg = '',
      cu = 0,
      fr = 0,
      pb;
    const comp = () => {
      F = sigFlow(S.act, S.d, S.v, 7);
      k = 0;
      cu = 0;
      msg =
        'Weights drawn at random with variance ' +
        f2(S.v) +
        ' ÷ (number of inputs). Nothing has been sent through yet. Press Send the signal through or One layer.';
    };
    const end = () => 2 * S.d,
      rel = (l) => F.gr[l] / F.gr[S.d - 1];
    const verdict = () => {
      const a = F.sd[S.d - 1],
        b = rel(0),
        sat = S.act !== 'relu' && a > 0.85 * (S.act === 'sigmoid' ? 0.5 : 1);
      return a < 0.02
        ? 'The signal has faded away'
        : a > 30
          ? 'The signal has exploded'
          : sat
            ? 'Neurons are jammed at their limits (saturated)'
            : b < 0.01
              ? 'Outputs look fine but the learning signal dies'
              : b > 100
                ? 'The learning signal explodes'
                : 'Healthy: the signal survives in both directions';
    };
    const step = () => {
      if (k >= end()) {
        k = 0;
        cu = 0;
        msg = 'Start again.';
        return;
      }
      k++;
      if (k <= S.d) {
        const l = k - 1;
        cu = k === 1 ? 1 : 2;
        msg =
          'Forward, layer ' +
          k +
          ': the outputs have spread ' +
          sci(F.sd[l]) +
          (l
            ? ' (the layer before had ' +
              sci(F.sd[l - 1]) +
              ', so this layer multiplied it by ' +
              f2(F.sd[l] / F.sd[l - 1]) +
              ')'
            : '') +
          '.';
      } else {
        const l = 2 * S.d - k;
        cu = k >= end() ? 4 : 3;
        msg =
          'Backward, layer ' +
          (l + 1) +
          ': the learning signal is ' +
          sci(rel(l)) +
          ' of its strength at the last layer.';
        if (k >= end()) msg += ' Done. Verdict: ' + verdict() + '.';
      }
    };
    const draw = () => {
      const t = T(),
        D = S.d,
        nf = Math.min(k, D),
        nbk = Math.max(0, k - D);
      {
        const { g, w, h } = fit(c),
          cw = w / D,
          bounded = S.act !== 'relu',
          lo = S.act === 'sigmoid' ? 0 : bounded ? -1 : 0;
        g.font = MONO;
        for (let l = 0; l < D; l++) {
          const x0 = l * cw + 6,
            ww = cw - 12,
            y1 = h - 30,
            on = l < nf;
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.strokeRect(x0, 14, ww, y1 - 14);
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('layer ' + (l + 1), x0 + ww / 2, h - 14);
          if (!on) continue;
          const a = F.acts[l],
            nb = 22;
          let hi = 1;
          if (!bounded) {
            hi = 1e-9;
            for (const v of a) hi = Math.max(hi, v);
            hi = Math.max(hi, 1e-6);
          }
          const cnt = new Array(nb).fill(0);
          for (const v of a) cnt[clamp(Math.floor(((v - lo) / (hi - lo)) * nb), 0, nb - 1)]++;
          const cm = Math.max(...cnt);
          g.fillStyle = l === nf - 1 && k <= D ? t.gold : t.accent;
          for (let q = 0; q < nb; q++) {
            const bh = (cnt[q] / cm) * (y1 - 24);
            g.fillRect(x0 + (q * ww) / nb, y1 - bh, ww / nb - 0.6, bh);
          }
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(bounded ? String(lo) : '0', x0, h - 2);
          g.textAlign = 'right';
          g.fillText(bounded ? '1' : sci(hi), x0 + ww, h - 2);
        }
      }
      const bars = (cv, vals, n, from, ttl, ref) => {
        const { g, w, h } = fit(cv),
          m = { l: 54, r: 10, t: 20, b: 24 },
          bw = (w - m.l - m.r) / D;
        let lo = -3,
          hi = 1;
        vals.forEach((v) => {
          const q = Math.log10(v + 1e-300);
          lo = Math.min(lo, Math.floor(q));
          hi = Math.max(hi, Math.ceil(q));
        });
        lo = Math.max(lo, -14);
        hi = Math.min(hi, 14);
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
          const on = from === 'L' ? l < n : l >= D - n,
            x = m.l + l * bw + bw * 0.15;
          g.fillStyle = on ? (ref ? t.ok : t.accent) : mix(t.ink, t.ink, 0, 0.07);
          const y1 = Y(1),
            y2 = on ? Y(vals[l]) : y1;
          g.fillRect(x, Math.min(y1, y2) - (on ? 0 : 2), bw * 0.7, Math.max(4, Math.abs(y2 - y1)));
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText(String(l + 1), x + bw * 0.35, h - m.b + 14);
        }
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(ttl, m.l, 12);
      };
      bars(cf, F.sd, nf, 'L', 'spread of the outputs (each line is ×10) →');
      bars(
        cb,
        F.gr.map((_, l) => rel(l)),
        nbk,
        'R',
        '← learning signal, relative to the last layer',
        true,
      );
      live.textContent = msg;
      uCur(root, 'wi', cu);
      setR(root, 'v', f2(S.v));
      setR(root, 's1', sci(F.sd[0]));
      setR(root, 'sL', nf >= D ? sci(F.sd[D - 1]) : '–');
      setR(root, 'g', k >= end() ? sci(rel(0)) : '–');
      setR(root, 'vd', k >= end() ? verdict() : '–');
      root
        .querySelectorAll('#wi-mode button')
        .forEach((b) =>
          b.setAttribute('aria-pressed', Math.abs(MV[b.dataset.v] - S.v) < 0.01 ? 'true' : 'false'),
        );
    };
    comp();
    k = end();
    const anim = Anim(() => {
      fr++;
      if (fr % 26) return;
      step();
      draw();
      if (k >= end()) pb.stop();
    });
    pb = playBtn($('#wi-play', root), anim);
    runCtl(pb, root, 'wi', {
      done: () => k >= end(),
      restart: () => {
        k = 0;
      },
    });
    $('#wi-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#wi-reset', root).onclick = () => {
      pb.stop();
      comp();
      draw();
    };
    const vb = bind(root, 'wi-v', (v) => {
      S.v = v;
      pb.stop();
      comp();
      k = end();
      draw();
    });
    bind(
      root,
      'wi-d',
      (v) => {
        S.d = v;
        pb.stop();
        comp();
        k = end();
        draw();
      },
      (v) => String(v),
    );
    $('#wi-act', root).onchange = (e) => {
      S.act = e.target.value;
      pb.stop();
      comp();
      draw();
      pb.start();
    };
    $('#wi-mode', root).addEventListener('click', (e) => {
      const v = e.target.dataset && e.target.dataset.v;
      if (!v) return;
      S.v = MV[v];
      vb.set(S.v);
      pb.stop();
      comp();
      draw();
      pb.start();
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
