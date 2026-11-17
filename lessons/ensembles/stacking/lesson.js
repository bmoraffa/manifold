/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $, MONO, clamp, f2, fit } from '../../../js/core/shared.js';
import { clfLesson, sig } from '../../classification/shared.js';
import { fitLogit, growTree, miniField, pct, treeP } from '../shared.js';

export default clfLesson({
  id: 'stacking',
  ds: ['moons', 'circles', 'overlap', 'blobs', 'xor'],
  S: { view: 3 },
  n: 140,
  nc: 40,

  setup(api) {
    const S = api.S,
      root = api.root;
    let base = null,
      meta = [0, 0, 0, 0],
      accs = [],
      fr = 0,
      hold = [{}, {}, {}],
      ver = 0;
    const NM = ['a straight line', '7 nearest neighbours', 'a small tree', 'the stacked model'];
    const mkBase = (T) => {
      const wl = fitLogit((x, y) => [1, x, y], T, 400),
        tq = growTree(T, { depth: 3, minleaf: 2 });
      return [
        (x, y) => sig(wl[0] + wl[1] * x + wl[2] * y),
        (x, y) => {
          const d = T.map((p) => [(p.x - x) ** 2 + (p.y - y) ** 2, p.c]);
          d.sort((a, b) => a[0] - b[0]);
          const k = Math.min(7, d.length);
          let s = 0;
          for (let i = 0; i < k; i++) s += d[i][1];
          return s / k;
        },
        (x, y) => treeP(tq, x, y),
      ];
    };
    const stack = (x, y) =>
      sig(meta[0] + meta[1] * base[0](x, y) + meta[2] * base[1](x, y) + meta[3] * base[2](x, y));
    $('#stacking-view', root).addEventListener('click', (e) => {
      const v = e.target.dataset && e.target.dataset.v;
      if (v != null) {
        api.pause();
        S.view = +v;
        api.redraw();
      }
    });
    return {
      fit() {
        const T = api.train,
          te = api.test;
        if (T.length < 15) return;
        base = mkBase(T);
        const K = 5,
          Z = [],
          Y = [];
        for (let f = 0; f < K; f++) {
          const bs = mkBase(T.filter((_, i) => i % K !== f));
          T.forEach((p, i) => {
            if (i % K === f) {
              Z.push(bs.map((b) => b(p.x, p.y)));
              Y.push(p.c);
            }
          });
        }
        meta = [0, 0, 0, 0];
        for (let it = 0; it < 2500; it++) {
          const g = [0, 0, 0, 0];
          Z.forEach((z, i) => {
            const e = sig(meta[0] + meta[1] * z[0] + meta[2] * z[1] + meta[3] * z[2]) - Y[i];
            g[0] += e;
            for (let j = 0; j < 3; j++) g[j + 1] += e * z[j];
          });
          for (let j = 0; j < 4; j++) meta[j] -= 1.2 * (g[j] / Z.length + (j ? 0.01 * meta[j] : 0));
        }
        const ac = (f) =>
          te.length
            ? te.filter((p) => (f(p.x, p.y) >= 0.5 ? 1 : 0) === p.c).length / te.length
            : NaN;
        accs = [ac(base[0]), ac(base[1]), ac(base[2]), ac(stack)];
        hold = [{}, {}, {}];
        ver++;
      },
      p: (x, y) =>
        !base ? 0.5 : S.view < 3 ? clamp(base[S.view](x, y), 0.001, 0.999) : stack(x, y),
      tick() {
        if (++fr % 170) return false;
        S.view = (S.view + 1) % 4;
        return true;
      },
      status() {
        if (!base) return '';
        return S.view < 3
          ? 'Showing base model ' +
              (S.view + 1) +
              ' of 3: ' +
              NM[S.view] +
              '. On its own it scores ' +
              pct(accs[S.view]) +
              ' on the test set.'
          : 'Showing the stacked model. The combiner computes ' +
              f2(meta[0]) +
              ' + ' +
              f2(meta[1]) +
              ' × (line) + ' +
              f2(meta[2]) +
              ' × (neighbours) + ' +
              f2(meta[3]) +
              ' × (tree) and squashes it into a probability. It scores ' +
              pct(accs[3]) +
              ' on the test set; the best base model scores ' +
              pct(Math.max(accs[0], accs[1], accs[2])) +
              '.';
      },
      cur() {
        return S.view < 3 ? 0 : 4;
      },
      right(t) {
        const { g, w, h } = fit(api.cr);
        if (!base) return;
        const sz = Math.min((w - 40) / 3 - 8, h * 0.52),
          y0 = 20,
          labs = ['Straight line', '7 neighbours', 'Small tree'];
        g.font = MONO;
        g.textAlign = 'center';
        const mxw = Math.max(0.5, ...meta.slice(1).map(Math.abs)),
          by = y0 + sz + 66,
          bh = h - by - 26;
        for (let k = 0; k < 3; k++) {
          const x = 20 + k * (sz + 18);
          miniField(g, x, y0, sz, base[k], t, hold[k], 30);
          if (S.view === k) {
            g.strokeStyle = t.ink;
            g.lineWidth = 3;
            g.strokeRect(x - 2, y0 - 2, sz + 4, sz + 4);
          }
          g.fillStyle = t.ink;
          g.fillText(labs[k], x + sz / 2, y0 - 6);
          g.fillStyle = t.muted;
          g.fillText('test ' + pct(accs[k]), x + sz / 2, y0 + sz + 14);
          const bw = sz * 0.5,
            v = meta[k + 1],
            bx = x + sz / 2 - bw / 2,
            zero = by + bh * 0.74,
            hh = (Math.abs(v) / mxw) * bh * 0.56;
          g.fillStyle = v >= 0 ? t.gold : t.ok;
          g.fillRect(bx, v >= 0 ? zero - hh : zero, bw, hh);
          g.fillStyle = t.ink;
          g.fillText(
            'weight ' + f2(v),
            x + sz / 2,
            (v >= 0 ? zero - hh : zero + hh) + (v >= 0 ? -6 : 14),
          );
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(x, zero);
          g.lineTo(x + sz, zero);
          g.stroke();
        }
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('how much the combiner relies on each:', 20, y0 + sz + 36);
      },
      read(set) {
        for (let k = 0; k < 4; k++) set('a' + k, pct(accs[k]));
        set('mw', meta.slice(1).map(f2).join(', '));
        root
          .querySelectorAll('#stacking-view button')
          .forEach((b) =>
            b.setAttribute('aria-pressed', +b.dataset.v === S.view ? 'true' : 'false'),
          );
      },
    };
  },
});
