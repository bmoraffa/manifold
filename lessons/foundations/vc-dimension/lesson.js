/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  SANS,
  T,
  bind,
  clamp,
  fit,
  mix,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { canvasXY, dotAt, segBind, uCur, uLine, uView } from '../../unsupervised/shared.js';
import { g2Note } from '../../generative/shared.js';

export default {
  init(root) {
    const c = $('#vc-c', root),
      cr = $('#vc-r', root),
      live = $('#vc-live', root),
      S = { h: 'line', n: 3, sp: 2 },
      VC = { int: 2, line: 3, rect: 4 },
      GOOD = {
        int: [
          [-2, 0],
          [1, 0],
          [2.6, 0],
          [-3, 0],
          [0, 0],
          [3.4, 0],
        ],
        line: [
          [-1.8, -1],
          [1.9, -0.8],
          [0.1, 1.6],
          [2.4, 1.4],
          [-2.6, 1.2],
          [0, -2.2],
        ],
        rect: [
          [0, 1.8],
          [2.2, 0],
          [0, -1.8],
          [-2.2, 0],
          [0.2, 0.1],
          [1.4, 1.3],
        ],
      };
    let P = [],
      k = 0,
      res = [],
      hyp = null,
      fin = false,
      msg = '',
      cu = 0,
      pb,
      chk = () => false,
      V = null,
      drag = -1,
      fr = 0,
      firstFail = -1;
    const lab = (m, i) => (m >> i) & 1;
    const find = (m) => {
      const n = S.n,
        pos = [],
        neg = [];
      for (let i = 0; i < n; i++) (lab(m, i) ? pos : neg).push(P[i]);
      if (S.h === 'int') {
        if (!pos.length) return { t: 'int', a: 9, b: 9 };
        const a = Math.min(...pos.map((p) => p[0])),
          b = Math.max(...pos.map((p) => p[0]));
        return neg.some((p) => p[0] >= a - 1e-9 && p[0] <= b + 1e-9)
          ? null
          : { t: 'int', a: a - 0.25, b: b + 0.25 };
      }
      if (S.h === 'rect') {
        if (!pos.length) return { t: 'rect', none: true };
        const x0 = Math.min(...pos.map((p) => p[0])),
          x1 = Math.max(...pos.map((p) => p[0])),
          y0 = Math.min(...pos.map((p) => p[1])),
          y1 = Math.max(...pos.map((p) => p[1]));
        return neg.some(
          (p) => p[0] >= x0 - 1e-9 && p[0] <= x1 + 1e-9 && p[1] >= y0 - 1e-9 && p[1] <= y1 + 1e-9,
        )
          ? null
          : { t: 'rect', x0: x0 - 0.22, x1: x1 + 0.22, y0: y0 - 0.22, y1: y1 + 0.22 };
      }
      if (!pos.length || !neg.length) return { t: 'line', nx: 1, ny: 0, c: pos.length ? -9 : 9 };
      let best = null,
        bg = 1e-6;
      for (let a = 0; a < 720; a++) {
        const th = (a * Math.PI) / 360,
          nx = Math.cos(th),
          ny = Math.sin(th),
          lo = Math.min(...pos.map((p) => p[0] * nx + p[1] * ny)),
          hi = Math.max(...neg.map((p) => p[0] * nx + p[1] * ny));
        if (lo - hi > bg) {
          bg = lo - hi;
          best = { t: 'line', nx, ny, c: (lo + hi) / 2 };
        }
      }
      return best;
    };
    const place = (kind) => {
      const n = 6;
      if (kind === 'good') P = GOOD[S.h].map((p) => p.slice());
      else if (kind === 'row')
        P = Array.from({ length: n }, (_, i) => [
          -3 + 1.2 * i,
          S.h === 'int' ? 0 : -1.2 + 0.48 * i,
        ]);
      else
        P = Array.from({ length: n }, () => [
          -3.3 + 6.6 * Math.random(),
          S.h === 'int' ? 0 : -2.3 + 4.6 * Math.random(),
        ]);
    };
    const reset = () => {
      k = 0;
      res = [];
      hyp = null;
      fin = false;
      firstFail = -1;
      cu = 0;
      msg =
        'There are ' +
        (1 << S.n) +
        ' ways to label ' +
        S.n +
        ' point' +
        (S.n === 1 ? '' : 's') +
        '. Press Run to try them all, or Next labelling to go one at a time.';
    };
    const str = (m) => Array.from({ length: S.n }, (_, i) => (lab(m, i) ? '+' : '−')).join(' ');
    const step = () => {
      if (fin) return;
      const tot = 1 << S.n;
      hyp = find(k);
      res.push(!!hyp);
      if (!hyp && firstFail < 0) firstFail = k;
      cu = 2;
      msg =
        'Labelling ' +
        (k + 1) +
        ' of ' +
        tot +
        ' (' +
        str(k) +
        '): ' +
        (hyp ? 'matched.' : 'no member of the family can do this.');
      k++;
      if (k >= tot) {
        fin = true;
        const ok = res.filter(Boolean).length,
          sh = ok === tot;
        cu = sh ? 3 : 4;
        msg =
          'All ' +
          tot +
          ' labellings tried: ' +
          ok +
          ' matched. ' +
          (sh
            ? 'These ' +
              S.n +
              ' points are shattered, so the VC dimension is at least ' +
              S.n +
              '.' +
              (S.n === VC[S.h]
                ? ' It is exactly ' +
                  S.n +
                  ': no arrangement of ' +
                  (S.n + 1) +
                  ' points can be shattered.'
                : '')
            : 'These points are not shattered. ' +
              (S.n <= VC[S.h]
                ? 'Another arrangement of ' + S.n + ' points can be: try A good arrangement.'
                : 'No arrangement of ' +
                  S.n +
                  ' points can be, because the VC dimension of this family is ' +
                  VC[S.h] +
                  '.'));
      }
    };
    const draw = () => {
      const t = T();
      V = uView(c);
      const { g, w, h, X, Y } = V,
        m = fin ? (firstFail >= 0 ? firstFail : k - 1) : Math.max(0, k - 1),
        show = fin && firstFail >= 0 ? null : hyp;
      if (S.h === 'int') {
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X(-4), Y(0));
        g.lineTo(X(4), Y(0));
        g.stroke();
      }
      if (show && k > 0) {
        g.fillStyle = mix(t.gold, t.gold, 0, 0.16);
        g.strokeStyle = t.gold;
        g.lineWidth = 2;
        if (show.t === 'int') {
          if (show.a < 5) {
            g.fillRect(X(show.a), Y(0.5), X(show.b) - X(show.a), Y(-0.5) - Y(0.5));
            g.strokeRect(X(show.a), Y(0.5), X(show.b) - X(show.a), Y(-0.5) - Y(0.5));
          }
        } else if (show.t === 'rect') {
          if (!show.none) {
            g.fillRect(X(show.x0), Y(show.y1), X(show.x1) - X(show.x0), Y(show.y0) - Y(show.y1));
            g.strokeRect(X(show.x0), Y(show.y1), X(show.x1) - X(show.x0), Y(show.y0) - Y(show.y1));
          }
        } else {
          const { nx, ny, c: cc } = show,
            px = nx * cc,
            py = ny * cc,
            L = 12;
          g.beginPath();
          g.moveTo(X(px - ny * L), Y(py + nx * L));
          g.lineTo(X(px + ny * L), Y(py - nx * L));
          g.lineTo(X(px + ny * L + nx * L), Y(py - nx * L + ny * L));
          g.lineTo(X(px - ny * L + nx * L), Y(py + nx * L + ny * L));
          g.closePath();
          g.fill();
          g.beginPath();
          g.moveTo(X(px - ny * L), Y(py + nx * L));
          g.lineTo(X(px + ny * L), Y(py - nx * L));
          g.stroke();
        }
      }
      for (let i = 0; i < S.n; i++) {
        const on = k > 0 || fin ? lab(m, i) : 0;
        g.lineWidth = 2.5;
        dotAt(g, X(P[i][0]), Y(P[i][1]), 10, k > 0 ? (on ? t.gold : t.ink) : t.muted, t.bg);
        g.font = SANS;
        g.fillStyle = t.bg;
        g.textAlign = 'center';
        g.fillText(k > 0 ? (on ? '+' : '−') : String(i + 1), X(P[i][0]), Y(P[i][1]) + 4.5);
      }
      g2Note(
        g,
        t,
        h,
        k === 0
          ? 'drag the points, then press Run'
          : fin && firstFail >= 0
            ? 'showing the first labelling that cannot be matched: ' + str(firstFail)
            : hyp
              ? 'gold area: the positive side of the matching model'
              : 'no model matches this labelling',
      );
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.5,
          tot = 1 << S.n,
          cols = tot <= 8 ? tot : tot <= 16 ? 8 : tot <= 32 ? 8 : 16,
          rows = Math.ceil(tot / cols),
          tw = Math.min(46, (w - 40) / cols),
          th = Math.min(34, (h1 - 34) / rows),
          ox = (w - cols * tw) / 2,
          oy = 26;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('all ' + tot + ' labellings', 8, 13);
        for (let q = 0; q < tot; q++) {
          const x = ox + (q % cols) * tw,
            y = oy + Math.floor(q / cols) * th,
            dn = q < res.length;
          g.strokeStyle = q === k - 1 && !fin ? t.gold : t.line;
          g.lineWidth = q === k - 1 && !fin ? 2.5 : 1;
          g.fillStyle = dn
            ? res[q]
              ? mix(t.accent, t.accent, 0, 0.55)
              : mix(t.cut, t.cut, 0, 0.2)
            : 'transparent';
          g.fillRect(x + 2, y + 2, tw - 4, th - 4);
          g.strokeRect(x + 2, y + 2, tw - 4, th - 4);
          if (dn && !res[q]) {
            g.strokeStyle = t.cut;
            g.lineWidth = 2;
            g.beginPath();
            g.moveTo(x + 6, y + 6);
            g.lineTo(x + tw - 6, y + th - 6);
            g.moveTo(x + tw - 6, y + 6);
            g.lineTo(x + 6, y + th - 6);
            g.stroke();
          }
          if (tw > 30) {
            for (let i = 0; i < S.n; i++)
              dotAt(
                g,
                x + tw / 2 + (i - (S.n - 1) / 2) * Math.min(7, (tw - 10) / S.n),
                y + th / 2,
                2,
                lab(q, i) ? t.gold : t.ink,
              );
          }
        }
        const gf = (n) =>
            S.h === 'int'
              ? (n * (n - 1)) / 2 + n + 1
              : S.h === 'line'
                ? n <= 1
                  ? 2
                  : n * n - n + 2
                : [0, 1, 2, 3, 4].reduce(
                    (s, i) =>
                      s +
                      (i <= n
                        ? [
                            1,
                            n,
                            (n * (n - 1)) / 2,
                            (n * (n - 1) * (n - 2)) / 6,
                            (n * (n - 1) * (n - 2) * (n - 3)) / 24,
                          ][i]
                        : 0),
                    0,
                  ),
          NS = [1, 2, 3, 4, 5, 6, 7, 8],
          A = uLine(g, t, 0, h1, w, h - h1, [], {
            title:
              'labellings on n points, n = 1 to 8 (log scale): family (solid), all 2ⁿ (dashed)',
            n: 8,
            ym: 8,
            y1: 1,
          });
        g.strokeStyle = t.cut;
        g.lineWidth = 1.6;
        g.setLineDash([6, 4]);
        g.beginPath();
        NS.forEach((n, i) => (i ? g.lineTo(A.X(i), A.Y(n)) : g.moveTo(A.X(i), A.Y(n))));
        g.stroke();
        g.setLineDash([]);
        g.strokeStyle = t.accent;
        g.lineWidth = 2.4;
        g.beginPath();
        NS.forEach((n, i) => {
          const v = Math.log2(Math.min(gf(n), 2 ** n));
          i ? g.lineTo(A.X(i), A.Y(v)) : g.moveTo(A.X(i), A.Y(v));
        });
        g.stroke();
        dotAt(g, A.X(S.n - 1), A.Y(Math.log2(Math.min(gf(S.n), 2 ** S.n))), 5, t.gold, t.bg);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.setLineDash([2, 4]);
        g.beginPath();
        g.moveTo(A.X(VC[S.h] - 1), A.Y(1));
        g.lineTo(A.X(VC[S.h] - 1), A.Y(8));
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('VC dimension', A.X(VC[S.h] - 1) + 5, A.Y(7.6));
      }
      const ok = res.filter(Boolean).length;
      live.textContent = msg;
      uCur(root, 'vc', cu);
      setR(root, 'n', S.n);
      setR(root, 'done', res.length + ' of ' + (1 << S.n) + (fin ? ' (finished)' : ''));
      setR(root, 'ok', String(ok));
      setR(root, 'sh', fin ? (ok === 1 << S.n ? 'Yes' : 'No') : '–');
      setR(root, 'vc', VC[S.h]);
      setR(root, 'fail', firstFail >= 0 ? str(firstFail) : fin ? 'none' : '–');
    };
    place('good');
    reset();
    if (RM) {
      while (!fin) step();
    }
    const anim = Anim(() => {
      if (fin) {
        pb.stop();
        return;
      }
      if (++fr % (S.sp === 1 ? 40 : S.sp === 2 ? (S.n > 4 ? 5 : 14) : 2)) return;
      step();
      draw();
      chk();
    });
    pb = playBtn($('#vc-play', root), anim);
    chk = runCtl(pb, root, 'vc', {
      done: () => fin,
      restart: reset,
      count: () => k,
      what: 'labellings',
    });
    const go = () => {
      pb.stop();
      reset();
      draw();
      fr = 0;
      pb.start();
    };
    $('#vc-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#vc-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    $('#vc-new', root).onclick = () => {
      place('rand');
      go();
    };
    $('#vc-row', root).onclick = () => {
      place('row');
      go();
    };
    $('#vc-best', root).onclick = () => {
      place('good');
      go();
    };
    bind(
      root,
      'vc-n',
      (v) => {
        S.n = v;
        go();
      },
      (v) => String(+v),
    );
    bind(
      root,
      'vc-sp',
      (v) => {
        S.sp = v;
      },
      (v) => ['', 'slow', 'steady', 'fast'][+v],
    );
    segBind(root, 'vc-h', S.h, (v) => {
      S.h = v;
      place('good');
      go();
    });
    c.style.cursor = 'grab';
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', (e) => {
      if (!V) return;
      const [x, y] = canvasXY(c, V, e);
      drag = P.slice(0, S.n).findIndex((p) => Math.hypot((p[0] - x) * V.s, (p[1] - y) * V.s) < 16);
      if (drag >= 0) {
        c.setPointerCapture(e.pointerId);
        pb.stop();
      }
    });
    c.addEventListener('pointermove', (e) => {
      if (drag < 0 || !V) return;
      const [x, y] = canvasXY(c, V, e);
      P[drag] = [clamp(x, -3.7, 3.7), S.h === 'int' ? 0 : clamp(y, -2.7, 2.7)];
      reset();
      while (!fin) step();
      draw();
    });
    const end = () => {
      drag = -1;
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        fr = 0;
        pb.start();
      },
    };
  },
};
