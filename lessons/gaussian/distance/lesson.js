/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  SANS,
  T,
  bind,
  clamp,
  drawMap,
  eig,
  f2,
  grab,
  playBtn,
  setR,
  shapeBind,
} from '../../../js/core/shared.js';

export default {
  init(root) {
    const c = $('#md-c', root),
      G8 = { sx: 1.7, sy: 0.85, rho: 0.72 },
      S = { t: 0, A: null, B: null };
    let Wm,
      last = null;
    const prep = () => {
      const e = eig(G8),
        c1 = Math.cos(e.th),
        s1 = Math.sin(e.th),
        i1 = 1 / Math.sqrt(e.l1),
        i2 = 1 / Math.sqrt(e.l2);
      Wm = [c1 * c1 * i1 + s1 * s1 * i2, c1 * s1 * (i1 - i2), s1 * s1 * i1 + c1 * c1 * i2];
      return [
        [2 * c1, 2 * s1],
        [-2 * s1, 2 * c1],
      ];
    };
    const place = () => {
      const p = prep();
      S.A = p[0];
      S.B = p[1];
    };
    place();
    const M = () => [1 - S.t + S.t * Wm[0], S.t * Wm[1], 1 - S.t + S.t * Wm[2]];
    const mah = (p) => Math.hypot(Wm[0] * p[0] + Wm[1] * p[1], Wm[1] * p[0] + Wm[2] * p[1]);
    const draw = () => {
      prep();
      const t = T(),
        m = M(),
        tr = (x, y) => [m[0] * x + m[1] * y, m[1] * x + m[2] * y],
        D = drawMap(c, t, G8, { tr }),
        g = D.g;
      last = D;
      const ra = Math.hypot(S.A[0], S.A[1]);
      g.setLineDash([5, 4]);
      g.strokeStyle = t.ink;
      g.lineWidth = 1.3;
      g.beginPath();
      for (let k = 0; k <= 72; k++) {
        const th = (k / 72) * 2 * Math.PI,
          p = D.X(ra * Math.cos(th), ra * Math.sin(th));
        k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]);
      }
      g.stroke();
      g.setLineDash([]);
      g.lineWidth = 2.6;
      g.strokeStyle = t.ok;
      D.ring(mah(S.A));
      g.stroke();
      g.strokeStyle = t.gold;
      D.ring(mah(S.B));
      g.stroke();
      const o = D.X(0, 0);
      [
        [S.A, t.ok, 'A'],
        [S.B, t.gold, 'B'],
      ].forEach(([p, col, nm]) => {
        const q = D.X(p[0], p[1]);
        g.strokeStyle = col;
        g.lineWidth = 2;
        D.seg(o, q);
        g.fillStyle = col;
        g.beginPath();
        g.arc(q[0], q[1], 9, 0, 7);
        g.fill();
        g.strokeStyle = t.bg;
        g.lineWidth = 2.5;
        g.stroke();
        g.fillStyle = t.ink;
        g.font = SANS;
        g.textAlign = 'left';
        g.fillText(nm, q[0] + 12, q[1] - 10);
      });
      const ma = mah(S.A),
        mb = mah(S.B);
      setR(root, 'ae', f2(Math.hypot(S.A[0], S.A[1])));
      setR(root, 'am', f2(ma));
      setR(root, 'be', f2(Math.hypot(S.B[0], S.B[1])));
      setR(root, 'bm', f2(mb));
      setR(root, 'pa', (100 * (1 - Math.exp((-ma * ma) / 2))).toFixed(1) + '%');
      setR(root, 'pb', (100 * (1 - Math.exp((-mb * mb) / 2))).toFixed(1) + '%');
    };
    const ts = bind(root, 'md-t', (v) => {
      S.t = v;
      draw();
    });
    let ph = 0;
    shapeBind(root, 'md', G8, () => draw());
    const anim = Anim(() => {
      ph += 1 / 60 / 9;
      const u = (1 - Math.cos(ph * 2 * Math.PI)) / 2;
      S.t = clamp(u * 1.3 - 0.15, 0, 1);
      ts.set(S.t);
      draw();
    });
    const pb = playBtn($('#md-play', root), anim);
    grab(ts.el, pb);
    $('#md-reset', root).onclick = () => {
      place();
      draw();
    };
    let drag = null;
    const toData = (ev) => {
      const r = c.getBoundingClientRect(),
        u = (ev.clientX - r.left - last.w / 2) / last.s,
        v = -(ev.clientY - r.top - last.h / 2) / last.s,
        m = M(),
        det = m[0] * m[2] - m[1] * m[1];
      return [
        clamp((m[2] * u - m[1] * v) / det, -3.9, 3.9),
        clamp((-m[1] * u + m[0] * v) / det, -3.9, 3.9),
      ];
    };
    c.addEventListener('pointerdown', (ev) => {
      if (!last) return;
      const r = c.getBoundingClientRect(),
        px = ev.clientX - r.left,
        py = ev.clientY - r.top,
        a = last.X(S.A[0], S.A[1]),
        b = last.X(S.B[0], S.B[1]),
        da = Math.hypot(a[0] - px, a[1] - py),
        db = Math.hypot(b[0] - px, b[1] - py);
      drag = da <= db ? 'A' : 'B';
      S[drag] = toData(ev);
      c.setPointerCapture(ev.pointerId);
      draw();
    });
    c.addEventListener('pointermove', (ev) => {
      if (drag) {
        S[drag] = toData(ev);
        draw();
      }
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      c.addEventListener(n, () => {
        drag = null;
      }),
    );
    return { draw, stop: pb.stop, auto: pb.start };
  },
};
