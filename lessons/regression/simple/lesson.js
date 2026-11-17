/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  clipPlot,
  f2,
  f3,
  fit,
  markTP,
  mix,
  playBtn,
  plot,
  ramp,
  randnFrom,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { parseXY, pfmt, tcrit, tp2 } from '../shared.js';

export default {
  init(root) {
    const c = $('#sr-c', root),
      cl = $('#sr-l', root),
      st = $('#sr-status', root),
      note = $('#sr-note', root);
    const S = { m: 0.8, b: 0.5, noise: 1, n: 40, lr: 0.08, res: true, sq: false },
      XR = [-3.5, 3.5],
      YR = [-7, 7],
      BR = [-3, 3],
      AR = [-5, 5];
    let seed = 3,
      pts = [],
      b0 = -3.5,
      b1 = -2,
      path = [],
      wait = 0,
      fr = 0,
      pb,
      PS = null,
      LM = null;
    const gen = () => {
      const r = rng(seed),
        rn = randnFrom(r);
      pts = [];
      for (let i = 0; i < S.n; i++) {
        const x = r() * 6 - 3;
        pts.push({ x, y: S.b + S.m * x + S.noise * rn(), test: r() < 0.3 });
      }
      if (pts.filter((p) => !p.test).length < 3)
        pts.forEach((p, i) => {
          if (i < 3) p.test = false;
        });
    };
    const mom = (P) => {
      let n = P.length,
        sx = 0,
        sy = 0,
        sxx = 0,
        sxy = 0,
        syy = 0;
      for (const p of P) {
        sx += p.x;
        sy += p.y;
        sxx += p.x * p.x;
        sxy += p.x * p.y;
        syy += p.y * p.y;
      }
      n = Math.max(1, n);
      return { n: P.length, mx: sx / n, my: sy / n, mxx: sxx / n, mxy: sxy / n, myy: syy / n };
    };
    const mse = (M, a, b) =>
      a * a + 2 * a * b * M.mx + b * b * M.mxx - 2 * a * M.my - 2 * b * M.mxy + M.myy;
    const ols = (M) => {
      const v = M.mxx - M.mx * M.mx,
        s = v > 1e-12 ? (M.mxy - M.mx * M.my) / v : 0;
      return [M.my - s * M.mx, s];
    };
    const begin = (a, b) => {
      b0 = a;
      b1 = b;
      path = [[a, b]];
      wait = 0;
      note.textContent = '';
    };
    const step = () => {
      const M = mom(pts.filter((p) => !p.test)),
        g0 = 2 * (b0 + b1 * M.mx - M.my),
        g1 = 2 * (b0 * M.mx + b1 * M.mxx - M.mxy);
      b0 -= S.lr * g0;
      b1 -= S.lr * g1;
      if (!isFinite(b0) || Math.abs(b0) > 40 || Math.abs(b1) > 40) {
        note.textContent =
          'The learning rate is too large: every step overshoots the valley and the loss grows. Lower it and run again.';
        b0 = clamp(b0 || 0, -40, 40);
        b1 = clamp(b1 || 0, -40, 40);
        path.push([b0, b1]);
        return 'div';
      }
      path.push([b0, b1]);
      if (path.length > 600) path.shift();
      return Math.hypot(g0, g1) < 2e-3 ? 'done' : 'go';
    };
    gen();
    begin(b0, b1);
    const tp = testPoint(root, 'sr', ['Your x'], [1.5], [[-3.5, 3.5]]);
    tp.on = () => draw();
    const draw = () => {
      const t = T(),
        trn = pts.filter((p) => !p.test),
        tst = pts.filter((p) => p.test),
        M = mom(trn),
        Mt = mom(tst),
        best = ols(M);
      {
        const { g, w, h } = fit(c),
          P = plot(g, w, h, XR, YR, t, { yticks: [-6, -3, 0, 3, 6], xlabel: 'x' });
        PS = P;
        clipPlot(g, P);
        if (S.res)
          for (const p of trn) {
            const yh = b0 + b1 * p.x;
            if (S.sq) {
              const d = Math.abs(P.Y(yh) - P.Y(p.y));
              g.fillStyle = mix(t.ok, t.ok, 0, 0.13);
              g.fillRect(P.X(p.x), Math.min(P.Y(yh), P.Y(p.y)), d, d);
            }
            g.strokeStyle = mix(t.ok, t.ok, 0, 0.75);
            g.lineWidth = 1.3;
            g.beginPath();
            g.moveTo(P.X(p.x), P.Y(p.y));
            g.lineTo(P.X(p.x), P.Y(yh));
            g.stroke();
          }
        g.strokeStyle = t.gold;
        g.lineWidth = 3.2;
        g.beginPath();
        g.moveTo(P.X(XR[0]), P.Y(b0 + b1 * XR[0]));
        g.lineTo(P.X(XR[1]), P.Y(b0 + b1 * XR[1]));
        g.stroke();
        g.strokeStyle = t.muted;
        g.lineWidth = 1.3;
        for (const p of tst) {
          g.beginPath();
          g.arc(P.X(p.x), P.Y(p.y), 4, 0, 7);
          g.stroke();
        }
        g.fillStyle = t.accent;
        for (const p of trn) {
          g.beginPath();
          g.arc(P.X(p.x), P.Y(p.y), 4.2, 0, 7);
          g.fill();
        }
        markTP(g, P.X(tp.v[0]), P.Y(b0 + b1 * tp.v[0]), t);
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.muted;
        g.fillText('filled = train, hollow = test', P.m.l + 8, P.m.t + 8);
      }
      {
        const { g, w, h } = fit(cl),
          m = { l: 44, r: 10, t: 10, b: 30 },
          X = (v) => m.l + ((v - BR[0]) / (BR[1] - BR[0])) * (w - m.l - m.r),
          Y = (v) => h - m.b - ((v - AR[0]) / (AR[1] - AR[0])) * (h - m.t - m.b);
        LM = { m, w, h };
        const lmin = mse(M, best[0], best[1]),
          lmax = Math.max(
            mse(M, AR[0], BR[0]),
            mse(M, AR[0], BR[1]),
            mse(M, AR[1], BR[0]),
            mse(M, AR[1], BR[1]),
          ),
          NC = 44,
          cw = (w - m.l - m.r) / NC,
          ch = (h - m.t - m.b) / NC;
        for (let i = 0; i < NC; i++)
          for (let j = 0; j < NC; j++) {
            const s = BR[0] + ((i + 0.5) / NC) * (BR[1] - BR[0]),
              a = AR[0] + ((j + 0.5) / NC) * (AR[1] - AR[0]),
              v = clamp((mse(M, a, s) - lmin) / (lmax - lmin || 1), 0, 1);
            g.fillStyle = ramp(t, Math.pow(1 - Math.sqrt(v), 1.6));
            g.fillRect(m.l + i * cw - 0.3, h - m.b - (j + 1) * ch - 0.3, cw + 0.6, ch + 0.6);
          }
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        for (let v = -3; v <= 3; v++) g.fillText(String(v), X(v), h - m.b + 14);
        g.fillText('slope', w / 2 + m.l / 2, h - 3);
        g.textAlign = 'right';
        for (let v = -4; v <= 4; v += 2) g.fillText(String(v), m.l - 6, Y(v) + 4);
        g.save();
        g.translate(11, h / 2);
        g.rotate(-Math.PI / 2);
        g.textAlign = 'center';
        g.fillText('intercept', 0, 0);
        g.restore();
        g.save();
        g.beginPath();
        g.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
        g.clip();
        g.strokeStyle = t.ink;
        g.lineWidth = 1.6;
        g.beginPath();
        path.forEach((q, i) => (i ? g.lineTo(X(q[1]), Y(q[0])) : g.moveTo(X(q[1]), Y(q[0]))));
        g.stroke();
        g.fillStyle = t.ink;
        for (const q of path.slice(-60)) {
          g.beginPath();
          g.arc(X(q[1]), Y(q[0]), 1.8, 0, 7);
          g.fill();
        }
        const bx = X(best[1]),
          by = Y(best[0]);
        g.strokeStyle = t.ink;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(bx - 7, by);
        g.lineTo(bx + 7, by);
        g.moveTo(bx, by - 7);
        g.lineTo(bx, by + 7);
        g.stroke();
        g.fillStyle = t.ok;
        g.strokeStyle = t.bg;
        g.lineWidth = 2;
        g.beginPath();
        g.arc(X(b1), Y(b0), 7, 0, 7);
        g.fill();
        g.stroke();
        g.restore();
        g.fillStyle = t.ink;
        g.textAlign = 'left';
        g.fillText(
          '+ lowest loss',
          clamp(bx + 10, m.l + 4, w - 110),
          clamp(by - 9, m.t + 12, h - m.b - 6),
        );
      }
      const n = M.n,
        sxx = n * (M.mxx - M.mx * M.mx),
        rss = n * mse(M, best[0], best[1]),
        tss = n * (M.myy - M.my * M.my);
      setR(root, 'k', String(path.length - 1));
      setR(root, 'cur', f2(b0) + ', ' + f2(b1));
      setR(root, 'tr', f3(mse(M, b0, b1)));
      setR(root, 'te', Mt.n ? f3(mse(Mt, b0, b1)) : '–');
      setR(root, 'best', f2(best[0]) + ', ' + f2(best[1]));
      setR(root, 'r2', tss > 1e-12 ? f3(1 - rss / tss) : '–');
      if (n > 2 && sxx > 1e-9) {
        const se = Math.sqrt(rss / (n - 2) / sxx),
          tv = se > 1e-12 ? best[1] / se : Infinity,
          tc = tcrit(n - 2);
        setR(root, 'se', f3(se));
        setR(root, 't', isFinite(tv) ? f2(tv) : '∞');
        setR(root, 'p', isFinite(tv) ? pfmt(tp2(tv, n - 2)) : '< 0.0001');
        setR(root, 'ci', '[' + f2(best[1] - tc * se) + ', ' + f2(best[1] + tc * se) + ']');
        const xq = tp.v[0],
          yq = best[0] + best[1] * xq,
          hw = tc * Math.sqrt((rss / (n - 2)) * (1 + 1 / n + (xq - M.mx) ** 2 / sxx));
        tp.out.textContent =
          'At x = ' +
          f2(xq) +
          ' the current line predicts y = ' +
          f2(b0 + b1 * xq) +
          '. The best-fit line predicts ' +
          f2(yq) +
          ', and a new observation there should fall between ' +
          f2(yq - hw) +
          ' and ' +
          f2(yq + hw) +
          ' (95% prediction interval).';
      } else ['se', 't', 'p', 'ci'].forEach((k) => setR(root, k, '–'));
    };
    let fin = false,
      nst = 0,
      chk = () => false;
    const anim = Anim(() => {
      fr++;
      if (fr % 3) return;
      const r = step();
      nst++;
      if (r === 'done') {
        fin = true;
        draw();
        pb.stop();
        note.textContent =
          'Converged after ' +
          nst +
          ' steps: the slope of the loss is essentially zero, so the line has stopped moving.';
        return;
      }
      if (r === 'div') {
        fin = true;
        draw();
        pb.stop();
        return;
      }
      draw();
      chk();
    });
    pb = playBtn($('#sr-play', root), anim);
    chk = runCtl(pb, root, 'sr', {
      done: () => fin,
      restart: () => {
        fin = false;
        nst = 0;
        note.textContent = '';
        begin(Math.random() * 8 - 4, Math.random() * 5 - 2.5);
      },
      count: () => nst,
    });
    $('#sr-play', root).addEventListener('click', () => {
      if (anim.on && (Math.abs(b0) >= 40 || Math.abs(b1) >= 40)) begin(-3.5, -2);
    });
    $('#sr-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#sr-jump', root).onclick = () => {
      pb.stop();
      const o = ols(mom(pts.filter((p) => !p.test)));
      b0 = o[0];
      b1 = o[1];
      path.push([b0, b1]);
      note.textContent = '';
      draw();
    };
    bind(root, 'sr-lr', (v) => {
      S.lr = v;
    });
    const regen = () => {
      st.textContent = '';
      gen();
      draw();
    };
    bind(root, 'sr-m', (v) => {
      S.m = v;
      regen();
    });
    bind(root, 'sr-b', (v) => {
      S.b = v;
      regen();
    });
    bind(root, 'sr-s', (v) => {
      S.noise = v;
      regen();
    });
    bind(
      root,
      'sr-n',
      (v) => {
        S.n = v;
        regen();
      },
      (v) => String(v),
    );
    $('#sr-new', root).onclick = () => {
      seed++;
      regen();
    };
    $('#sr-res', root).onchange = (e) => {
      S.res = e.target.checked;
      draw();
    };
    $('#sr-sq', root).onchange = (e) => {
      S.sq = e.target.checked;
      draw();
    };
    $('#sr-file', root).onchange = (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const fr2 = new FileReader();
      fr2.onerror = () => {
        st.textContent = 'That file could not be read. Try saving it again as plain CSV.';
      };
      fr2.onload = () => {
        const p = parseXY(fr2.result);
        if (typeof p === 'string') {
          st.textContent = p;
          return;
        }
        let x0 = 1 / 0,
          x1 = -1 / 0,
          y0 = 1 / 0,
          y1 = -1 / 0;
        for (const q of p) {
          x0 = Math.min(x0, q[0]);
          x1 = Math.max(x1, q[0]);
          y0 = Math.min(y0, q[1]);
          y1 = Math.max(y1, q[1]);
        }
        const r = rng(4);
        pts = p.map((q) => ({
          x: x1 > x0 ? -3 + (6 * (q[0] - x0)) / (x1 - x0) : 0,
          y: y1 > y0 ? -5 + (10 * (q[1] - y0)) / (y1 - y0) : 0,
          test: r() < 0.3,
        }));
        st.textContent =
          'Showing ' +
          pts.length +
          ' rows from ' +
          f.name.slice(0, 30) +
          ', rescaled to fit the plot.';
        begin(-3.5, -2);
        draw();
        pb.start();
      };
      fr2.readAsText(f);
      e.target.value = '';
    };
    let drag = -1;
    const at = (e) => {
      const r = c.getBoundingClientRect(),
        px = e.clientX - r.left,
        py = e.clientY - r.top;
      return {
        px,
        py,
        x: XR[0] + ((px - PS.m.l) / (PS.w - PS.m.l - PS.m.r)) * (XR[1] - XR[0]),
        y: YR[0] + ((PS.h - PS.m.b - py) / (PS.h - PS.m.t - PS.m.b)) * (YR[1] - YR[0]),
      };
    };
    c.addEventListener('pointerdown', (e) => {
      if (!PS) return;
      const a = at(e);
      let bi = -1,
        bd = 18;
      pts.forEach((p, i) => {
        const d = Math.hypot(PS.X(p.x) - a.px, PS.Y(p.y) - a.py);
        if (d < bd) {
          bd = d;
          bi = i;
        }
      });
      if (bi < 0) {
        pts.push({ x: clamp(a.x, -3.4, 3.4), y: clamp(a.y, -6.8, 6.8), test: false });
        bi = pts.length - 1;
      }
      drag = bi;
      c.setPointerCapture(e.pointerId);
      draw();
    });
    c.addEventListener('pointermove', (e) => {
      if (drag < 0) return;
      const a = at(e);
      pts[drag].x = clamp(a.x, -3.4, 3.4);
      pts[drag].y = clamp(a.y, -6.8, 6.8);
      draw();
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      c.addEventListener(n, () => {
        drag = -1;
      }),
    );
    cl.addEventListener('click', (e) => {
      if (!LM) return;
      const r = cl.getBoundingClientRect(),
        m = LM.m,
        s = BR[0] + ((e.clientX - r.left - m.l) / (LM.w - m.l - m.r)) * (BR[1] - BR[0]),
        a = AR[0] + ((LM.h - m.b - (e.clientY - r.top)) / (LM.h - m.t - m.b)) * (AR[1] - AR[0]);
      begin(clamp(a, AR[0], AR[1]), clamp(s, BR[0], BR[1]));
      draw();
      pb.start();
    });
    return { draw, stop: () => pb.stop(), auto: () => pb.start() };
  },
};
