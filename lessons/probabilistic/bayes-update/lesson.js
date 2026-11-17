/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  eqShow,
  f2,
  f3,
  fit,
  lgam,
  markTP,
  mix,
  playBtn,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#by-c', root),
      cr = $('#by-r', root),
      live = $('#by-live', root),
      eq = $('#by-eq', root),
      S = { p: 0.3, m: 0.5, n0: 2, pr: 'flat' },
      PR = { flat: [1, 1], mild: [5, 5], stub: [48, 12] },
      MAXN = 1000,
      NG = 400;
    let a0 = 1,
      b0 = 1,
      D = [],
      k = 0,
      HM = [],
      HL = [],
      HU = [],
      H3 = [[], [], []],
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      fin = false,
      bm,
      bn;
    const pdf = (a, b) => {
      const o = new Float64Array(NG + 1),
        lb = lgam(a) + lgam(b) - lgam(a + b);
      for (let i = 0; i <= NG; i++) {
        const x = clamp(i / NG, 1e-6, 1 - 1e-6);
        o[i] = Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lb);
      }
      return o;
    };
    const ci = (a, b) => {
      const d = pdf(a, b);
      let tot = 0;
      for (let i = 0; i <= NG; i++) tot += d[i];
      let s = 0,
        lo = 0,
        hi = 1,
        gl = false;
      for (let i = 0; i <= NG; i++) {
        s += d[i] / tot;
        if (!gl && s >= 0.025) {
          lo = i / NG;
          gl = true;
        }
        if (s >= 0.975) {
          hi = i / NG;
          break;
        }
      }
      return [lo, hi];
    };
    const push = () => {
      const n = D.length,
        a = a0 + k,
        b = b0 + n - k,
        [lo, hi] = ci(a, b);
      HM.push(a / (a + b));
      HL.push(lo);
      HU.push(hi);
      Object.values(PR).forEach((q, i) => H3[i].push((q[0] + k) / (q[0] + q[1] + n)));
    };
    const build = () => {
      D = [];
      k = 0;
      HM = [];
      HL = [];
      HU = [];
      H3 = [[], [], []];
      fin = false;
      cu = 0;
      push();
      msg =
        'The prior is Beta(' +
        f2(a0) +
        ', ' +
        f2(b0) +
        '): the owner’s best guess is ' +
        pct(a0 / (a0 + b0)) +
        ', held as firmly as if ' +
        f2(a0 + b0) +
        ' customers had already been asked. No real customer has answered yet.';
      eqShow(
        eq,
        ['No answers yet. The belief is the prior, Beta(' + f2(a0) + ', ' + f2(b0) + ').'],
        0,
      );
    };
    const add = (y) => {
      if (fin || D.length >= MAXN) return;
      const n0 = D.length,
        a = a0 + k,
        b = b0 + n0 - k;
      D.push(y);
      k += y;
      const a2 = a0 + k,
        b2 = b0 + D.length - k;
      push();
      const lo = HL[HL.length - 1],
        hi = HU[HU.length - 1];
      cu = y ? 2 : 2;
      eqShow(
        eq,
        [
          'Customer ' + D.length + ' says <b>' + (y ? 'yes' : 'no') + '</b>.',
          'belief before: Beta(' +
            f2(a) +
            ', ' +
            f2(b) +
            ')     best estimate ' +
            f2(a) +
            ' ÷ (' +
            f2(a) +
            ' + ' +
            f2(b) +
            ') = ' +
            f3(a / (a + b)),
          (y ? 'a yes adds 1 to the first number' : 'a no adds 1 to the second number') +
            ':   belief after: <b>Beta(' +
            f2(a2) +
            ', ' +
            f2(b2) +
            ')</b>',
          'best estimate ' +
            f2(a2) +
            ' ÷ (' +
            f2(a2) +
            ' + ' +
            f2(b2) +
            ') = <b>' +
            f3(a2 / (a2 + b2)) +
            '</b>     95% of the belief between ' +
            f2(lo) +
            ' and ' +
            f2(hi) +
            ' (width ' +
            f2(hi - lo) +
            ')',
          'for comparison, the plain frequency: ' + k + ' ÷ ' + D.length + ' = ' + f3(k / D.length),
        ],
        0,
      );
      msg =
        'Customer ' +
        D.length +
        ' said ' +
        (y ? 'yes' : 'no') +
        '. The best estimate moved from ' +
        pct(a / (a + b)) +
        ' to ' +
        pct(a2 / (a2 + b2)) +
        ', and 95% of the belief now lies between ' +
        pct(lo) +
        ' and ' +
        pct(hi) +
        '.';
      if (hi - lo < 0.1 || D.length >= MAXN) {
        fin = true;
        cu = 5;
        msg =
          (hi - lo < 0.1
            ? 'Stopped after ' +
              D.length +
              ' customers: 95% of the belief now fits in a range narrower than 0.1, from ' +
              pct(lo) +
              ' to ' +
              pct(hi) +
              '.'
            : 'Stopped at the limit of ' + MAXN + ' customers.') +
          ' Best estimate ' +
          pct(a2 / (a2 + b2)) +
          '. The true share is ' +
          pct(S.p) +
          ', which is ' +
          (S.p >= lo && S.p <= hi ? 'inside' : 'outside') +
          ' that range.';
      }
    };
    const tp = testPoint(root, 'by', ['A share to ask about'], [0.5], [[0, 1]]);
    const draw = () => {
      const t = T(),
        n = D.length,
        a = a0 + k,
        b = b0 + n - k;
      {
        const { g, w, h } = fit(c),
          m = { l: 14, r: 14, t: 40, b: 26 },
          X = (v) => m.l + v * (w - m.l - m.r),
          pr = pdf(a0, b0),
          po = pdf(a, b),
          lk = n ? pdf(k + 1, n - k + 1) : null,
          ym = Math.max(...po, ...pr.map((v) => Math.min(v, 6))) * 1.05,
          Y = (v) => h - m.b - (Math.min(v, ym) / ym) * (h - m.t - m.b),
          lo = HL[HL.length - 1],
          hi = HU[HU.length - 1];
        g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
        g.beginPath();
        g.moveTo(X(lo), Y(0));
        for (let i = 0; i <= NG; i++) {
          const x = i / NG;
          if (x >= lo && x <= hi) g.lineTo(X(x), Y(po[i]));
        }
        g.lineTo(X(hi), Y(0));
        g.closePath();
        g.fill();
        const line = (d, col, lw, dash) => {
          g.strokeStyle = col;
          g.lineWidth = lw;
          g.setLineDash(dash || []);
          g.beginPath();
          for (let i = 0; i <= NG; i++) i ? g.lineTo(X(i / NG), Y(d[i])) : g.moveTo(X(0), Y(d[i]));
          g.stroke();
          g.setLineDash([]);
        };
        if (lk) line(lk, t.ok, 1.3);
        line(pr, t.muted, 1.8, [6, 5]);
        line(po, t.accent, 2.6);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h - m.b);
        g.lineTo(w - m.r, h - m.b);
        g.stroke();
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) g.fillText(100 * v + '%', X(v), h - 8);
        g.setLineDash([4, 4]);
        g.strokeStyle = t.gold;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(X(S.p), m.t - 6);
        g.lineTo(X(S.p), h - m.b);
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = t.cut;
        g.textAlign = S.p > 0.8 ? 'right' : 'left';
        g.fillText('true share', X(S.p) + (S.p > 0.8 ? -5 : 5), m.t + 4);
        markTP(g, X(tp.v[0]), h - m.b, t, false);
        const show = D.slice(-Math.floor((w - 28) / 9)),
          x0 = 14;
        show.forEach((y, i) => {
          g.beginPath();
          g.arc(x0 + i * 9 + 4, 14, 3.4, 0, 7);
          if (y) {
            g.fillStyle = t.accent;
            g.fill();
          } else {
            g.strokeStyle = t.muted;
            g.lineWidth = 1.2;
            g.stroke();
          }
        });
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          n
            ? 'answers, oldest to newest: filled = yes, open = no' +
                (n > show.length ? ' (the last ' + show.length + ')' : '')
            : 'no answers yet',
          14,
          31,
        );
      }
      {
        const { g, w, h } = fit(cr),
          nn = Math.max(30, HM.length),
          A = uLine(g, t, 0, 0, w, h / 2, HM, {
            title: 'this owner: best estimate and 95% range, by customers asked',
            n: nn,
            ym: 1,
            col: t.accent,
          });
        g.fillStyle = mix(t.accent, t.accent, 0, 0.2);
        g.beginPath();
        HU.forEach((v, i) => (i ? g.lineTo(A.X(i), A.Y(v)) : g.moveTo(A.X(i), A.Y(v))));
        for (let i = HL.length - 1; i >= 0; i--) g.lineTo(A.X(i), A.Y(HL[i]));
        g.closePath();
        g.fill();
        const tl = (L, y0) => {
          g.setLineDash([4, 4]);
          g.strokeStyle = t.gold;
          g.lineWidth = 1.4;
          g.beginPath();
          g.moveTo(L.m.l, L.Y(S.p));
          g.lineTo(w - L.m.r, L.Y(S.p));
          g.stroke();
          g.setLineDash([]);
        };
        tl(A);
        const col = [t.accent, t.ok, t.cut],
          nm = ['no opinion', 'mild hunch', 'stubborn'];
        let B = null;
        H3.forEach((d, i) => {
          B = uLine(g, t, 0, h / 2, w, h / 2, d, {
            title: i ? '' : 'three owners, same answers: their best estimates',
            n: nn,
            ym: 1,
            col: col[i],
          });
        });
        tl(B);
        g.font = MONO;
        H3.forEach((d, i) => {
          g.fillStyle = col[i];
          g.textAlign = 'right';
          g.fillText(nm[i], w - 14, h / 2 + 38 + i * 13);
        });
        g.fillStyle = t.muted;
        g.fillText('customers asked →', w - 12, h - 5);
      }
      live.textContent = msg;
      uCur(root, 'by', cu);
      const po = pdf(a, b);
      let tot = 0,
        ab = 0;
      for (let i = 0; i <= NG; i++) {
        tot += po[i];
        if (i / NG > tp.v[0]) ab += po[i];
      }
      tp.out.textContent =
        'Given everything so far, the probability that the true share is above ' +
        pct(tp.v[0]) +
        ' is ' +
        pct(ab / tot) +
        '. The probability that the next customer says yes is ' +
        pct(a / (a + b)) +
        '.';
      setR(root, 'n', n + (fin ? ' (stopped)' : ''));
      setR(root, 'k', String(k));
      setR(root, 'mean', pct(a / (a + b)));
      setR(root, 'ci', pct(HL[HL.length - 1]) + ' and ' + pct(HU[HU.length - 1]));
      setR(root, 'fr', n ? pct(k / n) : '–');
      setR(root, 'pb', 'Beta(' + f2(a) + ', ' + f2(b) + ')');
    };
    tp.on = () => draw();
    build();
    if (RM) {
      while (!fin) add(Math.random() < S.p ? 1 : 0);
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      if (fr % (D.length < 12 ? 26 : D.length < 60 ? 7 : 2)) return;
      add(Math.random() < S.p ? 1 : 0);
      draw();
      chk();
    });
    pb = playBtn($('#by-play', root), anim);
    chk = runCtl(pb, root, 'by', {
      done: () => fin,
      restart: () => build(),
      count: () => D.length,
      what: 'customers',
    });
    $('#by-step', root).onclick = () => {
      pb.stop();
      if (fin) build();
      else add(Math.random() < S.p ? 1 : 0);
      draw();
    };
    $('#by-reset', root).onclick = () => {
      pb.stop();
      build();
      draw();
    };
    $('#by-yes', root).onclick = () => {
      pb.stop();
      add(1);
      draw();
    };
    $('#by-no', root).onclick = () => {
      pb.stop();
      add(0);
      draw();
    };
    const setP = () => {
      a0 = S.m * S.n0;
      b0 = (1 - S.m) * S.n0;
      if (pb) pb.stop();
      build();
      draw();
    };
    bm = bind(
      root,
      'by-m',
      (v) => {
        S.m = +v;
        setP();
      },
      f2,
    );
    bn = bind(
      root,
      'by-n',
      (v) => {
        S.n0 = +v;
        setP();
      },
      (v) => String(v),
    );
    bind(
      root,
      'by-p',
      (v) => {
        S.p = +v;
        if (pb) {
          pb.stop();
          build();
          draw();
        }
      },
      f2,
    );
    segBind(root, 'by-pr', S.pr, (v) => {
      const q = PR[v];
      S.m = q[0] / (q[0] + q[1]);
      S.n0 = q[0] + q[1];
      bm.set(S.m);
      bn.set(S.n0);
      a0 = q[0];
      b0 = q[1];
      pb.stop();
      build();
      draw();
      pb.start();
    });
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        build();
        draw();
        pb.start();
      },
    };
  },
};
