/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  clamp,
  f2,
  fit,
  markTP,
  mix,
  playBtn,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../../js/core/shared.js';
import {
  UC,
  d2,
  kmRun,
  kmSeed,
  segBind,
  uCur,
  uDataBind,
  uFrame,
  uInv,
  uLine,
  uView,
} from '../shared.js';

export default {
  init(root) {
    const c = $('#km-c', root),
      cr = $('#km-r', root),
      live = $('#km-live', root),
      S = { K: 5, init: 'rand', ds: 'five', n: 250, seed: 3 },
      NR = 150;
    let P = [],
      C = [],
      prev = null,
      A = [],
      ph = 'seed',
      it = 0,
      hist = [],
      msg = '',
      run = 1,
      R = rng(1),
      mv = 0,
      wait = 0,
      fr = 0,
      pb,
      race = null,
      elbow = [],
      cu = -1,
      drag = -1,
      CH = [],
      tv = 0,
      tvd = -1;
    const near = (p) => {
      let b = 0,
        bd = 1e18;
      for (let k = 0; k < C.length; k++) {
        const d = d2(p, C[k]);
        if (d < bd) {
          bd = d;
          b = k;
        }
      }
      return [b, bd];
    };
    const J = () => {
      let s = 0;
      P.forEach((p, i) => {
        if (A[i] >= 0 && C[A[i]]) s += d2(p, C[A[i]]);
      });
      return s;
    };
    const snap = (lab) => {
      const Cs = C.map((q) => ({ x: q.x, y: q.y }));
      let j = 0;
      for (const p of P) {
        let bd = 1e18;
        for (const q of Cs) bd = Math.min(bd, d2(p, q));
        j += bd;
      }
      CH.push({ lab, C: Cs, J: j });
      tv++;
    };
    const doRace = () => {
      if (P.length < S.K) {
        race = null;
        return;
      }
      const r = rng(77 + S.seed * 3 + S.K),
        go = (pp) => {
          const o = [];
          for (let i = 0; i < NR; i++) o.push(kmRun(P, kmSeed(P, S.K, pp, r)));
          return o;
        },
        a = go(false),
        b = go(true);
      const best = Math.min(...a.map((q) => q.J), ...b.map((q) => q.J)),
        sum = (o) => ({
          J: o.map((q) => q.J),
          ok: o.filter((q) => q.J <= best * 1.02 + 1e-9).length / o.length,
          it: o.reduce((s, q) => s + q.it, 0) / o.length,
        });
      race = {
        best,
        a: sum(a),
        b: sum(b),
        max: Math.max(...a.map((q) => q.J), ...b.map((q) => q.J)),
      };
    };
    const doElbow = () => {
      const r = rng(5 + S.seed);
      elbow = [];
      for (let k = 1; k <= 9; k++) {
        let m = 1e99;
        if (P.length >= k)
          for (let i = 0; i < 5; i++) m = Math.min(m, kmRun(P, kmSeed(P, k, true, r)).J);
        elbow.push(m < 1e98 ? m : NaN);
      }
    };
    const reset = () => {
      R = rng(1000 + run * 7 + S.seed);
      C = [];
      prev = null;
      A = new Array(P.length).fill(-1);
      ph = 'seed';
      it = 0;
      hist = [];
      mv = 0;
      wait = 0;
      cu = -1;
      CH = [];
      tv++;
      msg =
        S.init === 'own'
          ? 'No centres yet. Click on the left picture ' +
            S.K +
            ' times to place the starting centres yourself.'
          : 'No centres yet. ' +
            (S.init === 'pp'
              ? 'k-means++ will place them one at a time.'
              : 'A random start picks ' + S.K + ' data points all at once.') +
            ' Press Run or One step.';
    };
    const step = () => {
      if (P.length < S.K) return;
      if (ph === 'done') {
        run++;
        reset();
        return;
      }
      if (ph === 'seed') {
        cu = 0;
        if (S.init === 'own') {
          msg =
            'Your own start: click on the left picture to place centre ' +
            (C.length + 1) +
            ' of ' +
            S.K +
            '.';
          return;
        }
        if (S.init === 'rand') {
          C = kmSeed(P, S.K, false, R);
          snap('start');
          ph = 'assign';
          msg =
            'Random start: ' +
            C.length +
            ' data points were picked at random to be the first centres. Nothing stops two of them from landing in the same group.';
        } else {
          if (!C.length) {
            const i = Math.floor(R() * P.length);
            C.push({ x: P[i].x, y: P[i].y });
            msg =
              'k-means++: the first centre is a data point picked at random. Every other point is now drawn in proportion to its squared distance from that centre: the bigger the dot, the more likely it is to be picked next.';
          } else {
            const D = P.map((p) => near(p)[1]);
            let tot = 0;
            for (const v of D) tot += v;
            if (tot <= 0) {
              ph = 'assign';
              return;
            }
            let u = R() * tot,
              i = 0;
            while (i < P.length - 1 && (u -= D[i]) > 0) i++;
            const srt = D.slice().sort((a, b) => a - b);
            let lo = 0;
            for (let k = 0; k < srt.length / 2; k++) lo += srt[k];
            C.push({ x: P[i].x, y: P[i].y });
            msg =
              'k-means++: centre ' +
              C.length +
              ' of ' +
              S.K +
              ' was drawn with probability proportional to squared distance from the nearest existing centre. The point picked was ' +
              f2(Math.sqrt(D[i])) +
              ' away. The half of the points closest to an existing centre shared only ' +
              ((100 * lo) / tot).toFixed(1) +
              '% of the chance.';
          }
          if (C.length >= S.K) {
            ph = 'assign';
            snap('start');
            msg += ' All ' + S.K + ' centres are placed.';
          }
        }
        return;
      }
      if (ph === 'assign') {
        cu = 1;
        let ch = 0;
        P.forEach((p, i) => {
          const b = near(p)[0];
          if (A[i] !== b) {
            A[i] = b;
            ch++;
          }
        });
        const j = J();
        hist.push(j);
        if (!ch && it > 0) {
          ph = 'done';
          cu = 3;
          msg =
            'No point changed cluster, so the centres will not move again. Converged after ' +
            it +
            ' round' +
            (it === 1 ? '' : 's') +
            ' with sum of squares ' +
            f2(j) +
            '.' +
            (race
              ? j <= race.best * 1.02 + 1e-9
                ? ' This matches the best arrangement found in ' + 2 * NR + ' other runs.'
                : ' The best known arrangement has ' +
                  f2(race.best) +
                  ': this start was trapped in a worse local minimum.'
              : '');
        } else {
          ph = 'move';
          msg =
            'Assign: every point takes the colour of its nearest centre. ' +
            ch +
            ' point' +
            (ch === 1 ? '' : 's') +
            ' changed cluster. Sum of squared distances: ' +
            f2(j) +
            '.';
        }
        return;
      }
      if (ph === 'move') {
        cu = 2;
        prev = C.map((q) => ({ ...q }));
        const Sx = C.map(() => [0, 0, 0]);
        P.forEach((p, i) => {
          const s = Sx[A[i]];
          s[0] += p.x;
          s[1] += p.y;
          s[2]++;
        });
        let far = 0,
          emp = 0;
        C.forEach((q, k) => {
          if (Sx[k][2]) {
            const nx = Sx[k][0] / Sx[k][2],
              ny = Sx[k][1] / Sx[k][2];
            far = Math.max(far, Math.hypot(nx - q.x, ny - q.y));
            q.x = nx;
            q.y = ny;
          } else emp++;
        });
        it++;
        const j = J();
        hist.push(j);
        mv = RM ? 0 : 18;
        ph = 'assign';
        snap('after round ' + it);
        msg =
          'Move: each centre jumps to the average of its own points. The largest move was ' +
          f2(far) +
          '. Sum of squared distances: ' +
          f2(j) +
          '.' +
          (emp
            ? ' ' +
              emp +
              ' centre' +
              (emp === 1 ? ' has' : 's have') +
              ' no points and stay' +
              (emp === 1 ? 's' : '') +
              ' put.'
            : '');
      }
    };
    const tp = testPoint(
      root,
      'km',
      ['Your x', 'Your y'],
      [0.5, 0.5],
      [
        [-3.8, 3.8],
        [-2.85, 2.85],
      ],
    );
    const draw = () => {
      const t = T(),
        col = UC(t);
      {
        const V = uView(c),
          { g, w, h, X, Y, s } = V;
        if (C.length === S.K && ph !== 'seed') {
          const cs = 9;
          for (let py = 0; py < h; py += cs)
            for (let px = 0; px < w; px += cs) {
              const b = near({ x: (px + cs / 2 - w / 2) / s, y: -(py + cs / 2 - h / 2) / s })[0];
              g.fillStyle = mix(col[b % 8], col[b % 8], 0, 0.11);
              g.fillRect(px, py, cs, cs);
            }
        }
        uFrame(
          V,
          t,
          S.ds === 'shop' ? 'visits per month (standardised)' : 'x',
          S.ds === 'shop' ? 'average basket (standardised)' : 'y',
        );
        const sh = ph === 'seed' && S.init === 'pp' && C.length > 0;
        let D = null,
          dm = 1;
        if (sh) {
          D = P.map((p) => near(p)[1]);
          dm = Math.max(1e-9, ...D);
        }
        P.forEach((p, i) => {
          g.beginPath();
          if (sh) {
            g.arc(X(p.x), Y(p.y), 1.4 + 8.5 * Math.sqrt(D[i] / dm), 0, 7);
            g.fillStyle = mix(t.ink, t.ink, 0, 0.34);
          } else {
            g.arc(X(p.x), Y(p.y), 3.4, 0, 7);
            g.fillStyle = A[i] >= 0 ? col[A[i] % 8] : mix(t.ink, t.ink, 0, 0.42);
          }
          g.fill();
        });
        if (CH.length > 1)
          C.forEach((q, k) => {
            g.strokeStyle = col[k % 8];
            g.lineWidth = 2;
            g.beginPath();
            CH.forEach((sn, i) => {
              const z = sn.C[k];
              if (!z) return;
              i ? g.lineTo(X(z.x), Y(z.y)) : g.moveTo(X(z.x), Y(z.y));
            });
            g.stroke();
            CH.forEach((sn, i) => {
              const z = sn.C[k];
              if (!z) return;
              g.fillStyle = t.bg;
              g.beginPath();
              g.arc(X(z.x), Y(z.y), i ? 3 : 5, 0, 7);
              g.fill();
              g.stroke();
            });
          });
        const u = prev && mv > 0 ? 1 - mv / 18 : 1;
        C.forEach((q, k) => {
          const o = prev && prev[k] ? prev[k] : q,
            x = X(o.x + (q.x - o.x) * u),
            y = Y(o.y + (q.y - o.y) * u);
          if (prev && prev[k]) {
            g.strokeStyle = t.ink;
            g.lineWidth = 1.4;
            g.setLineDash([4, 4]);
            g.beginPath();
            g.moveTo(X(o.x), Y(o.y));
            g.lineTo(x, y);
            g.stroke();
            g.setLineDash([]);
          }
          g.fillStyle = col[k % 8];
          g.strokeStyle = t.bg;
          g.lineWidth = 5;
          g.beginPath();
          g.rect(x - 8, y - 8, 16, 16);
          g.stroke();
          g.fill();
          g.strokeStyle = t.ink;
          g.lineWidth = 1.6;
          g.strokeRect(x - 8, y - 8, 16, 16);
        });
        if (C.length === S.K) markTP(g, X(tp.v[0]), Y(tp.v[1]), t);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          sh
            ? 'bigger dot = more likely to be the next centre'
            : 'run ' + run + ', ' + (S.init === 'pp' ? 'k-means++ start' : 'random start'),
          8,
          h - 8,
        );
      }
      {
        const { g, w, h } = fit(cr),
          bh = h / 3;
        const L = uLine(g, t, 0, 0, w, bh, hist, {
          title: 'sum of squared distances after each step of this run',
          dots: true,
          n: Math.max(8, hist.length),
          ym: Math.max(race ? race.best * 1.6 : 1, ...hist),
        });
        if (race) {
          g.setLineDash([5, 4]);
          g.strokeStyle = t.gold;
          g.lineWidth = 1.4;
          g.beginPath();
          g.moveTo(L.m.l, L.Y(race.best));
          g.lineTo(w - L.m.r, L.Y(race.best));
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.cut;
          g.textAlign = 'right';
          g.fillText('best known', w - L.m.r, L.Y(race.best) - 5);
          const y0 = bh,
            m = { l: 44, r: 12, t: 40, b: 20 },
            nb = 16,
            lo = race.best,
            hi = Math.max(race.max, lo * 1.0001 + 1e-9),
            bw = (w - m.l - m.r) / nb,
            H = bh - m.t - m.b,
            cnt = (o) => {
              const q = new Array(nb).fill(0);
              for (const v of o.J) q[Math.min(nb - 1, Math.floor(((v - lo) / (hi - lo)) * nb))]++;
              return q;
            },
            ca = cnt(race.a),
            cb = cnt(race.b),
            cm = Math.max(1, ...ca, ...cb);
          for (let i = 0; i < nb; i++) {
            g.fillStyle = t.ok;
            g.fillRect(
              m.l + i * bw + 1,
              y0 + bh - m.b - (ca[i] / cm) * H,
              bw / 2 - 1.5,
              (ca[i] / cm) * H,
            );
            g.fillStyle = t.accent;
            g.fillRect(
              m.l + i * bw + bw / 2,
              y0 + bh - m.b - (cb[i] / cm) * H,
              bw / 2 - 1.5,
              (cb[i] / cm) * H,
            );
          }
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, y0 + bh - m.b);
          g.lineTo(w - m.r, y0 + bh - m.b);
          g.stroke();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.ok;
          g.fillText(
            'random start: ' + (100 * race.a.ok).toFixed(0) + '% of ' + NR + ' runs reach the best',
            8,
            y0 + 13,
          );
          g.fillStyle = t.accent;
          g.fillText(
            'k-means++ start: ' +
              (100 * race.b.ok).toFixed(0) +
              '% of ' +
              NR +
              ' runs reach the best',
            8,
            y0 + 27,
          );
          g.fillStyle = t.muted;
          g.fillText('best', m.l, y0 + bh - 5);
          g.textAlign = 'right';
          g.fillText('final sum of squares: worse →', w - m.r, y0 + bh - 5);
        }
        const E = uLine(g, t, 0, 2 * bh, w, bh, elbow, {
          title: 'lowest sum of squares found for K = 1 … 9: look for the bend',
          dots: true,
          col: t.muted,
        });
        g.font = MONO;
        g.textAlign = 'center';
        elbow.forEach((v, i) => {
          g.fillStyle = t.muted;
          g.fillText(String(i + 1), E.X(i), h - 5);
          if (i + 1 === S.K && isFinite(v)) {
            g.fillStyle = t.gold;
            g.beginPath();
            g.arc(E.X(i), E.Y(v), 6, 0, 7);
            g.fill();
          }
        });
      }
      live.textContent = msg;
      uCur(root, 'km', cu);
      if (tvd !== tv) {
        tvd = tv;
        const tb = $('#km-tab', root);
        tb.innerHTML = !CH.length
          ? '<p class="status">The centroids are not all placed yet.</p>'
          : '<table><thead><tr><th>when</th>' +
            CH[0].C.map(
              (_, k) =>
                '<th><i style="background:' + col[k % 8] + '"></i>centroid ' + (k + 1) + '</th>',
            ).join('') +
            '<th>largest move</th><th>sum of squares</th></tr></thead><tbody>' +
            CH.map(
              (sn, i) =>
                '<tr><td>' +
                sn.lab +
                '</td>' +
                sn.C.map((z) => '<td>(' + f2(z.x) + ', ' + f2(z.y) + ')</td>').join('') +
                '<td>' +
                (i
                  ? f2(
                      Math.max(
                        ...sn.C.map((z, k) =>
                          Math.hypot(z.x - CH[i - 1].C[k].x, z.y - CH[i - 1].C[k].y),
                        ),
                      ),
                    )
                  : '–') +
                '</td><td>' +
                f2(sn.J) +
                '</td></tr>',
            ).join('') +
            '</tbody></table>';
        tb.scrollTop = tb.scrollHeight;
      }
      if (C.length === S.K) {
        const [b, bd] = near({ x: tp.v[0], y: tp.v[1] });
        tp.out.textContent =
          'Your point is ' +
          f2(Math.sqrt(bd)) +
          ' from centre ' +
          (b + 1) +
          ', which is its nearest, so k-means puts it in cluster ' +
          (b + 1) +
          '.';
      } else tp.out.textContent = 'Place the centres first: press One step.';
      setR(root, 'ph', String(it));
      setR(root, 'J', hist.length ? f2(hist[hist.length - 1]) : '–');
      setR(root, 'best', race ? f2(race.best) : '–');
      setR(root, 'ra', race ? (100 * race.a.ok).toFixed(0) + '%' : '–');
      setR(root, 'rp', race ? (100 * race.b.ok).toFixed(0) + '%' : '–');
      setR(root, 'ia', race ? race.a.it.toFixed(1) + ' / ' + race.b.it.toFixed(1) : '–');
    };
    tp.on = () => draw();
    const gen = uDataBind(root, 'km', S, () => {
      P = gen();
      doRace();
      doElbow();
      run = 1;
      reset();
      draw();
    });
    P = gen();
    doRace();
    doElbow();
    reset();
    const anim = Anim(() => {
      fr++;
      if (mv > 0) {
        mv--;
        draw();
        return;
      }
      if (wait > 0) {
        wait--;
        if (!wait) {
          run++;
          reset();
          draw();
        }
        return;
      }
      if (fr % (ph === 'seed' ? 44 : 30)) return;
      step();
      draw();
      chk();
    });
    pb = playBtn($('#km-play', root), anim);
    const chk = runCtl(pb, root, 'km', {
      done: () => ph === 'done',
      restart: () => {
        run++;
        reset();
      },
      count: () => it,
      what: 'rounds',
    });
    $('#km-step', root).onclick = () => {
      pb.stop();
      mv = 0;
      step();
      draw();
    };
    $('#km-reset', root).onclick = () => {
      pb.stop();
      run++;
      reset();
      draw();
    };
    bind(
      root,
      'km-K',
      (v) => {
        S.K = v;
        pb.stop();
        doRace();
        reset();
        draw();
      },
      (v) => String(v),
    );
    segBind(root, 'km-init', S.init, (v) => {
      S.init = v;
      pb.stop();
      reset();
      draw();
    });
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', (e) => {
      const r = c.getBoundingClientRect(),
        s = Math.min(r.width / 8, r.height / 6);
      drag = -1;
      C.forEach((q, k) => {
        if (
          Math.hypot(
            r.width / 2 + q.x * s - (e.clientX - r.left),
            r.height / 2 - q.y * s - (e.clientY - r.top),
          ) < 16
        )
          drag = k;
      });
      if (drag >= 0) {
        pb.stop();
        mv = 0;
        prev = null;
        c.setPointerCapture(e.pointerId);
      } else if (S.init === 'own' && ph === 'seed' && C.length < S.K) {
        const q = uInv(c, e);
        C.push({ x: clamp(q[0], -3.9, 3.9), y: clamp(q[1], -2.9, 2.9) });
        cu = 0;
        if (C.length >= S.K) {
          ph = 'assign';
          snap('your start');
          msg = 'All ' + S.K + ' centres are placed where you chose. Press Run or One step.';
        } else
          msg =
            'Centre ' +
            C.length +
            ' placed at (' +
            f2(q[0]) +
            ', ' +
            f2(q[1]) +
            '). Click to place centre ' +
            (C.length + 1) +
            ' of ' +
            S.K +
            '.';
        draw();
      }
    });
    c.addEventListener('pointermove', (e) => {
      if (drag < 0) return;
      const q = uInv(c, e);
      C[drag].x = clamp(q[0], -3.9, 3.9);
      C[drag].y = clamp(q[1], -2.9, 2.9);
      if (C.length === S.K) {
        P.forEach((p, i) => {
          A[i] = near(p)[0];
        });
        ph = 'move';
        cu = 1;
        msg =
          'You moved centre ' +
          (drag + 1) +
          '. The points have been reassigned to their nearest centres: sum of squared distances ' +
          f2(J()) +
          '. Press One step to let the centres move.';
      }
      draw();
    });
    ['pointerup', 'pointercancel'].forEach((n) =>
      c.addEventListener(n, () => {
        if (drag >= 0 && C.length === S.K) {
          snap('you moved centroid ' + (drag + 1));
          draw();
        }
        drag = -1;
      }),
    );
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        reset();
        draw();
        pb.start();
      },
    };
  },
};
