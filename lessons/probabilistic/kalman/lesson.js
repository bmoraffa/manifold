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
  mix,
  playBtn,
  randnFrom,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { pct } from '../../ensembles/shared.js';
import { uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#kf-c', root),
      cr = $('#kf-r', root),
      live = $('#kf-live', root),
      eq = $('#kf-eq', root),
      S = { r: 4, q: 0.15 },
      TT = 80,
      RN = randnFrom(Math.random);
    let TP = [],
      TV = [],
      Y = [],
      EP = [],
      SG = [],
      KG = [],
      xh,
      Pm,
      t = 0,
      drop = [-1, -1],
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      fin = false,
      cur = null,
      seR = 0,
      nR = 0,
      seF = 0,
      outAt = -1;
    const gen = () => {
      TP = [];
      TV = [];
      let p = 0,
        v = 2.2;
      for (let i = 0; i <= TT; i++) {
        TP.push(p);
        TV.push(v);
        v = clamp(v + 0.22 * RN(), 0.2, 4.5);
        p += v;
      }
      Y = TP.map((q) => q + S.r * RN());
    };
    const reset = () => {
      xh = [Y[0], 0];
      Pm = [
        [S.r * S.r, 0],
        [0, 9],
      ];
      EP = [xh[0]];
      SG = [S.r];
      KG = [1];
      t = 0;
      fin = false;
      cu = 0;
      cur = null;
      seR = (Y[0] - TP[0]) ** 2;
      nR = 1;
      seF = seR;
      drop = [-1, -1];
      outAt = -1;
      msg =
        'Second 0: the filter starts at the first GPS reading and has no idea of the speed. Press One second or Run.';
      eqShow(
        eq,
        [
          'The first reading is taken as the starting estimate: position ' +
            f2(Y[0]) +
            ' m, speed unknown (taken as 0, with a large uncertainty).',
        ],
        0,
      );
    };
    const step = () => {
      if (fin) return;
      t++;
      const q = S.q,
        R = S.r * S.r,
        p0 = xh[0],
        v0 = xh[1],
        s0 = Math.sqrt(Pm[0][0]);
      const xp = [xh[0] + xh[1], xh[1]],
        a = Pm[0][0],
        b = Pm[0][1],
        d = Pm[1][1],
        Pp = [
          [a + 2 * b + d + q / 3, b + d + q / 2],
          [b + d + q / 2, d + q],
        ];
      const has = !(t >= drop[0] && t <= drop[1]);
      if (t === outAt) Y[t] = TP[t] + (Math.random() < 0.5 ? -1 : 1) * 7 * S.r;
      const L = [
        'Second ' + t + '.',
        '<b>predict</b>: position ' +
          f2(p0) +
          ' + 1 s × speed ' +
          f2(v0) +
          ' = ' +
          f2(xp[0]) +
          ' m      uncertainty ± ' +
          f2(s0) +
          ' grows to ± ' +
          f2(Math.sqrt(Pp[0][0])),
      ];
      if (has) {
        const y = Y[t],
          inn = y - xp[0],
          Sv = Pp[0][0] + R,
          K0 = Pp[0][0] / Sv,
          K1 = Pp[1][0] / Sv;
        xh = [xp[0] + K0 * inn, xp[1] + K1 * inn];
        Pm = [
          [(1 - K0) * Pp[0][0], (1 - K0) * Pp[0][1]],
          [Pp[1][0] - K1 * Pp[0][0], Pp[1][1] - K1 * Pp[0][1]],
        ];
        KG.push(K0);
        cur = { xp: xp[0], sp: Math.sqrt(Pp[0][0]), y, x: xh[0], s: Math.sqrt(Pm[0][0]) };
        seR += (y - TP[t]) ** 2;
        nR++;
        seF += (xh[0] - TP[t]) ** 2;
        L.push(
          '<b>reading</b>: GPS says ' +
            f2(y) +
            ' m (± ' +
            f2(S.r) +
            ').  surprise = ' +
            f2(y) +
            ' − ' +
            f2(xp[0]) +
            ' = ' +
            f2(inn) +
            ' m',
          '<b>gain</b> = prediction variance ÷ (prediction variance + GPS variance) = ' +
            f2(Pp[0][0]) +
            ' ÷ (' +
            f2(Pp[0][0]) +
            ' + ' +
            f2(R) +
            ') = <b>' +
            f3(K0) +
            '</b>',
          '<b>update</b>: position ' +
            f2(xp[0]) +
            ' + ' +
            f3(K0) +
            ' × ' +
            f2(inn) +
            ' = <b>' +
            f2(xh[0]) +
            ' m</b>      uncertainty ± ' +
            f2(Math.sqrt(Pm[0][0])),
          'speed corrected too: ' +
            f2(xp[1]) +
            ' + ' +
            f3(K1) +
            ' × ' +
            f2(inn) +
            ' = ' +
            f2(xh[1]) +
            ' m/s',
        );
        cu = 4;
        msg =
          'Second ' +
          t +
          ': the prediction was ' +
          f2(xp[0]) +
          ' m, the GPS said ' +
          f2(y) +
          ' m, and the filter believed ' +
          pct(K0) +
          ' of the difference. Estimate ' +
          f2(xh[0]) +
          ' m; the drone is really at ' +
          f2(TP[t]) +
          ' m.' +
          (t === outAt
            ? ' That reading was a wild one, ' + f2(Math.abs(y - TP[t])) + ' m off.'
            : '');
      } else {
        xh = xp;
        Pm = Pp;
        KG.push(NaN);
        cur = { xp: xp[0], sp: Math.sqrt(Pp[0][0]), y: NaN, x: xp[0], s: Math.sqrt(Pp[0][0]) };
        L.push(
          '<b>no reading</b>: the signal is lost. The prediction is all there is.',
          'estimate ' + f2(xh[0]) + ' m, uncertainty ± ' + f2(Math.sqrt(Pm[0][0])) + ' and growing',
        );
        cu = 5;
        msg =
          'Second ' +
          t +
          ': no GPS. The filter coasts at ' +
          f2(xh[1]) +
          ' m/s and its uncertainty has grown to ± ' +
          f2(Math.sqrt(Pm[0][0])) +
          ' m.';
      }
      EP.push(xh[0]);
      SG.push(Math.sqrt(Pm[0][0]));
      eqShow(eq, L, 0);
      if (t >= TT) {
        fin = true;
        msg +=
          ' The flight is over. Typical error of the raw GPS: ' +
          f2(Math.sqrt(seR / nR)) +
          ' m. Typical error of the filter on the same seconds: ' +
          f2(Math.sqrt(seF / nR)) +
          ' m.';
      }
    };
    const draw = () => {
      const tt = T();
      {
        const { g, w, h } = fit(c),
          m = { l: 44, r: 12, t: 14, b: 24 },
          lo = Math.min(...TP) - 3 * S.r,
          hi = Math.max(...TP) + 3 * S.r,
          X = (i) => m.l + (i / TT) * (w - m.l - m.r),
          Yv = (v) => h - m.b - ((v - lo) / (hi - lo)) * (h - m.t - m.b);
        g.font = MONO;
        g.strokeStyle = tt.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h - m.b);
        g.lineTo(w - m.r, h - m.b);
        g.moveTo(m.l, m.t);
        g.lineTo(m.l, h - m.b);
        g.stroke();
        if (drop[0] >= 0) {
          g.fillStyle = mix(tt.cut, tt.cut, 0, 0.13);
          g.fillRect(
            X(drop[0] - 0.5),
            m.t,
            X(Math.min(TT, drop[1] + 0.5)) - X(drop[0] - 0.5),
            h - m.t - m.b,
          );
          g.fillStyle = tt.cut;
          g.textAlign = 'center';
          g.fillText('no signal', (X(drop[0]) + X(Math.min(TT, drop[1]))) / 2, m.t + 12);
        }
        g.fillStyle = mix(tt.accent, tt.accent, 0, 0.22);
        g.beginPath();
        for (let i = 0; i <= t; i++)
          i ? g.lineTo(X(i), Yv(EP[i] + 2 * SG[i])) : g.moveTo(X(i), Yv(EP[i] + 2 * SG[i]));
        for (let i = t; i >= 0; i--) g.lineTo(X(i), Yv(EP[i] - 2 * SG[i]));
        g.closePath();
        g.fill();
        g.strokeStyle = tt.ink;
        g.lineWidth = 1.2;
        g.beginPath();
        for (let i = 0; i <= t; i++) i ? g.lineTo(X(i), Yv(TP[i])) : g.moveTo(X(i), Yv(TP[i]));
        g.stroke();
        g.fillStyle = mix(tt.ink, tt.ink, 0, 0.5);
        for (let i = 0; i <= t; i++) {
          if (i >= drop[0] && i <= drop[1]) continue;
          g.beginPath();
          g.arc(X(i), Yv(clamp(Y[i], lo, hi)), 2.6, 0, 7);
          g.fill();
        }
        g.strokeStyle = tt.accent;
        g.lineWidth = 2.6;
        g.beginPath();
        for (let i = 0; i <= t; i++) i ? g.lineTo(X(i), Yv(EP[i])) : g.moveTo(X(i), Yv(EP[i]));
        g.stroke();
        g.fillStyle = tt.gold;
        g.beginPath();
        g.arc(X(t), Yv(EP[t]), 5, 0, 7);
        g.fill();
        g.fillStyle = tt.muted;
        g.textAlign = 'right';
        g.fillText(lo.toFixed(0) + ' m', m.l - 4, h - m.b);
        g.fillText(hi.toFixed(0) + ' m', m.l - 4, m.t + 8);
        g.textAlign = 'center';
        for (const k of [0, 20, 40, 60, 80]) g.fillText(k + ' s', X(k), h - 7);
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.52,
          m = { l: 14, r: 14, t: 26, b: 22 };
        g.font = MONO;
        g.fillStyle = tt.muted;
        g.textAlign = 'left';
        g.fillText('this second: prediction (dashed), reading, and their blend', 8, 13);
        if (cur) {
          const cs = [cur.xp, cur.x].concat(isFinite(cur.y) ? [cur.y] : []),
            sp = Math.max(cur.sp, S.r) * 3.2,
            mid = (Math.min(...cs) + Math.max(...cs)) / 2,
            lo = mid - sp - Math.abs(cs[0] - mid),
            hi = mid + sp + Math.abs(cs[0] - mid),
            X = (v) => m.l + ((v - lo) / (hi - lo)) * (w - m.l - m.r),
            pk = 1 / Math.min(cur.s, cur.sp, S.r),
            Yv = (v) => h1 - m.b - (v / pk) * (h1 - m.t - m.b),
            bell = (mu, sd, col, lw, dash, fill) => {
              g.beginPath();
              for (let i = 0; i <= 120; i++) {
                const v = lo + ((hi - lo) * i) / 120,
                  d = Math.exp(-0.5 * ((v - mu) / sd) ** 2) / sd;
                i ? g.lineTo(X(v), Yv(d)) : g.moveTo(X(v), Yv(d));
              }
              if (fill) {
                g.lineTo(X(hi), Yv(0));
                g.lineTo(X(lo), Yv(0));
                g.closePath();
                g.fillStyle = fill;
                g.fill();
              }
              g.strokeStyle = col;
              g.lineWidth = lw;
              g.setLineDash(dash || []);
              g.stroke();
              g.setLineDash([]);
            };
          bell(cur.x, cur.s, tt.accent, 2.6, null, mix(tt.accent, tt.accent, 0, 0.25));
          bell(cur.xp, cur.sp, tt.muted, 1.8, [6, 4]);
          if (isFinite(cur.y)) bell(cur.y, S.r, tt.ok, 1.8);
          g.strokeStyle = tt.ink;
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(X(TP[t]), h1 - m.b);
          g.lineTo(X(TP[t]), h1 - m.b - 12);
          g.stroke();
          g.fillStyle = tt.muted;
          g.textAlign = 'center';
          g.fillText('true position', clamp(X(TP[t]), 50, w - 50), h1 - 6);
          g.textAlign = 'right';
          g.fillStyle = tt.muted;
          g.fillText('prediction', w - 14, m.t + 6);
          g.fillStyle = tt.ok;
          g.fillText(isFinite(cur.y) ? 'GPS reading' : 'no reading', w - 14, m.t + 20);
          g.fillStyle = tt.accent;
          g.fillText('blend', w - 14, m.t + 34);
        } else {
          g.textAlign = 'center';
          g.fillText('the curves appear at the first second', w / 2, h1 / 2);
        }
        const L = uLine(g, tt, 0, h1, w, h - h1, [], {
          title: 'gain: the share of each new reading that is believed',
          n: TT + 1,
          ym: 1,
        });
        g.strokeStyle = tt.cut;
        g.lineWidth = 2.2;
        g.beginPath();
        let pen = false;
        KG.forEach((v, i) => {
          if (!isFinite(v)) {
            pen = false;
            return;
          }
          pen ? g.lineTo(L.X(i), L.Y(v)) : g.moveTo(L.X(i), L.Y(v));
          pen = true;
        });
        g.stroke();
        g.fillStyle = tt.muted;
        g.textAlign = 'right';
        g.fillText('seconds →', w - 12, h - 5);
      }
      live.textContent = msg;
      uCur(root, 'kf', cu);
      const kk = KG[KG.length - 1];
      setR(root, 't', t + ' of ' + TT);
      setR(root, 'k', isFinite(kk) ? f3(kk) : 'no reading');
      setR(root, 'sg', f2(SG[SG.length - 1]));
      setR(root, 'v', f2(xh[1]) + ' / ' + f2(TV[t]));
      setR(root, 'er', f2(Math.sqrt(seR / nR)) + ' m');
      setR(root, 'ef', t ? f2(Math.sqrt(seF / nR)) + ' m' : '–');
    };
    gen();
    reset();
    if (RM) {
      while (!fin) step();
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      if (fr % (t < 6 ? 40 : 9)) return;
      step();
      draw();
      if (fin) pb.stop();
    });
    pb = playBtn($('#kf-play', root), anim);
    runCtl(pb, root, 'kf', { done: () => fin, restart: () => reset() });
    $('#kf-step', root).onclick = () => {
      pb.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#kf-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    $('#kf-new', root).onclick = () => {
      pb.stop();
      gen();
      reset();
      draw();
      pb.start();
    };
    $('#kf-drop', root).onclick = () => {
      if (fin || t > TT - 4) {
        msg = 'The flight is over. Press Start over, then lose the signal while it is running.';
        draw();
        return;
      }
      drop = [t + 1, t + 12];
      msg =
        'The GPS signal will be missing from second ' +
        (t + 1) +
        ' to second ' +
        Math.min(TT, t + 12) +
        '. Watch the band.';
      draw();
    };
    $('#kf-out', root).onclick = () => {
      if (fin) {
        msg = 'The flight is over. Press Start over first.';
        draw();
        return;
      }
      outAt = t + 1;
      msg = 'The next reading will be wildly wrong.';
      draw();
    };
    bind(
      root,
      'kf-r2',
      (v) => {
        S.r = +v;
        if (pb) {
          pb.stop();
          gen();
          reset();
          draw();
        }
      },
      f2,
    );
    bind(
      root,
      'kf-q',
      (v) => {
        S.q = +v;
        if (pb) {
          pb.stop();
          reset();
          draw();
        }
      },
      f2,
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
