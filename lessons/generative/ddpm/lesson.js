/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
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
import { segBind, uCur } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#dp-c', root),
      cr = $('#dp-r', root),
      live = $('#dp-live', root),
      st = $('#dp-st', root),
      sel = $('#dp-pic', root),
      S = {
        mode: 'round',
        pic: 1,
        sch: 'lin',
        T: 100,
        b0: 0.001,
        b1: 0.2,
        turn: 60,
        v: 0,
        rs: 100,
      },
      N = 32,
      DM = N * N * 3,
      RN = randnFrom(Math.random),
      SN = { lin: 'linear', cos: 'cosine', con: 'constant' };
    let GAL = [],
      AB = [1],
      ABo = [1],
      x = new Float64Array(DM),
      x0h = null,
      wts = [],
      t = 0,
      phase = 'fwd',
      tau = [],
      ti = 0,
      K = 75,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      eps0 = null,
      wTurn = null,
      aTurn = 1,
      res = -1,
      rc = [NaN, NaN],
      tb = null;
    const fromCanvas = (o) => {
      const d = o.getContext('2d').getImageData(0, 0, N, N).data,
        v = new Float64Array(DM);
      for (let i = 0; i < N * N; i++)
        for (let ch = 0; ch < 3; ch++) v[i * 3 + ch] = d[i * 4 + ch] / 127.5 - 1;
      return v;
    };
    const builtin = () => {
      const mk = (fn) => {
        const o = document.createElement('canvas');
        o.width = o.height = N;
        const g = o.getContext('2d');
        fn(g);
        return fromCanvas(o);
      };
      return [
        [
          'A house in a field',
          mk((g) => {
            let gr = g.createLinearGradient(0, 0, 0, 20);
            gr.addColorStop(0, '#3f79c4');
            gr.addColorStop(1, '#bfe0f5');
            g.fillStyle = gr;
            g.fillRect(0, 0, 32, 32);
            g.fillStyle = '#ffd95a';
            g.beginPath();
            g.arc(25, 7, 4, 0, 7);
            g.fill();
            g.fillStyle = '#4f9a47';
            g.fillRect(0, 20, 32, 12);
            g.fillStyle = '#efe6d4';
            g.fillRect(8, 14, 12, 10);
            g.fillStyle = '#b5452f';
            g.beginPath();
            g.moveTo(6, 15);
            g.lineTo(14, 7);
            g.lineTo(22, 15);
            g.fill();
            g.fillStyle = '#5a3a22';
            g.fillRect(12, 18, 4, 6);
          }),
        ],
        [
          'A sunflower',
          mk((g) => {
            g.fillStyle = '#8fd0ef';
            g.fillRect(0, 0, 32, 32);
            g.fillStyle = '#3f8f3f';
            g.fillRect(15, 16, 2, 16);
            g.fillStyle = '#f6c52c';
            for (let k = 0; k < 10; k++) {
              const a = (k / 10) * 2 * Math.PI;
              g.beginPath();
              g.ellipse(16 + 7 * Math.cos(a), 12 + 7 * Math.sin(a), 4, 2.2, a, 0, 7);
              g.fill();
            }
            g.fillStyle = '#6b3f1d';
            g.beginPath();
            g.arc(16, 12, 5, 0, 7);
            g.fill();
          }),
        ],
        [
          'A sailing boat',
          mk((g) => {
            let gr = g.createLinearGradient(0, 0, 0, 20);
            gr.addColorStop(0, '#f6b26b');
            gr.addColorStop(1, '#ffe8b8');
            g.fillStyle = gr;
            g.fillRect(0, 0, 32, 32);
            g.fillStyle = '#1f5f8b';
            g.fillRect(0, 21, 32, 11);
            g.fillStyle = '#ffffff';
            g.beginPath();
            g.moveTo(16, 4);
            g.lineTo(16, 19);
            g.lineTo(26, 19);
            g.fill();
            g.fillStyle = '#e04a3a';
            g.beginPath();
            g.moveTo(14, 7);
            g.lineTo(14, 19);
            g.lineTo(7, 19);
            g.fill();
            g.fillStyle = '#4a2f1b';
            g.beginPath();
            g.moveTo(5, 20);
            g.lineTo(27, 20);
            g.lineTo(23, 25);
            g.lineTo(9, 25);
            g.fill();
          }),
        ],
        [
          'A cat’s face',
          mk((g) => {
            g.fillStyle = '#2d2a4a';
            g.fillRect(0, 0, 32, 32);
            g.fillStyle = '#e8963a';
            g.beginPath();
            g.moveTo(6, 12);
            g.lineTo(8, 3);
            g.lineTo(14, 9);
            g.moveTo(26, 12);
            g.lineTo(24, 3);
            g.lineTo(18, 9);
            g.fill();
            g.beginPath();
            g.arc(16, 17, 11, 0, 7);
            g.fill();
            g.fillStyle = '#fff';
            g.beginPath();
            g.arc(11, 15, 3, 0, 7);
            g.arc(21, 15, 3, 0, 7);
            g.fill();
            g.fillStyle = '#1c1c1c';
            g.beginPath();
            g.arc(11, 15, 1.4, 0, 7);
            g.arc(21, 15, 1.4, 0, 7);
            g.fill();
            g.fillStyle = '#d4607a';
            g.beginPath();
            g.moveTo(14.5, 20);
            g.lineTo(17.5, 20);
            g.lineTo(16, 22);
            g.fill();
          }),
        ],
      ];
    };
    const other = () => (S.sch === 'cos' ? 'lin' : 'cos');
    const mkAB = (sch) => {
      const T = S.T,
        a = [1];
      if (sch === 'cos') {
        const s = 0.008,
          f = (k) => Math.cos((((k / T + s) / (1 + s)) * Math.PI) / 2) ** 2;
        for (let k = 1; k <= T; k++) a.push(a[k - 1] * (1 - Math.min(0.999, 1 - f(k) / f(k - 1))));
      } else
        for (let k = 1; k <= T; k++) {
          const b = sch === 'lin' ? S.b0 + ((S.b1 - S.b0) * (k - 1)) / Math.max(1, T - 1) : S.b1;
          a.push(a[k - 1] * (1 - clamp(b, 1e-5, 0.999)));
        }
      return a;
    };
    const recog = (a) => {
      if (S.mode === 'gen' || !eps0) return NaN;
      const y = GAL[S.pic][1],
        D = [],
        C = [];
      for (const q of GAL) {
        let d = 0,
          cc = 0;
        const g = q[1];
        for (let i = 0; i < DM; i++) {
          const u = y[i] - g[i];
          d += u * u;
          cc += u * eps0[i];
        }
        D.push(d);
        C.push(cc);
      }
      let n = 0;
      for (let k = 1; k < a.length; k++) {
        const ak = a[k],
          lg = D.map(
            (d, j) =>
              -(ak * d + 2 * Math.sqrt(ak * (1 - ak)) * C[j]) / (2 * Math.max(1e-12, 1 - ak)),
          ),
          m = Math.max(...lg),
          z = lg.reduce((s, v) => s + Math.exp(v - m), 0);
        if (Math.exp(lg[S.pic] - m) / z > 0.5) n++;
      }
      return n;
    };
    const sched = () => {
      AB = mkAB(S.sch);
      ABo = mkAB(other());
      K = clamp(Math.round((S.turn / 100) * S.T), 1, S.T);
      if (tb) {
        tb.el.max = S.T;
      }
      rc = [recog(AB), recog(ABo)];
    };
    const fillSel = () => {
      sel.innerHTML = GAL.map((q, i) => '<option value="' + i + '">' + q[0] + '</option>').join('');
      sel.value = S.pic;
    };
    const denoise = () => {
      const a = AB[t],
        sa = Math.sqrt(a),
        lg = GAL.map((q) => {
          let s = 0;
          const y = q[1];
          for (let i = 0; i < DM; i++) {
            const d = x[i] - sa * y[i];
            s += d * d;
          }
          return -s / (2 * Math.max(1e-9, 1 - a));
        }),
        m = Math.max(...lg),
        e = lg.map((v) => Math.exp(v - m)),
        z = e.reduce((p, q) => p + q, 0);
      wts = e.map((v) => v / z);
      x0h = new Float64Array(DM);
      GAL.forEach((q, k) => {
        const w = wts[k];
        if (w > 1e-6) for (let i = 0; i < DM; i++) x0h[i] += w * q[1][i];
      });
    };
    const mkTau = (from) => {
      const R = Math.max(1, Math.round((S.rs / 100) * from)),
        o = [];
      for (let j = 0; j <= R; j++) {
        const v = Math.round((from * (R - j)) / R);
        if (!o.length || o[o.length - 1] !== v) o.push(v);
      }
      return o;
    };
    const start = (newNoise) => {
      if (newNoise || !eps0 || eps0.length !== DM)
        eps0 = Float64Array.from({ length: DM }, () => RN());
      sched();
      cu = 0;
      res = -1;
      wTurn = null;
      ti = 0;
      if (S.mode === 'round') {
        phase = 'fwd';
        t = 0;
        x = Float64Array.from(GAL[S.pic][1]);
        msg =
          'This is the clean picture, “' +
          GAL[S.pic][0] +
          '”, at step 0. Press Run: noise is added for ' +
          K +
          ' of the ' +
          S.T +
          ' steps of the ' +
          SN[S.sch] +
          ' schedule, then the denoiser works its way back.';
      } else {
        phase = 'rev';
        t = S.T;
        x = Float64Array.from({ length: DM }, () => RN());
        tau = mkTau(S.T);
        msg =
          'Pure static at step ' +
          S.T +
          '. It was not made from any picture, so there is nothing to recover. Press Run and see which of the ' +
          GAL.length +
          ' known pictures forms.' +
          (Math.sqrt(AB[S.T]) > 0.03
            ? ' Note: this schedule does not end in pure static. At its last step ' +
              pct(Math.sqrt(AB[S.T])) +
              ' of a picture would still be left, so starting from pure static does not match what the denoiser expects, and the result is skewed toward some pictures. A schedule must destroy everything for generation to be fair.'
            : '');
      }
      denoise();
      if (tb) tb.set(S.mode === 'round' ? 0 : S.T);
    };
    const fin = () => phase === 'done';
    const step = () => {
      if (fin()) {
        start(false);
        return;
      }
      if (phase === 'fwd') {
        const b = 1 - AB[t + 1] / AB[t],
          s1 = Math.sqrt(1 - b),
          s2 = Math.sqrt(b);
        for (let i = 0; i < DM; i++) x[i] = s1 * x[i] + s2 * RN();
        t++;
        denoise();
        cu = t >= K ? 1 : 0;
        const bi = wts.indexOf(Math.max(...wts));
        msg =
          'Forward step ' +
          t +
          ' of ' +
          K +
          ': the picture was shrunk by a factor of ' +
          f3(s1) +
          ' and noise of size ' +
          f3(s2) +
          ' was added (β = ' +
          f3(b) +
          '). ' +
          pct(Math.sqrt(AB[t])) +
          ' of the original is left.';
        if (t >= K) {
          phase = 'rev';
          wTurn = wts.slice();
          aTurn = AB[t];
          tau = mkTau(K);
          ti = 0;
          msg +=
            ' This is the turning point. From what is left, the denoiser gives “' +
            GAL[S.pic][0] +
            '” a probability of ' +
            pct(wts[S.pic]) +
            (bi !== S.pic ? ', and “' + GAL[bi][0] + '” ' + pct(wts[bi]) : '') +
            '. Now the reverse process starts from exactly this noisy picture.';
        }
      } else {
        const tt = tau[ti],
          s = tau[ti + 1],
          at = AB[tt],
          as = AB[s],
          bp = 1 - at / as,
          c0 = (Math.sqrt(as) * bp) / Math.max(1e-12, 1 - at),
          c1 = (Math.sqrt(1 - bp) * (1 - as)) / Math.max(1e-12, 1 - at),
          bt = ((1 - as) / Math.max(1e-12, 1 - at)) * bp,
          sg =
            s > 0
              ? Math.sqrt(
                  Math.exp(
                    S.v * Math.log(Math.max(bp, 1e-20)) + (1 - S.v) * Math.log(Math.max(bt, 1e-20)),
                  ),
                )
              : 0;
        for (let i = 0; i < DM; i++) x[i] = c0 * x0h[i] + c1 * x[i] + sg * RN();
        ti++;
        t = s;
        denoise();
        const bi = wts.indexOf(Math.max(...wts));
        cu = t <= 0 ? 4 : 3;
        msg =
          'Reverse, from step ' +
          tt +
          ' to step ' +
          s +
          ': the new picture is ' +
          f3(c0) +
          ' × the denoiser’s guess + ' +
          f3(c1) +
          ' × the current picture, plus fresh noise of size ' +
          f3(sg) +
          '. The denoiser’s guess is ' +
          (wts[bi] > 0.98 ? 'now firmly' : pct(wts[bi])) +
          ' “' +
          GAL[bi][0] +
          '”.';
        if (t <= 0) {
          phase = 'done';
          res = bi;
          if (S.mode === 'gen') {
            cu = 5;
            msg +=
              ' Finished: “' +
              GAL[bi][0] +
              '”, made from nothing but static. Run it again and a different picture may form.';
          } else if (bi === S.pic)
            msg +=
              ' Finished: “' +
              GAL[bi][0] +
              '” came back, the picture you started with. At the turning point ' +
              pct(Math.sqrt(aTurn)) +
              ' of it was left, enough for the denoiser to recognise it.';
          else
            msg +=
              ' Finished: it came back as “' +
              GAL[bi][0] +
              '”, not “' +
              GAL[S.pic][0] +
              '”. At the turning point only ' +
              pct(Math.sqrt(aTurn)) +
              ' of the original was left and the denoiser gave it a probability of ' +
              pct(wTurn[S.pic]) +
              '. Too little survived to tell which picture it had been. Turn back earlier, or use a schedule that keeps more.';
        }
      }
      if (tb) tb.set(t);
    };
    const jump = (k) => {
      if (S.mode !== 'round') return;
      phase = 'fwd';
      res = -1;
      t = clamp(k, 0, S.T);
      const a = AB[t],
        sa = Math.sqrt(a),
        sn = Math.sqrt(1 - a),
        y = GAL[S.pic][1];
      for (let i = 0; i < DM; i++) x[i] = sa * y[i] + sn * eps0[i];
      denoise();
      cu = 0;
      msg =
        'Jumped straight to forward step ' +
        t +
        ' with the one-jump formula: ' +
        f3(sa) +
        ' × picture + ' +
        f3(sn) +
        ' × noise. The denoiser gives “' +
        GAL[S.pic][0] +
        '” a probability of ' +
        pct(wts[S.pic]) +
        ' here.' +
        (t >= K
          ? ' This is at or past your turning point: press One step to start the reverse process from here.'
          : '');
      if (t >= K) {
        phase = 'rev';
        wTurn = wts.slice();
        aTurn = a;
        tau = mkTau(t);
        ti = 0;
      }
    };
    const blit = (g, v, X0, Y0, sz) => {
      const cz = sz / N;
      for (let i = 0; i < N * N; i++) {
        g.fillStyle =
          'rgb(' +
          [0, 1, 2].map((ch) => Math.round(clamp((v[i * 3 + ch] + 1) * 127.5, 0, 255))).join(',') +
          ')';
        g.fillRect(X0 + (i % N) * cz, Y0 + Math.floor(i / N) * cz, cz + 0.4, cz + 0.4);
      }
    };
    const mixv = (a, buf) => {
      const sa = Math.sqrt(a),
        sn = Math.sqrt(1 - a),
        y = GAL[S.pic][1];
      for (let i = 0; i < DM; i++) buf[i] = sa * y[i] + sn * eps0[i];
      return buf;
    };
    const draw = () => {
      const tt = T(),
        Tn = S.T;
      {
        const { g, w, h } = fit(c),
          sz = Math.min(w, h) - 8,
          X0 = (w - sz) / 2,
          Y0 = (h - sz) / 2;
        blit(g, x, X0, Y0, sz);
        if (S.mode === 'round') {
          const s2 = sz * 0.2;
          g.fillStyle = tt.bg;
          g.fillRect(X0, Y0, s2 + 8, s2 + 22);
          blit(g, GAL[S.pic][1], X0 + 4, Y0 + 4, s2);
          g.font = MONO;
          g.fillStyle = tt.ink;
          g.textAlign = 'center';
          g.fillText('original', X0 + 4 + s2 / 2, Y0 + s2 + 17);
        }
      }
      {
        const { g, w, h } = fit(cr),
          sa = h * 0.24;
        g.font = MONO;
        g.fillStyle = tt.muted;
        g.textAlign = 'left';
        g.fillText('the denoiser’s guess at the clean picture, and its belief in each', 8, 13);
        blit(g, x0h, 8, 20, sa - 8);
        const bx = sa + 14,
          rh = Math.min(30, (sa - 8) / GAL.length);
        GAL.forEach((q, k) => {
          const y = 20 + k * rh;
          blit(g, q[1], bx, y + 1, rh - 3);
          g.fillStyle = mix(tt.ink, tt.ink, 0, 0.08);
          g.fillRect(bx + rh + 4, y + rh * 0.25, w - bx - rh - 60, rh * 0.5);
          g.fillStyle = k === S.pic && S.mode === 'round' ? tt.gold : tt.accent;
          g.fillRect(bx + rh + 4, y + rh * 0.25, (w - bx - rh - 60) * wts[k], rh * 0.5);
          g.fillStyle = tt.muted;
          g.textAlign = 'right';
          g.fillText((100 * wts[k]).toFixed(0) + '%', w - 8, y + rh * 0.62);
        });
        const ys = sa + 30,
          sw = Math.min((w - 110) / 5 - 6, h * 0.135),
          buf = new Float64Array(DM);
        [
          [AB, S.sch],
          [ABo, other()],
        ].forEach(([a, nm], r) => {
          const y = ys + r * (sw + 16);
          g.fillStyle = r ? tt.muted : tt.ink;
          g.textAlign = 'left';
          g.fillText(r ? 'alternative:' : 'yours:', 8, y + sw / 2 - 4);
          g.fillText(SN[nm], 8, y + sw / 2 + 9);
          for (let j = 0; j < 5; j++) {
            const k = Math.round((j * Tn) / 4),
              X0 = 92 + j * (sw + 6);
            blit(g, mixv(a[k], buf), X0, y, sw);
            if (!r) {
              g.fillStyle = tt.muted;
              g.textAlign = 'center';
              g.fillText(j === 4 ? 'all steps' : j ? j * 25 + '%' : 'start', X0 + sw / 2, y - 4);
            }
          }
        });
        const yc = ys + 2 * (sw + 16) + 14,
          m = { l: 36, r: 58, b: 20 },
          X = (k) => m.l + (k / Tn) * (w - m.l - m.r),
          Y = (v) => h - m.b - v * (h - yc - m.b - 6);
        g.strokeStyle = tt.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h - m.b);
        g.lineTo(w - m.r, h - m.b);
        g.stroke();
        g.textAlign = 'right';
        g.fillStyle = tt.muted;
        g.fillText('1', m.l - 5, Y(1) + 4);
        g.fillText('0', m.l - 5, Y(0) + 4);
        g.setLineDash([5, 4]);
        g.strokeStyle = tt.muted;
        g.lineWidth = 1.6;
        g.beginPath();
        for (let k = 0; k <= Tn; k++)
          k ? g.lineTo(X(k), Y(Math.sqrt(ABo[k]))) : g.moveTo(X(k), Y(Math.sqrt(ABo[k])));
        g.stroke();
        g.setLineDash([]);
        [
          [(k) => Math.sqrt(AB[k]), tt.accent, 'picture'],
          [(k) => Math.sqrt(1 - AB[k]), tt.ok, 'noise'],
        ].forEach(([f, cl, lab]) => {
          g.strokeStyle = cl;
          g.lineWidth = 2.4;
          g.beginPath();
          for (let k = 0; k <= Tn; k++) k ? g.lineTo(X(k), Y(f(k))) : g.moveTo(X(k), Y(f(k)));
          g.stroke();
          g.fillStyle = cl;
          g.textAlign = 'left';
          g.fillText(lab, w - m.r + 4, Y(f(Tn)) + 4);
        });
        if (S.mode === 'round') {
          g.setLineDash([3, 3]);
          g.strokeStyle = tt.ink;
          g.lineWidth = 1.3;
          g.beginPath();
          g.moveTo(X(K), Y(0));
          g.lineTo(X(K), Y(1.04));
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = tt.ink;
          g.textAlign = 'center';
          g.fillText('turn back', X(K), yc + 2);
        }
        g.strokeStyle = tt.gold;
        g.lineWidth = 2.5;
        g.beginPath();
        g.moveTo(X(t), Y(0));
        g.lineTo(X(t), Y(1));
        g.stroke();
        g.fillStyle = tt.muted;
        g.textAlign = 'left';
        g.fillText(
          'share of picture and of noise in the mix; dashed: picture under the ' +
            SN[other()] +
            ' schedule',
          8,
          h - 4,
        );
        g.textAlign = 'center';
      }
      live.textContent = msg;
      uCur(root, 'dp', cu);
      const bi = wts.indexOf(Math.max(...wts));
      setR(
        root,
        't',
        t +
          ' of ' +
          Tn +
          (phase === 'fwd' ? ' (adding noise)' : phase === 'rev' ? ' (removing noise)' : ''),
      );
      setR(root, 'sig', (100 * Math.sqrt(AB[t])).toFixed(1) + '%');
      setR(root, 'nz', (100 * Math.sqrt(1 - AB[t])).toFixed(1) + '%');
      setR(
        root,
        'to',
        wts[bi] < 0.4
          ? 'no particular picture yet'
          : '“' + GAL[bi][0] + '” (' + (100 * wts[bi]).toFixed(0) + '%)',
      );
      setR(
        root,
        'rc',
        isFinite(rc[0])
          ? rc[0] +
              ' of ' +
              Tn +
              ' (' +
              SN[S.sch] +
              ') / ' +
              rc[1] +
              ' of ' +
              Tn +
              ' (' +
              SN[other()] +
              ')'
          : '–',
      );
      setR(
        root,
        'res',
        res < 0
          ? '–'
          : S.mode === 'gen'
            ? 'Generated “' + GAL[res][0] + '”'
            : res === S.pic
              ? 'The same picture came back'
              : 'A different picture came back: “' + GAL[res][0] + '”',
      );
    };
    tb = bind(
      root,
      'dp-t',
      (v) => {
        pb.stop();
        if (S.mode === 'round') {
          jump(+v);
          draw();
        } else {
          tb.set(t);
          msg =
            'The step slider belongs to the forward direction, where any step can be computed in one jump. Choose Noise This Picture, Then Denoise It to use it.';
          draw();
        }
      },
      (v) => String(v),
    );
    GAL = builtin();
    fillSel();
    start(true);
    const anim = Anim(() => {
      fr++;
      const slow = phase === 'fwd' ? t < 5 : ti < 3 || t < 4;
      if (fr % (slow ? 9 : Math.max(2, Math.round(260 / S.T)))) return;
      step();
      draw();
      if (fin()) pb.stop();
    });
    pb = playBtn($('#dp-play', root), anim);
    runCtl(pb, root, 'dp', { done: fin, restart: () => start(false) });
    const re = (nn) => {
      pb.stop();
      start(!!nn);
      draw();
    };
    $('#dp-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#dp-reset', root).onclick = () => re(false);
    $('#dp-new', root).onclick = () => {
      re(true);
      msg = 'Fresh random noise has been drawn. ' + msg;
      draw();
    };
    segBind(root, 'dp-dir', S.mode, (v) => {
      S.mode = v;
      pb.stop();
      start(false);
      draw();
      pb.start();
    });
    segBind(root, 'dp-sch', S.sch, (v) => {
      S.sch = v;
      re(false);
    });
    sel.onchange = (e) => {
      S.pic = +e.target.value;
      re(false);
    };
    bind(
      root,
      'dp-T',
      (v) => {
        S.T = +v;
        re(false);
      },
      (v) => String(v),
    );
    bind(
      root,
      'dp-b0',
      (v) => {
        S.b0 = +v;
        if (S.sch === 'cos') {
          sched();
          draw();
        } else re(false);
      },
      (v) => (+v).toFixed(4),
    );
    bind(
      root,
      'dp-b1',
      (v) => {
        S.b1 = +v;
        if (S.sch === 'cos') {
          sched();
          draw();
        } else re(false);
      },
      f2,
    );
    bind(
      root,
      'dp-turn',
      (v) => {
        S.turn = +v;
        re(false);
      },
      (v) => v + '%',
    );
    bind(
      root,
      'dp-v',
      (v) => {
        S.v = +v;
      },
      f2,
    );
    bind(
      root,
      'dp-rs',
      (v) => {
        S.rs = +v;
        if (phase === 'rev' && ti === 0) tau = mkTau(tau[0]);
      },
      (v) => v + '%',
    );
    $('#dp-file', root).onchange = (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (!f) return;
      const bad = () => {
          st.textContent = 'That file could not be opened as a picture. Try a JPEG or PNG file.';
        },
        use = (im) => {
          const iw = im.width,
            ih = im.height;
          if (!iw || !ih) {
            bad();
            return;
          }
          const o = document.createElement('canvas');
          o.width = o.height = N;
          const s = Math.min(iw, ih);
          o.getContext('2d').drawImage(im, (iw - s) / 2, (ih - s) / 2, s, s, 0, 0, N, N);
          GAL = GAL.filter((q) => !q[2]);
          GAL.push(['Your picture', fromCanvas(o), true]);
          S.pic = GAL.length - 1;
          fillSel();
          S.mode = 'round';
          root
            .querySelectorAll('#dp-dir button')
            .forEach((b) =>
              b.setAttribute('aria-pressed', b.dataset.v === 'round' ? 'true' : 'false'),
            );
          re(false);
          st.textContent =
            'Loaded ' +
            f.name +
            ', cropped to a square and shrunk to 32 × 32 pixels. It stays in your browser. The model now knows ' +
            GAL.length +
            ' pictures.';
          draw();
        };
      if (window.createImageBitmap) createImageBitmap(f).then(use, bad);
      else {
        const r2 = new FileReader();
        r2.onerror = bad;
        r2.onload = () => {
          const im = new Image();
          im.onload = () => use(im);
          im.onerror = bad;
          im.src = r2.result;
        };
        r2.readAsDataURL(f);
      }
    };
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        start(false);
        draw();
        pb.start();
      },
    };
  },
};
