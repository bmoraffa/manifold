/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  bind,
  eqShow,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#td-c', root),
      cr = $('#td-r', root),
      live = $('#td-live', root),
      eq = $('#td-eq', root),
      S = { al: 0.1, sp: 2 },
      N = 5,
      TRUE = [1, 2, 3, 4, 5].map((v) => v / 6),
      MAXE = 100,
      NM = 'ABCDE';
    let VT,
      VM,
      pos,
      path,
      ep = 0,
      ET = [],
      EM = [],
      AT = [],
      AM = [],
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      fin = false,
      lastEnd = '–',
      lastN = 0,
      eqT = [],
      eqM = [];
    const rms = (V) => Math.sqrt(V.reduce((s, v, i) => s + (v - TRUE[i]) ** 2, 0) / N);
    const batch = () => {
      AT = new Array(MAXE + 1).fill(0);
      AM = new Array(MAXE + 1).fill(0);
      const R = 100;
      for (let r = 0; r < R; r++) {
        const a = new Array(N).fill(0.5),
          b = new Array(N).fill(0.5);
        AT[0] += rms(a) / R;
        AM[0] += rms(b) / R;
        for (let e = 1; e <= MAXE; e++) {
          let p = 2;
          const vis = [];
          for (;;) {
            vis.push(p);
            const q = p + (Math.random() < 0.5 ? -1 : 1),
              rw = q >= N ? 1 : 0,
              nv = q < 0 || q >= N ? 0 : a[q];
            a[p] += S.al * (rw + nv - a[p]);
            p = q;
            if (p < 0 || p >= N) break;
          }
          const G0 = p >= N ? 1 : 0;
          for (const s of vis) b[s] += S.al * (G0 - b[s]);
          AT[e] += rms(a) / R;
          AM[e] += rms(b) / R;
        }
      }
    };
    const build = () => {
      VT = new Array(N).fill(0.5);
      VM = new Array(N).fill(0.5);
      pos = 2;
      path = [2];
      ep = 0;
      ET = [rms(VT)];
      EM = [rms(VM)];
      fin = false;
      cu = 0;
      lastEnd = '–';
      lastN = 0;
      eqT = [];
      eqM = [];
      batch();
      msg =
        'Both methods start by guessing 0.5 for every square. The walker is on C. Press One step or Run.';
      eqShow(eq, ['No steps yet.'], 0);
    };
    const step = (quiet) => {
      if (fin) return;
      const p = pos,
        q = p + (Math.random() < 0.5 ? -1 : 1),
        off = q < 0 || q >= N,
        rw = q >= N ? 1 : 0,
        nv = off ? 0 : VT[q],
        old = VT[p],
        nw = old + S.al * (rw + nv - old);
      VT[p] = nw;
      eqT = [
        '<b>TD</b>, step ' +
          path.length +
          ' of walk ' +
          (ep + 1) +
          ': from ' +
          NM[p] +
          ' to ' +
          (off ? (q < 0 ? 'off the left edge' : 'off the right edge') : NM[q]) +
          ', reward ' +
          rw +
          '.',
        '   target = reward + value of next = ' +
          rw +
          ' + ' +
          (off ? '0 (the walk is over)' : f3(nv)) +
          ' = ' +
          f3(rw + nv),
        '   V(' +
          NM[p] +
          ') = ' +
          f3(old) +
          ' + ' +
          f2(S.al) +
          ' × (' +
          f3(rw + nv) +
          ' − ' +
          f3(old) +
          ') = <b>' +
          f3(nw) +
          '</b>',
      ];
      pos = q;
      cu = 2;
      if (!off) {
        path.push(q);
        eqM = ['<b>Monte Carlo</b>: waiting. It cannot update anything until the walk ends.'];
        msg =
          'The walker stepped from ' +
          NM[p] +
          ' to ' +
          NM[q] +
          '. TD moved V(' +
          NM[p] +
          ') from ' +
          f3(old) +
          ' to ' +
          f3(nw) +
          '. Monte Carlo is still waiting.';
      } else {
        const G0 = rw,
          vis = path.slice(),
          ch = [];
        for (const s of vis) {
          const o = VM[s];
          VM[s] += S.al * (G0 - VM[s]);
          if (ch.length < 4)
            ch.push(
              'V(' +
                NM[s] +
                ') = ' +
                f3(o) +
                ' + ' +
                f2(S.al) +
                ' × (' +
                G0 +
                ' − ' +
                f3(o) +
                ') = <b>' +
                f3(VM[s]) +
                '</b>',
            );
        }
        eqM = [
          '<b>Monte Carlo</b>: the walk ended ' +
            (G0 ? 'on the right, so the return is 1' : 'on the left, so the return is 0') +
            ' for all ' +
            vis.length +
            ' squares visited.',
        ].concat(
          ch.map((x) => '   ' + x),
          vis.length > 4 ? ['   … and ' + (vis.length - 4) + ' more in the same way'] : [],
        );
        ep++;
        ET.push(rms(VT));
        EM.push(rms(VM));
        lastEnd = G0 ? 'on the right (reward 1)' : 'on the left (reward 0)';
        lastN = vis.length;
        cu = 4;
        msg =
          'Walk ' +
          ep +
          ' ended ' +
          lastEnd +
          ' after ' +
          lastN +
          ' steps. Monte Carlo now updated every visited square toward ' +
          G0 +
          '. Error: TD ' +
          f3(ET[ep]) +
          ', Monte Carlo ' +
          f3(EM[ep]) +
          '.';
        pos = 2;
        path = [2];
        if (ep >= MAXE) {
          fin = true;
          msg =
            'Finished: 100 walks. Error of TD ' +
            f3(ET[ep]) +
            ', of Monte Carlo ' +
            f3(EM[ep]) +
            ' in this run. Averaged over 100 runs the errors after 100 walks are ' +
            f3(AT[MAXE]) +
            ' and ' +
            f3(AM[MAXE]) +
            '. With a fixed α neither settles exactly.';
        }
      }
      if (!quiet) eqShow(eq, eqT.concat(eqM), 0);
    };
    const draw = () => {
      const t = T();
      {
        const { g, w, h } = fit(c),
          m = 30,
          cw = (w - 2 * m) / (N + 2),
          y1 = 54,
          X = (i) => m + (i + 1.5) * cw;
        g.font = MONO;
        g.strokeStyle = t.muted;
        g.lineWidth = 1.5;
        g.beginPath();
        g.moveTo(X(-1), y1);
        g.lineTo(X(N), y1);
        g.stroke();
        for (let i = -1; i <= N; i++) {
          const end = i < 0 || i >= N;
          g.beginPath();
          if (end) g.rect(X(i) - 15, y1 - 15, 30, 30);
          else g.arc(X(i), y1, 17, 0, 7);
          g.fillStyle = end
            ? i < 0
              ? t.bg
              : mix(t.gold, t.gold, 0, 0.6)
            : path.includes(i)
              ? mix(t.accent, t.accent, 0, 0.25)
              : t.bg;
          g.fill();
          g.strokeStyle = t.ink;
          g.stroke();
          g.fillStyle = t.ink;
          g.textAlign = 'center';
          g.font = '600 13px ' + MONO.split('px ')[1];
          g.fillText(end ? (i < 0 ? '0' : '+1') : NM[i], X(i), y1 + 5);
        }
        g.fillStyle = t.ink;
        g.beginPath();
        g.arc(X(pos), y1 - 30, 7, 0, 7);
        g.fill();
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        g.fillText('walker', X(pos), y1 - 42);
        const y0 = h - 44,
          H = h - y1 - 100,
          Y = (v) => y0 - v * H;
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        for (const v of [0, 0.5, 1]) {
          g.beginPath();
          g.moveTo(X(0) - cw / 2, Y(v));
          g.lineTo(X(N - 1) + cw / 2, Y(v));
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText(f2(v), X(0) - cw / 2 - 4, Y(v) + 4);
        }
        for (let i = 0; i < N; i++) {
          const bw = cw * 0.26;
          g.fillStyle = t.accent;
          g.fillRect(X(i) - bw - 2, Y(VT[i]), bw, Y(0) - Y(VT[i]));
          g.fillStyle = t.cut;
          g.fillRect(X(i) + 2, Y(VM[i]), bw, Y(0) - Y(VM[i]));
          g.strokeStyle = t.ink;
          g.lineWidth = 2.5;
          g.beginPath();
          g.moveTo(X(i) - bw - 8, Y(TRUE[i]));
          g.lineTo(X(i) + bw + 8, Y(TRUE[i]));
          g.stroke();
          g.textAlign = 'center';
          g.fillStyle = t.ink;
          g.fillText(NM[i], X(i), y0 + 13);
          g.fillStyle = t.muted;
          g.fillText('true ' + f2(TRUE[i]), X(i), y0 + 25);
          g.fillText(f2(VT[i]) + ' | ' + f2(VM[i]), X(i), y0 + 37);
        }
        g.textAlign = 'left';
        g.fillStyle = t.accent;
        g.fillText('TD', X(0) - cw / 2, y1 + 38);
        g.fillStyle = t.cut;
        g.fillText('Monte Carlo', X(0) - cw / 2 + 30, y1 + 38);
      }
      {
        const { g, w, h } = fit(cr),
          two = (y0, H, a, b, title) => {
            const L = uLine(g, t, 0, y0, w, H, a, { title, n: MAXE + 1, ym: 0.3, col: t.accent });
            g.strokeStyle = t.cut;
            g.lineWidth = 2.2;
            g.beginPath();
            b.forEach((v, i) =>
              i ? g.lineTo(L.X(i), L.Y(Math.min(v, 0.3))) : g.moveTo(L.X(i), L.Y(Math.min(v, 0.3))),
            );
            g.stroke();
            g.font = MONO;
            g.textAlign = 'right';
            g.fillStyle = t.accent;
            g.fillText('TD', w - 14, y0 + 38);
            g.fillStyle = t.cut;
            g.fillText('Monte Carlo', w - 14, y0 + 52);
          };
        two(
          0,
          h / 2,
          ET.map((v) => Math.min(v, 0.3)),
          EM,
          'error of the five estimates, this run',
        );
        two(h / 2, h / 2, AT, AM, 'the same, averaged over 100 runs with α = ' + f2(S.al));
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('walks →', w - 12, h - 5);
      }
      live.textContent = msg;
      uCur(root, 'td', cu);
      const mean = (a) => a.slice(1).reduce((x, y) => x + y, 0) / MAXE;
      setR(root, 'ep', ep + ' of ' + MAXE);
      setR(root, 'et', f3(rms(VT)));
      setR(root, 'em', f3(rms(VM)));
      setR(root, 'lw', lastEnd);
      setR(root, 'ls', lastN ? String(lastN) : '–');
      setR(root, 'avg', f3(mean(AT)) + ' / ' + f3(mean(AM)));
    };
    build();
    if (RM) {
      while (!fin) step(true);
    }
    const anim = Anim(() => {
      fr++;
      if (fin) {
        pb.stop();
        return;
      }
      if (S.sp === 1) {
        if (fr % 30) return;
        step(false);
      } else if (S.sp === 2) {
        if (ep < 3) {
          if (fr % 8) return;
          step(false);
        } else {
          if (fr % 3) return;
          const e0 = ep;
          while (ep === e0 && !fin && chk.remaining() > 0) step(false);
        }
      } else {
        const e0 = ep;
        while (ep < e0 + 2 && !fin && chk.remaining() > 0) step(false);
      }
      draw();
      chk();
    });
    pb = playBtn($('#td-play', root), anim);
    chk = runCtl(pb, root, 'td', {
      done: () => fin,
      restart: () => build(),
      count: () => ep,
      what: 'walks',
    });
    $('#td-step', root).onclick = () => {
      pb.stop();
      if (fin) build();
      else step(false);
      draw();
    };
    $('#td-reset', root).onclick = () => {
      pb.stop();
      build();
      draw();
    };
    bind(
      root,
      'td-al',
      (v) => {
        S.al = +v;
        if (!pb) return;
        pb.stop();
        build();
        draw();
      },
      f2,
    );
    bind(
      root,
      'td-sp',
      (v) => {
        S.sp = +v;
      },
      (v) => ['', 'one step at a time', 'steady', 'fast'][+v],
    );
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
