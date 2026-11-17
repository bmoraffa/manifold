/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  bind,
  clamp,
  eqShow,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { g2Note } from '../../generative/shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart, uView } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#mc-c', root),
      cr = $('#mc-r', root),
      live = $('#mc-live', root),
      eq = $('#mc-eq', root),
      ol = $('#mc-steps', root),
      settings = { al: 'rw', tg: 'corr', s: 0.8, L: 15, cor: true, sp: 2 },
      MAXN = 8000,
      GW = 64,
      GH = 48,
      XR = 4,
      YR = 3,
      NB = 32,
      RN = randnFrom(Math.random);
    const LP = {
      corr: (x, y) =>
        (-(x * x - (2 * 0.85 * x * y) / 0.9 + (y * y) / 0.81) / (2 * (1 - 0.85 * 0.85))) * 0.8,
      banana: (x, y) => -((x * x) / 4.5 + 2.2 * (y + 0.9 - 0.35 * x * x) ** 2),
      two: (x, y) =>
        Math.log(
          Math.exp(-((x + 1.9) ** 2 + (y + 0.2) ** 2) / 0.55) +
            Math.exp(-((x - 1.9) ** 2 + (y - 0.2) ** 2) / 0.55) +
            1e-300,
        ),
    };
    const NAME = {
        rw: 'Random-walk Metropolis',
        mh: 'Metropolis-Hastings (independence proposal)',
        lang: 'Langevin',
        hmc: 'Hamiltonian Monte Carlo',
        nuts: 'No-U-Turn sampler',
      },
      DEF = { rw: 0.8, mh: 1.8, lang: 0.45, hmc: 0.2, nuts: 0.4 },
      SL = {
        rw: 'Stride: typical size of a proposed step',
        mh: 'Width of the fixed proposal bell',
        lang: 'Step size ε',
        hmc: 'Leapfrog step size ε',
        nuts: 'Leapfrog step size ε',
      };
    const HOW = {
      rw: [
        'Start the walker anywhere.',
        'Propose a move: the current place plus a random step. The stride slider sets how big steps tend to be.',
        'Compare the height of the target at the proposed place with the height at the current place.',
        'If the proposed place is higher, go there. If it is lower, go there only with probability equal to the ratio of the two heights. Otherwise stay put.',
        'Write down where the walker is now, even if it did not move. That is one sample.',
        'Repeat thousands of times. The written-down places are distributed like the target.',
      ],
      mh: [
        'Start the walker anywhere.',
        'Propose a place drawn from one fixed wide bell centred on the picture, wherever the walker is standing.',
        'Work out the ratio of target heights, as before.',
        'Multiply by the Hastings correction: how likely the bell is to propose the current place, divided by how likely it is to propose the new one.',
        'Accept with probability equal to that product (or 1 if it is larger). Otherwise stay put.',
        'Write down where the walker is now. Repeat.',
      ],
      lang: [
        'Start the walker anywhere.',
        'Feel the slope where the walker stands.',
        'Propose a move: a short drift uphill plus a random step.',
        'With the correction on, work out the target ratio times the Hastings correction for the uphill lean, and accept or reject. With it off, always move.',
        '—',
        'Write down where the walker is now. Repeat.',
      ],
      hmc: [
        'Start the walker anywhere.',
        'Give it a random kick: a fresh velocity in a random direction.',
        'Let it roll like a ball for the chosen number of leapfrog steps, feeling the slope at every step.',
        'Compare the total energy (height term plus speed term) at the end of the path with the start. Leapfrog nearly conserves it.',
        'Accept the end of the path with probability exp(energy at start − energy at end). Otherwise go back to the start.',
        'Write down where the walker is now. Repeat.',
      ],
      nuts: [
        'Start the walker anywhere.',
        'Give it a random kick, and draw a random energy threshold below its current level.',
        'Grow the path: pick forwards or backwards at random and double the path in that direction.',
        'Stop when the two ends of the path start to approach each other (a U-turn), or a point falls far below the threshold, or after six doublings.',
        'Pick the new place at random from the points on the path that sit above the threshold.',
        'Write down where the walker is now. Repeat.',
      ],
    };
    let lp,
      dens = null,
      dmax = 1,
      targetMarginalX = null,
      mxT = 0,
      targetRightMass = 0,
      x = 0,
      y = 0,
      XS = [],
      YS = [],
      nacc = 0,
      nev = 0,
      prop = null,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
      fin = false,
      err = NaN,
      P = null,
      hist = null,
      sB,
      cachedEss = null;
    const gr = (a, b) => {
        const e = 1e-4;
        return [(lp(a + e, b) - lp(a - e, b)) / (2 * e), (lp(a, b + e) - lp(a, b - e)) / (2 * e)];
      },
      out = (a, b) => Math.abs(a) > XR || Math.abs(b) > YR,
      ht = (l) => (l < -1e8 ? '0 (outside the picture)' : Math.exp(l).toExponential(2));
    const setup = () => {
      lp = LP[settings.tg];
      dens = new Float64Array(GW * GH);
      targetMarginalX = new Float64Array(NB);
      let tot = 0,
        sx = 0,
        rr = 0;
      for (let r = 0; r < GH; r++)
        for (let q = 0; q < GW; q++) {
          const px = -XR + (2 * XR * (q + 0.5)) / GW,
            py = YR - (2 * YR * (r + 0.5)) / GH,
            d = Math.exp(lp(px, py));
          dens[r * GW + q] = d;
          tot += d;
          sx += d * px;
          if (px > 0) rr += d;
          targetMarginalX[Math.min(NB - 1, Math.floor(((px + XR) / (2 * XR)) * NB))] += d;
        }
      dmax = Math.max(...dens);
      for (let i = 0; i < NB; i++) targetMarginalX[i] /= tot;
      mxT = sx / tot;
      targetRightMass = rr / tot;
    };
    const bin = (v) => Math.min(NB - 1, Math.max(0, Math.floor(((v + XR) / (2 * XR)) * NB)));
    const ui = () => {
      ol.innerHTML = HOW[settings.al]
        .filter((s) => s !== '—')
        .map((s) => '<li>' + s + '</li>')
        .join('');
      $('#mc-sl', root).textContent = SL[settings.al];
      const cb = $('#mc-cor', root),
        has = settings.al === 'mh' || settings.al === 'lang' || settings.al === 'hmc';
      cb.disabled = !has;
      cb.closest('label').style.opacity = has ? 1 : 0.45;
      $('#mc-corl', root).textContent =
        settings.al === 'mh'
          ? 'Include the Hastings correction'
          : settings.al === 'lang'
            ? 'Accept or reject each move (off: unadjusted Langevin)'
            : settings.al === 'hmc'
              ? 'Accept or reject the end of each path'
              : 'Apply the correction (not used by this walker)';
      $('#mc-L', root).closest('label').style.opacity = settings.al === 'hmc' ? 1 : 0.45;
      $('#mc-L', root).disabled = settings.al !== 'hmc';
    };
    const build = (keep) => {
      if (!keep) {
        x = settings.tg === 'two' ? -1.9 : -2.6;
        y = settings.tg === 'two' ? -0.2 : 1.6;
      }
      XS = [x];
      YS = [y];
      nacc = 0;
      nev = 0;
      prop = null;
      fin = false;
      activeStepIndex = 0;
      err = NaN;
      cachedEss = null;
      hist = new Float64Array(NB);
      hist[bin(x)]++;
      msg =
        NAME[settings.al] +
        '. The walker starts at (' +
        f2(x) +
        ', ' +
        f2(y) +
        '). Press One step to follow a single decision, or Run.';
      eqShow(eq, ['No steps yet.'], 0);
    };
    const leap = (st, e) => {
      let g0 = gr(st[0], st[1]);
      const px = st[2] + 0.5 * e * g0[0],
        py = st[3] + 0.5 * e * g0[1],
        nx = st[0] + e * px,
        ny = st[1] + e * py;
      g0 = gr(nx, ny);
      nev++;
      return [nx, ny, px + 0.5 * e * g0[0], py + 0.5 * e * g0[1]];
    };
    const lj = (st) =>
      (out(st[0], st[1]) ? -1e9 : lp(st[0], st[1])) - 0.5 * (st[2] * st[2] + st[3] * st[3]);
    const moves = {
      rw(quiet) {
        const px = x + settings.s * RN(),
          py = y + settings.s * RN(),
          l0 = lp(x, y),
          l1 = out(px, py) ? -1e9 : lp(px, py),
          ratio = Math.exp(Math.min(0, l1 - l0)),
          u = Math.random(),
          ok = u < ratio;
        nev++;
        prop = {
          path: [
            [x, y],
            [px, py],
          ],
          ok,
        };
        if (!quiet)
          eqShow(
            eq,
            [
              'Step ' +
                XS.length +
                '. The walker is at (' +
                f2(x) +
                ', ' +
                f2(y) +
                '), where the target has height ' +
                ht(l0) +
                '.',
              'Proposed place: (' + f2(px) + ', ' + f2(py) + '), height ' + ht(l1) + '.',
              'ratio = proposed height ÷ current height = <b>' +
                (l1 - l0 > 0 ? 'more than 1' : f3(Math.exp(l1 - l0))) +
                '</b>',
              l1 >= l0
                ? 'The proposed place is higher, so the move is <b>accepted</b>.'
                : 'It is lower, so roll a number between 0 and 1: ' +
                  f3(u) +
                  '. That is ' +
                  (ok
                    ? 'below the ratio, so the move is <b>accepted</b>.'
                    : 'above the ratio, so the move is <b>rejected</b>. The walker stays, and its current place is written down again.'),
            ],
            0,
          );
        return ok ? [px, py] : null;
      },
      mh(quiet) {
        const s = settings.s,
          px = s * RN(),
          py = s * RN(),
          l0 = lp(x, y),
          l1 = out(px, py) ? -1e9 : lp(px, py),
          lq = (a, b) => -(a * a + b * b) / (2 * s * s),
          hc = lq(x, y) - lq(px, py),
          la = l1 - l0 + (settings.cor ? hc : 0),
          u = Math.random(),
          ok = u < Math.exp(Math.min(0, la));
        nev++;
        prop = {
          path: [
            [x, y],
            [px, py],
          ],
          ok,
        };
        if (!quiet)
          eqShow(
            eq,
            [
              'Step ' +
                XS.length +
                '. The walker is at (' +
                f2(x) +
                ', ' +
                f2(y) +
                '). The fixed bell proposes (' +
                f2(px) +
                ', ' +
                f2(py) +
                ').',
              'target ratio = proposed height ÷ current height = <b>' +
                (l1 < -1e8 ? '0' : Math.exp(l1 - l0).toPrecision(3)) +
                '</b>',
              'Hastings correction = chance the bell proposes the current place ÷ chance it proposes the new one = <b>' +
                Math.exp(hc).toPrecision(3) +
                '</b>' +
                (settings.cor ? '' : ' (switched off, so it is left out)'),
              'acceptance probability = min(1, ' +
                (settings.cor ? 'ratio × correction' : 'ratio alone') +
                ') = <b>' +
                f3(Math.exp(Math.min(0, la))) +
                '</b>',
              'Rolled ' + f3(u) + ': the move is <b>' + (ok ? 'accepted' : 'rejected') + '</b>.',
            ],
            0,
          );
        return ok ? [px, py] : null;
      },
      lang(quiet) {
        const e = settings.s,
          g0 = gr(x, y),
          mx = x + 0.5 * e * e * g0[0],
          my = y + 0.5 * e * e * g0[1],
          px = mx + e * RN(),
          py = my + e * RN();
        nev++;
        if (!settings.cor) {
          const cx = clamp(px, -XR, XR),
            cy = clamp(py, -YR, YR);
          prop = {
            path: [
              [x, y],
              [mx, my],
              [cx, cy],
            ],
            ok: true,
          };
          if (!quiet)
            eqShow(
              eq,
              [
                'Step ' +
                  XS.length +
                  '. The walker is at (' +
                  f2(x) +
                  ', ' +
                  f2(y) +
                  '). The slope there is (' +
                  f2(g0[0]) +
                  ', ' +
                  f2(g0[1]) +
                  ').',
                'uphill drift = ε²/2 × slope = (' +
                  f2(0.5 * e * e * g0[0]) +
                  ', ' +
                  f2(0.5 * e * e * g0[1]) +
                  ')',
                'random step of typical size ε = ' + f2(e) + ' added on top',
                'New place: <b>(' +
                  f2(cx) +
                  ', ' +
                  f2(cy) +
                  ')</b>. There is no test, so the walker always moves. This is the unadjusted Langevin algorithm.',
              ],
              0,
            );
          return [cx, cy];
        }
        const l0 = lp(x, y),
          l1 = out(px, py) ? -1e9 : lp(px, py),
          g1 = gr(px, py),
          bx = px + 0.5 * e * e * g1[0],
          by = py + 0.5 * e * e * g1[1],
          lf = -((px - mx) ** 2 + (py - my) ** 2) / (2 * e * e),
          lb = -((x - bx) ** 2 + (y - by) ** 2) / (2 * e * e),
          hc = lb - lf,
          la = l1 - l0 + hc,
          u = Math.random(),
          ok = u < Math.exp(Math.min(0, la));
        prop = {
          path: [
            [x, y],
            [mx, my],
            [px, py],
          ],
          ok,
        };
        if (!quiet)
          eqShow(
            eq,
            [
              'Step ' +
                XS.length +
                '. The walker is at (' +
                f2(x) +
                ', ' +
                f2(y) +
                '). The slope there is (' +
                f2(g0[0]) +
                ', ' +
                f2(g0[1]) +
                ').',
              'Proposal = current place + uphill drift (' +
                f2(0.5 * e * e * g0[0]) +
                ', ' +
                f2(0.5 * e * e * g0[1]) +
                ') + random step = (' +
                f2(px) +
                ', ' +
                f2(py) +
                ')',
              'target ratio = <b>' +
                (l1 < -1e8 ? '0' : Math.exp(l1 - l0).toPrecision(3)) +
                '</b>, Hastings correction for the uphill lean = <b>' +
                Math.exp(hc).toPrecision(3) +
                '</b>',
              'acceptance probability = min(1, ratio × correction) = <b>' +
                f3(Math.exp(Math.min(0, la))) +
                '</b>',
              'Rolled ' + f3(u) + ': the move is <b>' + (ok ? 'accepted' : 'rejected') + '</b>.',
            ],
            0,
          );
        return ok ? [px, py] : null;
      },
      hmc(quiet) {
        let st = [x, y, RN(), RN()];
        const p0 = [st[2], st[3]],
          H0 = -lj(st),
          path = [[x, y]];
        for (let i = 0; i < settings.L; i++) {
          st = leap(st, settings.s);
          path.push([st[0], st[1]]);
        }
        const H1 = -lj(st),
          u = Math.random(),
          bad = out(st[0], st[1]),
          ok = !bad && (!settings.cor || u < Math.exp(Math.min(0, H0 - H1)));
        prop = { path, ok };
        if (!quiet)
          eqShow(
            eq,
            [
              'Step ' +
                XS.length +
                '. The walker is at (' +
                f2(x) +
                ', ' +
                f2(y) +
                ') and is kicked with velocity (' +
                f2(p0[0]) +
                ', ' +
                f2(p0[1]) +
                ').',
              'energy at the start = −log height + ½ speed² = <b>' + f2(H0) + '</b>',
              'It rolls for ' +
                settings.L +
                ' leapfrog steps of size ' +
                f2(settings.s) +
                ' and ends at (' +
                f2(st[0]) +
                ', ' +
                f2(st[1]) +
                ')' +
                (bad ? ', outside the picture' : '') +
                '.',
              'energy at the end = <b>' +
                (bad ? '∞' : f2(H1)) +
                '</b>, a change of ' +
                (bad ? '∞' : (H1 - H0 >= 0 ? '+' : '') + f3(H1 - H0)),
              bad
                ? 'The move is <b>rejected</b>.'
                : settings.cor
                  ? 'acceptance probability = min(1, exp(−change)) = <b>' +
                    f3(Math.exp(Math.min(0, H0 - H1))) +
                    '</b>. Rolled ' +
                    f3(u) +
                    ': <b>' +
                    (ok ? 'accepted' : 'rejected') +
                    '</b>.'
                  : 'The test is switched off, so the walker <b>moves</b> regardless.',
            ],
            0,
          );
        return ok ? [st[0], st[1]] : null;
      },
      nuts(quiet) {
        const p0 = [RN(), RN()],
          s0 = [x, y, p0[0], p0[1]],
          J0 = lj(s0),
          logu = J0 + Math.log(Math.random() || 1e-300),
          e = settings.s,
          fw = [],
          bw = [];
        let why = '',
          div = false;
        const nou = (a, b) =>
          (b[0] - a[0]) * a[2] + (b[1] - a[1]) * a[3] >= 0 &&
          (b[0] - a[0]) * b[2] + (b[1] - a[1]) * b[3] >= 0;
        const tree = (st, v, j) => {
          if (j === 0) {
            const s1 = leap(st, v * e),
              J = lj(s1);
            (v > 0 ? fw : bw).push([s1[0], s1[1]]);
            const s = logu < J + 1000;
            if (!s) div = true;
            return { m: s1, p: s1, c: s1, n: logu <= J ? 1 : 0, s };
          }
          const a = tree(st, v, j - 1);
          if (!a.s) return a;
          const b = tree(v < 0 ? a.m : a.p, v, j - 1),
            m = v < 0 ? b.m : a.m,
            p = v < 0 ? a.p : b.p,
            n = a.n + b.n;
          return {
            m,
            p,
            c: n > 0 && Math.random() < b.n / n ? b.c : a.c,
            n,
            s: b.s && nou(m, p),
          };
        };
        let m = s0,
          p = s0,
          cand = s0,
          n = 1,
          j = 0,
          s = true;
        while (s && j < 6) {
          const v = Math.random() < 0.5 ? -1 : 1,
            r = tree(v < 0 ? m : p, v, j);
          if (v < 0) m = r.m;
          else p = r.p;
          if (r.s && Math.random() < r.n / n) cand = r.c;
          n += r.n;
          const nt = nou(m, p);
          if (!r.s)
            why = div
              ? 'a point on the path fell far below the energy threshold'
              : 'a U-turn appeared inside the newest half of the path';
          else if (!nt) why = 'the two ends of the path began to approach each other';
          s = r.s && nt;
          j++;
        }
        if (s) why = 'it reached the limit of six doublings';
        const path = bw
            .slice()
            .reverse()
            .concat([[x, y]], fw),
          moved = cand !== s0;
        prop = { path, ok: true, pick: [cand[0], cand[1]], nuts: true };
        if (!quiet)
          eqShow(
            eq,
            [
              'Step ' +
                XS.length +
                '. The walker is at (' +
                f2(x) +
                ', ' +
                f2(y) +
                ') and is kicked with velocity (' +
                f2(p0[0]) +
                ', ' +
                f2(p0[1]) +
                ').',
              'The path grew in <b>' +
                j +
                '</b> round' +
                (j === 1 ? '' : 's') +
                ' of doubling, forwards or backwards at random each time, to ' +
                (fw.length + bw.length) +
                ' leapfrog steps.',
              'It stopped growing because ' + why + '.',
              '<b>' +
                n +
                '</b> of the ' +
                (fw.length + bw.length + 1) +
                ' points on the path sit above the energy threshold and were allowed.',
              'New place, picked from the allowed points: <b>(' +
                f2(cand[0]) +
                ', ' +
                f2(cand[1]) +
                ')</b>' +
                (moved ? '' : ' — the starting point itself') +
                '.',
            ],
            0,
          );
        return moved ? [cand[0], cand[1]] : null;
      },
    };
    const step = (quiet) => {
      if (fin) return;
      const x0 = x,
        y0 = y,
        r = moves[settings.al](quiet),
        ok = !!r;
      if (ok) {
        x = r[0];
        y = r[1];
        nacc++;
      }
      if (!quiet) {
        activeStepIndex =
          settings.al === 'rw' ? 3 : settings.al === 'mh' ? 4 : settings.al === 'lang' ? 3 : 4;
        msg =
          'Step ' +
          XS.length +
          ': ' +
          (ok
            ? 'moved to (' +
              f2(x) +
              ', ' +
              f2(y) +
              ')' +
              (settings.al === 'hmc' || settings.al === 'nuts'
                ? ', ' +
                  f2(Math.hypot(x - x0, y - y0)) +
                  ' away, along a path of ' +
                  (prop.path.length - 1) +
                  ' leapfrog steps.'
                : '.')
            : (settings.al === 'nuts'
                ? 'the starting point was picked from the path, so the walker stayed at ('
                : 'the proposal was rejected, so the walker stayed at (') +
              f2(x) +
              ', ' +
              f2(y) +
              ') and that place was recorded again.');
      }
      XS.push(x);
      YS.push(y);
      hist[bin(x)]++;
      cachedEss = null;
      const n = XS.length;
      if (n % 100 === 0 || n >= MAXN) {
        let e = 0;
        for (let i = 0; i < NB; i++) e += Math.abs(hist[i] / n - targetMarginalX[i]);
        err = e;
        if ((n >= 400 && e < 0.08) || n >= MAXN) {
          fin = true;
          activeStepIndex = 5;
          const es = ess();
          msg =
            NAME[settings.al] +
            '. ' +
            (e < 0.08
              ? 'Stopped after ' + (n - 1) + ' steps: the bars are within 0.08 of the true line.'
              : 'Stopped at the limit of ' +
                MAXN +
                ' steps with a gap of ' +
                f2(e) +
                ' still left' +
                (settings.tg === 'two' && Math.abs(rightShare() - targetRightMass) > 0.15
                  ? '. The walker has not divided its time correctly between the two islands'
                  : !settings.cor &&
                      (settings.al === 'mh' || settings.al === 'lang' || settings.al === 'hmc')
                    ? '. With the correction off this walker samples from a slightly different landscape, and no amount of running fixes that'
                    : '') +
                '.') +
            ' ' +
            pct(nacc / (n - 1)) +
            ' of moves changed place, the samples are worth about ' +
            Math.round(es) +
            ' independent ones, and they cost ' +
            nev.toLocaleString() +
            ' look-ups.';
        }
      }
    };
    const rightShare = () => {
      let r = 0;
      for (const v of XS) if (v > 0) r++;
      return r / XS.length;
    };
    const ess = () => {
      if (cachedEss != null) return cachedEss;
      const n = XS.length;
      if (n < 50) return NaN;
      let m = 0;
      for (const v of XS) m += v / n;
      let v0 = 0;
      for (const v of XS) v0 += (v - m) ** 2 / n;
      if (v0 < 1e-12) return (cachedEss = 1);
      let s = 0;
      const K = Math.min(600, Math.floor(n / 3));
      for (let k = 1; k < K; k++) {
        let cc = 0;
        for (let i = 0; i + k < n; i++) cc += (XS[i] - m) * (XS[i + k] - m);
        const r = cc / (n - k) / v0;
        if (r < 0.02) break;
        s += r;
      }
      return (cachedEss = Math.min(n, n / (1 + 2 * s)));
    };
    const draw = () => {
      const colors = readThemeColors(),
        n = XS.length;
      {
        const V = uView(c),
          { g, w, h, X, Y } = V;
        P = V;
        for (let r = 0; r < GH; r++)
          for (let q = 0; q < GW; q++) {
            const v = dens[r * GW + q] / dmax;
            if (v < 0.01) continue;
            g.fillStyle = mix(
              colors.accent,
              colors.accent,
              0,
              Math.min(0.85, Math.pow(v, 0.55) * 0.8),
            );
            const xa = Math.round(X(-XR + (2 * XR * q) / GW)),
              xb = Math.round(X(-XR + (2 * XR * (q + 1)) / GW)),
              ya = Math.round(Y(YR - (2 * YR * r) / GH)),
              yb = Math.round(Y(YR - (2 * YR * (r + 1)) / GH));
            g.fillRect(xa, ya, xb - xa, yb - ya);
          }
        if (settings.al === 'mh') {
          g.strokeStyle = mix(colors.cut, colors.cut, 0, 0.55);
          g.lineWidth = 1.2;
          g.setLineDash([4, 5]);
          for (const k of [1, 2]) {
            g.beginPath();
            g.ellipse(X(0), Y(0), k * settings.s * V.s, k * settings.s * V.s, 0, 0, 7);
            g.stroke();
          }
          g.setLineDash([]);
        }
        g.fillStyle = mix(colors.ink, colors.ink, 0, 0.45);
        const st = Math.max(1, Math.floor(n / 1500));
        for (let i = 0; i < n; i += st) {
          g.beginPath();
          g.arc(X(XS[i]), Y(YS[i]), 1.5, 0, 7);
          g.fill();
        }
        g.strokeStyle = colors.ink;
        g.lineWidth = 1;
        g.globalAlpha = 0.6;
        g.beginPath();
        for (let i = Math.max(0, n - 60); i < n; i++)
          i === Math.max(0, n - 60) ? g.moveTo(X(XS[i]), Y(YS[i])) : g.lineTo(X(XS[i]), Y(YS[i]));
        g.stroke();
        g.globalAlpha = 1;
        if (prop) {
          const pp = prop.path,
            e = pp[pp.length - 1];
          g.strokeStyle = prop.ok ? colors.gold : colors.cut;
          g.lineWidth = 2.5;
          g.setLineDash(prop.ok ? [] : [5, 4]);
          g.beginPath();
          pp.forEach((q, i) => (i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1]))));
          g.stroke();
          g.setLineDash([]);
          if (pp.length > 3) {
            g.fillStyle = prop.ok ? colors.gold : colors.cut;
            for (const q of pp) {
              g.beginPath();
              g.arc(X(q[0]), Y(q[1]), 2.2, 0, 7);
              g.fill();
            }
          }
          if (prop.nuts) {
            for (const q of [pp[0], e]) {
              g.strokeStyle = colors.cut;
              g.lineWidth = 2;
              g.beginPath();
              g.arc(X(q[0]), Y(q[1]), 5, 0, 7);
              g.stroke();
            }
            g.font = MONO;
            g.fillStyle = colors.muted;
            g.textAlign = 'left';
            g.fillText('path ends', X(e[0]) + 8, Y(e[1]) + 4);
          } else {
            g.beginPath();
            g.arc(X(e[0]), Y(e[1]), 5, 0, 7);
            g.stroke();
            g.font = MONO;
            g.fillStyle = prop.ok ? colors.cut : colors.muted;
            g.textAlign = 'left';
            g.fillText(
              prop.ok ? (settings.cor || settings.al === 'rw' ? 'accepted' : 'moved') : 'rejected',
              X(e[0]) + 8,
              Y(e[1]) + 4,
            );
          }
        }
        g.fillStyle = colors.gold;
        g.strokeStyle = colors.bg;
        g.lineWidth = 2;
        g.beginPath();
        g.arc(X(x), Y(y), 7, 0, 7);
        g.fill();
        g.stroke();
        g2Note(
          g,
          colors,
          h,
          'the gold dot is the walker' +
            (settings.al === 'mh' ? '; dashed rings: the fixed proposal bell' : ''),
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.46,
          A = drawSeriesChart(ctx, colors, 0, 0, w, h1, [], {
            title: 'left-to-right position at every step',
            n: Math.max(300, n),
            y1: -XR,
            ym: XR,
          });
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        const s2 = Math.max(1, Math.floor(n / 1200));
        for (let i = 0; i < n; i += s2)
          i ? ctx.lineTo(A.X(i), A.Y(XS[i])) : ctx.moveTo(A.X(i), A.Y(XS[i]));
        ctx.stroke();
        const m = { l: 44, r: 12, t: 26, b: 20 },
          y0 = h1,
          H = h - h1,
          bw = (w - m.l - m.r) / NB,
          mx = Math.max(...targetMarginalX, ...Array.from(hist).map((v) => v / n)) * 1.08,
          Yh = (v) => y0 + H - m.b - (v / mx) * (H - m.t - m.b);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          'time spent at each left-to-right position (bars) and the truth (line)',
          8,
          y0 + 13,
        );
        for (let i = 0; i < NB; i++) {
          ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.65);
          ctx.fillRect(m.l + i * bw + 0.5, Yh(hist[i] / n), bw - 1, Yh(0) - Yh(hist[i] / n));
        }
        ctx.strokeStyle = colors.cut;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (let i = 0; i < NB; i++)
          i
            ? ctx.lineTo(m.l + (i + 0.5) * bw, Yh(targetMarginalX[i]))
            : ctx.moveTo(m.l + 0.5 * bw, Yh(targetMarginalX[i]));
        ctx.stroke();
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, y0 + H - m.b);
        ctx.lineTo(w - m.r, y0 + H - m.b);
        ctx.stroke();
        ctx.fillStyle = colors.muted;
        ctx.fillText('left', m.l, h - 5);
        ctx.textAlign = 'right';
        ctx.fillText('right', w - m.r, h - 5);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'mc', activeStepIndex);
      let mx = 0;
      for (const v of XS) mx += v / n;
      const es = fin || !anim.on || n % 200 < 41 ? ess() : cacheE0;
      cacheE0 = es;
      setReadout(root, 'n', n - 1 + (fin ? ' (stopped)' : ''));
      setReadout(root, 'acc', n > 1 ? pct(nacc / (n - 1)) : '–');
      setReadout(root, 'mx', f2(mx) + ' / ' + f2(mxT));
      setReadout(root, 'ess', isFinite(es) ? String(Math.round(es)) : '–');
      setReadout(root, 'cost', nev.toLocaleString());
      setReadout(root, 'eff', isFinite(es) && nev > 0 ? ((1000 * es) / nev).toFixed(0) : '–');
      setReadout(root, 'err', isFinite(err) ? f2(err) : '–');
      setReadout(root, 'isl', pct(rightShare()) + ' / ' + pct(targetRightMass));
    };
    let cacheE0 = NaN;
    setup();
    ui();
    build(false);
    if (prefersReducedMotion) {
      while (!fin) step(true);
    }
    const anim = Anim(() => {
      frameCount++;
      if (fin) {
        playback.stop();
        return;
      }
      const hv = settings.al === 'hmc' || settings.al === 'nuts';
      if (settings.sp === 1) {
        if (frameCount % 34) return;
        step(false);
      } else if (settings.sp === 2) {
        if (XS.length < 25 || hv) {
          if (frameCount % (hv && XS.length >= 25 ? 3 : 10)) return;
          step(false);
        } else for (let i = 0; i < 10 && !fin && checkRunStop.remaining() > 0; i++) step(i < 9);
      } else {
        const k = hv ? 10 : 40;
        for (let i = 0; i < k && !fin && checkRunStop.remaining() > 0; i++) step(i < k - 1);
      }
      draw();
      checkRunStop();
    });
    playback = playBtn($('#mc-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'mc', {
      done: () => fin,
      restart: () => build(false),
      count: () => XS.length - 1,
      what: 'steps',
    });
    $('#mc-step', root).onclick = () => {
      playback.stop();
      if (fin) build(false);
      else step(false);
      draw();
    };
    $('#mc-reset', root).onclick = () => {
      playback.stop();
      build(false);
      draw();
    };
    sB = bind(
      root,
      'mc-s',
      (v) => {
        settings.s = +v;
      },
      f2,
    );
    bind(
      root,
      'mc-L',
      (v) => {
        settings.L = +v;
      },
      (v) => String(+v),
    );
    bind(
      root,
      'mc-sp',
      (v) => {
        settings.sp = +v;
      },
      (v) => ['', 'one step at a time', 'steady', 'fast'][+v],
    );
    $('#mc-cor', root).onchange = (e) => {
      settings.cor = e.target.checked;
      playback.stop();
      build(false);
      draw();
    };
    segBind(root, 'mc-al', settings.al, (v) => {
      settings.al = v;
      settings.s = DEF[v];
      sB.set(settings.s);
      playback.stop();
      ui();
      build(false);
      draw();
      playback.start();
    });
    segBind(root, 'mc-tg', settings.tg, (v) => {
      settings.tg = v;
      playback.stop();
      setup();
      build(false);
      draw();
      playback.start();
    });
    c.style.cursor = 'crosshair';
    c.addEventListener('click', (e) => {
      if (!P) return;
      const r = c.getBoundingClientRect();
      x = clamp((e.clientX - r.left - P.w / 2) / P.s, -XR, XR);
      y = clamp(-(e.clientY - r.top - P.h / 2) / P.s, -YR, YR);
      playback.stop();
      build(true);
      msg =
        'The walker has been put at (' +
        f2(x) +
        ', ' +
        f2(y) +
        ') and its diary cleared. Press Run or One step.';
      draw();
    });
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        build(false);
        draw();
        playback.start();
      },
    };
  },
};
