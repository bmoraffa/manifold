/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  SANS,
  readThemeColors,
  bind,
  clamp,
  prepareCanvas,
  mix,
  playBtn,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import {
  canvasXY,
  dotAt,
  segBind,
  highlightLessonStep,
  drawSeriesChart,
  uView,
} from '../../unsupervised/shared.js';
import { g2Note } from '../../generative/shared.js';

export default {
  init(root) {
    const c = $('#vc-c', root),
      cr = $('#vc-r', root),
      live = $('#vc-live', root),
      settings = { h: 'line', n: 3, sp: 2 },
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
      nextLabelingMask = 0,
      matchesByLabeling = [],
      hyp = null,
      fin = false,
      msg = '',
      activeStepIndex = 0,
      playback,
      checkRunStop = () => false,
      V = null,
      drag = -1,
      frameCount = 0,
      firstFail = -1;
    const lab = (m, i) => (m >> i) & 1;
    const findMatchingHypothesis = (m) => {
      const n = settings.n,
        pos = [],
        neg = [];
      for (let i = 0; i < n; i++) (lab(m, i) ? pos : neg).push(P[i]);
      if (settings.h === 'int') {
        if (!pos.length) return { t: 'int', a: 9, b: 9 };
        const a = Math.min(...pos.map((p) => p[0])),
          b = Math.max(...pos.map((p) => p[0]));
        return neg.some((p) => p[0] >= a - 1e-9 && p[0] <= b + 1e-9)
          ? null
          : { t: 'int', a: a - 0.25, b: b + 0.25 };
      }
      if (settings.h === 'rect') {
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
      if (kind === 'good') P = GOOD[settings.h].map((p) => p.slice());
      else if (kind === 'row')
        P = Array.from({ length: n }, (_, i) => [
          -3 + 1.2 * i,
          settings.h === 'int' ? 0 : -1.2 + 0.48 * i,
        ]);
      else
        P = Array.from({ length: n }, () => [
          -3.3 + 6.6 * Math.random(),
          settings.h === 'int' ? 0 : -2.3 + 4.6 * Math.random(),
        ]);
    };
    const reset = () => {
      nextLabelingMask = 0;
      matchesByLabeling = [];
      hyp = null;
      fin = false;
      firstFail = -1;
      activeStepIndex = 0;
      msg =
        'There are ' +
        (1 << settings.n) +
        ' ways to label ' +
        settings.n +
        ' point' +
        (settings.n === 1 ? '' : 's') +
        '. Press Run to try them all, or Next labelling to go one at a time.';
    };
    const str = (m) =>
      Array.from({ length: settings.n }, (_, i) => (lab(m, i) ? '+' : '−')).join(' ');
    const step = () => {
      if (fin) return;
      const tot = 1 << settings.n;
      hyp = findMatchingHypothesis(nextLabelingMask);
      matchesByLabeling.push(!!hyp);
      if (!hyp && firstFail < 0) firstFail = nextLabelingMask;
      activeStepIndex = 2;
      msg =
        'Labelling ' +
        (nextLabelingMask + 1) +
        ' of ' +
        tot +
        ' (' +
        str(nextLabelingMask) +
        '): ' +
        (hyp ? 'matched.' : 'no member of the family can do this.');
      nextLabelingMask++;
      if (nextLabelingMask >= tot) {
        fin = true;
        const ok = matchesByLabeling.filter(Boolean).length,
          sh = ok === tot;
        activeStepIndex = sh ? 3 : 4;
        msg =
          'All ' +
          tot +
          ' labellings tried: ' +
          ok +
          ' matched. ' +
          (sh
            ? 'These ' +
              settings.n +
              ' points are shattered, so the VC dimension is at least ' +
              settings.n +
              '.' +
              (settings.n === VC[settings.h]
                ? ' It is exactly ' +
                  settings.n +
                  ': no arrangement of ' +
                  (settings.n + 1) +
                  ' points can be shattered.'
                : '')
            : 'These points are not shattered. ' +
              (settings.n <= VC[settings.h]
                ? 'Another arrangement of ' + settings.n + ' points can be: try A good arrangement.'
                : 'No arrangement of ' +
                  settings.n +
                  ' points can be, because the VC dimension of this family is ' +
                  VC[settings.h] +
                  '.'));
      }
    };
    const draw = () => {
      const colors = readThemeColors();
      V = uView(c);
      const { g, w, h, X, Y } = V,
        m = fin
          ? firstFail >= 0
            ? firstFail
            : nextLabelingMask - 1
          : Math.max(0, nextLabelingMask - 1),
        show = fin && firstFail >= 0 ? null : hyp;
      if (settings.h === 'int') {
        g.strokeStyle = colors.line;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X(-4), Y(0));
        g.lineTo(X(4), Y(0));
        g.stroke();
      }
      if (show && nextLabelingMask > 0) {
        g.fillStyle = mix(colors.gold, colors.gold, 0, 0.16);
        g.strokeStyle = colors.gold;
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
      for (let i = 0; i < settings.n; i++) {
        const on = nextLabelingMask > 0 || fin ? lab(m, i) : 0;
        g.lineWidth = 2.5;
        dotAt(
          g,
          X(P[i][0]),
          Y(P[i][1]),
          10,
          nextLabelingMask > 0 ? (on ? colors.gold : colors.ink) : colors.muted,
          colors.bg,
        );
        g.font = SANS;
        g.fillStyle = colors.bg;
        g.textAlign = 'center';
        g.fillText(
          nextLabelingMask > 0 ? (on ? '+' : '−') : String(i + 1),
          X(P[i][0]),
          Y(P[i][1]) + 4.5,
        );
      }
      g2Note(
        g,
        colors,
        h,
        nextLabelingMask === 0
          ? 'drag the points, then press Run'
          : fin && firstFail >= 0
            ? 'showing the first labelling that cannot be matched: ' + str(firstFail)
            : hyp
              ? 'gold area: the positive side of the matching model'
              : 'no model matches this labelling',
      );
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          tot = 1 << settings.n,
          cols = tot <= 8 ? tot : tot <= 16 ? 8 : tot <= 32 ? 8 : 16,
          rows = Math.ceil(tot / cols),
          tw = Math.min(46, (w - 40) / cols),
          th = Math.min(34, (h1 - 34) / rows),
          ox = (w - cols * tw) / 2,
          oy = 26;
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('all ' + tot + ' labellings', 8, 13);
        for (let q = 0; q < tot; q++) {
          const x = ox + (q % cols) * tw,
            y = oy + Math.floor(q / cols) * th,
            dn = q < matchesByLabeling.length;
          ctx.strokeStyle = q === nextLabelingMask - 1 && !fin ? colors.gold : colors.line;
          ctx.lineWidth = q === nextLabelingMask - 1 && !fin ? 2.5 : 1;
          ctx.fillStyle = dn
            ? matchesByLabeling[q]
              ? mix(colors.accent, colors.accent, 0, 0.55)
              : mix(colors.cut, colors.cut, 0, 0.2)
            : 'transparent';
          ctx.fillRect(x + 2, y + 2, tw - 4, th - 4);
          ctx.strokeRect(x + 2, y + 2, tw - 4, th - 4);
          if (dn && !matchesByLabeling[q]) {
            ctx.strokeStyle = colors.cut;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x + 6, y + 6);
            ctx.lineTo(x + tw - 6, y + th - 6);
            ctx.moveTo(x + tw - 6, y + 6);
            ctx.lineTo(x + 6, y + th - 6);
            ctx.stroke();
          }
          if (tw > 30) {
            for (let i = 0; i < settings.n; i++)
              dotAt(
                ctx,
                x + tw / 2 + (i - (settings.n - 1) / 2) * Math.min(7, (tw - 10) / settings.n),
                y + th / 2,
                2,
                lab(q, i) ? colors.gold : colors.ink,
              );
          }
        }
        const gf = (n) =>
            settings.h === 'int'
              ? (n * (n - 1)) / 2 + n + 1
              : settings.h === 'line'
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
          A = drawSeriesChart(ctx, colors, 0, h1, w, h - h1, [], {
            title:
              'labellings on n points, n = 1 to 8 (log scale): family (solid), all 2ⁿ (dashed)',
            n: 8,
            ym: 8,
            y1: 1,
          });
        ctx.strokeStyle = colors.cut;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        NS.forEach((n, i) => (i ? ctx.lineTo(A.X(i), A.Y(n)) : ctx.moveTo(A.X(i), A.Y(n))));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        NS.forEach((n, i) => {
          const v = Math.log2(Math.min(gf(n), 2 ** n));
          i ? ctx.lineTo(A.X(i), A.Y(v)) : ctx.moveTo(A.X(i), A.Y(v));
        });
        ctx.stroke();
        dotAt(
          ctx,
          A.X(settings.n - 1),
          A.Y(Math.log2(Math.min(gf(settings.n), 2 ** settings.n))),
          5,
          colors.gold,
          colors.bg,
        );
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(A.X(VC[settings.h] - 1), A.Y(1));
        ctx.lineTo(A.X(VC[settings.h] - 1), A.Y(8));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('VC dimension', A.X(VC[settings.h] - 1) + 5, A.Y(7.6));
      }
      const ok = matchesByLabeling.filter(Boolean).length;
      live.textContent = msg;
      highlightLessonStep(root, 'vc', activeStepIndex);
      setReadout(root, 'n', settings.n);
      setReadout(
        root,
        'done',
        matchesByLabeling.length + ' of ' + (1 << settings.n) + (fin ? ' (finished)' : ''),
      );
      setReadout(root, 'ok', String(ok));
      setReadout(root, 'sh', fin ? (ok === 1 << settings.n ? 'Yes' : 'No') : '–');
      setReadout(root, 'vc', VC[settings.h]);
      setReadout(root, 'fail', firstFail >= 0 ? str(firstFail) : fin ? 'none' : '–');
    };
    place('good');
    reset();
    if (prefersReducedMotion) {
      while (!fin) step();
    }
    const anim = Anim(() => {
      if (fin) {
        playback.stop();
        return;
      }
      if (
        ++frameCount % (settings.sp === 1 ? 40 : settings.sp === 2 ? (settings.n > 4 ? 5 : 14) : 2)
      )
        return;
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#vc-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'vc', {
      done: () => fin,
      restart: reset,
      count: () => nextLabelingMask,
      what: 'labellings',
    });
    const go = () => {
      playback.stop();
      reset();
      draw();
      frameCount = 0;
      playback.start();
    };
    $('#vc-step', root).onclick = () => {
      playback.stop();
      if (fin) reset();
      else step();
      draw();
    };
    $('#vc-reset', root).onclick = () => {
      playback.stop();
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
        settings.n = v;
        go();
      },
      (v) => String(+v),
    );
    bind(
      root,
      'vc-sp',
      (v) => {
        settings.sp = v;
      },
      (v) => ['', 'slow', 'steady', 'fast'][+v],
    );
    segBind(root, 'vc-h', settings.h, (v) => {
      settings.h = v;
      place('good');
      go();
    });
    c.style.cursor = 'grab';
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', (e) => {
      if (!V) return;
      const [x, y] = canvasXY(c, V, e);
      drag = P.slice(0, settings.n).findIndex(
        (p) => Math.hypot((p[0] - x) * V.s, (p[1] - y) * V.s) < 16,
      );
      if (drag >= 0) {
        c.setPointerCapture(e.pointerId);
        playback.stop();
      }
    });
    c.addEventListener('pointermove', (e) => {
      if (drag < 0 || !V) return;
      const [x, y] = canvasXY(c, V, e);
      P[drag] = [clamp(x, -3.7, 3.7), settings.h === 'int' ? 0 : clamp(y, -2.7, 2.7)];
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
      stop: () => playback.stop(),
      auto: () => {
        reset();
        draw();
        frameCount = 0;
        playback.start();
      },
    };
  },
};
