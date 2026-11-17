/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  readThemeColors,
  bind,
  f2,
  prepareCanvas,
  markTP,
  mix,
  playBtn,
  rng,
  attachRunLimit,
  setReadout,
  testPoint,
} from '../../../js/core/shared.js';
import {
  UC,
  d2,
  highlightLessonStep,
  uDataBind,
  uFrame,
  drawSeriesChart,
  uView,
} from '../shared.js';

export default {
  init(root) {
    const c = $('#db-c', root),
      cr = $('#db-r', root),
      live = $('#db-live', root),
      settings = { eps: 0.3, m: 5, ds: 'moons', n: 220, seed: 2, bg: 14 };
    let P = [],
      lab = [],
      pointTypes = [],
      inQ = [],
      queue = [],
      order = [],
      scan = 0,
      nc = 0,
      activeClusterId = -1,
      currentPointIndex = -1,
      nb = [],
      done = false,
      vis = 0,
      msg = '',
      activeStepIndex = 0,
      wait = 0,
      frameCount = 0,
      playback,
      kd = [],
      run = 1;
    const nbr = (x, y) => {
      const e2 = settings.eps * settings.eps,
        o = [];
      for (let j = 0; j < P.length; j++) if ((P[j].x - x) ** 2 + (P[j].y - y) ** 2 <= e2) o.push(j);
      return o;
    };
    const kdist = () => {
      const k = Math.min(settings.m, P.length) - 1;
      kd = P.map((p) => {
        const d = P.map((q) => d2(p, q)).sort((a, b) => a - b);
        return Math.sqrt(d[Math.max(0, k)]);
      }).sort((a, b) => b - a);
    };
    const reset = () => {
      const n = P.length,
        r = rng(60 + run * 3);
      lab = new Array(n).fill(-2);
      pointTypes = new Array(n).fill(0);
      inQ = new Array(n).fill(0);
      queue = [];
      order = P.map((_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      scan = 0;
      nc = 0;
      activeClusterId = -1;
      currentPointIndex = -1;
      nb = [];
      done = false;
      vis = 0;
      activeStepIndex = -1;
      wait = 0;
      msg = 'No point has been visited yet. Press Run or One point.';
    };
    const step = () => {
      if (done) {
        run++;
        reset();
        return;
      }
      while (queue.length) {
        const q = queue.shift();
        if (lab[q] >= 0) continue;
        currentPointIndex = q;
        nb = nbr(P[q].x, P[q].y);
        activeStepIndex = 3;
        if (lab[q] === -1) {
          lab[q] = activeClusterId;
          pointTypes[q] = 2;
          msg =
            'This point was marked as noise earlier, but it lies within ε of a core point of cluster ' +
            (activeClusterId + 1) +
            '. It becomes a border point of that cluster.' +
            (queue.length
              ? ''
              : ' The queue is now empty: cluster ' + (activeClusterId + 1) + ' is complete.');
          if (!queue.length) activeStepIndex = 4;
          return;
        }
        lab[q] = activeClusterId;
        vis++;
        if (nb.length >= settings.m) {
          pointTypes[q] = 1;
          let add = 0;
          for (const j of nb)
            if ((lab[j] === -2 || lab[j] === -1) && !inQ[j]) {
              inQ[j] = 1;
              queue.push(j);
              add++;
            }
          msg =
            'Growing cluster ' +
            (activeClusterId + 1) +
            ': this point has ' +
            nb.length +
            ' points within ε, at least minPts = ' +
            settings.m +
            ', so it is a core point. ' +
            add +
            ' new neighbour' +
            (add === 1 ? ' joins' : 's join') +
            ' the queue, which now holds ' +
            queue.length +
            '.';
        } else {
          pointTypes[q] = 2;
          msg =
            'Growing cluster ' +
            (activeClusterId + 1) +
            ': this point has only ' +
            nb.length +
            ' points within ε, fewer than minPts = ' +
            settings.m +
            '. It joins the cluster as a border point, and the cluster does not spread through it. ' +
            queue.length +
            ' left in the queue.';
        }
        if (!queue.length) {
          msg += ' The queue is empty: cluster ' + (activeClusterId + 1) + ' is complete.';
          activeStepIndex = 4;
        }
        return;
      }
      while (scan < order.length && lab[order[scan]] !== -2) scan++;
      if (scan >= order.length) {
        done = true;
        currentPointIndex = -1;
        nb = [];
        activeStepIndex = 4;
        const no = lab.filter((v) => v === -1).length;
        msg =
          'Every point has been visited. DBSCAN found ' +
          nc +
          ' cluster' +
          (nc === 1 ? '' : 's') +
          ' and left ' +
          no +
          ' point' +
          (no === 1 ? '' : 's') +
          ' as noise, without being told how many clusters to look for.';
        return;
      }
      const p = order[scan];
      currentPointIndex = p;
      nb = nbr(P[p].x, P[p].y);
      vis++;
      if (nb.length < settings.m) {
        lab[p] = -1;
        pointTypes[p] = 3;
        activeStepIndex = 1;
        msg =
          'New point: only ' +
          nb.length +
          ' point' +
          (nb.length === 1 ? '' : 's') +
          ' within ε (itself included), fewer than minPts = ' +
          settings.m +
          '. Marked as noise for now. A cluster that grows this far may still claim it as a border point.';
      } else {
        activeClusterId = nc++;
        lab[p] = activeClusterId;
        pointTypes[p] = 1;
        for (const j of nb)
          if ((lab[j] === -2 || lab[j] === -1) && !inQ[j]) {
            inQ[j] = 1;
            queue.push(j);
          }
        activeStepIndex = 2;
        msg =
          'New point: ' +
          nb.length +
          ' points within ε, at least minPts = ' +
          settings.m +
          ', so it is a core point. Cluster ' +
          (activeClusterId + 1) +
          ' starts here, and its ' +
          queue.length +
          ' unclaimed neighbours go into the queue.';
      }
    };
    const fin = () => {
      let k = 0;
      while (!done && k++ < 1e5) step();
      currentPointIndex = -1;
      nb = [];
    };
    const tp = testPoint(
      root,
      'db',
      ['Your x', 'Your y'],
      [0, 0],
      [
        [-3.8, 3.8],
        [-2.85, 2.85],
      ],
    );
    const draw = () => {
      const colors = readThemeColors(),
        col = UC(colors);
      {
        const V = uView(c),
          { g, w, h, X, Y, s } = V;
        uFrame(V, colors, 'x', 'y');
        if (currentPointIndex >= 0) {
          g.beginPath();
          g.arc(X(P[currentPointIndex].x), Y(P[currentPointIndex].y), settings.eps * s, 0, 7);
          g.fillStyle = mix(colors.ink, colors.ink, 0, 0.07);
          g.fill();
          g.strokeStyle = colors.ink;
          g.lineWidth = 1.6;
          g.stroke();
        }
        P.forEach((p, i) => {
          const x = X(p.x),
            y = Y(p.y),
            k = lab[i];
          if (pointTypes[i] === 3 && k === -1) {
            g.strokeStyle = colors.muted;
            g.lineWidth = 1.6;
            g.beginPath();
            g.moveTo(x - 3.5, y - 3.5);
            g.lineTo(x + 3.5, y + 3.5);
            g.moveTo(x + 3.5, y - 3.5);
            g.lineTo(x - 3.5, y + 3.5);
            g.stroke();
          } else if (k >= 0 && pointTypes[i] === 1) {
            g.fillStyle = col[k % 8];
            g.beginPath();
            g.arc(x, y, 4.2, 0, 7);
            g.fill();
          } else if (k >= 0) {
            g.strokeStyle = col[k % 8];
            g.lineWidth = 2;
            g.fillStyle = colors.bg;
            g.beginPath();
            g.arc(x, y, 3.6, 0, 7);
            g.fill();
            g.stroke();
          } else {
            g.fillStyle = mix(colors.ink, colors.ink, 0, 0.32);
            g.beginPath();
            g.arc(x, y, 2.8, 0, 7);
            g.fill();
          }
          if (inQ[i] && lab[i] < 0 && !done) {
            g.strokeStyle = colors.gold;
            g.lineWidth = 1.8;
            g.beginPath();
            g.arc(x, y, 6.5, 0, 7);
            g.stroke();
          }
        });
        if (currentPointIndex >= 0) {
          g.strokeStyle = colors.ink;
          g.lineWidth = 2.4;
          g.beginPath();
          g.arc(X(P[currentPointIndex].x), Y(P[currentPointIndex].y), 7.5, 0, 7);
          g.stroke();
        }
        {
          const x = X(tp.v[0]),
            y = Y(tp.v[1]);
          g.setLineDash([4, 4]);
          g.strokeStyle = colors.muted;
          g.lineWidth = 1.2;
          g.beginPath();
          g.arc(x, y, settings.eps * s, 0, 7);
          g.stroke();
          g.setLineDash([]);
          markTP(g, x, y, colors);
        }
        g.font = MONO;
        g.fillStyle = colors.muted;
        g.textAlign = 'left';
        g.fillText('gold ring = waiting in the queue', 8, h - 8);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          ym = Math.max(1.3, kd.length ? kd[0] : 1) * 1.05,
          L = drawSeriesChart(ctx, colors, 0, 0, w, h, kd, {
            title:
              'distance to the ' +
              settings.m +
              (settings.m === 2 ? 'nd' : settings.m === 3 ? 'rd' : 'th') +
              ' nearest point (itself counted), largest first',
            ym,
            col: colors.accent,
          }),
          y = L.Y(settings.eps);
        ctx.setLineDash([7, 5]);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(L.m.l, y);
        ctx.lineTo(w - L.m.r, y);
        ctx.stroke();
        ctx.setLineDash([]);
        const above = kd.filter((v) => v > settings.eps).length;
        ctx.font = MONO;
        ctx.fillStyle = colors.cut;
        ctx.textAlign = 'right';
        ctx.fillText(
          'ε = ' +
            f2(settings.eps) +
            ': ' +
            above +
            ' of ' +
            kd.length +
            ' points lie above and cannot be core',
          w - L.m.r,
          y - 6,
        );
        ctx.fillStyle = colors.muted;
        ctx.fillText('points, most isolated first →', w - L.m.r, h - 5);
      }
      live.textContent = msg;
      highlightLessonStep(root, 'db', activeStepIndex);
      {
        const q = nbr(tp.v[0], tp.v[1]),
          cores = q.filter((j) => pointTypes[j] === 1),
          cnt = q.length + 1;
        let s =
          'Your point would have ' +
          cnt +
          ' point' +
          (cnt === 1 ? '' : 's') +
          ' within ε, counting itself. ';
        if (!done) s += 'Run to the end to see which cluster claims it.';
        else if (cnt >= settings.m && cores.length)
          s +=
            'That reaches minPts, so it would be a core point of cluster ' +
            (lab[cores[0]] + 1) +
            '.';
        else if (cnt >= settings.m)
          s +=
            'That reaches minPts, so it would be a core point and start a cluster of its own with its neighbours.';
        else if (cores.length)
          s +=
            'That is below minPts, but a core point of cluster ' +
            (lab[cores[0]] + 1) +
            ' is within reach, so it would be a border point of that cluster.';
        else s += 'That is below minPts and no core point is within reach: it would be noise.';
        tp.out.textContent = s;
      }
      setReadout(root, 'v', vis + ' of ' + P.length);
      setReadout(root, 'cl', String(nc));
      setReadout(root, 'co', String(pointTypes.filter((v) => v === 1).length));
      setReadout(root, 'bo', String(pointTypes.filter((v) => v === 2).length));
      setReadout(root, 'no', String(lab.filter((v) => v === -1).length));
    };
    tp.on = () => draw();
    const redo = () => {
      playback.stop();
      kdist();
      reset();
      fin();
      draw();
    };
    const gen = uDataBind(root, 'db', settings, () => {
      P = gen();
      redo();
    });
    P = gen();
    kdist();
    reset();
    fin();
    const anim = Anim(() => {
      frameCount++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          run++;
          reset();
          draw();
        }
        return;
      }
      if (frameCount % (vis < 12 ? 26 : 4)) return;
      step();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#db-play', root), anim);
    const checkRunStop = attachRunLimit(playback, root, 'db', {
      done: () => done,
      restart: () => {
        run++;
        reset();
      },
    });
    $('#db-step', root).onclick = () => {
      playback.stop();
      step();
      draw();
    };
    $('#db-reset', root).onclick = () => {
      playback.stop();
      run++;
      reset();
      draw();
    };
    $('#db-fin', root).onclick = () => {
      playback.stop();
      if (done) reset();
      fin();
      draw();
    };
    bind(root, 'db-eps', (v) => {
      settings.eps = v;
      redo();
    });
    bind(
      root,
      'db-m',
      (v) => {
        settings.m = v;
        redo();
      },
      (v) => String(v),
    );
    bind(
      root,
      'db-bg',
      (v) => {
        settings.bg = v;
        P = gen();
        redo();
      },
      (v) => String(v),
    );
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        reset();
        draw();
        playback.start();
      },
    };
  },
};
