/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
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
import { UC, d2, uCur, uDataBind, uFrame, uLine, uView } from '../shared.js';

export default {
  init(root) {
    const c = $('#db-c', root),
      cr = $('#db-r', root),
      live = $('#db-live', root),
      S = { eps: 0.3, m: 5, ds: 'moons', n: 220, seed: 2, bg: 14 };
    let P = [],
      lab = [],
      typ = [],
      inQ = [],
      queue = [],
      order = [],
      scan = 0,
      nc = 0,
      curC = -1,
      cur = -1,
      nb = [],
      done = false,
      vis = 0,
      msg = '',
      cu = 0,
      wait = 0,
      fr = 0,
      pb,
      kd = [],
      run = 1;
    const nbr = (x, y) => {
      const e2 = S.eps * S.eps,
        o = [];
      for (let j = 0; j < P.length; j++) if ((P[j].x - x) ** 2 + (P[j].y - y) ** 2 <= e2) o.push(j);
      return o;
    };
    const kdist = () => {
      const k = Math.min(S.m, P.length) - 1;
      kd = P.map((p) => {
        const d = P.map((q) => d2(p, q)).sort((a, b) => a - b);
        return Math.sqrt(d[Math.max(0, k)]);
      }).sort((a, b) => b - a);
    };
    const reset = () => {
      const n = P.length,
        r = rng(60 + run * 3);
      lab = new Array(n).fill(-2);
      typ = new Array(n).fill(0);
      inQ = new Array(n).fill(0);
      queue = [];
      order = P.map((_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      scan = 0;
      nc = 0;
      curC = -1;
      cur = -1;
      nb = [];
      done = false;
      vis = 0;
      cu = -1;
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
        cur = q;
        nb = nbr(P[q].x, P[q].y);
        cu = 3;
        if (lab[q] === -1) {
          lab[q] = curC;
          typ[q] = 2;
          msg =
            'This point was marked as noise earlier, but it lies within ε of a core point of cluster ' +
            (curC + 1) +
            '. It becomes a border point of that cluster.' +
            (queue.length
              ? ''
              : ' The queue is now empty: cluster ' + (curC + 1) + ' is complete.');
          if (!queue.length) cu = 4;
          return;
        }
        lab[q] = curC;
        vis++;
        if (nb.length >= S.m) {
          typ[q] = 1;
          let add = 0;
          for (const j of nb)
            if ((lab[j] === -2 || lab[j] === -1) && !inQ[j]) {
              inQ[j] = 1;
              queue.push(j);
              add++;
            }
          msg =
            'Growing cluster ' +
            (curC + 1) +
            ': this point has ' +
            nb.length +
            ' points within ε, at least minPts = ' +
            S.m +
            ', so it is a core point. ' +
            add +
            ' new neighbour' +
            (add === 1 ? ' joins' : 's join') +
            ' the queue, which now holds ' +
            queue.length +
            '.';
        } else {
          typ[q] = 2;
          msg =
            'Growing cluster ' +
            (curC + 1) +
            ': this point has only ' +
            nb.length +
            ' points within ε, fewer than minPts = ' +
            S.m +
            '. It joins the cluster as a border point, and the cluster does not spread through it. ' +
            queue.length +
            ' left in the queue.';
        }
        if (!queue.length) {
          msg += ' The queue is empty: cluster ' + (curC + 1) + ' is complete.';
          cu = 4;
        }
        return;
      }
      while (scan < order.length && lab[order[scan]] !== -2) scan++;
      if (scan >= order.length) {
        done = true;
        cur = -1;
        nb = [];
        cu = 4;
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
      cur = p;
      nb = nbr(P[p].x, P[p].y);
      vis++;
      if (nb.length < S.m) {
        lab[p] = -1;
        typ[p] = 3;
        cu = 1;
        msg =
          'New point: only ' +
          nb.length +
          ' point' +
          (nb.length === 1 ? '' : 's') +
          ' within ε (itself included), fewer than minPts = ' +
          S.m +
          '. Marked as noise for now. A cluster that grows this far may still claim it as a border point.';
      } else {
        curC = nc++;
        lab[p] = curC;
        typ[p] = 1;
        for (const j of nb)
          if ((lab[j] === -2 || lab[j] === -1) && !inQ[j]) {
            inQ[j] = 1;
            queue.push(j);
          }
        cu = 2;
        msg =
          'New point: ' +
          nb.length +
          ' points within ε, at least minPts = ' +
          S.m +
          ', so it is a core point. Cluster ' +
          (curC + 1) +
          ' starts here, and its ' +
          queue.length +
          ' unclaimed neighbours go into the queue.';
      }
    };
    const fin = () => {
      let k = 0;
      while (!done && k++ < 1e5) step();
      cur = -1;
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
      const t = T(),
        col = UC(t);
      {
        const V = uView(c),
          { g, w, h, X, Y, s } = V;
        uFrame(V, t, 'x', 'y');
        if (cur >= 0) {
          g.beginPath();
          g.arc(X(P[cur].x), Y(P[cur].y), S.eps * s, 0, 7);
          g.fillStyle = mix(t.ink, t.ink, 0, 0.07);
          g.fill();
          g.strokeStyle = t.ink;
          g.lineWidth = 1.6;
          g.stroke();
        }
        P.forEach((p, i) => {
          const x = X(p.x),
            y = Y(p.y),
            k = lab[i];
          if (typ[i] === 3 && k === -1) {
            g.strokeStyle = t.muted;
            g.lineWidth = 1.6;
            g.beginPath();
            g.moveTo(x - 3.5, y - 3.5);
            g.lineTo(x + 3.5, y + 3.5);
            g.moveTo(x + 3.5, y - 3.5);
            g.lineTo(x - 3.5, y + 3.5);
            g.stroke();
          } else if (k >= 0 && typ[i] === 1) {
            g.fillStyle = col[k % 8];
            g.beginPath();
            g.arc(x, y, 4.2, 0, 7);
            g.fill();
          } else if (k >= 0) {
            g.strokeStyle = col[k % 8];
            g.lineWidth = 2;
            g.fillStyle = t.bg;
            g.beginPath();
            g.arc(x, y, 3.6, 0, 7);
            g.fill();
            g.stroke();
          } else {
            g.fillStyle = mix(t.ink, t.ink, 0, 0.32);
            g.beginPath();
            g.arc(x, y, 2.8, 0, 7);
            g.fill();
          }
          if (inQ[i] && lab[i] < 0 && !done) {
            g.strokeStyle = t.gold;
            g.lineWidth = 1.8;
            g.beginPath();
            g.arc(x, y, 6.5, 0, 7);
            g.stroke();
          }
        });
        if (cur >= 0) {
          g.strokeStyle = t.ink;
          g.lineWidth = 2.4;
          g.beginPath();
          g.arc(X(P[cur].x), Y(P[cur].y), 7.5, 0, 7);
          g.stroke();
        }
        {
          const x = X(tp.v[0]),
            y = Y(tp.v[1]);
          g.setLineDash([4, 4]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1.2;
          g.beginPath();
          g.arc(x, y, S.eps * s, 0, 7);
          g.stroke();
          g.setLineDash([]);
          markTP(g, x, y, t);
        }
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('gold ring = waiting in the queue', 8, h - 8);
      }
      {
        const { g, w, h } = fit(cr),
          ym = Math.max(1.3, kd.length ? kd[0] : 1) * 1.05,
          L = uLine(g, t, 0, 0, w, h, kd, {
            title:
              'distance to the ' +
              S.m +
              (S.m === 2 ? 'nd' : S.m === 3 ? 'rd' : 'th') +
              ' nearest point (itself counted), largest first',
            ym,
            col: t.accent,
          }),
          y = L.Y(S.eps);
        g.setLineDash([7, 5]);
        g.strokeStyle = t.gold;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(L.m.l, y);
        g.lineTo(w - L.m.r, y);
        g.stroke();
        g.setLineDash([]);
        const above = kd.filter((v) => v > S.eps).length;
        g.font = MONO;
        g.fillStyle = t.cut;
        g.textAlign = 'right';
        g.fillText(
          'ε = ' +
            f2(S.eps) +
            ': ' +
            above +
            ' of ' +
            kd.length +
            ' points lie above and cannot be core',
          w - L.m.r,
          y - 6,
        );
        g.fillStyle = t.muted;
        g.fillText('points, most isolated first →', w - L.m.r, h - 5);
      }
      live.textContent = msg;
      uCur(root, 'db', cu);
      {
        const q = nbr(tp.v[0], tp.v[1]),
          cores = q.filter((j) => typ[j] === 1),
          cnt = q.length + 1;
        let s =
          'Your point would have ' +
          cnt +
          ' point' +
          (cnt === 1 ? '' : 's') +
          ' within ε, counting itself. ';
        if (!done) s += 'Run to the end to see which cluster claims it.';
        else if (cnt >= S.m && cores.length)
          s +=
            'That reaches minPts, so it would be a core point of cluster ' +
            (lab[cores[0]] + 1) +
            '.';
        else if (cnt >= S.m)
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
      setR(root, 'v', vis + ' of ' + P.length);
      setR(root, 'cl', String(nc));
      setR(root, 'co', String(typ.filter((v) => v === 1).length));
      setR(root, 'bo', String(typ.filter((v) => v === 2).length));
      setR(root, 'no', String(lab.filter((v) => v === -1).length));
    };
    tp.on = () => draw();
    const redo = () => {
      pb.stop();
      kdist();
      reset();
      fin();
      draw();
    };
    const gen = uDataBind(root, 'db', S, () => {
      P = gen();
      redo();
    });
    P = gen();
    kdist();
    reset();
    fin();
    const anim = Anim(() => {
      fr++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          run++;
          reset();
          draw();
        }
        return;
      }
      if (fr % (vis < 12 ? 26 : 4)) return;
      step();
      draw();
      chk();
    });
    pb = playBtn($('#db-play', root), anim);
    const chk = runCtl(pb, root, 'db', {
      done: () => done,
      restart: () => {
        run++;
        reset();
      },
    });
    $('#db-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#db-reset', root).onclick = () => {
      pb.stop();
      run++;
      reset();
      draw();
    };
    $('#db-fin', root).onclick = () => {
      pb.stop();
      if (done) reset();
      fin();
      draw();
    };
    bind(root, 'db-eps', (v) => {
      S.eps = v;
      redo();
    });
    bind(
      root,
      'db-m',
      (v) => {
        S.m = v;
        redo();
      },
      (v) => String(v),
    );
    bind(
      root,
      'db-bg',
      (v) => {
        S.bg = v;
        P = gen();
        redo();
      },
      (v) => String(v),
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
