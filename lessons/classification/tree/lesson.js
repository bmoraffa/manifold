/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $, MONO, clamp, prepareCanvas, mix } from '../../../js/core/shared.js';
import { clfLesson } from '../shared.js';
import { pct } from '../../ensembles/shared.js';

export default clfLesson({
  id: 'tree',
  fromZero: true,
  hard: true,

  ds: ['moons', 'blobs', 'overlap', 'circles', 'xor'],
  stepBtn: true,
  S: { depth: 6, minleaf: 1 },
  n: 160,
  nc: 60,
  setup(api) {
    const S = api.S;
    let root = null,
      totalSplits = 0,
      revealedSplits = 0,
      wait = 0,
      frameCount = 0,
      bd = 1,
      fresh = true,
      phase = 'grow',
      pmsg = '',
      ph = [];
    const hasActiveSplit = (nd) => !!(nd.sp && nd.sp.ord < revealedSplits && !nd.cut);
    const gini = (n, n1) => (n ? 2 * (n1 / n) * (1 - n1 / n) : 0);
    const build = (T, D, ML) => {
      const mk = (P, dp, box) => ({
          P,
          n: P.length,
          n1: P.filter((p) => p.c).length,
          dp,
          box,
          sp: null,
        }),
        rt = mk(T, 0, [-3, 3, -3, 3]),
        q = [rt];
      let ord = 0;
      while (q.length) {
        const nd = q.shift();
        if (nd.dp >= D || nd.n1 === 0 || nd.n1 === nd.n || nd.n < 2 * ML) continue;
        let best = null;
        const I = gini(nd.n, nd.n1);
        for (const f of ['x', 'y']) {
          const s = nd.P.slice().sort((a, b) => a[f] - b[f]);
          let l1 = 0;
          for (let i = 1; i < s.length; i++) {
            l1 += s[i - 1].c;
            if (s[i][f] === s[i - 1][f] || i < ML || s.length - i < ML) continue;
            const r1 = nd.n1 - l1,
              gn = I - (i / nd.n) * gini(i, l1) - ((nd.n - i) / nd.n) * gini(nd.n - i, r1);
            if (!best || gn > best.gn) best = { gn, f, thr: (s[i][f] + s[i - 1][f]) / 2, i, s };
          }
        }
        if (!best || (best.gn <= 1e-9 && nd.dp > 0)) continue;
        const b = nd.box,
          L = mk(
            best.s.slice(0, best.i),
            nd.dp + 1,
            best.f === 'x' ? [b[0], best.thr, b[2], b[3]] : [b[0], b[1], b[2], best.thr],
          ),
          Rr = mk(
            best.s.slice(best.i),
            nd.dp + 1,
            best.f === 'x' ? [best.thr, b[1], b[2], b[3]] : [b[0], b[1], best.thr, b[3]],
          );
        nd.sp = { f: best.f, thr: best.thr, gn: best.gn, ord: ord++, L, R: Rr };
        q.push(L, Rr);
      }
      return { rt, n: ord };
    };
    const leaf = (rt, x, y, lim) => {
      let nd = rt;
      while (nd.sp && nd.sp.ord < lim && !nd.cut)
        nd = (nd.sp.f === 'x' ? x : y) <= nd.sp.thr ? nd.sp.L : nd.sp.R;
      return nd;
    };
    const accOf = (P) =>
      P.length
        ? P.filter((p) => {
            const l = leaf(root, p.x, p.y, revealedSplits);
            return (l.n1 / l.n >= 0.5 ? 1 : 0) === p.c;
          }).length / P.length
        : NaN;
    const prune = () => {
      if (!root) return false;
      revealedSplits = totalSplits;
      let best = null;
      const info = (nd) => {
        if (!hasActiveSplit(nd)) return { L: 1, R: Math.min(nd.n1, nd.n - nd.n1) };
        const a = info(nd.sp.L),
          b = info(nd.sp.R),
          L = a.L + b.L,
          R = a.R + b.R,
          gq = (Math.min(nd.n1, nd.n - nd.n1) - R) / (L - 1);
        if (!best || gq < best.g - 1e-12 || (Math.abs(gq - best.g) < 1e-12 && L < best.L))
          best = { nd, g: gq, L, add: Math.min(nd.n1, nd.n - nd.n1) - R };
        return { L, R };
      };
      const tot = info(root);
      if (!best) return false;
      if (!ph.length) ph.push({ L: tot.L, te: accOf(api.test) });
      best.nd.cut = true;
      phase = 'prune';
      const left = tot.L - best.L + 1,
        te = accOf(api.test);
      ph.push({ L: left, te });
      pmsg =
        'Pruning step ' +
        (ph.length - 1) +
        ': removed the question “' +
        best.nd.sp.f +
        ' ≤ ' +
        best.nd.sp.thr.toFixed(2) +
        '” and ' +
        (best.L > 2 ? 'the ' + (best.L - 2) + ' below it' : 'nothing below it') +
        '. That merges ' +
        best.L +
        ' regions into one and adds ' +
        best.add +
        (best.add === 1 ? ' training mistake' : ' training mistakes') +
        ', the cheapest cut available (' +
        best.g.toFixed(2) +
        ' mistakes per region saved). The tree now has ' +
        left +
        (left === 1 ? ' region' : ' regions') +
        '. Train accuracy ' +
        pct(accOf(api.train)) +
        ', test accuracy ' +
        pct(te) +
        '.';
      return true;
    };
    const dsl = api.bind(
      'depth',
      'depth',
      (v) => String(v),
      () => {
        api.refit();
      },
    );
    api.bind('minleaf', 'minleaf', (v) => String(v));
    $('#tree-prune', api.root).onclick = () => {
      api.pause();
      if (!prune())
        api.note(
          'Nothing is left to prune: the tree is a single region. Press Start over or Show the Full Tree.',
        );
      else api.note('');
      api.redraw();
    };
    $('#tree-full', api.root).onclick = () => {
      api.pause();
      api.note('');
      const un = (nd) => {
        nd.cut = false;
        if (nd.sp) {
          un(nd.sp.L);
          un(nd.sp.R);
        }
      };
      if (root) un(root);
      revealedSplits = totalSplits;
      phase = 'grow';
      pmsg = '';
      ph = [];
      api.redraw();
    };
    return {
      fit() {
        const T = api.train,
          te = api.test;
        if (!T.length) return;
        const b = build(T, S.depth, S.minleaf);
        root = b.rt;
        totalSplits = b.n;
        phase = 'grow';
        pmsg = '';
        ph = [];
        if (fresh) {
          revealedSplits = totalSplits;
          fresh = false;
        } else revealedSplits = Math.min(revealedSplits, totalSplits);
        let be = -1;
        bd = 1;
        for (let d = 1; d <= 7; d++) {
          const r = build(T, d, S.minleaf),
            ok = te.filter((p) => {
              const l = leaf(r.rt, p.x, p.y, 1e9);
              return (l.n1 / l.n >= 0.5 ? 1 : 0) === p.c;
            }).length;
          if (ok > be) {
            be = ok;
            bd = d;
          }
        }
      },
      reset() {
        revealedSplits = 0;
        wait = 0;
        phase = 'grow';
        pmsg = '';
        ph = [];
        const un = (nd) => {
          nd.cut = false;
          if (nd.sp) {
            un(nd.sp.L);
            un(nd.sp.R);
          }
        };
        if (root) un(root);
      },
      p: (x, y) => {
        if (!root) return 0.5;
        const l = leaf(root, x, y, revealedSplits);
        return l.n ? clamp(l.n1 / l.n, 0.02, 0.98) * 0.999 + (l.n1 * 2 === l.n ? 1e-4 : 0) : 0.5;
      },
      stepOnce() {
        if (phase === 'prune') {
          if (!prune()) this.reset();
        } else if (revealedSplits < totalSplits) revealedSplits++;
        else prune();
      },
      done: () => wait > 0,
      tick() {
        frameCount++;
        if (wait > 0) return false;
        if (frameCount % 38) return false;
        if (phase === 'grow' && revealedSplits < totalSplits) revealedSplits++;
        else if (!prune()) {
          wait = 260;
        }
        return true;
      },
      overlay(g, X, Y, t) {
        const walk = (nd) => {
          if (!hasActiveSplit(nd)) return;
          const b = nd.box,
            s = nd.sp;
          g.strokeStyle = t.ink;
          g.lineWidth = s.ord === revealedSplits - 1 ? 3.4 : 1.4;
          g.beginPath();
          if (s.f === 'x') {
            g.moveTo(X(s.thr), Y(b[2]));
            g.lineTo(X(s.thr), Y(b[3]));
          } else {
            g.moveTo(X(b[0]), Y(s.thr));
            g.lineTo(X(b[1]), Y(s.thr));
          }
          g.stroke();
          walk(s.L);
          walk(s.R);
        };
        if (root) walk(root);
      },
      right(t) {
        const { g: ctx, w, h } = prepareCanvas(api.cr);
        if (!root) return;
        const leaves = [],
          lay = (nd, dp) => {
            nd._d = dp;
            if (hasActiveSplit(nd)) {
              lay(nd.sp.L, dp + 1);
              lay(nd.sp.R, dp + 1);
              nd._x = (nd.sp.L._x + nd.sp.R._x) / 2;
            } else {
              nd._x = leaves.length;
              leaves.push(nd);
            }
          };
        lay(root, 0);
        let md = 0;
        const all = [],
          col = (nd) => {
            all.push(nd);
            md = Math.max(md, nd._d);
            if (hasActiveSplit(nd)) {
              col(nd.sp.L);
              col(nd.sp.R);
            }
          };
        col(root);
        const X = (v) => 56 + (leaves.length > 1 ? v / (leaves.length - 1) : 0.5) * (w - 112),
          Y = (d) => 26 + (md ? d / md : 0) * (h - 64),
          small = leaves.length > 14;
        ctx.strokeStyle = t.muted;
        ctx.lineWidth = 1.2;
        for (const nd of all)
          if (hasActiveSplit(nd)) {
            [nd.sp.L, nd.sp.R].forEach((ch) => {
              ctx.beginPath();
              ctx.moveTo(X(nd._x), Y(nd._d));
              ctx.lineTo(X(ch._x), Y(ch._d));
              ctx.stroke();
            });
          }
        ctx.textAlign = 'center';
        for (const nd of all) {
          const x = X(nd._x),
            y = Y(nd._d),
            inner = hasActiveSplit(nd);
          if (inner) {
            const tx = nd.sp.f + ' ≤ ' + nd.sp.thr.toFixed(1);
            ctx.font = small ? '10px "JetBrains Mono",monospace' : MONO;
            const tw = ctx.measureText(tx).width + 10;
            ctx.fillStyle = t.bg;
            ctx.strokeStyle = nd.sp.ord === revealedSplits - 1 ? t.ink : t.muted;
            ctx.lineWidth = nd.sp.ord === revealedSplits - 1 ? 2.4 : 1;
            ctx.beginPath();
            ctx.rect(x - tw / 2, y - 10, tw, 20);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = t.ink;
            ctx.fillText(tx, x, y + 4);
          } else {
            const pb2 = nd.n ? nd.n1 / nd.n : 0.5;
            ctx.fillStyle = mix(
              pb2 >= 0.5 ? t.gold : t.accent,
              t.bg,
              1 - Math.abs(pb2 - 0.5) * 2,
              1,
            );
            ctx.strokeStyle = pb2 >= 0.5 ? t.gold : t.accent;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, small ? 6 : 10, 0, 7);
            ctx.fill();
            ctx.stroke();
            if (!small) {
              ctx.font = MONO;
              ctx.fillStyle = t.muted;
              ctx.fillText(nd.n - nd.n1 + ' A · ' + nd.n1 + ' B', x, y + 25);
            }
          }
        }
        ctx.font = MONO;
        ctx.fillStyle = t.muted;
        ctx.textAlign = 'left';
        ctx.fillText('yes ↙   no ↘', 8, 12);
      },
      status() {
        if (!root) return '';
        if (phase === 'prune' && pmsg) return pmsg;
        if (!revealedSplits)
          return (
            'All ' +
            root.n +
            ' training points are in one region: ' +
            (root.n - root.n1) +
            ' of class A and ' +
            root.n1 +
            ' of class B. Press Grow the tree or One step.'
          );
        let f = null;
        const find = (nd) => {
          if (!nd.sp) return;
          if (nd.sp.ord === revealedSplits - 1) f = nd;
          find(nd.sp.L);
          find(nd.sp.R);
        };
        find(root);
        if (!f) return '';
        const sp = f.sp;
        return (
          'Question ' +
          revealedSplits +
          ' of ' +
          totalSplits +
          ': is ' +
          sp.f +
          ' ≤ ' +
          sp.thr.toFixed(2) +
          '? Of every possible cut in this region of ' +
          f.n +
          ' points, this one lowers the impurity most (by ' +
          sp.gn.toFixed(3) +
          '). Yes: ' +
          (sp.L.n - sp.L.n1) +
          ' A and ' +
          sp.L.n1 +
          ' B. No: ' +
          (sp.R.n - sp.R.n1) +
          ' A and ' +
          sp.R.n1 +
          ' B.' +
          (revealedSplits >= totalSplits ? ' No allowed cut is left: the tree is finished.' : '')
        );
      },
      cur() {
        return phase === 'prune' ? 5 : !revealedSplits ? 0 : revealedSplits >= totalSplits ? 4 : 2;
      },
      read(set) {
        let lv = 0;
        const cnt = (nd) => {
          if (hasActiveSplit(nd)) {
            cnt(nd.sp.L);
            cnt(nd.sp.R);
          } else lv++;
        };
        if (root) cnt(root);
        set('sp', revealedSplits + ' of ' + totalSplits);
        set('lv', String(lv));
        set('bd', String(bd));
        let bq = null;
        for (const q of ph) if (!bq || q.te >= bq.te) bq = q;
        set('bp', bq ? bq.L + ' (test ' + pct(bq.te) + ')' : 'prune to find out');
      },
    };
  },
});
