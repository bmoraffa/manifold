/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  clipPlot,
  escapeHTML,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  plot,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { parseXgbCSV, xgbBuildTree } from '../shared.js';
import { sig } from '../../classification/shared.js';

export default {
  init(root) {
    const cL = $('#xgb-l', root),
      cT = $('#xgb-t', root),
      cI = $('#xgb-i', root),
      cP = $('#xgb-p', root),
      live = $('#xgb-live', root),
      st = $('#xgb-status', root),
      info = $('#xgb-data', root),
      tsel = $('#xgb-target', root),
      tpBox = $('#xgb-tp', root),
      tpo = $('#xgb-tpo', root);
    const S = { B: 60, eta: 0.3, depth: 3, gam: 0.05, lam: 1, sub: 1 };
    let D = null,
      raw = null,
      F = null,
      g = null,
      h = null,
      trees = [],
      base = 0,
      eTr = [],
      eTe = [],
      imp = [],
      pruned = 0,
      last = null,
      lastPr = 0,
      wait = 0,
      fr = 0,
      pb,
      msg = '',
      tpv = [];
    const presets = {
      house() {
        const r = rng(17),
          rn = randnFrom(r),
          names = [
            'floor area (m²)',
            'bedrooms',
            'age (years)',
            'km to centre',
            'street number',
            'price (thousands)',
          ],
          rows = [];
        for (let i = 0; i < 420; i++) {
          const a = 50 + 200 * r(),
            b = 1 + Math.floor(r() * 5),
            ag = 60 * r(),
            d = 30 * r(),
            sn = 1 + Math.floor(r() * 200);
          rows.push([
            a,
            b,
            ag,
            d,
            sn,
            60 + 1.7 * a + 14 * b - 0.9 * ag - 3.2 * d + (a > 150 && d < 8 ? 70 : 0) + 24 * rn(),
          ]);
        }
        return { names, rows, t: 5 };
      },
      loan() {
        const r = rng(23),
          rn = randnFrom(r),
          names = [
            'income (thousands)',
            'debt ÷ income',
            'late payments',
            'age',
            'years as customer',
            'defaulted',
          ],
          rows = [];
        for (let i = 0; i < 500; i++) {
          const inc = 20 + 100 * r(),
            dr = r(),
            lp = Math.floor(6 * r() * r()),
            ag = 20 + 50 * r(),
            yc = 15 * r(),
            z =
              -3.2 +
              4.2 * dr +
              0.75 * lp -
              0.012 * inc +
              (dr > 0.6 && lp >= 2 ? 1.4 : 0) +
              0.5 * rn();
          rows.push([inc, dr, lp, ag, yc, r() < sig(z) ? 1 : 0]);
        }
        return { names, rows, t: 5 };
      },
    };
    const setData = (names, rows, t) => {
      const fi = names
          .map((_, j) => j)
          .filter((j) => j !== t)
          .slice(0, 12),
        X = rows.map((rw) => fi.map((j) => rw[j])),
        yr = rows.map((rw) => rw[t]),
        u = [...new Set(yr)],
        clf = u.length === 2,
        lo = Math.min(...u),
        n = X.length;
      let mu = 0,
        sd = 1;
      if (!clf) {
        mu = yr.reduce((a, b) => a + b, 0) / n;
        sd = Math.sqrt(yr.reduce((a, b) => a + (b - mu) ** 2, 0) / n) || 1;
      }
      const y = clf ? yr.map((v) => (v === lo ? 0 : 1)) : yr.map((v) => (v - mu) / sd),
        r = rng(8),
        ix = X.map((_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [ix[i], ix[j]] = [ix[j], ix[i]];
      }
      const k = Math.round(n * 0.7),
        tr = ix.slice(0, k),
        te = ix.slice(k),
        order = fi.map((_, f) => tr.slice().sort((a, b) => X[a][f] - X[b][f]));
      D = {
        names: fi.map((j) => names[j]),
        tname: names[t],
        X,
        y,
        yr,
        clf,
        mu,
        sd,
        tr,
        te,
        order,
        cls: clf ? [lo, Math.max(...u)] : null,
      };
      raw = { names, rows, t };
      tsel.innerHTML = names
        .map(
          (nm, j) =>
            `<option value="${j}"${j === t ? ' selected' : ''}>${escapeHTML(nm.slice(0, 28))}</option>`,
        )
        .join('');
      info.textContent =
        n +
        ' rows, ' +
        D.names.length +
        ' features. Predicting “' +
        D.tname +
        '”' +
        (clf
          ? ', a yes/no outcome (' + D.cls[0] + ' or ' + D.cls[1] + '), so the loss is log loss.'
          : ', a number, so the loss is the root-mean-square error.') +
        ' ' +
        tr.length +
        ' rows train the model; ' +
        te.length +
        ' are held out.';
      tpv = D.names.map((_, f) => {
        const s = tr.map((q) => X[q][f]).sort((a, b) => a - b);
        return s[s.length >> 1];
      });
      tpBox.innerHTML = D.names
        .map(
          (nm, f) =>
            `<label class="ctl" for="xgb-tp${f}"><span>${escapeHTML(nm.slice(0, 24))}</span><input type="number" class="num" id="xgb-tp${f}" step="any" value="${+tpv[f].toFixed(2)}"></label>`,
        )
        .join('');
      D.names.forEach((_, f) => {
        $('#xgb-tp' + f, root).addEventListener('input', (e) => {
          const v = parseFloat(e.target.value);
          if (isFinite(v)) {
            tpv[f] = v;
            draw();
          }
        });
      });
      reset();
    };
    const tw = (nd, x) => {
        while (nd.sp) nd = x[nd.sp.f] <= nd.sp.thr ? nd.sp.L : nd.sp.R;
        return nd.w;
      },
      Fx = (x) => {
        let s = base;
        for (const tq of trees) s += tq.eta * tw(tq.root, x);
        return s;
      };
    const loss = (rows) => {
      let s = 0;
      if (D.clf) {
        for (const q of rows) {
          const p = clamp(sig(F[q]), 1e-9, 1 - 1e-9);
          s -= D.y[q] ? Math.log(p) : Math.log(1 - p);
        }
        return s / rows.length;
      }
      for (const q of rows) s += (F[q] - D.y[q]) ** 2;
      return D.sd * Math.sqrt(s / rows.length);
    };
    function reset() {
      const n = D.X.length;
      trees = [];
      eTr = [];
      eTe = [];
      imp = D.names.map(() => 0);
      pruned = 0;
      last = null;
      lastPr = 0;
      wait = 0;
      const m = D.tr.reduce((a, q) => a + D.y[q], 0) / D.tr.length;
      base = D.clf ? Math.log(clamp(m, 0.01, 0.99) / (1 - clamp(m, 0.01, 0.99))) : m;
      F = new Float64Array(n).fill(base);
      g = new Float64Array(n);
      h = new Float64Array(n);
      msg =
        'Round 0: every row gets the same prediction, ' +
        (D.clf
          ? 'a ' +
            (100 * sig(base)).toFixed(1) +
            '% chance (the share of “' +
            D.cls[1] +
            '” in the training rows)'
          : 'the average ' + f2(D.mu + D.sd * base)) +
        '. Held-out loss ' +
        f3(loss(D.te)) +
        '.';
    }
    const lvs = (nd) => (nd.sp ? lvs(nd.sp.L) + lvs(nd.sp.R) : 1);
    const add = () => {
      const tr = D.tr;
      for (const q of tr) {
        if (D.clf) {
          const p = sig(F[q]);
          g[q] = p - D.y[q];
          h[q] = Math.max(p * (1 - p), 1e-6);
        } else {
          g[q] = F[q] - D.y[q];
          h[q] = 1;
        }
      }
      const rows = S.sub >= 1 ? tr.slice() : tr.filter(() => Math.random() < S.sub);
      if (rows.length < 8) return;
      const bt = xgbBuildTree(D.X, D.order, g, h, rows, S),
        tq = { root: bt.root, eta: S.eta };
      trees.push(tq);
      last = tq;
      lastPr = bt.np;
      pruned += bt.np;
      for (let q = 0; q < D.X.length; q++) F[q] += S.eta * tw(tq.root, D.X[q]);
      const wk = (nd) => {
        if (nd.sp) {
          imp[nd.sp.f] += nd.sp.gain;
          wk(nd.sp.L);
          wk(nd.sp.R);
        }
      };
      wk(tq.root);
      eTr.push(loss(tr));
      eTe.push(loss(D.te));
      const k = trees.length,
        sp = tq.root.sp;
      msg =
        'Round ' +
        k +
        ': computed g and h for ' +
        rows.length +
        ' rows and grew a tree. ' +
        (sp
          ? 'Its first question is “' +
            D.names[sp.f] +
            ' ≤ ' +
            +sp.thr.toPrecision(4) +
            '”, with gain ' +
            f2(sp.gain) +
            '. '
          : bt.np
            ? 'Every split it tried was pruned, so this tree is a single leaf and changes almost nothing. '
            : 'No useful split was found, so this tree is a single leaf. ') +
        (bt.np
          ? 'Pruning removed ' +
            bt.np +
            (bt.np === 1 ? ' split' : ' splits') +
            ' whose gain was below γ = ' +
            f2(S.gam) +
            '. '
          : 'Nothing needed pruning. ') +
        'It has ' +
        lvs(tq.root) +
        ' leaves. Added ' +
        f2(S.eta) +
        ' × the tree. Held-out loss ' +
        (k > 1 ? f3(eTe[k - 2]) + ' → ' : '') +
        f3(eTe[k - 1]) +
        (k > 5 && eTe[k - 1] > Math.min(...eTe) * 1.02
          ? ', which is above its best: the model is now overfitting.'
          : '.');
    };
    const draw = () => {
      const t = T(),
        k = trees.length;
      {
        const { g: gc, w, h: hh } = fit(cL);
        let ym = 0;
        for (const v of eTr.concat(eTe)) ym = Math.max(ym, v);
        ym = ym || 1;
        const P = plot(gc, w, hh, [0, Math.max(10, S.B)], [0, ym * 1.1], t, {
          xstep: S.B > 120 ? 50 : S.B > 50 ? 20 : S.B > 20 ? 10 : 5,
          xlabel: 'round',
        });
        gc.font = MONO;
        gc.fillStyle = t.muted;
        gc.textAlign = 'right';
        gc.fillText(+ym.toPrecision(3) + '', P.m.l - 4, P.Y(ym) + 4);
        gc.fillText('0', P.m.l - 4, P.Y(0) + 4);
        clipPlot(gc, P);
        let bi = 0;
        eTe.forEach((v, i) => {
          if (v < eTe[bi]) bi = i;
        });
        if (k) {
          gc.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
          gc.lineWidth = 8;
          gc.beginPath();
          gc.moveTo(P.X(bi + 1), P.Y(0));
          gc.lineTo(P.X(bi + 1), P.Y(ym * 1.1));
          gc.stroke();
        }
        [
          [eTr, t.accent],
          [eTe, t.ok],
        ].forEach(([E, cl]) => {
          gc.strokeStyle = cl;
          gc.lineWidth = 2.4;
          gc.beginPath();
          E.forEach((v, i) => (i ? gc.lineTo(P.X(i + 1), P.Y(v)) : gc.moveTo(P.X(i + 1), P.Y(v))));
          gc.stroke();
        });
        gc.restore();
        gc.textAlign = 'left';
        gc.fillStyle = t.accent;
        gc.fillText('training', P.m.l + 50, P.m.t + 8);
        gc.fillStyle = t.ok;
        gc.fillText('held-out', P.m.l + 50, P.m.t + 23);
        gc.fillStyle = t.cut;
        gc.fillText('gold band = best round', P.m.l + 50, P.m.t + 38);
      }
      {
        const { g: gc, w, h: hh } = fit(cT);
        gc.font = MONO;
        if (!last) {
          gc.fillStyle = t.muted;
          gc.textAlign = 'center';
          gc.fillText('No tree yet. Press Add trees or One round.', w / 2, hh / 2);
        } else {
          const leaves = [],
            all = [];
          let md = 0;
          const lay = (nd, dp) => {
            nd._d = dp;
            md = Math.max(md, dp);
            all.push(nd);
            if (nd.sp) {
              lay(nd.sp.L, dp + 1);
              lay(nd.sp.R, dp + 1);
              nd._x = (nd.sp.L._x + nd.sp.R._x) / 2;
            } else {
              nd._x = leaves.length;
              leaves.push(nd);
            }
          };
          lay(last.root, 0);
          const X = (v) => 60 + (leaves.length > 1 ? v / (leaves.length - 1) : 0.5) * (w - 120),
            Y = (d) => 58 + (md ? d / md : 0) * (hh - 112),
            sm = leaves.length > 8;
          gc.strokeStyle = t.muted;
          gc.lineWidth = 1.2;
          for (const nd of all)
            if (nd.sp)
              [nd.sp.L, nd.sp.R].forEach((ch) => {
                gc.beginPath();
                gc.moveTo(X(nd._x), Y(nd._d));
                gc.lineTo(X(ch._x), Y(ch._d));
                gc.stroke();
              });
          gc.textAlign = 'center';
          for (const nd of all) {
            const x = X(nd._x),
              y = Y(nd._d);
            if (nd.sp) {
              const tx = D.names[nd.sp.f].slice(0, sm ? 9 : 16) + ' ≤ ' + +nd.sp.thr.toPrecision(3);
              gc.font = sm ? '10px "JetBrains Mono",monospace' : MONO;
              const tw2 = gc.measureText(tx).width + 10;
              gc.fillStyle = t.bg;
              gc.strokeStyle = t.ink;
              gc.lineWidth = 1.2;
              gc.beginPath();
              gc.rect(x - tw2 / 2, y - 10, tw2, 20);
              gc.fill();
              gc.stroke();
              gc.fillStyle = t.ink;
              gc.fillText(tx, x, y + 4);
              gc.fillStyle = t.cut;
              gc.fillText('gain ' + f2(nd.sp.gain), x, y + 23);
            } else {
              const up = nd.w >= 0;
              gc.fillStyle = up ? t.gold : t.accent;
              gc.beginPath();
              gc.arc(x, y, sm ? 6 : 9, 0, 7);
              gc.fill();
              gc.font = sm ? '10px "JetBrains Mono",monospace' : MONO;
              gc.fillStyle = t.ink;
              gc.fillText((nd.w >= 0 ? '+' : '') + nd.w.toFixed(2), x, y + 24);
              gc.fillStyle = t.muted;
              gc.fillText(nd.n + ' rows', x, y + 37);
              if (nd.cut) {
                gc.fillStyle = t.ok;
                gc.fillText(
                  '✂ ' +
                    (sm ? '' : D.names[nd.cut.f].slice(0, 10) + ' ') +
                    f2(nd.cut.gain) +
                    ' < γ',
                  x,
                  y - 15,
                );
              }
            }
          }
          gc.textAlign = 'left';
          gc.fillStyle = t.muted;
          gc.fillText('yes ↙  no ↘   leaf value = change to the prediction (standardised)', 8, 12);
        }
      }
      {
        const { g: gc, w, h: hh } = fit(cI),
          tot = imp.reduce((a, b) => a + b, 0) || 1,
          ord = D.names.map((_, f) => f).sort((a, b) => imp[b] - imp[a]),
          rh = Math.min(34, (hh - 16) / ord.length),
          lw = Math.min(190, w * 0.4);
        gc.font = MONO;
        ord.forEach((f, i) => {
          const y = 8 + i * rh,
            v = imp[f] / tot;
          gc.fillStyle = t.ink;
          gc.textAlign = 'right';
          gc.fillText(D.names[f].slice(0, 24), lw - 8, y + rh / 2 + 4);
          gc.fillStyle = mix(t.muted, t.muted, 0, 0.14);
          gc.fillRect(lw, y + rh * 0.2, w - lw - 56, rh * 0.6);
          gc.fillStyle = t.gold;
          gc.fillRect(lw, y + rh * 0.2, (w - lw - 56) * v, rh * 0.6);
          gc.fillStyle = t.muted;
          gc.textAlign = 'left';
          gc.fillText((100 * v).toFixed(0) + '%', w - 50, y + rh / 2 + 4);
        });
      }
      {
        const { g: gc, w, h: hh } = fit(cP);
        if (D.clf) {
          const P = plot(gc, w, hh, [0, 1], [-0.6, 1.6], t, {
            xstep: 0.25,
            xlabel: 'predicted chance of “' + String(D.cls[1]).slice(0, 10) + '”',
          });
          gc.font = MONO;
          gc.fillStyle = t.muted;
          gc.textAlign = 'right';
          gc.fillText(String(D.cls[1]).slice(0, 5), P.m.l - 4, P.Y(1) + 4);
          gc.fillText(String(D.cls[0]).slice(0, 5), P.m.l - 4, P.Y(0) + 4);
          clipPlot(gc, P);
          gc.setLineDash([5, 4]);
          gc.strokeStyle = t.muted;
          gc.beginPath();
          gc.moveTo(P.X(0.5), P.Y(-0.6));
          gc.lineTo(P.X(0.5), P.Y(1.6));
          gc.stroke();
          gc.setLineDash([]);
          const r = rng(4);
          for (const q of D.te) {
            const p = sig(F[q]);
            gc.fillStyle = mix(D.y[q] ? t.gold : t.accent, t.bg, 0, 0.75);
            gc.beginPath();
            gc.arc(P.X(p), P.Y(D.y[q] + (r() - 0.5) * 0.5), 3.4, 0, 7);
            gc.fill();
          }
          gc.restore();
          gc.textAlign = 'left';
          gc.fillStyle = t.muted;
          gc.fillText(
            'upper band = truly “' +
              String(D.cls[1]).slice(0, 8) +
              '”; right of the dashed line = predicted so',
            P.m.l + 6,
            P.m.t + 6,
          );
        } else {
          let lo = 1 / 0,
            hi = -1 / 0;
          for (const q of D.te) {
            const p = D.mu + D.sd * F[q];
            lo = Math.min(lo, p, D.yr[q]);
            hi = Math.max(hi, p, D.yr[q]);
          }
          const pad = (hi - lo) * 0.06 || 1,
            P = plot(gc, w, hh, [lo - pad, hi + pad], [lo - pad, hi + pad], t, {
              xstep: (hi - lo) / 4 || 1,
              xlabel: 'actual',
            });
          gc.font = MONO;
          gc.fillStyle = t.muted;
          gc.textAlign = 'right';
          [lo, hi].forEach((v) => gc.fillText(+v.toPrecision(3) + '', P.m.l - 4, P.Y(v) + 4));
          clipPlot(gc, P);
          gc.setLineDash([5, 4]);
          gc.strokeStyle = t.muted;
          gc.beginPath();
          gc.moveTo(P.X(lo - pad), P.Y(lo - pad));
          gc.lineTo(P.X(hi + pad), P.Y(hi + pad));
          gc.stroke();
          gc.setLineDash([]);
          gc.fillStyle = mix(t.accent, t.accent, 0, 0.7);
          for (const q of D.te) {
            gc.beginPath();
            gc.arc(P.X(D.yr[q]), P.Y(D.mu + D.sd * F[q]), 3.4, 0, 7);
            gc.fill();
          }
          gc.restore();
          gc.textAlign = 'left';
          gc.fillStyle = t.muted;
          gc.fillText(
            'predicted (up) against actual (across); on the dashed line = exact',
            P.m.l + 6,
            P.m.t + 6,
          );
        }
      }
      live.textContent = msg;
      const cu = !k ? 0 : k >= S.B ? 5 : 4;
      root
        .querySelectorAll('#xgb-steps li')
        .forEach((e, i) => e.setAttribute('aria-current', i === cu ? 'true' : 'false'));
      const fx = Fx(tpv);
      tpo.textContent = D.clf
        ? 'For this row the model gives a ' +
          (100 * sig(fx)).toFixed(1) +
          '% chance of “' +
          D.cls[1] +
          '” and predicts “' +
          (sig(fx) >= 0.5 ? D.cls[1] : D.cls[0]) +
          '”.'
        : 'For this row the model predicts “' + D.tname + '” = ' + f2(D.mu + D.sd * fx) + '.';
      let bi = 0;
      eTe.forEach((v, i) => {
        if (v < eTe[bi]) bi = i;
      });
      setR(root, 'k', k + ' of ' + S.B);
      setR(root, 'tr', k ? f3(eTr[k - 1]) : f3(loss(D.tr)));
      setR(root, 'te', k ? f3(eTe[k - 1]) : f3(loss(D.te)));
      setR(root, 'best', k ? String(bi + 1) : '–');
      setR(root, 'lv', last ? String(lvs(last.root)) : '–');
      setR(root, 'pr', String(pruned));
      setR(
        root,
        'acc',
        D.clf
          ? (
              (100 * D.te.filter((q) => (F[q] >= 0 ? 1 : 0) === D.y[q]).length) /
              D.te.length
            ).toFixed(1) + '%'
          : 'not a yes/no target',
      );
    };
    const p0 = presets.house();
    setData(p0.names, p0.rows, p0.t);
    for (let i = 0; i < 20; i++) add();
    const anim = Anim(() => {
      fr++;
      if (wait > 0) {
        wait--;
        if (!wait) {
          reset();
          draw();
        }
        return;
      }
      if (fr % (trees.length < 6 ? 40 : 8)) return;
      if (trees.length < S.B) {
        add();
        draw();
      } else pb.stop();
    });
    pb = playBtn($('#xgb-play', root), anim);
    runCtl(pb, root, 'xgb', { done: () => trees.length >= S.B, restart: () => reset() });
    $('#xgb-step', root).onclick = () => {
      pb.stop();
      if (trees.length >= S.B) reset();
      else add();
      draw();
    };
    $('#xgb-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    const rs = () => {
      reset();
      draw();
    };
    bind(
      root,
      'xgb-B',
      (v) => {
        S.B = v;
        rs();
      },
      (v) => String(v),
    );
    bind(root, 'xgb-eta', (v) => {
      S.eta = v;
      rs();
    });
    bind(
      root,
      'xgb-depth',
      (v) => {
        S.depth = v;
        rs();
      },
      (v) => String(v),
    );
    bind(root, 'xgb-gam', (v) => {
      S.gam = v;
      rs();
    });
    bind(
      root,
      'xgb-lam',
      (v) => {
        S.lam = v;
        rs();
      },
      (v) => (+v).toFixed(1),
    );
    bind(root, 'xgb-sub', (v) => {
      S.sub = v;
      rs();
    });
    $('#xgb-ds', root).onchange = (e) => {
      st.textContent = '';
      const p = presets[e.target.value]();
      setData(p.names, p.rows, p.t);
      draw();
    };
    tsel.onchange = () => {
      setData(raw.names, raw.rows, +tsel.value);
      draw();
    };
    $('#xgb-file', root).onchange = (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const rd2 = new FileReader();
      rd2.onerror = () => {
        st.textContent = 'That file could not be read. Try saving it again as plain CSV.';
      };
      rd2.onload = () => {
        const parsed = parseXgbCSV(rd2.result);
        if (typeof parsed === 'string') {
          st.textContent = parsed;
          return;
        }
        const { names, rows, skipped } = parsed;
        st.textContent =
          'Loaded ' +
          f.name.slice(0, 30) +
          ': ' +
          rows.length +
          ' rows.' +
          (skipped > 0
            ? ' ' +
              skipped +
              ' columns were skipped (text, mostly empty, or beyond the 13-column limit).'
            : '') +
          ' The last numeric column is the target; change it in “Column to predict”.';
        st.style.color = 'var(--ink)';
        let o = $('option[value="csv"]', $('#xgb-ds', root));
        if (!o) {
          o = document.createElement('option');
          o.value = 'csv';
          $('#xgb-ds', root).appendChild(o);
        }
        o.textContent = 'Your CSV: ' + f.name.slice(0, 24);
        $('#xgb-ds', root).value = 'csv';
        pb.stop();
        setData(names, rows, names.length - 1);
        draw();
      };
      rd2.readAsText(f);
      e.target.value = '';
    };
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
