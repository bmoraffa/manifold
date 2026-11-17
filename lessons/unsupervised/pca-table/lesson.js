/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  SANS,
  T,
  bind,
  clamp,
  f2,
  fit,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { jacobi, uCur, uNum } from '../shared.js';

export default {
  init(root) {
    const chh = $('#pt-h', root),
      cs = $('#pt-s', root),
      cb = $('#pt-b', root),
      cl = $('#pt-l', root),
      live = $('#pt-live', root),
      st = $('#pt-st', root),
      S = { ds: 'exam', std: true, k: 1 };
    let names = [],
      X = [],
      n = 0,
      d = 0,
      mu = [],
      sd = [],
      Z = [],
      Cm = [],
      E = null,
      tot = 1,
      sc = [],
      ph = 0,
      msg = '',
      cu = -1,
      wait = 0,
      fr = 0,
      pb,
      custom = null;
    const sim = (kind) => {
      const r = rng(kind === 'exam' ? 21 : 34),
        rn = randnFrom(r),
        rows = [];
      for (let i = 0; i < 220; i++) {
        if (kind === 'exam') {
          const gq = rn(),
            q = rn(),
            w = rn(),
            cl2 = (x) => clamp(Math.round(x), 0, 100);
          rows.push([
            cl2(62 + 9 * (0.6 * gq + 0.7 * q) + 5 * rn()),
            cl2(58 + 11 * (0.6 * gq + 0.75 * q) + 5 * rn()),
            cl2(60 + 10 * (0.55 * gq + 0.65 * q) + 6 * rn()),
            cl2(65 + 8 * (0.6 * gq + 0.7 * w) + 5 * rn()),
            cl2(63 + 9 * (0.55 * gq + 0.7 * w) + 6 * rn()),
            cl2(66 + 8 * (0.5 * gq + 0.75 * w) + 5 * rn()),
          ]);
        } else {
          const s = rn(),
            b = rn(),
            r1 = (x) => Math.round(x * 10) / 10;
          rows.push([
            r1(170 + 9 * s + 2 * rn()),
            r1(171 + 9.5 * s + 2.5 * rn()),
            r1(80 + 5 * s + 1.5 * rn()),
            r1(70 + 7 * s + 9 * b + 3 * rn()),
            r1(82 + 3 * s + 9 * b + 3 * rn()),
            r1(42 + 1.8 * s + 0.6 * rn()),
          ]);
        }
      }
      return {
        names:
          kind === 'exam'
            ? ['Algebra', 'Calculus', 'Physics', 'Literature', 'History', 'Languages']
            : ['Height cm', 'Arm span cm', 'Leg cm', 'Weight kg', 'Waist cm', 'Shoe size'],
        X: rows,
      };
    };
    const comp = () => {
      const D = custom || sim(S.ds);
      names = D.names;
      X = D.X;
      n = X.length;
      d = names.length;
      mu = names.map((_, j) => X.reduce((s, r) => s + r[j], 0) / n);
      sd = names.map(
        (_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / (n - 1)) || 1,
      );
      Z = X.map((r) => r.map((v, j) => (v - mu[j]) / (S.std ? sd[j] : 1)));
      Cm = names.map((_, j) =>
        names.map((_, k) => {
          let s = 0;
          for (const r of Z) s += r[j] * r[k];
          return s / (n - 1);
        }),
      );
      E = jacobi(Cm);
      tot = E.val.reduce((a, b) => a + Math.max(0, b), 0) || 1;
      sc = Z.map((r) => [0, 1].map((k) => r.reduce((s, v, j) => s + v * E.vec[k][j], 0)));
      S.k = clamp(S.k, 1, d);
      kb.el.max = d;
      kb.set(S.k);
    };
    const nrev = () => clamp(ph - 2, 0, d),
      end = () => 3 + d,
      cum = (k) => E.val.slice(0, k).reduce((a, b) => a + Math.max(0, b), 0) / tot;
    const reset = () => {
      ph = 0;
      cu = -1;
      wait = 0;
      msg =
        'The table has ' +
        n +
        ' rows and ' +
        d +
        ' columns: ' +
        names.join(', ') +
        '. No column is singled out as the thing to predict. Press Run or One step.';
    };
    const top = (v, m) =>
      v
        .map((x, j) => [x, j])
        .sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]))
        .slice(0, m)
        .map((q) => names[q[1]] + ' (' + (q[0] >= 0 ? '+' : '−') + Math.abs(q[0]).toFixed(2) + ')')
        .join(', ');
    const say = () => {
      if (ph === 1) {
        cu = 0;
        const hi = sd.indexOf(Math.max(...sd)),
          lo = sd.indexOf(Math.min(...sd));
        msg = S.std
          ? 'Standardise: “' +
            names[hi] +
            '” had the largest standard deviation (' +
            uNum(sd[hi]) +
            ') and “' +
            names[lo] +
            '” the smallest (' +
            uNum(sd[lo]) +
            '). Each column has now had its mean subtracted and been divided by its standard deviation, so all of them have mean 0 and spread 1 and none dominates because of its units.'
          : 'Centre only: each column had its mean subtracted but keeps its own units. “' +
            names[hi] +
            '” has the largest standard deviation (' +
            uNum(sd[hi]) +
            ') against ' +
            uNum(sd[lo]) +
            ' for “' +
            names[lo] +
            '”, so it will dominate the first component.';
      } else if (ph === 2) {
        cu = 1;
        let bj = 0,
          bk = 1,
          bv = -1;
        for (let j = 0; j < d; j++)
          for (let k = j + 1; k < d; k++) {
            const r = Cm[j][k] / Math.sqrt(Cm[j][j] * Cm[k][k]);
            if (Math.abs(r) > bv) {
              bv = Math.abs(r);
              bj = j;
              bk = k;
            }
          }
        msg =
          (S.std ? 'Correlation' : 'Covariance') +
          ' matrix: one number for each of the ' +
          (d * (d - 1)) / 2 +
          ' pairs of columns. The most strongly related pair is “' +
          names[bj] +
          '” and “' +
          names[bk] +
          '”, with correlation ' +
          f2(Cm[bj][bk] / Math.sqrt(Cm[bj][bj] * Cm[bk][bk])) +
          '. Blocks of strong colour are groups of columns that move together.';
      } else if (ph < end()) {
        const k = ph - 2;
        cu = k === d ? 3 : 2;
        msg =
          'Component ' +
          k +
          ': the direction of greatest spread' +
          (k > 1 ? ' at right angles to the ' + (k - 1) + ' before it' : '') +
          '. Its largest weights are on ' +
          top(E.vec[k - 1], 3) +
          '. It carries ' +
          ((100 * Math.max(0, E.val[k - 1])) / tot).toFixed(1) +
          '% of the total spread, and the first ' +
          k +
          ' together carry ' +
          (100 * cum(k)).toFixed(1) +
          '%.';
        if (k <= d) {
          S.k = k;
          kb.set(k);
        }
      } else {
        cu = 4;
        msg =
          'Project: every row is replaced by its scores on the first two components. Two numbers per row now keep ' +
          (100 * cum(2)).toFixed(1) +
          '% of the spread of all ' +
          d +
          ' columns. Each dot on the lower left is one row of the table.';
      }
    };
    const step = () => {
      if (ph >= end()) {
        reset();
        return;
      }
      ph++;
      say();
    };
    const fin = () => {
      ph = end();
      say();
    };
    const draw = () => {
      const t = T(),
        short = (s) => (s.length > 10 ? s.slice(0, 9) + '…' : s);
      {
        const { g, w, h } = fit(chh);
        $('#pt-hcap', root).textContent =
          ph < 2
            ? ph === 0
              ? 'The first rows of the table, as given'
              : S.std
                ? 'The same rows after standardising'
                : 'The same rows after centring'
            : S.std
              ? 'Correlation matrix: green pairs rise together, red pairs trade off'
              : 'Covariance matrix: green pairs rise together, red pairs trade off';
        g.font = MONO;
        if (ph < 2) {
          const nc = Math.min(d, 7),
            cw = (w - 16) / nc,
            rows = Math.min(n, Math.floor((h - 40) / 20));
          g.textAlign = 'right';
          g.fillStyle = t.ink;
          for (let j = 0; j < nc; j++) g.fillText(short(names[j]), 8 + (j + 1) * cw - 6, 18);
          g.strokeStyle = t.line;
          g.beginPath();
          g.moveTo(8, 26);
          g.lineTo(w - 8, 26);
          g.stroke();
          g.fillStyle = t.muted;
          for (let i = 0; i < rows; i++)
            for (let j = 0; j < nc; j++) {
              const v = ph ? Z[i][j] : X[i][j];
              g.fillStyle = ph ? (v >= 0 ? t.accent : t.ok) : t.muted;
              g.fillText(
                ph ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2) : uNum(v),
                8 + (j + 1) * cw - 6,
                44 + i * 20,
              );
            }
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            '… ' + (n - rows) + ' more rows' + (d > nc ? ', ' + (d - nc) + ' more columns' : ''),
            8,
            h - 6,
          );
        } else {
          const lm = Math.min(86, w * 0.2),
            tm = 20,
            cz = Math.min((w - lm - 8) / d, (h - tm - 8) / d);
          for (let j = 0; j < d; j++) {
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText(short(names[j]), lm - 6, tm + (j + 0.5) * cz + 4);
            g.textAlign = 'center';
            g.fillText(names[j].slice(0, cz > 52 ? 7 : 3), lm + (j + 0.5) * cz, tm - 6);
            for (let k = 0; k < d; k++) {
              const r = Cm[j][k] / Math.sqrt(Cm[j][j] * Cm[k][k] || 1),
                cl2 = r >= 0 ? t.accent : t.ok;
              g.fillStyle = mix(cl2, cl2, 0, Math.min(1, Math.abs(r)) * 0.9);
              g.fillRect(lm + k * cz + 1, tm + j * cz + 1, cz - 2, cz - 2);
              if (cz > 30) {
                g.fillStyle = Math.abs(r) > 0.55 ? t.bg : t.ink;
                g.fillText(r.toFixed(2), lm + (k + 0.5) * cz, tm + (j + 0.5) * cz + 4);
              }
            }
          }
        }
      }
      {
        const { g, w, h } = fit(cs),
          m = { l: 44, r: 14, t: 16, b: 26 },
          bw = (w - m.l - m.r) / d,
          Y = (v) => h - m.b - v * (h - m.t - m.b),
          nr = nrev();
        g.font = MONO;
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h - m.b);
        g.lineTo(w - m.r, h - m.b);
        g.stroke();
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        [0, 0.5, 1].forEach((v) => g.fillText(100 * v + '%', m.l - 5, Y(v) + 4));
        g.setLineDash([3, 4]);
        g.strokeStyle = t.line;
        g.beginPath();
        g.moveTo(m.l, Y(0.9));
        g.lineTo(w - m.r, Y(0.9));
        g.stroke();
        g.setLineDash([]);
        g.textAlign = 'right';
        g.fillText('90%', w - m.r, Y(0.9) - 4);
        for (let k = 0; k < d; k++) {
          const v = Math.max(0, E.val[k]) / tot,
            x = m.l + k * bw + bw * 0.18,
            bwid = bw * 0.64;
          g.textAlign = 'center';
          g.fillStyle = t.muted;
          g.fillText(String(k + 1), x + bwid / 2, h - m.b + 15);
          if (k < nr) {
            g.fillStyle = k + 1 === S.k ? t.gold : mix(t.gold, t.gold, 0, 0.55);
            g.fillRect(x, Y(v), bwid, Y(0) - Y(v));
            const tall = Y(0) - Y(v) > 26;
            g.fillStyle = tall ? t.bg : t.ink;
            g.fillText((100 * v).toFixed(0) + '%', x + bwid / 2, tall ? Y(v) + 15 : Y(v) - 5);
          } else {
            g.strokeStyle = t.line;
            g.setLineDash([3, 3]);
            g.strokeRect(x, Y(0.02), bwid, Y(0) - Y(0.02));
            g.setLineDash([]);
          }
        }
        if (nr) {
          g.strokeStyle = t.accent;
          g.lineWidth = 2.2;
          g.beginPath();
          for (let k = 0; k < nr; k++) {
            const x = m.l + (k + 0.5) * bw,
              y = Y(cum(k + 1));
            k ? g.lineTo(x, y) : g.moveTo(x, y);
          }
          g.stroke();
          g.fillStyle = t.accent;
          for (let k = 0; k < nr; k++) {
            g.beginPath();
            g.arc(m.l + (k + 0.5) * bw, Y(cum(k + 1)), 3.4, 0, 7);
            g.fill();
          }
        } else {
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('no components found yet', w / 2, h / 2);
        }
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('component', w - m.r, h - 3);
      }
      {
        const { g, w, h } = fit(cb);
        g.font = MONO;
        if (nrev() < 2) {
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('this picture needs the first two components', w / 2, h / 2);
        } else {
          let mx = 1e-9,
            my = 1e-9;
          for (const q of sc) {
            mx = Math.max(mx, Math.abs(q[0]));
            my = Math.max(my, Math.abs(q[1]));
          }
          const X2 = (v) => w / 2 + (v / mx) * (w / 2 - 30),
            Y2 = (v) => h / 2 - (v / my) * (h / 2 - 26);
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(8, h / 2);
          g.lineTo(w - 8, h / 2);
          g.moveTo(w / 2, 8);
          g.lineTo(w / 2, h - 8);
          g.stroke();
          g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
          for (const q of sc) {
            g.beginPath();
            g.arc(X2(q[0]), Y2(q[1]), 2.6, 0, 7);
            g.fill();
          }
          let lm = 1e-9;
          const put = [];
          for (let j = 0; j < d; j++) lm = Math.max(lm, Math.hypot(E.vec[0][j], E.vec[1][j]));
          for (let j = 0; j < d; j++) {
            const ax = (E.vec[0][j] / lm) * 0.82,
              ay = (E.vec[1][j] / lm) * 0.82,
              x = w / 2 + ax * (w / 2 - 30),
              y = h / 2 - ay * (h / 2 - 26);
            g.strokeStyle = t.ok;
            g.lineWidth = 2;
            g.beginPath();
            g.moveTo(w / 2, h / 2);
            g.lineTo(x, y);
            g.stroke();
            g.fillStyle = t.ok;
            g.beginPath();
            g.arc(x, y, 3.5, 0, 7);
            g.fill();
            g.font = SANS;
            g.fillStyle = t.ink;
            g.textAlign = ax >= 0 ? 'left' : 'right';
            const lx = clamp(x + (ax >= 0 ? 7 : -7), 50, w - 50);
            let ly = clamp(y + (ay >= 0 ? -6 : 14), 14, h - 6);
            for (let q = 0; q < 8; q++) {
              if (put.some((p) => Math.abs(p[0] - lx) < 80 && Math.abs(p[1] - ly) < 14))
                ly += ay >= 0 ? -14 : 14;
              else break;
            }
            put.push([lx, ly]);
            g.fillText(short(names[j]), lx, ly);
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('component 1 →', w - 8, h / 2 - 6);
          g.textAlign = 'left';
          g.fillText('↑ component 2', w / 2 + 6, 14);
        }
      }
      {
        const { g, w, h } = fit(cl),
          k = Math.min(S.k, Math.max(1, nrev()));
        $('#pt-lcap', root).textContent = 'Weights of component ' + k + ' on each column';
        g.font = MONO;
        if (!nrev()) {
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('no components found yet', w / 2, h / 2);
        } else {
          const lm = Math.min(110, w * 0.26),
            rh = Math.min(34, (h - 30) / d),
            x0 = lm + (w - lm - 50) / 2,
            sx = (w - lm - 50) / 2;
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(x0, 8);
          g.lineTo(x0, 14 + d * rh);
          g.stroke();
          for (let j = 0; j < d; j++) {
            const v = E.vec[k - 1][j],
              y = 14 + j * rh;
            g.fillStyle = t.ink;
            g.textAlign = 'right';
            g.fillText(names[j].slice(0, 14), lm - 8, y + rh / 2 + 4);
            g.fillStyle = v >= 0 ? t.accent : t.ok;
            g.fillRect(v >= 0 ? x0 : x0 + v * sx, y + rh * 0.2, Math.abs(v) * sx, rh * 0.6);
            g.fillStyle = t.muted;
            g.textAlign = v >= 0 ? 'left' : 'right';
            g.fillText(
              (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2),
              x0 + v * sx + (v >= 0 ? 5 : -5),
              y + rh / 2 + 4,
            );
          }
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText(
            'score = sum of weight × ' + (S.std ? 'standardised' : 'centred') + ' value',
            w / 2,
            h - 6,
          );
        }
      }
      live.textContent = msg;
      uCur(root, 'pt', cu);
      const nr = nrev();
      setR(root, 'n', String(n));
      setR(root, 'd', String(d));
      setR(root, 'p1', nr >= 1 ? (100 * cum(1)).toFixed(1) + '%' : '–');
      setR(root, 'p2', nr >= 2 ? (100 * cum(2)).toFixed(1) + '%' : '–');
      let k90 = 1;
      while (k90 < d && cum(k90) < 0.9) k90++;
      setR(root, 'k90', nr >= d ? k90 + ' of ' + d : '–');
    };
    const kb = bind(
      root,
      'pt-k',
      (v) => {
        S.k = clamp(v, 1, d || 1);
        pb && pb.stop();
        draw();
      },
      (v) => String(v),
    );
    comp();
    fin();
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
      if (fr % 64) return;
      step();
      draw();
      chk();
    });
    pb = playBtn($('#pt-play', root), anim);
    const chk = runCtl(pb, root, 'pt', { done: () => ph >= end(), restart: () => reset() });
    $('#pt-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#pt-reset', root).onclick = () => {
      pb.stop();
      reset();
      draw();
    };
    $('#pt-std', root).onchange = (e) => {
      S.std = e.target.checked;
      pb.stop();
      comp();
      fin();
      draw();
    };
    $('#pt-ds', root).onchange = (e) => {
      S.ds = e.target.value;
      custom = null;
      st.textContent = '';
      pb.stop();
      comp();
      fin();
      draw();
    };
    $('#pt-file', root).onchange = (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (!f) return;
      const fr2 = new FileReader();
      fr2.onerror = () => {
        st.textContent =
          'That file could not be read. Save it again as plain CSV and upload it once more.';
      };
      fr2.onload = () => {
        const r = parseTable(fr2.result);
        if (typeof r === 'string') {
          st.textContent = r;
          return;
        }
        custom = r;
        pb.stop();
        comp();
        fin();
        st.textContent =
          'Loaded ' +
          f.name +
          ': ' +
          r.X.length +
          ' rows and ' +
          r.names.length +
          ' numeric columns.' +
          (r.skipped > 0
            ? ' ' +
              r.skipped +
              ' column' +
              (r.skipped === 1 ? ' was' : 's were') +
              ' skipped (text, mostly empty, or beyond the 12-column limit).'
            : '');
        draw();
      };
      fr2.readAsText(f);
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

export function parseTable(text) {
  try {
    const lines = String(text)
      .split(/\r?\n/)
      .filter((l) => l.trim())
      .map((l) => l.split(/[,;\t]/).map((v) => v.trim().replace(/^"|"$/g, '').trim()));
    if (lines.length < 11)
      return 'This file has fewer than 10 data rows. PCA needs more rows than that to say anything. Add rows and upload again.';
    const isNum = (v) => v !== '' && isFinite(+v),
      head = lines[0].some((v) => !isNum(v)),
      body = head ? lines.slice(1) : lines,
      nc = lines.reduce((n, l) => Math.max(n, l.length), 0),
      cols = [];
    for (let j = 0; j < nc && cols.length < 12; j++) {
      let ok = 0;
      for (const r of body) if (isNum(r[j] || '')) ok++;
      if (ok / body.length > 0.9) cols.push(j);
    }
    if (cols.length < 3)
      return 'Fewer than three numeric columns were found. PCA on a table needs at least three. Check that the file is comma-separated and upload again.';
    const rows = [];
    for (const r of body) {
      const v = cols.map((j) => +(r[j] || NaN));
      if (v.every(isFinite)) rows.push(v);
      if (rows.length >= 2000) break;
    }
    if (rows.length < 10)
      return 'Fewer than 10 complete rows were found after dropping rows with gaps. Fill the gaps and upload again.';
    return {
      names: cols.map((j, i) =>
        head && lines[0][j] ? lines[0][j].slice(0, 18) : 'column ' + (i + 1),
      ),
      X: rows,
      skipped: nc - cols.length,
    };
  } catch {
    return 'That file could not be parsed. Try a smaller plain CSV file.';
  }
}
