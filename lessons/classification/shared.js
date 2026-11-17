/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  clamp,
  f2,
  fit,
  hex,
  markTP,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
  testPoint,
} from '../../js/core/shared.js';
import { renderFamily } from '../../js/app/lesson-view.js';

const sig = (z) => 1 / (1 + Math.exp(-z));

function mkData(kind, n, noise, seed, o = {}) {
  const r = rng(seed),
    rn = randnFrom(r),
    P = [],
    sd = 0.45 + 0.6 * noise;
  for (let i = 0; i < n; i++) {
    let c = i % 2,
      x,
      y;
    if (kind === 'blobs' || kind === 'overlap') {
      const d = kind === 'overlap' ? 0.55 : 1.15,
        rho = o.corr || 0,
        q = Math.sqrt(1 - rho * rho),
        s = (c ? o.sb || 1 : 1) * sd * 1.2,
        a = rn(),
        b = rn();
      x = (c ? d : -d) + s * a;
      y = (c ? d * 0.7 : -d * 0.7) + s * (rho * a + q * b);
    } else if (kind === 'moons') {
      const a = r() * Math.PI,
        j = 0.1 + 0.4 * noise;
      x = (c ? 0.75 - 1.5 * Math.cos(a) : -0.75 + 1.5 * Math.cos(a)) + j * rn();
      y = (c ? 0.35 - 1.3 * Math.sin(a) : -0.35 + 1.3 * Math.sin(a)) + j * rn();
    } else if (kind === 'circles') {
      const a = r() * 2 * Math.PI,
        j = 0.08 + 0.4 * noise,
        rad = c ? 0.75 * Math.sqrt(r()) : 2.05 + 0.25 * rn();
      x = rad * Math.cos(a) + j * rn();
      y = rad * Math.sin(a) + j * rn();
    } else if (kind === 'xor') {
      x = r() * 5 - 2.5;
      y = r() * 5 - 2.5;
      c = x * y > 0 ? 1 : 0;
      const j = 0.5 * noise;
      x += j * rn();
      y += j * rn();
    } else {
      c = i % 3;
      const m = [
        [-1.4, -0.9],
        [1.4, -0.9],
        [0, 1.4],
      ][c];
      x = m[0] + sd * rn();
      y = m[1] + sd * rn();
    }
    P.push({ x: clamp(x, -2.9, 2.9), y: clamp(y, -2.9, 2.9), c, test: r() < 0.3 });
  }
  return P;
}

function gfit(T, c) {
  const P = T.filter((p) => p.c === c),
    n = Math.max(1, P.length);
  let mx = 0,
    my = 0;
  for (const p of P) {
    mx += p.x;
    my += p.y;
  }
  mx /= n;
  my /= n;
  let a = 0,
    b = 0,
    d = 0;
  for (const p of P) {
    const u = p.x - mx,
      v = p.y - my;
    a += u * u;
    b += u * v;
    d += v * v;
  }
  return { n: P.length, mx, my, a: a / n + 1e-4, b: b / n, d: d / n + 1e-4 };
}

const glog = (G, x, y) => {
  const det = G.a * G.d - G.b * G.b,
    u = x - G.mx,
    v = y - G.my;
  return -0.5 * Math.log(det) - (0.5 * (G.d * u * u - 2 * G.b * u * v + G.a * v * v)) / det;
};

function clfLesson(o) {
  const id = o.id;
  const names = o.multi ? ['A', 'B', 'C'] : ['A', 'B'];
  return {
    ...o,
    renderStage(parts, context) {
      return renderFamily(new URL('./shared.html', import.meta.url), id, parts, context);
    },
    init(root) {
      const c = $('#' + id + '-c', root),
        cr = $('#' + id + '-r', root),
        noteEl = $('#' + id + '-note', root),
        S = { ds: o.ds[0], nz: 0.3, n: o.n || 120, addc: 0, seed: o.seed || 1, ...(o.S || {}) };
      let pts = [],
        probe = [0.6, 0.4],
        PV = null,
        pb,
        draw,
        M,
        FO = null;
      const gen = () => {
        pts = mkData(S.ds, S.n, S.nz, S.seed, S);
      };
      gen();
      const api = {
        root,
        S,
        id,
        cr,
        get pts() {
          return pts;
        },
        get train() {
          return pts.filter((p) => !p.test);
        },
        get test() {
          return pts.filter((p) => p.test);
        },
        get probe() {
          return probe;
        },
        redraw: () => draw(),
        note: (s) => {
          noteEl.textContent = s;
        },
        refit: () => {
          M.fit && M.fit();
          draw();
        },
        regen: () => {
          gen();
          M.reset && M.reset();
          M.fit && M.fit();
          draw();
        },
        bind: (sfx, key, fm, cb) =>
          bind(
            root,
            id + '-' + sfx,
            (v) => {
              S[key] = v;
              (cb || api.refit)();
            },
            fm,
          ),
      };
      const tp = testPoint(
        root,
        id,
        ['Your x', 'Your y'],
        [0.5, 0.5],
        [
          [-3, 3],
          [-3, 3],
        ],
      );
      tp.on = () => draw();
      const liveEl = $('#' + id + '-live', root),
        stepEls = [...root.querySelectorAll('#' + id + '-steps li')];
      M = o.setup(api);
      M.reset && M.reset();
      M.fit && M.fit();
      const cls = (v) => (Array.isArray(v) ? v.indexOf(Math.max(...v)) : v >= 0.5 ? 1 : 0),
        acc = (P) =>
          P.length
            ? ((100 * P.filter((p) => cls(M.p(p.x, p.y)) === p.c).length) / P.length).toFixed(1) +
              '%'
            : '–';
      api.cls = cls;
      draw = () => {
        const t = T(),
          { g, w, h } = fit(c),
          R = 3,
          X = (v) => w / 2 + (v * w) / (2 * R),
          Y = (v) => h / 2 - (v * h) / (2 * R),
          NC = o.nc || 50,
          cw = w / NC,
          ch = h / NC,
          cc = [t.accent, t.gold, t.ok],
          V = [];
        PV = { w, h };
        if (!FO) {
          FO = document.createElement('canvas');
          FO.width = FO.height = NC;
        }
        const fg = FO.getContext('2d'),
          im = fg.createImageData(NC, NC),
          rgb = cc.map(hex);
        for (let i = 0; i < NC; i++)
          for (let j = 0; j < NC; j++) {
            const v = M.p(-R + (2 * R * (i + 0.5)) / NC, -R + (2 * R * (j + 0.5)) / NC);
            let k, al;
            if (Array.isArray(v)) {
              k = cls(v);
              al = clamp(0.06 + 0.5 * (v[k] - 0.34), 0.06, 0.4);
            } else {
              V.push(v);
              k = v >= 0.5 ? 1 : 0;
              al = 0.05 + 0.7 * Math.abs(v - 0.5);
            }
            const q = ((NC - 1 - j) * NC + i) * 4;
            im.data[q] = rgb[k][0];
            im.data[q + 1] = rgb[k][1];
            im.data[q + 2] = rgb[k][2];
            im.data[q + 3] = Math.round(255 * al);
          }
        fg.putImageData(im, 0, 0);
        g.imageSmoothingEnabled = true;
        g.drawImage(FO, 0, 0, w, h);
        if (V.length) {
          g.strokeStyle = t.ink;
          g.lineWidth = 2.2;
          g.beginPath();
          const px = (i) => (i + 0.5) * cw,
            py = (j) => h - (j + 0.5) * ch,
            L = 0.5;
          for (let i = 0; i < NC - 1; i++)
            for (let j = 0; j < NC - 1; j++) {
              const a = V[i * NC + j],
                b = V[(i + 1) * NC + j],
                c2 = V[(i + 1) * NC + j + 1],
                d = V[i * NC + j + 1];
              if (a < L === b < L && b < L === c2 < L && c2 < L === d < L) continue;
              const ps = [];
              const E = (v1, v2, x1, y1, x2, y2) => {
                if (v1 < L !== v2 < L) {
                  const u = (L - v1) / (v2 - v1 || 1e-9);
                  ps.push(x1 + (x2 - x1) * u, y1 + (y2 - y1) * u);
                }
              };
              E(a, b, px(i), py(j), px(i + 1), py(j));
              E(b, c2, px(i + 1), py(j), px(i + 1), py(j + 1));
              E(c2, d, px(i + 1), py(j + 1), px(i), py(j + 1));
              E(d, a, px(i), py(j + 1), px(i), py(j));
              if (ps.length >= 4) {
                g.moveTo(ps[0], ps[1]);
                g.lineTo(ps[2], ps[3]);
                if (ps.length === 8) {
                  g.moveTo(ps[4], ps[5]);
                  g.lineTo(ps[6], ps[7]);
                }
              }
            }
          g.stroke();
        }
        M.overlay && M.overlay(g, X, Y, t, w, h);
        for (const p of pts) {
          if (p.test) {
            g.strokeStyle = cc[p.c];
            g.lineWidth = 2;
            g.beginPath();
            g.arc(X(p.x), Y(p.y), 4, 0, 7);
            g.stroke();
          } else {
            g.fillStyle = cc[p.c];
            g.strokeStyle = t.bg;
            g.lineWidth = 1.3;
            g.beginPath();
            g.arc(X(p.x), Y(p.y), 4.6, 0, 7);
            g.fill();
            g.stroke();
          }
        }
        g.font = MONO;
        g.textAlign = 'left';
        let lx = 8;
        names.forEach((n, i) => {
          g.fillStyle = i === 1 ? t.cut : cc[i];
          g.fillText('● class ' + n, lx, h - 8);
          lx += 74;
        });
        g.fillStyle = t.muted;
        g.fillText('○ test', lx, h - 8);
        markTP(g, X(tp.v[0]), Y(tp.v[1]), t);
        {
          const v = M.p(tp.v[0], tp.v[1]),
            k = cls(v);
          tp.out.textContent =
            'At (' +
            f2(tp.v[0]) +
            ', ' +
            f2(tp.v[1]) +
            ') the model predicts class ' +
            'ABC'[k] +
            '. ' +
            (Array.isArray(v)
              ? 'Probabilities: ' +
                v.map((q, i) => 'ABC'[i] + ' ' + (100 * q).toFixed(1) + '%').join(', ') +
                '.'
              : o.hard
                ? ''
                : 'Probability of class B: ' + (100 * v).toFixed(1) + '%.');
        }
        liveEl.textContent = M.status ? M.status() : o.still || '';
        const cu = M.cur ? M.cur() : -1;
        stepEls.forEach((e, i) => e.setAttribute('aria-current', i === cu ? 'true' : 'false'));
        M.right && M.right(t);
        setR(root, 'tr', acc(api.train));
        setR(root, 'te', acc(api.test));
        M.read && M.read((k, v) => setR(root, k, v));
        root
          .querySelectorAll('#' + id + '-add button')
          .forEach((b) =>
            b.setAttribute('aria-pressed', +b.dataset.c === S.addc ? 'true' : 'false'),
          );
      };
      let chk = () => false;
      const anim = Anim(() => {
        if (M.tick(chk.remaining()) !== false) draw();
        if (chk()) draw();
      });
      pb = playBtn($('#' + id + '-play', root), anim);
      api.pause = () => pb.stop();
      chk = runCtl(pb, root, id, {
        done: () => !!(M.done && M.done()),
        restart: () => {
          api.note('');
          M.reset && M.reset();
          M.fit && M.fit();
        },
        count: M.count,
        what: o.maxWhat,
      });
      const sb = $('#' + id + '-step', root);
      if (sb)
        sb.onclick = () => {
          pb.stop();
          M.stepOnce();
          draw();
        };
      $('#' + id + '-reset', root).onclick = () => {
        api.note('');
        M.reset && M.reset();
        M.fit && M.fit();
        draw();
      };
      $('#' + id + '-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        api.note('');
        api.regen();
      };
      bind(root, id + '-nz', (v) => {
        S.nz = v;
        api.regen();
      });
      bind(
        root,
        id + '-n',
        (v) => {
          S.n = v;
          api.regen();
        },
        (v) => String(v),
      );
      $('#' + id + '-new', root).onclick = () => {
        S.seed++;
        api.note('');
        api.regen();
      };
      $('#' + id + '-add', root).addEventListener('click', (e) => {
        const k = e.target.dataset && e.target.dataset.c;
        if (k != null) {
          S.addc = +k;
          draw();
        }
      });
      let drag = -1,
        dragT = false;
      const at = (e) => {
        const r = c.getBoundingClientRect();
        return [
          clamp((e.clientX - r.left - PV.w / 2) / (PV.w / 6), -2.95, 2.95),
          clamp(-(e.clientY - r.top - PV.h / 2) / (PV.h / 6), -2.95, 2.95),
          e.clientX - r.left,
          e.clientY - r.top,
        ];
      };
      c.addEventListener('pointerdown', (e) => {
        if (!PV) return;
        const a = at(e);
        if (S.addc < 0) {
          tp.set([a[0], a[1]]);
          dragT = true;
          c.setPointerCapture(e.pointerId);
          draw();
          return;
        }
        let bi = -1,
          bd = 14;
        pts.forEach((p, i) => {
          const d = Math.hypot(
            PV.w / 2 + (p.x * PV.w) / 6 - a[2],
            PV.h / 2 - (p.y * PV.h) / 6 - a[3],
          );
          if (d < bd) {
            bd = d;
            bi = i;
          }
        });
        if (bi < 0) {
          pts.push({ x: a[0], y: a[1], c: S.addc, test: false });
          bi = pts.length - 1;
        }
        drag = bi;
        c.setPointerCapture(e.pointerId);
        M.fit && M.fit();
        draw();
      });
      c.addEventListener('pointermove', (e) => {
        if (!PV) return;
        const a = at(e);
        if (dragT) {
          tp.set([a[0], a[1]]);
          draw();
        } else if (drag >= 0) {
          pts[drag].x = a[0];
          pts[drag].y = a[1];
          M.fit && M.fit();
          draw();
        } else if (o.probe) {
          probe = [a[0], a[1]];
          draw();
        }
      });
      ['pointerup', 'pointercancel'].forEach((n) =>
        c.addEventListener(n, () => {
          drag = -1;
          dragT = false;
        }),
      );
      return {
        draw,
        stop() {
          pb.stop();
          M.stop && M.stop();
        },
        auto() {
          if (o.fromZero && M.reset) {
            M.reset();
            api.note('');
            draw();
          }
          pb.start();
          M.start && M.start();
        },
      };
    },
  };
}

const sweepTick = (get, set, lo, hi, sec) => {
  let ph = null;
  return () => {
    if (ph === null) {
      const u = clamp((get() - lo) / (hi - lo), 0, 1);
      ph = Math.acos(1 - 2 * u) / (2 * Math.PI);
    }
    ph += 1 / 60 / sec;
    set(lo + ((hi - lo) * (1 - Math.cos(ph * 2 * Math.PI))) / 2);
    return true;
  };
};

const lineAt = (g, X, Y, w, cst) => {
  const pts = [];
  if (Math.abs(w[2]) > Math.abs(w[1])) {
    for (const x of [-3, 3]) pts.push([x, (cst - w[0] - w[1] * x) / w[2]]);
  } else if (Math.abs(w[1]) > 1e-9) {
    for (const y of [-3, 3]) pts.push([(cst - w[0] - w[2] * y) / w[1], y]);
  }
  if (pts.length) {
    g.beginPath();
    g.moveTo(X(pts[0][0]), Y(pts[0][1]));
    g.lineTo(X(pts[1][0]), Y(pts[1][1]));
    g.stroke();
  }
};

export { sig, mkData, gfit, glog, clfLesson, sweepTick, lineAt };
