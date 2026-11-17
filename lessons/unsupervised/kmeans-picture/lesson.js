/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  T,
  bind,
  fit,
  markTP,
  playBtn,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { uCur, uLine, uNum } from '../shared.js';

export default {
  init(root) {
    const co = $('#kp-o', root),
      cq = $('#kp-q', root),
      cc = $('#kp-c', root),
      ce = $('#kp-e', root),
      live = $('#kp-live', root),
      st = $('#kp-st', root),
      tpo = $('#kp-tpo', root),
      S = { K: 6, az: 35 };
    let src = null,
      out = null,
      W = 0,
      H = 0,
      N = 0,
      px = null,
      C = [],
      A = null,
      ph = 'seed',
      it = 0,
      hist = [],
      msg = '',
      cu = 0,
      wait = 0,
      fr = 0,
      pb,
      run = 1,
      samp = [],
      uniq = 0,
      lastCh = 0,
      pick = -1,
      spin = false;
    const builtin = () => {
      const w = 160,
        h = 120,
        o = document.createElement('canvas');
      o.width = w;
      o.height = h;
      const g = o.getContext('2d'),
        r = rng(5);
      let gr = g.createLinearGradient(0, 0, 0, h * 0.62);
      gr.addColorStop(0, '#2a5b9c');
      gr.addColorStop(0.6, '#8fc0e8');
      gr.addColorStop(1, '#f6d9a8');
      g.fillStyle = gr;
      g.fillRect(0, 0, w, h);
      gr = g.createRadialGradient(w * 0.76, h * 0.24, 2, w * 0.76, h * 0.24, h * 0.24);
      gr.addColorStop(0, '#fff6c8');
      gr.addColorStop(0.35, '#ffd45e');
      gr.addColorStop(1, 'rgba(255,200,80,0)');
      g.fillStyle = gr;
      g.fillRect(0, 0, w, h);
      g.fillStyle = 'rgba(255,255,255,.85)';
      [
        [30, 22, 13],
        [44, 19, 10],
        [55, 24, 9],
        [110, 36, 8],
        [120, 33, 10],
      ].forEach((q) => {
        g.beginPath();
        g.ellipse(q[0], q[1], q[2], q[2] * 0.5, 0, 0, 7);
        g.fill();
      });
      g.fillStyle = '#6f8fa6';
      g.beginPath();
      g.moveTo(0, h);
      for (let x = 0; x <= w; x += 2)
        g.lineTo(
          x,
          h * 0.56 - h * 0.09 * Math.sin((x / w) * 5 + 1) - h * 0.03 * Math.sin((x / w) * 13),
        );
      g.lineTo(w, h);
      g.fill();
      gr = g.createLinearGradient(0, h * 0.58, 0, h);
      gr.addColorStop(0, '#6aa84f');
      gr.addColorStop(1, '#1f4a22');
      g.fillStyle = gr;
      g.beginPath();
      g.moveTo(0, h);
      for (let x = 0; x <= w; x += 2)
        g.lineTo(x, h * 0.69 - h * 0.06 * Math.sin((x / w) * 3.2 + 2.2));
      g.lineTo(w, h);
      g.fill();
      g.fillStyle = '#efe6d4';
      g.fillRect(34, 62, 36, 26);
      g.fillStyle = '#b5452f';
      g.beginPath();
      g.moveTo(29, 63);
      g.lineTo(52, 44);
      g.lineTo(75, 63);
      g.fill();
      g.fillStyle = '#5a3a22';
      g.fillRect(48, 72, 9, 16);
      g.fillStyle = '#ffd97a';
      g.fillRect(38, 68, 7, 7);
      g.fillRect(60, 68, 7, 7);
      g.fillStyle = '#4a3020';
      g.fillRect(112, 66, 5, 26);
      [
        ['#2f7d32', 114, 58, 15],
        ['#3f9a3d', 106, 64, 10],
        ['#286b2c', 123, 64, 10],
      ].forEach((q) => {
        g.fillStyle = q[0];
        g.beginPath();
        g.arc(q[1], q[2], q[3], 0, 7);
        g.fill();
      });
      for (let i = 0; i < 70; i++) {
        g.fillStyle = ['#e8443a', '#f2c230', '#f5f0e6', '#d65aa8'][i % 4];
        g.beginPath();
        g.arc(r() * w, h * 0.8 + r() * h * 0.19, 1.3, 0, 7);
        g.fill();
      }
      return o;
    };
    const load = (o) => {
      src = o;
      W = o.width;
      H = o.height;
      N = W * H;
      const d = o.getContext('2d').getImageData(0, 0, W, H).data;
      px = new Float32Array(N * 3);
      const seen = new Set();
      for (let i = 0; i < N; i++) {
        px[i * 3] = d[i * 4];
        px[i * 3 + 1] = d[i * 4 + 1];
        px[i * 3 + 2] = d[i * 4 + 2];
        seen.add((d[i * 4] << 16) | (d[i * 4 + 1] << 8) | d[i * 4 + 2]);
      }
      uniq = seen.size;
      out = document.createElement('canvas');
      out.width = W;
      out.height = H;
      const r = rng(3);
      samp = [];
      for (let i = 0; i < 1500; i++) samp.push(Math.floor(r() * N));
      pick = -1;
      tpo.textContent = '';
      run = 1;
      reset();
    };
    const dist = (i, q) =>
      (px[i * 3] - q[0]) ** 2 + (px[i * 3 + 1] - q[1]) ** 2 + (px[i * 3 + 2] - q[2]) ** 2;
    const nearC = (i) => {
      let b = 0,
        bd = 1e18;
      for (let k = 0; k < C.length; k++) {
        const d = dist(i, C[k]);
        if (d < bd) {
          bd = d;
          b = k;
        }
      }
      return [b, bd];
    };
    const paint = () => {
      const g = out.getContext('2d'),
        im = g.createImageData(W, H),
        d = im.data;
      for (let i = 0; i < N; i++) {
        const q = A && A[i] < C.length ? C[A[i]] : null;
        d[i * 4] = q ? q[0] : px[i * 3];
        d[i * 4 + 1] = q ? q[1] : px[i * 3 + 1];
        d[i * 4 + 2] = q ? q[2] : px[i * 3 + 2];
        d[i * 4 + 3] = 255;
      }
      g.putImageData(im, 0, 0);
    };
    const reset = () => {
      C = [];
      A = null;
      ph = 'seed';
      it = 0;
      hist = [];
      cu = 0;
      wait = 0;
      lastCh = 0;
      msg =
        'The picture has been read as ' +
        N.toLocaleString('en-US') +
        ' points in colour space. Press Run or One step to pick ' +
        S.K +
        ' starting colours.';
      if (out) out.getContext('2d').drawImage(src, 0, 0);
    };
    const step = () => {
      if (!px) return;
      if (ph === 'done') {
        run++;
        reset();
        return;
      }
      if (ph === 'seed') {
        const r = rng(40 + run * 11 + S.K),
          f = Math.floor(r() * N);
        C = [[px[f * 3], px[f * 3 + 1], px[f * 3 + 2]]];
        const D = new Float64Array(N);
        for (let i = 0; i < N; i++) D[i] = dist(i, C[0]);
        while (C.length < S.K) {
          let tot = 0;
          for (let i = 0; i < N; i++) tot += D[i];
          if (tot <= 0) break;
          let u = r() * tot,
            i = 0;
          while (i < N - 1 && (u -= D[i]) > 0) i++;
          const q = [px[i * 3], px[i * 3 + 1], px[i * 3 + 2]];
          C.push(q);
          for (let j = 0; j < N; j++) {
            const d = dist(j, q);
            if (d < D[j]) D[j] = d;
          }
        }
        A = null;
        ph = 'assign';
        cu = 1;
        msg =
          'k-means++ picked ' +
          C.length +
          ' starting colours from the pixels, each one likely to be far in colour from those already chosen. They are the rings in the colour cube.';
        return;
      }
      if (ph === 'assign') {
        cu = 2;
        if (!A) A = new Uint8Array(N).fill(255);
        let ch = 0,
          J = 0;
        for (let i = 0; i < N; i++) {
          const [b, bd] = nearC(i);
          if (A[i] !== b) {
            A[i] = b;
            ch++;
          }
          J += bd;
        }
        lastCh = ch;
        hist.push(Math.sqrt(J / N));
        paint();
        if ((ch === 0 && it > 0) || it >= 120) {
          ph = 'done';
          cu = 4;
          msg =
            (ch === 0
              ? 'Converged: no pixel changed colour, so the centres will not move again. It took ' +
                it +
                ' rounds.'
              : 'Reached the limit of 120 rounds with ' + ch + ' pixels still changing.') +
            ' Average colour error ' +
            uNum(Math.sqrt(J / N)) +
            ' on a 0 to 255 scale, using ' +
            C.length +
            ' colours in place of ' +
            uniq.toLocaleString('en-US') +
            '.';
        } else {
          ph = 'move';
          msg =
            'Assign: every pixel was repainted with the nearest of the ' +
            C.length +
            ' colours. ' +
            ch.toLocaleString('en-US') +
            ' pixels changed. Average colour error ' +
            uNum(Math.sqrt(J / N)) +
            '.';
        }
        return;
      }
      if (ph === 'move') {
        cu = 3;
        const Sx = C.map(() => [0, 0, 0, 0]);
        for (let i = 0; i < N; i++) {
          const s = Sx[A[i]];
          s[0] += px[i * 3];
          s[1] += px[i * 3 + 1];
          s[2] += px[i * 3 + 2];
          s[3]++;
        }
        let far = 0;
        C.forEach((q, k) => {
          const s = Sx[k];
          if (s[3]) {
            const nq = [s[0] / s[3], s[1] / s[3], s[2] / s[3]];
            far = Math.max(far, Math.hypot(nq[0] - q[0], nq[1] - q[1], nq[2] - q[2]));
            C[k] = nq;
          }
        });
        let J = 0;
        for (let i = 0; i < N; i++) J += dist(i, C[A[i]]);
        hist.push(Math.sqrt(J / N));
        it++;
        paint();
        ph = 'assign';
        msg =
          'Move: each of the ' +
          C.length +
          ' colours became the average of the pixels using it. The largest shift was ' +
          uNum(far) +
          ' colour units. Average colour error ' +
          uNum(Math.sqrt(J / N)) +
          '.';
      }
    };
    const rect = (w, h, fh) => {
      const s = Math.min(w / W, (h * fh) / H);
      return [(w - W * s) / 2, (h * fh - H * s) / 2, W * s, H * s];
    };
    const draw = () => {
      if (!px) return;
      const t = T(),
        hx = (q) =>
          'rgb(' + Math.round(q[0]) + ',' + Math.round(q[1]) + ',' + Math.round(q[2]) + ')';
      {
        const { g, w, h } = fit(co),
          R = rect(w, h, 0.88);
        g.imageSmoothingEnabled = false;
        g.drawImage(src, R[0], R[1], R[2], R[3]);
        if (pick >= 0) {
          const x = R[0] + (((pick % W) + 0.5) * R[2]) / W,
            y = R[1] + ((Math.floor(pick / W) + 0.5) * R[3]) / H;
          g.strokeStyle = '#fff';
          g.lineWidth = 3;
          g.beginPath();
          g.arc(x, y, 8, 0, 7);
          g.stroke();
          g.strokeStyle = '#000';
          g.lineWidth = 1.4;
          g.stroke();
        }
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          W + ' × ' + H + ' pixels, ' + uniq.toLocaleString('en-US') + ' different colours',
          R[0],
          h * 0.88 + 16,
        );
      }
      {
        const { g, w, h } = fit(cq),
          R = rect(w, h, 0.88);
        g.imageSmoothingEnabled = false;
        g.drawImage(out, R[0], R[1], R[2], R[3]);
        const K = C.length,
          sw = Math.min(34, R[2] / Math.max(1, K));
        C.forEach((q, k) => {
          g.fillStyle = hx(q);
          g.fillRect(R[0] + k * sw, h * 0.88 + 5, sw - 2, h * 0.12 - 8);
        });
        if (!K) {
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('no colours chosen yet', R[0], h * 0.88 + 16);
        }
      }
      {
        const { g, w, h } = fit(cc),
          a = (S.az * Math.PI) / 180,
          ca = Math.cos(a),
          sa = Math.sin(a),
          sc = Math.min(w * 0.62, h) * 0.5,
          pr = (r, gg, b) => {
            const x = r / 255 - 0.5,
              y = b / 255 - 0.5,
              z = gg / 255 - 0.5,
              xr = x * ca - y * sa,
              yr = x * sa + y * ca;
            return [w / 2 + sc * xr, h / 2 - sc * (z * 0.92 + yr * 0.36)];
          };
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        const cs = [0, 255];
        for (const i of cs)
          for (const j of cs) {
            [
              [
                [0, i, j],
                [255, i, j],
              ],
              [
                [i, 0, j],
                [i, 255, j],
              ],
              [
                [i, j, 0],
                [i, j, 255],
              ],
            ].forEach((e) => {
              const p = pr(...e[0]),
                q = pr(...e[1]);
              g.beginPath();
              g.moveTo(p[0], p[1]);
              g.lineTo(q[0], q[1]);
              g.stroke();
            });
          }
        g.font = MONO;
        g.textAlign = 'center';
        [
          ['black', [0, 0, 0]],
          ['red', [255, 0, 0]],
          ['green', [0, 255, 0]],
          ['blue', [0, 0, 255]],
          ['white', [255, 255, 255]],
        ].forEach((q) => {
          const p = pr(...q[1]);
          g.fillStyle = t.muted;
          g.fillText(q[0], p[0], p[1] + (q[1][1] ? -6 : 14));
        });
        for (const i of samp) {
          const q = A && A[i] < C.length ? C[A[i]] : [px[i * 3], px[i * 3 + 1], px[i * 3 + 2]],
            p = pr(px[i * 3], px[i * 3 + 1], px[i * 3 + 2]);
          g.fillStyle = hx(q);
          g.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3);
        }
        C.forEach((q) => {
          const p = pr(...q);
          g.fillStyle = hx(q);
          g.strokeStyle = t.bg;
          g.lineWidth = 5;
          g.beginPath();
          g.arc(p[0], p[1], 8, 0, 7);
          g.stroke();
          g.fill();
          g.strokeStyle = t.ink;
          g.lineWidth = 1.8;
          g.stroke();
        });
        if (pick >= 0) {
          const p = pr(px[pick * 3], px[pick * 3 + 1], px[pick * 3 + 2]);
          markTP(g, p[0], p[1], t, false);
        }
      }
      {
        const { g, w, h } = fit(ce);
        const L = uLine(g, t, 0, 0, w, h, hist, {
          title: 'average colour error after each assign and each move',
          dots: true,
          n: Math.max(8, hist.length),
          col: t.ok,
        });
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'right';
        g.fillText('step', w - L.m.r, h - 5);
      }
      live.textContent = msg;
      uCur(root, 'kp', cu);
      if (pick >= 0) {
        const q = [px[pick * 3], px[pick * 3 + 1], px[pick * 3 + 2]];
        tpo.textContent =
          'The pixel you clicked is ' +
          hx(q) +
          '.' +
          (C.length
            ? (() => {
                const [b, bd] = nearC(pick);
                return (
                  ' Its nearest palette colour is number ' +
                  (b + 1) +
                  ', ' +
                  hx(C[b]) +
                  ', ' +
                  uNum(Math.sqrt(bd)) +
                  ' colour units away, so it is repainted with that.'
                );
              })()
            : ' Press One step to choose the palette.');
      }
      const bb = Math.max(1, Math.ceil(Math.log2(Math.max(2, S.K))));
      setR(root, 'n', N.toLocaleString('en-US'));
      setR(root, 'uc', uniq.toLocaleString('en-US'));
      setR(root, 'k', C.length ? String(C.length) : '–');
      setR(root, 'bits', '24 → ' + bb + ' (' + ((100 * bb) / 24).toFixed(0) + '% of the size)');
      setR(root, 'err', hist.length ? uNum(hist[hist.length - 1]) : '–');
      setR(root, 'it', String(it));
    };
    load(builtin());
    for (let i = 0; i < 260 && ph !== 'done'; i++) step();
    const anim = Anim(() => {
      fr++;
      if (spin) {
        S.az = (S.az + 0.35) % 360;
        azb.set(S.az);
      }
      if (wait > 0) {
        wait--;
        if (!wait) {
          run++;
          reset();
        }
        draw();
        return;
      }
      if (fr % (it < 3 ? 34 : 9) === 0) {
        step();
        draw();
        chk();
        return;
      }
      draw();
    });
    pb = playBtn($('#kp-play', root), anim);
    const ps = pb.stop,
      pst = pb.start;
    pb.stop = () => {
      spin = false;
      ps();
    };
    pb.start = () => {
      spin = true;
      pst();
    };
    const chk = runCtl(pb, root, 'kp', {
      done: () => ph === 'done',
      restart: () => {
        run++;
        reset();
      },
      count: () => it,
      what: 'rounds',
    });
    $('#kp-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#kp-reset', root).onclick = () => {
      pb.stop();
      run++;
      reset();
      draw();
    };
    bind(
      root,
      'kp-K',
      (v) => {
        S.K = v;
        pb.stop();
        reset();
        for (let i = 0; i < 260 && ph !== 'done'; i++) step();
        draw();
      },
      (v) => String(v),
    );
    const azb = bind(
      root,
      'kp-az',
      (v) => {
        S.az = v;
        draw();
      },
      (v) => Math.round(v) + '°',
    );
    $('#kp-built', root).onclick = () => {
      pb.stop();
      st.textContent = '';
      load(builtin());
      draw();
    };
    $('#kp-file', root).onchange = (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (!f) return;
      const use = (im) => {
          const iw = im.width,
            ih = im.height;
          if (!iw || !ih) {
            st.textContent = 'That picture could not be read. Try a JPEG or PNG file.';
            return;
          }
          const s = Math.min(1, 160 / Math.max(iw, ih)),
            o = document.createElement('canvas');
          o.width = Math.max(8, Math.round(iw * s));
          o.height = Math.max(8, Math.round(ih * s));
          o.getContext('2d').drawImage(im, 0, 0, o.width, o.height);
          pb.stop();
          load(o);
          st.textContent =
            'Loaded ' +
            f.name +
            ', shrunk to ' +
            o.width +
            ' × ' +
            o.height +
            ' pixels so that every step stays fast. The picture stays in your browser.';
          draw();
        },
        bad = () => {
          st.textContent = 'That file could not be opened as a picture. Try a JPEG or PNG file.';
        };
      if (window.createImageBitmap) createImageBitmap(f).then(use, bad);
      else {
        const fr2 = new FileReader();
        fr2.onerror = bad;
        fr2.onload = () => {
          const im = new Image();
          im.onload = () => use(im);
          im.onerror = bad;
          im.src = fr2.result;
        };
        fr2.readAsDataURL(f);
      }
    };
    co.addEventListener('click', (e) => {
      const r = co.getBoundingClientRect(),
        R = rect(r.width, r.height, 0.88),
        x = Math.floor(((e.clientX - r.left - R[0]) / R[2]) * W),
        y = Math.floor(((e.clientY - r.top - R[1]) / R[3]) * H);
      if (x < 0 || y < 0 || x >= W || y >= H) return;
      pick = y * W + x;
      draw();
    });
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
