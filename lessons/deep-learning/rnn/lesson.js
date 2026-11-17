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
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#rn-c', root),
      cr = $('#rn-r', root),
      live = $('#rn-live', root),
      outEl = $('#rn-out', root),
      st = $('#rn-st', root),
      settings = { text: 'hello world', H: 12 },
      EPM = 2500;
    let voc = [],
      seq = [],
      V = 0,
      n = 0,
      P,
      M,
      ep = 0,
      LL = [],
      trained = false,
      generatedCharIds = [],
      hs = [],
      probs = [],
      done = false,
      msg = '',
      activeStepIndex = 0,
      frameCount = 0,
      t0 = 0,
      playback,
      checkRunStop = () => false,
      run = 1,
      ts = 0;
    const mat = (r, cn, sd, rn) =>
        Array.from({ length: r }, () => Float64Array.from({ length: cn }, () => sd * rn())),
      zl = (A) => A.map((a) => new Float64Array(a.length));
    const build = () => {
      const tx =
        (settings.text
          .toLowerCase()
          .replace(/[^a-z ]/g, '')
          .replace(/\s+/g, ' ')
          .trim() || 'hello world') + ' ';
      voc = [...new Set(tx)].sort();
      V = voc.length;
      seq = [...tx].map((ch) => voc.indexOf(ch));
      n = seq.length;
      const rn = randnFrom(rng(run * 7 + 3)),
        H = settings.H;
      P = {
        Wxh: mat(H, V, 0.4, rn),
        Whh: mat(H, H, 0.4 / Math.sqrt(H), rn),
        bh: [new Float64Array(H)],
        Why: mat(V, H, 0.4, rn),
        by: [new Float64Array(V)],
      };
      M = {};
      for (const k in P) M[k] = [zl(P[k]), zl(P[k])];
      ep = 0;
      LL = [];
      trained = false;
      generatedCharIds = [];
      hs = [];
      probs = [];
      done = false;
      ts = 0;
      activeStepIndex = 0;
      msg =
        'Stage 1, learning. The phrase is “' +
        tx +
        '” (' +
        n +
        ' characters, counting the space that ends it, using ' +
        V +
        ' different ones). The network has random weights and has read nothing yet.';
      outEl.textContent = 'What it writes now: (nothing yet)';
    };
    const cell = (x, hp) => {
      const H = settings.H,
        h = new Float64Array(H);
      for (let j = 0; j < H; j++) {
        let s = P.bh[0][j] + P.Wxh[j][x];
        for (let i = 0; i < H; i++) s += P.Whh[j][i] * hp[i];
        h[j] = Math.tanh(s);
      }
      const y = new Float64Array(V);
      let mx = -1e9;
      for (let k = 0; k < V; k++) {
        let s = P.by[0][k];
        for (let j = 0; j < H; j++) s += P.Why[k][j] * h[j];
        y[k] = s;
        mx = Math.max(mx, s);
      }
      let z = 0;
      for (let k = 0; k < V; k++) {
        y[k] = Math.exp(y[k] - mx);
        z += y[k];
      }
      for (let k = 0; k < V; k++) y[k] /= z;
      return { h, y };
    };
    const write = (m) => {
      let h = new Float64Array(settings.H),
        x = seq[0],
        s = voc[x];
      for (let t = 1; t < m; t++) {
        const q = cell(x, h);
        h = q.h;
        x = q.y.indexOf(Math.max(...q.y));
        s += voc[x];
      }
      return s;
    };
    const epoch = () => {
      const H = settings.H,
        G = {};
      for (const k in P) G[k] = zl(P[k]);
      const Hs = [new Float64Array(H)],
        Ys = [];
      let loss = 0;
      for (let t = 0; t < n; t++) {
        const q = cell(seq[t], Hs[t]);
        Hs.push(q.h);
        Ys.push(q.y);
        loss -= Math.log(q.y[seq[(t + 1) % n]] + 1e-12);
      }
      let dn = new Float64Array(H);
      for (let t = n - 1; t >= 0; t--) {
        const dy = Float64Array.from(Ys[t]);
        dy[seq[(t + 1) % n]] -= 1;
        const h = Hs[t + 1],
          dh = Float64Array.from(dn);
        for (let k = 0; k < V; k++) {
          G.by[0][k] += dy[k];
          for (let j = 0; j < H; j++) {
            G.Why[k][j] += dy[k] * h[j];
            dh[j] += P.Why[k][j] * dy[k];
          }
        }
        const dz = new Float64Array(H);
        dn = new Float64Array(H);
        for (let j = 0; j < H; j++) {
          dz[j] = dh[j] * (1 - h[j] * h[j]);
          G.bh[0][j] += dz[j];
          G.Wxh[j][seq[t]] += dz[j];
          for (let i = 0; i < H; i++) {
            G.Whh[j][i] += dz[j] * Hs[t][i];
            dn[i] += P.Whh[j][i] * dz[j];
          }
        }
      }
      ts++;
      const c1 = 1 - Math.pow(0.9, ts),
        c2 = 1 - Math.pow(0.999, ts);
      for (const k in P)
        P[k].forEach((row, r) => {
          const g = G[k][r],
            m = M[k][0][r],
            v = M[k][1][r];
          for (let i = 0; i < row.length; i++) {
            const gi = clamp(g[i] / n, -1, 1);
            m[i] = 0.9 * m[i] + 0.1 * gi;
            v[i] = 0.999 * v[i] + 0.001 * gi * gi;
            row[i] -= (0.008 * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
          }
        });
      ep++;
      LL.push(loss / n);
    };
    const countCorrectNextChars = () => {
      let h = new Float64Array(settings.H),
        ok = 0;
      for (let t = 0; t < n; t++) {
        const q = cell(seq[t], h);
        h = q.h;
        if (q.y.indexOf(Math.max(...q.y)) === seq[(t + 1) % n]) ok++;
      }
      return ok;
    };
    const learn = (k) => {
      for (let i = 0; i < k && !trained; i++) {
        epoch();
        const L = LL[ep - 1];
        if (L < 0.03 || ep >= EPM || (ep > 400 && Math.abs(LL[ep - 1] - LL[ep - 201]) < 0.002 * L))
          trained = true;
      }
      const w = write(n);
      outEl.textContent = 'What it writes now: “' + w + '”';
      activeStepIndex = 0;
      msg =
        'Stage 1, learning. It has read the phrase ' +
        ep +
        ' times. Error per letter: ' +
        f3(LL[ep - 1]) +
        '. It gets ' +
        countCorrectNextChars() +
        ' of the ' +
        n +
        ' next letters right.' +
        (trained
          ? LL[ep - 1] < 0.03
            ? ' Converged: the error is close to zero, so learning stops. Now it will write the phrase from memory.'
            : ' The error has stopped falling' +
              (countCorrectNextChars() < n
                ? ', and it still makes mistakes: this memory is too small for this phrase.'
                : '.') +
              ' Now it will write from memory.'
          : '');
    };
    const generateNextChar = () => {
      if (!generatedCharIds.length) {
        generatedCharIds = [seq[0]];
        hs = [new Float64Array(settings.H)];
        probs = [];
        activeStepIndex = 1;
        msg =
          'Stage 2, writing. It is given only the first letter, “' +
          voc[seq[0]] +
          '”, and an empty memory.';
        t0 = frameCount;
        return;
      }
      const x = generatedCharIds[generatedCharIds.length - 1],
        q = cell(x, hs[hs.length - 1]),
        nx = q.y.indexOf(Math.max(...q.y));
      hs.push(q.h);
      probs.push(q.y);
      generatedCharIds.push(nx);
      t0 = frameCount;
      activeStepIndex = generatedCharIds.length < 4 ? 2 + ((generatedCharIds.length - 2) % 3) : 4;
      const srt = [...q.y].map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
      msg =
        'Step ' +
        (generatedCharIds.length - 1) +
        ': read “' +
        voc[x] +
        '”, mixed it with the memory of “' +
        generatedCharIds
          .slice(0, -2)
          .map((i) => voc[i])
          .join('') +
        '”, and from the new memory judged the next letter: “' +
        voc[srt[0][1]] +
        '” ' +
        (100 * srt[0][0]).toFixed(0) +
        '%' +
        (srt[1] ? ', “' + voc[srt[1][1]] + '” ' + (100 * srt[1][0]).toFixed(0) + '%' : '') +
        '. It writes “' +
        voc[nx] +
        '”' +
        (nx === seq[(generatedCharIds.length - 1) % n]
          ? '.'
          : ', which is wrong: the phrase has “' +
            voc[seq[(generatedCharIds.length - 1) % n]] +
            '” here.');
      outEl.textContent = 'Written so far: “' + generatedCharIds.map((i) => voc[i]).join('') + '”';
      if (generatedCharIds.length >= 2 * n) {
        done = true;
        t0 = frameCount - 200;
        msg += ' It has written the phrase twice over, so it stops.';
      }
    };
    const step = () => {
      if (done) {
        run++;
        build();
        return;
      }
      if (!trained) learn(10);
      else generateNextChar();
    };
    const draw = () => {
      const colors = readThemeColors(),
        u = prefersReducedMotion ? 1 : clamp((frameCount - t0) / 40, 0, 1);
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          NC = Math.min(9, Math.max(1, generatedCharIds.length)),
          from = Math.max(0, generatedCharIds.length - NC),
          cw = (w - 20) / 9,
          H = settings.H,
          yI = h - 34,
          yH0 = h * 0.3,
          hh = h * 0.36,
          ch = hh / H,
          yO = h * 0.13;
        ctx.font = MONO;
        if (!generatedCharIds.length) {
          ctx.fillStyle = colors.muted;
          ctx.textAlign = 'center';
          ctx.fillText(
            trained
              ? 'ready to write: press Run or One step'
              : 'learning… the writing starts when the error stops falling',
            w / 2,
            h / 2,
          );
        }
        for (let q = 0; q < NC && generatedCharIds.length; q++) {
          const tI = from + q,
            x = 10 + q * cw + cw / 2,
            cur = tI === generatedCharIds.length - 1,
            hv = hs[tI + 1];
          ctx.fillStyle = cur ? colors.gold : colors.bg;
          ctx.strokeStyle = colors.ink;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.rect(x - 15, yI - 15, 30, 30);
          ctx.fill();
          ctx.stroke();
          ctx.font = SANS;
          ctx.fillStyle = cur ? colors.bg : colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText(
            voc[generatedCharIds[tI]] === ' ' ? '␣' : voc[generatedCharIds[tI]],
            x,
            yI + 5,
          );
          ctx.font = MONO;
          if (hv) {
            const al = tI === generatedCharIds.length - 2 ? u : 1;
            ctx.strokeStyle = colors.muted;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(x, yI - 16);
            ctx.lineTo(x, yH0 + hh + 3);
            ctx.moveTo(x, yH0 - 3);
            ctx.lineTo(x, yO + 16);
            ctx.stroke();
            for (let j = 0; j < H; j++) {
              const v = hv[j];
              ctx.fillStyle =
                v >= 0
                  ? mix(colors.bg, colors.gold, v * al)
                  : mix(colors.bg, colors.accent, -v * al);
              ctx.fillRect(x - 13, yH0 + j * ch, 26, Math.max(1, ch - 0.7));
            }
            ctx.strokeStyle = colors.ink;
            ctx.lineWidth = 1.2;
            ctx.strokeRect(x - 13, yH0, 26, hh);
            if (q < NC - 1 || true) {
              ctx.strokeStyle = colors.ink;
              ctx.lineWidth = 1.6;
              ctx.beginPath();
              ctx.moveTo(x + 14, yH0 + hh / 2);
              ctx.lineTo(x + cw - 14, yH0 + hh / 2);
              ctx.stroke();
              ctx.beginPath();
              ctx.moveTo(x + cw - 14, yH0 + hh / 2);
              ctx.lineTo(x + cw - 20, yH0 + hh / 2 - 4);
              ctx.lineTo(x + cw - 20, yH0 + hh / 2 + 4);
              ctx.fill();
            }
            if (al > 0.7) {
              const pr = probs[tI],
                b = generatedCharIds[tI + 1];
              ctx.fillStyle = colors.bg;
              ctx.strokeStyle = colors.ok;
              ctx.lineWidth = 1.4;
              ctx.beginPath();
              ctx.rect(x - 15, yO - 15, 30, 30);
              ctx.fill();
              ctx.stroke();
              ctx.font = SANS;
              ctx.fillStyle = colors.ok;
              ctx.fillText(voc[b] === ' ' ? '␣' : voc[b], x, yO + 5);
              ctx.font = MONO;
              ctx.fillStyle = colors.muted;
              ctx.fillText((100 * pr[b]).toFixed(0) + '%', x, yO - 20);
            }
            if (tI === generatedCharIds.length - 2 && u < 1) {
              ctx.fillStyle = colors.gold;
              ctx.beginPath();
              ctx.arc(x, yI - 16 - (yI - 16 - yH0 - hh) * Math.min(1, u * 2), 5, 0, 7);
              ctx.fill();
              ctx.beginPath();
              ctx.arc(x - cw + 14 + (cw - 28) * Math.min(1, u * 2), yH0 + hh / 2, 5, 0, 7);
              ctx.fill();
            }
          }
        }
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('expects next', 4, yO - 30 < 10 ? 10 : yO - 30);
        ctx.fillText('memory', 4, yH0 - 6);
        ctx.fillText('reads', 4, h - 6);
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.5,
          pr = probs.length ? probs[probs.length - 1] : null,
          bw = (w - 30) / Math.max(1, V);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          pr
            ? 'after reading “' +
                voc[generatedCharIds[generatedCharIds.length - 2]] +
                '”: chance of each letter coming next'
            : 'chance of each letter coming next (appears while writing)',
          8,
          13,
        );
        for (let k = 0; k < V; k++) {
          const x = 15 + k * bw,
            p = pr ? pr[k] * u : 0,
            top = h1 - 24 - p * (h1 - 52);
          ctx.fillStyle =
            pr && k === generatedCharIds[generatedCharIds.length - 1] ? colors.gold : colors.accent;
          ctx.fillRect(x + bw * 0.15, top, bw * 0.7, h1 - 24 - top);
          ctx.font = SANS;
          ctx.fillStyle = colors.ink;
          ctx.textAlign = 'center';
          ctx.fillText(voc[k] === ' ' ? '␣' : voc[k], x + bw / 2, h1 - 8);
          ctx.font = MONO;
          if (p > 0.04) {
            ctx.fillStyle = colors.muted;
            ctx.fillText((100 * p).toFixed(0), x + bw / 2, top - 4);
          }
        }
        drawSeriesChart(ctx, colors, 0, h1, w, h - h1, LL, {
          title: 'error per letter, by number of readings',
          n: Math.max(100, LL.length),
          col: colors.ok,
          ym: Math.max(1, LL[0] || 1),
        });
      }
      live.textContent = msg;
      highlightLessonStep(root, 'rn', activeStepIndex);
      setReadout(
        root,
        'ph',
        done ? 'Finished' : trained ? 'Writing' : ep ? 'Learning' : 'Not started',
      );
      setReadout(root, 'ep', String(ep));
      setReadout(root, 'L', ep ? f3(LL[ep - 1]) : '–');
      setReadout(root, 'acc', ep ? countCorrectNextChars() + ' of ' + n : '–');
      setReadout(
        root,
        'np',
        String(settings.H * V + settings.H * settings.H + settings.H + V * settings.H + V),
      );
    };
    build();
    if (prefersReducedMotion) {
      learn(EPM);
      while (!done) generateNextChar();
    }
    const anim = Anim(() => {
      frameCount++;
      if (!trained) {
        if (frameCount % (ep < 40 ? 4 : 2) === 0) {
          learn(Math.min(ep < 40 ? 1 : ep < 200 ? 2 : 5, checkRunStop.remaining()));
          draw();
          if (checkRunStop()) return;
        }
        return;
      }
      if (frameCount - t0 > (generatedCharIds.length < 5 ? 95 : 46)) {
        generateNextChar();
        if (done) {
          draw();
          playback.stop();
          return;
        }
      }
      draw();
    });
    playback = playBtn($('#rn-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'rn', {
      done: () => done,
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'readings',
    });
    $('#rn-step', root).onclick = () => {
      playback.stop();
      step();
      t0 = frameCount - 100;
      draw();
    };
    $('#rn-reset', root).onclick = () => {
      playback.stop();
      run++;
      build();
      draw();
    };
    $('#rn-text', root).addEventListener('change', (e) => {
      settings.text = e.target.value;
      playback.stop();
      run++;
      build();
      st.textContent = /[^a-zA-Z ]/.test(e.target.value)
        ? 'Only letters and spaces are used; other characters were dropped.'
        : '';
      draw();
      playback.start();
    });
    bind(
      root,
      'rn-H',
      (v) => {
        settings.H = v;
        playback.stop();
        run++;
        build();
        draw();
      },
      (v) => String(v),
    );
    return {
      draw,
      stop: () => playback.stop(),
      auto: () => {
        run++;
        build();
        draw();
        playback.start();
      },
    };
  },
};
