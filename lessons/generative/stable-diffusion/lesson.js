/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  SANS,
  T,
  bind,
  clamp,
  f2,
  f3,
  fit,
  letterData,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { UC, segBind, uCur, uLine } from '../../unsupervised/shared.js';
import { sig } from '../../classification/shared.js';

export default {
  init(root) {
    const cp = $('#sd-p', root),
      c = $('#sd-c', root),
      cr = $('#sd-r', root),
      live = $('#sd-live', root),
      S = { pr: 0, w: 3 },
      Tn = 40,
      VS = 700,
      DS = 2500,
      M = 8,
      R = 3,
      BE = Array.from({ length: Tn }, (_, i) => 0.002 + (0.198 * i) / (Tn - 1)),
      AB = [],
      RN = randnFrom(Math.random);
    {
      let p = 1;
      for (const b of BE) {
        p *= 1 - b;
        AB.push(p);
      }
    }
    let X,
      XF,
      lab,
      LT,
      enc,
      dec,
      net,
      vs = 0,
      ds = 0,
      LV = [],
      LD = [],
      Z = [],
      zsd = 1,
      CEN = [],
      zs = null,
      lv = Tn,
      path = [],
      phase = 'vae',
      done = false,
      msg = '',
      cu = 0,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      sm = [];
    const d0 = letterData(1, 0.04);
    X = d0.X.map((x) => x.map((v) => clamp(Math.round(v), 0, 1)));
    lab = d0.lab;
    LT = d0.CH;
    XF = new Float64Array(X.length * 35);
    X.forEach((x, i) =>
      x.forEach((v, j) => {
        XF[i * 35 + j] = v;
      }),
    );
    $('#sd-pr', root).innerHTML = LT.map(
      (ch, i) =>
        '<button class="btn" type="button" data-v="' + i + '">“the letter ' + ch + '”</button>',
    ).join('');
    const build = () => {
      const q = randnFrom(rng(run * 9 + 4));
      enc = AD.mlp([35, 24, 4], q, 'tanh');
      dec = AD.mlp([2, 24, 35], q, 'tanh');
      net = AD.mlp([17, 64, 64, 2], q, 'relu');
      vs = ds = 0;
      LV = [];
      LD = [];
      sm = [];
      Z = [];
      zs = null;
      path = [];
      lv = Tn;
      phase = 'vae';
      done = false;
      cu = 0;
      msg =
        'Nothing is trained yet. First the compressor will learn to squeeze each picture into two numbers. Press Run.';
    };
    const decode = (z) => {
      const o = dec.fwd(AD.input([z[0] * zsd, z[1] * zsd], [1, 2]));
      AD.clear();
      return Float64Array.from(o.v, sig);
    };
    const emb = (k) => {
      const o = [];
      for (let q = 0; q < 4; q++) {
        const f = ((k + 1) / Tn) * Math.PI * 2 ** q * 0.5;
        o.push(Math.sin(f), Math.cos(f));
      }
      return o;
    };
    const trainV = (k) => {
      const n = X.length;
      for (let q = 0; q < k && vs < VS; q++) {
        const h = enc.fwd(AD.input(XF, [n, 35])),
          mu = AD.cols(h, 0, 2),
          lg = AD.cols(h, 2, 4),
          z = AD.reparam(
            mu,
            lg,
            Float64Array.from({ length: 2 * n }, () => RN()),
          ),
          rec = AD.bce(dec.fwd(z), XF),
          loss = AD.wsum(rec, AD.kl(mu, lg, n), 1 / 35);
        LV.push(rec.v[0] * 35);
        AD.backward(loss);
        AD.adam(enc.ps.concat(dec.ps), 0.006, ++vs, 1);
      }
      const o = enc.fwd(AD.input(XF, [n, 35]));
      AD.clear();
      let s = 0;
      Z = [];
      for (let i = 0; i < n; i++) {
        Z.push([o.v[i * 4], o.v[i * 4 + 1]]);
        s += o.v[i * 4] ** 2 + o.v[i * 4 + 1] ** 2;
      }
      zsd = Math.sqrt(s / (2 * n)) || 1;
      Z = Z.map((z) => [z[0] / zsd, z[1] / zsd]);
      CEN = LT.map((_, k) => {
        const q = Z.filter((_, i) => lab[i] === k);
        return [
          q.reduce((a, z) => a + z[0], 0) / q.length,
          q.reduce((a, z) => a + z[1], 0) / q.length,
        ];
      });
      cu = 0;
      msg =
        'Stage 1, the compressor: epoch ' +
        vs +
        ' of ' +
        VS +
        '. Each picture is squeezed to 2 numbers and rebuilt; the error is ' +
        f2(LV[vs - 1]) +
        ' per picture. On the left, the pictures are sorting themselves by letter.';
      if (vs >= VS) {
        phase = 'den';
        cu = 1;
        msg =
          'The compressor is trained and frozen. Every picture now has a code of 2 numbers and a caption. Stage 2: train the denoiser on those codes.';
      }
    };
    const trainD = (k) => {
      const B = 128;
      for (let q = 0; q < k && ds < DS; q++) {
        const Xb = new Float64Array(B * 17),
          E = new Float64Array(B * 2);
        for (let i = 0; i < B; i++) {
          const j = Math.floor(Math.random() * Z.length),
            tt = Math.floor(Math.random() * Tn),
            e0 = RN(),
            e1 = RN(),
            s = Math.sqrt(AB[tt]),
            nq = Math.sqrt(1 - AB[tt]),
            e = emb(tt);
          Xb[i * 17] = s * Z[j][0] + nq * e0;
          Xb[i * 17 + 1] = s * Z[j][1] + nq * e1;
          for (let u = 0; u < 8; u++) Xb[i * 17 + 2 + u] = e[u];
          Xb[i * 17 + 10 + (Math.random() < 0.15 ? 6 : lab[j])] = 1;
          E[2 * i] = e0;
          E[2 * i + 1] = e1;
        }
        const l = AD.mse(net.fwd(AD.input(Xb, [B, 17])), E);
        LD.push(l.v[0]);
        AD.backward(l);
        AD.adam(net.ps, 0.002, ++ds, 1);
        if (ds % 20 === 0) {
          let s = 0;
          for (let j = ds - 20; j < ds; j++) s += LD[j];
          sm.push(s / 20);
        }
      }
      cu = 2;
      msg =
        'Stage 2, the denoiser: step ' +
        ds +
        ' of ' +
        DS +
        '. It sees a noisy code, the noise level and (usually) the caption, and guesses the noise. Error ' +
        f3(sm.length ? sm[sm.length - 1] : LD[ds - 1]) +
        '. It never sees a pixel.';
      if (ds >= DS) startGen();
    };
    const startGen = () => {
      phase = 'gen';
      zs = Float64Array.from({ length: 2 * M }, () => RN());
      lv = Tn;
      done = false;
      path = [[zs[0], zs[1]]];
      cu = 3;
      msg =
        'Stage 3, generating “the letter ' +
        LT[S.pr] +
        '” with guidance ' +
        f2(S.w) +
        '. Eight codes start as pure noise. Decoded, they are meaningless smudges.';
    };
    const eps = (k, cond) => {
      const Xb = new Float64Array(M * 17),
        e = emb(k);
      for (let i = 0; i < M; i++) {
        Xb[i * 17] = zs[2 * i];
        Xb[i * 17 + 1] = zs[2 * i + 1];
        for (let u = 0; u < 8; u++) Xb[i * 17 + 2 + u] = e[u];
        Xb[i * 17 + 10 + cond] = 1;
      }
      const o = net.fwd(AD.input(Xb, [M, 17]));
      AD.clear();
      return o.v;
    };
    const cls = (i) => {
        let b = 0,
          bd = 1e9;
        CEN.forEach((q, k) => {
          const d = (q[0] - zs[2 * i]) ** 2 + (q[1] - zs[2 * i + 1]) ** 2;
          if (d < bd) {
            bd = d;
            b = k;
          }
        });
        return b;
      },
      match = () => {
        let ok = 0;
        for (let i = 0; i < M; i++) if (cls(i) === S.pr) ok++;
        return ok;
      };
    const gstep = () => {
      if (lv <= 0) return;
      const k = lv - 1,
        eu = eps(k, 6),
        ec = eps(k, S.pr),
        b = BE[k],
        c1 = b / Math.sqrt(1 - AB[k]),
        s1 = 1 / Math.sqrt(1 - b),
        s2 = k > 0 ? Math.sqrt(b) : 0;
      for (let i = 0; i < 2 * M; i++) {
        const e = eu[i] + S.w * (ec[i] - eu[i]);
        zs[i] = s1 * (zs[i] - c1 * e) + s2 * RN();
      }
      lv--;
      path.push([zs[0], zs[1]]);
      cu = lv <= 0 ? 4 : 3;
      msg =
        'Generating, noise level ' +
        lv +
        ' of ' +
        Tn +
        ': the denoiser was asked twice about each code, once with the prompt “the letter ' +
        LT[S.pr] +
        '” and once with no prompt, and the difference was multiplied by ' +
        f2(S.w) +
        '.' +
        (lv <= 0
          ? ' Finished. The decoder turned the eight codes into pictures: ' +
            match() +
            ' of 8 match the prompt.'
          : '');
      if (lv <= 0) done = true;
    };
    const step = () => {
      if (phase === 'vae') trainV(20);
      else if (phase === 'den') trainD(50);
      else if (done) startGen();
      else gstep();
    };
    const glyph = (g, v, x, y, cz, t, al = 1) => {
      for (let k = 0; k < 35; k++) {
        g.fillStyle = mix(t.bg, t.ink, clamp(v[k], 0, 1), al);
        g.fillRect(x + (k % 5) * cz, y + Math.floor(k / 5) * cz, cz - 0.4, cz - 0.4);
      }
    };
    const draw = () => {
      const t = T(),
        col = UC(t);
      {
        const { g, w, h } = fit(cp),
          bx = [
            ['prompt', '“the letter ' + LT[S.pr] + '”'],
            ['text encoder', 'caption → numbers'],
            ['denoiser', 'in latent space, × ' + Tn],
            ['decoder', 'latent → pixels'],
            ['picture', '5 × 7 pixels'],
          ],
          act = phase === 'vae' ? 3 : phase === 'den' ? 2 : done ? 4 : 2,
          bw = w / 5.6,
          by = h * 0.2,
          bh = h * 0.6;
        g.font = MONO;
        bx.forEach((q, i) => {
          const x = w * 0.02 + (i * (w * 0.96 - bw)) / 4,
            on = i === act || (phase === 'gen' && !done && i <= 2 && i === 2);
          g.fillStyle = i === act ? t.gold : t.bg;
          g.strokeStyle = t.ink;
          g.lineWidth = 1.5;
          g.beginPath();
          g.rect(x, by, bw, bh);
          g.fill();
          g.stroke();
          g.font = SANS;
          g.fillStyle = i === act ? t.bg : t.ink;
          g.textAlign = 'center';
          g.fillText(q[0], x + bw / 2, by + bh * 0.42);
          g.font = MONO;
          g.fillStyle = i === act ? t.bg : t.muted;
          g.fillText(q[1], x + bw / 2, by + bh * 0.74);
          if (i < 4) {
            const xa = x + bw + 4,
              xb = w * 0.02 + ((i + 1) * (w * 0.96 - bw)) / 4 - 4,
              ym = by + bh / 2;
            g.strokeStyle = t.muted;
            g.lineWidth = 1.4;
            g.beginPath();
            g.moveTo(xa, ym);
            g.lineTo(xb, ym);
            g.lineTo(xb - 6, ym - 4);
            g.moveTo(xb, ym);
            g.lineTo(xb - 6, ym + 4);
            g.stroke();
            if (phase === 'gen' && pb && pb.on) {
              g.fillStyle = t.gold;
              g.beginPath();
              g.arc(xa + (xb - xa) * ((fr / 40 + i * 0.25) % 1), ym, 4, 0, 7);
              g.fill();
            }
          }
        });
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          phase === 'vae'
            ? 'now training: the compressor (encoder and decoder)'
            : phase === 'den'
              ? 'now training: the denoiser, on codes only'
              : 'now generating',
          w * 0.02,
          h - 4,
        );
        g.textAlign = 'right';
        g.fillText(
          'real Stable Diffusion: CLIP text encoder → U-Net → VAE decoder',
          w * 0.98,
          h - 4,
        );
      }
      {
        const { g, w, h } = fit(c),
          sc = (Math.min(w, h) / 2 / R) * 0.96,
          X2 = (v) => w / 2 + v * sc,
          Y2 = (v) => h / 2 - v * sc;
        if (vs > 0) {
          const NG = 9;
          for (let a = 0; a < NG; a++)
            for (let b = 0; b < NG; b++) {
              const zx = -R + (2 * R * (b + 0.5)) / NG,
                zy = R - (2 * R * (a + 0.5)) / NG;
              glyph(g, decode([zx, zy]), X2(zx) - 9, Y2(zy) - 12, 3.6, t, 0.18);
            }
        }
        Z.forEach((z, i) => {
          g.font = '600 12px "Hanken Grotesk",sans-serif';
          g.fillStyle = mix(
            col[lab[i] % 8],
            col[lab[i] % 8],
            0,
            phase === 'gen' && lab[i] !== S.pr ? 0.3 : 0.9,
          );
          g.textAlign = 'center';
          g.fillText(LT[lab[i]], X2(z[0]), Y2(z[1]) + 4);
        });
        if (zs) {
          g.strokeStyle = t.gold;
          g.lineWidth = 1.6;
          g.beginPath();
          path.forEach((p, i) => (i ? g.lineTo(X2(p[0]), Y2(p[1])) : g.moveTo(X2(p[0]), Y2(p[1]))));
          g.stroke();
          for (let i = 0; i < M; i++) {
            const x = X2(clamp(zs[2 * i], -R, R)),
              y = Y2(clamp(zs[2 * i + 1], -R, R));
            g.fillStyle = t.gold;
            g.strokeStyle = t.ink;
            g.lineWidth = i ? 1 : 2.2;
            g.beginPath();
            g.arc(x, y, i ? 5 : 7, 0, 7);
            g.fill();
            g.stroke();
          }
        }
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          vs
            ? 'faint pictures: what each region of the latent space decodes to'
            : 'the latent space is empty until the compressor is trained',
          8,
          h - 8,
        );
      }
      {
        const { g, w, h } = fit(cr),
          cz = Math.min((w - 30) / M / 6.4, (h * 0.34) / 7.6),
          gw = cz * 6.4,
          x0 = (w - gw * M) / 2 + cz * 0.5;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          zs
            ? 'the eight codes, decoded' + (done ? '' : ' (noise level ' + lv + ')')
            : 'pictures appear here when generation starts',
          8,
          13,
        );
        if (zs)
          for (let i = 0; i < M; i++) {
            glyph(g, decode([zs[2 * i], zs[2 * i + 1]]), x0 + i * gw, 24, cz, t);
            const ok = done && cls(i) === S.pr;
            g.strokeStyle = ok ? t.gold : t.line;
            g.lineWidth = ok ? 2.6 : 1;
            g.strokeRect(x0 + i * gw - 2, 22, 5 * cz + 3, 7 * cz + 3);
            if (done) {
              g.fillStyle = ok ? t.cut : t.muted;
              g.textAlign = 'center';
              g.fillText(LT[cls(i)], x0 + i * gw + 2.5 * cz, 24 + 7 * cz + 16);
            }
          }
        const y3 = 24 + 7 * cz + 30;
        if (phase === 'vae')
          uLine(g, t, 0, y3, w, h - y3, LV, {
            title: 'compressor: reconstruction error per picture',
            n: VS,
            col: t.ok,
            ym: Math.max(24, LV[0] || 24),
          });
        else
          uLine(g, t, 0, y3, w, h - y3, sm, {
            title: 'denoiser: error at guessing the noise in a code',
            n: DS / 20,
            col: t.ok,
            ym: Math.max(1.05, sm[0] || 1.05),
          });
      }
      live.textContent = msg;
      uCur(root, 'sd', cu);
      setR(
        root,
        'ph',
        phase === 'vae'
          ? vs
            ? '1: training the compressor'
            : 'Not started'
          : phase === 'den'
            ? '2: training the denoiser'
            : done
              ? 'Finished'
              : '3: generating',
      );
      setR(root, 'pr', '“the letter ' + LT[S.pr] + '”');
      setR(root, 'lv', phase === 'gen' ? lv + ' of ' + Tn : '–');
      setR(root, 'ok', phase === 'gen' && done ? match() + ' of 8' : '–');
      setR(root, 'cmp', '2 (the picture has 35)');
      setR(
        root,
        'L',
        phase === 'vae' ? (vs ? f2(LV[vs - 1]) : '–') : sm.length ? f3(sm[sm.length - 1]) : '–',
      );
    };
    build();
    if (RM) {
      trainV(VS);
      trainD(DS);
      while (!done) gstep();
    }
    const anim = Anim(() => {
      fr++;
      if (phase === 'vae') {
        trainV(vs < 20 ? 1 : 3);
        draw();
        return;
      }
      if (phase === 'den') {
        trainD(Math.min(ds < 40 ? 1 : 6, chk.remaining()));
        draw();
        chk();
        return;
      }
      if (fr % (lv > Tn - 5 ? 24 : 8) === 0 && !done) gstep();
      draw();
      if (done) pb.stop();
    });
    pb = playBtn($('#sd-play', root), anim);
    chk = runCtl(pb, root, 'sd', {
      done: () => phase === 'gen' && done,
      restart: () => startGen(),
      count: () => ds,
      what: 'training steps',
    });
    $('#sd-step', root).onclick = () => {
      pb.stop();
      step();
      draw();
    };
    $('#sd-reset', root).onclick = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    const regen = () => {
      if (phase !== 'gen') {
        draw();
        return;
      }
      pb.stop();
      startGen();
      draw();
      pb.start();
    };
    $('#sd-gen', root).onclick = regen;
    segBind(root, 'sd-pr', String(S.pr), (v) => {
      S.pr = +v;
      regen();
    });
    bind(
      root,
      'sd-w',
      (v) => {
        S.w = v;
        if (phase === 'gen' && !pb.on) {
          startGen();
          draw();
          pb.start();
        }
      },
      f2,
    );
    return {
      draw,
      stop: () => pb.stop(),
      auto: () => {
        run++;
        build();
        draw();
        pb.start();
      },
    };
  },
};
