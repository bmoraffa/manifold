/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  RM,
  T,
  clamp,
  eqShow,
  f2,
  f3,
  fit,
  mix,
  playBtn,
  randnFrom,
  rng,
  runCtl,
  setR,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { gwArrow } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, uCur, uLine } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const c = $('#aq-c', root),
      cr = $('#aq-r', root),
      live = $('#aq-live', root),
      eq = $('#aq-eq', root),
      S = { m: 'dqn' },
      CAP = 200,
      MAXE = 400,
      GA = 0.99,
      rn0 = randnFrom(Math.random);
    let net,
      tgt,
      actor,
      critic,
      buf,
      steps,
      tr,
      ta,
      ep,
      LEN,
      st,
      n,
      phase,
      msg,
      cu,
      fr = 0,
      pb,
      chk = () => false,
      run = 1,
      lastA = 0,
      ES,
      EA,
      kick = 0,
      demoN = 0,
      exL = null,
      pmap = null,
      pmAt = -1;
    const reset = () => [0, 1, 2, 3].map(() => (Math.random() - 0.5) * 0.1);
    const phys = (s, a, extra = 0) => {
      const g = 9.8,
        mc = 1,
        mp = 0.1,
        l = 0.5,
        F = a ? 10 : -10,
        dt = 0.02,
        [x, xd, th, thd0] = s,
        thd = thd0 + extra,
        ct = Math.cos(th),
        sn = Math.sin(th),
        tmp = (F + mp * l * thd * thd * sn) / (mc + mp),
        tha = (g * sn - ct * tmp) / (l * (4 / 3 - (mp * ct * ct) / (mc + mp))),
        xa = tmp - (mp * l * tha * ct) / (mc + mp),
        q = [x + dt * xd, xd + dt * xa, th + dt * thd, thd + dt * tha];
      return { s: q, done: Math.abs(q[0]) > 2.4 || Math.abs(q[2]) > 0.2095 };
    };
    const feat = (s) => [s[0] / 2.4, s[1] / 2, s[2] / 0.21, s[3] / 2];
    const qv = (f) => {
      const o = net.fwd(AD.input(f, [1, 4]));
      AD.clear();
      return o.v;
    };
    const p1 = (f) => {
      const o = actor.fwd(AD.input(f, [1, 4]));
      AD.clear();
      return 1 / (1 + Math.exp(o.v[0] - o.v[1]));
    };
    const eps = () => Math.max(0.05, 1 - steps / 2500);
    const greedy = (f) =>
      S.m === 'dqn'
        ? (() => {
            const q = qv(f);
            return q[1] > q[0] ? 1 : 0;
          })()
        : p1(f) > 0.5
          ? 1
          : 0;
    const build = () => {
      const rn = randnFrom(rng(run * 7 + 1));
      net = AD.mlp([4, 32, 32, 2], rn, 'relu');
      tgt = AD.mlp([4, 32, 32, 2], rn, 'relu');
      net.ps.forEach((p, i) => tgt.ps[i].v.set(p.v));
      actor = AD.mlp([4, 32, 2], rn, 'tanh');
      critic = AD.mlp([4, 32, 1], rn, 'tanh');
      buf = [];
      steps = 0;
      tr = 0;
      ta = 0;
      ep = 0;
      LEN = [];
      st = reset();
      n = 0;
      phase = 'train';
      cu = 0;
      ES = [];
      EA = [];
      kick = 0;
      exL = null;
      pmAt = -1;
      msg = 'A network with random weights is in control. It has never seen a pole. Press Run.';
      eqShow(eq, ['No training yet.'], 0);
    };
    const solved = () => LEN.length >= 20 && LEN.slice(-20).reduce((a, b) => a + b, 0) / 20 >= 180;
    const learnDQN = () => {
      const B = 32,
        X = new Float64Array(B * 4),
        X2 = new Float64Array(B * 4),
        bs = [];
      for (let i = 0; i < B; i++) {
        const e = buf[Math.floor(Math.random() * buf.length)];
        bs.push(e);
        X.set(e[0], i * 4);
        X2.set(e[3], i * 4);
      }
      const q2 = tgt.fwd(AD.input(X2, [B, 4]));
      AD.clear();
      const out = net.fwd(AD.input(X, [B, 4])),
        Y = Float64Array.from(out.v);
      for (let i = 0; i < B; i++) {
        const e = bs[i],
          mxq = Math.max(q2.v[2 * i], q2.v[2 * i + 1]),
          tg = e[4] ? -1 : 1 + GA * mxq,
          pr = out.v[i * 2 + e[1]];
        if (i === 0)
          exL = [
            'A step drawn from the memory: the pole leaned ' +
              f2(e[0][2] * 12) +
              '° and the push was to the ' +
              (e[1] ? 'right' : 'left') +
              '.',
            e[4]
              ? 'The pole fell on that step, so the episode ended there:  target = <b>−1</b>'
              : 'The pole stayed up: reward 1. The target copy rates the next situation at max(' +
                f2(q2.v[0]) +
                ', ' +
                f2(q2.v[1]) +
                ') = ' +
                f2(mxq),
            e[4]
              ? ''
              : 'target = reward + γ × best next = 1 + 0.99 × ' +
                f2(mxq) +
                ' = <b>' +
                f2(tg) +
                '</b>',
            'the network had predicted ' +
              f2(pr) +
              ' for that push, so its weights are nudged to move the prediction ' +
              (tg > pr ? 'up' : 'down') +
              ' by a little.',
            'The same is done for 31 other remembered steps at once.',
          ].filter((x) => x);
        Y[i * 2 + e[1]] = pr + clamp(tg - pr, -2, 2);
      }
      AD.backward(AD.mse(out, Y));
      AD.adam(net.ps, 0.002, ++tr, 1);
      if (tr % 150 === 0) net.ps.forEach((p, i) => tgt.ps[i].v.set(p.v));
    };
    const learnAC = (term) => {
      const B = EA.length;
      let Gt = 0;
      if (!term) {
        const v = critic.fwd(AD.input(feat(st), [1, 4]));
        AD.clear();
        Gt = v.v[0];
      }
      const Gs = new Float64Array(B);
      for (let i = B - 1; i >= 0; i--) {
        Gt = 0.05 + GA * Gt;
        Gs[i] = Gt;
      }
      const X = Float64Array.from(ES.flat()),
        v = critic.fwd(AD.input(X, [B, 4])),
        adv = new Float64Array(B);
      for (let i = 0; i < B; i++) adv[i] = Gs[i] - v.v[i];
      const k = Math.floor(B / 2),
        pk = p1(ES[k]);
      AD.clear();
      const v2 = critic.fwd(AD.input(X, [B, 4]));
      AD.backward(AD.mse(v2, Gs));
      AD.adam(critic.ps, 0.01, ++ta, 1);
      let sd = 0,
        mn = 0;
      for (const x of adv) mn += x / B;
      for (const x of adv) sd += (x - mn) ** 2 / B;
      sd = Math.sqrt(sd) + 1e-6;
      const raw = adv[k];
      for (let i = 0; i < B; i++) adv[i] /= sd;
      AD.backward(AD.pgl(actor.fwd(AD.input(X, [B, 4])), EA, adv));
      AD.adam(actor.ps, 0.01, ta, 1);
      const pk2 = p1(ES[k]),
        pa = EA[k] ? pk : 1 - pk,
        pa2 = EA[k] ? pk2 : 1 - pk2;
      exL = [
        'The episode lasted ' +
          B +
          ' steps. Take step ' +
          (k + 1) +
          ' as an example: the pole leaned ' +
          f2(ES[k][2] * 12) +
          '° and the push was to the ' +
          (EA[k] ? 'right' : 'left') +
          '.',
        'steps the pole stayed up after that, discounted and scaled: G = ' +
          f3(Gs[k]) +
          '     the critic had predicted ' +
          f3(v.v[k]),
        'advantage = G − prediction = <b>' +
          f3(raw) +
          '</b>: ' +
          (raw > 0 ? 'better' : 'worse') +
          ' than the critic expected',
        'so this step votes to make that push ' +
          (raw > 0 ? 'more' : 'less') +
          ' likely in that situation (its probability was ' +
          pct(pa) +
          ').',
        'All ' +
          B +
          ' steps of the episode vote together in one update of the actor. The critic then moves its predictions toward what actually happened.',
      ];
    };
    const envStep = () => {
      const f = feat(st);
      let a;
      if (S.m === 'dqn') {
        a = Math.random() < eps() ? (Math.random() < 0.5 ? 0 : 1) : greedy(f);
      } else a = Math.random() < p1(f) ? 1 : 0;
      const o = phys(st, a);
      n++;
      steps++;
      lastA = a;
      if (S.m === 'dqn') {
        buf.push([f, a, 0, feat(o.s), o.done]);
        if (buf.length > 10000) buf.shift();
        if (buf.length >= 200 && steps % 2 === 0) learnDQN();
      } else {
        ES.push(f);
        EA.push(a);
      }
      st = o.s;
      if (o.done || n >= CAP) {
        if (S.m === 'ac') {
          learnAC(o.done);
          ES = [];
          EA = [];
        }
        LEN.push(n);
        ep++;
        const len = n,
          av = LEN.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, LEN.length);
        st = reset();
        n = 0;
        cu = S.m === 'dqn' ? 2 : 3;
        msg =
          'Episode ' +
          ep +
          ': the pole stayed up for ' +
          len +
          ' of ' +
          CAP +
          ' steps' +
          (len >= CAP ? ', the full time' : '') +
          '. Average of the last 20 episodes: ' +
          f2(av) +
          '.' +
          (S.m === 'dqn' ? ' Random pushes now: ' + pct(eps()) + ' of the time.' : '');
        if (exL) eqShow(eq, exL, 0);
        if (solved() || ep >= MAXE) {
          phase = 'done';
          cu = 4;
          msg =
            (solved()
              ? 'Trained: the last 20 episodes averaged ' +
                f2(av) +
                ' steps, above the mark of 180. It took ' +
                ep +
                ' episodes and ' +
                steps +
                ' steps of experience.'
              : 'Stopped at the limit of ' +
                MAXE +
                ' episodes with an average of ' +
                f2(av) +
                ' steps. This run did not reach the mark: press Start over for a different one.') +
            ' Press a shove button to test the controller.';
        }
        return true;
      }
      return false;
    };
    const demoStep = () => {
      const a = greedy(feat(st)),
        o = phys(st, a, kick);
      kick = 0;
      lastA = a;
      st = o.s;
      demoN++;
      if (o.done || demoN >= 400) {
        const fell = o.done;
        phase = 'done';
        $('#aq-tpo', root).textContent = fell
          ? 'The pole fell after ' +
            demoN +
            ' steps. That shove was more than this controller had learned to handle.'
          : 'The controller kept the pole up for 400 steps after the shove.';
        st = reset();
        return true;
      }
      return false;
    };
    const shove = (d) => {
      if (phase === 'train') {
        $('#aq-tpo', root).textContent = 'Train the controller first: press Run.';
        return;
      }
      pb.stop();
      phase = 'demo';
      demoN = 0;
      st = reset();
      kick = d * 0.6;
      cu = 5;
      $('#aq-tpo', root).textContent =
        'The pole was shoved to the ' +
        (d > 0 ? 'right' : 'left') +
        '. The trained controller is in charge, with no random pushes.';
      pb.start();
    };
    const draw = () => {
      const t = T();
      {
        const { g, w, h } = fit(c),
          y0 = h * 0.72,
          sc = (w - 60) / 4.8,
          cx = w / 2 + st[0] * sc,
          L = h * 0.42;
        g.strokeStyle = t.muted;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(w / 2 - 2.4 * sc, y0 + 14);
        g.lineTo(w / 2 + 2.4 * sc, y0 + 14);
        g.stroke();
        g.lineWidth = 1;
        for (const e of [-2.4, 2.4]) {
          g.beginPath();
          g.moveTo(w / 2 + e * sc, y0 - 10);
          g.lineTo(w / 2 + e * sc, y0 + 22);
          g.stroke();
        }
        g.fillStyle = mix(t.accent, t.accent, 0, 0.85);
        g.fillRect(cx - 34, y0 - 12, 68, 24);
        g.fillStyle = t.ink;
        for (const d of [-20, 20]) {
          g.beginPath();
          g.arc(cx + d, y0 + 14, 6, 0, 7);
          g.fill();
        }
        g.strokeStyle = t.gold;
        g.lineWidth = 9;
        g.lineCap = 'round';
        g.beginPath();
        g.moveTo(cx, y0 - 12);
        g.lineTo(cx + Math.sin(st[2]) * L, y0 - 12 - Math.cos(st[2]) * L);
        g.stroke();
        g.lineCap = 'butt';
        g.fillStyle = t.ink;
        g.beginPath();
        g.arc(cx, y0 - 12, 5, 0, 7);
        g.fill();
        gwArrow(g, cx + (lastA ? 40 : -40), y0, lastA ? 1 : 3, 26, t.cut, 3);
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          phase === 'train'
            ? 'training, episode ' + (ep + 1) + ', step ' + n
            : phase === 'demo'
              ? 'testing the trained controller, step ' + demoN
              : 'stopped',
          8,
          14,
        );
        g.fillText(
          'pole angle ' + f2((st[2] * 180) / Math.PI) + '°   cart position ' + f2(st[0]),
          8,
          h - 8,
        );
        g.textAlign = 'right';
        g.fillText(
          'the episode ends if the pole passes 12° or the cart reaches an end mark',
          w - 8,
          y0 + 40,
        );
      }
      {
        const { g, w, h } = fit(cr),
          h1 = h * 0.46,
          A = uLine(g, t, 0, 0, w, h1, LEN, {
            title: 'steps the pole stayed up, per episode',
            n: Math.max(60, LEN.length),
            ym: CAP,
            col: t.accent,
          });
        g.setLineDash([5, 4]);
        g.strokeStyle = t.gold;
        g.lineWidth = 1.3;
        g.beginPath();
        g.moveTo(A.m.l, A.Y(180));
        g.lineTo(w - A.m.r, A.Y(180));
        g.stroke();
        g.setLineDash([]);
        g.font = MONO;
        g.fillStyle = t.cut;
        g.textAlign = 'right';
        g.fillText('mark to reach: average of 180', w - A.m.r, A.Y(180) + 13);
        const GWd = 30,
          GHt = 16;
        if (pmAt !== ep || !pmap) {
          pmAt = ep;
          const X = new Float64Array(GWd * GHt * 4);
          for (let r = 0; r < GHt; r++)
            for (let q = 0; q < GWd; q++) {
              const i = r * GWd + q;
              X[i * 4 + 2] = -1 + (2 * (q + 0.5)) / GWd;
              X[i * 4 + 3] = 1 - (2 * (r + 0.5)) / GHt;
            }
          const o = (S.m === 'dqn' ? net : actor).fwd(AD.input(X, [GWd * GHt, 4]));
          AD.clear();
          pmap = new Float32Array(GWd * GHt);
          for (let i = 0; i < GWd * GHt; i++) pmap[i] = o.v[2 * i + 1] - o.v[2 * i];
        }
        const x0 = 46,
          y0 = h1 + 22,
          W = w - x0 - 14,
          H = h - y0 - 26;
        let mx = 1e-9;
        for (const v of pmap) mx = Math.max(mx, Math.abs(v));
        for (let r = 0; r < GHt; r++)
          for (let q = 0; q < GWd; q++) {
            const v = pmap[r * GWd + q],
              col = v > 0 ? t.accent : t.cut,
              xa = Math.round(x0 + (q * W) / GWd),
              xb = Math.round(x0 + ((q + 1) * W) / GWd),
              ya = Math.round(y0 + (r * H) / GHt),
              yb = Math.round(y0 + ((r + 1) * H) / GHt);
            g.fillStyle = mix(col, col, 0, 0.25 + 0.7 * Math.min(1, (Math.abs(v) / mx) * 2));
            g.fillRect(xa, ya, xb - xa, yb - ya);
          }
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('its choice for every angle and falling speed', 8, h1 + 14);
        g.textAlign = 'center';
        g.fillText('leaning left', x0 + W * 0.15, h - 8);
        g.fillText('upright', x0 + W / 2, h - 8);
        g.fillText('leaning right', x0 + W * 0.85, h - 8);
        g.save();
        g.translate(14, y0 + H / 2);
        g.rotate(-Math.PI / 2);
        g.fillText('falling left ← → falling right', 0, 0);
        g.restore();
        g.fillStyle = t.accent;
        g.textAlign = 'right';
        g.fillText('■ push right', w - 14, h1 + 14);
        g.fillStyle = t.cut;
        g.fillText('■ push left', w - 110, h1 + 14);
        const px = x0 + ((clamp(st[2] / 0.21, -1, 1) + 1) / 2) * W,
          py = y0 + ((1 - clamp(st[3] / 2, -1, 1)) / 2) * H;
        g.fillStyle = t.ink;
        g.strokeStyle = t.bg;
        g.lineWidth = 2;
        g.beginPath();
        g.arc(px, py, 5, 0, 7);
        g.fill();
        g.stroke();
      }
      live.textContent = msg;
      uCur(root, 'aq', cu);
      const av = LEN.length
        ? LEN.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, LEN.length)
        : NaN;
      setR(
        root,
        'ph',
        phase === 'train'
          ? steps
            ? 'Training'
            : 'Not started'
          : phase === 'demo'
            ? 'Testing'
            : 'Finished',
      );
      setR(root, 'ep', String(ep));
      setR(root, 'll', LEN.length ? LEN[LEN.length - 1] + ' of ' + CAP : '–');
      setR(root, 'av', isFinite(av) ? f2(av) : '–');
      setR(root, 'ts', String(steps));
      setR(
        root,
        'ex',
        S.m === 'dqn' ? pct(eps()) + ' random pushes' : 'drawn from the actor’s own probabilities',
      );
    };
    build();
    if (RM) {
      while (phase === 'train') envStep();
    }
    const anim = Anim(() => {
      fr++;
      if (phase === 'done') {
        pb.stop();
        return;
      }
      if (phase === 'demo') {
        for (let i = 0; i < 2 && chk.remaining() > 0; i++)
          if (demoStep()) {
            draw();
            pb.stop();
            return;
          }
        draw();
        return;
      }
      const k = ep < 3 ? 2 : 14;
      for (let i = 0; i < k && phase === 'train' && chk.remaining() > 0; i++) envStep();
      draw();
      chk();
    });
    pb = playBtn($('#aq-play', root), anim);
    chk = runCtl(pb, root, 'aq', {
      done: () => phase === 'done',
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'episodes',
    });
    $('#aq-step', root).onclick = () => {
      pb.stop();
      if (phase !== 'train') {
        run++;
        build();
      } else {
        let k = 0;
        while (!envStep() && k++ < CAP + 5);
      }
      draw();
    };
    $('#aq-reset', root).onclick = () => {
      pb.stop();
      run++;
      build();
      draw();
    };
    segBind(root, 'aq-m', S.m, (v) => {
      S.m = v;
      pb.stop();
      run++;
      build();
      draw();
      pb.start();
    });
    $('#aq-kl', root).onclick = () => shove(-1);
    $('#aq-kr', root).onclick = () => shove(1);
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
