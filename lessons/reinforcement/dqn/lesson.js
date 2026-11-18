/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  MONO,
  prefersReducedMotion,
  readThemeColors,
  clamp,
  eqShow,
  f2,
  f3,
  prepareCanvas,
  mix,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
  setReadout,
} from '../../../js/core/shared.js';
import { AD } from '../../../js/math/autodiff.js';
import { gwArrow } from '../shared.js';
import { pct } from '../../ensembles/shared.js';
import { segBind, highlightLessonStep, drawSeriesChart } from '../../unsupervised/shared.js';

export default {
  init(root) {
    const mainCanvas = $('#aq-c', root),
      cr = $('#aq-r', root),
      live = $('#aq-live', root),
      eq = $('#aq-eq', root),
      S = { m: 'dqn' },
      CAP = 200,
      MAXE = 400,
      GA = 0.99,
      rn0 = randnFrom(Math.random);
    let net,
      targetNetwork,
      actor,
      critic,
      replayBuffer,
      steps,
      tr,
      ta,
      ep,
      episodeLengths,
      st,
      n,
      phase,
      msg,
      activeStepIndex,
      frameCount = 0,
      playback,
      checkRunStop = () => false,
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
      targetNetwork = AD.mlp([4, 32, 32, 2], rn, 'relu');
      net.ps.forEach((p, i) => targetNetwork.ps[i].v.set(p.v));
      actor = AD.mlp([4, 32, 2], rn, 'tanh');
      critic = AD.mlp([4, 32, 1], rn, 'tanh');
      replayBuffer = [];
      steps = 0;
      tr = 0;
      ta = 0;
      ep = 0;
      episodeLengths = [];
      st = reset();
      n = 0;
      phase = 'train';
      activeStepIndex = 0;
      ES = [];
      EA = [];
      kick = 0;
      exL = null;
      pmAt = -1;
      msg = 'A network with random weights is in control. It has never seen a pole. Press Run.';
      eqShow(eq, ['No training yet.'], 0);
    };
    const solved = () =>
      episodeLengths.length >= 20 &&
      episodeLengths.slice(-20).reduce((a, b) => a + b, 0) / 20 >= 180;
    const learnDQN = () => {
      const B = 32,
        X = new Float64Array(B * 4),
        X2 = new Float64Array(B * 4),
        bs = [];
      for (let i = 0; i < B; i++) {
        const e = replayBuffer[Math.floor(Math.random() * replayBuffer.length)];
        bs.push(e);
        X.set(e[0], i * 4);
        X2.set(e[3], i * 4);
      }
      const q2 = targetNetwork.fwd(AD.input(X2, [B, 4]));
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
      if (tr % 150 === 0) net.ps.forEach((p, i) => targetNetwork.ps[i].v.set(p.v));
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
        replayBuffer.push([f, a, 0, feat(o.s), o.done]);
        if (replayBuffer.length > 10000) replayBuffer.shift();
        if (replayBuffer.length >= 200 && steps % 2 === 0) learnDQN();
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
        episodeLengths.push(n);
        ep++;
        const len = n,
          av =
            episodeLengths.slice(-20).reduce((a, b) => a + b, 0) /
            Math.min(20, episodeLengths.length);
        st = reset();
        n = 0;
        activeStepIndex = S.m === 'dqn' ? 2 : 3;
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
          activeStepIndex = 4;
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
      playback.stop();
      phase = 'demo';
      demoN = 0;
      st = reset();
      kick = d * 0.6;
      activeStepIndex = 5;
      $('#aq-tpo', root).textContent =
        'The pole was shoved to the ' +
        (d > 0 ? 'right' : 'left') +
        '. The trained controller is in charge, with no random pushes.';
      playback.start();
    };
    const draw = () => {
      const colors = readThemeColors();
      {
        const { g: ctx, w, h } = prepareCanvas(mainCanvas),
          y0 = h * 0.72,
          sc = (w - 60) / 4.8,
          cx = w / 2 + st[0] * sc,
          L = h * 0.42;
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(w / 2 - 2.4 * sc, y0 + 14);
        ctx.lineTo(w / 2 + 2.4 * sc, y0 + 14);
        ctx.stroke();
        ctx.lineWidth = 1;
        for (const e of [-2.4, 2.4]) {
          ctx.beginPath();
          ctx.moveTo(w / 2 + e * sc, y0 - 10);
          ctx.lineTo(w / 2 + e * sc, y0 + 22);
          ctx.stroke();
        }
        ctx.fillStyle = mix(colors.accent, colors.accent, 0, 0.85);
        ctx.fillRect(cx - 34, y0 - 12, 68, 24);
        ctx.fillStyle = colors.ink;
        for (const d of [-20, 20]) {
          ctx.beginPath();
          ctx.arc(cx + d, y0 + 14, 6, 0, 7);
          ctx.fill();
        }
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 9;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(cx, y0 - 12);
        ctx.lineTo(cx + Math.sin(st[2]) * L, y0 - 12 - Math.cos(st[2]) * L);
        ctx.stroke();
        ctx.lineCap = 'butt';
        ctx.fillStyle = colors.ink;
        ctx.beginPath();
        ctx.arc(cx, y0 - 12, 5, 0, 7);
        ctx.fill();
        gwArrow(ctx, cx + (lastA ? 40 : -40), y0, lastA ? 1 : 3, 26, colors.cut, 3);
        ctx.font = MONO;
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText(
          phase === 'train'
            ? 'training, episode ' + (ep + 1) + ', step ' + n
            : phase === 'demo'
              ? 'testing the trained controller, step ' + demoN
              : 'stopped',
          8,
          14,
        );
        ctx.fillText(
          'pole angle ' + f2((st[2] * 180) / Math.PI) + '°   cart position ' + f2(st[0]),
          8,
          h - 8,
        );
        ctx.textAlign = 'right';
        ctx.fillText(
          'the episode ends if the pole passes 12° or the cart reaches an end mark',
          w - 8,
          y0 + 40,
        );
      }
      {
        const { g: ctx, w, h } = prepareCanvas(cr),
          h1 = h * 0.46,
          A = drawSeriesChart(ctx, colors, 0, 0, w, h1, episodeLengths, {
            title: 'steps the pole stayed up, per episode',
            n: Math.max(60, episodeLengths.length),
            ym: CAP,
            col: colors.accent,
          });
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = colors.gold;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(A.m.l, A.Y(180));
        ctx.lineTo(w - A.m.r, A.Y(180));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = MONO;
        ctx.fillStyle = colors.cut;
        ctx.textAlign = 'right';
        ctx.fillText('mark to reach: average of 180', w - A.m.r, A.Y(180) + 13);
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
              col = v > 0 ? colors.accent : colors.cut,
              xa = Math.round(x0 + (q * W) / GWd),
              xb = Math.round(x0 + ((q + 1) * W) / GWd),
              ya = Math.round(y0 + (r * H) / GHt),
              yb = Math.round(y0 + ((r + 1) * H) / GHt);
            ctx.fillStyle = mix(col, col, 0, 0.25 + 0.7 * Math.min(1, (Math.abs(v) / mx) * 2));
            ctx.fillRect(xa, ya, xb - xa, yb - ya);
          }
        ctx.fillStyle = colors.muted;
        ctx.textAlign = 'left';
        ctx.fillText('its choice for every angle and falling speed', 8, h1 + 14);
        ctx.textAlign = 'center';
        ctx.fillText('leaning left', x0 + W * 0.15, h - 8);
        ctx.fillText('upright', x0 + W / 2, h - 8);
        ctx.fillText('leaning right', x0 + W * 0.85, h - 8);
        ctx.save();
        ctx.translate(14, y0 + H / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('falling left ← → falling right', 0, 0);
        ctx.restore();
        ctx.fillStyle = colors.accent;
        ctx.textAlign = 'right';
        ctx.fillText('■ push right', w - 14, h1 + 14);
        ctx.fillStyle = colors.cut;
        ctx.fillText('■ push left', w - 110, h1 + 14);
        const px = x0 + ((clamp(st[2] / 0.21, -1, 1) + 1) / 2) * W,
          py = y0 + ((1 - clamp(st[3] / 2, -1, 1)) / 2) * H;
        ctx.fillStyle = colors.ink;
        ctx.strokeStyle = colors.bg;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, 7);
        ctx.fill();
        ctx.stroke();
      }
      live.textContent = msg;
      highlightLessonStep(root, 'aq', activeStepIndex);
      const av = episodeLengths.length
        ? episodeLengths.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, episodeLengths.length)
        : NaN;
      setReadout(
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
      setReadout(root, 'ep', String(ep));
      setReadout(
        root,
        'll',
        episodeLengths.length ? episodeLengths[episodeLengths.length - 1] + ' of ' + CAP : '–',
      );
      setReadout(root, 'av', isFinite(av) ? f2(av) : '–');
      setReadout(root, 'ts', String(steps));
      setReadout(
        root,
        'ex',
        S.m === 'dqn' ? pct(eps()) + ' random pushes' : 'drawn from the actor’s own probabilities',
      );
    };
    build();
    if (prefersReducedMotion) {
      while (phase === 'train') envStep();
    }
    const anim = Anim(() => {
      frameCount++;
      if (phase === 'done') {
        playback.stop();
        return;
      }
      if (phase === 'demo') {
        for (let i = 0; i < 2 && checkRunStop.remaining() > 0; i++)
          if (demoStep()) {
            draw();
            playback.stop();
            return;
          }
        draw();
        return;
      }
      const k = ep < 3 ? 2 : 14;
      for (let i = 0; i < k && phase === 'train' && checkRunStop.remaining() > 0; i++) envStep();
      draw();
      checkRunStop();
    });
    playback = playBtn($('#aq-play', root), anim);
    checkRunStop = attachRunLimit(playback, root, 'aq', {
      done: () => phase === 'done',
      restart: () => {
        run++;
        build();
      },
      count: () => ep,
      what: 'episodes',
    });
    $('#aq-step', root).onclick = () => {
      playback.stop();
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
      playback.stop();
      run++;
      build();
      draw();
    };
    segBind(root, 'aq-m', S.m, (v) => {
      S.m = v;
      playback.stop();
      run++;
      build();
      draw();
      playback.start();
    });
    $('#aq-kl', root).onclick = () => shove(-1);
    $('#aq-kr', root).onclick = () => shove(1);
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
