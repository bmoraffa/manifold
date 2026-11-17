/* Manifold — Reinforcement learning.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ reinforcement learning ============ */
const GWD = [
    [-1, 0],
    [0, 1],
    [1, 0],
    [0, -1],
  ],
  GWN = ['up', 'right', 'down', 'left'],
  GWS = ['↑', '→', '↓', '←'];
const rlN = (v) =>
  (v < 0 ? '−' : '') +
  (Math.abs(v) >= 100
    ? Math.abs(v).toFixed(0)
    : Math.abs(v) >= 9.995
      ? Math.abs(v).toFixed(1)
      : Math.abs(v).toFixed(2));
function gwMake(rows, o = {}) {
  const R = rows.length,
    C = rows[0].length,
    G = {
      R,
      C,
      rows,
      n: R * C,
      rs: o.rs != null ? o.rs : -1,
      rg: o.rg != null ? o.rg : 10,
      rp: o.rp != null ? o.rp : -10,
      slip: o.slip || 0,
      back: !!o.back,
    };
  G.ch = (s) => rows[Math.floor(s / C)][s % C];
  G.name = (s) => String.fromCharCode(65 + Math.floor(s / C)) + ((s % C) + 1);
  G.wall = (s) => G.ch(s) === '#';
  G.term = (s) => G.ch(s) === 'G' || (G.ch(s) === 'X' && !G.back);
  G.start = rows.join('').indexOf('S');
  G.states = [...Array(R * C).keys()].filter((s) => !G.wall(s));
  G.move = (s, a) => {
    const r = Math.floor(s / C) + GWD[a][0],
      c = (s % C) + GWD[a][1];
    if (r < 0 || c < 0 || r >= R || c >= C || rows[r][c] === '#') return s;
    return r * C + c;
  };
  G.rew = (s2) => (G.ch(s2) === 'G' ? G.rg : G.ch(s2) === 'X' ? G.rp : G.rs);
  G.land = (s2) => (G.back && G.ch(s2) === 'X' ? G.start : s2);
  G.outcomes = (s, a) => {
    const sl = G.slip;
    return sl
      ? [
          [1 - sl, G.move(s, a)],
          [sl / 2, G.move(s, (a + 1) % 4)],
          [sl / 2, G.move(s, (a + 3) % 4)],
        ]
      : [[1, G.move(s, a)]];
  };
  G.step = (s, a) => {
    let u = Math.random(),
      raw = s,
      slipped = false;
    const O = G.outcomes(s, a);
    for (let i = 0; i < O.length; i++) {
      u -= O[i][0];
      if (u <= 0 || i === O.length - 1) {
        raw = O[i][1];
        slipped = i > 0;
        break;
      }
    }
    return { s2: G.land(raw), raw, r: G.rew(raw), done: G.term(raw), slipped, bump: raw === s };
  };
  return G;
}
function gwQstar(G, gam) {
  const Q = new Float64Array(G.n * 4);
  for (let it = 0; it < 3000; it++) {
    let d = 0;
    for (const s of G.states) {
      if (G.term(s) || G.ch(s) === 'X') continue;
      for (let a = 0; a < 4; a++) {
        let v = 0;
        for (const [p, raw] of G.outcomes(s, a)) {
          const s2 = G.land(raw);
          v +=
            p *
            (G.rew(raw) +
              (G.term(raw)
                ? 0
                : gam * Math.max(Q[s2 * 4], Q[s2 * 4 + 1], Q[s2 * 4 + 2], Q[s2 * 4 + 3])));
        }
        d = Math.max(d, Math.abs(v - Q[s * 4 + a]));
        Q[s * 4 + a] = v;
      }
    }
    if (d < 1e-9) break;
  }
  return Q;
}
const gwBest = (Q, s) => {
  let b = 0;
  for (let a = 1; a < 4; a++) if (Q[s * 4 + a] > Q[s * 4 + b] + 1e-12) b = a;
  return b;
};
const gwMax = (Q, s) => Math.max(Q[s * 4], Q[s * 4 + 1], Q[s * 4 + 2], Q[s * 4 + 3]);
function gwRoute(G, Q, s0, cap = 60) {
  const path = [s0],
    acts = [];
  let s = s0,
    ret = 0;
  const seen = new Set([s0]);
  for (let i = 0; i < cap; i++) {
    if (G.term(s)) return { path, acts, ret, ok: G.ch(s) === 'G' };
    const a = gwBest(Q, s),
      raw = G.move(s, a);
    ret += G.rew(raw);
    acts.push(a);
    if (G.ch(raw) === 'X' && G.back)
      return { path: path.concat([raw]), acts, ret, ok: false, fell: true };
    s = raw;
    path.push(s);
    if (seen.has(s) && !G.term(s)) return { path, acts, ret, ok: false, loop: true };
    seen.add(s);
  }
  return { path, acts, ret, ok: G.ch(s) === 'G' };
}
function gwArrow(g, cx, cy, a, len, col, lw = 2) {
  const dx = GWD[a][1],
    dy = GWD[a][0],
    x2 = cx + dx * len,
    y2 = cy + dy * len,
    hs = Math.max(3, Math.min(7, len * 0.45));
  g.strokeStyle = col;
  g.fillStyle = col;
  g.lineWidth = lw;
  g.beginPath();
  g.moveTo(cx, cy);
  g.lineTo(x2, y2);
  g.stroke();
  g.beginPath();
  g.moveTo(x2 + dx * hs * 0.6, y2 + dy * hs * 0.6);
  g.lineTo(x2 - dx * hs * 0.5 - dy * hs * 0.6, y2 - dy * hs * 0.5 + dx * hs * 0.6);
  g.lineTo(x2 - dx * hs * 0.5 + dy * hs * 0.6, y2 - dy * hs * 0.5 - dx * hs * 0.6);
  g.closePath();
  g.fill();
}
function gwDraw(g, t, x0, y0, w, h, G, o = {}) {
  const cs = Math.floor(Math.min(w / G.C, h / G.R)),
    ox = Math.round(x0 + (w - cs * G.C) / 2),
    oy = Math.round(y0 + (h - cs * G.R) / 2),
    ctr = (s) => [ox + (s % G.C) * cs + cs / 2, oy + Math.floor(s / G.C) * cs + cs / 2],
    fam = MONO.split('px ')[1],
    fs = Math.max(8, Math.min(13, Math.round(cs * 0.16)));
  let vm = 1e-9;
  if (o.Q)
    for (const s of G.states)
      if (!G.term(s)) for (let a = 0; a < 4; a++) vm = Math.max(vm, Math.abs(o.Q[s * 4 + a]));
  if (o.V) for (const s of G.states) if (!G.term(s)) vm = Math.max(vm, Math.abs(o.V[s]));
  if (o.vm) vm = o.vm;
  for (let s = 0; s < G.n; s++) {
    const x = ox + (s % G.C) * cs,
      y = oy + Math.floor(s / G.C) * cs,
      ch = G.ch(s),
      cx = x + cs / 2,
      cy = y + cs / 2;
    if (ch === '#') {
      g.fillStyle = mix(t.ink, t.ink, 0, 0.78);
      g.fillRect(x, y, cs, cs);
      continue;
    }
    if (ch === 'G' || ch === 'X') {
      g.fillStyle = mix(
        ch === 'G' ? t.gold : t.cut,
        t.bg,
        ch === 'G' ? 0.25 : 0.1,
        ch === 'G' ? 0.75 : 0.55,
      );
      g.fillRect(x, y, cs, cs);
      g.fillStyle = t.ink;
      g.textAlign = 'center';
      g.font = '600 ' + Math.max(11, Math.round(cs * 0.2)) + 'px ' + fam;
      if (ch === 'G' && G.rg <= 0) {
        g.fillText('goal', cx, cy + 4);
      } else {
        g.fillText((ch === 'G' ? '+' : '−') + Math.abs(ch === 'G' ? G.rg : G.rp), cx, cy + 2);
        g.font = Math.max(8, Math.round(cs * 0.12)) + 'px ' + fam;
        g.fillText(ch === 'G' ? 'goal' : o.xname || 'pit', cx, cy + cs * 0.27);
      }
    } else if (o.Q) {
      const b = gwBest(o.Q, s),
        C4 = [
          [x, y],
          [x + cs, y],
          [x + cs, y + cs],
          [x, y + cs],
        ];
      let any = false;
      for (let a = 0; a < 4; a++) {
        const v = o.Q[s * 4 + a];
        if (v !== 0) any = true;
        const col = v >= 0 ? t.accent : t.cut;
        g.fillStyle = mix(col, col, 0, v === 0 ? 0 : 0.07 + 0.6 * Math.min(1, Math.abs(v) / vm));
        g.beginPath();
        g.moveTo(cx, cy);
        g.lineTo(C4[a][0], C4[a][1]);
        g.lineTo(C4[(a + 1) % 4][0], C4[(a + 1) % 4][1]);
        g.closePath();
        g.fill();
      }
      g.strokeStyle = t.line;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x, y);
      g.lineTo(x + cs, y + cs);
      g.moveTo(x + cs, y);
      g.lineTo(x, y + cs);
      g.stroke();
      if (o.hot && o.hot[0] === s) {
        const a = o.hot[1];
        g.strokeStyle = t.gold;
        g.lineWidth = 3;
        g.beginPath();
        g.moveTo(cx, cy);
        g.lineTo(C4[a][0], C4[a][1]);
        g.lineTo(C4[(a + 1) % 4][0], C4[(a + 1) % 4][1]);
        g.closePath();
        g.stroke();
      }
      if (cs >= 54 && !o.nonum) {
        g.textAlign = 'center';
        for (let a = 0; a < 4; a++) {
          g.font = (any && a === b ? '700 ' : '') + fs + 'px ' + fam;
          g.fillStyle = any && a === b ? t.ink : t.muted;
          g.fillText(
            rlN(o.Q[s * 4 + a]),
            cx + GWD[a][1] * cs * 0.3,
            cy + GWD[a][0] * cs * 0.33 + fs * 0.35,
          );
        }
      } else if (any) gwArrow(g, cx, cy, b, cs * 0.26, t.ink, 1.6);
    } else if (o.V) {
      const v = o.V[s],
        col = v >= 0 ? t.accent : t.cut;
      g.fillStyle = mix(col, col, 0, v === 0 ? 0 : 0.07 + 0.6 * Math.min(1, Math.abs(v) / vm));
      g.fillRect(x, y, cs, cs);
      g.textAlign = 'center';
      g.font = fs + 1 + 'px ' + fam;
      g.fillStyle = t.ink;
      g.fillText(rlN(v), cx, y + cs - 7);
      if (o.pol && o.pol[s] >= 0)
        gwArrow(
          g,
          cx - GWD[o.pol[s]][1] * cs * 0.14,
          cy - 4 - GWD[o.pol[s]][0] * cs * 0.14,
          o.pol[s],
          cs * 0.28,
          t.ink,
          2,
        );
    } else if (o.probs) {
      for (let a = 0; a < 4; a++) {
        const p = o.probs[s * 4 + a];
        if (p > 0.02)
          gwArrow(
            g,
            cx,
            cy,
            a,
            cs * 0.42 * p,
            mix(t.accent, t.accent, 0, 0.3 + 0.7 * p),
            1 + 3 * p,
          );
      }
      g.fillStyle = t.ink;
      g.beginPath();
      g.arc(cx, cy, 2, 0, 7);
      g.fill();
    }
    if (ch !== 'G' && ch !== 'X' && !o.noname) {
      g.font = Math.max(8, Math.round(cs * 0.11)) + 'px ' + fam;
      g.fillStyle = t.muted;
      g.textAlign = 'left';
      g.fillText(G.name(s) + (ch === 'S' ? ' start' : ''), x + 3, y + Math.max(9, cs * 0.12));
    }
    g.strokeStyle = t.muted;
    g.lineWidth = 1;
    g.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
    if (o.cur === s) {
      g.strokeStyle = t.gold;
      g.lineWidth = 3;
      g.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
    }
  }
  if (o.trail && o.trail.length > 1) {
    g.strokeStyle = mix(t.gold, t.gold, 0, o.thin ? 0.55 : 0.85);
    g.lineWidth = o.thin ? 2 : Math.max(2, cs * 0.05);
    g.lineJoin = 'round';
    g.beginPath();
    o.trail.forEach((s, i) => {
      const p = ctr(s);
      i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]);
    });
    g.stroke();
  }
  if (o.sel != null && o.sel >= 0) {
    const p = ctr(o.sel);
    g.strokeStyle = t.ink;
    g.lineWidth = 2.5;
    g.setLineDash([5, 4]);
    g.strokeRect(p[0] - cs / 2 + 3, p[1] - cs / 2 + 3, cs - 6, cs - 6);
    g.setLineDash([]);
  }
  if (o.agent != null && o.agent >= 0) {
    const p = ctr(o.agent),
      r = Math.max(5, cs * 0.13);
    g.fillStyle = o.acol || t.ink;
    g.strokeStyle = t.bg;
    g.lineWidth = 2.5;
    g.beginPath();
    g.arc(p[0], p[1], r, 0, 7);
    g.fill();
    g.stroke();
  }
  return {
    cs,
    ox,
    oy,
    ctr,
    at: (px, py) => {
      const c = Math.floor((px - ox) / cs),
        r = Math.floor((py - oy) / cs);
      return r < 0 || c < 0 || r >= G.R || c >= G.C ? -1 : r * G.C + c;
    },
  };
}
const gwClick = (c, fn) =>
  c.addEventListener('click', (e) => {
    const r = c.getBoundingClientRect();
    fn(e.clientX - r.left, e.clientY - r.top);
  });
(() => {
  const N = (v, sh) => ({ v, g: new Float64Array(v.length), sh });
  AD.pgl = (Lg, act, adv) => {
    const B = Lg.sh[0],
      V = Lg.sh[1],
      P = new Float64Array(B * V),
      o = N(new Float64Array(1), [1]);
    for (let i = 0; i < B; i++) {
      let mx = -1e30;
      for (let j = 0; j < V; j++) mx = Math.max(mx, Lg.v[i * V + j]);
      let z = 0;
      for (let j = 0; j < V; j++) {
        P[i * V + j] = Math.exp(Lg.v[i * V + j] - mx);
        z += P[i * V + j];
      }
      for (let j = 0; j < V; j++) P[i * V + j] /= z;
      o.v[0] -= (adv[i] * Math.log(P[i * V + act[i]] + 1e-12)) / B;
    }
    o.P = P;
    return AD._op(o, () => {
      for (let i = 0; i < B; i++)
        for (let j = 0; j < V; j++)
          Lg.g[i * V + j] += (o.g[0] * adv[i] * (P[i * V + j] - (j === act[i] ? 1 : 0))) / B;
    });
  };
})();
const RLL = [
  {
    id: 'bandits',
    tab: 'Multi-armed bandits',
    title: 'Multi-armed bandits: try new things, or stick with what works?',
    lede: 'Five food trucks park outside your office. Each has its own hidden chance of serving you a good lunch, and you have 1,500 lunches to spend. Every day at the best truck you have found so far is a day you learn nothing about the others. Every day spent testing another truck may be a wasted lunch. This is the simplest problem in reinforcement learning, and the same dilemma sits inside every harder one. Four strategies play it side by side here, and you can play against them.',
    stage: `<div class="row seg" id="mb-s" role="group" aria-label="Whose choices to watch"><button class="btn" type="button" data-v="0">Watch: Greedy</button><button class="btn" type="button" data-v="1">Watch: ε-Greedy</button><button class="btn" type="button" data-v="2">Watch: UCB</button><button class="btn" type="button" data-v="3">Watch: Thompson Sampling</button><button class="btn" type="button" data-v="4">Play Yourself</button></div>
 <div class="stage"><figure class="fig"><figcaption>The five trucks as the watched player sees them: its estimate of each truck’s success rate (bar), how many times it has eaten there, and how unsure it still is. Gold frame: today’s choice</figcaption><canvas class="short" id="mb-c" role="img" aria-label="Estimated success rate, count and uncertainty for each of five options"></canvas></figure>
 <figure class="fig"><figcaption>Top: regret, the good lunches each player has missed compared with always going to the best truck. Lower is better. Bottom: how often each has chosen the best truck</figcaption><canvas class="short" id="mb-r" role="img" aria-label="Cumulative regret and share of best choices for each strategy"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>How the watched player made today’s choice</figcaption><div class="eq" id="mb-eq" aria-live="polite" style="min-height:10em;margin-top:6px"></div></figure>
 ${uHow('mb', ['Each truck has a fixed chance of a good lunch. Nobody knows the five chances.', 'Each day every player picks one truck by its own rule, using only its own past lunches.', 'It gets a good lunch or a bad one, and updates its record for that truck: visits, and good lunches.', 'Greedy always picks the truck with the best record so far. ε-greedy does the same but picks at random a share ε of the time.', 'UCB adds a bonus to trucks it has rarely visited, and picks the highest total. Thompson sampling draws a plausible success rate for every truck from what it knows, and picks the highest draw.', 'After 1,500 days, compare how many good lunches each player missed.'])}
 <div class="controls">${uBtns('mb', 'Run', 'One day', 'days')}
 ${grp('Settings', slider('mb-e', 'ε for ε-greedy: share of random choices', 0, 0.5, 0.01, 0.1) + slider('mb-sp', 'Speed', 1, 3, 1, 2) + `<div class="row"><button class="btn" id="mb-rev" type="button">Reveal the True Chances</button></div>`)}</div>
 <dl class="read">${rd('d', 'Days so far')}${rd('w', 'Watched player')}${rd('g', 'Its good lunches')}${rd('rg', 'Its regret: good lunches missed', 'hl')}${rd('bs', 'Share of its days at the best truck', 'hl2')}${rd('fav', 'Truck it currently rates highest')}</dl>`,
    tryit:
      'First choose Play Yourself and click on trucks for 40 or 50 days. Which truck do you think is best, and how sure are you? Press Reveal the True Chances to check. Now start over and press Run, several times. Greedy is a gamble: in most runs it happens to settle on the best truck early and looks unbeatable, and in the rest it locks on to a worse truck after one unlucky lunch and its regret climbs in a straight line for good. ε-greedy never locks on, and never stops wasting a fixed share of its days either. Watch UCB: which trucks get the large bonus, and what happens to the bonus as the days pass? Watch Thompson sampling: the bell shapes narrow for the trucks it visits. Which two strategies have a regret curve that bends over and flattens?',
    math: String.raw`<h4>Intuition</h4><p>You move to a new town with five restaurants on your street. On the first night you pick one at random and it is good. On the second night, do you go back? If you always return to the best place you know, you may eat for years at the third-best restaurant in town because of one lucky evening. If you keep trying everything for ever, you keep eating bad dinners you could have avoided. The sensible course is in between: try the others enough to be reasonably sure, and try them less and less as the evidence comes in. The strategies on this page are different precise versions of "enough".</p>
 <h4>The problem</h4><p>There are \(K\) options, called arms after the levers of slot machines. Arm \(k\) pays 1 with unknown probability \(\mu_k\) and 0 otherwise. On each of \(T\) days you choose one arm \(a_t\) and see only its result. The best arm has \(\mu^\star=\max_k\mu_k\). The <b>regret</b> after \(T\) days is the expected reward given up by not always choosing it:</p>$$\text{Regret}(T)=\sum_{t=1}^{T}\big(\mu^\star-\mu_{a_t}\big).$$<p>A strategy that keeps choosing a worse arm a fixed share of the time has regret that grows in a straight line. A good strategy has regret that grows more and more slowly, like \(\log T\), which is the best that can be achieved.</p>
 <h4>Greedy and ε-greedy</h4><p>Keep the average result \(\hat\mu_k\) of each arm and its number of pulls \(n_k\). <b>Greedy</b> pulls \(\arg\max_k\hat\mu_k\). After trying each arm once it never explores again, so an unlucky first result on the best arm can banish it for good. <b>ε-greedy</b> pulls a uniformly random arm with probability \(\varepsilon\) and the greedy arm otherwise. It always finds the best arm in the end, and it always goes on spending the share \(\varepsilon\) on random pulls, most of them on arms it already knows are worse. Its regret is a straight line with a smaller slope.</p>
 <h4>UCB: optimism in the face of uncertainty</h4><p>The upper confidence bound rule (Auer and others, 2002) gives each arm a score: its average plus a bonus that is large when the arm has been tried little,</p>$$\text{score}_k=\hat\mu_k+\sqrt{\frac{\ln t}{2\,n_k}},$$<p>and pulls the highest score. An arm is chosen either because it looks good or because too little is known about it to rule it out. Each pull shrinks that arm's bonus. The \(\ln t\) makes every bonus creep up slowly, so no arm is abandoned for ever, and it is revisited more and more rarely. Exploration is directed at the arms where it can still change the decision. The original UCB1 rule uses the larger bonus \(\sqrt{2\ln t/n_k}\), which is safer in theory and explores noticeably more than is needed on a problem this small.</p>
 <h4>Thompson sampling: act on a plausible guess</h4><p>Keep a full belief about each \(\mu_k\). Starting from "anything between 0 and 1 is equally likely", after \(w_k\) good and \(l_k\) bad results the belief is a Beta distribution,</p>$$\mu_k\sim\mathrm{Beta}(1+w_k,\ 1+l_k).$$<p>Each day, draw one value from every arm's belief and pull the arm whose draw is highest (Thompson, 1933). A rarely tried arm has a wide belief, so its draw is sometimes high and it gets tried. A well-known poor arm has a narrow belief around a low value and almost never wins the draw. The rule has no setting to tune, and it chooses each arm with exactly the probability that the arm is the best one given the evidence.</p>
 <h4>Why this matters beyond lunch</h4><p>Websites choose which headline or advert to show with these rules, clinical trials use them to give fewer patients the worse treatment, and recommendation systems use them to try new items. In the rest of this module the same tension returns as the ε in Q-learning: a robot that only does what currently looks best never finds out that something else is better.</p>
 <h4>What this page runs</h4><p>Five arms with success chances 0.20, 0.35, 0.50, 0.60 and 0.80, in a random order that is hidden until you ask. All four strategies face the same five trucks for 1,500 days, each with its own lunches. Regret is computed from the true chances, so it is exact and does not depend on the luck of individual lunches. One run is one sample, so press Start over a few times. Averaged over 300 runs of this exact setup, the regret after 1,500 days is about 99 for greedy (usually under 5, occasionally over 300), 68 for ε-greedy with ε = 0.1, 33 for UCB and 22 for Thompson sampling.</p>`,
    q: 'After 1,500 days ε-greedy with ε = 0.1 has found the best truck and rates it highest. Its regret is still rising steadily. Why, and what simple change would fix it?',
    a: 'It still chooses a random truck on one day in ten, and four times out of five that random truck is not the best one. Those days are wasted for ever, however certain it has become, so its regret rises by a fixed amount per day. The fix is to explore less as evidence accumulates, for example by shrinking ε over time. UCB and Thompson sampling do this automatically: their exploration fades exactly as fast as their uncertainty does.',
    init(root) {
      const c = $('#mb-c', root),
        cr = $('#mb-r', root),
        live = $('#mb-live', root),
        eq = $('#mb-eq', root),
        S = { w: 2, eps: 0.1, sp: 2, rev: false },
        K = 5,
        TMAX = 1500,
        NM = ['Greedy', 'ε-greedy', 'UCB', 'Thompson sampling', 'You'];
      let MU,
        best,
        PL,
        t = 0,
        ty = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        P = null;
      const rnorm = randnFrom(Math.random),
        rgam = (k) => {
          const d = k - 1 / 3,
            cc = 1 / Math.sqrt(9 * d);
          for (;;) {
            let x, v;
            do {
              x = rnorm();
              v = 1 + cc * x;
            } while (v <= 0);
            v = v * v * v;
            const u = Math.random();
            if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
          }
        },
        rbeta = (a, b) => {
          const x = rgam(a),
            y = rgam(b);
          return x / (x + y);
        };
      const build = () => {
        MU = [0.2, 0.35, 0.5, 0.6, 0.8]
          .map((v) => [Math.random(), v])
          .sort((a, b) => a[0] - b[0])
          .map((v) => v[1]);
        best = MU.indexOf(Math.max(...MU));
        PL = NM.map(() => ({
          n: new Int32Array(K),
          w: new Int32Array(K),
          reg: [0],
          bs: 0,
          good: 0,
          last: -1,
          lw: 0,
          info: null,
        }));
        t = 0;
        ty = 0;
        fin = false;
        cu = 0;
        S.rev = false;
        msg =
          'Five trucks, five hidden chances of a good lunch. Press Run to let the four strategies play, or choose Play Yourself and click a truck.';
        eqShow(eq, ['No lunches yet.'], 0);
      };
      const est = (p, k) => (p.n[k] ? p.w[k] / p.n[k] : 0);
      const choose = (i, p, day) => {
        const un = [...Array(K).keys()].filter((k) => !p.n[k]);
        if (i === 0) {
          if (un.length)
            return {
              k: un[0],
              why: 'It has not tried truck ' + (un[0] + 1) + ' yet, so it tries it once.',
            };
          let b = 0;
          for (let k = 1; k < K; k++) if (est(p, k) > est(p, b)) b = k;
          return {
            k: b,
            why:
              'Best record so far: truck ' +
              (b + 1) +
              ' with ' +
              pct(est(p, b)) +
              '. Greedy goes there and nowhere else.',
          };
        }
        if (i === 1) {
          if (Math.random() < S.eps) {
            const k = Math.floor(Math.random() * K);
            return {
              k,
              why:
                'It rolled a number below ε = ' +
                f2(S.eps) +
                ', so it <b>explores</b>: a truck chosen at random, number ' +
                (k + 1) +
                '.',
            };
          }
          let b = 0;
          for (let k = 1; k < K; k++) if (est(p, k) > est(p, b)) b = k;
          return {
            k: b,
            why:
              'It <b>exploits</b>: the best record so far is truck ' +
              (b + 1) +
              ' with ' +
              pct(est(p, b)) +
              '.',
          };
        }
        if (i === 2) {
          if (un.length)
            return {
              k: un[0],
              why:
                'It has not tried truck ' +
                (un[0] + 1) +
                ' yet. An untried truck has an unlimited bonus, so it goes there.',
            };
          const sc = [...Array(K).keys()].map(
            (k) => est(p, k) + Math.sqrt(Math.log(day) / (2 * p.n[k])),
          );
          let b = 0;
          for (let k = 1; k < K; k++) if (sc[k] > sc[b]) b = k;
          return { k: b, sc, es: [...Array(K).keys()].map((k) => est(p, k)), why: null };
        }
        const dr = [...Array(K).keys()].map((k) => rbeta(1 + p.w[k], 1 + p.n[k] - p.w[k]));
        let b = 0;
        for (let k = 1; k < K; k++) if (dr[k] > dr[b]) b = k;
        return { k: b, dr, why: null };
      };
      const pull = (i, k, info) => {
        const p = PL[i],
          win = Math.random() < MU[k] ? 1 : 0;
        p.n[k]++;
        p.w[k] += win;
        p.good += win;
        p.last = k;
        p.lw = win;
        p.info = info;
        p.reg.push(p.reg[p.reg.length - 1] + MU[best] - MU[k]);
        if (k === best) p.bs++;
      };
      const explain = (i) => {
        const p = PL[i],
          o = p.info,
          k = p.last,
          day = i === 4 ? ty : t,
          L = ['Day ' + day + ', ' + NM[i] + '.'];
        if (i === 4) L.push('You chose truck ' + (k + 1) + '.');
        else if (o.why) L.push(o.why);
        else if (o.sc) {
          for (let j = 0; j < K; j++)
            L.push(
              'truck ' +
                (j + 1) +
                ':  record ' +
                f3(o.es[j]) +
                '  +  bonus √(ln ' +
                day +
                ' ÷ (2 × ' +
                (p.n[j] - (j === k ? 1 : 0)) +
                '))  =  ' +
                f3(o.sc[j]) +
                (j === k ? '   <b>← highest score</b>' : ''),
            );
        } else {
          for (let j = 0; j < K; j++)
            L.push(
              'truck ' +
                (j + 1) +
                ':  ' +
                (p.w[j] - (j === k ? p.lw : 0)) +
                ' good, ' +
                (p.n[j] - p.w[j] - (j === k ? 1 - p.lw : 0)) +
                ' bad so far.  A plausible success rate drawn from that: ' +
                f3(o.dr[j]) +
                (j === k ? '   <b>← highest draw</b>' : ''),
            );
        }
        L.push(
          'Lunch at truck ' +
            (k + 1) +
            ': <b>' +
            (p.lw ? 'good' : 'bad') +
            '</b>. Its record there is now ' +
            p.w[k] +
            ' good out of ' +
            p.n[k] +
            ' = ' +
            pct(est(p, k)) +
            '.',
        );
        return L;
      };
      const day = (quiet) => {
        if (fin) return;
        t++;
        for (let i = 0; i < 4; i++) {
          const o = choose(i, PL[i], t);
          if (i === 2 && o.sc) {
            /* scores use counts before the pull */
          }
          pull(i, o.k, o);
        }
        if (!quiet && S.w < 4) {
          eqShow(eq, explain(S.w), 0);
          const p = PL[S.w];
          cu = S.w < 2 ? 3 : 4;
          msg =
            'Day ' +
            t +
            ': ' +
            NM[S.w] +
            ' went to truck ' +
            (p.last + 1) +
            ' and had a ' +
            (p.lw ? 'good' : 'bad') +
            ' lunch. Its regret so far is ' +
            f2(p.reg[p.reg.length - 1]) +
            ' good lunches.';
        }
        if (t >= TMAX) {
          fin = true;
          cu = 5;
          const o = [0, 1, 2, 3].sort((a, b) => PL[a].reg[t] - PL[b].reg[t]);
          msg =
            'Finished: 1,500 days. Good lunches missed, least first: ' +
            o.map((i) => NM[i] + ' ' + f2(PL[i].reg[t])).join(', ') +
            '. The best truck was number ' +
            (best + 1) +
            ', where ' +
            pct(MU[best]) +
            ' of lunches are good.' +
            (ty ? ' You missed ' + f2(PL[4].reg[ty]) + ' in ' + ty + ' days.' : '');
          S.rev = true;
        }
      };
      const draw = () => {
        const T0 = T(),
          col = UC(T0),
          w0 = S.w,
          p = PL[w0];
        {
          const { g, w, h } = fit(c),
            m = { l: 36, r: 10, t: 26, b: 54 },
            cw = (w - m.l - m.r) / K,
            Y = (v) => h - m.b - v * (h - m.t - m.b);
          g.font = MONO;
          g.fillStyle = T0.muted;
          g.textAlign = 'left';
          g.fillText(
            NM[w0] +
              (w0 === 4 ? ': click a truck to have lunch there' : '’s view of the five trucks'),
            8,
            14,
          );
          g.strokeStyle = T0.line;
          g.lineWidth = 1;
          for (const v of [0, 0.25, 0.5, 0.75, 1]) {
            g.beginPath();
            g.moveTo(m.l, Y(v));
            g.lineTo(w - m.r, Y(v));
            g.stroke();
            g.textAlign = 'right';
            g.fillText(100 * v + '%', m.l - 5, Y(v) + 4);
          }
          for (let k = 0; k < K; k++) {
            const x = m.l + k * cw,
              cx = x + cw / 2,
              e = est(p, k),
              bw = cw * 0.36;
            g.fillStyle = mix(T0.accent, T0.accent, 0, p.n[k] ? 0.8 : 0.15);
            g.fillRect(cx - bw / 2, Y(e), bw, Y(0) - Y(e));
            if (w0 === 2 && p.n[k] && t > 0) {
              const u = Math.min(1.25, e + Math.sqrt(Math.log(Math.max(2, t)) / (2 * p.n[k])));
              g.strokeStyle = T0.cut;
              g.lineWidth = 2;
              g.beginPath();
              g.moveTo(cx, Y(e));
              g.lineTo(cx, Y(Math.min(1.12, u)));
              g.moveTo(cx - 8, Y(Math.min(1.12, u)));
              g.lineTo(cx + 8, Y(Math.min(1.12, u)));
              g.stroke();
            }
            if (w0 === 3 || w0 === 4) {
              const a = 1 + p.w[k],
                b = 1 + p.n[k] - p.w[k];
              let mx = 0;
              const D = [];
              for (let i = 0; i <= 60; i++) {
                const x2 = (i + 0.5) / 61,
                  d = Math.exp((a - 1) * Math.log(x2) + (b - 1) * Math.log(1 - x2));
                D.push(d);
                mx = Math.max(mx, d);
              }
              g.fillStyle = mix(T0.cut, T0.cut, 0, 0.45);
              g.beginPath();
              g.moveTo(cx + bw / 2 + 3, Y(0));
              D.forEach((d, i) =>
                g.lineTo(cx + bw / 2 + 3 + (d / mx) * cw * 0.26, Y((i + 0.5) / 61)),
              );
              g.lineTo(cx + bw / 2 + 3, Y(1));
              g.closePath();
              g.fill();
              if (w0 === 3 && p.info && p.info.dr) {
                g.fillStyle = T0.ink;
                g.beginPath();
                g.arc(cx + bw / 2 + 3, Y(p.info.dr[k]), 3.5, 0, 7);
                g.fill();
              }
            }
            if (S.rev) {
              g.setLineDash([5, 4]);
              g.strokeStyle = T0.gold;
              g.lineWidth = 2;
              g.beginPath();
              g.moveTo(x + 6, Y(MU[k]));
              g.lineTo(x + cw - 6, Y(MU[k]));
              g.stroke();
              g.setLineDash([]);
            }
            if (p.last === k) {
              g.strokeStyle = T0.gold;
              g.lineWidth = 3;
              g.strokeRect(x + 4, m.t - 4, cw - 8, h - m.t - 12);
              g.fillStyle = p.lw ? T0.ok : T0.muted;
              g.textAlign = 'center';
              g.fillText(p.lw ? 'good lunch' : 'bad lunch', cx, m.t + 10);
            }
            g.textAlign = 'center';
            g.fillStyle = T0.ink;
            g.font = '600 12px ' + MONO.split('px ')[1];
            g.fillText('truck ' + (k + 1), cx, h - m.b + 16);
            g.font = MONO;
            g.fillStyle = T0.muted;
            g.fillText(p.n[k] ? p.w[k] + ' good of ' + p.n[k] : 'not tried', cx, h - m.b + 30);
            g.fillText(p.n[k] ? pct(e) : '', cx, h - m.b + 43);
          }
          P = { m, cw };
          g.textAlign = 'left';
          g.fillStyle = T0.muted;
          if (w0 === 2) g.fillText('whisker: record plus UCB bonus', m.l + 4, m.t + 10 + 0 * 12);
          if (w0 >= 3)
            g.fillText(
              'shape: belief about the true chance' + (w0 === 3 ? '; dot: today’s draw' : ''),
              m.l + 4,
              h - 6,
            );
          if (S.rev) {
            g.fillStyle = T0.cut;
            g.textAlign = 'right';
            g.fillText('dashed: the true chance', w - m.r, 14);
          }
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.62,
            n = TMAX + 1,
            ym = Math.max(12, ...PL.map((q) => q.reg[q.reg.length - 1])),
            ids = [0, 1, 2, 3].concat(ty ? [4] : []);
          let L = null;
          ids.forEach((i, j) => {
            L = uLine(g, T0, 0, 0, w, h1, PL[i].reg, {
              title: j ? '' : 'regret: good lunches missed so far',
              n,
              ym,
              col: col[i],
            });
          });
          g.font = MONO;
          ids.forEach((i) => {
            const q = PL[i],
              k = q.reg.length - 1;
            g.fillStyle = col[i];
            g.textAlign = 'left';
            g.fillText(NM[i], Math.min(L.X(k) + 5, w - 128), L.Y(q.reg[k]) - 6);
          });
          const y0 = h1 + 18,
            bh = Math.min(17, (h - h1 - 26) / 5);
          g.fillStyle = T0.muted;
          g.textAlign = 'left';
          g.fillText('share of days spent at the best truck', 8, h1 + 12);
          ids.forEach((i, j) => {
            const q = PL[i],
              d = i === 4 ? ty : t,
              sh = d ? q.bs / d : 0,
              x0 = 120;
            g.fillStyle = col[i];
            g.fillRect(x0, y0 + j * bh + 1, Math.max(1, (w - x0 - 60) * sh), bh - 3);
            g.fillStyle = T0.muted;
            g.textAlign = 'right';
            g.fillText(NM[i].replace(' sampling', ''), x0 - 6, y0 + j * bh + bh - 5);
            g.textAlign = 'left';
            g.fillText(d ? pct(sh) : '–', x0 + (w - x0 - 60) * sh + 5, y0 + j * bh + bh - 5);
          });
        }
        live.textContent = msg;
        uCur(root, 'mb', cu);
        const d = w0 === 4 ? ty : t;
        let fv = -1;
        for (let k = 0; k < K; k++) if (p.n[k] && (fv < 0 || est(p, k) > est(p, fv))) fv = k;
        setR(root, 'd', t + ' of ' + TMAX + (ty ? ' (you: ' + ty + ')' : ''));
        setR(root, 'w', NM[w0]);
        setR(root, 'g', p.good + ' of ' + d);
        setR(root, 'rg', f2(p.reg[p.reg.length - 1]));
        setR(root, 'bs', d ? (S.rev ? pct(p.bs / d) : 'hidden until you reveal') : '–');
        setR(root, 'fav', fv < 0 ? '–' : 'truck ' + (fv + 1) + ' (' + pct(est(p, fv)) + ')');
      };
      build();
      if (RM) {
        while (!fin) day(true);
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (S.sp === 1) {
          if (fr % 40) return;
          day(false);
        } else if (S.sp === 2) {
          for (let i = 0; i < 3 && !fin; i++) day(i < 2);
        } else {
          for (let i = 0; i < 15 && !fin; i++) day(i < 14);
        }
        draw();
        chk();
      });
      pb = playBtn($('#mb-play', root), anim);
      chk = runCtl(pb, root, 'mb', {
        done: () => fin,
        restart: () => build(),
        count: () => t,
        what: 'days',
      });
      $('#mb-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else day(false);
        draw();
      };
      $('#mb-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      $('#mb-rev', root).onclick = () => {
        S.rev = !S.rev;
        draw();
      };
      segBind(root, 'mb-s', String(S.w), (v) => {
        S.w = +v;
        if (S.w === 4) {
          pb.stop();
          msg =
            'Your turn: click a truck in the left picture to have lunch there. Your regret is drawn with the others.';
        }
        if (PL[S.w].info || (S.w === 4 && ty))
          eqShow(eq, PL[S.w].last >= 0 ? explain(S.w) : ['No lunches yet.'], 0);
        draw();
      });
      bind(
        root,
        'mb-e',
        (v) => {
          S.eps = +v;
        },
        f2,
      );
      bind(
        root,
        'mb-sp',
        (v) => {
          S.sp = +v;
        },
        (v) => ['', 'slow', 'steady', 'fast'][+v],
      );
      gwClick(c, (x, y) => {
        if (S.w !== 4 || !P || ty >= TMAX) return;
        const k = Math.floor((x - P.m.l) / P.cw);
        if (k < 0 || k >= K) return;
        ty++;
        pull(4, k, {});
        eqShow(eq, explain(4), 0);
        msg =
          'Day ' +
          ty +
          ' for you: truck ' +
          (k + 1) +
          ', a ' +
          (PL[4].lw ? 'good' : 'bad') +
          ' lunch. You have missed ' +
          f2(PL[4].reg[ty]) +
          ' good lunches so far compared with always choosing the best truck.';
        draw();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          build();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'value-iteration',
    tab: 'Value and policy iteration',
    title: 'Value and policy iteration: planning when you know the rules',
    lede: 'Before learning by trial and error, take the easier case: the robot has a complete map. It knows every square, what each move does, and what each reward is. No experience is needed, only thinking. The thinking is one small calculation repeated over and over: the value of a square is the best you can get by making one move and then collecting the value of wherever you land. Watch values spread out from the goal like water, and the arrows fall into place behind them.',
    stage: `<div class="row seg" id="vi-m" role="group" aria-label="Method"><button class="btn" type="button" data-v="vi">Value Iteration</button><button class="btn" type="button" data-v="pi">Policy Iteration</button></div>
 <div class="stage"><figure class="fig"><figcaption>The number in each square is its current value: the total reward expected from there. The arrow is the best move given those values. Gold frame: the square being recomputed. Click a square to see its calculation</figcaption><canvas class="short" id="vi-c" role="img" aria-label="Grid world with the value and best move of every square"></canvas></figure>
 <figure class="fig"><figcaption>Top: the largest change in any value during each sweep. Bottom: how many arrows changed in each sweep</figcaption><canvas class="short" id="vi-r" role="img" aria-label="Largest value change and number of policy changes per sweep"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The calculation for one square</figcaption><div class="eq" id="vi-eq" aria-live="polite" style="min-height:11em;margin-top:6px"></div></figure>
 ${uHow('vi', ['Start with every value at zero.', 'Value iteration: take a square. For each of the four moves, work out reward + γ × the value of where the move leads, averaged over where it might slip to.', 'The new value of the square is the best of those four numbers, and the arrow points at the move that gave it.', 'Do this for every square: that is one sweep. Repeat sweeps until no value changes by more than 0.001.', 'Policy iteration instead fixes all the arrows, works out the values of following exactly those arrows, then redraws each arrow toward the best move, and repeats until no arrow changes.'])}
 <div class="controls">${uBtns('vi', 'Run', 'One step', 'sweeps')}
 ${grp('The world', `<div class="row seg" id="vi-w" role="group" aria-label="World"><button class="btn" type="button" data-v="room">The Room with a Pit</button><button class="btn" type="button" data-v="maze">The Maze</button></div>` + slider('vi-sl', 'Slippery floor: chance a move slides sideways', 0, 0.5, 0.05, 0.2) + slider('vi-ga', 'Discount γ', 0.5, 1, 0.01, 0.9) + slider('vi-rs', 'Reward for each ordinary step', -3, 0, 0.25, -1))}</div>
 <dl class="read">${rd('me', 'Method')}${rd('sw', 'Sweeps so far')}${rd('dl', 'Largest change in the last sweep', 'hl2')}${rd('vs', 'Value of the start square', 'hl')}${rd('rt', 'Route from the start, following the arrows')}${rd('pc', 'Arrows changed in the last sweep')}</dl>`,
    tryit:
      'Press One step repeatedly in the first sweep and read each calculation. Only squares next to the goal or the pit get interesting values at first. Why? Press Run and watch the values travel outward, one square per sweep. Notice the arrows settle long before the numbers do. Now set the reward for each step to 0 and the slippery slider to 0.3, and look at C4, the square under the pit: its arrow points down, into the wall. Then set the step reward to −3. With a very costly step, what does the robot next to the pit prefer? Finally switch to Policy Iteration: how many rounds does it need, compared with the number of sweeps value iteration took?',
    math: String.raw`<h4>Intuition</h4><p>Suppose you want to know how many minutes each junction in a town is from the railway station, and you have a map. Write 0 on the station. Every junction next to it is "one street away": write that down. Every junction next to those is "one more street". Keep going and the numbers flood outward from the station until the map is covered. Once every junction has its number, finding the way from anywhere is trivial: at each junction, walk toward the neighbour with the smallest number. Value iteration is this flood, with rewards in place of minutes and with the complication that a move may not go where you intended.</p>
 <h4>The setting: a Markov decision process</h4><p>There is a set of states \(s\), a set of actions \(a\), transition probabilities \(P(s'\mid s,a)\) saying where each action leads, a reward \(r(s,a,s')\), and a discount \(\gamma\). A <b>policy</b> \(\pi\) says which action to take in each state. The <b>value</b> \(V^\pi(s)\) is the expected total discounted reward from \(s\) when following \(\pi\). The goal is the policy whose values are highest everywhere.</p>
 <h4>The Bellman optimality equation</h4><p>The values of the best policy satisfy a consistency condition (Bellman, 1957). The value of a state is the best, over actions, of the immediate reward plus the discounted value of the next state:</p>$$V^\star(s)=\max_a\sum_{s'}P(s'\mid s,a)\,\big[r(s,a,s')+\gamma V^\star(s')\big].$$<p>This is one equation for every state, all tied together, and the \(\max\) makes them impossible to solve by ordinary algebra.</p>
 <h4>Value iteration</h4><p>Turn the equation into an update. Start from \(V_0=0\) and compute, for every state,</p>$$V_{k+1}(s)=\max_a\sum_{s'}P(s'\mid s,a)\,\big[r(s,a,s')+\gamma V_k(s')\big].$$<p>The panel under the pictures writes this out for one square: four lines, one per move, then the best. Each sweep brings every value closer to \(V^\star\) by at least the factor \(\gamma\):</p>$$\max_s\big|V_{k+1}(s)-V^\star(s)\big|\le\gamma\,\max_s\big|V_k(s)-V^\star(s)\big|.$$<p>That is why the top-right curve falls steadily, and why a \(\gamma\) near 1 needs more sweeps. Information travels one square per sweep, so a goal ten squares away takes at least ten sweeps to be felt.</p>
 <h4>From values to behaviour</h4><p>Given values, the best move in each state is the one that achieved the maximum:</p>$$\pi(s)=\arg\max_a\sum_{s'}P(s'\mid s,a)\,\big[r(s,a,s')+\gamma V(s')\big].$$<p>The arrows usually stop changing long before the values have converged, because the arrow only depends on which neighbour is better, not on by how much.</p>
 <h4>Policy iteration</h4><p>Howard (1960) used that observation. Alternate two steps. <b>Evaluate</b>: with the arrows fixed, find the values of following them, by sweeping the simpler update without a \(\max\) until it settles,</p>$$V^\pi(s)=\sum_{s'}P(s'\mid s,\pi(s))\,\big[r+\gamma V^\pi(s')\big].$$<p><b>Improve</b>: redraw every arrow toward the best move under those values. Each round gives a policy at least as good as the last, and there are finitely many policies, so it stops, usually after very few rounds. Each round costs many sweeps, though, as the readout shows.</p>
 <h4>What the settings do</h4><p><b>Slippery floor</b>: with chance \(p\) a move slides to one side or the other. Next to the pit, when steps are cheap, the robot then prefers a move that cannot slide into it, even walking into a wall. <b>Step reward</b>: at 0 there is no hurry and the robot takes the safest route however long. At \(-3\) life is so costly that a risky short cut becomes worthwhile, and for squares far from the goal even the pit can be preferable to a long walk. <b>\(\gamma\)</b>: small values make distant rewards nearly worthless.</p>
 <h4>Why this is not yet learning</h4><p>Everything here needs \(P\) and \(r\) to be known: the robot is planning, not learning. In most real problems nobody hands you the rules. The next tabs keep the same Bellman idea and replace the known probabilities with experience: Monte Carlo and TD learning estimate values from sampled walks, and Q-learning does so while also choosing the moves.</p>`,
    q: 'In the room, set the reward for each step to 0 and the slippery floor to 0.3. The best move in C4, the square directly under the pit, is now “down”, straight into the wall, which gets the robot nowhere. Why is that the best move?',
    a: 'Because a move can slide sideways. The robot wants to go left, away from the pit. But a move to the left can slide up, and up from C4 is the pit. A move down pushes into the wall and keeps the robot where it is, and its two possible slides are left, to safety, and right, into another wall. So “down” can never enter the pit, and sooner or later it slides the robot to the left. When steps cost nothing, waiting for that lucky slide is better than taking a real chance of losing 10. Give each step a cost of 1 and the patience stops paying: click C4 to compare the four numbers.',
    init(root) {
      const c = $('#vi-c', root),
        cr = $('#vi-r', root),
        live = $('#vi-live', root),
        eq = $('#vi-eq', root),
        S = { m: 'vi', w: 'room', sl: 0.2, ga: 0.9, rs: -1 },
        W = {
          room: ['...G', '.#.X', 'S...'],
          maze: ['S..#..', '.#...#', '.#.#..', '...#.X', '.#...G'],
        };
      let G,
        V,
        Vn,
        pol,
        order = [],
        ci = 0,
        sw = 0,
        DL = [],
        PCH = [],
        dcur = 0,
        pch = 0,
        cur = -1,
        sel = -1,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        P = null,
        evs = 0;
      const qa = (s, a, Vx) => {
        let v = 0;
        for (const [p, s2] of G.outcomes(s, a))
          v += p * (G.rew(s2) + (G.term(s2) ? 0 : S.ga * Vx[s2]));
        return v;
      };
      const bestA = (s, Vx) => {
        let b = 0,
          bv = -1e30;
        for (let a = 0; a < 4; a++) {
          const v = qa(s, a, Vx);
          if (v > bv + 1e-10) {
            bv = v;
            b = a;
          }
        }
        return [b, bv];
      };
      const build = () => {
        G = gwMake(W[S.w], { slip: S.sl, rs: S.rs });
        V = new Float64Array(G.n);
        Vn = new Float64Array(G.n);
        pol = new Int8Array(G.n).fill(-1);
        order = G.states.filter((s) => !G.term(s));
        if (S.m === 'pi') for (const s of order) pol[s] = 0;
        ci = 0;
        sw = 0;
        DL = [];
        PCH = [];
        dcur = 0;
        pch = 0;
        cur = -1;
        fin = false;
        cu = 0;
        evs = 0;
        msg =
          S.m === 'vi'
            ? 'Every value is zero and there are no arrows yet. Press One step to recompute one square, or Run.'
            : 'Policy iteration starts from an arbitrary set of arrows: every square says “up”. Press One step for one round of evaluate and improve.';
        eqShow(eq, ['Nothing has been computed yet.'], 0);
      };
      const lines = (s, Vx, head) => {
        const L = [head];
        let bv = -1e30,
          b = 0;
        for (let a = 0; a < 4; a++) {
          const O = G.outcomes(s, a),
            v = qa(s, a, Vx);
          if (v > bv + 1e-10) {
            bv = v;
            b = a;
          }
          L.push(
            GWS[a] +
              ' ' +
              GWN[a].padEnd(5) +
              ': ' +
              O.map(
                ([p, s2]) =>
                  (O.length > 1 ? f2(p) + ' × ' : '') +
                  '(' +
                  rlN(G.rew(s2)) +
                  (G.term(s2) ? '' : ' + ' + f2(S.ga) + ' × ' + rlN(Vx[s2])) +
                  ')' +
                  (O.length > 1 ? '' : '   [lands in ' + G.name(s2) + ']'),
              ).join(' + ') +
              ' = ' +
              rlN(v),
          );
        }
        return { L, b, bv };
      };
      const cell = () => {
        if (fin) return;
        const s = order[ci],
          r = lines(
            s,
            V,
            'Sweep ' +
              (sw + 1) +
              ', square <b>' +
              G.name(s) +
              '</b>. Its value before: ' +
              rlN(V[s]) +
              '. Each move is worth reward + γ × value of where it lands:',
          ),
          d = Math.abs(r.bv - V[s]);
        Vn[s] = r.bv;
        if (pol[s] !== r.b) {
          pch++;
          pol[s] = r.b;
        }
        dcur = Math.max(dcur, d);
        cur = s;
        r.L.push(
          'best: ' +
            GWS[r.b] +
            ' ' +
            GWN[r.b] +
            '.  New value of ' +
            G.name(s) +
            ' = <b>' +
            rlN(r.bv) +
            '</b>   (changed by ' +
            rlN(d) +
            ')',
        );
        eqShow(eq, r.L, 0);
        cu = 2;
        msg =
          'Square ' +
          G.name(s) +
          ': the best of its four moves is ' +
          GWN[r.b] +
          ', worth ' +
          rlN(r.bv) +
          '. Its value changed by ' +
          rlN(d) +
          '.';
        ci++;
        if (ci >= order.length) endSweep();
      };
      const endSweep = () => {
        for (const s of order) V[s] = Vn[s];
        sw++;
        DL.push(dcur);
        PCH.push(pch);
        const d = dcur,
          pc = pch;
        dcur = 0;
        pch = 0;
        ci = 0;
        cu = 3;
        msg =
          'Sweep ' +
          sw +
          ' is complete: the largest change in any value was ' +
          f3(d) +
          ', and ' +
          pc +
          ' arrow' +
          (pc === 1 ? '' : 's') +
          ' changed.';
        if (d < 1e-3) {
          fin = true;
          msg += ' That is below 0.001, so the values have converged after ' + sw + ' sweeps.';
        }
      };
      const sweep = () => {
        if (fin) return;
        while (true) {
          const s = order[ci],
            [b, bv] = bestA(s, V);
          dcur = Math.max(dcur, Math.abs(bv - V[s]));
          Vn[s] = bv;
          if (pol[s] !== b) {
            pch++;
            pol[s] = b;
          }
          ci++;
          if (ci >= order.length) {
            cur = -1;
            endSweep();
            break;
          }
        }
      };
      const round = () => {
        if (fin) return;
        let n = 0;
        for (; n < 2000; n++) {
          let d = 0;
          for (const s of order) {
            const v = qa(s, pol[s], V);
            d = Math.max(d, Math.abs(v - V[s]));
            Vn[s] = v;
          }
          for (const s of order) V[s] = Vn[s];
          if (d < 1e-4) {
            n++;
            break;
          }
        }
        evs += n;
        const ch = [];
        for (const s of order) {
          const [b] = bestA(s, V),
            old = pol[s];
          if (qa(s, b, V) > qa(s, old, V) + 1e-9) {
            pol[s] = b;
            ch.push(G.name(s) + ' ' + GWS[old] + '→' + GWS[b]);
          }
        }
        sw++;
        DL.push(n);
        PCH.push(ch.length);
        cur = -1;
        cu = 4;
        eqShow(
          eq,
          [
            'Round ' + sw + ' of policy iteration.',
            '<b>Evaluate</b>: with the arrows held fixed, the values were swept ' +
              n +
              ' times until they stopped changing.',
            '<b>Improve</b>: every square looked at its four moves under those values. ' +
              ch.length +
              ' square' +
              (ch.length === 1 ? '' : 's') +
              ' found a better move than its arrow.',
            ch.length
              ? 'changed: ' + ch.slice(0, 10).join(',  ') + (ch.length > 10 ? ', …' : '')
              : 'No arrow changed, so these arrows are the best possible.',
          ],
          0,
        );
        msg =
          'Round ' +
          sw +
          ': evaluating the arrows took ' +
          n +
          ' sweeps, then ' +
          ch.length +
          ' arrow' +
          (ch.length === 1 ? '' : 's') +
          ' changed.';
        if (!ch.length) {
          fin = true;
          msg +=
            ' Policy iteration has converged after ' +
            sw +
            ' rounds and ' +
            evs +
            ' evaluation sweeps in total.';
        }
      };
      const show = () => {
        if (sel < 0 || G.wall(sel) || G.term(sel)) return;
        const r = lines(
          sel,
          V,
          'Square <b>' +
            G.name(sel) +
            '</b> with the values as they are now (' +
            rlN(V[sel]) +
            '). Each move is worth reward + γ × value of where it lands:',
        );
        r.L.push('best: ' + GWS[r.b] + ' ' + GWN[r.b] + ', worth <b>' + rlN(r.bv) + '</b>');
        eqShow(eq, r.L, 0);
      };
      const draw = () => {
        const t = T(),
          D = new Float64Array(V);
        if (S.m === 'vi') for (let i = 0; i < ci; i++) D[order[i]] = Vn[order[i]];
        {
          const { g, w, h } = fit(c);
          P = gwDraw(g, t, 6, 6, w - 12, h - 12, G, { V: D, pol, cur, sel });
        }
        {
          const { g, w, h } = fit(cr);
          if (S.m === 'vi') {
            uLine(g, t, 0, 0, w, h / 2, DL, {
              title: 'largest change in any value, per sweep',
              n: Math.max(12, DL.length),
              col: t.accent,
              dots: DL.length < 50,
              ym: Math.max(1, ...DL),
            });
            uLine(g, t, 0, h / 2, w, h / 2, PCH, {
              title: 'arrows that changed, per sweep',
              n: Math.max(12, PCH.length),
              col: t.ok,
              dots: PCH.length < 50,
              ym: Math.max(4, ...PCH),
            });
          } else {
            uLine(g, t, 0, 0, w, h / 2, DL, {
              title: 'evaluation sweeps needed, per round',
              n: Math.max(6, DL.length),
              col: t.accent,
              dots: true,
              ym: Math.max(10, ...DL),
            });
            uLine(g, t, 0, h / 2, w, h / 2, PCH, {
              title: 'arrows that changed, per round',
              n: Math.max(6, PCH.length),
              col: t.ok,
              dots: true,
              ym: Math.max(4, ...PCH),
            });
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText(S.m === 'vi' ? 'sweeps →' : 'rounds →', w - 12, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'vi', cu);
        let rt = '–';
        if (pol[G.start] >= 0) {
          const Qx = new Float64Array(G.n * 4);
          for (const s of order) if (pol[s] >= 0) Qx[s * 4 + pol[s]] = 1;
          const R = gwRoute(G, Qx, G.start);
          rt = R.ok
            ? R.acts.map((a) => GWS[a]).join(' ') + ' (' + R.acts.length + ' steps)'
            : R.loop
              ? 'goes round in a circle so far'
              : 'ends in the pit';
        }
        setR(root, 'me', S.m === 'vi' ? 'Value iteration' : 'Policy iteration');
        setR(
          root,
          'sw',
          S.m === 'vi'
            ? sw + (fin ? ' (converged)' : '')
            : sw + ' rounds, ' + evs + ' evaluation sweeps' + (fin ? ' (converged)' : ''),
        );
        setR(root, 'dl', S.m === 'vi' && DL.length ? f3(DL[DL.length - 1]) : '–');
        setR(root, 'vs', rlN(D[G.start]));
        setR(root, 'rt', rt);
        setR(root, 'pc', PCH.length ? String(PCH[PCH.length - 1]) : '–');
      };
      build();
      if (RM) {
        while (!fin) S.m === 'vi' ? sweep() : round();
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (S.m === 'vi') {
          if (sw < 2) {
            if (fr % 9) return;
            cell();
          } else {
            if (fr % 10) return;
            sweep();
          }
        } else {
          if (fr % 70) return;
          round();
        }
        draw();
        chk();
      });
      pb = playBtn($('#vi-play', root), anim);
      chk = runCtl(pb, root, 'vi', {
        done: () => fin,
        restart: () => build(),
        count: () => sw,
        what: 'sweeps',
      });
      $('#vi-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else S.m === 'vi' ? cell() : round();
        draw();
      };
      $('#vi-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      const re = () => {
        if (!pb) return;
        pb.stop();
        build();
        draw();
        pb.start();
      };
      segBind(root, 'vi-m', S.m, (v) => {
        S.m = v;
        re();
      });
      segBind(root, 'vi-w', S.w, (v) => {
        S.w = v;
        sel = -1;
        re();
      });
      bind(
        root,
        'vi-sl',
        (v) => {
          S.sl = +v;
          re();
        },
        f2,
      );
      bind(
        root,
        'vi-ga',
        (v) => {
          S.ga = +v;
          re();
        },
        f2,
      );
      bind(
        root,
        'vi-rs',
        (v) => {
          S.rs = +v;
          re();
        },
        f2,
      );
      gwClick(c, (x, y) => {
        if (!P) return;
        const q = P.at(x, y);
        if (q < 0 || G.wall(q) || G.term(q)) return;
        pb.stop();
        sel = q;
        show();
        draw();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          build();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'td-learning',
    tab: 'Monte Carlo and TD learning',
    title: 'Monte Carlo and TD learning: wait for the end, or learn as you go?',
    lede: 'Now the map is gone. A walker wanders at random along a row of five squares until it falls off one end: nothing on the left, a reward of 1 on the right. How good is it to be on each square? Two ways of finding out from experience are compared. Monte Carlo waits until the walk is over and then tells every square it visited how things turned out. Temporal-difference learning does not wait: after every single step it corrects one square using its current guess about the next. That small change is the central idea of reinforcement learning.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Top: the walk. Below: the value of each square. Short dark line: the true value. Left bar of each pair: TD’s estimate. Right bar: Monte Carlo’s</figcaption><canvas class="short" id="td-c" role="img" aria-label="Random walk and value estimates of each state by two methods"></canvas></figure>
 <figure class="fig"><figcaption>Top: the error of each method in this one run. Bottom: the same, averaged over 100 separate runs with the current α</figcaption><canvas class="short" id="td-r" role="img" aria-label="Estimation error of TD and Monte Carlo, in this run and on average"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The latest updates, worked out</figcaption><div class="eq" id="td-eq" aria-live="polite" style="min-height:10em;margin-top:6px"></div></figure>
 ${uHow('td', ['Every square starts with the guess 0.5. The walker starts in the middle, C.', 'It steps left or right with equal chance.', 'TD, after each step: nudge the value of the square just left toward reward + value of the square just entered.', 'Keep stepping until the walker falls off an end. The return of the walk is 1 if it left on the right and 0 if on the left.', 'Monte Carlo, now that the walk is over: nudge the value of every square that was visited toward that return.', 'Start a new walk from C. Repeat for 100 walks.'])}
 <div class="controls">${uBtns('td', 'Run', 'One step', 'walks')}
 ${grp('Learning', slider('td-al', 'Step size α for both methods', 0.01, 0.5, 0.01, 0.1) + slider('td-sp', 'Speed', 1, 3, 1, 2))}</div>
 <dl class="read">${rd('ep', 'Walks finished')}${rd('et', 'TD: error now', 'hl')}${rd('em', 'Monte Carlo: error now', 'hl2')}${rd('lw', 'Last walk ended')}${rd('ls', 'Steps in the last walk')}${rd('avg', 'Average error over walks 1 to 100, across 100 runs: TD / MC')}</dl>`,
    tryit:
      'Step through the first walk. After each step TD changes one bar straight away, by a small amount, while Monte Carlo does nothing. When the walk ends, Monte Carlo moves every visited square at once, all in the same direction. Which of the two is jumpier? Press Run and compare the two error curves, then look at the average of 100 runs below. Now change α. At 0.02 both learn slowly and smoothly. At 0.3 both are quick and noisy. Is there an α at which Monte Carlo beats TD in the averaged chart? Why can neither error reach zero with a fixed α?',
    math: String.raw`<h4>Intuition</h4><p>You drive home from work every day and want to predict how long the trip takes. One method: note the time when you leave, and when you get home, compare the real total with what you had predicted at each point along the way. That is Monte Carlo. Nothing is learned until you are on your own doorstep. The other method: you leave expecting 30 minutes, reach the motorway, and find it jammed. At that moment you already know the trip will be nearer 45. Why wait until you are home to correct your prediction for "just left work"? Correct it now, using your new and better guess. That is temporal-difference learning: learning a guess from a later guess.</p>
 <h4>What is being estimated</h4><p>For a fixed way of behaving, here walking at random, the value \(V(s)\) of a state is the expected total reward from that state onward. In this walk the only reward is 1 for leaving on the right, so \(V(s)\) is the probability of finishing on the right. The true values of the five squares A to E are \(\tfrac16,\tfrac26,\tfrac36,\tfrac46,\tfrac56\). Neither method is told this.</p>
 <h4>Monte Carlo</h4><p>Run a whole episode. For each state \(s_t\) visited, the <b>return</b> \(G_t\) is the total reward actually collected from that point to the end. Move the estimate toward it:</p>$$V(s_t)\leftarrow V(s_t)+\alpha\,\big[G_t-V(s_t)\big].$$<p>The target \(G_t\) is an honest sample of what the value is supposed to predict, so on average it is exactly right. It is also very noisy: here it is either 0 or 1, never the 0.5 that the middle square is really worth.</p>
 <h4>Temporal-difference learning, TD(0)</h4><p>After a single step from \(s_t\) to \(s_{t+1}\) with reward \(r_{t+1}\), use the current estimate of the next state in place of the rest of the episode (Sutton, 1988):</p>$$V(s_t)\leftarrow V(s_t)+\alpha\,\big[\underbrace{r_{t+1}+\gamma V(s_{t+1})}_{\text{TD target}}-V(s_t)\big].$$<p>The bracket is the TD error \(\delta_t\). The target leans on \(V(s_{t+1})\), which is itself only a guess, so it is biased while the guesses are poor. This use of an estimate to improve an estimate is called <b>bootstrapping</b>. In return the target varies much less than a full return, because it depends on one random step, not on a whole random walk.</p>
 <h4>The trade</h4><p>Monte Carlo: no bias, high variance, must wait for the end, and cannot be used at all if episodes never end. TD: some bias early on, low variance, learns at every step. On problems like this one TD usually reaches a given accuracy sooner, as the averaged chart shows for most values of \(\alpha\). The two are the ends of a range: \(n\)-step methods and TD(\(\lambda\)) look a few steps ahead before bootstrapping.</p>
 <h4>Why the error does not reach zero</h4><p>With a fixed \(\alpha\) every update moves the estimate by a fixed fraction toward a noisy target, so the estimates keep rattling around the true values for ever. A smaller \(\alpha\) rattles less and takes longer to arrive. Exact convergence needs \(\alpha\) to shrink over time.</p>
 <h4>What this page runs</h4><p>The five-state random walk from Sutton and Barto's book. Both methods see exactly the same walks. The error shown is the root-mean-square difference between the five estimates and the five true values. The bottom chart repeats the whole experiment 100 times with fresh walks and averages the curves, because one run is too noisy to judge. There is no discounting, \(\gamma=1\).</p>
 <h4>Where this leads</h4><p>Replace "value of a state" by "value of a move in a state" and let the learner choose its own moves, and TD learning becomes SARSA and Q-learning, the next two tabs.</p>`,
    q: 'The walker goes C → D → E → off the right edge, earning 1. Monte Carlo then raises the values of C, D and E. TD, during that same walk, may have lowered the value of C. How can that be, when the walk ended well?',
    a: 'TD updated C at the moment the walker stepped from C to D, long before the outcome was known. Its target was the reward for that step, 0, plus the current value of D. If D’s estimate happened to be lower than C’s at that moment, C was moved down. TD does not know how the walk ends. It only compares neighbouring guesses. The good news from the right edge reaches E at once, reaches D the next time a walk passes from D to E, and reaches C later still. Monte Carlo delivers the news to all three at once, along with all the luck in that one walk.',
    init(root) {
      const c = $('#td-c', root),
        cr = $('#td-r', root),
        live = $('#td-live', root),
        eq = $('#td-eq', root),
        S = { al: 0.1, sp: 2 },
        N = 5,
        TRUE = [1, 2, 3, 4, 5].map((v) => v / 6),
        MAXE = 100,
        NM = 'ABCDE';
      let VT,
        VM,
        pos,
        path,
        ep = 0,
        ET = [],
        EM = [],
        AT = [],
        AM = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        lastEnd = '–',
        lastN = 0,
        eqT = [],
        eqM = [];
      const rms = (V) => Math.sqrt(V.reduce((s, v, i) => s + (v - TRUE[i]) ** 2, 0) / N);
      const batch = () => {
        AT = new Array(MAXE + 1).fill(0);
        AM = new Array(MAXE + 1).fill(0);
        const R = 100;
        for (let r = 0; r < R; r++) {
          const a = new Array(N).fill(0.5),
            b = new Array(N).fill(0.5);
          AT[0] += rms(a) / R;
          AM[0] += rms(b) / R;
          for (let e = 1; e <= MAXE; e++) {
            let p = 2;
            const vis = [];
            for (;;) {
              vis.push(p);
              const q = p + (Math.random() < 0.5 ? -1 : 1),
                rw = q >= N ? 1 : 0,
                nv = q < 0 || q >= N ? 0 : a[q];
              a[p] += S.al * (rw + nv - a[p]);
              p = q;
              if (p < 0 || p >= N) break;
            }
            const G0 = p >= N ? 1 : 0;
            for (const s of vis) b[s] += S.al * (G0 - b[s]);
            AT[e] += rms(a) / R;
            AM[e] += rms(b) / R;
          }
        }
      };
      const build = () => {
        VT = new Array(N).fill(0.5);
        VM = new Array(N).fill(0.5);
        pos = 2;
        path = [2];
        ep = 0;
        ET = [rms(VT)];
        EM = [rms(VM)];
        fin = false;
        cu = 0;
        lastEnd = '–';
        lastN = 0;
        eqT = [];
        eqM = [];
        batch();
        msg =
          'Both methods start by guessing 0.5 for every square. The walker is on C. Press One step or Run.';
        eqShow(eq, ['No steps yet.'], 0);
      };
      const step = (quiet) => {
        if (fin) return;
        const p = pos,
          q = p + (Math.random() < 0.5 ? -1 : 1),
          off = q < 0 || q >= N,
          rw = q >= N ? 1 : 0,
          nv = off ? 0 : VT[q],
          old = VT[p],
          nw = old + S.al * (rw + nv - old);
        VT[p] = nw;
        eqT = [
          '<b>TD</b>, step ' +
            path.length +
            ' of walk ' +
            (ep + 1) +
            ': from ' +
            NM[p] +
            ' to ' +
            (off ? (q < 0 ? 'off the left edge' : 'off the right edge') : NM[q]) +
            ', reward ' +
            rw +
            '.',
          '   target = reward + value of next = ' +
            rw +
            ' + ' +
            (off ? '0 (the walk is over)' : f3(nv)) +
            ' = ' +
            f3(rw + nv),
          '   V(' +
            NM[p] +
            ') = ' +
            f3(old) +
            ' + ' +
            f2(S.al) +
            ' × (' +
            f3(rw + nv) +
            ' − ' +
            f3(old) +
            ') = <b>' +
            f3(nw) +
            '</b>',
        ];
        pos = q;
        cu = 2;
        if (!off) {
          path.push(q);
          eqM = ['<b>Monte Carlo</b>: waiting. It cannot update anything until the walk ends.'];
          msg =
            'The walker stepped from ' +
            NM[p] +
            ' to ' +
            NM[q] +
            '. TD moved V(' +
            NM[p] +
            ') from ' +
            f3(old) +
            ' to ' +
            f3(nw) +
            '. Monte Carlo is still waiting.';
        } else {
          const G0 = rw,
            vis = path.slice(),
            ch = [];
          for (const s of vis) {
            const o = VM[s];
            VM[s] += S.al * (G0 - VM[s]);
            if (ch.length < 4)
              ch.push(
                'V(' +
                  NM[s] +
                  ') = ' +
                  f3(o) +
                  ' + ' +
                  f2(S.al) +
                  ' × (' +
                  G0 +
                  ' − ' +
                  f3(o) +
                  ') = <b>' +
                  f3(VM[s]) +
                  '</b>',
              );
          }
          eqM = [
            '<b>Monte Carlo</b>: the walk ended ' +
              (G0 ? 'on the right, so the return is 1' : 'on the left, so the return is 0') +
              ' for all ' +
              vis.length +
              ' squares visited.',
          ].concat(
            ch.map((x) => '   ' + x),
            vis.length > 4 ? ['   … and ' + (vis.length - 4) + ' more in the same way'] : [],
          );
          ep++;
          ET.push(rms(VT));
          EM.push(rms(VM));
          lastEnd = G0 ? 'on the right (reward 1)' : 'on the left (reward 0)';
          lastN = vis.length;
          cu = 4;
          msg =
            'Walk ' +
            ep +
            ' ended ' +
            lastEnd +
            ' after ' +
            lastN +
            ' steps. Monte Carlo now updated every visited square toward ' +
            G0 +
            '. Error: TD ' +
            f3(ET[ep]) +
            ', Monte Carlo ' +
            f3(EM[ep]) +
            '.';
          pos = 2;
          path = [2];
          if (ep >= MAXE) {
            fin = true;
            msg =
              'Finished: 100 walks. Error of TD ' +
              f3(ET[ep]) +
              ', of Monte Carlo ' +
              f3(EM[ep]) +
              ' in this run. Averaged over 100 runs the errors after 100 walks are ' +
              f3(AT[MAXE]) +
              ' and ' +
              f3(AM[MAXE]) +
              '. With a fixed α neither settles exactly.';
          }
        }
        if (!quiet) eqShow(eq, eqT.concat(eqM), 0);
      };
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c),
            m = 30,
            cw = (w - 2 * m) / (N + 2),
            y1 = 54,
            X = (i) => m + (i + 1.5) * cw;
          g.font = MONO;
          g.strokeStyle = t.muted;
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(X(-1), y1);
          g.lineTo(X(N), y1);
          g.stroke();
          for (let i = -1; i <= N; i++) {
            const end = i < 0 || i >= N;
            g.beginPath();
            if (end) g.rect(X(i) - 15, y1 - 15, 30, 30);
            else g.arc(X(i), y1, 17, 0, 7);
            g.fillStyle = end
              ? i < 0
                ? t.bg
                : mix(t.gold, t.gold, 0, 0.6)
              : path.includes(i)
                ? mix(t.accent, t.accent, 0, 0.25)
                : t.bg;
            g.fill();
            g.strokeStyle = t.ink;
            g.stroke();
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.font = '600 13px ' + MONO.split('px ')[1];
            g.fillText(end ? (i < 0 ? '0' : '+1') : NM[i], X(i), y1 + 5);
          }
          g.fillStyle = t.ink;
          g.beginPath();
          g.arc(X(pos), y1 - 30, 7, 0, 7);
          g.fill();
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('walker', X(pos), y1 - 42);
          const y0 = h - 44,
            H = h - y1 - 100,
            Y = (v) => y0 - v * H;
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          for (const v of [0, 0.5, 1]) {
            g.beginPath();
            g.moveTo(X(0) - cw / 2, Y(v));
            g.lineTo(X(N - 1) + cw / 2, Y(v));
            g.stroke();
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText(f2(v), X(0) - cw / 2 - 4, Y(v) + 4);
          }
          for (let i = 0; i < N; i++) {
            const bw = cw * 0.26;
            g.fillStyle = t.accent;
            g.fillRect(X(i) - bw - 2, Y(VT[i]), bw, Y(0) - Y(VT[i]));
            g.fillStyle = t.cut;
            g.fillRect(X(i) + 2, Y(VM[i]), bw, Y(0) - Y(VM[i]));
            g.strokeStyle = t.ink;
            g.lineWidth = 2.5;
            g.beginPath();
            g.moveTo(X(i) - bw - 8, Y(TRUE[i]));
            g.lineTo(X(i) + bw + 8, Y(TRUE[i]));
            g.stroke();
            g.textAlign = 'center';
            g.fillStyle = t.ink;
            g.fillText(NM[i], X(i), y0 + 13);
            g.fillStyle = t.muted;
            g.fillText('true ' + f2(TRUE[i]), X(i), y0 + 25);
            g.fillText(f2(VT[i]) + ' | ' + f2(VM[i]), X(i), y0 + 37);
          }
          g.textAlign = 'left';
          g.fillStyle = t.accent;
          g.fillText('TD', X(0) - cw / 2, y1 + 38);
          g.fillStyle = t.cut;
          g.fillText('Monte Carlo', X(0) - cw / 2 + 30, y1 + 38);
        }
        {
          const { g, w, h } = fit(cr),
            two = (y0, H, a, b, title) => {
              const L = uLine(g, t, 0, y0, w, H, a, { title, n: MAXE + 1, ym: 0.3, col: t.accent });
              g.strokeStyle = t.cut;
              g.lineWidth = 2.2;
              g.beginPath();
              b.forEach((v, i) =>
                i
                  ? g.lineTo(L.X(i), L.Y(Math.min(v, 0.3)))
                  : g.moveTo(L.X(i), L.Y(Math.min(v, 0.3))),
              );
              g.stroke();
              g.font = MONO;
              g.textAlign = 'right';
              g.fillStyle = t.accent;
              g.fillText('TD', w - 14, y0 + 38);
              g.fillStyle = t.cut;
              g.fillText('Monte Carlo', w - 14, y0 + 52);
            };
          two(
            0,
            h / 2,
            ET.map((v) => Math.min(v, 0.3)),
            EM,
            'error of the five estimates, this run',
          );
          two(h / 2, h / 2, AT, AM, 'the same, averaged over 100 runs with α = ' + f2(S.al));
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('walks →', w - 12, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'td', cu);
        const mean = (a) => a.slice(1).reduce((x, y) => x + y, 0) / MAXE;
        setR(root, 'ep', ep + ' of ' + MAXE);
        setR(root, 'et', f3(rms(VT)));
        setR(root, 'em', f3(rms(VM)));
        setR(root, 'lw', lastEnd);
        setR(root, 'ls', lastN ? String(lastN) : '–');
        setR(root, 'avg', f3(mean(AT)) + ' / ' + f3(mean(AM)));
      };
      build();
      if (RM) {
        while (!fin) step(true);
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (S.sp === 1) {
          if (fr % 30) return;
          step(false);
        } else if (S.sp === 2) {
          if (ep < 3) {
            if (fr % 8) return;
            step(false);
          } else {
            if (fr % 3) return;
            const e0 = ep;
            while (ep === e0 && !fin) step(false);
          }
        } else {
          const e0 = ep;
          while (ep < e0 + 2 && !fin) step(false);
        }
        draw();
        chk();
      });
      pb = playBtn($('#td-play', root), anim);
      chk = runCtl(pb, root, 'td', {
        done: () => fin,
        restart: () => build(),
        count: () => ep,
        what: 'walks',
      });
      $('#td-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else step(false);
        draw();
      };
      $('#td-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      bind(
        root,
        'td-al',
        (v) => {
          S.al = +v;
          if (!pb) return;
          pb.stop();
          build();
          draw();
        },
        f2,
      );
      bind(
        root,
        'td-sp',
        (v) => {
          S.sp = +v;
        },
        (v) => ['', 'one step at a time', 'steady', 'fast'][+v],
      );
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          build();
          draw();
          pb.start();
        },
      };
    },
  },
  {
    id: 'q-learning',
    tab: 'Q-learning',
    title: 'Q-learning: fill in the table, one step at a time',
    lede: 'A robot is dropped into a room it has never seen. Nobody tells it where the goal is, where the pit is, or what its moves do. All it gets is a number after every step: −1 for an ordinary step, +10 for reaching the goal, −10 for falling into the pit. Q-learning turns that trickle of numbers into a table that says, for every square and every move, how good that move is. On this page you can watch every single entry of the table being computed, with the arithmetic written out.',
    stage: `<div class="row seg" id="ql-w" role="group" aria-label="Which world"><button class="btn" type="button" data-v="hall">The Hallway: 5 Squares in a Row</button><button class="btn" type="button" data-v="room">The Room with a Pit</button><button class="btn" type="button" data-v="maze">The Maze</button></div>
 <div class="stage"><figure class="fig"><figcaption>The world. The dot is the robot. Every square is cut into four triangles, one for each move, holding the current value of making that move from that square. The best move is in bold. Click any square to ask the table about it</figcaption><canvas class="short" id="ql-c" role="img" aria-label="Grid world with the value of each move in each square"></canvas></figure>
 <figure class="fig"><figcaption>The Q-table: one row for every square, one column for every move. The gold cell was just updated. Bold is the best move in that row</figcaption><div class="ctab qtab" id="ql-tab"></div></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The last update, worked out in full</figcaption><div class="eq" id="ql-eq" aria-live="polite" style="min-height:11.5em;margin-top:6px"></div></figure>
 <div class="stage" style="margin-top:22px"><figure class="fig"><figcaption>Left: the total reward collected in each episode. Right: how far the table still is from the true values, on average</figcaption><canvas class="wide" id="ql-r" role="img" aria-label="Reward per episode and distance of the table from the true values"></canvas></figure>
 <div><p class="tpout" id="ql-tpo" role="status" style="margin-top:26px"></p></div></div>
 ${uHow('ql', ['Start with a table of zeros: the robot knows nothing. Put it on the start square.', 'Choose a move. Most of the time take the move with the highest value in the current row. With probability ε pick a random move instead, to explore.', 'Make the move. See which square you land in and what reward you get.', 'Look along the row of the new square and take its highest value: the best that can be expected from there.', 'Target = reward + γ × that best value. Move the old table entry a fraction α of the way toward the target.', 'If the episode has ended, go back to the start square. Repeat from step 2 until the table stops changing.'])}
 <div class="controls">${uBtns('ql', 'Run', 'One step', 'episodes')}
 ${grp('Speed and episodes', slider('ql-sp', 'Speed', 1, 4, 1, 2) + `<div class="row"><button class="btn" id="ql-ep" type="button">Finish This Episode</button></div>`)}
 ${grp('Learning', slider('ql-al', 'Learning rate α: how far to move toward the target', 0.05, 1, 0.05, 0.5) + slider('ql-ga', 'Discount γ: how much the future counts', 0, 1, 0.05, 0.9) + slider('ql-ep2', 'Exploration ε: share of random moves', 0, 1, 0.05, 0.2))}
 ${grp('The floor', `<div class="row seg" id="ql-sl" role="group" aria-label="Floor"><button class="btn" type="button" data-v="0">Firm: Moves Always Work</button><button class="btn" type="button" data-v="0.2">Slippery: 1 Move in 5 Slides Sideways</button></div>`)}</div>
 <dl class="read">${rd('ep', 'Episodes finished')}${rd('st', 'Steps in this episode')}${rd('lr', 'Reward of the last finished episode')}${rd('rt', 'Best route from the start, according to the table', 'hl')}${rd('ok', 'Entries within 0.1 of their true value', 'hl2')}${rd('dm', 'Largest change in the table over the last 25 episodes')}</dl>`,
    tryit:
      'Start with The Hallway and press One step again and again, reading the worked update each time. The first time the robot reaches the goal, exactly one entry becomes positive. Which one? In the next episode, watch that value leak one square further back. That backward leak is the whole algorithm. Now choose The Room with a Pit and press Run. When it stops, click squares to ask the table for the best route. Then experiment: set ε to 0 and start over. Does the robot still find the best route, and why might it not? Set γ to 0: what can the robot no longer see? Set α to 1 on the slippery floor and watch the values refuse to settle.',
    math: String.raw`<h4>Intuition</h4><p>Imagine learning your way round a new city with no map, by keeping a notebook. On each page is one street corner, and on that page four numbers: how good it has turned out to be to go north, east, south or west from there. At first every number is zero. You walk. Each time you take a step you look at the page for the corner you have just arrived at, find its best number, and use it to correct the number for the step you just took: "going east from the bakery got me to the fountain, and the fountain is a good place to be, so east from the bakery is good too". You never need to know the whole route. Good news spreads backwards through the notebook, one corner at a time, from the places where something actually happened.</p>
 <h4>What the table means</h4><p>\(Q(s,a)\) is an estimate of the total reward the robot will collect from now on if it is in square \(s\), makes move \(a\), and behaves as well as possible afterwards. Rewards further in the future are counted a little less, by a factor \(\gamma\) for each step of delay:</p>$$Q^\star(s,a)=\mathbb E\big[r_1+\gamma r_2+\gamma^2r_3+\dots\ \big|\ s,a,\ \text{then act optimally}\big].$$<p>If the table were correct, behaving well would be easy: in every square, make the move with the highest value. Learning the table is the whole problem.</p>
 <h4>The update</h4><p>The robot is in \(s\), makes move \(a\), receives reward \(r\) and lands in \(s'\). It then changes one entry of the table:</p>$$Q(s,a)\ \leftarrow\ Q(s,a)+\alpha\Big[\underbrace{r+\gamma\max_{a'}Q(s',a')}_{\text{target}}-Q(s,a)\Big].$$<p>The <b>target</b> is a better guess of what \(Q(s,a)\) should be, made one step later: the reward that actually arrived plus the best value the table currently offers from the new square. The bracket is the <b>temporal-difference error</b>: how much better or worse things turned out than the table said. The learning rate \(\alpha\) decides how far to move: with \(\alpha=1\) the old value is thrown away, with \(\alpha=0.1\) it moves a tenth of the way. If \(s'\) ends the episode there is no future, and the target is \(r\) alone. The panel under the pictures writes these lines out with the numbers of the step just taken (Watkins, 1989).</p>
 <h4>A worked example</h4><p>In the hallway with \(\alpha=0.5\) and \(\gamma=0.9\), suppose the robot stands on the square next to the goal, with the whole table at zero, and steps right. It reaches the goal and receives \(+10\). The goal ends the episode, so</p>$$Q(\text{A4},\rightarrow)\leftarrow0+0.5\,[\,10-0\,]=5 .$$<p>In a later episode it steps right from A3 into A4, with reward \(-1\). The best value now on offer in A4 is 5, so</p>$$Q(\text{A3},\rightarrow)\leftarrow0+0.5\,[\,-1+0.9\times5-0\,]=1.75 .$$<p>The reward at the goal has reached a square from which the goal cannot be seen. After enough visits the values settle at \(10\), \(8\), \(6.2\), \(4.58\) going back from the goal: each is \(-1+0.9\times\) the one in front of it.</p>
 <h4>Explore or exploit</h4><p>Always taking the move that looks best is called exploiting. A robot that only exploits never tries the moves it wrongly believes are bad, so it can settle on a poor route for ever. With <b>ε-greedy</b> behaviour it makes a random move with probability \(\varepsilon\) and the best-looking move otherwise. Every entry of the table then keeps being tested.</p>
 <h4>Why it learns the best behaviour while behaving imperfectly</h4><p>The target uses \(\max_{a'}Q(s',a')\), the value of the best next move, whatever move the robot actually makes next. So the table describes the best possible behaviour even while the robot is stumbling around at random one step in five. This is called <b>off-policy</b> learning, and it is the difference from SARSA in the next tab. Provided every square and move keeps being visited and \(\alpha\) is reduced gradually, the table is guaranteed to converge to \(Q^\star\).</p>
 <h4>The three settings</h4><p><b>\(\alpha\)</b>: large values learn fast and are thrown about by luck when the floor is slippery. Small values are steady and slow. <b>\(\gamma\)</b>: at 0 only the immediate reward counts, and since every ordinary step pays \(-1\), the robot cannot tell a step toward the goal from a step away. Near 1, distant rewards count almost fully. <b>\(\varepsilon\)</b>: at 0 nothing is explored. At 1 the robot moves at random, which still fills in the table correctly but collects very little reward on the way.</p>
 <h4>How this page decides it has finished</h4><p>The true values are known here, because the page also solves the world exactly with the method of the Value and Policy Iteration tab. They are used only to draw the right-hand chart. Learning stops when no entry has changed by more than 0.01 during 25 episodes in a row, or after 400 episodes. On the slippery floor with a fixed \(\alpha\) the entries never stop jiggling, so the limit is what ends the run.</p>
 <h4>Where the table stops being enough</h4><p>A table needs one row for every situation. A 3 by 4 room has 11. Chess has more positions than there are atoms in the Earth. For such problems the table is replaced by a neural network that takes the situation as input and outputs the values of the moves, trained on the same target. That is the deep Q-network in the last tab of this module.</p>`,
    q: 'After the very first episode in The Room with a Pit, almost every entry the robot touched is negative, even on squares that lie on the best route. Why, and why does that not ruin the learning?',
    a: 'During the first episode nothing in the table knows about the goal yet, so every ordinary step is updated toward a target of −1 plus γ times zero. The squares along the route are therefore marked as slightly bad. Only the very last step, into the goal, gets a positive value. In later episodes that positive value is passed back one square per visit, and it soon outweighs the small negative numbers. The early negative values even help: they make squares the robot has already tried look slightly worse than squares it has not, which pushes it to try new moves.',
    init(root) {
      const c = $('#ql-c', root),
        cr = $('#ql-r', root),
        live = $('#ql-live', root),
        eq = $('#ql-eq', root),
        tab = $('#ql-tab', root),
        S = { w: 'room', al: 0.5, ga: 0.9, eps: 0.2, slip: 0, sp: 2 },
        W = {
          hall: ['S...G'],
          room: ['...G', '.#.X', 'S...'],
          maze: ['S..#..', '.#...#', '.#.#..', '...#.X', '.#...G'],
        },
        MAXE = 400,
        CAP = 200;
      let G,
        Q,
        Qs,
        s,
        ep = 0,
        stp = 0,
        epR = 0,
        RET = [],
        ERR = [],
        DM = [],
        dmx = 0,
        trail = [],
        last = null,
        sel = -1,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        P = null,
        tv = 0,
        tvd = -1,
        best0 = NaN,
        lastSeq = 0,
        lastL = null;
      const gap = () => {
        let e = 0,
          n = 0;
        for (const q of G.states) {
          if (G.term(q)) continue;
          for (let a = 0; a < 4; a++) {
            e += Math.abs(Q[q * 4 + a] - Qs[q * 4 + a]);
            n++;
          }
        }
        return e / n;
      };
      const okShare = () => {
        let k = 0,
          n = 0;
        for (const q of G.states) {
          if (G.term(q)) continue;
          for (let a = 0; a < 4; a++) {
            if (Math.abs(Q[q * 4 + a] - Qs[q * 4 + a]) <= 0.1) k++;
            n++;
          }
        }
        return k + ' of ' + n;
      };
      const truth = () => {
        Qs = gwQstar(G, S.ga);
        const R = gwRoute(gwMake(G.rows, { slip: 0 }), Qs, G.start);
        best0 = R.ok ? R.ret : NaN;
      };
      const build = () => {
        G = gwMake(W[S.w], { slip: S.slip });
        Q = new Float64Array(G.n * 4);
        truth();
        s = G.start;
        ep = 0;
        stp = 0;
        epR = 0;
        RET = [];
        ERR = [gap()];
        DM = [];
        dmx = 0;
        trail = [s];
        last = null;
        fin = false;
        cu = 0;
        tv++;
        sel = -1;
        msg =
          'The table is all zeros and the robot is on the start square, ' +
          G.name(s) +
          '. It knows nothing about this world. Press One step to follow each update, or Run.';
        eqShow(
          eq,
          [
            'Nothing has been updated yet.',
            'Press <b>One step</b>: the robot makes one move and one entry of the table is recomputed here, line by line.',
          ],
          0,
        );
      };
      const step = (quiet) => {
        if (fin) return;
        const row = [0, 1, 2, 3].map((a) => Q[s * 4 + a]),
          mx = Math.max(...row),
          explore = Math.random() < S.eps;
        let a,
          ties = [0, 1, 2, 3].filter((k) => row[k] >= mx - 1e-12);
        if (explore) a = Math.floor(Math.random() * 4);
        else a = ties[Math.floor(Math.random() * ties.length)];
        const o = G.step(s, a),
          old = Q[s * 4 + a],
          s2 = o.s2,
          nx = o.done ? 0 : gwMax(Q, s2),
          tg = o.r + S.ga * nx,
          nw = old + S.al * (tg - old);
        Q[s * 4 + a] = nw;
        dmx = Math.max(dmx, Math.abs(nw - old));
        stp++;
        epR += o.r;
        trail.push(s2);
        last = { s, a, s2, old, nw, ep: ep + 1, stp };
        tv++;
        {
          const L = [
            'Episode ' +
              (ep + 1) +
              ', step ' +
              stp +
              '. The robot is in <b>' +
              G.name(s) +
              '</b>. Its row of the table: ' +
              row.map((v, k) => GWS[k] + ' ' + rlN(v)).join('   '),
            explore
              ? 'It rolled a random number below ε = ' +
                f2(S.eps) +
                ', so it <b>explores</b>: a random move, ' +
                GWS[a] +
                ' ' +
                GWN[a] +
                '.'
              : 'It <b>exploits</b>: the highest value in the row is ' +
                rlN(mx) +
                (ties.length > 1
                  ? ', shared by ' + ties.length + ' moves, so one of them is picked at random: '
                  : ', which is ') +
                GWS[a] +
                ' ' +
                GWN[a] +
                '.',
            (o.slipped ? 'The floor was slippery and it slid sideways. ' : '') +
              (o.bump
                ? 'It bumped into a wall and stayed in ' + G.name(s2)
                : 'It lands in ' + G.name(s2)) +
              '. Reward: <b>' +
              rlN(o.r).replace('.00', '') +
              '</b>.',
            o.done
              ? G.name(s2) + ' ends the episode, so there is no future to add: best next value = 0'
              : 'Best value on offer from ' +
                G.name(s2) +
                ':  max( ' +
                [0, 1, 2, 3]
                  .map(
                    (k) => GWS[k] + ' ' + rlN(Q[s2 * 4 + k] - (s2 === s && k === a ? nw - old : 0)),
                  )
                  .join(',  ') +
                ' ) = ' +
                rlN(nx),
            'target = reward + γ × best next = ' +
              rlN(o.r) +
              ' + ' +
              f2(S.ga) +
              ' × ' +
              rlN(nx) +
              ' = <b>' +
              rlN(tg) +
              '</b>',
            'new Q(' +
              G.name(s) +
              ', ' +
              GWS[a] +
              ') = old + α × (target − old) = ' +
              rlN(old) +
              ' + ' +
              f2(S.al) +
              ' × (' +
              rlN(tg) +
              ' − ' +
              rlN(old) +
              ') = <b>' +
              rlN(nw) +
              '</b>',
          ];
          lastL = L;
          if (!quiet) eqShow(eq, L, 0);
          cu = o.done ? 5 : 4;
          msg =
            'The entry for ' +
            GWS[a] +
            ' ' +
            GWN[a] +
            ' in row ' +
            G.name(s) +
            ' changed from ' +
            rlN(old) +
            ' to ' +
            rlN(nw) +
            '.' +
            (o.done
              ? (G.ch(o.raw) === 'G'
                  ? ' The robot reached the goal.'
                  : ' The robot fell into the pit.') +
                ' Episode ' +
                (ep + 1) +
                ' is over with a total reward of ' +
                rlN(epR).replace('.00', '') +
                '. Back to the start.'
              : '');
        }
        s = s2;
        if (o.done || stp >= CAP) {
          RET.push(epR);
          ERR.push(gap());
          DM.push(dmx);
          ep++;
          const lastR = epR;
          dmx = 0;
          stp = 0;
          epR = 0;
          s = G.start;
          trail = [s];
          const w = DM.slice(-25);
          if (ep >= MAXE || (DM.length >= 25 && Math.max(...w) < 0.01)) {
            fin = true;
            cu = 5;
            const R = gwRoute(G, Q, G.start);
            msg =
              (ep >= MAXE
                ? 'Stopped at the limit of ' +
                  MAXE +
                  ' episodes: the table is still changing by up to ' +
                  f2(Math.max(...w)) +
                  ' per episode' +
                  (S.slip ? ', which is what a slippery floor and a fixed α produce' : '') +
                  '.'
                : 'The table has stopped changing: for 25 episodes in a row no entry moved by more than 0.01. Finished after ' +
                  ep +
                  ' episodes.') +
              ' ' +
              okShare() +
              ' entries are within 0.1 of their true values' +
              (S.slip
                ? '. A smaller α would steady them. '
                : ': the ones still off belong to moves the robot rarely makes. ') +
              (R.ok
                ? 'Following the bold moves from the start reaches the goal in ' +
                  R.acts.length +
                  ' steps.'
                : 'Following the bold moves from the start does not reach the goal.');
          }
        }
      };
      const tableHTML = (t) => {
        let h =
          '<table><thead><tr><th>square</th>' +
          GWS.map((x, k) => '<th>' + x + ' ' + GWN[k] + '</th>').join('') +
          '<th>best move</th></tr></thead><tbody>';
        for (const q of G.states) {
          if (G.term(q)) {
            h +=
              '<tr><td>' +
              G.name(q) +
              '</td><td colspan="5" style="color:var(--muted)">' +
              (G.ch(q) === 'G'
                ? 'goal: the episode ends, +' + G.rg
                : 'pit: the episode ends, −' + Math.abs(G.rp)) +
              '</td></tr>';
            continue;
          }
          const b = gwBest(Q, q),
            any = [0, 1, 2, 3].some((a) => Q[q * 4 + a] !== 0);
          h +=
            '<tr' +
            (q === sel ? ' style="outline:2px dashed var(--ink)"' : '') +
            '><td>' +
            (q === s ? '● ' : '') +
            G.name(q) +
            '</td>' +
            [0, 1, 2, 3]
              .map((a) => {
                const hot = last && last.s === q && last.a === a && lastSeq === tv;
                return (
                  '<td' +
                  (hot ? ' style="background:' + mix(t.gold, t.gold, 0, 0.45) + '"' : '') +
                  '>' +
                  (any && a === b ? '<b>' + rlN(Q[q * 4 + a]) + '</b>' : rlN(Q[q * 4 + a])) +
                  '</td>'
                );
              })
              .join('') +
            '<td>' +
            (any ? GWS[b] + ' ' + GWN[b] : '–') +
            '</td></tr>';
        }
        return h + '</tbody></table>';
      };
      const ask = () => {
        const el = $('#ql-tpo', root);
        if (sel < 0 || G.wall(sel)) {
          el.textContent = 'Click a square in the picture to ask the table about it.';
          return;
        }
        if (G.term(sel)) {
          el.textContent = G.name(sel) + ' ends the episode. Nothing is learned for it.';
          return;
        }
        const b = gwBest(Q, sel),
          any = [0, 1, 2, 3].some((a) => Q[sel * 4 + a] !== 0),
          R = gwRoute(G, Q, sel),
          tb = gwBest(Qs, sel);
        el.textContent = !any
          ? 'Square ' +
            G.name(sel) +
            ': every entry is still 0. The robot has not learned anything here yet.'
          : 'Square ' +
            G.name(sel) +
            ': the table says the best move is ' +
            GWS[b] +
            ' ' +
            GWN[b] +
            ', worth ' +
            rlN(Q[sel * 4 + b]) +
            '. ' +
            (R.ok
              ? 'Following the bold moves from here reaches the goal in ' +
                R.acts.length +
                ' step' +
                (R.acts.length > 1 ? 's' : '') +
                ' (' +
                R.acts.map((a) => GWS[a]).join(' ') +
                ') for a total reward of ' +
                rlN(R.ret).replace('.00', '') +
                '.'
              : R.loop
                ? 'Following the bold moves from here goes round in a circle: the table is not good enough yet.'
                : 'Following the bold moves from here ends in the pit.') +
            ' The true value of the best move here is ' +
            rlN(Qs[sel * 4 + tb]) +
            ' (' +
            GWS[tb] +
            ' ' +
            GWN[tb] +
            ').';
      };
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c);
          P = gwDraw(g, t, 6, 6, w - 12, h - 12, G, {
            Q,
            agent: s,
            trail,
            sel,
            hot: last && lastSeq === tv ? [last.s, last.a] : null,
          });
        }
        {
          const { g, w, h } = fit(cr),
            n = Math.max(30, RET.length),
            lo = Math.min(-20, ...RET.slice(-n)),
            A = uLine(
              g,
              t,
              0,
              0,
              w / 2,
              h,
              RET.map((v) => Math.max(v, lo)),
              {
                title: 'reward of each episode',
                n,
                y1: lo,
                ym: Math.max(10, isFinite(best0) ? best0 : 10),
                col: t.accent,
              },
            );
          if (isFinite(best0)) {
            g.setLineDash([5, 4]);
            g.strokeStyle = t.gold;
            g.lineWidth = 1.3;
            g.beginPath();
            g.moveTo(A.m.l, A.Y(best0));
            g.lineTo(w / 2 - A.m.r, A.Y(best0));
            g.stroke();
            g.setLineDash([]);
            g.fillStyle = t.cut;
            g.textAlign = 'right';
            g.fillText(
              'dashed: best possible' + (S.slip ? ' without sliding' : ''),
              w / 2 - A.m.r,
              h - A.m.b - 8,
            );
          }
          uLine(g, t, w / 2, 0, w / 2, h, ERR, {
            title: 'average gap: table against true values',
            n: n + 1,
            col: t.ok,
            ym: Math.max(0.5, ERR[0]),
          });
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('episodes →', w - 12, h - 5);
        }
        if (tvd !== tv) {
          tvd = tv;
          tab.innerHTML = tableHTML(t);
          ask();
        }
        live.textContent = msg;
        uCur(root, 'ql', cu);
        const R = gwRoute(G, Q, G.start),
          w25 = DM.slice(-25);
        setR(root, 'ep', ep + (fin ? ' (stopped)' : ''));
        setR(root, 'st', String(stp));
        setR(root, 'lr', RET.length ? rlN(RET[RET.length - 1]).replace('.00', '') : '–');
        setR(
          root,
          'rt',
          R.ok
            ? R.acts.map((a) => GWS[a]).join(' ') + ' (' + R.acts.length + ' steps)'
            : 'not found yet',
        );
        setR(root, 'ok', okShare());
        setR(root, 'dm', w25.length ? f3(Math.max(...w25)) : '–');
      };
      const one = (q) => {
        step(q);
        lastSeq = tv;
      };
      build();
      if (RM) {
        while (!fin) step(true);
        lastSeq = -1;
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (S.sp === 1) {
          if (fr % 34) return;
          one(false);
        } else if (S.sp === 2) {
          if (fr % 7) return;
          one(false);
        } else if (S.sp === 3) {
          for (let i = 0; i < 4 && !fin; i++) one(true);
          if (lastL) eqShow(eq, lastL, 0);
        } else {
          const e0 = ep;
          while (ep < e0 + 2 && !fin) one(true);
          if (lastL) eqShow(eq, lastL, 0);
        }
        draw();
        chk();
      });
      pb = playBtn($('#ql-play', root), anim);
      chk = runCtl(pb, root, 'ql', {
        done: () => fin,
        restart: () => build(),
        count: () => ep,
        what: 'episodes',
      });
      $('#ql-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else one(false);
        draw();
      };
      $('#ql-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      $('#ql-ep', root).onclick = () => {
        pb.stop();
        if (fin) {
          draw();
          return;
        }
        const e0 = ep;
        let k = 0;
        while (ep === e0 && !fin && k++ < CAP + 2) one(false);
        draw();
      };
      segBind(root, 'ql-w', S.w, (v) => {
        S.w = v;
        pb.stop();
        build();
        draw();
      });
      segBind(root, 'ql-sl', String(S.slip), (v) => {
        S.slip = +v;
        pb.stop();
        build();
        draw();
      });
      bind(
        root,
        'ql-sp',
        (v) => {
          S.sp = +v;
        },
        (v) => ['', 'slow, to read each update', 'steady', 'fast', 'two episodes at a time'][+v],
      );
      bind(
        root,
        'ql-al',
        (v) => {
          S.al = +v;
        },
        f2,
      );
      bind(
        root,
        'ql-ep2',
        (v) => {
          S.eps = +v;
        },
        f2,
      );
      bind(
        root,
        'ql-ga',
        (v) => {
          S.ga = +v;
          if (!pb) return;
          truth();
          ERR = [gap()];
          RET = [];
          DM = [];
          ep = 0;
          fin = false;
          tv++;
          msg =
            'γ changed, so the true values changed too. The charts start again. The table keeps what it has learned.';
          draw();
        },
        f2,
      );
      gwClick(c, (x, y) => {
        if (!P) return;
        const q = P.at(x, y);
        if (q < 0 || G.wall(q)) return;
        sel = q;
        tv++;
        draw();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          build();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'sarsa',
    tab: 'SARSA and the cliff',
    title: 'SARSA and Q-learning: two robots on a cliff edge',
    lede: 'Two robots learn the same walk: from the bottom-left corner to the bottom-right, with a cliff along the whole way between. Falling costs 100 and sends the robot back to the start. Both use a table and both explore by making an occasional random move. Their update rules differ in a single term. One learns the shortest path, right along the edge. The other learns to keep its distance. Which is the wiser depends on a question that matters far beyond this grid: are you being judged while you learn?',
    stage: `<div class="stage"><figure class="fig"><figcaption>Top: Q-learning. Bottom: SARSA. The gold line is the route each table currently recommends. The dot is the robot, still exploring</figcaption><canvas class="short" id="sa-c" role="img" aria-label="Two cliff worlds with the routes learned by Q-learning and SARSA"></canvas></figure>
 <figure class="fig"><figcaption>Top: the reward each robot actually collected, averaged over the last 20 episodes. Bottom: how often each fell off the cliff</figcaption><canvas class="short" id="sa-r" role="img" aria-label="Reward per episode and number of falls for both methods"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The same step, updated by each rule</figcaption><div class="eq" id="sa-eq" aria-live="polite" style="min-height:9.5em;margin-top:6px"></div></figure>
 ${uHow('sa', ['Both robots start with a table of zeros and behave the same way: the best-looking move most of the time, a random move with probability ε.', 'Each makes a move, gets a reward and lands in a new square.', 'Q-learning asks: what is the best move from the new square? Its target is reward + γ × the value of that best move.', 'SARSA first chooses its next move, random slips included, and asks: what is the value of the move I am actually going to make? Its target is reward + γ × that value.', 'Each moves its table entry a fraction α of the way toward its own target.', 'Repeat for 500 episodes.'])}
 <div class="controls">${uBtns('sa', 'Run', 'One step', 'episodes')}
 ${grp('Learning', slider('sa-ep2', 'Exploration ε: share of random moves', 0, 0.5, 0.02, 0.1) + slider('sa-al', 'Learning rate α', 0.05, 1, 0.05, 0.5) + slider('sa-sp', 'Speed', 1, 3, 1, 2))}</div>
 <dl class="read">${rd('ep', 'Episodes finished')}${rd('rq', 'Q-learning: average reward, last 50 episodes', 'hl')}${rd('rs', 'SARSA: average reward, last 50 episodes', 'hl2')}${rd('pq', 'Q-learning: recommended route')}${rd('ps', 'SARSA: recommended route')}${rd('fq', 'Falls so far: Q-learning / SARSA')}</dl>`,
    tryit:
      'Press Run and watch the gold routes. Which robot learns to hug the cliff, and which goes the long way round? Now compare the reward curves: the robot with the shorter route earns less. How can that be? (Count the falls.) Slide ε down to 0.02 and start over, then to 0: what happens to the difference between the two? Slide it up to 0.3. Finally step slowly and read the two updates side by side on a step where the robot explored: that is the only kind of step on which the two rules disagree.',
    math: String.raw`<h4>Intuition</h4><p>Two apprentices learn to carry trays through a crowded kitchen. Both are clumsy: one step in ten goes somewhere they did not intend. The first reasons, "the best route is the short one past the hot stove, and that is what I will learn", and gives no thought to its own clumsiness. The second reasons, "given how I actually walk, how do routes work out for me?", and finds that the route past the stove regularly ends in a burn. It learns to go the long way round. The first has learned the route that is best for a waiter who never stumbles. The second has learned the route that is best for the waiter it currently is.</p>
 <h4>The two updates</h4><p>After the step \(s,a\to r,s'\), Q-learning uses the best next move in its target, and SARSA uses the next move \(a'\) that it has actually chosen:</p>$$\text{Q-learning:}\quad Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]$$$$\text{SARSA:}\quad Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\,Q(s',a')-Q(s,a)\big].$$<p>The name SARSA spells out what one update needs: \(s,a,r,s',a'\). When the next move is the best-looking one, the two targets are identical. They differ only when the next move is an exploratory one, which happens with probability \(\varepsilon\).</p>
 <h4>On-policy and off-policy</h4><p>SARSA's table answers the question "how good is this move, given that I will go on behaving as I do now, random moves included?" It evaluates the policy it is following, so it is called <b>on-policy</b>. Q-learning's table answers "how good is this move, given that I behave perfectly from the next step on?" It learns about a policy it is not following, so it is <b>off-policy</b>.</p>
 <h4>Why the cliff separates them</h4><p>On a square at the cliff edge, a random move has a one in four chance of stepping off, at a cost of 100. SARSA's targets include those accidents, so the values of edge squares are dragged down and the route moves away from the edge. Q-learning's targets ignore them, because the best move from an edge square is never to jump. Its table is right about the shortest path. But while it learns, it still explores, walks along the edge, and falls. That is why its reward during learning is lower although its route is better.</p>
 <h4>Which is right?</h4><p>If learning happens in a simulator and only the final behaviour is used, with exploration switched off, Q-learning's route is the one you want. If mistakes during learning cost something real, as for a robot that can break or a system that serves customers while it learns, SARSA's caution is worth having. As \(\varepsilon\to0\) the accidents disappear and the two learn the same thing: try it.</p>
 <h4>What this page runs</h4><p>The cliff walk of Sutton and Barto, here 4 squares high and 10 wide. Every step costs 1, the cliff costs 100 and returns the robot to the start, and the goal ends the episode. There is no discounting, \(\gamma=1\). Both robots run for 500 episodes. The reward curve shows the average of the last 20 episodes, since single episodes swing wildly whenever a robot falls.</p>`,
    q: 'With ε = 0.1, Q-learning’s recommended route is 11 steps and SARSA’s is longer. A classmate says SARSA has “learned the wrong answer”. In what sense is SARSA’s answer correct?',
    a: 'SARSA has learned the true values of moves for a robot that keeps making a random move one time in ten. For that robot, walking along the edge really is worse, because the expected cost of falls outweighs the steps saved. SARSA’s route is the best one available to the robot as it actually behaves. Q-learning’s route is the best one for a robot that will stop exploring. Neither is wrong. They answer different questions, and the right one to ask depends on whether the exploring will ever be switched off.',
    init(root) {
      const c = $('#sa-c', root),
        cr = $('#sa-r', root),
        live = $('#sa-live', root),
        eq = $('#sa-eq', root),
        S = { eps: 0.1, al: 0.5, sp: 2 },
        ROWS = ['..........', '..........', '..........', 'SXXXXXXXXG'],
        MAXE = 500,
        CAP = 400;
      let G,
        A = [],
        ep = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false;
      const pick = (Q, s) => {
        if (Math.random() < S.eps) return { a: Math.floor(Math.random() * 4), ex: true };
        const row = [0, 1, 2, 3].map((a) => Q[s * 4 + a]),
          mx = Math.max(...row),
          ties = [0, 1, 2, 3].filter((k) => row[k] >= mx - 1e-12);
        return { a: ties[Math.floor(Math.random() * ties.length)], ex: false };
      };
      const mk = (sarsa) => {
        const o = {
          sarsa,
          Q: new Float64Array(G.n * 4),
          s: G.start,
          R: 0,
          n: 0,
          RET: [],
          FALL: [0],
          falls: 0,
          trail: [G.start],
          done: false,
          last: null,
        };
        o.nx = pick(o.Q, o.s);
        return o;
      };
      const build = () => {
        G = gwMake(ROWS, { rp: -100, rg: -1, back: true });
        A = [mk(false), mk(true)];
        ep = 0;
        fin = false;
        cu = 0;
        msg = 'Two tables of zeros, two robots in the bottom-left corner. Press Run.';
        eqShow(
          eq,
          ['Nothing has been updated yet. Press <b>One step</b> to see both rules applied.'],
          0,
        );
      };
      const stepA = (o) => {
        if (o.done) return;
        const s = o.s,
          a = o.nx.a,
          ex = o.nx.ex,
          r = G.step(s, a),
          s2 = r.s2,
          old = o.Q[s * 4 + a];
        let nxt = null,
          tv;
        if (r.done) tv = 0;
        else if (o.sarsa) {
          nxt = pick(o.Q, s2);
          tv = o.Q[s2 * 4 + nxt.a];
        } else tv = gwMax(o.Q, s2);
        const tg = r.r - 0 + tv,
          nw = old + S.al * (tg - old);
        o.Q[s * 4 + a] = nw;
        o.R += r.r;
        o.n++;
        if (G.ch(r.raw) === 'X') {
          o.falls++;
          o.trail = [G.start];
        } else o.trail.push(s2);
        o.last = {
          s,
          a,
          ex,
          r: r.r,
          s2,
          old,
          nw,
          tv,
          tg,
          nxt,
          done: r.done,
          fell: G.ch(r.raw) === 'X',
        };
        o.s = s2;
        if (!o.sarsa && !r.done) nxt = pick(o.Q, s2);
        o.nx = nxt;
        if (r.done || o.n >= CAP) {
          o.done = true;
        }
      };
      const endEp = () => {
        for (const o of A) {
          o.RET.push(o.R);
          o.FALL.push(o.falls);
          o.R = 0;
          o.n = 0;
          o.s = G.start;
          o.trail = [G.start];
          o.done = false;
          o.nx = pick(o.Q, o.s);
        }
        ep++;
        if (ep >= MAXE) {
          fin = true;
          cu = 5;
          const rq = gwRoute(G, A[0].Q, G.start),
            rs = gwRoute(G, A[1].Q, G.start),
            av = (o) => o.RET.slice(-50).reduce((x, y) => x + y, 0) / 50;
          msg =
            'Finished: 500 episodes each. Q-learning recommends a route of ' +
            (rq.ok ? rq.acts.length + ' steps' : 'no complete route') +
            ' and averaged ' +
            f2(av(A[0])) +
            ' per episode over the last 50. SARSA recommends ' +
            (rs.ok ? rs.acts.length + ' steps' : 'no complete route') +
            ' and averaged ' +
            f2(av(A[1])) +
            '. Falls: ' +
            A[0].falls +
            ' and ' +
            A[1].falls +
            '.';
        }
      };
      const line = (o) => {
        const l = o.last;
        if (!l) return '';
        const nm = o.sarsa ? 'SARSA' : 'Q-learning';
        return [
          '<b>' +
            nm +
            '</b>: in ' +
            G.name(l.s) +
            ' it ' +
            (l.ex ? 'explored' : 'took its best move') +
            ', ' +
            GWS[l.a] +
            '. ' +
            (l.fell
              ? 'It stepped off the cliff: reward −100, back to the start.'
              : 'Reward ' + rlN(l.r).replace('.00', '') + ', now in ' + G.name(l.s2) + '.'),
          l.done
            ? '   the goal ends the episode: target = ' + rlN(l.r)
            : o.sarsa
              ? '   its next move will be ' +
                GWS[l.nxt.a] +
                (l.nxt.ex ? ' (a random, exploring move)' : ' (its best move)') +
                ', worth ' +
                rlN(l.tv) +
                ':  target = ' +
                rlN(l.r) +
                ' + ' +
                rlN(l.tv) +
                ' = <b>' +
                rlN(l.tg) +
                '</b>'
              : '   the best move from there is worth ' +
                rlN(l.tv) +
                ':  target = ' +
                rlN(l.r) +
                ' + ' +
                rlN(l.tv) +
                ' = <b>' +
                rlN(l.tg) +
                '</b>',
          '   Q(' +
            G.name(l.s) +
            ', ' +
            GWS[l.a] +
            ') = ' +
            rlN(l.old) +
            ' + ' +
            f2(S.al) +
            ' × (' +
            rlN(l.tg) +
            ' − ' +
            rlN(l.old) +
            ') = <b>' +
            rlN(l.nw) +
            '</b>',
        ];
      };
      const stepBoth = (quiet) => {
        if (fin) return;
        for (const o of A) stepA(o);
        if (!quiet) {
          eqShow(eq, line(A[0]).concat(line(A[1])), 0);
          const d = A[1].last && A[1].last.nxt && A[1].last.nxt.ex;
          cu = d ? 3 : 4;
          msg =
            'Episode ' +
            (ep + 1) +
            '. ' +
            (A.some((o) => o.last && o.last.fell) ? 'A robot fell off the cliff. ' : '') +
            (d
              ? 'SARSA’s next move is an exploring one, so its target uses the value of that random move. Q-learning would have used the best move here.'
              : 'Each robot moved one square and updated one entry of its table.');
        }
        if (A.every((o) => o.done)) endEp();
      };
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c),
            hh = (h - 34) / 2;
          g.font = MONO;
          A.forEach((o, i) => {
            const y0 = 16 + i * (hh + 18),
              R = gwRoute(G, o.Q, G.start);
            g.font = MONO;
            g.fillStyle = t.ink;
            g.textAlign = 'left';
            g.fillText(
              (i ? 'SARSA' : 'Q-LEARNING') +
                ': ' +
                (R.ok ? 'recommended route, ' + R.acts.length + ' steps' : 'no complete route yet'),
              6,
              y0 - 4,
            );
            gwDraw(g, t, 4, y0, w - 8, hh, G, {
              Q: o.Q,
              agent: o.s,
              trail: R.ok ? R.path : null,
              noname: true,
              nonum: true,
              xname: 'cliff',
              vm: 30,
              acol: i ? t.ok : t.accent,
            });
          });
        }
        {
          const { g, w, h } = fit(cr),
            sm = (o) =>
              o.RET.map((_, i) => {
                const a = o.RET.slice(Math.max(0, i - 19), i + 1);
                return Math.max(-120, a.reduce((x, y) => x + y, 0) / a.length);
              }),
            n = MAXE,
            L = uLine(g, t, 0, 0, w, h * 0.56, sm(A[0]), {
              title: 'reward per episode while learning (average of the last 20)',
              n,
              y1: -120,
              ym: 0,
              col: t.accent,
            }),
            s1 = sm(A[1]);
          g.strokeStyle = t.ok;
          g.lineWidth = 2.2;
          g.beginPath();
          s1.forEach((v, i) => (i ? g.lineTo(L.X(i), L.Y(v)) : g.moveTo(L.X(i), L.Y(v))));
          g.stroke();
          g.font = MONO;
          g.textAlign = 'right';
          g.fillStyle = t.accent;
          g.fillText('Q-learning', w - 14, h * 0.56 - 40);
          g.fillStyle = t.ok;
          g.fillText('SARSA', w - 14, h * 0.56 - 26);
          const fm = Math.max(10, A[0].falls, A[1].falls),
            L2 = uLine(g, t, 0, h * 0.56, w, h * 0.44, A[0].FALL, {
              title: 'falls off the cliff so far',
              n: n + 1,
              ym: fm,
              col: t.accent,
            });
          g.strokeStyle = t.ok;
          g.lineWidth = 2.2;
          g.beginPath();
          A[1].FALL.forEach((v, i) =>
            i ? g.lineTo(L2.X(i), L2.Y(v)) : g.moveTo(L2.X(i), L2.Y(v)),
          );
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('episodes →', w - 12, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'sa', cu);
        const av = (o) =>
            o.RET.length
              ? f2(o.RET.slice(-50).reduce((x, y) => x + y, 0) / Math.min(50, o.RET.length))
              : '–',
          rt = (o) => {
            const R = gwRoute(G, o.Q, G.start);
            if (!R.ok) return 'not found yet';
            const edge = R.path.filter((q) => Math.floor(q / G.C) === 2).length;
            return R.acts.length + ' steps, ' + edge + ' of them along the cliff edge';
          };
        setR(root, 'ep', ep + (fin ? ' (stopped)' : ''));
        setR(root, 'rq', av(A[0]));
        setR(root, 'rs', av(A[1]));
        setR(root, 'pq', rt(A[0]));
        setR(root, 'ps', rt(A[1]));
        setR(root, 'fq', A[0].falls + ' / ' + A[1].falls);
      };
      build();
      if (RM) {
        while (!fin) stepBoth(true);
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (S.sp === 1) {
          if (fr % 20) return;
          stepBoth(false);
        } else if (S.sp === 2) {
          if (ep < 1) {
            for (let i = 0; i < 3 && ep < 1; i++) stepBoth(false);
          } else {
            const e0 = ep;
            let k = 0;
            while (ep === e0 && !fin && k++ < 2 * CAP) stepBoth(true);
          }
        } else {
          const e0 = ep;
          let k = 0;
          while (ep < e0 + 5 && !fin && k++ < 12 * CAP) stepBoth(true);
        }
        draw();
        chk();
      });
      pb = playBtn($('#sa-play', root), anim);
      chk = runCtl(pb, root, 'sa', {
        done: () => fin,
        restart: () => build(),
        count: () => ep,
        what: 'episodes',
      });
      $('#sa-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else stepBoth(false);
        draw();
      };
      $('#sa-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      bind(
        root,
        'sa-ep2',
        (v) => {
          S.eps = +v;
        },
        f2,
      );
      bind(
        root,
        'sa-al',
        (v) => {
          S.al = +v;
        },
        f2,
      );
      bind(
        root,
        'sa-sp',
        (v) => {
          S.sp = +v;
        },
        (v) => ['', 'one step at a time', 'one episode per frame', 'five episodes per frame'][+v],
      );
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          build();
          draw();
          pb.start();
        },
      };
    },
  },
  {
    id: 'policy-gradient',
    tab: 'Policy gradient',
    title: 'Policy gradient: adjust the habits directly',
    lede: 'Q-learning learns how good every move is and then picks the best. There is a more direct idea: skip the values, and learn the behaviour itself. Give every square four preferences, one per move, and choose moves at random in proportion to them. After each episode, strengthen the moves that were followed by a good outcome and weaken the ones followed by a bad one. Here the arrows in each square show the current probabilities, and you can watch a robot that starts by wandering aimlessly make up its mind.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The policy. In every square the four arrows show how likely each move is: longer and darker means more likely. The gold line is the last episode. Click a square to see its numbers</figcaption><canvas class="short" id="pg-c" role="img" aria-label="Grid world with the probability of each move in each square"></canvas></figure>
 <figure class="fig"><figcaption>Top: the total reward of each episode (faint) and its average over 30 episodes (bold). Bottom: the four probabilities in the selected square</figcaption><canvas class="short" id="pg-r" role="img" aria-label="Reward per episode and the action probabilities of one square"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>One update from the last episode, worked out</figcaption><div class="eq" id="pg-eq" aria-live="polite" style="min-height:10.5em;margin-top:6px"></div></figure>
 ${uHow('pg', ['Every square has four preferences, all zero at first, so all four moves are equally likely.', 'Play one whole episode, choosing each move at random according to the current probabilities.', 'For every step of the episode, work out the return: the total discounted reward from that step to the end.', 'Compare the return with what is usual for that square, the baseline. Better than usual counts as good, worse as bad.', 'Raise the preference of the move that was made if the result was good, lower it if bad, by an amount proportional to how good or bad. The other three moves shift the opposite way.', 'Repeat until the average reward stops improving.'])}
 <div class="controls">${uBtns('pg', 'Run', 'One episode', 'episodes')}
 ${grp('Learning', slider('pg-al', 'Step size α', 0.01, 0.3, 0.01, 0.1) + `<div class="row seg" id="pg-b" role="group" aria-label="Baseline"><button class="btn" type="button" data-v="1">With a Baseline</button><button class="btn" type="button" data-v="0">Without a Baseline</button></div>` + slider('pg-sp', 'Speed', 1, 3, 1, 2))}</div>
 <dl class="read">${rd('ep', 'Episodes finished')}${rd('lr', 'Reward of the last episode')}${rd('av', 'Average reward, last 30 episodes', 'hl')}${rd('op', 'Best possible reward')}${rd('ps', 'Chance of choosing a best move, averaged over the squares of the last episode', 'hl2')}${rd('ls', 'Steps in the last episode')}</dl>`,
    tryit:
      'Press One episode a few times. The first episodes are long and aimless, and they usually end badly: read the worked update to see a move being made less likely. Press Run and watch the arrows grow in one direction, square by square, starting near the goal. Click the start square and watch its four bars in the bottom-right picture. When learning stops, start over with Without a Baseline, two or three times. Most runs now settle into a poor habit and never reach the best reward. Then go back to the baseline and try a step size of 0.3: fast, and in a good share of runs it commits to a poor habit and cannot undo it. Why is that harder to recover from here than in Q-learning?',
    math: String.raw`<h4>Intuition</h4><p>Think of training a dog. You do not hand the dog a table of how valuable each action is in each situation. You wait until it does something, and if what follows is good, you reward it, and it becomes a little more inclined to do the same thing in the same situation. Policy gradient is exactly that, done with arithmetic: behave, see how it turned out, and make the things you did a little more or a little less likely.</p>
 <h4>A policy with adjustable preferences</h4><p>Each state \(s\) has one number \(\theta_{s,a}\) for each move, and the probability of a move comes from a softmax:</p>$$\pi_\theta(a\mid s)=\frac{e^{\theta_{s,a}}}{\sum_{b}e^{\theta_{s,b}}} .$$<p>With all preferences equal every move is equally likely. Raising one preference makes its move more likely and all the others less. The behaviour is random by construction, so exploration is built in and fades by itself as the preferences grow apart.</p>
 <h4>What is being maximised</h4><p>The objective is the expected return of an episode played with the policy, \(J(\theta)=\mathbb E_{\pi_\theta}[G_0]\). The aim is to climb its gradient. The <b>policy gradient theorem</b> says this gradient can be estimated from sampled episodes without knowing how the world works:</p>$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\Big[\sum_t G_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big],\qquad G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}.$$
 <h4>REINFORCE</h4><p>Williams (1992) turned this into an algorithm: play an episode, then for every step move the preferences along \(G_t\nabla\log\pi\). For the softmax above the gradient is simple, and the update for the visited state \(s_t\) is</p>$$\theta_{s_t,a}\leftarrow\theta_{s_t,a}+\alpha\,\big(G_t-b(s_t)\big)\,\big(\mathbb 1[a=a_t]-\pi_\theta(a\mid s_t)\big)\quad\text{for each move }a.$$<p>For the move that was made, the last factor is \(1-\pi\), positive. For the others it is \(-\pi\), negative. So a good outcome pushes the chosen move up and the rest down. The push is larger when the chosen move was unlikely, because a surprising success is more informative.</p>
 <h4>The baseline</h4><p>Without \(b\), every return is compared with zero. In a world where almost all returns are negative, every move ever made is pushed down, and only the differences between large pushes carry information. That works on average and is very noisy. Subtracting a <b>baseline</b> \(b(s)\), the usual return from that state, changes the question from "was the result good?" to "was it better than usual?". It does not change the expected gradient, because \(\sum_a\pi(a\mid s)\,\nabla\log\pi(a\mid s)=0\), and it reduces the variance a great deal. Here the baseline is a running average of the returns seen from each square. On this page the difference is stark: in 40 test runs with a step size of 0.1, every run with the baseline reached the best reward, typically in about 75 episodes, and 27 of the 40 runs without it were still stuck after 1,500.</p>
 <h4>Compared with Q-learning</h4><p>Policy gradient works unchanged when moves are continuous, such as a steering angle, and when the best behaviour is itself random. It has no \(\max\) over moves to compute. Its weaknesses are the ones visible here. It needs whole episodes before learning anything. Its updates are noisy. And it is on-policy: when the probabilities for some move have collapsed to nearly zero, that move is almost never tried again, so a bad habit formed early can be hard to unlearn. Q-learning with ε-greedy keeps testing every move for ever.</p>
 <h4>What this page runs</h4><p>The room with a pit, with a firm floor and \(\gamma=0.95\). Each episode is cut off at 60 steps. Learning stops when the average reward of the last 30 episodes is within 1 of the best possible for 30 episodes in a row, or after 1,500 episodes. The next tab replaces the running-average baseline with a learned critic, which gives the actor-critic method.</p>`,
    q: 'An episode ends in the pit with a total reward of −14. With no baseline, the preference of every move made in that episode is lowered, including moves that were perfectly sensible. Why does learning still work?',
    a: 'Each single update is indeed a poor judge of each move. But the sensible moves also appear in the episodes that end well, where they are raised, and across many episodes they are followed by better returns on average than the foolish moves are. The updates are proportional to the returns, so on average the sensible moves come out ahead. The price of relying on averages is noise and slow learning. A baseline removes the part of every return that has nothing to do with the move, so each update points more nearly in the right direction.',
    init(root) {
      const c = $('#pg-c', root),
        cr = $('#pg-r', root),
        live = $('#pg-live', root),
        eq = $('#pg-eq', root),
        S = { al: 0.1, base: 1, sp: 2 },
        GA = 0.95,
        CAP = 60,
        MAXE = 1500;
      let G,
        TH,
        B,
        BN,
        ep = 0,
        RET = [],
        lastPath = [],
        lastN = 0,
        sel,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        P = null,
        best0 = 0,
        bpath = [],
        okrun = 0,
        QS = null;
      const prob = (s) => {
        const z = [0, 1, 2, 3].map((a) => TH[s * 4 + a]),
          mx = Math.max(...z),
          e = z.map((v) => Math.exp(v - mx)),
          sm = e.reduce((a, b) => a + b, 0);
        return e.map((v) => v / sm);
      };
      const build = () => {
        G = gwMake(['...G', '.#.X', 'S...']);
        TH = new Float64Array(G.n * 4);
        B = new Float64Array(G.n);
        BN = new Int32Array(G.n);
        ep = 0;
        RET = [];
        lastPath = [G.start];
        lastN = 0;
        sel = G.start;
        fin = false;
        cu = 0;
        okrun = 0;
        const Qs = gwQstar(G, GA),
          R = gwRoute(G, Qs, G.start);
        QS = Qs;
        best0 = R.ret;
        bpath = R.path.slice(0, -1).map((s, i) => [s, R.acts[i]]);
        msg =
          'All preferences are zero: in every square the four moves are equally likely, as the equal arrows show. Press One episode or Run.';
        eqShow(eq, ['No episode has been played yet.'], 0);
      };
      const episode = (quiet) => {
        if (fin) return;
        let s = G.start;
        const tr = [];
        let tot = 0;
        for (let i = 0; i < CAP; i++) {
          const p = prob(s);
          let u = Math.random(),
            a = 3;
          for (let k = 0; k < 4; k++) {
            u -= p[k];
            if (u <= 0) {
              a = k;
              break;
            }
          }
          const o = G.step(s, a);
          tr.push({ s, a, r: o.r, p: p[a] });
          tot += o.r;
          s = o.s2;
          if (o.done) break;
        }
        let Gt = 0;
        for (let i = tr.length - 1; i >= 0; i--) {
          Gt = tr[i].r + GA * Gt;
          tr[i].G = Gt;
        }
        let ex = null;
        const pick = Math.floor(Math.random() * Math.min(tr.length, 6));
        tr.forEach((e, i) => {
          const b = S.base ? B[e.s] : 0,
            adv = e.G - b,
            p = prob(e.s),
            old = TH[e.s * 4 + e.a];
          for (let k = 0; k < 4; k++) TH[e.s * 4 + k] += S.al * adv * ((k === e.a ? 1 : 0) - p[k]);
          if (i === pick)
            ex = { e, b, adv, p: p[e.a], old, nw: TH[e.s * 4 + e.a], p2: prob(e.s)[e.a], i };
          BN[e.s]++;
          B[e.s] += (e.G - B[e.s]) / Math.min(BN[e.s], 20);
        });
        ep++;
        RET.push(tot);
        lastPath = [G.start].concat(tr.map((e, i) => (i + 1 < tr.length ? tr[i + 1].s : s)));
        lastN = tr.length;
        const how =
          G.ch(s) === 'G'
            ? 'reached the goal'
            : G.ch(s) === 'X'
              ? 'fell into the pit'
              : 'ran out of time after ' + CAP + ' steps';
        if (!quiet && ex) {
          const e = ex.e;
          eqShow(
            eq,
            [
              'Episode ' +
                ep +
                ' ' +
                how +
                ' after ' +
                tr.length +
                ' steps, total reward ' +
                rlN(tot).replace('.00', '') +
                '. Take step ' +
                (ex.i + 1) +
                ' as an example:',
              'in <b>' +
                G.name(e.s) +
                '</b> the robot chose ' +
                GWS[e.a] +
                ' ' +
                GWN[e.a] +
                ', which had probability ' +
                pct(ex.p) +
                '.',
              'return from that step to the end:  G = <b>' +
                rlN(e.G) +
                '</b>' +
                (S.base
                  ? '     usual return from ' + G.name(e.s) + ' (baseline): ' + rlN(ex.b)
                  : '     no baseline: compared with 0'),
              'better or worse than usual:  G − baseline = ' +
                rlN(e.G) +
                ' − ' +
                rlN(ex.b) +
                ' = <b>' +
                rlN(ex.adv) +
                '</b>',
              'preference for ' +
                GWS[e.a] +
                ' in ' +
                G.name(e.s) +
                ':  ' +
                rlN(ex.old) +
                ' + ' +
                f2(S.al) +
                ' × ' +
                rlN(ex.adv) +
                ' × (1 − ' +
                f2(ex.p) +
                ') = <b>' +
                rlN(ex.nw) +
                '</b>',
              'so the probability of ' +
                GWS[e.a] +
                ' in ' +
                G.name(e.s) +
                ' went from ' +
                pct(ex.p) +
                ' to <b>' +
                pct(ex.p2) +
                '</b>. The other three moves shifted the opposite way.',
            ],
            0,
          );
        }
        const av = RET.slice(-30).reduce((a, b) => a + b, 0) / Math.min(30, RET.length);
        cu = 4;
        msg =
          'Episode ' +
          ep +
          ' ' +
          how +
          ': total reward ' +
          rlN(tot).replace('.00', '') +
          '. Every move made in it was strengthened or weakened according to what followed. Average over the last 30 episodes: ' +
          f2(av) +
          ' (best possible: ' +
          rlN(best0).replace('.00', '') +
          ').';
        if (RET.length >= 30 && av >= best0 - 1) okrun++;
        else okrun = 0;
        if (okrun >= 30 || ep >= MAXE) {
          fin = true;
          cu = 5;
          msg =
            (ep >= MAXE
              ? 'Stopped at the limit of ' +
                MAXE +
                ' episodes with an average reward of ' +
                f2(av) +
                '.'
              : 'Converged after ' +
                ep +
                ' episodes: for 30 episodes in a row the average reward has been within 1 of the best possible (' +
                rlN(best0).replace('.00', '') +
                '). It is now ' +
                f2(av) +
                '.') +
            ' On the squares of its last episode it chooses a best move ' +
            pct(pbest()) +
            ' of the time.';
        }
      };
      const pbest = () => {
        const sq = [...new Set(lastPath.filter((q) => !G.term(q)))];
        if (!sq.length) return NaN;
        let t2 = 0;
        for (const q of sq) {
          const p = prob(q),
            m = gwMax(QS, q);
          for (let a = 0; a < 4; a++) if (QS[q * 4 + a] >= m - 1e-9) t2 += p[a];
        }
        return t2 / sq.length;
      };
      const draw = () => {
        const t = T(),
          PR = new Float64Array(G.n * 4);
        for (const s of G.states) if (!G.term(s)) prob(s).forEach((p, a) => (PR[s * 4 + a] = p));
        {
          const { g, w, h } = fit(c);
          P = gwDraw(g, t, 6, 6, w - 12, h - 12, G, {
            probs: PR,
            trail: lastPath,
            thin: true,
            sel,
          });
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.56,
            n = Math.max(60, RET.length),
            lo = -70,
            A = uLine(g, t, 0, 0, w, h1, [], {
              title: 'total reward of each episode, and its average over 30',
              n,
              y1: lo,
              ym: 10,
            });
          g.strokeStyle = mix(t.accent, t.accent, 0, 0.3);
          g.lineWidth = 1;
          g.beginPath();
          RET.forEach((v, i) =>
            i ? g.lineTo(A.X(i), A.Y(Math.max(lo, v))) : g.moveTo(A.X(i), A.Y(Math.max(lo, v))),
          );
          g.stroke();
          g.strokeStyle = t.accent;
          g.lineWidth = 2.4;
          g.beginPath();
          RET.forEach((_, i) => {
            const a = RET.slice(Math.max(0, i - 29), i + 1),
              v = Math.max(lo, a.reduce((x, y) => x + y, 0) / a.length);
            i ? g.lineTo(A.X(i), A.Y(v)) : g.moveTo(A.X(i), A.Y(v));
          });
          g.stroke();
          g.setLineDash([5, 4]);
          g.strokeStyle = t.gold;
          g.lineWidth = 1.3;
          g.beginPath();
          g.moveTo(A.m.l, A.Y(best0));
          g.lineTo(w - A.m.r, A.Y(best0));
          g.stroke();
          g.setLineDash([]);
          g.font = MONO;
          g.fillStyle = t.cut;
          g.textAlign = 'right';
          g.fillText('best possible', w - A.m.r, A.Y(best0) + 13);
          const p = G.term(sel) ? [0, 0, 0, 0] : prob(sel),
            y0 = h1 + 24,
            H = h - h1 - 48,
            bw = (w - 80) / 4;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('square ' + G.name(sel) + ': probability of each move', 8, h1 + 14);
          for (let a = 0; a < 4; a++) {
            const x = 50 + a * bw;
            g.fillStyle = mix(t.accent, t.accent, 0, 0.35 + 0.65 * p[a]);
            g.fillRect(x, y0 + H - p[a] * H, bw * 0.6, p[a] * H);
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(GWS[a] + ' ' + GWN[a] + '  ' + pct(p[a]), x + bw * 0.3, h - 8);
          }
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(40, y0 + H);
          g.lineTo(w - 20, y0 + H);
          g.stroke();
        }
        live.textContent = msg;
        uCur(root, 'pg', cu);
        const av = RET.length
          ? RET.slice(-30).reduce((a, b) => a + b, 0) / Math.min(30, RET.length)
          : NaN;
        setR(root, 'ep', ep + (fin ? ' (stopped)' : ''));
        setR(root, 'lr', RET.length ? rlN(RET[RET.length - 1]).replace('.00', '') : '–');
        setR(root, 'av', isFinite(av) ? f2(av) : '–');
        setR(root, 'op', rlN(best0).replace('.00', ''));
        setR(root, 'ps', pct(pbest()));
        setR(root, 'ls', lastN ? String(lastN) : '–');
      };
      build();
      if (RM) {
        while (!fin) episode(true);
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (S.sp === 1) {
          if (fr % 45) return;
          episode(false);
        } else if (S.sp === 2) {
          if (ep < 5) {
            if (fr % 20) return;
            episode(false);
          } else {
            if (fr % 2) return;
            episode(false);
          }
        } else {
          for (let i = 0; i < 5 && !fin; i++) episode(i < 4);
        }
        draw();
        chk();
      });
      pb = playBtn($('#pg-play', root), anim);
      chk = runCtl(pb, root, 'pg', {
        done: () => fin,
        restart: () => build(),
        count: () => ep,
        what: 'episodes',
      });
      $('#pg-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else episode(false);
        draw();
      };
      $('#pg-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      bind(
        root,
        'pg-al',
        (v) => {
          S.al = +v;
        },
        f2,
      );
      bind(
        root,
        'pg-sp',
        (v) => {
          S.sp = +v;
        },
        (v) => ['', 'one episode at a time', 'steady', 'fast'][+v],
      );
      segBind(root, 'pg-b', String(S.base), (v) => {
        S.base = +v;
        pb.stop();
        build();
        draw();
        pb.start();
      });
      gwClick(c, (x, y) => {
        if (!P) return;
        const q = P.at(x, y);
        if (q < 0 || G.wall(q) || G.term(q)) return;
        sel = q;
        draw();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          build();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'dqn',
    tab: 'Actor-critic and DQN',
    title: 'Actor-critic and DQN: when the table would never fit',
    lede: 'A cart runs on a short track with a pole hinged on top, and the only controls are a push to the left and a push to the right. Keep the pole up for 200 steps. The situation is described by four continuous numbers, so there is no finite list of squares and no table can hold the answer. The table is replaced by a neural network. Two famous ways of training it are shown: a deep Q-network, which learns the value of each push, and an actor-critic, which learns the behaviour directly with a second network as its coach. Watch a controller go from dropping the pole in a tenth of a second to balancing it indefinitely.',
    stage: `<div class="row seg" id="aq-m" role="group" aria-label="Method"><button class="btn" type="button" data-v="dqn">Deep Q-Network (DQN)</button><button class="btn" type="button" data-v="ac">Actor-Critic</button></div>
 <div class="stage"><figure class="fig"><figcaption>The cart and pole, controlled by the network as it is right now. The arrow shows the push it chose</figcaption><canvas class="short" id="aq-c" role="img" aria-label="Cart and pole being balanced"></canvas></figure>
 <figure class="fig"><figcaption>Top: how many steps the pole stayed up in each episode. Bottom: what the network has learned to do for every lean of the pole (across) and every speed of falling (up)</figcaption><canvas class="short" id="aq-r" role="img" aria-label="Episode length and the learned policy over pole angle and angular velocity"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>One training example, worked out</figcaption><div class="eq" id="aq-eq" aria-live="polite" style="min-height:9.5em;margin-top:6px"></div></figure>
 ${uHow('aq', ['The network sees four numbers: where the cart is, how fast it moves, how far the pole leans, how fast it is falling. Every step the pole stays up earns a reward.', 'Act: choose a push using the network, with some randomness so that both pushes get tried.', 'DQN stores every step in a memory. It then draws 32 old steps at random and, for each, moves its predicted value toward reward + γ × the best value of the next situation, as judged by a slowly updated copy of itself.', 'Actor-critic waits for the episode to end. Its critic network predicts how long the pole will stay up from each situation. Pushes that led to a longer run than the critic predicted are made more likely, the others less.', 'Repeat until the last 20 episodes average at least 180 of the 200 possible steps.', 'Then watch the trained controller, and give the pole a shove.'])}
 <div class="controls">${uBtns('aq', 'Run', 'One episode', 'episodes')}
 ${grp('Test the controller', `<div class="row"><button class="btn" id="aq-kl" type="button">Shove the Pole Left</button><button class="btn" id="aq-kr" type="button">Shove the Pole Right</button></div><p class="tpout" id="aq-tpo" role="status"></p>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ep', 'Episodes finished')}${rd('ll', 'Length of the last episode')}${rd('av', 'Average length, last 20 episodes', 'hl')}${rd('ts', 'Steps of experience so far', 'hl2')}${rd('ex', 'Randomness in its choices now')}</dl>`,
    tryit:
      'Press Run with DQN. For the first episodes the pole falls almost at once. Watch the bottom-right picture: it starts as noise and becomes two clean regions with a slanted border. What rule does that border express, in words? When training stops, press the two shove buttons and see how hard a push the controller can recover from. Now switch to Actor-Critic and compare: how many episodes and how many steps of experience does each method need? Run each two or three times with Start over. Which one is more consistent from run to run?',
    math: String.raw`<h4>Intuition</h4><p>Balance a broom upright on your palm. You could not write down a table of what to do for every possible lean and every possible speed: there are infinitely many. Yet you learn it in a minute, because nearby situations call for nearby responses, and your brain fills in the gaps. A neural network does the same for the robot. It is shown some situations and generalises to the ones in between. The two methods here differ in what the network is asked to output: a score for each push, or the push itself.</p>
 <h4>From a table to a function</h4><p>The state is \(s=(x,\dot x,\theta,\dot\theta)\). In place of a table entry \(Q(s,a)\) there is a network \(Q_w(s,a)\) with weights \(w\), taking the four numbers in and giving one value per push out. Changing the weights to fix one situation also changes the answer for similar situations. That is the point, and also the source of all the trouble: the careful one-entry-at-a-time argument that made the table converge no longer applies.</p>
 <h4>Deep Q-network</h4><p>DQN (Mnih and others, 2015) keeps the Q-learning target and trains the network to predict it, minimising the squared error</p>$$L(w)=\Big(r+\gamma\max_{a'}Q_{w^-}(s',a')-Q_w(s,a)\Big)^2 .$$<p>Two additions make this stable. A <b>replay memory</b> stores past steps, and training uses random handfuls of them. Consecutive steps are almost identical, and training on them in order would make the network forget everything but the last few seconds. A <b>target network</b> \(Q_{w^-}\), a copy of the weights that is refreshed only now and then, computes the target. Without it the network would be chasing a target that moves every time it learns. When the pole falls the episode ends and the target is just the final reward, here \(-1\).</p>
 <h4>Actor-critic</h4><p>The <b>actor</b> is a policy network \(\pi_\theta(a\mid s)\) giving the probability of each push, trained by the policy gradient of the previous tab. The <b>critic</b> is a value network \(V_\phi(s)\) that predicts the return from each situation and serves as the baseline:</p>$$\theta\leftarrow\theta+\alpha\,\big(G_t-V_\phi(s_t)\big)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t),\qquad \phi\leftarrow\phi-\beta\,\nabla_\phi\big(G_t-V_\phi(s_t)\big)^2 .$$<p>The quantity \(G_t-V_\phi(s_t)\) is the <b>advantage</b>: how much better the run went than the critic expected. A push followed by a better-than-expected run is made more likely. The critic turns the vague signal "the pole stayed up 40 steps" into the sharp one "that was 15 more than usual from there".</p>
 <h4>How they compare</h4><p>DQN is off-policy: it can reuse every old step many times, so it is sparing with experience, and it needs the replay memory and target network to stay stable. It only works for a small set of distinct actions, because of the \(\max\). Actor-critic is on-policy: it learns from each episode once and then discards it, but each update is cheap, it has no \(\max\), and it carries over directly to continuous actions such as a steering angle. Modern methods mix the two ideas.</p>
 <h4>Reading the policy picture</h4><p>The bottom-right picture fixes the cart in the middle and at rest, and colours each combination of pole angle and falling speed by the push the network prefers. A good controller pushes toward the side the pole is falling to, which moves the cart back underneath it. The border is slanted because speed matters as well as position: a pole leaning slightly left and swinging quickly to the right needs a push to the right.</p>
 <h4>What this page trains</h4><p>The standard cart-pole simulation, with episodes ending when the pole passes 12 degrees, the cart leaves the track, or 200 steps are reached. DQN: a network with two hidden layers of 32 neurons, a memory of 10,000 steps, a batch of 32 every second step, the target copy refreshed every 150 updates, and exploration falling from 100% to 5% over the first 2,500 steps. Actor-critic: an actor and a critic with one hidden layer of 32 neurons each, updated once after every episode. Training stops when the last 20 episodes average 180 steps, or after 400 episodes. Both methods are sensitive to luck, and an occasional run fails to reach the mark.</p>`,
    q: 'DQN trains on steps drawn at random from a memory of old experience, some of it collected thousands of steps ago by a much worse version of the network. Why is it legitimate to learn from such stale experience, when the actor-critic must throw each episode away?',
    a: 'A stored step records a fact about the world: in this situation, this push led to that situation and that reward. That fact is true whoever made the push and however badly they were behaving. The Q-learning target then asks only for the best value available from the next situation, using the current network, so nothing in the update depends on how the old agent would have continued. That is what off-policy means. The actor-critic update is an average over what the current policy does. Experience gathered by an older policy describes a different average, so using it without correction would point the update in the wrong direction.',
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
      const solved = () =>
        LEN.length >= 20 && LEN.slice(-20).reduce((a, b) => a + b, 0) / 20 >= 180;
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
          S.m === 'dqn'
            ? pct(eps()) + ' random pushes'
            : 'drawn from the actor’s own probabilities',
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
          for (let i = 0; i < 2; i++)
            if (demoStep()) {
              draw();
              pb.stop();
              return;
            }
          draw();
          return;
        }
        const k = ep < 3 ? 2 : 14;
        for (let i = 0; i < k && phase === 'train'; i++) envStep();
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
  },
];
