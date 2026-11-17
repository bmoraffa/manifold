/* Manifold — Probabilistic methods.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ probabilistic methods ============ */
function pmLG(x) {
  const cf = [
    76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
    0.001208650973866179, -0.000005395239384953,
  ];
  let y = x,
    t = x + 5.5;
  t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015;
  for (let j = 0; j < 6; j++) s += cf[j] / ++y;
  return -t + Math.log((2.5066282746310005 * s) / x);
}
const PML = [
  {
    id: 'bayes-update',
    tab: 'Bayesian updating',
    title: 'Bayesian updating: change your mind by the right amount',
    lede: 'A café adds a new drink and wants to know what share of its customers like it. Before anyone has tried it, the owner has a hunch. Then customers answer one by one: yes or no. Bayesian updating is the rule for combining the hunch with the answers. It never gives a single number. It gives a whole curve of belief over every possible share, and the curve narrows as the evidence comes in. Watch three owners with very different hunches being pulled to the same conclusion.',
    stage: `<div class="row seg" id="by-pr" role="group" aria-label="The owner's starting belief"><button class="btn" type="button" data-v="flat">No Opinion: Anything Is Possible</button><button class="btn" type="button" data-v="mild">Mild Hunch: About Half</button><button class="btn" type="button" data-v="stub">Stubborn: Sure It Is About 80%</button></div>
 <div class="stage"><figure class="fig"><figcaption>Belief about the share of customers who like the drink. Dashed: before any answers. Filled: now. The shaded part holds 95% of the belief. The thin line is what the answers alone say. Dots along the top: the answers so far</figcaption><canvas class="short" id="by-c" role="img" aria-label="Prior, likelihood and posterior over the unknown share"></canvas></figure>
 <figure class="fig"><figcaption>Top: this owner’s best estimate and 95% range after each answer. Bottom: the best estimates of all three owners, given the same answers</figcaption><canvas class="short" id="by-r" role="img" aria-label="Posterior mean and interval over time, and three priors compared"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The last update, worked out</figcaption><div class="eq" id="by-eq" aria-live="polite" style="min-height:9.5em;margin-top:6px"></div></figure>
 ${uHow('by', ['Before any answers, write down a belief about the share: a curve over every value from 0 to 1. This is the prior.', 'A customer answers yes or no.', 'For every possible share, ask: how likely was that answer if this were the true share? Multiply the belief at that share by that likelihood.', 'Rescale the curve so that its area is 1 again. This is the posterior.', 'The posterior becomes the prior for the next customer. Repeat.', 'Stop when the belief is narrow enough for the decision at hand: here, when 95% of it fits in a range 0.1 wide.'])}
 <div class="controls">${uBtns('by', 'Run', 'One customer', 'customers')}
 ${grp('Add answers yourself', `<div class="row"><button class="btn" id="by-yes" type="button">A Customer Says Yes</button><button class="btn" id="by-no" type="button">A Customer Says No</button></div>`)}
 ${grp('The world and the prior', slider('by-p', 'The true share of customers who like it (the owner cannot see this)', 0.02, 0.98, 0.01, 0.3) + slider('by-m', 'Prior: the owner’s best guess', 0.05, 0.95, 0.01, 0.5) + slider('by-n', 'Prior: how sure, as a number of imagined earlier customers', 2, 100, 1, 2))}</div>
 <dl class="read">${rd('n', 'Customers asked')}${rd('k', 'Said yes')}${rd('mean', 'Best estimate now (posterior mean)', 'hl')}${rd('ci', '95% of the belief lies between', 'hl2')}${rd('fr', 'Plain frequency: yes ÷ asked')}${rd('pb', 'Posterior')}</dl>`,
    tryit:
      'Press One customer a few times with No Opinion and read the worked update: each yes adds 1 to the first number of the curve, each no adds 1 to the second. That is the entire calculation. Press Run and watch the curve tighten around the true share. Now choose Stubborn and start over. How many customers does it take before the owner’s estimate is closer to the truth than to the original 80%? Look at the bottom-right chart: when do the three owners agree to within a few percent? Then use the two buttons to feed in a streak of ten yes answers by hand and see how each owner reacts.',
    math: String.raw`<h4>Intuition</h4><p>A doctor sees a patient with a cough. Before any test, she thinks a common cold is far more likely than anything serious, because colds are common. A test comes back, and it shifts her opinion, a lot if the result would be very surprising under one of the explanations, a little otherwise. A second test shifts it again, starting from where the first left it. She never throws away what she knew before, and she never ignores the evidence. Bayes' rule is the exact arithmetic of that way of thinking: what you believe afterwards is what you believed before, reweighted by how well each possibility explains what you saw.</p>
 <h4>Bayes' rule</h4><p>Let \(\theta\) be the unknown share and \(D\) the data. The rule is</p>$$\underbrace{p(\theta\mid D)}_{\text{posterior}}=\frac{\overbrace{p(D\mid\theta)}^{\text{likelihood}}\ \overbrace{p(\theta)}^{\text{prior}}}{p(D)},\qquad p(D)=\int p(D\mid\theta)\,p(\theta)\,d\theta .$$<p>The prior is the belief before the data. The likelihood says how probable the data would be for each value of \(\theta\). The denominator is only there to make the posterior's area equal 1. Posterior is proportional to likelihood times prior: that sentence is the whole method.</p>
 <h4>The café: a Beta prior and yes-or-no answers</h4><p>If the true share is \(\theta\), a customer says yes with probability \(\theta\). After \(k\) yes answers among \(n\) customers the likelihood is \(\theta^k(1-\theta)^{n-k}\). For the prior, use the Beta family:</p>$$p(\theta)=\mathrm{Beta}(\theta;a,b)\propto\theta^{a-1}(1-\theta)^{b-1}.$$<p>Multiplying the two gives another curve of the same shape,</p>$$p(\theta\mid D)\propto\theta^{a+k-1}(1-\theta)^{b+n-k-1}=\mathrm{Beta}(\theta;\ a+k,\ b+n-k).$$<p>So updating is addition: every yes adds 1 to \(a\), every no adds 1 to \(b\). A prior that reproduces itself like this is called <b>conjugate</b>. The prior behaves exactly like \(a+b\) imagined earlier customers, \(a\) of whom said yes. "No opinion" is \(\mathrm{Beta}(1,1)\), a flat line.</p>
 <h4>Reading the posterior</h4><p>The mean of \(\mathrm{Beta}(a,b)\) is \(a/(a+b)\), so the best estimate after the data is</p>$$\mathbb E[\theta\mid D]=\frac{a+k}{a+b+n}=\underbrace{\frac{a+b}{a+b+n}}_{\text{weight on the prior}}\cdot\frac{a}{a+b}+\underbrace{\frac{n}{a+b+n}}_{\text{weight on the data}}\cdot\frac{k}{n}.$$<p>It is a weighted average of the prior guess and the plain frequency, and the weight on the data grows with every customer. That is why the stubborn owner is pulled to the truth slowly and the owner with no opinion almost at once, and why all three end in the same place. The 95% range, called a <b>credible interval</b>, is the middle 95% of the posterior's area. Its width shrinks like \(1/\sqrt n\).</p>
 <h4>The same number is also a prediction</h4><p>The probability that the next customer says yes, taking the uncertainty about \(\theta\) into account, is the posterior mean \((a+k)/(a+b+n)\). With no opinion and no data it is \(\tfrac12\). After one yes it is \(\tfrac23\), not 1: one happy customer does not prove that everyone will be happy. The plain frequency would have said 100%.</p>
 <h4>Why bother with a prior?</h4><p>With a great deal of data the prior stops mattering, as the bottom chart shows. With little data it matters a lot, and that is when it is most useful: it keeps the estimate sensible after three answers, and it is the natural place to put real knowledge, such as "new drinks at this café are usually liked by 20 to 40 percent of people". The cost is that a confident wrong prior takes many observations to overturn. A prior should be as sure as you actually are, and no surer.</p>
 <h4>Beyond the café</h4><p>The same recipe, prior times likelihood, underlies spam filters, medical diagnosis, A/B testing, the Thompson sampling of the Multi-Armed Bandits tab, and Gaussian process regression. For most models the product has no tidy formula like the Beta one. Then the posterior is explored by sampling, which is the subject of the next tab, MCMC.</p>
 <h4>What this page runs</h4><p>Customers are drawn at random with the true share you set. The run stops when the 95% credible interval is narrower than 0.1, or after 1,000 customers. The interval is computed by adding up the area under the curve numerically.</p>`,
    q: 'With no opinion at the start, the first customer says yes. The plain frequency is now 1 out of 1, or 100%. The Bayesian estimate is 2/3. Which is the better forecast for the second customer, and why?',
    a: 'The Bayesian one. A frequency of 100% claims that every future customer will say yes, on the strength of one answer, and would be proved wrong by the first no. The posterior after one yes is Beta(2, 1): it has shifted toward high values and still gives real weight to everything from about 0.2 upward. Its mean, 2/3, says “probably more people like it than not, and I am far from sure”. That is what one answer actually tells you. As the answers accumulate the two estimates approach each other, and after a few hundred customers they agree to the second decimal place.',
    init(root) {
      const c = $('#by-c', root),
        cr = $('#by-r', root),
        live = $('#by-live', root),
        eq = $('#by-eq', root),
        S = { p: 0.3, m: 0.5, n0: 2, pr: 'flat' },
        PR = { flat: [1, 1], mild: [5, 5], stub: [48, 12] },
        MAXN = 1000,
        NG = 400;
      let a0 = 1,
        b0 = 1,
        D = [],
        k = 0,
        HM = [],
        HL = [],
        HU = [],
        H3 = [[], [], []],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        bm,
        bn;
      const pdf = (a, b) => {
        const o = new Float64Array(NG + 1),
          lb = pmLG(a) + pmLG(b) - pmLG(a + b);
        for (let i = 0; i <= NG; i++) {
          const x = clamp(i / NG, 1e-6, 1 - 1e-6);
          o[i] = Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lb);
        }
        return o;
      };
      const ci = (a, b) => {
        const d = pdf(a, b);
        let tot = 0;
        for (let i = 0; i <= NG; i++) tot += d[i];
        let s = 0,
          lo = 0,
          hi = 1,
          gl = false;
        for (let i = 0; i <= NG; i++) {
          s += d[i] / tot;
          if (!gl && s >= 0.025) {
            lo = i / NG;
            gl = true;
          }
          if (s >= 0.975) {
            hi = i / NG;
            break;
          }
        }
        return [lo, hi];
      };
      const push = () => {
        const n = D.length,
          a = a0 + k,
          b = b0 + n - k,
          [lo, hi] = ci(a, b);
        HM.push(a / (a + b));
        HL.push(lo);
        HU.push(hi);
        Object.values(PR).forEach((q, i) => H3[i].push((q[0] + k) / (q[0] + q[1] + n)));
      };
      const build = () => {
        D = [];
        k = 0;
        HM = [];
        HL = [];
        HU = [];
        H3 = [[], [], []];
        fin = false;
        cu = 0;
        push();
        msg =
          'The prior is Beta(' +
          f2(a0) +
          ', ' +
          f2(b0) +
          '): the owner’s best guess is ' +
          pct(a0 / (a0 + b0)) +
          ', held as firmly as if ' +
          f2(a0 + b0) +
          ' customers had already been asked. No real customer has answered yet.';
        eqShow(
          eq,
          ['No answers yet. The belief is the prior, Beta(' + f2(a0) + ', ' + f2(b0) + ').'],
          0,
        );
      };
      const add = (y) => {
        if (fin || D.length >= MAXN) return;
        const n0 = D.length,
          a = a0 + k,
          b = b0 + n0 - k;
        D.push(y);
        k += y;
        const a2 = a0 + k,
          b2 = b0 + D.length - k;
        push();
        const lo = HL[HL.length - 1],
          hi = HU[HU.length - 1];
        cu = y ? 2 : 2;
        eqShow(
          eq,
          [
            'Customer ' + D.length + ' says <b>' + (y ? 'yes' : 'no') + '</b>.',
            'belief before: Beta(' +
              f2(a) +
              ', ' +
              f2(b) +
              ')     best estimate ' +
              f2(a) +
              ' ÷ (' +
              f2(a) +
              ' + ' +
              f2(b) +
              ') = ' +
              f3(a / (a + b)),
            (y ? 'a yes adds 1 to the first number' : 'a no adds 1 to the second number') +
              ':   belief after: <b>Beta(' +
              f2(a2) +
              ', ' +
              f2(b2) +
              ')</b>',
            'best estimate ' +
              f2(a2) +
              ' ÷ (' +
              f2(a2) +
              ' + ' +
              f2(b2) +
              ') = <b>' +
              f3(a2 / (a2 + b2)) +
              '</b>     95% of the belief between ' +
              f2(lo) +
              ' and ' +
              f2(hi) +
              ' (width ' +
              f2(hi - lo) +
              ')',
            'for comparison, the plain frequency: ' +
              k +
              ' ÷ ' +
              D.length +
              ' = ' +
              f3(k / D.length),
          ],
          0,
        );
        msg =
          'Customer ' +
          D.length +
          ' said ' +
          (y ? 'yes' : 'no') +
          '. The best estimate moved from ' +
          pct(a / (a + b)) +
          ' to ' +
          pct(a2 / (a2 + b2)) +
          ', and 95% of the belief now lies between ' +
          pct(lo) +
          ' and ' +
          pct(hi) +
          '.';
        if (hi - lo < 0.1 || D.length >= MAXN) {
          fin = true;
          cu = 5;
          msg =
            (hi - lo < 0.1
              ? 'Stopped after ' +
                D.length +
                ' customers: 95% of the belief now fits in a range narrower than 0.1, from ' +
                pct(lo) +
                ' to ' +
                pct(hi) +
                '.'
              : 'Stopped at the limit of ' + MAXN + ' customers.') +
            ' Best estimate ' +
            pct(a2 / (a2 + b2)) +
            '. The true share is ' +
            pct(S.p) +
            ', which is ' +
            (S.p >= lo && S.p <= hi ? 'inside' : 'outside') +
            ' that range.';
        }
      };
      const tp = testPoint(root, 'by', ['A share to ask about'], [0.5], [[0, 1]]);
      const draw = () => {
        const t = T(),
          n = D.length,
          a = a0 + k,
          b = b0 + n - k;
        {
          const { g, w, h } = fit(c),
            m = { l: 14, r: 14, t: 40, b: 26 },
            X = (v) => m.l + v * (w - m.l - m.r),
            pr = pdf(a0, b0),
            po = pdf(a, b),
            lk = n ? pdf(k + 1, n - k + 1) : null,
            ym = Math.max(...po, ...pr.map((v) => Math.min(v, 6))) * 1.05,
            Y = (v) => h - m.b - (Math.min(v, ym) / ym) * (h - m.t - m.b),
            lo = HL[HL.length - 1],
            hi = HU[HU.length - 1];
          g.fillStyle = mix(t.accent, t.accent, 0, 0.55);
          g.beginPath();
          g.moveTo(X(lo), Y(0));
          for (let i = 0; i <= NG; i++) {
            const x = i / NG;
            if (x >= lo && x <= hi) g.lineTo(X(x), Y(po[i]));
          }
          g.lineTo(X(hi), Y(0));
          g.closePath();
          g.fill();
          const line = (d, col, lw, dash) => {
            g.strokeStyle = col;
            g.lineWidth = lw;
            g.setLineDash(dash || []);
            g.beginPath();
            for (let i = 0; i <= NG; i++)
              i ? g.lineTo(X(i / NG), Y(d[i])) : g.moveTo(X(0), Y(d[i]));
            g.stroke();
            g.setLineDash([]);
          };
          if (lk) line(lk, t.ok, 1.3);
          line(pr, t.muted, 1.8, [6, 5]);
          line(po, t.accent, 2.6);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h - m.b);
          g.lineTo(w - m.r, h - m.b);
          g.stroke();
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) g.fillText(100 * v + '%', X(v), h - 8);
          g.setLineDash([4, 4]);
          g.strokeStyle = t.gold;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(X(S.p), m.t - 6);
          g.lineTo(X(S.p), h - m.b);
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.cut;
          g.textAlign = S.p > 0.8 ? 'right' : 'left';
          g.fillText('true share', X(S.p) + (S.p > 0.8 ? -5 : 5), m.t + 4);
          markTP(g, X(tp.v[0]), h - m.b, t, false);
          const show = D.slice(-Math.floor((w - 28) / 9)),
            x0 = 14;
          show.forEach((y, i) => {
            g.beginPath();
            g.arc(x0 + i * 9 + 4, 14, 3.4, 0, 7);
            if (y) {
              g.fillStyle = t.accent;
              g.fill();
            } else {
              g.strokeStyle = t.muted;
              g.lineWidth = 1.2;
              g.stroke();
            }
          });
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            n
              ? 'answers, oldest to newest: filled = yes, open = no' +
                  (n > show.length ? ' (the last ' + show.length + ')' : '')
              : 'no answers yet',
            14,
            31,
          );
        }
        {
          const { g, w, h } = fit(cr),
            nn = Math.max(30, HM.length),
            A = uLine(g, t, 0, 0, w, h / 2, HM, {
              title: 'this owner: best estimate and 95% range, by customers asked',
              n: nn,
              ym: 1,
              col: t.accent,
            });
          g.fillStyle = mix(t.accent, t.accent, 0, 0.2);
          g.beginPath();
          HU.forEach((v, i) => (i ? g.lineTo(A.X(i), A.Y(v)) : g.moveTo(A.X(i), A.Y(v))));
          for (let i = HL.length - 1; i >= 0; i--) g.lineTo(A.X(i), A.Y(HL[i]));
          g.closePath();
          g.fill();
          const tl = (L, y0) => {
            g.setLineDash([4, 4]);
            g.strokeStyle = t.gold;
            g.lineWidth = 1.4;
            g.beginPath();
            g.moveTo(L.m.l, L.Y(S.p));
            g.lineTo(w - L.m.r, L.Y(S.p));
            g.stroke();
            g.setLineDash([]);
          };
          tl(A);
          const col = [t.accent, t.ok, t.cut],
            nm = ['no opinion', 'mild hunch', 'stubborn'];
          let B = null;
          H3.forEach((d, i) => {
            B = uLine(g, t, 0, h / 2, w, h / 2, d, {
              title: i ? '' : 'three owners, same answers: their best estimates',
              n: nn,
              ym: 1,
              col: col[i],
            });
          });
          tl(B);
          g.font = MONO;
          H3.forEach((d, i) => {
            g.fillStyle = col[i];
            g.textAlign = 'right';
            g.fillText(nm[i], w - 14, h / 2 + 38 + i * 13);
          });
          g.fillStyle = t.muted;
          g.fillText('customers asked →', w - 12, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'by', cu);
        const po = pdf(a, b);
        let tot = 0,
          ab = 0;
        for (let i = 0; i <= NG; i++) {
          tot += po[i];
          if (i / NG > tp.v[0]) ab += po[i];
        }
        tp.out.textContent =
          'Given everything so far, the probability that the true share is above ' +
          pct(tp.v[0]) +
          ' is ' +
          pct(ab / tot) +
          '. The probability that the next customer says yes is ' +
          pct(a / (a + b)) +
          '.';
        setR(root, 'n', n + (fin ? ' (stopped)' : ''));
        setR(root, 'k', String(k));
        setR(root, 'mean', pct(a / (a + b)));
        setR(root, 'ci', pct(HL[HL.length - 1]) + ' and ' + pct(HU[HU.length - 1]));
        setR(root, 'fr', n ? pct(k / n) : '–');
        setR(root, 'pb', 'Beta(' + f2(a) + ', ' + f2(b) + ')');
      };
      tp.on = () => draw();
      build();
      if (RM) {
        while (!fin) add(Math.random() < S.p ? 1 : 0);
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (fr % (D.length < 12 ? 26 : D.length < 60 ? 7 : 2)) return;
        add(Math.random() < S.p ? 1 : 0);
        draw();
        chk();
      });
      pb = playBtn($('#by-play', root), anim);
      chk = runCtl(pb, root, 'by', {
        done: () => fin,
        restart: () => build(),
        count: () => D.length,
        what: 'customers',
      });
      $('#by-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else add(Math.random() < S.p ? 1 : 0);
        draw();
      };
      $('#by-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      $('#by-yes', root).onclick = () => {
        pb.stop();
        add(1);
        draw();
      };
      $('#by-no', root).onclick = () => {
        pb.stop();
        add(0);
        draw();
      };
      const setP = () => {
        a0 = S.m * S.n0;
        b0 = (1 - S.m) * S.n0;
        if (pb) pb.stop();
        build();
        draw();
      };
      bm = bind(
        root,
        'by-m',
        (v) => {
          S.m = +v;
          setP();
        },
        f2,
      );
      bn = bind(
        root,
        'by-n',
        (v) => {
          S.n0 = +v;
          setP();
        },
        (v) => String(v),
      );
      bind(
        root,
        'by-p',
        (v) => {
          S.p = +v;
          if (pb) {
            pb.stop();
            build();
            draw();
          }
        },
        f2,
      );
      segBind(root, 'by-pr', S.pr, (v) => {
        const q = PR[v];
        S.m = q[0] / (q[0] + q[1]);
        S.n0 = q[0] + q[1];
        bm.set(S.m);
        bn.set(S.n0);
        a0 = q[0];
        b0 = q[1];
        pb.stop();
        build();
        draw();
        pb.start();
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
    id: 'mcmc',
    tab: 'MCMC',
    title: 'MCMC: explore a landscape you can only feel underfoot',
    lede: 'Bayes’ rule often leaves you with a belief you can write down and cannot use: a formula for how plausible each answer is, but no way to draw examples from it or average over it. Markov chain Monte Carlo is the workaround. A walker wanders over the landscape of plausibility, and the places it visits, counted up, turn out to be exactly a sample from the belief. Five walkers live on this page, from the blind random stepper of 1953 to the No-U-Turn sampler that modern Bayesian software runs. They differ in one thing only: how the next place is proposed.',
    stage: `<div class="row seg" id="mc-al" role="group" aria-label="Sampler"><button class="btn" type="button" data-v="rw">Random-Walk Metropolis</button><button class="btn" type="button" data-v="mh">Metropolis-Hastings</button><button class="btn" type="button" data-v="lang">Langevin</button><button class="btn" type="button" data-v="hmc">Hamiltonian</button><button class="btn" type="button" data-v="nuts">No-U-Turn</button></div>
 <div class="row seg" id="mc-tg" role="group" aria-label="Target distribution" style="margin-top:8px"><button class="btn" type="button" data-v="corr">A Slanted Bell</button><button class="btn" type="button" data-v="banana">A Banana</button><button class="btn" type="button" data-v="two">Two Islands</button></div>
 <div class="stage"><figure class="fig"><figcaption>Shading: the target, darker where it is more probable. Dots: every place the walker has stood. Thin line: its last 60 moves. Thick line: the move it has just proposed, with the whole path it travelled for the Hamiltonian and No-U-Turn walkers. Click anywhere to put the walker there</figcaption><canvas class="short" id="mc-c" role="img" aria-label="Target density, samples and the proposed move"></canvas></figure>
 <figure class="fig"><figcaption>Top: the walker’s left-to-right position over time. A healthy walk looks like a fuzzy band. Bottom: how often it has stood at each left-to-right position (bars), against the true answer (line)</figcaption><canvas class="short" id="mc-r" role="img" aria-label="Trace of the chain and histogram against the true marginal"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The last decision, worked out</figcaption><div class="eq" id="mc-eq" aria-live="polite" style="min-height:9.5em;margin-top:6px"></div></figure>
 <div class="howto"><div><h3>How it works, step by step</h3><ol id="mc-steps"></ol></div><div><h3>What just happened</h3><p class="live" id="mc-live" role="status"></p></div></div>
 <div class="controls">${uBtns('mc', 'Run', 'One step', 'steps')}
 ${grp('The walker', slider('mc-s', '<span id="mc-sl">Stride</span>', 0.05, 4, 0.05, 0.8) + slider('mc-L', 'Leapfrog steps in one move (Hamiltonian only)', 1, 40, 1, 15) + `<label class="check" for="mc-cor"><input type="checkbox" id="mc-cor" checked><span id="mc-corl">Apply the correction</span></label>` + slider('mc-sp', 'Speed', 1, 3, 1, 2))}</div>
 <dl class="read">${rd('n', 'Steps so far')}${rd('acc', 'Share of proposals accepted', 'hl2')}${rd('mx', 'Average position: walker / truth')}${rd('ess', 'Worth this many independent samples', 'hl')}${rd('cost', 'Look-ups of height or slope used')}${rd('eff', 'Independent samples per 1,000 look-ups', 'hl')}${rd('err', 'Gap between the bars and the true line')}${rd('isl', 'Share of time on the right-hand side: walker / truth')}</dl>`,
    tryit:
      'Start with Random-Walk Metropolis. Press One step several times and read each decision. When is a downhill move accepted? Run it on the Slanted Bell, then set the stride to 0.05 and to 4 and watch the trace. Now switch to Langevin and press One step: the proposal leans uphill. Choose the Banana, switch off the correction and raise the step size to 0.6. Do the bars still match the line, however long you wait? Switch the correction back on. Choose Hamiltonian on the Banana and step once: the thick line is the path of a ball rolled across the landscape. Raise the leapfrog steps to 40 and watch some paths curl back to where they began. Then choose No-U-Turn and see where each path is cut. Compare the last two readouts across all five walkers on the Banana. Finally try every walker on Two Islands. Which one crosses, and why that one?',
    math: String.raw`<h4>Intuition</h4><p>You are set down on a hilly island in thick fog and asked to draw a map showing how high the ground is everywhere. You cannot see anything. All you have is an altimeter that tells you the height where you stand. Here is a way to do it: wander about, and keep a diary of where you are once a minute. Whenever you consider a step, check the altimeter there. If it is higher, take the step. If it is lower, take it only sometimes, less often the bigger the drop. You will spend most of your time on the high ground and visit the low ground occasionally, and in the long run the number of diary entries in each place is proportional to its height. The diary is the map. Nobody ever needed the total volume of the island. The better walkers on this page add one more instrument: they can also feel which way the ground slopes, and use it to decide where to try next.</p>
 <h4>The problem</h4><p>We want samples from a density \(\pi(\mathbf x)\) that we can only evaluate up to a constant: \(\pi(\mathbf x)=\tilde\pi(\mathbf x)/Z\) with \(Z\) unknown. This is the usual position after Bayes' rule, where \(\tilde\pi\) is prior times likelihood and \(Z\) is the awkward integral in the denominator. With samples \(\mathbf x_1,\dots,\mathbf x_N\) any average can be estimated:</p>$$\mathbb E_\pi[f(\mathbf x)]\approx\frac1N\sum_{i=1}^{N}f(\mathbf x_i).$$
 <h4>Random-walk Metropolis</h4><p>From the current point \(\mathbf x\), propose \(\mathbf x'=\mathbf x+\sigma\,\boldsymbol\varepsilon\) with \(\boldsymbol\varepsilon\sim\mathcal N(\mathbf 0,\mathbf I)\). Accept the proposal with probability</p>$$A=\min\!\Big(1,\ \frac{\tilde\pi(\mathbf x')}{\tilde\pi(\mathbf x)}\Big).$$<p>If it is accepted the walker moves to \(\mathbf x'\). If not, it stays at \(\mathbf x\), and \(\mathbf x\) is recorded again. The unknown \(Z\) cancels in the ratio, which is the entire trick (Metropolis and others, 1953).</p>
 <h4>Metropolis-Hastings</h4><p>The random walk is fair: a step from \(\mathbf x\) to \(\mathbf x'\) is proposed exactly as often as the step back. Hastings (1970) removed that restriction. Let \(q(\mathbf x'\mid\mathbf x)\) be <i>any</i> rule for proposing, and accept with probability</p>$$A=\min\!\Big(1,\ \frac{\tilde\pi(\mathbf x')}{\tilde\pi(\mathbf x)}\cdot\frac{q(\mathbf x\mid\mathbf x')}{q(\mathbf x'\mid\mathbf x)}\Big).$$<p>The second fraction is the <b>Hastings correction</b>. It asks how much easier it is to propose the move than to propose its reverse, and cancels the unfairness. Every other sampler on this page is Metropolis-Hastings with a cleverer \(q\). The Metropolis-Hastings button here uses the simplest unfair proposal there is, an <b>independence sampler</b>: it ignores where the walker stands and proposes from one fixed wide bell centred on the picture, \(q(\mathbf x')=\mathcal N(\mathbf 0,s^2\mathbf I)\). The centre is proposed far more often than the edges, and the correction \(q(\mathbf x)/q(\mathbf x')\) pays that back. Switch the correction off and the bars pile up in the middle.</p>
 <h4>Langevin dynamics</h4><p>A blind proposal wastes most of its tries. If the slope \(\nabla\log\tilde\pi\) is available, lean the proposal uphill:</p>$$\mathbf x'=\mathbf x+\frac{\epsilon^2}{2}\,\nabla\log\tilde\pi(\mathbf x)+\epsilon\,\boldsymbol\xi,\qquad\boldsymbol\xi\sim\mathcal N(\mathbf 0,\mathbf I).$$<p>This is one small time step of the Langevin diffusion \(d\mathbf x=\tfrac12\nabla\log\pi\,dt+d\mathbf W\), a random motion whose long-run distribution is exactly \(\pi\). Taken with a finite step and no check, it is the <b>unadjusted Langevin algorithm</b>: simple, and slightly wrong, with an error that grows with \(\epsilon\). Push \(\epsilon\) further and it becomes unstable: on a narrow ridge each drift overshoots the top by more than the last, and the walker is thrown to the edge of the picture. Adding the Metropolis-Hastings test, with \(q(\mathbf x'\mid\mathbf x)=\mathcal N\big(\mathbf x+\tfrac{\epsilon^2}{2}\nabla\log\tilde\pi(\mathbf x),\,\epsilon^2\mathbf I\big)\), removes the error. That is the <b>Metropolis-adjusted Langevin algorithm</b>, MALA. The same uphill-plus-noise update, with a learned slope, is what generates pictures in the score-matching lesson.</p>
 <h4>Hamiltonian Monte Carlo</h4><p>Langevin takes one slope-guided step and then re-randomizes, so it still diffuses. Hamiltonian Monte Carlo keeps going in the same direction. Think of the walker as a frictionless ball on the surface \(U(\mathbf x)=-\log\tilde\pi(\mathbf x)\), a bowl whose bottom is the most probable place. Give the ball a random kick \(\mathbf p\sim\mathcal N(\mathbf 0,\mathbf I)\) and let it roll. Its total energy</p>$$H(\mathbf x,\mathbf p)=U(\mathbf x)+\tfrac12\lVert\mathbf p\rVert^2$$<p>stays constant along the true path. The path is computed with the <b>leapfrog</b> rule, repeated \(L\) times:</p>$$\mathbf p\leftarrow\mathbf p-\tfrac{\epsilon}{2}\nabla U(\mathbf x),\qquad\mathbf x\leftarrow\mathbf x+\epsilon\,\mathbf p,\qquad\mathbf p\leftarrow\mathbf p-\tfrac{\epsilon}{2}\nabla U(\mathbf x).$$<p>Leapfrog is reversible and preserves volume, so the Hastings correction reduces to the change in energy. The end of the path is accepted with probability</p>$$A=\min\!\big(1,\ \exp(H_{\text{start}}-H_{\text{end}})\big).$$<p>Because leapfrog nearly conserves energy, \(A\) is close to 1 even for a path that crosses the whole distribution. That is the point: far-away proposals that are almost always accepted (Duane and others, 1987; Neal, 2011).</p>
 <h4>The No-U-Turn sampler</h4><p>Hamiltonian Monte Carlo has an awkward setting: the number of leapfrog steps. Too few and it behaves like Langevin. Too many and the ball swings round and comes back, having paid for a long path to end where it started. The No-U-Turn sampler (Hoffman and Gelman, 2014) chooses the length afresh at every move. It grows the path forwards and backwards in time, doubling its length each round, and stops as soon as the two ends begin to approach each other:</p>$$(\mathbf x^{+}-\mathbf x^{-})\cdot\mathbf p^{-}<0\quad\text{or}\quad(\mathbf x^{+}-\mathbf x^{-})\cdot\mathbf p^{+}<0.$$<p>The new place is then drawn from the points on the path, each allowed with probability set by its energy, in a way that keeps detailed balance. Growing in both directions is what makes the move reversible. There is no accept or reject at the end and no path length to tune.</p>
 <h4>Why they all work</h4><p>Each rule satisfies <b>detailed balance</b>: for any two points, the long-run flow of the walker from \(\mathbf x\) to \(\mathbf x'\) equals the flow back,</p>$$\pi(\mathbf x)\,P(\mathbf x\to\mathbf x')=\pi(\mathbf x')\,P(\mathbf x'\to\mathbf x).$$<p>If the walker is already distributed like \(\pi\), one more step leaves it distributed like \(\pi\). As long as it can eventually reach every region, it converges to that distribution from any start. Recording the repeats when a proposal is rejected is essential: leave them out and the high ground is under-counted.</p>
 <h4>How many samples do you really have?</h4><p>The samples are not independent. The <b>effective sample size</b> says how many independent samples would give the same accuracy as the \(N\) correlated ones:</p>$$\mathrm{ESS}=\frac{N}{1+2\sum_{k\ge1}\rho_k},$$<p>where \(\rho_k\) is the correlation between samples \(k\) steps apart. Steps are not equally expensive, though. A random-walk step looks up the height once. A Hamiltonian move with 15 leapfrog steps looks up the slope 15 times. The honest comparison is effective samples per look-up, which is the last-but-two readout. Run the five walkers on the Banana and compare. In the two dimensions of this page the result is humbling: Hamiltonian Monte Carlo beats the random walk by a factor of two or three, the No-U-Turn sampler roughly ties with it, and the independence proposal beats everything, because the whole picture fits under one wide bell. None of that survives in higher dimension. In hundreds of dimensions no fixed bell covers the target, a random walk must shrink its stride towards nothing, and Hamiltonian paths still travel far. That is where the gradient methods earn their cost. Note also what the effective sample size does not measure: whether the walker is sampling the right landscape. With the correction switched off a walker can move briskly and be wrong.</p>
 <h4>Separate islands</h4><p>If the target has two regions separated by low ground, a local walker stays on the island where it started for a very long time. Its samples look perfectly healthy and describe half of the answer. Slopes do not help: they point back up the hill the walker is already on, and a rolling ball needs a kick with enough energy to clear the gap, which is rare. On this page only the independence proposal crosses freely, because it does not care where the walker stands, and it works only because the picture is small enough for one wide bell to cover. Defences in real problems include running several chains from different starts and checking that they agree.</p>
 <h4>What this page runs</h4><p>All five samplers in two dimensions, with slopes computed by finite differences. The true left-to-right distribution is known here, because in two dimensions it can be added up on a grid, and it is used for the comparison line. A run stops when the bars are within 0.08 of that line in total area, or after 8,000 steps. The No-U-Turn sampler is the slice-sampling version from the original paper, with at most six doublings (64 leapfrog steps) and a fixed step size; production software also tunes the step size during warm-up.</p>`,
    q: 'On the Banana, raising the Hamiltonian walker from 15 to 40 leapfrog steps makes every move nearly three times as expensive, and the samples per 1,000 look-ups go down, not up. Where does the extra work go? And what does the No-U-Turn walker do about it?',
    a: 'A path of 15 steps already carries the ball most of the way across the Banana, so its end is nearly independent of its start. The extra 25 steps cannot make the sample any more independent than that. Often they do harm: the ball runs up the far side of the valley, turns round and rolls back, and a long, costly path ends near its own beginning. The right length also differs from place to place and from kick to kick, so no single setting is right everywhere. The No-U-Turn walker watches the two ends of its path as it grows and stops when they begin to approach each other, which is the moment more steps stop adding distance. It never has to be told a length. On this small picture you can see what that convenience costs: it builds the path in both directions and uses one point from it, so per look-up it does no better here than a well-chosen fixed length. Its value is that in a real problem nobody knows the well-chosen length in advance.',
    init(root) {
      const c = $('#mc-c', root),
        cr = $('#mc-r', root),
        live = $('#mc-live', root),
        eq = $('#mc-eq', root),
        ol = $('#mc-steps', root),
        S = { al: 'rw', tg: 'corr', s: 0.8, L: 15, cor: true, sp: 2 },
        MAXN = 8000,
        GW = 64,
        GH = 48,
        XR = 4,
        YR = 3,
        NB = 32,
        RN = randnFrom(Math.random);
      const LP = {
        corr: (x, y) =>
          (-(x * x - (2 * 0.85 * x * y) / 0.9 + (y * y) / 0.81) / (2 * (1 - 0.85 * 0.85))) * 0.8,
        banana: (x, y) => -((x * x) / 4.5 + 2.2 * (y + 0.9 - 0.35 * x * x) ** 2),
        two: (x, y) =>
          Math.log(
            Math.exp(-((x + 1.9) ** 2 + (y + 0.2) ** 2) / 0.55) +
              Math.exp(-((x - 1.9) ** 2 + (y - 0.2) ** 2) / 0.55) +
              1e-300,
          ),
      };
      const NAME = {
          rw: 'Random-walk Metropolis',
          mh: 'Metropolis-Hastings (independence proposal)',
          lang: 'Langevin',
          hmc: 'Hamiltonian Monte Carlo',
          nuts: 'No-U-Turn sampler',
        },
        DEF = { rw: 0.8, mh: 1.8, lang: 0.45, hmc: 0.2, nuts: 0.4 },
        SL = {
          rw: 'Stride: typical size of a proposed step',
          mh: 'Width of the fixed proposal bell',
          lang: 'Step size ε',
          hmc: 'Leapfrog step size ε',
          nuts: 'Leapfrog step size ε',
        };
      const HOW = {
        rw: [
          'Start the walker anywhere.',
          'Propose a move: the current place plus a random step. The stride slider sets how big steps tend to be.',
          'Compare the height of the target at the proposed place with the height at the current place.',
          'If the proposed place is higher, go there. If it is lower, go there only with probability equal to the ratio of the two heights. Otherwise stay put.',
          'Write down where the walker is now, even if it did not move. That is one sample.',
          'Repeat thousands of times. The written-down places are distributed like the target.',
        ],
        mh: [
          'Start the walker anywhere.',
          'Propose a place drawn from one fixed wide bell centred on the picture, wherever the walker is standing.',
          'Work out the ratio of target heights, as before.',
          'Multiply by the Hastings correction: how likely the bell is to propose the current place, divided by how likely it is to propose the new one.',
          'Accept with probability equal to that product (or 1 if it is larger). Otherwise stay put.',
          'Write down where the walker is now. Repeat.',
        ],
        lang: [
          'Start the walker anywhere.',
          'Feel the slope where the walker stands.',
          'Propose a move: a short drift uphill plus a random step.',
          'With the correction on, work out the target ratio times the Hastings correction for the uphill lean, and accept or reject. With it off, always move.',
          '—',
          'Write down where the walker is now. Repeat.',
        ],
        hmc: [
          'Start the walker anywhere.',
          'Give it a random kick: a fresh velocity in a random direction.',
          'Let it roll like a ball for the chosen number of leapfrog steps, feeling the slope at every step.',
          'Compare the total energy (height term plus speed term) at the end of the path with the start. Leapfrog nearly conserves it.',
          'Accept the end of the path with probability exp(energy at start − energy at end). Otherwise go back to the start.',
          'Write down where the walker is now. Repeat.',
        ],
        nuts: [
          'Start the walker anywhere.',
          'Give it a random kick, and draw a random energy threshold below its current level.',
          'Grow the path: pick forwards or backwards at random and double the path in that direction.',
          'Stop when the two ends of the path start to approach each other (a U-turn), or a point falls far below the threshold, or after six doublings.',
          'Pick the new place at random from the points on the path that sit above the threshold.',
          'Write down where the walker is now. Repeat.',
        ],
      };
      let lp,
        dens = null,
        dmax = 1,
        margT = null,
        mxT = 0,
        rT = 0,
        x = 0,
        y = 0,
        XS = [],
        YS = [],
        nacc = 0,
        nev = 0,
        prop = null,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        err = NaN,
        P = null,
        hist = null,
        sB,
        cacheE = null;
      const gr = (a, b) => {
          const e = 1e-4;
          return [(lp(a + e, b) - lp(a - e, b)) / (2 * e), (lp(a, b + e) - lp(a, b - e)) / (2 * e)];
        },
        out = (a, b) => Math.abs(a) > XR || Math.abs(b) > YR,
        ht = (l) => (l < -1e8 ? '0 (outside the picture)' : Math.exp(l).toExponential(2));
      const setup = () => {
        lp = LP[S.tg];
        dens = new Float64Array(GW * GH);
        margT = new Float64Array(NB);
        let tot = 0,
          sx = 0,
          rr = 0;
        for (let r = 0; r < GH; r++)
          for (let q = 0; q < GW; q++) {
            const px = -XR + (2 * XR * (q + 0.5)) / GW,
              py = YR - (2 * YR * (r + 0.5)) / GH,
              d = Math.exp(lp(px, py));
            dens[r * GW + q] = d;
            tot += d;
            sx += d * px;
            if (px > 0) rr += d;
            margT[Math.min(NB - 1, Math.floor(((px + XR) / (2 * XR)) * NB))] += d;
          }
        dmax = Math.max(...dens);
        for (let i = 0; i < NB; i++) margT[i] /= tot;
        mxT = sx / tot;
        rT = rr / tot;
      };
      const bin = (v) => Math.min(NB - 1, Math.max(0, Math.floor(((v + XR) / (2 * XR)) * NB)));
      const ui = () => {
        ol.innerHTML = HOW[S.al]
          .filter((s) => s !== '—')
          .map((s) => '<li>' + s + '</li>')
          .join('');
        $('#mc-sl', root).textContent = SL[S.al];
        const cb = $('#mc-cor', root),
          has = S.al === 'mh' || S.al === 'lang' || S.al === 'hmc';
        cb.disabled = !has;
        cb.closest('label').style.opacity = has ? 1 : 0.45;
        $('#mc-corl', root).textContent =
          S.al === 'mh'
            ? 'Include the Hastings correction'
            : S.al === 'lang'
              ? 'Accept or reject each move (off: unadjusted Langevin)'
              : S.al === 'hmc'
                ? 'Accept or reject the end of each path'
                : 'Apply the correction (not used by this walker)';
        $('#mc-L', root).closest('label').style.opacity = S.al === 'hmc' ? 1 : 0.45;
        $('#mc-L', root).disabled = S.al !== 'hmc';
      };
      const build = (keep) => {
        if (!keep) {
          x = S.tg === 'two' ? -1.9 : -2.6;
          y = S.tg === 'two' ? -0.2 : 1.6;
        }
        XS = [x];
        YS = [y];
        nacc = 0;
        nev = 0;
        prop = null;
        fin = false;
        cu = 0;
        err = NaN;
        cacheE = null;
        hist = new Float64Array(NB);
        hist[bin(x)]++;
        msg =
          NAME[S.al] +
          '. The walker starts at (' +
          f2(x) +
          ', ' +
          f2(y) +
          '). Press One step to follow a single decision, or Run.';
        eqShow(eq, ['No steps yet.'], 0);
      };
      const leap = (st, e) => {
        let g0 = gr(st[0], st[1]);
        const px = st[2] + 0.5 * e * g0[0],
          py = st[3] + 0.5 * e * g0[1],
          nx = st[0] + e * px,
          ny = st[1] + e * py;
        g0 = gr(nx, ny);
        nev++;
        return [nx, ny, px + 0.5 * e * g0[0], py + 0.5 * e * g0[1]];
      };
      const lj = (st) =>
        (out(st[0], st[1]) ? -1e9 : lp(st[0], st[1])) - 0.5 * (st[2] * st[2] + st[3] * st[3]);
      const moves = {
        rw(q) {
          const px = x + S.s * RN(),
            py = y + S.s * RN(),
            l0 = lp(x, y),
            l1 = out(px, py) ? -1e9 : lp(px, py),
            ratio = Math.exp(Math.min(0, l1 - l0)),
            u = Math.random(),
            ok = u < ratio;
          nev++;
          prop = {
            path: [
              [x, y],
              [px, py],
            ],
            ok,
          };
          if (!q)
            eqShow(
              eq,
              [
                'Step ' +
                  XS.length +
                  '. The walker is at (' +
                  f2(x) +
                  ', ' +
                  f2(y) +
                  '), where the target has height ' +
                  ht(l0) +
                  '.',
                'Proposed place: (' + f2(px) + ', ' + f2(py) + '), height ' + ht(l1) + '.',
                'ratio = proposed height ÷ current height = <b>' +
                  (l1 - l0 > 0 ? 'more than 1' : f3(Math.exp(l1 - l0))) +
                  '</b>',
                l1 >= l0
                  ? 'The proposed place is higher, so the move is <b>accepted</b>.'
                  : 'It is lower, so roll a number between 0 and 1: ' +
                    f3(u) +
                    '. That is ' +
                    (ok
                      ? 'below the ratio, so the move is <b>accepted</b>.'
                      : 'above the ratio, so the move is <b>rejected</b>. The walker stays, and its current place is written down again.'),
              ],
              0,
            );
          return ok ? [px, py] : null;
        },
        mh(q) {
          const s = S.s,
            px = s * RN(),
            py = s * RN(),
            l0 = lp(x, y),
            l1 = out(px, py) ? -1e9 : lp(px, py),
            lq = (a, b) => -(a * a + b * b) / (2 * s * s),
            hc = lq(x, y) - lq(px, py),
            la = l1 - l0 + (S.cor ? hc : 0),
            u = Math.random(),
            ok = u < Math.exp(Math.min(0, la));
          nev++;
          prop = {
            path: [
              [x, y],
              [px, py],
            ],
            ok,
          };
          if (!q)
            eqShow(
              eq,
              [
                'Step ' +
                  XS.length +
                  '. The walker is at (' +
                  f2(x) +
                  ', ' +
                  f2(y) +
                  '). The fixed bell proposes (' +
                  f2(px) +
                  ', ' +
                  f2(py) +
                  ').',
                'target ratio = proposed height ÷ current height = <b>' +
                  (l1 < -1e8 ? '0' : Math.exp(l1 - l0).toPrecision(3)) +
                  '</b>',
                'Hastings correction = chance the bell proposes the current place ÷ chance it proposes the new one = <b>' +
                  Math.exp(hc).toPrecision(3) +
                  '</b>' +
                  (S.cor ? '' : ' (switched off, so it is left out)'),
                'acceptance probability = min(1, ' +
                  (S.cor ? 'ratio × correction' : 'ratio alone') +
                  ') = <b>' +
                  f3(Math.exp(Math.min(0, la))) +
                  '</b>',
                'Rolled ' + f3(u) + ': the move is <b>' + (ok ? 'accepted' : 'rejected') + '</b>.',
              ],
              0,
            );
          return ok ? [px, py] : null;
        },
        lang(q) {
          const e = S.s,
            g0 = gr(x, y),
            mx = x + 0.5 * e * e * g0[0],
            my = y + 0.5 * e * e * g0[1],
            px = mx + e * RN(),
            py = my + e * RN();
          nev++;
          if (!S.cor) {
            const cx = clamp(px, -XR, XR),
              cy = clamp(py, -YR, YR);
            prop = {
              path: [
                [x, y],
                [mx, my],
                [cx, cy],
              ],
              ok: true,
            };
            if (!q)
              eqShow(
                eq,
                [
                  'Step ' +
                    XS.length +
                    '. The walker is at (' +
                    f2(x) +
                    ', ' +
                    f2(y) +
                    '). The slope there is (' +
                    f2(g0[0]) +
                    ', ' +
                    f2(g0[1]) +
                    ').',
                  'uphill drift = ε²/2 × slope = (' +
                    f2(0.5 * e * e * g0[0]) +
                    ', ' +
                    f2(0.5 * e * e * g0[1]) +
                    ')',
                  'random step of typical size ε = ' + f2(e) + ' added on top',
                  'New place: <b>(' +
                    f2(cx) +
                    ', ' +
                    f2(cy) +
                    ')</b>. There is no test, so the walker always moves. This is the unadjusted Langevin algorithm.',
                ],
                0,
              );
            return [cx, cy];
          }
          const l0 = lp(x, y),
            l1 = out(px, py) ? -1e9 : lp(px, py),
            g1 = gr(px, py),
            bx = px + 0.5 * e * e * g1[0],
            by = py + 0.5 * e * e * g1[1],
            lf = -((px - mx) ** 2 + (py - my) ** 2) / (2 * e * e),
            lb = -((x - bx) ** 2 + (y - by) ** 2) / (2 * e * e),
            hc = lb - lf,
            la = l1 - l0 + hc,
            u = Math.random(),
            ok = u < Math.exp(Math.min(0, la));
          prop = {
            path: [
              [x, y],
              [mx, my],
              [px, py],
            ],
            ok,
          };
          if (!q)
            eqShow(
              eq,
              [
                'Step ' +
                  XS.length +
                  '. The walker is at (' +
                  f2(x) +
                  ', ' +
                  f2(y) +
                  '). The slope there is (' +
                  f2(g0[0]) +
                  ', ' +
                  f2(g0[1]) +
                  ').',
                'Proposal = current place + uphill drift (' +
                  f2(0.5 * e * e * g0[0]) +
                  ', ' +
                  f2(0.5 * e * e * g0[1]) +
                  ') + random step = (' +
                  f2(px) +
                  ', ' +
                  f2(py) +
                  ')',
                'target ratio = <b>' +
                  (l1 < -1e8 ? '0' : Math.exp(l1 - l0).toPrecision(3)) +
                  '</b>, Hastings correction for the uphill lean = <b>' +
                  Math.exp(hc).toPrecision(3) +
                  '</b>',
                'acceptance probability = min(1, ratio × correction) = <b>' +
                  f3(Math.exp(Math.min(0, la))) +
                  '</b>',
                'Rolled ' + f3(u) + ': the move is <b>' + (ok ? 'accepted' : 'rejected') + '</b>.',
              ],
              0,
            );
          return ok ? [px, py] : null;
        },
        hmc(q) {
          let st = [x, y, RN(), RN()];
          const p0 = [st[2], st[3]],
            H0 = -lj(st),
            path = [[x, y]];
          for (let i = 0; i < S.L; i++) {
            st = leap(st, S.s);
            path.push([st[0], st[1]]);
          }
          const H1 = -lj(st),
            u = Math.random(),
            bad = out(st[0], st[1]),
            ok = !bad && (!S.cor || u < Math.exp(Math.min(0, H0 - H1)));
          prop = { path, ok };
          if (!q)
            eqShow(
              eq,
              [
                'Step ' +
                  XS.length +
                  '. The walker is at (' +
                  f2(x) +
                  ', ' +
                  f2(y) +
                  ') and is kicked with velocity (' +
                  f2(p0[0]) +
                  ', ' +
                  f2(p0[1]) +
                  ').',
                'energy at the start = −log height + ½ speed² = <b>' + f2(H0) + '</b>',
                'It rolls for ' +
                  S.L +
                  ' leapfrog steps of size ' +
                  f2(S.s) +
                  ' and ends at (' +
                  f2(st[0]) +
                  ', ' +
                  f2(st[1]) +
                  ')' +
                  (bad ? ', outside the picture' : '') +
                  '.',
                'energy at the end = <b>' +
                  (bad ? '∞' : f2(H1)) +
                  '</b>, a change of ' +
                  (bad ? '∞' : (H1 - H0 >= 0 ? '+' : '') + f3(H1 - H0)),
                bad
                  ? 'The move is <b>rejected</b>.'
                  : S.cor
                    ? 'acceptance probability = min(1, exp(−change)) = <b>' +
                      f3(Math.exp(Math.min(0, H0 - H1))) +
                      '</b>. Rolled ' +
                      f3(u) +
                      ': <b>' +
                      (ok ? 'accepted' : 'rejected') +
                      '</b>.'
                    : 'The test is switched off, so the walker <b>moves</b> regardless.',
              ],
              0,
            );
          return ok ? [st[0], st[1]] : null;
        },
        nuts(q) {
          const p0 = [RN(), RN()],
            s0 = [x, y, p0[0], p0[1]],
            J0 = lj(s0),
            logu = J0 + Math.log(Math.random() || 1e-300),
            e = S.s,
            fw = [],
            bw = [];
          let why = '',
            div = false;
          const nou = (a, b) =>
            (b[0] - a[0]) * a[2] + (b[1] - a[1]) * a[3] >= 0 &&
            (b[0] - a[0]) * b[2] + (b[1] - a[1]) * b[3] >= 0;
          const tree = (st, v, j) => {
            if (j === 0) {
              const s1 = leap(st, v * e),
                J = lj(s1);
              (v > 0 ? fw : bw).push([s1[0], s1[1]]);
              const s = logu < J + 1000;
              if (!s) div = true;
              return { m: s1, p: s1, c: s1, n: logu <= J ? 1 : 0, s };
            }
            const a = tree(st, v, j - 1);
            if (!a.s) return a;
            const b = tree(v < 0 ? a.m : a.p, v, j - 1),
              m = v < 0 ? b.m : a.m,
              p = v < 0 ? a.p : b.p,
              n = a.n + b.n;
            return {
              m,
              p,
              c: n > 0 && Math.random() < b.n / n ? b.c : a.c,
              n,
              s: b.s && nou(m, p),
            };
          };
          let m = s0,
            p = s0,
            cand = s0,
            n = 1,
            j = 0,
            s = true;
          while (s && j < 6) {
            const v = Math.random() < 0.5 ? -1 : 1,
              r = tree(v < 0 ? m : p, v, j);
            if (v < 0) m = r.m;
            else p = r.p;
            if (r.s && Math.random() < r.n / n) cand = r.c;
            n += r.n;
            const nt = nou(m, p);
            if (!r.s)
              why = div
                ? 'a point on the path fell far below the energy threshold'
                : 'a U-turn appeared inside the newest half of the path';
            else if (!nt) why = 'the two ends of the path began to approach each other';
            s = r.s && nt;
            j++;
          }
          if (s) why = 'it reached the limit of six doublings';
          const path = bw
              .slice()
              .reverse()
              .concat([[x, y]], fw),
            moved = cand !== s0;
          prop = { path, ok: true, pick: [cand[0], cand[1]], nuts: true };
          if (!q)
            eqShow(
              eq,
              [
                'Step ' +
                  XS.length +
                  '. The walker is at (' +
                  f2(x) +
                  ', ' +
                  f2(y) +
                  ') and is kicked with velocity (' +
                  f2(p0[0]) +
                  ', ' +
                  f2(p0[1]) +
                  ').',
                'The path grew in <b>' +
                  j +
                  '</b> round' +
                  (j === 1 ? '' : 's') +
                  ' of doubling, forwards or backwards at random each time, to ' +
                  (fw.length + bw.length) +
                  ' leapfrog steps.',
                'It stopped growing because ' + why + '.',
                '<b>' +
                  n +
                  '</b> of the ' +
                  (fw.length + bw.length + 1) +
                  ' points on the path sit above the energy threshold and were allowed.',
                'New place, picked from the allowed points: <b>(' +
                  f2(cand[0]) +
                  ', ' +
                  f2(cand[1]) +
                  ')</b>' +
                  (moved ? '' : ' — the starting point itself') +
                  '.',
              ],
              0,
            );
          return moved ? [cand[0], cand[1]] : null;
        },
      };
      const step = (quiet) => {
        if (fin) return;
        const x0 = x,
          y0 = y,
          r = moves[S.al](quiet),
          ok = !!r;
        if (ok) {
          x = r[0];
          y = r[1];
          nacc++;
        }
        if (!quiet) {
          cu = S.al === 'rw' ? 3 : S.al === 'mh' ? 4 : S.al === 'lang' ? 3 : 4;
          msg =
            'Step ' +
            XS.length +
            ': ' +
            (ok
              ? 'moved to (' +
                f2(x) +
                ', ' +
                f2(y) +
                ')' +
                (S.al === 'hmc' || S.al === 'nuts'
                  ? ', ' +
                    f2(Math.hypot(x - x0, y - y0)) +
                    ' away, along a path of ' +
                    (prop.path.length - 1) +
                    ' leapfrog steps.'
                  : '.')
              : (S.al === 'nuts'
                  ? 'the starting point was picked from the path, so the walker stayed at ('
                  : 'the proposal was rejected, so the walker stayed at (') +
                f2(x) +
                ', ' +
                f2(y) +
                ') and that place was recorded again.');
        }
        XS.push(x);
        YS.push(y);
        hist[bin(x)]++;
        cacheE = null;
        const n = XS.length;
        if (n % 100 === 0 || n >= MAXN) {
          let e = 0;
          for (let i = 0; i < NB; i++) e += Math.abs(hist[i] / n - margT[i]);
          err = e;
          if ((n >= 400 && e < 0.08) || n >= MAXN) {
            fin = true;
            cu = 5;
            const es = ess();
            msg =
              NAME[S.al] +
              '. ' +
              (e < 0.08
                ? 'Stopped after ' + (n - 1) + ' steps: the bars are within 0.08 of the true line.'
                : 'Stopped at the limit of ' +
                  MAXN +
                  ' steps with a gap of ' +
                  f2(e) +
                  ' still left' +
                  (S.tg === 'two' && Math.abs(rightShare() - rT) > 0.15
                    ? '. The walker has not divided its time correctly between the two islands'
                    : !S.cor && (S.al === 'mh' || S.al === 'lang' || S.al === 'hmc')
                      ? '. With the correction off this walker samples from a slightly different landscape, and no amount of running fixes that'
                      : '') +
                  '.') +
              ' ' +
              pct(nacc / (n - 1)) +
              ' of moves changed place, the samples are worth about ' +
              Math.round(es) +
              ' independent ones, and they cost ' +
              nev.toLocaleString() +
              ' look-ups.';
          }
        }
      };
      const rightShare = () => {
        let r = 0;
        for (const v of XS) if (v > 0) r++;
        return r / XS.length;
      };
      const ess = () => {
        if (cacheE != null) return cacheE;
        const n = XS.length;
        if (n < 50) return NaN;
        let m = 0;
        for (const v of XS) m += v / n;
        let v0 = 0;
        for (const v of XS) v0 += (v - m) ** 2 / n;
        if (v0 < 1e-12) return (cacheE = 1);
        let s = 0;
        const K = Math.min(600, Math.floor(n / 3));
        for (let k = 1; k < K; k++) {
          let cc = 0;
          for (let i = 0; i + k < n; i++) cc += (XS[i] - m) * (XS[i + k] - m);
          const r = cc / (n - k) / v0;
          if (r < 0.02) break;
          s += r;
        }
        return (cacheE = Math.min(n, n / (1 + 2 * s)));
      };
      const draw = () => {
        const t = T(),
          n = XS.length;
        {
          const V = uView(c),
            { g, w, h, X, Y } = V;
          P = V;
          for (let r = 0; r < GH; r++)
            for (let q = 0; q < GW; q++) {
              const v = dens[r * GW + q] / dmax;
              if (v < 0.01) continue;
              g.fillStyle = mix(t.accent, t.accent, 0, Math.min(0.85, Math.pow(v, 0.55) * 0.8));
              const xa = Math.round(X(-XR + (2 * XR * q) / GW)),
                xb = Math.round(X(-XR + (2 * XR * (q + 1)) / GW)),
                ya = Math.round(Y(YR - (2 * YR * r) / GH)),
                yb = Math.round(Y(YR - (2 * YR * (r + 1)) / GH));
              g.fillRect(xa, ya, xb - xa, yb - ya);
            }
          if (S.al === 'mh') {
            g.strokeStyle = mix(t.cut, t.cut, 0, 0.55);
            g.lineWidth = 1.2;
            g.setLineDash([4, 5]);
            for (const k of [1, 2]) {
              g.beginPath();
              g.ellipse(X(0), Y(0), k * S.s * V.s, k * S.s * V.s, 0, 0, 7);
              g.stroke();
            }
            g.setLineDash([]);
          }
          g.fillStyle = mix(t.ink, t.ink, 0, 0.45);
          const st = Math.max(1, Math.floor(n / 1500));
          for (let i = 0; i < n; i += st) {
            g.beginPath();
            g.arc(X(XS[i]), Y(YS[i]), 1.5, 0, 7);
            g.fill();
          }
          g.strokeStyle = t.ink;
          g.lineWidth = 1;
          g.globalAlpha = 0.6;
          g.beginPath();
          for (let i = Math.max(0, n - 60); i < n; i++)
            i === Math.max(0, n - 60) ? g.moveTo(X(XS[i]), Y(YS[i])) : g.lineTo(X(XS[i]), Y(YS[i]));
          g.stroke();
          g.globalAlpha = 1;
          if (prop) {
            const pp = prop.path,
              e = pp[pp.length - 1];
            g.strokeStyle = prop.ok ? t.gold : t.cut;
            g.lineWidth = 2.5;
            g.setLineDash(prop.ok ? [] : [5, 4]);
            g.beginPath();
            pp.forEach((q, i) => (i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1]))));
            g.stroke();
            g.setLineDash([]);
            if (pp.length > 3) {
              g.fillStyle = prop.ok ? t.gold : t.cut;
              for (const q of pp) {
                g.beginPath();
                g.arc(X(q[0]), Y(q[1]), 2.2, 0, 7);
                g.fill();
              }
            }
            if (prop.nuts) {
              for (const q of [pp[0], e]) {
                g.strokeStyle = t.cut;
                g.lineWidth = 2;
                g.beginPath();
                g.arc(X(q[0]), Y(q[1]), 5, 0, 7);
                g.stroke();
              }
              g.font = MONO;
              g.fillStyle = t.muted;
              g.textAlign = 'left';
              g.fillText('path ends', X(e[0]) + 8, Y(e[1]) + 4);
            } else {
              g.beginPath();
              g.arc(X(e[0]), Y(e[1]), 5, 0, 7);
              g.stroke();
              g.font = MONO;
              g.fillStyle = prop.ok ? t.cut : t.muted;
              g.textAlign = 'left';
              g.fillText(
                prop.ok ? (S.cor || S.al === 'rw' ? 'accepted' : 'moved') : 'rejected',
                X(e[0]) + 8,
                Y(e[1]) + 4,
              );
            }
          }
          g.fillStyle = t.gold;
          g.strokeStyle = t.bg;
          g.lineWidth = 2;
          g.beginPath();
          g.arc(X(x), Y(y), 7, 0, 7);
          g.fill();
          g.stroke();
          g2Note(
            g,
            t,
            h,
            'the gold dot is the walker' +
              (S.al === 'mh' ? '; dashed rings: the fixed proposal bell' : ''),
          );
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.46,
            A = uLine(g, t, 0, 0, w, h1, [], {
              title: 'left-to-right position at every step',
              n: Math.max(300, n),
              y1: -XR,
              ym: XR,
            });
          g.strokeStyle = t.accent;
          g.lineWidth = 1.1;
          g.beginPath();
          const s2 = Math.max(1, Math.floor(n / 1200));
          for (let i = 0; i < n; i += s2)
            i ? g.lineTo(A.X(i), A.Y(XS[i])) : g.moveTo(A.X(i), A.Y(XS[i]));
          g.stroke();
          const m = { l: 44, r: 12, t: 26, b: 20 },
            y0 = h1,
            H = h - h1,
            bw = (w - m.l - m.r) / NB,
            mx = Math.max(...margT, ...Array.from(hist).map((v) => v / n)) * 1.08,
            Yh = (v) => y0 + H - m.b - (v / mx) * (H - m.t - m.b);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            'time spent at each left-to-right position (bars) and the truth (line)',
            8,
            y0 + 13,
          );
          for (let i = 0; i < NB; i++) {
            g.fillStyle = mix(t.accent, t.accent, 0, 0.65);
            g.fillRect(m.l + i * bw + 0.5, Yh(hist[i] / n), bw - 1, Yh(0) - Yh(hist[i] / n));
          }
          g.strokeStyle = t.cut;
          g.lineWidth = 2.2;
          g.beginPath();
          for (let i = 0; i < NB; i++)
            i
              ? g.lineTo(m.l + (i + 0.5) * bw, Yh(margT[i]))
              : g.moveTo(m.l + 0.5 * bw, Yh(margT[i]));
          g.stroke();
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, y0 + H - m.b);
          g.lineTo(w - m.r, y0 + H - m.b);
          g.stroke();
          g.fillStyle = t.muted;
          g.fillText('left', m.l, h - 5);
          g.textAlign = 'right';
          g.fillText('right', w - m.r, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'mc', cu);
        let mx = 0;
        for (const v of XS) mx += v / n;
        const es = fin || !anim.on || n % 200 < 41 ? ess() : cacheE0;
        cacheE0 = es;
        setR(root, 'n', n - 1 + (fin ? ' (stopped)' : ''));
        setR(root, 'acc', n > 1 ? pct(nacc / (n - 1)) : '–');
        setR(root, 'mx', f2(mx) + ' / ' + f2(mxT));
        setR(root, 'ess', isFinite(es) ? String(Math.round(es)) : '–');
        setR(root, 'cost', nev.toLocaleString());
        setR(root, 'eff', isFinite(es) && nev > 0 ? ((1000 * es) / nev).toFixed(0) : '–');
        setR(root, 'err', isFinite(err) ? f2(err) : '–');
        setR(root, 'isl', pct(rightShare()) + ' / ' + pct(rT));
      };
      let cacheE0 = NaN;
      setup();
      ui();
      build(false);
      if (RM) {
        while (!fin) step(true);
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        const hv = S.al === 'hmc' || S.al === 'nuts';
        if (S.sp === 1) {
          if (fr % 34) return;
          step(false);
        } else if (S.sp === 2) {
          if (XS.length < 25 || hv) {
            if (fr % (hv && XS.length >= 25 ? 3 : 10)) return;
            step(false);
          } else for (let i = 0; i < 10 && !fin; i++) step(i < 9);
        } else {
          const k = hv ? 10 : 40;
          for (let i = 0; i < k && !fin; i++) step(i < k - 1);
        }
        draw();
        chk();
      });
      pb = playBtn($('#mc-play', root), anim);
      chk = runCtl(pb, root, 'mc', {
        done: () => fin,
        restart: () => build(false),
        count: () => XS.length - 1,
        what: 'steps',
      });
      $('#mc-step', root).onclick = () => {
        pb.stop();
        if (fin) build(false);
        else step(false);
        draw();
      };
      $('#mc-reset', root).onclick = () => {
        pb.stop();
        build(false);
        draw();
      };
      sB = bind(
        root,
        'mc-s',
        (v) => {
          S.s = +v;
        },
        f2,
      );
      bind(
        root,
        'mc-L',
        (v) => {
          S.L = +v;
        },
        (v) => String(+v),
      );
      bind(
        root,
        'mc-sp',
        (v) => {
          S.sp = +v;
        },
        (v) => ['', 'one step at a time', 'steady', 'fast'][+v],
      );
      $('#mc-cor', root).onchange = (e) => {
        S.cor = e.target.checked;
        pb.stop();
        build(false);
        draw();
      };
      segBind(root, 'mc-al', S.al, (v) => {
        S.al = v;
        S.s = DEF[v];
        sB.set(S.s);
        pb.stop();
        ui();
        build(false);
        draw();
        pb.start();
      });
      segBind(root, 'mc-tg', S.tg, (v) => {
        S.tg = v;
        pb.stop();
        setup();
        build(false);
        draw();
        pb.start();
      });
      c.style.cursor = 'crosshair';
      c.addEventListener('click', (e) => {
        if (!P) return;
        const r = c.getBoundingClientRect();
        x = clamp((e.clientX - r.left - P.w / 2) / P.s, -XR, XR);
        y = clamp(-(e.clientY - r.top - P.h / 2) / P.s, -YR, YR);
        pb.stop();
        build(true);
        msg =
          'The walker has been put at (' +
          f2(x) +
          ', ' +
          f2(y) +
          ') and its diary cleared. Press Run or One step.';
        draw();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          build(false);
          draw();
          pb.start();
        },
      };
    },
  },
  {
    id: 'hmm',
    tab: 'Hidden Markov models',
    title: 'Hidden Markov models: guess what you cannot see from what you can',
    lede: 'A friend who lives far away tells you every evening what she did that day: went for a walk, went shopping, or stayed in and cleaned. She never mentions the weather. Can you work out, day by day, whether it was sunny or rainy there? You know two things: weather tends to last, and she walks more in the sun. A hidden Markov model turns those two facts into a calculation. The same calculation recognises speech from sound, finds genes in DNA and follows a phone’s position between GPS fixes.',
    stage: `<div class="row seg" id="hm-v" role="group" aria-label="Which question"><button class="btn" type="button" data-v="fwd">Belief Day by Day (Forward Algorithm)</button><button class="btn" type="button" data-v="vit">The Single Best Story (Viterbi)</button></div>
 <div class="stage"><figure class="fig"><figcaption>Top row: what she did each day. Click a day to change it. Then two rows of circles, sunny and rainy: the fuller the circle, the more the model believes it. The gold line is the model’s answer</figcaption><canvas class="short" id="hm-c" role="img" aria-label="Observations and the belief in each hidden state over time"></canvas></figure>
 <figure class="fig"><figcaption>Top: the probability of rain on each day, as judged on the day itself and as judged afterwards with all the evidence. Bottom: the two tables the model uses</figcaption><canvas class="short" id="hm-r" role="img" aria-label="Probability of rain over time and the model's transition and emission tables"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The calculation for the current day</figcaption><div class="eq" id="hm-eq" aria-live="polite" style="min-height:9.5em;margin-top:6px"></div></figure>
 ${uHow('hm', ['Day 1: start from even odds for sun and rain, and weigh each by how likely today’s activity is in that weather.', 'Each later day, first predict: push yesterday’s belief through the “weather tends to last” table. This is the belief before hearing from her.', 'Then update: multiply the belief in each weather by how likely her activity is in that weather, and rescale so the two add up to 1.', 'Repeat to the last day. These are the day-by-day beliefs.', 'Viterbi instead keeps, for each weather on each day, the single most probable story that ends there, and remembers which weather it came from.', 'At the last day, pick the better ending and trace the remembered links backwards. That is the single best story.'])}
 <div class="controls">${uBtns('hm', 'Run', 'One day')}
 ${grp('The model', slider('hm-st', 'Weather tends to last: chance tomorrow is the same as today', 0.5, 0.98, 0.01, 0.8) + slider('hm-em', 'How strongly the weather shows in what she does', 0, 1, 0.05, 1))}
 ${grp('The data', `<div class="row"><button class="btn" id="hm-new" type="button">A New Fortnight</button><button class="btn" id="hm-tr" type="button">Show the True Weather</button></div>`)}</div>
 <dl class="read">${rd('d', 'Day')}${rd('b', 'Belief today: sunny / rainy', 'hl')}${rd('af', 'Days guessed right, judging each day as it comes')}${rd('as', 'Days guessed right, judging afterwards with all the evidence', 'hl2')}${rd('av', 'Days guessed right by the single best story')}${rd('lk', 'Probability of the whole fortnight of activities under the model')}</dl>`,
    tryit:
      'Press One day and read the calculation: predict, then update. Find a day where she went shopping, which she does equally often in any weather. What does the belief do on such a day? Press Run to the end, then press Show the True Weather. Compare the two lines on the right: on which days does hindsight change the answer, and why can a later walk change your mind about an earlier day? Click days in the top row to make her clean on a single day in the middle of a sunny spell: is one day enough to convince the model it rained? Lower the first slider to 0.5 and try again. Switch to The Single Best Story and compare its answer with the day-by-day one.',
    math: String.raw`<h4>Intuition</h4><p>You are in a room with no windows, and every so often a colleague walks in. Sometimes they carry a wet umbrella. One wet umbrella makes rain likely. A dry colleague five minutes later does not make you think the rain has stopped, because rain rarely stops that quickly and not everyone carries an umbrella. You are combining two kinds of knowledge without noticing: how the hidden thing behaves over time, and how it shows itself. A hidden Markov model writes both down as tables and does the combining exactly.</p>
 <h4>The model</h4><p>There is a hidden state \(z_t\) for each day, here sunny or rainy, and an observation \(x_t\), here walk, shop or clean. Two assumptions make the model tractable. The state depends only on yesterday's state, through a <b>transition</b> table \(A_{ij}=P(z_t=j\mid z_{t-1}=i)\). The observation depends only on today's state, through an <b>emission</b> table \(B_j(x)=P(x_t=x\mid z_t=j)\). With a starting distribution \(\pi\), the probability of a whole hidden story together with the observations is</p>$$P(z_{1:T},x_{1:T})=\pi_{z_1}B_{z_1}(x_1)\prod_{t=2}^{T}A_{z_{t-1}z_t}\,B_{z_t}(x_t).$$
 <h4>The forward algorithm: belief day by day</h4><p>Let \(\alpha_t(j)=P(z_t=j\mid x_{1:t})\) be the belief on day \(t\) given everything heard so far. It is computed from yesterday's belief in two moves:</p>$$\text{predict:}\quad \bar\alpha_t(j)=\sum_i\alpha_{t-1}(i)\,A_{ij},\qquad\text{update:}\quad \alpha_t(j)=\frac{\bar\alpha_t(j)\,B_j(x_t)}{\sum_k\bar\alpha_t(k)\,B_k(x_t)}.$$<p>Predict blurs the belief a little, because the weather might have changed overnight. Update sharpens it with today's evidence. This is Bayes' rule applied once per day, with yesterday's posterior serving as today's prior. The cost is proportional to the number of days, although the number of possible stories doubles with every day. The normalising sums, multiplied together, give \(P(x_{1:T})\), the probability of all the observations, shown in the last readout.</p>
 <h4>Hindsight: smoothing</h4><p>Once the fortnight is over, earlier days can be judged again using what came after. A second pass runs backwards, \(\beta_t(i)=\sum_jA_{ij}B_j(x_{t+1})\beta_{t+1}(j)\), and the two are combined:</p>$$P(z_t=j\mid x_{1:T})\propto\alpha_t(j)\,\beta_t(j).$$<p>This is the forward-backward algorithm, and it is the second line in the top-right chart. It is never less informed than the day-by-day belief, and it is usually right more often.</p>
 <h4>Viterbi: the single best story</h4><p>The most probable whole sequence is a different question, and choosing the most probable state on each day separately does not answer it: the separate choices might string together into a story the model considers nearly impossible. Viterbi (1967) keeps, for each state on each day, the probability of the best story ending there,</p>$$\delta_t(j)=\max_i\ \delta_{t-1}(i)\,A_{ij}\,B_j(x_t),$$<p>and remembers which \(i\) achieved the maximum. At the end it takes the larger \(\delta_T\) and follows the remembered links back to day 1. It is the forward algorithm with a maximum in place of a sum.</p>
 <h4>What the two sliders do</h4><p>When the weather is very persistent, one odd activity is explained away as an off day, and the belief hardly moves. When persistence is 0.5 the weather has no memory, and each day is judged by its own activity alone. When the second slider is at 0 her activities say nothing about the weather, and the belief stays at even odds whatever she does. The model is only as informative as its two tables.</p>
 <h4>Where the tables come from</h4><p>Here they are given. In practice they are learned from sequences of observations alone with the Baum-Welch algorithm, a form of the expectation-maximisation method from the Gaussian Mixtures tab: guess the tables, use forward-backward to estimate the hidden states, re-estimate the tables from those, and repeat.</p>
 <h4>What this page runs</h4><p>Fourteen days. The true weather is drawn from the model itself, then her activities from the weather. In sunny weather she walks 60% of the time, shops 30% and cleans 10%, and in rain the reverse, when the second slider is at 1. All probabilities are computed exactly.</p>`,
    q: 'On day 6 the day-by-day belief said 70% sunny. After the fortnight is over, the hindsight line says day 6 was 75% rainy. Nothing about day 6 itself has changed. Where did the new information come from?',
    a: 'From days 7 onward. Weather tends to last, so what happened after day 6 is evidence about day 6 itself. If she cleaned on days 7, 8 and 9, those days were very probably rainy, and a rainy spell probably did not begin exactly on day 7: it is quite likely to have started a day earlier, with day 6 being a rainy day on which she happened to shop or walk. The forward pass could not know this on day 6. The backward pass carries the later evidence back through the “weather tends to last” table.',
    init(root) {
      const c = $('#hm-c', root),
        cr = $('#hm-r', root),
        live = $('#hm-live', root),
        eq = $('#hm-eq', root),
        S = { v: 'fwd', st: 0.8, em: 1, show: false },
        TT = 14,
        ON = ['walk', 'shop', 'clean'],
        SN = ['sunny', 'rainy'];
      let Z = [],
        X = [],
        AL = [],
        PRD = [],
        DE = [],
        BP = [],
        SM = [],
        VP = [],
        t = 0,
        ll = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        fin = false,
        P = null;
      const Bm = () => {
          const e = S.em,
            u = 1 / 3;
          return [
            [u + e * (0.6 - u), u + e * (0.3 - u), u + e * (0.1 - u)],
            [u + e * (0.1 - u), u + e * (0.3 - u), u + e * (0.6 - u)],
          ];
        },
        A = () => [
          [S.st, 1 - S.st],
          [1 - S.st, S.st],
        ];
      const gen = () => {
        const a = A(),
          b = Bm();
        Z = [];
        X = [];
        let z = Math.random() < 0.5 ? 0 : 1;
        for (let i = 0; i < TT; i++) {
          if (i) z = Math.random() < a[z][z] ? z : 1 - z;
          Z.push(z);
          const u = Math.random();
          X.push(u < b[z][0] ? 0 : u < b[z][0] + b[z][1] ? 1 : 2);
        }
      };
      const smooth = () => {
        const a = A(),
          b = Bm(),
          al = [],
          cs = [];
        let p = [0.5, 0.5];
        for (let i = 0; i < TT; i++) {
          const pr = i
              ? [p[0] * a[0][0] + p[1] * a[1][0], p[0] * a[0][1] + p[1] * a[1][1]]
              : [0.5, 0.5],
            u = [pr[0] * b[0][X[i]], pr[1] * b[1][X[i]]],
            s = u[0] + u[1];
          p = [u[0] / s, u[1] / s];
          al.push(p);
          cs.push(s);
        }
        let be = [1, 1];
        const out = new Array(TT);
        for (let i = TT - 1; i >= 0; i--) {
          const g = [al[i][0] * be[0], al[i][1] * be[1]],
            s = g[0] + g[1];
          out[i] = [g[0] / s, g[1] / s];
          if (i) {
            const nb = [0, 1].map(
              (q) => a[q][0] * b[0][X[i]] * be[0] + a[q][1] * b[1][X[i]] * be[1],
            );
            const m = nb[0] + nb[1];
            be = [nb[0] / m, nb[1] / m];
          }
        }
        return out;
      };
      const reset = () => {
        AL = [];
        PRD = [];
        DE = [];
        BP = [];
        VP = [];
        t = 0;
        ll = 0;
        fin = false;
        cu = 0;
        SM = smooth();
        msg =
          'Fourteen evenings of reports are in the top row. Nothing has been worked out yet. Press One day or Run.';
        eqShow(eq, ['No days processed yet.'], 0);
      };
      const step = () => {
        if (fin) return;
        const a = A(),
          b = Bm(),
          x = X[t],
          d = t + 1,
          L = [
            'Day ' +
              d +
              ': she went to <b>' +
              (x === 0 ? 'walk' : x === 1 ? 'the shops' : 'clean the house') +
              '</b>.',
          ];
        {
          const p = t ? AL[t - 1] : [0.5, 0.5],
            pr = t
              ? [p[0] * a[0][0] + p[1] * a[1][0], p[0] * a[0][1] + p[1] * a[1][1]]
              : [0.5, 0.5],
            u = [pr[0] * b[0][x], pr[1] * b[1][x]],
            s = u[0] + u[1];
          PRD.push(pr);
          AL.push([u[0] / s, u[1] / s]);
          ll += Math.log(s);
          if (S.v === 'fwd') {
            L.push(
              t
                ? 'predict: P(sunny today) = ' +
                    f3(p[0]) +
                    ' × ' +
                    f2(a[0][0]) +
                    ' + ' +
                    f3(p[1]) +
                    ' × ' +
                    f2(a[1][0]) +
                    ' = <b>' +
                    f3(pr[0]) +
                    '</b>,  so P(rainy) = ' +
                    f3(pr[1])
                : 'before hearing from her: sunny 0.500, rainy 0.500',
              'weigh by her activity:  sunny ' +
                f3(pr[0]) +
                ' × P(' +
                ON[x] +
                ' | sunny) ' +
                f2(b[0][x]) +
                ' = ' +
                f3(u[0]) +
                '     rainy ' +
                f3(pr[1]) +
                ' × P(' +
                ON[x] +
                ' | rainy) ' +
                f2(b[1][x]) +
                ' = ' +
                f3(u[1]),
              'rescale to add up to 1:  sunny ' +
                f3(u[0]) +
                ' ÷ ' +
                f3(s) +
                ' = <b>' +
                f3(u[0] / s) +
                '</b>     rainy <b>' +
                f3(u[1] / s) +
                '</b>',
            );
            cu = t ? 2 : 0;
            msg =
              'Day ' +
              d +
              ' (' +
              ON[x] +
              '): before her report the model expected sun with probability ' +
              pct(pr[0]) +
              '. After it, ' +
              pct(u[0] / s) +
              '.' +
              (Math.abs(b[0][x] - b[1][x]) < 0.02
                ? ' This activity is equally likely in any weather, so the report changed nothing: only the prediction step moved the belief.'
                : '');
          }
        }
        {
          const p = t ? DE[t - 1] : null,
            de = [0, 0],
            bp = [0, 0];
          for (let j = 0; j < 2; j++) {
            if (!t) {
              de[j] = 0.5 * b[j][x];
              bp[j] = -1;
            } else {
              const c0 = p[0] * a[0][j],
                c1 = p[1] * a[1][j];
              bp[j] = c0 >= c1 ? 0 : 1;
              de[j] = Math.max(c0, c1) * b[j][x];
            }
          }
          const s = de[0] + de[1];
          DE.push([de[0] / s, de[1] / s]);
          BP.push(bp);
          if (S.v === 'vit') {
            if (t)
              for (let j = 0; j < 2; j++)
                L.push(
                  'best story ending ' +
                    SN[j] +
                    ' today:  max( from sunny ' +
                    f3(p[0]) +
                    ' × ' +
                    f2(a[0][j]) +
                    ',  from rainy ' +
                    f3(p[1]) +
                    ' × ' +
                    f2(a[1][j]) +
                    ' ) × P(' +
                    ON[x] +
                    ' | ' +
                    SN[j] +
                    ') ' +
                    f2(b[j][x]) +
                    '   → came from <b>' +
                    SN[bp[j]] +
                    '</b>',
                );
            else
              L.push(
                'day 1 has no yesterday:  sunny 0.5 × ' +
                  f2(b[0][x]) +
                  ',  rainy 0.5 × ' +
                  f2(b[1][x]),
              );
            L.push('scores, rescaled:  sunny ' + f3(de[0] / s) + '   rainy ' + f3(de[1] / s));
            cu = 4;
            msg =
              'Day ' +
              d +
              ' (' +
              ON[x] +
              '): for each weather, the best story ending in it today was extended by one day and the link back was remembered.';
          }
        }
        t++;
        if (t >= TT) {
          fin = true;
          let j = DE[TT - 1][0] >= DE[TT - 1][1] ? 0 : 1;
          VP = new Array(TT);
          for (let i = TT - 1; i >= 0; i--) {
            VP[i] = j;
            j = BP[i][j];
          }
          const ok = acc();
          cu = S.v === 'vit' ? 5 : 3;
          if (S.v === 'vit')
            L.push(
              'Last day reached. The better ending is ' +
                SN[VP[TT - 1]] +
                '. Following the remembered links back gives the gold line.',
            );
          msg +=
            ' The fortnight is finished. ' +
            (S.v === 'vit'
              ? 'The single best story gets ' + ok[2] + ' of ' + TT + ' days right.'
              : 'Judging each day as it came, the model gets ' +
                ok[0] +
                ' of ' +
                TT +
                ' days right. With hindsight it gets ' +
                ok[1] +
                '.') +
            ' Press Show the True Weather to compare.';
        }
        eqShow(eq, L, 0);
      };
      const acc = () => {
        let f = 0,
          s = 0,
          v = 0;
        for (let i = 0; i < TT; i++) {
          if (AL[i] && (AL[i][1] > 0.5 ? 1 : 0) === Z[i]) f++;
          if ((SM[i][1] > 0.5 ? 1 : 0) === Z[i]) s++;
          if (VP.length && VP[i] === Z[i]) v++;
        }
        return [f, s, v];
      };
      const draw = () => {
        const tt = T();
        {
          const { g, w, h } = fit(c),
            m = 58,
            cw = (w - m - 10) / TT,
            yo = 34,
            ys = [h * 0.42, h * 0.66],
            R = Math.min(cw * 0.4, 26),
            fam = MONO.split('px ')[1];
          P = { m, cw, yo };
          g.font = MONO;
          g.textAlign = 'right';
          g.fillStyle = tt.muted;
          g.fillText('she did', m - 8, yo + 4);
          g.fillText('sunny', m - 8, ys[0] + 4);
          g.fillText('rainy', m - 8, ys[1] + 4);
          if (S.show) g.fillText('truth', m - 8, h * 0.86 + 4);
          for (let i = 0; i < TT; i++) {
            const cx = m + (i + 0.5) * cw;
            if (i === t - 1) {
              g.fillStyle = mix(tt.gold, tt.gold, 0, 0.18);
              g.fillRect(cx - cw / 2, 8, cw, h - 30);
            }
            g.textAlign = 'center';
            g.fillStyle = tt.ink;
            g.font = '600 ' + Math.max(9, Math.min(12, cw * 0.26)) + 'px ' + fam;
            g.fillText(ON[X[i]], cx, yo + 4);
            g.strokeStyle = tt.line;
            g.lineWidth = 1;
            g.strokeRect(cx - cw / 2 + 2, yo - 14, cw - 4, 26);
            g.font = MONO;
            g.fillStyle = tt.muted;
            g.fillText(String(i + 1), cx, h - 8);
            for (let j = 0; j < 2; j++) {
              const pr = i < t ? (S.v === 'fwd' ? AL[i][j] : DE[i][j]) : 0;
              g.beginPath();
              g.arc(cx, ys[j], R, 0, 7);
              g.strokeStyle = tt.muted;
              g.lineWidth = 1;
              g.stroke();
              if (i < t) {
                g.beginPath();
                g.arc(cx, ys[j], R * Math.sqrt(pr), 0, 7);
                g.fillStyle = j ? mix(tt.accent, tt.accent, 0, 0.85) : mix(tt.cut, tt.cut, 0, 0.85);
                g.fill();
                if (R > 15) {
                  g.fillStyle = tt.ink;
                  g.fillText((100 * pr).toFixed(0), cx, ys[j] + R + 12);
                }
              }
            }
            if (S.v === 'vit' && i < t && i > 0)
              for (let j = 0; j < 2; j++) {
                const cx0 = m + (i - 0.5) * cw;
                g.strokeStyle = mix(tt.ink, tt.ink, 0, 0.35);
                g.lineWidth = 1;
                g.beginPath();
                g.moveTo(cx0 + R, ys[BP[i][j]]);
                g.lineTo(cx - R, ys[j]);
                g.stroke();
              }
            if (S.show) {
              g.fillStyle = Z[i] ? tt.accent : tt.cut;
              g.fillRect(cx - cw / 2 + 3, h * 0.86 - 9, cw - 6, 18);
              g.fillStyle = tt.bg;
              g.fillText(Z[i] ? 'rain' : 'sun', cx, h * 0.86 + 4);
            }
          }
          const path = S.v === 'vit' ? (fin ? VP : []) : AL.map((p) => (p[1] > 0.5 ? 1 : 0));
          if (path.length) {
            g.strokeStyle = tt.gold;
            g.lineWidth = 3;
            g.lineJoin = 'round';
            g.beginPath();
            path.forEach((j, i) => {
              const cx = m + (i + 0.5) * cw;
              i ? g.lineTo(cx, ys[j]) : g.moveTo(cx, ys[j]);
            });
            g.stroke();
          }
          g.textAlign = 'left';
          g.fillStyle = tt.muted;
          g.fillText('day', 8, h - 8);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5,
            L = uLine(
              g,
              tt,
              0,
              0,
              w,
              h1,
              AL.map((p) => p[1]),
              {
                title: 'probability of rain on each day',
                n: TT,
                ym: 1,
                col: tt.accent,
                dots: true,
              },
            );
          if (fin) {
            g.strokeStyle = tt.ok;
            g.lineWidth = 2.2;
            g.beginPath();
            SM.forEach((p, i) => (i ? g.lineTo(L.X(i), L.Y(p[1])) : g.moveTo(L.X(i), L.Y(p[1]))));
            g.stroke();
          }
          if (S.show)
            Z.forEach((z, i) => {
              if (z) {
                g.fillStyle = mix(tt.accent, tt.accent, 0, 0.14);
                g.fillRect(
                  L.X(i) - (w - L.m.l - L.m.r) / (TT - 1) / 2,
                  L.Y(1),
                  (w - L.m.l - L.m.r) / (TT - 1),
                  L.Y(0) - L.Y(1),
                );
              }
            });
          g.font = MONO;
          g.textAlign = 'right';
          g.fillStyle = tt.ok;
          g.fillText(fin ? 'with hindsight' : '(hindsight: at the end)', w - 12, 13);
          g.fillStyle = tt.accent;
          g.fillText('on the day', w - 12 - (fin ? 104 : 160), 13);
          const a = A(),
            b = Bm(),
            y0 = h1 + 18,
            fam = MONO.split('px ')[1];
          g.textAlign = 'left';
          g.fillStyle = tt.muted;
          g.fillText('how the weather changes overnight', 8, y0);
          g.fillText('what she does in each weather', w / 2 + 4, y0);
          g.font = '12px ' + fam;
          g.fillStyle = tt.ink;
          [
            ['', '→ sunny', '→ rainy'],
            ['sunny', f2(a[0][0]), f2(a[0][1])],
            ['rainy', f2(a[1][0]), f2(a[1][1])],
          ].forEach((r, i) =>
            r.forEach((v, j) => g.fillText(v, 12 + j * ((w / 2 - 20) / 3), y0 + 22 + i * 20)),
          );
          [
            ['', 'walk', 'shop', 'clean'],
            ['sunny', ...b[0].map(f2)],
            ['rainy', ...b[1].map(f2)],
          ].forEach((r, i) =>
            r.forEach((v, j) =>
              g.fillText(v, w / 2 + 8 + j * ((w / 2 - 20) / 4), y0 + 22 + i * 20),
            ),
          );
        }
        live.textContent = msg;
        uCur(root, 'hm', cu);
        const ok = acc(),
          cur = t ? (S.v === 'fwd' ? AL[t - 1] : DE[t - 1]) : null;
        setR(root, 'd', t + ' of ' + TT);
        setR(root, 'b', cur ? pct(cur[0]) + ' / ' + pct(cur[1]) : '–');
        setR(
          root,
          'af',
          fin && S.show ? ok[0] + ' of ' + TT : fin ? 'press Show the True Weather' : '–',
        );
        setR(
          root,
          'as',
          fin && S.show ? ok[1] + ' of ' + TT : fin ? 'press Show the True Weather' : '–',
        );
        setR(
          root,
          'av',
          fin && S.show ? ok[2] + ' of ' + TT : fin ? 'press Show the True Weather' : '–',
        );
        setR(root, 'lk', t ? Math.exp(ll).toExponential(2) : '–');
      };
      gen();
      reset();
      if (RM) {
        while (!fin) step();
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (fr % 46) return;
        step();
        draw();
        if (fin) pb.stop();
      });
      pb = playBtn($('#hm-play', root), anim);
      runCtl(pb, root, 'hm', { done: () => fin, restart: () => reset() });
      $('#hm-step', root).onclick = () => {
        pb.stop();
        if (fin) reset();
        else step();
        draw();
      };
      $('#hm-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      $('#hm-new', root).onclick = () => {
        pb.stop();
        gen();
        reset();
        draw();
        pb.start();
      };
      $('#hm-tr', root).onclick = (e) => {
        S.show = !S.show;
        e.target.textContent = S.show ? 'Hide the True Weather' : 'Show the True Weather';
        draw();
      };
      segBind(root, 'hm-v', S.v, (v) => {
        S.v = v;
        pb.stop();
        reset();
        draw();
        pb.start();
      });
      bind(
        root,
        'hm-st',
        (v) => {
          S.st = +v;
          if (pb) {
            pb.stop();
            reset();
            draw();
          }
        },
        f2,
      );
      bind(
        root,
        'hm-em',
        (v) => {
          S.em = +v;
          if (pb) {
            pb.stop();
            reset();
            draw();
          }
        },
        f2,
      );
      gwClick(c, (px, py) => {
        if (!P || py > P.yo + 16 || py < P.yo - 18) return;
        const i = Math.floor((px - P.m) / P.cw);
        if (i < 0 || i >= TT) return;
        pb.stop();
        X[i] = (X[i] + 1) % 3;
        reset();
        msg =
          'Day ' +
          (i + 1) +
          ' was changed to “' +
          ON[X[i]] +
          '”. The true weather was not changed. Press Run to work through the fortnight again.';
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
  },

  {
    id: 'kalman',
    tab: 'Kalman filter',
    title: 'The Kalman filter: trust the forecast and the sensor, each as much as it deserves',
    lede: 'A delivery drone flies along a street. Its GPS reports a position every second, and every report is off by a few metres. Plotting the raw reports gives a jagged, jumping line. But the drone cannot jump: it has a speed, and where it will be in a second is predictable. The Kalman filter blends the two sources, the prediction from physics and the noisy reading, weighting each by how much it can be trusted at that moment. It is the reason your phone’s blue dot moves smoothly, and it flew to the Moon.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Position along the street over time. Thin line: where the drone really is. Dots: what the GPS reports. Bold line: the filter’s estimate, with a band two standard deviations wide</figcaption><canvas class="short" id="kf-c" role="img" aria-label="True position, noisy measurements and the filtered estimate with its uncertainty"></canvas></figure>
 <figure class="fig"><figcaption>Top: this second’s three bell curves over position: the prediction (dashed), the GPS reading, and their blend. Bottom: how much weight the filter gives a new reading, second by second</figcaption><canvas class="short" id="kf-r" role="img" aria-label="Prediction, measurement and posterior distributions, and the Kalman gain over time"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>This second’s calculation</figcaption><div class="eq" id="kf-eq" aria-live="polite" style="min-height:10em;margin-top:6px"></div></figure>
 ${uHow('kf', ['Keep two things: a best estimate of position and speed, and a measure of how unsure you are about each.', 'Predict: move the position forward by one second of travel at the estimated speed. Become a little less sure, because the speed may have changed.', 'A GPS reading arrives. Compare it with the prediction: the difference is the surprise.', 'Work out the gain: the share of the surprise to believe. It is large when the prediction is shaky and the sensor is good, and small the other way round.', 'Update: shift the estimate by gain × surprise, and become more sure.', 'Repeat every second. With no reading, only predict, and watch the uncertainty grow.'])}
 <div class="controls">${uBtns('kf', 'Run', 'One second')}
 ${grp('How much to trust each source', slider('kf-r2', 'GPS error, in metres', 0.5, 12, 0.5, 4) + slider('kf-q', 'How much the drone’s speed is assumed to change', 0.01, 2, 0.01, 0.15))}
 ${grp('Make trouble', `<div class="row"><button class="btn" id="kf-drop" type="button">Lose the Signal for 12 Seconds</button><button class="btn" id="kf-out" type="button">One Wild Reading</button><button class="btn" id="kf-new" type="button">A New Flight</button></div>`)}</div>
 <dl class="read">${rd('t', 'Second')}${rd('k', 'Gain: share of the surprise believed', 'hl2')}${rd('sg', 'Uncertainty in position, ± metres')}${rd('v', 'Estimated speed / true speed, m/s')}${rd('er', 'Typical error of the raw GPS', 'hl3')}${rd('ef', 'Typical error of the filter, on the seconds with a reading', 'hl')}</dl>`,
    tryit:
      'Press One second a few times and read the calculation: predict, compare, weigh, update. Notice the gain: it starts near 1, when the filter knows nothing, and settles to a steady value. Press Run and compare the two error readouts. Then press Lose the Signal: the estimate coasts on at the last known speed and the band widens. What happens to the gain on the first reading after the signal returns? Press One Wild Reading and see how far the estimate is pulled. Now raise the GPS error slider to 12 and start over: the filter leans on its prediction. Set the second slider to its maximum: the filter believes the drone may change speed wildly, and follows every reading.',
    math: String.raw`<h4>Intuition</h4><p>You are a passenger on a train at night, trying to guess where you are. You know the train's speed, so you can keep a running estimate: "ten minutes at this speed, so about twelve kilometres from the last station". Your estimate drifts, because the speed varies a little. Now and then a station sign flashes past, half-read. If your running estimate is fresh and the sign was a blur, you mostly trust the estimate. If you have been guessing for an hour and the sign was clear, you mostly trust the sign. Either way you end up more certain than either source alone would make you. The Kalman filter is this weighing done optimally, every second.</p>
 <h4>The model</h4><p>The hidden state is position and speed, \(\mathbf x_t=(p_t,v_t)\). It moves by simple physics plus a random disturbance, and the sensor reports the position plus noise:</p>$$\mathbf x_t=F\,\mathbf x_{t-1}+\mathbf w_t,\quad \mathbf w_t\sim\mathcal N(\mathbf 0,Q),\qquad y_t=H\,\mathbf x_t+e_t,\quad e_t\sim\mathcal N(0,R),$$$$F=\begin{pmatrix}1&\Delta t\\0&1\end{pmatrix},\qquad H=\begin{pmatrix}1&0\end{pmatrix}.$$<p>This has the same shape as a hidden Markov model: a hidden state that evolves, and observations that depend on it. The difference is that the state is continuous and everything is Gaussian. Because Gaussians stay Gaussian under these operations, the belief is fully described by a mean \(\hat{\mathbf x}\) and a covariance \(P\).</p>
 <h4>Predict</h4>$$\hat{\mathbf x}^-_t=F\,\hat{\mathbf x}_{t-1},\qquad P^-_t=F\,P_{t-1}F^{\top}+Q .$$<p>The estimate moves forward by one step of physics. The uncertainty grows: \(Q\) is how much the world may have changed that we could not foresee.</p>
 <h4>Update</h4><p>The reading \(y_t\) differs from what was predicted by the <b>innovation</b>, or surprise, \(y_t-H\hat{\mathbf x}^-_t\). The <b>Kalman gain</b> decides how much of it to believe:</p>$$K_t=P^-_tH^{\top}\big(HP^-_tH^{\top}+R\big)^{-1},\qquad \hat{\mathbf x}_t=\hat{\mathbf x}^-_t+K_t\,\big(y_t-H\hat{\mathbf x}^-_t\big),\qquad P_t=(I-K_tH)\,P^-_t .$$<p>For the position alone the gain is a single number,</p>$$K=\frac{\sigma^2_{\text{prediction}}}{\sigma^2_{\text{prediction}}+\sigma^2_{\text{sensor}}},$$<p>the share of the total uncertainty that belongs to the prediction. If the prediction is poor, \(K\) is near 1 and the estimate jumps to the reading. If the sensor is poor, \(K\) is near 0 and the reading is nearly ignored. The top-right picture shows the same thing as curves: multiplying the prediction bell by the reading bell gives a narrower bell between them, nearer to the narrower of the two. This is Bayes' rule with a Gaussian prior and a Gaussian likelihood (Kalman, 1960).</p>
 <h4>How it learns the speed without measuring it</h4><p>The GPS reports position only. Yet \(P\) records that errors in position and in speed go together: if the drone is further along than predicted, it was probably flying faster than estimated. The gain has a second component that corrects the speed in proportion to the position surprise. After a few seconds the filter knows the speed quite well.</p>
 <h4>When the signal drops</h4><p>With no reading there is nothing to update, so only the predict step runs. The estimate coasts in a straight line and \(P\) grows every second, which the widening band shows. When readings return, the prediction is very uncertain, so the gain is high and the estimate snaps back.</p>
 <h4>The two settings are beliefs, not facts</h4><p>\(R\) and \(Q\) say how much the filter trusts the sensor and the physics. Set \(R\) too low and it chases every jitter. Set \(Q\) too low and it is over-confident in its straight line and lags behind real turns. A single wild reading pulls the estimate by \(K\) times the error, because a Gaussian model has no notion of an outlier: practical systems reject readings whose surprise is implausibly large.</p>
 <h4>Beyond straight lines</h4><p>When the physics or the sensor is not linear, the extended and unscented Kalman filters apply the same two steps to a local approximation. When the belief is not bell-shaped at all, particle filters carry a cloud of samples in place of a mean and covariance.</p>
 <h4>What this page runs</h4><p>An 80-second flight with a speed that wanders slowly. The GPS error slider sets both the real noise in the readings and the filter's \(R\). The second slider sets the filter's \(Q\) only. The two "typical error" readouts are root-mean-square distances from the true position, over the seconds on which a reading arrived.</p>`,
    q: 'After the signal has been lost for 12 seconds, the first reading that arrives moves the estimate almost all the way to the reading. Ten seconds later, a reading that is just as far from the estimate moves it only a little. The sensor has not changed. What has?',
    a: 'The filter’s confidence in its own prediction. During the outage only the predict step ran, and every second added uncertainty, so by the time the signal returned the prediction was far less trustworthy than the sensor. The gain is the prediction’s share of the total uncertainty, so it was close to 1 and the reading was believed almost entirely. Each reading after that shrank the uncertainty again, the gain fell back to its steady value, and a surprise of the same size was mostly attributed to sensor noise.',
    init(root) {
      const c = $('#kf-c', root),
        cr = $('#kf-r', root),
        live = $('#kf-live', root),
        eq = $('#kf-eq', root),
        S = { r: 4, q: 0.15 },
        TT = 80,
        RN = randnFrom(Math.random);
      let TP = [],
        TV = [],
        Y = [],
        EP = [],
        SG = [],
        KG = [],
        xh,
        Pm,
        t = 0,
        drop = [-1, -1],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        fin = false,
        cur = null,
        seR = 0,
        nR = 0,
        seF = 0,
        outAt = -1;
      const gen = () => {
        TP = [];
        TV = [];
        let p = 0,
          v = 2.2;
        for (let i = 0; i <= TT; i++) {
          TP.push(p);
          TV.push(v);
          v = clamp(v + 0.22 * RN(), 0.2, 4.5);
          p += v;
        }
        Y = TP.map((q) => q + S.r * RN());
      };
      const reset = () => {
        xh = [Y[0], 0];
        Pm = [
          [S.r * S.r, 0],
          [0, 9],
        ];
        EP = [xh[0]];
        SG = [S.r];
        KG = [1];
        t = 0;
        fin = false;
        cu = 0;
        cur = null;
        seR = (Y[0] - TP[0]) ** 2;
        nR = 1;
        seF = seR;
        drop = [-1, -1];
        outAt = -1;
        msg =
          'Second 0: the filter starts at the first GPS reading and has no idea of the speed. Press One second or Run.';
        eqShow(
          eq,
          [
            'The first reading is taken as the starting estimate: position ' +
              f2(Y[0]) +
              ' m, speed unknown (taken as 0, with a large uncertainty).',
          ],
          0,
        );
      };
      const step = () => {
        if (fin) return;
        t++;
        const q = S.q,
          R = S.r * S.r,
          p0 = xh[0],
          v0 = xh[1],
          s0 = Math.sqrt(Pm[0][0]);
        const xp = [xh[0] + xh[1], xh[1]],
          a = Pm[0][0],
          b = Pm[0][1],
          d = Pm[1][1],
          Pp = [
            [a + 2 * b + d + q / 3, b + d + q / 2],
            [b + d + q / 2, d + q],
          ];
        const has = !(t >= drop[0] && t <= drop[1]);
        if (t === outAt) Y[t] = TP[t] + (Math.random() < 0.5 ? -1 : 1) * 7 * S.r;
        const L = [
          'Second ' + t + '.',
          '<b>predict</b>: position ' +
            f2(p0) +
            ' + 1 s × speed ' +
            f2(v0) +
            ' = ' +
            f2(xp[0]) +
            ' m      uncertainty ± ' +
            f2(s0) +
            ' grows to ± ' +
            f2(Math.sqrt(Pp[0][0])),
        ];
        if (has) {
          const y = Y[t],
            inn = y - xp[0],
            Sv = Pp[0][0] + R,
            K0 = Pp[0][0] / Sv,
            K1 = Pp[1][0] / Sv;
          xh = [xp[0] + K0 * inn, xp[1] + K1 * inn];
          Pm = [
            [(1 - K0) * Pp[0][0], (1 - K0) * Pp[0][1]],
            [Pp[1][0] - K1 * Pp[0][0], Pp[1][1] - K1 * Pp[0][1]],
          ];
          KG.push(K0);
          cur = { xp: xp[0], sp: Math.sqrt(Pp[0][0]), y, x: xh[0], s: Math.sqrt(Pm[0][0]) };
          seR += (y - TP[t]) ** 2;
          nR++;
          seF += (xh[0] - TP[t]) ** 2;
          L.push(
            '<b>reading</b>: GPS says ' +
              f2(y) +
              ' m (± ' +
              f2(S.r) +
              ').  surprise = ' +
              f2(y) +
              ' − ' +
              f2(xp[0]) +
              ' = ' +
              f2(inn) +
              ' m',
            '<b>gain</b> = prediction variance ÷ (prediction variance + GPS variance) = ' +
              f2(Pp[0][0]) +
              ' ÷ (' +
              f2(Pp[0][0]) +
              ' + ' +
              f2(R) +
              ') = <b>' +
              f3(K0) +
              '</b>',
            '<b>update</b>: position ' +
              f2(xp[0]) +
              ' + ' +
              f3(K0) +
              ' × ' +
              f2(inn) +
              ' = <b>' +
              f2(xh[0]) +
              ' m</b>      uncertainty ± ' +
              f2(Math.sqrt(Pm[0][0])),
            'speed corrected too: ' +
              f2(xp[1]) +
              ' + ' +
              f3(K1) +
              ' × ' +
              f2(inn) +
              ' = ' +
              f2(xh[1]) +
              ' m/s',
          );
          cu = 4;
          msg =
            'Second ' +
            t +
            ': the prediction was ' +
            f2(xp[0]) +
            ' m, the GPS said ' +
            f2(y) +
            ' m, and the filter believed ' +
            pct(K0) +
            ' of the difference. Estimate ' +
            f2(xh[0]) +
            ' m; the drone is really at ' +
            f2(TP[t]) +
            ' m.' +
            (t === outAt
              ? ' That reading was a wild one, ' + f2(Math.abs(y - TP[t])) + ' m off.'
              : '');
        } else {
          xh = xp;
          Pm = Pp;
          KG.push(NaN);
          cur = { xp: xp[0], sp: Math.sqrt(Pp[0][0]), y: NaN, x: xp[0], s: Math.sqrt(Pp[0][0]) };
          L.push(
            '<b>no reading</b>: the signal is lost. The prediction is all there is.',
            'estimate ' +
              f2(xh[0]) +
              ' m, uncertainty ± ' +
              f2(Math.sqrt(Pm[0][0])) +
              ' and growing',
          );
          cu = 5;
          msg =
            'Second ' +
            t +
            ': no GPS. The filter coasts at ' +
            f2(xh[1]) +
            ' m/s and its uncertainty has grown to ± ' +
            f2(Math.sqrt(Pm[0][0])) +
            ' m.';
        }
        EP.push(xh[0]);
        SG.push(Math.sqrt(Pm[0][0]));
        eqShow(eq, L, 0);
        if (t >= TT) {
          fin = true;
          msg +=
            ' The flight is over. Typical error of the raw GPS: ' +
            f2(Math.sqrt(seR / nR)) +
            ' m. Typical error of the filter on the same seconds: ' +
            f2(Math.sqrt(seF / nR)) +
            ' m.';
        }
      };
      const draw = () => {
        const tt = T();
        {
          const { g, w, h } = fit(c),
            m = { l: 44, r: 12, t: 14, b: 24 },
            lo = Math.min(...TP) - 3 * S.r,
            hi = Math.max(...TP) + 3 * S.r,
            X = (i) => m.l + (i / TT) * (w - m.l - m.r),
            Yv = (v) => h - m.b - ((v - lo) / (hi - lo)) * (h - m.t - m.b);
          g.font = MONO;
          g.strokeStyle = tt.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h - m.b);
          g.lineTo(w - m.r, h - m.b);
          g.moveTo(m.l, m.t);
          g.lineTo(m.l, h - m.b);
          g.stroke();
          if (drop[0] >= 0) {
            g.fillStyle = mix(tt.cut, tt.cut, 0, 0.13);
            g.fillRect(
              X(drop[0] - 0.5),
              m.t,
              X(Math.min(TT, drop[1] + 0.5)) - X(drop[0] - 0.5),
              h - m.t - m.b,
            );
            g.fillStyle = tt.cut;
            g.textAlign = 'center';
            g.fillText('no signal', (X(drop[0]) + X(Math.min(TT, drop[1]))) / 2, m.t + 12);
          }
          g.fillStyle = mix(tt.accent, tt.accent, 0, 0.22);
          g.beginPath();
          for (let i = 0; i <= t; i++)
            i ? g.lineTo(X(i), Yv(EP[i] + 2 * SG[i])) : g.moveTo(X(i), Yv(EP[i] + 2 * SG[i]));
          for (let i = t; i >= 0; i--) g.lineTo(X(i), Yv(EP[i] - 2 * SG[i]));
          g.closePath();
          g.fill();
          g.strokeStyle = tt.ink;
          g.lineWidth = 1.2;
          g.beginPath();
          for (let i = 0; i <= t; i++) i ? g.lineTo(X(i), Yv(TP[i])) : g.moveTo(X(i), Yv(TP[i]));
          g.stroke();
          g.fillStyle = mix(tt.ink, tt.ink, 0, 0.5);
          for (let i = 0; i <= t; i++) {
            if (i >= drop[0] && i <= drop[1]) continue;
            g.beginPath();
            g.arc(X(i), Yv(clamp(Y[i], lo, hi)), 2.6, 0, 7);
            g.fill();
          }
          g.strokeStyle = tt.accent;
          g.lineWidth = 2.6;
          g.beginPath();
          for (let i = 0; i <= t; i++) i ? g.lineTo(X(i), Yv(EP[i])) : g.moveTo(X(i), Yv(EP[i]));
          g.stroke();
          g.fillStyle = tt.gold;
          g.beginPath();
          g.arc(X(t), Yv(EP[t]), 5, 0, 7);
          g.fill();
          g.fillStyle = tt.muted;
          g.textAlign = 'right';
          g.fillText(lo.toFixed(0) + ' m', m.l - 4, h - m.b);
          g.fillText(hi.toFixed(0) + ' m', m.l - 4, m.t + 8);
          g.textAlign = 'center';
          for (const k of [0, 20, 40, 60, 80]) g.fillText(k + ' s', X(k), h - 7);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.52,
            m = { l: 14, r: 14, t: 26, b: 22 };
          g.font = MONO;
          g.fillStyle = tt.muted;
          g.textAlign = 'left';
          g.fillText('this second: prediction (dashed), reading, and their blend', 8, 13);
          if (cur) {
            const cs = [cur.xp, cur.x].concat(isFinite(cur.y) ? [cur.y] : []),
              sp = Math.max(cur.sp, S.r) * 3.2,
              mid = (Math.min(...cs) + Math.max(...cs)) / 2,
              lo = mid - sp - Math.abs(cs[0] - mid),
              hi = mid + sp + Math.abs(cs[0] - mid),
              X = (v) => m.l + ((v - lo) / (hi - lo)) * (w - m.l - m.r),
              pk = 1 / Math.min(cur.s, cur.sp, S.r),
              Yv = (v) => h1 - m.b - (v / pk) * (h1 - m.t - m.b),
              bell = (mu, sd, col, lw, dash, fill) => {
                g.beginPath();
                for (let i = 0; i <= 120; i++) {
                  const v = lo + ((hi - lo) * i) / 120,
                    d = Math.exp(-0.5 * ((v - mu) / sd) ** 2) / sd;
                  i ? g.lineTo(X(v), Yv(d)) : g.moveTo(X(v), Yv(d));
                }
                if (fill) {
                  g.lineTo(X(hi), Yv(0));
                  g.lineTo(X(lo), Yv(0));
                  g.closePath();
                  g.fillStyle = fill;
                  g.fill();
                }
                g.strokeStyle = col;
                g.lineWidth = lw;
                g.setLineDash(dash || []);
                g.stroke();
                g.setLineDash([]);
              };
            bell(cur.x, cur.s, tt.accent, 2.6, null, mix(tt.accent, tt.accent, 0, 0.25));
            bell(cur.xp, cur.sp, tt.muted, 1.8, [6, 4]);
            if (isFinite(cur.y)) bell(cur.y, S.r, tt.ok, 1.8);
            g.strokeStyle = tt.ink;
            g.lineWidth = 1.5;
            g.beginPath();
            g.moveTo(X(TP[t]), h1 - m.b);
            g.lineTo(X(TP[t]), h1 - m.b - 12);
            g.stroke();
            g.fillStyle = tt.muted;
            g.textAlign = 'center';
            g.fillText('true position', clamp(X(TP[t]), 50, w - 50), h1 - 6);
            g.textAlign = 'right';
            g.fillStyle = tt.muted;
            g.fillText('prediction', w - 14, m.t + 6);
            g.fillStyle = tt.ok;
            g.fillText(isFinite(cur.y) ? 'GPS reading' : 'no reading', w - 14, m.t + 20);
            g.fillStyle = tt.accent;
            g.fillText('blend', w - 14, m.t + 34);
          } else {
            g.textAlign = 'center';
            g.fillText('the curves appear at the first second', w / 2, h1 / 2);
          }
          const L = uLine(g, tt, 0, h1, w, h - h1, [], {
            title: 'gain: the share of each new reading that is believed',
            n: TT + 1,
            ym: 1,
          });
          g.strokeStyle = tt.cut;
          g.lineWidth = 2.2;
          g.beginPath();
          let pen = false;
          KG.forEach((v, i) => {
            if (!isFinite(v)) {
              pen = false;
              return;
            }
            pen ? g.lineTo(L.X(i), L.Y(v)) : g.moveTo(L.X(i), L.Y(v));
            pen = true;
          });
          g.stroke();
          g.fillStyle = tt.muted;
          g.textAlign = 'right';
          g.fillText('seconds →', w - 12, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'kf', cu);
        const kk = KG[KG.length - 1];
        setR(root, 't', t + ' of ' + TT);
        setR(root, 'k', isFinite(kk) ? f3(kk) : 'no reading');
        setR(root, 'sg', f2(SG[SG.length - 1]));
        setR(root, 'v', f2(xh[1]) + ' / ' + f2(TV[t]));
        setR(root, 'er', f2(Math.sqrt(seR / nR)) + ' m');
        setR(root, 'ef', t ? f2(Math.sqrt(seF / nR)) + ' m' : '–');
      };
      gen();
      reset();
      if (RM) {
        while (!fin) step();
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (fr % (t < 6 ? 40 : 9)) return;
        step();
        draw();
        if (fin) pb.stop();
      });
      pb = playBtn($('#kf-play', root), anim);
      runCtl(pb, root, 'kf', { done: () => fin, restart: () => reset() });
      $('#kf-step', root).onclick = () => {
        pb.stop();
        if (fin) reset();
        else step();
        draw();
      };
      $('#kf-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      $('#kf-new', root).onclick = () => {
        pb.stop();
        gen();
        reset();
        draw();
        pb.start();
      };
      $('#kf-drop', root).onclick = () => {
        if (fin || t > TT - 4) {
          msg = 'The flight is over. Press Start over, then lose the signal while it is running.';
          draw();
          return;
        }
        drop = [t + 1, t + 12];
        msg =
          'The GPS signal will be missing from second ' +
          (t + 1) +
          ' to second ' +
          Math.min(TT, t + 12) +
          '. Watch the band.';
        draw();
      };
      $('#kf-out', root).onclick = () => {
        if (fin) {
          msg = 'The flight is over. Press Start over first.';
          draw();
          return;
        }
        outAt = t + 1;
        msg = 'The next reading will be wildly wrong.';
        draw();
      };
      bind(
        root,
        'kf-r2',
        (v) => {
          S.r = +v;
          if (pb) {
            pb.stop();
            gen();
            reset();
            draw();
          }
        },
        f2,
      );
      bind(
        root,
        'kf-q',
        (v) => {
          S.q = +v;
          if (pb) {
            pb.stop();
            reset();
            draw();
          }
        },
        f2,
      );
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
  },
  {
    id: 'recommender',
    tab: 'Recommender systems',
    title: 'Recommender systems: fill in the ratings nobody gave',
    lede: 'Ten people have rated some of ten films, each from 1 to 5. Most of the table is empty, because nobody has seen everything. A recommender has to guess the empty cells: would Ben like a film he has never seen? The idea that won a famous million-dollar prize is to describe every person and every film by a few hidden numbers, such as “how much action” and “how much romance”, learned only from the ratings that exist. Add your own ratings in the bottom row and get recommendations.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The ratings table. Bold numbers are real ratings. Faint numbers are the model’s guesses for empty cells. The bottom row is yours: click its cells to rate films from 1 to 5</figcaption><canvas class="short" id="rc-c" role="img" aria-label="Ratings matrix with observed ratings and predictions"></canvas></figure>
 <figure class="fig"><figcaption>Top: every person (circle) and film (square) placed by their first two hidden numbers. People sit near the films they like. Bottom: the model’s error on ratings it trains on, and on ratings hidden from it</figcaption><canvas class="short" id="rc-r" role="img" aria-label="Latent factor map and training and held-out error"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>One update, worked out</figcaption><div class="eq" id="rc-eq" aria-live="polite" style="min-height:9.5em;margin-top:6px"></div></figure>
 ${uHow('rc', ['Give every person and every film K hidden numbers, small and random at first. Set a few real ratings aside to test on later.', 'Predict a rating: the overall average, plus how generous this person is, plus how well liked this film is, plus the match between the person’s numbers and the film’s.', 'Take one real rating. Compare it with the prediction: the difference is the error.', 'Nudge the person’s numbers toward the film’s, and the film’s toward the person’s, in proportion to the error. Shrink everything slightly so that nothing grows without need.', 'Do this for every real rating: one pass. Repeat passes until the error on the hidden ratings stops falling.', 'Fill every empty cell with its prediction, and recommend each person their highest unseen films.'])}
 <div class="controls">${uBtns('rc', 'Run', 'One pass', 'passes')}
 ${grp('The model', slider('rc-k', 'Hidden numbers per person and per film, K', 1, 4, 1, 2) + slider('rc-l', 'Shrinkage λ: resistance to large numbers', 0, 0.3, 0.01, 0.15))}
 ${grp('The data', `<div class="row"><button class="btn" id="rc-new" type="button">New Ratings Table</button><button class="btn" id="rc-clr" type="button">Clear My Ratings</button></div><p class="tpout" id="rc-tpo" role="status"></p>`)}</div>
 <dl class="read">${rd('ep', 'Passes so far')}${rd('tr', 'Error on the ratings it trains on', 'hl2')}${rd('te', 'Error on the hidden ratings', 'hl')}${rd('bl', 'Error of always guessing the film’s average')}${rd('nr', 'Real ratings: used for training / hidden for testing')}${rd('np', 'Numbers the model learns')}</dl>`,
    tryit:
      'Press Run and watch the faint numbers settle. In the top-right map, the films separate into groups without anyone telling the model their genres. Which films are close together? Which people sit near which films? Now rate four or five films in the bottom row, giving the action films 5 and the romances 1, or the other way round, and press Run again. Where do you land on the map, and what does it recommend? Then set K to 4 and λ to 0 and start over. Watch the two error curves: the error on the training ratings falls almost to zero. Does the error on the hidden ratings follow it down? With a table this small the answer varies from table to table, so press New Ratings Table a few times before you decide which settings are best.',
    math: String.raw`<h4>Intuition</h4><p>You ask a well-read friend for a book. She does not look up statistics. She has a rough picture of your taste, "likes slow novels, dislikes violence", and a rough picture of each book along the same lines, and she looks for a match. Nobody ever filled in a questionnaire to produce those pictures. She formed them from the handful of books she knows you loved or hated. Matrix factorisation gives a computer the same two sets of pictures, as short lists of numbers, and learns both at once from the ratings alone.</p>
 <h4>The table and the model</h4><p>Let \(r_{ui}\) be the rating person \(u\) gave film \(i\), known only for some pairs. Each person gets a vector \(\mathbf p_u\in\mathbb R^K\) and each film a vector \(\mathbf q_i\in\mathbb R^K\). The predicted rating is</p>$$\hat r_{ui}=\mu+b_u+b_i+\mathbf p_u^{\top}\mathbf q_i .$$<p>\(\mu\) is the average of all ratings, \(b_u\) says whether this person rates high or low in general, \(b_i\) whether the film is generally liked, and the dot product is large when the person's tastes and the film's qualities point the same way. Stacking the vectors gives two thin matrices whose product approximates the whole table, which is where the name comes from.</p>
 <h4>What is minimised</h4><p>Only the known ratings enter the loss, with a penalty that keeps the numbers small:</p>$$\min_{b,\mathbf p,\mathbf q}\ \sum_{(u,i)\ \text{known}}\big(r_{ui}-\hat r_{ui}\big)^2+\lambda\big(\lVert\mathbf p_u\rVert^2+\lVert\mathbf q_i\rVert^2+b_u^2+b_i^2\big).$$<p>The empty cells are not treated as zeros. They are simply left out, and that is what distinguishes this from ordinary matrix decompositions such as PCA.</p>
 <h4>One update</h4><p>Stochastic gradient descent visits the known ratings one at a time. With error \(e_{ui}=r_{ui}-\hat r_{ui}\) and step size \(\eta\),</p>$$\mathbf p_u\leftarrow\mathbf p_u+\eta\,(e_{ui}\,\mathbf q_i-\lambda\,\mathbf p_u),\qquad \mathbf q_i\leftarrow\mathbf q_i+\eta\,(e_{ui}\,\mathbf p_u-\lambda\,\mathbf q_i),$$<p>and similarly for the two biases. If the person liked the film more than predicted, their vector moves toward the film's and the film's toward theirs, so the next prediction is higher. This is the approach Simon Funk published during the Netflix Prize, later refined by Koren, Bell and Volinsky (2009).</p>
 <h4>Why it can guess a cell it has never seen</h4><p>Ana and Ben have both rated several films, and where they overlap they agree. Their vectors therefore end up close together. Ana has also rated a film that Ben has not. The film's vector was shaped by Ana's rating, and since Ben's vector resembles Ana's, the dot product gives Ben a similar prediction. Information flows from person to person through the films they share, and from film to film through the people who rated both. This is called <b>collaborative filtering</b>: nothing about the content of the films is used.</p>
 <h4>Choosing K and λ</h4><p>More hidden numbers can describe taste more finely, and can also memorise the training ratings, noise included. The honest test is the error on ratings the model never saw. With a table this small, a large \(K\) with no shrinkage drives the training error toward zero while the hidden-rating error stays far above it: the model has memorised its training ratings, which is overfitting, exactly as in the Foundations module. \(\lambda\) pulls the vectors toward zero and guards against it. In repeated tests on this page a larger \(\lambda\) gave the lower hidden-rating error more often than not, and with only about twenty hidden ratings any single table can go either way.</p>
 <h4>What it cannot do</h4><p>A new person or a new film with no ratings has no vector worth the name: the <b>cold start</b> problem. Real systems fall back on content, such as genre or description, or on popularity, until ratings arrive. Most real feedback is also not a rating at all but a click, a purchase or a watch, where "no data" may mean "did not like" or "never saw". And a recommender that only reinforces what people already chose narrows what they see.</p>
 <h4>What this page runs</h4><p>Ten people, ten invented films of three kinds, and a table with roughly 75% of the ratings present, generated from hidden tastes plus noise. About one rating in four is set aside for testing. Training runs for at least 80 passes and stops when the error on those has not improved for 40 passes, or after 400. Your own ratings are visited four times in every pass, so that a handful of them is enough to place you on the map. The error shown is the root-mean-square difference in rating points.</p>`,
    q: 'A new film is added to the table with no ratings at all. What rating does the model predict for it, for every person, and why is that the best it can do?',
    a: 'It predicts each person’s usual level: the overall average plus that person’s own bias. The film’s bias and its hidden numbers have never been updated, because updates happen only when a rating for that film is visited, and with shrinkage they stay at or near zero. Nothing in the ratings table says anything about this film, so nothing better is available from ratings alone. To do better the system needs information from outside the table, such as the film’s genre, cast or description.',
    init(root) {
      const c = $('#rc-c', root),
        cr = $('#rc-r', root),
        live = $('#rc-live', root),
        eq = $('#rc-eq', root),
        S = { k: 2, l: 0.15 },
        US = ['Ana', 'Ben', 'Chloe', 'Dev', 'Emma', 'Farid', 'Grace', 'Hugo', 'Ines', 'Jun', 'You'],
        FM = [
          'Star Raiders',
          'Steel Horizon',
          'Night Chase',
          'Robot Dawn',
          'Letters from Paris',
          'Summer in Tuscany',
          'The Wedding Guest',
          'Autumn Hearts',
          'Spy in Love',
          'Laugh Track',
        ],
        FT = [
          [1.1, -0.6],
          [1, -0.4],
          [0.9, -0.2],
          [1.2, -0.7],
          [-0.7, 1.1],
          [-0.5, 1],
          [-0.6, 0.9],
          [-0.8, 1.2],
          [0.5, 0.6],
          [0.1, 0.2],
        ],
        NU = 11,
        NI = 10,
        ME = 10,
        MAXE = 400,
        ETA = 0.05;
      let R = [],
        TE = [],
        MY = new Array(NI).fill(0),
        mu = 3,
        bu,
        bi,
        Pu,
        Qi,
        ep = 0,
        TR = [],
        TS = [],
        best = 9,
        bestAt = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        seed = 1,
        PC = null,
        base = NaN;
      const gen = () => {
        const r = rng(seed * 13 + 5),
          rn = randnFrom(r);
        R = [];
        TE = [];
        for (let u = 0; u < ME; u++) {
          const a = u % 2 ? [0.2 + 1.1 * r(), 1.1 - 1.3 * r()] : [1.1 - 1.3 * r(), 0.2 + 1.1 * r()],
            gb = 0.5 * rn();
          for (let i = 0; i < NI; i++) {
            if (r() > 0.75) continue;
            const v = clamp(
              Math.round(3 + gb + 1.5 * (a[0] * FT[i][0] + a[1] * FT[i][1]) + 0.45 * rn()),
              1,
              5,
            );
            (r() < 0.25 ? TE : R).push([u, i, v]);
          }
        }
      };
      const all = () => R.concat(MY.map((v, i) => (v ? [ME, i, v] : null)).filter((x) => x));
      const pred = (u, i) => {
        let s = mu + bu[u] + bi[i];
        for (let k = 0; k < S.k; k++) s += Pu[u * 4 + k] * Qi[i * 4 + k];
        return s;
      };
      const rmse = (L) =>
        L.length
          ? Math.sqrt(L.reduce((s, [u, i, v]) => s + (v - pred(u, i)) ** 2, 0) / L.length)
          : NaN;
      const build = () => {
        const rn = randnFrom(rng(seed * 7 + S.k)),
          A = all();
        mu = A.reduce((s, q) => s + q[2], 0) / A.length;
        bu = new Float64Array(NU);
        bi = new Float64Array(NI);
        Pu = Float64Array.from({ length: NU * 4 }, () => 0.1 * rn());
        Qi = Float64Array.from({ length: NI * 4 }, () => 0.1 * rn());
        ep = 0;
        TR = [rmse(A)];
        TS = [rmse(TE)];
        best = 9;
        bestAt = 0;
        fin = false;
        cu = 0;
        const sm = new Float64Array(NI),
          ct = new Float64Array(NI);
        for (const [u, i, v] of A) {
          sm[i] += v;
          ct[i]++;
        }
        base = Math.sqrt(
          TE.reduce((s, [u, i, v]) => s + (v - (ct[i] ? sm[i] / ct[i] : mu)) ** 2, 0) /
            Math.max(1, TE.length),
        );
        msg =
          'The table holds ' +
          A.length +
          ' real ratings for training and ' +
          TE.length +
          ' set aside. Every person and film has ' +
          S.k +
          ' hidden number' +
          (S.k > 1 ? 's' : '') +
          ', random for now, so every guess is close to the overall average of ' +
          f2(mu) +
          '. Press Run.';
        eqShow(eq, ['No updates yet.'], 0);
      };
      const pass = () => {
        if (fin) return;
        const A0 = all(),
          A = A0.concat(
            A0.filter((q) => q[0] === ME),
            A0.filter((q) => q[0] === ME),
            A0.filter((q) => q[0] === ME),
          )
            .map((q) => [Math.random(), q])
            .sort((a, b) => a[0] - b[0])
            .map((q) => q[1]);
        let ex = null;
        A.forEach(([u, i, v], n) => {
          const p0 = pred(u, i),
            e = v - p0;
          if (n === 0) ex = { u, i, v, p0, e, pu: Pu[u * 4], qi: Qi[i * 4] };
          bu[u] += ETA * (e - S.l * bu[u]);
          bi[i] += ETA * (e - S.l * bi[i]);
          for (let k = 0; k < S.k; k++) {
            const a = Pu[u * 4 + k],
              b = Qi[i * 4 + k];
            Pu[u * 4 + k] += ETA * (e * b - S.l * a);
            Qi[i * 4 + k] += ETA * (e * a - S.l * b);
          }
          if (n === 0) {
            ex.pu2 = Pu[u * 4];
            ex.qi2 = Qi[i * 4];
            ex.p1 = pred(u, i);
          }
        });
        ep++;
        const tr = rmse(A0),
          te = rmse(TE);
        TR.push(tr);
        TS.push(te);
        if (te < best - 1e-4) {
          best = te;
          bestAt = ep;
        }
        if (ex)
          eqShow(
            eq,
            [
              'Pass ' +
                ep +
                ', first rating visited: <b>' +
                US[ex.u] +
                '</b> gave <b>' +
                FM[ex.i] +
                '</b> a ' +
                ex.v +
                '.',
              'the model predicted ' +
                f2(ex.p0) +
                ', so the error is ' +
                ex.v +
                ' − ' +
                f2(ex.p0) +
                ' = <b>' +
                f2(ex.e) +
                '</b>',
              US[ex.u] +
                '’s first hidden number: ' +
                f3(ex.pu) +
                ' + ' +
                ETA +
                ' × (' +
                f2(ex.e) +
                ' × the film’s ' +
                f3(ex.qi) +
                ' − ' +
                f2(S.l) +
                ' × ' +
                f3(ex.pu) +
                ') = <b>' +
                f3(ex.pu2) +
                '</b>',
              'the film’s first hidden number: ' +
                f3(ex.qi) +
                ' → <b>' +
                f3(ex.qi2) +
                '</b>, moved in the same way toward ' +
                US[ex.u] +
                '’s',
              'the prediction for this cell is now ' +
                f2(ex.p1) +
                (Math.abs(ex.v - ex.p1) < Math.abs(ex.e) ? ', closer to ' + ex.v : ''),
            ],
            0,
          );
        cu = 4;
        msg =
          'Pass ' +
          ep +
          ': all ' +
          A0.length +
          ' training ratings were visited. Error on them: ' +
          f3(tr) +
          ' rating points. Error on the ' +
          TE.length +
          ' hidden ratings: ' +
          f3(te) +
          '.';
        if ((ep >= 80 && ep - bestAt >= 40) || ep >= MAXE) {
          fin = true;
          cu = 5;
          msg =
            (ep >= MAXE
              ? 'Stopped at the limit of ' + MAXE + ' passes.'
              : 'Stopped after ' +
                ep +
                ' passes: the error on the hidden ratings has not improved for 40 passes. Its best value was ' +
                f3(best) +
                ' at pass ' +
                bestAt +
                '.') +
            ' Final error: ' +
            f3(tr) +
            ' on training ratings, ' +
            f3(te) +
            ' on hidden ones. Always guessing each film’s average would score ' +
            f3(base) +
            '.';
        }
      };
      const recs = () => {
        const el = $('#rc-tpo', root),
          n = MY.filter((v) => v).length;
        if (!n) {
          el.textContent =
            'Click cells in the bottom row of the table to rate films yourself. Each click raises the rating by one, and a sixth click clears it.';
          return;
        }
        const o = [...Array(NI).keys()]
          .filter((i) => !MY[i])
          .map((i) => [pred(ME, i), i])
          .sort((a, b) => b[0] - a[0]);
        el.textContent =
          'You have rated ' +
          n +
          ' film' +
          (n > 1 ? 's' : '') +
          '. ' +
          (ep < 5
            ? 'Press Run to train with your ratings included.'
            : o.length
              ? 'For you, the model’s highest guesses among the films you have not rated: ' +
                o
                  .slice(0, 3)
                  .map((q) => FM[q[1]] + ' (' + f2(clamp(q[0], 1, 5)) + ')')
                  .join(', ') +
                '. Its lowest: ' +
                FM[o[o.length - 1][1]] +
                ' (' +
                f2(clamp(o[o.length - 1][0], 1, 5)) +
                ').'
              : 'You have rated everything.');
      };
      const draw = () => {
        const t = T(),
          A = all(),
          known = new Map(A.map((q) => [q[0] * NI + q[1], q[2]])),
          held = new Map(TE.map((q) => [q[0] * NI + q[1], q[2]]));
        {
          const { g, w, h } = fit(c),
            lx = 54,
            ty = 122,
            cw = (w - lx - 6) / NI,
            rh = (h - ty - 6) / NU,
            fam = MONO.split('px ')[1];
          PC = { lx, ty, cw, rh };
          g.font = MONO;
          for (let i = 0; i < NI; i++) {
            g.save();
            g.translate(lx + (i + 0.5) * cw + 3, ty - 6);
            g.rotate(-Math.PI / 2.5);
            g.fillStyle = i < 4 ? t.cut : i < 8 ? t.accent : t.muted;
            g.textAlign = 'left';
            g.fillText(FM[i], 0, 0);
            g.restore();
          }
          for (let u = 0; u < NU; u++) {
            g.textAlign = 'right';
            g.fillStyle = u === ME ? t.cut : t.ink;
            g.font = (u === ME ? '600 ' : '') + '12px ' + fam;
            g.fillText(US[u], lx - 8, ty + (u + 0.5) * rh + 4);
            for (let i = 0; i < NI; i++) {
              const x = lx + i * cw,
                y = ty + u * rh,
                kv = known.get(u * NI + i),
                hv = held.get(u * NI + i),
                p = clamp(pred(u, i), 1, 5),
                v = kv || p,
                col = v >= 3 ? t.accent : t.cut;
              g.fillStyle = mix(
                col,
                col,
                0,
                (kv ? 0.2 : 0.07) + (Math.abs(v - 3) / 2) * (kv ? 0.6 : 0.3),
              );
              g.fillRect(x + 1, y + 1, cw - 2, rh - 2);
              g.textAlign = 'center';
              if (kv) {
                g.fillStyle = t.ink;
                g.font = '700 ' + Math.min(15, rh * 0.5) + 'px ' + fam;
                g.fillText(String(kv), x + cw / 2, y + rh / 2 + 5);
              } else {
                g.fillStyle = t.muted;
                g.font = Math.min(11, rh * 0.38) + 'px ' + fam;
                g.fillText(ep ? p.toFixed(1) : '·', x + cw / 2, y + rh / 2 + 4);
                if (hv) {
                  g.strokeStyle = t.gold;
                  g.lineWidth = 2;
                  g.strokeRect(x + 2.5, y + 2.5, cw - 5, rh - 5);
                  g.fillStyle = t.cut;
                  g.font = '9px ' + fam;
                  g.fillText('was ' + hv, x + cw / 2, y + rh - 4);
                }
              }
            }
            if (u === ME) {
              g.strokeStyle = t.cut;
              g.lineWidth = 1.5;
              g.setLineDash([4, 3]);
              g.strokeRect(lx + 0.5, ty + u * rh + 0.5, NI * cw - 1, rh - 1);
              g.setLineDash([]);
            }
          }
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.56,
            pts = [];
          for (let u = 0; u < NU; u++)
            if (u < ME || MY.some((v) => v))
              pts.push([Pu[u * 4], S.k > 1 ? Pu[u * 4 + 1] : 0, US[u], 0, u]);
          for (let i = 0; i < NI; i++)
            pts.push([Qi[i * 4], S.k > 1 ? Qi[i * 4 + 1] : 0, FM[i], 1, i]);
          const mx = Math.max(0.2, ...pts.map((p) => Math.abs(p[0]))) * 1.25,
            my = Math.max(0.2, ...pts.map((p) => Math.abs(p[1]))) * 1.25,
            X = (v) => w / 2 + (v / mx) * (w / 2 - 70),
            Y = (v) => 18 + (h1 - 30) / 2 - (v / my) * ((h1 - 40) / 2);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            'hidden number 1 (across) and 2 (up): circles are people, squares are films',
            8,
            13,
          );
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(X(-mx), Y(0));
          g.lineTo(X(mx), Y(0));
          g.moveTo(X(0), Y(-my));
          g.lineTo(X(0), Y(my));
          g.stroke();
          for (const p of pts) {
            const x = X(p[0]),
              y = Y(p[1]);
            if (p[3]) {
              g.fillStyle = p[4] < 4 ? t.cut : p[4] < 8 ? t.accent : t.muted;
              g.fillRect(x - 4, y - 4, 8, 8);
            } else {
              g.beginPath();
              g.arc(x, y, 4.5, 0, 7);
              g.fillStyle = p[4] === ME ? t.gold : t.bg;
              g.fill();
              g.strokeStyle = t.ink;
              g.lineWidth = 1.6;
              g.stroke();
            }
            g.fillStyle = p[3] ? t.muted : t.ink;
            g.textAlign = 'left';
            g.fillText(p[2], x + 7, y + 4);
          }
          const n = Math.max(60, TR.length),
            ym = Math.max(1.2, TR[0], TS[0] || 0),
            L = uLine(g, t, 0, h1, w, h - h1, TR, {
              title: 'error in rating points, by pass',
              n,
              ym,
              col: t.accent,
            });
          g.strokeStyle = t.cut;
          g.lineWidth = 2.2;
          g.beginPath();
          TS.forEach((v, i) =>
            i ? g.lineTo(L.X(i), L.Y(Math.min(v, ym))) : g.moveTo(L.X(i), L.Y(Math.min(v, ym))),
          );
          g.stroke();
          g.textAlign = 'right';
          g.fillStyle = t.accent;
          g.fillText('ratings it trains on', w - 14, h1 + 38);
          g.fillStyle = t.cut;
          g.fillText('ratings hidden from it', w - 14, h1 + 52);
        }
        live.textContent = msg;
        uCur(root, 'rc', cu);
        recs();
        setR(root, 'ep', ep + (fin ? ' (stopped)' : ''));
        setR(root, 'tr', f3(TR[TR.length - 1]));
        setR(root, 'te', f3(TS[TS.length - 1]));
        setR(root, 'bl', f3(base));
        setR(root, 'nr', A.length + ' / ' + TE.length);
        setR(root, 'np', String((NU + NI) * (S.k + 1)));
      };
      gen();
      build();
      if (RM) {
        while (!fin) pass();
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (fr % (ep < 6 ? 24 : 3)) return;
        pass();
        draw();
        chk();
      });
      pb = playBtn($('#rc-play', root), anim);
      chk = runCtl(pb, root, 'rc', {
        done: () => fin,
        restart: () => build(),
        count: () => ep,
        what: 'passes',
      });
      $('#rc-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else pass();
        draw();
      };
      $('#rc-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      $('#rc-new', root).onclick = () => {
        pb.stop();
        seed++;
        gen();
        build();
        draw();
        pb.start();
      };
      $('#rc-clr', root).onclick = () => {
        pb.stop();
        MY.fill(0);
        build();
        draw();
      };
      bind(
        root,
        'rc-k',
        (v) => {
          S.k = +v;
          if (pb) {
            pb.stop();
            build();
            draw();
            pb.start();
          }
        },
        (v) => String(v),
      );
      bind(
        root,
        'rc-l',
        (v) => {
          S.l = +v;
          if (pb) {
            pb.stop();
            build();
            draw();
            pb.start();
          }
        },
        f2,
      );
      gwClick(c, (px, py) => {
        if (!PC) return;
        const i = Math.floor((px - PC.lx) / PC.cw),
          u = Math.floor((py - PC.ty) / PC.rh);
        if (u !== ME || i < 0 || i >= NI) return;
        pb.stop();
        MY[i] = (MY[i] + 1) % 6;
        build();
        msg =
          'You ' +
          (MY[i] ? 'gave “' + FM[i] + '” a ' + MY[i] : 'cleared your rating of “' + FM[i] + '”') +
          '. The model starts again with your ratings included. Press Run.';
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
    id: 'time-series',
    tab: 'Time series',
    title: 'Time series: forecast tomorrow from the shape of yesterday',
    lede: 'An ice-cream shop has eight years of monthly sales. Sales climb a little every year and swing up every summer. Next month’s stock has to be ordered now. Forecasting from a series like this differs from everything else in this course in one way: the order of the data is the information. Four forecasters, from the laziest possible to a proper statistical model, are tested the only honest way, by walking through history and predicting each month before it is revealed.',
    stage: `<div class="row seg" id="ts-m" role="group" aria-label="Which forecaster to watch"><button class="btn" type="button" data-v="0">Last Value</button><button class="btn" type="button" data-v="1">Same Season Last Cycle</button><button class="btn" type="button" data-v="2">Holt-Winters Smoothing</button><button class="btn" type="button" data-v="3">Autoregression</button></div>
 <div class="stage"><figure class="fig"><figcaption>The series. Dark: what has been revealed so far. Faint: still hidden. Gold dots: the watched forecaster’s predictions, each made before that point was revealed. After the walk: its forecast for the future, with a rough range</figcaption><canvas class="short" id="ts-c" role="img" aria-label="Time series, one-step forecasts and the final forecast"></canvas></figure>
 <figure class="fig"><figcaption>Top: how strongly each value resembles the value a given number of steps earlier. Bottom: the average miss of each forecaster so far</figcaption><canvas class="short" id="ts-r" role="img" aria-label="Autocorrelation by lag and running error of four forecasting methods"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The watched forecaster’s calculation for this step</figcaption><div class="eq" id="ts-eq" aria-live="polite" style="min-height:8.5em;margin-top:6px"></div></figure>
 ${uHow('ts', ['Hide everything after the first stretch of history.', 'Each forecaster predicts the next value, using only what has been revealed.', 'Reveal the true value. Record how far off each forecast was.', 'Let each forecaster learn from the new value, then move one step forward.', 'Repeat to the end of the data. The forecaster with the smallest average miss has earned the most trust.', 'Now forecast the real future, as far ahead as needed. The further ahead, the wider the range of what could happen.'])}
 <div class="controls">${uBtns('ts', 'Run', 'One step')}
 ${grp('The forecasters', slider('ts-s', 'Length of one cycle (12 for monthly data with a yearly pattern)', 2, 24, 1, 12) + slider('ts-a', 'Holt-Winters: weight on the newest value, α', 0.05, 1, 0.05, 0.3) + slider('ts-p', 'Autoregression: how many past values it uses, p', 1, 24, 1, 13))}
 ${grp('The data', `<label class="ctl" for="ts-d"><span>Series</span><select id="ts-d"><option value="ice">Ice-cream sales: rising, with a yearly swing</option><option value="temp">Temperature: a yearly swing only</option><option value="walk">A share price: a random walk</option><option value="traffic">Website visits: a weekly rhythm (cycle of 7)</option></select></label><div class="row"><label class="btn file">Upload a CSV<input type="file" id="ts-file" accept=".csv,.txt,text/csv"></label></div><p class="status" id="ts-st" role="status"></p>`)}</div>
 <dl class="read">${rd('t', 'Points revealed')}${rd('f', 'Forecast for the next point', 'hl')}${rd('e0', 'Average miss: Last Value')}${rd('e1', 'Average miss: Same Season Last Cycle')}${rd('e2', 'Average miss: Holt-Winters', 'hl2')}${rd('e3', 'Average miss: Autoregression', 'hl3')}</dl>`,
    tryit:
      'Press Run on the ice-cream data and watch the four error lines. Which forecaster is worst, and why is Last Value so bad on data with seasons? Look at the top-right bars: at which gap do values most resemble each other? Switch the watched forecaster between the four during the walk. Now choose the share price and run again. Which forecaster wins? What does that tell you about predicting share prices from their own past? Choose the website visits and set the cycle length to 12, the wrong value: see what the seasonal methods do. Set it to 7. Finally type a number of steps ahead in the box and read the forecast and its range.',
    math: String.raw`<h4>Intuition</h4><p>A sailor forecasts the tide without any physics. Today's tide will be much like the one twelve and a half hours ago, a little like yesterday's at the same hour, and shaped by where we are in the month. A good forecast leans on the regular patterns in the past: the slow drift, the repeating cycle, and the fact that things do not usually jump. What is left over after those is noise, and no method can forecast noise. The skill is in telling which is which.</p>
 <h4>What a series is made of</h4><p>A useful way to think of a series \(y_t\) is as a sum of three parts:</p>$$y_t=\underbrace{T_t}_{\text{trend}}+\underbrace{S_t}_{\text{season}}+\underbrace{\varepsilon_t}_{\text{noise}},$$<p>where the season repeats with period \(m\): \(S_t=S_{t-m}\). The <b>autocorrelation</b> at lag \(k\) measures how alike \(y_t\) and \(y_{t-k}\) are, on a scale from \(-1\) to \(1\). The bars at the top right show it for every lag. A peak at lag 12 says the series repeats every 12 steps. Bars that fade slowly from lag 1 say the series drifts.</p>
 <h4>Two benchmarks</h4><p><b>Last value</b>: \(\hat y_{t+1}=y_t\). <b>Seasonal naive</b>: \(\hat y_{t+1}=y_{t+1-m}\), the value one cycle ago. Neither has anything to learn. Any serious method must beat both, and surprisingly often it does not. For a random walk, where each value is the last one plus pure noise, the last value is the best possible forecast.</p>
 <h4>Holt-Winters exponential smoothing</h4><p>Keep three running estimates: a level \(\ell\), a slope \(b\) and one seasonal offset \(s\) for each position in the cycle. Each new value updates them by a weighted average of "what the new value says" and "what I thought before":</p>$$\ell_t=\alpha\,(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1}),\quad b_t=\beta\,(\ell_t-\ell_{t-1})+(1-\beta)\,b_{t-1},\quad s_t=\gamma\,(y_t-\ell_t)+(1-\gamma)\,s_{t-m},$$$$\hat y_{t+h}=\ell_t+h\,b_t+s_{t+h-m}.$$<p>With \(\alpha\) near 1 the level chases every wobble. Near 0 it barely moves. Old data is never thrown away, only given exponentially less weight, which is where the name comes from (Holt, 1957; Winters, 1960).</p>
 <h4>Autoregression</h4><p>An AR(\(p\)) model predicts the next value as a weighted sum of the last \(p\):</p>$$\hat y_{t+1}=c+\phi_1y_t+\phi_2y_{t-1}+\dots+\phi_py_{t-p+1}.$$<p>The weights are found by least squares, exactly as in the Regression module, with the past values as the features. If \(p\) reaches back a whole cycle, the model can learn the seasonal pattern by itself, as a large weight near lag \(m\). ARIMA models extend this by first differencing the series to remove trend and by adding terms for past forecast errors.</p>
 <h4>Testing honestly: walking forward</h4><p>Shuffling the data into random training and test sets, as elsewhere in this course, would let the model see the future. The honest test respects time: at each step fit on the past only, forecast one step, then reveal it. The average absolute miss over the walk,</p>$$\mathrm{MAE}=\frac1n\sum_t\big|y_t-\hat y_t\big|,$$<p>is what the bottom chart tracks.</p>
 <h4>Forecasting further ahead</h4><p>For \(h\) steps ahead, Holt-Winters extends its line and repeats its seasonal offsets, and the autoregression feeds its own forecasts back in as if they were data. Errors accumulate, so the range of plausible outcomes widens with \(h\). The range drawn on this page is a rough one: plus or minus twice the method's typical one-step miss, multiplied by \(\sqrt h\). It is right for a random walk and a cautious overestimate for strongly seasonal data.</p>
 <h4>What this page runs</h4><p>The walk starts after the first 40% of the series, or after three full cycles if that is longer. Holt-Winters uses your \(\alpha\) with \(\beta=0.05\) and \(\gamma=0.25\). The autoregression is refitted at every step on everything revealed so far. An uploaded file should have one number per line, or its first numeric column is used.</p>`,
    q: 'On the share-price series, the sophisticated autoregression does no better than simply repeating the last value. Does that mean the autoregression is badly built?',
    a: 'No. It means there is nothing in the past of that series for any method to exploit. In a random walk each change is independent of everything before it, so the best forecast of the next value is the current one, and every extra weight the autoregression estimates is fitting noise. It ends up with a weight close to 1 on the last value and weights near 0 on the rest, which is the Last Value forecaster with some added jitter. A benchmark that cannot be beaten is informative: it tells you the series has no usable memory beyond its last value.',
    init(root) {
      const c = $('#ts-c', root),
        cr = $('#ts-r', root),
        live = $('#ts-live', root),
        eq = $('#ts-eq', root),
        st = $('#ts-st', root),
        S = { m: 2, s: 12, a: 0.3, p: 13, d: 'ice' },
        NM = ['Last Value', 'Same Season Last Cycle', 'Holt-Winters', 'Autoregression'],
        HZ = 24;
      let Yd = [],
        t0 = 0,
        t = 0,
        F = [[], [], [], []],
        AE = [[0], [0], [0], [0]],
        ER = [[], [], [], []],
        hw = null,
        phi = null,
        ACF = [],
        FUT = null,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        fin = false,
        custom = null,
        unit = '';
      const mk = (d) => {
        const r = rng(d === 'ice' ? 11 : d === 'temp' ? 23 : d === 'walk' ? 37 : 51),
          rn = randnFrom(r),
          o = [];
        if (d === 'ice')
          for (let i = 0; i < 96; i++)
            o.push(
              40 +
                0.45 * i +
                18 * Math.sin((((i % 12) - 3) / 12) * 2 * Math.PI) +
                6 * Math.max(0, Math.sin((((i % 12) - 3) / 12) * 2 * Math.PI)) +
                3.2 * rn(),
            );
        else if (d === 'temp')
          for (let i = 0; i < 96; i++)
            o.push(12 + 9 * Math.sin((((i % 12) - 3.5) / 12) * 2 * Math.PI) + 1.6 * rn());
        else if (d === 'walk') {
          let v = 100;
          for (let i = 0; i < 120; i++) {
            v += 2.2 * rn();
            o.push(v);
          }
        } else
          for (let i = 0; i < 112; i++)
            o.push(500 + 0.8 * i + [60, 80, 70, 65, 40, -150, -170][i % 7] + 25 * rn());
        return o;
      };
      const acf = (y) => {
        const n = y.length,
          m = y.reduce((a, b) => a + b, 0) / n;
        let v = 0;
        for (const q of y) v += (q - m) ** 2;
        const o = [];
        for (let k = 1; k <= 24; k++) {
          let s = 0;
          for (let i = k; i < n; i++) s += (y[i] - m) * (y[i - k] - m);
          o.push(v ? s / v : 0);
        }
        return o;
      };
      const hwInit = () => {
        const m = S.s,
          y = Yd,
          n = Math.min(t0, 2 * m),
          L = y.slice(0, m).reduce((a, b) => a + b, 0) / m,
          L2 = n >= 2 * m ? y.slice(m, 2 * m).reduce((a, b) => a + b, 0) / m : L;
        hw = {
          l: L,
          b: n >= 2 * m ? (L2 - L) / m : 0,
          s: Array.from({ length: m }, (_, i) => y[i] - L),
        };
        for (let i = 0; i < t0; i++) hwUp(y[i], i);
      };
      const hwUp = (y, i) => {
        const m = S.s,
          a = S.a,
          so = hw.s[i % m],
          l0 = hw.l;
        hw.l = a * (y - so) + (1 - a) * (hw.l + hw.b);
        hw.b = 0.05 * (hw.l - l0) + 0.95 * hw.b;
        hw.s[i % m] = 0.25 * (y - hw.l) + 0.75 * so;
      };
      const arFit = (n) => {
        const p = Math.min(S.p, Math.floor(n / 3)),
          k = p + 1,
          A = Array.from({ length: k }, () => new Float64Array(k)),
          b = new Float64Array(k);
        for (let i = p; i < n; i++) {
          const x = [1];
          for (let j = 1; j <= p; j++) x.push(Yd[i - j]);
          for (let r = 0; r < k; r++) {
            b[r] += x[r] * Yd[i];
            for (let q = 0; q < k; q++) A[r][q] += x[r] * x[q];
          }
        }
        for (let r = 1; r < k; r++) A[r][r] += 1e-6 * (A[r][r] + 1);
        for (let i = 0; i < k; i++) {
          let pv = i;
          for (let r = i + 1; r < k; r++) if (Math.abs(A[r][i]) > Math.abs(A[pv][i])) pv = r;
          [A[i], A[pv]] = [A[pv], A[i]];
          [b[i], b[pv]] = [b[pv], b[i]];
          const d = A[i][i] || 1e-12;
          for (let r = i + 1; r < k; r++) {
            const f = A[r][i] / d;
            for (let q = i; q < k; q++) A[r][q] -= f * A[i][q];
            b[r] -= f * b[i];
          }
        }
        const w = new Float64Array(k);
        for (let i = k - 1; i >= 0; i--) {
          let s = b[i];
          for (let q = i + 1; q < k; q++) s -= A[i][q] * w[q];
          w[i] = s / (A[i][i] || 1e-12);
        }
        return w;
      };
      const fc = (n) => {
        const m = S.s,
          p = phi.length - 1;
        let ar = phi[0];
        for (let j = 1; j <= p; j++) ar += phi[j] * Yd[n - j];
        return [Yd[n - 1], n - m >= 0 ? Yd[n - m] : Yd[n - 1], hw.l + hw.b + hw.s[n % m], ar];
      };
      const build = () => {
        Yd = custom || mk(S.d);
        const n = Yd.length;
        t0 = clamp(Math.max(Math.round(n * 0.4), 3 * S.s), 8, n - 6);
        t = t0;
        F = [[], [], [], []];
        AE = [[], [], [], []];
        ER = [[], [], [], []];
        FUT = null;
        fin = false;
        cu = 0;
        ACF = acf(Yd);
        hwInit();
        phi = arFit(t);
        msg =
          'The first ' +
          t0 +
          ' of ' +
          n +
          ' points are revealed. The rest are hidden. Each forecaster will now predict the next point before it is shown. Press One step or Run.';
        eqShow(eq, ['The walk has not started.'], 0);
      };
      const explain = (n, f) => {
        const m = S.s,
          w = S.m,
          L = [
            'Predicting point ' +
              (n + 1) +
              ' from the first ' +
              n +
              '. Watching: <b>' +
              NM[w] +
              '</b>.',
          ];
        if (w === 0) L.push('forecast = the last value seen = <b>' + f2(f[0]) + '</b>');
        else if (w === 1)
          L.push(
            'forecast = the value ' +
              m +
              ' steps ago (point ' +
              (n + 1 - m) +
              ') = <b>' +
              f2(f[1]) +
              '</b>',
          );
        else if (w === 2)
          L.push(
            'level ' +
              f2(hw.l) +
              '  +  slope ' +
              f3(hw.b) +
              '  +  offset for this place in the cycle ' +
              f2(hw.s[n % m]) +
              '  =  <b>' +
              f2(f[2]) +
              '</b>',
          );
        else {
          const p = phi.length - 1,
            top = [...Array(p).keys()]
              .map((j) => [Math.abs(phi[j + 1] * Yd[n - 1 - j]), j + 1])
              .sort((a, b) => b[0] - a[0])
              .slice(0, 3)
              .sort((a, b) => a[1] - b[1]);
          L.push(
            'constant ' +
              f2(phi[0]) +
              top
                .map(
                  (q) =>
                    '  +  ' +
                    f3(phi[q[1]]) +
                    ' × value ' +
                    q[1] +
                    ' step' +
                    (q[1] > 1 ? 's' : '') +
                    ' back (' +
                    f2(Yd[n - q[1]]) +
                    ')',
                )
                .join('') +
              (p > 3 ? '  +  ' + (p - 3) + ' smaller terms' : '') +
              '  =  <b>' +
              f2(f[3]) +
              '</b>',
          );
          L.push(
            'largest weights: ' +
              [...Array(p).keys()]
                .map((j) => [Math.abs(phi[j + 1]), j + 1])
                .sort((a, b) => b[0] - a[0])
                .slice(0, 3)
                .map((q) => 'lag ' + q[1] + ': ' + f2(phi[q[1]]))
                .join(',  '),
          );
        }
        return L;
      };
      const step = () => {
        if (fin) return;
        const n = t,
          f = fc(n),
          y = Yd[n],
          L = explain(n, f);
        for (let k = 0; k < 4; k++) {
          F[k].push(f[k]);
          ER[k].push(Math.abs(y - f[k]));
          AE[k].push(ER[k].reduce((a, b) => a + b, 0) / ER[k].length);
        }
        L.push(
          'the true value is revealed: ' +
            f2(y) +
            '.  This forecaster missed by <b>' +
            f2(Math.abs(y - f[S.m])) +
            '</b>',
        );
        eqShow(eq, L, 0);
        hwUp(y, n);
        t++;
        phi = arFit(t);
        cu = 2;
        const bestk = [0, 1, 2, 3].sort(
          (a, b) => AE[a][AE[a].length - 1] - AE[b][AE[b].length - 1],
        )[0];
        msg =
          'Point ' +
          t +
          ' revealed: ' +
          f2(y) +
          '. ' +
          NM[S.m] +
          ' had forecast ' +
          f2(f[S.m]) +
          '. Smallest average miss so far: ' +
          NM[bestk] +
          ', ' +
          f2(AE[bestk][AE[bestk].length - 1]) +
          '.';
        if (t >= Yd.length) {
          fin = true;
          cu = 5;
          future();
          msg =
            'The walk is finished. Average miss: ' +
            [0, 1, 2, 3].map((k) => NM[k] + ' ' + f2(AE[k][AE[k].length - 1])).join(', ') +
            '. The forecast for the next ' +
            HZ +
            ' points by ' +
            NM[S.m] +
            ' is drawn on the right of the series.';
        }
      };
      const future = () => {
        const n = Yd.length,
          m = S.s,
          ext = Yd.slice(),
          p = phi.length - 1,
          o = [];
        for (let h = 1; h <= HZ; h++) {
          let v;
          if (S.m === 0) v = Yd[n - 1];
          else if (S.m === 1) v = ext[n + h - 1 - m];
          else if (S.m === 2) v = hw.l + h * hw.b + hw.s[(n + h - 1) % m];
          else {
            v = phi[0];
            for (let j = 1; j <= p; j++) v += phi[j] * ext[n + h - 1 - j];
          }
          ext.push(v);
          o.push(v);
        }
        const e = ER[S.m],
          sd = Math.sqrt(e.reduce((a, b) => a + b * b, 0) / Math.max(1, e.length));
        FUT = { v: o, sd };
      };
      const tp = testPoint(root, 'ts', ['Steps ahead'], [6], [[1, HZ]]);
      const draw = () => {
        const tt = T(),
          n = Yd.length,
          tot = n + HZ,
          nf = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : f2(v));
        {
          const { g, w, h } = fit(c),
            m = { l: 44, r: 10, t: 12, b: 22 },
            all = Yd.concat(FUT ? FUT.v : []),
            lo = Math.min(...all),
            hi = Math.max(...all),
            pd = (hi - lo) * 0.3 || 1,
            X = (i) => m.l + (i / (tot - 1)) * (w - m.l - m.r),
            Yv = (v) => h - m.b - ((v - lo + pd) / (hi - lo + 2 * pd)) * (h - m.t - m.b);
          g.font = MONO;
          g.strokeStyle = tt.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h - m.b);
          g.lineTo(w - m.r, h - m.b);
          g.stroke();
          g.fillStyle = tt.muted;
          g.textAlign = 'right';
          g.fillText(nf(lo), m.l - 4, Yv(lo) + 4);
          g.fillText(nf(hi), m.l - 4, Yv(hi) + 4);
          g.save();
          g.beginPath();
          g.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
          g.clip();
          g.fillStyle = mix(tt.ink, tt.ink, 0, 0.06);
          g.fillRect(X(0), m.t, X(t0 - 1) - X(0), h - m.t - m.b);
          g.strokeStyle = mix(tt.ink, tt.ink, 0, 0.2);
          g.lineWidth = 1.2;
          g.beginPath();
          for (let i = Math.max(0, t - 1); i < n; i++)
            i === Math.max(0, t - 1) ? g.moveTo(X(i), Yv(Yd[i])) : g.lineTo(X(i), Yv(Yd[i]));
          g.stroke();
          g.strokeStyle = tt.ink;
          g.lineWidth = 1.8;
          g.beginPath();
          for (let i = 0; i < t; i++) i ? g.lineTo(X(i), Yv(Yd[i])) : g.moveTo(X(i), Yv(Yd[i]));
          g.stroke();
          const f = F[S.m];
          g.strokeStyle = mix(tt.gold, tt.gold, 0, 0.8);
          g.lineWidth = 1.4;
          g.beginPath();
          f.forEach((v, i) => (i ? g.lineTo(X(t0 + i), Yv(v)) : g.moveTo(X(t0 + i), Yv(v))));
          g.stroke();
          g.fillStyle = tt.gold;
          f.forEach((v, i) => {
            g.beginPath();
            g.arc(X(t0 + i), Yv(v), 2.6, 0, 7);
            g.fill();
          });
          if (!fin && t < n) {
            const nx = fc(t)[S.m];
            g.strokeStyle = tt.cut;
            g.lineWidth = 2;
            g.beginPath();
            g.arc(X(t), Yv(nx), 6, 0, 7);
            g.stroke();
          }
          if (FUT) {
            g.fillStyle = mix(tt.accent, tt.accent, 0, 0.2);
            g.beginPath();
            FUT.v.forEach((v, i) => {
              const y = Yv(v + 2 * FUT.sd * Math.sqrt(i + 1));
              i ? g.lineTo(X(n + i), y) : g.moveTo(X(n + i), y);
            });
            for (let i = HZ - 1; i >= 0; i--)
              g.lineTo(X(n + i), Yv(FUT.v[i] - 2 * FUT.sd * Math.sqrt(i + 1)));
            g.closePath();
            g.fill();
            g.strokeStyle = tt.accent;
            g.lineWidth = 2.4;
            g.beginPath();
            g.moveTo(X(n - 1), Yv(Yd[n - 1]));
            FUT.v.forEach((v, i) => g.lineTo(X(n + i), Yv(v)));
            g.stroke();
            const hh = clamp(Math.round(tp.v[0]), 1, HZ);
            markTP(g, X(n + hh - 1), Yv(FUT.v[hh - 1]), tt, false);
          }
          g.restore();
          g.fillStyle = tt.muted;
          g.textAlign = 'left';
          g.fillText('history given at the start', X(0) + 4, m.t + 10);
          g.textAlign = 'right';
          g.fillText(fin ? 'the future →' : '', w - m.r, m.t + 10);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.42,
            m = { l: 44, r: 12, t: 26, b: 20 },
            bw = (w - m.l - m.r) / 24,
            Y0 = m.t + (h1 - m.t - m.b) / 2,
            sc = (h1 - m.t - m.b) / 2;
          g.font = MONO;
          g.fillStyle = tt.muted;
          g.textAlign = 'left';
          g.fillText('resemblance to the value k steps earlier (autocorrelation)', 8, 13);
          ACF.forEach((v, i) => {
            g.fillStyle = i + 1 === S.s ? tt.gold : mix(tt.accent, tt.accent, 0, 0.7);
            g.fillRect(m.l + i * bw + 1, v >= 0 ? Y0 - v * sc : Y0, bw - 2, Math.abs(v) * sc);
            if ((i + 1) % 2 === 0 || i === 0) {
              g.fillStyle = i + 1 === S.s ? tt.cut : tt.muted;
              g.textAlign = 'center';
              g.fillText(String(i + 1), m.l + (i + 0.5) * bw, h1 - 4);
            }
          });
          g.strokeStyle = tt.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, Y0);
          g.lineTo(w - m.r, Y0);
          g.stroke();
          g.textAlign = 'right';
          g.fillStyle = tt.muted;
          g.fillText('1', m.l - 5, Y0 - sc + 4);
          g.fillText('−1', m.l - 5, Y0 + sc + 4);
          const col = [tt.muted, tt.ok, tt.accent, tt.cut],
            nn = Math.max(10, n - t0 + 1),
            ym =
              Math.max(
                1e-6,
                ...AE.map((a) => (a.length ? Math.max(...a.slice(Math.min(a.length - 1, 4))) : 0)),
              ) * 1.1;
          let L = null;
          AE.forEach((a, k) => {
            L = uLine(
              g,
              tt,
              0,
              h1,
              w - 150,
              h - h1,
              a.map((v) => Math.min(v, ym)),
              { title: k ? '' : 'average miss so far (lower is better)', n: nn, ym, col: col[k] },
            );
          });
          const SH = ['Last Value', 'Same Season', 'Holt-Winters', 'Autoregression'];
          AE.forEach((a, k) => {
            g.fillStyle = col[k];
            g.textAlign = 'left';
            g.font = (k === S.m ? '700 ' : '') + '11px ' + MONO.split('px ')[1];
            g.fillText(SH[k], w - 146, h1 + 40 + k * 30);
            g.fillText(a.length ? nf(a[a.length - 1]) : '–', w - 146, h1 + 53 + k * 30);
          });
        }
        live.textContent = msg;
        uCur(root, 'ts', cu);
        if (FUT) {
          const hh = clamp(Math.round(tp.v[0]), 1, HZ),
            v = FUT.v[hh - 1],
            r = 2 * FUT.sd * Math.sqrt(hh);
          tp.out.textContent =
            hh +
            ' step' +
            (hh > 1 ? 's' : '') +
            ' ahead, ' +
            NM[S.m] +
            ' forecasts ' +
            f2(v) +
            '. A rough range for what will actually happen: ' +
            f2(v - r) +
            ' to ' +
            f2(v + r) +
            '.';
        } else
          tp.out.textContent =
            'Finish the walk first: the forecast for the future appears at the end.';
        setR(root, 't', t + ' of ' + n);
        setR(root, 'f', t < n ? f2(fc(t)[S.m]) : FUT ? f2(FUT.v[0]) : '–');
        for (let k = 0; k < 4; k++)
          setR(root, 'e' + k, AE[k].length ? f2(AE[k][AE[k].length - 1]) : '–');
      };
      tp.on = () => draw();
      build();
      if (RM) {
        while (!fin) step();
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (fr % (t - t0 < 5 ? 30 : 5)) return;
        step();
        draw();
        if (fin) pb.stop();
      });
      pb = playBtn($('#ts-play', root), anim);
      runCtl(pb, root, 'ts', { done: () => fin, restart: () => build() });
      const re = (go) => {
        if (!pb) return;
        pb.stop();
        build();
        draw();
        if (go) pb.start();
      };
      $('#ts-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else step();
        draw();
      };
      $('#ts-reset', root).onclick = () => re(false);
      const bs = bind(
        root,
        'ts-s',
        (v) => {
          S.s = +v;
          re(false);
        },
        (v) => String(v),
      );
      bind(
        root,
        'ts-a',
        (v) => {
          S.a = +v;
          re(false);
        },
        f2,
      );
      bind(
        root,
        'ts-p',
        (v) => {
          S.p = +v;
          re(false);
        },
        (v) => String(v),
      );
      segBind(root, 'ts-m', String(S.m), (v) => {
        S.m = +v;
        if (fin) future();
        draw();
      });
      $('#ts-d', root).onchange = (e) => {
        S.d = e.target.value;
        custom = null;
        st.textContent = '';
        const m = S.d === 'traffic' ? 7 : 12;
        S.s = m;
        bs.set(m);
        re(true);
      };
      $('#ts-file', root).onchange = (e) => {
        const f = e.target.files[0];
        e.target.value = '';
        if (!f) return;
        const r = new FileReader();
        r.onerror = () => {
          st.textContent = 'That file could not be read. Save it as plain CSV and try again.';
        };
        r.onload = () => {
          const rows = String(r.result)
            .split(/\r?\n/)
            .map((l) => l.split(/[,;\t]/).map((x) => parseFloat(x)));
          let best = [];
          for (let cI = 0; cI < 6; cI++) {
            const col = rows.map((q) => q[cI]).filter((v) => isFinite(v));
            if (col.length > best.length) best = col;
          }
          if (best.length < 30) {
            st.textContent =
              'At least 30 numbers are needed, one per line or in one column. This file has ' +
              best.length +
              '.';
            return;
          }
          custom = best.slice(0, 400);
          st.textContent =
            'Loaded ' +
            f.name +
            ': ' +
            custom.length +
            ' values from its longest numeric column. Set the cycle length to match your data.';
          re(true);
        };
        r.readAsText(f);
      };
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
    id: 'explainability',
    tab: 'Explainability',
    title: 'Explainability: ask a model why',
    lede: 'A neural network has learned to estimate house prices from five facts about a house. It prices one particular house at well over 400 thousand. Why? “Because of 600 weights” is true and useless. A useful answer splits the estimate into parts: so much because it is large, so much off because it is old, so much because it is close to the centre. Shapley values are the one fair way to make that split. Describe a house with the sliders and watch its price being explained.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Top: one order of revealing the house’s features to the model, from the average house to this one. Bottom: each feature’s share of the price, averaged over all the orders tried so far. The dark tick is the exact answer</figcaption><canvas class="short" id="xp-c" role="img" aria-label="One feature ordering as a waterfall, and the running Shapley value estimates"></canvas></figure>
 <figure class="fig"><figcaption>Top: which features matter most across 80 different houses. Bottom: how the estimate for this house would change if one feature were different, and the same on average over many houses</figcaption><canvas class="short" id="xp-r" role="img" aria-label="Global feature importance and a dependence curve for one feature"></canvas></figure></div>
 <figure class="fig" style="margin-top:18px"><figcaption>The last order tried, worked out</figcaption><div class="eq" id="xp-eq" aria-live="polite" style="min-height:10em;margin-top:6px"></div></figure>
 ${uHow('xp', ['Start from the model’s estimate for an average house: nothing about this house is known yet.', 'Pick a random order for the five features. Reveal the first one: replace the average house’s value with this house’s, and see how the estimate moves.', 'Reveal the next feature, and the next, noting each move. After the fifth, the estimate is the one for this house.', 'That was one order. A feature’s effect depends on which others were already revealed, so try another order.', 'Average each feature’s moves over all the orders tried. That average is its Shapley value.', 'Stop when the averages have settled. They add up exactly to the difference between this house and the average house.'])}
 <div class="controls">${uBtns('xp', 'Run', 'One order', 'orders')}
 ${grp('Describe the house', slider('xp-f0', 'Floor area, m²', 40, 220, 5, 150) + slider('xp-f1', 'Age, years', 0, 90, 1, 12) + slider('xp-f2', 'Distance to the centre, km', 0, 30, 0.5, 4) + slider('xp-f3', 'Bedrooms', 1, 6, 1, 4) + slider('xp-f4', 'Garden (0 = no, 1 = yes)', 0, 1, 1, 1))}
 ${grp('The curve at the bottom right', `<label class="ctl" for="xp-ft"><span>Feature to vary</span><select id="xp-ft"><option value="0">Floor area</option><option value="1">Age</option><option value="2">Distance to the centre</option><option value="3">Bedrooms</option><option value="4">Garden</option></select></label>`)}</div>
 <dl class="read">${rd('pr', 'The model’s estimate for this house', 'hl')}${rd('ba', 'Its estimate for the average house')}${rd('n', 'Orders tried')}${rd('top', 'Feature that adds the most')}${rd('bot', 'Feature that takes away the most')}${rd('gap', 'Largest gap between the running averages and the exact values', 'hl2')}</dl>`,
    tryit:
      'Press One order several times. Floor area appears in a different position each time: is its contribution the same every time? That variation is why one order is not an explanation. Press Run and watch the bars settle onto the dark ticks. Check the arithmetic: the average house’s estimate plus the five bars equals this house’s estimate. Now move the garden slider to 0 and back. Then set the distance to 25 km and compare the garden’s share with before. In this model a garden is worth more close to the centre. Can a single “importance of the garden” number be right for every house? Look at the top-right chart for the answer the model gives on average.',
    math: String.raw`<h4>Intuition</h4><p>Four friends share a taxi home, each living further along the same road, and the fare comes to 40. How should they split it? Charging each the fare to their own door adds up to far more than 40. Splitting evenly is unfair to the one who gets out first. One fair scheme imagines the friends joining the ride one at a time in some order, charges each the extra cost they add, and then averages over every possible order of joining. A model's estimate is the fare, the features are the passengers, and the Shapley value is each feature's fair share.</p>
 <h4>What is being explained</h4><p>Take a trained model \(f\) and one input \(\mathbf x\), here a house with five features. The <b>base value</b> is the average prediction over typical houses, \(\phi_0=\mathbb E[f(\mathbf X)]\). An explanation is a set of numbers \(\phi_1,\dots,\phi_d\), one per feature, that account for the difference:</p>$$f(\mathbf x)=\phi_0+\sum_{j=1}^{d}\phi_j .$$
 <h4>The value of knowing some features</h4><p>For a subset \(S\) of features, let \(v(S)\) be the model's average prediction when the features in \(S\) are fixed to this house's values and the others are drawn from typical houses:</p>$$v(S)=\mathbb E\big[f(\mathbf x_S,\mathbf X_{\bar S})\big].$$<p>So \(v(\varnothing)=\phi_0\) and \(v(\text{all})=f(\mathbf x)\). Revealing feature \(j\) after the set \(S\) changes the prediction by \(v(S\cup\{j\})-v(S)\).</p>
 <h4>The Shapley value</h4><p>That change depends on \(S\). The Shapley value (Shapley, 1953) averages it over every order in which the features could be revealed:</p>$$\phi_j=\sum_{S\subseteq\{1..d\}\setminus\{j\}}\frac{|S|!\,(d-|S|-1)!}{d!}\,\big[v(S\cup\{j\})-v(S)\big].$$<p>The weight is the share of orderings in which exactly the features in \(S\) come before \(j\). With 5 features there are 32 subsets and the sum can be computed exactly, which gives the dark ticks. With hundreds of features it cannot, and the practical method is the one animated here: sample random orders and average. Lundberg and Lee (2017) made this the basis of the SHAP library.</p>
 <h4>Why this split and no other</h4><p>It is the only attribution that satisfies four reasonable demands at once. <b>Efficiency</b>: the shares add up to the difference from the base value. <b>Symmetry</b>: two features that always contribute equally get equal shares. <b>Dummy</b>: a feature that never changes the prediction gets zero. <b>Additivity</b>: the explanation of a sum of models is the sum of their explanations.</p>
 <h4>Interactions</h4><p>If the model were a plain weighted sum of the features, every order would give the same moves, and one order would be enough. The moves differ between orders because features interact: in this model a garden adds more to a large house near the centre than to a small one far out. The Shapley value shares each interaction equally among the features involved. That is fair, and it also means the garden's number is specific to this house.</p>
 <h4>Local and global</h4><p>A Shapley explanation is <b>local</b>: it is about one house. Averaging the sizes \(|\phi_j|\) over many houses gives a <b>global</b> ranking of features, the top-right chart. The curve at the bottom right is a different tool. The line for "this house" changes one feature and holds the others at this house's values. The line for "on average" does the same for many houses and averages, which is called a partial dependence plot.</p>
 <h4>What an explanation is not</h4><p>These numbers explain the <b>model</b>, not the world. If the model has learned that houses with gardens cost more, the garden gets a positive share, whether or not building a garden would raise a price. They are not causes. When features are strongly related, such as floor area and bedrooms, replacing one while keeping the other creates houses that do not exist, and the credit between the two is divided somewhat arbitrarily. And a faithful explanation of a wrong model is still an explanation of something wrong.</p>
 <h4>What this page runs</h4><p>A network with two hidden layers of 24 neurons, trained when the page opens on 700 invented houses whose prices follow a formula with interactions and some noise. "Typical houses" are a fixed sample of 60 of them. The run stops when no running average has moved by more than 0.3 thousand over 60 orders, or after 1,500 orders.</p>`,
    q: 'In one order, floor area is revealed first and moves the estimate by +70. In another it is revealed last and moves it by +95. Which number is the true effect of floor area for this house?',
    a: 'Neither, and both. Each is the correct answer to a slightly different question: “how much does knowing the area change the estimate when nothing else is known?” and “how much does it change when everything else is already known?”. They differ because area interacts with the other features. A large house close to the centre gains more from its size than a house whose location is still treated as average. There is no single order that is the right one, which is why the Shapley value averages over all of them. The average is a fair summary, and the spread between orders is itself a sign of how strongly the feature interacts.',
    init(root) {
      const c = $('#xp-c', root),
        cr = $('#xp-r', root),
        live = $('#xp-live', root),
        eq = $('#xp-eq', root),
        S = { x: [150, 12, 4, 4, 1], ft: 0 },
        NF = 5,
        FN = ['floor area', 'age', 'distance', 'bedrooms', 'garden'],
        UN = [' m²', ' years', ' km', '', ''],
        RG = [
          [40, 220],
          [0, 90],
          [0, 30],
          [1, 6],
          [0, 1],
        ],
        MAXN = 1500,
        NB = 60;
      let net,
        MU,
        SD,
        BG = [],
        base = 0,
        pr = 0,
        EX = [],
        SUM,
        n = 0,
        EST,
        HIST = [],
        last = null,
        GI = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        fin = false,
        VC = null;
      const truth = (a, r) => {
        const [s, ag, d, b, gd] = a,
          near = 1 - d / 34;
        return (
          45 +
          1.5 * s +
          1.1 * s * near -
          0.9 * ag +
          0.004 * ag * ag * 2 -
          3.2 * d +
          9 * b +
          gd * (8 + 0.12 * s * near) +
          (r ? 9 * r() : 0)
        );
      };
      const sampleH = (r) => {
        const s = 45 + 170 * r(),
          ag = 85 * r() * r(),
          d = 30 * r(),
          b = clamp(Math.round(1 + s / 45 + 0.8 * (r() - 0.5) * 2), 1, 6),
          gd = r() < 0.3 + 0.4 * (d / 30) ? 1 : 0;
        return [s, ag, d, b, gd];
      };
      const fz = (rows) => {
        const B = rows.length,
          X = new Float64Array(B * NF);
        rows.forEach((q, i) => {
          for (let j = 0; j < NF; j++) X[i * NF + j] = (q[j] - MU[j]) / SD[j];
        });
        const o = net.fwd(AD.input(X, [B, NF]));
        AD.clear();
        return Array.from(o.v, (v) => v * 100);
      };
      const val = (mask) => {
        const key = mask;
        if (VC.has(key)) return VC.get(key);
        const rows = BG.map((b) => b.map((v, j) => ((mask >> j) & 1 ? S.x[j] : v))),
          o = fz(rows),
          m = o.reduce((a, b) => a + b, 0) / o.length;
        VC.set(key, m);
        return m;
      };
      const train = () => {
        const r = rng(31),
          rn = randnFrom(r),
          D = Array.from({ length: 700 }, () => sampleH(r)),
          Yv = D.map((q) => truth(q, rn) / 100);
        MU = [0, 1, 2, 3, 4].map((j) => D.reduce((s, q) => s + q[j], 0) / D.length);
        SD = [0, 1, 2, 3, 4].map(
          (j) => Math.sqrt(D.reduce((s, q) => s + (q[j] - MU[j]) ** 2, 0) / D.length) || 1,
        );
        net = AD.mlp([NF, 24, 24, 1], randnFrom(rng(7)), 'tanh');
        const Xs = D.map((q) => q.map((v, j) => (v - MU[j]) / SD[j]));
        for (let ts = 1; ts <= 900; ts++) {
          const B = 64,
            X = new Float64Array(B * NF),
            Yb = new Float64Array(B);
          for (let i = 0; i < B; i++) {
            const k = Math.floor(r() * D.length);
            X.set(Xs[k], i * NF);
            Yb[i] = Yv[k];
          }
          AD.backward(AD.mse(net.fwd(AD.input(X, [B, NF])), Yb));
          AD.adam(net.ps, 0.01 * Math.pow(0.15, ts / 900), ts, 1);
        }
        BG = D.slice(0, NB);
        const keep = S.x;
        GI = new Array(NF).fill(0);
        const H = D.slice(100, 180);
        for (const hh of H) {
          S.x = hh;
          VC = new Map();
          const e = exact();
          for (let j = 0; j < NF; j++) GI[j] += Math.abs(e[j]) / H.length;
        }
        S.x = keep;
      };
      const exact = () => {
        const fact = [1, 1, 2, 6, 24, 120],
          o = new Array(NF).fill(0);
        for (let j = 0; j < NF; j++)
          for (let m = 0; m < 32; m++) {
            if ((m >> j) & 1) continue;
            let sz = 0;
            for (let q = 0; q < NF; q++) if ((m >> q) & 1) sz++;
            o[j] += ((fact[sz] * fact[NF - sz - 1]) / 120) * (val(m | (1 << j)) - val(m));
          }
        return o;
      };
      const build = () => {
        VC = new Map();
        base = val(0);
        pr = val(31);
        EX = exact();
        SUM = new Float64Array(NF);
        EST = new Float64Array(NF);
        n = 0;
        HIST = [];
        last = null;
        fin = false;
        cu = 0;
        msg =
          'The model estimates this house at ' +
          f2(pr) +
          ' thousand. For an average house it says ' +
          f2(base) +
          '. The difference of ' +
          f2(pr - base) +
          ' has to be shared out among the five features. Press One order or Run.';
        eqShow(eq, ['No orders tried yet.'], 0);
      };
      const order = () => {
        if (fin) return;
        const o = [0, 1, 2, 3, 4]
          .map((j) => [Math.random(), j])
          .sort((a, b) => a[0] - b[0])
          .map((q) => q[1]);
        let m = 0,
          v = base;
        const st = [];
        for (const j of o) {
          m |= 1 << j;
          const v2 = val(m);
          st.push({ j, from: v, to: v2 });
          SUM[j] += v2 - v;
          v = v2;
        }
        n++;
        const prev = Float64Array.from(EST);
        for (let j = 0; j < NF; j++) EST[j] = SUM[j] / n;
        HIST.push(Math.max(...[0, 1, 2, 3, 4].map((j) => Math.abs(EST[j] - prev[j]))));
        last = st;
        eqShow(
          eq,
          [
            'Order ' + n + ': ' + o.map((j) => FN[j]).join('  →  '),
            'start from the average house: ' + f2(base),
          ].concat(
            st.map(
              (q) =>
                'reveal ' +
                FN[q.j] +
                ' = ' +
                S.x[q.j] +
                UN[q.j] +
                ':   ' +
                f2(q.from) +
                ' → ' +
                f2(q.to) +
                '   (<b>' +
                (q.to >= q.from ? '+' : '−') +
                f2(Math.abs(q.to - q.from)) +
                '</b>)   average for ' +
                FN[q.j] +
                ' over ' +
                n +
                ' order' +
                (n > 1 ? 's' : '') +
                ': ' +
                f2(EST[q.j]),
            ),
          ),
          0,
        );
        cu = n < 2 ? 2 : 4;
        const gp = Math.max(...[0, 1, 2, 3, 4].map((j) => Math.abs(EST[j] - EX[j])));
        msg =
          'Order ' +
          n +
          ': ' +
          o.map((j) => FN[j]).join(', ') +
          '. The running averages are within ' +
          f2(gp) +
          ' thousand of the exact Shapley values.';
        if ((n >= 80 && Math.max(...HIST.slice(-60)) < 0.3) || n >= MAXN) {
          fin = true;
          cu = 5;
          const idx = [0, 1, 2, 3, 4].sort((a, b) => EST[b] - EST[a]);
          msg =
            (n >= MAXN
              ? 'Stopped at the limit of ' + MAXN + ' orders.'
              : 'Stopped after ' +
                n +
                ' orders: no average has moved by more than 0.3 over the last 60.') +
            ' Check: ' +
            f2(base) +
            ' (average house) ' +
            [0, 1, 2, 3, 4]
              .map((j) => (EST[j] >= 0 ? '+ ' : '− ') + f2(Math.abs(EST[j])))
              .join(' ') +
            ' = ' +
            f2(base + EST.reduce((a, b) => a + b, 0)) +
            ', and the model’s estimate is ' +
            f2(pr) +
            '. The largest share is ' +
            FN[idx[0]] +
            '.';
        }
      };
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c),
            h1 = h * 0.44,
            m = { l: 92, r: 16 },
            all = [base, pr].concat(last ? last.flatMap((q) => [q.from, q.to]) : []),
            lo = Math.min(...all) - 12,
            hi = Math.max(...all) + 12,
            X = (v) => m.l + ((v - lo) / (hi - lo)) * (w - m.l - m.r),
            rh = (h1 - 30) / 6;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            last
              ? 'order ' + n + ': from the average house to this house, one feature at a time'
              : 'the first order appears here',
            8,
            13,
          );
          g.textAlign = 'right';
          g.fillStyle = t.muted;
          g.fillText('average house', m.l - 8, 24 + rh * 0.5 + 4);
          g.strokeStyle = t.muted;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(X(base), 20);
          g.lineTo(X(base), 20 + rh);
          g.stroke();
          g.textAlign = 'left';
          g.fillText(f2(base), X(base) + 5, 24 + rh * 0.5 + 4);
          if (last)
            last.forEach((q, i) => {
              const y = 20 + (i + 1) * rh,
                up = q.to >= q.from;
              g.fillStyle = up ? t.accent : t.cut;
              g.fillRect(
                Math.min(X(q.from), X(q.to)),
                y + 3,
                Math.max(2, Math.abs(X(q.to) - X(q.from))),
                rh - 6,
              );
              g.fillStyle = t.ink;
              g.textAlign = 'right';
              g.fillText(FN[q.j], m.l - 8, y + rh / 2 + 4);
              g.textAlign = up ? 'left' : 'right';
              g.fillStyle = t.muted;
              g.fillText(
                (up ? '+' : '−') + f2(Math.abs(q.to - q.from)),
                X(q.to) + (up ? 5 : -5),
                y + rh / 2 + 4,
              );
              g.strokeStyle = t.line;
              g.lineWidth = 1;
              g.beginPath();
              g.moveTo(X(q.to), y + rh - 3);
              g.lineTo(X(q.to), y + rh + 3);
              g.stroke();
            });
          const y0 = h1 + 22,
            H = h - y0 - 8,
            r2 = H / NF,
            mx = Math.max(10, ...EX.map(Math.abs), ...Array.from(EST).map(Math.abs)) * 1.25,
            X2 = (v) => m.l + (w - m.l - m.r) / 2 + ((v / mx) * (w - m.l - m.r)) / 2;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            'each feature’s share of the price, in thousands: average over ' +
              n +
              ' order' +
              (n === 1 ? '' : 's'),
            8,
            h1 + 12,
          );
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(X2(0), y0);
          g.lineTo(X2(0), y0 + H);
          g.stroke();
          for (let j = 0; j < NF; j++) {
            const y = y0 + j * r2,
              v = EST[j];
            g.fillStyle = v >= 0 ? t.accent : t.cut;
            if (n) g.fillRect(Math.min(X2(0), X2(v)), y + 4, Math.abs(X2(v) - X2(0)), r2 - 8);
            g.strokeStyle = t.ink;
            g.lineWidth = 3;
            g.beginPath();
            g.moveTo(X2(EX[j]), y + 1);
            g.lineTo(X2(EX[j]), y + r2 - 1);
            g.stroke();
            g.fillStyle = t.ink;
            g.textAlign = 'right';
            g.fillText(FN[j], m.l - 8, y + r2 / 2 - 2);
            g.fillStyle = t.muted;
            g.fillText(S.x[j] + UN[j], m.l - 8, y + r2 / 2 + 11);
            g.textAlign = v >= 0 ? 'left' : 'right';
            g.fillText(
              n ? (v >= 0 ? '+' : '−') + f2(Math.abs(v)) : '',
              X2(v) + (v >= 0 ? 6 : -6),
              y + r2 / 2 + 4,
            );
          }
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.4,
            m = { l: 92, r: 40 },
            rh = (h1 - 26) / NF,
            mx = Math.max(...GI) * 1.1;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('average size of each feature’s share, over 80 houses (thousands)', 8, 13);
          for (let j = 0; j < NF; j++) {
            const y = 20 + j * rh;
            g.fillStyle = j === S.ft ? t.gold : mix(t.accent, t.accent, 0, 0.75);
            g.fillRect(m.l, y + 3, ((w - m.l - m.r) * GI[j]) / mx, rh - 6);
            g.fillStyle = t.ink;
            g.textAlign = 'right';
            g.fillText(FN[j], m.l - 8, y + rh / 2 + 4);
            g.textAlign = 'left';
            g.fillStyle = t.muted;
            g.fillText(f2(GI[j]), m.l + ((w - m.l - m.r) * GI[j]) / mx + 5, y + rh / 2 + 4);
          }
          const j = S.ft,
            NP = j >= 3 ? (j === 3 ? 6 : 2) : 30,
            xs = Array.from(
              { length: NP },
              (_, i) => RG[j][0] + ((RG[j][1] - RG[j][0]) * i) / (NP - 1),
            ),
            ice = fz(xs.map((v) => S.x.map((q, k) => (k === j ? v : q)))),
            pdp = xs.map((v) => {
              const o = fz(BG.map((b) => b.map((q, k) => (k === j ? v : q))));
              return o.reduce((a, b) => a + b, 0) / o.length;
            }),
            lo = Math.min(...ice, ...pdp),
            hi = Math.max(...ice, ...pdp),
            L = uLine(g, t, 0, h1, w, h - h1, ice, {
              title: 'estimate as ' + FN[j] + ' changes, everything else held fixed',
              n: NP,
              y1: lo - 5,
              ym: hi + 5,
              col: t.accent,
              dots: NP < 8,
            });
          g.strokeStyle = t.muted;
          g.lineWidth = 2;
          g.setLineDash([6, 4]);
          g.beginPath();
          pdp.forEach((v, i) => (i ? g.lineTo(L.X(i), L.Y(v)) : g.moveTo(L.X(i), L.Y(v))));
          g.stroke();
          g.setLineDash([]);
          const xi = ((S.x[j] - RG[j][0]) / (RG[j][1] - RG[j][0])) * (NP - 1);
          g.fillStyle = t.gold;
          g.beginPath();
          g.arc(L.X(xi), L.Y(pr), 6, 0, 7);
          g.fill();
          g.textAlign = 'right';
          g.fillStyle = t.accent;
          g.fillText('this house', w - 14, h - 48);
          g.fillStyle = t.muted;
          g.fillText('dashed: on average over typical houses', w - 14, h - 34);
          g.textAlign = 'left';
          g.fillText(RG[j][0] + UN[j], L.m.l, h - 5);
          g.textAlign = 'right';
          g.fillText(RG[j][1] + UN[j], w - L.m.r, h - 5);
        }
        live.textContent = msg;
        uCur(root, 'xp', cu);
        const src = n ? Array.from(EST) : EX,
          idx = [0, 1, 2, 3, 4].sort((a, b) => src[b] - src[a]);
        setR(root, 'pr', f2(pr) + ' thousand');
        setR(root, 'ba', f2(base) + ' thousand');
        setR(root, 'n', n + (fin ? ' (stopped)' : ''));
        setR(root, 'top', n ? FN[idx[0]] + ' (+' + f2(src[idx[0]]) + ')' : '–');
        setR(
          root,
          'bot',
          n
            ? src[idx[4]] < 0
              ? FN[idx[4]] + ' (−' + f2(-src[idx[4]]) + ')'
              : 'none takes away'
            : '–',
        );
        setR(
          root,
          'gap',
          n ? f2(Math.max(...[0, 1, 2, 3, 4].map((j) => Math.abs(EST[j] - EX[j])))) : '–',
        );
      };
      train();
      build();
      if (RM) {
        while (!fin) order();
      }
      const anim = Anim(() => {
        fr++;
        if (fin) {
          pb.stop();
          return;
        }
        if (fr % (n < 6 ? 42 : n < 30 ? 8 : 2)) return;
        order();
        draw();
        chk();
      });
      pb = playBtn($('#xp-play', root), anim);
      chk = runCtl(pb, root, 'xp', {
        done: () => fin,
        restart: () => build(),
        count: () => n,
        what: 'orders',
      });
      $('#xp-step', root).onclick = () => {
        pb.stop();
        if (fin) build();
        else order();
        draw();
      };
      $('#xp-reset', root).onclick = () => {
        pb.stop();
        build();
        draw();
      };
      for (let j = 0; j < NF; j++)
        bind(
          root,
          'xp-f' + j,
          (v) => {
            S.x[j] = +v;
            if (pb) {
              pb.stop();
              build();
              draw();
            }
          },
          (v) => (j === 4 ? (+v ? 'yes' : 'no') : String(v)),
        );
      $('#xp-ft', root).onchange = (e) => {
        S.ft = +e.target.value;
        draw();
      };
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
];
