/* Manifold — The Gaussian.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ the Gaussian lessons ============ */
const Z250 = (() => {
  const r = randnFrom(rng(11)),
    a = [];
  for (let i = 0; i < 250; i++) a.push([r(), r()]);
  return a;
})();
const LESSONS = [
  {
    id: 'galton',
    tab: 'Galton board',
    title: 'A board of pegs makes a bell',
    lede: 'Each ball hits a peg and bounces left or right at random, row after row. No single ball is predictable. The pile at the bottom always is.',
    stage: `<div class="stage one"><figure class="fig"><canvas class="tall" id="gb-c" role="img" aria-label="Balls falling through rows of pegs and piling into a bell shape"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="gb-play" type="button">Play</button><button class="btn" id="gb-reset" type="button">Empty the board</button></div>${slider('gb-rate', 'Balls per second', 1, 80, 1, 14)}`)}
 ${grp('The board', slider('gb-rows', 'Rows of pegs', 3, 16, 1, 12) + slider('gb-p', 'Chance of bouncing right', 0.15, 0.85, 0.01, 0.5))}</div>
 <dl class="read">${rd('n', 'Balls landed')}${rd('m', 'Average slot', 'hl3')}${rd('pm', 'Predicted average', 'hl')}${rd('s', 'Spread of the pile', 'hl3')}${rd('ps', 'Predicted spread', 'hl')}</dl>`,
    tryit:
      'Follow one ball with your eyes and guess its slot. Then tilt the odds to 0.7: the pile moves, but does it keep its shape? Finally drop to 3 rows. Is it still a bell?',
    math: String.raw`<h4>Intuition</h4><p>Every peg is a coin flip. A ball's final slot is just the number of times it went right, so a ball that lands in slot \(k\) after \(n\) rows made exactly \(k\) right turns. There is one path to the far edge and a huge number of paths to the middle, and every individual path is equally likely when \(p=\tfrac12\). The pile is tall where paths are many.</p>
 <h4>The formula</h4><p>Counting paths gives the binomial distribution:</p>$$P(K=k)=\binom{n}{k}\,p^{k}(1-p)^{n-k},\qquad \mathbb{E}[K]=np,\quad \operatorname{Var}(K)=np(1-p).$$
 <p>For many rows the binomial is very close to a Gaussian with the same mean and variance (the de Moivre–Laplace theorem), which is the curve drawn over the pile:</p>$$\binom{n}{k}p^{k}(1-p)^{n-k}\;\approx\;\frac{1}{\sqrt{2\pi\,np(1-p)}}\exp\!\left(-\frac{(k-np)^2}{2\,np(1-p)}\right).$$
 <h4>Reading it</h4><p>The board is \(n\) slots wide, but the pile is only about \(\sqrt{np(1-p)}\) wide. Doubling the rows doubles the width of the board and widens the pile by only \(\sqrt2\). Sums of many small independent pushes concentrate, and their shape is always this one.</p>`,
    q: 'Why do so few balls reach the outer slots?',
    a: 'To land at the far right a ball must bounce right every single time, and only one path does that. There are many different paths to the middle slots, so most balls end up there.',
    init(root) {
      const c = $('#gb-c', root),
        S = { rows: 12, p: 0.5, rate: 14 };
      let balls,
        cnt,
        n,
        acc = 0;
      const reset = () => {
        balls = [];
        cnt = new Array(S.rows + 1).fill(0);
        n = 0;
      };
      reset();
      const draw = () => {
        const t = T(),
          { g, w, h } = fit(c),
          top = 14,
          pegH = h * 0.5,
          dy = pegH / S.rows,
          dx = Math.min((w - 30) / (S.rows + 2), dy * 1.5),
          cx = w / 2,
          binTop = top + pegH + 8,
          binH = h - binTop - 24,
          br = clamp(dx * 0.2, 2.2, 5.5);
        g.fillStyle = t.muted;
        for (let r = 0; r < S.rows; r++)
          for (let i = 0; i <= r; i++) {
            g.beginPath();
            g.arc(cx + (i - r / 2) * dx, top + (r + 0.5) * dy, Math.min(2.6, dx * 0.09), 0, 7);
            g.fill();
          }
        let mx = 1;
        for (const k of cnt) mx = Math.max(mx, k);
        const unit = Math.min(br * 1.6, binH / mx),
          bw = dx * 0.78;
        for (let k = 0; k <= S.rows; k++) {
          const x = cx + (k - S.rows / 2) * dx,
            hh = cnt[k] * unit;
          g.fillStyle = ramp(t, 0.25 + (0.75 * cnt[k]) / mx);
          g.fillRect(x - bw / 2, binTop + binH - hh, bw, hh);
        }
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(cx - (S.rows / 2 + 0.6) * dx, binTop + binH);
        g.lineTo(cx + (S.rows / 2 + 0.6) * dx, binTop + binH);
        g.stroke();
        if (n > 20) {
          const mu = S.rows * S.p,
            sd = Math.sqrt(S.rows * S.p * (1 - S.p));
          g.beginPath();
          for (let i = 0; i <= 120; i++) {
            const k = -0.6 + ((S.rows + 1.2) * i) / 120,
              y = binTop + binH - n * npdf(k, mu, sd) * unit,
              x = cx + (k - S.rows / 2) * dx;
            i ? g.lineTo(x, y) : g.moveTo(x, y);
          }
          g.strokeStyle = t.ok;
          g.lineWidth = 2.6;
          g.stroke();
        }
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        for (let k = 0; k <= S.rows; k += S.rows > 10 ? 2 : 1)
          g.fillText(String(k), cx + (k - S.rows / 2) * dx, binTop + binH + 15);
        g.fillStyle = t.gold;
        for (const b of balls) {
          let x, y;
          if (b.r < S.rows) {
            const r0 = Math.floor(b.r),
              f = b.r - r0,
              k0 = b.pre[r0],
              x0 = cx + (k0 - r0 / 2) * dx,
              y0 = top + (r0 + 0.5) * dy - br - 3,
              e = f * f * (3 - 2 * f);
            x = x0 + (b.dirs[r0] - 0.5) * dx * e;
            y = y0 + dy * f - Math.sin(Math.PI * f) * dy * 0.28;
          } else {
            const f = Math.min(1, (b.r - S.rows) / 3);
            x = cx + (b.k - S.rows / 2) * dx;
            const ya = top + (S.rows + 0.5) * dy - br - 3;
            y = ya + (binTop + binH - cnt[b.k] * unit - br - ya) * f * f;
          }
          g.beginPath();
          g.arc(x, y, br, 0, 7);
          g.fill();
        }
        let s1 = 0,
          s2 = 0;
        for (let k = 0; k <= S.rows; k++) {
          s1 += k * cnt[k];
          s2 += k * k * cnt[k];
        }
        const m = n ? s1 / n : 0;
        setR(root, 'n', String(n));
        setR(root, 'm', n ? f2(m) : '–');
        setR(root, 's', n > 1 ? f2(Math.sqrt(Math.max(0, s2 / n - m * m))) : '–');
        setR(root, 'pm', f2(S.rows * S.p));
        setR(root, 'ps', f2(Math.sqrt(S.rows * S.p * (1 - S.p))));
      };
      const spawn = () => {
        const dirs = [],
          pre = [0];
        let k = 0;
        for (let r = 0; r < S.rows; r++) {
          const d = Math.random() < S.p ? 1 : 0;
          dirs.push(d);
          k += d;
          pre.push(k);
        }
        balls.push({ r: 0, k, dirs, pre });
      };
      const anim = Anim(() => {
        acc += S.rate / 60;
        while (acc >= 1) {
          acc--;
          if (balls.length < 500) spawn();
        }
        balls = balls.filter((b) => {
          b.r += 0.085;
          if (b.r >= S.rows + 3) {
            cnt[b.k]++;
            n++;
            return false;
          }
          return true;
        });
        draw();
      });
      const pb = playBtn($('#gb-play', root), anim);
      bind(
        root,
        'gb-rate',
        (v) => {
          S.rate = v;
        },
        (v) => String(v),
      );
      bind(
        root,
        'gb-rows',
        (v) => {
          S.rows = v;
          reset();
          draw();
        },
        (v) => String(v),
      );
      bind(root, 'gb-p', (v) => {
        S.p = v;
        reset();
        draw();
      });
      $('#gb-reset', root).onclick = () => {
        reset();
        draw();
      };
      return { draw, stop: pb.stop, auto: pb.start };
    },
  },

  {
    id: 'bell',
    tab: 'Bell curve',
    title: 'One curve, two numbers',
    lede: 'The mean μ slides the curve left and right. The standard deviation σ makes it wide and low or narrow and tall. The area under it is always 1.',
    stage: `<div class="stage one"><figure class="fig"><canvas class="wide" id="bell-c" role="img" aria-label="Bell curve with shaded area"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="bell-play" type="button">Animate σ</button></div>`)}
 ${grp('The curve', slider('bell-mu', 'Mean μ', -4, 4, 0.1, 0) + slider('bell-sig', 'Standard deviation σ', 0.4, 3, 0.05, 1))}${grp('Shaded band', slider('bell-k', 'Width, in units of σ', 0.1, 3.5, 0.1, 1))}</div>
 <dl class="read">${rd('p', 'Area inside the shaded band', 'hl2')}${rd('int', 'Shaded interval')}${rd('var', 'Variance σ²')}${rd('peak', 'Peak height', 'hl3')}</dl>`,
    tryit:
      'Before you move anything: if you double σ, what happens to the peak height? Then set the band to 1, 2 and 3 and read the shaded area each time.',
    math: String.raw`<h4>Intuition</h4><p>The Gaussian says: values near the mean are common, and they get rare very quickly as you move away. How quickly is set by \(\sigma\), which acts as the unit of distance. A value two \(\sigma\) away is equally surprising whether \(\sigma\) is 0.1 or 100.</p>
 <h4>The formula</h4>$$p(x)=\frac{1}{\sigma\sqrt{2\pi}}\,\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
 <p>Read it from the inside out. \(z=(x-\mu)/\sigma\) is the distance from the mean in units of \(\sigma\). Squaring makes left and right the same. The exponential \(e^{-z^2/2}\) turns distance into rarity. The factor in front only fixes the total area at 1.</p>
 <h4>Why the peak drops when σ grows</h4><p>The area must stay 1. Stretching the curve sideways by a factor of 2 would double its area, so the height has to halve: the peak is \(1/(\sigma\sqrt{2\pi})\). The curve changes from bending down to bending up exactly at \(\mu\pm\sigma\), which is where the little bar marked σ ends.</p>
 <h4>The 68–95–99.7 rule</h4>$$P\big(|X-\mu|<k\sigma\big)=\operatorname{erf}\!\left(\frac{k}{\sqrt2}\right)\quad\Rightarrow\quad 68.3\%,\;95.4\%,\;99.7\%\ \text{for }k=1,2,3.$$<p>These numbers do not depend on \(\mu\) or \(\sigma\), because every Gaussian is the standard one, \(N(0,1)\), shifted and stretched.</p>`,
    q: 'A value sits 2.5σ above the mean. Roughly how unusual is that?',
    a: 'Set the band to 2.5: about 98.8% of the area is inside, so about 1.2% is outside, split between two tails. Roughly 0.6% of values land that far above the mean.',
    init(root) {
      const c = $('#bell-c', root),
        S = { mu: 0, sig: 1, k: 1 };
      const draw = () => {
        const t = T(),
          { g, w, h } = fit(c),
          xr = [-8, 8],
          P = plot(g, w, h, xr, [0, 1.05], t, { yticks: [0.2, 0.4, 0.6, 0.8, 1], xlabel: 'x' }),
          f = (x) => npdf(x, S.mu, S.sig);
        clipPlot(g, P);
        [3, 2, 1].forEach((j) => {
          area(g, P, f, Math.max(xr[0], S.mu - j * S.sig), Math.min(xr[1], S.mu + j * S.sig));
          g.fillStyle = mix(t.accent, t.accent, 0, 0.1);
          g.fill();
        });
        const a = Math.max(xr[0], S.mu - S.k * S.sig),
          b = Math.min(xr[1], S.mu + S.k * S.sig);
        area(g, P, f, a, b);
        g.fillStyle = mix(t.ok, t.ok, 0, 0.28);
        g.fill();
        g.strokeStyle = t.ok;
        g.lineWidth = 1.5;
        [a, b].forEach((x) => {
          g.beginPath();
          g.moveTo(P.X(x), P.Y(0));
          g.lineTo(P.X(x), P.Y(f(x)));
          g.stroke();
        });
        curve(g, P, f, xr[0], xr[1]);
        g.strokeStyle = t.accent;
        g.lineWidth = 3.2;
        g.stroke();
        g.setLineDash([4, 4]);
        g.strokeStyle = t.muted;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(P.X(S.mu), P.Y(0));
        g.lineTo(P.X(S.mu), P.Y(f(S.mu)));
        g.stroke();
        g.setLineDash([]);
        const yi = f(S.mu + S.sig);
        g.strokeStyle = t.ink;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(P.X(S.mu), P.Y(yi));
        g.lineTo(P.X(S.mu + S.sig), P.Y(yi));
        g.stroke();
        g.font = SANS;
        g.fillStyle = t.ink;
        g.textAlign = 'center';
        g.fillText('σ', P.X(S.mu + S.sig / 2), P.Y(yi) - 6);
        g.fillText('μ', P.X(S.mu), P.Y(f(S.mu)) - 8);
        g.restore();
        setR(root, 'p', (100 * erf(S.k / Math.SQRT2)).toFixed(1) + '%');
        setR(root, 'int', '[' + f2(S.mu - S.k * S.sig) + ', ' + f2(S.mu + S.k * S.sig) + ']');
        setR(root, 'var', f2(S.sig * S.sig));
        setR(root, 'peak', f3(f(S.mu)));
      };
      bind(
        root,
        'bell-mu',
        (v) => {
          S.mu = v;
          draw();
        },
        (v) => (+v).toFixed(1),
      );
      const sg = bind(root, 'bell-sig', (v) => {
        S.sig = v;
        draw();
      });
      bind(
        root,
        'bell-k',
        (v) => {
          S.k = v;
          draw();
        },
        (v) => (+v).toFixed(1),
      );
      const pb = playBtn($('#bell-play', root), sweeper(sg.el, 0.4, 3));
      grab(sg.el, pb);
      return { draw, stop: pb.stop, auto: pb.start };
    },
  },

  {
    id: 'sampling',
    tab: 'Sampling',
    title: 'Samples build the curve',
    lede: 'A distribution is a recipe for random numbers. Choose μ and σ, draw one number at a time and stack them. The pile is rough at first and settles onto the curve.',
    stage: `<div class="stage one"><figure class="fig"><canvas class="wide" id="sm-c" role="img" aria-label="Histogram of samples under the bell curve"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="sm-play" type="button">Play</button><button class="btn" id="sm-step" type="button">Draw one</button><button class="btn" id="sm-reset" type="button">Reset</button></div>${slider('sm-speed', 'Samples per frame', 1, 200, 1, 2)}`)}
 ${grp('The distribution', slider('sm-mu', 'Mean μ', -3, 3, 0.1, 0) + slider('sm-sig', 'Standard deviation σ', 0.4, 2, 0.05, 1))}${grp('The histogram', slider('sm-bins', 'Number of bins', 10, 96, 2, 48))}</div>
 <dl class="read">${rd('n', 'Samples drawn')}${rd('m', 'Sample mean', 'hl3')}${rd('s', 'Sample standard deviation', 'hl3')}${rd('se', 'Expected error of the mean, σ / √n', 'hl')}</dl>`,
    tryit:
      'Reset, then draw ten samples one at a time and look at the sample mean. Then press Play. Compare the error of the mean you see with the predicted σ / √n. Try very few and very many bins.',
    math: String.raw`<h4>Intuition</h4><p>You never see a distribution, only numbers drawn from it. The histogram is your picture of it, and with more samples the picture sharpens. This is the law of large numbers: averages of samples converge to the true values.</p>
 <h4>The estimates</h4>$$\bar x=\frac1n\sum_{i=1}^n x_i,\qquad s^2=\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2$$
 <p>The \(n-1\) is there because the deviations are measured from \(\bar x\), which was itself fitted to the same data and so sits a little closer to the points than the true mean does. Dividing by \(n-1\) corrects that.</p>
 <h4>How fast it converges</h4><p>The sample mean is itself a random number. Across repeated experiments it is distributed as</p>$$\bar x\sim N\!\left(\mu,\ \frac{\sigma^2}{n}\right),$$<p>so its typical error is \(\sigma/\sqrt n\). To halve the error you need four times the data. That square root shows up everywhere in machine learning, from error bars to the noise in stochastic gradient descent.</p>
 <h4>Bins</h4><p>Each bar is a count divided by \(n\) and by the bin width. Narrow bins follow the shape closely but each holds few samples and is noisy. Wide bins are steady but blur the shape. That is a first look at the bias–variance trade-off.</p>`,
    q: 'Why do the bars jump around so much when n is small?',
    a: 'Each bar is a count divided by n. With few samples one extra point changes a bar a lot. The noise in each bar shrinks like 1 / √n.',
    init(root) {
      const c = $('#sm-c', root),
        xr = [-6, 6],
        S = { mu: 0, sig: 1, nb: 48 };
      let cnt, n, s1, s2, parts, bw;
      const reset = () => {
        bw = 12 / S.nb;
        cnt = new Array(S.nb).fill(0);
        n = 0;
        s1 = 0;
        s2 = 0;
        parts = [];
      };
      reset();
      const land = (x) => {
        n++;
        s1 += x;
        s2 += x * x;
        const b = Math.floor((x - xr[0]) / bw);
        if (b >= 0 && b < S.nb) cnt[b]++;
      };
      const smp = () => S.mu + S.sig * randn(),
        sp = bind(
          root,
          'sm-speed',
          () => {},
          (v) => String(v),
        );
      const draw = () => {
        const t = T(),
          { g, w, h } = fit(c),
          pk = npdf(0, 0, S.sig),
          P = plot(g, w, h, xr, [0, Math.max(0.5, pk * 1.5)], t, {
            yticks: [0.2, 0.4, 0.6, 0.8, 1],
            xlabel: 'x',
          });
        clipPlot(g, P);
        for (let b = 0; b < S.nb; b++) {
          if (!cnt[b]) continue;
          const d = cnt[b] / (n * bw),
            x0 = P.X(xr[0] + b * bw),
            x1 = P.X(xr[0] + (b + 1) * bw);
          g.fillStyle = ramp(t, clamp(0.25 + (0.75 * d) / pk, 0, 1), 0.9);
          g.fillRect(x0 + 0.5, P.Y(d), x1 - x0 - 1, P.Y(0) - P.Y(d));
        }
        curve(g, P, (x) => npdf(x, S.mu, S.sig), xr[0], xr[1]);
        g.strokeStyle = t.ink;
        g.lineWidth = 2.2;
        g.stroke();
        g.fillStyle = t.gold;
        for (const p of parts) {
          g.beginPath();
          g.arc(P.X(p.x), P.m.t + (P.Y(0) - P.m.t) * p.y * p.y, 4, 0, 7);
          g.fill();
        }
        g.restore();
        const m = n ? s1 / n : 0;
        setR(root, 'n', String(n));
        setR(root, 'm', n ? f3(m) : '–');
        setR(root, 's', n > 1 ? f3(Math.sqrt(Math.max(0, (s2 - n * m * m) / (n - 1)))) : '–');
        setR(root, 'se', n ? f3(S.sig / Math.sqrt(n)) : '–');
      };
      const anim = Anim(() => {
        const r = +sp.el.value;
        if (r <= 8) for (let i = 0; i < r; i++) parts.push({ x: smp(), y: 0 });
        else for (let i = 0; i < r; i++) land(smp());
        parts = parts.filter((p) => {
          p.y += 0.05;
          if (p.y >= 1) {
            land(p.x);
            return false;
          }
          return true;
        });
        draw();
      });
      const pb = playBtn($('#sm-play', root), anim);
      bind(
        root,
        'sm-mu',
        (v) => {
          S.mu = v;
          reset();
          draw();
        },
        (v) => (+v).toFixed(1),
      );
      bind(root, 'sm-sig', (v) => {
        S.sig = v;
        reset();
        draw();
      });
      bind(
        root,
        'sm-bins',
        (v) => {
          S.nb = v;
          reset();
          draw();
        },
        (v) => String(v),
      );
      $('#sm-step', root).onclick = () => {
        land(smp());
        draw();
      };
      $('#sm-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      return { draw, stop: pb.stop, auto: pb.start };
    },
  },

  {
    id: 'clt',
    tab: 'Central limit theorem',
    title: 'Why the bell curve is everywhere',
    lede: 'Start from something that is not bell-shaped at all. Average a few draws, and repeat. The averages pile up into a bell curve.',
    stage: `<div class="stage one"><figure class="fig"><canvas class="wide" id="clt-c" role="img" aria-label="Histogram of sample averages approaching a bell curve"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="clt-play" type="button">Play</button><button class="btn" id="clt-reset" type="button">Reset</button></div>${slider('clt-speed', 'Averages per frame', 1, 100, 1, 10)}`)}
 ${grp('The experiment', `<label class="ctl" for="clt-src"><span>Each draw comes from</span><select id="clt-src"><option value="die">Rolling a die</option><option value="exp">Waiting times (skewed)</option><option value="two">Two separate bumps</option><option value="uni">A flat range</option></select></label>${slider('clt-n', 'Draws averaged, n', 0, 6, 1, 1)}`)}</div>
 <dl class="read">${rd('N', 'Averages collected')}${rd('sd', 'Spread of the averages', 'hl3')}${rd('pred', 'Predicted σ / √n', 'hl2')}</dl>`,
    tryit:
      'Set n = 1: you just see the starting shape. Raise n to 2, then 5, then 30. At which n does the skewed one start to look symmetric? Which starting shape needs the most draws?',
    math: String.raw`<h4>Intuition</h4><p>When you add independent random numbers, extreme totals need every term to be extreme in the same direction, which is rare. Middling totals can happen in a great many ways. Adding smooths out whatever shape you started with, and what is left after enough smoothing is always the same curve.</p>
 <h4>The theorem</h4><p>Let \(X_1,\dots,X_n\) be independent draws from any distribution with mean \(\mu\) and finite variance \(\sigma^2\), and let \(\bar X_n\) be their average. Then</p>$$\frac{\sqrt n\,(\bar X_n-\mu)}{\sigma}\ \xrightarrow{\ d\ }\ N(0,1)\qquad\text{so}\qquad \bar X_n\approx N\!\left(\mu,\frac{\sigma^2}{n}\right).$$
 <h4>Why the Gaussian and not something else</h4><p>The density of a sum is the convolution of the densities. The Gaussian is the shape that convolution leaves unchanged: a Gaussian plus a Gaussian is a Gaussian. Every other finite-variance shape is pulled toward it. The mean and variance survive the averaging; all other details of the starting shape are washed out.</p>
 <h4>How fast</h4><p>The remaining error shrinks like \(1/\sqrt n\) and is proportional to how lopsided the start was (the Berry–Esseen bound). Symmetric starts look Gaussian by \(n\approx 5\); the skewed one needs closer to 30.</p>
 <h4>Why it matters</h4><p>Measurement noise, the average loss over a mini-batch and the sum of many small effects in a real system are all sums. That is the usual justification for assuming Gaussian noise.</p>`,
    q: 'The averages get narrower as n grows. By how much when n goes from 5 to 20?',
    a: 'The spread is σ / √n. Going from 5 to 20 multiplies n by 4, so the spread halves.',
    init(root) {
      const c = $('#clt-c', root),
        NS = [1, 2, 3, 5, 10, 30, 100];
      const SRC = {
        die: {
          mu: 3.5,
          sd: Math.sqrt(35 / 12),
          xr: [1, 6],
          disc: true,
          draw: () => 1 + Math.floor(Math.random() * 6),
        },
        exp: { mu: 1, sd: 1, xr: [0, 4], draw: () => -Math.log(1 - Math.random()) },
        two: {
          mu: 0,
          sd: Math.sqrt(4.25),
          xr: [-4, 4],
          draw: () => (Math.random() < 0.5 ? -2 : 2) + 0.5 * randn(),
        },
        uni: { mu: 0, sd: Math.sqrt(3), xr: [-3, 3], draw: () => Math.random() * 6 - 3 },
      };
      const S = { src: 'die', n: 2 };
      let cnt, N, s1, s2, lo, bw, nb;
      const reset = () => {
        const sr = SRC[S.src];
        if (sr.disc && S.n <= 30) {
          bw = 1 / S.n;
          lo = 1 - bw / 2;
          nb = 5 * S.n + 1;
        } else {
          nb = 80;
          lo = sr.xr[0];
          bw = (sr.xr[1] - sr.xr[0]) / nb;
        }
        cnt = new Array(nb).fill(0);
        N = 0;
        s1 = 0;
        s2 = 0;
      };
      reset();
      const add = () => {
        const sr = SRC[S.src];
        let a = 0;
        for (let i = 0; i < S.n; i++) a += sr.draw();
        a /= S.n;
        N++;
        s1 += a;
        s2 += a * a;
        const b = Math.floor((a - lo) / bw);
        if (b >= 0 && b < nb) cnt[b]++;
      };
      const draw = () => {
        const t = T(),
          { g, w, h } = fit(c),
          sr = SRC[S.src],
          sd = sr.sd / Math.sqrt(S.n),
          gp = npdf(0, 0, sd);
        let ym = 1.35 * gp;
        if (N > 300) {
          let mx = 0;
          for (const k of cnt) mx = Math.max(mx, k / (N * bw));
          ym = Math.max(ym, mx * 1.1);
        }
        const xr = [sr.xr[0] - 0.5, sr.xr[1] + 0.5],
          P = plot(g, w, h, xr, [0, ym], t, { xlabel: 'average of n draws' });
        clipPlot(g, P);
        for (let b = 0; b < nb; b++) {
          if (!cnt[b]) continue;
          const d = cnt[b] / (N * bw),
            x0 = P.X(lo + b * bw),
            x1 = P.X(lo + (b + 1) * bw);
          g.fillStyle = ramp(t, clamp(0.25 + (0.75 * d) / gp, 0, 1), 0.9);
          g.fillRect(x0 + 0.5, P.Y(d), Math.max(1, x1 - x0 - 1), P.Y(0) - P.Y(d));
        }
        g.setLineDash([7, 5]);
        curve(g, P, (x) => npdf(x, sr.mu, sd), xr[0], xr[1]);
        g.strokeStyle = t.ok;
        g.lineWidth = 2.4;
        g.stroke();
        g.setLineDash([]);
        g.restore();
        const iw = 128,
          ih = 58,
          ix = P.m.l + 10,
          iy = P.m.t + 2;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('one draw looks like', ix, iy + 9);
        g.fillStyle = mix(t.ink, t.ink, 0, 0.55);
        if (S.src === 'die') for (let k = 0; k < 6; k++) g.fillRect(ix + k * 18, iy + 26, 13, 26);
        else {
          g.beginPath();
          g.moveTo(ix, iy + ih - 6);
          for (let k = 0; k <= 60; k++) {
            const u = k / 60,
              v =
                S.src === 'exp'
                  ? Math.exp(-3.2 * u)
                  : S.src === 'uni'
                    ? 0.7
                    : Math.exp(-((u - 0.25) ** 2) / 0.008) + Math.exp(-((u - 0.75) ** 2) / 0.008);
            g.lineTo(ix + u * (iw - 16), iy + ih - 6 - v * 34);
          }
          g.lineTo(ix + iw - 16, iy + ih - 6);
          g.closePath();
          g.fill();
        }
        const m = N ? s1 / N : 0;
        setR(root, 'N', String(N));
        setR(root, 'sd', N > 1 ? f3(Math.sqrt(Math.max(0, s2 / N - m * m))) : '–');
        setR(root, 'pred', f3(sd));
      };
      const sp = bind(
        root,
        'clt-speed',
        () => {},
        (v) => String(v),
      );
      const anim = Anim(() => {
        const r = +sp.el.value;
        for (let i = 0; i < r; i++) add();
        draw();
      });
      const pb = playBtn($('#clt-play', root), anim);
      bind(
        root,
        'clt-n',
        (v) => {
          S.n = NS[v];
          reset();
          draw();
        },
        (v) => String(NS[v]),
      );
      $('#clt-src', root).onchange = (e) => {
        S.src = e.target.value;
        reset();
        draw();
      };
      $('#clt-reset', root).onclick = () => {
        reset();
        draw();
      };
      return { draw, stop: pb.stop, auto: pb.start };
    },
  },

  {
    id: 'hill',
    tab: 'Two variables: the hill',
    title: 'Two variables make a hill',
    lede: 'With two variables the bell curve becomes a hill. Rotate it, reshape it, then cut it with a flat plane: every cut is an ellipse, and the ellipses stacked on a flat map are the contour plot.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The hill, cut at one height</figcaption><canvas class="d3" id="hill-c3"></canvas>${viewbar}</figure><figure class="fig"><figcaption>Seen from above</figcaption><canvas class="sq" id="hill-c2" role="img" aria-label="Contour map of the hill"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="hill-play" type="button">Lower the cut</button><button class="btn" id="hill-clear" type="button">Clear traces</button></div>`)}
 ${grp('The cut', slider('hill-h', 'Height, as a share of the peak', 0.03, 0.97, 0.01, 0.9))}${grp('Shape of the hill', shapeHTML('hill', G))}</div>
 <dl class="read">${rd('h', 'Density at the cut', 'hl2')}${rd('r', 'Contour radius, in σ units')}${rd('in', 'Probability inside the contour', 'hl')}</dl>`,
    tryit:
      'Watch each ellipse drop onto the map as the cut comes down. Drag the hill until you look straight down on it. Then set ρ to 0 and make σₓ equal σᵧ: what shape are the cuts now?',
    math: String.raw`<h4>Intuition</h4><p>In one dimension, the height of the curve depends on how far you are from the mean, measured in units of \(\sigma\). In two dimensions it is the same idea, except that "how far" now depends on direction: one step along the long side of the hill costs less height than one step across it.</p>
 <h4>The formula</h4>$$p(\mathbf x)=\frac{1}{2\pi\sqrt{|\Sigma|}}\exp\!\left(-\tfrac12\,(\mathbf x-\boldsymbol\mu)^{\!\top}\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\right)$$
 <p>The quantity \(d^2=(\mathbf x-\boldsymbol\mu)^{\top}\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\) plays the role of \(z^2\). It is a squared distance that has been corrected for the shape of the hill.</p>
 <h4>Why every cut is an ellipse</h4><p>The height depends on \(\mathbf x\) only through \(d\). A flat cut fixes the height, so it fixes \(d\), and the set of points with \(d\) constant is \(\{\mathbf x:(\mathbf x-\boldsymbol\mu)^{\top}\Sigma^{-1}(\mathbf x-\boldsymbol\mu)=d^2\}\): an ellipse. A cut at a fraction \(h\) of the peak has</p>$$e^{-d^2/2}=h\quad\Longrightarrow\quad d=\sqrt{-2\ln h}.$$
 <h4>How much probability is inside</h4><p>In two dimensions \(d^2\) follows a chi-squared distribution with 2 degrees of freedom, whose distribution function is \(1-e^{-d^2/2}\). So the probability inside the contour is</p>$$P(\text{inside})=1-e^{-d^2/2}=1-h.$$<p>A cut at 10% of the peak encloses 90% of the probability. The 1σ ellipse holds only 39%, not 68%: in two dimensions there is more room far from the centre.</p>`,
    q: 'Where on the contour map is the hill steepest?',
    a: 'Where the contour lines are closest together. For equally spaced heights that is along the short axis of the ellipses, at about one σ from the centre.',
    init(root) {
      const c3 = $('#hill-c3', root),
        c2 = $('#hill-c2', root),
        S = { h: 0.9, trace: [] };
      let draw,
        wait = 0;
      const hill = Hill(c3, () => draw(), $('.viewbar', root));
      draw = () => {
        const t = T(),
          r = Math.sqrt(-2 * Math.log(S.h));
        hill.draw(t, G, { h: S.h });
        drawMap(c2, t, G, { rings: S.trace, ring: r });
        const pk = 1 / (2 * Math.PI * G.sx * G.sy * Math.sqrt(1 - G.rho * G.rho));
        setR(root, 'h', f3(S.h * pk));
        setR(root, 'r', f2(r));
        setR(root, 'in', (100 * (1 - S.h)).toFixed(0) + '%');
      };
      const hs = bind(root, 'hill-h', (v) => {
          S.h = v;
          draw();
        }),
        sh = shapeBind(root, 'hill', G, () => draw()),
        LV = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.05];
      const anim = Anim(() => {
        if (wait > 0) {
          wait--;
          if (!wait) {
            S.h = 0.97;
            S.trace = [];
            hs.set(S.h);
            draw();
          }
          return;
        }
        const nh = S.h - 0.0035;
        LV.forEach((l) => {
          if (S.h > l && nh <= l) S.trace.push(Math.sqrt(-2 * Math.log(l)));
        });
        if (nh < 0.035) {
          S.h = 0.035;
          wait = 110;
        } else S.h = nh;
        hs.set(S.h);
        draw();
      });
      const pb = playBtn($('#hill-play', root), anim);
      grab(hs.el, pb);
      $('#hill-clear', root).onclick = () => {
        S.trace = [];
        draw();
      };
      return {
        draw,
        stop() {
          pb.stop();
          hill.spin(false);
        },
        auto() {
          pb.start();
          hill.spin(true);
        },
        sync: sh.sync,
      };
    },
  },

  {
    id: 'covariance',
    tab: 'Covariance matrix',
    title: 'The covariance matrix shapes everything',
    lede: 'Type your own matrix, or pick a shape, and watch the cloud, the contours and the hill morph to match. Isotropic gives circles. Diagonal stretches along an axis. Off-diagonal entries tilt it.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Samples, contours and principal axes</figcaption><canvas class="sq" id="cov-c2" role="img" aria-label="Scatter of samples with contour ellipses and axes"></canvas></figure><figure class="fig"><figcaption>The hill</figcaption><canvas class="d3" id="cov-c3"></canvas>${viewbar}</figure></div>
 <div class="controls">${grp('Type the matrix', `<div class="mx"><span class="mx-l">Σ =</span><div class="mx-grid"><input type="number" id="cov-a" step="0.1" min="0.2" max="4" aria-label="Variance of x"><input type="number" id="cov-b" step="0.1" min="-4" max="4" aria-label="Covariance of x and y"><input type="number" id="cov-b2" disabled aria-label="Covariance, mirrored"><input type="number" id="cov-d" step="0.1" min="0.2" max="4" aria-label="Variance of y"></div></div><p class="status" id="cov-status" role="status"></p>`)}
 ${grp('Or pick a shape', `<div class="row" id="cov-presets"><button class="btn" type="button" data-p="1,0,1">Isotropic</button><button class="btn" type="button" data-p="2.4,0,2.4">Isotropic, wide</button><button class="btn" type="button" data-p="3,0,0.5">Diagonal</button><button class="btn" type="button" data-p="1.8,1.1,1.2">Positive tilt</button><button class="btn" type="button" data-p="1.8,-1.1,1.2">Negative tilt</button><button class="btn" type="button" data-p="2,1.9,2">Nearly a line</button></div><div class="row"><button class="btn go" id="cov-play" type="button">Tour the shapes</button></div>`)}
 ${grp('Or use sliders', shapeHTML('cov', G))}</div>
 <dl class="read">${rd('shape', 'Shape')}${rd('l1', 'Long axis variance λ₁', 'hl2')}${rd('l2', 'Short axis variance λ₂')}${rd('th', 'Tilt of the long axis')}${rd('det', 'Determinant |Σ|', 'hl3')}</dl>`,
    tryit:
      'Type 1, 0, 1 for an isotropic Gaussian. Change only the top-left entry to 3. Then put 0.9 in the off-diagonal. Last, try an off-diagonal of 2 with ones on the diagonal and read the message.',
    math: String.raw`<h4>Intuition</h4><p>The covariance matrix answers two questions at once. The diagonal says how much each variable varies on its own. The off-diagonal says whether they vary together: positive when a large \(x\) tends to come with a large \(y\), negative when it comes with a small one, zero when knowing one tells you nothing about the other.</p>
 <h4>The matrix</h4>$$\Sigma=\begin{pmatrix}\sigma_x^2 & \rho\,\sigma_x\sigma_y\\[2pt] \rho\,\sigma_x\sigma_y & \sigma_y^2\end{pmatrix},\qquad \Sigma_{12}=\mathbb{E}\big[(x-\mu_x)(y-\mu_y)\big],\qquad \rho=\frac{\Sigma_{12}}{\sigma_x\sigma_y}\in[-1,1].$$
 <h4>Three families</h4><p><b>Isotropic</b>, \(\Sigma=\sigma^2 I\): the same spread in every direction, circular contours, one number to learn. <b>Diagonal</b>, \(\Sigma=\operatorname{diag}(\sigma_x^2,\sigma_y^2)\): independent variables, ellipses lined up with the axes, one number per dimension. <b>Full</b>: any tilt, \(d(d+1)/2\) numbers in \(d\) dimensions. Models choose between these all the time: naive Bayes assumes diagonal, a standard VAE and diffusion noise are isotropic.</p>
 <h4>What makes a matrix valid</h4><p>The variance in any direction \(\mathbf u\) is \(\mathbf u^{\top}\Sigma\,\mathbf u\), and a variance cannot be negative. So \(\Sigma\) must be symmetric and positive definite, which in two dimensions means \(\Sigma_{11}>0\), \(\Sigma_{22}>0\) and \(\Sigma_{12}^2<\Sigma_{11}\Sigma_{22}\). That last condition is the same as \(|\rho|<1\).</p>
 <h4>Axes of the ellipse</h4>$$\Sigma=Q\,\Lambda\,Q^{\top},\qquad \Lambda=\operatorname{diag}(\lambda_1,\lambda_2).$$<p>The columns of \(Q\) (the eigenvectors) are the directions of the ellipse axes, and the axis lengths are proportional to \(\sqrt{\lambda_1}\) and \(\sqrt{\lambda_2}\). Finding these axes from data is exactly PCA. The determinant \(|\Sigma|=\lambda_1\lambda_2\) measures the area of the ellipse, \(\pi\sqrt{|\Sigma|}\) for the 1σ contour, and sets the peak height \(1/(2\pi\sqrt{|\Sigma|})\).</p>
 <h4>How the samples are made</h4><p>Draw \(\mathbf z\sim N(\mathbf 0,I)\) and set \(\mathbf x=\boldsymbol\mu+L\mathbf z\) with \(LL^{\top}=\Sigma\). The dots in the picture are the same \(\mathbf z\) every time, pushed through a different \(L\). This "reparameterisation" is what lets a VAE learn through a sampling step.</p>`,
    q: 'Two matrices have the same determinant. Do they give the same shape?',
    a: 'No. The determinant fixes the area of the ellipse, not its shape. A circle and a long thin ellipse can have the same area. Compare 1, 0, 1 with 3, 0, 0.33.',
    init(root) {
      const c3 = $('#cov-c3', root),
        c2 = $('#cov-c2', root),
        st = $('#cov-status', root),
        ia = $('#cov-a', root),
        ib = $('#cov-b', root),
        ib2 = $('#cov-b2', root),
        id = $('#cov-d', root);
      let draw,
        tw = null,
        pb;
      const hill = Hill(c3, () => draw(), $('.viewbar', root));
      const fields = () => {
        const e = eig(G),
          ae = document.activeElement;
        if (ae !== ia) ia.value = f2(e.a);
        if (ae !== ib) ib.value = f2(e.b);
        ib2.value = f2(e.b);
        if (ae !== id) id.value = f2(e.d);
      };
      const sh = shapeBind(root, 'cov', G, () => {
          fields();
          draw();
        }),
        syncUI = () => {
          sh.sync();
          fields();
        };
      draw = () => {
        const t = T(),
          e = eig(G);
        drawMap(c2, t, G, { pts: Z250, eig: true });
        hill.draw(t, G, {});
        setR(
          root,
          'shape',
          Math.abs(G.rho) < 0.03
            ? Math.abs(G.sx - G.sy) < 0.04
              ? 'Isotropic (circles)'
              : 'Diagonal (axis-aligned)'
            : 'Full (tilted)',
        );
        setR(root, 'l1', f3(e.l1));
        setR(root, 'l2', f3(e.l2));
        setR(root, 'th', ((e.th * 180) / Math.PI).toFixed(0) + '°');
        setR(root, 'det', f3(e.l1 * e.l2));
      };
      const tweenTo = (tg) => {
        const s = { ...G };
        let k = 0;
        if (tw) tw.stop();
        tw = Anim(() => {
          k++;
          const u = Math.min(1, k / 40),
            e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u);
          G.sx = s.sx + (tg.sx - s.sx) * e;
          G.sy = s.sy + (tg.sy - s.sy) * e;
          G.rho = s.rho + (tg.rho - s.rho) * e;
          syncUI();
          draw();
          if (u >= 1) tw.stop();
        });
        tw.start();
      };
      const fromM = (a, b, d) => {
        if (!(isFinite(a) && isFinite(b) && isFinite(d)))
          return 'Fill in all three entries with numbers.';
        if (a < 0.2 || d < 0.2 || a > 4 || d > 4)
          return 'The two diagonal entries are variances. Keep each between 0.2 and 4.';
        const lim = Math.sqrt(a * d);
        if (Math.abs(b) >= lim)
          return (
            'Not a valid covariance matrix: the off-diagonal entry must be smaller in size than √(Σ₁₁·Σ₂₂) = ' +
            f2(lim) +
            '. Otherwise one direction would have negative variance.'
          );
        if (Math.abs(b) > 0.97 * lim)
          return (
            'That is valid but too close to a flat line to draw. Keep the off-diagonal entry under ' +
            f2(0.97 * lim) +
            ' in size.'
          );
        return { sx: Math.sqrt(a), sy: Math.sqrt(d), rho: b / lim };
      };
      const typed = () => {
        pb.stop();
        const r = fromM(parseFloat(ia.value), parseFloat(ib.value), parseFloat(id.value));
        if (typeof r === 'string') {
          st.textContent = r;
          return;
        }
        st.textContent = '';
        tweenTo(r);
      };
      [ia, ib, id].forEach((i) => i.addEventListener('change', typed));
      const PRE = [...root.querySelectorAll('#cov-presets button')].map((b) =>
        b.dataset.p.split(',').map(Number),
      );
      let pi = 0,
        fr = 0;
      const tour = Anim(() => {
        if (fr++ % 130 === 0) {
          const p = PRE[pi++ % PRE.length];
          st.textContent = '';
          tweenTo(fromM(p[0], p[1], p[2]));
        }
      });
      pb = playBtn($('#cov-play', root), tour);
      $('#cov-presets', root).addEventListener('click', (e) => {
        const p = e.target.dataset && e.target.dataset.p;
        if (!p) return;
        pb.stop();
        st.textContent = '';
        const v = p.split(',').map(Number);
        tweenTo(fromM(v[0], v[1], v[2]));
      });
      sh.els.forEach((el) => grab(el, pb));
      return {
        draw,
        stop() {
          pb.stop();
          fr = 0;
          if (tw) tw.stop();
          hill.spin(false);
        },
        auto() {
          fr = 60;
          pb.start();
        },
        sync: syncUI,
      };
    },
  },

  {
    id: 'slices',
    tab: 'Marginals and conditionals',
    title: 'Slices and shadows',
    lede: 'Cut the hill with a vertical wall at x = x₀. The cut face is a bell curve again: what you know about y once x is fixed. The shadows on the back walls are what you know about each variable alone.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The hill, sliced at x = x₀</figcaption><canvas class="d3" id="sl-c3"></canvas>${viewbar}</figure><figure class="fig"><figcaption>y on its own, and y given x₀</figcaption><canvas class="short" id="sl-c1" role="img" aria-label="Conditional and marginal density of y"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="sl-play" type="button">Slide the cut</button></div>`)}
 ${grp('The slice', slider('sl-x', 'Position x₀', -3, 3, 0.05, 1))}${grp('Shape of the hill', shapeHTML('sl', G))}</div>
 <dl class="read">${rd('m', 'Mean of y given x₀', 'hl2')}${rd('s', 'Std of y given x₀', 'hl2')}${rd('ms', 'Std of y alone', 'hl3')}${rd('r2', 'Variance explained, ρ²', 'hl')}</dl>`,
    tryit:
      'Watch the slice as the cut slides. Its centre moves but its width does not. Now push ρ toward 0.95: what happens to the width? What happens at ρ = 0?',
    math: String.raw`<h4>Intuition</h4><p>A <b>marginal</b> answers "what do I know about \(y\) if I ignore \(x\)?" You get it by squashing the hill flat against a wall. A <b>conditional</b> answers "what do I know about \(y\) now that I have seen \(x=x_0\)?" You get it by slicing the hill at \(x_0\) and rescaling the slice so its area is 1. For a Gaussian, both are Gaussian again. Very few distributions have this property, and it is why Gaussians make exact inference possible.</p>
 <h4>The marginal</h4>$$p(y)=\int p(x,y)\,dx=N\big(\mu_y,\ \sigma_y^2\big)$$<p>To marginalise a Gaussian you simply drop the rows and columns you do not care about.</p>
 <h4>The conditional</h4>$$y\mid x_0\ \sim\ N\!\Big(\ \mu_y+\rho\,\frac{\sigma_y}{\sigma_x}\,(x_0-\mu_x)\ ,\ \ \sigma_y^2\,(1-\rho^2)\ \Big)$$
 <p><b>The mean is a straight line in \(x_0\).</b> Its slope \(\rho\,\sigma_y/\sigma_x\) is the slope of the least-squares regression line. Linear regression is this formula.</p>
 <p><b>The variance does not depend on \(x_0\).</b> Wherever you cut, the slice has the same width. And it is smaller than the marginal variance by the factor \(1-\rho^2\): the fraction \(\rho^2\) of the variance of \(y\) has been explained by \(x\). That is the \(R^2\) of regression.</p>
 <h4>In any number of dimensions</h4>$$\boldsymbol\mu_{a\mid b}=\boldsymbol\mu_a+\Sigma_{ab}\Sigma_{bb}^{-1}(\mathbf x_b-\boldsymbol\mu_b),\qquad \Sigma_{a\mid b}=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}.$$<p>Gaussian processes, Kalman filters and the denoising step of a diffusion model all apply these two lines.</p>`,
    q: 'If ρ = 0, what does knowing x tell you about y?',
    a: 'Nothing. The slice has the same centre and width wherever you cut, and it matches the shadow on the wall. For Gaussians, zero correlation means independence.',
    init(root) {
      const c3 = $('#sl-c3', root),
        c1 = $('#sl-c1', root),
        S = { x0: 1 };
      let draw;
      const hill = Hill(c3, () => draw(), $('.viewbar', root), { marg: true });
      draw = () => {
        const t = T();
        hill.draw(t, G, { v: S.x0 });
        const m = ((G.rho * G.sy) / G.sx) * S.x0,
          sd = G.sy * Math.sqrt(1 - G.rho * G.rho);
        const { g, w, h } = fit(c1),
          ym = Math.max(1, npdf(0, 0, sd) * 1.15),
          P = plot(g, w, h, [-4, 4], [0, ym], t, { xlabel: 'y' });
        clipPlot(g, P);
        g.setLineDash([6, 4]);
        curve(g, P, (y) => npdf(y, 0, G.sy), -4, 4);
        g.strokeStyle = t.accent;
        g.lineWidth = 2.4;
        g.stroke();
        g.setLineDash([]);
        area(g, P, (y) => npdf(y, m, sd), -4, 4);
        g.fillStyle = mix(t.ok, t.ok, 0, 0.24);
        g.fill();
        curve(g, P, (y) => npdf(y, m, sd), -4, 4);
        g.strokeStyle = t.ok;
        g.lineWidth = 3;
        g.stroke();
        g.restore();
        g.font = MONO;
        g.textAlign = 'left';
        g.fillStyle = t.ok;
        g.fillText('p(y | x₀)', P.m.l + 8, P.m.t + 10);
        g.fillStyle = t.accent;
        g.fillText('p(y), dashed', P.m.l + 8, P.m.t + 25);
        setR(root, 'm', f2(m));
        setR(root, 's', f3(sd));
        setR(root, 'ms', f3(G.sy));
        setR(root, 'r2', (100 * G.rho * G.rho).toFixed(0) + '%');
      };
      const xs = bind(root, 'sl-x', (v) => {
          S.x0 = v;
          draw();
        }),
        sh = shapeBind(root, 'sl', G, () => draw());
      const pb = playBtn($('#sl-play', root), sweeper(xs.el, -2.6, 2.6, 7));
      grab(xs.el, pb);
      return {
        draw,
        stop() {
          pb.stop();
          hill.spin(false);
        },
        auto: pb.start,
        sync: sh.sync,
      };
    },
  },

  {
    id: 'fitting',
    tab: 'Fitting to data',
    title: 'Fitting a Gaussian to data',
    lede: 'Go the other way: start from data and find the Gaussian. Build your own data set, watch points arrive one at a time, and see the fitted ellipses settle. Held-out test points show whether the fit generalises.',
    stage: `<div class="stage one"><figure class="fig"><canvas class="wide" id="fit-c" role="img" aria-label="Scatter plot with fitted Gaussian ellipses"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="fit-play" type="button">Play</button><button class="btn" id="fit-step" type="button">Add one</button><button class="btn" id="fit-reset" type="button">Reset</button></div>${slider('fit-speed', 'Speed', 1, 20, 1, 3)}`)}
 ${grp('The data', `<label class="ctl" for="fit-ds"><span>Data set</span><select id="fit-ds"><option value="tilted">Tilted cloud</option><option value="round">Round cloud</option><option value="narrow">Narrow, strongly correlated</option><option value="clumps">Two clumps (not Gaussian)</option><option value="custom">Your own settings</option></select></label><div class="row"><button class="btn" id="fit-new" type="button">New random sample</button><label class="btn file">Upload a CSV<input type="file" id="fit-file" accept=".csv,.txt,text/csv"></label></div><p class="status" id="fit-status" role="status"></p>`)}
 ${grp('The true Gaussian', slider('fit-sx', 'Spread of x, σₓ', 0.3, 2.5, 0.05, 1.5) + slider('fit-sy', 'Spread of y, σᵧ', 0.3, 2.5, 0.05, 0.8) + slider('fit-rho', 'Correlation ρ', -0.97, 0.97, 0.01, 0.7) + slider('fit-n', 'Number of points', 20, 500, 10, 220))}</div>
 <dl class="read">${rd('n', 'Training points used')}${rd('mu', 'Estimated mean')}${rd('cov', 'Estimated σₓ, σᵧ, ρ')}${rd('tr', 'Train: average log-likelihood', 'hl2')}${rd('te', 'Test: average log-likelihood', 'hl3')}</dl>`,
    tryit:
      'Compare train and test log-likelihood after 5 points, then after 100. When are they far apart? Set the number of points to 20 and press “New random sample” a few times: how much does the fit change? Then try “Two clumps”.',
    math: String.raw`<h4>Intuition</h4><p>Maximum likelihood asks one question: of all possible Gaussians, which one would have made this data least surprising? It slides and reshapes the hill until the data points sit as high on it as possible, on average.</p>
 <h4>The likelihood</h4><p>For independent points the probability of the whole data set is a product, so its logarithm is a sum:</p>$$\ell(\boldsymbol\mu,\Sigma)=\sum_{i=1}^n\ln p(\mathbf x_i)=-\frac n2\ln\big|2\pi\Sigma\big|-\frac12\sum_{i=1}^n(\mathbf x_i-\boldsymbol\mu)^{\top}\Sigma^{-1}(\mathbf x_i-\boldsymbol\mu).$$
 <p>The second term rewards hills that sit close to the points. The first term penalises hills that are spread out, since a wide hill is low everywhere. The best fit balances the two.</p>
 <h4>The answer</h4><p>Setting the derivatives to zero gives the estimates in closed form:</p>$$\hat{\boldsymbol\mu}=\frac1n\sum_{i=1}^n\mathbf x_i,\qquad \hat\Sigma=\frac1n\sum_{i=1}^n(\mathbf x_i-\hat{\boldsymbol\mu})(\mathbf x_i-\hat{\boldsymbol\mu})^{\top}.$$<p>The sample mean and the sample covariance. No iteration is needed, which is rare; most models in later topics need gradient descent to do this same job.</p>
 <h4>Train against test</h4><p>The fit is tuned to the training points, so they score better under it than fresh points do. With \(n\) points and \(k\) fitted numbers (here \(k=5\)), the gap in average log-likelihood is roughly \(k/n\). It is large for small \(n\), and it closes as data arrives. This gap is overfitting in its simplest form. With fewer than 3 points \(\hat\Sigma\) is singular and the likelihood is infinite, which the next lesson fixes.</p>
 <h4>When the model is wrong</h4><p>Maximum likelihood finds the best Gaussian, not a good model. For two clumps it returns one wide hill whose peak sits in the empty gap.</p><p>A CSV needs at least two numeric columns. The first two are used.</p>`,
    q: 'For “Two clumps”, the fitted peak sits where there is almost no data. What does that tell you?',
    a: 'Maximum likelihood finds the best Gaussian, not a good model. One Gaussian cannot have two peaks. That is the reason for mixtures of Gaussians, which come later.',
    init(root) {
      const c = $('#fit-c', root),
        st = $('#fit-status', root),
        sel = $('#fit-ds', root);
      const DS = {
        tilted: { mu: [1, -0.5], sx: 1.5, sy: 0.8, rho: 0.7 },
        round: { mu: [0, 0], sx: 1, sy: 1, rho: 0 },
        narrow: { mu: [-0.5, 0.5], sx: 1.6, sy: 1.2, rho: -0.92 },
      };
      const Pm = { mu: [1, -0.5], sx: 1.5, sy: 0.8, rho: 0.7, n: 220 };
      let seed = 7,
        train = [],
        test = [],
        shown = 0,
        truth = null,
        acc = 0,
        V = {},
        custom = null,
        pb;
      const mle = (p) => {
        const n = p.length;
        let mx = 0,
          my = 0;
        for (const q of p) {
          mx += q[0];
          my += q[1];
        }
        mx /= n;
        my /= n;
        let a = 0,
          b = 0,
          d = 0;
        for (const q of p) {
          const u = q[0] - mx,
            v = q[1] - my;
          a += u * u;
          b += u * v;
          d += v * v;
        }
        return { mx, my, a: a / n, b: b / n, d: d / n, n };
      };
      const ll = (p, F) => {
        const det = F.a * F.d - F.b * F.b;
        if (!(det > 1e-12) || !p.length) return NaN;
        let s = 0;
        for (const q of p) {
          const u = q[0] - F.mx,
            v = q[1] - F.my;
          s += (F.d * u * u - 2 * F.b * u * v + F.a * v * v) / det;
        }
        return -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - (0.5 * s) / p.length;
      };
      const load = (pts, tr, fixed) => {
        const r = rng(5),
          p = pts.slice();
        for (let i = p.length - 1; i > 0; i--) {
          const j = Math.floor(r() * (i + 1));
          [p[i], p[j]] = [p[j], p[i]];
        }
        const k = Math.round(p.length * 0.7);
        train = p.slice(0, k);
        test = p.slice(k);
        truth = tr;
        if (fixed) V = fixed;
        else {
          let x0 = 1 / 0,
            x1 = -1 / 0,
            y0 = 1 / 0,
            y1 = -1 / 0;
          for (const q of p) {
            x0 = Math.min(x0, q[0]);
            x1 = Math.max(x1, q[0]);
            y0 = Math.min(y0, q[1]);
            y1 = Math.max(y1, q[1]);
          }
          V = {
            cx: (x0 + x1) / 2,
            cy: (y0 + y1) / 2,
            hx: ((x1 - x0) / 2) * 1.15 || 1,
            hy: ((y1 - y0) / 2) * 1.15 || 1,
          };
        }
      };
      const gen = () => {
        const r = rng(seed),
          rn = randnFrom(r),
          p = [];
        if (sel.value === 'clumps') {
          for (let i = 0; i < Pm.n; i++) {
            const k = r() < 0.5;
            p.push([(k ? -2 : 2) + 0.6 * rn(), (k ? -1 : 1.2) + 0.6 * rn()]);
          }
          load(p, null, { cx: 0, cy: 0, hx: 7.5, hy: 3.6 });
        } else if (sel.value === 'csv' && custom) load(custom, null);
        else {
          const q = Math.sqrt(1 - Pm.rho * Pm.rho);
          for (let i = 0; i < Pm.n; i++) {
            const a = rn(),
              b = rn();
            p.push([Pm.mu[0] + Pm.sx * a, Pm.mu[1] + Pm.sy * (Pm.rho * a + q * b)]);
          }
          load(
            p,
            {
              mx: Pm.mu[0],
              my: Pm.mu[1],
              a: Pm.sx * Pm.sx,
              b: Pm.rho * Pm.sx * Pm.sy,
              d: Pm.sy * Pm.sy,
            },
            { cx: Pm.mu[0], cy: Pm.mu[1], hx: 7.5, hy: 3.6 },
          );
        }
      };
      gen();
      const draw = () => {
        const t = T(),
          { g, w, h } = fit(c),
          m = { l: 44, r: 14, t: 12, b: 26 };
        const sc = Math.min((w - m.l - m.r) / (2 * V.hx), (h - m.t - m.b) / (2 * V.hy)),
          ox = m.l + (w - m.l - m.r) / 2,
          oy = m.t + (h - m.t - m.b) / 2;
        const X = (x) => ox + (x - V.cx) * sc,
          Y = (y) => oy - (y - V.cy) * sc;
        g.font = MONO;
        g.fillStyle = t.muted;
        g.textAlign = 'center';
        const hx = (w - m.l - m.r) / 2 / sc,
          hy = (h - m.t - m.b) / 2 / sc;
        g.strokeStyle = t.line;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(m.l, h - m.b);
        g.lineTo(w - m.r, h - m.b);
        g.moveTo(m.l, m.t);
        g.lineTo(m.l, h - m.b);
        g.stroke();
        [-1, 0, 1].forEach((k) => {
          g.fillText((V.cx + k * hx * 0.9).toPrecision(3), X(V.cx + k * hx * 0.9), h - m.b + 16);
        });
        g.textAlign = 'right';
        [-1, 0, 1].forEach((k) => {
          g.fillText((V.cy + k * hy * 0.85).toPrecision(3), m.l - 5, Y(V.cy + k * hy * 0.85) + 4);
        });
        g.save();
        g.beginPath();
        g.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
        g.clip();
        g.strokeStyle = t.muted;
        g.lineWidth = 1.2;
        for (const q of test) {
          g.beginPath();
          g.arc(X(q[0]), Y(q[1]), 3.2, 0, 7);
          g.stroke();
        }
        g.fillStyle = mix(t.accent, t.accent, 0, 0.85);
        for (let i = 0; i < shown; i++) {
          g.beginPath();
          g.arc(
            X(train[i][0]),
            Y(train[i][1]),
            i === shown - 1 && shown < train.length ? 6 : 3.4,
            0,
            7,
          );
          g.fill();
        }
        if (truth) {
          g.setLineDash([6, 4]);
          g.strokeStyle = t.ink;
          g.lineWidth = 1.6;
          ellPath(g, X, Y, truth.mx, truth.my, truth.a, truth.b, truth.d, 2);
          g.stroke();
          g.setLineDash([]);
        }
        let F = null;
        if (shown >= 3) {
          F = mle(train.slice(0, shown));
          if (F.a * F.d - F.b * F.b > 1e-12) {
            g.strokeStyle = t.ok;
            g.lineWidth = 2.8;
            ellPath(g, X, Y, F.mx, F.my, F.a, F.b, F.d, 1);
            g.stroke();
            ellPath(g, X, Y, F.mx, F.my, F.a, F.b, F.d, 2);
            g.stroke();
            g.fillStyle = mix(t.ok, t.ok, 0, 0.07);
            g.fill();
            g.beginPath();
            g.moveTo(X(F.mx) - 7, Y(F.my));
            g.lineTo(X(F.mx) + 7, Y(F.my));
            g.moveTo(X(F.mx), Y(F.my) - 7);
            g.lineTo(X(F.mx), Y(F.my) + 7);
            g.stroke();
          } else F = null;
        }
        g.restore();
        g.textAlign = 'left';
        g.fillStyle = t.ok;
        g.fillText('fit: 1σ and 2σ', m.l + 10, m.t + 12);
        if (truth) {
          g.fillStyle = t.ink;
          g.fillText('truth: 2σ, dashed', m.l + 10, m.t + 27);
        }
        g.fillStyle = t.muted;
        g.fillText('hollow = test', m.l + 10, m.t + (truth ? 42 : 27));
        setR(root, 'n', shown + ' of ' + train.length);
        if (F) {
          const sx = Math.sqrt(F.a),
            sy = Math.sqrt(F.d);
          setR(root, 'mu', '(' + F.mx.toPrecision(3) + ', ' + F.my.toPrecision(3) + ')');
          setR(
            root,
            'cov',
            sx.toPrecision(3) + ', ' + sy.toPrecision(3) + ', ' + f2(F.b / (sx * sy)),
          );
          setR(root, 'tr', f3(ll(train.slice(0, shown), F)));
          setR(root, 'te', f3(ll(test, F)));
        } else ['mu', 'cov', 'tr', 'te'].forEach((k) => setR(root, k, '–'));
      };
      const sp = bind(
        root,
        'fit-speed',
        () => {},
        (v) => String(v),
      );
      const anim = Anim(() => {
        acc += +sp.el.value / 12;
        const k = Math.floor(acc);
        acc -= k;
        shown = Math.min(train.length, shown + k);
        draw();
        if (shown >= train.length) pb.stop();
      });
      pb = playBtn($('#fit-play', root), anim);
      $('#fit-play', root).addEventListener('click', () => {
        if (anim.on && shown >= train.length) {
          shown = 0;
          acc = 0;
        }
      });
      $('#fit-step', root).onclick = () => {
        shown = Math.min(train.length, shown + 1);
        draw();
      };
      $('#fit-reset', root).onclick = () => {
        pb.stop();
        shown = 0;
        acc = 0;
        draw();
      };
      const sx = bind(root, 'fit-sx', (v) => {
          Pm.sx = v;
          live();
        }),
        sy = bind(root, 'fit-sy', (v) => {
          Pm.sy = v;
          live();
        }),
        sr = bind(root, 'fit-rho', (v) => {
          Pm.rho = v;
          live();
        }),
        sn = bind(
          root,
          'fit-n',
          (v) => {
            Pm.n = v;
            live(true);
          },
          (v) => String(v),
        );
      function live(isN) {
        pb.stop();
        if (!(isN && sel.value === 'clumps') && sel.value !== 'custom') {
          sel.value = 'custom';
          Pm.mu = [0, 0];
        }
        gen();
        shown = train.length;
        draw();
      }
      sel.onchange = () => {
        st.textContent = '';
        const d = DS[sel.value];
        if (d) {
          Object.assign(Pm, { mu: d.mu.slice(), sx: d.sx, sy: d.sy, rho: d.rho });
          sx.set(Pm.sx);
          sy.set(Pm.sy);
          sr.set(Pm.rho);
        }
        if (sel.value === 'custom') Pm.mu = [0, 0];
        if (sel.value === 'csv' && !custom) sel.value = 'custom';
        gen();
        shown = 0;
        acc = 0;
        draw();
        pb.start();
      };
      $('#fit-new', root).onclick = () => {
        seed++;
        gen();
        shown = 0;
        acc = 0;
        draw();
        pb.start();
      };
      $('#fit-file', root).onchange = (e) => {
        const f = e.target.files[0];
        if (!f) return;
        const fr = new FileReader();
        fr.onerror = () => {
          st.textContent = 'That file could not be read. Try saving it again as plain CSV.';
        };
        fr.onload = () => {
          const p = parseXY(fr.result, 5000);
          if (typeof p === 'string') {
            st.textContent = p;
            return;
          }
          custom = p;
          let o = $('option[value="csv"]', sel);
          if (!o) {
            o = document.createElement('option');
            o.value = 'csv';
            sel.appendChild(o);
          }
          o.textContent = 'Your CSV: ' + f.name.slice(0, 24);
          sel.value = 'csv';
          st.textContent = '';
          gen();
          shown = 0;
          acc = 0;
          draw();
          pb.start();
        };
        fr.readAsText(f);
        e.target.value = '';
      };
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'bayes',
    tab: 'MLE, MAP and Bayes',
    title: 'Three ways to estimate a Gaussian',
    lede: 'MLE trusts only the data. MAP starts from a prior guess and lets the data pull it away. The Bayesian view keeps every Gaussian that is still plausible. Look at them one at a time, then together.',
    stage: `<div class="row seg" id="by-mode" role="group" aria-label="What to show"><button class="btn" type="button" data-m="mle">MLE</button><button class="btn" type="button" data-m="map">MAP</button><button class="btn" type="button" data-m="bayes">Bayesian</button><button class="btn" type="button" data-m="all">All three</button></div>
 <p class="cap" id="by-cap" role="status"></p>
 <div class="stage"><figure class="fig"><figcaption>The data and the fitted Gaussians (2σ outlines)</figcaption><canvas class="short" id="by-c" style="touch-action:none;cursor:move" role="img" aria-label="Data points with prior, MLE, MAP and Bayesian estimates"></canvas></figure>
 <figure class="fig"><figcaption>Belief about the mean, along x</figcaption><canvas class="short" id="by-c2" role="img" aria-label="Prior, data and posterior curves for the mean"></canvas></figure></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="by-play" type="button">Play</button><button class="btn" id="by-step" type="button">Add one point</button><button class="btn" id="by-reset" type="button">Reset</button></div>`)}
 ${grp('The prior (drag its mean on the picture)', slider('by-k', 'Trust in the prior mean, κ₀', 0.1, 40, 0.1, 6) + slider('by-v', 'Trust in the prior shape, ν₀', 4, 60, 1, 10) + slider('by-s', 'Prior guess of the spread', 0.4, 2.2, 0.05, 1.3))}
 ${grp('The data', slider('by-ds', 'True spread', 0.4, 1.6, 0.05, 1) + slider('by-dr', 'True correlation ρ', -0.9, 0.9, 0.01, 0.6) + `<div class="row"><button class="btn" id="by-new" type="button">New random data</button></div>`)}</div>
 <dl class="read">${rd('n', 'Data points')}${rd('mle', 'MLE mean', 'hl2')}${rd('map', 'MAP mean', 'hl')}${rd('w', 'Share of the vote held by the data')}${rd('pm', 'Prior mean')}</dl>`,
    tryit:
      'Reset and stay on MAP. Add points one at a time and watch the gold dot slide along the dotted line from the prior toward the data. Then drag the prior far away and raise κ₀: how many points until the data wins? Switch to Bayesian and watch the bundle tighten.',
    math: String.raw`<h4>Intuition</h4><p>Three attitudes to the same data. <b>MLE</b>: believe only what you have seen. <b>MAP</b>: start with a guess, and move away from it only as far as the evidence pushes you. <b>Bayesian</b>: never commit to a single answer; keep every Gaussian that could plausibly have produced the data, weighted by how plausible it is. With little data these differ a lot. With plenty they agree.</p>
 <h4>Bayes' rule for parameters</h4>$$\underbrace{p(\boldsymbol\mu,\Sigma\mid\mathcal D)}_{\text{posterior}}\ \propto\ \underbrace{p(\mathcal D\mid\boldsymbol\mu,\Sigma)}_{\text{likelihood}}\ \times\ \underbrace{p(\boldsymbol\mu,\Sigma)}_{\text{prior}}$$<p>The right-hand picture is this line for the mean: the grey prior curve times the burgundy data curve gives the gold posterior curve. MLE is the peak of the data curve. MAP is the peak of the posterior curve. The Bayesian view keeps the whole gold curve, width included.</p>
 <h4>The Normal-Inverse-Wishart prior</h4>$$\Sigma\sim\mathcal{IW}(\Lambda_0,\nu_0),\qquad \boldsymbol\mu\mid\Sigma\sim N\!\big(\boldsymbol\mu_0,\ \Sigma/\kappa_0\big)$$
 <p>This is the conjugate prior for a Gaussian with unknown mean and covariance, meaning the posterior has the same form. Think of it as imaginary data you saw before the experiment: \(\boldsymbol\mu_0\) is where you believe the mean is and \(\kappa_0\) is how many observations that belief is worth; \(\Lambda_0\) encodes the shape you expect and \(\nu_0\) how many observations that is worth. Here \(\Lambda_0=\nu_0\,s^2 I\) for the spread \(s\) you set, an isotropic guess.</p>
 <h4>The update</h4><p>After \(n\) points with mean \(\bar{\mathbf x}\) and scatter \(S=\sum_i(\mathbf x_i-\bar{\mathbf x})(\mathbf x_i-\bar{\mathbf x})^{\top}\):</p>$$\kappa_n=\kappa_0+n,\qquad \nu_n=\nu_0+n,\qquad \boldsymbol\mu_n=\frac{\kappa_0\,\boldsymbol\mu_0+n\,\bar{\mathbf x}}{\kappa_0+n}$$$$\Lambda_n=\Lambda_0+S+\frac{\kappa_0\,n}{\kappa_0+n}\,(\bar{\mathbf x}-\boldsymbol\mu_0)(\bar{\mathbf x}-\boldsymbol\mu_0)^{\top}$$
 <p>Real and imaginary observations are simply added. The posterior mean is a weighted average: the data holds a share \(n/(\kappa_0+n)\) of the vote, which is the bar above the curves. The last term of \(\Lambda_n\) adds extra spread when the data disagrees with the prior mean.</p>
 <h4>The three estimates</h4>$$\text{MLE: }\ \hat{\boldsymbol\mu}=\bar{\mathbf x},\ \ \hat\Sigma=\frac{S}{n}\qquad\qquad \text{MAP: }\ \hat{\boldsymbol\mu}=\boldsymbol\mu_n,\ \ \hat\Sigma=\frac{\Lambda_n}{\nu_n+d+2}\quad(d=2)$$
 <p>MAP is MLE with a pull toward the prior. This pull is regularisation: the MAP estimate for a regression weight under a Gaussian prior is ridge regression. As \(\kappa_0,\nu_0\to0\) the pull vanishes and MAP becomes MLE.</p>
 <h4>The Bayesian answer</h4><p>The uncertainty about the mean, with \(\Sigma\) integrated out, is a Student-t (the gold curve and the shaded patch):</p>$$\boldsymbol\mu\mid\mathcal D\ \sim\ t_{\nu_n-d+1}\!\left(\boldsymbol\mu_n,\ \frac{\Lambda_n}{\kappa_n(\nu_n-d+1)}\right)$$<p>and the prediction for a new point averages over every plausible Gaussian (the thin ellipses):</p>$$p(\mathbf x_{\text{new}}\mid\mathcal D)=\int p(\mathbf x_{\text{new}}\mid\boldsymbol\mu,\Sigma)\,p(\boldsymbol\mu,\Sigma\mid\mathcal D)\,d\boldsymbol\mu\,d\Sigma=t_{\nu_n-d+1}\!\left(\boldsymbol\mu_n,\ \frac{(\kappa_n+1)\,\Lambda_n}{\kappa_n(\nu_n-d+1)}\right)$$
 <p>It has heavier tails than any single Gaussian. The extra width is the model saying "I am not yet sure of my own parameters", and it fades as \(n\) grows.</p>`,
    q: 'With only two data points, which estimate would you trust, and why?',
    a: 'Not MLE: with two points its covariance is a flat line and it claims certainty it does not have. MAP gives a usable ellipse because the prior fills in what the data cannot yet say, and the Bayesian picture shows how unsure the estimate still is.',
    init(root) {
      const c = $('#by-c', root),
        c2 = $('#by-c2', root),
        cap = $('#by-cap', root),
        TR = { mx: 1.3, my: 0.6, s: 1, rho: 0.6 },
        S = { n: 0, m0: [-1.8, -0.9], k0: 6, v0: 10, s0: 1.3, mode: 'map' };
      let seed = 21,
        data = [];
      const mk = () => {
        const rn = randnFrom(rng(seed)),
          q = Math.sqrt(1 - TR.rho * TR.rho);
        data = [];
        for (let i = 0; i < 120; i++) {
          const a = rn(),
            b = rn();
          data.push([TR.mx + 1.15 * TR.s * a, TR.my + 0.7 * TR.s * (TR.rho * a + q * b)]);
        }
      };
      mk();
      let samples = [],
        dirty = true,
        view = null,
        acc = 0,
        fr = 0,
        pb;
      const tpdf = (x, nu, loc, sc) => {
        const z = (x - loc) / sc;
        return (
          (Math.exp(lgam((nu + 1) / 2) - lgam(nu / 2)) / (Math.sqrt(nu * Math.PI) * sc)) *
          Math.pow(1 + (z * z) / nu, -(nu + 1) / 2)
        );
      };
      const post = () => {
        const n = S.n;
        let mx = 0,
          my = 0;
        for (let i = 0; i < n; i++) {
          mx += data[i][0];
          my += data[i][1];
        }
        if (n) {
          mx /= n;
          my /= n;
        }
        let a = 0,
          b = 0,
          d = 0;
        for (let i = 0; i < n; i++) {
          const u = data[i][0] - mx,
            v = data[i][1] - my;
          a += u * u;
          b += u * v;
          d += v * v;
        }
        const kn = S.k0 + n,
          vn = S.v0 + n,
          l0 = S.v0 * S.s0 * S.s0,
          dx = mx - S.m0[0],
          dy = my - S.m0[1],
          sh = n ? (S.k0 * n) / kn : 0;
        return {
          n,
          mx,
          my,
          a,
          b,
          d,
          kn,
          vn,
          l0,
          mun: n ? [(S.k0 * S.m0[0] + n * mx) / kn, (S.k0 * S.m0[1] + n * my) / kn] : S.m0.slice(),
          La: l0 + a + sh * dx * dx,
          Lb: b + sh * dx * dy,
          Ld: l0 + d + sh * dy * dy,
        };
      };
      const sampleNIW = (P) => {
        const det = P.La * P.Ld - P.Lb * P.Lb,
          ia = P.Ld / det,
          ib = -P.Lb / det,
          idd = P.La / det,
          l11 = Math.sqrt(ia),
          l21 = ib / l11,
          l22 = Math.sqrt(Math.max(1e-12, idd - l21 * l21));
        const a11 = Math.sqrt(2 * gammaS(P.vn / 2)),
          a21 = randn(),
          a22 = Math.sqrt(2 * gammaS((P.vn - 1) / 2)),
          m11 = l11 * a11,
          m21 = l21 * a11 + l22 * a21,
          m22 = l22 * a22;
        const wa = m11 * m11,
          wb = m11 * m21,
          wd = m21 * m21 + m22 * m22,
          wdet = wa * wd - wb * wb,
          sa = wd / wdet,
          sb = -wb / wdet,
          sd = wa / wdet;
        const c11 = Math.sqrt(sa / P.kn),
          c21 = sb / P.kn / c11,
          c22 = Math.sqrt(Math.max(1e-12, sd / P.kn - c21 * c21)),
          z1 = randn(),
          z2 = randn();
        return { mx: P.mun[0] + c11 * z1, my: P.mun[1] + c21 * z1 + c22 * z2, a: sa, b: sb, d: sd };
      };
      const draw = () => {
        const t = T(),
          P = post(),
          M = S.mode,
          sM = M === 'mle' || M === 'all',
          sP = M !== 'mle',
          sA = M === 'map' || M === 'all',
          sB = M === 'bayes' || M === 'all',
          hasM = P.n >= 3 && P.a * P.d - P.b * P.b > 1e-9;
        {
          const { g, w, h } = fit(c),
            sc = Math.min(w / 9.6, h / 6.6),
            X = (x) => w / 2 + x * sc,
            Y = (y) => h / 2 - y * sc;
          view = { w, h, sc };
          const lab = (s, mx, my, d, col) => {
            g.font = SANS;
            g.fillStyle = col;
            g.textAlign = 'center';
            g.fillText(s, clamp(X(mx), 40, w - 40), clamp(Y(my + 2 * Math.sqrt(d)) - 7, 13, h - 6));
          };
          if (dirty) {
            samples = [];
            for (let i = 0; i < 12; i++) samples.push(sampleNIW(P));
            dirty = false;
          }
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          for (let k = -5; k <= 5; k++) {
            g.beginPath();
            g.moveTo(X(k), 0);
            g.lineTo(X(k), h);
            g.moveTo(0, Y(k));
            g.lineTo(w, Y(k));
            g.stroke();
          }
          const tsx = 1.15 * TR.s,
            tsy = 0.7 * TR.s;
          g.setLineDash([7, 5]);
          g.strokeStyle = mix(t.ink, t.ink, 0, 0.45);
          g.lineWidth = 1.3;
          ellPath(g, X, Y, TR.mx, TR.my, tsx * tsx, TR.rho * tsx * tsy, tsy * tsy, 2);
          g.stroke();
          g.setLineDash([]);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('truth, dashed', X(TR.mx), Math.min(h - 6, Y(TR.my - 2 * tsy) + 15));
          if (sB) {
            g.lineWidth = 1;
            g.strokeStyle = mix(t.gold, t.gold, 0, 0.5);
            for (const s of samples) {
              ellPath(g, X, Y, s.mx, s.my, s.a, s.b, s.d, 2);
              g.stroke();
            }
          }
          g.fillStyle = mix(t.accent, t.accent, 0, 0.9);
          for (let i = 0; i < S.n; i++) {
            g.beginPath();
            g.arc(
              X(data[i][0]),
              Y(data[i][1]),
              i === S.n - 1 && S.n < data.length ? 6.5 : 3.2,
              0,
              7,
            );
            g.fill();
          }
          const hx = X(S.m0[0]),
            hy = Y(S.m0[1]);
          if (sP) {
            g.setLineDash([3, 5]);
            g.strokeStyle = t.muted;
            g.lineWidth = 1.6;
            ellPath(g, X, Y, S.m0[0], S.m0[1], S.s0 * S.s0, 0, S.s0 * S.s0, 2);
            g.stroke();
            g.setLineDash([]);
            lab('prior guess', S.m0[0], S.m0[1], S.s0 * S.s0, t.muted);
            if (P.n && sA) {
              g.strokeStyle = t.muted;
              g.lineWidth = 1.2;
              g.setLineDash([2, 4]);
              g.beginPath();
              g.moveTo(hx, hy);
              g.lineTo(X(P.mx), Y(P.my));
              g.stroke();
              g.setLineDash([]);
            }
          }
          if (sM) {
            if (hasM) {
              g.strokeStyle = t.ok;
              g.lineWidth = 3;
              ellPath(g, X, Y, P.mx, P.my, P.a / P.n, P.b / P.n, P.d / P.n, 2);
              g.stroke();
              lab('MLE', P.mx, P.my, P.d / P.n, t.ok);
            }
            if (P.n) {
              g.fillStyle = t.ok;
              g.beginPath();
              g.arc(X(P.mx), Y(P.my), 5.5, 0, 7);
              g.fill();
            }
          } else if (P.n && sA) {
            g.strokeStyle = t.ok;
            g.lineWidth = 2;
            g.beginPath();
            g.arc(X(P.mx), Y(P.my), 5, 0, 7);
            g.stroke();
            g.font = MONO;
            g.fillStyle = t.ok;
            g.textAlign = 'left';
            g.fillText('data mean', X(P.mx) + 9, Y(P.my) + 4);
          }
          if (sB) {
            const dn = P.kn * (P.vn - 1);
            ellPath(g, X, Y, P.mun[0], P.mun[1], P.La / dn, P.Lb / dn, P.Ld / dn, 2);
            g.fillStyle = mix(t.gold, t.gold, 0, 0.4);
            g.fill();
            g.strokeStyle = t.gold;
            g.lineWidth = 1.5;
            g.stroke();
            if (M === 'bayes') {
              g.font = MONO;
              g.fillStyle = t.cut;
              g.textAlign = 'left';
              g.fillText('where the mean could be', X(P.mun[0]) + 10, Y(P.mun[1]) - 10);
            }
          }
          if (sA) {
            const dn = P.vn + 4;
            g.strokeStyle = t.gold;
            g.lineWidth = 3.2;
            ellPath(g, X, Y, P.mun[0], P.mun[1], P.La / dn, P.Lb / dn, P.Ld / dn, 2);
            g.stroke();
            lab('MAP', P.mun[0], P.mun[1], P.Ld / dn, t.cut);
            g.fillStyle = t.gold;
            g.beginPath();
            g.arc(X(P.mun[0]), Y(P.mun[1]), 6, 0, 7);
            g.fill();
          }
          if (sP) {
            g.fillStyle = t.bg;
            g.strokeStyle = t.muted;
            g.lineWidth = 2.5;
            g.beginPath();
            g.arc(hx, hy, 7.5, 0, 7);
            g.fill();
            g.stroke();
          }
        }
        {
          const { g, w, h } = fit(c2),
            share = P.n / (S.k0 + P.n),
            bx = 38,
            bw = w - 50;
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.muted;
          g.fillText('who has the vote', bx, 12);
          g.fillStyle = mix(t.muted, t.muted, 0, 0.45);
          g.fillRect(bx, 18, bw * (1 - share), 12);
          g.fillStyle = t.accent;
          g.fillRect(bx + bw * (1 - share), 18, bw * share, 12);
          g.fillStyle = t.muted;
          g.fillText('prior κ₀ = ' + (+S.k0).toFixed(1), bx, 44);
          g.textAlign = 'right';
          g.fillStyle = t.accent;
          g.fillText('data n = ' + P.n, bx + bw, 44);
          g.save();
          g.translate(0, 52);
          const hh = h - 52,
            pri = (x) => tpdf(x, S.v0 - 1, S.m0[0], Math.sqrt(P.l0 / (S.k0 * (S.v0 - 1)))),
            pos = (x) => tpdf(x, P.vn - 1, P.mun[0], Math.sqrt(P.La / (P.kn * (P.vn - 1))));
          const sdD = P.n >= 2 ? Math.sqrt(P.a / (P.n - 1) / P.n) : 0,
            dat = (x) => npdf(x, P.mx, sdD);
          let ym = 0.3;
          if (sP) ym = Math.max(ym, pri(S.m0[0]), pos(P.mun[0]));
          if (sdD > 0 && M !== 'bayes') ym = Math.max(ym, dat(P.mx));
          ym = Math.min(ym * 1.12, 8);
          const Pl = plot(g, w, hh, [-4.8, 4.8], [0, ym], t, { xlabel: 'mean, x-coordinate' });
          clipPlot(g, Pl);
          if (sP) {
            g.setLineDash([3, 5]);
            curve(g, Pl, pri, -4.8, 4.8);
            g.strokeStyle = t.muted;
            g.lineWidth = 2;
            g.stroke();
            g.setLineDash([]);
          }
          if (sdD > 0 && M !== 'bayes') {
            curve(g, Pl, dat, -4.8, 4.8, 520);
            g.strokeStyle = t.ok;
            g.lineWidth = 2.4;
            g.stroke();
          }
          if (sP) {
            area(g, Pl, pos, -4.8, 4.8, 520);
            g.fillStyle = mix(t.gold, t.gold, 0, 0.3);
            g.fill();
            curve(g, Pl, pos, -4.8, 4.8, 520);
            g.strokeStyle = t.gold;
            g.lineWidth = 3;
            g.stroke();
          }
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          let ly = 70;
          const L = [];
          if (sP) L.push([t.muted, 'prior, dotted']);
          if (M !== 'bayes')
            L.push([t.ok, sdD > 0 ? 'data alone (peak = MLE)' : 'data alone (needs 2 points)']);
          if (sP)
            L.push([
              t.cut,
              M === 'bayes' ? 'posterior (keep all of it)' : 'posterior (peak = MAP)',
            ]);
          L.forEach(([col, s]) => {
            g.fillStyle = col;
            g.fillText(s, 46, ly);
            ly += 15;
          });
        }
        const pc = Math.round((100 * P.n) / (S.k0 + P.n));
        cap.textContent =
          M === 'mle'
            ? hasM
              ? 'MLE: the Gaussian that fits these ' +
                P.n +
                ' points best. It ignores any prior belief.'
              : 'MLE has no ellipse yet. It needs at least 3 points and has nothing to say before that.'
            : M === 'map'
              ? P.n
                ? 'MAP: the prior guess, pulled toward the data. With ' +
                  P.n +
                  ' points the data holds ' +
                  pc +
                  '% of the vote, so the gold dot sits ' +
                  pc +
                  '% of the way from the prior to the data mean.'
                : 'MAP: with no data yet, the estimate is the prior guess itself. Add points and watch it move.'
              : M === 'bayes'
                ? 'Bayesian: each thin outline is one Gaussian that could still explain the data. The shaded patch is where the true mean probably lies. Both tighten as points arrive.'
                : 'All three. Burgundy is MLE, gold is MAP, thin gold outlines are the Bayesian bundle. They disagree with little data and merge with a lot.';
        setR(root, 'n', String(P.n) + ' of ' + data.length);
        setR(root, 'mle', P.n ? '(' + f2(P.mx) + ', ' + f2(P.my) + ')' : '–');
        setR(root, 'map', '(' + f2(P.mun[0]) + ', ' + f2(P.mun[1]) + ')');
        setR(root, 'w', pc + '%');
        setR(root, 'pm', '(' + f2(S.m0[0]) + ', ' + f2(S.m0[1]) + ')');
      };
      const chg = () => {
        dirty = true;
        draw();
      };
      const setMode = (m) => {
        S.mode = m;
        root
          .querySelectorAll('#by-mode button')
          .forEach((b) => b.setAttribute('aria-pressed', b.dataset.m === m ? 'true' : 'false'));
        chg();
      };
      $('#by-mode', root).addEventListener('click', (e) => {
        const m = e.target.dataset && e.target.dataset.m;
        if (m) setMode(m);
      });
      const anim = Anim(() => {
        fr++;
        acc += S.n < 10 ? 0.022 : S.n < 40 ? 0.1 : 0.4;
        if (acc >= 1) {
          acc = 0;
          if (S.n < data.length) {
            S.n++;
            dirty = true;
          }
        }
        if (fr % 18 === 0) dirty = true;
        draw();
      });
      pb = playBtn($('#by-play', root), anim);
      $('#by-play', root).addEventListener('click', () => {
        if (anim.on && S.n >= data.length) {
          S.n = 0;
          dirty = true;
        }
      });
      $('#by-step', root).onclick = () => {
        pb.stop();
        if (S.n < data.length) S.n++;
        chg();
      };
      $('#by-reset', root).onclick = () => {
        pb.stop();
        S.n = 0;
        acc = 0;
        chg();
      };
      bind(
        root,
        'by-k',
        (v) => {
          S.k0 = v;
          chg();
        },
        (v) => (+v).toFixed(1),
      );
      bind(
        root,
        'by-v',
        (v) => {
          S.v0 = v;
          chg();
        },
        (v) => String(v),
      );
      bind(root, 'by-s', (v) => {
        S.s0 = v;
        chg();
      });
      bind(root, 'by-ds', (v) => {
        TR.s = v;
        mk();
        chg();
      });
      bind(root, 'by-dr', (v) => {
        TR.rho = v;
        mk();
        chg();
      });
      $('#by-new', root).onclick = () => {
        seed++;
        mk();
        S.n = 0;
        acc = 0;
        chg();
        pb.start();
      };
      let drag = false;
      const mv = (e) => {
        if (!view) return;
        const r = c.getBoundingClientRect();
        S.m0 = [
          clamp((e.clientX - r.left - view.w / 2) / view.sc, -4.4, 4.4),
          clamp(-(e.clientY - r.top - view.h / 2) / view.sc, -2.9, 2.9),
        ];
        chg();
      };
      c.addEventListener('pointerdown', (e) => {
        if (S.mode === 'mle') return;
        drag = true;
        c.setPointerCapture(e.pointerId);
        mv(e);
      });
      c.addEventListener('pointermove', (e) => {
        if (drag) mv(e);
      });
      ['pointerup', 'pointercancel'].forEach((n) =>
        c.addEventListener(n, () => {
          drag = false;
        }),
      );
      setMode('map');
      return { draw, stop: () => pb.stop(), auto: () => pb.start() };
    },
  },

  {
    id: 'distance',
    tab: 'Mahalanobis distance',
    title: 'Distance that follows the shape',
    lede: 'Points A and B are the same straight-line distance from the centre. But A lies along the cloud and B lies across it, so B is far more surprising. Mahalanobis distance measures distance in units of spread.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Drag A and B</figcaption><canvas class="sq" id="md-c" style="touch-action:none;cursor:pointer" role="img" aria-label="Two draggable points on a tilted Gaussian"></canvas></figure>
 <div><dl class="read" style="margin-top:22px">${rd('ae', 'A: straight-line distance')}${rd('am', 'A: Mahalanobis distance', 'hl2')}${rd('be', 'B: straight-line distance')}${rd('bm', 'B: Mahalanobis distance', 'hl')}${rd('pa', 'Share of the data closer than A', 'hl2')}${rd('pb', 'Share of the data closer than B', 'hl')}</dl></div></div>
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="md-play" type="button">Whiten</button><button class="btn" id="md-reset" type="button">Reset points</button></div>${slider('md-t', 'Whiten the space', 0, 1, 0.01, 0)}`)}${grp('Shape of the cloud', shapeHTML('md', { sx: 1.7, sy: 0.85, rho: 0.72 }))}</div>`,
    tryit:
      'Watch the space squeeze until the ellipses become circles, then relax again. When it is fully whitened, compare the two rings with the ordinary distances on screen. Then make the cloud isotropic: do the two distances still differ?',
    math: String.raw`<h4>Intuition</h4><p>Is a point far from the centre? It depends which way you went. Along the long axis of the cloud the data spreads widely, so two units is nothing unusual. Across the short axis two units puts you well outside the cloud. A useful distance should count steps in units of the spread in that direction.</p>
 <h4>The formula</h4>$$d_M(\mathbf x)=\sqrt{(\mathbf x-\boldsymbol\mu)^{\top}\,\Sigma^{-1}\,(\mathbf x-\boldsymbol\mu)}$$
 <p>This is the multivariate version of the z-score \(|x-\mu|/\sigma\). If \(\Sigma=\sigma^2 I\) it reduces to ordinary distance divided by \(\sigma\).</p>
 <h4>Whitening</h4><p>Apply the change of variables \(\mathbf z=\Sigma^{-1/2}(\mathbf x-\boldsymbol\mu)\). In the new coordinates the cloud is round with unit variance, and</p>$$d_M(\mathbf x)=\lVert\mathbf z\rVert.$$<p>Mahalanobis distance is ordinary distance after the space has been stretched until the data looks isotropic. That is what the animation does. Whitening inputs before training is the same operation, and it is why feature scaling matters for methods based on distance such as k-nearest neighbours.</p>
 <h4>Distance and density</h4><p>The Gaussian density is \(p(\mathbf x)\propto e^{-d_M^2/2}\), so points at the same Mahalanobis distance are equally likely: the rings are contours. In two dimensions the share of the data closer than \(d\) is \(1-e^{-d^2/2}\), which gives 39% inside \(d=1\), 86% inside \(d=2\) and 98.9% inside \(d=3\).</p>`,
    q: 'An anomaly detector flags points with Mahalanobis distance above 3. Why is that better than a straight-line threshold?',
    a: 'A straight-line threshold draws a circle, which is too generous along the short axis and too strict along the long one. The Mahalanobis threshold draws the 3σ ellipse, which matches where the data actually lives.',
    init(root) {
      const c = $('#md-c', root),
        G8 = { sx: 1.7, sy: 0.85, rho: 0.72 },
        S = { t: 0, A: null, B: null };
      let Wm,
        last = null;
      const prep = () => {
        const e = eig(G8),
          c1 = Math.cos(e.th),
          s1 = Math.sin(e.th),
          i1 = 1 / Math.sqrt(e.l1),
          i2 = 1 / Math.sqrt(e.l2);
        Wm = [c1 * c1 * i1 + s1 * s1 * i2, c1 * s1 * (i1 - i2), s1 * s1 * i1 + c1 * c1 * i2];
        return [
          [2 * c1, 2 * s1],
          [-2 * s1, 2 * c1],
        ];
      };
      const place = () => {
        const p = prep();
        S.A = p[0];
        S.B = p[1];
      };
      place();
      const M = () => [1 - S.t + S.t * Wm[0], S.t * Wm[1], 1 - S.t + S.t * Wm[2]];
      const mah = (p) => Math.hypot(Wm[0] * p[0] + Wm[1] * p[1], Wm[1] * p[0] + Wm[2] * p[1]);
      const draw = () => {
        prep();
        const t = T(),
          m = M(),
          tr = (x, y) => [m[0] * x + m[1] * y, m[1] * x + m[2] * y],
          D = drawMap(c, t, G8, { tr }),
          g = D.g;
        last = D;
        const ra = Math.hypot(S.A[0], S.A[1]);
        g.setLineDash([5, 4]);
        g.strokeStyle = t.ink;
        g.lineWidth = 1.3;
        g.beginPath();
        for (let k = 0; k <= 72; k++) {
          const th = (k / 72) * 2 * Math.PI,
            p = D.X(ra * Math.cos(th), ra * Math.sin(th));
          k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]);
        }
        g.stroke();
        g.setLineDash([]);
        g.lineWidth = 2.6;
        g.strokeStyle = t.ok;
        D.ring(mah(S.A));
        g.stroke();
        g.strokeStyle = t.gold;
        D.ring(mah(S.B));
        g.stroke();
        const o = D.X(0, 0);
        [
          [S.A, t.ok, 'A'],
          [S.B, t.gold, 'B'],
        ].forEach(([p, col, nm]) => {
          const q = D.X(p[0], p[1]);
          g.strokeStyle = col;
          g.lineWidth = 2;
          D.seg(o, q);
          g.fillStyle = col;
          g.beginPath();
          g.arc(q[0], q[1], 9, 0, 7);
          g.fill();
          g.strokeStyle = t.bg;
          g.lineWidth = 2.5;
          g.stroke();
          g.fillStyle = t.ink;
          g.font = SANS;
          g.textAlign = 'left';
          g.fillText(nm, q[0] + 12, q[1] - 10);
        });
        const ma = mah(S.A),
          mb = mah(S.B);
        setR(root, 'ae', f2(Math.hypot(S.A[0], S.A[1])));
        setR(root, 'am', f2(ma));
        setR(root, 'be', f2(Math.hypot(S.B[0], S.B[1])));
        setR(root, 'bm', f2(mb));
        setR(root, 'pa', (100 * (1 - Math.exp((-ma * ma) / 2))).toFixed(1) + '%');
        setR(root, 'pb', (100 * (1 - Math.exp((-mb * mb) / 2))).toFixed(1) + '%');
      };
      const ts = bind(root, 'md-t', (v) => {
        S.t = v;
        draw();
      });
      let ph = 0;
      shapeBind(root, 'md', G8, () => draw());
      const anim = Anim(() => {
        ph += 1 / 60 / 9;
        const u = (1 - Math.cos(ph * 2 * Math.PI)) / 2;
        S.t = clamp(u * 1.3 - 0.15, 0, 1);
        ts.set(S.t);
        draw();
      });
      const pb = playBtn($('#md-play', root), anim);
      grab(ts.el, pb);
      $('#md-reset', root).onclick = () => {
        place();
        draw();
      };
      let drag = null;
      const toData = (ev) => {
        const r = c.getBoundingClientRect(),
          u = (ev.clientX - r.left - last.w / 2) / last.s,
          v = -(ev.clientY - r.top - last.h / 2) / last.s,
          m = M(),
          det = m[0] * m[2] - m[1] * m[1];
        return [
          clamp((m[2] * u - m[1] * v) / det, -3.9, 3.9),
          clamp((-m[1] * u + m[0] * v) / det, -3.9, 3.9),
        ];
      };
      c.addEventListener('pointerdown', (ev) => {
        if (!last) return;
        const r = c.getBoundingClientRect(),
          px = ev.clientX - r.left,
          py = ev.clientY - r.top,
          a = last.X(S.A[0], S.A[1]),
          b = last.X(S.B[0], S.B[1]),
          da = Math.hypot(a[0] - px, a[1] - py),
          db = Math.hypot(b[0] - px, b[1] - py);
        drag = da <= db ? 'A' : 'B';
        S[drag] = toData(ev);
        c.setPointerCapture(ev.pointerId);
        draw();
      });
      c.addEventListener('pointermove', (ev) => {
        if (drag) {
          S[drag] = toData(ev);
          draw();
        }
      });
      ['pointerup', 'pointercancel'].forEach((n) =>
        c.addEventListener(n, () => {
          drag = null;
        }),
      );
      return { draw, stop: pb.stop, auto: pb.start };
    },
  },

  {
    id: 'next',
    tab: 'Where it comes back',
    title: 'Where the Gaussian comes back',
    lede: 'Almost every later topic leans on one of the pictures you just played with.',
    stage: `<dl class="links">
 <div><dt><small>Regression</small>Noise around a line</dt><dd>Least squares is maximum likelihood when the errors are Gaussian. The slice picture, a mean that moves in a straight line, is linear regression. Ridge regression is its MAP version.</dd></div>
 <div><dt><small>Classification</small>One hill per class</dt><dd>Fit a Gaussian to each class and compare heights. Equal covariances give a straight boundary (LDA), different ones a curved boundary (QDA).</dd></div>
 <div><dt><small>Unsupervised learning</small>Several hills at once</dt><dd>A mixture of Gaussians fixes the two-clumps failure. PCA is the long and short axes of the covariance ellipse.</dd></div>
 <div><dt><small>Regression</small>Gaussian processes</dt><dd>Slicing a Gaussian in many dimensions gives a prediction with an uncertainty band. It is the conditional formula, used on functions.</dd></div>
 <div><dt><small>Neural networks</small>Starting weights</dt><dd>Weights are usually drawn from a Gaussian. Its spread decides whether signals die out or blow up as they pass through layers.</dd></div>
 <div><dt><small>Generative AI</small>VAEs and diffusion</dt><dd>A VAE squeezes data toward an isotropic Gaussian. Diffusion adds Gaussian noise step by step until only an isotropic Gaussian is left, then learns to walk back.</dd></div>
 </dl>`,
    tryit:
      'Go back to “Marginals and conditionals” and set ρ to 0.9. The centre of the slice traces a line as the cut slides. That line is the regression line.',
    math: String.raw`<h4>Three facts worth remembering</h4><p><b>Sums stay Gaussian.</b> Adding independent Gaussians gives a Gaussian, and adding almost anything else tends toward one (Galton board, central limit theorem).</p><p><b>Slices and shadows stay Gaussian.</b> Conditioning and marginalising never leave the family, so inference can be done with matrix algebra instead of integrals.</p><p><b>Two objects describe everything.</b> A mean vector and a covariance matrix, both of which can be estimated in closed form:</p>$$\mathbf x\sim N(\boldsymbol\mu,\Sigma)\quad\Longrightarrow\quad A\mathbf x+\mathbf b\sim N\big(A\boldsymbol\mu+\mathbf b,\ A\Sigma A^{\top}\big).$$<p>That last line, a Gaussian pushed through a linear map stays Gaussian, is the single most used identity in the topics that follow.</p>`,
    q: 'Which lesson showed the formula behind linear regression?',
    a: 'Marginals and conditionals. The mean of y given x₀ is μᵧ + ρ (σᵧ/σₓ)(x₀ − μₓ), a straight line in x₀ with slope ρ σᵧ/σₓ.',
    init() {
      return { draw() {} };
    },
  },
];
