/* Manifold — Generative AI.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ generative AI ============ */
const GEN = [
  {
    id: 'autoregressive',
    tab: 'Autoregressive models',
    title: 'Autoregressive models: write one letter, then ask what comes next',
    lede: 'Every chatbot writes the same way: it looks at what has been written so far, works out how likely each possible next piece is, picks one, and repeats. That is all “autoregressive” means. Here the model is as simple as it can be: it has read a short text and counted which letter tends to follow which. Watch it write, then give it your own text to learn from.',
    stage: `<p class="cap" id="ar-out" role="status" style="font-family:var(--font-mono);min-height:5.5em;white-space:pre-wrap"></p>
 <div class="stage"><figure class="fig"><figcaption>How likely the model thinks each next letter is, given the letters in the gold box. The one it picked is gold</figcaption><canvas class="short" id="ar-c" role="img" aria-label="Next-character probabilities"></canvas></figure>
 <figure class="fig"><figcaption>Top: what the model has learned from the text, for the current context. Bottom: how surprised the model is by the real text, for each memory length</figcaption><canvas class="short" id="ar-r" role="img" aria-label="Counts for the current context and perplexity by context length"></canvas></figure></div>
 ${uHow('ar', ['Learn: read the text once. For every run of k letters, count which letter came next.', 'Write: look at the last k letters written so far.', 'Look up the counts for that context and turn them into probabilities.', 'Pick the next letter at random according to those probabilities. Temperature makes the choice bolder or safer.', 'Add the letter to the text and go back to step 2.'])}
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="ar-play" type="button">Write</button><button class="btn" id="ar-step" type="button">One letter</button><button class="btn" id="ar-reset" type="button">Start over</button></div>`)}
 ${grp('The model', slider('ar-k', 'Memory: letters of context it looks at', 0, 6, 1, 3) + slider('ar-temp', 'Temperature', 0.2, 2, 0.05, 1))}
 ${grp('The text it learns from', `<textarea id="ar-text" rows="4" style="width:100%;font:inherit;font-size:.85rem;border:1px solid var(--line);background:none;padding:6px" aria-label="Training text"></textarea><p class="status" id="ar-st" role="status"></p>`)}</div>
 <dl class="read">${rd('ctx', 'Context right now')}${rd('opt', 'Different letters seen after it')}${rd('pick', 'Letter picked', 'hl')}${rd('n', 'Letters in the training text')}${rd('pp', 'Perplexity on the training text at this memory', 'hl2')}</dl>`,
    tryit:
      'Set the memory to 0 and write: a soup of letters in roughly English proportions. Raise it to 1, 2, 3, 4: syllables appear, then words, then phrases. At 6 it mostly copies the text back. Why? Then put the memory at 3 and move the temperature to 0.2 and to 2. Finally paste a paragraph in another language, or a song you know, and see what it writes.',
    math: String.raw`<h4>Intuition</h4><p>Play a game: I say "once upon a", and you say the next word. You did not need to think. You have heard that phrase finished the same way many times. An autoregressive model plays this game with itself. It writes a piece, reads back what it has written, guesses the next piece, and continues. The only thing that changes from the toy on this page to the largest language model is how it makes the guess: by counting here, with a Transformer there.</p>
 <h4>The idea in one line</h4><p>The probability of a whole sequence can always be split into a product of next-item probabilities, by the chain rule:</p>$$p(x_1,x_2,\dots,x_T)=\prod_{t=1}^{T}p\big(x_t\mid x_1,\dots,x_{t-1}\big).$$<p>So a model that can answer "what comes next, given everything so far?" is a complete model of sequences. To generate, sample \(x_1\), then \(x_2\) given \(x_1\), and so on.</p>
 <h4>The model on this page</h4><p>It assumes the next letter depends only on the previous \(k\) letters, and estimates the probability by counting in the training text:</p>$$p\big(x_t\mid x_{t-k},\dots,x_{t-1}\big)\approx\frac{\text{count}(x_{t-k}\dots x_{t-1}\,x_t)}{\text{count}(x_{t-k}\dots x_{t-1})}.$$<p>This is an n-gram model with \(n=k+1\). Training is one pass of counting. A neural language model replaces the table of counts with a network, which lets it use thousands of items of context and share what it learns between similar contexts.</p>
 <h4>Temperature</h4><p>Before sampling, each probability is raised to the power \(1/\tau\) and the results are rescaled to add up to 1:</p>$$p_\tau(x)\propto p(x)^{1/\tau}.$$<p>With \(\tau&lt;1\) likely letters become more likely still, and the text is safe and repetitive. With \(\tau&gt;1\) the choices even out, and the text is varied and error-prone. \(\tau\to0\) always takes the single most likely letter.</p>
 <h4>Memory against data</h4><p>Longer context gives better guesses but splits the text into more and more distinct contexts, each seen fewer times. With a short text and \(k=6\), most contexts were seen once, so the model has exactly one continuation to offer and reproduces the text. That is overfitting. <b>Perplexity</b>, \(\exp\big(-\tfrac1T\sum_t\log p(x_t\mid\text{context})\big)\), is the model's average number of equally likely choices per letter. On the training text it always falls as \(k\) grows, which is why it must be measured on text the model has not seen.</p>
 <h4>Why it matters</h4><p>GPT-style models, WaveNet for audio and PixelCNN for images are all autoregressive. Their strengths: an exact likelihood and simple training (predict the next item). Their weakness: generation is one item at a time, and an early mistake cannot be taken back.</p>`,
    q: 'With a memory of 6 letters the output is nearly a copy of the training text. Is that a better model than memory 3?',
    a: 'It is a better memoriser, not a better model. With a short training text almost every 6-letter context occurred once, so only one continuation has a non-zero count and the model can do nothing but replay the text. It has the lowest perplexity on the training text and would do badly on new text, where most 6-letter contexts never appeared at all. Memory 3 has seen each context many times and has learned patterns it can recombine.',
    init(root) {
      const c = $('#ar-c', root),
        cr = $('#ar-r', root),
        live = $('#ar-live', root),
        outEl = $('#ar-out', root),
        ta = $('#ar-text', root),
        st = $('#ar-st', root),
        S = { k: 3, temp: 1 },
        MAXL = 320,
        DEF =
          'the cat sat on the mat and looked at the rain. the rain fell on the roof and on the road and on the old stone wall. a dog ran past the wall and the cat ran after the dog. then the sun came out and the cat sat in the sun and the dog sat in the shade. the rain had stopped and the road was wet and the wall was warm.';
      let text = '',
        M = {},
        out = '',
        last = null,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        pps = [];
      const learn = () => {
        text = (ta.value || DEF).toLowerCase().replace(/\s+/g, ' ').slice(0, 6000);
        if (text.length < 40) {
          st.textContent =
            'That text is very short. Paste at least a sentence or two; using the built-in text for now.';
          text = DEF;
        } else st.textContent = '';
        M = {};
        for (let k = 0; k <= 6; k++) {
          const m = (M[k] = {});
          for (let i = k; i < text.length; i++) {
            const cx = text.slice(i - k, i),
              t = m[cx] || (m[cx] = {});
            t[text[i]] = (t[text[i]] || 0) + 1;
          }
        }
        pps = [];
        for (let k = 0; k <= 6; k++) {
          let s = 0,
            n = 0;
          for (let i = k; i < text.length; i++) {
            const t = M[k][text.slice(i - k, i)];
            let tot = 0;
            for (const q in t) tot += t[q];
            s -= Math.log(t[text[i]] / tot);
            n++;
          }
          pps.push(Math.exp(s / n));
        }
      };
      const dist = () => {
        let k = Math.min(S.k, out.length),
          t;
        for (; ; k--) {
          t = M[k][out.slice(out.length - k)];
          if (t || k === 0) break;
        }
        const ent = Object.entries(t || { ' ': 1 });
        let z = 0;
        const pr = ent.map(([ch, n]) => {
          const p = Math.pow(n, 1 / S.temp);
          z += p;
          return [ch, p, n];
        });
        pr.forEach((q) => {
          q[1] /= z;
        });
        pr.sort((a, b) => b[1] - a[1]);
        return { pr, k, ctx: out.slice(out.length - k) };
      };
      const reset = () => {
        out = text.slice(0, Math.max(1, S.k));
        last = null;
        cu = 0;
        msg =
          'The model has read ' +
          text.length +
          ' letters and counted what follows every run of ' +
          S.k +
          '. It starts from “' +
          out +
          '”. Press Write or One letter.';
        outEl.textContent = out;
      };
      const step = () => {
        if (out.length >= MAXL) {
          reset();
          return;
        }
        const d = dist();
        let u = Math.random(),
          pick = d.pr[d.pr.length - 1];
        for (const q of d.pr) {
          if ((u -= q[1]) <= 0) {
            pick = q;
            break;
          }
        }
        last = { d, pick };
        out += pick[0];
        cu = 2 + (out.length % 3);
        outEl.textContent = out;
        msg =
          'Context “' +
          d.ctx +
          '”' +
          (d.k < S.k ? ' (shortened, because the longer context never appeared in the text)' : '') +
          ': in the text it was followed by ' +
          d.pr.length +
          ' different letter' +
          (d.pr.length === 1 ? '' : 's') +
          '. The model picked “' +
          (pick[0] === ' ' ? 'space' : pick[0]) +
          '”, which had probability ' +
          (100 * pick[1]).toFixed(0) +
          '%.' +
          (out.length >= MAXL ? ' It has written ' + MAXL + ' letters, so it stops.' : '');
      };
      const draw = () => {
        const t = T(),
          d = last ? last.d : dist(),
          sh = d.pr.slice(0, 14);
        {
          const { g, w, h } = fit(c),
            bw = (w - 30) / 14;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('after', 10, 16);
          g.font = SANS;
          const cx = '“' + d.ctx.replace(/ /g, '␣') + '”',
            tw = g.measureText(cx).width + 16;
          g.fillStyle = t.gold;
          g.fillRect(52, 2, tw, 22);
          g.fillStyle = t.bg;
          g.fillText(cx, 60, 18);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.fillText('the next letter is…', 60 + tw + 4, 16);
          sh.forEach((q, i) => {
            const x = 15 + i * bw,
              top = h - 30 - q[1] * (h - 90),
              pk = last && q === last.pick;
            g.fillStyle = pk ? t.gold : t.accent;
            g.fillRect(x + bw * 0.14, top, bw * 0.72, h - 30 - top);
            g.font = SANS;
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.fillText(q[0] === ' ' ? '␣' : q[0], x + bw / 2, h - 12);
            g.font = MONO;
            g.fillStyle = t.muted;
            g.fillText((100 * q[1]).toFixed(0) + '%', x + bw / 2, top - 5);
          });
          if (d.pr.length > 14) {
            g.textAlign = 'right';
            g.fillText('+' + (d.pr.length - 14) + ' more', w - 8, 44);
          }
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.48;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            'the counts behind those bars: after “' + d.ctx.replace(/ /g, '␣') + '” the text had',
            8,
            13,
          );
          const tot = d.pr.reduce((s, q) => s + q[2], 0);
          sh.slice(0, 7).forEach((q, i) => {
            const y = 24 + i * ((h1 - 30) / 7);
            g.fillStyle = t.ink;
            g.fillText(
              '“' +
                (q[0] === ' ' ? '␣' : q[0]) +
                '”  ' +
                q[2] +
                ' time' +
                (q[2] === 1 ? '' : 's') +
                ' of ' +
                tot,
              14,
              y + 11,
            );
          });
          const m = { l: 44, r: 12, t: 22, b: 22 },
            bw = (w - m.l - m.r) / 7,
            ym = Math.max(...pps) * 1.1,
            Y = (v) => h - m.b - (v / ym) * (h - h1 - m.t - m.b);
          g.fillStyle = t.muted;
          g.fillText(
            'perplexity on the training text (lower = less surprised), by memory',
            8,
            h1 + 12,
          );
          pps.forEach((v, k) => {
            const x = m.l + k * bw + bw * 0.15;
            g.fillStyle = k === S.k ? t.gold : t.ok;
            g.fillRect(x, Y(v), bw * 0.7, h - m.b - Y(v));
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(String(k), x + bw * 0.35, h - m.b + 14);
            g.fillText(v.toFixed(1), x + bw * 0.35, Y(v) - 4);
          });
          g.textAlign = 'right';
          g.fillText('letters of memory', w - m.r, h - 3);
        }
        live.textContent = msg;
        uCur(root, 'ar', cu);
        setR(root, 'ctx', '“' + d.ctx.replace(/ /g, '␣') + '”');
        setR(root, 'opt', String(d.pr.length));
        setR(
          root,
          'pick',
          last
            ? '“' +
                (last.pick[0] === ' ' ? '␣' : last.pick[0]) +
                '” (' +
                (100 * last.pick[1]).toFixed(0) +
                '%)'
            : '–',
        );
        setR(root, 'n', String(text.length));
        setR(root, 'pp', pps[S.k].toFixed(2));
      };
      ta.value = DEF;
      learn();
      reset();
      for (let i = 0; i < 120; i++) step();
      const anim = Anim(() => {
        fr++;
        if (fr % (out.length < S.k + 8 ? 40 : 5)) return;
        step();
        draw();
        if (out.length >= MAXL) pb.stop();
      });
      pb = playBtn($('#ar-play', root), anim);
      runCtl(pb, root, 'ar', { done: () => out.length >= MAXL, restart: () => reset() });
      $('#ar-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#ar-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      bind(
        root,
        'ar-k',
        (v) => {
          S.k = v;
          pb.stop();
          reset();
          draw();
          pb.start();
        },
        (v) => String(v),
      );
      bind(
        root,
        'ar-temp',
        (v) => {
          S.temp = v;
          last = null;
          if (!pb.on) draw();
        },
        f2,
      );
      ta.addEventListener('change', () => {
        pb.stop();
        learn();
        reset();
        draw();
        pb.start();
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
    id: 'vae',
    tab: 'Variational autoencoders',
    title: 'Variational autoencoders: a map with no empty regions',
    lede: 'An ordinary autoencoder gives each picture a spot on a map, but the space between the spots is no-man’s-land: decode a random point and you get rubbish. A variational autoencoder makes two changes. Each picture becomes a small cloud instead of a point, and all the clouds are pulled toward the centre. The map fills in, and now any point you pick decodes to something sensible. That makes it a generator.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The map of codes. Each letter is where one picture’s cloud is centred; the rings mark where random draws will come from. Faint pictures show what every region decodes to. Drag the diamond</figcaption><canvas class="short" id="va-c" role="img" aria-label="Latent space of the variational autoencoder"></canvas></figure>
 <figure class="fig"><figcaption>Top: originals and what comes back. Middle: brand-new pictures decoded from random points inside the rings. Bottom: the two parts of the loss</figcaption><canvas class="short" id="va-r" role="img" aria-label="Reconstructions, generated samples and the loss terms"></canvas></figure></div>
 ${uHow('va', ['Encoder: turn a picture into a cloud on the map: a centre and a width for each of the two code numbers.', 'Draw one random point from that cloud.', 'Decoder: turn that point back into a picture.', 'Loss, part 1: how different is the output from the original? Part 2: how far is the cloud from a standard bell at the centre of the map?', 'Adjust all weights to lower the sum, and repeat. To generate, skip the encoder: pick a random point from the standard bell and decode it.'])}
 <div class="controls">${uBtns('va', 'Train', 'Ten epochs', 'epochs')}
 ${grp('The model', slider('va-beta', 'Weight β on keeping the clouds near the centre', 0, 6, 0.1, 1) + `<div class="row"><button class="btn" id="va-samp" type="button">Draw New Random Pictures</button></div>`)}
 ${grp('The data', slider('va-flip', 'Noise: chance that a pixel is flipped', 0, 0.2, 0.01, 0.04) + `<label class="check"><input type="checkbox" id="va-col" checked> Show the true letter (the model never sees it)</label>`)}</div>
 <dl class="read">${rd('ep', 'Epochs')}${rd('rec', 'Reconstruction error per picture', 'hl2')}${rd('kl', 'Distance of the clouds from the centre (KL)', 'hl3')}${rd('in', 'Codes that fall inside the outer ring', 'hl')}${rd('st', 'Status')}</dl>`,
    tryit:
      'Train with β = 1 and look at the row of generated pictures: most are recognisable letters. Set β to 0 and train again. That is a plain autoencoder: the codes fly far outside the rings, and random draws decode to smudges. Now set β to 6: the clouds collapse onto the centre, everything decodes to the same blur. Somewhere between, the map is both organised and full. That tension is the whole idea.',
    math: String.raw`<h4>Intuition</h4><p>Imagine a library where every book has a shelf position. In a badly run library each book sits at an exact spot and the gaps between them are empty: reach in at random and you grab air. In a well-run library each book has a small zone, the zones are packed side by side around the entrance, and similar books are neighbours. Now any spot you reach for holds a book, and walking along a shelf takes you smoothly from one subject to the next. A variational autoencoder (VAE) trains the librarian to shelve like that.</p>
 <h4>The model</h4><p>The encoder maps a picture \(\mathbf x\) to the parameters of a Gaussian over codes, and one code is sampled from it:</p>$$q(\mathbf z\mid\mathbf x)=\mathcal N\big(\boldsymbol\mu(\mathbf x),\operatorname{diag}\boldsymbol\sigma^2(\mathbf x)\big),\qquad\mathbf z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\varepsilon,\quad\boldsymbol\varepsilon\sim\mathcal N(\mathbf 0,\mathbf I).$$<p>Writing the sample as \(\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\varepsilon\) is the <b>reparameterisation trick</b>: the randomness sits in \(\boldsymbol\varepsilon\), outside the network, so gradients can pass through \(\boldsymbol\mu\) and \(\boldsymbol\sigma\). The decoder maps \(\mathbf z\) to a picture.</p>
 <h4>The loss</h4>$$\mathcal L=\underbrace{-\mathbb E_{q}\big[\log p(\mathbf x\mid\mathbf z)\big]}_{\text{reconstruction}}+\beta\,\underbrace{\mathrm{KL}\big(q(\mathbf z\mid\mathbf x)\,\Vert\,\mathcal N(\mathbf 0,\mathbf I)\big)}_{\text{keep the clouds near the centre}},\qquad\mathrm{KL}=\tfrac12\sum_j\big(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\big).$$<p>The first term wants each picture's cloud to be small and far from the others, so that pictures can be told apart. The second wants every cloud to be the same standard bell at the centre. Neither wins. The compromise is clouds that overlap a little and together fill the bell, which is what makes random draws from \(\mathcal N(\mathbf 0,\mathbf I)\) decode well. With \(\beta=1\) the negative of \(\mathcal L\) is the <b>evidence lower bound</b> (ELBO) on \(\log p(\mathbf x)\).</p>
 <h4>Generating</h4><p>Draw \(\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)\) and run the decoder. No picture is needed as input. Because neighbouring codes decode to similar pictures, moving along a line in code space morphs one picture into another.</p>
 <h4>The trade-off, and what came next</h4><p>VAE samples tend to be blurry: the decoder hedges across everything a code might mean. Raising \(\beta\) organises the map but loses detail; lowering it does the reverse. VAEs are still everywhere, mostly as compressors: latent diffusion models such as Stable Diffusion run their diffusion inside the code space of a VAE, and decode at the end.</p>
 <h4>What this page trains</h4><p>A real VAE on 150 letter pictures of 35 pixels, with a two-number code, encoder \(35\to24\to(\boldsymbol\mu,\log\boldsymbol\sigma^2)\) and decoder \(2\to24\to35\), trained by backpropagation with Adam until the loss stops falling.</p>`,
    q: 'With β = 0 the reconstruction error is lower than with β = 1. Why would anyone accept the higher error?',
    a: 'Because the purpose is to generate, not to copy. With β = 0 nothing ties the codes to the centre, so they spread wherever is convenient and leave gaps. A random draw from the standard bell then lands where the decoder was never trained, and decodes to rubbish. The KL term costs some reconstruction accuracy and buys a code space in which every region means something.',
    init(root) {
      const c = $('#va-c', root),
        cr = $('#va-r', root),
        live = $('#va-live', root),
        S = { beta: 1, flip: 0.04, col: true, seed: 1 },
        R = 3.2;
      let X,
        XF,
        lab,
        LT,
        enc,
        dec,
        ps,
        ts = 0,
        RC = [],
        KL = [],
        conv = false,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        MU = [],
        SD = [],
        pr = [0.5, 0.5],
        zs = [],
        drag = false;
      const gen = () => {
          const d = letterData(S.seed, S.flip);
          X = d.X.map((x) => x.map((v) => clamp(Math.round(v), 0, 1)));
          lab = d.lab;
          LT = d.CH;
          XF = new Float64Array(X.length * 35);
          X.forEach((x, i) =>
            x.forEach((v, j) => {
              XF[i * 35 + j] = v;
            }),
          );
        },
        newz = () => {
          const rn = randnFrom(Math.random);
          zs = Array.from({ length: 8 }, () => [rn(), rn()]);
        };
      const build = () => {
        const q = randnFrom(rng(run * 7 + 2));
        enc = AD.mlp([35, 24, 4], q, 'tanh');
        dec = AD.mlp([2, 24, 35], q, 'tanh');
        ps = enc.ps.concat(dec.ps);
        ts = 0;
        RC = [];
        KL = [];
        conv = false;
        cu = 0;
        codes();
        msg =
          'Random weights. Every picture’s cloud sits near the middle and everything decodes to grey. Press Train.';
      };
      const codes = () => {
        const o = enc.fwd(AD.input(XF, [X.length, 35]));
        AD.clear();
        MU = [];
        SD = [];
        for (let i = 0; i < X.length; i++) {
          MU.push([o.v[i * 4], o.v[i * 4 + 1]]);
          SD.push([Math.exp(0.5 * o.v[i * 4 + 2]), Math.exp(0.5 * o.v[i * 4 + 3])]);
        }
      };
      const decode = (z) => {
        const o = dec.fwd(AD.input(z, [1, 2]));
        AD.clear();
        return Float64Array.from(o.v, sig);
      };
      const train = (k) => {
        const n = X.length,
          rn = randnFrom(Math.random);
        for (let q = 0; q < k && !conv; q++) {
          const h = enc.fwd(AD.input(XF, [n, 35])),
            mu = AD.cols(h, 0, 2),
            lv = AD.cols(h, 2, 4),
            z = AD.reparam(
              mu,
              lv,
              Float64Array.from({ length: 2 * n }, () => rn()),
            ),
            rec = AD.bce(dec.fwd(z), XF),
            kl = AD.kl(mu, lv, n),
            loss = AD.wsum(rec, kl, S.beta / 35);
          RC.push(rec.v[0] * 35);
          KL.push(kl.v[0]);
          AD.backward(loss);
          AD.adam(ps, 0.006, ++ts, 1);
          if (
            (ts > 300 &&
              Math.abs(RC[ts - 1] + S.beta * KL[ts - 1] - RC[ts - 201] - S.beta * KL[ts - 201]) <
                0.01 * (RC[ts - 1] + S.beta * KL[ts - 1])) ||
            ts >= 2500
          )
            conv = true;
        }
        codes();
        cu = conv ? 4 : ts % 4;
        msg =
          'Epoch ' +
          ts +
          ': every picture became a cloud, one point was drawn from each and decoded. Reconstruction error ' +
          f2(RC[ts - 1]) +
          ' per picture; distance of the clouds from the centre ' +
          f2(KL[ts - 1]) +
          '.' +
          (conv
            ? ' Converged: the total has stopped falling. The second row on the right shows pictures made from random points.'
            : '');
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
          const { g, w, h } = fit(c),
            sc = (Math.min(w, h) / 2 / R) * 0.96,
            X2 = (v) => w / 2 + v * sc,
            Y2 = (v) => h / 2 - v * sc,
            NG = 9;
          for (let a = 0; a < NG; a++)
            for (let b = 0; b < NG; b++) {
              const zx = -R + (2 * R * (b + 0.5)) / NG,
                zy = R - (2 * R * (a + 0.5)) / NG;
              glyph(g, decode([zx, zy]), X2(zx) - 9, Y2(zy) - 12, 3.6, t, 0.22);
            }
          [1, 2].forEach((r) => {
            g.strokeStyle = mix(t.gold, t.gold, 0, r === 1 ? 0.9 : 0.5);
            g.lineWidth = 1.6;
            g.setLineDash(r === 2 ? [5, 5] : []);
            g.beginPath();
            g.arc(X2(0), Y2(0), r * sc, 0, 7);
            g.stroke();
          });
          g.setLineDash([]);
          MU.forEach((m, i) => {
            const x = X2(m[0]),
              y = Y2(m[1]);
            if (x < 0 || y < 0 || x > w || y > h) return;
            g.fillStyle = mix(col[lab[i] % 8], col[lab[i] % 8], 0, 0.12);
            g.beginPath();
            g.ellipse(x, y, Math.max(1, SD[i][0] * sc), Math.max(1, SD[i][1] * sc), 0, 0, 7);
            g.fill();
            if (S.col) {
              g.font = '600 12px "Hanken Grotesk",sans-serif';
              g.fillStyle = col[lab[i] % 8];
              g.textAlign = 'center';
              g.fillText(LT[lab[i]], x, y + 4);
            } else {
              g.fillStyle = mix(t.ink, t.ink, 0, 0.6);
              g.beginPath();
              g.arc(x, y, 2.5, 0, 7);
              g.fill();
            }
          });
          markTP(g, X2(pr[0]), Y2(pr[1]), t, false);
          g.font = MONO;
          g.fillStyle = t.cut;
          g.textAlign = 'left';
          g.fillText('rings: where random draws come from (1 and 2 standard deviations)', 8, h - 8);
          c._g = { w, h, sc };
        }
        {
          const { g, w, h } = fit(cr),
            cz = Math.min((w - 60) / 8 / 6.2, (h * 0.15) / 7.4),
            gw = cz * 6.2,
            x0 = 50;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('in', 6, 12 + 3.5 * cz);
          g.fillText('out', 6, 18 + 10.5 * cz);
          g.fillStyle = t.cut;
          g.fillText('new', 6, 26 + 17.5 * cz);
          const o = enc.fwd(AD.input(XF, [X.length, 35]));
          AD.clear();
          for (let k = 0; k < 6; k++) {
            const i = k * 25;
            glyph(g, X[i], x0 + k * gw, 6, cz, t);
            glyph(g, decode([o.v[i * 4], o.v[i * 4 + 1]]), x0 + k * gw, 12 + 7 * cz, cz, t);
          }
          glyph(g, decode(pr), x0 + 6.6 * gw, 6, cz * 1.5, t);
          g.fillStyle = t.muted;
          g.fillText('diamond', x0 + 6.6 * gw, 18 + 10.5 * cz + 4);
          zs.forEach((z, k) => {
            glyph(g, decode(z), x0 + k * gw, 20 + 14 * cz, cz, t);
            g.strokeStyle = t.gold;
            g.lineWidth = 1;
            g.strokeRect(x0 + k * gw - 1.5, 18.5 + 14 * cz, 5 * cz + 2, 7 * cz + 2);
          });
          const y3 = 34 + 21 * cz,
            U = uLine(g, t, 0, y3, w, h - y3, RC, {
              title: 'reconstruction error (red) and β × cloud distance (green)',
              n: Math.max(200, RC.length),
              col: t.ok,
              ym: Math.max(24, RC[0] || 24),
            });
          g.strokeStyle = t.accent;
          g.lineWidth = 2;
          g.beginPath();
          KL.forEach((v, i) =>
            i ? g.lineTo(U.X(i), U.Y(S.beta * v)) : g.moveTo(U.X(i), U.Y(S.beta * v)),
          );
          g.stroke();
        }
        live.textContent = msg;
        uCur(root, 'va', cu);
        setR(root, 'ep', String(ts));
        setR(root, 'rec', ts ? f2(RC[ts - 1]) : '–');
        setR(root, 'kl', ts ? f2(KL[ts - 1]) : '–');
        setR(root, 'in', MU.filter((m) => Math.hypot(m[0], m[1]) < 2).length + ' of ' + MU.length);
        setR(root, 'st', conv ? 'Converged' : ts ? 'Training' : 'Not started');
      };
      gen();
      newz();
      build();
      if (RM) while (!conv) train(50);
      const redo = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      const anim = Anim(() => {
        fr++;
        if (ts < 30 && fr % 4) return;
        train(Math.min(ts < 30 ? 1 : ts < 300 ? 2 : 5, chk.remaining()));
        draw();
        chk();
      });
      pb = playBtn($('#va-play', root), anim);
      chk = runCtl(pb, root, 'va', {
        done: () => conv,
        restart: () => {
          run++;
          build();
        },
        count: () => ts,
        what: 'epochs',
      });
      $('#va-step', root).onclick = () => {
        pb.stop();
        if (conv) {
          run++;
          build();
        } else train(10);
        draw();
      };
      $('#va-reset', root).onclick = redo;
      $('#va-samp', root).onclick = () => {
        newz();
        if (!pb.on) draw();
      };
      bind(
        root,
        'va-beta',
        (v) => {
          S.beta = v;
          pb.stop();
          run++;
          build();
          draw();
          pb.start();
        },
        f2,
      );
      bind(root, 'va-flip', (v) => {
        S.flip = v;
        gen();
        redo();
      });
      $('#va-col', root).onchange = (e) => {
        S.col = e.target.checked;
        draw();
      };
      const mv = (e) => {
        const q = c._g;
        if (!q) return;
        const r = c.getBoundingClientRect();
        pr = [
          clamp((e.clientX - r.left - q.w / 2) / q.sc, -R, R),
          clamp(-(e.clientY - r.top - q.h / 2) / q.sc, -R, R),
        ];
        if (!pb.on) draw();
      };
      c.style.touchAction = 'none';
      c.addEventListener('pointerdown', (e) => {
        drag = true;
        c.setPointerCapture(e.pointerId);
        mv(e);
      });
      c.addEventListener('pointermove', (e) => {
        if (drag) mv(e);
      });
      ['pointerup', 'pointercancel'].forEach((n2) =>
        c.addEventListener(n2, () => {
          drag = false;
        }),
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
  },

  {
    id: 'gan',
    tab: 'GANs',
    title: 'GANs: a forger and a detective train each other',
    lede: 'Two networks play against each other. The generator turns random numbers into fake data and has never seen the real thing. The discriminator sees both and tries to tell them apart. Each time the detective gets better, the forger has to improve, and the other way round. Watch the gold fakes chase the dark real points across the plane, and watch what can go wrong.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Dark dots: real data. Gold dots: what the generator makes right now. Background: the discriminator’s opinion of every spot (green “looks real”, red “looks fake”)</figcaption><canvas class="short" id="ga-c" role="img" aria-label="Real and generated points with the discriminator’s decision field"></canvas></figure>
 <figure class="fig"><figcaption>Top: the two losses. Bottom: how often the discriminator is right about real points and about fakes</figcaption><canvas class="short" id="ga-r" role="img" aria-label="Generator and discriminator losses and discriminator accuracy"></canvas></figure></div>
 ${uHow('ga', ['The generator turns a batch of random numbers into points. At first they land anywhere.', 'Discriminator’s turn: it is shown real points and fakes, told which is which, and adjusts to tell them apart better.', 'Generator’s turn: it makes new fakes and is told only how real the discriminator found them. It adjusts to fool it more.', 'Repeat. The background changes as the detective learns; the gold dots move as the forger responds.', 'There is no finish line: the aim is a balance where the fakes sit on the real data and the detective can only guess.'])}
 <div class="controls">${uBtns('ga', 'Train', '20 steps', 'steps')}
 ${grp('The data', `<label class="ctl" for="ga-ds"><span>Real data</span><select id="ga-ds"><option value="ring">Six separate clusters in a ring</option><option value="moons">Two moons</option><option value="line">One blob</option></select></label>`)}
 ${grp('The model', slider('ga-dk', 'Discriminator turns for each generator turn', 1, 5, 1, 1) + slider('ga-lr', 'Learning rate (×0.001)', 0.5, 6, 0.1, 2))}</div>
 <dl class="read">${rd('step', 'Steps')}${rd('ld', 'Discriminator loss', 'hl3')}${rd('lg', 'Generator loss', 'hl2')}${rd('dr', 'Real points the discriminator calls real')}${rd('df', 'Fakes the discriminator calls real')}${rd('md', 'Clusters the generator covers', 'hl')}</dl>`,
    tryit:
      'Run it on the six clusters and count how many the gold dots cover as training goes on. They often lock onto two or three first, then jump: that is mode collapse, the best-known GAN failure. Watch the two loss curves: neither goes to zero, and that is correct. Raise the learning rate to 6 and train again. Then try One Blob, where there is nothing to collapse onto.',
    math: String.raw`<h4>Intuition</h4><p>A forger paints fake banknotes and a detective inspects them. The forger never sees a real note. All they learn is what the detective says: "fake, the ink is wrong". They fix the ink. The detective, now seeing better fakes, learns to check the paper. The forger fixes the paper. Round after round both improve, and in the end the fakes are indistinguishable from the real thing. Neither could have got there alone. A generative adversarial network (GAN) sets up that contest between two networks.</p>
 <h4>The two players</h4><p>The <b>generator</b> \(G\) maps random noise \(\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)\) to a data point \(G(\mathbf z)\). The <b>discriminator</b> \(D\) maps a point to the probability that it is real. They play</p>$$\min_G\max_D\;\;\mathbb E_{\mathbf x\sim\text{data}}\big[\log D(\mathbf x)\big]+\mathbb E_{\mathbf z}\big[\log\big(1-D(G(\mathbf z))\big)\big].$$<p>\(D\) wants both terms large: real points scored near 1, fakes near 0. \(G\) wants the second term small.</p>
 <h4>Training</h4><p>Alternate two steps. <b>Discriminator step:</b> ordinary classification, real against fake, with \(G\) held fixed. <b>Generator step:</b> hold \(D\) fixed and backpropagate through it into \(G\). In practice \(G\) maximises \(\log D(G(\mathbf z))\), which gives stronger gradients early on than minimising \(\log(1-D(G(\mathbf z)))\). The generator's only teaching signal is the gradient of the discriminator's opinion: which way should each fake move to look more real?</p>
 <h4>The ideal end point</h4><p>For a fixed generator the best discriminator is \(D^*(\mathbf x)=\dfrac{p_{\text{data}}(\mathbf x)}{p_{\text{data}}(\mathbf x)+p_G(\mathbf x)}\). If the generator matches the data exactly, \(D^*=\tfrac12\) everywhere: the detective is reduced to tossing a coin, and both losses settle near \(\log2\approx0.69\) and \(2\log2\approx1.39\). So unlike every other lesson on this site, a falling loss is not the goal, and there is no test for convergence. This page stops after a fixed number of steps.</p>
 <h4>What goes wrong</h4><ul><li><b>Mode collapse.</b> The generator finds a few outputs that fool the detective and produces only those. The detective catches up, the generator jumps elsewhere, and they chase each other. Nothing in the objective rewards covering all of the data.</li><li><b>Instability.</b> Two networks optimising against each other can oscillate or diverge, and the balance is sensitive to learning rates.</li><li><b>Vanishing signal.</b> If the detective becomes too good, its opinion is flat near the fakes and the generator learns nothing.</li></ul>
 <h4>Where GANs stand</h4><p>From 2014 to about 2020 GANs produced the sharpest generated images, and they are still used where speed matters, since generation is one forward pass. Diffusion models have largely replaced them for images because they train stably and cover the data better.</p>
 <h4>What this page trains</h4><p>Two real networks, each with two hidden layers of 32 neurons, trained with Adam. The generator maps two random numbers to a point in the plane.</p>`,
    q: 'After long training the discriminator’s accuracy is about 50% on both real and fake points. Has training failed?',
    a: 'No, that is the best possible outcome. If the generator’s points are distributed exactly like the real data, no rule can tell them apart, and the best the discriminator can do is guess. A discriminator that stays near 100% accuracy means the generator is still producing detectable fakes; one near 50% means either success or, less happily, that the discriminator is too weak to notice differences that are there.',
    init(root) {
      const c = $('#ga-c', root),
        cr = $('#ga-r', root),
        live = $('#ga-live', root),
        S = { ds: 'ring', dk: 1, lr: 2 },
        MAXS = 3000,
        B = 64,
        RN = randnFrom(Math.random);
      let G,
        D,
        ts = 0,
        tg = 0,
        td = 0,
        LD = [],
        LG = [],
        AR = [],
        AF = [],
        fake = [],
        real = [],
        done = false,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1;
      const samp = () => {
        if (S.ds === 'ring') {
          const k = Math.floor(Math.random() * 6),
            a = (k / 6) * 2 * Math.PI;
          return [1.3 * Math.cos(a) + 0.09 * RN(), 1.3 * Math.sin(a) + 0.09 * RN()];
        }
        if (S.ds === 'moons') {
          const a = Math.random() * Math.PI;
          return Math.random() < 0.5
            ? [Math.cos(a) * 1.1 - 0.5 + 0.07 * RN(), -0.3 + Math.sin(a) * 1.0 + 0.07 * RN()]
            : [0.5 - Math.cos(a) * 1.1 + 0.07 * RN(), 0.3 - Math.sin(a) * 1.0 + 0.07 * RN()];
        }
        return [0.5 + 0.35 * RN(), 0.3 + 0.25 * RN()];
      };
      const build = () => {
        G = AD.mlp([2, 32, 32, 2], randnFrom(rng(run * 3 + 1)), 'relu');
        D = AD.mlp([2, 32, 32, 1], randnFrom(rng(run * 3 + 2)), 'relu');
        ts = tg = td = 0;
        LD = [];
        LG = [];
        AR = [];
        AF = [];
        done = false;
        cu = 0;
        real = Array.from({ length: 260 }, samp);
        view();
        msg =
          'The generator has random weights and has never seen the data: its gold dots are a clump in the wrong place. Press Train.';
      };
      const view = () => {
        const z = Float64Array.from({ length: 400 }, () => RN()),
          f = G.fwd(AD.input(z, [200, 2]));
        AD.clear();
        fake = [];
        for (let i = 0; i < 200; i++) fake.push([f.v[2 * i], f.v[2 * i + 1]]);
      };
      const dOn = (P) => {
        const x = new Float64Array(P.length * 2);
        P.forEach((p, i) => {
          x[2 * i] = p[0];
          x[2 * i + 1] = p[1];
        });
        const o = D.fwd(AD.input(x, [P.length, 2]));
        AD.clear();
        return o.v;
      };
      const modes = () => {
        if (S.ds !== 'ring') return null;
        const h = new Set();
        fake.forEach((p) => {
          for (let k = 0; k < 6; k++) {
            const a = (k / 6) * 2 * Math.PI;
            if (Math.hypot(p[0] - 1.3 * Math.cos(a), p[1] - 1.3 * Math.sin(a)) < 0.32) h.add(k);
          }
        });
        return h.size;
      };
      const train = (k) => {
        const ones = new Float64Array(B).fill(1),
          zeros = new Float64Array(B),
          lr = S.lr * 0.001;
        for (let q = 0; q < k && !done; q++) {
          let ld = 0;
          for (let d = 0; d < S.dk; d++) {
            const z = Float64Array.from({ length: 2 * B }, () => RN()),
              rl = new Float64Array(2 * B);
            for (let i = 0; i < B; i++) {
              const p = samp();
              rl[2 * i] = p[0];
              rl[2 * i + 1] = p[1];
            }
            const fk = G.fwd(AD.input(z, [B, 2]));
            AD.clear();
            let l = AD.bce(D.fwd(AD.input(rl, [B, 2])), ones);
            ld = l.v[0];
            AD.backward(l);
            l = AD.bce(D.fwd(AD.input(fk.v, [B, 2])), zeros);
            ld += l.v[0];
            AD.backward(l);
            AD.adam(D.ps, lr, ++td, 1);
          }
          const z2 = Float64Array.from({ length: 2 * B }, () => RN()),
            l = AD.bce(D.fwd(G.fwd(AD.input(z2, [B, 2]))), ones);
          AD.backward(l);
          AD.adam(G.ps, lr, ++tg, 1);
          for (const p of D.ps) p.g.fill(0);
          ts++;
          LD.push(ld);
          LG.push(l.v[0]);
          if (ts >= MAXS) done = true;
        }
        view();
        const a = dOn(real),
          b = dOn(fake);
        AR.push(a.filter((v) => v > 0).length / a.length);
        AF.push(b.filter((v) => v > 0).length / b.length);
        cu = done ? 4 : 1 + (ts % 3);
        const m = modes();
        msg =
          'Step ' +
          ts +
          ': the discriminator trained on 64 real and 64 fake points, then the generator trained to fool it. Right now the discriminator calls ' +
          pct(AR[AR.length - 1]) +
          ' of real points real and is fooled by ' +
          pct(AF[AF.length - 1]) +
          ' of the fakes.' +
          (m !== null ? ' The gold dots cover ' + m + ' of the 6 clusters.' : '') +
          (done
            ? ' It has had its ' +
              MAXS +
              ' steps and stops here. A GAN has no loss that says “finished”.'
            : '');
      };
      const draw = () => {
        const t = T();
        {
          const V = uView(c),
            { g, w, h, X, Y, s } = V,
            cs = 14,
            nx = Math.ceil(w / cs),
            ny = Math.ceil(h / cs),
            P = [];
          for (let a = 0; a < ny; a++)
            for (let b = 0; b < nx; b++)
              P.push([((b + 0.5) * cs - w / 2) / s / 1.9, -((a + 0.5) * cs - h / 2) / s / 1.9]);
          const dv = dOn(P);
          P.forEach((p, i) => {
            const pr2 = sig(dv[i]);
            g.fillStyle = mix(
              pr2 >= 0.5 ? t.accent : t.ok,
              t.bg,
              0,
              0.05 + 0.3 * Math.abs(pr2 - 0.5) * 2,
            );
            g.fillRect((i % nx) * cs, Math.floor(i / nx) * cs, cs, cs);
          });
          const X2 = (v) => X(v * 1.9),
            Y2 = (v) => Y(v * 1.9);
          g.fillStyle = mix(t.ink, t.ink, 0, 0.75);
          real.forEach((p) => {
            g.beginPath();
            g.arc(X2(p[0]), Y2(p[1]), 2.6, 0, 7);
            g.fill();
          });
          g.fillStyle = t.gold;
          g.strokeStyle = t.bg;
          g.lineWidth = 0.8;
          fake.forEach((p) => {
            g.beginPath();
            g.arc(X2(p[0]), Y2(p[1]), 3.4, 0, 7);
            g.fill();
            g.stroke();
          });
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5,
            ym = Math.max(2.5, ...LG.slice(-400), ...LD.slice(-400)),
            U = uLine(
              g,
              t,
              0,
              0,
              w,
              h1,
              LD.map((v) => Math.min(v, ym)),
              {
                title: 'losses: discriminator (green), generator (red)',
                n: Math.max(300, LD.length),
                col: t.accent,
                ym,
              },
            );
          g.strokeStyle = t.ok;
          g.lineWidth = 1.8;
          g.beginPath();
          LG.forEach((v, i) =>
            i ? g.lineTo(U.X(i), U.Y(Math.min(v, ym))) : g.moveTo(U.X(i), U.Y(Math.min(v, ym))),
          );
          g.stroke();
          const U2 = uLine(g, t, 0, h1, w, h - h1, AR, {
            title: 'discriminator says “real” for: real points (green), fakes (gold)',
            n: Math.max(100, AR.length),
            col: t.accent,
            ym: 1,
          });
          g.strokeStyle = t.gold;
          g.lineWidth = 2;
          g.beginPath();
          AF.forEach((v, i) => (i ? g.lineTo(U2.X(i), U2.Y(v)) : g.moveTo(U2.X(i), U2.Y(v))));
          g.stroke();
          g.setLineDash([4, 4]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(U2.X(0), U2.Y(0.5));
          g.lineTo(w - 12, U2.Y(0.5));
          g.stroke();
          g.setLineDash([]);
        }
        live.textContent = msg;
        uCur(root, 'ga', cu);
        const m = modes();
        setR(root, 'step', ts + ' of ' + MAXS);
        setR(root, 'ld', ts ? f2(LD[ts - 1]) : '–');
        setR(root, 'lg', ts ? f2(LG[ts - 1]) : '–');
        setR(root, 'dr', AR.length ? pct(AR[AR.length - 1]) : '–');
        setR(root, 'df', AF.length ? pct(AF[AF.length - 1]) : '–');
        setR(root, 'md', m === null ? '(only for the ring)' : m + ' of 6');
      };
      build();
      if (RM) train(MAXS);
      const redo = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      const anim = Anim(() => {
        fr++;
        if (ts < 40 && fr % 3) return;
        train(Math.min(ts < 40 ? 1 : 4, chk.remaining()));
        draw();
        chk();
      });
      pb = playBtn($('#ga-play', root), anim);
      chk = runCtl(pb, root, 'ga', {
        done: () => done,
        restart: () => {
          run++;
          build();
        },
        count: () => ts,
        what: 'steps',
      });
      $('#ga-step', root).onclick = () => {
        pb.stop();
        if (done) {
          run++;
          build();
        } else train(20);
        draw();
      };
      $('#ga-reset', root).onclick = redo;
      $('#ga-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        redo();
        pb.start();
      };
      bind(
        root,
        'ga-dk',
        (v) => {
          S.dk = v;
        },
        (v) => String(v),
      );
      bind(
        root,
        'ga-lr',
        (v) => {
          S.lr = v;
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
  },
  {
    id: 'ddpm',
    tab: 'DDPM and IDDPM on a picture',
    title: 'DDPM and improved DDPM: bury a picture in noise, then bring it back',
    lede: 'Stir a drop of ink into water and it spreads until nothing of the drop is left. That direction is easy. A diffusion model learns the other direction: it removes a little noise at a time until a picture appears. Choose a picture, watch it dissolve, then watch the denoiser bring it back. Whether the same picture returns depends on how much of it survived, and that is decided by the noise schedule, which you can redesign here: the original DDPM schedule, the improved one from IDDPM, or your own.',
    stage: `<div class="row seg" id="dp-dir" role="group" aria-label="What to do"><button class="btn" type="button" data-v="round">Noise This Picture, Then Denoise It</button><button class="btn" type="button" data-v="gen">Generate a Picture from Pure Static</button></div>
 <div class="stage"><figure class="fig"><figcaption id="dp-cap">The picture at the current step. The small picture in the corner is the original</figcaption><canvas class="sq" id="dp-c" role="img" aria-label="The image at the current diffusion step"></canvas></figure>
 <figure class="fig"><figcaption>Top: the denoiser’s best guess at the clean picture, and which known picture it thinks this is. Middle: the same picture a quarter, half, three quarters and all the way through, under your schedule and under the alternative. Bottom: how much picture and how much noise is in the mix at each step</figcaption><canvas class="sq" id="dp-r" role="img" aria-label="Denoiser estimate, belief in each known picture, schedule comparison strips and the noise schedule curves"></canvas></figure></div>
 ${uHow('dp', ['Forward, one step: shrink the picture slightly toward grey and add a little fresh random noise. The noise schedule says how much at each step.', 'Repeat up to the turning point you chose. The further you go, the less of the original is left in the mix.', 'Reverse: ask the denoiser, “given this noisy picture, what was the clean one?” Its answer blends the pictures it knows, weighted by how well each one fits.', 'Step toward that guess and add back a little fresh noise. The reverse-noise setting says how much.', 'Repeat down to step 0. If enough of the original survived at the turning point, the same picture comes back.', 'Generate: start from pure static instead. There is no original to recover, so any picture the model knows can appear.'])}
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="dp-play" type="button">Run</button><button class="btn" id="dp-step" type="button">One step</button><button class="btn" id="dp-reset" type="button">Start over</button></div>` + slider('dp-t', 'Jump to a forward step', 0, 100, 1, 0) + slider('dp-turn', 'Turn back after this share of the steps', 10, 100, 5, 60))}
 ${grp('The noise schedule', `<div class="row seg" id="dp-sch" role="group" aria-label="Noise schedule"><button class="btn" type="button" data-v="lin">Linear (DDPM)</button><button class="btn" type="button" data-v="cos">Cosine (IDDPM)</button><button class="btn" type="button" data-v="con">Constant</button></div>` + slider('dp-T', 'Number of steps T', 20, 200, 10, 100) + slider('dp-b0', 'Linear: noise added at the first step, β₁', 0.0005, 0.02, 0.0005, 0.001) + slider('dp-b1', 'Linear and constant: noise at the last step, β_T', 0.02, 0.5, 0.01, 0.2))}
 ${grp('The reverse process', slider('dp-v', 'Noise put back at each reverse step (0 = least, 1 = most)', 0, 1, 0.05, 0) + slider('dp-rs', 'Reverse steps used, as a share of the forward steps', 5, 100, 5, 100) + `<div class="row"><button class="btn" id="dp-new" type="button">New Random Noise</button></div>`)}
 ${grp('The pictures the model knows', `<label class="ctl" for="dp-pic"><span>Picture to start from</span><select id="dp-pic"></select></label><div class="row"><label class="btn file">Upload a Picture<input type="file" id="dp-file" accept="image/*"></label></div><p class="status" id="dp-st" role="status"></p>`)}</div>
 <dl class="read">${rd('t', 'Step')}${rd('sig', 'Share of the original left (√ᾱ)', 'hl3')}${rd('nz', 'Share of noise (√(1−ᾱ))', 'hl2')}${rd('to', 'The denoiser thinks this is', 'hl')}${rd('rc', 'Steps on which the denoiser can still recognise the original: your schedule / the alternative')}${rd('res', 'Result')}</dl>`,
    tryit:
      'Choose the sunflower and press Run. The picture is noised for 60% of the steps of the linear DDPM schedule and then denoised: the sunflower comes back. Now move “Turn back” to 100% and run a few times. Sometimes a different picture comes back. Read the two share readouts at the turning point to see why. Next compare the two strips in the middle of the right-hand picture: under the linear schedule the picture is already static halfway through, and under the cosine schedule it fades evenly. That waste is what IDDPM fixed. Design your own schedule with the β sliders and watch the curves and the “still recognisable” readout. Then lower the reverse steps to 10%: does the denoiser still get home? Finally choose Generate a Picture from Pure Static and run it several times.',
    math: String.raw`<h4>Intuition</h4><p>A sculptor starts with a rough block and removes a little at a time until the statue appears. Nobody could carve a statue in a single blow, but each small cut is easy: look at what is there, decide what does not belong, take it off. A diffusion model generates the same way. It starts from a block of static and removes a little noise at each step. It learns what "does not belong" by first watching real pictures being buried in noise, where it knows exactly what noise was added. If you bury a picture only part of the way, enough of it shows through for the sculptor to recover that same picture. Bury it completely and the sculptor is free to carve any statue it knows.</p>
 <h4>The forward process: adding noise</h4><p>Let \(\mathbf x_0\) be a picture with pixel values scaled to \([-1,1]\). Each step mixes in a little Gaussian noise:</p>$$\mathbf x_t=\sqrt{1-\beta_t}\;\mathbf x_{t-1}+\sqrt{\beta_t}\;\boldsymbol\varepsilon_t,\qquad\boldsymbol\varepsilon_t\sim\mathcal N(\mathbf 0,\mathbf I).$$<p>The numbers \(\beta_1,\dots,\beta_T\) are the <b>noise schedule</b>. Because Gaussians combine neatly, any step can be reached from the original in one jump. With \(\bar\alpha_t=\prod_{s\le t}(1-\beta_s)\),</p>$$\mathbf x_t=\sqrt{\bar\alpha_t}\;\mathbf x_0+\sqrt{1-\bar\alpha_t}\;\boldsymbol\varepsilon .$$<p>\(\sqrt{\bar\alpha_t}\) is how much picture is left and \(\sqrt{1-\bar\alpha_t}\) how much noise: the two solid curves at the lower right. The step slider uses this formula to jump anywhere.</p>
 <h4>The reverse process: removing noise</h4><p>A network \(\boldsymbol\varepsilon_\theta(\mathbf x_t,t)\) is trained to predict the noise that was added, by minimising</p>$$\mathcal L_{\text{simple}}=\mathbb E_{\mathbf x_0,\,t,\,\boldsymbol\varepsilon}\Big[\big\lVert\boldsymbol\varepsilon-\boldsymbol\varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,\mathbf x_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon,\;t\big)\big\rVert^2\Big].$$<p>Predicting the noise is the same as predicting the clean picture, since \(\hat{\mathbf x}_0=(\mathbf x_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon_\theta)/\sqrt{\bar\alpha_t}\). One reverse step then draws \(\mathbf x_{t-1}\) from a Gaussian whose centre lies between the current noisy picture and the guess:</p>$$\mathbf x_{t-1}=\underbrace{\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat{\mathbf x}_0+\frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,\mathbf x_t}_{\text{mean}}+\sigma_t\,\mathbf z,\qquad\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I).$$<p>This is DDPM, denoising diffusion probabilistic models (Ho, Jain and Abbeel, 2020). They used a schedule in which \(\beta_t\) rises in a straight line, and \(\sigma_t^2=\beta_t\).</p>
 <h4>Why the picture you chose comes back, or does not</h4><p>The reverse process only sees the noisy picture. When you turn back early, \(\sqrt{\bar\alpha_t}\) is still well above zero, the original shows through, and the denoiser's guess is that picture. At the very end of a schedule \(\bar\alpha_T\) is practically zero: the noisy picture is static that would look the same whichever picture you began with. Nothing is left to say it was the sunflower, so the reverse process is free to end on any picture it knows. That is not a fault. It is exactly what makes diffusion a <b>generator</b>: begin from fresh static and a new sample appears. Starting the reverse process from a partly noised picture, as the first button does, is how diffusion models are used to edit or restore an existing image.</p>
 <h4>IDDPM, part 1: the cosine schedule</h4><p>Nichol and Dhariwal (2021) noticed that the linear schedule destroys small pictures far too quickly. Look at the middle strips: with the linear schedule the picture is gone about halfway through, so the remaining steps are spent turning static into static and teach the network very little. They proposed choosing \(\bar\alpha_t\) directly, as a smooth curve that falls gently at both ends:</p>$$\bar\alpha_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\!\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big),\qquad \beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}},$$<p>with a small offset \(s=0.008\) and \(\beta_t\) capped at 0.999. Information is removed at a nearly even pace from the first step to the last. The readout "steps on which the denoiser can still recognise the original" measures the difference. The ideal denoiser on this page sees through far more noise than your eye does, so judge the strips by eye and the readout by number.</p>
 <h4>IDDPM, part 2: how much noise to put back</h4><p>For the size \(\sigma_t\) of the fresh noise in the reverse step there are two natural choices. The largest sensible value is \(\sigma_t^2=\beta_t\). The smallest is the variance the step would have if the clean picture were known exactly,</p>$$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t .$$<p>DDPM fixed one of them. IDDPM lets the network output a number \(v\) between 0 and 1 for every step and pixel, and uses a blend of the two on a logarithmic scale:</p>$$\sigma_t^2=\exp\!\big(v\,\log\beta_t+(1-v)\,\log\tilde\beta_t\big).$$<p>On this page you set \(v\) by hand with the reverse-noise slider. The two ends differ most in the last few steps, where \(\tilde\beta_t\) goes to zero and \(\beta_t\) does not. To train \(v\), IDDPM adds a small multiple of the full likelihood bound to the simple loss, \(\mathcal L_{\text{hybrid}}=\mathcal L_{\text{simple}}+\lambda\,\mathcal L_{\text{vlb}}\).</p>
 <h4>IDDPM, part 3: fewer reverse steps</h4><p>With a well-chosen variance, sampling does not have to visit every step. Pick a shorter list of steps \(t_1>t_2>\dots\), treat each jump from \(t\) to \(s\) as one step with \(\beta'=1-\bar\alpha_t/\bar\alpha_s\), and apply the same formulas. A model trained with 4,000 steps could then be sampled in about 100. The "reverse steps used" slider does this. The DDIM and Samplers tab takes the idea further.</p>
 <h4>Why the first guesses are blurry</h4><p>The best possible answer to "what clean picture is behind this static?" is the average of all the pictures it could have been, weighted by how likely each is. At high noise many pictures are plausible, so the average is a blur. As noise is removed the candidates narrow and the guess sharpens. The fresh noise added at each step helps decide which candidate wins.</p>
 <h4>What this page computes</h4><p>The forward process, the schedules, the reverse mean and the reverse variance are exactly the formulas above. For the denoiser this page does not train a network. It uses the <b>mathematically ideal denoiser for a tiny data set</b>: when the model "knows" only the \(K\) pictures in the list, the best guess is</p>$$\hat{\mathbf x}_0=\sum_{k=1}^{K}w_k\,\mathbf x^{(k)},\qquad w_k\propto\exp\!\Big(-\frac{\lVert\mathbf x_t-\sqrt{\bar\alpha_t}\,\mathbf x^{(k)}\rVert^2}{2(1-\bar\alpha_t)}\Big),$$<p>the bars at the top right. Because it knows so few pictures, it can pick out the original from astonishingly little: a picture has thousands of pixels, and a faint trace spread over all of them adds up. A real diffusion model replaces this sum with a U-Net trained on millions of pictures. When such a model denoises a heavily noised photo it returns a picture like the original, not the original itself, and it can produce pictures that are in none of its training data. The next tab trains a denoising network on simple data. The built-in pictures are drawn by code, and an uploaded photo is shrunk to 32 × 32 pixels.</p>`,
    q: 'You noise the sunflower all the way to the last step of the linear schedule and denoise it, and a cat comes back. A classmate says the denoiser made a mistake. Did it?',
    a: 'No. At the last step of that schedule the share of the original left in the mix is practically zero, so the noisy picture is static that could have come from any picture with almost equal probability. The information that it was a sunflower has been destroyed, not hidden, and no denoiser, however good, can read what is no longer there. From that point the reverse process is generating a new sample, and which picture it lands on is decided by the random noise. Turn back earlier, or use a schedule that still keeps a trace at the turning point, and the sunflower returns.',
    init(root) {
      const c = $('#dp-c', root),
        cr = $('#dp-r', root),
        live = $('#dp-live', root),
        st = $('#dp-st', root),
        sel = $('#dp-pic', root),
        S = {
          mode: 'round',
          pic: 1,
          sch: 'lin',
          T: 100,
          b0: 0.001,
          b1: 0.2,
          turn: 60,
          v: 0,
          rs: 100,
        },
        N = 32,
        DM = N * N * 3,
        RN = randnFrom(Math.random),
        SN = { lin: 'linear', cos: 'cosine', con: 'constant' };
      let GAL = [],
        AB = [1],
        ABo = [1],
        x = new Float64Array(DM),
        x0h = null,
        wts = [],
        t = 0,
        phase = 'fwd',
        tau = [],
        ti = 0,
        K = 75,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        eps0 = null,
        wTurn = null,
        aTurn = 1,
        res = -1,
        rc = [NaN, NaN],
        tb = null;
      const fromCanvas = (o) => {
        const d = o.getContext('2d').getImageData(0, 0, N, N).data,
          v = new Float64Array(DM);
        for (let i = 0; i < N * N; i++)
          for (let ch = 0; ch < 3; ch++) v[i * 3 + ch] = d[i * 4 + ch] / 127.5 - 1;
        return v;
      };
      const builtin = () => {
        const mk = (fn) => {
          const o = document.createElement('canvas');
          o.width = o.height = N;
          const g = o.getContext('2d');
          fn(g);
          return fromCanvas(o);
        };
        return [
          [
            'A house in a field',
            mk((g) => {
              let gr = g.createLinearGradient(0, 0, 0, 20);
              gr.addColorStop(0, '#3f79c4');
              gr.addColorStop(1, '#bfe0f5');
              g.fillStyle = gr;
              g.fillRect(0, 0, 32, 32);
              g.fillStyle = '#ffd95a';
              g.beginPath();
              g.arc(25, 7, 4, 0, 7);
              g.fill();
              g.fillStyle = '#4f9a47';
              g.fillRect(0, 20, 32, 12);
              g.fillStyle = '#efe6d4';
              g.fillRect(8, 14, 12, 10);
              g.fillStyle = '#b5452f';
              g.beginPath();
              g.moveTo(6, 15);
              g.lineTo(14, 7);
              g.lineTo(22, 15);
              g.fill();
              g.fillStyle = '#5a3a22';
              g.fillRect(12, 18, 4, 6);
            }),
          ],
          [
            'A sunflower',
            mk((g) => {
              g.fillStyle = '#8fd0ef';
              g.fillRect(0, 0, 32, 32);
              g.fillStyle = '#3f8f3f';
              g.fillRect(15, 16, 2, 16);
              g.fillStyle = '#f6c52c';
              for (let k = 0; k < 10; k++) {
                const a = (k / 10) * 2 * Math.PI;
                g.beginPath();
                g.ellipse(16 + 7 * Math.cos(a), 12 + 7 * Math.sin(a), 4, 2.2, a, 0, 7);
                g.fill();
              }
              g.fillStyle = '#6b3f1d';
              g.beginPath();
              g.arc(16, 12, 5, 0, 7);
              g.fill();
            }),
          ],
          [
            'A sailing boat',
            mk((g) => {
              let gr = g.createLinearGradient(0, 0, 0, 20);
              gr.addColorStop(0, '#f6b26b');
              gr.addColorStop(1, '#ffe8b8');
              g.fillStyle = gr;
              g.fillRect(0, 0, 32, 32);
              g.fillStyle = '#1f5f8b';
              g.fillRect(0, 21, 32, 11);
              g.fillStyle = '#ffffff';
              g.beginPath();
              g.moveTo(16, 4);
              g.lineTo(16, 19);
              g.lineTo(26, 19);
              g.fill();
              g.fillStyle = '#e04a3a';
              g.beginPath();
              g.moveTo(14, 7);
              g.lineTo(14, 19);
              g.lineTo(7, 19);
              g.fill();
              g.fillStyle = '#4a2f1b';
              g.beginPath();
              g.moveTo(5, 20);
              g.lineTo(27, 20);
              g.lineTo(23, 25);
              g.lineTo(9, 25);
              g.fill();
            }),
          ],
          [
            'A cat’s face',
            mk((g) => {
              g.fillStyle = '#2d2a4a';
              g.fillRect(0, 0, 32, 32);
              g.fillStyle = '#e8963a';
              g.beginPath();
              g.moveTo(6, 12);
              g.lineTo(8, 3);
              g.lineTo(14, 9);
              g.moveTo(26, 12);
              g.lineTo(24, 3);
              g.lineTo(18, 9);
              g.fill();
              g.beginPath();
              g.arc(16, 17, 11, 0, 7);
              g.fill();
              g.fillStyle = '#fff';
              g.beginPath();
              g.arc(11, 15, 3, 0, 7);
              g.arc(21, 15, 3, 0, 7);
              g.fill();
              g.fillStyle = '#1c1c1c';
              g.beginPath();
              g.arc(11, 15, 1.4, 0, 7);
              g.arc(21, 15, 1.4, 0, 7);
              g.fill();
              g.fillStyle = '#d4607a';
              g.beginPath();
              g.moveTo(14.5, 20);
              g.lineTo(17.5, 20);
              g.lineTo(16, 22);
              g.fill();
            }),
          ],
        ];
      };
      const other = () => (S.sch === 'cos' ? 'lin' : 'cos');
      const mkAB = (sch) => {
        const T = S.T,
          a = [1];
        if (sch === 'cos') {
          const s = 0.008,
            f = (k) => Math.cos((((k / T + s) / (1 + s)) * Math.PI) / 2) ** 2;
          for (let k = 1; k <= T; k++)
            a.push(a[k - 1] * (1 - Math.min(0.999, 1 - f(k) / f(k - 1))));
        } else
          for (let k = 1; k <= T; k++) {
            const b = sch === 'lin' ? S.b0 + ((S.b1 - S.b0) * (k - 1)) / Math.max(1, T - 1) : S.b1;
            a.push(a[k - 1] * (1 - clamp(b, 1e-5, 0.999)));
          }
        return a;
      };
      const recog = (a) => {
        if (S.mode === 'gen' || !eps0) return NaN;
        const y = GAL[S.pic][1],
          D = [],
          C = [];
        for (const q of GAL) {
          let d = 0,
            cc = 0;
          const g = q[1];
          for (let i = 0; i < DM; i++) {
            const u = y[i] - g[i];
            d += u * u;
            cc += u * eps0[i];
          }
          D.push(d);
          C.push(cc);
        }
        let n = 0;
        for (let k = 1; k < a.length; k++) {
          const ak = a[k],
            lg = D.map(
              (d, j) =>
                -(ak * d + 2 * Math.sqrt(ak * (1 - ak)) * C[j]) / (2 * Math.max(1e-12, 1 - ak)),
            ),
            m = Math.max(...lg),
            z = lg.reduce((s, v) => s + Math.exp(v - m), 0);
          if (Math.exp(lg[S.pic] - m) / z > 0.5) n++;
        }
        return n;
      };
      const sched = () => {
        AB = mkAB(S.sch);
        ABo = mkAB(other());
        K = clamp(Math.round((S.turn / 100) * S.T), 1, S.T);
        if (tb) {
          tb.el.max = S.T;
        }
        rc = [recog(AB), recog(ABo)];
      };
      const fillSel = () => {
        sel.innerHTML = GAL.map((q, i) => '<option value="' + i + '">' + q[0] + '</option>').join(
          '',
        );
        sel.value = S.pic;
      };
      const denoise = () => {
        const a = AB[t],
          sa = Math.sqrt(a),
          lg = GAL.map((q) => {
            let s = 0;
            const y = q[1];
            for (let i = 0; i < DM; i++) {
              const d = x[i] - sa * y[i];
              s += d * d;
            }
            return -s / (2 * Math.max(1e-9, 1 - a));
          }),
          m = Math.max(...lg),
          e = lg.map((v) => Math.exp(v - m)),
          z = e.reduce((p, q) => p + q, 0);
        wts = e.map((v) => v / z);
        x0h = new Float64Array(DM);
        GAL.forEach((q, k) => {
          const w = wts[k];
          if (w > 1e-6) for (let i = 0; i < DM; i++) x0h[i] += w * q[1][i];
        });
      };
      const mkTau = (from) => {
        const R = Math.max(1, Math.round((S.rs / 100) * from)),
          o = [];
        for (let j = 0; j <= R; j++) {
          const v = Math.round((from * (R - j)) / R);
          if (!o.length || o[o.length - 1] !== v) o.push(v);
        }
        return o;
      };
      const start = (newNoise) => {
        if (newNoise || !eps0 || eps0.length !== DM)
          eps0 = Float64Array.from({ length: DM }, () => RN());
        sched();
        cu = 0;
        res = -1;
        wTurn = null;
        ti = 0;
        if (S.mode === 'round') {
          phase = 'fwd';
          t = 0;
          x = Float64Array.from(GAL[S.pic][1]);
          msg =
            'This is the clean picture, “' +
            GAL[S.pic][0] +
            '”, at step 0. Press Run: noise is added for ' +
            K +
            ' of the ' +
            S.T +
            ' steps of the ' +
            SN[S.sch] +
            ' schedule, then the denoiser works its way back.';
        } else {
          phase = 'rev';
          t = S.T;
          x = Float64Array.from({ length: DM }, () => RN());
          tau = mkTau(S.T);
          msg =
            'Pure static at step ' +
            S.T +
            '. It was not made from any picture, so there is nothing to recover. Press Run and see which of the ' +
            GAL.length +
            ' known pictures forms.' +
            (Math.sqrt(AB[S.T]) > 0.03
              ? ' Note: this schedule does not end in pure static. At its last step ' +
                pct(Math.sqrt(AB[S.T])) +
                ' of a picture would still be left, so starting from pure static does not match what the denoiser expects, and the result is skewed toward some pictures. A schedule must destroy everything for generation to be fair.'
              : '');
        }
        denoise();
        if (tb) tb.set(S.mode === 'round' ? 0 : S.T);
      };
      const fin = () => phase === 'done';
      const step = () => {
        if (fin()) {
          start(false);
          return;
        }
        if (phase === 'fwd') {
          const b = 1 - AB[t + 1] / AB[t],
            s1 = Math.sqrt(1 - b),
            s2 = Math.sqrt(b);
          for (let i = 0; i < DM; i++) x[i] = s1 * x[i] + s2 * RN();
          t++;
          denoise();
          cu = t >= K ? 1 : 0;
          const bi = wts.indexOf(Math.max(...wts));
          msg =
            'Forward step ' +
            t +
            ' of ' +
            K +
            ': the picture was shrunk by a factor of ' +
            f3(s1) +
            ' and noise of size ' +
            f3(s2) +
            ' was added (β = ' +
            f3(b) +
            '). ' +
            pct(Math.sqrt(AB[t])) +
            ' of the original is left.';
          if (t >= K) {
            phase = 'rev';
            wTurn = wts.slice();
            aTurn = AB[t];
            tau = mkTau(K);
            ti = 0;
            msg +=
              ' This is the turning point. From what is left, the denoiser gives “' +
              GAL[S.pic][0] +
              '” a probability of ' +
              pct(wts[S.pic]) +
              (bi !== S.pic ? ', and “' + GAL[bi][0] + '” ' + pct(wts[bi]) : '') +
              '. Now the reverse process starts from exactly this noisy picture.';
          }
        } else {
          const tt = tau[ti],
            s = tau[ti + 1],
            at = AB[tt],
            as = AB[s],
            bp = 1 - at / as,
            c0 = (Math.sqrt(as) * bp) / Math.max(1e-12, 1 - at),
            c1 = (Math.sqrt(1 - bp) * (1 - as)) / Math.max(1e-12, 1 - at),
            bt = ((1 - as) / Math.max(1e-12, 1 - at)) * bp,
            sg =
              s > 0
                ? Math.sqrt(
                    Math.exp(
                      S.v * Math.log(Math.max(bp, 1e-20)) +
                        (1 - S.v) * Math.log(Math.max(bt, 1e-20)),
                    ),
                  )
                : 0;
          for (let i = 0; i < DM; i++) x[i] = c0 * x0h[i] + c1 * x[i] + sg * RN();
          ti++;
          t = s;
          denoise();
          const bi = wts.indexOf(Math.max(...wts));
          cu = t <= 0 ? 4 : 3;
          msg =
            'Reverse, from step ' +
            tt +
            ' to step ' +
            s +
            ': the new picture is ' +
            f3(c0) +
            ' × the denoiser’s guess + ' +
            f3(c1) +
            ' × the current picture, plus fresh noise of size ' +
            f3(sg) +
            '. The denoiser’s guess is ' +
            (wts[bi] > 0.98 ? 'now firmly' : pct(wts[bi])) +
            ' “' +
            GAL[bi][0] +
            '”.';
          if (t <= 0) {
            phase = 'done';
            res = bi;
            if (S.mode === 'gen') {
              cu = 5;
              msg +=
                ' Finished: “' +
                GAL[bi][0] +
                '”, made from nothing but static. Run it again and a different picture may form.';
            } else if (bi === S.pic)
              msg +=
                ' Finished: “' +
                GAL[bi][0] +
                '” came back, the picture you started with. At the turning point ' +
                pct(Math.sqrt(aTurn)) +
                ' of it was left, enough for the denoiser to recognise it.';
            else
              msg +=
                ' Finished: it came back as “' +
                GAL[bi][0] +
                '”, not “' +
                GAL[S.pic][0] +
                '”. At the turning point only ' +
                pct(Math.sqrt(aTurn)) +
                ' of the original was left and the denoiser gave it a probability of ' +
                pct(wTurn[S.pic]) +
                '. Too little survived to tell which picture it had been. Turn back earlier, or use a schedule that keeps more.';
          }
        }
        if (tb) tb.set(t);
      };
      const jump = (k) => {
        if (S.mode !== 'round') return;
        phase = 'fwd';
        res = -1;
        t = clamp(k, 0, S.T);
        const a = AB[t],
          sa = Math.sqrt(a),
          sn = Math.sqrt(1 - a),
          y = GAL[S.pic][1];
        for (let i = 0; i < DM; i++) x[i] = sa * y[i] + sn * eps0[i];
        denoise();
        cu = 0;
        msg =
          'Jumped straight to forward step ' +
          t +
          ' with the one-jump formula: ' +
          f3(sa) +
          ' × picture + ' +
          f3(sn) +
          ' × noise. The denoiser gives “' +
          GAL[S.pic][0] +
          '” a probability of ' +
          pct(wts[S.pic]) +
          ' here.' +
          (t >= K
            ? ' This is at or past your turning point: press One step to start the reverse process from here.'
            : '');
        if (t >= K) {
          phase = 'rev';
          wTurn = wts.slice();
          aTurn = a;
          tau = mkTau(t);
          ti = 0;
        }
      };
      const blit = (g, v, X0, Y0, sz) => {
        const cz = sz / N;
        for (let i = 0; i < N * N; i++) {
          g.fillStyle =
            'rgb(' +
            [0, 1, 2]
              .map((ch) => Math.round(clamp((v[i * 3 + ch] + 1) * 127.5, 0, 255)))
              .join(',') +
            ')';
          g.fillRect(X0 + (i % N) * cz, Y0 + Math.floor(i / N) * cz, cz + 0.4, cz + 0.4);
        }
      };
      const mixv = (a, buf) => {
        const sa = Math.sqrt(a),
          sn = Math.sqrt(1 - a),
          y = GAL[S.pic][1];
        for (let i = 0; i < DM; i++) buf[i] = sa * y[i] + sn * eps0[i];
        return buf;
      };
      const draw = () => {
        const tt = T(),
          Tn = S.T;
        {
          const { g, w, h } = fit(c),
            sz = Math.min(w, h) - 8,
            X0 = (w - sz) / 2,
            Y0 = (h - sz) / 2;
          blit(g, x, X0, Y0, sz);
          if (S.mode === 'round') {
            const s2 = sz * 0.2;
            g.fillStyle = tt.bg;
            g.fillRect(X0, Y0, s2 + 8, s2 + 22);
            blit(g, GAL[S.pic][1], X0 + 4, Y0 + 4, s2);
            g.font = MONO;
            g.fillStyle = tt.ink;
            g.textAlign = 'center';
            g.fillText('original', X0 + 4 + s2 / 2, Y0 + s2 + 17);
          }
        }
        {
          const { g, w, h } = fit(cr),
            sa = h * 0.24;
          g.font = MONO;
          g.fillStyle = tt.muted;
          g.textAlign = 'left';
          g.fillText('the denoiser’s guess at the clean picture, and its belief in each', 8, 13);
          blit(g, x0h, 8, 20, sa - 8);
          const bx = sa + 14,
            rh = Math.min(30, (sa - 8) / GAL.length);
          GAL.forEach((q, k) => {
            const y = 20 + k * rh;
            blit(g, q[1], bx, y + 1, rh - 3);
            g.fillStyle = mix(tt.ink, tt.ink, 0, 0.08);
            g.fillRect(bx + rh + 4, y + rh * 0.25, w - bx - rh - 60, rh * 0.5);
            g.fillStyle = k === S.pic && S.mode === 'round' ? tt.gold : tt.accent;
            g.fillRect(bx + rh + 4, y + rh * 0.25, (w - bx - rh - 60) * wts[k], rh * 0.5);
            g.fillStyle = tt.muted;
            g.textAlign = 'right';
            g.fillText((100 * wts[k]).toFixed(0) + '%', w - 8, y + rh * 0.62);
          });
          const ys = sa + 30,
            sw = Math.min((w - 110) / 5 - 6, h * 0.135),
            buf = new Float64Array(DM);
          [
            [AB, S.sch],
            [ABo, other()],
          ].forEach(([a, nm], r) => {
            const y = ys + r * (sw + 16);
            g.fillStyle = r ? tt.muted : tt.ink;
            g.textAlign = 'left';
            g.fillText(r ? 'alternative:' : 'yours:', 8, y + sw / 2 - 4);
            g.fillText(SN[nm], 8, y + sw / 2 + 9);
            for (let j = 0; j < 5; j++) {
              const k = Math.round((j * Tn) / 4),
                X0 = 92 + j * (sw + 6);
              blit(g, mixv(a[k], buf), X0, y, sw);
              if (!r) {
                g.fillStyle = tt.muted;
                g.textAlign = 'center';
                g.fillText(j === 4 ? 'all steps' : j ? j * 25 + '%' : 'start', X0 + sw / 2, y - 4);
              }
            }
          });
          const yc = ys + 2 * (sw + 16) + 14,
            m = { l: 36, r: 58, b: 20 },
            X = (k) => m.l + (k / Tn) * (w - m.l - m.r),
            Y = (v) => h - m.b - v * (h - yc - m.b - 6);
          g.strokeStyle = tt.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h - m.b);
          g.lineTo(w - m.r, h - m.b);
          g.stroke();
          g.textAlign = 'right';
          g.fillStyle = tt.muted;
          g.fillText('1', m.l - 5, Y(1) + 4);
          g.fillText('0', m.l - 5, Y(0) + 4);
          g.setLineDash([5, 4]);
          g.strokeStyle = tt.muted;
          g.lineWidth = 1.6;
          g.beginPath();
          for (let k = 0; k <= Tn; k++)
            k ? g.lineTo(X(k), Y(Math.sqrt(ABo[k]))) : g.moveTo(X(k), Y(Math.sqrt(ABo[k])));
          g.stroke();
          g.setLineDash([]);
          [
            [(k) => Math.sqrt(AB[k]), tt.accent, 'picture'],
            [(k) => Math.sqrt(1 - AB[k]), tt.ok, 'noise'],
          ].forEach(([f, cl, lab]) => {
            g.strokeStyle = cl;
            g.lineWidth = 2.4;
            g.beginPath();
            for (let k = 0; k <= Tn; k++) k ? g.lineTo(X(k), Y(f(k))) : g.moveTo(X(k), Y(f(k)));
            g.stroke();
            g.fillStyle = cl;
            g.textAlign = 'left';
            g.fillText(lab, w - m.r + 4, Y(f(Tn)) + 4);
          });
          if (S.mode === 'round') {
            g.setLineDash([3, 3]);
            g.strokeStyle = tt.ink;
            g.lineWidth = 1.3;
            g.beginPath();
            g.moveTo(X(K), Y(0));
            g.lineTo(X(K), Y(1.04));
            g.stroke();
            g.setLineDash([]);
            g.fillStyle = tt.ink;
            g.textAlign = 'center';
            g.fillText('turn back', X(K), yc + 2);
          }
          g.strokeStyle = tt.gold;
          g.lineWidth = 2.5;
          g.beginPath();
          g.moveTo(X(t), Y(0));
          g.lineTo(X(t), Y(1));
          g.stroke();
          g.fillStyle = tt.muted;
          g.textAlign = 'left';
          g.fillText(
            'share of picture and of noise in the mix; dashed: picture under the ' +
              SN[other()] +
              ' schedule',
            8,
            h - 4,
          );
          g.textAlign = 'center';
        }
        live.textContent = msg;
        uCur(root, 'dp', cu);
        const bi = wts.indexOf(Math.max(...wts));
        setR(
          root,
          't',
          t +
            ' of ' +
            Tn +
            (phase === 'fwd' ? ' (adding noise)' : phase === 'rev' ? ' (removing noise)' : ''),
        );
        setR(root, 'sig', (100 * Math.sqrt(AB[t])).toFixed(1) + '%');
        setR(root, 'nz', (100 * Math.sqrt(1 - AB[t])).toFixed(1) + '%');
        setR(
          root,
          'to',
          wts[bi] < 0.4
            ? 'no particular picture yet'
            : '“' + GAL[bi][0] + '” (' + (100 * wts[bi]).toFixed(0) + '%)',
        );
        setR(
          root,
          'rc',
          isFinite(rc[0])
            ? rc[0] +
                ' of ' +
                Tn +
                ' (' +
                SN[S.sch] +
                ') / ' +
                rc[1] +
                ' of ' +
                Tn +
                ' (' +
                SN[other()] +
                ')'
            : '–',
        );
        setR(
          root,
          'res',
          res < 0
            ? '–'
            : S.mode === 'gen'
              ? 'Generated “' + GAL[res][0] + '”'
              : res === S.pic
                ? 'The same picture came back'
                : 'A different picture came back: “' + GAL[res][0] + '”',
        );
      };
      tb = bind(
        root,
        'dp-t',
        (v) => {
          pb.stop();
          if (S.mode === 'round') {
            jump(+v);
            draw();
          } else {
            tb.set(t);
            msg =
              'The step slider belongs to the forward direction, where any step can be computed in one jump. Choose Noise This Picture, Then Denoise It to use it.';
            draw();
          }
        },
        (v) => String(v),
      );
      GAL = builtin();
      fillSel();
      start(true);
      const anim = Anim(() => {
        fr++;
        const slow = phase === 'fwd' ? t < 5 : ti < 3 || t < 4;
        if (fr % (slow ? 9 : Math.max(2, Math.round(260 / S.T)))) return;
        step();
        draw();
        if (fin()) pb.stop();
      });
      pb = playBtn($('#dp-play', root), anim);
      runCtl(pb, root, 'dp', { done: fin, restart: () => start(false) });
      const re = (nn) => {
        pb.stop();
        start(!!nn);
        draw();
      };
      $('#dp-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#dp-reset', root).onclick = () => re(false);
      $('#dp-new', root).onclick = () => {
        re(true);
        msg = 'Fresh random noise has been drawn. ' + msg;
        draw();
      };
      segBind(root, 'dp-dir', S.mode, (v) => {
        S.mode = v;
        pb.stop();
        start(false);
        draw();
        pb.start();
      });
      segBind(root, 'dp-sch', S.sch, (v) => {
        S.sch = v;
        re(false);
      });
      sel.onchange = (e) => {
        S.pic = +e.target.value;
        re(false);
      };
      bind(
        root,
        'dp-T',
        (v) => {
          S.T = +v;
          re(false);
        },
        (v) => String(v),
      );
      bind(
        root,
        'dp-b0',
        (v) => {
          S.b0 = +v;
          if (S.sch === 'cos') {
            sched();
            draw();
          } else re(false);
        },
        (v) => (+v).toFixed(4),
      );
      bind(
        root,
        'dp-b1',
        (v) => {
          S.b1 = +v;
          if (S.sch === 'cos') {
            sched();
            draw();
          } else re(false);
        },
        f2,
      );
      bind(
        root,
        'dp-turn',
        (v) => {
          S.turn = +v;
          re(false);
        },
        (v) => v + '%',
      );
      bind(
        root,
        'dp-v',
        (v) => {
          S.v = +v;
        },
        f2,
      );
      bind(
        root,
        'dp-rs',
        (v) => {
          S.rs = +v;
          if (phase === 'rev' && ti === 0) tau = mkTau(tau[0]);
        },
        (v) => v + '%',
      );
      $('#dp-file', root).onchange = (e) => {
        const f = e.target.files[0];
        e.target.value = '';
        if (!f) return;
        const bad = () => {
            st.textContent = 'That file could not be opened as a picture. Try a JPEG or PNG file.';
          },
          use = (im) => {
            const iw = im.width,
              ih = im.height;
            if (!iw || !ih) {
              bad();
              return;
            }
            const o = document.createElement('canvas');
            o.width = o.height = N;
            const s = Math.min(iw, ih);
            o.getContext('2d').drawImage(im, (iw - s) / 2, (ih - s) / 2, s, s, 0, 0, N, N);
            GAL = GAL.filter((q) => !q[2]);
            GAL.push(['Your picture', fromCanvas(o), true]);
            S.pic = GAL.length - 1;
            fillSel();
            S.mode = 'round';
            root
              .querySelectorAll('#dp-dir button')
              .forEach((b) =>
                b.setAttribute('aria-pressed', b.dataset.v === 'round' ? 'true' : 'false'),
              );
            re(false);
            st.textContent =
              'Loaded ' +
              f.name +
              ', cropped to a square and shrunk to 32 × 32 pixels. It stays in your browser. The model now knows ' +
              GAL.length +
              ' pictures.';
            draw();
          };
        if (window.createImageBitmap) createImageBitmap(f).then(use, bad);
        else {
          const r2 = new FileReader();
          r2.onerror = bad;
          r2.onload = () => {
            const im = new Image();
            im.onload = () => use(im);
            im.onerror = bad;
            im.src = r2.result;
          };
          r2.readAsDataURL(f);
        }
      };
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          start(false);
          draw();
          pb.start();
        },
      };
    },
  },
  {
    id: 'ddpm2d',
    tab: 'Train a diffusion model',
    title: 'Train a diffusion model: teach a network to undo noise',
    lede: 'The previous tab used an ideal denoiser. Here a real network learns the job from scratch. The “pictures” are single points in a plane, arranged in a shape, so that you can see everything at once. The network is shown noisy points and asked what noise was added. Once it can answer, hand it a cloud of pure noise and watch 400 points walk back into the shape.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Faint dots: the real data. Gold dots: points being generated. Arrows: which way the network says “less noisy” is, at the current noise level</figcaption><canvas class="short" id="d2-c" role="img" aria-label="Data, generated samples and the learned denoising field"></canvas></figure>
 <figure class="fig"><figcaption>Top: the real data at four noise levels, which is what the network is trained on. Bottom: its error at guessing the noise</figcaption><canvas class="short" id="d2-r" role="img" aria-label="Noised data at several steps and the training loss"></canvas></figure></div>
 ${uHow('d2', ['Take a batch of real points. Give each a random noise level t and add that much noise.', 'Ask the network: “this noisy point, at level t: what noise was added?” Compare with the truth and backpropagate.', 'Repeat for three thousand batches.', 'Generate: start 400 points as pure noise at the highest level.', 'At each level, the network estimates the noise in every point, a little of it is removed, and a little fresh noise is added. After 40 levels the points form the shape.'])}
 <div class="controls">${uBtns('d2', 'Run', 'One step', 'training steps')}
 ${grp('The data', `<label class="ctl" for="d2-ds"><span>Shape to learn</span><select id="d2-ds"><option value="moons">Two moons</option><option value="ring">A ring</option><option value="spiral">A spiral</option><option value="blobs">Three blobs</option></select></label><div class="row"><button class="btn" id="d2-gen" type="button">Generate Again</button></div>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ts', 'Training steps')}${rd('L', 'Error at guessing the noise', 'hl2')}${rd('lv', 'Noise level while generating')}${rd('near', 'Generated points that land on the shape', 'hl')}</dl>`,
    tryit:
      'Let it train and generate, then press Generate Again a few times: new points each time, always forming the shape. Now switch to the spiral and press Start over, and during training press One step to move 50 steps at a time, generating at 100, 500 and 3000 steps by pressing Generate Again. How does the quality of the generated shape track the error curve? Watch the arrows during generation: far from the data they all point inward; near the end they point at the nearest part of the shape.',
    math: String.raw`<h4>Intuition</h4><p>Hide a friend's sketch under a thin sheet of tracing paper covered in random speckles and ask someone to remove the speckles. With one sheet it is easy. Stack forty sheets and the sketch is invisible, but the job can still be done one sheet at a time, as long as the person has learned what sketches look like and so can tell speckle from line. Training a diffusion model is practising that single-sheet job at every thickness of the stack. Generating is starting with forty sheets of speckle and nothing underneath, and peeling.</p>
 <h4>Training</h4><p>Pick a real point \(\mathbf x_0\), a level \(t\) uniformly from \(1,\dots,T\), and noise \(\boldsymbol\varepsilon\sim\mathcal N(\mathbf 0,\mathbf I)\). Form the noisy point in one jump, \(\mathbf x_t=\sqrt{\bar\alpha_t}\,\mathbf x_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon\), and train the network to recover the noise:</p>$$\mathcal L=\big\lVert\boldsymbol\varepsilon-\boldsymbol\varepsilon_\theta(\mathbf x_t,t)\big\rVert^2 .$$<p>That is the entire training loop. No adversary, no encoder: plain regression with a target that is known exactly, because we added the noise ourselves. This is why diffusion models train so stably.</p>
 <h4>Why the error never reaches zero</h4><p>Many different clean points and noises give the same noisy point, so the noise cannot be known exactly. The best possible answer is the average over all the ways \(\mathbf x_t\) could have arisen, and the remaining error is irreducible, as in the Bias and Variance lesson. A loss that flattens at a value well above zero is what success looks like here. This page trains for a fixed 3,000 steps, by which time it has flattened.</p>
 <h4>What the network has really learned</h4><p>The predicted noise points away from the data. Its negative, scaled, is the <b>score</b>, the direction in which the probability of the noisy data rises fastest:</p>$$\nabla_{\mathbf x}\log p_t(\mathbf x)\approx-\frac{\boldsymbol\varepsilon_\theta(\mathbf x,t)}{\sqrt{1-\bar\alpha_t}} .$$<p>The arrows in the picture are this field. At high noise it is simple and points toward the middle of the data. At low noise it is detailed and points at the nearest part of the shape. Learning one field per noise level, from coarse to fine, is what lets generation start from anywhere.</p>
 <h4>Generating</h4><p>Start from \(\mathbf x_T\sim\mathcal N(\mathbf 0,\mathbf I)\) and apply, for \(t=T,\dots,1\),</p>$$\mathbf x_{t-1}=\frac{1}{\sqrt{1-\beta_t}}\Big(\mathbf x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\varepsilon_\theta(\mathbf x_t,t)\Big)+\sqrt{\beta_t}\,\mathbf z .$$<p>Each point follows the arrows a short way and is jostled by fresh noise \(\mathbf z\). The jostling matters: without it the points would bunch up along the centre of the shape.</p>
 <h4>What this page trains</h4><p>A real network with two hidden layers of 64 neurons. Its input is the two coordinates of the noisy point and eight numbers that encode the noise level as sine and cosine waves. It is trained with Adam on batches of 128, with \(T=40\) levels. Image models do the same with a U-Net in place of this small network and a picture in place of the point.</p>`,
    q: 'After training, the error at guessing the noise is about 0.3, not 0. Yet the generated points land on the shape. How can a network that is “30% wrong” generate well?',
    a: 'The error is an average over all noise levels, and at high noise the question has no exact answer: the same noisy point could have come from many clean points, so even a perfect network must be wrong by a fixed amount. What matters for generation is that the network’s answer is the correct average, which gives the right direction to move in at every level. Following correct average directions for forty small steps leads onto the data.',
    init(root) {
      const c = $('#d2-c', root),
        cr = $('#d2-r', root),
        live = $('#d2-live', root),
        S = { ds: 'moons' },
        Tn = 40,
        MAXS = 3000,
        M = 400,
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
      let net,
        ts = 0,
        LL = [],
        xs = null,
        lv = Tn,
        phase = 'train',
        data = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        sm = [],
        done = false;
      const samp = () => {
        const R = Math.random;
        if (S.ds === 'moons') {
          const a = R() * Math.PI;
          return R() < 0.5
            ? [Math.cos(a) - 0.5 + 0.06 * RN(), -0.25 + Math.sin(a) * 0.9 + 0.06 * RN()]
            : [0.5 - Math.cos(a) + 0.06 * RN(), 0.25 - Math.sin(a) * 0.9 + 0.06 * RN()];
        }
        if (S.ds === 'ring') {
          const a = R() * 2 * Math.PI;
          return [1.1 * Math.cos(a) + 0.06 * RN(), 1.1 * Math.sin(a) + 0.06 * RN()];
        }
        if (S.ds === 'spiral') {
          const u = R(),
            a = u * 4 * Math.PI;
          return [
            (0.25 + 1.05 * u) * Math.cos(a) + 0.04 * RN(),
            (0.25 + 1.05 * u) * Math.sin(a) + 0.04 * RN(),
          ];
        }
        const k = Math.floor(R() * 3),
          m = [
            [-1, -0.5],
            [1, -0.5],
            [0, 0.9],
          ][k];
        return [m[0] + 0.18 * RN(), m[1] + 0.18 * RN()];
      };
      const emb = (k) => {
        const o = [];
        for (let q = 0; q < 4; q++) {
          const f = ((k + 1) / Tn) * Math.PI * 2 ** q * 0.5;
          o.push(Math.sin(f), Math.cos(f));
        }
        return o;
      };
      const pred = (P, k) => {
        const n = P.length / 2,
          X = new Float64Array(n * 10),
          e = emb(k);
        for (let i = 0; i < n; i++) {
          X[i * 10] = P[2 * i];
          X[i * 10 + 1] = P[2 * i + 1];
          for (let q = 0; q < 8; q++) X[i * 10 + 2 + q] = e[q];
        }
        const o = net.fwd(AD.input(X, [n, 10]));
        AD.clear();
        return o.v;
      };
      const build = () => {
        net = AD.mlp([10, 64, 64, 2], randnFrom(rng(run * 7 + 4)), 'relu');
        ts = 0;
        LL = [];
        sm = [];
        phase = 'train';
        xs = null;
        lv = Tn;
        done = false;
        cu = 0;
        data = Array.from({ length: 500 }, samp);
        msg =
          'A network with random weights. It cannot tell noise from data yet. Press Run to train it.';
      };
      const train = (k) => {
        const B = 128;
        for (let q = 0; q < k && ts < MAXS; q++) {
          const X = new Float64Array(B * 10),
            E = new Float64Array(B * 2);
          for (let i = 0; i < B; i++) {
            const p = samp(),
              tt = Math.floor(Math.random() * Tn),
              e0 = RN(),
              e1 = RN(),
              s = Math.sqrt(AB[tt]),
              nq = Math.sqrt(1 - AB[tt]),
              e = emb(tt);
            X[i * 10] = s * p[0] + nq * e0;
            X[i * 10 + 1] = s * p[1] + nq * e1;
            for (let j = 0; j < 8; j++) X[i * 10 + 2 + j] = e[j];
            E[2 * i] = e0;
            E[2 * i + 1] = e1;
          }
          const l = AD.mse(net.fwd(AD.input(X, [B, 10])), E);
          LL.push(l.v[0]);
          AD.backward(l);
          AD.adam(net.ps, 0.002, ++ts, 1);
          if (ts % 20 === 0) {
            let s = 0;
            for (let j = ts - 20; j < ts; j++) s += LL[j];
            sm.push(s / 20);
          }
        }
        cu = ts % 2 ? 0 : 1;
        msg =
          'Training step ' +
          ts +
          ' of ' +
          MAXS +
          ': 128 real points were given random amounts of noise, and the network guessed the noise in each. Average error over the last 20 steps: ' +
          f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
          '.';
        if (ts >= MAXS) {
          startGen();
          cu = 3;
          msg =
            'Training is finished. Now generate: ' +
            M +
            ' points of pure noise have been placed at the highest noise level. Watch them walk.';
        }
      };
      const startGen = () => {
        phase = 'gen';
        xs = Float64Array.from({ length: 2 * M }, () => RN());
        lv = Tn;
        done = false;
      };
      const gstep = () => {
        if (lv <= 0) return;
        const k = lv - 1,
          e = pred(xs, k),
          b = BE[k],
          c1 = b / Math.sqrt(1 - AB[k]),
          s1 = 1 / Math.sqrt(1 - b),
          s2 = k > 0 ? Math.sqrt(b) : 0;
        for (let i = 0; i < 2 * M; i++) xs[i] = s1 * (xs[i] - c1 * e[i]) + s2 * RN();
        lv--;
        cu = 4;
        msg =
          'Generating, noise level ' +
          lv +
          ' of ' +
          Tn +
          ': the network estimated the noise in each of the ' +
          M +
          ' points, a little of it was removed, and ' +
          (k > 0 ? 'a little fresh noise was added.' : 'no noise was added on this last step.') +
          (lv <= 0
            ? ' Finished: ' +
              pct(near()) +
              ' of the generated points lie on the shape. None of them is a copy of a training point.'
            : '');
        if (lv <= 0) done = true;
      };
      const near = () => {
        if (!xs) return NaN;
        let ok = 0;
        for (let i = 0; i < M; i++) {
          let b = 1e9;
          for (const q of data)
            b = Math.min(b, (q[0] - xs[2 * i]) ** 2 + (q[1] - xs[2 * i + 1]) ** 2);
          if (b < 0.15 * 0.15) ok++;
        }
        return ok / M;
      };
      const step = () => {
        if (phase === 'train') train(50);
        else if (done) {
          if (ts < MAXS) {
            phase = 'train';
            xs = null;
            train(50);
          } else {
            startGen();
            msg = 'A fresh cloud of pure noise. Press Run or One step.';
          }
        } else gstep();
      };
      const draw = () => {
        const t = T();
        {
          const V = uView(c),
            { g, w, h, X, Y } = V,
            sc = 1.75,
            X2 = (v) => X(v * sc),
            Y2 = (v) => Y(v * sc);
          g.fillStyle = mix(t.ink, t.ink, 0, 0.28);
          data.forEach((p) => {
            g.beginPath();
            g.arc(X2(p[0]), Y2(p[1]), 2, 0, 7);
            g.fill();
          });
          if (ts > 0) {
            const k = phase === 'gen' ? Math.max(0, lv - 1) : Math.floor((fr / 8) % Tn),
              G = [];
            for (let a = 0; a < 10; a++)
              for (let b = 0; b < 14; b++)
                G.push(-2.1 + (4.2 * (b + 0.5)) / 14, 1.55 - (3.1 * (a + 0.5)) / 10);
            const e = pred(Float64Array.from(G), k);
            g.strokeStyle = mix(t.accent, t.accent, 0, 0.6);
            g.lineWidth = 1.2;
            for (let i = 0; i < G.length / 2; i++) {
              const x = X2(G[2 * i]),
                y = Y2(G[2 * i + 1]),
                dx = -e[2 * i],
                dy = -e[2 * i + 1],
                m = Math.hypot(dx, dy) || 1,
                L = Math.min(15, 7 * m);
              g.beginPath();
              g.moveTo(x, y);
              g.lineTo(x + (dx / m) * L, y - (dy / m) * L);
              g.stroke();
              g.beginPath();
              g.arc(x + (dx / m) * L, y - (dy / m) * L, 1.6, 0, 7);
              g.fillStyle = t.accent;
              g.fill();
            }
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText('arrows shown for noise level ' + (k + 1), 8, h - 8);
          }
          if (xs) {
            g.fillStyle = t.gold;
            g.strokeStyle = t.bg;
            g.lineWidth = 0.7;
            for (let i = 0; i < M; i++) {
              const x = X2(xs[2 * i]),
                y = Y2(xs[2 * i + 1]);
              if (x < -5 || y < -5 || x > w + 5 || y > h + 5) continue;
              g.beginPath();
              g.arc(x, y, 3, 0, 7);
              g.fill();
              g.stroke();
            }
          }
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.42,
            LV = [0, 8, 20, 40],
            pw = (w - 20) / 4;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('the real data with more and more noise added: what training sees', 8, 13);
          LV.forEach((k, q) => {
            const x0 = 10 + q * pw,
              cx = x0 + pw / 2,
              cy = 20 + (h1 - 40) / 2,
              s = Math.min(pw, h1 - 40) / 5.6,
              a = k ? AB[k - 1] : 1,
              r = rng(q * 7 + 3),
              rn = randnFrom(r);
            g.strokeStyle = t.line;
            g.lineWidth = 1;
            g.strokeRect(x0 + 2, 20, pw - 4, h1 - 40);
            g.fillStyle = mix(t.ink, t.ink, 0, 0.55);
            for (let i = 0; i < 160; i++) {
              const p = data[i],
                x = Math.sqrt(a) * p[0] + Math.sqrt(1 - a) * rn(),
                y = Math.sqrt(a) * p[1] + Math.sqrt(1 - a) * rn();
              if (Math.abs(x) > 2.7 || Math.abs(y) > 2.6) continue;
              g.beginPath();
              g.arc(cx + x * s, cy - y * s, 1.5, 0, 7);
              g.fill();
            }
            g.fillStyle = phase === 'gen' && Math.abs(lv - k) < 5 ? t.cut : t.muted;
            g.textAlign = 'center';
            g.fillText('level ' + k, cx, h1 - 6);
          });
          uLine(g, t, 0, h1, w, h - h1, sm, {
            title: 'error at guessing the noise (average of every 20 steps)',
            n: MAXS / 20,
            col: t.ok,
            ym: Math.max(1.05, sm[0] || 1.05),
          });
        }
        live.textContent = msg;
        uCur(root, 'd2', cu);
        setR(
          root,
          'ph',
          phase === 'train' ? (ts ? 'Training' : 'Not started') : done ? 'Finished' : 'Generating',
        );
        setR(root, 'ts', ts + ' of ' + MAXS);
        setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
        setR(root, 'lv', phase === 'gen' ? lv + ' of ' + Tn : '–');
        setR(root, 'near', phase === 'gen' && done ? pct(near()) : '–');
      };
      build();
      if (RM) {
        train(MAXS);
        while (!done) gstep();
      }
      const anim = Anim(() => {
        fr++;
        if (phase === 'train') {
          train(Math.min(ts < 60 ? 1 : 6, chk.remaining()));
          draw();
          chk();
          return;
        }
        if (fr % (lv > Tn - 4 ? 26 : 9) === 0 && !done) gstep();
        draw();
        if (done) pb.stop();
      });
      pb = playBtn($('#d2-play', root), anim);
      chk = runCtl(pb, root, 'd2', {
        done: () => phase === 'gen' && done,
        restart: () => {
          if (ts < MAXS) {
            phase = 'train';
            xs = null;
            done = false;
          } else startGen();
        },
        count: () => ts,
        what: 'training steps',
      });
      $('#d2-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#d2-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#d2-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        pb.stop();
        run++;
        build();
        draw();
        pb.start();
      };
      $('#d2-gen', root).onclick = () => {
        if (!ts) {
          msg = 'Train the network first.';
          draw();
          return;
        }
        pb.stop();
        startGen();
        msg =
          'A fresh cloud of pure noise' +
          (ts < MAXS ? ', using the network as trained so far (' + ts + ' steps).' : '.');
        draw();
        pb.start();
      };
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
  {
    id: 'stable-diffusion',
    tab: 'Stable Diffusion',
    title: 'Stable Diffusion in miniature: from a prompt to a picture',
    lede: 'Stable Diffusion is three ideas bolted together. Compress pictures into a small code, so the hard work happens on a few numbers instead of a million pixels. Run diffusion on that code. And steer the diffusion with a prompt. This page builds all three for real, small enough to train in your browser: choose a prompt, set how strongly to follow it, and watch eight pictures form.',
    stage: `<div class="row seg" id="sd-pr" role="group" aria-label="Prompt"></div>
 <div class="stage one"><figure class="fig"><figcaption>The pipeline. The gold box is the part that is working right now</figcaption><canvas class="tall" id="sd-p" style="aspect-ratio:16/2.5" role="img" aria-label="Stable Diffusion pipeline: prompt, text encoder, denoiser in latent space, decoder, picture"></canvas></figure></div>
 <div class="stage" style="margin-top:14px"><figure class="fig"><figcaption>The latent space: every picture compressed to two numbers. Letters mark where the training pictures sit. Gold dots are the eight codes being generated; the line is the path of the first one</figcaption><canvas class="short" id="sd-c" role="img" aria-label="Latent space with generated codes walking from noise to the prompted region"></canvas></figure>
 <figure class="fig"><figcaption>Top: the eight codes run through the decoder at this moment. A gold frame means it matches the prompt. Bottom: the training error of the part being trained</figcaption><canvas class="short" id="sd-r" role="img" aria-label="Decoded pictures and training loss"></canvas></figure></div>
 ${uHow('sd', ['Train the compressor (a VAE): squeeze each 35-pixel picture to a code of 2 numbers and back.', 'Encode every training picture. From here on the pictures are put away: only their codes and their captions are used.', 'Train the denoiser on the codes: add noise to a code, show it the noisy code and the caption, and ask what noise was added. One time in seven the caption is hidden.', 'Generate: start eight codes as pure noise. At each step ask the denoiser twice, with the prompt and without, and push the answer further in the direction the prompt suggests (guidance).', 'After 40 steps, run the finished codes through the decoder. Only now do pixels appear.'])}
 <div class="controls">${uBtns('sd', 'Run', 'One step', 'training steps')}
 ${grp('Generation', slider('sd-w', 'Guidance: how strongly to follow the prompt', 0, 8, 0.1, 3) + `<div class="row"><button class="btn" id="sd-gen" type="button">Generate Again</button></div>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('pr', 'Prompt')}${rd('lv', 'Noise level while generating')}${rd('ok', 'Generated pictures that match the prompt', 'hl')}${rd('cmp', 'Numbers the denoiser works on, per picture')}${rd('L', 'Training error now', 'hl2')}</dl>`,
    tryit:
      'Let it train both parts and generate. Then press each prompt in turn: the gold dots walk to a different region of the map each time, and a different letter comes out. Set guidance to 0 and generate: the prompt is ignored and you get a random mix of letters. Raise it to 3, then 8: more of the eight match, and they bunch tightly together. What is lost at very high guidance?',
    math: String.raw`<h4>Intuition</h4><p>An architect does not design a building by placing bricks. She works on a small floor plan, where moving a wall is one pencil stroke, and only at the end is the plan turned into a building. And she works to a brief: "three bedrooms, lots of light". Stable Diffusion generates pictures the same way. It does its diffusion on a small plan of the picture, called a latent, steered by a brief, the prompt. Only at the very end does a decoder turn the finished latent into pixels.</p>
 <h4>Part 1: compress (the VAE)</h4><p>An autoencoder is trained first and then frozen. Its encoder \(\mathcal E\) maps a picture \(\mathbf x\) to a latent \(\mathbf z=\mathcal E(\mathbf x)\), and its decoder \(\mathcal D\) maps a latent back to a picture. In the real model a \(512\times512\times3\) picture (786,432 numbers) becomes a \(64\times64\times4\) latent (16,384 numbers), 48 times smaller. Diffusion in that space is far cheaper than in pixels, which is what made the model usable on an ordinary graphics card. Here 35 pixels become 2 numbers.</p>
 <h4>Part 2: diffuse in latent space</h4><p>Everything from the DDPM tabs applies unchanged, with \(\mathbf z\) in place of the picture. Noise is added to latents, \(\mathbf z_t=\sqrt{\bar\alpha_t}\,\mathbf z_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon\), and a denoiser is trained to predict it, now with the caption \(c\) as an extra input:</p>$$\mathcal L=\mathbb E\Big[\big\lVert\boldsymbol\varepsilon-\boldsymbol\varepsilon_\theta(\mathbf z_t,\,t,\,c)\big\rVert^2\Big].$$<p>In the real model the denoiser is a U-Net and the caption enters through attention layers: every position in the latent attends to the words of the prompt (cross-attention).</p>
 <h4>Part 3: follow the prompt (classifier-free guidance)</h4><p>During training the caption is replaced by an empty one a fraction of the time, so the same network learns both "denoise toward this caption" and "denoise toward any picture". When generating, both are computed, and the difference between them is exaggerated by a guidance scale \(w\):</p>$$\hat{\boldsymbol\varepsilon}=\boldsymbol\varepsilon_\theta(\mathbf z_t,t,\varnothing)+w\,\Big[\boldsymbol\varepsilon_\theta(\mathbf z_t,t,c)-\boldsymbol\varepsilon_\theta(\mathbf z_t,t,\varnothing)\Big].$$<p>\(w=0\) ignores the prompt. \(w=1\) is the plain conditional model. \(w&gt;1\) pushes harder toward what the prompt adds, which makes results match the prompt more reliably and makes them less varied. Very large \(w\) gives over-saturated, distorted pictures; values around 7 are typical for images.</p>
 <h4>The whole pipeline</h4><ol><li>The prompt is turned into vectors by a text encoder.</li><li>A latent is filled with random noise.</li><li>For each step, the denoiser predicts the noise twice (with and without the prompt), guidance combines them, and the sampler removes a little noise.</li><li>The decoder turns the final latent into a picture.</li></ol>
 <h4>This page beside the real thing</h4><table><thead><tr><th></th><th>Here</th><th>Stable Diffusion 1.5</th></tr></thead><tbody><tr><td>Picture</td><td>5 × 7 pixels, black and white</td><td>512 × 512 pixels, colour</td></tr><tr><td>Latent</td><td>2 numbers</td><td>64 × 64 × 4 = 16,384 numbers</td></tr><tr><td>Prompt</td><td>one of six fixed captions</td><td>any text, up to 77 tokens</td></tr><tr><td>Text encoder</td><td>a lookup: one switch per caption</td><td>CLIP, a Transformer (123 million weights)</td></tr><tr><td>Denoiser</td><td>a small network, about 5,400 weights</td><td>a U-Net with attention (860 million weights)</td></tr><tr><td>Sampling steps</td><td>40</td><td>20 to 50, with faster samplers</td></tr><tr><td>Training data</td><td>150 pictures</td><td>about two billion captioned images</td></tr></tbody></table><p>The three parts and the way they connect are the same. What a model of this size cannot do is combine ideas it has never seen together, which is where the scale of the real model matters.</p>`,
    q: 'Why is the caption hidden from the denoiser in a fraction of the training examples? It seems to throw information away.',
    a: 'So that one network learns two things: how to denoise given a caption, and how to denoise with no caption at all. Guidance needs both at generation time, because it works by comparing the two answers and exaggerating their difference. Without the caption-free examples the network would have no sensible answer for an empty prompt, and there would be nothing to compare against.',
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
            path.forEach((p, i) =>
              i ? g.lineTo(X2(p[0]), Y2(p[1])) : g.moveTo(X2(p[0]), Y2(p[1])),
            );
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
  },
];

/* ---- extra autodiff pieces and shared drawing for the later generative lessons ---- */
(() => {
  const N = (v, sh) => ({ v, g: new Float64Array(v.length), sh }),
    T = AD._op;
  AD.mean = (A) => {
    const n = A.v.length,
      o = N(new Float64Array(1), [1]);
    for (let i = 0; i < n; i++) o.v[0] += A.v[i] / n;
    return T(o, () => {
      for (let i = 0; i < n; i++) A.g[i] += o.g[0] / n;
    });
  };
  AD.hcat = (A, B) => {
    const m = A.sh[0],
      o = N(new Float64Array(2 * m), [m, 2]);
    for (let i = 0; i < m; i++) {
      o.v[2 * i] = A.v[i];
      o.v[2 * i + 1] = B.v[i];
    }
    return T(o, () => {
      for (let i = 0; i < m; i++) {
        A.g[i] += o.g[2 * i];
        B.g[i] += o.g[2 * i + 1];
      }
    });
  };
  AD.aff = (x, sr, tt, c) => {
    const n = x.v.length,
      o = N(new Float64Array(n), x.sh),
      s = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      s[i] = c * Math.tanh(sr.v[i]);
      o.v[i] = x.v[i] * Math.exp(s[i]) + tt.v[i];
    }
    o.s = s;
    o.sr = sr;
    o.c = c;
    return T(o, () => {
      for (let i = 0; i < n; i++) {
        const e = Math.exp(s[i]),
          q = o.g[i];
        x.g[i] += q * e;
        tt.g[i] += q;
        sr.g[i] += q * x.v[i] * e * c * (1 - (s[i] / c) ** 2);
      }
    });
  };
  AD.nfl = (z, Ls) => {
    const m = z.sh[0],
      o = N(new Float64Array(1), [1]);
    for (let i = 0; i < z.v.length; i++) o.v[0] += (0.5 * z.v[i] * z.v[i]) / m;
    for (const L of Ls) for (let i = 0; i < m; i++) o.v[0] -= L.s[i] / m;
    o.v[0] += Math.log(2 * Math.PI);
    return T(o, () => {
      const q = o.g[0] / m;
      for (let i = 0; i < z.v.length; i++) z.g[i] += q * z.v[i];
      for (const L of Ls)
        for (let i = 0; i < m; i++) L.sr.g[i] -= q * L.c * (1 - (L.s[i] / L.c) ** 2);
    });
  };
  AD.ebl = (Ed, Es, lam) => {
    const n = Ed.v.length,
      k = Es.v.length,
      o = N(new Float64Array(1), [1]);
    for (let i = 0; i < n; i++) o.v[0] += (Ed.v[i] + lam * Ed.v[i] ** 2) / n;
    for (let i = 0; i < k; i++) o.v[0] += (-Es.v[i] + lam * Es.v[i] ** 2) / k;
    return T(o, () => {
      for (let i = 0; i < n; i++) Ed.g[i] += (o.g[0] * (1 + 2 * lam * Ed.v[i])) / n;
      for (let i = 0; i < k; i++) Es.g[i] += (o.g[0] * (-1 + 2 * lam * Es.v[i])) / k;
    });
  };
  AD.mce = (Lg, V, tg, mk) => {
    const R = mk.length,
      P = new Float64Array(Lg.v.length),
      o = N(new Float64Array(1), [1]);
    let cnt = 0;
    for (let i = 0; i < R; i++) if (mk[i]) cnt++;
    cnt = Math.max(1, cnt);
    for (let i = 0; i < R; i++) {
      let mx = -1e30;
      for (let j = 0; j < V; j++) mx = Math.max(mx, Lg.v[i * V + j]);
      let zz = 0;
      for (let j = 0; j < V; j++) {
        const e = Math.exp(Lg.v[i * V + j] - mx);
        P[i * V + j] = e;
        zz += e;
      }
      for (let j = 0; j < V; j++) P[i * V + j] /= zz;
      if (mk[i]) o.v[0] -= Math.log(P[i * V + tg[i]] + 1e-12) / cnt;
    }
    o.P = P;
    return T(o, () => {
      for (let i = 0; i < R; i++)
        if (mk[i])
          for (let j = 0; j < V; j++)
            Lg.g[i * V + j] += (o.g[0] * (P[i * V + j] - (j === tg[i] ? 1 : 0))) / cnt;
    });
  };
})();
const G2K = { moons: 'Two moons', ring: 'A ring', spiral: 'A spiral', blobs: 'Three blobs' },
  G2S = 1.75;
const g2Opts = (def, ks = Object.keys(G2K)) =>
  ks.map((k) => `<option value="${k}"${k === def ? ' selected' : ''}>${G2K[k]}</option>`).join('');
function g2Samp(ds, RN) {
  const R = Math.random;
  if (ds === 'moons') {
    const a = R() * Math.PI;
    return R() < 0.5
      ? [Math.cos(a) - 0.5 + 0.06 * RN(), -0.25 + Math.sin(a) * 0.9 + 0.06 * RN()]
      : [0.5 - Math.cos(a) + 0.06 * RN(), 0.25 - Math.sin(a) * 0.9 + 0.06 * RN()];
  }
  if (ds === 'ring') {
    const a = R() * 2 * Math.PI;
    return [1.1 * Math.cos(a) + 0.06 * RN(), 1.1 * Math.sin(a) + 0.06 * RN()];
  }
  if (ds === 'spiral') {
    const u = R(),
      a = u * 4 * Math.PI;
    return [
      (0.25 + 1.05 * u) * Math.cos(a) + 0.04 * RN(),
      (0.25 + 1.05 * u) * Math.sin(a) + 0.04 * RN(),
    ];
  }
  const k = Math.floor(R() * 3),
    m = [
      [-1, -0.5],
      [1, -0.5],
      [0, 0.9],
    ][k];
  return [m[0] + 0.18 * RN(), m[1] + 0.18 * RN()];
}
function g2Near(xs, M, data, r = 0.15) {
  let ok = 0;
  for (let i = 0; i < M; i++) {
    let b = 1e9;
    for (const q of data) {
      const d = (q[0] - xs[2 * i]) ** 2 + (q[1] - xs[2 * i + 1]) ** 2;
      if (d < b) b = d;
    }
    if (b < r * r) ok++;
  }
  return ok / M;
}
const g2Emb = (u) => {
  const o = [];
  for (let q = 0; q < 4; q++) {
    const f = u * Math.PI * 2 ** q * 0.5;
    o.push(Math.sin(f), Math.cos(f));
  }
  return o;
};
function g2Pred(net, P, e) {
  const n = P.length / 2,
    X = new Float64Array(n * 10);
  for (let i = 0; i < n; i++) {
    X[i * 10] = P[2 * i];
    X[i * 10 + 1] = P[2 * i + 1];
    for (let q = 0; q < 8; q++) X[i * 10 + 2 + q] = e[q];
  }
  const o = net.fwd(AD.input(X, [n, 10]));
  AD.clear();
  return o.v;
}
const G2G = (() => {
  const G = [];
  for (let a = 0; a < 10; a++)
    for (let b = 0; b < 14; b++)
      G.push(-2.1 + (4.2 * (b + 0.5)) / 14, 1.55 - (3.1 * (a + 0.5)) / 10);
  return Float64Array.from(G);
})();
function g2Data(g, V, t, data, al = 0.28, r = 2) {
  g.fillStyle = mix(t.ink, t.ink, 0, al);
  for (const p of data) {
    g.beginPath();
    g.arc(V.X(p[0] * G2S), V.Y(p[1] * G2S), r, 0, 7);
    g.fill();
  }
}
function g2Arrows(g, V, e, col, sgn = 1, al = 0.6, k = 7, mx = 15) {
  g.strokeStyle = mix(col, col, 0, al);
  g.fillStyle = col;
  g.lineWidth = 1.2;
  for (let i = 0; i < G2G.length / 2; i++) {
    const x = V.X(G2G[2 * i] * G2S),
      y = V.Y(G2G[2 * i + 1] * G2S),
      dx = sgn * e[2 * i],
      dy = sgn * e[2 * i + 1],
      m = Math.hypot(dx, dy) || 1,
      L = Math.min(mx, k * m);
    g.beginPath();
    g.moveTo(x, y);
    g.lineTo(x + (dx / m) * L, y - (dy / m) * L);
    g.stroke();
    g.beginPath();
    g.arc(x + (dx / m) * L, y - (dy / m) * L, 1.6, 0, 7);
    g.fill();
  }
}
function g2Pts(g, V, t, xs, M, col, r = 3) {
  g.fillStyle = col || t.gold;
  g.strokeStyle = t.bg;
  g.lineWidth = 0.7;
  for (let i = 0; i < M; i++) {
    const x = V.X(xs[2 * i] * G2S),
      y = V.Y(xs[2 * i + 1] * G2S);
    if (x < -5 || y < -5 || x > V.w + 5 || y > V.h + 5) continue;
    g.beginPath();
    g.arc(x, y, r, 0, 7);
    g.fill();
    g.stroke();
  }
}
function g2Trails(g, V, t, tr) {
  g.lineWidth = 1.6;
  g.lineJoin = 'round';
  for (const p of tr) {
    if (p.length < 2) continue;
    g.strokeStyle = mix(t.accent, t.accent, 0, 0.85);
    g.beginPath();
    p.forEach((q, i) =>
      i ? g.lineTo(V.X(q[0] * G2S), V.Y(q[1] * G2S)) : g.moveTo(V.X(q[0] * G2S), V.Y(q[1] * G2S)),
    );
    g.stroke();
    g.fillStyle = t.bg;
    g.beginPath();
    g.arc(V.X(p[0][0] * G2S), V.Y(p[0][1] * G2S), 3.5, 0, 7);
    g.fill();
    g.stroke();
  }
}
const g2Note = (g, t, h, s) => {
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'left';
  g.fillText(s, 8, h - 8);
};
function g2Bench(g, t, x0, y0, w, h, NS, series, cur, title) {
  let L = null;
  series.forEach((s, i) => {
    L = uLine(g, t, x0, y0, w, h - 12, s.ys, {
      title: i ? '' : title,
      n: NS.length,
      ym: 1,
      col: s.col,
      dots: true,
    });
  });
  g.font = MONO;
  g.textAlign = 'center';
  NS.forEach((n, i) => {
    g.fillStyle = i === cur ? t.cut : t.muted;
    g.fillText(String(n), L.X(i), y0 + h - 6);
  });
  if (cur >= 0) {
    g.strokeStyle = mix(t.gold, t.gold, 0, 0.7);
    g.lineWidth = 1.2;
    g.setLineDash([4, 4]);
    g.beginPath();
    g.moveTo(L.X(cur), y0 + 26);
    g.lineTo(L.X(cur), y0 + h - 32);
    g.stroke();
    g.setLineDash([]);
  }
  g.textAlign = 'right';
  series.forEach((s, i) => {
    g.fillStyle = s.col;
    g.fillText(s.lab, x0 + w - 14, y0 + h - 40 - 15 * (series.length - 1 - i));
  });
  return L;
}
function g2Levels(g, t, w, h1, data, specs, title) {
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'left';
  g.fillText(title, 8, 13);
  const pw = (w - 20) / specs.length;
  specs.forEach((sp, q) => {
    const x0 = 10 + q * pw,
      cx = x0 + pw / 2,
      cy = 20 + (h1 - 40) / 2,
      s = Math.min(pw, h1 - 40) / 5.6,
      rn = randnFrom(rng(q * 7 + 3));
    g.strokeStyle = t.line;
    g.lineWidth = 1;
    g.strokeRect(x0 + 2, 20, pw - 4, h1 - 40);
    g.fillStyle = mix(t.ink, t.ink, 0, 0.55);
    for (let i = 0; i < 160 && i < data.length; i++) {
      const p = data[i],
        x = sp.a * p[0] + sp.s * rn(),
        y = sp.a * p[1] + sp.s * rn();
      if (Math.abs(x) > 2.7 || Math.abs(y) > 2.6) continue;
      g.beginPath();
      g.arc(cx + x * s, cy - y * s, 1.5, 0, 7);
      g.fill();
    }
    g.fillStyle = sp.hi ? t.cut : t.muted;
    g.textAlign = 'center';
    g.fillText(sp.lab, cx, h1 - 6);
  });
}
function g2Panel(g, t, x0, y0, w, h, title, zoom = 1) {
  g.font = MONO;
  g.fillStyle = t.muted;
  g.textAlign = 'left';
  g.fillText(title, x0 + 8, y0 + 13);
  const s = Math.min((w - 16) / 4.6, (h - 26) / 3.5) * zoom,
    cx = x0 + w / 2,
    cy = y0 + 20 + (h - 24) / 2;
  g.save();
  g.beginPath();
  g.rect(x0 + 2, y0 + 18, w - 4, h - 20);
  g.clip();
  return { X: (v) => cx + (v * s) / G2S, Y: (v) => cy - (v * s) / G2S, w: x0 + w, h: y0 + h, s, g };
}
const g2Uni = (n) =>
  Float64Array.from({ length: 2 * n }, (_, i) => (Math.random() * 2 - 1) * (i % 2 ? 1.65 : 2.2));
/* normalizing flow: rotations and affine coupling layers */
function nfMake(seed, K = 8, H = 32) {
  const rn = randnFrom(rng(seed)),
    L = [];
  for (let k = 0; k < K; k++) {
    const a = Math.PI / 3,
      net = AD.mlp([1, H, H, 2], rn, 'relu'),
      w = net.ps[2];
    for (let i = 0; i < w.v.length; i++) w.v[i] *= 0.1;
    L.push({ net, a, R: AD.input([Math.cos(a), Math.sin(a), -Math.sin(a), Math.cos(a)], [2, 2]) });
  }
  return { L, ps: L.flatMap((l) => l.net.ps) };
}
function nfFwd(F, X, m) {
  let h = AD.input(X, [m, 2]);
  const As = [];
  for (const l of F.L) {
    h = AD.mm(h, l.R);
    const a = AD.cols(h, 0, 1),
      b = AD.cols(h, 1, 2),
      st = l.net.fwd(a),
      y = AD.aff(b, AD.cols(st, 0, 1), AD.cols(st, 1, 2), 1.2);
    As.push(y);
    h = AD.hcat(a, y);
  }
  return { z: h, As };
}
function nfLogp(F, X, m) {
  const { z, As } = nfFwd(F, X, m);
  AD.clear();
  const o = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    o[i] = -0.5 * (z.v[2 * i] ** 2 + z.v[2 * i + 1] ** 2) - Math.log(2 * Math.PI);
    for (const A of As) o[i] += A.s[i];
  }
  return o;
}
function nfInv(F, Z, m) {
  const h = Float64Array.from(Z),
    out = [Float64Array.from(h)];
  /* out[0] is z; then, for each layer undone, the cloud after the stretch is undone and after the turn is undone */ for (
    let k = F.L.length - 1;
    k >= 0;
    k--
  ) {
    const l = F.L[k],
      a = new Float64Array(m);
    for (let i = 0; i < m; i++) a[i] = h[2 * i];
    const st = l.net.fwd(AD.input(a, [m, 1]));
    AD.clear();
    const c = Math.cos(l.a),
      s = Math.sin(l.a);
    for (let i = 0; i < m; i++)
      h[2 * i + 1] = (h[2 * i + 1] - st.v[2 * i + 1]) * Math.exp(-1.2 * Math.tanh(st.v[2 * i]));
    out.push(Float64Array.from(h));
    for (let i = 0; i < m; i++) {
      const a0 = h[2 * i],
        b = h[2 * i + 1];
      h[2 * i] = a0 * c + b * s;
      h[2 * i + 1] = -a0 * s + b * c;
    }
    out.push(Float64Array.from(h));
  }
  return out;
}
/* energy-based model */
function ebE(net, P, n) {
  const o = net.fwd(AD.input(P, [n, 2]));
  AD.clear();
  return o.v;
}
function ebGrad(net, P, n) {
  const X = AD.input(P, [n, 2]),
    l = AD.mean(net.fwd(X));
  AD.backward(l);
  for (const p of net.ps) p.g.fill(0);
  const G = X.g;
  for (let i = 0; i < G.length; i++) G[i] *= n;
  return G;
}
function ebLang(net, P, n, k, a, sg, RN) {
  for (let q = 0; q < k; q++) {
    const G = ebGrad(net, P, n);
    for (let i = 0; i < n; i++) {
      P[2 * i] = clamp(P[2 * i] - a * clamp(G[2 * i], -6, 6) + sg * RN(), -2.25, 2.25);
      P[2 * i + 1] = clamp(P[2 * i + 1] - a * clamp(G[2 * i + 1], -6, 6) + sg * RN(), -1.7, 1.7);
    }
  }
}

GEN.push(
  {
    id: 'ddim',
    tab: 'DDIM and samplers',
    title: 'DDIM and samplers: the same network, far fewer steps',
    lede: 'The diffusion model in the earlier tabs walks through every noise level, one small step at a time. Real image models have a thousand levels, so that walk is slow. A sampler is the rule for getting from noise to data, and you can change it after training without touching the network. Here one network is trained once, with 100 noise levels, and then asked to finish the job in 100 steps, in 10, or in 3.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Faint dots: the real data. Gold dots: points being generated. Thin lines: the routes taken by fourteen of them, from the open circle where each one starts</figcaption><canvas class="short" id="dm-c" role="img" aria-label="Data, generated points and the routes they take"></canvas></figure>
 <figure class="fig"><figcaption>Top: where the network currently thinks each point will end up. Bottom: how many points land on the shape for each number of steps</figcaption><canvas class="short" id="dm-r" role="img" aria-label="The network's guess of the finished points and quality against number of steps"></canvas></figure></div>
 ${uHow('dm', ['Train as in the previous tab: add a random amount of noise to real points and teach the network to guess that noise. There are 100 noise levels.', 'Measure: generate with 1, 2, 3 and up to 100 steps, with and without added randomness, and count the points that land on the shape.', 'Generate: start from pure noise and choose a short list of levels to visit, for example 100, 80, 60, 40, 20, 0.', 'At each stop the network guesses the noise. Removing all of that noise gives a guess of the finished point.', 'Jump: put the guess back at the next level on the list by adding the right amount of noise. DDIM reuses the noise the network just guessed. The DDPM rule mixes in fresh random noise.', 'After the last jump the points are finished.'])}
 <div class="controls">${uBtns('dm', 'Run', 'One step', 'training steps')}
 ${grp('The sampler', slider('dm-n', 'Number of sampler steps', 0, 10, 1, 6) + slider('dm-eta', 'Fresh randomness at each step, η', 0, 1, 0.1, 0) + `<div class="row"><button class="btn" id="dm-same" type="button">Same Noise Again</button><button class="btn" id="dm-new" type="button">New Noise</button></div>`)}
 ${grp('The data', `<label class="ctl" for="dm-ds"><span>Shape to learn</span><select id="dm-ds">${g2Opts('ring')}</select></label>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ts', 'Training steps')}${rd('nc', 'Times the network was called')}${rd('near', 'Generated points that land on the shape', 'hl')}${rd('mv', 'Average shift from the last run with the same starting noise', 'hl2')}</dl>`,
    tryit:
      'Let it train and measure. With η at 0, set the steps to 100 and note how many points land on the shape, then try 20, 10, 5 and 2. How few steps can you get away with? Now press Same Noise Again several times: the shift readout stays at 0.00, because nothing random is left in the walk. Move η to 1 and repeat: the same starting noise now ends somewhere new each time, and the routes turn jittery. Now look at the two lines in the chart. Where do both fall, and how far apart are they at 5 steps and at 50? Last, watch the top-right picture on a 10-step run: how early does the network already know roughly where the points belong?',
    math: String.raw`<h4>Intuition</h4><p>Think of walking down a foggy mountain to a village. The DDPM rule is a cautious walker who takes a thousand tiny steps and is nudged sideways by a random gust after each one. DDIM is a walker who, at every stop, asks a guide "where do you think the village is?", then strides a long way in that direction and asks again. The guide is the same in both cases: it is the trained network. The first few answers are vague ("somewhere down there"), and later answers are sharp, so a handful of well-spaced questions is enough. Because there are no gusts, starting from the same spot always leads to the same door.</p>
 <h4>Two things the network's answer gives you</h4><p>At level \(t\) the noisy point is \(\mathbf x_t=\sqrt{\bar\alpha_t}\,\mathbf x_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon\). The network returns \(\hat{\boldsymbol\varepsilon}=\boldsymbol\varepsilon_\theta(\mathbf x_t,t)\). Solving the same equation for \(\mathbf x_0\) turns that into a guess of the finished point:</p>$$\hat{\mathbf x}_0=\frac{\mathbf x_t-\sqrt{1-\bar\alpha_t}\,\hat{\boldsymbol\varepsilon}}{\sqrt{\bar\alpha_t}} .$$<p>The top-right picture plots \(\hat{\mathbf x}_0\) for every point. Early on it is a blurry average of the data. It sharpens as the level falls.</p>
 <h4>The jump</h4><p>To move from level \(t\) to any lower level \(s\), not only \(t-1\), rebuild a noisy point at level \(s\) from the guess:</p>$$\mathbf x_s=\sqrt{\bar\alpha_s}\,\hat{\mathbf x}_0+\sqrt{1-\bar\alpha_s-\sigma^2}\;\hat{\boldsymbol\varepsilon}+\sigma\,\mathbf z,\qquad \mathbf z\sim\mathcal N(\mathbf 0,\mathbf I),$$$$\sigma=\eta\,\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}}\,\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_s}} .$$<p>The total noise in \(\mathbf x_s\) is always \(1-\bar\alpha_s\), which is what level \(s\) should contain. The slider \(\eta\) decides how much of it is new. With \(\eta=1\) and every level visited this is exactly the DDPM step from the previous tab. With \(\eta=0\) no new noise enters: this is <b>DDIM</b> (Song, Meng and Ermon, 2021).</p>
 <h4>Why skipping levels is allowed</h4><p>Training only ever used the one-jump formula for \(\mathbf x_t\) given \(\mathbf x_0\). It never used the chain of small steps. DDIM's observation is that many different step-by-step processes share those same one-jump distributions, so a network trained for one of them is already trained for all of them. Choosing a sampler is choosing among those processes after the fact. That is why the page trains once and then moves the two sliders freely.</p>
 <h4>What you give up and what you gain</h4><p>Each jump treats the network's current guess as if it stayed correct across the whole jump. Over a short jump that is nearly true. Over a long jump the direction should have bent, so the error grows as the number of steps falls: the bottom-right chart shows quality dropping at the left. With \(\eta=0\) the routes are smooth curves. With \(\eta=1\) every step also adds a random kick, and the routes are jagged. In large image models the \(\eta=0\) rule usually keeps its quality better when the steps are few, which is why DDIM made fast sampling practical. On this small two-dimensional problem the two lines in the chart stay close to each other. The clear effect here is the one both lines share: below about five steps, quality collapses.</p>
 <h4>The same noise gives the same picture</h4><p>With \(\eta=0\) the only randomness is the starting point \(\mathbf x_T\). Generation becomes a fixed map from noise to data. Three useful things follow. A result can be reproduced from its seed. Sliding smoothly between two starting noises gives a smooth blend between two outputs. And the map can be run backwards, from a real picture to the noise that would produce it, which is how many image-editing methods begin.</p>
 <h4>Other samplers</h4><p>As the steps shrink, the \(\eta=0\) rule traces the solution of an ordinary differential equation, called the probability-flow ODE. Once generation is seen as solving an ODE, any numerical solver can be used. Euler's method is the plain version. Heun's method and the DPM-Solver family look at the slope more than once per step and reach the same quality with fewer network calls. The sampler names in image tools, such as "Euler", "Heun", "DPM++" and "DDIM", are these choices. The network is the same in every case.</p>
 <h4>What this page trains</h4><p>The same small network as the previous tab, two hidden layers of 64 neurons, trained for 2,500 batches of 128 with \(T=100\) noise levels. The quality number is the share of generated points that fall within 0.1 of a real data point. For the chart it is measured on 200 fresh points for each setting, so it wobbles by a few percent from one training run to the next.</p>`,
    q: 'A classmate says: "DDIM with 10 steps must need a network trained with 10 noise levels." What is wrong with that?',
    a: 'The network is trained with all 100 levels and is never retrained. Training only teaches it to guess the noise in a point at any single level. How those guesses are strung together into a walk is decided afterwards by the sampler. DDIM visits 10 of the 100 levels and uses the network’s guess at each one to jump straight to the next, so the same weights serve 100 steps, 10 steps or 3.',
    init(root) {
      const c = $('#dm-c', root),
        cr = $('#dm-r', root),
        live = $('#dm-live', root),
        S = { ds: 'ring', ni: 6, eta: 0 },
        Tn = 100,
        MAXS = 2500,
        M = 400,
        NS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100],
        BE = Array.from({ length: Tn }, (_, i) => 0.0008 + (0.0792 * i) / (Tn - 1)),
        AB = [],
        RN = randnFrom(Math.random);
      {
        let p = 1;
        for (const b of BE) {
          p *= 1 - b;
          AB.push(p);
        }
      }
      let net,
        ts = 0,
        LL = [],
        sm = [],
        phase = 'train',
        data = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        z0 = null,
        xs = null,
        x0h = null,
        seq = [],
        si = 0,
        done = false,
        tr = [],
        prevOut = null,
        moved = NaN,
        Q = [[], []],
        bi = 0,
        nearV = NaN;
      const ab = (k) => (k > 0 ? AB[k - 1] : 1),
        pred = (P, k) => g2Pred(net, P, g2Emb(k / Tn));
      const build = () => {
        net = AD.mlp([10, 64, 64, 2], randnFrom(rng(run * 7 + 4)), 'relu');
        ts = 0;
        LL = [];
        sm = [];
        phase = 'train';
        xs = null;
        x0h = null;
        done = false;
        cu = 0;
        Q = [NS.map(() => NaN), NS.map(() => NaN)];
        bi = 0;
        z0 = null;
        prevOut = null;
        moved = NaN;
        nearV = NaN;
        tr = [];
        si = 0;
        data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
        msg =
          'A network with random weights. Press Run: it trains first, then the page tries it with different numbers of sampler steps.';
      };
      const train = (k) => {
        const B = 128;
        for (let q = 0; q < k && ts < MAXS; q++) {
          const X = new Float64Array(B * 10),
            E = new Float64Array(B * 2);
          for (let i = 0; i < B; i++) {
            const p = g2Samp(S.ds, RN),
              tt = 1 + Math.floor(Math.random() * Tn),
              e0 = RN(),
              e1 = RN(),
              s = Math.sqrt(ab(tt)),
              nq = Math.sqrt(1 - ab(tt)),
              e = g2Emb(tt / Tn);
            X[i * 10] = s * p[0] + nq * e0;
            X[i * 10 + 1] = s * p[1] + nq * e1;
            for (let j = 0; j < 8; j++) X[i * 10 + 2 + j] = e[j];
            E[2 * i] = e0;
            E[2 * i + 1] = e1;
          }
          const l = AD.mse(net.fwd(AD.input(X, [B, 10])), E);
          LL.push(l.v[0]);
          AD.backward(l);
          AD.adam(net.ps, 0.003 * Math.pow(0.1, ts / MAXS), ++ts, 1);
          if (ts % 20 === 0) {
            let s = 0;
            for (let j = ts - 20; j < ts; j++) s += LL[j];
            sm.push(s / 20);
          }
        }
        cu = 0;
        msg =
          'Training step ' +
          ts +
          ' of ' +
          MAXS +
          ': 128 real points were given random amounts of noise, from level 1 to level ' +
          Tn +
          ', and the network guessed the noise in each. Average error over the last 20 steps: ' +
          f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
          '.';
        if (ts >= MAXS) {
          phase = 'bench';
          bi = 0;
          cu = 1;
          msg =
            'Training is finished and the network will not change again. Next it is used with eleven different numbers of sampler steps, with and without fresh randomness.';
        }
      };
      const mkSeq = (N) => {
        const s = [];
        for (let j = N; j >= 0; j--) s.push(Math.round((j * Tn) / N));
        return s;
      };
      const sstep = (x, k, kp, eta) => {
        const e = pred(x, k),
          a = ab(k),
          ap = ab(kp),
          sa = Math.sqrt(a),
          sn = Math.sqrt(1 - a),
          sig = kp > 0 ? eta * Math.sqrt((1 - ap) / (1 - a)) * Math.sqrt(1 - a / ap) : 0,
          dir = Math.sqrt(Math.max(0, 1 - ap - sig * sig)),
          sp = Math.sqrt(ap),
          h = new Float64Array(x.length);
        for (let i = 0; i < x.length; i++) {
          const x0 = clamp((x[i] - sn * e[i]) / sa, -2.5, 2.5),
            e2 = (x[i] - sa * x0) / sn;
          h[i] = x0;
          x[i] = sp * x0 + dir * e2 + (sig ? sig * RN() : 0);
        }
        return h;
      };
      const sampleAll = (N, eta, Mm) => {
        const x = Float64Array.from({ length: 2 * Mm }, () => RN()),
          s = mkSeq(N);
        for (let j = 0; j < N; j++) sstep(x, s[j], s[j + 1], eta);
        return g2Near(x, Mm, data, 0.1);
      };
      const startGen = (fresh) => {
        if (fresh || !z0) {
          z0 = Float64Array.from({ length: 2 * M }, () => RN());
          prevOut = null;
          moved = NaN;
        }
        xs = Float64Array.from(z0);
        x0h = null;
        seq = mkSeq(NS[S.ni]);
        si = 0;
        done = false;
        nearV = NaN;
        tr = Array.from({ length: 14 }, (_, i) => [[xs[2 * i], xs[2 * i + 1]]]);
      };
      const bench = () => {
        const j = bi >> 1,
          e = bi & 1;
        Q[e][j] = sampleAll(NS[j], e, 200);
        bi++;
        cu = 1;
        msg =
          'Measuring: ' +
          NS[j] +
          ' step' +
          (NS[j] > 1 ? 's' : '') +
          ' with η = ' +
          e +
          (e ? ' (fresh noise at every step, the DDPM rule)' : ' (no fresh noise, the DDIM rule)') +
          ' put ' +
          pct(Q[e][j]) +
          ' of 200 new points on the shape.';
        if (bi >= 2 * NS.length) {
          phase = 'gen';
          startGen(true);
          cu = 2;
          msg =
            'All settings are measured: see the chart at the bottom right. Now watch one run. ' +
            M +
            ' points of pure noise will visit these levels: ' +
            seq.join(', ') +
            '.';
        }
      };
      const gstep = () => {
        if (done || !xs) return;
        const k = seq[si],
          kp = seq[si + 1];
        x0h = sstep(xs, k, kp, S.eta);
        si++;
        tr.forEach((p, i) => p.push([xs[2 * i], xs[2 * i + 1]]));
        cu = 4;
        msg =
          'Step ' +
          si +
          ' of ' +
          NS[S.ni] +
          ': the network guessed the noise at level ' +
          k +
          '. Removing it gave the guessed finished points (top right). They were put back at level ' +
          kp +
          (kp === 0
            ? ', which is no noise at all.'
            : S.eta > 0
              ? ', using the guessed noise mixed with fresh random noise.'
              : ', using the same guessed noise and nothing random.');
        if (si >= seq.length - 1) {
          done = true;
          cu = 5;
          nearV = g2Near(xs, M, data, 0.1);
          if (prevOut) {
            let d = 0;
            for (let i = 0; i < M; i++)
              d += Math.hypot(xs[2 * i] - prevOut[2 * i], xs[2 * i + 1] - prevOut[2 * i + 1]);
            moved = d / M;
          }
          prevOut = Float64Array.from(xs);
          msg +=
            ' Finished after ' +
            si +
            ' call' +
            (si > 1 ? 's' : '') +
            ' of the network: ' +
            pct(nearV) +
            ' of the points lie on the shape.' +
            (isFinite(moved)
              ? ' Compared with the last run from the same starting noise, each point ended ' +
                f2(moved) +
                ' away on average.'
              : '');
        }
      };
      const step = () => {
        if (phase === 'train') train(50);
        else if (phase === 'bench') bench();
        else if (done) {
          startGen(false);
          cu = 2;
          msg = 'The same starting noise again. Press Run or One step.';
        } else gstep();
      };
      const draw = () => {
        const t = T();
        {
          const V = uView(c),
            { g, h } = V;
          g2Data(g, V, t, data);
          if (xs) {
            g2Trails(g, V, t, tr);
            g2Pts(g, V, t, xs, M, t.gold);
          }
          g2Note(
            g,
            t,
            h,
            phase === 'gen'
              ? NS[S.ni] +
                  ' step' +
                  (NS[S.ni] > 1 ? 's' : '') +
                  ', η = ' +
                  f2(S.eta) +
                  (xs ? ', now at noise level ' + seq[si] : '')
              : phase === 'bench'
                ? 'measuring each number of steps'
                : 'training the network',
          );
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.52;
          if (phase === 'gen' && x0h) {
            const P = g2Panel(
              g,
              t,
              0,
              0,
              w,
              h1,
              'the network’s guess of where each point will finish, at this step',
              1.22,
            );
            g2Data(g, P, t, data, 0.22, 1.5);
            g2Pts(g, P, t, x0h, M, t.accent, 2);
            g.restore();
          } else
            g2Levels(
              g,
              t,
              w,
              h1,
              data,
              [0, 25, 50, 100].map((k) => ({
                a: Math.sqrt(ab(k)),
                s: Math.sqrt(1 - ab(k)),
                lab: 'level ' + k,
              })),
              'the real data at four of the 100 noise levels: what training sees',
            );
          if (phase === 'train')
            uLine(g, t, 0, h1, w, h - h1, sm, {
              title: 'error at guessing the noise (average of every 20 steps)',
              n: MAXS / 20,
              col: t.ok,
              ym: Math.max(1.05, sm[0] || 1.05),
            });
          else
            g2Bench(
              g,
              t,
              0,
              h1,
              w,
              h - h1,
              NS,
              [
                { ys: Q[1], col: t.cut, lab: 'η = 1: fresh noise each step' },
                { ys: Q[0], col: t.accent, lab: 'η = 0: DDIM' },
              ],
              phase === 'gen' ? S.ni : -1,
              'share of points on the shape, by number of sampler steps',
            );
        }
        live.textContent = msg;
        uCur(root, 'dm', cu);
        setR(
          root,
          'ph',
          phase === 'train'
            ? ts
              ? 'Training'
              : 'Not started'
            : phase === 'bench'
              ? 'Measuring'
              : done
                ? 'Finished'
                : 'Generating',
        );
        setR(root, 'ts', ts + ' of ' + MAXS);
        setR(root, 'nc', phase === 'gen' ? si + ' of ' + NS[S.ni] : '–');
        setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
        setR(root, 'mv', isFinite(moved) && done ? f2(moved) : '–');
      };
      build();
      if (RM) {
        train(MAXS);
        while (phase === 'bench') bench();
        while (!done) gstep();
      }
      const anim = Anim(() => {
        fr++;
        if (phase === 'train') {
          train(Math.min(ts < 60 ? 1 : 5, chk.remaining()));
          draw();
          chk();
          return;
        }
        if (phase === 'bench') {
          if (fr % 4 === 0) {
            bench();
            draw();
          }
          return;
        }
        if (fr % clamp(Math.round(220 / NS[S.ni]), 4, 40) === 0 && !done) gstep();
        draw();
        if (done) pb.stop();
      });
      pb = playBtn($('#dm-play', root), anim);
      chk = runCtl(pb, root, 'dm', {
        done: () => phase === 'gen' && done,
        restart: () => startGen(false),
        count: () => ts,
        what: 'training steps',
      });
      const regen = (fresh) => {
        if (phase !== 'gen') {
          msg = 'The sampler is used once training and measuring are finished. Press Run.';
          draw();
          return;
        }
        pb.stop();
        startGen(fresh);
        draw();
        pb.start();
      };
      $('#dm-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#dm-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#dm-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        pb.stop();
        run++;
        build();
        draw();
        pb.start();
      };
      $('#dm-same', root).onclick = () => regen(false);
      $('#dm-new', root).onclick = () => regen(true);
      bind(
        root,
        'dm-n',
        (v) => {
          S.ni = +v;
          if (pb && phase === 'gen') regen(false);
        },
        (v) => String(NS[+v]),
      );
      bind(
        root,
        'dm-eta',
        (v) => {
          S.eta = +v;
          if (pb && phase === 'gen') regen(false);
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
  },

  {
    id: 'flow-matching',
    tab: 'Flow matching',
    title: 'Flow matching: draw a straight line from noise to data',
    lede: 'Diffusion reaches data by a winding, noisy road. Flow matching asks for the simplest road there is: pair each noise point with a data point, join them with a straight line, and teach a network the direction of travel along those lines. Generating is then following the arrows from noise for a few steps. Many recent image and video models are trained this way.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Faint dots: the real data. Arrows: the velocity the network has learned, at the time shown. Gold dots: points being carried from noise to data. Thin lines: fourteen of their routes</figcaption><canvas class="short" id="fm-c" role="img" aria-label="Data, the learned velocity field and points flowing along it"></canvas></figure>
 <figure class="fig"><figcaption>Top: what training sees. Each line joins one noise point (open) to one data point (filled). Bottom: the training error, then quality for each number of steps</figcaption><canvas class="short" id="fm-r" role="img" aria-label="Training pairs, the loss and quality against number of steps"></canvas></figure></div>
 ${uHow('fm', ['Draw a random noise point and a random real point. Join them with a straight line.', 'Pick a random time between 0 and 1 and stand that far along the line. The correct velocity there is simply “data point minus noise point”.', 'Ask the network for the velocity at that place and time, compare with the correct one, and backpropagate. Repeat for 2,500 batches.', 'Measure: generate with 1, 2, 3 and up to 40 steps and count the points that land on the shape.', 'Generate: start from pure noise at time 0. At each step ask the network for the velocity and move every point a short way along it.', 'At time 1 the points have arrived.'])}
 <div class="controls">${uBtns('fm', 'Run', 'One step', 'training steps')}
 ${grp('The sampler', slider('fm-n', 'Number of steps from noise to data', 0, 8, 1, 5) + `<div class="row seg" id="fm-sol" role="group" aria-label="Solver"><button class="btn" type="button" data-v="euler">Euler: One Look per Step</button><button class="btn" type="button" data-v="mid">Midpoint: Two Looks per Step</button></div><div class="row"><button class="btn" id="fm-same" type="button">Same Noise Again</button><button class="btn" id="fm-new" type="button">New Noise</button></div>`)}
 ${grp('The data', `<label class="ctl" for="fm-ds"><span>Shape to learn</span><select id="fm-ds">${g2Opts('ring')}</select></label>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ts', 'Training steps')}${rd('L', 'Error at guessing the velocity', 'hl2')}${rd('nc', 'Times the network was called')}${rd('near', 'Generated points that land on the shape', 'hl')}${rd('st', 'How straight the routes are (1 = ruler-straight)')}</dl>`,
    tryit:
      'Watch the top-right picture during training: every arrow there is known exactly, so the network has a clear target at every step. When generation starts, look at the routes on the left. They are not perfectly straight even though every training line was. Why not? (Hint: many lines cross at the same place, and the network can only give one answer there.) Now set the steps to 1: every point jumps in a single stride. Where do they land? Try 2, 4 and 8. Switch to Midpoint and compare 4 steps of Midpoint with 8 steps of Euler: both call the network eight times. Which lands more points on the shape?',
    math: String.raw`<h4>Intuition</h4><p>Picture a crowd spread randomly over a field, and a set of chairs arranged in a ring. Give each person one chair at random and tell them to walk straight to it at steady speed, arriving after one minute. Now film the crowd from above. At every spot and every moment there is a typical direction in which people are moving. Flow matching trains a network to report that typical direction. Afterwards the people and their assignments are no longer needed: drop a new crowd on the field, tell everyone to follow the reported directions, and after one minute they are sitting in a ring.</p>
 <h4>The straight-line path</h4><p>Take noise \(\mathbf x_0\sim\mathcal N(\mathbf 0,\mathbf I)\) and a real point \(\mathbf x_1\). For a time \(t\in[0,1]\) define the point on the line between them and its velocity:</p>$$\mathbf x_t=(1-t)\,\mathbf x_0+t\,\mathbf x_1,\qquad \frac{d\mathbf x_t}{dt}=\mathbf x_1-\mathbf x_0 .$$<p>The velocity does not depend on \(t\): each person walks at constant speed. This is the choice used by rectified flow (Liu, Gong and Liu, 2023) and by conditional flow matching (Lipman and others, 2023).</p>
 <h4>Training</h4><p>The network \(\mathbf v_\theta(\mathbf x,t)\) is trained by plain regression onto that velocity:</p>$$\mathcal L=\mathbb E_{\,t,\ \mathbf x_0,\ \mathbf x_1}\big\lVert \mathbf v_\theta(\mathbf x_t,t)-(\mathbf x_1-\mathbf x_0)\big\rVert^2 .$$<p>As with diffusion, the target is known exactly because we built the path ourselves. There is no noise schedule to design and no \(\bar\alpha_t\).</p>
 <h4>What the network actually learns</h4><p>Many different lines pass through the same place at the same time, each with its own velocity, and the network can give only one answer. The answer that minimises a squared error is the average:</p>$$\mathbf v^\star(\mathbf x,t)=\mathbb E\big[\mathbf x_1-\mathbf x_0\ \big|\ \mathbf x_t=\mathbf x\big].$$<p>So the error can never reach zero, for the same reason as in the diffusion tab. The remarkable fact is that following this averaged field still moves the whole noise distribution onto the whole data distribution. Individual routes bend where lines crossed, because the field cannot let two routes cross, but the crowd as a whole ends in the right place.</p>
 <h4>Generating</h4><p>Solve the ordinary differential equation \(d\mathbf x/dt=\mathbf v_\theta(\mathbf x,t)\) from \(t=0\) to \(t=1\). With \(N\) steps of Euler's method and \(h=1/N\):</p>$$\mathbf x_{t+h}=\mathbf x_t+h\,\mathbf v_\theta(\mathbf x_t,t).$$<p>The midpoint method spends two network calls per step: it takes half a step, looks at the velocity there, and uses that velocity for the full step,</p>$$\mathbf x_{t+h}=\mathbf x_t+h\,\mathbf v_\theta\big(\mathbf x_t+\tfrac h2\mathbf v_\theta(\mathbf x_t,t),\ t+\tfrac h2\big).$$<p>Nothing random is added along the way, so the same starting noise always gives the same result, as with DDIM.</p>
 <h4>Why straightness matters</h4><p>Euler's method is exact on a straight route travelled at constant speed, whatever the step size. The straighter the learned routes, the fewer steps are needed. The readout compares the straight-line distance from start to finish with the length of the route actually taken. One further idea, called reflow, generates noise and data pairs with the trained model and retrains on those pairs. Their lines cross less, the routes become straighter, and good samples can then be made in one or two steps.</p>
 <h4>How it relates to diffusion</h4><p>A diffusion model's \(\eta=0\) sampler also solves an ODE from noise to data. The difference is the path that training prescribes between the two ends: curved and tied to a noise schedule for diffusion, straight for flow matching. One can be rewritten as the other with a change of time and scale. The network's output is a velocity here and a noise estimate there, and each can be computed from the other.</p>
 <h4>What this page trains</h4><p>A network with two hidden layers of 64 neurons, taking the two coordinates and eight numbers that encode the time, trained with Adam for 2,500 batches of 128. Training times are drawn a little more often from the data end of the line, where the field has the most detail. The quality number is the share of generated points within 0.1 of a real data point, measured on 200 fresh points for each setting in the chart.</p>`,
    q: 'With one Euler step, every point moves from its starting place by exactly the velocity the network gives at time 0. Why does that produce a blob near the middle of the data instead of the shape?',
    a: 'At time 0 a point is pure noise and carries no information about which data point it was paired with, because the pairing was random. The best single answer for its velocity is therefore the average over all data points: “head for the centre of the data”. One step with that velocity sends everything toward the centre. Only later, when a point has moved and its position starts to say something about where it is heading, does the field split toward the different parts of the shape. That needs more than one step.',
    init(root) {
      const c = $('#fm-c', root),
        cr = $('#fm-r', root),
        live = $('#fm-live', root),
        S = { ds: 'ring', ni: 5, sol: 'euler' },
        MAXS = 2500,
        M = 400,
        NS = [1, 2, 3, 4, 6, 8, 12, 20, 40],
        RN = randnFrom(Math.random);
      let net,
        ts = 0,
        LL = [],
        sm = [],
        phase = 'train',
        data = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        z0 = null,
        xs = null,
        si = 0,
        done = false,
        tr = [],
        Q = [[], []],
        bi = 0,
        nearV = NaN,
        plen = null,
        calls = 0,
        strt = NaN,
        pairs = [];
      const pred = (P, u) => g2Pred(net, P, g2Emb(u));
      const build = () => {
        net = AD.mlp([10, 64, 64, 2], randnFrom(rng(run * 7 + 4)), 'relu');
        ts = 0;
        LL = [];
        sm = [];
        phase = 'train';
        xs = null;
        done = false;
        cu = 0;
        Q = [NS.map(() => NaN), NS.map(() => NaN)];
        bi = 0;
        z0 = null;
        nearV = NaN;
        strt = NaN;
        tr = [];
        si = 0;
        calls = 0;
        data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
        pairs = Array.from({ length: 9 }, (_, i) => ({ a: [RN(), RN()], b: data[i] }));
        msg = 'A network with random weights: its arrows mean nothing yet. Press Run to train it.';
      };
      const train = (k) => {
        const B = 128;
        for (let q = 0; q < k && ts < MAXS; q++) {
          const X = new Float64Array(B * 10),
            E = new Float64Array(B * 2);
          for (let i = 0; i < B; i++) {
            const p = g2Samp(S.ds, RN),
              u = Math.pow(Math.random(), 0.6),
              a0 = RN(),
              a1 = RN(),
              e = g2Emb(u);
            X[i * 10] = (1 - u) * a0 + u * p[0];
            X[i * 10 + 1] = (1 - u) * a1 + u * p[1];
            for (let j = 0; j < 8; j++) X[i * 10 + 2 + j] = e[j];
            E[2 * i] = p[0] - a0;
            E[2 * i + 1] = p[1] - a1;
          }
          const l = AD.mse(net.fwd(AD.input(X, [B, 10])), E);
          LL.push(l.v[0]);
          AD.backward(l);
          AD.adam(net.ps, 0.004 * Math.pow(0.1, ts / MAXS), ++ts, 1);
          if (ts % 20 === 0) {
            let s = 0;
            for (let j = ts - 20; j < ts; j++) s += LL[j];
            sm.push(s / 20);
          }
        }
        cu = Math.floor(ts / 100) % 3;
        msg =
          'Training step ' +
          ts +
          ' of ' +
          MAXS +
          ': 128 noise points were each joined to a random real point. The network was asked for the velocity at a random place along each line and compared with the true one. Average error over the last 20 steps: ' +
          f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
          '.';
        if (ts >= MAXS) {
          phase = 'bench';
          bi = 0;
          cu = 3;
          msg =
            'Training is finished. Next the network is used with nine different numbers of steps and with both solvers.';
        }
      };
      const ode = (x, j, N, sol) => {
        const dt = 1 / N,
          u = j / N,
          v = pred(x, u);
        if (sol === 'mid') {
          const xm = x.map((q, i) => q + (v[i] * dt) / 2),
            v2 = pred(xm, u + dt / 2);
          for (let i = 0; i < x.length; i++) x[i] += v2[i] * dt;
        } else for (let i = 0; i < x.length; i++) x[i] += v[i] * dt;
      };
      const sampleAll = (N, sol, Mm) => {
        const x = Float64Array.from({ length: 2 * Mm }, () => RN());
        for (let j = 0; j < N; j++) ode(x, j, N, sol);
        return g2Near(x, Mm, data, 0.1);
      };
      const startGen = (fresh) => {
        if (fresh || !z0) z0 = Float64Array.from({ length: 2 * M }, () => RN());
        xs = Float64Array.from(z0);
        si = 0;
        done = false;
        nearV = NaN;
        strt = NaN;
        calls = 0;
        plen = new Float64Array(M);
        tr = Array.from({ length: 14 }, (_, i) => [[xs[2 * i], xs[2 * i + 1]]]);
      };
      const bench = () => {
        const j = bi >> 1,
          e = bi & 1;
        Q[e][j] = sampleAll(NS[j], e ? 'mid' : 'euler', 200);
        bi++;
        cu = 3;
        msg =
          'Measuring: ' +
          NS[j] +
          ' step' +
          (NS[j] > 1 ? 's' : '') +
          ' of ' +
          (e
            ? 'Midpoint (' + 2 * NS[j] + ' network calls)'
            : 'Euler (' + NS[j] + ' network call' + (NS[j] > 1 ? 's' : '') + ')') +
          ' put ' +
          pct(Q[e][j]) +
          ' of 200 new points on the shape.';
        if (bi >= 2 * NS.length) {
          phase = 'gen';
          startGen(true);
          cu = 4;
          msg =
            'All settings are measured: see the chart at the bottom right. Now watch one run: ' +
            M +
            ' points of pure noise at time 0.';
        }
      };
      const gstep = () => {
        if (done || !xs) return;
        const N = NS[S.ni],
          o = Float64Array.from(xs);
        ode(xs, si, N, S.sol);
        si++;
        calls += S.sol === 'mid' ? 2 : 1;
        for (let i = 0; i < M; i++)
          plen[i] += Math.hypot(xs[2 * i] - o[2 * i], xs[2 * i + 1] - o[2 * i + 1]);
        tr.forEach((p, i) => p.push([xs[2 * i], xs[2 * i + 1]]));
        cu = 4;
        msg =
          'Step ' +
          si +
          ' of ' +
          N +
          ': the network gave a velocity for every point at time ' +
          f2((si - 1) / N) +
          (S.sol === 'mid' ? ', each point looked again from half a step ahead,' : '') +
          ' and every point moved along it for a time of ' +
          f2(1 / N) +
          '. The time is now ' +
          f2(si / N) +
          '.';
        if (si >= N) {
          done = true;
          cu = 5;
          nearV = g2Near(xs, M, data, 0.1);
          let s = 0;
          for (let i = 0; i < M; i++)
            s += Math.hypot(xs[2 * i] - z0[2 * i], xs[2 * i + 1] - z0[2 * i + 1]) / (plen[i] || 1);
          strt = s / M;
          msg +=
            ' Arrived: ' +
            pct(nearV) +
            ' of the points lie on the shape, after ' +
            calls +
            ' call' +
            (calls > 1 ? 's' : '') +
            ' of the network.';
        }
      };
      const step = () => {
        if (phase === 'train') train(50);
        else if (phase === 'bench') bench();
        else if (done) {
          startGen(false);
          cu = 4;
          msg = 'The same starting noise again, at time 0. Press Run or One step.';
        } else gstep();
      };
      const draw = () => {
        const t = T(),
          N = NS[S.ni],
          u = phase === 'gen' ? Math.min(1, si / N) : (Math.floor(fr / 3) % 100) / 100;
        {
          const V = uView(c),
            { g, h } = V;
          g2Data(g, V, t, data);
          if (ts > 0) g2Arrows(g, V, pred(G2G, Math.min(u, 0.999)), t.accent, 1, 0.55, 6);
          if (xs) {
            g2Trails(g, V, t, tr);
            g2Pts(g, V, t, xs, M, t.gold);
          }
          g2Note(
            g,
            t,
            h,
            (ts > 0 ? 'arrows shown for time ' + f2(u) : 'not trained yet') +
              (phase === 'gen'
                ? ', ' +
                  N +
                  ' step' +
                  (N > 1 ? 's' : '') +
                  ' of ' +
                  (S.sol === 'mid' ? 'Midpoint' : 'Euler')
                : ''),
          );
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5,
            P = g2Panel(
              g,
              t,
              0,
              0,
              w,
              h1,
              'training pairs at time ' + f2(u) + ': the arrow is the right answer at the gold dot',
            );
          for (const p of pairs) {
            const ax = P.X(p.a[0]),
              ay = P.Y(p.a[1]),
              bx = P.X(p.b[0]),
              by = P.Y(p.b[1]),
              mx = ax + (bx - ax) * u,
              my = ay + (by - ay) * u,
              d = Math.hypot(bx - ax, by - ay) || 1;
            g.strokeStyle = mix(t.ink, t.ink, 0, 0.3);
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(ax, ay);
            g.lineTo(bx, by);
            g.stroke();
            g.fillStyle = t.bg;
            g.strokeStyle = t.ink;
            g.beginPath();
            g.arc(ax, ay, 3.5, 0, 7);
            g.fill();
            g.stroke();
            g.fillStyle = t.ink;
            g.beginPath();
            g.arc(bx, by, 3.5, 0, 7);
            g.fill();
            g.strokeStyle = t.accent;
            g.lineWidth = 2;
            g.beginPath();
            g.moveTo(mx, my);
            g.lineTo(mx + ((bx - ax) / d) * 16, my + ((by - ay) / d) * 16);
            g.stroke();
            g.fillStyle = t.gold;
            g.beginPath();
            g.arc(mx, my, 4, 0, 7);
            g.fill();
          }
          g.restore();
          if (phase === 'train')
            uLine(g, t, 0, h1, w, h - h1, sm, {
              title: 'error at guessing the velocity (average of every 20 steps)',
              n: MAXS / 20,
              col: t.ok,
              ym: Math.max(2.2, sm[0] || 2.2),
            });
          else
            g2Bench(
              g,
              t,
              0,
              h1,
              w,
              h - h1,
              NS,
              [
                { ys: Q[1], col: t.cut, lab: 'Midpoint: two calls per step' },
                { ys: Q[0], col: t.accent, lab: 'Euler: one call per step' },
              ],
              phase === 'gen' ? S.ni : -1,
              'share of points on the shape, by number of steps',
            );
        }
        live.textContent = msg;
        uCur(root, 'fm', cu);
        setR(
          root,
          'ph',
          phase === 'train'
            ? ts
              ? 'Training'
              : 'Not started'
            : phase === 'bench'
              ? 'Measuring'
              : done
                ? 'Finished'
                : 'Generating',
        );
        setR(root, 'ts', ts + ' of ' + MAXS);
        setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
        setR(root, 'nc', phase === 'gen' ? String(calls) : '–');
        setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
        setR(root, 'st', isFinite(strt) ? f2(strt) : '–');
      };
      build();
      if (RM) {
        train(MAXS);
        while (phase === 'bench') bench();
        while (!done) gstep();
      }
      const anim = Anim(() => {
        fr++;
        if (phase === 'train') {
          train(Math.min(ts < 60 ? 1 : 5, chk.remaining()));
          draw();
          chk();
          return;
        }
        if (phase === 'bench') {
          if (fr % 4 === 0) {
            bench();
            draw();
          }
          return;
        }
        if (fr % clamp(Math.round(200 / NS[S.ni]), 5, 40) === 0 && !done) gstep();
        draw();
        if (done) pb.stop();
      });
      pb = playBtn($('#fm-play', root), anim);
      chk = runCtl(pb, root, 'fm', {
        done: () => phase === 'gen' && done,
        restart: () => startGen(false),
        count: () => ts,
        what: 'training steps',
      });
      const regen = (fresh) => {
        if (phase !== 'gen') {
          msg = 'The sampler is used once training and measuring are finished. Press Run.';
          draw();
          return;
        }
        pb.stop();
        startGen(fresh);
        draw();
        pb.start();
      };
      $('#fm-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#fm-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#fm-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        pb.stop();
        run++;
        build();
        draw();
        pb.start();
      };
      $('#fm-same', root).onclick = () => regen(false);
      $('#fm-new', root).onclick = () => regen(true);
      bind(
        root,
        'fm-n',
        (v) => {
          S.ni = +v;
          if (pb && phase === 'gen') regen(false);
        },
        (v) => String(NS[+v]),
      );
      segBind(root, 'fm-sol', S.sol, (v) => {
        S.sol = v;
        if (phase === 'gen') regen(false);
      });
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
  {
    id: 'score-matching',
    tab: 'Score matching',
    title: 'Score matching: learn which way is “more likely”',
    lede: 'Suppose you could stand anywhere in the plane and ask one question: in which direction does the data get denser? That arrow is called the score. With arrows everywhere you can generate data by dropping points at random and letting them climb, with a little jitter. The surprise is that the arrows can be learned without ever knowing the density itself, by practising one simple job: removing noise.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Faint dots: the real data. Arrows: the score the network has learned at the noise level shown. Gold dots: points that are climbing the arrows</figcaption><canvas class="short" id="sm-c" role="img" aria-label="Data, the learned score field and samples"></canvas></figure>
 <figure class="fig"><figcaption>Top: the exact score at the same noise level, worked out from the data for comparison. Bottom: the training error, then how well the learned arrows match the exact ones at each level</figcaption><canvas class="short" id="sm-r" role="img" aria-label="Exact score field, loss and agreement by noise level"></canvas></figure></div>
 ${uHow('sm', ['Take real points. Choose one of ten noise levels for each and add that much noise.', 'Ask the network which way leads back toward the clean point. The right answer is known, because we added the noise ourselves. Backpropagate.', 'Repeat for 2,500 batches. The network now gives an arrow at every place, for every noise level.', 'Generate: scatter points at random over the whole picture.', 'Langevin step: move each point a little way along its arrow and add a little random jitter. Repeat.', 'Annealing: start with the arrows for the largest noise level, which reach everywhere, and work down to the smallest, which are precise.'])}
 <div class="controls">${uBtns('sm', 'Run', 'One step', 'training steps')}
 ${grp('Generating', `<div class="row seg" id="sm-mode" role="group" aria-label="Which arrows to follow"><button class="btn" type="button" data-v="ann">All Ten Levels, Large to Small</button><button class="btn" type="button" data-v="one">Only the Level Shown</button></div>` + slider('sm-lv', 'Noise level shown (1 = most noise)', 1, 10, 1, 5) + `<div class="row"><button class="btn" id="sm-gen" type="button">Generate Again</button></div>`)}
 ${grp('The data', `<label class="ctl" for="sm-ds"><span>Shape to learn</span><select id="sm-ds">${g2Opts('blobs')}</select></label>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ts', 'Training steps')}${rd('L', 'Error at guessing the noise', 'hl2')}${rd('sg', 'Noise level shown, and its size σ')}${rd('ag', 'Learned arrows within 25° of the exact ones, at that level')}${rd('near', 'Generated points that land on the shape', 'hl')}</dl>`,
    tryit:
      'After training, drag the noise-level slider from 1 to 10 and compare the two arrow pictures. At level 1 the arrows are simple: everything points at the middle of the data. At level 10 they are detailed near the data. What do the learned arrows do far from the data at level 10, and how does the bar chart show it? Now choose Only the Level Shown with level 10 and press Generate Again. Do all the points find the shape? Try level 1, then a middle level such as 5 or 6. On these small shapes a well-chosen middle level can do about as well as the full ladder. What would you have to know in advance to choose it? Then switch back to All Ten Levels: which problem of the smallest level does starting large solve, and which problem of the largest level does finishing small solve?',
    math: String.raw`<h4>Intuition</h4><p>Imagine a dark hillside with a few campfires. You cannot see the fires, but everywhere you stand you can feel which way is warmer. Follow the warmth and you reach a fire. The score is that "warmer this way" arrow, with data density in place of heat. There is one difficulty: far from every fire you feel almost nothing, and what you do feel is unreliable. The fix is to begin with huge bonfires whose warmth reaches the whole hillside, walk toward them, and then shrink the fires in stages. Each stage only has to guide you from "roughly right" to "a bit more exactly right".</p>
 <h4>The score</h4><p>For a density \(p(\mathbf x)\) the score is the gradient of its logarithm:</p>$$\mathbf s(\mathbf x)=\nabla_{\mathbf x}\log p(\mathbf x).$$<p>It points toward higher density, and it is longer where the density changes faster. Writing a model as \(p(\mathbf x)=e^{-E(\mathbf x)}/Z\) shows why the score is convenient: the awkward constant \(Z\), which makes probabilities sum to one, disappears when you take the gradient of the logarithm. A network can output a score without anyone ever computing \(Z\).</p>
 <h4>Learning it by denoising</h4><p>We do not know \(p\), so we cannot compute its score to train against. Vincent (2011) showed a way around this. Blur the data with Gaussian noise of size \(\sigma\): \(\tilde{\mathbf x}=\mathbf x+\sigma\boldsymbol\varepsilon\). The score of the blurred density \(p_\sigma\) is the minimiser of</p>$$\mathcal L(\sigma)=\mathbb E_{\mathbf x,\boldsymbol\varepsilon}\Big\lVert\,\mathbf s_\theta(\tilde{\mathbf x},\sigma)+\frac{\boldsymbol\varepsilon}{\sigma}\Big\rVert^2 .$$<p>In words: the arrow at a noisy point should point back along the noise that was added, \(-\boldsymbol\varepsilon/\sigma=(\mathbf x-\tilde{\mathbf x})/\sigma^2\). Many clean points could have produced the same noisy point, and the average of all their "way back" arrows is exactly \(\nabla\log p_\sigma\). This page trains the network to output \(\hat{\boldsymbol\varepsilon}\) and uses \(\mathbf s_\theta=-\hat{\boldsymbol\varepsilon}/\sigma\), which is the same loss multiplied by \(\sigma^2\) so that every level counts equally.</p>
 <h4>Checking against the exact answer</h4><p>For the 500 data points on this page, the blurred density is a mixture of 500 small Gaussians, one on each point, and its score has a closed form:</p>$$\nabla\log p_\sigma(\mathbf x)=\frac{1}{\sigma^2}\sum_i w_i(\mathbf x)\,(\mathbf x_i-\mathbf x),\qquad w_i(\mathbf x)\propto\exp\!\Big(-\frac{\lVert\mathbf x-\mathbf x_i\rVert^2}{2\sigma^2}\Big).$$<p>That is the top-right picture. The network never sees this formula. The bar chart counts how many of its arrows fall within 25° of it.</p>
 <h4>Generating: Langevin dynamics</h4><p>Given a score, repeat</p>$$\mathbf x\leftarrow\mathbf x+\frac{\alpha}{2}\,\mathbf s_\theta(\mathbf x,\sigma)+\sqrt{\alpha}\,\mathbf z,\qquad\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I).$$<p>Climb the arrow a little, then jitter. Without the jitter every point would end on a peak of the density. With it, and with a small step \(\alpha\), the points settle into samples spread according to \(p_\sigma\) itself.</p>
 <h4>Why one noise level is a poor bet</h4><p>With a small \(\sigma\) the blurred density is almost the real one, but training only ever showed the network points close to the data. Far away, where random starting points land, the arrows were never practised and cannot be trusted. Even the exact score has a second problem there: between two separate groups it is nearly blind to how many points each group holds, so the groups receive the wrong shares. With a large \(\sigma\) the arrows are reliable everywhere, but they lead to a blurred version of the data. On the two-dimensional shapes of this page a middle level happens to work about as well as the whole ladder, as you can check. For images, the distance from a random start to the data is vastly larger than the detail that has to be resolved, and no single level covers both.</p>
 <h4>Annealing</h4><p>Song and Ermon (2019) train one network on a ladder of levels \(\sigma_1>\sigma_2>\dots>\sigma_L\) and sample by running Langevin dynamics at each level in turn, with step \(\alpha_i\propto\sigma_i^2\), using the result of one level as the start of the next. The large levels move the points across empty space. The small levels sharpen them. This page uses ten levels from 1.2 down to 0.05, each 0.7 times the one before, with twelve Langevin steps per level.</p>
 <h4>The link to diffusion</h4><p>A diffusion model's noise prediction is this same quantity under another name: \(\nabla\log p_t(\mathbf x)=-\boldsymbol\varepsilon_\theta(\mathbf x,t)/\sqrt{1-\bar\alpha_t}\). Its many noise levels are the ladder, and its sampling loop is annealed Langevin dynamics with a particular choice of step sizes. Song and others (2021) joined the two views into one framework in which the noise level changes continuously.</p>`,
    q: 'The loss compares the network’s arrow with the direction back to one particular clean point. But many clean points could have produced the same noisy point. Why does training on these conflicting targets give the right answer?',
    a: 'Because the loss is a squared error, and the single arrow that minimises a squared error against many targets is their average. At a given noisy point the targets are the “way back” arrows to every clean point that could have produced it, weighted by how likely each one is. That weighted average is exactly the gradient of the log of the blurred density, which is the score. The individual targets conflict, but their mean is what we wanted all along. It is the same reason the error never falls to zero.',
    init(root) {
      const c = $('#sm-c', root),
        cr = $('#sm-r', root),
        live = $('#sm-live', root),
        S = { ds: 'blobs', lv: 4, mode: 'ann' },
        L = 10,
        SG = Array.from({ length: L }, (_, i) => 1.2 * Math.pow(0.05 / 1.2, i / (L - 1))),
        MAXS = 2500,
        M = 400,
        KS = 12,
        CC = 0.2,
        RN = randnFrom(Math.random);
      let net,
        ts = 0,
        LL = [],
        sm = [],
        phase = 'train',
        data = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        xs = null,
        si = 0,
        done = false,
        AG = [],
        nearV = NaN,
        curL = 0,
        exC = null,
        exK = '';
      const eps = (P, i) => g2Pred(net, P, g2Emb(i / (L - 1)));
      const exact = (P, sg) => {
        const n = P.length / 2,
          o = new Float64Array(2 * n),
          iv = 1 / (2 * sg * sg),
          d = new Float64Array(data.length);
        for (let i = 0; i < n; i++) {
          const x = P[2 * i],
            y = P[2 * i + 1];
          let mn = 1e99;
          for (let j = 0; j < data.length; j++) {
            d[j] = ((data[j][0] - x) ** 2 + (data[j][1] - y) ** 2) * iv;
            if (d[j] < mn) mn = d[j];
          }
          let W = 0,
            sx = 0,
            sy = 0;
          for (let j = 0; j < data.length; j++) {
            const w = Math.exp(mn - d[j]);
            W += w;
            sx += w * (data[j][0] - x);
            sy += w * (data[j][1] - y);
          }
          o[2 * i] = sx / W / (sg * sg);
          o[2 * i + 1] = sy / W / (sg * sg);
        }
        return o;
      };
      const exFor = (i) => {
        const k = run + '/' + i;
        if (exK !== k) {
          exK = k;
          exC = exact(G2G, SG[i]);
        }
        return exC;
      };
      const agree = (i) => {
        const e = eps(G2G, i),
          ex = exact(G2G, SG[i]),
          n = G2G.length / 2;
        let ok = 0;
        for (let k = 0; k < n; k++) {
          const ax = -e[2 * k],
            ay = -e[2 * k + 1],
            bx = ex[2 * k],
            by = ex[2 * k + 1];
          if ((ax * bx + ay * by) / (Math.hypot(ax, ay) * Math.hypot(bx, by) || 1) > 0.906) ok++;
        }
        return ok / n;
      };
      const build = () => {
        net = AD.mlp([10, 64, 64, 2], randnFrom(rng(run * 7 + 4)), 'relu');
        ts = 0;
        LL = [];
        sm = [];
        phase = 'train';
        xs = null;
        done = false;
        cu = 0;
        AG = [];
        nearV = NaN;
        si = 0;
        curL = 0;
        data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
        msg =
          'A network with random weights: its arrows point nowhere useful yet. The top-right picture shows what they should become. Press Run to train it.';
      };
      const train = (k) => {
        const B = 128;
        for (let q = 0; q < k && ts < MAXS; q++) {
          const X = new Float64Array(B * 10),
            E = new Float64Array(B * 2);
          for (let i = 0; i < B; i++) {
            const p = g2Samp(S.ds, RN),
              lv = Math.floor(Math.random() * L),
              e0 = RN(),
              e1 = RN(),
              e = g2Emb(lv / (L - 1));
            X[i * 10] = p[0] + SG[lv] * e0;
            X[i * 10 + 1] = p[1] + SG[lv] * e1;
            for (let j = 0; j < 8; j++) X[i * 10 + 2 + j] = e[j];
            E[2 * i] = e0;
            E[2 * i + 1] = e1;
          }
          const l = AD.mse(net.fwd(AD.input(X, [B, 10])), E);
          LL.push(l.v[0]);
          AD.backward(l);
          AD.adam(net.ps, 0.003 * Math.pow(0.1, ts / MAXS), ++ts, 1);
          if (ts % 20 === 0) {
            let s = 0;
            for (let j = ts - 20; j < ts; j++) s += LL[j];
            sm.push(s / 20);
          }
        }
        cu = Math.floor(ts / 100) % 2;
        msg =
          'Training step ' +
          ts +
          ' of ' +
          MAXS +
          ': 128 real points were given noise at random levels and the network guessed the way back for each. Average error over the last 20 steps: ' +
          f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
          '.';
        if (ts >= MAXS) {
          AG = SG.map((_, i) => agree(i));
          phase = 'gen';
          startGen();
          cu = 3;
          msg =
            'Training is finished. The bars at the bottom right compare the learned arrows with the exact ones at each level. Now generate: ' +
            M +
            ' points have been scattered at random.';
        }
      };
      const startGen = () => {
        xs = g2Uni(M);
        si = 0;
        done = false;
        nearV = NaN;
        curL = S.mode === 'ann' ? 0 : S.lv;
      };
      const gstep = () => {
        if (done || !xs) return;
        const i = S.mode === 'ann' ? Math.min(L - 1, Math.floor(si / KS)) : S.lv,
          sg = SG[i],
          e = eps(xs, i),
          a = (CC * sg) / 2,
          b = sg * Math.sqrt(CC);
        for (let j = 0; j < 2 * M; j++)
          xs[j] = clamp(xs[j] - a * e[j] + b * RN(), j % 2 ? -1.68 : -2.25, j % 2 ? 1.68 : 2.25);
        si++;
        curL = i;
        cu = S.mode === 'ann' && si % KS === 0 ? 5 : 4;
        msg =
          'Langevin step ' +
          si +
          ' of ' +
          L * KS +
          ', using the arrows for noise level ' +
          (i + 1) +
          ' (σ = ' +
          f2(sg) +
          '): every point moved a short way along its arrow and was jittered.' +
          (S.mode === 'ann' && si % KS === 0 && si < L * KS
            ? ' Twelve steps are done at this level, so the next step moves to level ' +
              (i + 2) +
              '.'
            : '');
        if (si >= L * KS) {
          const e2 = eps(xs, i);
          for (let j = 0; j < 2 * M; j++) xs[j] -= sg * e2[j];
          done = true;
          nearV = g2Near(xs, M, data);
          let wall = 0;
          for (let j = 0; j < M; j++)
            if (Math.abs(xs[2 * j]) > 2.2 || Math.abs(xs[2 * j + 1]) > 1.63) wall++;
          msg =
            'Finished after ' +
            L * KS +
            ' Langevin steps' +
            (S.mode === 'ann' ? ' through all ten levels' : ' at level ' + (i + 1) + ' only') +
            ', with the last of the jitter removed at the end: ' +
            pct(nearV) +
            ' of the points lie on the shape' +
            (wall ? ', and ' + wall + ' are stuck at the edge of the picture' : '') +
            '.' +
            (S.mode === 'one' ? ' Compare with All Ten Levels.' : '');
        }
      };
      const step = () => {
        if (phase === 'train') train(50);
        else if (done) {
          startGen();
          cu = 3;
          msg = 'A fresh scatter of random points. Press Run or One step.';
        } else gstep();
      };
      const draw = () => {
        const t = T(),
          lv = phase === 'gen' && xs && si > 0 && !done && S.mode === 'ann' ? curL : S.lv;
        {
          const V = uView(c),
            { g, h } = V;
          g2Data(g, V, t, data);
          if (ts > 0) g2Arrows(g, V, eps(G2G, lv), t.accent, -1, 0.6, 7);
          if (xs) g2Pts(g, V, t, xs, M, t.gold);
          g2Note(
            g,
            t,
            h,
            (ts > 0 ? 'learned' : 'untrained') +
              ' arrows for noise level ' +
              (lv + 1) +
              ' of ' +
              L +
              ', σ = ' +
              f2(SG[lv]),
          );
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.56,
            P = g2Panel(
              g,
              t,
              0,
              0,
              w,
              h1,
              'the exact score at noise level ' + (lv + 1) + ', computed from the 500 data points',
            );
          g2Data(g, P, t, data, 0.25, 1.5);
          g2Arrows(g, P, exFor(lv), t.ok, 1, 0.7, 7 * SG[lv], 9);
          g.restore();
          if (phase === 'train')
            uLine(g, t, 0, h1, w, h - h1, sm, {
              title: 'error at guessing the noise (average of every 20 steps)',
              n: MAXS / 20,
              col: t.ok,
              ym: Math.max(1.05, sm[0] || 1.05),
            });
          else {
            const y0 = h1,
              H = h - h1,
              m = { l: 44, r: 12, t: 26, b: 22 },
              bw = (w - m.l - m.r) / L;
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText('learned arrows within 25° of the exact ones, by noise level', 8, y0 + 13);
            g.strokeStyle = t.muted;
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(m.l, y0 + H - m.b);
            g.lineTo(w - m.r, y0 + H - m.b);
            g.stroke();
            g.textAlign = 'right';
            g.fillText('100%', m.l - 5, y0 + m.t + 4);
            g.fillText('0%', m.l - 5, y0 + H - m.b + 4);
            AG.forEach((v, i) => {
              const bh = v * (H - m.t - m.b);
              g.fillStyle = i === lv ? t.gold : mix(t.accent, t.accent, 0, 0.75);
              g.fillRect(m.l + i * bw + 3, y0 + H - m.b - bh, bw - 6, bh);
              g.fillStyle = i === lv ? t.cut : t.muted;
              g.textAlign = 'center';
              g.fillText(String(i + 1), m.l + i * bw + bw / 2, y0 + H - 6);
              if (bh > 16) {
                g.fillStyle = t.bg;
                g.fillText((100 * v).toFixed(0), m.l + i * bw + bw / 2, y0 + H - m.b - bh + 13);
              }
            });
          }
        }
        live.textContent = msg;
        uCur(root, 'sm', cu);
        setR(
          root,
          'ph',
          phase === 'train' ? (ts ? 'Training' : 'Not started') : done ? 'Finished' : 'Generating',
        );
        setR(root, 'ts', ts + ' of ' + MAXS);
        setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
        setR(root, 'sg', lv + 1 + ' of ' + L + ', σ = ' + f2(SG[lv]));
        setR(root, 'ag', AG.length ? pct(AG[lv]) : '–');
        setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
      };
      build();
      if (RM) {
        train(MAXS);
        while (!done) gstep();
      }
      const anim = Anim(() => {
        fr++;
        if (phase === 'train') {
          train(Math.min(ts < 60 ? 1 : 5, chk.remaining()));
          draw();
          chk();
          return;
        }
        if (fr % 3 === 0 && !done) gstep();
        draw();
        if (done) pb.stop();
      });
      pb = playBtn($('#sm-play', root), anim);
      chk = runCtl(pb, root, 'sm', {
        done: () => phase === 'gen' && done,
        restart: () => startGen(),
        count: () => ts,
        what: 'training steps',
      });
      const regen = () => {
        if (phase !== 'gen') {
          msg = 'Points can be generated once training is finished. Press Run.';
          draw();
          return;
        }
        pb.stop();
        startGen();
        draw();
        pb.start();
      };
      $('#sm-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#sm-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#sm-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        pb.stop();
        run++;
        build();
        draw();
        pb.start();
      };
      $('#sm-gen', root).onclick = regen;
      segBind(root, 'sm-mode', S.mode, (v) => {
        S.mode = v;
        if (phase === 'gen') regen();
      });
      bind(
        root,
        'sm-lv',
        (v) => {
          S.lv = +v - 1;
          if (!pb) return;
          if (phase === 'gen' && S.mode === 'one') regen();
          else draw();
        },
        (v) => String(v),
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
  },

  {
    id: 'normalizing-flows',
    tab: 'Normalizing flows',
    title: 'Normalizing flows: bend a bell curve into the data',
    lede: 'Start with the simplest cloud there is, a round Gaussian blob, printed on a rubber sheet. Stretch and turn the sheet, a little at a time, until the blob has the shape of the data. If every stretch can be undone, two things come for free: new samples, by pushing fresh blob points through the stretches, and the exact probability of any point, by pulling it back to the blob and keeping count of how much the sheet was stretched there.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Faint dots: the real data. Gold dots: 350 points from a round Gaussian after the stretches. Thin lines: a square grid drawn on the Gaussian, carried along by the same stretches</figcaption><canvas class="short" id="nf-c" role="img" aria-label="Data, samples pushed through the flow and the warped grid"></canvas></figure>
 <figure class="fig"><figcaption>Top: the exact probability density the model assigns to every place (darker is more likely). Bottom: the training loss</figcaption><canvas class="short" id="nf-r" role="img" aria-label="Exact model density and the training loss"></canvas></figure></div>
 ${uHow('nf', ['Take a batch of real points and pull them backwards through the six layers, toward the Gaussian.', 'Score the result: the points should land where a round Gaussian is dense, and the layers get credit wherever they had to squeeze space to bring them there. This score is the exact log-probability of the data.', 'Backpropagate to raise that log-probability. Repeat for 1,500 batches.', 'Generate: draw points from a round Gaussian.', 'Push them forwards through the layers, one at a time. Each layer slides and stretches points along one axis by an amount that depends on the other axis, then turns the picture by 60°.', 'After the last layer the cloud has the shape of the data.'])}
 <div class="controls">${uBtns('nf', 'Run', 'One step', 'training steps')}
 ${grp('Layer by layer', slider('nf-ly', 'Show the cloud after this many layers', 0, 6, 1, 6) + `<div class="row"><button class="btn" id="nf-gen" type="button">Play the Layers Again</button></div>`)}
 ${grp('The data', `<label class="ctl" for="nf-ds"><span>Shape to learn</span><select id="nf-ds">${g2Opts('moons')}</select></label>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ts', 'Training steps')}${rd('L', 'Loss: minus the average log-probability of the data', 'hl2')}${rd('base', 'The same loss for a plain round Gaussian, with no layers')}${rd('ly', 'Layers applied in the left picture')}${rd('near', 'Generated points that land on the shape', 'hl')}</dl>`,
    tryit:
      'Watch the left picture during training: the round blob is pulled into the shape while the grid shows where the sheet is being stretched and squeezed. Where are the grid cells smallest, and what does the density picture show there? After training, drag the layer slider from 0 to 6 to see what each layer adds. Then type coordinates for your own point: one on the shape, one in the gap between the two moons. How do the two densities compare? Finally switch to A Ring. The model must cover the ring with a sheet that has no hole in it. Look closely at the density: where did it hide the seam?',
    math: String.raw`<h4>Intuition</h4><p>Draw a round target on a sheet of rubber, densest at the bullseye. Now stretch the sheet. Where you stretch it, the ink thins out, and the density falls. Where you bunch it up, the ink concentrates, and the density rises. The total amount of ink never changes. A normalizing flow is a recipe for stretching, learned so that the ink ends up where the data is. Because you know exactly how much each spot was stretched, you know exactly how dense the ink is there. No other model in this module can tell you the exact probability of a point. This one can.</p>
 <h4>Change of variables</h4><p>Let \(f\) be an invertible map that carries a data point \(\mathbf x\) back to a point \(\mathbf z=f(\mathbf x)\) with a standard Gaussian density \(p_Z\). Then the density of the data is</p>$$\log p_X(\mathbf x)=\log p_Z\big(f(\mathbf x)\big)+\log\left|\det\frac{\partial f}{\partial\mathbf x}\right| .$$<p>The first term asks whether \(\mathbf x\) is pulled back to a likely place under the bell. The second is the bookkeeping for stretched rubber: the determinant of the Jacobian says how much \(f\) expands or shrinks a small area around \(\mathbf x\). Training maximises the average of this quantity over the data, which is maximum likelihood with nothing approximated.</p>
 <h4>The difficulty, and the trick</h4><p>For a general network, the inverse does not exist and the determinant costs far too much to compute. Coupling layers (Dinh and others, 2015 and 2017) are designed so that both are easy. Split the coordinates into two parts. Leave the first as it is, and move the second by an amount that depends only on the first:</p>$$y_1=x_1,\qquad y_2=x_2\,e^{\,s(x_1)}+t(x_1).$$<p>Here \(s\) and \(t\) are the two outputs of an ordinary small network. They can be as complicated as you like, and they are never inverted. To undo the layer, note that \(x_1=y_1\) is still available, so \(s\) and \(t\) can be recomputed, and then \(x_2=(y_2-t(y_1))\,e^{-s(y_1)}\). The Jacobian is triangular, so its determinant is the product of its diagonal:</p>$$\log\left|\det\frac{\partial\mathbf y}{\partial\mathbf x}\right|=s(x_1).$$
 <h4>Stacking layers</h4><p>One coupling layer leaves \(x_1\) untouched, so between layers the axes are changed. With many dimensions the two halves swap roles. In two dimensions this page turns the plane by 60° before each layer, which costs nothing because a turn does not change areas. For a chain \(f=f_K\circ\dots\circ f_1\) the log-determinants add:</p>$$\log p_X(\mathbf x)=\log p_Z(\mathbf z)+\sum_{k=1}^{K}s_k .$$<p>The loss shown is minus the average of this over a batch.</p>
 <h4>Sampling and density, both exact</h4><p>To sample, draw \(\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)\) and apply the inverses in reverse order: this is the animation on the left. To evaluate the density at a point, apply the layers forwards and add up the \(s_k\): this is the top-right picture, and the readout for your own point. A density that can be evaluated exactly is useful for spotting unusual inputs, for compression, and for comparing models fairly.</p>
 <h4>What it cannot do</h4><p>A smooth invertible map cannot tear the sheet or glue parts of it together. A blob has no hole, so its image has no hole. To cover a ring, or two separate groups, the model must stretch a thin film of density across the gap. You can see that film in the density picture, and some generated points land on it. More layers make the film thinner but never remove it. The second limit is cost: every layer must keep the same number of dimensions as the data, with no bottleneck.</p>
 <h4>Where flows went next</h4><p>Glow applied coupling layers to images. Autoregressive flows make each coordinate depend on all the earlier ones. Continuous normalizing flows replace the stack of layers with an ODE, \(d\mathbf x/dt=\mathbf v_\theta(\mathbf x,t)\), and flow matching, which has its own tab, is the modern way of training those without ever computing a determinant.</p>
 <h4>What this page trains</h4><p>Six coupling layers. Each has its own network with two hidden layers of 32 neurons, whose scale output is kept between −1.2 and 1.2. Training uses Adam on batches of 96 for 1,500 steps.</p>`,
    q: 'The model gives the exact density at any point. A student proposes using it as a spam filter for data: flag any input whose density is low. In the ring data set, which points would this wrongly accept?',
    a: 'Points on the thin film that the model had to stretch across the hole or the gap. A flow cannot tear the Gaussian apart, so some density always connects regions that the data keeps separate. A point on that bridge receives a moderate density even though no real data ever appears there. The exact density is exact for the model, not for the world: it is only as good as the fit.',
    init(root) {
      const c = $('#nf-c', root),
        cr = $('#nf-r', root),
        live = $('#nf-live', root),
        S = { ds: 'moons' },
        K = 6,
        MAXS = 1500,
        M = 350,
        GW = 40,
        GH = 30,
        RN = randnFrom(Math.random);
      let F,
        ts = 0,
        LL = [],
        sm = [],
        phase = 'train',
        data = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        Z,
        GZ,
        gn = 0,
        st = null,
        gs = null,
        gl = 2 * K,
        done = false,
        dens = null,
        dmax = 1,
        base = NaN,
        lpD = null,
        nearV = NaN,
        lastD = 0;
      const mkGrid = () => {
        const P = [];
        for (let a = -2; a <= 2.001; a += 0.5)
          for (let b = 0; b <= 48; b++) P.push(a, -2 + (4 * b) / 48);
        for (let a = -2; a <= 2.001; a += 0.5)
          for (let b = 0; b <= 48; b++) P.push(-2 + (4 * b) / 48, a);
        gn = P.length / 2;
        return Float64Array.from(P);
      };
      const refresh = (full) => {
        st = nfInv(F, Z, M);
        gs = nfInv(F, GZ, gn);
        if (full) {
          lastD = ts;
          const G = new Float64Array(2 * GW * GH);
          for (let r = 0; r < GH; r++)
            for (let q = 0; q < GW; q++) {
              G[2 * (r * GW + q)] = (-4 + (8 * (q + 0.5)) / GW) / G2S;
              G[2 * (r * GW + q) + 1] = (3 - (6 * (r + 0.5)) / GH) / G2S;
            }
          const lp = nfLogp(F, G, GW * GH);
          dens = lp.map(Math.exp);
          dmax = Math.max(1e-9, ...dens);
          const D = new Float64Array(2 * data.length);
          data.forEach((p, i) => {
            D[2 * i] = p[0];
            D[2 * i + 1] = p[1];
          });
          lpD = nfLogp(F, D, data.length);
        }
      };
      const build = () => {
        F = nfMake(run * 5 + 3, K, 32);
        ts = 0;
        LL = [];
        sm = [];
        phase = 'train';
        done = false;
        cu = 0;
        gl = 2 * K;
        nearV = NaN;
        data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
        Z = Float64Array.from({ length: 2 * M }, () => RN());
        GZ = mkGrid();
        let b = 0;
        for (const p of data) b += (0.5 * (p[0] * p[0] + p[1] * p[1])) / data.length;
        base = b + Math.log(2 * Math.PI);
        refresh(true);
        msg =
          'Six layers that do almost nothing yet: the gold cloud is still a round Gaussian and the grid is still square. Press Run to train them.';
      };
      const train = (k) => {
        const B = 96;
        for (let q = 0; q < k && ts < MAXS; q++) {
          const X = new Float64Array(2 * B);
          for (let i = 0; i < B; i++) {
            const p = g2Samp(S.ds, RN);
            X[2 * i] = p[0];
            X[2 * i + 1] = p[1];
          }
          const { z, As } = nfFwd(F, X, B),
            l = AD.nfl(z, As);
          LL.push(l.v[0]);
          AD.backward(l);
          AD.adam(F.ps, ts < 1100 ? 0.004 : 0.0015, ++ts, 1);
          if (ts % 20 === 0) {
            let s = 0;
            for (let j = ts - 20; j < ts; j++) s += LL[j];
            sm.push(s / 20);
          }
        }
        refresh(ts - lastD >= 30 || ts >= MAXS);
        cu = Math.floor(ts / 60) % 3;
        msg =
          'Training step ' +
          ts +
          ' of ' +
          MAXS +
          ': 96 real points were pulled back through the layers toward the Gaussian, their exact log-probability was computed, and the layers were adjusted to raise it. Loss over the last 20 steps: ' +
          f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
          ' (a plain round Gaussian scores ' +
          f3(base) +
          ').';
        if (ts >= MAXS) {
          phase = 'gen';
          gl = 0;
          done = false;
          cu = 3;
          nearV = g2Near(st[2 * K], M, data);
          msg =
            'Training is finished. The left picture is back at the start: 350 points from a round Gaussian, with no layers applied. Press Run to send them through the six layers.';
        }
      };
      const gstep = () => {
        if (done) return;
        gl = Math.min(2 * K, Math.floor(gl + 1e-6) + 1);
        const ly = Math.ceil(gl / 2);
        cu = 4;
        msg =
          gl % 2
            ? 'Layer ' +
              ly +
              ' of ' +
              K +
              ', first half: every point slides and stretches up or down, by an amount that depends on how far left or right it is. Grid cells that shrink here are where the density rises.'
            : 'Layer ' +
              ly +
              ' of ' +
              K +
              ', second half: the whole picture is turned by 60°' +
              (ly < K ? ', so that the next layer stretches along a new direction' : '') +
              '. A turn does not change any area, so it does not change any density.';
        if (gl >= 2 * K) {
          done = true;
          cu = 5;
          msg += ' All six layers are applied: ' + pct(nearV) + ' of the points lie on the shape.';
        }
      };
      const step = () => {
        if (phase === 'train') train(50);
        else if (done) {
          gl = 0;
          done = false;
          cu = 3;
          msg = 'Back to the round Gaussian, with no layers applied. Press Run or One step.';
        } else gstep();
      };
      const at = (sn, i, n, o) => {
        const j = Math.min(2 * K - 1, Math.floor(gl)),
          u = gl >= 2 * K ? 1 : gl - j,
          A = sn[j],
          B = sn[j + 1];
        if (j % 2 === 0) {
          o[0] = A[2 * i] + (B[2 * i] - A[2 * i]) * u;
          o[1] = A[2 * i + 1] + (B[2 * i + 1] - A[2 * i + 1]) * u;
        } else {
          const th = (-Math.PI / 3) * u,
            cs = Math.cos(th),
            sn2 = Math.sin(th),
            x = A[2 * i],
            y = A[2 * i + 1];
          o[0] = x * cs - y * sn2;
          o[1] = x * sn2 + y * cs;
        }
        return o;
      };
      const tp = testPoint(
        root,
        'nf',
        ['Your x', 'Your y'],
        [0, 0],
        [
          [-2.2, 2.2],
          [-1.6, 1.6],
        ],
      );
      const draw = () => {
        const t = T(),
          o = [0, 0];
        {
          const V = uView(c),
            { g, h } = V;
          g2Data(g, V, t, data);
          if (gs) {
            g.strokeStyle = mix(t.accent, t.accent, 0, 0.38);
            g.lineWidth = 1;
            g.lineJoin = 'round';
            for (let ln = 0; ln < gn / 49; ln++) {
              g.beginPath();
              for (let b = 0; b < 49; b++) {
                at(gs, ln * 49 + b, gn, o);
                const x = V.X(o[0] * G2S),
                  y = V.Y(o[1] * G2S);
                b ? g.lineTo(x, y) : g.moveTo(x, y);
              }
              g.stroke();
            }
          }
          if (st) {
            const P = new Float64Array(2 * M);
            for (let i = 0; i < M; i++) {
              at(st, i, M, o);
              P[2 * i] = o[0];
              P[2 * i + 1] = o[1];
            }
            g2Pts(g, V, t, P, M, t.gold);
          }
          markTP(g, V.X(tp.v[0] * G2S), V.Y(tp.v[1] * G2S), t);
          g2Note(
            g,
            t,
            h,
            phase === 'train'
              ? 'all ' + K + ' layers applied, as trained so far'
              : f2(gl / 2) + ' of ' + K + ' layers applied',
          );
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.56,
            P = g2Panel(
              g,
              t,
              0,
              0,
              w,
              h1,
              'exact density of the model: darker means more probable',
              1.2,
            );
          if (dens) {
            for (let r = 0; r < GH; r++)
              for (let q = 0; q < GW; q++) {
                const v = dens[r * GW + q] / dmax;
                if (v < 0.012) continue;
                g.fillStyle = mix(t.accent, t.accent, 0, Math.min(0.95, Math.pow(v, 0.6)));
                const xa = Math.round(P.X(-4 + (8 * q) / GW)),
                  xb = Math.round(P.X(-4 + (8 * (q + 1)) / GW)),
                  ya = Math.round(P.Y(3 - (6 * r) / GH)),
                  yb = Math.round(P.Y(3 - (6 * (r + 1)) / GH));
                g.fillRect(xa, ya, xb - xa, yb - ya);
              }
          }
          g2Data(g, P, t, data, 0.3, 1.2);
          markTP(g, P.X(tp.v[0] * G2S), P.Y(tp.v[1] * G2S), t, false);
          g.restore();
          const Lp = uLine(g, t, 0, h1, w, h - h1, sm, {
            title: 'minus the average log-probability of the data (lower is better)',
            n: MAXS / 20,
            col: t.ok,
            ym: Math.max(base * 1.08, sm[0] || 0),
            y1: Math.min(0, ...sm),
          });
          g.setLineDash([5, 4]);
          g.strokeStyle = t.gold;
          g.lineWidth = 1.4;
          g.beginPath();
          g.moveTo(Lp.m.l, Lp.Y(base));
          g.lineTo(w - Lp.m.r, Lp.Y(base));
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.cut;
          g.textAlign = 'right';
          g.fillText('a plain round Gaussian', w - Lp.m.r, Lp.Y(base) + 14);
        }
        live.textContent = msg;
        uCur(root, 'nf', cu);
        {
          const lp = nfLogp(F, Float64Array.of(tp.v[0], tp.v[1]), 1)[0];
          let lo = 0;
          if (lpD) for (const v of lpD) if (v < lp) lo++;
          tp.out.textContent = ts
            ? 'The model’s probability density at your point is ' +
              (Math.exp(lp) < 0.001 ? Math.exp(lp).toExponential(1) : f3(Math.exp(lp))) +
              ' (log-density ' +
              f2(lp) +
              '). That is higher than the density at ' +
              (lpD ? ((100 * lo) / lpD.length).toFixed(0) : '–') +
              '% of the real data points.'
            : 'Train the model first: press Run.';
        }
        setR(
          root,
          'ph',
          phase === 'train'
            ? ts
              ? 'Training'
              : 'Not started'
            : done
              ? 'Finished'
              : 'Applying the layers',
        );
        setR(root, 'ts', ts + ' of ' + MAXS);
        setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
        setR(root, 'base', f3(base));
        setR(root, 'ly', (phase === 'train' ? K : f2(gl / 2)) + ' of ' + K);
        setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
      };
      tp.on = () => draw();
      build();
      if (RM) {
        train(MAXS);
        gl = 2 * K;
        done = true;
      }
      const anim = Anim(() => {
        fr++;
        if (phase === 'train') {
          train(Math.min(ts < 40 ? 1 : 4, chk.remaining()));
          draw();
          chk();
          return;
        }
        if (!done) {
          const n = gl + 1 / 45;
          if (Math.floor(n + 1e-9) > Math.floor(gl + 1e-9) || n >= 2 * K) {
            gstep();
          } else gl = n;
        }
        draw();
        if (done) pb.stop();
      });
      pb = playBtn($('#nf-play', root), anim);
      chk = runCtl(pb, root, 'nf', {
        done: () => phase === 'gen' && done,
        restart: () => {
          gl = 0;
          done = false;
        },
        count: () => ts,
        what: 'training steps',
      });
      $('#nf-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#nf-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#nf-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        pb.stop();
        run++;
        build();
        draw();
        pb.start();
      };
      $('#nf-gen', root).onclick = () => {
        if (phase !== 'gen') {
          msg = 'The layers can be played one by one once training is finished. Press Run.';
          draw();
          return;
        }
        pb.stop();
        gl = 0;
        done = false;
        draw();
        pb.start();
      };
      bind(
        root,
        'nf-ly',
        (v) => {
          if (!pb) return;
          if (phase !== 'gen') {
            draw();
            return;
          }
          pb.stop();
          gl = 2 * +v;
          done = gl >= 2 * K;
          cu = 4;
          msg = 'Showing the cloud after ' + v + ' of ' + K + ' layers.';
          draw();
        },
        (v) => String(v),
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
  },
  {
    id: 'energy-based',
    tab: 'Energy-based models',
    title: 'Energy-based models: carve valleys where the data lives',
    lede: 'Give every possible point a single number, its energy, and agree that low energy means “this looks like data”. The model is then a landscape: valleys under the data, high ground everywhere else. Nothing forces the numbers to add up to one, which makes the model very free and also makes it awkward to train. Watch the landscape being carved, then drop marbles on it and let them roll.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Shading: the energy landscape, darker for lower energy (the higher half of the picture is left blank). Faint dots: the real data. Small dots: the model’s own points during training. Gold dots: points being generated</figcaption><canvas class="short" id="eb-c" role="img" aria-label="Energy landscape with data, the model's points and generated points"></canvas></figure>
 <figure class="fig"><figcaption>Top: the height of the landscape along the dashed line. Bottom: energy at the model’s own points minus energy at the real data. It is below zero while the model’s points keep finding valleys that hold no data</figcaption><canvas class="short" id="eb-r" role="img" aria-label="A slice through the energy landscape and the energy gap during training"></canvas></figure></div>
 ${uHow('eb', ['Keep 96 points that belong to the model: its current idea of what data looks like. They start scattered at random.', 'Let those points roll downhill on the energy landscape for five small steps, with a little jitter.', 'Take 96 real points. Lower the landscape under the real points and raise it under the model’s points.', 'Repeat. Where the model’s points already sit on the data the two pushes cancel, and the landscape stops changing there.', 'Generate: scatter new points at random and let them roll downhill, with jitter, for 200 steps.', 'Switch the jitter off for a few last steps so that each point settles in its valley.'])}
 <div class="controls">${uBtns('eb', 'Run', 'One step', 'training steps')}
 ${grp('Looking at the landscape', slider('eb-sl', 'Height of the dashed line', -1.5, 1.5, 0.05, 0.25) + `<div class="row"><button class="btn" id="eb-gen" type="button">Generate Again</button></div>`)}
 ${grp('The data', `<label class="ctl" for="eb-ds"><span>Shape to learn</span><select id="eb-ds">${g2Opts('moons')}</select></label>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ts', 'Training steps')}${rd('ed', 'Average energy at real points')}${rd('es', 'Average energy at the model’s own points')}${rd('gs', 'Rolling steps while generating')}${rd('near', 'Generated points that land on the shape', 'hl')}</dl>`,
    tryit:
      'During training, watch the small dots: they are the model’s own guesses, and they chase the data. Every place they visit that holds no data gets pushed up, which is how the high ground is built. Move the dashed line so that it crosses the data and look at the slice: how many valleys does it show? After training, press Generate Again and watch the gold points roll. Do any get caught in a dip that is not on the data? Then type in your own point, once on the shape and once far away, and compare the two energies.',
    math: String.raw`<h4>Intuition</h4><p>Lay a rubber sheet over a table and decide that marbles should collect where the data is. You are allowed two moves. Press the sheet down wherever a real data point sits. Then release a handful of marbles, see where they roll to, and pull the sheet up at those places. If the marbles rolled onto the data, the press and the pull cancel and nothing changes. If they rolled somewhere wrong, that wrong valley is filled in. Repeat until marbles only ever come to rest on data.</p>
 <h4>From energy to probability</h4><p>An energy-based model is any function \(E_\theta(\mathbf x)\) that returns one number. It defines a density through</p>$$p_\theta(\mathbf x)=\frac{e^{-E_\theta(\mathbf x)}}{Z_\theta},\qquad Z_\theta=\int e^{-E_\theta(\mathbf x)}\,d\mathbf x .$$<p>Lower energy means higher probability. The constant \(Z_\theta\) makes the total equal one. It is an integral over every possible input, which for images cannot be computed. Every difficulty with these models comes from \(Z_\theta\), and every training method is a way of avoiding it.</p>
 <h4>The gradient of the likelihood</h4><p>Maximising the log-likelihood of the data gives a gradient with two parts:</p>$$\nabla_\theta\,\mathbb E_{\text{data}}\big[\log p_\theta(\mathbf x)\big]=-\,\mathbb E_{\text{data}}\big[\nabla_\theta E_\theta(\mathbf x)\big]+\mathbb E_{\mathbf x\sim p_\theta}\big[\nabla_\theta E_\theta(\mathbf x)\big].$$<p>The first part lowers the energy at real data. The second, which comes from differentiating \(\log Z_\theta\), raises the energy at samples from the model itself. So the loss that is actually minimised is</p>$$\mathcal L=\mathbb E_{\text{data}}\big[E_\theta(\mathbf x)\big]-\mathbb E_{\text{model}}\big[E_\theta(\mathbf x)\big].$$<p>The bottom-right chart plots the negative of this, the gap between the two averages. When the model's samples cannot be told from data, the two averages are equal, the gap is zero and the gradient vanishes.</p>
 <h4>Where the model's samples come from</h4><p>The second part needs samples from \(p_\theta\). These come from Langevin dynamics, which needs only the slope of the energy and never \(Z_\theta\):</p>$$\mathbf x\leftarrow\mathbf x-\frac{\alpha}{2}\,\nabla_{\mathbf x}E_\theta(\mathbf x)+\sqrt{\alpha}\,\mathbf z .$$<p>Roll downhill, jitter, repeat. The downhill direction \(-\nabla_{\mathbf x}E_\theta\) is the score from the Score Matching tab: an energy-based model is a score model whose arrows are guaranteed to be the slope of one surface. Running the chain until it forgets its start would take far too long, so this page uses <b>persistent contrastive divergence</b> (Tieleman, 2008): the 96 points are kept from one training step to the next and moved only five steps each time. One in twenty is thrown back to a random place so that distant regions keep being checked.</p>
 <h4>Keeping it stable</h4><p>The loss only cares about differences in energy, so nothing stops both averages from drifting off to large values. A small penalty on the squared energies, \(0.1\,(\mathbb E_{\text{data}}[E^2]+\mathbb E_{\text{model}}[E^2])\), holds them near zero. That is why the energies shown stay small. The step size and jitter used here are chosen to look clear and train quickly. They are not matched as the formula above requires, so the points settle a little more tightly than the exact density would.</p>
 <h4>What the freedom buys</h4><p>Any network can be an energy. There is no decoder, no invertibility requirement and no fixed order of variables. Energies combine by addition: the sum of two energies is low only where both are low, which gives a model of "satisfies both". The energy itself is a ready-made measure of how unusual an input is. The price is that both training and sampling need a slow inner loop of Langevin steps, and that the result is sensitive to the settings of that loop.</p>
 <h4>How it connects to the rest of the module</h4><p>The critic of a GAN is close to an energy, with a generator network standing in for the Langevin sampler. A diffusion model learns the slope \(-\nabla E\) directly at many noise levels and never needs the inner loop during training, which is the main reason diffusion models displaced energy-based models for images. Boltzmann machines and Hopfield networks are energy-based models with particular forms for \(E\).</p>
 <h4>What this page trains</h4><p>A network with two inputs, two hidden layers of 48 neurons and one output, trained with Adam for 800 steps. The landscape is drawn by evaluating that network on a grid.</p>`,
    q: 'Training lowers the energy at real data. Why is that alone not enough? What would the landscape look like if the second half of the update, raising the energy at the model’s own points, were left out?',
    a: 'Lowering the energy at data says nothing about anywhere else. The cheapest way to do it is to lower the energy everywhere: a flat landscape, or one that slopes away to minus infinity, gives low energy at the data and equally low energy at every other point, so it has learned nothing. A valley only exists relative to higher ground around it. The second half builds that higher ground, and it does so precisely where the model currently and wrongly believes data might be.',
    init(root) {
      const c = $('#eb-c', root),
        cr = $('#eb-r', root),
        live = $('#eb-live', root),
        S = { ds: 'moons', sl: 0.25 },
        MAXS = 800,
        NP = 96,
        M = 300,
        GW = 64,
        GH = 48,
        A = 0.03,
        SGM = 0.05,
        GENS = 200,
        RN = randnFrom(Math.random);
      let net,
        ts = 0,
        GAP = [],
        gsm = [],
        phase = 'train',
        data = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        P = null,
        xs = null,
        si = 0,
        done = false,
        EG = null,
        ED = null,
        emin = 0,
        emax = 1,
        ecap = 1,
        nearV = NaN,
        edm = NaN,
        esm = NaN,
        lastE = -99;
      const refreshE = () => {
        const G = new Float64Array(2 * GW * GH);
        for (let r = 0; r < GH; r++)
          for (let q = 0; q < GW; q++) {
            G[2 * (r * GW + q)] = (-4 + (8 * (q + 0.5)) / GW) / G2S;
            G[2 * (r * GW + q) + 1] = (3 - (6 * (r + 0.5)) / GH) / G2S;
          }
        EG = Float64Array.from(ebE(net, G, GW * GH));
        emin = Math.min(...EG);
        emax = Math.max(emin + 1e-6, ...EG);
        ecap = Math.max(emin + 1e-6, Float64Array.from(EG).sort()[Math.floor(0.5 * EG.length)]);
        const D = new Float64Array(2 * data.length);
        data.forEach((p, i) => {
          D[2 * i] = p[0];
          D[2 * i + 1] = p[1];
        });
        ED = Float64Array.from(ebE(net, D, data.length));
        lastE = ts;
      };
      const build = () => {
        net = AD.mlp([2, 48, 48, 1], randnFrom(rng(run * 3 + 5)), 'relu');
        ts = 0;
        GAP = [];
        gsm = [];
        phase = 'train';
        xs = null;
        done = false;
        cu = 0;
        nearV = NaN;
        edm = NaN;
        esm = NaN;
        si = 0;
        data = Array.from({ length: 500 }, () => g2Samp(S.ds, RN));
        P = g2Uni(NP);
        refreshE();
        msg =
          'A network with random weights gives a landscape that has nothing to do with the data. The small dots are the model’s own points, scattered at random. Press Run.';
      };
      const train = (k) => {
        for (let q = 0; q < k && ts < MAXS; q++) {
          const X = new Float64Array(2 * NP);
          for (let i = 0; i < NP; i++) {
            const p = g2Samp(S.ds, RN);
            X[2 * i] = p[0] + 0.03 * RN();
            X[2 * i + 1] = p[1] + 0.03 * RN();
          }
          for (let i = 0; i < NP; i++)
            if (Math.random() < 0.05) {
              P[2 * i] = (Math.random() * 2 - 1) * 2.2;
              P[2 * i + 1] = (Math.random() * 2 - 1) * 1.65;
            }
          ebLang(net, P, NP, 5, A, SGM, RN);
          const Ed = net.fwd(AD.input(X, [NP, 2])),
            Es = net.fwd(AD.input(P, [NP, 2])),
            l = AD.ebl(Ed, Es, 0.1);
          let md = 0,
            ms = 0;
          for (let i = 0; i < NP; i++) {
            md += Ed.v[i] / NP;
            ms += Es.v[i] / NP;
          }
          AD.backward(l);
          AD.adam(net.ps, 0.003, ++ts, 1);
          GAP.push(ms - md);
          edm = md;
          esm = ms;
          if (ts % 10 === 0) {
            let s = 0;
            for (let j = ts - 10; j < ts; j++) s += GAP[j];
            gsm.push(s / 10);
          }
        }
        if (ts - lastE >= 8 || ts >= MAXS) refreshE();
        cu = 1 + (Math.floor(ts / 60) % 3);
        msg =
          'Training step ' +
          ts +
          ' of ' +
          MAXS +
          ': the model’s 96 points rolled downhill for five small steps. Then the landscape was lowered under 96 real points and raised under the model’s points. Average energy: ' +
          f3(edm) +
          ' at the real points, ' +
          f3(esm) +
          ' at the model’s.';
        if (ts >= MAXS) {
          phase = 'gen';
          startGen();
          cu = 4;
          msg =
            'Training is finished and the landscape is fixed. ' +
            M +
            ' new points have been scattered at random. Press Run to let them roll.';
        }
      };
      const startGen = () => {
        xs = g2Uni(M);
        si = 0;
        done = false;
        nearV = NaN;
      };
      const gstep = (k) => {
        if (done || !xs) return;
        ebLang(net, xs, M, k, A, SGM, RN);
        si += k;
        cu = 4;
        msg =
          'Rolling step ' +
          si +
          ' of ' +
          GENS +
          ': every point moved a short way downhill and was jittered.';
        if (si >= GENS) {
          ebLang(net, xs, M, 20, A, 0, RN);
          done = true;
          cu = 5;
          nearV = g2Near(xs, M, data);
          msg =
            'Finished: ' +
            GENS +
            ' downhill steps with jitter, then 20 without. ' +
            pct(nearV) +
            ' of the points came to rest on the shape. They all began scattered at random.';
        }
      };
      const step = () => {
        if (phase === 'train') train(20);
        else if (done) {
          startGen();
          cu = 4;
          msg = 'A fresh scatter of random points. Press Run or One step.';
        } else gstep(10);
      };
      const tp = testPoint(
        root,
        'eb',
        ['Your x', 'Your y'],
        [0, 0],
        [
          [-2.2, 2.2],
          [-1.6, 1.6],
        ],
      );
      const draw = () => {
        const t = T();
        {
          const V = uView(c),
            { g, w, h } = V,
            cw = w / GW,
            ch = h / GH;
          if (EG)
            for (let r = 0; r < GH; r++)
              for (let q = 0; q < GW; q++) {
                const u = 1 - clamp((EG[r * GW + q] - emin) / (ecap - emin), 0, 1);
                if (u < 0.03) continue;
                g.fillStyle = mix(t.accent, t.accent, 0, 0.8 * Math.pow(u, 1.2));
                const xa = Math.round(V.X(-4 + (8 * q) / GW)),
                  xb = Math.round(V.X(-4 + (8 * (q + 1)) / GW)),
                  ya = Math.round(V.Y(3 - (6 * r) / GH)),
                  yb = Math.round(V.Y(3 - (6 * (r + 1)) / GH));
                g.fillRect(xa, ya, xb - xa, yb - ya);
              }
          g.setLineDash([6, 5]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(V.X(-4), V.Y(S.sl * G2S));
          g.lineTo(V.X(4), V.Y(S.sl * G2S));
          g.stroke();
          g.setLineDash([]);
          g2Data(g, V, t, data, 0.4);
          if (phase === 'train' && P) g2Pts(g, V, t, P, NP, t.cut, 2.4);
          if (xs) g2Pts(g, V, t, xs, M, t.gold);
          markTP(g, V.X(tp.v[0] * G2S), V.Y(tp.v[1] * G2S), t);
          g2Note(
            g,
            t,
            h,
            phase === 'train'
              ? 'small dots: the model’s own 96 points'
              : 'gold dots: ' + M + ' generated points',
          );
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5,
            NPF = 70,
            F = new Float64Array(2 * NPF);
          for (let i = 0; i < NPF; i++) {
            F[2 * i] = -2.25 + (4.5 * i) / (NPF - 1);
            F[2 * i + 1] = S.sl;
          }
          const pe = Array.from(ebE(net, F, NPF)),
            lo = Math.min(...pe),
            hi = Math.max(lo + 1e-6, ...pe);
          const Lp = uLine(g, t, 0, 0, w, h1, pe, {
            title: 'energy along the dashed line: valleys are where the model expects data',
            col: t.accent,
            y1: lo,
            ym: hi,
          });
          g.fillStyle = t.ink;
          for (const p of data)
            if (Math.abs(p[1] - S.sl) < 0.07) {
              const x = Lp.X(((p[0] + 2.25) / 4.5) * (NPF - 1));
              g.fillRect(x - 0.8, h1 - Lp.m.b - 7, 1.6, 7);
            }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('ticks: real data near the line', w - Lp.m.r, h1 - 5);
          const G2 = uLine(g, t, 0, h1, w, h - h1, gsm, {
            title: 'energy: model’s points minus real points (average of 10 steps)',
            n: MAXS / 10,
            col: t.ok,
            y1: Math.min(-0.05, ...gsm),
            ym: Math.max(0.1, ...gsm),
          });
          g.setLineDash([5, 4]);
          g.strokeStyle = t.gold;
          g.lineWidth = 1.3;
          g.beginPath();
          g.moveTo(G2.m.l, G2.Y(0));
          g.lineTo(w - G2.m.r, G2.Y(0));
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.cut;
          g.textAlign = 'right';
          g.fillText('dashed line: zero, where samples look like data', w - G2.m.r, h - G2.m.b - 7);
        }
        live.textContent = msg;
        uCur(root, 'eb', cu);
        {
          const e = ebE(net, Float64Array.of(tp.v[0], tp.v[1]), 1)[0];
          let lo = 0;
          if (ED) for (const v of ED) if (v < e) lo++;
          tp.out.textContent = ts
            ? 'The energy at your point is ' +
              f3(e) +
              '. ' +
              (ED ? ((100 * lo) / ED.length).toFixed(0) : '–') +
              '% of the real data points have lower energy than that. On this landscape the lowest energy anywhere in view is ' +
              f3(emin) +
              ' and the highest is ' +
              f3(emax) +
              '.'
            : 'Train the model first: press Run.';
        }
        setR(
          root,
          'ph',
          phase === 'train' ? (ts ? 'Training' : 'Not started') : done ? 'Finished' : 'Generating',
        );
        setR(root, 'ts', ts + ' of ' + MAXS);
        setR(root, 'ed', isFinite(edm) ? f3(edm) : '–');
        setR(root, 'es', isFinite(esm) ? f3(esm) : '–');
        setR(root, 'gs', phase === 'gen' ? Math.min(si, GENS) + ' of ' + GENS : '–');
        setR(root, 'near', isFinite(nearV) ? pct(nearV) : '–');
      };
      tp.on = () => draw();
      build();
      if (RM) {
        train(MAXS);
        while (!done) gstep(10);
      }
      const anim = Anim(() => {
        fr++;
        if (phase === 'train') {
          train(Math.min(ts < 30 ? 1 : 2, chk.remaining()));
          draw();
          chk();
          return;
        }
        if (!done) gstep(1);
        draw();
        if (done) pb.stop();
      });
      pb = playBtn($('#eb-play', root), anim);
      chk = runCtl(pb, root, 'eb', {
        done: () => phase === 'gen' && done,
        restart: () => startGen(),
        count: () => ts,
        what: 'training steps',
      });
      $('#eb-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#eb-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#eb-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        pb.stop();
        run++;
        build();
        draw();
        pb.start();
      };
      $('#eb-gen', root).onclick = () => {
        if (phase !== 'gen') {
          msg = 'Points can be generated once training is finished. Press Run.';
          draw();
          return;
        }
        pb.stop();
        startGen();
        draw();
        pb.start();
      };
      bind(
        root,
        'eb-sl',
        (v) => {
          S.sl = +v;
          if (pb) draw();
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
  },

  {
    id: 'discrete-diffusion',
    tab: 'Discrete diffusion',
    title: 'Discrete diffusion: fill in the blanks, a few at a time',
    lede: 'Gaussian noise suits pictures, where a pixel can be a little brighter or darker. A letter cannot be a little more “q”. For words, tokens and other things that come in separate kinds, the noise is different: tokens are hidden behind a blank. The model learns to guess what is under the blanks, and it generates by starting with everything hidden and revealing a few tokens at a time. Unlike a language model that writes left to right, it fills in the whole thing at once, in any order.',
    stage: `<div class="row seg" id="dd-kind" role="group" aria-label="What to generate"><button class="btn" type="button" data-v="pic">Letter Pictures: 35 Black-or-White Squares</button><button class="btn" type="button" data-v="word">Five-Letter Words: 5 Letters out of 26</button></div>
 <div class="stage"><figure class="fig"><figcaption>Eight samples being generated. Tinted cells are still hidden, and the strength of the tint is the network’s current belief. Gold frames mark what was revealed in the last step</figcaption><canvas class="short" id="dd-c" role="img" aria-label="Eight samples being revealed step by step"></canvas></figure>
 <figure class="fig"><figcaption>Top: one training example with more and more of it hidden. Bottom: the training error, then quality for each number of reveal steps</figcaption><canvas class="short" id="dd-r" role="img" aria-label="Forward masking of one example, the loss, and quality against number of steps"></canvas></figure></div>
 ${uHow('dd', ['Take a real example. Choose a random fraction between 0 and 1 and hide that fraction of its tokens behind a blank.', 'Show the network the partly hidden example and ask, for every blank, how likely each token is. Score it only on the blanks and backpropagate. Repeat for 1,500 batches.', 'Measure: generate with different numbers of reveal steps and count how many samples are real.', 'Generate: start with everything hidden.', 'Ask the network for its beliefs about every blank. Reveal a few blanks by drawing a token for each from those beliefs. The rest stay hidden.', 'Ask again: the tokens just revealed change the beliefs about the rest. Repeat until nothing is hidden.'])}
 <div class="controls">${uBtns('dd', 'Run', 'One step', 'training steps')}
 ${grp('Generating', slider('dd-n', 'Number of reveal steps', 0, 7, 1, 5) + `<div class="row seg" id="dd-ord" role="group" aria-label="Which blanks to reveal first"><button class="btn" type="button" data-v="conf">Most Confident First</button><button class="btn" type="button" data-v="rand">Random Order</button></div><div class="row"><button class="btn" id="dd-gen" type="button">Generate Again</button></div>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ts', 'Training steps')}${rd('L', 'Error at guessing hidden tokens', 'hl2')}${rd('rs', 'Reveal steps')}${rd('hid', 'Tokens still hidden in each sample')}${rd('ok', 'Samples that came out real', 'hl')}</dl>`,
    tryit:
      'Let it train, then watch a run with Letter Pictures. Early on the tint is the same in all eight samples: with nothing revealed the network can only report how often each square is black on average. After the first few reveals the samples go their separate ways. Now set the reveal steps to 1 and press Generate Again: every square is drawn at once. What comes out, and why? Raise the steps until the letters are clean. Compare Most Confident First with Random Order at a middle setting. Then switch to Five-Letter Words and try 1 step and 5 steps. With one step, are the letters at least plausible for their positions?',
    math: String.raw`<h4>Intuition</h4><p>Think of a crossword with no clues, or the game where letters of a hidden phrase are turned over one at a time. With every square blank, all you can say is that E is common and Q is rare. Turn over one letter and the neighbours become easier. Turn over a few more and the rest is nearly forced. A discrete diffusion model is trained to play this game from any position: given some squares, state the odds for all the others. To write something new it starts from an empty board, lets its own odds decide a few squares, looks again, and carries on.</p>
 <h4>The forward process: hiding</h4><p>Let \(\mathbf x=(x^1,\dots,x^L)\) be a sequence of tokens. At time \(t\in[0,1]\) each token is independently replaced by a special blank token \(\texttt{[M]}\) with probability \(t\) and otherwise left alone:</p>$$q(x_t^i\mid x^i)=\begin{cases}1-t & x_t^i=x^i\\[2pt] t & x_t^i=\texttt{[M]}\end{cases}$$<p>At \(t=0\) nothing is hidden and at \(t=1\) everything is. A blank never turns back into a token on its own, so this is called an absorbing process. It plays the part that adding Gaussian noise plays for pictures.</p>
 <h4>Training</h4><p>The network sees the partly hidden sequence and outputs, for every position, a distribution over the vocabulary. It is scored with cross-entropy on the hidden positions only:</p>$$\mathcal L=\mathbb E_{\,t,\ \mathbf x,\ \mathbf x_t}\Big[\,w(t)\sum_{i:\ x_t^i=\texttt{[M]}}-\log p_\theta\big(x^i\mid\mathbf x_t\big)\Big].$$<p>With the weight \(w(t)=1/t\) this is a bound on the log-likelihood of the data (Sahoo and others, 2024; Shi and others, 2024). This page uses \(w(t)=1\) and averages over the hidden positions, which trains the same network a little less exactly. If you know BERT, this is its training task, with one change: the fraction hidden is not fixed at 15% but ranges over everything from none to all. That range is what turns a fill-in-the-blank model into a generator.</p>
 <h4>Generating: revealing</h4><p>Start from \(\mathbf x_1=(\texttt{[M]},\dots,\texttt{[M]})\). Choose \(N\) steps. At each step run the network once, pick some of the positions that are still hidden, and for each draw a token from the network's distribution for that position. With a linear schedule, \(L/N\) positions are revealed per step. Which positions go first is a free choice. Random Order is what the forward process implies. Most Confident First, used by MaskGIT (Chang and others, 2022), reveals the positions where the network's top choice has the highest probability.</p>
 <h4>Why the number of steps matters</h4><p>The network outputs one distribution per position. It does not output a joint distribution over several positions at once. Tokens revealed in the same step are therefore drawn <b>independently</b>: none of them knows what the others turned out to be.</p>$$p_\theta(x^i,x^j\mid\mathbf x_t)\ \text{is approximated by}\ p_\theta(x^i\mid\mathbf x_t)\,p_\theta(x^j\mid\mathbf x_t).$$<p>With one step, every position is drawn from its own average behaviour, and the result is a blend of all the training examples: letters that are each plausible for their place but do not form a word, or a picture with pieces of several letters. With \(N=L\) steps exactly one token is revealed at a time, each one sees all the earlier ones, and nothing is approximated. In between is the trade: fewer network calls against more tokens that had to be chosen blind. The chart measures it.</p>
 <h4>How it compares with writing left to right</h4><p>An autoregressive model is the special case in which one token is revealed per step in a fixed left-to-right order. Discrete diffusion relaxes both parts: any order, and several tokens per step. That allows generation in far fewer network calls than there are tokens, and it makes filling a gap in the middle of a text as natural as continuing it. The cost is the independence error above, and the fact that a revealed token is never revisited in this simple form.</p>
 <h4>Other kinds of discrete noise</h4><p>Hiding is not the only choice. D3PM (Austin and others, 2021) also studies replacing a token with a uniformly random one, or with a similar one. SEDD (Lou and others, 2024) learns ratios of probabilities, a discrete counterpart of the score. Hiding has become the common choice for text because it is simple and works well.</p>
 <h4>What this page trains</h4><p>One network with two hidden layers of 96 neurons. Its input marks each position as one of the tokens or as hidden. For Letter Pictures the data is 150 slightly noisy pictures of the letters A, E, H, O, T and X on a 5 by 7 grid, and a sample counts as real if it is within two squares of one of those letters. For Five-Letter Words the data is 64 common words, and a sample counts as real if it is one of them. A list that short can only be memorised, so the words tab shows the process, not creativity.</p>`,
    q: 'With one reveal step the pictures come out as a blur of several letters, yet the network that produced them is exactly the one that draws clean letters in 35 steps. What does the one-step failure tell you about what the network outputs?',
    a: 'It outputs a separate probability for each hidden square, given what is visible. With nothing visible those probabilities are just how often each square is black across all six letters. Drawing all 35 squares from them at once treats the squares as independent, and independent squares cannot agree on which letter they are part of. The knowledge that squares go together is in the network, but it is only used when a revealed square is fed back in as input for the next step.',
    init(root) {
      const c = $('#dd-c', root),
        cr = $('#dd-r', root),
        live = $('#dd-live', root),
        S = { kind: 'pic', ni: 5, ord: 'conf' },
        G = 8,
        KD = {
          pic: {
            Ln: 35,
            V: 2,
            NS: [1, 2, 3, 5, 7, 12, 18, 35],
            def: 5,
            what: 'a real letter',
            unit: 'squares',
          },
          word: { Ln: 5, V: 26, NS: [1, 2, 3, 4, 5], def: 4, what: 'a real word', unit: 'letters' },
        },
        MAXS = 1500,
        WORDS =
          'apple bread chair dance eagle flame grape house juice knife lemon mouse night ocean piano queen river snake table uncle voice water zebra beach cloud dream earth field green heart light money music paper plant radio sheep smile stone sugar tiger train truck watch world young brain bring candy clean clock drink floor fruit glass happy horse lunch magic party pizza robot shirt storm'.split(
            ' ',
          );
      let K,
        net,
        ts = 0,
        LL = [],
        sm = [],
        phase = 'train',
        seqs = [],
        clean = [],
        CHN = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        X = null,
        PR = null,
        si = 0,
        rev = 0,
        done = false,
        Q = [[], []],
        bi = 0,
        okV = NaN,
        just = [],
        ex = null,
        exO = null,
        res = [];
      const W = () => K.Ln * (K.V + 1);
      const enc = (rows) => {
        const B = rows.length,
          w = W(),
          A = new Float64Array(B * w);
        rows.forEach((r, b) => {
          for (let i = 0; i < K.Ln; i++) A[b * w + i * (K.V + 1) + (r[i] < 0 ? K.V : r[i])] = 1;
        });
        return A;
      };
      const probs = (rows) => {
        const B = rows.length,
          o = net.fwd(AD.input(enc(rows), [B, W()]));
        AD.clear();
        const V = K.V,
          P = new Float64Array(o.v.length);
        for (let i = 0; i < B * K.Ln; i++) {
          let mx = -1e30;
          for (let j = 0; j < V; j++) mx = Math.max(mx, o.v[i * V + j]);
          let z = 0;
          for (let j = 0; j < V; j++) {
            P[i * V + j] = Math.exp(o.v[i * V + j] - mx);
            z += P[i * V + j];
          }
          for (let j = 0; j < V; j++) P[i * V + j] /= z;
        }
        return P;
      };
      const match = (r) => {
        if (S.kind === 'word')
          return WORDS.indexOf(r.map((v) => String.fromCharCode(97 + v)).join(''));
        let b = -1,
          bd = 3;
        clean.forEach((g, k) => {
          let d = 0;
          for (let i = 0; i < 35; i++) if (g[i] !== r[i]) d++;
          if (d < bd) {
            bd = d;
            b = k;
          }
        });
        return b;
      };
      const build = () => {
        K = KD[S.kind];
        if (S.kind === 'pic') {
          const d = letterData(run, 0.02);
          seqs = d.X.map((x) => x.map((v) => (v > 0.5 ? 1 : 0)));
          CHN = d.CH;
          const q = letterData(1, 0);
          clean = d.CH.map((_, k) => q.X[k * 25].map((v) => (v > 0.5 ? 1 : 0)));
        } else {
          seqs = WORDS.map((w) => [...w].map((ch) => ch.charCodeAt(0) - 97));
          clean = [];
        }
        net = AD.mlp([W(), 96, 96, K.Ln * K.V], randnFrom(rng(run * 9 + 2)), 'relu');
        ts = 0;
        LL = [];
        sm = [];
        phase = 'train';
        X = null;
        PR = null;
        done = false;
        cu = 0;
        Q = [K.NS.map(() => NaN), K.NS.map(() => NaN)];
        bi = 0;
        okV = NaN;
        just = [];
        res = [];
        si = 0;
        rev = 0;
        ex = seqs[Math.floor(Math.random() * seqs.length)];
        exO = Array.from({ length: K.Ln }, (_, i) => i).sort(() => Math.random() - 0.5);
        msg =
          'A network with random weights. It has ' +
          K.Ln +
          ' positions to fill, each with one of ' +
          K.V +
          ' tokens. Press Run to train it.';
      };
      const train = (k) => {
        const B = 32,
          Ln = K.Ln;
        for (let q = 0; q < k && ts < MAXS; q++) {
          const rows = [],
            tg = new Int32Array(B * Ln),
            mk = new Uint8Array(B * Ln);
          for (let b = 0; b < B; b++) {
            const s = seqs[Math.floor(Math.random() * seqs.length)],
              u = Math.random(),
              r = s.slice();
            let any = 0;
            for (let i = 0; i < Ln; i++) {
              tg[b * Ln + i] = s[i];
              if (Math.random() < u) {
                r[i] = -1;
                mk[b * Ln + i] = 1;
                any = 1;
              }
            }
            if (!any) {
              const i = Math.floor(Math.random() * Ln);
              r[i] = -1;
              mk[b * Ln + i] = 1;
            }
            rows.push(r);
          }
          const l = AD.mce(net.fwd(AD.input(enc(rows), [B, W()])), K.V, tg, mk);
          LL.push(l.v[0]);
          AD.backward(l);
          AD.adam(net.ps, 0.003, ++ts, 1);
          if (ts % 20 === 0) {
            let s = 0;
            for (let j = ts - 20; j < ts; j++) s += LL[j];
            sm.push(s / 20);
          }
        }
        cu = Math.floor(ts / 100) % 2;
        msg =
          'Training step ' +
          ts +
          ' of ' +
          MAXS +
          ': 32 real examples each had a random share of their ' +
          K.unit +
          ' hidden, and the network guessed what was under every blank. Average error over the last 20 steps: ' +
          f3(sm.length ? sm[sm.length - 1] : LL[ts - 1]) +
          ' (guessing blindly would score ' +
          f3(Math.log(K.V)) +
          ').';
        if (ts >= MAXS) {
          phase = 'bench';
          bi = 0;
          cu = 2;
          msg =
            'Training is finished. Next the network generates with each number of reveal steps, in both orders.';
        }
      };
      const reveal = (rows, P, cnt, ord) => {
        const out = [],
          V = K.V,
          Ln = K.Ln;
        rows.forEach((r, b) => {
          const hid = [];
          for (let i = 0; i < Ln; i++)
            if (r[i] < 0) {
              let m = 0;
              for (let j = 0; j < V; j++) m = Math.max(m, P[(b * Ln + i) * V + j]);
              hid.push([i, ord === 'conf' ? m + 1e-6 * Math.random() : Math.random()]);
            }
          hid.sort((a, b2) => b2[1] - a[1]);
          for (let q = 0; q < cnt && q < hid.length; q++) {
            const i = hid[q][0];
            let u = Math.random(),
              tk = 0;
            for (; tk < V - 1; tk++) {
              u -= P[(b * Ln + i) * V + tk];
              if (u <= 0) break;
            }
            r[i] = tk;
            out.push(b * Ln + i);
          }
        });
        return out;
      };
      const sampleAll = (N, ord, B) => {
        const rows = Array.from({ length: B }, () => new Array(K.Ln).fill(-1));
        let rv = 0;
        for (let j = 1; j <= N; j++) {
          const tgt = Math.round((K.Ln * j) / N);
          reveal(rows, probs(rows), tgt - rv, ord);
          rv = tgt;
        }
        let ok = 0;
        for (const r of rows) if (match(r) >= 0) ok++;
        return ok / B;
      };
      const startGen = () => {
        X = Array.from({ length: G }, () => new Array(K.Ln).fill(-1));
        PR = probs(X);
        si = 0;
        rev = 0;
        done = false;
        just = [];
        okV = NaN;
        res = [];
      };
      const bench = () => {
        const j = bi >> 1,
          e = bi & 1,
          N = K.NS[j];
        Q[e][j] = sampleAll(N, e ? 'conf' : 'rand', 64);
        bi++;
        cu = 2;
        msg =
          'Measuring: ' +
          N +
          ' reveal step' +
          (N > 1 ? 's' : '') +
          ', ' +
          (e ? 'most confident first' : 'random order') +
          ': ' +
          pct(Q[e][j]) +
          ' of 64 samples came out as ' +
          K.what +
          '.';
        if (bi >= 2 * K.NS.length) {
          phase = 'gen';
          startGen();
          cu = 3;
          msg =
            'All settings are measured: see the chart at the bottom right. Now watch one run. Eight samples, everything hidden. The tint shows what the network believes before anything is revealed, and it is the same for all eight.';
        }
      };
      const gstep = () => {
        if (done || !X) return;
        const N = K.NS[S.ni],
          tgt = Math.round((K.Ln * (si + 1)) / N),
          cnt = tgt - rev;
        just = reveal(X, PR, cnt, S.ord);
        rev = tgt;
        si++;
        cu = 4;
        msg =
          'Reveal step ' +
          si +
          ' of ' +
          N +
          ': the network was run once on each sample. ' +
          cnt +
          ' hidden ' +
          (cnt === 1 ? K.unit.slice(0, -1) + ' was' : K.unit + ' were') +
          ' chosen in each (' +
          (S.ord === 'conf' ? 'where the network was most sure' : 'at random') +
          ') and filled by drawing from its beliefs' +
          (cnt > 1 ? ', each drawn without knowing the others.' : '.') +
          ' ' +
          (K.Ln - rev) +
          ' remain hidden.';
        if (si >= N) {
          done = true;
          cu = 5;
          res = X.map(match);
          let ok = 0;
          for (const v of res) if (v >= 0) ok++;
          okV = ok / G;
          msg += ' Finished: ' + ok + ' of ' + G + ' samples are ' + K.what + '.';
        } else {
          PR = probs(X);
          cu = 5;
        }
      };
      const step = () => {
        if (phase === 'train') train(50);
        else if (phase === 'bench') bench();
        else if (done) {
          startGen();
          cu = 3;
          msg = 'Eight fresh samples with everything hidden. Press Run or One step.';
        } else gstep();
      };
      const cell = (g, t, x, y, s, v, p, hot) => {
        if (v < 0) {
          g.fillStyle = mix(t.accent, t.accent, 0, 0.08 + 0.6 * p);
          g.fillRect(x, y, s - 1, s - 1);
        } else {
          g.fillStyle = v ? t.ink : t.bg;
          g.fillRect(x, y, s - 1, s - 1);
          if (!v) {
            g.strokeStyle = t.line;
            g.lineWidth = 1;
            g.strokeRect(x + 0.5, y + 0.5, s - 2, s - 2);
          }
        }
        if (hot) {
          g.strokeStyle = t.gold;
          g.lineWidth = 2.5;
          g.strokeRect(x + 0.5, y + 0.5, s - 2, s - 2);
        }
      };
      const tile = (g, t, x, y, s, v, P0, hot) => {
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        if (v < 0) {
          let b = 0;
          for (let j = 1; j < 26; j++) if (P0[j] > P0[b]) b = j;
          g.fillStyle = mix(t.accent, t.accent, 0, 0.08 + 0.3 * P0[b]);
          g.fillRect(x, y, s - 3, s - 3);
          g.fillStyle = mix(t.accent, t.accent, 0, 0.35 + 0.65 * P0[b]);
          g.font = '500 ' + Math.round(s * 0.42) + 'px ' + MONO.split('px ')[1];
          g.fillText(String.fromCharCode(97 + b), x + (s - 3) / 2, y + (s - 3) / 2 - s * 0.08);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.fillText((100 * P0[b]).toFixed(0) + '%', x + (s - 3) / 2, y + s * 0.74);
        } else {
          g.fillStyle = t.bg;
          g.fillRect(x, y, s - 3, s - 3);
          g.strokeStyle = t.ink;
          g.lineWidth = 1.2;
          g.strokeRect(x + 0.5, y + 0.5, s - 4, s - 4);
          g.fillStyle = t.ink;
          g.font = '600 ' + Math.round(s * 0.55) + 'px ' + MONO.split('px ')[1];
          g.fillText(String.fromCharCode(97 + v), x + (s - 3) / 2, y + (s - 3) / 2);
        }
        if (hot) {
          g.strokeStyle = t.gold;
          g.lineWidth = 3;
          g.strokeRect(x + 0.5, y + 0.5, s - 4, s - 4);
        }
        g.textBaseline = 'alphabetic';
      };
      const draw = () => {
        const t = T(),
          Ln = K.Ln,
          V = K.V,
          N = K.NS[S.ni],
          hot = new Set(just);
        {
          const { g, w, h } = fit(c);
          g.font = MONO;
          if (!X) {
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(
              phase === 'bench'
                ? 'measuring each number of reveal steps'
                : 'the eight samples appear here once the network is trained',
              w / 2,
              h / 2,
            );
          } else if (S.kind === 'pic') {
            const s = Math.floor(Math.min((w - 50) / (4 * 5 + 3 * 1.2), (h - 70) / (2 * 7 + 1.6)));
            const ox = (w - (4 * 5 * s + 3 * 1.2 * s)) / 2,
              oy = (h - (2 * 7 * s + 1.6 * s)) / 2 - 6;
            for (let b = 0; b < G; b++) {
              const x0 = ox + (b % 4) * 6.2 * s,
                y0 = oy + Math.floor(b / 4) * 8.6 * s;
              for (let i = 0; i < 35; i++)
                cell(
                  g,
                  t,
                  x0 + (i % 5) * s,
                  y0 + Math.floor(i / 5) * s,
                  s,
                  X[b][i],
                  PR ? PR[(b * Ln + i) * V + 1] : 0,
                  hot.has(b * Ln + i),
                );
              if (done) {
                g.font = MONO;
                g.textAlign = 'center';
                g.fillStyle = res[b] >= 0 ? t.ok : t.cut;
                g.fillText(
                  res[b] >= 0 ? 'reads as ' + CHN[res[b]] : 'not a letter',
                  x0 + 2.5 * s,
                  y0 + 7 * s + 15,
                );
              }
            }
          } else {
            const s = Math.floor(Math.min((w - 70) / (2 * 5 + 1.4), (h - 60) / (4 * 1.5)));
            const ox = (w - (2 * 5 * s + 1.4 * s)) / 2,
              oy = (h - 4 * 1.5 * s) / 2;
            for (let b = 0; b < G; b++) {
              const x0 = ox + (b % 2) * 6.4 * s,
                y0 = oy + Math.floor(b / 2) * 1.5 * s;
              for (let i = 0; i < 5; i++)
                tile(
                  g,
                  t,
                  x0 + i * s,
                  y0,
                  s,
                  X[b][i],
                  PR ? PR.subarray((b * Ln + i) * V, (b * Ln + i + 1) * V) : null,
                  hot.has(b * Ln + i),
                );
              if (done) {
                g.font = MONO;
                g.textAlign = 'left';
                g.fillStyle = res[b] >= 0 ? t.ok : t.cut;
                g.fillText(res[b] >= 0 ? 'a real word' : 'not a word', x0, y0 + s + 9);
              }
            }
          }
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.44,
            FR = [0, 0.25, 0.5, 0.75, 1],
            pw = (w - 20) / 5;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('one training example, hidden more and more: what training sees', 8, 13);
          FR.forEach((f, q) => {
            const nh = Math.round(f * Ln),
              hid = new Set(exO.slice(0, nh)),
              cx = 10 + q * pw + pw / 2;
            if (S.kind === 'pic') {
              const s = Math.floor(Math.min((pw - 14) / 5, (h1 - 48) / 7)),
                x0 = cx - 2.5 * s,
                y0 = 22;
              for (let i = 0; i < 35; i++) {
                const x = x0 + (i % 5) * s,
                  y = y0 + Math.floor(i / 5) * s;
                if (hid.has(i)) {
                  g.fillStyle = mix(t.accent, t.accent, 0, 0.22);
                  g.fillRect(x, y, s - 1, s - 1);
                } else {
                  g.fillStyle = ex[i] ? t.ink : t.bg;
                  g.fillRect(x, y, s - 1, s - 1);
                  if (!ex[i]) {
                    g.strokeStyle = t.line;
                    g.lineWidth = 1;
                    g.strokeRect(x + 0.5, y + 0.5, s - 2, s - 2);
                  }
                }
              }
            } else {
              const rh = (h1 - 24) / 5,
                s = Math.floor(Math.min(rh - 4, 30)),
                x0 = w / 2 - 2.5 * s - 50,
                y0 = 20 + q * rh;
              for (let i = 0; i < 5; i++) {
                const x = x0 + i * s;
                g.textAlign = 'center';
                if (hid.has(i)) {
                  g.fillStyle = mix(t.accent, t.accent, 0, 0.22);
                  g.fillRect(x, y0, s - 2, s - 2);
                } else {
                  g.strokeStyle = t.ink;
                  g.lineWidth = 1;
                  g.strokeRect(x + 0.5, y0 + 0.5, s - 3, s - 3);
                  g.fillStyle = t.ink;
                  g.font = '600 ' + Math.round(s * 0.6) + 'px ' + MONO.split('px ')[1];
                  g.fillText(String.fromCharCode(97 + ex[i]), x + (s - 2) / 2, y0 + s * 0.68);
                }
              }
              g.font = MONO;
              g.fillStyle = t.muted;
              g.textAlign = 'left';
              g.fillText(Math.round(f * 100) + '% hidden', x0 + 5 * s + 12, y0 + s * 0.62);
              return;
            }
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(Math.round(f * 100) + '% hidden', cx, h1 - 6);
          });
          if (phase === 'train')
            uLine(g, t, 0, h1, w, h - h1, sm, {
              title: 'error at guessing hidden tokens (average of every 20 steps)',
              n: MAXS / 20,
              col: t.ok,
              ym: Math.max(Math.log(V) * 1.05, sm[0] || 0),
            });
          else
            g2Bench(
              g,
              t,
              0,
              h1,
              w,
              h - h1,
              K.NS,
              [
                { ys: Q[0], col: t.cut, lab: 'random order' },
                { ys: Q[1], col: t.accent, lab: 'most confident first' },
              ],
              phase === 'gen' ? S.ni : -1,
              'share of samples that are ' + K.what + ', by number of reveal steps',
            );
        }
        live.textContent = msg;
        uCur(root, 'dd', cu);
        setR(
          root,
          'ph',
          phase === 'train'
            ? ts
              ? 'Training'
              : 'Not started'
            : phase === 'bench'
              ? 'Measuring'
              : done
                ? 'Finished'
                : 'Generating',
        );
        setR(root, 'ts', ts + ' of ' + MAXS);
        setR(root, 'L', sm.length ? f3(sm[sm.length - 1]) : '–');
        setR(root, 'rs', phase === 'gen' ? si + ' of ' + N : '–');
        setR(root, 'hid', phase === 'gen' ? Ln - rev + ' of ' + Ln : '–');
        setR(root, 'ok', isFinite(okV) ? Math.round(okV * G) + ' of ' + G : '–');
      };
      const sl = $('#dd-n', root),
        slo = $('#dd-n-o', root),
        setSl = () => {
          sl.max = K.NS.length - 1;
          S.ni = K.def;
          sl.value = K.def;
          if (slo) slo.textContent = String(K.NS[S.ni]);
        };
      build();
      setSl();
      if (RM) {
        train(MAXS);
        while (phase === 'bench') bench();
        while (!done) gstep();
      }
      const anim = Anim(() => {
        fr++;
        if (phase === 'train') {
          train(Math.min(ts < 60 ? 1 : 5, chk.remaining()));
          draw();
          chk();
          return;
        }
        if (phase === 'bench') {
          if (fr % 5 === 0) {
            bench();
            draw();
          }
          return;
        }
        if (fr % clamp(Math.round(260 / K.NS[S.ni]), 14, 70) === 0 && !done) gstep();
        draw();
        if (done) pb.stop();
      });
      pb = playBtn($('#dd-play', root), anim);
      chk = runCtl(pb, root, 'dd', {
        done: () => phase === 'gen' && done,
        restart: () => startGen(),
        count: () => ts,
        what: 'training steps',
      });
      const regen = () => {
        if (phase !== 'gen') {
          msg = 'Samples can be generated once training and measuring are finished. Press Run.';
          draw();
          return;
        }
        pb.stop();
        startGen();
        draw();
        pb.start();
      };
      $('#dd-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#dd-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#dd-gen', root).onclick = regen;
      bind(
        root,
        'dd-n',
        (v) => {
          if (!pb) return;
          S.ni = clamp(+v, 0, K.NS.length - 1);
          if (phase === 'gen') regen();
          else draw();
        },
        (v) => String((K || KD.pic).NS[Math.min(+v, (K || KD.pic).NS.length - 1)]),
      );
      segBind(root, 'dd-ord', S.ord, (v) => {
        S.ord = v;
        if (phase === 'gen') regen();
      });
      segBind(root, 'dd-kind', S.kind, (v) => {
        S.kind = v;
        pb.stop();
        run++;
        build();
        setSl();
        draw();
        pb.start();
      });
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
);
{
  const o = [
    'autoregressive',
    'vae',
    'gan',
    'ddpm',
    'ddpm2d',
    'ddim',
    'stable-diffusion',
    'score-matching',
    'flow-matching',
    'normalizing-flows',
    'energy-based',
    'discrete-diffusion',
  ];
  GEN.sort((a, b) => o.indexOf(a.id) - o.indexOf(b.id));
}
