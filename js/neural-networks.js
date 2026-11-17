/* Manifold — Neural networks.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ neural networks ============ */
const ACT = {
    tanh: [
      Math.tanh,
      (z) => {
        const a = Math.tanh(z);
        return 1 - a * a;
      },
    ],
    relu: [(z) => (z > 0 ? z : 0), (z) => (z > 0 ? 1 : 0)],
    sigmoid: [
      sig,
      (z) => {
        const a = sig(z);
        return a * (1 - a);
      },
    ],
    leaky: [(z) => (z > 0 ? z : 0.1 * z), (z) => (z > 0 ? 1 : 0.1)],
  },
  ACTN = { tanh: 'tanh', relu: 'ReLU', sigmoid: 'Sigmoid', leaky: 'Leaky ReLU' };
function NNet(sizes, o = {}) {
  const act = o.act || 'tanh',
    out = o.out || 'sig',
    r = randnFrom(rng(o.seed || 1)),
    L = sizes.length - 1,
    W = [],
    B = [],
    mW = [],
    vW = [],
    mB = [],
    vB = [],
    f = ACT[act][0],
    df = ACT[act][1];
  let ts = 0;
  for (let l = 0; l < L; l++) {
    const n = sizes[l],
      m = sizes[l + 1],
      sd = (o.scale || 1) * Math.sqrt((act === 'relu' || act === 'leaky' ? 2 : 1) / n);
    W.push(Array.from({ length: m }, () => Float64Array.from({ length: n }, () => sd * r())));
    B.push(new Float64Array(m));
    mW.push(W[l].map(() => new Float64Array(n)));
    vW.push(W[l].map(() => new Float64Array(n)));
    mB.push(new Float64Array(m));
    vB.push(new Float64Array(m));
  }
  const fwd = (x, mask) => {
    const A = [x],
      Z = [];
    for (let l = 0; l < L; l++) {
      const m = sizes[l + 1],
        z = new Float64Array(m),
        a = new Float64Array(m),
        last = l === L - 1,
        p = A[l];
      for (let j = 0; j < m; j++) {
        let s = B[l][j];
        const w = W[l][j];
        for (let i = 0; i < w.length; i++) s += w[i] * p[i];
        z[j] = s;
        a[j] = last ? (out === 'sig' ? sig(s) : s) : f(s) * (mask ? mask[l][j] : 1);
      }
      Z.push(z);
      A.push(a);
    }
    return { A, Z };
  };
  const train = (X, Y, lr, q = {}) => {
    const gW = W.map((wl) => wl.map((w) => new Float64Array(w.length))),
      gB = B.map((b) => new Float64Array(b.length)),
      p = q.drop || 0;
    let loss = 0,
      lastMask = null;
    for (let n = 0; n < X.length; n++) {
      let mask = null;
      if (p > 0) {
        mask = sizes
          .slice(1, L)
          .map((m) =>
            Float64Array.from({ length: m }, () => (Math.random() < p ? 0 : 1 / (1 - p))),
          );
        lastMask = mask;
      }
      const { A, Z } = fwd(X[n], mask),
        y = Y[n];
      let d;
      if (y.length) {
        const O = A[L];
        d = new Float64Array(O.length);
        for (let k = 0; k < O.length; k++) {
          d[k] = O[k] - y[k];
          loss +=
            out === 'sig'
              ? -(y[k] * Math.log(O[k] + 1e-9) + (1 - y[k]) * Math.log(1 - O[k] + 1e-9))
              : 0.5 * d[k] * d[k];
        }
      } else {
        const ov = A[L][0];
        loss +=
          out === 'sig'
            ? -(y * Math.log(ov + 1e-9) + (1 - y) * Math.log(1 - ov + 1e-9))
            : 0.5 * (ov - y) ** 2;
        d = Float64Array.of(ov - y);
      }
      for (let l = L - 1; l >= 0; l--) {
        const pa = A[l];
        for (let j = 0; j < d.length; j++) {
          gB[l][j] += d[j];
          const g = gW[l][j],
            dj = d[j];
          for (let i = 0; i < pa.length; i++) g[i] += dj * pa[i];
        }
        if (l > 0) {
          const nd = new Float64Array(sizes[l]);
          for (let i = 0; i < nd.length; i++) {
            let s = 0;
            for (let j = 0; j < d.length; j++) s += W[l][j][i] * d[j];
            nd[i] = s * df(Z[l - 1][i]) * (mask ? mask[l - 1][i] : 1);
          }
          d = nd;
        }
      }
    }
    ts++;
    const N = X.length,
      c1 = 1 - Math.pow(0.9, ts),
      c2 = 1 - Math.pow(0.999, ts),
      up = (par, g, m, v) => {
        for (let i = 0; i < par.length; i++) {
          const gi = g[i] / N;
          if (q.sgd) par[i] -= lr * gi;
          else {
            m[i] = 0.9 * m[i] + 0.1 * gi;
            v[i] = 0.999 * v[i] + 0.001 * gi * gi;
            par[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
          }
        }
      };
    for (let l = q.freeze || 0; l < L; l++) {
      for (let j = 0; j < W[l].length; j++) up(W[l][j], gW[l][j], mW[l][j], vW[l][j]);
      up(B[l], gB[l], mB[l], vB[l]);
    }
    return { loss: loss / N, mask: lastMask };
  };
  return { W, B, sizes, L, fwd, train, p: (x) => fwd(x).A[L][0], p2: (x) => fwd(x).A[L] };
}
function nnData(kind, n, nz, seed) {
  const r = rng(seed * 19 + 3),
    rn = randnFrom(r),
    P = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    let x, y;
    if (kind === 'circles') {
      const a = r() * 2 * Math.PI,
        rad = c ? 2 + 0.25 * rn() * (1 + 2 * nz) : 0.7 * Math.sqrt(r());
      x = rad * Math.cos(a) + nz * 0.5 * rn();
      y = rad * Math.sin(a) + nz * 0.5 * rn();
    } else if (kind === 'moons') {
      const a = r() * Math.PI;
      x = c ? 0.7 - 1.6 * Math.cos(a) : -0.7 + 1.6 * Math.cos(a);
      y = c ? 0.4 - 1.5 * Math.sin(a) : -0.4 + 1.5 * Math.sin(a);
      x += nz * rn();
      y += nz * rn();
    } else if (kind === 'xor') {
      x = (r() * 2.4 + 0.2) * (r() < 0.5 ? -1 : 1);
      y = (r() * 2.4 + 0.2) * (r() < 0.5 ? -1 : 1);
      x += nz * 0.6 * rn();
      y += nz * 0.6 * rn();
      P.push({
        x: clamp(x, -2.9, 2.9),
        y: clamp(y, -2.9, 2.9),
        c: x * y > 0 ? 1 : 0,
        test: i % 4 === 3,
      });
      continue;
    } else if (kind === 'spiral') {
      const u = r(),
        a = u * 3.6 + c * Math.PI,
        rad = 0.3 + 2.3 * u;
      x = rad * Math.cos(a) + nz * 0.35 * rn();
      y = rad * Math.sin(a) + nz * 0.35 * rn();
    } else {
      x = (c ? 1.2 : -1.2) + (0.6 + nz) * rn();
      y = (c ? 0.8 : -0.8) + (0.6 + nz) * rn();
    }
    P.push({ x: clamp(x, -2.9, 2.9), y: clamp(y, -2.9, 2.9), c, test: i % 4 === 3 });
  }
  return P;
}
function netPos(sizes, x0, y0, w, h) {
  return sizes.map((m, l) =>
    Array.from({ length: m }, (_, j) => [
      x0 + (sizes.length > 1 ? l / (sizes.length - 1) : 0.5) * w,
      y0 + h / 2 + (j - (m - 1) / 2) * Math.min(52, h / Math.max(m, 1)),
    ]),
  );
}
function drawEdges(g, W, pos, t, ph, col) {
  let wm = 1e-9;
  W.forEach((wl) =>
    wl.forEach((w) =>
      w.forEach((v) => {
        wm = Math.max(wm, Math.abs(v));
      }),
    ),
  );
  W.forEach((wl, l) =>
    wl.forEach((w, j) => {
      if (!pos[l + 1][j]) return;
      w.forEach((v, i) => {
        if (!pos[l][i]) return;
        const a = pos[l][i],
          b = pos[l + 1][j],
          al = Math.min(1, Math.abs(v) / wm),
          cl = v >= 0 ? t.accent : t.ok;
        g.strokeStyle = mix(cl, cl, 0, 0.12 + 0.78 * al);
        g.lineWidth = 0.6 + 3.2 * al;
        g.beginPath();
        g.moveTo(a[0], a[1]);
        g.lineTo(b[0], b[1]);
        g.stroke();
        if (ph != null && al > 0.12) {
          const u = (((ph + i * 0.17 + j * 0.31 + l * 0.4) % 1) + 1) % 1;
          g.fillStyle = col || t.gold;
          g.beginPath();
          g.arc(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, 1.6 + 2 * al, 0, 7);
          g.fill();
        }
      });
    }),
  );
}
function sigFlow(act, depth, base, seed) {
  const n = 40,
    bt = 50,
    r = randnFrom(rng(seed)),
    f = ACT[act][0],
    df = ACT[act][1],
    W = [],
    acts = [],
    Zs = [];
  let A = [];
  for (let b = 0; b < bt; b++) A.push(Float64Array.from({ length: n }, () => r()));
  for (let l = 0; l < depth; l++) {
    const sd = Math.sqrt(base / n),
      w = Array.from({ length: n }, () => Float64Array.from({ length: n }, () => sd * r()));
    W.push(w);
    const Z = [],
      NA = [];
    for (let b = 0; b < bt; b++) {
      const z = new Float64Array(n),
        a = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += w[j][i] * A[b][i];
        z[j] = s;
        a[j] = f(s);
      }
      Z.push(z);
      NA.push(a);
    }
    Zs.push(Z);
    A = NA;
    const fl = new Float32Array(n * bt);
    A.forEach((a, b) => fl.set(a, b * n));
    acts.push(fl);
  }
  const gr = new Array(depth).fill(0);
  let D = [];
  for (let b = 0; b < bt; b++) D.push(Float64Array.from({ length: n }, () => r()));
  for (let l = depth - 1; l >= 0; l--) {
    let s = 0;
    const ND = [];
    for (let b = 0; b < bt; b++) {
      const dz = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        dz[j] = D[b][j] * df(Zs[l][b][j]);
        s += dz[j] * dz[j];
      }
      const da = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let q = 0;
        for (let j = 0; j < n; j++) q += W[l][j][i] * dz[j];
        da[i] = q;
      }
      ND.push(da);
    }
    gr[l] = Math.sqrt(s / (n * bt));
    D = ND;
  }
  const sd = acts.map((a) => {
    let m = 0,
      s = 0;
    for (const v of a) m += v;
    m /= a.length;
    for (const v of a) s += (v - m) ** 2;
    return Math.sqrt(s / a.length);
  });
  return { acts, gr, sd };
}
const sci = (v) =>
  v === 0
    ? '0'
    : Math.abs(v) >= 0.01 && Math.abs(v) < 1000
      ? (+v.toPrecision(3)).toString()
      : v.toExponential(1).replace('e', '×10^');

const NNL = [
  {
    id: 'mlp',
    tab: 'Multilayer perceptron',
    title: 'Watch a network learn a shape',
    lede: 'A single neuron can only draw one straight edge. Put a few of them in a layer, feed that layer into another, and the edges combine into curves, rings and spirals. Here you can watch that happen: the picture inside each neuron shows what that neuron has learned to respond to.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The network’s answer at every position: green side or gold side. Filled dots are training points, rings are test points</figcaption><canvas class="short" id="nn-c" role="img" aria-label="Decision regions of the network with the data"></canvas></figure>
 <figure class="fig"><figcaption>Top: the network. Each small square shows where that neuron is active. Line colour is the sign of the weight, thickness its size. Bottom: loss by epoch</figcaption><canvas class="sq" id="nn-r" role="img" aria-label="Network diagram with per-neuron activity and the loss curves"></canvas></figure></div>
 ${uHow('nn', ['Start with random weights. The answer is a meaningless smear.', 'Forward pass: every point flows through the network, left to right, and comes out as a number between 0 and 1.', 'Loss: compare those numbers with the true colours.', 'Backward pass: work out how each weight should change to lower the loss, and nudge them all.', 'Repeat. Stop when the loss stops falling.'])}
 <div class="controls">${uBtns('nn', 'Train', 'One epoch', 'epochs')}
 ${grp('The network', slider('nn-hl', 'Hidden layers', 1, 3, 1, 2) + slider('nn-hw', 'Neurons in each hidden layer', 1, 8, 1, 5) + `<label class="ctl" for="nn-act"><span>Activation</span><select id="nn-act"><option value="tanh">tanh</option><option value="relu">ReLU</option><option value="sigmoid">Sigmoid</option></select></label>` + slider('nn-lr', 'Learning rate', 0.003, 0.2, 0.001, 0.015))}
 ${grp('The data', `<label class="ctl" for="nn-ds"><span>Shape to learn</span><select id="nn-ds"><option value="circles">A ring around a core</option><option value="moons">Two moons</option><option value="xor">Checkerboard</option><option value="spiral">Two spirals</option><option value="blobs">Two clouds</option></select></label>` + slider('nn-nz', 'Noise', 0, 0.6, 0.01, 0.2) + slider('nn-n', 'Number of points', 60, 400, 2, 200) + `<div class="row"><button class="btn" id="nn-new" type="button">New random sample</button></div>`)}</div>
 <dl class="read">${rd('ep', 'Epochs')}${rd('lt', 'Training loss', 'hl3')}${rd('le', 'Test loss', 'hl2')}${rd('at', 'Training accuracy')}${rd('ae', 'Test accuracy', 'hl')}${rd('np', 'Weights and biases in the network')}${rd('st', 'Status')}</dl>`,
    tryit:
      'On the ring, set one hidden layer with 1 neuron, then 2, then 3, and train each time. How many neurons does it take to close a ring, and what do their little pictures look like? Then choose Two Spirals: one layer of 8 struggles, while three layers of 8 can do it. Watch the pictures inside the neurons get more complicated from left to right.',
    math: String.raw`<h4>Intuition</h4><p>Think of a sculptor who may only make straight cuts. One cut divides the block in two: that is one neuron. Three cuts can leave a triangle standing. With enough cuts, and a second sculptor who combines the pieces the first one made, any shape can be carved. A neural network works the same way. The first hidden layer makes straight cuts through the input space, and each later layer combines the pieces from the layer before into more elaborate regions.</p>
 <h4>One neuron</h4><p>A neuron takes its inputs, multiplies each by a weight, adds a bias, and passes the total through an activation function \(\phi\):</p>$$a=\phi\big(w_1x_1+w_2x_2+b\big).$$<p>The set of points where \(w_1x_1+w_2x_2+b=0\) is a straight line, and the neuron is "on" on one side and "off" on the other. Look at any neuron in the first hidden layer: its picture is always a straight edge.</p>
 <h4>Layers</h4><p>A layer is many neurons reading the same inputs. Stacking the weights into a matrix \(\mathbf W^{(l)}\) and the biases into a vector \(\mathbf b^{(l)}\),</p>$$\mathbf a^{(0)}=\mathbf x,\qquad\mathbf a^{(l)}=\phi\big(\mathbf W^{(l)}\mathbf a^{(l-1)}+\mathbf b^{(l)}\big),\qquad\hat y=\sigma\big(\mathbf w^{\top}\mathbf a^{(L)}+b\big),$$<p>where the final \(\sigma\) is the logistic function, so that \(\hat y\) is the probability of the gold class. A neuron in layer 2 takes a weighted sum of the straight-edge pictures from layer 1 and bends the result, which is how corners and closed regions appear.</p>
 <h4>Why the activation must bend</h4><p>If \(\phi\) were the identity, each layer would be a linear map, and a stack of linear maps is one linear map: \(\mathbf W^{(2)}\mathbf W^{(1)}\mathbf x\). The whole network would be a single straight cut, however deep. The non-linear activation is what lets layers build on one another.</p>
 <h4>What training does</h4><p>The loss is the average log loss (cross-entropy) over the training points,</p>$$\mathcal L=-\frac1n\sum_{i=1}^{n}\big[y_i\log\hat y_i+(1-y_i)\log(1-\hat y_i)\big].$$<p>Each epoch computes the gradient of \(\mathcal L\) with respect to every weight by backpropagation (its own tab) and moves the weights a small step against it. This page uses the Adam optimiser from the Foundations module. Training stops here when the loss has stopped falling.</p>
 <h4>Width and depth</h4><p>More neurons in a layer give more cuts to work with. More layers let cuts be combined into pieces, and pieces into shapes, so a deep network can describe some shapes with far fewer neurons than a wide shallow one. The count of weights grows quickly: a layer of \(m\) neurons reading \(n\) inputs has \(m(n+1)\) parameters. When the network is much larger than the data needs, training loss keeps falling while test loss turns back up: the same overfitting you saw with polynomials.</p>`,
    q: 'A network with one hidden layer of 2 tanh neurons cannot separate the ring from its core. Why not?',
    a: 'Each first-layer neuron makes one straight cut, so two neurons give two lines. The output can only combine "which side of line 1" and "which side of line 2", and no region bounded by two lines is closed: it always runs off to infinity somewhere. Closing a region around the core takes at least three cuts, which is why three neurons are the minimum.',
    init(root) {
      const c = $('#nn-c', root),
        cr = $('#nn-r', root),
        live = $('#nn-live', root),
        S = { hl: 2, hw: 5, act: 'tanh', lr: 0.015, ds: 'circles', nz: 0.2, n: 200, seed: 3 };
      let pts = [],
        net,
        ep = 0,
        LT = [],
        LE = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        conv = false,
        run = 1;
      const tr = () => pts.filter((p) => !p.test),
        te = () => pts.filter((p) => p.test),
        ll = (P) =>
          P.length
            ? -P.reduce((s, p) => {
                const o = net.p([p.x, p.y]);
                return s + (p.c ? Math.log(o + 1e-9) : Math.log(1 - o + 1e-9));
              }, 0) / P.length
            : NaN,
        ac = (P) =>
          P.length
            ? P.filter((p) => (net.p([p.x, p.y]) >= 0.5 ? 1 : 0) === p.c).length / P.length
            : NaN;
      const build = () => {
        net = NNet([2, ...Array(S.hl).fill(S.hw), 1], { act: S.act, seed: S.seed * 7 + run * 13 });
        ep = 0;
        LT = [];
        LE = [];
        conv = false;
        cu = 0;
        msg =
          'Random weights: the network has not seen any data yet, and its answer is a smear. Press Train or One epoch.';
      };
      const epoch = () => {
        const T = tr(),
          l = net.train(
            T.map((p) => [p.x, p.y]),
            T.map((p) => p.c),
            S.lr,
          ).loss;
        ep++;
        LT.push(l);
        LE.push(ll(te()));
        cu = 3;
        if (
          (ep > 120 && Math.abs(LT[ep - 1] - LT[ep - 61]) < 4e-3 * LT[ep - 1]) ||
          l < 0.012 ||
          ep >= 4000
        )
          conv = true;
        msg =
          'Epoch ' +
          ep +
          ': all ' +
          T.length +
          ' training points went forward through the network, the loss came out at ' +
          f3(l) +
          (ep > 1 ? ' (it was ' + f3(LT[ep - 2]) + ')' : '') +
          ', and every weight was nudged to lower it. Training accuracy ' +
          pct(ac(T)) +
          ', test accuracy ' +
          pct(ac(te())) +
          '.' +
          (conv
            ? ep >= 4000
              ? ' Reached the limit of 4000 epochs.'
              : ' Converged: the loss has stopped falling, so training stops here.'
            : '');
        if (conv) cu = 4;
      };
      const tp = testPoint(
        root,
        'nn',
        ['Your x', 'Your y'],
        [0, 0],
        [
          [-3, 3],
          [-3, 3],
        ],
      );
      const draw = () => {
        const t = T(),
          ca = hex(t.accent),
          cg = hex(t.gold);
        {
          const { g, w, h } = fit(c),
            cs = 8,
            X = (x) => w / 2 + (x * w) / 6,
            Y = (y) => h / 2 - (y * h) / 6;
          for (let py = 0; py < h; py += cs)
            for (let px = 0; px < w; px += cs) {
              const p = net.p([((px + cs / 2 - w / 2) * 6) / w, (-(py + cs / 2 - h / 2) * 6) / h]),
                q = Math.abs(p - 0.5) * 2;
              g.fillStyle = mix(p < 0.5 ? t.accent : t.gold, t.bg, 0, 0.1 + 0.42 * q);
              g.fillRect(px, py, cs, cs);
            }
          for (const p of pts) {
            const cl = p.c ? t.gold : t.accent;
            g.beginPath();
            g.arc(X(p.x), Y(p.y), p.test ? 3.4 : 3.6, 0, 7);
            if (p.test) {
              g.fillStyle = t.bg;
              g.fill();
              g.strokeStyle = cl;
              g.lineWidth = 1.8;
              g.stroke();
            } else {
              g.fillStyle = cl;
              g.fill();
              g.strokeStyle = t.bg;
              g.lineWidth = 0.8;
              g.stroke();
            }
          }
          markTP(g, X(tp.v[0]), Y(tp.v[1]), t);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.62,
            pos = netPos(net.sizes, 34, 8, w - 68, h1 - 16),
            NC = 9,
            act = [];
          drawEdges(g, net.W, pos, t, pb && pb.on ? fr / 50 : null);
          for (let a = 0; a < NC; a++)
            for (let b = 0; b < NC; b++)
              act.push(net.fwd([-3 + (6 * (b + 0.5)) / NC, 3 - (6 * (a + 0.5)) / NC]).A);
          pos.forEach((col, l) =>
            col.forEach((q, j) => {
              const sz = l === 0 ? 22 : 30,
                x0 = q[0] - sz / 2,
                y0 = q[1] - sz / 2,
                cz = sz / NC;
              let lo = 1e9,
                hi = -1e9;
              if (l > 0 && l < net.L)
                for (const A of act) {
                  lo = Math.min(lo, A[l][j]);
                  hi = Math.max(hi, A[l][j]);
                }
              for (let k = 0; k < NC * NC; k++) {
                let v =
                  l === 0
                    ? j === 0
                      ? ((k % NC) + 0.5) / NC
                      : 1 - (Math.floor(k / NC) + 0.5) / NC
                    : l === net.L
                      ? act[k][l][j]
                      : (act[k][l][j] - lo) / (hi - lo || 1);
                g.fillStyle =
                  'rgb(' + ca.map((u, i) => Math.round(u + (cg[i] - u) * v)).join(',') + ')';
                g.fillRect(x0 + (k % NC) * cz, y0 + Math.floor(k / NC) * cz, cz + 0.6, cz + 0.6);
              }
              g.strokeStyle = t.ink;
              g.lineWidth = 1.3;
              g.strokeRect(x0, y0, sz, sz);
            }),
          );
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('x, y', pos[0][0][0], h1 - 1);
          g.fillText('answer', pos[net.L][0][0], h1 - 1);
          if (net.L > 1) g.fillText('hidden layers', w / 2, h1 - 1);
          const ym = Math.max(0.72, ...LT.slice(0, 3), ...LE.slice(0, 3)),
            U = uLine(g, t, 0, h1, w, h - h1, LT, {
              title: 'loss: training (green) and test (red)',
              n: Math.max(60, LT.length),
              ym,
              col: t.accent,
            });
          if (LE.length) {
            g.strokeStyle = t.ok;
            g.lineWidth = 2;
            g.beginPath();
            LE.forEach((v, i) =>
              i ? g.lineTo(U.X(i), U.Y(Math.min(v, ym))) : g.moveTo(U.X(i), U.Y(Math.min(v, ym))),
            );
            g.stroke();
          }
        }
        live.textContent = msg;
        uCur(root, 'nn', cu);
        const o = net.p(tp.v);
        tp.out.textContent =
          'At (' +
          f2(tp.v[0]) +
          ', ' +
          f2(tp.v[1]) +
          ') the network outputs ' +
          f3(o) +
          ', so it says ' +
          (o >= 0.5 ? 'gold' : 'green') +
          ' with ' +
          (100 * Math.max(o, 1 - o)).toFixed(1) +
          '% confidence.';
        let np = 0;
        net.W.forEach((wl, l) => {
          np += wl.length * (wl[0].length + 1);
        });
        setR(root, 'ep', String(ep));
        setR(root, 'lt', LT.length ? f3(LT[ep - 1]) : '–');
        setR(root, 'le', LE.length ? f3(LE[ep - 1]) : '–');
        setR(root, 'at', pct(ac(tr())));
        setR(root, 'ae', pct(ac(te())));
        setR(root, 'np', String(np));
        setR(root, 'st', conv ? 'Converged' : ep ? 'Training' : 'Not started');
      };
      tp.on = () => draw();
      const gen = () => {
          pts = nnData(S.ds, S.n, S.nz, S.seed);
        },
        redo = () => {
          pb.stop();
          run++;
          build();
          draw();
        };
      gen();
      build();
      if (RM) for (let i = 0; i < 4000 && !conv; i++) epoch();
      const anim = Anim(() => {
        fr++;
        if (fr % (ep < 20 ? 8 : ep < 90 ? 3 : 1)) {
          draw();
          return;
        }
        for (let i = 0; i < (ep < 400 ? 1 : 3) && !conv; i++) epoch();
        draw();
        chk();
      });
      pb = playBtn($('#nn-play', root), anim);
      chk = runCtl(pb, root, 'nn', {
        done: () => conv,
        restart: () => {
          run++;
          build();
        },
        count: () => ep,
        what: 'epochs',
      });
      $('#nn-step', root).onclick = () => {
        pb.stop();
        if (conv) {
          run++;
          build();
        } else epoch();
        draw();
      };
      $('#nn-reset', root).onclick = redo;
      bind(
        root,
        'nn-hl',
        (v) => {
          S.hl = v;
          redo();
        },
        (v) => String(v),
      );
      bind(
        root,
        'nn-hw',
        (v) => {
          S.hw = v;
          redo();
        },
        (v) => String(v),
      );
      $('#nn-act', root).onchange = (e) => {
        S.act = e.target.value;
        redo();
      };
      bind(
        root,
        'nn-lr',
        (v) => {
          S.lr = v;
        },
        f3,
      );
      $('#nn-ds', root).onchange = (e) => {
        S.ds = e.target.value;
        gen();
        redo();
      };
      bind(root, 'nn-nz', (v) => {
        S.nz = v;
        gen();
        redo();
      });
      bind(
        root,
        'nn-n',
        (v) => {
          S.n = v;
          gen();
          redo();
        },
        (v) => String(v),
      );
      $('#nn-new', root).onclick = () => {
        S.seed++;
        gen();
        redo();
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
    id: 'backprop',
    tab: 'Backpropagation',
    title: 'Backpropagation: follow one example through, and back',
    lede: 'Training a network means answering one question for every weight: if I nudge this weight up a little, does the error go up or down, and by how much? Backpropagation answers it for all weights in one backward sweep. Here a tiny network decides whether to take an umbrella, and every number is on the screen.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The network. Numbers in the circles are values, numbers on the lines are weights. Gold packets carry values forward; red packets carry blame backward</figcaption><canvas class="short" id="bp-c" role="img" aria-label="A two-input, two-hidden-neuron network with values, weights and gradients"></canvas>
 <div class="eq" id="bp-eq" role="status" aria-label="The calculation for this step"></div></figure>
 <figure class="fig"><figcaption>Top: the answer against the truth. Then: the gradient of every weight as it becomes known, the error after each update, and the path of every weight</figcaption><canvas class="sq" id="bp-r" role="img" aria-label="Prediction, gradients, loss curve and weight paths"></canvas></figure></div>
 <figure class="fig" style="margin-top:20px"><figcaption>All nine numbers the network can change: where each stands, which way it should move, and where it moves to</figcaption><div class="ctab" id="bp-tab"></div></figure>
 ${uHow('bp', ['Put the inputs in.', 'Forward: hidden neuron 1 takes a weighted sum of the inputs and squashes it. Then hidden neuron 2 does the same.', 'Forward: the output neuron does the same with the two hidden values. That is the network’s answer.', 'Compare the answer with what really happened: the loss.', 'Backward: how much is the output neuron to blame?', 'Backward: so how should each weight into the output change?', 'Backward: pass the blame to hidden neuron 1, then to hidden neuron 2, along the weights that connect them.', 'Backward: so how should each weight into the hidden neurons change?', 'Update: move every weight a small step against its gradient. Then start again.'])}
 <div class="controls">${uBtns('bp', 'Run', 'One step', 'updates')}
 ${grp('The example', slider('bp-x1', 'Cloud cover (0 = clear, 1 = overcast)', 0, 1, 0.01, 0.8) + slider('bp-x2', 'Humidity (0 = dry, 1 = damp)', 0, 1, 0.01, 0.6) + `<div class="row seg" id="bp-y" role="group" aria-label="What really happened"><button class="btn" type="button" data-v="1">It Rained</button><button class="btn" type="button" data-v="0">It Stayed Dry</button></div>`)}
 ${grp('The model', slider('bp-lr', 'Learning rate', 0.1, 4, 0.1, 1.5) + `<label class="check"><input type="checkbox" id="bp-slow" checked> Show every calculation for the first two updates, then speed up</label>`)}</div>
 <dl class="read">${rd('o', 'Network’s answer: chance of rain', 'hl')}${rd('y', 'What really happened')}${rd('L', 'Loss ½(answer − truth)²', 'hl2')}${rd('u', 'Updates so far')}${rd('st', 'Status')}</dl>`,
    tryit:
      'Press Start over and use One step. Each press does one small calculation: watch the packets travel, then read the lines that appear under the picture, where the formula is filled in with the numbers from the picture. Twelve presses make one full update. Then press Run. After two full updates it speeds up, and you can watch the loss fall and the nine weights drift until it converges. Switch to It Stayed Dry: which weights reverse direction?',
    math: String.raw`<h4>Intuition</h4><p>A restaurant serves a dish that is too salty. The head chef does not retrain the whole kitchen. She asks who added salt, and how much each contributed. The cook who made the sauce is mostly responsible; the one who salted the pasta water, a little. Each of them then asks the same of whoever supplied their ingredients. Blame is passed backward along the same routes the ingredients travelled forward, shared out in proportion to how much each route mattered. That is backpropagation. The dish is the output, the saltiness is the error, and the routes are the weights.</p>
 <h4>The forward pass</h4><p>This network has inputs \(x_1,x_2\), two hidden neurons with \(\tanh\), and one sigmoid output:</p>$$z_j=w_{j1}x_1+w_{j2}x_2+b_j,\quad h_j=\tanh z_j,\qquad z_o=v_1h_1+v_2h_2+c,\quad\hat y=\sigma(z_o),\qquad L=\tfrac12(\hat y-y)^2 .$$
 <h4>The backward pass is the chain rule</h4><p>We want \(\partial L/\partial\) (every weight). Define the "blame" of a neuron as the derivative of the loss with respect to its total input, \(\delta=\partial L/\partial z\). For the output neuron,</p>$$\delta_o=\frac{\partial L}{\partial\hat y}\cdot\frac{\partial\hat y}{\partial z_o}=(\hat y-y)\;\hat y(1-\hat y).$$<p>A weight's gradient is the blame of the neuron it feeds, times the value that travelled along it:</p>$$\frac{\partial L}{\partial v_j}=\delta_o\,h_j,\qquad\frac{\partial L}{\partial c}=\delta_o .$$<p>The blame of a hidden neuron is the output's blame, carried back along the connecting weight and multiplied by the slope of the hidden neuron's own activation:</p>$$\delta_j=\delta_o\,v_j\,\big(1-h_j^2\big),\qquad\frac{\partial L}{\partial w_{ji}}=\delta_j\,x_i,\qquad\frac{\partial L}{\partial b_j}=\delta_j .$$
 <h4>The general rule</h4><p>For any layered network the same two lines repeat from the last layer to the first:</p>$$\boldsymbol\delta^{(l)}=\big(\mathbf W^{(l+1)\top}\boldsymbol\delta^{(l+1)}\big)\odot\phi'\big(\mathbf z^{(l)}\big),\qquad\frac{\partial L}{\partial\mathbf W^{(l)}}=\boldsymbol\delta^{(l)}\,\mathbf a^{(l-1)\top}.$$<p>One forward pass stores the values, one backward pass reuses them, and the cost of getting every gradient is about the same as computing the output once. Without this, each weight would need its own forward pass, which for a network with a million weights would be a million times slower.</p>
 <h4>The update</h4><p>Every parameter then takes a step against its gradient, \(\theta\leftarrow\theta-\eta\,\partial L/\partial\theta\), with learning rate \(\eta\). The table lists all nine. On this page the loop stops when the loss is below 0.0005, where the answer is within about three percentage points of the truth.</p>
 <h4>What to notice</h4><p>The gradient of a weight is a product of three kinds of factor: an error, some weights along the way, and slopes of activations. If the output is saturated, \(\hat y(1-\hat y)\) is near zero and everything behind it learns slowly even when the answer is badly wrong. If an input is 0, the weights leaving it get zero gradient and do not move. And a neuron connected to the output by a large weight receives a large share of the blame. The Activation Functions tab shows what these products do in a deep network.</p>`,
    q: 'Set cloud cover to 0 and step through. The two weights leaving that input have gradient exactly 0, whatever the error. Why, and is that a problem?',
    a: 'The gradient of a weight is the blame of the neuron it feeds times the value carried along it, and that value is 0. Changing the weight would change nothing about this prediction, so this example gives no information about it. It is not a problem in practice: other training examples have non-zero cloud cover and will move those weights. It is one reason inputs are usually centred so that they are not stuck at zero.',
    init(root) {
      const c = $('#bp-c', root),
        cr = $('#bp-r', root),
        live = $('#bp-live', root),
        tab = $('#bp-tab', root),
        eq = $('#bp-eq', root),
        S = { x: [0.8, 0.6], y: 1, lr: 1.5, slow: true };
      let w,
        b,
        v,
        cc,
        ph = 0,
        V = {},
        hist = [],
        WH = [],
        upd = 0,
        msg = '',
        fr = 0,
        t0 = 0,
        pb,
        chk = () => false,
        old = null,
        nl = 1;
      const s4 = (q) => (q >= 0 ? '+' : '−') + Math.abs(q).toFixed(3),
        n3 = (q) => (q < 0 ? '(−' + Math.abs(q).toFixed(3) + ')' : q.toFixed(3)),
        n2 = (q) => (q < 0 ? '(−' + Math.abs(q).toFixed(2) + ')' : q.toFixed(2)),
        B = (q) => '<b>' + q + '</b>';
      const all = () => [w[0][0], w[0][1], b[0], w[1][0], w[1][1], b[1], v[0], v[1], cc],
        NM = ['w₁₁', 'w₁₂', 'b₁', 'w₂₁', 'w₂₂', 'b₂', 'v₁', 'v₂', 'c'],
        LB = [
          'hidden 1 ← cloud cover',
          'hidden 1 ← humidity',
          'hidden 1 bias',
          'hidden 2 ← cloud cover',
          'hidden 2 ← humidity',
          'hidden 2 bias',
          'output ← hidden 1',
          'output ← hidden 2',
          'output bias',
        ];
      const calc = () => {
        const z = [0, 1].map((j) => w[j][0] * S.x[0] + w[j][1] * S.x[1] + b[j]),
          h = z.map(Math.tanh),
          zo = v[0] * h[0] + v[1] * h[1] + cc,
          o = sig(zo),
          L = 0.5 * (o - S.y) ** 2,
          dO = (o - S.y) * o * (1 - o),
          dH = [0, 1].map((j) => dO * v[j] * (1 - h[j] * h[j]));
        V = {
          z,
          h,
          zo,
          o,
          L,
          dO,
          dH,
          G: [
            dH[0] * S.x[0],
            dH[0] * S.x[1],
            dH[0],
            dH[1] * S.x[0],
            dH[1] * S.x[1],
            dH[1],
            dO * h[0],
            dO * h[1],
            dO,
          ],
        };
      };
      const show = (lines) => {
        nl = lines.length;
        eqShow(eq, lines, S.slow ? 480 : 0);
        t0 = fr;
      };
      const start = () => {
        w = [
          [0.5, -0.4],
          [-0.3, 0.6],
        ];
        b = [0.1, -0.1];
        v = [0.4, -0.5];
        cc = -0.2;
        ph = 0;
        hist = [];
        upd = 0;
        old = null;
        calc();
        hist.push(V.L);
        WH = [all()];
        msg =
          'The example: cloud cover ' +
          f2(S.x[0]) +
          ', humidity ' +
          f2(S.x[1]) +
          '. What really happened: ' +
          (S.y ? 'it rained' : 'it stayed dry') +
          '. The weights are an arbitrary starting guess. Press Run or One step.';
        show([
          'x₁ = ' + B(f2(S.x[0])) + '   (cloud cover)',
          'x₂ = ' + B(f2(S.x[1])) + '   (humidity)',
          'y  = ' + B(S.y) + '   (what really happened)',
        ]);
      };
      const done = () => (V.L < 5e-4 && ph === 0 && upd > 0) || upd >= 600;
      const gk = () => (ph >= 10 ? 9 : ph >= 9 ? [0, 1, 2, 6, 7, 8] : ph >= 6 ? [6, 7, 8] : []);
      const step = () => {
        calc();
        ph = ph >= 11 ? 1 : ph + 1;
        const x = S.x,
          q = V;
        if (ph === 1 || ph === 2) {
          const j = ph - 1,
            s = j ? '₂' : '₁';
          msg =
            'Hidden neuron ' +
            (j + 1) +
            ' multiplies each input by its weight, adds them up with its bias, and squashes the total with tanh.';
          show([
            'z' + s + ' = w' + s + '₁·x₁ + w' + s + '₂·x₂ + b' + s,
            '   = ' +
              n2(w[j][0]) +
              '·' +
              f2(x[0]) +
              ' + ' +
              n2(w[j][1]) +
              '·' +
              f2(x[1]) +
              ' + ' +
              n2(b[j]),
            '   = ' + B(n3(q.z[j])),
            'h' + s + ' = tanh(z' + s + ') = tanh(' + n3(q.z[j]) + ') = ' + B(n3(q.h[j])),
          ]);
        } else if (ph === 3) {
          msg =
            'The output neuron does the same with the two hidden values, then a sigmoid turns the total into a probability: a ' +
            (100 * q.o).toFixed(0) +
            '% chance of rain.';
          show([
            'z = v₁·h₁ + v₂·h₂ + c',
            '  = ' +
              n2(v[0]) +
              '·' +
              n3(q.h[0]) +
              ' + ' +
              n2(v[1]) +
              '·' +
              n3(q.h[1]) +
              ' + ' +
              n2(cc),
            '  = ' + B(n3(q.zo)),
            'ŷ = 1 / (1 + e^(−z)) = ' + B(f3(q.o)),
          ]);
        } else if (ph === 4) {
          msg =
            'The loss measures how far the answer is from the truth. Everything after this asks: which weights caused it?';
          show(['L = ½ (ŷ − y)²', '  = ½ (' + f3(q.o) + ' − ' + S.y + ')²', '  = ' + B(f3(q.L))]);
        } else if (ph === 5) {
          msg =
            'Blame at the output: the error times the slope of the sigmoid. ' +
            (q.dO < 0
              ? 'It is negative, so the output’s input should go up.'
              : 'It is positive, so the output’s input should go down.');
          show([
            'δ = (ŷ − y) · ŷ · (1 − ŷ)',
            '  = (' + f3(q.o) + ' − ' + S.y + ') · ' + f3(q.o) + ' · ' + f3(1 - q.o),
            '  = ' + B(s4(q.dO)),
          ]);
        } else if (ph === 6) {
          msg =
            'A weight’s gradient is the blame of the neuron it feeds times the value that travelled along it. That gives the three gradients at the output.';
          show([
            '∂L/∂v₁ = δ · h₁ = ' + n3(q.dO) + ' · ' + n3(q.h[0]) + ' = ' + B(s4(q.G[6])),
            '∂L/∂v₂ = δ · h₂ = ' + n3(q.dO) + ' · ' + n3(q.h[1]) + ' = ' + B(s4(q.G[7])),
            '∂L/∂c  = δ = ' + B(s4(q.G[8])),
          ]);
        } else if (ph === 7 || ph === 8) {
          const j = ph - 7,
            s = j ? '₂' : '₁';
          msg =
            'The blame travels back to hidden neuron ' +
            (j + 1) +
            ' along the weight that connects it to the output, and is scaled by the slope of that neuron’s tanh.';
          show([
            'δ' + s + ' = δ · v' + s + ' · (1 − h' + s + '²)',
            '   = ' + n3(q.dO) + ' · ' + n2(v[j]) + ' · (1 − ' + n3(q.h[j]) + '²)',
            '   = ' + n3(q.dO) + ' · ' + n2(v[j]) + ' · ' + f3(1 - q.h[j] ** 2),
            '   = ' + B(s4(q.dH[j])),
          ]);
        } else if (ph === 9 || ph === 10) {
          const j = ph - 9,
            s = j ? '₂' : '₁';
          msg =
            'Blame of hidden neuron ' +
            (j + 1) +
            ' times each input gives the gradients of the three weights that feed it.';
          show([
            '∂L/∂w' +
              s +
              '₁ = δ' +
              s +
              ' · x₁ = ' +
              n3(q.dH[j]) +
              ' · ' +
              f2(x[0]) +
              ' = ' +
              B(s4(q.G[3 * j])),
            '∂L/∂w' +
              s +
              '₂ = δ' +
              s +
              ' · x₂ = ' +
              n3(q.dH[j]) +
              ' · ' +
              f2(x[1]) +
              ' = ' +
              B(s4(q.G[3 * j + 1])),
            '∂L/∂b' + s + '  = δ' + s + ' = ' + B(s4(q.G[3 * j + 2])),
          ]);
        } else {
          const o0 = q.o,
            L0 = q.L,
            A0 = all();
          old = { A: A0, G: q.G.slice() };
          for (let j = 0; j < 2; j++) {
            v[j] -= S.lr * q.G[6 + j];
            b[j] -= S.lr * q.G[3 * j + 2];
            for (let i = 0; i < 2; i++) w[j][i] -= S.lr * q.G[3 * j + i];
          }
          cc -= S.lr * q.G[8];
          upd++;
          calc();
          hist.push(V.L);
          const A1 = all();
          WH.push(A1);
          ph = 0;
          msg =
            'Update ' +
            upd +
            ': every weight moved against its gradient. Running the same example forward again, the answer goes from ' +
            f3(o0) +
            ' to ' +
            f3(V.o) +
            ' and the loss from ' +
            f3(L0) +
            ' to ' +
            f3(V.L) +
            '.' +
            (done()
              ? upd >= 600
                ? ' Reached the limit of 600 updates.'
                : ' Converged: the answer is within a few points of the truth, so it stops here.'
              : '');
          show(
            ['new weight = old weight − η · gradient,   η = ' + f2(S.lr)].concat(
              NM.map(
                (nm, i) =>
                  nm +
                  (nm.length < 3 ? ' ' : '') +
                  ' : ' +
                  n3(A0[i]) +
                  ' − ' +
                  f2(S.lr) +
                  '·' +
                  n3(old.G[i]) +
                  ' = ' +
                  B(n3(A1[i])),
              ),
            ),
          );
        }
      };
      const full = () => {
        const sl = S.slow;
        S.slow = false;
        do {
          step();
        } while (ph !== 0);
        S.slow = sl;
      };
      const draw = () => {
        const t = T(),
          known = gk(),
          kn = (i) => known === 9 || (Array.isArray(known) && known.includes(i));
        {
          const { g, w: W, h } = fit(c),
            P = {
              i: [
                [W * 0.13, h * 0.3],
                [W * 0.13, h * 0.72],
              ],
              h: [
                [W * 0.5, h * 0.25],
                [W * 0.5, h * 0.77],
              ],
              o: [W * 0.87, h * 0.51],
            },
            R = 27,
            u = RM ? 1 : clamp((fr - t0) / 70, 0, 1),
            e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u);
          g.font = MONO;
          const edge = (a, b2, wt, gi, act, dir, pk) => {
            g.strokeStyle = act ? t.ink : mix(t.ink, t.ink, 0, 0.3);
            g.lineWidth = act ? 2.6 : 1.2;
            g.beginPath();
            g.moveTo(a[0], a[1]);
            g.lineTo(b2[0], b2[1]);
            g.stroke();
            const lx = a[0] + (b2[0] - a[0]) * 0.28,
              ly = a[1] + (b2[1] - a[1]) * 0.28;
            g.font = MONO;
            g.fillStyle = t.bg;
            g.fillRect(lx - 21, ly - 9, 42, 17);
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.fillText(f2(wt), lx, ly + 4);
            if (kn(gi)) {
              const gx2 = a[0] + (b2[0] - a[0]) * 0.7,
                gy = a[1] + (b2[1] - a[1]) * 0.7;
              g.fillStyle = t.bg;
              g.fillRect(gx2 - 27, gy - 9, 54, 17);
              g.fillStyle = t.ok;
              g.fillText(s4(V.G[gi]), gx2, gy + 4);
            }
            if (act && u < 1) {
              const q = dir > 0 ? e : 1 - e,
                px = a[0] + (b2[0] - a[0]) * q,
                py = a[1] + (b2[1] - a[1]) * q,
                cl = dir > 0 ? t.gold : t.ok;
              g.font = SANS;
              const tw = g.measureText(pk).width + 14;
              g.fillStyle = cl;
              g.beginPath();
              g.rect(px - tw / 2, py - 11, tw, 22);
              g.fill();
              g.fillStyle = t.bg;
              g.fillText(pk, px, py + 5);
            }
          };
          for (let j = 0; j < 2; j++)
            for (let i = 0; i < 2; i++) {
              const fw = ph === 1 + j,
                bk = ph === 9 + j;
              edge(
                P.i[i],
                P.h[j],
                w[j][i],
                3 * j + i,
                fw || bk,
                bk ? -1 : 1,
                bk ? s4(V.dH[j]) : f2(S.x[i]) + '×' + f2(w[j][i]),
              );
            }
          for (let j = 0; j < 2; j++) {
            const fw = ph === 3,
              bk = ph === 6 || ph === 7 + j;
            edge(
              P.h[j],
              P.o,
              v[j],
              6 + j,
              fw || bk,
              bk ? -1 : 1,
              bk ? s4(V.dO) : f3(V.h[j]) + '×' + f2(v[j]),
            );
          }
          const node = (p, val, lab, on, blame) => {
            g.fillStyle = on ? mix(t.gold, t.bg, 0, 0.9) : t.bg;
            g.strokeStyle = t.ink;
            g.lineWidth = on ? 2.8 : 1.5;
            g.beginPath();
            g.arc(p[0], p[1], on ? R + 3 * Math.sin(Math.PI * u) : R, 0, 7);
            g.fill();
            g.stroke();
            g.font = SANS;
            g.fillStyle = on ? t.bg : t.ink;
            g.textAlign = 'center';
            g.fillText(val, p[0], p[1] + 5);
            g.font = MONO;
            g.fillStyle = t.muted;
            g.fillText(lab, p[0], p[1] - R - 9);
            if (blame != null) {
              g.fillStyle = t.ok;
              g.fillText('blame ' + s4(blame), p[0], p[1] + R + 16);
            }
          };
          const seen = ph === 0 && upd > 0;
          node(P.i[0], f2(S.x[0]), 'cloud cover', ph === 0 && !upd);
          node(P.i[1], f2(S.x[1]), 'humidity', ph === 0 && !upd);
          node(
            P.h[0],
            ph >= 1 || seen ? f3(V.h[0]) : '?',
            'hidden 1',
            (ph === 1 && u > 0.6) || ph === 7,
            ph >= 7 ? V.dH[0] : null,
          );
          node(
            P.h[1],
            ph >= 2 || seen ? f3(V.h[1]) : '?',
            'hidden 2',
            (ph === 2 && u > 0.6) || ph === 8,
            ph >= 8 ? V.dH[1] : null,
          );
          node(
            P.o,
            ph >= 3 || seen ? f3(V.o) : '?',
            'chance of rain',
            (ph === 3 && u > 0.6) || ph === 4 || ph === 5,
            ph >= 5 ? V.dO : null,
          );
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            ph >= 1 && ph <= 4
              ? 'forward pass →'
              : ph >= 5
                ? '← backward pass'
                : seen
                  ? 'weights updated'
                  : '',
            8,
            h - 8,
          );
          g.textAlign = 'right';
          g.fillText('black on lines: weight.  red: its gradient', W - 8, h - 8);
        }
        {
          const { g, w: W, h } = fit(cr),
            o = V.o,
            x0 = 24,
            bw = W - 48,
            y0 = 30;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('the network’s answer (gold) and the truth (black line)', 8, 13);
          g.fillStyle = mix(t.ink, t.ink, 0, 0.1);
          g.fillRect(x0, y0, bw, 20);
          g.fillStyle = t.gold;
          g.fillRect(x0, y0, bw * o, 20);
          g.strokeStyle = t.ink;
          g.lineWidth = 3;
          g.beginPath();
          g.moveTo(x0 + bw * S.y, y0 - 6);
          g.lineTo(x0 + bw * S.y, y0 + 26);
          g.stroke();
          g.fillStyle = t.ink;
          g.textAlign = 'center';
          g.fillText(
            (100 * o).toFixed(0) + '% rain',
            clamp(x0 + bw * o, x0 + 30, x0 + bw - 30),
            y0 + 38,
          );
          const ya = 84,
            ha = h * 0.3,
            rh = (ha - 18) / 9,
            G = ph === 0 && old ? old.G : V.G,
            gm = Math.max(0.02, ...G.map(Math.abs)),
            xc = W * 0.58,
            sx = W * 0.36;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            ph === 0 && old
              ? 'gradients used in the last update'
              : 'gradient of each weight (appears when computed)',
            8,
            ya,
          );
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(xc, ya + 8);
          g.lineTo(xc, ya + 10 + 9 * rh);
          g.stroke();
          for (let i = 0; i < 9; i++) {
            const y = ya + 10 + i * rh,
              on = (ph === 0 && old) || kn(i);
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText(NM[i], xc - sx - 6, y + rh * 0.7);
            if (on) {
              const q = G[i];
              g.fillStyle = q < 0 ? t.accent : t.ok;
              g.fillRect(q < 0 ? xc + (q / gm) * sx : xc, y + 2, (Math.abs(q) / gm) * sx, rh - 4);
              g.fillStyle = t.muted;
              g.textAlign = q < 0 ? 'right' : 'left';
              g.fillText(s4(q), xc + (q / gm) * sx + (q < 0 ? -4 : 4), y + rh * 0.7);
            }
          }
          g.fillStyle = t.accent;
          g.textAlign = 'left';
          g.fillText('← raise this weight', xc - sx, ya + 18 + 9 * rh);
          g.fillStyle = t.ok;
          g.textAlign = 'right';
          g.fillText('lower it →', xc + sx, ya + 18 + 9 * rh);
          const yb = ya + ha + 22,
            hb = (h - yb) / 2;
          uLine(g, t, 0, yb, W, hb, hist, {
            title: 'loss after each update',
            n: Math.max(20, hist.length),
            col: t.ok,
            dots: hist.length < 40,
            ym: Math.max(0.02, hist[0] || 0.1),
          });
          let lo = 1e9,
            hi = -1e9;
          for (const r of WH)
            for (const q of r) {
              lo = Math.min(lo, q);
              hi = Math.max(hi, q);
            }
          const pad = (hi - lo) * 0.08 + 0.05,
            col = UC(t);
          const U = uLine(g, t, 0, yb + hb, W, hb, [], {
            title: 'the nine weights, update by update',
            n: Math.max(20, WH.length),
            y1: lo - pad,
            ym: hi + pad,
          });
          for (let i = 0; i < 9; i++) {
            g.strokeStyle = col[i % 8];
            g.lineWidth = 1.6;
            g.beginPath();
            WH.forEach((r, k) => (k ? g.lineTo(U.X(k), U.Y(r[i])) : g.moveTo(U.X(k), U.Y(r[i]))));
            g.stroke();
          }
        }
        const O = ph === 0 && old ? old : null,
          A = all();
        tab.innerHTML =
          '<table><thead><tr><th>weight</th><th></th><th>' +
          (O ? 'before' : 'value now') +
          '</th><th>gradient</th><th>' +
          (O ? 'after: before − ' + f2(S.lr) + ' × gradient' : '') +
          '</th></tr></thead><tbody>' +
          LB.map(
            (lb, i) =>
              '<tr><td>' +
              NM[i] +
              '</td><td>' +
              lb +
              '</td><td>' +
              f3(O ? O.A[i] : A[i]) +
              '</td><td>' +
              (O ? s4(O.G[i]) : kn(i) ? s4(V.G[i]) : 'not known yet') +
              '</td><td>' +
              (O ? f3(A[i]) : '') +
              '</td></tr>',
          ).join('') +
          '</tbody></table>';
        live.textContent = msg;
        uCur(root, 'bp', ph === 0 ? (upd ? 8 : 0) : [0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7][ph]);
        setR(root, 'o', (100 * V.o).toFixed(1) + '%');
        setR(root, 'y', S.y ? 'It rained (1)' : 'It stayed dry (0)');
        setR(root, 'L', f3(V.L) + (V.L < 0.001 ? ' (' + V.L.toExponential(1) + ')' : ''));
        setR(root, 'u', String(upd));
        setR(root, 'st', done() ? 'Converged' : upd ? 'Learning' : 'Not started');
      };
      const fx = Anim(() => {
        fr++;
        draw();
        if (fr - t0 > 80) fx.stop();
      });
      start();
      const anim = Anim(() => {
        fr++;
        if (S.slow && upd < 2) {
          if (fr - t0 > Math.max(150, nl * 30 + 110)) {
            step();
            if (chk()) {
              draw();
              return;
            }
          }
          draw();
          return;
        }
        if (fr % (upd < 10 ? 10 : 3) === 0) {
          full();
          draw();
          chk();
          return;
        }
      });
      pb = playBtn($('#bp-play', root), anim);
      chk = runCtl(pb, root, 'bp', {
        done,
        restart: () => start(),
        count: () => upd,
        what: 'updates',
      });
      $('#bp-step', root).onclick = () => {
        pb.stop();
        if (done()) start();
        else step();
        fx.start();
      };
      $('#bp-reset', root).onclick = () => {
        pb.stop();
        start();
        draw();
      };
      const re = () => {
        pb.stop();
        start();
        draw();
      };
      bind(root, 'bp-x1', (q) => {
        S.x[0] = q;
        re();
      });
      bind(root, 'bp-x2', (q) => {
        S.x[1] = q;
        re();
      });
      bind(
        root,
        'bp-lr',
        (q) => {
          S.lr = q;
        },
        f2,
      );
      segBind(root, 'bp-y', String(S.y), (q) => {
        S.y = +q;
        re();
      });
      $('#bp-slow', root).onchange = (e) => {
        S.slow = e.target.checked;
      };
      return {
        draw,
        stop: () => {
          pb.stop();
          fx.stop();
        },
        auto: () => {
          start();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'activations',
    tab: 'Activation functions',
    title: 'The bend in every neuron',
    lede: 'After a neuron adds up its inputs, it passes the total through a small curve called the activation function. That curve is what lets a network learn anything other than straight lines. Its slope also decides how well the learning signal survives the trip back through many layers. Send the signal back and watch what each curve does to it.',
    stage: `<div class="row seg" id="af-act" role="group" aria-label="Activation function"><button class="btn" type="button" data-v="sigmoid">Sigmoid</button><button class="btn" type="button" data-v="tanh">tanh</button><button class="btn" type="button" data-v="relu">ReLU</button><button class="btn" type="button" data-v="leaky">Leaky ReLU</button></div>
 <div class="stage"><figure class="fig"><figcaption>The curve (green) and its slope (red). The dot slides along: the slope is how much of the learning signal gets through at that point</figcaption><canvas class="short" id="af-c" role="img" aria-label="Activation function and its derivative"></canvas></figure>
 <figure class="fig"><figcaption>The learning signal travelling backward through a deep network. Each bar is its strength on arriving at that layer, compared with where it started</figcaption><canvas class="short" id="af-r" role="img" aria-label="Gradient strength by layer"></canvas></figure></div>
 ${uHow('af', ['Build a deep stack of layers with random weights and this activation, and send data forward through it.', 'Start the learning signal (the gradient) at the last layer with strength 1.', 'Step back one layer: the signal is multiplied by the weights and by the slope of the activation at every neuron.', 'Keep stepping back. Slopes below 1 shrink it every time.', 'Read the strength that reaches the first layer. That is how fast the first layer can learn.'])}
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="af-play" type="button">Send the signal back</button><button class="btn" id="af-step" type="button">One layer</button><button class="btn" id="af-reset" type="button">Start over</button></div>`)}
 ${grp('The network', slider('af-d', 'Number of layers', 3, 24, 1, 12) + slider('af-w', 'Size of the weights (1 = standard)', 0.4, 2.5, 0.05, 1))}</div>
 <dl class="read">${rd('f', 'Curve at the dot')}${rd('s', 'Slope at the dot', 'hl2')}${rd('mx', 'Largest slope this curve ever has')}${rd('g1', 'Signal reaching the first layer', 'hl')}${rd('pl', 'Average change per layer')}</dl>`,
    tryit:
      'Choose Sigmoid with 12 layers and send the signal back. How much of it reaches the first layer? Try tanh, then ReLU. Now make the network 24 layers deep and compare again. Finally, with Sigmoid, raise the size of the weights to 2.5: does that rescue it? Look at where the dot spends its time on the left picture to see why not.',
    math: String.raw`<h4>Intuition</h4><p>Photocopy a page, then photocopy the copy, and keep going. If each copy keeps only a quarter of the ink, the tenth copy is blank. If each keeps all of it, the tenth is as sharp as the first. Training a deep network sends a learning signal backward through every layer, and at each layer the signal is multiplied by the slope of the activation function. A curve whose slope is always small is a bad photocopier.</p>
 <h4>The four curves</h4>$$\sigma(z)=\frac{1}{1+e^{-z}},\qquad\tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}},\qquad\mathrm{ReLU}(z)=\max(0,z),\qquad\mathrm{LeakyReLU}(z)=\max(0.1z,\;z).$$<p>Their slopes are</p>$$\sigma'(z)=\sigma(z)\big(1-\sigma(z)\big)\le\tfrac14,\qquad\tanh'(z)=1-\tanh^2(z)\le1,\qquad\mathrm{ReLU}'(z)=\begin{cases}1&z>0\\0&z\le0\end{cases},\qquad\mathrm{LeakyReLU}'(z)=\begin{cases}1&z>0\\0.1&z\le0.\end{cases}$$
 <h4>Why the slope matters</h4><p>Backpropagation gives the blame at layer \(l\) as</p>$$\boldsymbol\delta^{(l)}=\big(\mathbf W^{(l+1)\top}\boldsymbol\delta^{(l+1)}\big)\odot\phi'\big(\mathbf z^{(l)}\big).$$<p>Going back through \(k\) layers multiplies by \(k\) weight matrices and \(k\) sets of slopes. With the sigmoid every slope is at most \(\tfrac14\), so unless the weights are large the signal shrinks geometrically: this is the <b>vanishing gradient</b>. The first layers then learn thousands of times more slowly than the last, and a deep sigmoid network barely trains. Making the weights larger does not help, because large inputs push the sigmoid into its flat ends, where the slope is nearly zero (<b>saturation</b>).</p>
 <h4>Why ReLU changed things</h4><p>For positive inputs ReLU has slope exactly 1, so the signal passes through unchanged, and nothing saturates on that side. About half the neurons are off for any given input, and those pass nothing, but the paths that remain open carry the signal at full strength. With suitably scaled weights the overall strength stays roughly constant however deep the network. ReLU is also cheap to compute. Its weakness is the <b>dead neuron</b>: one whose input is negative for every training example has slope 0 everywhere and never updates again. Leaky ReLU keeps a small slope on the negative side so that such a neuron can recover.</p>
 <h4>Choosing one</h4><ul><li><b>Hidden layers:</b> ReLU or a close relative (Leaky ReLU, GELU, SiLU) by default.</li><li><b>tanh:</b> still used where outputs centred on zero matter, as in recurrent networks. It is better than the sigmoid for hidden layers because its slope reaches 1 and its output averages near zero.</li><li><b>Sigmoid:</b> for an output that must be a probability, and for gates. Avoid it in deep hidden layers.</li><li><b>Output layer:</b> chosen by the task, not by the trainability of the network: sigmoid for yes or no, softmax for one of many, none for a number.</li></ul>
 <h4>What this page computes</h4><p>The right-hand picture is a real calculation: a stack of layers 40 neurons wide with random weights of variance (size)²/40, fifty random inputs sent forward, and a random signal sent backward by the formula above. Each bar is the root-mean-square size of the signal at that layer. The Weight Initialization tab shows how the size of the weights should be chosen for each activation.</p>`,
    q: 'With the sigmoid, the gradient shrinks by roughly a factor of four or more per layer. A colleague suggests multiplying the learning rate of the early layers by a large number to compensate. What is wrong with that?',
    a: 'The factor differs from layer to layer, from weight to weight and from one moment in training to the next, so no fixed multiplier is right for long. A gradient that has shrunk by a factor of a million has also lost its information to rounding and noise. Fixing the cause works better: use an activation whose slope is 1 over much of its range, initialise the weights to preserve the signal, and add skip connections or normalisation in very deep networks.',
    init(root) {
      const c = $('#af-c', root),
        cr = $('#af-r', root),
        live = $('#af-live', root),
        S = { act: 'sigmoid', d: 12, w: 1 };
      let F = null,
        shown = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        ph = 0;
      const comp = () => {
        F = sigFlow(S.act, S.d, S.w * S.w, 11);
        shown = 0;
        cu = 0;
        msg =
          'A stack of ' +
          S.d +
          ' layers with ' +
          ACTN[S.act] +
          ' is ready, and data has been sent forward through it. Press Send the signal back or One layer.';
      };
      const rel = (l) => F.gr[l] / F.gr[S.d - 1];
      const step = () => {
        if (shown >= S.d) {
          shown = 0;
          cu = 0;
          return;
        }
        shown++;
        const l = S.d - shown;
        cu = shown === 1 ? 1 : shown >= S.d ? 4 : shown === 2 ? 2 : 3;
        msg =
          shown === 1
            ? 'The signal starts at layer ' + S.d + ', the last one, with strength 1.'
            : 'Layer ' +
              (l + 1) +
              ': going back one layer multiplied the signal by ' +
              sci(F.gr[l] / F.gr[l + 1]) +
              '. It is now ' +
              sci(rel(l)) +
              ' of what it was at the last layer.';
        if (shown >= S.d)
          msg +=
            ' That is what reaches the first layer after ' +
            S.d +
            ' layers: ' +
            (rel(0) < 0.01
              ? 'almost nothing. The first layers of this network would learn extremely slowly.'
              : rel(0) > 100
                ? 'far too much. This network’s first updates would be wildly large.'
                : 'a healthy share. Every layer of this network can learn.');
      };
      const draw = () => {
        const t = T(),
          f = ACT[S.act][0],
          df = ACT[S.act][1],
          z = 4.6 * Math.sin(ph);
        {
          const { g, w, h } = fit(c),
            P = plot(
              g,
              w,
              h,
              [-5, 5],
              [-1.3, S.act === 'relu' || S.act === 'leaky' ? 3.2 : 1.3],
              t,
              { xstep: 1, yticks: [-1, 0, 1], xlabel: 'total input z' },
            );
          clipPlot(g, P);
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(P.X(0), P.Y(-2));
          g.lineTo(P.X(0), P.Y(4));
          g.stroke();
          curve(g, P, df, -5, 5, 400);
          g.strokeStyle = t.ok;
          g.lineWidth = 2.2;
          g.stroke();
          curve(g, P, f, -5, 5, 400);
          g.strokeStyle = t.accent;
          g.lineWidth = 3;
          g.stroke();
          const s = df(z);
          g.strokeStyle = t.gold;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(P.X(z - 1.1), P.Y(f(z) - 1.1 * s));
          g.lineTo(P.X(z + 1.1), P.Y(f(z) + 1.1 * s));
          g.stroke();
          g.setLineDash([3, 3]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(P.X(z), P.Y(f(z)));
          g.lineTo(P.X(z), P.Y(s));
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.accent;
          g.beginPath();
          g.arc(P.X(z), P.Y(f(z)), 6.5, 0, 7);
          g.fill();
          g.fillStyle = t.ok;
          g.beginPath();
          g.arc(P.X(z), P.Y(s), 5, 0, 7);
          g.fill();
          g.restore();
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.accent;
          g.fillText('green: ' + ACTN[S.act] + '(z)', P.m.l + 8, P.m.t + 8);
          g.fillStyle = t.ok;
          g.fillText('red: its slope = share of the signal let through', P.m.l + 8, P.m.t + 22);
          g.fillStyle = t.cut;
          g.fillText('gold: the tangent at the dot', P.m.l + 8, P.m.t + 36);
        }
        {
          const { g, w, h } = fit(cr),
            m = { l: 54, r: 12, t: 20, b: 36 },
            D = S.d,
            bw = (w - m.l - m.r) / D;
          let lo = -6,
            hi = 1;
          for (let l = 0; l < D; l++) {
            const v = Math.log10(rel(l) + 1e-300);
            lo = Math.min(lo, Math.floor(v));
            hi = Math.max(hi, Math.ceil(v));
          }
          lo = Math.max(lo, -14);
          const Y = (v) =>
            h - m.b - ((clamp(Math.log10(v + 1e-300), lo, hi) - lo) / (hi - lo)) * (h - m.t - m.b);
          g.font = MONO;
          g.textAlign = 'right';
          for (let e = lo; e <= hi; e += Math.max(1, Math.round((hi - lo) / 7))) {
            g.strokeStyle = e === 0 ? t.muted : t.line;
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(m.l, Y(10 ** e));
            g.lineTo(w - m.r, Y(10 ** e));
            g.stroke();
            g.fillStyle = t.muted;
            g.fillText(e === 0 ? '1' : '10^' + e, m.l - 5, Y(10 ** e) + 4);
          }
          for (let l = 0; l < D; l++) {
            const on = l >= D - shown,
              x = m.l + l * bw + bw * 0.14,
              v = rel(l),
              y1 = Y(1),
              y2 = Y(v);
            g.fillStyle = on
              ? l === D - shown
                ? t.gold
                : v < 0.01
                  ? t.ok
                  : t.accent
              : mix(t.ink, t.ink, 0, 0.07);
            if (on) g.fillRect(x, Math.min(y1, y2), bw * 0.72, Math.max(3, Math.abs(y2 - y1)));
            else g.fillRect(x, y1 - 2, bw * 0.72, 4);
            if (D <= 16 || l % 2 === 0) {
              g.fillStyle = t.muted;
              g.textAlign = 'center';
              g.fillText(String(l + 1), x + bw * 0.36, h - m.b + 14);
            }
          }
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('first layer', m.l, h - 5);
          g.textAlign = 'right';
          g.fillText('last layer: the signal starts here ←', w - m.r, h - 5);
          g.textAlign = 'left';
          g.fillText('strength compared with the start (each line is ×10)', m.l, 12);
        }
        live.textContent = msg;
        uCur(root, 'af', cu);
        setR(root, 'f', f3(f(z)));
        setR(root, 's', f3(df(z)));
        setR(root, 'mx', S.act === 'sigmoid' ? '0.25' : '1');
        setR(
          root,
          'g1',
          shown >= S.d ? sci(rel(0)) + ' of the start' : 'send the signal back to see',
        );
        setR(root, 'pl', '× ' + sci(Math.pow(rel(0), 1 / Math.max(1, S.d - 1))));
      };
      comp();
      shown = S.d;
      const anim = Anim(() => {
        fr++;
        ph += 0.02;
        if (fr % 34 === 0 && shown < S.d) step();
        draw();
        if (shown >= S.d && fr % 34 === 0) pb.stop();
      });
      pb = playBtn($('#af-play', root), anim);
      runCtl(pb, root, 'af', {
        done: () => shown >= S.d,
        restart: () => {
          shown = 0;
        },
      });
      $('#af-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#af-reset', root).onclick = () => {
        pb.stop();
        comp();
        draw();
      };
      segBind(root, 'af-act', S.act, (v) => {
        S.act = v;
        pb.stop();
        comp();
        draw();
        pb.start();
      });
      bind(
        root,
        'af-d',
        (v) => {
          S.d = v;
          pb.stop();
          comp();
          shown = S.d;
          draw();
        },
        (v) => String(v),
      );
      bind(root, 'af-w', (v) => {
        S.w = v;
        pb.stop();
        comp();
        shown = S.d;
        draw();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          comp();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'init',
    tab: 'Weight initialization',
    title: 'Where to start: the first weights decide whether it can learn',
    lede: 'Before training, every weight is set to a small random number. How small matters a great deal. Too small, and the signal fades to nothing on its way through the layers. Too large, and it blows up or jams every neuron at its limit. Send a signal through ten layers and see which starting scale keeps it alive.',
    stage: `<div class="row seg" id="wi-mode" role="group" aria-label="Starting scale of the weights"><button class="btn" type="button" data-v="small">Too Small</button><button class="btn" type="button" data-v="xavier">Xavier</button><button class="btn" type="button" data-v="he">He</button><button class="btn" type="button" data-v="large">Too Large</button></div>
 <div class="stage one"><figure class="fig"><figcaption>What the neurons in each layer output, as a histogram, for 50 random inputs. The signal enters on the left</figcaption><canvas class="wide" id="wi-c" role="img" aria-label="Histograms of neuron outputs in each layer"></canvas></figure></div>
 <div class="stage" style="margin-top:20px"><figure class="fig"><figcaption>Forward: spread of the outputs in each layer</figcaption><canvas class="tall" id="wi-f" role="img" aria-label="Standard deviation of activations by layer"></canvas></figure>
 <figure class="fig"><figcaption>Backward: strength of the learning signal reaching each layer</figcaption><canvas class="tall" id="wi-b" role="img" aria-label="Gradient strength by layer"></canvas></figure></div>
 ${uHow('wi', ['Draw every weight at random with the chosen spread.', 'Send 50 random inputs into the first layer.', 'Forward, one layer at a time: each neuron adds up its weighted inputs and applies the activation. Watch the histogram.', 'At the last layer, start a learning signal and send it backward, one layer at a time.', 'Compare the first layer with the last: a healthy start keeps both the outputs and the learning signal about the same size everywhere.'])}
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="wi-play" type="button">Send the signal through</button><button class="btn" id="wi-step" type="button">One layer</button><button class="btn" id="wi-reset" type="button">Start over</button></div>`)}
 ${grp('The network', `<label class="ctl" for="wi-act"><span>Activation</span><select id="wi-act"><option value="tanh">tanh</option><option value="relu">ReLU</option><option value="sigmoid">Sigmoid</option></select></label>` + slider('wi-v', 'Variance of the weights × number of inputs', 0.05, 8, 0.05, 1) + slider('wi-d', 'Number of layers', 4, 16, 1, 10))}</div>
 <dl class="read">${rd('v', 'Weight variance × inputs')}${rd('s1', 'Spread of outputs, first layer')}${rd('sL', 'Spread of outputs, last layer', 'hl')}${rd('g', 'Learning signal at the first layer, relative to the last', 'hl2')}${rd('vd', 'Verdict')}</dl>`,
    tryit:
      'With tanh, press Too Small and watch the histograms collapse to a spike at zero. Press Too Large: every neuron is jammed at −1 or +1. Xavier keeps a healthy spread. Now switch the activation to ReLU and press Xavier again: the signal fades, because ReLU throws away half of it at every layer. Press He. What number did He change, and why that number?',
    math: String.raw`<h4>Intuition</h4><p>Line up ten people to pass a message by whispering. If each person speaks a little more quietly than they heard, the last one hears nothing. If each speaks a little louder, the last one is being shouted at and cannot make out a word. The message survives only if each person repeats it at the volume they heard it. The layers of a network are that line of people, the starting weights set each one's volume, and the goal of a good initialisation is to make every layer pass the signal on at the strength it arrived.</p>
 <h4>How the spread changes from layer to layer</h4><p>Take a neuron with \(n\) inputs, each of variance \(\operatorname{Var}(a)\), and weights drawn independently with mean 0 and variance \(\operatorname{Var}(w)\). Its total input \(z=\sum_i w_ia_i\) has</p>$$\operatorname{Var}(z)=n\,\operatorname{Var}(w)\,\mathbb E[a^2].$$<p>So each layer multiplies the size of the signal by the factor \(n\operatorname{Var}(w)\), adjusted for what the activation does. Across \(L\) layers the factors multiply. A factor of 0.5 per layer leaves \(0.5^{10}\approx0.001\) after ten layers; a factor of 2 gives about 1000. Only a factor of 1 is stable.</p>
 <h4>Xavier (Glorot) initialisation</h4><p>Near zero, \(\tanh(z)\approx z\), so the activation passes the signal through nearly unchanged and the factor is \(n\operatorname{Var}(w)\). Setting it to 1 gives</p>$$\operatorname{Var}(w)=\frac{1}{n_{\text{in}}}\qquad\text{or, balancing forward and backward,}\qquad\operatorname{Var}(w)=\frac{2}{n_{\text{in}}+n_{\text{out}}} .$$
 <h4>He (Kaiming) initialisation</h4><p>ReLU sets every negative input to zero. For inputs symmetric around zero that removes half of \(\mathbb E[a^2]\), so the factor is \(\tfrac12 n\operatorname{Var}(w)\). Setting that to 1 gives</p>$$\operatorname{Var}(w)=\frac{2}{n_{\text{in}}} .$$<p>The slider on this page is \(n\operatorname{Var}(w)\): Xavier is 1 and He is 2.</p>
 <h4>The backward direction</h4><p>The learning signal obeys the same arithmetic going the other way: at each layer it is multiplied by the weights and by the slope of the activation. If the forward signal vanishes, so does the gradient (<b>vanishing gradients</b>), and the early layers do not learn. If it explodes, the first update throws the weights far away (<b>exploding gradients</b>). With tanh and weights that are too large there is a third failure: the forward signal looks fine in size, because tanh is bounded, but every neuron sits at −1 or +1 where the slope is nearly zero, so the gradient still dies. The histograms piled up at the two ends are the sign of that.</p>
 <h4>Two things that are never done</h4><p>Starting all weights at zero, or at the same value, makes every neuron in a layer compute the same thing and receive the same gradient, so they stay identical for ever: the layer behaves as one neuron. Randomness is needed to <b>break the symmetry</b>. Biases, on the other hand, are usually started at zero.</p>
 <h4>In practice</h4><p>Libraries pick He for ReLU-type layers and Xavier for tanh by default. Batch normalisation and residual connections make deep networks far less sensitive to the starting scale, by rescaling the signal at every layer or by giving it a path around the layers.</p>`,
    q: 'A ten-layer ReLU network is initialised with Xavier variance 1/n instead of He variance 2/n. Roughly how much smaller is the signal at the last layer than it should be?',
    a: 'With ReLU each layer multiplies the variance of the signal by one half of n × Var(w). With He that is ½ × 2 = 1. With Xavier it is ½ × 1 = ½. After ten layers the variance is (½)¹⁰, about a thousandth, so the typical size of the signal (the standard deviation) is about 1/32 of what it should be. The gradient is reduced by a similar factor on the way back.',
    init(root) {
      const c = $('#wi-c', root),
        cf = $('#wi-f', root),
        cb = $('#wi-b', root),
        live = $('#wi-live', root),
        S = { act: 'tanh', v: 1, d: 10 },
        MV = { small: 0.2, xavier: 1, he: 2, large: 6 };
      let F = null,
        k = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb;
      const comp = () => {
        F = sigFlow(S.act, S.d, S.v, 7);
        k = 0;
        cu = 0;
        msg =
          'Weights drawn at random with variance ' +
          f2(S.v) +
          ' ÷ (number of inputs). Nothing has been sent through yet. Press Send the signal through or One layer.';
      };
      const end = () => 2 * S.d,
        rel = (l) => F.gr[l] / F.gr[S.d - 1];
      const verdict = () => {
        const a = F.sd[S.d - 1],
          b = rel(0),
          sat = S.act !== 'relu' && a > 0.85 * (S.act === 'sigmoid' ? 0.5 : 1);
        return a < 0.02
          ? 'The signal has faded away'
          : a > 30
            ? 'The signal has exploded'
            : sat
              ? 'Neurons are jammed at their limits (saturated)'
              : b < 0.01
                ? 'Outputs look fine but the learning signal dies'
                : b > 100
                  ? 'The learning signal explodes'
                  : 'Healthy: the signal survives in both directions';
      };
      const step = () => {
        if (k >= end()) {
          k = 0;
          cu = 0;
          msg = 'Start again.';
          return;
        }
        k++;
        if (k <= S.d) {
          const l = k - 1;
          cu = k === 1 ? 1 : 2;
          msg =
            'Forward, layer ' +
            k +
            ': the outputs have spread ' +
            sci(F.sd[l]) +
            (l
              ? ' (the layer before had ' +
                sci(F.sd[l - 1]) +
                ', so this layer multiplied it by ' +
                f2(F.sd[l] / F.sd[l - 1]) +
                ')'
              : '') +
            '.';
        } else {
          const l = 2 * S.d - k;
          cu = k >= end() ? 4 : 3;
          msg =
            'Backward, layer ' +
            (l + 1) +
            ': the learning signal is ' +
            sci(rel(l)) +
            ' of its strength at the last layer.';
          if (k >= end()) msg += ' Done. Verdict: ' + verdict() + '.';
        }
      };
      const draw = () => {
        const t = T(),
          D = S.d,
          nf = Math.min(k, D),
          nbk = Math.max(0, k - D);
        {
          const { g, w, h } = fit(c),
            cw = w / D,
            bounded = S.act !== 'relu',
            lo = S.act === 'sigmoid' ? 0 : bounded ? -1 : 0;
          g.font = MONO;
          for (let l = 0; l < D; l++) {
            const x0 = l * cw + 6,
              ww = cw - 12,
              y1 = h - 30,
              on = l < nf;
            g.strokeStyle = t.line;
            g.lineWidth = 1;
            g.strokeRect(x0, 14, ww, y1 - 14);
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText('layer ' + (l + 1), x0 + ww / 2, h - 14);
            if (!on) continue;
            const a = F.acts[l],
              nb = 22;
            let hi = 1;
            if (!bounded) {
              hi = 1e-9;
              for (const v of a) hi = Math.max(hi, v);
              hi = Math.max(hi, 1e-6);
            }
            const cnt = new Array(nb).fill(0);
            for (const v of a) cnt[clamp(Math.floor(((v - lo) / (hi - lo)) * nb), 0, nb - 1)]++;
            const cm = Math.max(...cnt);
            g.fillStyle = l === nf - 1 && k <= D ? t.gold : t.accent;
            for (let q = 0; q < nb; q++) {
              const bh = (cnt[q] / cm) * (y1 - 24);
              g.fillRect(x0 + (q * ww) / nb, y1 - bh, ww / nb - 0.6, bh);
            }
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText(bounded ? String(lo) : '0', x0, h - 2);
            g.textAlign = 'right';
            g.fillText(bounded ? '1' : sci(hi), x0 + ww, h - 2);
          }
        }
        const bars = (cv, vals, n, from, ttl, ref) => {
          const { g, w, h } = fit(cv),
            m = { l: 54, r: 10, t: 20, b: 24 },
            bw = (w - m.l - m.r) / D;
          let lo = -3,
            hi = 1;
          vals.forEach((v) => {
            const q = Math.log10(v + 1e-300);
            lo = Math.min(lo, Math.floor(q));
            hi = Math.max(hi, Math.ceil(q));
          });
          lo = Math.max(lo, -14);
          hi = Math.min(hi, 14);
          const Y = (v) =>
            h - m.b - ((clamp(Math.log10(v + 1e-300), lo, hi) - lo) / (hi - lo)) * (h - m.t - m.b);
          g.font = MONO;
          for (let e = lo; e <= hi; e += Math.max(1, Math.round((hi - lo) / 6))) {
            g.strokeStyle = e === 0 ? t.muted : t.line;
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(m.l, Y(10 ** e));
            g.lineTo(w - m.r, Y(10 ** e));
            g.stroke();
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText(e === 0 ? '1' : '10^' + e, m.l - 5, Y(10 ** e) + 4);
          }
          for (let l = 0; l < D; l++) {
            const on = from === 'L' ? l < n : l >= D - n,
              x = m.l + l * bw + bw * 0.15;
            g.fillStyle = on ? (ref ? t.ok : t.accent) : mix(t.ink, t.ink, 0, 0.07);
            const y1 = Y(1),
              y2 = on ? Y(vals[l]) : y1;
            g.fillRect(
              x,
              Math.min(y1, y2) - (on ? 0 : 2),
              bw * 0.7,
              Math.max(4, Math.abs(y2 - y1)),
            );
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(String(l + 1), x + bw * 0.35, h - m.b + 14);
          }
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(ttl, m.l, 12);
        };
        bars(cf, F.sd, nf, 'L', 'spread of the outputs (each line is ×10) →');
        bars(
          cb,
          F.gr.map((_, l) => rel(l)),
          nbk,
          'R',
          '← learning signal, relative to the last layer',
          true,
        );
        live.textContent = msg;
        uCur(root, 'wi', cu);
        setR(root, 'v', f2(S.v));
        setR(root, 's1', sci(F.sd[0]));
        setR(root, 'sL', nf >= D ? sci(F.sd[D - 1]) : '–');
        setR(root, 'g', k >= end() ? sci(rel(0)) : '–');
        setR(root, 'vd', k >= end() ? verdict() : '–');
        root
          .querySelectorAll('#wi-mode button')
          .forEach((b) =>
            b.setAttribute(
              'aria-pressed',
              Math.abs(MV[b.dataset.v] - S.v) < 0.01 ? 'true' : 'false',
            ),
          );
      };
      comp();
      k = end();
      const anim = Anim(() => {
        fr++;
        if (fr % 26) return;
        step();
        draw();
        if (k >= end()) pb.stop();
      });
      pb = playBtn($('#wi-play', root), anim);
      runCtl(pb, root, 'wi', {
        done: () => k >= end(),
        restart: () => {
          k = 0;
        },
      });
      $('#wi-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#wi-reset', root).onclick = () => {
        pb.stop();
        comp();
        draw();
      };
      const vb = bind(root, 'wi-v', (v) => {
        S.v = v;
        pb.stop();
        comp();
        k = end();
        draw();
      });
      bind(
        root,
        'wi-d',
        (v) => {
          S.d = v;
          pb.stop();
          comp();
          k = end();
          draw();
        },
        (v) => String(v),
      );
      $('#wi-act', root).onchange = (e) => {
        S.act = e.target.value;
        pb.stop();
        comp();
        draw();
        pb.start();
      };
      $('#wi-mode', root).addEventListener('click', (e) => {
        const v = e.target.dataset && e.target.dataset.v;
        if (!v) return;
        S.v = MV[v];
        vb.set(S.v);
        pb.stop();
        comp();
        draw();
        pb.start();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          comp();
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'dropout',
    tab: 'Dropout',
    title: 'Dropout: train with half the team missing',
    lede: 'A large network given a small, noisy set of examples will memorise them, mistakes included. Dropout is a simple cure: during every training step, switch off a random share of the neurons. No neuron can rely on any other being there, so the network has to learn the pattern in many overlapping ways. Two identical networks train here side by side, one with dropout and one without.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Left half: trained normally. Right half: trained with dropout. Same data, same starting weights. Dots are the training points; a few carry the wrong colour on purpose</figcaption><canvas class="wide" id="dr-c" role="img" aria-label="Decision regions without and with dropout"></canvas></figure>
 <figure class="fig"><figcaption>Top: the dropout network on this step; crossed-out neurons are switched off. Bottom: error on new data for both networks</figcaption><canvas class="short" id="dr-r" role="img" aria-label="Network with dropped neurons and the test loss curves"></canvas></figure></div>
 ${uHow('dr', ['Give two identical networks the same small, noisy training set.', 'Normal training: every neuron takes part in every step.', 'Dropout training: before each example, switch off each hidden neuron with probability p, and scale up the ones that remain.', 'Both networks update their weights. Repeat for many epochs.', 'To predict, switch every neuron back on. Compare the two networks on data they have not seen.'])}
 <div class="controls">${uBtns('dr', 'Train both', 'One epoch', 'epochs')}
 ${grp('The model', slider('dr-p', 'Share of neurons switched off, p', 0, 0.8, 0.05, 0.5) + slider('dr-hw', 'Neurons in each of the two hidden layers', 4, 32, 1, 24))}
 ${grp('The data', slider('dr-n', 'Training points', 20, 200, 2, 60) + slider('dr-fl', 'Share of training points with the wrong label', 0, 0.3, 0.01, 0.12) + `<div class="row"><button class="btn" id="dr-new" type="button">New random sample</button></div>`)}</div>
 <dl class="read">${rd('ep', 'Epochs')}${rd('a0', 'Normal: accuracy on training / new data', 'hl2')}${rd('a1', 'Dropout: accuracy on training / new data', 'hl3')}${rd('l0', 'Normal: loss on new data')}${rd('l1', 'Dropout: loss on new data', 'hl')}</dl>`,
    tryit:
      'Run it and watch the left half: after a while it grows small islands around the wrongly labelled points, and the red curve on the right turns upward. The right half stays smooth. Set p to 0 and train again: the two halves now agree. Set p to 0.8: is more always better? Then raise the training points to 200. Does dropout still matter as much?',
    math: String.raw`<h4>Intuition</h4><p>Imagine a team where one brilliant member knows everything and the rest have learned to pass every question to her. The team does well until the day she is away. Now imagine a manager who sends a random half of the team home each day. Nobody can depend on anyone in particular, so everyone has to become competent, and knowledge ends up spread across the team with plenty of overlap. On the day that matters, everyone comes in. That is dropout: the team is the layer, the members are neurons, and the days with absences are training steps.</p>
 <h4>What it does during training</h4><p>For each training example and each hidden neuron, draw a mask \(m_j\) that is 0 with probability \(p\) and 1 otherwise, and replace the neuron's output \(a_j\) by</p>$$\tilde a_j=\frac{m_j}{1-p}\,a_j .$$<p>The division by \(1-p\) (called inverted dropout) keeps the expected output the same, \(\mathbb E[\tilde a_j]=a_j\), so the next layer sees inputs of the usual size. Backpropagation uses the same mask: a neuron that was off receives no gradient on that step.</p>
 <h4>What it does at prediction time</h4><p>Nothing. All neurons are on and no scaling is needed, because the scaling was already applied in training. The decision regions drawn on this page are the prediction-time network.</p>
 <h4>Why it works</h4><p>Three ways to see it, all saying the same thing.</p><ul><li><b>No co-adaptation.</b> A neuron cannot rely on a specific partner, because that partner is absent a share \(p\) of the time. Fragile arrangements of neurons that fit a single odd training point do not survive.</li><li><b>An ensemble for the price of one.</b> Each mask defines a different thinned network. With \(H\) hidden neurons there are \(2^H\) of them, all sharing weights. Training with dropout trains them all a little, and predicting with every neuron on approximates averaging their answers. Averaging many models reduces variance, as bagging did.</li><li><b>Noise as a penalty.</b> Adding noise to the hidden layer makes solutions that depend on large, finely balanced weights perform badly, which pushes the network toward smaller, more evenly spread weights, much as an L2 penalty does.</li></ul>
 <h4>The cost</h4><p>Dropout makes the training loss noisier and higher, and training takes longer. If \(p\) is too large the network does not have enough capacity left on each step and underfits. Typical values are 0.5 for large fully connected layers and 0.1 to 0.3 elsewhere. With plenty of data there is little overfitting to cure and dropout helps less.</p>
 <h4>Batch normalisation, briefly</h4><p>Batch normalisation is a different technique that is often mentioned alongside dropout. It rescales the inputs of each layer to mean 0 and variance 1 over the current mini-batch, then lets the layer learn its own scale and shift. Its main effect is to make deep networks train faster and more reliably, and the noise from using batch statistics also regularises a little. It does not have its own animation on this site yet.</p>`,
    q: 'With dropout the training loss is higher than without, yet the test loss is lower. How can a model that fits the training data worse be the better model?',
    a: 'The training set here contains wrongly labelled points. Fitting them perfectly means bending the boundary around each one, and those bends are wrong for new data. Dropout makes such narrow, single-point fits unstable, because the few neurons that would encode them are often switched off, so the network keeps only the broad pattern that many neurons agree on. Its higher training loss is mostly the mislabelled points it declined to memorise.',
    init(root) {
      const c = $('#dr-c', root),
        cr = $('#dr-r', root),
        live = $('#dr-live', root),
        S = { p: 0.5, hw: 24, n: 60, fl: 0.12, seed: 2 },
        EPM = 500;
      let tr = [],
        te = [],
        N0,
        N1,
        ep = 0,
        H = [[], [], [], []],
        mask = null,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1;
      const gen = () => {
        const a = nnData('moons', Math.round((S.n * 4) / 3) + 2, 0.28, S.seed)
            .filter((p) => !p.test)
            .slice(0, S.n),
          r = rng(S.seed * 5 + 1);
        tr = a.map((p) => ({ ...p, bad: r() < S.fl }));
        tr.forEach((p) => {
          if (p.bad) p.c = 1 - p.c;
        });
        te = nnData('moons', 400, 0.28, S.seed + 77);
      };
      const build = () => {
        const sz = [2, S.hw, S.hw, 1],
          sd = S.seed * 3 + run * 17;
        N0 = NNet(sz, { act: 'relu', seed: sd });
        N1 = NNet(sz, { act: 'relu', seed: sd });
        ep = 0;
        H = [[], [], [], []];
        mask = null;
        cu = 0;
        msg =
          'Two networks with the same ' +
          (2 * S.hw + S.hw * S.hw + S.hw + 2 * S.hw + 1) +
          ' starting weights and the same ' +
          tr.length +
          ' training points, ' +
          tr.filter((p) => p.bad).length +
          ' of which carry the wrong label. Press Train both or One epoch.';
      };
      const ll = (N, P) =>
          -P.reduce((s, p) => {
            const o = N.p([p.x, p.y]);
            return s + (p.c ? Math.log(o + 1e-9) : Math.log(1 - o + 1e-9));
          }, 0) / P.length,
        ac = (N, P) => P.filter((p) => (N.p([p.x, p.y]) >= 0.5 ? 1 : 0) === p.c).length / P.length;
      const epoch = () => {
        const X = tr.map((p) => [p.x, p.y]),
          Y = tr.map((p) => p.c),
          a = N0.train(X, Y, 0.02),
          b = N1.train(X, Y, 0.02, { drop: S.p });
        mask = b.mask;
        ep++;
        H[0].push(a.loss);
        H[1].push(b.loss);
        H[2].push(ll(N0, te));
        H[3].push(ll(N1, te));
        cu = ep < 3 ? 2 : 3;
        const off = mask ? mask.reduce((s, m) => s + m.filter((v) => v === 0).length, 0) : 0;
        msg =
          'Epoch ' +
          ep +
          ': on the last example ' +
          off +
          ' of the ' +
          2 * S.hw +
          ' hidden neurons in the dropout network were switched off. Loss on new data: normal ' +
          f3(H[2][ep - 1]) +
          ', dropout ' +
          f3(H[3][ep - 1]) +
          '.' +
          (ep >= EPM
            ? ' Both have had their ' +
              EPM +
              ' epochs, so training stops. ' +
              (H[3][ep - 1] < H[2][ep - 1]
                ? 'Dropout ends with the lower error on new data.'
                : 'Here dropout did not help: there was little overfitting to cure, or p is too large.')
            : '');
        if (ep >= EPM) cu = 4;
      };
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c),
            hw = w / 2,
            cs = 7;
          [N0, N1].forEach((N, q) => {
            const x0 = q * hw,
              X = (x) => x0 + hw / 2 + (x * hw) / 6.4,
              Y = (y) => h / 2 - (y * h) / 6.4;
            g.save();
            g.beginPath();
            g.rect(x0, 0, hw, h);
            g.clip();
            for (let py = 0; py < h; py += cs)
              for (let px = 0; px < hw; px += cs) {
                const p = N.p([
                    ((px + cs / 2 - hw / 2) * 6.4) / hw,
                    (-(py + cs / 2 - h / 2) * 6.4) / h,
                  ]),
                  s = Math.abs(p - 0.5) * 2;
                g.fillStyle = mix(p < 0.5 ? t.accent : t.gold, t.bg, 0, 0.12 + 0.42 * s);
                g.fillRect(x0 + px, py, cs, cs);
              }
            for (const p of tr) {
              g.beginPath();
              g.arc(X(p.x), Y(p.y), 3.6, 0, 7);
              g.fillStyle = p.c ? t.gold : t.accent;
              g.fill();
              g.strokeStyle = p.bad ? t.ink : t.bg;
              g.lineWidth = p.bad ? 2 : 0.8;
              g.stroke();
            }
            g.restore();
            g.font = SANS;
            g.fillStyle = t.ink;
            g.textAlign = 'left';
            g.fillText(q ? 'With dropout, p = ' + f2(S.p) : 'Without dropout', x0 + 10, 18);
          });
          g.strokeStyle = t.ink;
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(hw, 0);
          g.lineTo(hw, h);
          g.stroke();
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('black ring = wrong label', w - 8, h - 8);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5,
            sh = Math.min(10, S.hw),
            pos = netPos([2, sh, sh, 1], 30, 4, w - 60, h1 - 20);
          drawEdges(
            g,
            [
              N1.W[0].slice(0, sh),
              N1.W[1].slice(0, sh).map((q) => q.slice(0, sh)),
              [N1.W[2][0].slice(0, sh)],
            ],
            pos,
            t,
            pb && pb.on ? fr / 45 : null,
          );
          pos.forEach((col, l) =>
            col.forEach((q, j) => {
              const off = mask && l > 0 && l < 3 && mask[l - 1][j] === 0;
              g.fillStyle = off ? t.bg : t.ink;
              g.strokeStyle = off ? t.muted : t.ink;
              g.lineWidth = 1.4;
              g.beginPath();
              g.arc(q[0], q[1], 6.5, 0, 7);
              g.fill();
              g.stroke();
              if (off) {
                g.strokeStyle = t.ok;
                g.lineWidth = 2;
                g.beginPath();
                g.moveTo(q[0] - 6, q[1] - 6);
                g.lineTo(q[0] + 6, q[1] + 6);
                g.moveTo(q[0] + 6, q[1] - 6);
                g.lineTo(q[0] - 6, q[1] + 6);
                g.stroke();
              }
            }),
          );
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText(
            S.hw > sh
              ? 'first ' + sh + ' of ' + S.hw + ' neurons in each hidden layer'
              : 'the dropout network',
            w / 2,
            h1 + 2,
          );
          const ym = 1.6,
            U = uLine(
              g,
              t,
              0,
              h1 + 8,
              w,
              h - h1 - 8,
              H[2].map((v) => Math.min(v, ym)),
              {
                title:
                  'loss on new data: normal (red), dropout (green). Dashed: on the training points',
                n: EPM,
                ym,
                col: t.ok,
              },
            );
          g.strokeStyle = t.accent;
          g.lineWidth = 2.4;
          g.beginPath();
          H[3].forEach((v, i) =>
            i ? g.lineTo(U.X(i), U.Y(Math.min(v, ym))) : g.moveTo(U.X(i), U.Y(Math.min(v, ym))),
          );
          g.stroke();
          [
            [H[0], t.ok],
            [H[1], t.accent],
          ].forEach((q) => {
            g.strokeStyle = mix(q[1], q[1], 0, 0.4);
            g.lineWidth = 1;
            g.setLineDash([3, 3]);
            g.beginPath();
            q[0].forEach((v, i) =>
              i ? g.lineTo(U.X(i), U.Y(Math.min(v, ym))) : g.moveTo(U.X(i), U.Y(Math.min(v, ym))),
            );
            g.stroke();
            g.setLineDash([]);
          });
        }
        live.textContent = msg;
        uCur(root, 'dr', cu);
        setR(root, 'ep', ep + ' of ' + EPM);
        setR(root, 'a0', pct(ac(N0, tr)) + ' / ' + pct(ac(N0, te)));
        setR(root, 'a1', pct(ac(N1, tr)) + ' / ' + pct(ac(N1, te)));
        setR(root, 'l0', ep ? f3(H[2][ep - 1]) : '–');
        setR(root, 'l1', ep ? f3(H[3][ep - 1]) : '–');
      };
      gen();
      build();
      if (RM) for (let i = 0; i < EPM; i++) epoch();
      const redo = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      const anim = Anim(() => {
        fr++;
        if (ep < 12 && fr % 8) return;
        for (let i = 0; i < (ep < 12 ? 1 : 2) && ep < EPM; i++) epoch();
        draw();
        chk();
      });
      pb = playBtn($('#dr-play', root), anim);
      chk = runCtl(pb, root, 'dr', {
        done: () => ep >= EPM,
        restart: () => {
          run++;
          build();
        },
        count: () => ep,
        what: 'epochs',
      });
      $('#dr-step', root).onclick = () => {
        pb.stop();
        if (ep >= EPM) {
          run++;
          build();
        } else epoch();
        draw();
      };
      $('#dr-reset', root).onclick = redo;
      bind(root, 'dr-p', (v) => {
        S.p = v;
        redo();
      });
      bind(
        root,
        'dr-hw',
        (v) => {
          S.hw = v;
          redo();
        },
        (v) => String(v),
      );
      bind(
        root,
        'dr-n',
        (v) => {
          S.n = v;
          gen();
          redo();
        },
        (v) => String(v),
      );
      bind(root, 'dr-fl', (v) => {
        S.fl = v;
        gen();
        redo();
      });
      $('#dr-new', root).onclick = () => {
        S.seed++;
        gen();
        redo();
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
    id: 'mcdropout',
    tab: 'Bayesian dropout',
    title: 'Bayesian dropout: a network that knows when it is guessing',
    lede: 'An ordinary network gives one answer and sounds equally sure everywhere, even far from anything it was trained on. A small change fixes that: leave dropout switched on when predicting, and ask the same question many times. Each time a different random part of the network answers. Where the answers agree, the network knows. Where they scatter, it is guessing.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Dots: the training data, with a gap in the middle. Black: the ordinary prediction. Thin lines: predictions with dropout left on. Gold: their average. Green band: where 95% of them fall</figcaption><canvas class="short" id="mc-c" role="img" aria-label="Data, ordinary prediction and many dropout predictions with an uncertainty band"></canvas></figure>
 <figure class="fig"><figcaption>Top: all the answers given at your x so far. Bottom: how much the answers disagree at every x</figcaption><canvas class="short" id="mc-r" role="img" aria-label="Histogram of predictions at one point and uncertainty against x"></canvas></figure></div>
 ${uHow('mc', ['Train a network with dropout in the usual way.', 'Ordinary prediction: switch every neuron on and get one answer for each x.', 'Bayesian prediction: leave dropout on. Switch off a random share of neurons and predict the whole curve.', 'Do it again with a different random share. The curve comes out slightly different.', 'After many passes, the average is the prediction and the spread is the uncertainty.'])}
 <div class="controls">${uBtns('mc', 'Run', 'One pass', 'passes')}
 ${grp('The model', slider('mc-p', 'Share of neurons switched off, p', 0.05, 0.6, 0.01, 0.1) + slider('mc-hw', 'Neurons in each of the two hidden layers', 8, 64, 1, 48))}
 ${grp('The data', slider('mc-gap', 'Width of the gap with no data', 0, 3.2, 0.05, 2.2) + slider('mc-nz', 'Noise in the measurements', 0, 0.5, 0.01, 0.12) + `<div class="row"><button class="btn" id="mc-new" type="button">New random sample</button></div>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('k', 'Dropout passes so far')}${rd('sd', 'Uncertainty where there is data (±)', 'hl3')}${rd('sg', 'Uncertainty in the gap (±)', 'hl2')}${rd('so', 'Uncertainty beyond the data (±)')}${rd('ins', 'Training points inside the 95% band')}</dl>`,
    tryit:
      'Watch the thin lines pile up: they bunch together over the dots and fan out in the gap and past both ends. Move your x into the gap and read the histogram on the right, then move it onto the data. Widen the gap to 3.2, then close it to 0. Finally raise p to 0.5: the band gets wider everywhere. Is the network really less sure, or did you just turn up a dial?',
    math: String.raw`<h4>Intuition</h4><p>Ask one expert a question and you get one answer, with no way to tell a fact from a guess. Ask a panel of fifty, each of whom studied slightly different notes. On a question from the middle of the syllabus they all say the same thing. On a question nobody covered, the answers are all over the place, and that disagreement is the useful information: it tells you not to rely on the answer. Dropout already contains such a panel. Every random choice of which neurons to switch off is a slightly different network that shares the same training. Leaving dropout on at prediction time lets you poll them.</p>
 <h4>The recipe (Monte Carlo dropout)</h4><p>Train with dropout as usual. To predict at \(x\), run \(T\) forward passes, each with a fresh random mask \(\mathbf m_t\), giving \(\hat y_t=f(x;\mathbf m_t)\). Then</p>$$\text{prediction}=\bar y=\frac1T\sum_{t=1}^{T}\hat y_t,\qquad\text{model uncertainty}=s^2=\frac{1}{T-1}\sum_{t=1}^{T}\big(\hat y_t-\bar y\big)^2 .$$<p>The green band on this page is \(\bar y\pm1.96\,s\). The passes stop when adding more no longer changes the band.</p>
 <h4>Why it is called Bayesian</h4><p>A Bayesian treats the weights as uncertain: there is a whole distribution of weight settings consistent with the data, the posterior \(p(\mathbf w\mid\mathcal D)\), and a prediction should average over all of them,</p>$$p(y\mid x,\mathcal D)=\int p(y\mid x,\mathbf w)\,p(\mathbf w\mid\mathcal D)\,d\mathbf w .$$<p>That integral cannot be computed for a neural network. Gal and Ghahramani (2016) showed that training with dropout is mathematically equivalent to fitting an approximation \(q(\mathbf w)\) to the posterior, in which each neuron's outgoing weights are randomly either kept or set to zero. Each dropout pass is then one sample of weights from \(q\), and averaging passes is a Monte Carlo estimate of the integral. No change to the network or its training is required.</p>
 <h4>Two kinds of uncertainty</h4><p><b>Model (epistemic) uncertainty</b> comes from not having seen enough data. It is what the scatter of the passes measures, it is large in the gap and beyond the ends, and more data there would shrink it. <b>Noise (aleatoric) uncertainty</b> is the irreducible noise \(\sigma^2\) from the Bias and Variance lesson: the measurements themselves wobble, and no amount of data removes it. The total predictive variance is the sum,</p>$$\operatorname{Var}(y\mid x)\approx s^2(x)+\sigma^2 ,$$<p>so the band drawn here, which shows only \(s\), is the uncertainty about the curve, not about a new measurement.</p>
 <h4>Why it matters</h4><p>A model that can say "I do not know" can hand a case to a person, ask for more data where it is unsure (active learning), or decline to act. This is important in medicine, driving and anywhere a confident wrong answer is costly. An ordinary network extrapolates across the gap with a single smooth line and gives no warning.</p>
 <h4>Cautions</h4><ul><li>The size of the band depends on \(p\), which you chose. The method ranks regions by uncertainty much better than it gives the exact width. Check calibration on held-out data.</li><li>It tends to underestimate uncertainty, and far from the data it can still be overconfident.</li><li>It costs \(T\) forward passes per prediction.</li><li>Alternatives with the same aim: deep ensembles (train several networks from different starts and compare them), and Gaussian processes for small problems.</li></ul>`,
    q: 'Two regions have the same wide band. In one, collecting more data would shrink it; in the other it would not. Which kind of uncertainty is each, and which one does this method measure?',
    a: 'The one that shrinks with more data is model (epistemic) uncertainty: the network has not seen enough examples there. The one that does not is noise (aleatoric) uncertainty: the outcomes themselves vary. The scatter of dropout passes measures only the first. To capture the second, the network must also predict a noise level, or one must be estimated from the residuals.',
    init(root) {
      const c = $('#mc-c', root),
        cr = $('#mc-r', root),
        live = $('#mc-live', root),
        S = { p: 0.1, hw: 48, gap: 2.2, nz: 0.12, seed: 2 },
        NG = 110,
        EPM = 1200,
        f = (x) => Math.sin(1.6 * x) + 0.25 * x;
      let D = [],
        net,
        ep = 0,
        passes = [],
        sum,
        sq,
        k = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        det = null,
        last = 0,
        stable = 0,
        done = false,
        tph = [];
      const gx = (i) => -6 + (12 * i) / (NG - 1),
        gen = () => {
          const r = rng(S.seed * 9 + 4),
            rn = randnFrom(r);
          D = [];
          for (let i = 0; i < 46; i++) {
            const left = i % 2 === 0,
              x = left
                ? -2.6 + r() * (2.6 - S.gap / 2 - 0.15)
                : S.gap / 2 + 0.15 + r() * (2.6 - S.gap / 2 - 0.15);
            D.push({ x, y: f(x) + S.nz * rn() });
          }
        };
      const build = () => {
        net = NNet([1, S.hw, S.hw, 1], { act: 'relu', out: 'lin', seed: S.seed * 5 + run * 11 });
        ep = 0;
        passes = [];
        sum = new Float64Array(NG);
        sq = new Float64Array(NG);
        k = 0;
        det = null;
        done = false;
        stable = 0;
        tph = [];
        cu = 0;
        msg =
          'Stage 1: training a network with dropout (p = ' +
          f2(S.p) +
          ') on ' +
          D.length +
          ' measurements. Nothing lies in the gap between x = ' +
          f2(-S.gap / 2) +
          ' and ' +
          f2(S.gap / 2) +
          '.';
      };
      const sdAt = (i) =>
          k > 1 ? Math.sqrt(Math.max(0, (sq[i] - (sum[i] * sum[i]) / k) / (k - 1))) : 0,
        avg = (a, b) => {
          let s = 0,
            n = 0;
          for (let i = 0; i < NG; i++) {
            const x = gx(i);
            if (x >= a && x <= b) {
              s += sdAt(i);
              n++;
            }
          }
          return n ? s / n : 0;
        };
      const pass = () => {
        const mask = [0, 1].map(() =>
            Float64Array.from({ length: S.hw }, () => (Math.random() < S.p ? 0 : 1 / (1 - S.p))),
          ),
          ft = new Float32Array(NG);
        for (let i = 0; i < NG; i++) {
          const v = net.fwd([gx(i)], mask).A[3][0];
          ft[i] = v;
          sum[i] += v;
          sq[i] += v * v;
        }
        passes.push(ft);
        if (passes.length > 50) passes.shift();
        k++;
        tph.push(net.fwd([tp.v[0]], mask).A[3][0]);
        cu = k < 2 ? 2 : k < 5 ? 3 : 4;
        const tot = avg(-6, 6);
        if (k > 12 && Math.abs(tot - last) < 0.012 * tot) stable++;
        else stable = 0;
        last = tot;
        if (stable >= 14 || k >= 400) done = true;
        msg =
          'Pass ' +
          k +
          ': a random ' +
          mask.reduce((s, m) => s + m.filter((q) => !q).length, 0) +
          ' of the ' +
          2 * S.hw +
          ' hidden neurons were switched off and the rest drew this curve. So far the answers disagree by ±' +
          f2(avg(-2.5, -S.gap / 2 - 0.2)) +
          ' over the data and by ±' +
          f2(avg(-S.gap / 2 + 0.1, S.gap / 2 - 0.1) || 0) +
          ' in the gap.' +
          (done
            ? ' More passes no longer change the band, so it stops after ' + k + ' passes.'
            : '');
      };
      const tr = (n) => {
        for (let i = 0; i < n && ep < EPM; i++) {
          net.train(
            D.map((q) => [q.x]),
            D.map((q) => q.y),
            0.01,
            { drop: S.p },
          );
          ep++;
        }
        if (ep >= EPM && !det) {
          det = new Float32Array(NG);
          for (let i = 0; i < NG; i++) det[i] = net.p([gx(i)]);
          cu = 1;
          msg =
            'Stage 2: training is finished. The black line is the ordinary prediction, with every neuron on: one confident answer for every x, including across the gap. Now dropout is left on.';
        } else if (ep < EPM)
          msg = 'Stage 1: training with dropout, epoch ' + ep + ' of ' + EPM + '.';
      };
      const step = () => {
        if (done) {
          run++;
          build();
          return;
        }
        if (ep < EPM) tr(EPM);
        else pass();
      };
      const tp = testPoint(root, 'mc', ['Your x'], [0], [[-6, 6]]);
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c),
            P = plot(g, w, h, [-6, 6], [-2.6, 2.6], t, {
              xstep: 2,
              yticks: [-2, 0, 2],
              xlabel: 'x',
            });
          clipPlot(g, P);
          g.fillStyle = mix(t.ok, t.ok, 0, 0.07);
          g.fillRect(
            P.X(-S.gap / 2),
            P.Y(2.6),
            P.X(S.gap / 2) - P.X(-S.gap / 2),
            P.Y(-2.6) - P.Y(2.6),
          );
          g.fillRect(P.X(-6), P.Y(2.6), P.X(-2.6) - P.X(-6), P.Y(-2.6) - P.Y(2.6));
          g.fillRect(P.X(2.6), P.Y(2.6), P.X(6) - P.X(2.6), P.Y(-2.6) - P.Y(2.6));
          if (k > 1) {
            g.beginPath();
            for (let i = 0; i < NG; i++) {
              const y = sum[i] / k + 1.96 * sdAt(i);
              i ? g.lineTo(P.X(gx(i)), P.Y(y)) : g.moveTo(P.X(gx(i)), P.Y(y));
            }
            for (let i = NG - 1; i >= 0; i--)
              g.lineTo(P.X(gx(i)), P.Y(sum[i] / k - 1.96 * sdAt(i)));
            g.closePath();
            g.fillStyle = mix(t.accent, t.accent, 0, 0.2);
            g.fill();
          }
          passes.forEach((ft, q) => {
            const lastOne = q === passes.length - 1;
            g.strokeStyle = mix(t.accent, t.accent, 0, lastOne ? 1 : 0.22);
            g.lineWidth = lastOne ? 2 : 1;
            g.beginPath();
            for (let i = 0; i < NG; i++)
              i ? g.lineTo(P.X(gx(i)), P.Y(ft[i])) : g.moveTo(P.X(gx(i)), P.Y(ft[i]));
            g.stroke();
          });
          if (ep < EPM) {
            g.strokeStyle = t.ink;
            g.lineWidth = 2;
            g.beginPath();
            for (let i = 0; i < NG; i++) {
              const y = net.p([gx(i)]);
              i ? g.lineTo(P.X(gx(i)), P.Y(y)) : g.moveTo(P.X(gx(i)), P.Y(y));
            }
            g.stroke();
          }
          if (det) {
            g.strokeStyle = t.ink;
            g.lineWidth = 2;
            g.setLineDash(k ? [6, 4] : []);
            g.beginPath();
            for (let i = 0; i < NG; i++)
              i ? g.lineTo(P.X(gx(i)), P.Y(det[i])) : g.moveTo(P.X(gx(i)), P.Y(det[i]));
            g.stroke();
            g.setLineDash([]);
          }
          if (k > 1) {
            g.strokeStyle = t.gold;
            g.lineWidth = 3;
            g.beginPath();
            for (let i = 0; i < NG; i++)
              i ? g.lineTo(P.X(gx(i)), P.Y(sum[i] / k)) : g.moveTo(P.X(gx(i)), P.Y(sum[i] / k));
            g.stroke();
          }
          g.fillStyle = t.ink;
          for (const q of D) {
            g.beginPath();
            g.arc(P.X(q.x), P.Y(q.y), 3.2, 0, 7);
            g.fill();
          }
          const px = P.X(tp.v[0]);
          g.strokeStyle = t.ink;
          g.lineWidth = 1.2;
          g.beginPath();
          g.moveTo(px, P.Y(-2.6));
          g.lineTo(px, P.Y(2.6));
          g.stroke();
          g.restore();
          g.font = MONO;
          g.fillStyle = t.ok;
          g.textAlign = 'center';
          g.fillText('no data here', P.X(0), P.m.t + 10);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5,
            m = { l: 40, r: 12, t: 24, b: 20 };
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            'answers given at x = ' + f2(tp.v[0]) + ' by the ' + tph.length + ' passes',
            8,
            13,
          );
          const X = (v) => m.l + ((v + 2.6) / 5.2) * (w - m.l - m.r),
            nb = 40,
            cnt = new Array(nb).fill(0);
          for (const v of tph) cnt[clamp(Math.floor(((v + 2.6) / 5.2) * nb), 0, nb - 1)]++;
          const cm = Math.max(4, ...cnt);
          g.fillStyle = t.accent;
          cnt.forEach((q, i) =>
            g.fillRect(
              m.l + (i * (w - m.l - m.r)) / nb,
              h1 - m.b - (q / cm) * (h1 - m.t - m.b),
              (w - m.l - m.r) / nb - 1,
              (q / cm) * (h1 - m.t - m.b),
            ),
          );
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h1 - m.b);
          g.lineTo(w - m.r, h1 - m.b);
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          [-2, -1, 0, 1, 2].forEach((v) => g.fillText(String(v), X(v), h1 - m.b + 13));
          if (det) {
            const gi = Math.round(((tp.v[0] + 6) / 12) * (NG - 1));
            g.strokeStyle = t.ink;
            g.lineWidth = 2;
            g.setLineDash([6, 4]);
            g.beginPath();
            g.moveTo(X(det[gi]), m.t);
            g.lineTo(X(det[gi]), h1 - m.b);
            g.stroke();
            g.setLineDash([]);
            g.strokeStyle = t.ok;
            g.lineWidth = 1.5;
            g.beginPath();
            g.moveTo(X(f(tp.v[0])), m.t);
            g.lineTo(X(f(tp.v[0])), h1 - m.b);
            g.stroke();
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText('dashed: ordinary answer', w - m.r, m.t + 12);
            g.fillText('red: the truth', w - m.r, m.t + 25);
          }
          const sds = Array.from({ length: NG }, (_, i) => sdAt(i)),
            U = uLine(g, t, 0, h1, w, h - h1, sds, {
              title: 'disagreement between passes (±), at every x',
              n: NG,
              ym: Math.max(0.3, ...sds) * 1.1,
              col: t.accent,
            });
          g.fillStyle = t.ink;
          for (const q of D) {
            g.fillRect(U.X(((q.x + 6) / 12) * (NG - 1)) - 1, h - 17, 2, 7);
          }
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('ticks: where the training data is', w - 12, h - 3);
        }
        live.textContent = msg;
        uCur(root, 'mc', cu);
        const gi = Math.round(((tp.v[0] + 6) / 12) * (NG - 1));
        tp.out.textContent = !det
          ? 'The network is still training.'
          : k < 2
            ? 'Ordinary prediction at x = ' +
              f2(tp.v[0]) +
              ': ' +
              f2(det[gi]) +
              ', with nothing to say how sure it is.'
            : 'At x = ' +
              f2(tp.v[0]) +
              ' the ordinary network says ' +
              f2(det[gi]) +
              '. The ' +
              k +
              ' dropout passes average ' +
              f2(sum[gi] / k) +
              ' and disagree by ±' +
              f2(sdAt(gi)) +
              ', so a 95% range is ' +
              f2(sum[gi] / k - 1.96 * sdAt(gi)) +
              ' to ' +
              f2(sum[gi] / k + 1.96 * sdAt(gi)) +
              '. The truth is ' +
              f2(f(tp.v[0])) +
              '.';
        let ins = 0;
        if (k > 1)
          for (const q of D) {
            const i = Math.round(((q.x + 6) / 12) * (NG - 1));
            if (Math.abs(q.y - sum[i] / k) <= 1.96 * sdAt(i) + 1e-9) ins++;
          }
        setR(
          root,
          'ph',
          ep < EPM
            ? 'Training (' + ep + ' of ' + EPM + ' epochs)'
            : done
              ? 'Finished'
              : k
                ? 'Polling with dropout on'
                : 'Trained',
        );
        setR(root, 'k', String(k));
        setR(
          root,
          'sd',
          k > 1 ? f2((avg(-2.5, -S.gap / 2 - 0.2) + avg(S.gap / 2 + 0.2, 2.5)) / 2) : '–',
        );
        setR(root, 'sg', k > 1 && S.gap > 0.3 ? f2(avg(-S.gap / 2 + 0.1, S.gap / 2 - 0.1)) : '–');
        setR(root, 'so', k > 1 ? f2((avg(-6, -4.5) + avg(4.5, 6)) / 2) : '–');
        setR(root, 'ins', k > 1 ? ins + ' of ' + D.length : '–');
      };
      tp.on = () => {
        tph = [];
        if (det && k) {
          /* re-poll this x with fresh masks */ for (let q = 0; q < Math.min(k, 200); q++) {
            const mask = [0, 1].map(() =>
              Float64Array.from({ length: S.hw }, () => (Math.random() < S.p ? 0 : 1 / (1 - S.p))),
            );
            tph.push(net.fwd([tp.v[0]], mask).A[3][0]);
          }
        }
        draw();
      };
      gen();
      build();
      if (RM) {
        tr(EPM);
        for (let i = 0; i < 80; i++) pass();
      }
      const redo = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      const anim = Anim(() => {
        fr++;
        if (ep < EPM) {
          tr(12);
          draw();
          return;
        }
        if (fr % (k < 8 ? 34 : 5)) return;
        pass();
        draw();
        chk();
      });
      pb = playBtn($('#mc-play', root), anim);
      chk = runCtl(pb, root, 'mc', {
        done: () => done,
        restart: () => {
          run++;
          build();
        },
        count: () => k,
        what: 'passes',
      });
      $('#mc-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#mc-reset', root).onclick = redo;
      bind(root, 'mc-p', (v) => {
        S.p = v;
        redo();
      });
      bind(
        root,
        'mc-hw',
        (v) => {
          S.hw = v;
          redo();
        },
        (v) => String(v),
      );
      bind(root, 'mc-gap', (v) => {
        S.gap = v;
        gen();
        redo();
      });
      bind(root, 'mc-nz', (v) => {
        S.nz = v;
        gen();
        redo();
      });
      $('#mc-new', root).onclick = () => {
        S.seed++;
        gen();
        redo();
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
    id: 'universal',
    tab: 'Universal approximation',
    title: 'Enough neurons can draw any curve',
    lede: 'One hidden layer is enough, in principle, to copy any continuous curve as closely as you like. Each neuron contributes one simple piece, a bend or an S-shaped step, and the output adds the pieces up. Watch the pieces arrange themselves, add neurons one at a time, or draw your own curve and see how many neurons it takes.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Dashed: the curve to copy. Thin coloured lines: what each neuron contributes. Gold: their sum, the network’s output. Drag across the picture to draw your own curve</figcaption><canvas class="short" id="ua-c" role="img" aria-label="Target curve, the contribution of each neuron and their sum"></canvas></figure>
 <figure class="fig"><figcaption>Top: error by training step. Bottom: the best error reached with each number of neurons</figcaption><canvas class="short" id="ua-r" role="img" aria-label="Loss curve and error against number of neurons"></canvas></figure></div>
 ${uHow('ua', ['Each hidden neuron makes one simple piece: it scales and shifts the input, then bends it.', 'The output multiplies every piece by its own weight and adds them all up.', 'Compare the sum with the target at 100 positions: the error.', 'Adjust every scale, shift and weight a little to lower the error. Repeat until it stops falling.', 'If the fit is still poor, add another neuron and train again.'])}
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="ua-play" type="button">Train</button><button class="btn" id="ua-step" type="button">Twenty steps</button><button class="btn" id="ua-reset" type="button">Start over</button></div>${stopBox('ua', 'steps')}<div class="row"><button class="btn" id="ua-grow" type="button">Add Neurons One at a Time</button></div>`)}
 ${grp('The network', slider('ua-H', 'Hidden neurons', 1, 24, 1, 5) + `<label class="ctl" for="ua-act"><span>Shape of each piece</span><select id="ua-act"><option value="tanh">S-shaped step (tanh)</option><option value="relu">Bend (ReLU)</option></select></label>`)}
 ${grp('The curve to copy', `<label class="ctl" for="ua-tg"><span>Target</span><select id="ua-tg"><option value="wave">A wave</option><option value="hill">One hill</option><option value="step">A staircase</option><option value="sky">A city skyline</option><option value="own">Your own drawing</option></select></label><p class="status" id="ua-st" role="status"></p>`)}</div>
 <dl class="read">${rd('H', 'Hidden neurons')}${rd('np', 'Numbers the network can adjust')}${rd('ep', 'Training steps')}${rd('L', 'Error (mean squared)', 'hl')}${rd('st', 'Status')}</dl>`,
    tryit:
      'Press Add Neurons One at a Time on the wave and watch the bars on the right: how many neurons until the gold line sits on the dashed one? Then switch to the staircase with S-shaped pieces: count the steps in the staircase and the neurons needed. Change the pieces to Bends and try the skyline. Finally draw something of your own, the wilder the better, and find its number.',
    math: String.raw`<h4>Intuition</h4><p>A child with a box of identical wooden ramps can build the outline of any hill: a few ramps for a gentle one, many small ramps for a jagged one. The ramps are all the same simple shape. What changes is where each is placed, how steep it is, and whether it goes up or down. A neural network with one hidden layer builds curves this way. Each hidden neuron is one ramp or one step, and training is the work of sliding the pieces into place.</p>
 <h4>The network as a sum of pieces</h4><p>With one input \(x\), \(H\) hidden neurons and one output,</p>$$\hat f(x)=c+\sum_{j=1}^{H}v_j\,\phi\big(w_jx+b_j\big).$$<p>Inside each term, \(w_j\) sets how sharp the piece is, \(-b_j/w_j\) sets where along the axis it sits, and \(v_j\) sets its height and whether it points up or down. The thin coloured lines in the picture are the individual terms \(v_j\phi(w_jx+b_j)\), and the gold line is their sum.</p>
 <h4>Why steps are enough</h4><p>Take \(\phi=\tanh\) with a large \(w\): the piece is nearly a sharp step. Two steps at slightly different positions, one up and one down, make a narrow bump of any height. Any continuous curve can be approximated by a row of narrow bumps of the right heights, as a bar chart approximates a curve, and making the bars narrower makes the approximation as close as you like. With \(\phi=\mathrm{ReLU}\) each piece is a bend, sums of bends are piecewise-linear curves, and enough short straight segments follow any continuous curve.</p>
 <h4>The theorem</h4><p>For any continuous function \(f\) on a closed, bounded interval and any tolerance \(\varepsilon&gt;0\), there is a number of neurons \(H\) and a choice of weights such that</p>$$\big|\hat f(x)-f(x)\big|&lt;\varepsilon\quad\text{for every }x\text{ in the interval.}$$<p>This holds for any activation that is not a polynomial (Cybenko 1989; Hornik 1991; Leshno and others 1993), and in any number of input dimensions.</p>
 <h4>What the theorem does not say</h4><ul><li><b>How many neurons.</b> It may be very many. Rough or rapidly changing targets need more pieces, which you can check with the skyline or your own drawing. In high dimensions the number can grow exponentially.</li><li><b>That training will find the weights.</b> The theorem says good weights exist. Gradient descent may stop at a worse arrangement, and running the same setting twice here can give different errors.</li><li><b>Anything about new data.</b> Fitting 100 known points closely is not the same as predicting well away from them. That is the bias and variance story again.</li><li><b>That one layer is the best design.</b> Deep networks can represent many useful functions with far fewer neurons than a single wide layer, because later layers reuse the pieces built by earlier ones.</li></ul>`,
    q: 'The staircase target has four flat levels. About how many tanh neurons should be enough, and why does adding many more barely help?',
    a: 'Three, one for each jump between levels: a tanh neuron with a large input weight is one step, placed where it is needed and scaled to the height of that jump, and the output bias sets the first level. Once every jump has its own neuron the error is already close to zero, so extra neurons have nothing left to explain and receive almost no gradient.',
    init(root) {
      const c = $('#ua-c', root),
        cr = $('#ua-r', root),
        live = $('#ua-live', root),
        st = $('#ua-st', root),
        S = { H: 5, act: 'tanh', tg: 'wave' },
        NP = 100,
        TG = {
          wave: (x) => 0.8 * Math.sin(4.2 * x) + 0.3 * x,
          hill: (x) => 1.3 * Math.exp(-7 * x * x) - 0.5,
          step: (x) => (x < -0.5 ? -0.8 : x < 0 ? -0.1 : x < 0.5 ? 0.7 : 0.1),
          sky: (x) =>
            [0.2, 0.9, 0.4, -0.3, 0.6, -0.6, 0.1, 0.8, -0.1, 0.5][
              Math.min(9, Math.floor((x + 1) * 5))
            ],
        };
      let Tg = new Float64Array(NP),
        net,
        ep = 0,
        LL = [],
        conv = false,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        grow = false,
        best = {},
        painting = false,
        lastI = -1;
      const xi = (i) => -1 + (2 * i) / (NP - 1),
        setT = () => {
          if (S.tg !== 'own') for (let i = 0; i < NP; i++) Tg[i] = TG[S.tg](xi(i));
        };
      const build = () => {
        net = NNet([1, S.H, 1], {
          act: S.act,
          out: 'lin',
          seed: run * 31 + S.H * 7,
          scale: S.act === 'relu' ? 1.2 : 2.2,
        });
        for (let j = 0; j < S.H; j++) net.B[0][j] = (Math.random() - 0.5) * 3;
        ep = 0;
        LL = [];
        conv = false;
        cu = 0;
        msg =
          S.H +
          ' neuron' +
          (S.H === 1 ? '' : 's') +
          ' with random settings: the pieces are scattered and their sum looks nothing like the target. Press Train.';
      };
      const X = Array.from({ length: NP }, (_, i) => [xi(i) * 3]);
      const train = (k) => {
        for (let q = 0; q < k && !conv; q++) {
          const l = 2 * net.train(X, Tg, 0.02).loss;
          ep++;
          LL.push(l);
          if (
            (ep > 300 && Math.abs(LL[ep - 1] - LL[ep - 201]) < 0.01 * LL[ep - 1] + 1e-7) ||
            ep >= 6000
          )
            conv = true;
        }
        const l = LL[ep - 1];
        best[S.H] = Math.min(best[S.H] == null ? 1e9 : best[S.H], l);
        cu = conv ? 4 : 3;
        msg =
          'Step ' +
          ep +
          ': the ' +
          S.H +
          ' piece' +
          (S.H === 1 ? '' : 's') +
          ' shifted a little. Mean squared error ' +
          sci(l) +
          '.' +
          (conv
            ? (ep >= 6000
                ? ' Reached the limit of 6000 steps.'
                : ' Converged: the error has stopped falling with ' +
                  S.H +
                  ' neuron' +
                  (S.H === 1 ? '' : 's') +
                  '.') +
              (l > 0.004
                ? ' The fit is still rough: there are not enough pieces for this curve.'
                : ' The gold line follows the target closely.')
            : '');
      };
      const draw = () => {
        const t = T(),
          col = UC(t);
        {
          const { g, w, h } = fit(c),
            P = plot(g, w, h, [-1, 1], [-1.6, 1.6], t, {
              xstep: 0.5,
              yticks: [-1, 0, 1],
              xlabel: 'x',
            });
          clipPlot(g, P);
          const f = ACT[S.act][0];
          for (let j = 0; j < S.H; j++) {
            g.strokeStyle = mix(col[j % 8], col[j % 8], 0, 0.75);
            g.lineWidth = 1.3;
            g.beginPath();
            for (let i = 0; i < NP; i++) {
              const v = net.W[1][0][j] * f(net.W[0][j][0] * xi(i) * 3 + net.B[0][j]);
              i ? g.lineTo(P.X(xi(i)), P.Y(v)) : g.moveTo(P.X(xi(i)), P.Y(v));
            }
            g.stroke();
          }
          g.setLineDash([7, 5]);
          g.strokeStyle = t.ink;
          g.lineWidth = 2;
          g.beginPath();
          for (let i = 0; i < NP; i++)
            i ? g.lineTo(P.X(xi(i)), P.Y(Tg[i])) : g.moveTo(P.X(xi(i)), P.Y(Tg[i]));
          g.stroke();
          g.setLineDash([]);
          g.strokeStyle = t.gold;
          g.lineWidth = 3.4;
          g.beginPath();
          for (let i = 0; i < NP; i++) {
            const v = net.p(X[i]);
            i ? g.lineTo(P.X(xi(i)), P.Y(v)) : g.moveTo(P.X(xi(i)), P.Y(v));
          }
          g.stroke();
          {
            const x = tp.v[0],
              v = net.p([x * 3]);
            markTP(g, P.X(x), P.Y(v), t);
          }
          g.restore();
          c._P = P;
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5;
          const lo = LL.length ? Math.min(...LL) : 0;
          uLine(
            g,
            t,
            0,
            0,
            w,
            h1,
            LL.map((v) => Math.log10(v + 1e-9)),
            {
              title: 'error by training step (log scale): 10^…',
              n: Math.max(100, LL.length),
              y1: Math.min(-3, Math.floor(Math.log10(lo + 1e-9))),
              ym: 0.5,
              col: t.ok,
            },
          );
          const m = { l: 44, r: 12, t: 22, b: 22 },
            bw = (w - m.l - m.r) / 24,
            Y = (v) =>
              h - m.b - ((clamp(Math.log10(v + 1e-9), -4, 0) + 4) / 4) * (h - h1 - m.t - m.b);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('best error reached with each number of neurons', 8, h1 + 13);
          g.textAlign = 'right';
          g.fillText('1', m.l - 5, Y(1) + 4);
          g.fillText('10^-4', m.l - 5, Y(1e-4) + 4);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(m.l, h - m.b);
          g.lineTo(w - m.r, h - m.b);
          g.stroke();
          for (let k = 1; k <= 24; k++) {
            const x = m.l + (k - 1) * bw + bw * 0.15;
            if (best[k] != null) {
              g.fillStyle = k === S.H ? t.gold : t.accent;
              g.fillRect(x, Y(best[k]), bw * 0.7, h - m.b - Y(best[k]));
            }
            if (k % 2 === 1) {
              g.fillStyle = t.muted;
              g.textAlign = 'center';
              g.fillText(String(k), x + bw * 0.35, h - m.b + 14);
            }
          }
        }
        live.textContent = msg;
        uCur(root, 'ua', cu);
        const x = tp.v[0],
          gi = Math.round(((x + 1) / 2) * (NP - 1));
        tp.out.textContent =
          'At x = ' +
          f2(x) +
          ' the network outputs ' +
          f2(net.p([x * 3])) +
          ' and the target is ' +
          f2(Tg[gi]) +
          '.';
        setR(root, 'H', String(S.H));
        setR(root, 'np', String(3 * S.H + 1));
        setR(root, 'ep', String(ep));
        setR(root, 'L', LL.length ? sci(LL[ep - 1]) : '–');
        setR(
          root,
          'st',
          grow ? 'Adding neurons' : conv ? 'Converged' : ep ? 'Training' : 'Not started',
        );
      };
      const tp = testPoint(root, 'ua', ['Your x'], [0.3], [[-1, 1]]);
      tp.on = () => draw();
      setT();
      build();
      if (RM) train(6000);
      const anim = Anim(() => {
        fr++;
        if (ep < 40 && fr % 4) return;
        train(ep < 40 ? 2 : ep < 400 ? 4 : 10);
        if (conv && grow) {
          if (S.H < 12 && LL[ep - 1] > 3e-4) {
            S.H++;
            hb.set(S.H);
            run++;
            build();
            msg = 'Added a neuron: now ' + S.H + '. Training again from a fresh random start.';
          } else {
            grow = false;
            msg += ' Stopped adding neurons at ' + S.H + '.';
          }
        }
        draw();
        if (!grow) chk();
      });
      pb = playBtn($('#ua-play', root), anim);
      chk = runCtl(pb, root, 'ua', {
        done: () => conv && !grow,
        restart: () => {
          run++;
          build();
        },
        count: () => ep,
        what: 'steps',
      });
      const redo = () => {
        pb.stop();
        grow = false;
        run++;
        build();
        draw();
      };
      $('#ua-step', root).onclick = () => {
        pb.stop();
        grow = false;
        if (conv) {
          run++;
          build();
        } else train(20);
        draw();
      };
      $('#ua-reset', root).onclick = redo;
      $('#ua-grow', root).onclick = () => {
        pb.stop();
        best = {};
        S.H = 1;
        hb.set(1);
        run++;
        build();
        grow = true;
        pb.start();
      };
      const hb = bind(
        root,
        'ua-H',
        (v) => {
          S.H = v;
          redo();
        },
        (v) => String(v),
      );
      $('#ua-act', root).onchange = (e) => {
        S.act = e.target.value;
        best = {};
        redo();
      };
      $('#ua-tg', root).onchange = (e) => {
        S.tg = e.target.value;
        best = {};
        if (S.tg === 'own') st.textContent = 'Drag across the left picture to draw the curve.';
        else st.textContent = '';
        setT();
        redo();
      };
      const paint = (e) => {
        const P = c._P;
        if (!P) return;
        const r = c.getBoundingClientRect(),
          x = clamp(((e.clientX - r.left - P.m.l) / (P.w - P.m.l - P.m.r)) * 2 - 1, -1, 1),
          y = clamp(
            -((e.clientY - r.top - (P.h - P.m.b)) / (P.h - P.m.t - P.m.b)) * 3.2 - 1.6,
            -1.5,
            1.5,
          ),
          i = Math.round(((x + 1) / 2) * (NP - 1));
        if (lastI >= 0 && lastI !== i) {
          const a = Math.min(i, lastI),
            b = Math.max(i, lastI),
            ya = Tg[lastI];
          for (let q = a; q <= b; q++)
            Tg[q] = ya + (y - ya) * (i === lastI ? 1 : Math.abs(q - lastI) / Math.abs(i - lastI));
        } else Tg[i] = y;
        lastI = i;
      };
      c.style.touchAction = 'none';
      c.addEventListener('pointerdown', (e) => {
        pb.stop();
        grow = false;
        painting = true;
        lastI = -1;
        S.tg = 'own';
        $('#ua-tg', root).value = 'own';
        st.textContent = 'Your own curve. Press Train, or Add Neurons One at a Time.';
        c.setPointerCapture(e.pointerId);
        paint(e);
        draw();
      });
      c.addEventListener('pointermove', (e) => {
        if (!painting) return;
        paint(e);
        draw();
      });
      ['pointerup', 'pointercancel'].forEach((n) =>
        c.addEventListener(n, () => {
          if (painting) {
            painting = false;
            best = {};
            run++;
            build();
            draw();
          }
        }),
      );
      return {
        draw,
        stop: () => {
          pb.stop();
          grow = false;
        },
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
