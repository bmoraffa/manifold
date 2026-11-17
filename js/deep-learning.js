/* Manifold — Automatic differentiation and Deep learning.
   © 2026 Bahman Moraffah. All rights reserved. Load order matters: see index.html. */
/* ============ tiny reverse-mode autodiff for the trained deep-learning lessons ============ */
const AD = (() => {
  let tape = [];
  const N = (v, sh) => ({ v, g: new Float64Array(v.length), sh }),
    op = (o, bk) => {
      tape.push(bk);
      return o;
    };
  const param = (sh, sd, rn) => {
    const n = sh.reduce((a, b) => a * b, 1),
      p = N(
        Float64Array.from({ length: n }, () => sd * rn()),
        sh,
      );
    p.m = new Float64Array(n);
    p.s = new Float64Array(n);
    return p;
  };
  const mm = (A, B) => {
    const m = A.sh[0],
      k = A.sh[1],
      n = B.sh[1],
      o = N(new Float64Array(m * n), [m, n]),
      a = A.v,
      b = B.v,
      ov = o.v;
    for (let i = 0; i < m; i++)
      for (let l = 0; l < k; l++) {
        const x = a[i * k + l];
        if (x !== 0) for (let j = 0; j < n; j++) ov[i * n + j] += x * b[l * n + j];
      }
    return op(o, () => {
      const og = o.g,
        ag = A.g,
        bg = B.g;
      for (let i = 0; i < m; i++)
        for (let l = 0; l < k; l++) {
          const x = a[i * k + l];
          let s = 0;
          for (let j = 0; j < n; j++) {
            const q = og[i * n + j];
            s += q * b[l * n + j];
            bg[l * n + j] += x * q;
          }
          ag[i * k + l] += s;
        }
    });
  };
  const mmT = (A, B) => {
    const m = A.sh[0],
      k = A.sh[1],
      n = B.sh[0],
      o = N(new Float64Array(m * n), [m, n]),
      a = A.v,
      b = B.v;
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let l = 0; l < k; l++) s += a[i * k + l] * b[j * k + l];
        o.v[i * n + j] = s;
      }
    return op(o, () => {
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) {
          const q = o.g[i * n + j];
          if (q !== 0)
            for (let l = 0; l < k; l++) {
              A.g[i * k + l] += q * b[j * k + l];
              B.g[j * k + l] += q * a[i * k + l];
            }
        }
    });
  };
  const add = (A, B) => {
    const o = N(
      A.v.map((x, i) => x + B.v[i]),
      A.sh,
    );
    return op(o, () => {
      for (let i = 0; i < o.g.length; i++) {
        A.g[i] += o.g[i];
        B.g[i] += o.g[i];
      }
    });
  };
  const relu = (A) => {
    const o = N(
      A.v.map((x) => (x > 0 ? x : 0)),
      A.sh,
    );
    return op(o, () => {
      for (let i = 0; i < o.g.length; i++) if (A.v[i] > 0) A.g[i] += o.g[i];
    });
  };
  const gather = (E, idx) => {
    const d = E.sh[1],
      o = N(new Float64Array(idx.length * d), [idx.length, d]);
    idx.forEach((t, i) => {
      for (let j = 0; j < d; j++) o.v[i * d + j] = E.v[t * d + j];
    });
    return op(o, () => {
      idx.forEach((t, i) => {
        for (let j = 0; j < d; j++) E.g[t * d + j] += o.g[i * d + j];
      });
    });
  };
  const attn = (S, sc, causal) => {
    const T = S.sh[0],
      o = N(new Float64Array(T * T), [T, T]);
    for (let i = 0; i < T; i++) {
      let mx = -1e30;
      const lim = causal ? i : T - 1;
      for (let j = 0; j <= lim; j++) mx = Math.max(mx, S.v[i * T + j] * sc);
      let z = 0;
      for (let j = 0; j <= lim; j++) {
        const e = Math.exp(S.v[i * T + j] * sc - mx);
        o.v[i * T + j] = e;
        z += e;
      }
      for (let j = 0; j <= lim; j++) o.v[i * T + j] /= z;
    }
    return op(o, () => {
      for (let i = 0; i < T; i++) {
        const lim = causal ? i : T - 1;
        let d = 0;
        for (let j = 0; j <= lim; j++) d += o.v[i * T + j] * o.g[i * T + j];
        for (let j = 0; j <= lim; j++) S.g[i * T + j] += sc * o.v[i * T + j] * (o.g[i * T + j] - d);
      }
    });
  };
  const ce = (L, tg, from) => {
    const T = L.sh[0],
      V = L.sh[1],
      o = N(new Float64Array(1), [1]),
      Pm = new Float64Array(T * V),
      cnt = T - from;
    for (let t = from; t < T; t++) {
      let mx = -1e30;
      for (let j = 0; j < V; j++) mx = Math.max(mx, L.v[t * V + j]);
      let z = 0;
      for (let j = 0; j < V; j++) {
        const e = Math.exp(L.v[t * V + j] - mx);
        Pm[t * V + j] = e;
        z += e;
      }
      for (let j = 0; j < V; j++) Pm[t * V + j] /= z;
      o.v[0] -= Math.log(Pm[t * V + tg[t]] + 1e-12) / cnt;
    }
    o.P = Pm;
    return op(o, () => {
      for (let t = from; t < T; t++)
        for (let j = 0; j < V; j++)
          L.g[t * V + j] += (o.g[0] * (Pm[t * V + j] - (j === tg[t] ? 1 : 0))) / cnt;
    });
  };
  const conv3 = (X, W, b) => {
    const C = X.sh[0],
      H = X.sh[1],
      Wd = X.sh[2],
      O = W.sh[0],
      o = N(new Float64Array(O * H * Wd), [O, H, Wd]),
      x = X.v,
      w = W.v;
    for (let q = 0; q < O; q++)
      for (let y = 0; y < H; y++)
        for (let xx = 0; xx < Wd; xx++) {
          let s = b.v[q];
          for (let c = 0; c < C; c++)
            for (let dy = -1; dy <= 1; dy++) {
              const yy = y + dy;
              if (yy < 0 || yy >= H) continue;
              for (let dx = -1; dx <= 1; dx++) {
                const x2 = xx + dx;
                if (x2 < 0 || x2 >= Wd) continue;
                s += x[(c * H + yy) * Wd + x2] * w[((q * C + c) * 3 + dy + 1) * 3 + dx + 1];
              }
            }
          o.v[(q * H + y) * Wd + xx] = s;
        }
    return op(o, () => {
      for (let q = 0; q < O; q++)
        for (let y = 0; y < H; y++)
          for (let xx = 0; xx < Wd; xx++) {
            const g = o.g[(q * H + y) * Wd + xx];
            if (g === 0) continue;
            b.g[q] += g;
            for (let c = 0; c < C; c++)
              for (let dy = -1; dy <= 1; dy++) {
                const yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (let dx = -1; dx <= 1; dx++) {
                  const x2 = xx + dx;
                  if (x2 < 0 || x2 >= Wd) continue;
                  const xi = (c * H + yy) * Wd + x2,
                    wi = ((q * C + c) * 3 + dy + 1) * 3 + dx + 1;
                  W.g[wi] += g * x[xi];
                  X.g[xi] += g * w[wi];
                }
              }
          }
    });
  };
  const pool = (X) => {
    const C = X.sh[0],
      H = X.sh[1],
      Wd = X.sh[2],
      h = H / 2,
      w2 = Wd / 2,
      o = N(new Float64Array(C * h * w2), [C, h, w2]),
      ix = new Int32Array(C * h * w2);
    for (let c = 0; c < C; c++)
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w2; x++) {
          let b = -1e30,
            bi = 0;
          for (let a = 0; a < 4; a++) {
            const i = (c * H + 2 * y + (a >> 1)) * Wd + 2 * x + (a & 1);
            if (X.v[i] > b) {
              b = X.v[i];
              bi = i;
            }
          }
          o.v[(c * h + y) * w2 + x] = b;
          ix[(c * h + y) * w2 + x] = bi;
        }
    return op(o, () => {
      for (let i = 0; i < ix.length; i++) X.g[ix[i]] += o.g[i];
    });
  };
  const up = (X) => {
    const C = X.sh[0],
      h = X.sh[1],
      w2 = X.sh[2],
      H = h * 2,
      Wd = w2 * 2,
      o = N(new Float64Array(C * H * Wd), [C, H, Wd]);
    for (let c = 0; c < C; c++)
      for (let y = 0; y < H; y++)
        for (let x = 0; x < Wd; x++)
          o.v[(c * H + y) * Wd + x] = X.v[(c * h + (y >> 1)) * w2 + (x >> 1)];
    return op(o, () => {
      for (let c = 0; c < C; c++)
        for (let y = 0; y < H; y++)
          for (let x = 0; x < Wd; x++)
            X.g[(c * h + (y >> 1)) * w2 + (x >> 1)] += o.g[(c * H + y) * Wd + x];
    });
  };
  const cat = (A, B) => {
    const o = N(new Float64Array(A.v.length + B.v.length), [A.sh[0] + B.sh[0], A.sh[1], A.sh[2]]);
    o.v.set(A.v);
    o.v.set(B.v, A.v.length);
    return op(o, () => {
      const n = A.v.length;
      for (let i = 0; i < n; i++) A.g[i] += o.g[i];
      for (let i = 0; i < B.v.length; i++) B.g[i] += o.g[n + i];
    });
  };
  const bce = (L, tg) => {
    const n = L.v.length,
      o = N(new Float64Array(1), [1]);
    for (let i = 0; i < n; i++) {
      const z = L.v[i];
      o.v[0] += (Math.max(z, 0) - z * tg[i] + Math.log(1 + Math.exp(-Math.abs(z)))) / n;
    }
    return op(o, () => {
      for (let i = 0; i < n; i++) L.g[i] += (o.g[0] * (1 / (1 + Math.exp(-L.v[i])) - tg[i])) / n;
    });
  };
  const input = (v, sh) => N(Float64Array.from(v), sh),
    backward = (l) => {
      l.g[0] = 1;
      for (let i = tape.length - 1; i >= 0; i--) tape[i]();
      tape = [];
    },
    clear = () => {
      tape = [];
    };
  const adam = (ps, lr, t, bs) => {
    const c1 = 1 - Math.pow(0.9, t),
      c2 = 1 - Math.pow(0.999, t);
    for (const p of ps) {
      for (let i = 0; i < p.v.length; i++) {
        const g = p.g[i] / bs;
        p.m[i] = 0.9 * p.m[i] + 0.1 * g;
        p.s[i] = 0.999 * p.s[i] + 0.001 * g * g;
        p.v[i] -= (lr * (p.m[i] / c1)) / (Math.sqrt(p.s[i] / c2) + 1e-8);
      }
      p.g.fill(0);
    }
  };
  return {
    _op: op,
    param,
    mm,
    mmT,
    add,
    relu,
    gather,
    attn,
    ce,
    conv3,
    pool,
    up,
    cat,
    bce,
    input,
    backward,
    clear,
    adam,
  };
})();
(() => {
  const N = (v, sh) => ({ v, g: new Float64Array(v.length), sh }),
    T = AD._op;
  AD.addb = (A, b) => {
    const m = A.sh[0],
      n = A.sh[1],
      o = N(new Float64Array(m * n), [m, n]);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) o.v[i * n + j] = A.v[i * n + j] + b.v[j];
    return T(o, () => {
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) {
          A.g[i * n + j] += o.g[i * n + j];
          b.g[j] += o.g[i * n + j];
        }
    });
  };
  AD.tanh = (A) => {
    const o = N(A.v.map(Math.tanh), A.sh);
    return T(o, () => {
      for (let i = 0; i < o.g.length; i++) A.g[i] += o.g[i] * (1 - o.v[i] * o.v[i]);
    });
  };
  AD.mse = (A, Y) => {
    const n = A.v.length,
      o = N(new Float64Array(1), [1]);
    for (let i = 0; i < n; i++) o.v[0] += (A.v[i] - Y[i]) ** 2 / n;
    return T(o, () => {
      for (let i = 0; i < n; i++) A.g[i] += (o.g[0] * 2 * (A.v[i] - Y[i])) / n;
    });
  };
  AD.reparam = (mu, lv, eps) => {
    const o = N(
      mu.v.map((m, i) => m + Math.exp(0.5 * lv.v[i]) * eps[i]),
      mu.sh,
    );
    return T(o, () => {
      for (let i = 0; i < o.g.length; i++) {
        mu.g[i] += o.g[i];
        lv.g[i] += o.g[i] * 0.5 * Math.exp(0.5 * lv.v[i]) * eps[i];
      }
    });
  };
  AD.kl = (mu, lv, rows) => {
    const o = N(new Float64Array(1), [1]);
    for (let i = 0; i < mu.v.length; i++)
      o.v[0] += (0.5 * (mu.v[i] ** 2 + Math.exp(lv.v[i]) - lv.v[i] - 1)) / rows;
    return T(o, () => {
      for (let i = 0; i < mu.v.length; i++) {
        mu.g[i] += (o.g[0] * mu.v[i]) / rows;
        lv.g[i] += (o.g[0] * 0.5 * (Math.exp(lv.v[i]) - 1)) / rows;
      }
    });
  };
  AD.wsum = (a, b, w) => {
    const o = N(Float64Array.of(a.v[0] + w * b.v[0]), [1]);
    return T(o, () => {
      a.g[0] += o.g[0];
      b.g[0] += w * o.g[0];
    });
  };
  AD.cols = (A, from, to) => {
    const m = A.sh[0],
      n = A.sh[1],
      k = to - from,
      o = N(new Float64Array(m * k), [m, k]);
    for (let i = 0; i < m; i++) for (let j = 0; j < k; j++) o.v[i * k + j] = A.v[i * n + from + j];
    return T(o, () => {
      for (let i = 0; i < m; i++)
        for (let j = 0; j < k; j++) A.g[i * n + from + j] += o.g[i * k + j];
    });
  };
  AD.mlp = (sz, rn, act) => {
    const Ws = [],
      bs = [];
    for (let l = 0; l < sz.length - 1; l++) {
      Ws.push(AD.param([sz[l], sz[l + 1]], Math.sqrt((act === 'tanh' ? 1 : 2) / sz[l]), rn));
      bs.push(AD.param([sz[l + 1]], 0, rn));
    }
    const f = act === 'tanh' ? AD.tanh : AD.relu;
    return {
      ps: Ws.concat(bs),
      fwd: (X) => {
        let h = X;
        for (let l = 0; l < Ws.length; l++) {
          h = AD.addb(AD.mm(h, Ws[l]), bs[l]);
          if (l < Ws.length - 1) h = f(h);
        }
        return h;
      },
    };
  };
})();

/* ============ deep learning ============ */
const DLL = [
  {
    id: 'convolution',
    tab: 'Convolution and pooling',
    title: 'Convolution: one small window, slid over the whole picture',
    lede: 'A network that looks at pictures does not study every pixel separately. It slides a small window across the picture and, at each stop, asks one question: does this patch look like the pattern I am searching for? The answers form a new picture that lights up wherever the pattern was found. Draw on the picture, pick a pattern, and watch the window work.',
    stage: `<div class="row seg" id="cv-k" role="group" aria-label="Pattern to search for"><button class="btn" type="button" data-v="vert">Vertical Edges</button><button class="btn" type="button" data-v="horiz">Horizontal Edges</button><button class="btn" type="button" data-v="outline">Outlines</button><button class="btn" type="button" data-v="blur">Blur</button><button class="btn" type="button" data-v="sharp">Sharpen</button></div>
 <div class="stage one"><figure class="fig"><figcaption>Left: the picture, with the window. Middle: the patch under the window, the pattern (kernel), and their products. Right: the answers so far, and below them the pooled summary. Drag on the picture to draw</figcaption><canvas class="wide" id="cv-c" role="img" aria-label="Input picture with a sliding window, the kernel, the feature map and the pooled map"></canvas><div class="eq" id="cv-eq" role="status"></div></figure></div>
 ${uHow('cv', ['Place the 3 × 3 window on the top-left corner of the picture.', 'Multiply each of the nine pixels under the window by the matching number in the kernel, and add the nine products.', 'Write the sum into the matching cell of the output. If ReLU is on, write 0 for a negative sum.', 'Slide the window one pixel and repeat, row by row, until the whole picture is covered.', 'Pooling: cut the output into 2 × 2 blocks and keep only the largest number in each. The result is a quarter of the size.'])}
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="cv-play" type="button">Slide the window</button><button class="btn" id="cv-step" type="button">One position</button><button class="btn" id="cv-reset" type="button">Start over</button></div>`)}
 ${grp('The picture', `<label class="ctl" for="cv-img"><span>Picture</span><select id="cv-img"><option value="seven">The digit 7</option><option value="box">A square</option><option value="cross">A cross</option><option value="face">A smiling face</option><option value="diag">A diagonal stripe</option></select></label><div class="row"><button class="btn" id="cv-clear" type="button">Clear the picture</button></div>`)}
 ${grp('The layer', `<label class="check"><input type="checkbox" id="cv-relu" checked> Apply ReLU: keep only positive answers</label>`)}</div>
 <dl class="read">${rd('pos', 'Window position')}${rd('sum', 'Sum at this position', 'hl')}${rd('sz', 'Sizes: picture → output → pooled')}${rd('np', 'Numbers this layer learns')}${rd('st', 'Status')}</dl>`,
    tryit:
      'With the digit 7 and Vertical Edges, watch the output: it lights up on the upright stroke and stays dark on the flat top. Switch to Horizontal Edges and the opposite happens. Clear the picture and draw a single vertical line: where is the answer positive, where negative, and what does ReLU do to it? Then look at the pooled picture: move your line one pixel sideways and see how little the pooled picture changes.',
    math: String.raw`<h4>Intuition</h4><p>Imagine looking for your keys in a dark room with a small torch. You do not light the whole room at once. You sweep the torch across it and, wherever the beam lands, ask the same question: "does this look like keys?" The torch is small, the question never changes, and it works wherever the keys happen to be. A convolutional layer does this with a picture. The window is the torch beam, the kernel is the question, and the output is a map of where the answer was yes.</p>
 <h4>The operation</h4><p>Let \(X\) be the picture and \(K\) a \(3\times3\) kernel. The output at row \(i\), column \(j\) is</p>$$Y_{ij}=\sum_{a=0}^{2}\sum_{b=0}^{2}K_{ab}\,X_{i+a,\;j+b}\;+\;\text{bias}.$$<p>It is a dot product between the kernel and the patch of picture under it: large and positive when the patch resembles the kernel, near zero when it does not, negative when it is the reverse. An \(n\times n\) picture and a \(k\times k\) kernel give an output of size \((n-k+1)\times(n-k+1)\): here \(18\to16\). (This is strictly cross-correlation; deep learning libraries call it convolution.)</p>
 <h4>Why the kernels look as they do</h4><p>The vertical-edge kernel has negative numbers on the left and positive on the right, so it adds up "right minus left". On a flat area the two cancel to zero. Where the picture changes from dark to bright going rightward the sum is large. The blur kernel is nine equal weights, an average. In a trained network nobody writes the kernels: they are the weights, and gradient descent learns them. First-layer kernels of real networks do turn out to be edge and colour-blob detectors much like these.</p>
 <h4>Two ideas that make it work</h4><p><b>Local connections.</b> Each output looks at only nine pixels, not all 324. Nearby pixels are what form edges and corners. <b>Shared weights.</b> The same nine numbers are used at every position, so the layer has 9 weights and 1 bias whatever the size of the picture, and a pattern learned in one place is recognised everywhere. A fully connected layer from 324 pixels to 256 outputs would have 83,200 weights.</p>
 <h4>ReLU and pooling</h4><p>ReLU keeps positive answers and sets negative ones to zero: "the pattern is here" matters, "its opposite is here" is dropped. <b>Max pooling</b> then keeps the largest value in each \(2\times2\) block,</p>$$P_{ij}=\max\big(Y_{2i,2j},\,Y_{2i,2j+1},\,Y_{2i+1,2j},\,Y_{2i+1,2j+1}\big),$$<p>halving the width and height. It records that the pattern was found somewhere in the block without recording exactly where, which makes the result insensitive to small shifts and cuts the work for later layers by four.</p>
 <h4>Stacking layers</h4><p>A real convolutional network has many kernels per layer, each producing its own output map, and several layers in a row. A second layer slides its windows over the maps of the first, so it combines edges into corners and curves. Each layer sees a larger area of the original picture (its <b>receptive field</b>): edges, then parts, then whole objects.</p>`,
    q: 'A convolutional layer with a 3 × 3 kernel is applied to a 1000 × 1000 picture instead of an 18 × 18 one. How many weights does the layer have now?',
    a: 'Still nine, plus one bias. The kernel is the layer, and the same nine numbers are reused at every position. Only the amount of computation and the size of the output grow with the picture. This is why convolutional networks can handle large images with few parameters.',
    init(root) {
      const c = $('#cv-c', root),
        live = $('#cv-live', root),
        eq = $('#cv-eq', root),
        N = 18,
        M = 16,
        S = { k: 'vert', img: 'seven', relu: true },
        KS = {
          vert: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
          horiz: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
          outline: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
          blur: [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9),
          sharp: [0, -1, 0, -1, 5, -1, 0, -1, 0],
        };
      let img = new Float32Array(N * N),
        out = new Float32Array(M * M),
        pos = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        GEO = null,
        paint = false;
      const load = () => {
        const o = document.createElement('canvas');
        o.width = o.height = 72;
        const g = o.getContext('2d');
        g.fillStyle = '#000';
        g.fillRect(0, 0, 72, 72);
        g.strokeStyle = g.fillStyle = '#fff';
        g.lineWidth = 7;
        g.lineCap = 'round';
        g.lineJoin = 'round';
        if (S.img === 'seven') {
          g.beginPath();
          g.moveTo(18, 16);
          g.lineTo(54, 16);
          g.lineTo(32, 58);
          g.stroke();
        } else if (S.img === 'box') {
          g.strokeRect(18, 18, 36, 36);
        } else if (S.img === 'cross') {
          g.beginPath();
          g.moveTo(36, 12);
          g.lineTo(36, 60);
          g.moveTo(12, 36);
          g.lineTo(60, 36);
          g.stroke();
        } else if (S.img === 'face') {
          g.lineWidth = 5;
          g.beginPath();
          g.arc(36, 36, 24, 0, 7);
          g.stroke();
          g.beginPath();
          g.arc(27, 29, 3, 0, 7);
          g.arc(45, 29, 3, 0, 7);
          g.fill();
          g.beginPath();
          g.arc(36, 36, 13, 0.5, 2.64);
          g.stroke();
        } else {
          g.lineWidth = 11;
          g.beginPath();
          g.moveTo(8, 62);
          g.lineTo(62, 8);
          g.stroke();
        }
        const d = g.getImageData(0, 0, 72, 72).data;
        for (let y = 0; y < N; y++)
          for (let x = 0; x < N; x++) {
            let s = 0;
            for (let a = 0; a < 4; a++)
              for (let b = 0; b < 4; b++) s += d[((y * 4 + a) * 72 + x * 4 + b) * 4];
            img[y * N + x] = Math.round((s / 16 / 255) * 10) / 10;
          }
      };
      const K = () => KS[S.k],
        at = (p) => {
          const i = Math.floor(p / M),
            j = p % M,
            k = K();
          let s = 0;
          const pr = [];
          for (let a = 0; a < 3; a++)
            for (let b = 0; b < 3; b++) {
              const x = img[(i + a) * N + j + b];
              pr.push([x, k[a * 3 + b]]);
              s += x * k[a * 3 + b];
            }
          return { i, j, s, pr };
        };
      const comp = () => {
        for (let p = 0; p < M * M; p++) {
          const s = at(p).s;
          out[p] = S.relu ? Math.max(0, s) : s;
        }
      };
      const END = M * M + 64,
        reset = () => {
          pos = 0;
          cu = 0;
          msg =
            'The window is about to start at the top-left corner. Press Slide the window or One position.';
          eq.innerHTML = '';
        };
      const nf = (v) =>
        (v < 0 ? '(−' + Math.abs(v).toFixed(2) + ')' : v.toFixed(2)).replace('.00', '');
      const say = () => {
        if (pos <= M * M && pos > 0) {
          const q = at(pos - 1);
          cu = pos === 1 ? 1 : pos < M * M ? 3 : 2;
          msg =
            'Position ' +
            pos +
            ' of ' +
            M * M +
            ' (row ' +
            (q.i + 1) +
            ', column ' +
            (q.j + 1) +
            '): the nine products add up to ' +
            f2(q.s) +
            '.' +
            (S.relu && q.s < 0
              ? ' Negative, so ReLU writes 0.'
              : q.s > 1
                ? ' A strong match with the pattern.'
                : Math.abs(q.s) < 0.05
                  ? ' No match: this patch is flat.'
                  : '');
          eq.innerHTML =
            [0, 1, 2]
              .map(
                (a) =>
                  '<div class="on">' +
                  [0, 1, 2]
                    .map((b) => nf(q.pr[a * 3 + b][0]) + '·' + nf(q.pr[a * 3 + b][1]))
                    .join('  +  ') +
                  (a < 2 ? '  +' : '') +
                  '</div>',
              )
              .join('') +
            '<div class="on">= <b>' +
            f2(q.s) +
            '</b>' +
            (S.relu ? '  →  after ReLU: <b>' + f2(Math.max(0, q.s)) + '</b>' : '') +
            '</div>';
        } else if (pos > M * M) {
          const b = pos - M * M - 1,
            bi = Math.floor(b / 8),
            bj = b % 8,
            vals = [0, 1, 2, 3].map((q) => out[(2 * bi + (q >> 1)) * M + 2 * bj + (q & 1)]);
          cu = 4;
          msg =
            'Pooling block ' +
            (b + 1) +
            ' of 64: the four answers are ' +
            vals.map(f2).join(', ') +
            '. Keep the largest, ' +
            f2(Math.max(...vals)) +
            '.' +
            (pos >= END
              ? ' Finished: an 18 × 18 picture became a 16 × 16 map of where the pattern is, and then an 8 × 8 summary.'
              : '');
          eq.innerHTML =
            '<div class="on">max( ' +
            vals.map(f2).join(' ,  ') +
            ' ) = <b>' +
            f2(Math.max(...vals)) +
            '</b></div>';
        }
      };
      const step = () => {
        if (pos >= END) {
          reset();
          return;
        }
        pos++;
        say();
      };
      const draw = () => {
        const t = T(),
          { g, w, h } = fit(c),
          A = Math.min(h - 34, w * 0.36),
          ca = A / N,
          x0 = 8,
          y0 = 22,
          B = Math.min(h - 34, w * 0.3),
          cb = B / M,
          xo = w - B - w * 0.13 - 12,
          xp = w - w * 0.12 - 4,
          Pp = w * 0.12,
          cp = Pp / 8,
          xm = x0 + A + 18,
          wm = xo - xm - 18;
        GEO = { x0, y0, ca };
        g.font = MONO;
        const gray = (v) => mix(t.bg, t.ink, 0.07 + 0.93 * clamp(v, 0, 1)),
          div = (v, m) =>
            v >= 0 ? mix(t.bg, t.gold, clamp(v / m, 0, 1)) : mix(t.bg, t.ok, clamp(-v / m, 0, 1));
        let om = 1e-6;
        for (const v of out) om = Math.max(om, Math.abs(v));
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText('picture, 18 × 18', x0, 14);
        g.fillText('output, 16 × 16', xo, 14);
        g.fillText('pooled, 8 × 8', xp, 14);
        for (let p = 0; p < N * N; p++) {
          g.fillStyle = gray(img[p]);
          g.fillRect(x0 + (p % N) * ca, y0 + Math.floor(p / N) * ca, ca - 0.6, ca - 0.6);
        }
        const shown = Math.min(pos, M * M);
        for (let p = 0; p < M * M; p++) {
          g.fillStyle = p < shown ? div(out[p], om) : mix(t.ink, t.ink, 0, 0.05);
          g.fillRect(xo + (p % M) * cb, y0 + Math.floor(p / M) * cb, cb - 0.6, cb - 0.6);
        }
        const np = Math.max(0, pos - M * M);
        for (let b = 0; b < 64; b++) {
          const bi = Math.floor(b / 8),
            bj = b % 8;
          let v = -1e9;
          for (let q = 0; q < 4; q++)
            v = Math.max(v, out[(2 * bi + (q >> 1)) * M + 2 * bj + (q & 1)]);
          g.fillStyle = b < np ? div(v, om) : mix(t.ink, t.ink, 0, 0.05);
          g.fillRect(xp + bj * cp, y0 + bi * cp, cp - 0.6, cp - 0.6);
        }
        if (pos > 0 && pos <= M * M) {
          const q = at(pos - 1);
          g.strokeStyle = t.gold;
          g.lineWidth = 3;
          g.strokeRect(x0 + q.j * ca - 1, y0 + q.i * ca - 1, 3 * ca + 1.4, 3 * ca + 1.4);
          g.strokeRect(xo + q.j * cb - 1, y0 + q.i * cb - 1, cb + 1.4, cb + 1.4);
          const cz = Math.min(wm / 3.3, (h - 70) / 10.2),
            mx = xm + (wm - 3 * cz) / 2,
            k = K(),
            km = Math.max(...k.map(Math.abs));
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('patch under the window', xm + wm / 2, y0 + 6);
          for (let a = 0; a < 3; a++)
            for (let b = 0; b < 3; b++) {
              const X = mx + b * cz,
                Y1 = y0 + 12 + a * cz,
                Y2 = y0 + 30 + 3 * cz + a * cz,
                Y3 = y0 + 48 + 6 * cz + a * cz,
                pv = q.pr[a * 3 + b];
              g.fillStyle = gray(pv[0]);
              g.fillRect(X, Y1, cz - 2, cz - 2);
              g.fillStyle = pv[0] > 0.5 ? t.bg : t.ink;
              g.fillText(nf(pv[0]), X + cz / 2 - 1, Y1 + cz / 2 + 3);
              g.fillStyle = div(pv[1], km);
              g.fillRect(X, Y2, cz - 2, cz - 2);
              g.strokeStyle = t.line;
              g.lineWidth = 1;
              g.strokeRect(X, Y2, cz - 2, cz - 2);
              g.fillStyle = t.ink;
              g.fillText(nf(pv[1]), X + cz / 2 - 1, Y2 + cz / 2 + 3);
              const pd = pv[0] * pv[1];
              g.fillStyle = div(pd, Math.max(1, km));
              g.fillRect(X, Y3, cz - 2, cz - 2);
              g.strokeStyle = t.line;
              g.strokeRect(X, Y3, cz - 2, cz - 2);
              g.fillStyle = t.ink;
              g.fillText(nf(pd), X + cz / 2 - 1, Y3 + cz / 2 + 3);
            }
          g.fillStyle = t.muted;
          g.fillText('× the kernel', xm + wm / 2, y0 + 24 + 3 * cz);
          g.fillText('= products, which add up to', xm + wm / 2, y0 + 42 + 6 * cz);
          g.font = '600 20px "Hanken Grotesk",sans-serif';
          g.fillStyle = t.cut;
          g.fillText(f2(q.s), xm + wm / 2, Math.min(h - 6, y0 + 72 + 9 * cz));
          g.font = MONO;
          g.strokeStyle = mix(t.gold, t.gold, 0, 0.6);
          g.lineWidth = 1.2;
          g.beginPath();
          g.moveTo(x0 + (q.j + 3) * ca, y0 + (q.i + 1.5) * ca);
          g.lineTo(mx, y0 + 12 + 1.5 * cz);
          g.moveTo(mx + 3 * cz, y0 + 48 + 7.5 * cz);
          g.lineTo(xo + q.j * cb, y0 + (q.i + 0.5) * cb);
          g.stroke();
        } else if (pos > M * M) {
          const b = pos - M * M - 1,
            bi = Math.floor(b / 8),
            bj = b % 8;
          g.strokeStyle = t.gold;
          g.lineWidth = 3;
          g.strokeRect(xo + 2 * bj * cb - 1, y0 + 2 * bi * cb - 1, 2 * cb + 1.4, 2 * cb + 1.4);
          g.strokeRect(xp + bj * cp - 1, y0 + bi * cp - 1, cp + 1.4, cp + 1.4);
          g.lineWidth = 1.2;
          g.beginPath();
          g.moveTo(xo + (2 * bj + 2) * cb, y0 + (2 * bi + 1) * cb);
          g.lineTo(xp + bj * cp, y0 + (bi + 0.5) * cp);
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('keep the largest of each 2 × 2 block →', xm + wm / 2, h / 2);
        } else {
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('the 3 × 3 window will stop', xm + wm / 2, h / 2 - 8);
          g.fillText('at 256 positions', xm + wm / 2, h / 2 + 8);
        }
        g.fillStyle = t.muted;
        g.textAlign = 'left';
        g.fillText(
          'gold: pattern found.  red: its opposite found.',
          xo,
          Math.min(h - 4, y0 + B + 14),
        );
        live.textContent = msg;
        uCur(root, 'cv', cu);
        const q = pos > 0 && pos <= M * M ? at(pos - 1) : null;
        setR(
          root,
          'pos',
          pos > M * M ? 'pooling ' + (pos - M * M) + ' of 64' : pos + ' of ' + M * M,
        );
        setR(root, 'sum', q ? f2(q.s) : '–');
        setR(root, 'sz', '18 × 18 → 16 × 16 → 8 × 8');
        setR(root, 'np', '9 weights + 1 bias');
        setR(root, 'st', pos >= END ? 'Finished' : pos ? 'Sliding' : 'Not started');
      };
      load();
      comp();
      pos = END;
      say();
      const anim = Anim(() => {
        fr++;
        const slow = pos < 8 || (pos > M * M && pos < M * M + 5);
        if (fr % (slow ? 32 : pos > M * M ? 5 : 2)) return;
        step();
        draw();
        if (pos >= END) pb.stop();
      });
      pb = playBtn($('#cv-play', root), anim);
      runCtl(pb, root, 'cv', { done: () => pos >= END, restart: () => reset() });
      $('#cv-step', root).onclick = () => {
        pb.stop();
        step();
        draw();
      };
      $('#cv-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      const redo = () => {
        pb.stop();
        comp();
        reset();
        draw();
        pb.start();
      };
      segBind(root, 'cv-k', S.k, (v) => {
        S.k = v;
        redo();
      });
      $('#cv-img', root).onchange = (e) => {
        S.img = e.target.value;
        load();
        redo();
      };
      $('#cv-relu', root).onchange = (e) => {
        S.relu = e.target.checked;
        redo();
      };
      $('#cv-clear', root).onclick = () => {
        img.fill(0);
        pb.stop();
        comp();
        pos = END;
        say();
        draw();
      };
      const pt = (e) => {
        if (!GEO) return;
        const r = c.getBoundingClientRect(),
          x = Math.floor((e.clientX - r.left - GEO.x0) / GEO.ca),
          y = Math.floor((e.clientY - r.top - GEO.y0) / GEO.ca);
        if (x < 0 || y < 0 || x >= N || y >= N) return false;
        img[y * N + x] = 1;
        return true;
      };
      c.style.touchAction = 'none';
      c.addEventListener('pointerdown', (e) => {
        if (pt(e) === false) return;
        pb.stop();
        paint = true;
        c.setPointerCapture(e.pointerId);
        comp();
        pos = END;
        say();
        draw();
      });
      c.addEventListener('pointermove', (e) => {
        if (!paint) return;
        pt(e);
        comp();
        draw();
      });
      ['pointerup', 'pointercancel'].forEach((n) =>
        c.addEventListener(n, () => {
          if (paint) {
            paint = false;
            say();
            draw();
          }
        }),
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
    id: 'residual',
    tab: 'Residual connections',
    title: 'Residual connections: a shortcut around every layer',
    lede: 'Very deep networks used to be untrainable, because the learning signal faded before it reached the first layers. The cure was almost embarrassingly simple: let each block add its result to its own input, so that there is always a direct road past it. Send the learning signal back through a plain stack and a stack with shortcuts, and compare what arrives.',
    stage: `<div class="stage one"><figure class="fig"><figcaption>Top row: a plain stack of layers. Bottom row: the same layers with a shortcut around each. The learning signal starts on the right; the size of the dot is its strength</figcaption><canvas class="tall" id="rs-c" style="aspect-ratio:16/5" role="img" aria-label="Plain and residual stacks with the learning signal travelling backward"></canvas></figure></div>
 <div class="stage" style="margin-top:18px"><figure class="fig"><figcaption>Strength of the learning signal on arriving at each layer, compared with where it started</figcaption><canvas class="tall" id="rs-r" role="img" aria-label="Gradient strength by layer for plain and residual networks"></canvas></figure>
 <figure class="fig"><figcaption>What one block computes</figcaption><canvas class="tall" id="rs-d" role="img" aria-label="Diagram of a plain block and a residual block"></canvas></figure></div>
 ${uHow('rs', ['Build two stacks from the same random weights: one plain, one where every block adds its output to its input.', 'Start a learning signal of strength 1 at the last layer of both.', 'Plain stack: stepping back one layer multiplies the signal by the weights and the slope of the activation. It usually shrinks.', 'Residual stack: the signal splits. One part goes back through the layer as before. The other takes the shortcut, unchanged. They add.', 'Compare what reaches the first layer.'])}
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="rs-play" type="button">Send the signal back</button><button class="btn" id="rs-step" type="button">One layer</button><button class="btn" id="rs-reset" type="button">Start over</button></div>`)}
 ${grp('The network', slider('rs-d', 'Number of layers', 4, 60, 1, 20) + slider('rs-v', 'Size of the weights (1 = standard)', 0.3, 1.6, 0.05, 0.8) + `<label class="ctl" for="rs-act"><span>Activation</span><select id="rs-act"><option value="tanh">tanh</option><option value="relu">ReLU</option><option value="sigmoid">Sigmoid</option></select></label>`)}</div>
 <dl class="read">${rd('pl', 'Plain stack: signal at the first layer', 'hl2')}${rd('rs', 'With shortcuts: signal at the first layer', 'hl3')}${rd('pp', 'Plain: average change per layer')}${rd('rp', 'With shortcuts: average change per layer')}</dl>`,
    tryit:
      'Run it at 20 layers, then at 60. In the plain stack the dot shrinks to nothing; with shortcuts it arrives intact. Switch the activation to Sigmoid: the plain stack is hopeless, yet the shortcuts still deliver. Now raise the size of the weights to 1.6. What happens to the signal with shortcuts, and why do real residual networks also use normalisation?',
    math: String.raw`<h4>Intuition</h4><p>Think of a message passed along a line of sixty people, each of whom rewrites it in their own words. By the end it is unrecognisable. Now change one rule: each person passes on the original note untouched and staples their own comment to it. The original always arrives, and each person only has to add what they think is missing. A residual block works like that. Its layers do not have to reproduce their whole input, only to compute a correction to it, and the input itself travels along the shortcut.</p>
 <h4>The block</h4><p>A plain layer computes \(\mathbf a_{l+1}=F(\mathbf a_l)\), where \(F\) is "multiply by weights, apply the activation". A residual block computes</p>$$\mathbf a_{l+1}=\mathbf a_l+F(\mathbf a_l).$$<p>If the best thing a block can do is nothing, it only has to learn \(F=0\), which is easy. A plain layer would have to learn to copy its input exactly, which is hard.</p>
 <h4>Why the gradient survives</h4><p>Differentiate the block:</p>$$\frac{\partial\mathbf a_{l+1}}{\partial\mathbf a_l}=\mathbf I+\frac{\partial F}{\partial\mathbf a_l}.$$<p>Going back through \(L\) plain layers multiplies \(L\) factors \(\partial F/\partial\mathbf a\), and if each is a little smaller than 1 in size the product vanishes geometrically. With shortcuts each factor is \(\mathbf I+\partial F/\partial\mathbf a\). Multiplying these out gives the identity plus many other terms, so there is always a path from the loss to every layer along which the signal is multiplied by exactly 1. The gradient at layer \(l\) is</p>$$\frac{\partial\mathcal L}{\partial\mathbf a_l}=\frac{\partial\mathcal L}{\partial\mathbf a_L}\Big(\mathbf I+\text{terms that pass through one or more blocks}\Big).$$
 <h4>What it made possible</h4><p>Before residual connections (He and others, 2015), adding layers beyond about twenty made networks worse, even on the training data: an optimisation failure, not overfitting. With them, networks of 50, 150 and more than 1000 layers trained successfully. Every Transformer block also has two shortcuts, one around attention and one around its feed-forward layers, and U-Nets use long shortcuts between matching layers.</p>
 <h4>The other side</h4><p>The shortcut stops the signal from vanishing, but if the blocks' own contribution is large the sum can grow layer after layer, forward and backward. That is why residual networks pair shortcuts with batch or layer normalisation and with careful initialisation, often starting each block close to \(F=0\).</p>
 <h4>What this page computes</h4><p>Both rows use the same random weight matrices, 30 neurons wide. The residual row scales each block's contribution by one half. The bars are the root-mean-square size of the gradient at each layer, computed with the formulas above for thirty random inputs.</p>`,
    q: 'In a residual network, what does a block compute if all of its weights are zero, and why is that a good place to start training?',
    a: 'It computes a(l+1) = a(l) + 0: the block passes its input through unchanged. A network made of such blocks is the identity from first block to last, so the signal and the gradient both travel through perfectly at the start. Training then only has to add useful corrections, block by block, instead of first repairing a scrambled signal.',
    init(root) {
      const c = $('#rs-c', root),
        cr = $('#rs-r', root),
        cd = $('#rs-d', root),
        live = $('#rs-live', root),
        S = { d: 20, v: 0.8, act: 'tanh' };
      let Gp = [],
        Gr = [],
        k = 0,
        msg = '',
        cu = 0,
        fr = 0,
        t0 = 0,
        pb;
      const comp = () => {
        const n = 30,
          bt = 30,
          D = S.d,
          r = randnFrom(rng(5)),
          f = ACT[S.act][0],
          df = ACT[S.act][1],
          W = [],
          sdv = S.v / Math.sqrt(n),
          bta = 0.5;
        for (let l = 0; l < D; l++)
          W.push(
            Array.from({ length: n }, () => Float64Array.from({ length: n }, () => sdv * r())),
          );
        const flow = (res) => {
          let A = [];
          for (let b = 0; b < bt; b++) A.push(Float64Array.from({ length: n }, () => r()));
          const Zs = [];
          for (let l = 0; l < D; l++) {
            const Z = [],
              NA = [];
            for (let b = 0; b < bt; b++) {
              const z = new Float64Array(n),
                a = new Float64Array(n);
              for (let j = 0; j < n; j++) {
                let s = 0;
                for (let i = 0; i < n; i++) s += W[l][j][i] * A[b][i];
                z[j] = s;
                a[j] = res ? A[b][j] + bta * f(s) : f(s);
              }
              Z.push(z);
              NA.push(a);
            }
            Zs.push(Z);
            A = NA;
          }
          let Dl = [];
          for (let b = 0; b < bt; b++) Dl.push(Float64Array.from({ length: n }, () => r()));
          const G = new Array(D + 1).fill(0),
            rms = (Q) => {
              let s = 0;
              for (const q of Q) for (const u of q) s += u * u;
              return Math.sqrt(s / (n * bt));
            };
          G[D] = rms(Dl);
          for (let l = D - 1; l >= 0; l--) {
            const ND = [];
            for (let b = 0; b < bt; b++) {
              const da = new Float64Array(n);
              for (let i = 0; i < n; i++) {
                let q = 0;
                for (let j = 0; j < n; j++) q += W[l][j][i] * Dl[b][j] * df(Zs[l][b][j]);
                da[i] = res ? Dl[b][i] + bta * q : q;
              }
              ND.push(da);
            }
            Dl = ND;
            G[l] = rms(Dl);
          }
          return G.map((q) => q / G[D]);
        };
        Gp = flow(false);
        Gr = flow(true);
        k = 0;
        cu = 0;
        msg =
          'Two stacks of ' +
          S.d +
          ' layers are ready, built from the same weights. Press Send the signal back or One layer.';
      };
      const step = () => {
        if (k >= S.d) {
          k = 0;
          cu = 0;
          return;
        }
        k++;
        t0 = fr;
        const l = S.d - k;
        cu = k === 1 ? 2 : k >= S.d ? 4 : 3;
        msg =
          'Back to layer ' +
          (l + 1) +
          '. Plain: the signal was multiplied by ' +
          sci(Gp[l] / Gp[l + 1]) +
          ' and is now ' +
          sci(Gp[l]) +
          ' of its starting strength. With shortcuts: multiplied by ' +
          sci(Gr[l] / Gr[l + 1]) +
          ', now ' +
          sci(Gr[l]) +
          '.' +
          (k >= S.d
            ? ' At the first layer the plain stack has ' +
              sci(Gp[0]) +
              ' left and the stack with shortcuts has ' +
              sci(Gr[0]) +
              '.'
            : '');
      };
      const draw = () => {
        const t = T(),
          D = S.d,
          u = RM ? 1 : clamp((fr - t0) / 22, 0, 1);
        {
          const { g, w, h } = fit(c),
            m = 30,
            bw = (w - 2 * m) / D,
            rowY = [h * 0.27, h * 0.72],
            rad = (v) => clamp(13 + 3.4 * Math.log10(v + 1e-300), 1.5, 24);
          g.font = MONO;
          [
            [Gp, 'plain stack', t.ok, false],
            [Gr, 'with shortcuts', t.accent, true],
          ].forEach(([G, lab, cl, res], ri) => {
            const y = rowY[ri];
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText(lab, m, y - h * 0.2);
            g.strokeStyle = t.muted;
            g.lineWidth = 1.2;
            g.beginPath();
            g.moveTo(m, y);
            g.lineTo(w - m, y);
            g.stroke();
            for (let l = 0; l < D; l++) {
              const x = m + l * bw + bw * 0.2,
                ww = bw * 0.6;
              g.fillStyle = t.bg;
              g.strokeStyle = t.ink;
              g.lineWidth = 1.2;
              g.beginPath();
              g.rect(x, y - 11, ww, 22);
              g.fill();
              g.stroke();
              if (res) {
                g.strokeStyle = mix(cl, cl, 0, 0.8);
                g.lineWidth = 1.4;
                g.beginPath();
                g.moveTo(m + l * bw + bw * 0.06, y);
                g.quadraticCurveTo(
                  m + l * bw + bw * 0.5,
                  y - h * 0.17,
                  m + (l + 1) * bw - bw * 0.06,
                  y,
                );
                g.stroke();
              }
              if (D <= 24 || l % 5 === 0) {
                g.fillStyle = t.muted;
                g.textAlign = 'center';
                g.fillText(String(l + 1), x + ww / 2, y + 28);
              }
            }
            if (k > 0) {
              const pos = D - k + (1 - u),
                li = Math.min(D, Math.round(pos)),
                v = G[D - k] * u + G[Math.min(D, D - k + 1)] * (1 - u),
                x = m + pos * bw;
              g.fillStyle = cl;
              g.beginPath();
              g.arc(x, y, rad(v), 0, 7);
              g.fill();
              g.fillStyle = t.ink;
              g.textAlign = 'center';
              g.fillText(sci(G[D - k]), clamp(x, 60, w - 60), y + h * 0.19);
            } else {
              g.fillStyle = cl;
              g.beginPath();
              g.arc(w - m, y, rad(1), 0, 7);
              g.fill();
            }
          });
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('← the learning signal travels this way', w - m, h - 6);
        }
        {
          const { g, w, h } = fit(cr),
            m = { l: 54, r: 10, t: 16, b: 22 },
            bw = (w - m.l - m.r) / D;
          let lo = -3,
            hi = 1;
          [Gp, Gr].forEach((G) =>
            G.forEach((v) => {
              const q = Math.log10(v + 1e-300);
              lo = Math.min(lo, Math.floor(q));
              hi = Math.max(hi, Math.ceil(q));
            }),
          );
          lo = Math.max(lo, -16);
          hi = Math.min(hi, 12);
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
            const on = l >= D - k,
              x = m.l + l * bw;
            [
              [Gp, t.ok, 0.08],
              [Gr, t.accent, 0.5],
            ].forEach(([G, cl, o]) => {
              const y1 = Y(1),
                y2 = on ? Y(G[l]) : y1;
              g.fillStyle = on ? cl : mix(t.ink, t.ink, 0, 0.07);
              g.fillRect(
                x + bw * o,
                Math.min(y1, y2) - (on ? 0 : 1),
                bw * 0.4,
                Math.max(2, Math.abs(y2 - y1)),
              );
            });
          }
          g.textAlign = 'left';
          g.fillStyle = t.ok;
          g.fillText('red: plain', m.l + 6, m.t + 2);
          g.fillStyle = t.accent;
          g.fillText('green: with shortcuts', m.l + 90, m.t + 2);
          g.fillStyle = t.muted;
          g.fillText('first layer', m.l, h - 5);
          g.textAlign = 'right';
          g.fillText('last layer', w - m.r, h - 5);
        }
        {
          const { g, w, h } = fit(cd),
            bx = (x, y, tx) => {
              g.fillStyle = t.bg;
              g.strokeStyle = t.ink;
              g.lineWidth = 1.4;
              g.beginPath();
              g.rect(x - 58, y - 15, 116, 30);
              g.fill();
              g.stroke();
              g.font = MONO;
              g.fillStyle = t.ink;
              g.textAlign = 'center';
              g.fillText(tx, x, y + 4);
            },
            ar = (x1, y1, x2, y2) => {
              g.strokeStyle = t.ink;
              g.lineWidth = 1.4;
              g.beginPath();
              g.moveTo(x1, y1);
              g.lineTo(x2, y2);
              g.stroke();
            },
            ph = (fr / 70) % 1;
          g.font = MONO;
          const xa = w * 0.25,
            xb = w * 0.72;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('plain', xa, 14);
          g.fillText('with a shortcut', xb, 14);
          ar(xa, 28, xa, h - 24);
          bx(xa, h * 0.5, 'weights, activation');
          g.fillStyle = t.ink;
          g.fillText('input a', xa, 28);
          g.fillText('output F(a)', xa, h - 10);
          ar(xb, 28, xb, h - 24);
          bx(xb, h * 0.45, 'weights, activation');
          g.strokeStyle = t.accent;
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(xb, 40);
          g.bezierCurveTo(xb + 120, 40, xb + 120, h * 0.74, xb + 10, h * 0.74);
          g.stroke();
          g.fillStyle = t.bg;
          g.strokeStyle = t.ink;
          g.lineWidth = 1.4;
          g.beginPath();
          g.arc(xb, h * 0.74, 10, 0, 7);
          g.fill();
          g.stroke();
          g.fillStyle = t.ink;
          g.fillText('+', xb, h * 0.74 + 4);
          g.fillText('input a', xb, 28);
          g.fillText('output a + F(a)', xb, h - 10);
          g.fillStyle = t.accent;
          g.textAlign = 'left';
          g.fillText('shortcut: a, unchanged', Math.min(w - 150, xb + 62), h * 0.3);
          if (pb && pb.on) {
            g.fillStyle = t.gold;
            [xa, xb].forEach((x) => {
              g.beginPath();
              g.arc(x, h - 24 - (h - 52) * ph, 5, 0, 7);
              g.fill();
            });
            const q = 1 - ph,
              bz = (a, b2, cc, d) =>
                q ** 3 * a + 3 * q * q * ph * b2 + 3 * q * ph * ph * cc + ph ** 3 * d;
            g.fillStyle = t.accent;
            g.beginPath();
            g.arc(bz(xb + 10, xb + 120, xb + 120, xb), bz(h * 0.74, h * 0.74, 40, 40), 5, 0, 7);
            g.fill();
          }
        }
        live.textContent = msg;
        uCur(root, 'rs', cu);
        const fin = k >= D;
        setR(root, 'pl', fin ? sci(Gp[0]) : '–');
        setR(root, 'rs', fin ? sci(Gr[0]) : '–');
        setR(root, 'pp', '× ' + sci(Math.pow(Gp[0], 1 / D)));
        setR(root, 'rp', '× ' + sci(Math.pow(Gr[0], 1 / D)));
      };
      comp();
      k = S.d;
      const anim = Anim(() => {
        fr++;
        if (fr - t0 >= (S.d > 30 ? 10 : 24) && k < S.d) step();
        draw();
        if (k >= S.d && fr - t0 > 26) pb.stop();
      });
      pb = playBtn($('#rs-play', root), anim);
      runCtl(pb, root, 'rs', {
        done: () => k >= S.d,
        restart: () => {
          k = 0;
        },
      });
      $('#rs-step', root).onclick = () => {
        pb.stop();
        step();
        t0 = fr - 100;
        draw();
      };
      $('#rs-reset', root).onclick = () => {
        pb.stop();
        comp();
        draw();
      };
      const redo = () => {
        pb.stop();
        comp();
        draw();
        pb.start();
      };
      bind(
        root,
        'rs-d',
        (v) => {
          S.d = v;
          pb.stop();
          comp();
          k = S.d;
          draw();
        },
        (v) => String(v),
      );
      bind(root, 'rs-v', (v) => {
        S.v = v;
        pb.stop();
        comp();
        k = S.d;
        draw();
      });
      $('#rs-act', root).onchange = (e) => {
        S.act = e.target.value;
        redo();
      };
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
    id: 'rnn',
    tab: 'Recurrent networks',
    title: 'Recurrent networks: reading one letter at a time, with a memory',
    lede: 'Your phone suggests the next word as you type. To do that it has to remember what came before. A recurrent network reads a sequence one item at a time and keeps a small memory that it updates at every step. Here one learns to spell a phrase you choose, letter by letter, and then writes it back on its own.',
    stage: `<p class="cap" id="rn-out" role="status"></p>
 <div class="stage"><figure class="fig"><figcaption>Unrolled in time. Bottom: the letter read. Middle: the memory, one coloured cell per neuron. Top: the letter it expects next. The memory is passed along to the right</figcaption><canvas class="short" id="rn-c" role="img" aria-label="Recurrent network unrolled over time steps"></canvas></figure>
 <figure class="fig"><figcaption>Top: how likely it thinks each letter is to come next. Bottom: the error while it learns</figcaption><canvas class="short" id="rn-r" role="img" aria-label="Next-letter probabilities and the training loss"></canvas></figure></div>
 ${uHow('rn', ['Learn: read the phrase many times. At each letter, guess the next one, measure the error, and adjust the weights (backpropagation through time).', 'Write: start with the first letter and an empty memory.', 'Mix the letter just read with the memory so far. That mixture is the new memory.', 'From the new memory, work out how likely each letter is to come next, and take the most likely.', 'Feed that letter back in as the next input, and repeat.'])}
 <div class="controls">${uBtns('rn', 'Run', 'One step', 'readings')}
 ${grp('The phrase', `<label class="ctl" for="rn-text"><span>Phrase to learn (letters and spaces, up to 28)</span><input type="text" class="num" id="rn-text" maxlength="28" value="hello world" autocomplete="off" spellcheck="false"></label><p class="status" id="rn-st" role="status"></p>`)}
 ${grp('The model', slider('rn-H', 'Memory cells (hidden neurons)', 1, 24, 1, 12))}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('ep', 'Times it has read the phrase')}${rd('L', 'Error per letter', 'hl2')}${rd('acc', 'Letters it now predicts correctly', 'hl')}${rd('np', 'Weights')}</dl>`,
    tryit:
      'Let it learn “hello world” and watch the line of text above the pictures improve from nonsense. In the phrase, the letter “l” is followed once by “l”, once by “o” and once by “d”. Step through the writing and look at the memory column each time it reads an “l”: the cells differ, and that is how it knows which one it is on. Now set the memory to 1 cell and run again. Then type your own phrase.',
    math: String.raw`<h4>Intuition</h4><p>When you read a sentence you do not start afresh at every word. You carry a running sense of what the sentence has said so far, and each new word updates it. A recurrent network (RNN) has exactly one such running summary: a vector called the hidden state. At every step it takes the new item and the old summary, mixes them, and produces a new summary. Nothing else is remembered, so whatever the network needs to know about the past has to be squeezed into that vector.</p>
 <h4>One step</h4><p>Let \(\mathbf x_t\) be the letter at step \(t\), written as a vector with a 1 in that letter's position and 0 elsewhere. The memory \(\mathbf h_t\) and the prediction \(\hat{\mathbf y}_t\) are</p>$$\mathbf h_t=\tanh\big(\mathbf W_{xh}\mathbf x_t+\mathbf W_{hh}\mathbf h_{t-1}+\mathbf b_h\big),\qquad\hat{\mathbf y}_t=\operatorname{softmax}\big(\mathbf W_{hy}\mathbf h_t+\mathbf b_y\big).$$<p>The same three weight matrices are used at every step. That is what "recurrent" means, and it is why the network can read a sequence of any length with a fixed number of weights.</p>
 <h4>Learning: backpropagation through time</h4><p>The loss is the cross-entropy between the predicted next letter and the true one, added up over the phrase: \(\mathcal L=-\sum_t\log\hat y_t[\text{true next letter}]\). To get gradients, unroll the network in time, as in the left picture, and treat it as a deep network with one layer per step and shared weights. Blame flows from each prediction back through the memory to all earlier steps:</p>$$\frac{\partial\mathcal L}{\partial\mathbf h_t}=\mathbf W_{hy}^{\top}\big(\hat{\mathbf y}_t-\mathbf y_t\big)+\mathbf W_{hh}^{\top}\Big[\big(1-\mathbf h_{t+1}^2\big)\odot\frac{\partial\mathcal L}{\partial\mathbf h_{t+1}}\Big].$$<p>The gradient of each shared matrix is the sum of its gradients at all steps.</p>
 <h4>Why memory is needed</h4><p>In "hello world" the letter l is followed by l, by o and by d. A model that sees only the current letter cannot do better than guess among them. The RNN can, because by the time it reads each l its memory holds a trace of the letters before it. The number of memory cells limits how much can be kept: with too few, different histories collapse onto the same memory and the predictions blur.</p>
 <h4>The weakness, and LSTM and GRU</h4><p>Going back \(k\) steps multiplies the gradient by \(\mathbf W_{hh}^{\top}\) and by tanh slopes \(k\) times, the same repeated product as in the Activation Functions lesson. Over long gaps it vanishes, so a plain RNN learns short-range patterns well and long-range ones poorly. The <b>LSTM</b> and the <b>GRU</b> add gates, small sigmoid layers that decide at each step how much of the old memory to keep, how much new information to write and how much to reveal. The LSTM's memory update is \(\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t\): when the forget gate \(\mathbf f_t\) is near 1 the memory, and the gradient with it, passes through unchanged, much like a residual shortcut through time. Those gated versions are not animated on this page.</p>
 <h4>Where RNNs stand now</h4><p>For language, Transformers have largely replaced RNNs because attention lets every position look directly at every other. RNNs are still used for streaming signals and small devices, and the idea of a state carried through time appears throughout machine learning.</p>`,
    q: 'The network has learned “hello world” and is writing it back. It has just read the second “l”. The input is the same as for the first “l”, yet it now predicts “o” instead of “l”. Where is the difference stored?',
    a: 'In the memory (the hidden state). The prediction depends on the new memory, and the new memory is computed from the current letter and the previous memory. Before the first l the previous memory encoded “he”; before the second it encoded “hel”. Same input, different previous memory, so a different new memory and a different prediction.',
    init(root) {
      const c = $('#rn-c', root),
        cr = $('#rn-r', root),
        live = $('#rn-live', root),
        outEl = $('#rn-out', root),
        st = $('#rn-st', root),
        S = { text: 'hello world', H: 12 },
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
        gen = [],
        hs = [],
        probs = [],
        done = false,
        msg = '',
        cu = 0,
        fr = 0,
        t0 = 0,
        pb,
        chk = () => false,
        run = 1,
        ts = 0;
      const mat = (r, cn, sd, rn) =>
          Array.from({ length: r }, () => Float64Array.from({ length: cn }, () => sd * rn())),
        zl = (A) => A.map((a) => new Float64Array(a.length));
      const build = () => {
        const tx =
          (S.text
            .toLowerCase()
            .replace(/[^a-z ]/g, '')
            .replace(/\s+/g, ' ')
            .trim() || 'hello world') + ' ';
        voc = [...new Set(tx)].sort();
        V = voc.length;
        seq = [...tx].map((ch) => voc.indexOf(ch));
        n = seq.length;
        const rn = randnFrom(rng(run * 7 + 3)),
          H = S.H;
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
        gen = [];
        hs = [];
        probs = [];
        done = false;
        ts = 0;
        cu = 0;
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
        const H = S.H,
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
        let h = new Float64Array(S.H),
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
        const H = S.H,
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
      const right = () => {
        let h = new Float64Array(S.H),
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
          if (
            L < 0.03 ||
            ep >= EPM ||
            (ep > 400 && Math.abs(LL[ep - 1] - LL[ep - 201]) < 0.002 * L)
          )
            trained = true;
        }
        const w = write(n);
        outEl.textContent = 'What it writes now: “' + w + '”';
        cu = 0;
        msg =
          'Stage 1, learning. It has read the phrase ' +
          ep +
          ' times. Error per letter: ' +
          f3(LL[ep - 1]) +
          '. It gets ' +
          right() +
          ' of the ' +
          n +
          ' next letters right.' +
          (trained
            ? LL[ep - 1] < 0.03
              ? ' Converged: the error is close to zero, so learning stops. Now it will write the phrase from memory.'
              : ' The error has stopped falling' +
                (right() < n
                  ? ', and it still makes mistakes: this memory is too small for this phrase.'
                  : '.') +
                ' Now it will write from memory.'
            : '');
      };
      const wstep = () => {
        if (!gen.length) {
          gen = [seq[0]];
          hs = [new Float64Array(S.H)];
          probs = [];
          cu = 1;
          msg =
            'Stage 2, writing. It is given only the first letter, “' +
            voc[seq[0]] +
            '”, and an empty memory.';
          t0 = fr;
          return;
        }
        const x = gen[gen.length - 1],
          q = cell(x, hs[hs.length - 1]),
          nx = q.y.indexOf(Math.max(...q.y));
        hs.push(q.h);
        probs.push(q.y);
        gen.push(nx);
        t0 = fr;
        cu = gen.length < 4 ? 2 + ((gen.length - 2) % 3) : 4;
        const srt = [...q.y].map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
        msg =
          'Step ' +
          (gen.length - 1) +
          ': read “' +
          voc[x] +
          '”, mixed it with the memory of “' +
          gen
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
          (nx === seq[(gen.length - 1) % n]
            ? '.'
            : ', which is wrong: the phrase has “' + voc[seq[(gen.length - 1) % n]] + '” here.');
        outEl.textContent = 'Written so far: “' + gen.map((i) => voc[i]).join('') + '”';
        if (gen.length >= 2 * n) {
          done = true;
          t0 = fr - 200;
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
        else wstep();
      };
      const draw = () => {
        const t = T(),
          u = RM ? 1 : clamp((fr - t0) / 40, 0, 1);
        {
          const { g, w, h } = fit(c),
            NC = Math.min(9, Math.max(1, gen.length)),
            from = Math.max(0, gen.length - NC),
            cw = (w - 20) / 9,
            H = S.H,
            yI = h - 34,
            yH0 = h * 0.3,
            hh = h * 0.36,
            ch = hh / H,
            yO = h * 0.13;
          g.font = MONO;
          if (!gen.length) {
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(
              trained
                ? 'ready to write: press Run or One step'
                : 'learning… the writing starts when the error stops falling',
              w / 2,
              h / 2,
            );
          }
          for (let q = 0; q < NC && gen.length; q++) {
            const tI = from + q,
              x = 10 + q * cw + cw / 2,
              cur = tI === gen.length - 1,
              hv = hs[tI + 1];
            g.fillStyle = cur ? t.gold : t.bg;
            g.strokeStyle = t.ink;
            g.lineWidth = 1.4;
            g.beginPath();
            g.rect(x - 15, yI - 15, 30, 30);
            g.fill();
            g.stroke();
            g.font = SANS;
            g.fillStyle = cur ? t.bg : t.ink;
            g.textAlign = 'center';
            g.fillText(voc[gen[tI]] === ' ' ? '␣' : voc[gen[tI]], x, yI + 5);
            g.font = MONO;
            if (hv) {
              const al = tI === gen.length - 2 ? u : 1;
              g.strokeStyle = t.muted;
              g.lineWidth = 1.2;
              g.beginPath();
              g.moveTo(x, yI - 16);
              g.lineTo(x, yH0 + hh + 3);
              g.moveTo(x, yH0 - 3);
              g.lineTo(x, yO + 16);
              g.stroke();
              for (let j = 0; j < H; j++) {
                const v = hv[j];
                g.fillStyle = v >= 0 ? mix(t.bg, t.gold, v * al) : mix(t.bg, t.accent, -v * al);
                g.fillRect(x - 13, yH0 + j * ch, 26, Math.max(1, ch - 0.7));
              }
              g.strokeStyle = t.ink;
              g.lineWidth = 1.2;
              g.strokeRect(x - 13, yH0, 26, hh);
              if (q < NC - 1 || true) {
                g.strokeStyle = t.ink;
                g.lineWidth = 1.6;
                g.beginPath();
                g.moveTo(x + 14, yH0 + hh / 2);
                g.lineTo(x + cw - 14, yH0 + hh / 2);
                g.stroke();
                g.beginPath();
                g.moveTo(x + cw - 14, yH0 + hh / 2);
                g.lineTo(x + cw - 20, yH0 + hh / 2 - 4);
                g.lineTo(x + cw - 20, yH0 + hh / 2 + 4);
                g.fill();
              }
              if (al > 0.7) {
                const pr = probs[tI],
                  b = gen[tI + 1];
                g.fillStyle = t.bg;
                g.strokeStyle = t.ok;
                g.lineWidth = 1.4;
                g.beginPath();
                g.rect(x - 15, yO - 15, 30, 30);
                g.fill();
                g.stroke();
                g.font = SANS;
                g.fillStyle = t.ok;
                g.fillText(voc[b] === ' ' ? '␣' : voc[b], x, yO + 5);
                g.font = MONO;
                g.fillStyle = t.muted;
                g.fillText((100 * pr[b]).toFixed(0) + '%', x, yO - 20);
              }
              if (tI === gen.length - 2 && u < 1) {
                g.fillStyle = t.gold;
                g.beginPath();
                g.arc(x, yI - 16 - (yI - 16 - yH0 - hh) * Math.min(1, u * 2), 5, 0, 7);
                g.fill();
                g.beginPath();
                g.arc(x - cw + 14 + (cw - 28) * Math.min(1, u * 2), yH0 + hh / 2, 5, 0, 7);
                g.fill();
              }
            }
          }
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('expects next', 4, yO - 30 < 10 ? 10 : yO - 30);
          g.fillText('memory', 4, yH0 - 6);
          g.fillText('reads', 4, h - 6);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5,
            pr = probs.length ? probs[probs.length - 1] : null,
            bw = (w - 30) / Math.max(1, V);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            pr
              ? 'after reading “' +
                  voc[gen[gen.length - 2]] +
                  '”: chance of each letter coming next'
              : 'chance of each letter coming next (appears while writing)',
            8,
            13,
          );
          for (let k = 0; k < V; k++) {
            const x = 15 + k * bw,
              p = pr ? pr[k] * u : 0,
              top = h1 - 24 - p * (h1 - 52);
            g.fillStyle = pr && k === gen[gen.length - 1] ? t.gold : t.accent;
            g.fillRect(x + bw * 0.15, top, bw * 0.7, h1 - 24 - top);
            g.font = SANS;
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.fillText(voc[k] === ' ' ? '␣' : voc[k], x + bw / 2, h1 - 8);
            g.font = MONO;
            if (p > 0.04) {
              g.fillStyle = t.muted;
              g.fillText((100 * p).toFixed(0), x + bw / 2, top - 4);
            }
          }
          uLine(g, t, 0, h1, w, h - h1, LL, {
            title: 'error per letter, by number of readings',
            n: Math.max(100, LL.length),
            col: t.ok,
            ym: Math.max(1, LL[0] || 1),
          });
        }
        live.textContent = msg;
        uCur(root, 'rn', cu);
        setR(root, 'ph', done ? 'Finished' : trained ? 'Writing' : ep ? 'Learning' : 'Not started');
        setR(root, 'ep', String(ep));
        setR(root, 'L', ep ? f3(LL[ep - 1]) : '–');
        setR(root, 'acc', ep ? right() + ' of ' + n : '–');
        setR(root, 'np', String(S.H * V + S.H * S.H + S.H + V * S.H + V));
      };
      build();
      if (RM) {
        learn(EPM);
        while (!done) wstep();
      }
      const anim = Anim(() => {
        fr++;
        if (!trained) {
          if (fr % (ep < 40 ? 4 : 2) === 0) {
            learn(ep < 40 ? 1 : ep < 200 ? 2 : 5);
            draw();
            if (chk()) return;
          }
          return;
        }
        if (fr - t0 > (gen.length < 5 ? 95 : 46)) {
          wstep();
          if (done) {
            draw();
            pb.stop();
            return;
          }
        }
        draw();
      });
      pb = playBtn($('#rn-play', root), anim);
      chk = runCtl(pb, root, 'rn', {
        done: () => done,
        restart: () => {
          run++;
          build();
        },
        count: () => ep,
        what: 'readings',
      });
      $('#rn-step', root).onclick = () => {
        pb.stop();
        step();
        t0 = fr - 100;
        draw();
      };
      $('#rn-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#rn-text', root).addEventListener('change', (e) => {
        S.text = e.target.value;
        pb.stop();
        run++;
        build();
        st.textContent = /[^a-zA-Z ]/.test(e.target.value)
          ? 'Only letters and spaces are used; other characters were dropped.'
          : '';
        draw();
        pb.start();
      });
      bind(
        root,
        'rn-H',
        (v) => {
          S.H = v;
          pb.stop();
          run++;
          build();
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
    id: 'attention',
    tab: 'Attention',
    title: 'Attention: learning which word to listen to',
    lede: 'In “the cat sat on the mat because it was tired”, what does “it” mean? Change one word, “because it was soft”, and the answer flips to the mat. Here a small attention layer learns that skill from examples. Nobody tells it which words are animals or which adjectives suit them. Watch it work that out, then build your own sentences, including ones it never saw in training.',
    stage: `<div class="row" style="gap:10px;margin-bottom:8px;flex-wrap:wrap"><label class="ctl" style="grid-template-columns:auto 120px" for="at-a"><span>Animal</span><select id="at-a"></select></label><label class="ctl" style="grid-template-columns:auto 120px" for="at-p"><span>Place</span><select id="at-p"></select></label><label class="ctl" style="grid-template-columns:auto 120px" for="at-j"><span>It was…</span><select id="at-j"></select></label><label class="ctl" style="grid-template-columns:auto 230px" for="at-o"><span>Word order</span><select id="at-o"><option value="0">the animal sat on the place</option><option value="1">on the place sat the animal</option></select></label></div>
 <div class="stage"><figure class="fig"><figcaption>Your sentence. Arcs run from “it” to the words it listens to; thicker means more. Bars: first the match score of every word, then the shares after softmax</figcaption><canvas class="short" id="at-c" role="img" aria-label="Sentence with learned attention from the word it"></canvas><div class="eq" id="at-eq" role="status"></div></figure>
 <figure class="fig"><figcaption>Top: the map the layer has learned. Every word has a key (dot). Every adjective has a question (line from the centre). A question matches the keys that lie in its direction. Bottom: error and accuracy on unseen sentences</figcaption><canvas class="sq" id="at-r" role="img" aria-label="Learned keys and queries, and the training curves"></canvas></figure></div>
 ${uHow('at', ['Give every word a random key and every adjective a random question (query): two numbers each.', 'Take a training sentence. The question comes from its last word; compare it with every word’s key by a dot product.', 'Softmax turns the scores into shares. The share on the correct word should be high.', 'It is not, at first. Nudge the question toward the key of the right word, and that key toward the question; push the others away.', 'Repeat over all training sentences until the error stops falling. Then try sentences it has never seen.'])}
 <div class="controls">${uBtns('at', 'Train', 'Five passes', 'passes')}
 ${grp('The model', slider('at-tau', 'Sharpness at prediction: multiply the scores by', 0.2, 3, 0.05, 1) + `<div class="row"><button class="btn" id="at-exp" type="button">Explain This Sentence Step by Step</button></div>`)}</div>
 <dl class="read">${rd('ep', 'Passes over the training sentences')}${rd('L', 'Training error', 'hl2')}${rd('te', 'Unseen sentences answered correctly', 'hl')}${rd('ans', 'In your sentence, “it” is', 'hl3')}${rd('seen', 'Was your sentence in the training set?')}</dl>`,
    tryit:
      'Press Start over and watch the map: at first the dots are a jumble and “it” spreads its attention evenly. As training runs, animals drift one way and places the other, and each adjective’s line swings toward its kind. When it stops, switch “tired” to “soft” and watch the arcs jump. Change the word order: the answer follows the meaning, not the position. Find a sentence marked as never seen in training: is it still right? Then press Explain This Sentence.',
    math: String.raw`<h4>Intuition</h4><p>You are in a library with a question. Every book has a title on its spine. You compare your question with each title, pull out the few that match, and read mostly from those. Your question is the <b>query</b>, a title is a <b>key</b>, and what is inside the book is its <b>value</b>. Attention does this softly: instead of choosing a few books it reads a little from every book, in proportion to how well its title matched. In a sentence every word is at once a reader with a question and a book with a title.</p>
 <h4>The calculation</h4><p>Let word \(i\) have query \(\mathbf q_i\), and word \(j\) have key \(\mathbf k_j\) and value \(\mathbf v_j\), all vectors of dimension \(d\). The match score, the share and the output are</p>$$s_{ij}=\frac{\mathbf q_i\cdot\mathbf k_j}{\sqrt d},\qquad\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{m}e^{s_{im}}},\qquad\mathbf o_i=\sum_j\alpha_{ij}\,\mathbf v_j .$$<p>In matrix form, with the queries, keys and values stacked as rows of \(\mathbf Q,\mathbf K,\mathbf V\),</p>$$\operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V)=\operatorname{softmax}\!\Big(\frac{\mathbf Q\mathbf K^{\top}}{\sqrt d}\Big)\mathbf V .$$<p>On this page you see one row of it: the row for the word “it”.</p>
 <h4>Where queries, keys and values come from</h4><p>In a real network each word starts as one embedding vector \(\mathbf x_j\), and three learned matrices produce the three roles: \(\mathbf q_j=\mathbf W_Q\mathbf x_j\), \(\mathbf k_j=\mathbf W_K\mathbf x_j\), \(\mathbf v_j=\mathbf W_V\mathbf x_j\). Training adjusts those matrices so that useful words end up attending to each other. <b>On this page they are learned.</b> Every word has a key of two numbers and every adjective a query of two numbers, all starting at random, and gradient descent on the loss \(-\log\alpha_{\text{correct word}}\) moves them. Two numbers are used so that the result can be drawn as a map. One simplification remains: the query is read directly from the adjective at the end of the sentence. In a full Transformer the word “it” would first gather that adjective through an earlier attention layer and then ask its question.</p>
 <h4>Why divide by \(\sqrt d\), and what sharpness does</h4><p>Dot products of long vectors are large, and large scores push the softmax to put everything on one word, where its gradient is nearly zero. Dividing by \(\sqrt d\) keeps the scores at a sensible size. The sharpness slider multiplies the scores before the softmax: small values flatten the shares toward equal, large values approach picking the single best match.</p>
 <h4>Why attention replaced recurrence</h4><ul><li><b>Direct access.</b> Any word can read from any other in one step, however far apart. An RNN has to carry the information through every step in between.</li><li><b>Parallel.</b> All words are processed at once, not one after another.</li><li><b>Readable.</b> The shares \(\alpha_{ij}\) can be inspected, as here, although in deep models they are only a partial explanation of what the model does.</li></ul>
 <h4>What is still missing</h4><p>Attention on its own ignores word order: shuffle the sentence and the shares are the same. Transformers add position information to the embeddings. They also run several attentions side by side (<b>multi-head attention</b>), each free to look for a different relationship, and stack many layers with residual shortcuts between them.</p>`,
    q: 'If all the queries and keys were identical vectors, what would the attention shares be, and what would each word’s output become?',
    a: 'Every score would be the same number, so the softmax would give every word the same share, 1/n. Each output would be the plain average of all the values: every word would receive the same blurred summary of the sentence. Attention is useful only because different words ask different questions and offer different keys.',
    init(root) {
      const c = $('#at-c', root),
        cr = $('#at-r', root),
        live = $('#at-live', root),
        eq = $('#at-eq', root),
        S = { a: 0, p: 0, j: 0, o: 0, tau: 1 },
        AN = ['cat', 'dog', 'horse', 'bird', 'fox'],
        PL = ['mat', 'sofa', 'rug', 'bed', 'floor'],
        JA = ['tired', 'hungry', 'sleepy', 'happy'],
        JP = ['soft', 'warm', 'flat', 'dirty'],
        ADJ = JA.concat(JP),
        FW = ['the', 'sat', 'on', 'because', 'it', 'was'],
        WD = AN.concat(PL, FW, ADJ),
        wi = (w) => WD.indexOf(w);
      let Kt,
        Qt,
        TRN = [],
        TST = [],
        ep = 0,
        LL = [],
        AC = [],
        conv = false,
        msg = '',
        cu = 0,
        fr = 0,
        t0 = 0,
        pb,
        chk = () => false,
        run = 1,
        sub = 99,
        seen = new Set();
      const sent = (a, p, j, o) => {
        const w = o
          ? ['on', 'the', PL[p], 'sat', 'the', AN[a], 'because', 'it', 'was', ADJ[j]]
          : ['the', AN[a], 'sat', 'on', 'the', PL[p], 'because', 'it', 'was', ADJ[j]];
        return { w, ix: w.map(wi), tg: j < 4 ? (o ? 5 : 1) : o ? 2 : 5, j };
      };
      const fill = (id, arr) => {
        $('#' + id, root).innerHTML = arr
          .map((w, i) => '<option value="' + i + '">' + w + '</option>')
          .join('');
      };
      fill('at-a', AN);
      fill('at-p', PL);
      fill('at-j', ADJ);
      const build = () => {
        const r = rng(run * 17 + 3),
          rn = randnFrom(r);
        Kt = WD.map(() => [0.25 * rn(), 0.25 * rn()]);
        Qt = ADJ.map(() => [0.25 * rn(), 0.25 * rn()]);
        TRN = [];
        TST = [];
        seen = new Set();
        for (let a = 0; a < 5; a++)
          for (let p = 0; p < 5; p++)
            for (let j = 0; j < 8; j++)
              for (let o = 0; o < 2; o++) {
                const key = a + '-' + p + '-' + j + '-' + o;
                if (r() < 0.4) {
                  TRN.push(sent(a, p, j, o));
                  seen.add(key);
                } else TST.push(sent(a, p, j, o));
              }
        ep = 0;
        LL = [];
        AC = [];
        conv = false;
        cu = 0;
        sub = 99;
        eq.innerHTML = '';
        msg =
          'Random keys and questions. ' +
          TRN.length +
          ' sentences will be used for training and ' +
          TST.length +
          ' are held back. Right now “it” listens to every word about equally. Press Train.';
      };
      const att = (s, tau = 1) => {
        const q = Qt[s.j],
          sc = s.ix.map((i) => (q[0] * Kt[i][0] + q[1] * Kt[i][1]) * tau),
          m = Math.max(...sc),
          e = sc.map((v) => Math.exp(v - m)),
          z = e.reduce((x, y) => x + y, 0);
        return { sc, al: e.map((v) => v / z) };
      };
      const accOn = (D) =>
        D.filter((s) => {
          const al = att(s).al;
          return al.indexOf(Math.max(...al)) === s.tg;
        }).length / D.length;
      const pass = () => {
        let l = 0;
        const ord = TRN.map((_, i) => i);
        for (let i = ord.length - 1; i > 0; i--) {
          const k = Math.floor(Math.random() * (i + 1));
          [ord[i], ord[k]] = [ord[k], ord[i]];
        }
        for (const oi of ord) {
          const s = TRN[oi],
            { al } = att(s),
            q = Qt[s.j];
          l -= Math.log(al[s.tg] + 1e-12);
          const dq = [0, 0];
          s.ix.forEach((w, pz) => {
            const d = al[pz] - (pz === s.tg ? 1 : 0);
            dq[0] += d * Kt[w][0];
            dq[1] += d * Kt[w][1];
          });
          s.ix.forEach((w, pz) => {
            const d = al[pz] - (pz === s.tg ? 1 : 0);
            Kt[w][0] -= 0.012 * d * q[0];
            Kt[w][1] -= 0.012 * d * q[1];
          });
          q[0] -= 0.012 * dq[0];
          q[1] -= 0.012 * dq[1];
        }
        ep++;
        LL.push(l / TRN.length);
        AC.push(accOn(TST));
        if (
          LL[ep - 1] < 0.06 ||
          ep >= 400 ||
          (ep > 60 && Math.abs(LL[ep - 1] - LL[ep - 31]) < 0.01 * LL[ep - 1])
        )
          conv = true;
        cu = conv ? 4 : 1 + (ep % 3);
        msg =
          'Pass ' +
          ep +
          ': all ' +
          TRN.length +
          ' training sentences were shown once, and after each the keys and questions were nudged. Training error ' +
          f3(LL[ep - 1]) +
          '. On the ' +
          TST.length +
          ' sentences it has never seen, it picks the right word in ' +
          pct(AC[ep - 1]) +
          '.' +
          (conv ? ' Converged: training stops. Change the words above to test it.' : '');
      };
      const cur = () => sent(S.a, S.p, S.j, S.o);
      const explain = () => {
        const s = cur(),
          { sc, al } = att(s, S.tau),
          q = Qt[s.j],
          b = al.indexOf(Math.max(...al)),
          nf = (v) => (v < 0 ? '(−' + Math.abs(v).toFixed(2) + ')' : v.toFixed(2));
        if (sub <= 10) {
          const j = sub - 1,
            k = Kt[s.ix[j]];
          cu = 1;
          msg =
            'The question of “' +
            s.w[9] +
            '” is compared with the key of “' +
            s.w[j] +
            '”: score ' +
            f2(sc[j]) +
            '.';
          eq.innerHTML =
            '<div class="on">score(“' +
            s.w[j] +
            '”) = question · key × sharpness</div><div class="on">   = (' +
            nf(q[0]) +
            '·' +
            nf(k[0]) +
            ' + ' +
            nf(q[1]) +
            '·' +
            nf(k[1]) +
            ') × ' +
            f2(S.tau) +
            ' = <b>' +
            f2(sc[j]) +
            '</b></div>';
        } else if (sub === 11) {
          cu = 2;
          msg =
            'Softmax turns the ten scores into shares that add up to 1. “' +
            s.w[b] +
            '” gets ' +
            (100 * al[b]).toFixed(0) +
            '%.';
          eq.innerHTML =
            '<div class="on">share(“' +
            s.w[b] +
            '”) = e^' +
            f2(sc[b]) +
            ' / ( ' +
            sc
              .map((v) => 'e^' + f2(v))
              .slice(0, 4)
              .join(' + ') +
            ' + … ) = <b>' +
            f2(al[b]) +
            '</b></div>';
        } else {
          cu = 4;
          msg =
            'So in this sentence “it” means “' +
            s.w[b] +
            '”' +
            (b === s.tg
              ? ', which is right.'
              : ', which is wrong: it should be “' + s.w[s.tg] + '”.') +
            ' The layer was never given a rule; it placed the keys and questions so that the scores come out this way.';
          eq.innerHTML =
            '<div class="on">“it” takes ' +
            (100 * al[b]).toFixed(0) +
            '% of its new meaning from <b>“' +
            s.w[b] +
            '”</b></div>';
        }
      };
      const draw = () => {
        const t = T(),
          col = UC(t),
          s = cur(),
          { sc, al } = att(s, S.tau),
          n = 10,
          u = RM ? 1 : clamp((fr - t0) / 22, 0, 1),
          ex = sub <= 12,
          ns = ex ? Math.min(sub, 10) : 10,
          shares = !ex || sub > 10;
        {
          const { g, w, h } = fit(c),
            yw = h * 0.4,
            bw = (w - 20) / n,
            X = (j) => 10 + j * bw + bw / 2,
            sm = Math.max(0.6, ...sc.map(Math.abs)),
            by = h - 30,
            bh = h * 0.26;
          g.font = SANS;
          if (shares)
            for (let j = 0; j < n; j++) {
              if (j === 7) continue;
              const wt = al[j] * (sub === 11 ? u : 1);
              g.strokeStyle = mix(t.gold, t.gold, 0, 0.2 + 0.8 * wt);
              g.lineWidth = 0.5 + 13 * wt;
              g.beginPath();
              g.moveTo(X(7), yw - 16);
              g.quadraticCurveTo(
                (X(7) + X(j)) / 2,
                yw - 16 - Math.min(h * 0.32, Math.abs(X(j) - X(7)) * 0.45),
                X(j),
                yw - 16,
              );
              g.stroke();
            }
          for (let j = 0; j < n; j++) {
            const x = X(j),
              me = j === 7,
              tgt = j === s.tg;
            g.font = SANS;
            const tw = Math.max(34, g.measureText(s.w[j]).width + 12);
            g.fillStyle = me ? t.ink : t.bg;
            g.strokeStyle = tgt ? t.accent : t.ink;
            g.lineWidth = tgt ? 2.6 : 1.3;
            g.beginPath();
            g.rect(x - tw / 2, yw - 14, tw, 28);
            g.fill();
            g.stroke();
            g.fillStyle = me ? t.bg : t.ink;
            g.textAlign = 'center';
            g.fillText(s.w[j], x, yw + 5);
            g.font = MONO;
            if (!shares) {
              if (j < ns) {
                const v = sc[j] * (j === ns - 1 ? u : 1),
                  hh = (Math.abs(v) / sm) * bh * 0.5;
                g.fillStyle = v >= 0 ? t.accent : t.ok;
                g.fillRect(
                  x - bw * 0.28,
                  v >= 0 ? by - bh * 0.5 - hh : by - bh * 0.5,
                  bw * 0.56,
                  Math.max(1, hh),
                );
                g.fillStyle = t.muted;
                g.fillText(f2(sc[j]), x, by + 14);
              }
            } else {
              const v = al[j] * (sub === 11 ? u : 1),
                hh = v * bh;
              g.fillStyle = t.gold;
              g.fillRect(x - bw * 0.28, by - hh, bw * 0.56, Math.max(1, hh));
              g.fillStyle = t.muted;
              g.fillText((100 * al[j]).toFixed(0) + '%', x, by + 14);
            }
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            shares
              ? 'share of attention that “it” gives each word'
              : 'match score of each word’s key with the question',
            10,
            by - bh - 8,
          );
          g.fillStyle = t.accent;
          g.textAlign = 'right';
          g.fillText('green frame: the correct answer', w - 10, yw + 34);
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.58,
            ox = w / 2,
            oy = h1 / 2 + 6;
          let m = 1e-9;
          Kt.forEach((k) => {
            m = Math.max(m, Math.abs(k[0]), Math.abs(k[1]));
          });
          Qt.forEach((k) => {
            m = Math.max(m, Math.abs(k[0]), Math.abs(k[1]));
          });
          const sx = (w / 2 - 50) / m,
            sy = (h1 / 2 - 20) / m,
            P = (v) => [ox + v[0] * sx, oy - v[1] * sy];
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(8, oy);
          g.lineTo(w - 8, oy);
          g.moveTo(ox, 8);
          g.lineTo(ox, h1 - 4);
          g.stroke();
          ADJ.forEach((a, j) => {
            const p = P(Qt[j]),
              me = j === S.j;
            g.strokeStyle = me ? t.gold : mix(j < 4 ? t.accent : t.cut, t.bg, 0, 0.55);
            g.lineWidth = me ? 3.2 : 1.3;
            g.beginPath();
            g.moveTo(ox, oy);
            g.lineTo(p[0], p[1]);
            g.stroke();
            g.font = MONO;
            g.fillStyle = me ? t.cut : t.muted;
            g.textAlign = 'center';
            g.fillText(a + '?', p[0], p[1] + (p[1] < oy ? -5 : 12));
          });
          WD.forEach((wd, i) => {
            if (i >= 16) return;
            const p = P(Kt[i]),
              inS = s.ix.includes(i);
            g.fillStyle = i < 5 ? t.accent : i < 10 ? t.gold : t.muted;
            g.beginPath();
            g.arc(p[0], p[1], inS ? 5.5 : 3.5, 0, 7);
            g.fill();
            if (inS) {
              g.strokeStyle = t.ink;
              g.lineWidth = 1.5;
              g.stroke();
            }
            g.font = inS ? SANS : MONO;
            g.fillStyle = inS ? t.ink : t.muted;
            g.textAlign = 'left';
            g.fillText(wd, p[0] + 7, p[1] + 4);
          });
          g.font = MONO;
          g.textAlign = 'left';
          g.fillStyle = t.accent;
          g.fillText('● animals', 8, 14);
          g.fillStyle = t.cut;
          g.fillText('● places', 84, 14);
          g.fillStyle = t.muted;
          g.fillText('● other words     lines: each adjective’s question', 150, 14);
          const U = uLine(g, t, 0, h1, w, h - h1, LL, {
            title: 'training error (red) and accuracy on unseen sentences (green, 0 to 100%)',
            n: Math.max(40, LL.length),
            col: t.ok,
            ym: Math.max(2.4, LL[0] || 2.4),
          });
          if (AC.length) {
            g.strokeStyle = t.accent;
            g.lineWidth = 2.2;
            g.beginPath();
            AC.forEach((v, i) => {
              const y = U.Y(v * Math.max(2.4, LL[0] || 2.4));
              i ? g.lineTo(U.X(i), y) : g.moveTo(U.X(i), y);
            });
            g.stroke();
          }
        }
        live.textContent = msg;
        uCur(root, 'at', cu);
        const b = al.indexOf(Math.max(...al));
        setR(root, 'ep', String(ep));
        setR(root, 'L', ep ? f3(LL[ep - 1]) : '–');
        setR(root, 'te', ep ? pct(AC[ep - 1]) : pct(accOn(TST)));
        setR(
          root,
          'ans',
          '“' +
            s.w[b] +
            '” (' +
            (100 * al[b]).toFixed(0) +
            '%)' +
            (b === s.tg ? ', correct' : ', wrong'),
        );
        setR(
          root,
          'seen',
          seen.has(S.a + '-' + S.p + '-' + S.j + '-' + S.o) ? 'Yes' : 'No: never seen',
        );
      };
      build();
      if (RM) while (!conv) pass();
      const anim = Anim(() => {
        fr++;
        if (!conv) {
          if (fr % (ep < 8 ? 18 : 5) === 0) {
            pass();
            t0 = fr;
            if (conv) {
              sub = 1;
              explain();
            }
            chk();
          }
          draw();
          return;
        }
        if (sub <= 12) {
          if (fr - t0 > (sub <= 10 ? 14 : 110)) {
            sub++;
            t0 = fr;
            if (sub > 12) {
              draw();
              pb.stop();
              return;
            }
            explain();
          }
          draw();
          return;
        }
        draw();
        pb.stop();
      });
      pb = playBtn($('#at-play', root), anim);
      chk = runCtl(pb, root, 'at', {
        done: () => conv && sub > 12,
        restart: () => {
          run++;
          build();
        },
        count: () => ep,
        what: 'passes',
      });
      $('#at-step', root).onclick = () => {
        pb.stop();
        if (conv) {
          run++;
          build();
        } else for (let i = 0; i < 5 && !conv; i++) pass();
        draw();
      };
      $('#at-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#at-exp', root).onclick = () => {
        pb.stop();
        if (!conv) {
          msg = 'Train first: the keys and questions are still random.';
          draw();
          return;
        }
        sub = 1;
        t0 = fr;
        explain();
        const st = pb.start;
        anim.start();
        $('#at-play', root).textContent = 'Pause';
        $('#at-play', root).classList.add('on');
      };
      const chg = () => {
        sub = 99;
        eq.innerHTML = '';
        const s = cur(),
          al = att(s, S.tau).al,
          b = al.indexOf(Math.max(...al));
        if (conv)
          msg =
            '“' +
            s.w.join(' ') +
            '”: “it” gives ' +
            (100 * al[b]).toFixed(0) +
            '% of its attention to “' +
            s.w[b] +
            '”' +
            (b === s.tg ? ', the right word.' : ', which is wrong.') +
            (seen.has(S.a + '-' + S.p + '-' + S.j + '-' + S.o)
              ? ''
              : ' This exact sentence was never in the training set.');
        if (!pb.on) draw();
      };
      [
        ['at-a', 'a'],
        ['at-p', 'p'],
        ['at-j', 'j'],
        ['at-o', 'o'],
      ].forEach(([id, k]) => {
        $('#' + id, root).onchange = (e) => {
          S[k] = +e.target.value;
          chg();
        };
      });
      bind(
        root,
        'at-tau',
        (v) => {
          S.tau = v;
          chg();
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
    id: 'autoencoder',
    tab: 'Autoencoders',
    title: 'Autoencoders: squeeze a picture into two numbers, and get it back',
    lede: 'Give a network a picture, force it through a bottleneck of just two numbers, and ask for the same picture back on the other side. To succeed it has to discover on its own what matters about the pictures. Nobody tells it which letter is which, yet the letters sort themselves into groups on the map of those two numbers. Drag across the map to see what each spot turns into.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The map: where each picture lands on the two numbers of the bottleneck. Faint pictures show what every region decodes to. Drag the diamond anywhere</figcaption><canvas class="short" id="ae-c" role="img" aria-label="Two-dimensional code space of the autoencoder"></canvas></figure>
 <figure class="fig"><figcaption>Top row: six originals. Second row: what comes back after the squeeze. Then: the picture decoded from the diamond, and the error while learning</figcaption><canvas class="short" id="ae-r" role="img" aria-label="Originals, reconstructions, decoded probe and loss"></canvas></figure></div>
 ${uHow('ae', ['Take a 5 × 7 picture of a letter: 35 numbers.', 'Encoder: squeeze the 35 numbers through a hidden layer down to 2. That pair is the code, a point on the map.', 'Decoder: expand the 2 numbers back through a hidden layer to 35.', 'Compare the output with the original picture, pixel by pixel. That is the error.', 'Adjust all the weights to lower the error, and repeat over all the pictures until it stops falling.'])}
 <div class="controls">${uBtns('ae', 'Train', 'Ten epochs', 'epochs')}
 ${grp('The model', slider('ae-hw', 'Neurons in each hidden layer', 3, 24, 1, 14))}
 ${grp('The data', slider('ae-flip', 'Noise: chance that a pixel is flipped', 0, 0.2, 0.01, 0.04) + `<label class="check"><input type="checkbox" id="ae-col" checked> Show the true letter (the network never sees it)</label><div class="row"><button class="btn" id="ae-new" type="button">New random sample</button></div>`)}</div>
 <dl class="read">${rd('ep', 'Epochs')}${rd('L', 'Error per pixel', 'hl2')}${rd('px', 'Pixels reproduced correctly', 'hl')}${rd('cmp', 'Numbers per picture: before → after')}${rd('st', 'Status')}</dl>`,
    tryit:
      'Press Start over and train. At first every picture lands on the same spot and comes back as the same grey blur. Watch the points pull apart into groups, and the second row sharpen. When it has converged, drag the diamond slowly from the A group to the H group: the decoded picture morphs from one letter into the other. Untick Show the True Letter and train again to see that the grouping does not depend on the labels.',
    math: String.raw`<h4>Intuition</h4><p>Suppose you must describe a friend's face over a bad phone line using only two numbers. You could not send the picture. You would have to agree on a scheme first: perhaps the first number says how round the face is and the second how long the hair. A good scheme lets the listener redraw something close to the original. An autoencoder invents such a scheme by itself. The encoder is you, the two numbers are the phone line, the decoder is the listener, and the scheme is whatever makes the redrawn pictures match best.</p>
 <h4>The model</h4><p>An encoder \(f\) maps the input \(\mathbf x\in\mathbb R^{35}\) to a short code \(\mathbf z\in\mathbb R^{2}\), and a decoder \(g\) maps the code back:</p>$$\mathbf z=f(\mathbf x),\qquad\hat{\mathbf x}=g(\mathbf z),\qquad\mathcal L=\frac1n\sum_{i=1}^{n}\ell\big(\mathbf x_i,\;g(f(\mathbf x_i))\big).$$<p>Here both are small networks, \(35\to H\to2\) and \(2\to H\to35\), and \(\ell\) is the cross-entropy between each original pixel and the predicted one. The target is the input itself, so no labels are needed: this is unsupervised (more precisely, self-supervised) learning.</p>
 <h4>Why the bottleneck matters</h4><p>If the code were as large as the input, the network could copy its input straight through and learn nothing. Two numbers cannot hold 35 pixels, so the network must keep what distinguishes one picture from another and drop the rest. Pictures that share most of their pixels need nearly the same code, so they land close together on the map. That is why the letters form groups without ever being told they are letters.</p>
 <h4>Relation to PCA</h4><p>With no hidden layers and no activation, the best possible autoencoder with a two-number code finds the same plane as the first two principal components. The non-linear layers let it bend that plane into a curved surface that fits the data better. An autoencoder is a non-linear generalisation of PCA.</p>
 <h4>The decoder is a generator</h4><p>The decoder turns any point of the map into a picture, including points that no training picture landed on. Dragging the diamond between two groups gives pictures in between. In a plain autoencoder the regions between groups can decode to rubbish, because nothing trained them. The <b>variational autoencoder</b> fixes this by making the codes fill the map smoothly, which turns the decoder into a proper generative model. That is the start of the Generative AI module.</p>
 <h4>What autoencoders are used for</h4><ul><li><b>Compression and visualisation:</b> a short code in place of a long input.</li><li><b>Denoising:</b> train with a corrupted input and the clean picture as the target.</li><li><b>Anomaly detection:</b> an input unlike the training data is reconstructed badly, and the size of the error flags it.</li><li><b>Pre-training:</b> the encoder's code is a compact set of features for another task.</li></ul>`,
    q: 'After training, a picture of the letter K, which was never in the training set, is fed in. What do you expect to come out, and how could that be put to use?',
    a: 'Something that looks like a blend of the letters it does know, probably closest to H or X, because the decoder can only produce pictures of the kind it was trained on. The reconstruction will differ from the input in many pixels, so the error will be much larger than for a familiar letter. That is the basis of anomaly detection with autoencoders: a large reconstruction error signals an input unlike the training data.',
    init(root) {
      const c = $('#ae-c', root),
        cr = $('#ae-r', root),
        live = $('#ae-live', root),
        S = { hw: 14, flip: 0.04, col: true, seed: 1 };
      let X = [],
        lab = [],
        LT = [],
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
        pr = [0.2, 0.2],
        drag = false,
        codes = [];
      const gen = () => {
          const d = letterData(S.seed, S.flip);
          X = d.X.map((x) => Float64Array.from(x, (v) => clamp(Math.round(v), 0, 1)));
          lab = d.lab;
          LT = d.CH;
        },
        build = () => {
          net = NNet([35, S.hw, 2, S.hw, 35], { act: 'tanh', out: 'sig', seed: run * 9 + S.seed });
          ep = 0;
          LL = [];
          conv = false;
          cu = 0;
          enc();
          msg =
            'Random weights: every picture lands near the middle of the map and comes back as a grey blur. Press Train.';
        };
      const enc = () => {
          codes = X.map((x) => net.fwd(x).A[2]);
        },
        dec = (z) => {
          let a = z;
          for (let l = 2; l < 4; l++) {
            const m = net.sizes[l + 1],
              o = new Float64Array(m);
            for (let j = 0; j < m; j++) {
              let s = net.B[l][j];
              for (let i = 0; i < a.length; i++) s += net.W[l][j][i] * a[i];
              o[j] = l === 3 ? sig(s) : Math.tanh(s);
            }
            a = o;
          }
          return a;
        };
      const acc = () => {
        let ok = 0;
        X.forEach((x) => {
          const o = net.p2(x);
          for (let k = 0; k < 35; k++) if ((o[k] >= 0.5 ? 1 : 0) === x[k]) ok++;
        });
        return ok / (X.length * 35);
      };
      const train = (k) => {
        for (let i = 0; i < k && !conv; i++) {
          const l = net.train(X, X, 0.01).loss / 35;
          ep++;
          LL.push(l);
          if ((ep > 250 && Math.abs(LL[ep - 1] - LL[ep - 151]) < 0.012 * l) || ep >= 3000)
            conv = true;
        }
        enc();
        cu = conv ? 4 : ep < 4 ? 1 : 3;
        msg =
          'Epoch ' +
          ep +
          ': all ' +
          X.length +
          ' pictures were squeezed to two numbers and rebuilt. Error per pixel ' +
          f3(LL[ep - 1]) +
          '; ' +
          pct(acc()) +
          ' of pixels come back right.' +
          (conv
            ? ep >= 3000
              ? ' Reached the limit of 3000 epochs.'
              : ' Converged: the error has stopped falling, so training stops.'
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
            X2 = (v) => w / 2 + (v * (w / 2 - 14)) / 1.08,
            Y2 = (v) => h / 2 - (v * (h / 2 - 14)) / 1.08,
            NGd = 8;
          for (let a = 0; a < NGd; a++)
            for (let b = 0; b < NGd; b++) {
              const zx = -1 + (2 * (b + 0.5)) / NGd,
                zy = 1 - (2 * (a + 0.5)) / NGd;
              glyph(g, dec([zx, zy]), X2(zx) - 10, Y2(zy) - 14, 4, t, 0.22);
            }
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          g.strokeRect(X2(-1), Y2(1), X2(1) - X2(-1), Y2(-1) - Y2(1));
          codes.forEach((z, i) => {
            const x = X2(z[0]),
              y = Y2(z[1]);
            if (S.col) {
              g.font = '600 13px "Hanken Grotesk",sans-serif';
              g.fillStyle = col[lab[i] % 8];
              g.textAlign = 'center';
              g.fillText(LT[lab[i]], x, y + 4);
            } else {
              g.fillStyle = mix(t.ink, t.ink, 0, 0.6);
              g.beginPath();
              g.arc(x, y, 3, 0, 7);
              g.fill();
            }
          });
          markTP(g, X2(pr[0]), Y2(pr[1]), t, false);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('first number of the code →', X2(-1) + 4, Y2(-1) - 5);
          c._g = { X2, Y2, w, h };
        }
        {
          const { g, w, h } = fit(cr),
            cz = Math.min((w - 40) / 6 / 6, (h * 0.21) / 7.4),
            gw = 5 * cz,
            x0 = (w - 6 * gw * 1.25) / 2;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('in', 6, 16 + 3.5 * cz);
          g.fillText('out', 6, 28 + 10.5 * cz);
          for (let k = 0; k < 6; k++) {
            const i = k * 25,
              x = x0 + k * gw * 1.25;
            glyph(g, X[i], x, 10, cz, t);
            glyph(g, net.p2(X[i]), x, 22 + 7 * cz, cz, t);
          }
          const y3 = 38 + 14 * cz,
            pz = Math.min(9, (h - y3 - 8) / 7.2);
          glyph(g, dec(pr), 14, y3 + 10, pz, t);
          g.fillStyle = t.muted;
          g.fillText('the diamond', 14, y3 + 2);
          uLine(g, t, 5 * pz + 50, y3 - 4, w - 5 * pz - 54, h - y3 + 4, LL, {
            title: 'error per pixel',
            n: Math.max(100, LL.length),
            col: t.ok,
            ym: Math.max(0.7, LL[0] || 0.7),
          });
        }
        live.textContent = msg;
        uCur(root, 'ae', cu);
        setR(root, 'ep', String(ep));
        setR(root, 'L', ep ? f3(LL[ep - 1]) : '–');
        setR(root, 'px', pct(acc()));
        setR(root, 'cmp', '35 → 2');
        setR(root, 'st', conv ? 'Converged' : ep ? 'Training' : 'Not started');
      };
      gen();
      build();
      if (RM) train(3000);
      const redo = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      const anim = Anim(() => {
        fr++;
        if (ep < 30 && fr % 4) return;
        train(ep < 30 ? 1 : ep < 300 ? 3 : 6);
        draw();
        chk();
      });
      pb = playBtn($('#ae-play', root), anim);
      chk = runCtl(pb, root, 'ae', {
        done: () => conv,
        restart: () => {
          run++;
          build();
        },
        count: () => ep,
        what: 'epochs',
      });
      $('#ae-step', root).onclick = () => {
        pb.stop();
        if (conv) {
          run++;
          build();
        } else train(10);
        draw();
      };
      $('#ae-reset', root).onclick = redo;
      bind(
        root,
        'ae-hw',
        (v) => {
          S.hw = v;
          redo();
        },
        (v) => String(v),
      );
      bind(root, 'ae-flip', (v) => {
        S.flip = v;
        gen();
        redo();
      });
      $('#ae-col', root).onchange = (e) => {
        S.col = e.target.checked;
        draw();
      };
      $('#ae-new', root).onclick = () => {
        S.seed++;
        gen();
        redo();
      };
      const mv = (e) => {
        const q = c._g;
        if (!q) return;
        const r = c.getBoundingClientRect();
        pr = [
          clamp((e.clientX - r.left - q.w / 2) / ((q.w / 2 - 14) / 1.08), -1, 1),
          clamp(-(e.clientY - r.top - q.h / 2) / ((q.h / 2 - 14) / 1.08), -1, 1),
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
    id: 'embeddings',
    tab: 'Word embeddings',
    title: 'Word embeddings: words that keep the same company end up as neighbours',
    lede: 'A network cannot read letters; it needs numbers. The trick is to give every word a position on a map and let the positions be learned from text alone, by one rule: words that appear in the same company should sit close together. Here 28 words start scattered at random and sort themselves into royalty, animals, food and cities without ever being told those groups exist.',
    stage: `<div class="stage"><figure class="fig"><figcaption>The map of words. Gold line: a pair seen together in the text, being pulled closer. Red line: a random pair, being pushed apart. Click a word to see its nearest neighbours</figcaption><canvas class="short" id="we-c" role="img" aria-label="Two-dimensional word embeddings"></canvas></figure>
 <figure class="fig"><figcaption>Top: how similar every other word is to the one you clicked. Bottom: the error while learning</figcaption><canvas class="short" id="we-r" role="img" aria-label="Similarity to the selected word and the loss curve"></canvas></figure></div>
 ${uHow('we', ['Give every word a random position.', 'Take a pair of words that appear together in the text, such as “king” and “crown”.', 'Pull the two a little closer together.', 'Take a random word that did not appear with the first, and push the two a little apart.', 'Repeat over all the pairs, many times, until the map stops changing.'])}
 <div class="controls">${uBtns('we', 'Train', 'One pass', 'passes')}
 ${grp('The model', `<label class="check"><input type="checkbox" id="we-col" checked> Colour the words by their true group (the model never sees it)</label>` + slider('we-neg', 'Random pairs pushed apart for each pair pulled together', 0, 5, 1, 2))}</div>
 <dl class="read">${rd('ep', 'Passes over the text')}${rd('L', 'Error', 'hl2')}${rd('sel', 'Word you clicked')}${rd('nn', 'Its nearest neighbours', 'hl')}${rd('grp', 'Words whose nearest neighbour is in their own group')}</dl>`,
    tryit:
      'Press Start over and train with the colours off, so you see what the model sees: grey words drifting. Turn the colours on at the end. Click “queen” and read its neighbours. Then set the random pairs to 0 and train again: with only pulling and no pushing, what happens to the whole map?',
    math: String.raw`<h4>Intuition</h4><p>You can guess a lot about a stranger from the company they keep. If someone is always seen with a crown, a throne and a palace, you do not need a dictionary to guess their role. Words work the same way: "you shall know a word by the company it keeps" (J. R. Firth, 1957). Word embeddings turn that saying into arithmetic. No one defines any word. The positions come entirely from which words appear near which.</p>
 <h4>The model (skip-gram with negative sampling)</h4><p>Every word \(w\) gets a vector \(\mathbf u_w\), and a second vector \(\mathbf v_w\) for when it plays the role of context. For a pair \((w,c)\) seen together in the text and \(k\) randomly drawn words \(n_1,\dots,n_k\), the loss is</p>$$\mathcal L=-\log\sigma\big(\mathbf u_w\cdot\mathbf v_c\big)-\sum_{j=1}^{k}\log\sigma\big(-\mathbf u_w\cdot\mathbf v_{n_j}\big).$$<p>The first term is small when the dot product of a true pair is large, so true pairs are pulled into alignment. The second is small when the dot product with a random word is very negative, so random pairs are pushed apart. Gradient descent on this loss is the pulling and pushing in the animation. This is word2vec (Mikolov and others, 2013).</p>
 <h4>Why similar words end up together</h4><p>"King" and "queen" need never appear side by side. Both appear with "crown", "throne" and "palace", so both are pulled toward the same context vectors and arrive in the same region. Similarity of meaning is measured by the cosine of the angle between two vectors,</p>$$\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert},$$<p>which is what the bars on the right show.</p>
 <h4>Why the pushing is needed</h4><p>Pulling alone has a trivial solution: put every word at the same point. The random "negative" pairs give the model something to lose by doing that.</p>
 <h4>Real embeddings</h4><p>Real ones have 100 to several thousand dimensions and are learned from billions of words. In that much space, directions acquire meaning: the step from "man" to "woman" is close to the step from "king" to "queen". Two dimensions, as here, have room for groups but not for such directions. Embeddings also absorb the biases of the text they are trained on. In a Transformer the embedding table is the first layer, and each word's vector is then reshaped by its sentence, which a fixed table like this one cannot do: "bank" gets one position whether it is a river bank or a savings bank.</p>
 <h4>The text used here</h4><p>The "text" is a small list of word pairs built for this page: each of 16 words is paired with three context words of its group (for example king with crown, throne, palace). It is a toy corpus, not real text.</p>`,
    q: '“King” and “queen” end up close together although the text never contains the pair (king, queen). How?',
    a: 'Both words appear with the same context words: crown, throne, palace. Training pulls king toward the context vectors of those words and pulls queen toward the same three vectors. Two words pulled toward the same targets end up near each other. The embedding encodes shared company, not direct co-occurrence.',
    init(root) {
      const c = $('#we-c', root),
        cr = $('#we-r', root),
        live = $('#we-live', root),
        S = { col: true, neg: 2 },
        GR = [
          ['king', 'queen', 'prince', 'princess'],
          ['cat', 'dog', 'horse', 'bird'],
          ['apple', 'bread', 'cheese', 'rice'],
          ['paris', 'london', 'tokyo', 'rome'],
        ],
        CX = [
          ['crown', 'throne', 'palace'],
          ['fur', 'tail', 'paw'],
          ['eat', 'cook', 'tasty'],
          ['city', 'travel', 'map'],
        ];
      let W = [],
        G = [],
        PR = [],
        U,
        Vc,
        ep = 0,
        LL = [],
        conv = false,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        sel = 1,
        cur = null,
        PX = [];
      GR.forEach((g, k) => {
        g.forEach((w) => {
          W.push(w);
          G.push(k);
        });
        CX[k].forEach((w) => {
          W.push(w);
          G.push(k);
        });
      });
      GR.forEach((g, k) =>
        g.forEach((w) =>
          CX[k].forEach((x) => {
            PR.push([W.indexOf(w), W.indexOf(x)]);
            PR.push([W.indexOf(x), W.indexOf(w)]);
          }),
        ),
      );
      const N = W.length;
      const build = () => {
        const r = rng(run * 13 + 1);
        U = W.map(() => [(r() - 0.5) * 1.6, (r() - 0.5) * 1.6]);
        Vc = W.map(() => [(r() - 0.5) * 0.4, (r() - 0.5) * 0.4]);
        ep = 0;
        LL = [];
        conv = false;
        cu = 0;
        cur = null;
        msg =
          'Every word has a random position. The model knows nothing about any of them. Press Train or One pass.';
      };
      const upd = (a, b, y) => {
        const u = U[a],
          v = Vc[b],
          p = sig(u[0] * v[0] + u[1] * v[1]),
          g = (p - y) * 0.06,
          u0 = u[0],
          u1 = u[1];
        u[0] -= g * v[0];
        u[1] -= g * v[1];
        v[0] -= g * u0;
        v[1] -= g * u1;
        return -Math.log((y ? p : 1 - p) + 1e-9);
      };
      const one = () => {
        const pr = PR[Math.floor(Math.random() * PR.length)];
        let l = upd(pr[0], pr[1], 1),
          ng = -1;
        for (let k = 0; k < S.neg; k++) {
          let nw;
          do {
            nw = Math.floor(Math.random() * N);
          } while (G[nw] === G[pr[0]]);
          l += upd(pr[0], nw, 0);
          ng = nw;
        }
        cur = [pr[0], pr[1], ng];
        return l;
      };
      const cosv = (a, b) =>
          (U[a][0] * U[b][0] + U[a][1] * U[b][1]) /
          (Math.hypot(...U[a]) * Math.hypot(...U[b]) || 1),
        near = (a) =>
          W.map((_, i) => i)
            .filter((i) => i !== a)
            .sort((i, j) => cosv(a, j) - cosv(a, i));
      const dnear = (a) => {
        let b = -1,
          bd = 1e9;
        for (let i = 0; i < N; i++) {
          if (i === a) continue;
          const d = Math.hypot(U[a][0] - U[i][0], U[a][1] - U[i][1]);
          if (d < bd) {
            bd = d;
            b = i;
          }
        }
        return b;
      };
      const pass = () => {
        let l = 0;
        for (let i = 0; i < PR.length; i++) l += one();
        ep++;
        LL.push(l / PR.length);
        if ((ep > 60 && Math.abs(LL[ep - 1] - LL[ep - 41]) < 0.01 * LL[ep - 1]) || ep >= 600)
          conv = true;
        cu = conv ? 4 : 2 + (ep % 2);
        msg =
          'Pass ' +
          ep +
          ': all ' +
          PR.length +
          ' pairs from the text were pulled together once, each with ' +
          S.neg +
          ' random pair' +
          (S.neg === 1 ? '' : 's') +
          ' pushed apart. Last pair pulled: “' +
          W[cur[0]] +
          '” and “' +
          W[cur[1]] +
          '”. Error ' +
          f3(LL[ep - 1]) +
          '.' +
          (conv
            ? ep >= 600
              ? ' Reached the limit of 600 passes.'
              : ' Converged: the error has stopped falling, so the map has settled.'
            : '');
      };
      const draw = () => {
        const t = T(),
          col = UC(t);
        {
          const { g, w, h } = fit(c);
          let m = 1e-9;
          U.forEach((u) => {
            m = Math.max(m, Math.abs(u[0]), Math.abs(u[1]));
          });
          const X = (v) => w / 2 + (v / m) * (w / 2 - 46),
            Y = (v) => h / 2 - (v / m) * (h / 2 - 22);
          PX = U.map((u) => [X(u[0]), Y(u[1])]);
          if (cur && pb && pb.on) {
            g.lineWidth = 2.2;
            g.strokeStyle = t.gold;
            g.beginPath();
            g.moveTo(...PX[cur[0]]);
            g.lineTo(...PX[cur[1]]);
            g.stroke();
            if (cur[2] >= 0) {
              g.strokeStyle = t.ok;
              g.setLineDash([5, 4]);
              g.beginPath();
              g.moveTo(...PX[cur[0]]);
              g.lineTo(...PX[cur[2]]);
              g.stroke();
              g.setLineDash([]);
            }
          }
          W.forEach((wd, i) => {
            const ctx = i % 7 > 3;
            g.font =
              (i === sel ? '700 15px' : ctx ? '500 12px' : '600 13px') +
              ' "Hanken Grotesk",sans-serif';
            g.fillStyle = S.col ? col[G[i] % 8] : t.ink;
            g.globalAlpha = ctx && i !== sel ? 0.7 : 1;
            g.textAlign = 'center';
            g.fillText(wd, PX[i][0], PX[i][1] + 4);
            g.globalAlpha = 1;
            if (i === sel) {
              const tw = g.measureText(wd).width;
              g.strokeStyle = t.ink;
              g.lineWidth = 1.5;
              g.strokeRect(PX[i][0] - tw / 2 - 5, PX[i][1] - 11, tw + 10, 21);
            }
          });
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.56,
            nb = near(sel).slice(0, 9),
            rh = (h1 - 26) / 9;
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('words most similar to “' + W[sel] + '” (cosine similarity)', 8, 13);
          nb.forEach((i, k) => {
            const y = 22 + k * rh,
              v = cosv(sel, i);
            g.font = SANS;
            g.fillStyle = S.col ? col[G[i] % 8] : t.ink;
            g.textAlign = 'right';
            g.fillText(W[i], 88, y + rh * 0.7);
            g.fillStyle = mix(t.ink, t.ink, 0, 0.08);
            g.fillRect(96, y + 3, w - 150, rh - 6);
            g.fillStyle = G[i] === G[sel] ? t.gold : t.muted;
            g.fillRect(96, y + 3, (w - 150) * Math.max(0, v), rh - 6);
            g.font = MONO;
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText(f2(v), w - 48, y + rh * 0.7);
          });
          uLine(g, t, 0, h1, w, h - h1, LL, {
            title: 'error, by pass over the text',
            n: Math.max(60, LL.length),
            col: t.ok,
            ym: Math.max(1.5, LL[0] || 1.5),
          });
        }
        live.textContent = msg;
        uCur(root, 'we', cu);
        let ok = 0;
        for (let i = 0; i < N; i++) if (G[dnear(i)] === G[i]) ok++;
        setR(root, 'ep', String(ep));
        setR(root, 'L', ep ? f3(LL[ep - 1]) : '–');
        setR(root, 'sel', '“' + W[sel] + '”');
        setR(
          root,
          'nn',
          near(sel)
            .slice(0, 3)
            .map((i) => W[i])
            .join(', '),
        );
        setR(root, 'grp', ok + ' of ' + N);
      };
      build();
      if (RM) while (!conv) pass();
      const anim = Anim(() => {
        fr++;
        if (fr % (ep < 6 ? 14 : 5) === 0) pass();
        else if (fr % 2 === 0) one();
        draw();
        chk();
      });
      pb = playBtn($('#we-play', root), anim);
      chk = runCtl(pb, root, 'we', {
        done: () => conv,
        restart: () => {
          run++;
          build();
        },
        count: () => ep,
        what: 'passes',
      });
      $('#we-step', root).onclick = () => {
        pb.stop();
        if (conv) {
          run++;
          build();
        } else pass();
        draw();
      };
      $('#we-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      $('#we-col', root).onchange = (e) => {
        S.col = e.target.checked;
        draw();
      };
      bind(
        root,
        'we-neg',
        (v) => {
          S.neg = v;
          pb.stop();
          run++;
          build();
          draw();
        },
        (v) => String(v),
      );
      c.addEventListener('click', (e) => {
        const r = c.getBoundingClientRect(),
          x = e.clientX - r.left,
          y = e.clientY - r.top;
        let b = -1,
          bd = 900;
        PX.forEach((p, i) => {
          const d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
          if (d < bd) {
            bd = d;
            b = i;
          }
        });
        if (b >= 0) {
          sel = b;
          if (!pb.on) draw();
        }
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
    id: 'lstm',
    tab: 'LSTM and GRU',
    title: 'LSTM and GRU: a memory with a gate on it',
    lede: '“The keys, which I left on the table next to the lamp, ___ missing.” To choose “are” you must still remember “keys” ten words later, and not be thrown by “table” and “lamp”. A plain recurrent network rewrites its whole memory at every word, so the trace fades. An LSTM puts the memory on a conveyor belt with gates that decide what gets on and what is wiped. Follow one fact down both.',
    stage: `<div class="stage one"><figure class="fig"><figcaption>Top belt: a plain recurrent network. Bottom belt: an LSTM. The dot is the memory of “keys” (plural); its size is how much is left. Gates are drawn open or shut at each word</figcaption><canvas class="tall" id="ls-c" style="aspect-ratio:16/6" role="img" aria-label="Memory of a fact carried through a sentence by an RNN and by an LSTM"></canvas><div class="eq" id="ls-eq" role="status"></div></figure></div>
 <div class="stage" style="margin-top:18px"><figure class="fig"><figcaption>How much of “keys is plural” each network still holds, word by word</figcaption><canvas class="tall" id="ls-r" role="img" aria-label="Memory retained by step for RNN and LSTM"></canvas></figure>
 <figure class="fig"><figcaption>Inside one LSTM step</figcaption><canvas class="tall" id="ls-d" role="img" aria-label="Diagram of an LSTM cell with its gates"></canvas></figure></div>
 ${uHow('ls', ['Read “keys”. Both networks store the fact “the subject is plural” as +1.', 'Plain network, next word: the whole memory is multiplied by a weight and squashed. A little is lost every time.', 'LSTM, next word: the forget gate decides how much of the old memory to keep (near 1 = keep), and the input gate decides whether the new word may write (near 0 = no).', 'A distracting word arrives (“table”, singular). The plain network must mix it in. The LSTM keeps its input gate shut.', 'At the blank, read the memory: clearly positive means “are”.'])}
 <div class="controls">${grp('Animation', `<div class="row"><button class="btn go" id="ls-play" type="button">Read the sentence</button><button class="btn" id="ls-step" type="button">One word</button><button class="btn" id="ls-reset" type="button">Start over</button></div>`)}
 ${grp('The gates', slider('ls-f', 'LSTM forget gate: share of memory kept at each word', 0, 1, 0.01, 0.98) + slider('ls-i', 'LSTM input gate at the distracting words', 0, 1, 0.01, 0.03) + slider('ls-w', 'Plain network: weight on its own memory', 0.2, 1.6, 0.01, 0.9))}</div>
 <dl class="read">${rd('r', 'Plain network at the blank', 'hl2')}${rd('l', 'LSTM at the blank', 'hl3')}${rd('rg', 'Plain network: learning signal reaching “keys”')}${rd('lg', 'LSTM: learning signal reaching “keys”')}</dl>`,
    tryit:
      'Read the sentence through and compare the two dots at the blank. Now drag the forget gate down to 0.7: the LSTM forgets too. Put it back and open the input gate at the distracting words to 1: “table” and “lamp” overwrite the fact. So the LSTM is only as good as its gates, and the gates are what training learns. Last, raise the plain network’s weight above 1.2: does that rescue it?',
    math: String.raw`<h4>Intuition</h4><p>A plain recurrent network keeps its memory on a whiteboard that it wipes and rewrites, in slightly smudged form, at every single word. After fifteen rewrites the first note is illegible. An LSTM (long short-term memory) keeps a notebook instead. At each word it makes three small decisions: which old lines to cross out, whether to write anything new, and which lines to read aloud. If it decides "cross out nothing, write nothing", the note is exactly as it was. Remembering becomes the default, and forgetting is a choice.</p>
 <h4>The plain network</h4>$$h_t=\tanh\big(w\,h_{t-1}+u\,x_t\big),\qquad\frac{\partial h_T}{\partial h_1}=\prod_{t=2}^{T}w\,\big(1-h_t^2\big).$$<p>The memory passes through a multiplication and a squashing at every step, and the learning signal back to the first word is a product of \(T-1\) factors. If they are below 1 it vanishes. If \(w\) is made large to compensate, \(\tanh\) saturates and \(1-h^2\) goes to zero instead.</p>
 <h4>The LSTM</h4><p>The LSTM has a separate cell state \(c_t\), the conveyor belt, and three gates, each a sigmoid layer with outputs between 0 and 1:</p>$$f_t=\sigma(\cdot)\ \text{forget},\quad i_t=\sigma(\cdot)\ \text{input},\quad o_t=\sigma(\cdot)\ \text{output},\qquad c_t=f_t\,c_{t-1}+i_t\,\tilde c_t,\qquad h_t=o_t\tanh(c_t),$$<p>where \(\tilde c_t=\tanh(\cdot)\) is the candidate to be written and every \((\cdot)\) is a weighted sum of \(x_t\) and \(h_{t-1}\). The key line is the update of \(c_t\): the old memory is multiplied by \(f_t\) and added to, with no squashing. So</p>$$\frac{\partial c_T}{\partial c_1}=\prod_{t=2}^{T}f_t ,$$<p>and when the forget gates are near 1 the memory and its gradient pass through almost unchanged for as long as needed. It is the same idea as a residual shortcut, applied through time.</p>
 <h4>The GRU</h4><p>The gated recurrent unit merges the cell and hidden state and uses two gates. An update gate \(z_t\) blends old and new, \(h_t=(1-z_t)\,h_{t-1}+z_t\,\tilde h_t\), and a reset gate controls how much of the past feeds the candidate. It has fewer weights than an LSTM and usually performs about as well.</p>
 <h4>What this page simplifies</h4><p>In a real LSTM the gates are computed from the current word and the previous state by learned weights. <b>Here the gates are set by you with sliders</b>, and each network holds a single number, so that the effect of the gates can be seen in isolation: the forget gate is the same at every word, and the input gate is open at "keys" and set by the slider at the distracting words. The curves are exact for those settings.</p>`,
    q: 'With the forget gate at 0.98 the LSTM keeps about 75% of the fact after 14 words. Why does 0.98 matter so much more than, say, 0.90?',
    a: 'The fact is multiplied by the forget gate at every word, so what remains is the gate value raised to the number of words. 0.98 to the power 14 is about 0.75, while 0.90 to the power 14 is about 0.23. Small differences per step compound over long gaps, which is why LSTMs are often initialised with the forget gate biased toward 1.',
    init(root) {
      const c = $('#ls-c', root),
        cr = $('#ls-r', root),
        cd = $('#ls-d', root),
        live = $('#ls-live', root),
        eq = $('#ls-eq', root),
        S = { f: 0.98, i: 0.03, w: 0.9 },
        WD = [
          'The',
          'keys',
          ',',
          'which',
          'I',
          'left',
          'on',
          'the',
          'table',
          'next',
          'to',
          'the',
          'lamp',
          ',',
          '___',
        ],
        X = WD.map((w) => (w === 'keys' ? 1 : w === 'table' || w === 'lamp' ? -1 : 0)),
        T2 = WD.length;
      let H = [],
        C = [],
        GR = [],
        GL = [],
        k = 0,
        msg = '',
        cu = 0,
        fr = 0,
        t0 = 0,
        pb;
      const comp = () => {
        H = [0];
        C = [0];
        for (let t = 0; t < T2; t++) {
          const x = X[t];
          H.push(Math.tanh(S.w * H[t] + 1.2 * x));
          const ig = x > 0 ? 1 : x < 0 ? S.i : 0;
          C.push((t === 0 ? 0 : S.f) * C[t] + ig * x);
        }
        GR = new Array(T2 + 1).fill(1);
        GL = new Array(T2 + 1).fill(1);
        for (let t = T2 - 1; t >= 2; t--) {
          GR[t] = GR[t + 1] * S.w * (1 - H[t + 1] ** 2);
          GL[t] = GL[t + 1] * S.f;
        }
      };
      const verdict = (v) => (v > 0.3 ? '“are” (correct)' : v < -0.3 ? '“is” (wrong)' : 'unsure');
      const step = () => {
        if (k >= T2) {
          k = 0;
          cu = 0;
          eq.innerHTML = '';
          msg = 'Start again.';
          return;
        }
        k++;
        t0 = fr;
        const t = k - 1,
          x = X[t],
          ig = x > 0 ? 1 : x < 0 ? S.i : 0;
        if (x > 0) {
          cu = 0;
          msg =
            '“keys”: both networks store the fact that the subject is plural. Plain network memory ' +
            f2(H[k]) +
            ', LSTM memory ' +
            f2(C[k]) +
            '.';
        } else if (x < 0) {
          cu = 3;
          msg =
            '“' +
            WD[t] +
            '” is singular, a distraction. The plain network has to mix it in: ' +
            f2(H[k - 1]) +
            ' → ' +
            f2(H[k]) +
            '. The LSTM’s input gate is ' +
            (S.i < 0.2 ? 'nearly shut' : 'open ' + f2(S.i)) +
            ': ' +
            f2(C[k - 1]) +
            ' → ' +
            f2(C[k]) +
            '.';
        } else if (k >= T2) {
          cu = 4;
          msg =
            'The blank. Plain network memory ' +
            f2(H[k]) +
            ': it says ' +
            verdict(H[k]) +
            '. LSTM memory ' +
            f2(C[k]) +
            ': it says ' +
            verdict(C[k]) +
            '.';
        } else {
          cu = k % 2 ? 1 : 2;
          msg =
            '“' +
            WD[t] +
            '”: nothing to store. Plain network: ' +
            f2(H[k - 1]) +
            ' → ' +
            f2(H[k]) +
            ' (multiplied by its weight and squashed). LSTM: ' +
            f2(C[k - 1]) +
            ' → ' +
            f2(C[k]) +
            ' (multiplied by the forget gate, ' +
            f2(S.f) +
            ').';
        }
        eq.innerHTML =
          '<div class="on">plain:  h = tanh( w·h + u·x ) = tanh( ' +
          f2(S.w) +
          '·' +
          f2(H[k - 1]) +
          ' + 1.2·' +
          x +
          ' ) = <b>' +
          f2(H[k]) +
          '</b></div><div class="on">LSTM:   c = f·c + i·x = ' +
          f2(t === 0 ? 0 : S.f) +
          '·' +
          f2(C[k - 1]) +
          ' + ' +
          f2(ig) +
          '·' +
          x +
          ' = <b>' +
          f2(C[k]) +
          '</b></div>';
      };
      const draw = () => {
        const t = T(),
          u = RM ? 1 : clamp((fr - t0) / 26, 0, 1);
        {
          const { g, w, h } = fit(c),
            m = 34,
            cw = (w - 2 * m) / T2,
            rows = [
              [H, h * 0.3, t.ok, 'plain recurrent network'],
              [C, h * 0.72, t.accent, 'LSTM'],
            ];
          g.font = MONO;
          rows.forEach(([A, y, cl, lab], ri) => {
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText(lab, m, y - h * 0.17);
            g.strokeStyle = t.muted;
            g.lineWidth = ri ? 3 : 1.2;
            g.beginPath();
            g.moveTo(m, y);
            g.lineTo(w - m, y);
            g.stroke();
            for (let q = 0; q < T2; q++) {
              const x = m + (q + 0.5) * cw;
              g.font = q < k ? SANS : MONO;
              g.fillStyle = q === k - 1 ? t.ink : q < k ? t.muted : mix(t.ink, t.ink, 0, 0.3);
              g.textAlign = 'center';
              g.fillText(WD[q], x, y + h * 0.2);
              if (ri === 1) {
                const ig = X[q] > 0 ? 1 : X[q] < 0 ? S.i : 0;
                g.strokeStyle = X[q] ? t.ink : t.line;
                g.lineWidth = 1.4;
                g.beginPath();
                g.moveTo(x - 7, y + h * 0.11);
                g.lineTo(x - 7, y + 12);
                g.moveTo(x + 7, y + h * 0.11);
                g.lineTo(x + 7, y + 12);
                g.stroke();
                g.fillStyle = X[q] ? t.ok : t.line;
                g.fillRect(x - 7, y + 12, 14 * (1 - ig), 3);
              } else {
                g.strokeStyle = t.line;
                g.lineWidth = 1;
                g.beginPath();
                g.arc(x, y, 9, 0, 7);
                g.stroke();
              }
              if (q < k) {
                const v = A[q + 1],
                  al = q === k - 1 ? u : 1,
                  pv = A[q];
                const vv = pv + (v - pv) * al;
                g.fillStyle = vv >= 0 ? cl : t.muted;
                g.globalAlpha = q === k - 1 ? 1 : 0.28;
                g.beginPath();
                g.arc(x - (1 - al) * cw, y, 2 + 15 * Math.min(1, Math.abs(vv)), 0, 7);
                g.fill();
                g.globalAlpha = 1;
                if (q === k - 1) {
                  g.fillStyle = t.ink;
                  g.fillText(f2(v), x, y - h * 0.11);
                }
              }
            }
          });
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('grey dot: the memory has flipped to “singular”', w - m, h - 6);
        }
        {
          const { g, w, h } = fit(cr),
            m = { l: 40, r: 12, t: 16, b: 22 },
            X2 = (q) => m.l + (q / T2) * (w - m.l - m.r),
            Y2 = (v) => h - m.b - ((clamp(v, -1.1, 1.1) + 1.1) / 2.2) * (h - m.t - m.b);
          g.font = MONO;
          g.strokeStyle = t.line;
          g.lineWidth = 1;
          [-1, 0, 1].forEach((v) => {
            g.beginPath();
            g.moveTo(m.l, Y2(v));
            g.lineTo(w - m.r, Y2(v));
            g.stroke();
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText(String(v), m.l - 5, Y2(v) + 4);
          });
          [
            [H, t.ok],
            [C, t.accent],
          ].forEach(([A, cl]) => {
            g.strokeStyle = cl;
            g.lineWidth = 2.6;
            g.beginPath();
            for (let q = 1; q <= k; q++)
              q > 1 ? g.lineTo(X2(q), Y2(A[q])) : g.moveTo(X2(q), Y2(A[q]));
            g.stroke();
            if (k) {
              g.fillStyle = cl;
              g.beginPath();
              g.arc(X2(k), Y2(A[k]), 5, 0, 7);
              g.fill();
            }
          });
          g.textAlign = 'left';
          g.fillStyle = t.ok;
          g.fillText('red: plain network', m.l + 6, m.t);
          g.fillStyle = t.accent;
          g.fillText('green: LSTM', m.l + 150, m.t);
          g.fillStyle = t.muted;
          g.fillText('+1 = “plural”, −1 = “singular”', m.l + 6, h - 6);
        }
        {
          const { g, w, h } = fit(cd),
            y = h * 0.34,
            bx = (x, yy, tx, cl) => {
              g.fillStyle = t.bg;
              g.strokeStyle = cl || t.ink;
              g.lineWidth = 1.5;
              g.beginPath();
              g.rect(x - 46, yy - 14, 92, 28);
              g.fill();
              g.stroke();
              g.font = MONO;
              g.fillStyle = cl || t.ink;
              g.textAlign = 'center';
              g.fillText(tx, x, yy + 4);
            };
          g.font = MONO;
          g.strokeStyle = t.accent;
          g.lineWidth = 4;
          g.beginPath();
          g.moveTo(16, y);
          g.lineTo(w - 16, y);
          g.stroke();
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('old memory c', 16, y - 12);
          g.textAlign = 'right';
          g.fillText('new memory c', w - 16, y - 12);
          const xf = w * 0.32,
            xi = w * 0.62;
          [
            [xf, '×'],
            [xi, '+'],
          ].forEach(([x, s]) => {
            g.fillStyle = t.bg;
            g.strokeStyle = t.ink;
            g.lineWidth = 1.5;
            g.beginPath();
            g.arc(x, y, 12, 0, 7);
            g.fill();
            g.stroke();
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.font = SANS;
            g.fillText(s, x, y + 5);
            g.font = MONO;
            g.strokeStyle = t.ink;
            g.beginPath();
            g.moveTo(x, y + 12);
            g.lineTo(x, h * 0.66 - 14);
            g.stroke();
          });
          bx(xf, h * 0.66, 'forget ' + f2(S.f));
          bx(xi, h * 0.66, 'input × new', t.ok);
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('how much to keep', xf, h * 0.66 + 30);
          g.fillText('whether to write', xi, h * 0.66 + 30);
          g.fillText('current word and previous output feed every gate', w / 2, h - 8);
          if (pb && pb.on) {
            const p = (fr / 80) % 1;
            g.fillStyle = t.gold;
            g.beginPath();
            g.arc(16 + (w - 32) * p, y, 6, 0, 7);
            g.fill();
          }
        }
        live.textContent = msg;
        uCur(root, 'ls', cu);
        setR(root, 'r', f2(H[T2]) + ', ' + verdict(H[T2]));
        setR(root, 'l', f2(C[T2]) + ', ' + verdict(C[T2]));
        setR(root, 'rg', sci(Math.abs(GR[2])));
        setR(root, 'lg', sci(GL[2]));
      };
      comp();
      k = T2;
      const anim = Anim(() => {
        fr++;
        if (fr - t0 > (k < 3 ? 90 : 48) && k < T2) step();
        draw();
        if (k >= T2 && fr - t0 > 30) pb.stop();
      });
      pb = playBtn($('#ls-play', root), anim);
      runCtl(pb, root, 'ls', {
        done: () => k >= T2,
        restart: () => {
          k = 0;
        },
      });
      $('#ls-step', root).onclick = () => {
        pb.stop();
        step();
        t0 = fr - 100;
        draw();
      };
      $('#ls-reset', root).onclick = () => {
        pb.stop();
        k = 0;
        cu = 0;
        eq.innerHTML = '';
        msg = 'Press Read the sentence or One word.';
        draw();
      };
      const re = () => {
        pb.stop();
        comp();
        k = T2;
        msg =
          'With these settings the plain network ends at ' +
          f2(H[T2]) +
          ' (' +
          verdict(H[T2]) +
          ') and the LSTM at ' +
          f2(C[T2]) +
          ' (' +
          verdict(C[T2]) +
          ').';
        draw();
      };
      bind(root, 'ls-f', (v) => {
        S.f = v;
        re();
      });
      bind(root, 'ls-i', (v) => {
        S.i = v;
        re();
      });
      bind(root, 'ls-w', (v) => {
        S.w = v;
        re();
      });
      return {
        draw,
        stop: () => pb.stop(),
        auto: () => {
          comp();
          k = 0;
          draw();
          pb.start();
        },
      };
    },
  },

  {
    id: 'gnn',
    tab: 'Graph neural networks',
    title: 'Graph neural networks: tell me who your friends are',
    lede: 'Some data is not a table or a picture but a web of connections: friendships, molecules, roads, web pages. A graph neural network learns from such data by letting every node repeatedly ask its neighbours what they think and update its own view. Here two people in a small social network have told us which club they joined. Watch that knowledge spread along the friendships until everyone has an answer.',
    stage: `<div class="stage"><figure class="fig"><figcaption>A friendship network. Ringed nodes have a known answer. Colour is each node’s current belief: green club or gold club. Dots are messages. Click a node to change what is known about it</figcaption><canvas class="short" id="gn-c" role="img" aria-label="Graph with node beliefs and messages passing along edges"></canvas></figure>
 <figure class="fig"><figcaption>Every node’s belief, round by round. 1 = certainly gold, 0 = certainly green</figcaption><canvas class="short" id="gn-r" role="img" aria-label="Node beliefs over rounds of message passing"></canvas></figure></div>
 ${uHow('gn', ['Start: the labelled nodes know their club. Everyone else is undecided, at one half.', 'Message: every node sends its current belief along each of its friendships.', 'Aggregate: every node averages the messages it received together with its own belief.', 'Update: that average becomes its new belief. Labelled nodes keep their known answer.', 'Repeat. Each round carries information one friendship further. Stop when the beliefs no longer change.'])}
 <div class="controls">${uBtns('gn', 'Pass messages', 'One round', 'rounds')}
 ${grp('The model', `<label class="check"><input type="checkbox" id="gn-fix" checked> Labelled nodes keep their known answer</label><div class="row"><button class="btn" id="gn-clear" type="button">Forget all labels</button></div>`)}</div>
 <dl class="read">${rd('rd', 'Rounds')}${rd('ch', 'Largest change in the last round')}${rd('acc', 'Nodes placed in their true club', 'hl')}${rd('lab', 'Labelled nodes')}${rd('st', 'Status')}</dl>`,
    tryit:
      'Step through three rounds and watch the colour spread outward from the two ringed nodes, one friendship per round. Notice the nodes on the bridge between the two groups: they stay pale. Now untick “Labelled nodes keep their known answer” and run: every node drifts to the same colour. That is over-smoothing, and it is why very deep graph networks are hard to build. Then click nodes to label different people and see how few labels are enough.',
    math: String.raw`<h4>Intuition</h4><p>Suppose you move to a new town and want to know which football team a stranger supports. You do not know, but you know two of their friends, and each of those has friends whose team you do know. Ask around: each person forms a guess from their friends' guesses, and passes their own guess on. After a few rounds of gossip, a reasonable answer has reached everyone, because friends tend to agree. A graph neural network is this gossip, with learned rules for how to combine what is heard.</p>
 <h4>Message passing</h4><p>Let \(\mathbf h_i^{(t)}\) be the vector held by node \(i\) after round \(t\) and \(\mathcal N(i)\) its neighbours. One round, or one layer, is</p>$$\mathbf h_i^{(t+1)}=\phi\Big(\mathbf W\cdot\frac{1}{1+|\mathcal N(i)|}\Big[\mathbf h_i^{(t)}+\sum_{j\in\mathcal N(i)}\mathbf h_j^{(t)}\Big]\Big).$$<p>Each node averages its own vector with its neighbours', multiplies by a learned matrix \(\mathbf W\) and applies an activation. This is essentially the graph convolutional network of Kipf and Welling (2017). The same \(\mathbf W\) is used at every node, as a convolution reuses one kernel at every pixel, so the layer works on a graph of any size or shape.</p>
 <h4>What this page shows</h4><p>Each node holds one number, its belief that it is in the gold club, and the round is the plain average with no learned weights (\(\mathbf W=1\), no activation). Labelled nodes are reset to their known value after each round. <b>That is the untrained core of a graph network</b>, known on its own as label propagation. A trained network would also use each node's own features and learn \(\mathbf W\) so that the averaging emphasises what matters.</p>
 <h4>Depth is reach</h4><p>After one round a node knows about its friends; after two, about friends of friends. A network with \(k\) layers sees exactly \(k\) steps away. That is the graph version of the receptive field.</p>
 <h4>Over-smoothing</h4><p>Averaging with neighbours makes connected nodes more alike every round. With nothing to hold them apart, all nodes of a connected graph converge to the same value, and the network can no longer tell them apart. That is why practical graph networks are shallow, usually two to four layers, or use residual shortcuts back to each node's original features.</p>
 <h4>Where they are used</h4><p>Predicting properties of molecules (atoms are nodes, bonds are edges), recommending products (people and items), estimating travel times on road maps, detecting fraud in payment networks, and weather forecasting on a mesh of the globe. A Transformer is a graph network on a graph in which every token is connected to every other, with attention deciding the weight of each edge.</p>`,
    q: 'A node is five friendships away from the nearest labelled node. What is the smallest number of rounds after which its belief can differ from one half, and what does that say about choosing the number of layers?',
    a: 'Five. Information moves one friendship per round, so after four rounds nothing from a labelled node has reached it. In a graph network the number of layers sets how far each node can see, so it must be at least the distance over which the relevant information has to travel, but not much more, because each extra round also blurs neighbouring nodes together.',
    init(root) {
      const c = $('#gn-c', root),
        cr = $('#gn-r', root),
        live = $('#gn-live', root),
        S = { fix: true },
        NP = [
          [-2.6, 0.9],
          [-1.9, 1.7],
          [-1.5, 0.5],
          [-2.4, -0.4],
          [-1.3, -0.6],
          [-0.7, 1.2],
          [-0.6, 0.1],
          [-1.8, -1.5],
          [0.6, -0.2],
          [0.7, 1.1],
          [1.4, 0.3],
          [1.5, -1],
          [2.3, 1.2],
          [2.5, -0.2],
          [2.1, -1.6],
          [0.9, -1.7],
        ],
        ED = [
          [0, 1],
          [0, 2],
          [0, 3],
          [1, 2],
          [1, 5],
          [2, 4],
          [2, 5],
          [2, 6],
          [3, 4],
          [3, 7],
          [4, 6],
          [4, 7],
          [5, 6],
          [6, 8],
          [8, 9],
          [8, 10],
          [8, 11],
          [9, 10],
          [9, 12],
          [10, 12],
          [10, 13],
          [11, 13],
          [11, 14],
          [11, 15],
          [13, 14],
          [12, 13],
          [14, 15],
          [5, 9],
        ],
        TR = NP.map((_, i) => (i < 8 ? 0 : 1)),
        N = 16;
      let L,
        h,
        hist,
        rd = 0,
        ch = 1,
        conv = false,
        msg = '',
        cu = 0,
        fr = 0,
        t0 = -999,
        pb,
        chk = () => false,
        prev,
        PX = [];
      const nb = NP.map(() => []);
      ED.forEach((e) => {
        nb[e[0]].push(e[1]);
        nb[e[1]].push(e[0]);
      });
      const reset = () => {
        h = L.map((v) => (v < 0 ? 0.5 : v));
        prev = h.slice();
        hist = [h.slice()];
        rd = 0;
        ch = 1;
        conv = false;
        cu = 0;
        t0 = -999;
        msg =
          L.filter((v) => v >= 0).length +
          ' people have told us their club. Everyone else starts undecided at one half. Press Pass messages or One round.';
      };
      const round = () => {
        prev = h.slice();
        const nh = h.map((v, i) => {
          let s = v;
          for (const j of nb[i]) s += h[j];
          return s / (1 + nb[i].length);
        });
        if (S.fix)
          L.forEach((v, i) => {
            if (v >= 0) nh[i] = v;
          });
        ch = Math.max(...nh.map((v, i) => Math.abs(v - h[i])));
        h = nh;
        hist.push(h.slice());
        rd++;
        t0 = fr;
        if (ch < 0.004 || rd >= 80) conv = true;
        cu = conv ? 4 : 1 + (rd % 3);
        const und = h.filter((v) => Math.abs(v - 0.5) < 0.02).length;
        msg =
          'Round ' +
          rd +
          ': every node averaged its own belief with those of its friends. The largest change was ' +
          f3(ch) +
          '. ' +
          und +
          ' node' +
          (und === 1 ? ' is' : 's are') +
          ' still undecided.' +
          (conv
            ? (rd >= 80
                ? ' Reached the limit of 80 rounds.'
                : ' Converged: the beliefs have stopped changing.') +
              (Math.max(...h) - Math.min(...h) < 0.05
                ? ' Every node now holds almost the same value: the network has over-smoothed and can no longer tell the two groups apart.'
                : '')
            : '');
      };
      const draw = () => {
        const t = T(),
          u = RM ? 1 : clamp((fr - t0) / 36, 0, 1),
          cv = (v) => mix(t.accent, t.gold, clamp(v, 0, 1));
        {
          const V = uView(c),
            { g, w, h: hh, X, Y } = V;
          PX = NP.map((p) => [X(p[0] * 1.25), Y(p[1] * 1.25)]);
          g.strokeStyle = mix(t.ink, t.ink, 0, 0.3);
          g.lineWidth = 1.4;
          ED.forEach((e) => {
            g.beginPath();
            g.moveTo(...PX[e[0]]);
            g.lineTo(...PX[e[1]]);
            g.stroke();
          });
          if (u < 1 && rd > 0)
            ED.forEach((e) => {
              [
                [e[0], e[1]],
                [e[1], e[0]],
              ].forEach(([a, b]) => {
                g.fillStyle = cv(prev[a]);
                g.strokeStyle = t.bg;
                g.lineWidth = 1;
                g.beginPath();
                g.arc(
                  PX[a][0] + (PX[b][0] - PX[a][0]) * u,
                  PX[a][1] + (PX[b][1] - PX[a][1]) * u,
                  4.5,
                  0,
                  7,
                );
                g.fill();
                g.stroke();
              });
            });
          for (let i = 0; i < N; i++) {
            const v = rd > 0 ? prev[i] + (h[i] - prev[i]) * (u < 0.8 ? 0 : (u - 0.8) / 0.2) : h[i],
              [x, y] = PX[i];
            g.fillStyle =
              Math.abs(v - 0.5) < 0.02
                ? mix(t.ink, t.ink, 0, 0.25)
                : mix(v < 0.5 ? t.accent : t.gold, t.bg, 0, 0.25 + 0.75 * Math.abs(v - 0.5) * 2);
            g.strokeStyle = t.ink;
            g.lineWidth = L[i] >= 0 ? 3.4 : 1.2;
            g.beginPath();
            g.arc(x, y, 15, 0, 7);
            g.fill();
            g.stroke();
            g.font = MONO;
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.fillText(f2(v), x, y + 29);
          }
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('thick ring: club known.  click: unknown → green → gold', 8, hh - 8);
        }
        {
          const { g, w, h: hh } = fit(cr),
            m = { l: 36, r: 12, t: 16, b: 22 },
            nR = Math.max(12, hist.length - 1),
            X2 = (q) => m.l + (q / nR) * (w - m.l - m.r),
            Y2 = (v) => hh - m.b - v * (hh - m.t - m.b);
          g.font = MONO;
          [0, 0.5, 1].forEach((v) => {
            g.strokeStyle = t.line;
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(m.l, Y2(v));
            g.lineTo(w - m.r, Y2(v));
            g.stroke();
            g.fillStyle = t.muted;
            g.textAlign = 'right';
            g.fillText(String(v), m.l - 5, Y2(v) + 4);
          });
          for (let i = 0; i < N; i++) {
            g.strokeStyle = mix(TR[i] ? t.gold : t.accent, t.bg, 0, 0.85);
            g.lineWidth = L[i] >= 0 ? 3 : 1.5;
            g.beginPath();
            hist.forEach((r, q) => (q ? g.lineTo(X2(q), Y2(r[i])) : g.moveTo(X2(q), Y2(r[i]))));
            g.stroke();
          }
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('line colour: the node’s true club.  thick: labelled', m.l + 4, hh - 5);
        }
        live.textContent = msg;
        uCur(root, 'gn', cu);
        const ok = h.filter(
          (v, i) => Math.abs(v - 0.5) >= 0.02 && (v > 0.5 ? 1 : 0) === TR[i],
        ).length;
        setR(root, 'rd', String(rd));
        setR(root, 'ch', rd ? f3(ch) : '–');
        setR(root, 'acc', ok + ' of ' + N);
        setR(root, 'lab', String(L.filter((v) => v >= 0).length));
        setR(root, 'st', conv ? 'Converged' : rd ? 'Running' : 'Not started');
      };
      L = NP.map(() => -1);
      L[0] = 0;
      L[13] = 1;
      reset();
      if (RM) while (!conv) round();
      const anim = Anim(() => {
        fr++;
        if (fr - t0 > (rd < 4 ? 80 : 44) && !conv) {
          round();
          if (chk()) {
            draw();
            return;
          }
        }
        draw();
        if (conv && fr - t0 > 40) pb.stop();
      });
      pb = playBtn($('#gn-play', root), anim);
      chk = runCtl(pb, root, 'gn', {
        done: () => conv && fr - t0 > 40,
        restart: () => reset(),
        count: () => rd,
        what: 'rounds',
      });
      $('#gn-step', root).onclick = () => {
        pb.stop();
        if (conv) reset();
        else round();
        t0 = fr - 100;
        draw();
      };
      $('#gn-reset', root).onclick = () => {
        pb.stop();
        reset();
        draw();
      };
      $('#gn-fix', root).onchange = (e) => {
        S.fix = e.target.checked;
        pb.stop();
        reset();
        draw();
        pb.start();
      };
      $('#gn-clear', root).onclick = () => {
        pb.stop();
        L = NP.map(() => -1);
        reset();
        msg = 'No labels. Click a node to tell the network which club that person joined.';
        draw();
      };
      c.addEventListener('click', (e) => {
        const r = c.getBoundingClientRect(),
          x = e.clientX - r.left,
          y = e.clientY - r.top;
        PX.forEach((p, i) => {
          if (Math.hypot(p[0] - x, p[1] - y) < 18) {
            pb.stop();
            L[i] = L[i] < 0 ? 0 : L[i] === 0 ? 1 : -1;
            reset();
            draw();
            pb.start();
          }
        });
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
    id: 'transformer',
    tab: 'Transformers',
    title: 'Train a Transformer and watch it work out what to look at',
    lede: 'This is a real Transformer, small enough to train in your browser in seconds. Its job: read five letters, then write them out backwards. Nobody tells it how. Watch the attention square as it learns: a diagonal line appears, which is the network discovering that to write the first output it must look at the last input. Then give it your own letters.',
    stage: `<div class="row seg" id="tf-task" role="group" aria-label="Task"><button class="btn" type="button" data-v="rev">Task: Write It Backwards</button><button class="btn" type="button" data-v="copy">Task: Copy It</button></div>
 <div class="stage"><figure class="fig"><figcaption>Top: the letters going in, the ▸ that says “now answer”, and the letters it writes. Arcs: where the highlighted output is looking. Bottom: the attention square, one row per position that has to predict</figcaption><canvas class="short" id="tf-c" role="img" aria-label="Sequence with attention arcs and the learned attention matrix"></canvas></figure>
 <figure class="fig"><figcaption>Top: its confidence in each letter for the highlighted output. Middle: error while training. Bottom: share of fresh sequences it gets entirely right</figcaption><canvas class="short" id="tf-r" role="img" aria-label="Next-letter probabilities, loss and accuracy"></canvas></figure></div>
 <div class="stage one" style="margin-top:18px"><figure class="fig"><figcaption>The same sequence as tables of numbers, one row per position: letter-plus-position vectors going in, what attention adds, and what leaves the block. Arcs are the shortcuts</figcaption><canvas class="tall" id="tf-s" style="aspect-ratio:16/3.6" role="img" aria-label="Token vectors at each stage of the trained block"></canvas></figure></div>
 ${uHow('tf', ['Make up a random sequence of five letters, and the right answer for it.', 'Forward: letters become vectors, positions are added, attention lets each position gather from earlier ones, and a small network turns the result into a guess for the next letter.', 'Compare the guesses with the right answer: the loss.', 'Backward: backpropagation finds how every weight should change, including the ones that decide where attention looks.', 'Repeat with fresh sequences until it gets them all right. Then it answers sequences it has never seen.'])}
 <div class="controls">${uBtns('tf', 'Train', '25 steps', 'steps')}
 ${grp('Try it yourself', `<label class="ctl" for="tf-text"><span>Five letters from a to h</span><input type="text" class="num" id="tf-text" maxlength="5" value="cafeb" autocomplete="off" spellcheck="false"></label>` + slider('tf-pos', 'Output to inspect', 1, 5, 1, 1) + `<p class="status" id="tf-st" role="status"></p>`)}</div>
 <dl class="read">${rd('step', 'Training steps')}${rd('L', 'Error', 'hl2')}${rd('acc', 'Fresh sequences fully correct', 'hl')}${rd('out', 'Its answer for your letters', 'hl3')}${rd('look', 'The inspected output looks mostly at')}${rd('np', 'Weights in the model')}</dl>`,
    tryit:
      'Press Start over and watch the lower-left square while it trains: first a blur, then a line. For Write It Backwards the line runs from top right to bottom left; switch the task to Copy It and it runs the other way. Type your own five letters and move the Output to Inspect slider: the arcs show which input letter each output is reading. Before training, where does it look?',
    math: String.raw`<h4>Intuition</h4><p>Picture a meeting where everyone holds a page of notes about themselves. In each round two things happen. First, everyone looks around the room and copies onto their own page whatever is relevant from the others' pages: that is attention. Then everyone goes quiet and thinks over their own page alone, rewriting it: that is the feed-forward layer. Nobody throws their old page away; they add to it, which is the shortcut. After many rounds each page describes its owner in the light of everyone else in the room. A Transformer runs that meeting with words.</p>
 <h4>The stages</h4><p>With \(n\) tokens and vectors of size \(d\), stack the token vectors as the rows of an \(n\times d\) matrix.</p>$$\mathbf X=\mathbf E+\mathbf P\qquad\text{(embeddings plus positions)}$$$$\mathbf A=\operatorname{softmax}\!\Big(\frac{(\mathbf X\mathbf W_Q)(\mathbf X\mathbf W_K)^{\top}}{\sqrt d}+\mathbf M\Big),\qquad\mathbf Z=\mathbf A\,(\mathbf X\mathbf W_V)\qquad\text{(self-attention)}$$$$\mathbf R_1=\operatorname{norm}(\mathbf X+\mathbf Z)\qquad\text{(shortcut and normalise)}$$$$\mathbf F=\max(0,\;\mathbf R_1\mathbf W_1)\,\mathbf W_2,\qquad\mathbf R_2=\operatorname{norm}(\mathbf R_1+\mathbf F)\qquad\text{(feed-forward, shortcut, normalise)}$$<p>The mask \(\mathbf M\) is \(-\infty\) wherever a word would look at a later word and 0 elsewhere, which makes those attention shares exactly zero.</p>
 <h4>Positions</h4><p>Attention treats its inputs as a set: shuffle the words and each gets the same result. Order is restored by adding to each word a vector that depends only on its position. The original design uses waves of different lengths,</p>$$P_{pos,2i}=\sin\!\Big(\frac{pos}{10000^{2i/d}}\Big),\qquad P_{pos,2i+1}=\cos\!\Big(\frac{pos}{10000^{2i/d}}\Big),$$<p>so that each position has a distinct pattern. This page learns its position vectors instead, which works as well for short sequences.</p>
 <h4>Predicting the next word</h4><p>The final vector of the last token is compared with the embedding of every word in the vocabulary, \(\text{logits}=\mathbf r\,\mathbf E_{\text{vocab}}^{\top}\), and a softmax turns the scores into probabilities. Training adjusts every matrix so that the true next word gets a high probability, using backpropagation as in any network. With the mask, one pass through a sentence trains the prediction at every position at once.</p>
 <h4>A real model</h4><p>GPT-style models stack dozens of such blocks, use vectors of thousands of numbers, and run many attention "heads" side by side in each block, each with its own \(\mathbf W_Q,\mathbf W_K,\mathbf W_V\). The parts are the ones shown here.</p>
 <h4>What this page trains</h4><p>One block with one attention head, vectors of 16 numbers, and about 2,000 weights. For simplicity it leaves out normalisation and learns its position vectors. It is trained by backpropagation with the Adam optimiser on random five-letter sequences, 12 at a time, and stops when it writes every fresh sequence correctly.</p><p>What it has to discover is a rule about <b>positions</b>: to write output number \(i\) of the reversed sequence it must read input number \(6-i\). So the query and key matrices learn to use the position vectors, which is the line you see in the attention square. The value matrix learns to carry the letter itself across, and the final layer reads it out. Nothing in the code states this rule. It emerges because it lowers the loss.</p>`,
    q: 'Without the position signal, would “dog bites man” and “man bites dog” give the word “bites” different vectors at the end of the block?',
    a: 'No. Attention computes each word’s output from the set of all word vectors, weighted by how well they match. Without positions the same three vectors are present in both sentences, so every score, every share and every output is identical; the two sentences are indistinguishable to the block. Adding a different position vector to each slot is what lets it tell who did the biting.',
    init(root) {
      const c = $('#tf-c', root),
        cr = $('#tf-r', root),
        cs = $('#tf-s', root),
        live = $('#tf-live', root),
        st = $('#tf-st', root),
        S = { task: 'rev', text: 'cafeb', pos: 1 },
        n = 5,
        V = 8,
        d = 16,
        T2 = 2 * n,
        LT = 'abcdefgh';
      let P,
        ps,
        ts = 0,
        LL = [],
        AC = [],
        conv = false,
        good = 0,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        demo = null;
      const build = () => {
        const q = randnFrom(rng(run * 11 + 5)),
          pm = (a, b, sd) => AD.param([a, b], sd, q);
        P = {
          E: pm(V + 1, d, 0.3),
          Pos: pm(T2, d, 0.3),
          Wq: pm(d, d, 0.3),
          Wk: pm(d, d, 0.3),
          Wv: pm(d, d, 0.3),
          W1: pm(d, 2 * d, 0.25),
          W2: pm(2 * d, d, 0.25),
          Wu: pm(d, V + 1, 0.3),
        };
        ps = Object.values(P);
        ts = 0;
        LL = [];
        AC = [];
        conv = false;
        good = 0;
        cu = 0;
        run2();
        msg =
          'A Transformer with random weights. It has never seen the task. Its answer for “' +
          S.text +
          '” is “' +
          demo.out +
          '”. Press Train.';
      };
      const fwd = (tok, tg) => {
        const X = AD.add(
            AD.gather(P.E, tok),
            AD.gather(
              P.Pos,
              tok.map((_, i) => i),
            ),
          ),
          A = AD.attn(AD.mmT(AD.mm(X, P.Wq), AD.mm(X, P.Wk)), 1 / Math.sqrt(d), true),
          Z = AD.mm(A, AD.mm(X, P.Wv)),
          R1 = AD.add(X, Z),
          R2 = AD.add(R1, AD.mm(AD.relu(AD.mm(R1, P.W1)), P.W2)),
          L = AD.mm(R2, P.Wu);
        return { X, A, Z, R2, L, loss: tg ? AD.ce(L, tg, n) : null };
      };
      const ans = (x) => (S.task === 'rev' ? x.slice().reverse() : x.slice()),
        mk = () => {
          const x = Array.from({ length: n }, () => Math.floor(Math.random() * V)),
            y = ans(x),
            tg = new Array(T2).fill(0);
          for (let i = 0; i < n; i++) tg[n + i] = y[i];
          return { tok: x.concat([V], y.slice(0, n - 1)), tg };
        };
      const gen = (x) => {
        const tok = x.concat([V]);
        let o;
        for (let i = 0; i < n; i++) {
          o = fwd(tok);
          AD.clear();
          const r = (tok.length - 1) * (V + 1);
          let b = 0;
          for (let j = 0; j < V; j++) if (o.L.v[r + j] > o.L.v[r + b]) b = j;
          if (i < n - 1) tok.push(b);
          else tok.out = tok.slice(n + 1).concat([b]);
        }
        return { tok, o, out: tok.out };
      };
      const run2 = () => {
        const x = [...S.text].map((ch) => Math.max(0, LT.indexOf(ch)));
        while (x.length < n) x.push(0);
        const gq = gen(x),
          pr = [];
        for (let t = n; t < T2; t++) {
          const row = [];
          let mx = -1e30,
            z = 0;
          for (let j = 0; j <= V; j++) mx = Math.max(mx, gq.o.L.v[t * (V + 1) + j]);
          for (let j = 0; j <= V; j++) {
            const e = Math.exp(gq.o.L.v[t * (V + 1) + j] - mx);
            row.push(e);
            z += e;
          }
          pr.push(row.map((v) => v / z));
        }
        demo = {
          x,
          tok: gq.tok,
          A: gq.o.A.v,
          X: gq.o.X.v,
          Z: gq.o.Z.v,
          R2: gq.o.R2.v,
          pr,
          outI: gq.out,
          out: gq.out.map((i) => LT[i]).join(''),
          want: ans(x)
            .map((i) => LT[i])
            .join(''),
        };
      };
      const evalAcc = () => {
        let ok = 0;
        for (let k = 0; k < 40; k++) {
          const x = Array.from({ length: n }, () => Math.floor(Math.random() * V));
          if (gen(x).out.join() === ans(x).join()) ok++;
        }
        return ok / 40;
      };
      const train = (k) => {
        for (let q = 0; q < k && !conv; q++) {
          let l = 0;
          for (let b = 0; b < 12; b++) {
            const s = mk(),
              o = fwd(s.tok, s.tg);
            l += o.loss.v[0];
            AD.backward(o.loss);
          }
          AD.adam(ps, 0.004, ++ts, 12);
          LL.push(l / 12);
          if (ts % 10 === 0) {
            const a = evalAcc();
            AC.push(a);
            good = a >= 0.99 ? good + 1 : 0;
            if (good >= 3 || ts >= 4000) conv = true;
          }
        }
        run2();
        cu = conv ? 4 : 1 + (ts % 3);
        msg =
          'Step ' +
          ts +
          ': 12 fresh sequences went forward, the error was ' +
          f3(LL[ts - 1]) +
          ', and backpropagation adjusted all ' +
          ps.reduce((a, p) => a + p.v.length, 0) +
          ' weights. For “' +
          S.text +
          '” it now writes “' +
          demo.out +
          '” (the right answer is “' +
          demo.want +
          '”).' +
          (conv
            ? ts >= 4000
              ? ' Reached the limit of 4000 steps.'
              : ' Converged: it gets every fresh sequence right, so training stops. Type your own letters.'
            : '');
      };
      const draw = () => {
        const t = T(),
          D = demo,
          pi = n + S.pos - 1,
          lab = (i) => (i < n ? LT[D.x[i]] : i === n ? '▸' : LT[D.tok[i]]),
          u = 1;
        {
          const { g, w, h } = fit(c),
            bw = Math.min(46, (w - 20) / 11.6),
            x0 = (w - bw * 11.4) / 2,
            X = (i) => x0 + (i < n ? i : i === n ? i + 0.2 : i + 0.4) * bw + bw / 2,
            yw = h * 0.2,
            cell = Math.min((w - 140) / T2, (h * 0.62 - 30) / n),
            mx = (w - cell * T2) / 2 + 20,
            my = h * 0.4 + 6;
          g.font = SANS;
          for (let j = 0; j <= pi; j++) {
            const wt = D.A[pi * T2 + j];
            if (j === pi || wt < 0.02) continue;
            g.strokeStyle = mix(t.gold, t.gold, 0, 0.2 + 0.8 * wt);
            g.lineWidth = 0.5 + 10 * wt;
            const xo = pi + 1 < T2 ? X(pi + 1) : X(T2 - 1) + bw;
            g.beginPath();
            g.moveTo(xo, yw - 15);
            g.quadraticCurveTo(
              (xo + X(j)) / 2,
              yw - 15 - Math.min(h * 0.14, Math.abs(xo - X(j)) * 0.3),
              X(j),
              yw - 15,
            );
            g.stroke();
          }
          for (let i = 0; i < T2 + 1; i++) {
            const isOut = i > n,
              x = i < T2 ? X(i) : X(T2 - 1) + bw,
              ch = i <= n ? lab(i) : LT[D.outI[i - n - 1]],
              ok = isOut && ch === D.want[i - n - 1],
              hi = i === pi + 1 || (i === pi && false);
            if (i === T2) void 0;
            g.fillStyle =
              i === pi + 1 ? t.gold : isOut ? mix(ok ? t.accent : t.ok, t.bg, 0, 0.18) : t.bg;
            g.strokeStyle = isOut ? (ok ? t.accent : t.ok) : t.ink;
            g.lineWidth = i === pi + 1 ? 2.6 : 1.3;
            g.beginPath();
            g.rect(x - bw * 0.42, yw - 14, bw * 0.84, 28);
            g.fill();
            g.stroke();
            g.fillStyle = i === pi + 1 ? t.bg : t.ink;
            g.textAlign = 'center';
            g.fillText(ch, x, yw + 5);
          }
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'center';
          g.fillText('input', X(2), yw + 30);
          g.fillText('its output', X(n + 3), yw + 30);
          for (let r = 0; r < n; r++) {
            const ti = n + r;
            g.fillStyle = r === S.pos - 1 ? t.ink : t.muted;
            g.textAlign = 'right';
            g.fillText('output ' + (r + 1), mx - 6, my + (r + 0.65) * cell);
            for (let j = 0; j < T2; j++) {
              const v = j <= ti ? D.A[ti * T2 + j] : -1;
              g.fillStyle =
                v < 0 ? mix(t.ink, t.ink, 0, 0.13) : mix(t.bg, t.gold, Math.pow(v, 0.6));
              g.fillRect(mx + j * cell, my + r * cell, cell - 1.5, cell - 1.5);
            }
            if (r === S.pos - 1) {
              g.strokeStyle = t.ink;
              g.lineWidth = 1.6;
              g.strokeRect(mx - 1, my + r * cell - 1, cell * T2, cell);
            }
          }
          for (let j = 0; j < T2; j++) {
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(lab(j), mx + (j + 0.5) * cell, my + n * cell + 13);
          }
          g.textAlign = 'left';
          g.fillText('looks at →   (grey: not yet written, hidden by the mask)', mx, my - 7);
        }
        {
          const { g, w, h } = fit(cr),
            ha = h * 0.34,
            pr = D.pr[S.pos - 1],
            bw = (w - 30) / (V + 1);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText(
            'output ' +
              S.pos +
              ': confidence in each letter (should be “' +
              D.want[S.pos - 1] +
              '”)',
            8,
            13,
          );
          for (let j = 0; j <= V; j++) {
            const x = 15 + j * bw,
              p = pr[j],
              top = ha - 22 - p * (ha - 48);
            g.fillStyle = j < V && LT[j] === D.want[S.pos - 1] ? t.accent : t.gold;
            g.fillRect(x + bw * 0.15, top, bw * 0.7, ha - 22 - top);
            g.font = SANS;
            g.fillStyle = t.ink;
            g.textAlign = 'center';
            g.fillText(j < V ? LT[j] : '▸', x + bw / 2, ha - 6);
            g.font = MONO;
            if (p > 0.04) {
              g.fillStyle = t.muted;
              g.fillText((100 * p).toFixed(0), x + bw / 2, top - 4);
            }
          }
          const hb = (h - ha) / 2;
          uLine(g, t, 0, ha, w, hb, LL, {
            title: 'error, by training step',
            n: Math.max(200, LL.length),
            col: t.ok,
            ym: Math.max(2.2, LL[0] || 2.2),
          });
          uLine(g, t, 0, ha + hb, w, hb, AC, {
            title: 'fresh sequences written entirely correctly (checked every 10 steps)',
            n: Math.max(20, AC.length),
            col: t.accent,
            ym: 1,
            dots: AC.length < 40,
          });
        }
        {
          const { g, w, h } = fit(cs),
            lm = 64,
            tw = (w - lm - 10) / 3,
            cw = Math.min((tw * 0.8) / d, 22),
            chh = (h - 48) / T2,
            y0 = 30,
            TB = [
              [D.X, 'letter + position vectors, X'],
              [D.Z, 'what attention adds, Z'],
              [D.R2, 'leaving the block, R₂'],
            ];
          g.font = MONO;
          for (let i = 0; i < T2; i++) {
            g.fillStyle = i === pi ? t.ink : t.muted;
            g.textAlign = 'right';
            g.fillText(
              (i < n ? 'in ' : i === n ? '' : 'out ') + lab(i),
              lm - 8,
              y0 + (i + 0.7) * chh,
            );
          }
          TB.forEach((tb, q) => {
            const x0 = lm + q * tw;
            g.fillStyle = t.muted;
            g.textAlign = 'left';
            g.fillText(tb[1], x0, y0 - 10);
            let m = 1e-9;
            for (const v of tb[0]) m = Math.max(m, Math.abs(v));
            for (let i = 0; i < T2; i++)
              for (let j = 0; j < d; j++) {
                const v = tb[0][i * d + j] / m;
                g.fillStyle = mix(t.bg, v >= 0 ? t.gold : t.accent, Math.min(1, Math.abs(v) * 1.3));
                g.fillRect(x0 + j * cw, y0 + i * chh, cw - 0.8, chh - 0.8);
              }
            g.strokeStyle = t.ink;
            g.lineWidth = 1.4;
            g.strokeRect(x0 - 1, y0 + pi * chh - 1, cw * d + 1, chh + 1);
            if (q < 2) {
              const xa = x0 + cw * d + 4,
                xb = x0 + tw - 6,
                ym = y0 + (chh * T2) / 2;
              g.strokeStyle = t.muted;
              g.lineWidth = 1.2;
              g.beginPath();
              g.moveTo(xa, ym);
              g.lineTo(xb, ym);
              g.lineTo(xb - 5, ym - 4);
              g.moveTo(xb, ym);
              g.lineTo(xb - 5, ym + 4);
              g.stroke();
              g.fillStyle = t.muted;
              g.textAlign = 'center';
              g.fillText(q ? '+ X, feed-forward, +' : 'attention', (xa + xb) / 2, ym - 8);
            }
          });
        }
        live.textContent = msg;
        uCur(root, 'tf', cu);
        let b = 0;
        for (let j = 0; j <= pi; j++) if (D.A[pi * T2 + j] > D.A[pi * T2 + b]) b = j;
        setR(root, 'step', String(ts));
        setR(root, 'L', ts ? f3(LL[ts - 1]) : '–');
        setR(root, 'acc', AC.length ? pct(AC[AC.length - 1]) : '–');
        setR(
          root,
          'out',
          '“' + D.out + '”' + (D.out === D.want ? ' (correct)' : ' (should be “' + D.want + '”)'),
        );
        setR(
          root,
          'look',
          (b < n
            ? 'input letter ' + (b + 1) + ', “' + LT[D.x[b]] + '”'
            : b === n
              ? 'the ▸ sign'
              : 'its own earlier output “' + lab(b) + '”') +
            ' (' +
            (100 * D.A[pi * T2 + b]).toFixed(0) +
            '%)',
        );
        setR(root, 'np', String(ps.reduce((a, p) => a + p.v.length, 0)));
      };
      build();
      if (RM) while (!conv) train(20);
      const anim = Anim(() => {
        fr++;
        if (fr % (ts < 30 ? 5 : ts < 150 ? 2 : 1)) return;
        train(ts < 150 ? 1 : 3);
        draw();
        chk();
      });
      pb = playBtn($('#tf-play', root), anim);
      chk = runCtl(pb, root, 'tf', {
        done: () => conv,
        restart: () => {
          run++;
          build();
        },
        count: () => ts,
        what: 'steps',
      });
      $('#tf-step', root).onclick = () => {
        pb.stop();
        if (conv) {
          run++;
          build();
        } else train(25);
        draw();
      };
      $('#tf-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      segBind(root, 'tf-task', S.task, (v) => {
        S.task = v;
        pb.stop();
        run++;
        build();
        draw();
        pb.start();
      });
      bind(
        root,
        'tf-pos',
        (v) => {
          S.pos = v;
          draw();
        },
        (v) => String(v),
      );
      $('#tf-text', root).addEventListener('input', (e) => {
        const v = e.target.value.toLowerCase().replace(/[^a-h]/g, '');
        st.textContent = v.length < 5 ? 'Type five letters, using only a to h.' : '';
        if (v.length === 5) {
          S.text = v;
          run2();
          msg =
            'For “' +
            S.text +
            '” it writes “' +
            demo.out +
            '”' +
            (demo.out === demo.want
              ? ', which is correct' +
                (conv ? '. It was never trained on this particular sequence.' : '.')
              : '. The right answer is “' + demo.want + '”.');
          if (!pb.on) draw();
        }
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
    id: 'unet',
    tab: 'U-Net',
    title: 'Train a U-Net: find the shape in the noise, pixel by pixel',
    lede: 'Some tasks need an answer for every pixel: which pixels are tumour, which are road. Here two small convolutional networks learn, live, to pick out shapes from noisy pictures and mark every pixel as shape or background. They are identical except for one thing: one has skip connections and one does not. Watch their answers side by side, then draw your own shape and see what each makes of it.',
    stage: `<div class="stage"><figure class="fig"><figcaption>Each row is one picture the networks have never trained on. Columns: the noisy input, the true shape, the answer without skips, the answer with skips. The last row is yours: drag in its first box to draw</figcaption><canvas class="sq" id="un-c" role="img" aria-label="Noisy inputs, true masks and the outputs of both networks"></canvas></figure>
 <figure class="fig"><figcaption>Top: the U-shaped network. The picture shrinks to 4 × 4 and grows back; the green rungs are the skip connections. Bottom: overlap between each network’s answer and the truth on test pictures</figcaption><canvas class="sq" id="un-r" role="img" aria-label="U-Net architecture and the test overlap curves"></canvas></figure></div>
 ${uHow('un', ['Make a random picture of one or two thin shapes, add noise, and keep the clean shape as the answer.', 'Down: convolutions find edges, pooling halves the size, twice. At 4 × 4 the network knows roughly what is where and has lost the fine lines.', 'Up: enlarge twice back to 16 × 16. With skips, each level is also handed the feature maps from the same size on the way down.', 'Compare the output with the true shape, pixel by pixel, and backpropagate through every convolution.', 'Repeat with fresh pictures. Check both networks on pictures they have not seen.'])}
 <div class="controls">${uBtns('un', 'Train both', 'Ten steps', 'steps')}
 ${grp('The data', slider('un-nz', 'Noise in the input pictures', 0, 0.6, 0.01, 0.3) + `<div class="row"><button class="btn" id="un-clear" type="button">Clear my drawing</button></div>`)}</div>
 <dl class="read">${rd('step', 'Training steps')}${rd('i0', 'Without skips: overlap with the truth', 'hl2')}${rd('i1', 'With skips: overlap with the truth', 'hl3')}${rd('np', 'Weights: without / with skips')}${rd('st', 'Status')}</dl>`,
    tryit:
      'Press Start over and watch the two right-hand columns. Both begin as grey fog. The one with skips sharpens into thin clean lines; the one without stays blobby, because everything it knows had to pass through a 4 × 4 bottleneck. Draw a small circle in the bottom row, then a single thin line. Which network can draw a one-pixel line, and why can the other not?',
    math: String.raw`<h4>Intuition</h4><p>To describe a photograph over the phone you would first step back and say what it shows: "a face, in the middle". That overview needs the picture shrunk in your mind, with the detail ignored. But if the listener must then redraw it exactly, the overview is not enough: they need the fine lines back. A U-Net does both. One half steps back, level by level, to grasp the content. The other half redraws at full size, and at each level it is handed the detail that was set aside at the same level on the way down.</p>
 <h4>The shape</h4><p>The <b>encoder</b> (down path) alternates convolutions with pooling, halving the width and height each time and increasing the number of feature maps, so that later layers see large areas. The <b>decoder</b> (up path) alternates upsampling with convolutions back to full size. A <b>skip connection</b> copies the encoder's feature maps at each size and joins them to the decoder's maps of the same size:</p>$$\mathbf d_l=\operatorname{conv}\big(\big[\operatorname{up}(\mathbf d_{l+1}),\;\mathbf e_l\big]\big).$$<p>Drawn with the sizes shrinking then growing, the network forms a U, and the skips are the horizontal rungs (Ronneberger and others, 2015).</p>
 <h4>Why the skips are needed</h4><p>Halving a picture three times leaves one number for every 64 pixels. Exact positions of edges are gone and no amount of upsampling can bring them back: the decoder would have to invent them. The skip supplies them directly. The deep path answers "what is here?", the skip answers "exactly where?", and the decoder combines the two.</p>
 <h4>What this page trains</h4><p>Two real convolutional networks, trained live by backpropagation. Each has three \(3\times3\) convolution layers on the way down (4, 6 and 8 feature maps at sizes 16, 8 and 4), two on the way up and a final one that outputs one number per pixel, turned into a probability by a sigmoid. Pooling is \(2\times2\) max pooling and enlargement repeats each pixel. The loss is the cross-entropy between the output and the true shape at every pixel. The two networks start from the same random kernels and see the same pictures; the only difference is whether the encoder's maps are joined to the decoder's at sizes 8 and 16. The score shown is intersection over union: pixels both mark as shape, divided by pixels either marks.</p>
 <h4>Where U-Nets are used</h4><p>Segmenting medical scans (the original purpose), satellite images and road scenes; removing noise and sharpening images; and as the denoising network inside image diffusion models, which you will meet in the Generative AI module. There the input is a noisy picture and the output is an estimate of the noise, pixel for pixel.</p>`,
    q: 'A U-Net has its skip connections removed and is trained to outline cells in microscope images. What kind of mistakes would you expect in its output?',
    a: 'The right objects in roughly the right places, but with blurred, blobby outlines: neighbouring cells merged, thin gaps closed, small cells missed. The bottleneck keeps what is in the image and approximately where, and loses the exact positions of edges. Without the skips the decoder has to guess the boundaries from the coarse map.',
    init(root) {
      const c = $('#un-c', root),
        cr = $('#un-r', root),
        live = $('#un-live', root),
        S = { nz: 0.3 },
        MAXS = 700;
      let NA,
        NB,
        ts = 0,
        IA = [],
        IB = [],
        TE = [],
        pad = new Float64Array(256),
        OUT = [],
        done = false,
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        geo = null,
        paint = false;
      const mkNet = (skip, seed) => {
        const q = randnFrom(rng(seed)),
          cw = (o, ci) => AD.param([o, ci, 3, 3], Math.sqrt(2 / (ci * 9)), q),
          cb = (o) => AD.param([o], 0, q),
          P = {
            w1: cw(4, 1),
            b1: cb(4),
            w2: cw(6, 4),
            b2: cb(6),
            w3: cw(8, 6),
            b3: cb(8),
            w4: cw(6, skip ? 14 : 8),
            b4: cb(6),
            w5: cw(4, skip ? 10 : 6),
            b5: cb(4),
            w6: cw(1, 4),
            b6: cb(1),
          },
          ps = Object.values(P);
        return {
          ps,
          fwd: (img, tg) => {
            const x = AD.input(img, [1, 16, 16]),
              e1 = AD.relu(AD.conv3(x, P.w1, P.b1)),
              e2 = AD.relu(AD.conv3(AD.pool(e1), P.w2, P.b2)),
              bt = AD.relu(AD.conv3(AD.pool(e2), P.w3, P.b3)),
              u2 = AD.up(bt),
              d2 = AD.relu(AD.conv3(skip ? AD.cat(u2, e2) : u2, P.w4, P.b4)),
              u1 = AD.up(d2),
              d1 = AD.relu(AD.conv3(skip ? AD.cat(u1, e1) : u1, P.w5, P.b5)),
              L = AD.conv3(d1, P.w6, P.b6);
            return { L, loss: tg ? AD.bce(L, tg) : null };
          },
        };
      };
      const shape = (r, rn) => {
        const m = new Float64Array(256),
          k = 1 + Math.floor(r() * 2);
        for (let s = 0; s < k; s++) {
          const ty = Math.floor(r() * 3),
            cx = 3 + r() * 10,
            cy = 3 + r() * 10,
            rd = 2 + r() * 3.5;
          for (let y = 0; y < 16; y++)
            for (let x = 0; x < 16; x++) {
              const dx = x - cx,
                dy = y - cy;
              if (ty === 0) {
                if (Math.abs(Math.hypot(dx, dy) - rd) < 0.8) m[y * 16 + x] = 1;
              } else if (ty === 1) {
                if (Math.abs(Math.max(Math.abs(dx), Math.abs(dy)) - rd) < 0.7) m[y * 16 + x] = 1;
              } else if (
                Math.abs(dx * Math.cos(rd) + dy * Math.sin(rd)) < 0.7 &&
                Math.hypot(dx, dy) < rd + 2.5
              )
                m[y * 16 + x] = 1;
            }
        }
        return { m, x: m.map((v) => v + S.nz * rn()) };
      };
      const R = () => Math.random(),
        RN = randnFrom(Math.random);
      const iou = (N) => {
        let s = 0;
        for (const q of TE) {
          const o = N.fwd(q.x);
          AD.clear();
          let a = 0,
            b = 0;
          for (let i = 0; i < 256; i++) {
            const p = o.L.v[i] > 0 ? 1 : 0;
            if (p && q.m[i]) a++;
            if (p || q.m[i]) b++;
          }
          s += b ? a / b : 1;
        }
        return s / TE.length;
      };
      const outs = () => {
        OUT = [];
        const rows = TE.slice(0, 3)
          .map((q) => q.x)
          .concat([pad]);
        for (const x of rows) {
          const a = NA.fwd(x),
            b = NB.fwd(x);
          AD.clear();
          OUT.push([Float64Array.from(a.L.v, sig), Float64Array.from(b.L.v, sig)]);
        }
      };
      const build = () => {
        NA = mkNet(false, run * 7 + 1);
        NB = mkNet(true, run * 7 + 1);
        const r = rng(run * 5 + 9),
          rn = randnFrom(r);
        TE = [];
        for (let i = 0; i < 20; i++) TE.push(shape(r, rn));
        ts = 0;
        IA = [];
        IB = [];
        done = false;
        cu = 0;
        outs();
        msg = 'Two networks with random weights. Both answers are grey fog. Press Train both.';
      };
      const train = (k) => {
        for (let q = 0; q < k && !done; q++) {
          for (let b = 0; b < 5; b++) {
            const s = shape(R, RN);
            for (const N of [NA, NB]) {
              const o = N.fwd(s.x, s.m);
              AD.backward(o.loss);
            }
          }
          ts++;
          AD.adam(NA.ps, 0.006, ts, 5);
          AD.adam(NB.ps, 0.006, ts, 5);
          if (ts % 10 === 0 || ts === 1) {
            IA.push(iou(NA));
            IB.push(iou(NB));
          }
          if (ts >= MAXS) done = true;
        }
        outs();
        cu = done ? 4 : 1 + (ts % 3);
        const a = IA[IA.length - 1],
          b = IB[IB.length - 1];
        msg =
          'Step ' +
          ts +
          ' of ' +
          MAXS +
          ': both networks saw 5 fresh noisy pictures and adjusted their convolution kernels. On 20 test pictures, the answer overlaps the truth by ' +
          pct(a) +
          ' without skips and ' +
          pct(b) +
          ' with skips.' +
          (done
            ? ' Both have had their ' +
              MAXS +
              ' steps. ' +
              (b > a + 0.1
                ? 'The skips make the difference between a blur and a clean outline.'
                : 'They finish close: with this little noise or this few thin lines, the bottleneck was enough.')
            : '');
      };
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c),
            sz = Math.min((w - 30) / 4.3, (h - 34) / 4.25),
            gx = (w - sz * 4.3) / 2,
            cz = sz / 16,
            hd = ['noisy input', 'true shape', 'without skips', 'with skips'],
            rows = TE.slice(0, 3);
          g.font = MONO;
          hd.forEach((s, q) => {
            g.fillStyle = q === 2 ? t.ok : q === 3 ? t.accent : t.muted;
            g.textAlign = 'center';
            g.fillText(s, gx + q * sz * 1.1 + sz / 2, 12);
          });
          const pic = (A, x, y, cl) => {
            for (let i = 0; i < 256; i++) {
              g.fillStyle = mix(t.bg, t.ink, 0.05 + 0.95 * clamp(A[i], 0, 1));
              g.fillRect(x + (i % 16) * cz, y + Math.floor(i / 16) * cz, cz + 0.3, cz + 0.3);
            }
            g.strokeStyle = cl || t.line;
            g.lineWidth = cl ? 2 : 1;
            g.strokeRect(x - 0.5, y - 0.5, sz, sz);
          };
          for (let r = 0; r < 4; r++) {
            const y = 20 + r * sz * 1.06,
              mine = r === 3;
            pic(mine ? pad : rows[r].x, gx, y, mine ? t.gold : null);
            if (mine) {
              g.fillStyle = t.muted;
              g.textAlign = 'center';
              g.fillText('draw in the gold box', gx + sz * 1.1 + sz / 2, y + sz / 2);
              geo = { x: gx, y, cz };
            } else pic(rows[r].m, gx + sz * 1.1, y);
            pic(OUT[r][0], gx + 2 * sz * 1.1, y, t.ok);
            pic(OUT[r][1], gx + 3 * sz * 1.1, y, t.accent);
          }
        }
        {
          const { g, w, h } = fit(cr),
            h1 = h * 0.5,
            ph = (fr / 90) % 1,
            L = [
              [0.1, 0.16, 34, '16 × 16', '4 maps'],
              [0.27, 0.5, 24, '8 × 8', '6 maps'],
              [0.5, 0.82, 15, '4 × 4', '8 maps'],
              [0.73, 0.5, 24, '8 × 8', '6 maps'],
              [0.9, 0.16, 34, '16 × 16', '4 maps'],
            ].map((q) => [q[0] * w, q[1] * (h1 - 30) + 14, q[2], q[3], q[4]]);
          g.font = MONO;
          for (let q = 0; q < 4; q++) {
            g.strokeStyle = t.muted;
            g.lineWidth = 1.4;
            g.beginPath();
            g.moveTo(L[q][0], L[q][1]);
            g.lineTo(L[q + 1][0], L[q + 1][1]);
            g.stroke();
          }
          [
            [0, 4],
            [1, 3],
          ].forEach(([a, b]) => {
            g.strokeStyle = t.accent;
            g.lineWidth = 2.6;
            g.beginPath();
            g.moveTo(L[a][0] + L[a][2] / 2 + 4, L[a][1]);
            g.lineTo(L[b][0] - L[b][2] / 2 - 4, L[b][1]);
            g.stroke();
            g.fillStyle = t.accent;
            g.textAlign = 'center';
            g.fillText('skip', (L[a][0] + L[b][0]) / 2, L[a][1] - 6);
            if (pb && pb.on) {
              g.beginPath();
              g.arc(L[a][0] + (L[b][0] - L[a][0]) * ph, L[a][1], 4.5, 0, 7);
              g.fill();
            }
          });
          L.forEach((q, i) => {
            g.fillStyle = i === 2 ? t.gold : t.bg;
            g.strokeStyle = t.ink;
            g.lineWidth = 1.5;
            g.beginPath();
            g.rect(q[0] - q[2] / 2, q[1] - q[2] / 2, q[2], q[2]);
            g.fill();
            g.stroke();
            g.fillStyle = t.muted;
            g.textAlign = 'center';
            g.fillText(q[3], q[0], q[1] + q[2] / 2 + 13);
            g.fillText(q[4], q[0], q[1] + q[2] / 2 + 25);
          });
          if (pb && pb.on) {
            const k = Math.min(3, Math.floor(ph * 4)),
              uu = ph * 4 - k;
            g.fillStyle = t.gold;
            g.beginPath();
            g.arc(
              L[k][0] + (L[k + 1][0] - L[k][0]) * uu,
              L[k][1] + (L[k + 1][1] - L[k][1]) * uu,
              5,
              0,
              7,
            );
            g.fill();
          }
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('down: convolve, shrink', 8, h1 - 6);
          g.textAlign = 'right';
          g.fillText('up: enlarge, convolve', w - 8, h1 - 6);
          const U = uLine(g, t, 0, h1, w, h - h1, IB, {
            title: 'overlap with the truth on test pictures: with skips (green), without (red)',
            n: MAXS / 10 + 1,
            ym: 1,
            col: t.accent,
          });
          g.strokeStyle = t.ok;
          g.lineWidth = 2.2;
          g.beginPath();
          IA.forEach((v, i) => (i ? g.lineTo(U.X(i), U.Y(v)) : g.moveTo(U.X(i), U.Y(v))));
          g.stroke();
        }
        live.textContent = msg;
        uCur(root, 'un', cu);
        const cnt = (N) => N.ps.reduce((a, p) => a + p.v.length, 0);
        setR(root, 'step', ts + ' of ' + MAXS);
        setR(root, 'i0', IA.length ? pct(IA[IA.length - 1]) : '–');
        setR(root, 'i1', IB.length ? pct(IB[IB.length - 1]) : '–');
        setR(root, 'np', cnt(NA) + ' / ' + cnt(NB));
        setR(root, 'st', done ? 'Finished' : ts ? 'Training' : 'Not started');
      };
      build();
      if (RM) train(MAXS);
      const anim = Anim(() => {
        fr++;
        if (ts < 10 && fr % 6) {
          draw();
          return;
        }
        train(1);
        draw();
        chk();
      });
      pb = playBtn($('#un-play', root), anim);
      chk = runCtl(pb, root, 'un', {
        done: () => done,
        restart: () => {
          run++;
          build();
        },
        count: () => ts,
        what: 'steps',
      });
      $('#un-step', root).onclick = () => {
        pb.stop();
        if (done) {
          run++;
          build();
        } else train(10);
        draw();
      };
      $('#un-reset', root).onclick = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      bind(root, 'un-nz', (v) => {
        S.nz = v;
        pb.stop();
        run++;
        build();
        draw();
      });
      $('#un-clear', root).onclick = () => {
        pad.fill(0);
        outs();
        draw();
      };
      const pt = (e) => {
        if (!geo) return false;
        const r = c.getBoundingClientRect(),
          x = Math.floor((e.clientX - r.left - geo.x) / geo.cz),
          y = Math.floor((e.clientY - r.top - geo.y) / geo.cz);
        if (x < 0 || y < 0 || x >= 16 || y >= 16) return false;
        pad[y * 16 + x] = 1;
        return true;
      };
      c.style.touchAction = 'none';
      c.addEventListener('pointerdown', (e) => {
        if (pt(e) === false) return;
        paint = true;
        c.setPointerCapture(e.pointerId);
        outs();
        if (!pb.on) draw();
      });
      c.addEventListener('pointermove', (e) => {
        if (!paint) return;
        pt(e);
        outs();
        if (!pb.on) draw();
      });
      ['pointerup', 'pointercancel'].forEach((n2) =>
        c.addEventListener(n2, () => {
          if (paint) {
            paint = false;
            msg =
              'Your drawing went through both networks: compare the two boxes to its right. (Your drawing has no noise, so it is easier than the training pictures.)';
            draw();
          }
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
    id: 'transfer',
    tab: 'Transfer learning',
    title: 'Transfer learning: do not start from zero',
    lede: 'A child who can already read does not learn a new alphabet game from nothing. Networks can reuse knowledge too. First a network studies hundreds of labelled letter pictures. Then a new question arrives, “is this letter a vowel?”, with only a handful of examples. One learner starts from scratch. The other keeps what the first network learned to see and only learns the new answer. Watch which one copes.',
    stage: `<div class="stage"><figure class="fig"><figcaption>What the first layer looks for: one small picture per neuron (gold: wants ink here, red: wants none). Top: learned from hundreds of pictures. Bottom: learned from scratch on the few new examples</figcaption><canvas class="short" id="tl-c" role="img" aria-label="First-layer weights of the pretrained and the from-scratch networks"></canvas></figure>
 <figure class="fig"><figcaption>Top: all the examples the new task provides. Bottom: accuracy on new pictures while both learners train (average of four tries)</figcaption><canvas class="short" id="tl-r" role="img" aria-label="Few-shot examples and test accuracy of both learners"></canvas></figure></div>
 ${uHow('tl', ['Stage 1, pretraining: a network learns to name six letters from 300 noisy pictures. Its first layer learns to see strokes.', 'A new task arrives: say whether a letter is a vowel. There are only a few examples.', 'Learner A, from scratch: a fresh network with random weights is trained on those few examples.', 'Learner B, transfer: copy the pretrained first layer, freeze it, and train only a new last layer on the same few examples.', 'Test both on pictures neither has seen.'])}
 <div class="controls">${uBtns('tl', 'Run', 'Ten epochs', 'epochs')}
 ${grp('The new task', slider('tl-k', 'Examples per letter for the new task', 1, 12, 1, 1) + slider('tl-fl', 'Noise: chance that a pixel is flipped', 0.05, 0.3, 0.01, 0.2) + `<div class="row"><button class="btn" id="tl-new" type="button">New random sample</button></div>`)}</div>
 <dl class="read">${rd('ph', 'Stage')}${rd('pa', 'Pretrained network: letters named correctly')}${rd('sc', 'From scratch: accuracy on new pictures', 'hl2')}${rd('tf', 'Transfer: accuracy on new pictures', 'hl3')}${rd('n', 'Examples the new task provides')}${rd('w', 'Weights trained: scratch / transfer')}</dl>`,
    tryit:
      'Run it with 1 example per letter: six pictures in all. Compare the two rows of small pictures on the left: one set is clean strokes, the other is noise that happened to fit six examples. Then raise the examples to 12 per letter. The gap closes. When is starting from someone else’s network worth it, and when does it stop mattering?',
    math: String.raw`<h4>Intuition</h4><p>A chef who has cooked Italian food for ten years is asked to make a Thai dish from three recipes. A beginner given the same three recipes will struggle, because three recipes cannot teach knife skills, timing or how to taste. The chef already has all of that and only needs to learn what is new. Most of what a network learns on one task with a lot of data, how to see edges and strokes, how to read grammar, is useful for other tasks with little data.</p>
 <h4>The recipe</h4><p>Split a network into a <b>feature extractor</b> \(\phi\) (the early layers) and a <b>head</b> \(g\) (the last layer).</p><ol><li><b>Pretrain</b> \(g_A(\phi(\mathbf x))\) on task A, which has plenty of data.</li><li>For task B, keep \(\phi\), throw away \(g_A\) and attach a fresh head \(g_B\).</li><li>Train on task B. Either <b>freeze</b> \(\phi\) and train only \(g_B\) (as on this page), or <b>fine-tune</b>: train everything, with a small learning rate so that \(\phi\) changes only a little.</li></ol>
 <h4>Why it helps</h4><p>It is the bias and variance trade-off again. Training from scratch on six examples means fitting every weight, here 889 of them, to six pictures: enormous variance, and the first-layer pictures on the lower left show the result. Freezing \(\phi\) leaves only the head's 25 weights to fit, and they are fitted on features that already separate the letters cleanly. Far fewer free numbers and better inputs mean much lower variance. The price is bias: if task B needs features that task A never taught, a frozen \(\phi\) cannot supply them.</p>
 <h4>When it works, and when not</h4><ul><li><b>Works</b> when the two tasks share low-level structure: the same kind of pictures, the same language, the same sensor.</li><li><b>Helps most</b> when task B has little data. With plenty, training from scratch catches up, as the slider shows.</li><li><b>Can hurt</b> (negative transfer) when the tasks are unrelated, because the frozen features discard what the new task needs.</li><li><b>Freeze or fine-tune?</b> Freeze with very little data, fine-tune with more, and unfreeze from the last layer backward.</li></ul>
 <h4>Why this matters now</h4><p>Almost nobody trains a large model from zero. Vision models are pretrained on millions of images and language models on a large share of the public internet, then adapted with a few thousand examples, a few dozen, or just a written instruction. Those pretrained networks are called <b>foundation models</b>, and fine-tuning methods such as LoRA appear in the Language Models module.</p>
 <h4>About this demonstration</h4><p>Both tasks use the same six letters, so the new task is a new question about familiar pictures. That is the easiest kind of transfer. The accuracy curves are averaged over four independent draws of the few examples, because with so few examples a single try is mostly luck.</p>`,
    q: 'With 12 examples per letter the two learners end up level. Does that mean transfer learning was pointless there?',
    a: 'Not quite. It reached that accuracy while training only 25 weights instead of 889, so it was cheaper and faster, and it got there with far fewer epochs. But the accuracy advantage is gone: once the new task has enough data of its own, a network trained from scratch can learn good features for itself. Transfer pays most when data for the new task is scarce.',
    init(root) {
      const c = $('#tl-c', root),
        cr = $('#tl-r', root),
        live = $('#tl-live', root),
        S = { k: 1, fl: 0.2, seed: 1 },
        Hn = 24,
        PE = 200,
        FE = 260,
        RP = 4,
        VW = [1, 1, 0, 1, 0, 0];
      let A,
        Bt = [],
        Bte,
        pre,
        SC = [],
        TF = [],
        ep = 0,
        fe = 0,
        pa = 0,
        AS = [],
        AT = [],
        msg = '',
        cu = 0,
        fr = 0,
        pb,
        chk = () => false,
        run = 1,
        done = false;
      const bin = (d) => ({
          X: d.X.map((x) => Float64Array.from(x, (v) => clamp(Math.round(v), 0, 1))),
          lab: d.lab,
        }),
        oh = (l) => Float64Array.from({ length: 6 }, (_, q) => (q === l ? 1 : 0));
      const gen = () => {
        const a = bin(letterData(S.seed * 3 + 1, S.fl)),
          b = bin(letterData(S.seed * 3 + 2, S.fl));
        A = { X: a.X.concat(b.X), Y: a.lab.concat(b.lab).map(oh), lab: a.lab.concat(b.lab) };
        Bte = bin(letterData(S.seed * 3 + 900, S.fl));
        Bt = [];
        for (let r = 0; r < RP; r++) {
          const d = bin(letterData(S.seed * 7 + 50 + r, S.fl)),
            X = [],
            Y = [];
          for (let l = 0; l < 6; l++)
            for (let i = 0; i < S.k; i++) {
              X.push(d.X[l * 25 + i]);
              Y.push(Float64Array.of(VW[l]));
            }
          Bt.push({ X, Y });
        }
      };
      const build = () => {
        pre = NNet([35, Hn, 6], { act: 'relu', out: 'sig', seed: run * 5 + S.seed });
        SC = [];
        TF = [];
        ep = 0;
        fe = 0;
        pa = 0;
        AS = [];
        AT = [];
        done = false;
        cu = 0;
        msg =
          'Stage 1: a network with random weights is about to study ' +
          A.X.length +
          ' labelled letter pictures. Press Run.';
      };
      const acc6 = () => {
          let ok = 0;
          A.X.forEach((x, i) => {
            const o = pre.p2(x);
            if (o.indexOf(Math.max(...o)) === A.lab[i]) ok++;
          });
          return ok / A.X.length;
        },
        accB = (N) => {
          let ok = 0;
          Bte.X.forEach((x, i) => {
            if ((N.p2(x)[0] >= 0.5 ? 1 : 0) === VW[Bte.lab[i]]) ok++;
          });
          return ok / Bte.X.length;
        };
      const startB = () => {
        for (let r = 0; r < RP; r++) {
          const s = NNet([35, Hn, 1], { act: 'relu', out: 'sig', seed: run * 3 + r * 7 + 2 }),
            tn = NNet([35, Hn, 1], { act: 'relu', out: 'sig', seed: run * 3 + r * 7 + 2 });
          for (let j = 0; j < Hn; j++) {
            tn.W[0][j].set(pre.W[0][j]);
            tn.B[0][j] = pre.B[0][j];
          }
          SC.push(s);
          TF.push(tn);
        }
      };
      const train = (k) => {
        for (let i = 0; i < k && !done; i++) {
          if (ep < PE) {
            pre.train(A.X, A.Y, 0.01);
            ep++;
            if (ep % 10 === 0 || ep >= PE) pa = acc6();
            cu = 0;
            msg =
              'Stage 1, pretraining: epoch ' +
              ep +
              ' of ' +
              PE +
              '. It names ' +
              pct(pa) +
              ' of the ' +
              A.X.length +
              ' letter pictures correctly. Watch the top row of small pictures turn from noise into strokes.';
            if (ep >= PE) {
              startB();
              cu = 1;
              msg =
                'Pretraining is done: ' +
                pct(pa) +
                ' correct. Now the new task: vowel or not, from ' +
                6 * S.k +
                ' example' +
                (6 * S.k === 1 ? '' : 's') +
                '. Learner A starts from random weights. Learner B starts with the first layer you see at the top, frozen.';
            }
          } else {
            let a = 0,
              b = 0;
            for (let r = 0; r < RP; r++) {
              SC[r].train(Bt[r].X, Bt[r].Y, 0.01);
              TF[r].train(Bt[r].X, Bt[r].Y, 0.01, { freeze: 1 });
              a += accB(SC[r]) / RP;
              b += accB(TF[r]) / RP;
            }
            AS.push(a);
            AT.push(b);
            fe++;
            cu = fe % 2 ? 2 : 3;
            if (fe >= FE) done = true;
            msg =
              'Stage 2, the new task: epoch ' +
              fe +
              ' of ' +
              FE +
              '. On ' +
              Bte.X.length +
              ' new pictures, from scratch gets ' +
              pct(a) +
              ' right and transfer gets ' +
              pct(b) +
              '.' +
              (done
                ? ' Both have had their ' +
                  FE +
                  ' epochs. ' +
                  (b > a + 0.02
                    ? 'Transfer wins by ' +
                      (100 * (b - a)).toFixed(0) +
                      ' points: with so few examples, borrowed features beat features learned from almost nothing.'
                    : a > b + 0.02
                      ? 'From scratch wins here: there was enough data for it to learn its own features, and the frozen ones were a limit.'
                      : 'They finish level: with this many examples, starting from scratch catches up.')
                : '');
            if (done) cu = 4;
          }
        }
      };
      const draw = () => {
        const t = T();
        {
          const { g, w, h } = fit(c),
            cz = Math.min((w - 20) / 12 / 6.2, (h / 2 - 34) / 2 / 8),
            gw = 5 * cz,
            gh = 7 * cz,
            rowsOf = (N, y0, lab, al) => {
              g.font = MONO;
              g.fillStyle = t.muted;
              g.textAlign = 'left';
              g.fillText(lab, 8, y0 - 6);
              if (!N) {
                g.fillText('(not started yet)', 8, y0 + 20);
                return;
              }
              for (let j = 0; j < Hn; j++) {
                const x = 8 + (j % 12) * (gw + cz * 1.2),
                  y = y0 + Math.floor(j / 12) * (gh + cz * 1.5),
                  wv = N.W[0][j];
                let m = 1e-9;
                for (const v of wv) m = Math.max(m, Math.abs(v));
                for (let q = 0; q < 35; q++) {
                  const v = wv[q] / m;
                  g.fillStyle = mix(t.bg, v >= 0 ? t.gold : t.ok, Math.abs(v));
                  g.fillRect(x + (q % 5) * cz, y + Math.floor(q / 5) * cz, cz - 0.4, cz - 0.4);
                }
              }
            };
          rowsOf(
            pre,
            22,
            'first layer after pretraining on ' +
              A.X.length +
              ' pictures (copied to learner B, frozen)',
          );
          rowsOf(
            SC[0],
            h / 2 + 26,
            'first layer of learner A, trained from scratch on ' +
              6 * S.k +
              ' example' +
              (6 * S.k === 1 ? '' : 's'),
          );
        }
        {
          const { g, w, h } = fit(cr),
            B0 = Bt[0],
            sh = Math.min(B0.X.length, 18),
            cz = Math.min((w - 20) / sh / 6.2, (h * 0.2) / 7.6);
          g.font = MONO;
          g.fillStyle = t.muted;
          g.textAlign = 'left';
          g.fillText('the new task’s examples (first try): gold frame = vowel', 8, 13);
          for (let i = 0; i < sh; i++) {
            const x = 10 + i * cz * 6.2,
              y = 22;
            for (let q = 0; q < 35; q++) {
              g.fillStyle = mix(t.bg, t.ink, 0.06 + 0.94 * B0.X[i][q]);
              g.fillRect(x + (q % 5) * cz, y + Math.floor(q / 5) * cz, cz - 0.4, cz - 0.4);
            }
            g.strokeStyle = B0.Y[i][0] ? t.gold : t.muted;
            g.lineWidth = B0.Y[i][0] ? 2.4 : 1;
            g.strokeRect(x - 1.5, y - 1.5, 5 * cz + 2, 7 * cz + 2);
          }
          const y0 = h * 0.36,
            U = uLine(g, t, 0, y0, w, h - y0, AT, {
              title: 'accuracy on new pictures: transfer (green), from scratch (red)',
              n: FE,
              y1: 0.4,
              ym: 1,
              col: t.accent,
            });
          g.strokeStyle = t.ok;
          g.lineWidth = 2.2;
          g.beginPath();
          AS.forEach((v, i) => (i ? g.lineTo(U.X(i), U.Y(v)) : g.moveTo(U.X(i), U.Y(v))));
          g.stroke();
          g.setLineDash([4, 4]);
          g.strokeStyle = t.muted;
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(U.X(0), U.Y(0.5));
          g.lineTo(U.X(FE - 1), U.Y(0.5));
          g.stroke();
          g.setLineDash([]);
          g.fillStyle = t.muted;
          g.textAlign = 'right';
          g.fillText('dashed: a coin toss', w - 14, U.Y(0.5) - 4);
        }
        live.textContent = msg;
        uCur(root, 'tl', cu);
        setR(root, 'ph', done ? 'Finished' : ep < PE ? 'Pretraining' : 'New task');
        setR(root, 'pa', ep ? pct(pa) : '–');
        setR(root, 'sc', fe ? pct(AS[fe - 1]) : '–');
        setR(root, 'tf', fe ? pct(AT[fe - 1]) : '–');
        setR(root, 'n', String(6 * S.k));
        setR(root, 'w', 35 * Hn + Hn + Hn + 1 + ' / ' + (Hn + 1));
      };
      gen();
      build();
      if (RM) train(PE + FE);
      const redo = () => {
        pb.stop();
        run++;
        build();
        draw();
      };
      const anim = Anim(() => {
        fr++;
        if (ep < PE) {
          if (ep < 30 && fr % 3) return;
          train(ep < 30 ? 1 : 2);
          draw();
          return;
        }
        if (fr % (fe < 6 ? 16 : 2)) return;
        train(1);
        draw();
        chk();
      });
      pb = playBtn($('#tl-play', root), anim);
      chk = runCtl(pb, root, 'tl', {
        done: () => done,
        restart: () => {
          run++;
          build();
        },
        count: () => ep + fe,
        what: 'epochs',
      });
      $('#tl-step', root).onclick = () => {
        pb.stop();
        if (done) {
          run++;
          build();
        } else train(10);
        draw();
      };
      $('#tl-reset', root).onclick = redo;
      bind(
        root,
        'tl-k',
        (v) => {
          S.k = v;
          gen();
          redo();
        },
        (v) => String(v),
      );
      bind(root, 'tl-fl', (v) => {
        S.fl = v;
        gen();
        redo();
      });
      $('#tl-new', root).onclick = () => {
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
];
{
  const o = [
    'convolution',
    'residual',
    'unet',
    'autoencoder',
    'embeddings',
    'rnn',
    'lstm',
    'attention',
    'transformer',
    'gnn',
    'transfer',
  ];
  DLL.sort((a, b) => o.indexOf(a.id) - o.indexOf(b.id));
}
