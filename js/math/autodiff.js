/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */

const AD = (() => {
  let tape = [];
  const createTensor = (v, sh) => ({ v, g: new Float64Array(v.length), sh }),
    op = (output, backpropagate) => {
      tape.push(backpropagate);
      return output;
    };
  const param = (sh, sd, rn) => {
    const n = sh.reduce((a, b) => a * b, 1),
      p = createTensor(
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
      o = createTensor(new Float64Array(m * n), [m, n]),
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
      o = createTensor(new Float64Array(m * n), [m, n]),
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
    const o = createTensor(
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
    const o = createTensor(
      A.v.map((x) => (x > 0 ? x : 0)),
      A.sh,
    );
    return op(o, () => {
      for (let i = 0; i < o.g.length; i++) if (A.v[i] > 0) A.g[i] += o.g[i];
    });
  };
  const gather = (E, idx) => {
    const d = E.sh[1],
      o = createTensor(new Float64Array(idx.length * d), [idx.length, d]);
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
      o = createTensor(new Float64Array(T * T), [T, T]);
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
      o = createTensor(new Float64Array(1), [1]),
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
      o = createTensor(new Float64Array(O * H * Wd), [O, H, Wd]),
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
      o = createTensor(new Float64Array(C * h * w2), [C, h, w2]),
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
      o = createTensor(new Float64Array(C * H * Wd), [C, H, Wd]);
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
    const o = createTensor(new Float64Array(A.v.length + B.v.length), [
      A.sh[0] + B.sh[0],
      A.sh[1],
      A.sh[2],
    ]);
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
      o = createTensor(new Float64Array(1), [1]);
    for (let i = 0; i < n; i++) {
      const z = L.v[i];
      o.v[0] += (Math.max(z, 0) - z * tg[i] + Math.log(1 + Math.exp(-Math.abs(z)))) / n;
    }
    return op(o, () => {
      for (let i = 0; i < n; i++) L.g[i] += (o.g[0] * (1 / (1 + Math.exp(-L.v[i])) - tg[i])) / n;
    });
  };
  const input = (v, sh) => createTensor(Float64Array.from(v), sh),
    backward = (l) => {
      l.g[0] = 1;
      for (let i = tape.length - 1; i >= 0; i--) tape[i]();
      tape = [];
    },
    clear = () => {
      tape = [];
    };
  const adam = (ps, lr, t, gradientDivisor) => {
    const c1 = 1 - Math.pow(0.9, t),
      c2 = 1 - Math.pow(0.999, t);
    for (const p of ps) {
      for (let i = 0; i < p.v.length; i++) {
        const g = p.g[i] / gradientDivisor;
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

export { AD };
