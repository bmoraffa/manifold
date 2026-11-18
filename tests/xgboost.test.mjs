import test from 'node:test';
import assert from 'node:assert/strict';
import { loadSite } from './helpers/load-site.mjs';
const site = loadSite();
function build({ depth = 2, gam = 0, constant = false, rows = null } = {}) {
  return site.evaluate(`(() => {
    const X = Array.from({length:40}, (_,i) => ${constant ? '[0,0]' : '[i%2,(i>>1)%2]'});
    const g = X.map(([a,b]) => .5 - (a !== b ? 1 : 0));
    const order = [0,1].map(f => X.map((_,i)=>i).sort((a,b)=>X[a][f]-X[b][f]));
    return xgbBuildTree(X, order, g, X.map(()=>.25), ${JSON.stringify(rows)} || X.map((_,i)=>i), {depth:${depth},lam:0,gam:${gam}});
  })()`);
}
test('zero immediate gain can lead to useful XOR descendants', () => {
  const { root } = build();
  for (const [a, b, y] of [
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
  ]) {
    let n = root;
    while (n.sp) n = [a, b][n.sp.f] <= n.sp.thr ? n.sp.L : n.sp.R;
    assert.equal(n.w >= 0 ? 1 : 0, y);
  }
  assert.ok(!build({ depth: 1 }).root.sp?.L.sp);
  assert.equal(build({ constant: true }).root.sp, undefined);
  assert.ok(Number.isFinite(build({ constant: true }).root.w));
  assert.equal(build({ rows: [0, 1, 2, 3] }).root.n, 4);
  assert.equal(build({ gam: 1000 }).root.sp, null);
});

test('adjacent or very large feature values never create empty children', () => {
  for (const [lo, hi] of [
    [1.0000000000000002, 1.0000000000000004],
    [1e308, 1.5e308],
  ]) {
    const result = site.evaluate(`(() => {
      const X = [[${lo}],[${lo}],[${hi}],[${hi}]];
      return xgbBuildTree(X, [[0,1,2,3]], [-1,1,-1,1], [1,1,1,1], [0,1,2,3], {depth:1,lam:0,gam:0});
    })()`);
    assert.equal(result.root.sp.L.n, 2);
    assert.equal(result.root.sp.R.n, 2);
    assert.ok(Number.isFinite(result.root.sp.R.w));
  }
});
