export function createMathTypesetter({ window }) {
  const typesetRoots = new WeakSet();
  const typesettingByRoot = new WeakMap();
  let disposed = false;
  let typesettingQueue = Promise.resolve();

  function typeset(root) {
    if (disposed || typesetRoots.has(root) || !root.isConnected) return null;
    if (typesettingByRoot.has(root)) return typesettingByRoot.get(root);
    const math = window.MathJax;
    const prose = [...root.querySelectorAll('.prose')];
    if (!math?.typesetPromise || !prose.length) return null;
    // MathJax 3 reuses one document for every typesetting call.
    const typesettingPromise = typesettingQueue
      .then(async () => {
        if (disposed || !root.isConnected) return;
        await math.typesetPromise(prose);
        typesetRoots.add(root);
      })
      .catch(() => {})
      .finally(() => typesettingByRoot.delete(root));
    typesettingByRoot.set(root, typesettingPromise);
    typesettingQueue = typesettingPromise;
    return typesettingPromise;
  }

  function prepare(root) {
    const typesettingPromise = typeset(root);
    if (!typesettingPromise) return null;
    let timer;
    return Promise.race([
      typesettingPromise,
      new Promise((resolve) => {
        timer = setTimeout(resolve, 300);
      }),
    ]).finally(() => clearTimeout(timer));
  }

  function retry(root, isCurrent, redraw, attempt = 0) {
    if (disposed || !isCurrent() || typesetRoots.has(root) || !root.querySelector('.prose')) return;
    const typesettingPromise = typeset(root);
    if (typesettingPromise)
      typesettingPromise.then(() => {
        if (!disposed && isCurrent()) redraw();
      });
    else if (attempt < 60) setTimeout(() => retry(root, isCurrent, redraw, attempt + 1), 400);
  }
  return {
    prepare,
    retry,
    destroy() {
      disposed = true;
    },
  };
}
