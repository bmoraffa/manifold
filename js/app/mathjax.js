export function createMathTypesetter({ window }) {
  const completed = new WeakSet();
  const pending = new WeakMap();
  let disposed = false;
  let queue = Promise.resolve();

  function typeset(root) {
    if (disposed || completed.has(root) || !root.isConnected) return null;
    if (pending.has(root)) return pending.get(root);
    const math = window.MathJax;
    const prose = [...root.querySelectorAll('.prose')];
    if (!math?.typesetPromise || !prose.length) return null;
    // MathJax 3 reuses one document for every typesetting call.
    const work = queue
      .then(async () => {
        if (disposed || !root.isConnected) return;
        await math.typesetPromise(prose);
        completed.add(root);
      })
      .catch(() => {})
      .finally(() => pending.delete(root));
    pending.set(root, work);
    queue = work;
    return work;
  }

  function prepare(root) {
    const work = typeset(root);
    if (!work) return null;
    let timer;
    return Promise.race([
      work,
      new Promise((resolve) => {
        timer = setTimeout(resolve, 300);
      }),
    ]).finally(() => clearTimeout(timer));
  }

  function retry(root, isCurrent, redraw, attempt = 0) {
    if (disposed || !isCurrent() || completed.has(root) || !root.querySelector('.prose')) return;
    const work = typeset(root);
    if (work)
      work.then(() => {
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
