const EDGE = 64;

export function lessonMotion(from, to) {
  const sameTopic = from?.topic && from.topic === to.topic && !to.topic.about;
  return sameTopic
    ? { shift: to.index > from.index ? 30 : -30, duration: 280 }
    : { shift: 0, duration: 220 };
}

export function createTransitions({ window, document, content, nav }) {
  let layer;
  let animations = [];
  const listeners = new AbortController();
  const signal = listeners.signal;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  nav.style.setProperty('--lesson-fade-depth', `${EDGE}px`);
  const scrollEdge = () => nav.toggleAttribute('data-scrolled', window.scrollY > 1);
  const measure = () => {
    const height = nav.getBoundingClientRect().height;
    document.documentElement.style.setProperty(
      '--lesson-scroll-offset',
      height ? `${height + EDGE + 12}px` : '0px',
    );
  };
  const resize =
    typeof window.ResizeObserver === 'function' ? new window.ResizeObserver(measure) : null;
  resize?.observe(nav);
  window.addEventListener('scroll', scrollEdge, { passive: true, signal });
  window.addEventListener('resize', measure, { signal });
  document.addEventListener(
    'focusin',
    (event) => {
      if (nav.hidden || !content.contains(event.target)) return;
      const edge = nav.getBoundingClientRect().bottom + (window.scrollY > 1 ? EDGE : 0);
      if (event.target.getBoundingClientRect().top < edge)
        event.target.scrollIntoView({ block: 'start' });
    },
    { signal },
  );
  motion.addEventListener('change', cancel, { signal });
  scrollEdge();

  function cancel() {
    for (const animation of animations) animation.cancel();
    animations = [];
    layer?.remove();
    layer = undefined;
  }

  function snapshot(element) {
    const rect = element.getBoundingClientRect();
    const copy = element.cloneNode(true);
    const canvases = element.querySelectorAll('canvas');
    copy.querySelectorAll('canvas').forEach((canvas, i) => {
      const original = canvases[i];
      if (original.width && original.height) canvas.getContext('2d')?.drawImage(original, 0, 0);
    });
    copy.querySelectorAll('input[type="file"]').forEach((input) => {
      input.disabled = true;
    });
    copy.querySelectorAll('[id], [data-go]').forEach((node) => {
      node.removeAttribute('id');
      node.removeAttribute('data-go');
    });
    copy.removeAttribute('id');
    copy.classList.add('navigation-snapshot');
    copy.inert = true;
    copy.setAttribute('aria-hidden', 'true');
    Object.assign(copy.style, {
      top: `${rect.top}px`,
      left: `${rect.left}px`,
      width: `${rect.width}px`,
      maxWidth: 'none',
    });
    if (element.id === 'home') {
      copy.classList.add('navigation-snapshot-home');
      copy.style.height = `${rect.height}px`;
    }
    return copy;
  }

  async function run({ from, to, commit, isCurrent }) {
    cancel();
    if (!isCurrent()) return;
    const outgoing = from?.element;
    const animate =
      outgoing &&
      from.id !== to.id &&
      !document.hidden &&
      !motion.matches &&
      typeof to.element.animate === 'function';
    if (!animate) {
      commit();
      measure();
      scrollEdge();
      return;
    }
    let copy;
    try {
      copy = snapshot(outgoing);
      document.body.append(copy);
      layer = copy;
    } catch {
      copy?.remove();
      copy = undefined;
    }
    try {
      const ready = commit();
      measure();
      scrollEdge();
      const { shift, duration } = lessonMotion(from, to);
      const easing = 'cubic-bezier(.22,.61,.36,1)';
      const incoming = to.element.animate(
        [
          { opacity: 0, transform: `translateX(${shift}px)` },
          { opacity: 1, transform: 'none' },
        ],
        { duration, easing, fill: 'backwards' },
      );
      animations.push(incoming);
      if (ready) {
        incoming.pause();
        await Promise.race([ready, incoming.finished.catch(() => {})]);
        if (!isCurrent() || !animations.includes(incoming)) return;
        incoming.play();
      }
      if (copy)
        animations.push(
          copy.animate(
            [
              { opacity: 1, transform: 'none' },
              { opacity: 0, transform: `translateX(${shift * -0.8}px)` },
            ],
            { duration: 180, easing, fill: 'forwards' },
          ),
        );
      await Promise.allSettled(animations.map((animation) => animation.finished));
    } finally {
      if (isCurrent()) cancel();
      else copy?.remove();
    }
  }

  return {
    run,
    cancel,
    destroy() {
      cancel();
      listeners.abort();
      resize?.disconnect();
    },
  };
}
