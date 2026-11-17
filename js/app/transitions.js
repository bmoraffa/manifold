const NAV_FADE_DEPTH_PX = 64;

export function lessonMotion(from, to) {
  const sameTopic = from?.topic && from.topic === to.topic && !to.topic.isAbout;
  return sameTopic
    ? { offsetXPx: to.index > from.index ? 30 : -30, duration: 280 }
    : { offsetXPx: 0, duration: 220 };
}

export function createTransitions({ window, document, content, nav }) {
  let snapshotLayer;
  let animations = [];
  const listeners = new AbortController();
  const signal = listeners.signal;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  nav.style.setProperty('--lesson-fade-depth', `${NAV_FADE_DEPTH_PX}px`);
  const scrollEdge = () => nav.toggleAttribute('data-scrolled', window.scrollY > 1);
  const measure = () => {
    const height = nav.getBoundingClientRect().height;
    document.documentElement.style.setProperty(
      '--lesson-scroll-offset',
      height ? `${height + NAV_FADE_DEPTH_PX + 12}px` : '0px',
    );
  };
  const navResizeObserver =
    typeof window.ResizeObserver === 'function' ? new window.ResizeObserver(measure) : null;
  navResizeObserver?.observe(nav);
  window.addEventListener('scroll', scrollEdge, { passive: true, signal });
  window.addEventListener('resize', measure, { signal });
  document.addEventListener(
    'focusin',
    (event) => {
      if (nav.hidden || !content.contains(event.target)) return;
      const edge =
        nav.getBoundingClientRect().bottom + (window.scrollY > 1 ? NAV_FADE_DEPTH_PX : 0);
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
    snapshotLayer?.remove();
    snapshotLayer = undefined;
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
    const outgoingElement = from?.element;
    const shouldAnimate =
      outgoingElement &&
      from.id !== to.id &&
      !document.hidden &&
      !motion.matches &&
      typeof to.element.animate === 'function';
    if (!shouldAnimate) {
      commit();
      measure();
      scrollEdge();
      return;
    }
    let copy;
    try {
      copy = snapshot(outgoingElement);
      document.body.append(copy);
      snapshotLayer = copy;
    } catch {
      copy?.remove();
      copy = undefined;
    }
    try {
      const contentReady = commit();
      measure();
      scrollEdge();
      const { offsetXPx, duration } = lessonMotion(from, to);
      const easing = 'cubic-bezier(.22,.61,.36,1)';
      const incomingAnimation = to.element.animate(
        [
          { opacity: 0, transform: `translateX(${offsetXPx}px)` },
          { opacity: 1, transform: 'none' },
        ],
        { duration, easing, fill: 'backwards' },
      );
      animations.push(incomingAnimation);
      if (contentReady) {
        incomingAnimation.pause();
        await Promise.race([contentReady, incomingAnimation.finished.catch(() => {})]);
        if (!isCurrent() || !animations.includes(incomingAnimation)) return;
        incomingAnimation.play();
      }
      if (copy)
        animations.push(
          copy.animate(
            [
              { opacity: 1, transform: 'none' },
              { opacity: 0, transform: `translateX(${offsetXPx * -0.8}px)` },
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
      navResizeObserver?.disconnect();
    },
  };
}
