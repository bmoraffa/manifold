import { RM, SITE, TC, observeReducedMotion } from '../core/shared.js';
import { createHero } from './hero.js';
import { renderLesson } from './lesson-view.js';
import { createMathTypesetter } from './mathjax.js';

export function mountApp({
  document,
  window,
  catalog,
  loader,
  hero = createHero(),
  transition = { run: ({ commit }) => commit(), cancel() {} },
}) {
  const byId = (id) => document.getElementById(id);
  const home = byId('home'),
    lesson = byId('lesson'),
    modbar = byId('modbar');
  const tabs = byId('ltabs'),
    menu = byId('menu-b'),
    side = byId('side');
  const routes = new Map(),
    roots = new Map();
  const homeRoute = { id: 'home', element: home };
  const math = createMathTypesetter({ window });
  const listeners = new AbortController();
  const signal = listeners.signal;
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  const unwatch = observeReducedMotion(media);
  const status = document.createElement('div');
  status.className = 'navigation-status';
  status.hidden = true;
  status.setAttribute('role', 'status');
  lesson.before(status);

  let active = null;
  let desired = 0;
  let loadingTimer;
  function button(label, id, className) {
    const node = document.createElement('button');
    node.type = 'button';
    node.textContent = label;
    node.setAttribute('data-go', id);
    if (className) node.className = className;
    return node;
  }

  const brand = button(SITE, 'home', 'brand');
  const sub = document.createElement('p');
  sub.className = 'brand-sub';
  sub.textContent = 'Machine learning you can see move';
  side.replaceChildren(brand, sub);
  const about = document.createElement('div');
  about.className = 'side-about';
  for (const topic of catalog) {
    topic.lessons.forEach((entry, index) => {
      routes.set(entry.id, { id: entry.id, topic, index, entry, element: lesson });
      if (topic.about) about.append(button(entry.tab, entry.id, 'topic-b about-b'));
    });
    if (topic.lessons.length) routes.set(topic.id, routes.get(topic.lessons[0].id));
    if (!topic.about) side.append(button(TC(topic.t), topic.id, 'topic-b'));
  }
  side.append(about);

  function stop(route) {
    if (!route) return;
    if (route === homeRoute) hero.stop();
    else roots.get(route.id)?.runtime?.stop?.();
  }
  function redraw() {
    if (active === homeRoute) hero.draw();
    else roots.get(active?.id)?.runtime?.draw();
  }
  function syncHash(id) {
    if (window.location.hash !== '#' + id) window.history.replaceState(null, '', '#' + id);
  }
  function clearStatus() {
    clearTimeout(loadingTimer);
    status.hidden = true;
    status.replaceChildren();
  }
  function showRoute(route, root) {
    const onHome = route === homeRoute;
    document.body.classList.toggle('in-module', !onHome);
    home.hidden = !onHome;
    lesson.hidden = modbar.hidden = onHome;
    if (onHome) return;
    byId('mod-name').textContent = TC(route.topic.t);
    if (tabs.getAttribute('data-topic') !== route.topic.id) {
      tabs.setAttribute('data-topic', route.topic.id);
      tabs.replaceChildren(
        ...route.topic.lessons.map((entry) => {
          const tab = button(TC(entry.tab), entry.id);
          tab.setAttribute('role', 'tab');
          return tab;
        }),
      );
    }
    for (const tab of tabs.children)
      tab.setAttribute('aria-selected', String(tab.getAttribute('data-go') === route.id));
    if (lesson.firstElementChild !== root) lesson.replaceChildren(root);
  }
  function failure(route) {
    status.hidden = false;
    const message = document.createTextNode('Could not open this lesson. ');
    status.replaceChildren(
      message,
      button('Retry', route.id, 'link'),
      ' ',
      button('Reload', route.id, 'link'),
    );
    const reload = status.lastElementChild;
    reload.removeAttribute('data-go');
    reload.onclick = () => {
      syncHash(route.id);
      window.location.reload();
    };
    if (!active) status.append(' ', button('All topics', 'home', 'link'));
  }

  async function navigate(
    requested,
    { scrollToTop = true, updateHash = true, focus = false } = {},
  ) {
    const route = routes.get(requested) ?? homeRoute;
    const old = active;
    const token = ++desired;
    const isCurrent = () => desired === token;
    transition.cancel();
    clearStatus();
    let cached = roots.get(route.id);
    try {
      if (route !== homeRoute && !cached) {
        loadingTimer = setTimeout(() => {
          if (!isCurrent()) return;
          status.textContent = 'Loading lesson…';
          status.hidden = false;
        }, 150);
        const loaded = await loader.load(route.entry);
        if (!isCurrent()) return active?.id;
        const nextEntry = route.topic.lessons[route.index + 1];
        const root = await renderLesson(loaded, { document, nextEntry, loadText: loader.loadText });
        if (!isCurrent()) return active?.id;
        cached = { root, definition: loaded.definition };
      }
      await transition.run({
        from: old,
        to: route,
        isCurrent,
        commit() {
          if (!isCurrent()) return;
          stop(old);
          showRoute(route, cached?.root);
          if (cached) {
            if (!cached.runtime) cached.runtime = cached.definition.init(cached.root);
            cached.runtime.sync?.();
            roots.set(route.id, cached);
          }
          active = route;
          if (updateHash) syncHash(route.id);
          clearStatus();
          if (scrollToTop) window.scrollTo({ top: 0 });
          const selected = tabs.querySelector('[aria-selected="true"]');
          if (selected && tabs.scrollWidth > tabs.clientWidth)
            selected.scrollIntoView({ block: 'nearest', inline: 'nearest' });
          if (focus) {
            const heading = cached?.root.querySelector('h1');
            if (heading) heading.setAttribute('tabindex', '-1');
            const homeBrand = brand.getBoundingClientRect().height
              ? brand
              : (document.querySelector('.topbar .brand') ?? brand);
            (heading ?? homeBrand).focus({ preventScroll: true });
          }
          window.requestAnimationFrame(() => {
            if (!isCurrent()) return;
            if (route === homeRoute) {
              if (!document.hidden) hero.start();
            } else {
              cached.runtime.draw();
              if (!document.hidden && !RM) cached.runtime.auto?.();
              math.retry(cached.root, () => !signal.aborted && active === route, redraw);
            }
          });
          return cached ? math.prepare(cached.root) : null;
        },
      });
    } catch (error) {
      if (!isCurrent()) return active?.id;
      transition.cancel();
      clearStatus();
      if (old) {
        active = old;
        showRoute(old, roots.get(old.id)?.root);
        syncHash(old.id);
      } else {
        active = homeRoute;
        showRoute(homeRoute);
        syncHash(homeRoute.id);
        if (!document.hidden) hero.start();
      }
      failure(route);
      console.warn('Unable to open lesson', route.id, error);
    }
    return active?.id ?? 'home';
  }
  const on = (target, name, handler) => target?.addEventListener(name, handler, { signal });
  on(menu, 'click', () =>
    menu.setAttribute('aria-expanded', String(document.body.classList.toggle('menu-open'))),
  );
  on(document, 'click', (event) => {
    const target = event.target.closest?.('[data-go]');
    if (!target) return;
    document.body.classList.remove('menu-open');
    menu?.setAttribute('aria-expanded', 'false');
    const destination = target.getAttribute('data-go');
    const hidesTrigger = destination === 'home' ? active?.entry : active === homeRoute;
    void navigate(destination, {
      focus: Boolean(hidesTrigger || lesson.contains(target) || status.contains(target)),
    });
  });
  on(
    window,
    'hashchange',
    () => void navigate(window.location.hash.slice(1), { scrollToTop: false, updateHash: false }),
  );
  on(window, 'resize', redraw);
  on(window.matchMedia('(prefers-color-scheme: dark)'), 'change', redraw);
  on(document, 'visibilitychange', () => {
    if (document.hidden) stop(active);
    else if (active === homeRoute) hero.start();
  });
  on(media, 'change', () => {
    if (RM) {
      transition.cancel();
      stop(active);
      redraw();
    } else if (active === homeRoute && !document.hidden) hero.start();
  });
  const theme =
    typeof window.MutationObserver === 'function' ? new window.MutationObserver(redraw) : null;
  theme?.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  document.fonts?.ready.then(() => {
    if (!signal.aborted) {
      hero.rebuild();
      redraw();
    }
  });
  return {
    navigate,
    redraw,
    getActiveId: () => active?.id ?? 'home',
    destroy() {
      desired++;
      clearStatus();
      status.remove();
      stop(active);
      listeners.abort();
      unwatch();
      theme?.disconnect();
      math.destroy();
      transition.destroy?.();
    },
  };
}
