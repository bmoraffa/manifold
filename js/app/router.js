import { prefersReducedMotion, SITE, toTitleCase, observeReducedMotion } from '../core/shared.js';
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
  const homeView = byId('home'),
    lessonContainer = byId('lesson'),
    lessonNav = byId('modbar');
  const lessonTabs = byId('ltabs'),
    menuButton = byId('menu-b'),
    sidebar = byId('side');
  const routes = new Map(),
    lessonInstancesById = new Map();
  const homeRoute = { id: 'home', element: homeView };
  const math = createMathTypesetter({ window });
  const listeners = new AbortController();
  const signal = listeners.signal;
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  const unwatch = observeReducedMotion(media);
  const navigationStatus = document.createElement('div');
  navigationStatus.className = 'navigation-status';
  navigationStatus.hidden = true;
  navigationStatus.setAttribute('role', 'status');
  lessonContainer.before(navigationStatus);

  let activeRoute = null;
  let navigationVersion = 0;
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
  const brandTagline = document.createElement('p');
  brandTagline.className = 'brand-sub';
  brandTagline.textContent = 'Machine learning you can see move';
  sidebar.replaceChildren(brand, brandTagline);
  const aboutLinks = document.createElement('div');
  aboutLinks.className = 'side-about';
  for (const topic of catalog) {
    topic.lessons.forEach((entry, index) => {
      routes.set(entry.id, { id: entry.id, topic, index, entry, element: lessonContainer });
      if (topic.isAbout) aboutLinks.append(button(entry.tab, entry.id, 'topic-b about-b'));
    });
    if (topic.lessons.length) routes.set(topic.id, routes.get(topic.lessons[0].id));
    if (!topic.isAbout) sidebar.append(button(toTitleCase(topic.title), topic.id, 'topic-b'));
  }
  sidebar.append(aboutLinks);

  function stop(route) {
    if (!route) return;
    if (route === homeRoute) hero.stop();
    else lessonInstancesById.get(route.id)?.runtime?.stop?.();
  }
  function redraw() {
    if (activeRoute === homeRoute) hero.draw();
    else lessonInstancesById.get(activeRoute?.id)?.runtime?.draw();
  }
  function syncHash(id) {
    if (window.location.hash !== '#' + id) window.history.replaceState(null, '', '#' + id);
  }
  function clearStatus() {
    clearTimeout(loadingTimer);
    navigationStatus.hidden = true;
    navigationStatus.replaceChildren();
  }
  function showRoute(route, root) {
    const onHome = route === homeRoute;
    document.body.classList.toggle('in-module', !onHome);
    homeView.hidden = !onHome;
    lessonContainer.hidden = lessonNav.hidden = onHome;
    if (onHome) return;
    byId('mod-name').textContent = toTitleCase(route.topic.title);
    if (lessonTabs.getAttribute('data-topic') !== route.topic.id) {
      lessonTabs.setAttribute('data-topic', route.topic.id);
      lessonTabs.replaceChildren(
        ...route.topic.lessons.map((entry) => {
          const tab = button(toTitleCase(entry.tab), entry.id);
          tab.setAttribute('role', 'tab');
          return tab;
        }),
      );
    }
    for (const tab of lessonTabs.children)
      tab.setAttribute('aria-selected', String(tab.getAttribute('data-go') === route.id));
    if (lessonContainer.firstElementChild !== root) lessonContainer.replaceChildren(root);
  }
  function failure(route) {
    navigationStatus.hidden = false;
    const message = document.createTextNode('Could not open this lesson. ');
    navigationStatus.replaceChildren(
      message,
      button('Retry', route.id, 'link'),
      ' ',
      button('Reload', route.id, 'link'),
    );
    const reload = navigationStatus.lastElementChild;
    reload.removeAttribute('data-go');
    reload.onclick = () => {
      syncHash(route.id);
      window.location.reload();
    };
    if (!activeRoute) navigationStatus.append(' ', button('All topics', 'home', 'link'));
  }

  async function navigate(
    requested,
    { scrollToTop = true, updateHash = true, focus = false } = {},
  ) {
    const route = routes.get(requested) ?? homeRoute;
    const previousRoute = activeRoute;
    const requestVersion = ++navigationVersion;
    const isCurrent = () => navigationVersion === requestVersion;
    transition.cancel();
    clearStatus();
    let lessonInstance = lessonInstancesById.get(route.id);
    try {
      if (route !== homeRoute && !lessonInstance) {
        loadingTimer = setTimeout(() => {
          if (!isCurrent()) return;
          navigationStatus.textContent = 'Loading lesson…';
          navigationStatus.hidden = false;
        }, 150);
        const loaded = await loader.load(route.entry);
        if (!isCurrent()) return activeRoute?.id;
        const nextEntry = route.topic.lessons[route.index + 1];
        const root = await renderLesson(loaded, { document, nextEntry, loadText: loader.loadText });
        if (!isCurrent()) return activeRoute?.id;
        lessonInstance = { root, definition: loaded.definition };
      }
      await transition.run({
        from: previousRoute,
        to: route,
        isCurrent,
        commit() {
          if (!isCurrent()) return;
          stop(previousRoute);
          showRoute(route, lessonInstance?.root);
          if (lessonInstance) {
            if (!lessonInstance.runtime)
              lessonInstance.runtime = lessonInstance.definition.init(lessonInstance.root);
            lessonInstance.runtime.sync?.();
            lessonInstancesById.set(route.id, lessonInstance);
          }
          activeRoute = route;
          if (updateHash) syncHash(route.id);
          clearStatus();
          if (scrollToTop) window.scrollTo({ top: 0 });
          const selected = lessonTabs.querySelector('[aria-selected="true"]');
          if (selected && lessonTabs.scrollWidth > lessonTabs.clientWidth)
            selected.scrollIntoView({ block: 'nearest', inline: 'nearest' });
          if (focus) {
            const heading = lessonInstance?.root.querySelector('h1');
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
              lessonInstance.runtime.draw();
              if (!document.hidden && !prefersReducedMotion) lessonInstance.runtime.auto?.();
              math.retry(
                lessonInstance.root,
                () => !signal.aborted && activeRoute === route,
                redraw,
              );
            }
          });
          return lessonInstance ? math.prepare(lessonInstance.root) : null;
        },
      });
    } catch (error) {
      if (!isCurrent()) return activeRoute?.id;
      transition.cancel();
      clearStatus();
      if (previousRoute) {
        activeRoute = previousRoute;
        showRoute(previousRoute, lessonInstancesById.get(previousRoute.id)?.root);
        syncHash(previousRoute.id);
      } else {
        activeRoute = homeRoute;
        showRoute(homeRoute);
        syncHash(homeRoute.id);
        if (!document.hidden) hero.start();
      }
      failure(route);
      console.warn('Unable to open lesson', route.id, error);
    }
    return activeRoute?.id ?? 'home';
  }
  const on = (target, name, handler) => target?.addEventListener(name, handler, { signal });
  on(menuButton, 'click', () =>
    menuButton.setAttribute('aria-expanded', String(document.body.classList.toggle('menu-open'))),
  );
  on(document, 'click', (event) => {
    const navigationTrigger = event.target.closest?.('[data-go]');
    if (!navigationTrigger) return;
    document.body.classList.remove('menu-open');
    menuButton?.setAttribute('aria-expanded', 'false');
    const destination = navigationTrigger.getAttribute('data-go');
    const leavingTriggerView =
      destination === 'home' ? activeRoute?.entry : activeRoute === homeRoute;
    void navigate(destination, {
      focus: Boolean(
        leavingTriggerView ||
        lessonContainer.contains(navigationTrigger) ||
        navigationStatus.contains(navigationTrigger),
      ),
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
    if (document.hidden) stop(activeRoute);
    else if (activeRoute === homeRoute) hero.start();
  });
  on(media, 'change', () => {
    if (prefersReducedMotion) {
      transition.cancel();
      stop(activeRoute);
      redraw();
    } else if (activeRoute === homeRoute && !document.hidden) hero.start();
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
    getActiveId: () => activeRoute?.id ?? 'home',
    destroy() {
      navigationVersion++;
      clearStatus();
      navigationStatus.remove();
      stop(activeRoute);
      listeners.abort();
      unwatch();
      theme?.disconnect();
      math.destroy();
      transition.destroy?.();
    },
  };
}
