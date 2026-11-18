const preferenceKey = 'manifold-theme';

export function mountThemeSwitches({ document, window }) {
  const root = document.documentElement;
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const template = document.getElementById('theme-switch');
  const switches = [...document.querySelectorAll('[data-theme-anchor]')].map((anchor) => {
    const control = template.content.firstElementChild.cloneNode(true);
    anchor.after(control);
    return control;
  });

  try {
    const saved = window.localStorage.getItem(preferenceKey);
    if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;
  } catch {
    // Storage can be disabled. The switch still works for this page.
  }

  const isDark = () => (root.dataset.theme ? root.dataset.theme === 'dark' : systemTheme.matches);
  function sync() {
    const pressed = String(isDark());
    for (const control of switches) control.setAttribute('aria-pressed', pressed);
  }

  function toggleTheme() {
    const theme = isDark() ? 'light' : 'dark';
    root.dataset.theme = theme;
    sync();
    try {
      window.localStorage.setItem(preferenceKey, theme);
    } catch {
      // Keep the current choice if storage is unavailable.
    }
  }

  for (const control of switches) {
    control.addEventListener('click', () => {
      if (!reducedMotion.matches && document.startViewTransition) {
        const transition = document.startViewTransition(toggleTheme);
        // A quick second toggle can skip the first animation while it is preparing.
        void transition.ready.catch(() => {});
      } else {
        toggleTheme();
      }
    });
  }
  systemTheme.addEventListener('change', sync);
  sync();
}
