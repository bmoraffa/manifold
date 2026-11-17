const controlsUrl = new URL('../ui/controls.html', import.meta.url);

async function fetchSource(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url.pathname} (${response.status})`);
  return response.text();
}

export function createLessonLoader({ fetchText = fetchSource } = {}) {
  const texts = new Map();
  const lessons = new Map();
  function cached(cache, key, start) {
    if (!cache.has(key)) {
      const pending = Promise.resolve()
        .then(start)
        .catch((error) => {
          cache.delete(key);
          throw error;
        });
      cache.set(key, pending);
    }
    return cache.get(key);
  }
  const loadText = (url) => cached(texts, url.href, () => fetchText(url));
  return {
    loadText,
    load(entry) {
      return cached(lessons, entry, async () => {
        const [html, { default: definition }, controlsHTML] = await Promise.all([
          loadText(entry.viewUrl),
          entry.load(),
          loadText(controlsUrl),
        ]);
        if (typeof definition?.init !== 'function')
          throw new Error(`Missing lesson behavior: ${entry.id}`);
        return { entry, definition, html, controlsHTML };
      });
    },
  };
}
