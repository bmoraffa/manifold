import { TOPICS } from './app/catalog.js';
import { createLessonLoader } from './app/loader.js';
import { mountApp } from './app/router.js';
import { createTransitions } from './app/transitions.js';

const transition = createTransitions({
  window,
  document,
  content: document.getElementById('lesson'),
  nav: document.getElementById('modbar'),
});
const app = mountApp({
  document,
  window,
  catalog: TOPICS,
  loader: createLessonLoader(),
  transition,
});
void app.navigate(location.hash.slice(1), { scrollToTop: false, updateHash: false });
