import { TOPICS } from './app/catalog.js';
import { createLessonLoader } from './app/loader.js';
import { mountApp } from './app/router.js';
import { mountThemeSwitches } from './app/theme.js';
import { createTransitions } from './app/transitions.js';
import { mountContactForm } from './ui/contact-form.js';

const contactDialog = document.getElementById('contact-dialog');
mountContactForm(contactDialog.querySelector('[data-contact-form]'));
contactDialog.querySelector('input[name="name"]').setAttribute('autofocus', '');
document
  .getElementById('contact-trigger')
  .addEventListener('click', () => contactDialog.showModal());
contactDialog
  .querySelector('[data-close-contact]')
  .addEventListener('click', () => contactDialog.close());

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
mountThemeSwitches({ document, window });
void app.navigate(location.hash.slice(1), { scrollToTop: false, updateHash: false });
