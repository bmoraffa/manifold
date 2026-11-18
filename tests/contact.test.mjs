import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { TOPICS } from '../js/app/catalog.js';
import { createLessonLoader } from '../js/app/loader.js';
import { renderLesson } from '../js/app/lesson-view.js';
import { mountContactForm } from '../js/ui/contact-form.js';

test('the sitewide Contact Us button and About page share the Web3Forms form', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { document } = parseHTML(html);
  const button = document.querySelector('button[aria-controls="contact-dialog"]');
  assert.equal(button?.getAttribute('aria-label'), 'Contact Us');
  assert.equal(button.closest('.app'), null);
  const modalSlot = document.querySelector('dialog#contact-dialog [data-contact-form]');
  assert.ok(modalSlot);
  mountContactForm(modalSlot);
  const template = document.querySelector('template#contact-form');
  assert.ok(template);

  const about = TOPICS.find((topic) => topic.isAbout);
  const entry = about.lessons.find(({ id }) => id === 'contact-us');
  assert.ok(entry?.isAboutPage);
  const loader = createLessonLoader({ fetchText: (url) => readFile(url, 'utf8') });
  const loaded = await loader.load(entry);
  const root = await renderLesson(loaded, { document, loadText: loader.loadText });
  loaded.definition.init(root);
  assert.equal(root.querySelector('h1')?.textContent, 'Contact Us');
  const emailLink = root.querySelector('a[href="mailto:manifold.ai.tool@gmail.com"]');
  assert.equal(emailLink?.textContent, 'email us');
  assert.equal(modalSlot.querySelector('.contact-email a')?.outerHTML, emailLink.outerHTML);
  const feedback = root.querySelector('.contact-feedback a');
  assert.equal(
    feedback?.getAttribute('href'),
    'https://docs.google.com/forms/d/e/1FAIpQLScH9mwrNkUZqgA8CaJogIVm6H_CHn90Mvqp3FkKILPEv247yA/viewform',
  );
  assert.equal(modalSlot.querySelector('.contact-feedback a')?.outerHTML, feedback.outerHTML);
  assert.equal(
    feedback.closest('.contact-feedback').nextElementSibling,
    root.querySelector('.lede'),
  );
  assert.equal(modalSlot.querySelector('.contact-feedback'), modalSlot.firstElementChild);
  assert.ok(modalSlot.firstElementChild.nextElementSibling.matches('.contact-email'));

  const form = root.querySelector('form');
  assert.equal(form?.getAttribute('action'), 'https://api.web3forms.com/submit');
  assert.equal(form.getAttribute('method'), 'post');
  assert.ok(form.querySelector('input[type="hidden"][name="access_key"]')?.value);
  const fields = [...form.querySelectorAll('label > [required]')];
  assert.deepEqual(
    fields.map((field) => field.getAttribute('name')),
    ['name', 'email', 'message'],
  );
  assert.equal(fields[1].getAttribute('type'), 'email');
  assert.equal(fields[2].localName, 'textarea');
});

test('contact submission blocks duplicates, retains a failed message, and clears a sent message', async (t) => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { document, window } = parseHTML(html);
  const slot = document.querySelector('#contact-dialog [data-contact-form]');
  mountContactForm(slot);
  const form = slot.querySelector('form');
  const fields = form.querySelector('fieldset');
  const status = form.querySelector('[role="status"]');
  const reset = (form.reset = t.mock.fn());
  const values = new Map([
    ['name', 'Example Sender'],
    ['email', 'example@example.com'],
    ['message', 'Example message'],
  ]);
  t.mock.method(globalThis, 'FormData', function () {
    return values;
  });
  let respond;
  const fetch = t.mock.method(
    globalThis,
    'fetch',
    () => new Promise((resolve) => (respond = resolve)),
  );
  const submit = () => form.dispatchEvent(new window.Event('submit', { cancelable: true }));

  submit();
  submit();
  assert.equal(fields.disabled, true);
  assert.match(status.textContent, /Sending/);
  assert.equal(fetch.mock.callCount(), 1);
  const [url, request] = fetch.mock.calls[0].arguments;
  assert.equal(url, 'https://api.web3forms.com/submit');
  assert.equal(request.method, 'POST');
  assert.deepEqual(JSON.parse(request.body), Object.fromEntries(values));

  respond({ ok: false });
  await new Promise(setImmediate);
  assert.equal(fields.disabled, false);
  assert.match(status.textContent, /try again/);
  assert.equal(reset.mock.callCount(), 0);

  submit();
  respond({ ok: true, json: async () => ({ success: true }) });
  await new Promise(setImmediate);
  assert.equal(fields.disabled, false);
  assert.match(status.textContent, /message was sent/);
  assert.equal(reset.mock.callCount(), 1);
});
