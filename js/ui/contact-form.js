export function mountContactForm(container, { feedbackBefore } = {}) {
  const template = container.ownerDocument.getElementById('contact-form');
  const content = template.content.cloneNode(true);
  const form = content.querySelector('form');
  const fields = form.querySelector('fieldset');
  const status = form.querySelector('[role="status"]');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (fields.disabled) return;

    const body = JSON.stringify(Object.fromEntries(new FormData(form)));
    fields.disabled = true;
    status.textContent = 'Sending your message…';
    try {
      const response = await fetch(form.getAttribute('action'), {
        method: 'POST',
        headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
        body,
      });
      if (!response.ok || (await response.json()).success !== true) {
        throw new Error('Web3Forms did not accept the submission');
      }
      form.reset();
      status.textContent = 'Thanks. Your message was sent.';
    } catch {
      status.textContent = 'We couldn’t send your message. Please try again.';
    } finally {
      fields.disabled = false;
    }
  });
  if (feedbackBefore) feedbackBefore.before(content.querySelector('.contact-feedback'));
  container.append(content);
}
