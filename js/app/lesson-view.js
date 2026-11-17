import { toTitleCase } from '../core/shared.js';
import { createControls } from '../ui/controls.js';

function readLessonParts(document, html) {
  const sourceTemplate = document.createElement('template');
  sourceTemplate.innerHTML = html;
  return new Map(
    [...sourceTemplate.content.querySelectorAll('template[data-part]')].map((part) => [
      part.getAttribute('data-part'),
      part.content.cloneNode(true),
    ]),
  );
}

export async function renderLesson(
  { entry, definition, html, controlsHTML },
  { document, nextEntry, loadText },
) {
  const parts = readLessonParts(document, html);
  const controls = createControls(document, controlsHTML);
  const root = document.createElement('div');
  function appendPart(selector, name, required = true) {
    const part = parts.get(name);
    if (part) root.querySelector(selector).append(part);
    else if (required) throw new Error(`Missing ${name} in ${entry.id}`);
  }
  if (entry.isAboutPage) {
    root.className = 'about about-' + entry.id;
    root.innerHTML =
      '<p class="eyebrow">About Manifold</p><h1></h1><div class="rule"></div><p class="lede"></p><div class="about-body"></div>';
    root.querySelector('h1').textContent = entry.title;
    appendPart('.lede', 'lede');
    appendPart('.about-body', 'body');
    return root;
  }
  root.innerHTML = `<h1></h1><p class="lede"></p>${parts.has('intro') ? '<div class="prose intro"></div>' : ''}
  <template data-stage></template>
  <p class="try"><b>Try this.</b> </p>
  <div class="prose after"></div>
  <details><summary>Check yourself</summary><div class="prose"><p><b></b></p><p></p></div></details>
  <div class="pager"></div>`;
  root.querySelector('h1').textContent = toTitleCase(entry.title);
  appendPart('.lede', 'lede');
  appendPart('.intro', 'intro', false);
  appendPart('.try', 'tryit');
  appendPart('.after', 'math');
  appendPart('details p b', 'question');
  appendPart('details p:last-child', 'answer');
  const stage = definition.renderStage
    ? await definition.renderStage(parts, { document, loadText, controls })
    : parts.get('stage');
  if (!stage) throw new Error(`Missing stage in ${entry.id}`);
  root.querySelector('[data-stage]').replaceWith(stage);
  controls.expand(root);
  if (nextEntry) {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('data-go', nextEntry.id);
    button.textContent = `Next: ${toTitleCase(nextEntry.tab)} →`;
    root.querySelector('.pager').append(button);
  }
  return root;
}

export async function renderFamily(url, id, parts, { document, loadText }) {
  const sourceTemplate = document.createElement('template');
  sourceTemplate.innerHTML = await loadText(url);
  for (const node of sourceTemplate.content.querySelectorAll('[data-id-suffix]')) {
    node.id = id + node.getAttribute('data-id-suffix');
    node.removeAttribute('data-id-suffix');
  }
  for (const node of sourceTemplate.content.querySelectorAll('[data-for-suffix]')) {
    node.setAttribute('for', id + node.getAttribute('data-for-suffix'));
    node.removeAttribute('data-for-suffix');
  }
  for (const node of sourceTemplate.content.querySelectorAll('template[data-control-id-suffix]')) {
    node.setAttribute('data-id', id + node.getAttribute('data-control-id-suffix'));
    node.removeAttribute('data-control-id-suffix');
  }
  for (const node of sourceTemplate.content.querySelectorAll('template[data-lesson-id]')) {
    node.setAttribute('data-id', id);
    node.removeAttribute('data-lesson-id');
  }
  for (const slot of sourceTemplate.content.querySelectorAll('template[data-slot]')) {
    const part = parts.get(slot.getAttribute('data-slot'));
    if (part) slot.replaceWith(part);
    else slot.remove();
  }
  return sourceTemplate.content;
}
