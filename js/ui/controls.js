const controlsByDocument = new WeakMap();

export function controlsFor(root) {
  const controls = controlsByDocument.get(root.ownerDocument);
  if (!controls) throw new Error('Shared controls must be prepared before initializing a lesson');
  return controls;
}

export function createControls(document, html) {
  if (controlsByDocument.has(document)) return controlsByDocument.get(document);
  const sourceTemplate = document.createElement('template');
  sourceTemplate.innerHTML = html;
  const templates = new Map(
    [...sourceTemplate.content.querySelectorAll('template[data-template]')].map((template) => [
      template.getAttribute('data-template'),
      template,
    ]),
  );
  function create(type, attributes, children) {
    const template = templates.get(type);
    if (!template) throw new Error(`Unknown shared control: ${type}`);
    const fragment = template.content.cloneNode(true);
    const find = (selector) => fragment.querySelector(selector);
    const requireAttribute = (name) => {
      const value = attributes[name];
      if (value == null) throw new Error(`Missing ${name} on shared ${type} control`);
      return value;
    };
    if (type === 'slider') {
      const id = requireAttribute('id');
      const label = find('label'),
        input = find('input'),
        output = find('output');
      label.setAttribute('for', id);
      find('span').textContent = requireAttribute('label');
      input.id = id;
      for (const name of ['min', 'max', 'step', 'value']) {
        const value = requireAttribute(name);
        if (!Number.isFinite(Number(value))) throw new Error(`Invalid ${name} on slider ${id}`);
        input.setAttribute(name, value);
      }
      output.id = id + '-o';
      output.setAttribute('for', id);
    } else if (type === 'number') {
      const id = requireAttribute('id');
      find('label').setAttribute('for', id);
      find('span').textContent = requireAttribute('label');
      const input = find('input');
      input.id = id;
      for (const name of ['min', 'max', 'value']) input.setAttribute(name, requireAttribute(name));
    } else if (type === 'point-status') {
      find('p').id = requireAttribute('id');
    } else if (type === 'group') {
      find('legend').textContent = requireAttribute('label');
      if (children) find('fieldset').append(children);
    } else if (type === 'readout') {
      find('div').setAttribute('class', attributes.class || '');
      find('dt').textContent = requireAttribute('label');
      find('dd').setAttribute('data-r', requireAttribute('key'));
    } else if (type === 'run-limit') {
      const id = requireAttribute('id');
      find('span').textContent =
        `Stop after this many ${requireAttribute('unit')} (empty: run until it converges)`;
      find('label').setAttribute('for', id + '-max');
      find('input').id = id + '-max';
      find('input').setAttribute('aria-describedby', id + '-maxnote');
      find('p').id = id + '-maxnote';
    } else if (type === 'xy-upload') {
      const id = requireAttribute('id');
      find('input').id = id + '-file';
      find('p').id = id + '-st';
    }
    // Indentation in the shared template is outside the control's boundary.
    for (const node of [...fragment.childNodes]) {
      if (node.nodeType === 3 && !node.nodeValue.trim()) node.remove();
    }
    return fragment;
  }
  function expand(root) {
    for (const declaration of root.querySelectorAll('template[data-control]')) {
      const content = declaration.content.cloneNode(true);
      expand(content);
      const attributes = Object.fromEntries(
        declaration
          .getAttributeNames()
          .filter((name) => name.startsWith('data-'))
          .map((name) => [name.slice(5), declaration.getAttribute(name)]),
      );
      declaration.replaceWith(create(attributes.control, attributes, content));
    }
  }
  const controls = { expand, create };
  controlsByDocument.set(document, controls);
  return controls;
}
