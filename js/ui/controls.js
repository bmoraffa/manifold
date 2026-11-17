const owners = new WeakMap();

export function controlsFor(root) {
  const controls = owners.get(root.ownerDocument);
  if (!controls) throw new Error('Shared controls must be prepared before initializing a lesson');
  return controls;
}

export function createControls(document, html) {
  if (owners.has(document)) return owners.get(document);
  const source = document.createElement('template');
  source.innerHTML = html;
  const templates = new Map(
    [...source.content.querySelectorAll('template[data-template]')].map((template) => [
      template.getAttribute('data-template'),
      template,
    ]),
  );
  function create(type, attributes, children) {
    const template = templates.get(type);
    if (!template) throw new Error(`Unknown shared control: ${type}`);
    const fragment = template.content.cloneNode(true);
    const find = (selector) => fragment.querySelector(selector);
    const required = (name) => {
      const value = attributes[name];
      if (value == null) throw new Error(`Missing ${name} on shared ${type} control`);
      return value;
    };
    if (type === 'slider') {
      const id = required('id');
      const label = find('label'),
        input = find('input'),
        output = find('output');
      label.setAttribute('for', id);
      find('span').textContent = required('label');
      input.id = id;
      for (const name of ['min', 'max', 'step', 'value']) {
        const value = required(name);
        if (!Number.isFinite(Number(value))) throw new Error(`Invalid ${name} on slider ${id}`);
        input.setAttribute(name, value);
      }
      output.id = id + '-o';
      output.setAttribute('for', id);
    } else if (type === 'number') {
      const id = required('id');
      find('label').setAttribute('for', id);
      find('span').textContent = required('label');
      const input = find('input');
      input.id = id;
      for (const name of ['min', 'max', 'value']) input.setAttribute(name, required(name));
    } else if (type === 'point-status') {
      find('p').id = required('id');
    } else if (type === 'group') {
      find('legend').textContent = required('label');
      if (children) find('fieldset').append(children);
    } else if (type === 'readout') {
      find('div').setAttribute('class', attributes.class || '');
      find('dt').textContent = required('label');
      find('dd').setAttribute('data-r', required('key'));
    } else if (type === 'run-limit') {
      const id = required('id');
      find('span').textContent =
        `Stop after this many ${required('unit')} (empty: run until it converges)`;
      find('label').setAttribute('for', id + '-max');
      find('input').id = id + '-max';
      find('input').setAttribute('aria-describedby', id + '-maxnote');
      find('p').id = id + '-maxnote';
    } else if (type === 'xy-upload') {
      const id = required('id');
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
  owners.set(document, controls);
  return controls;
}
