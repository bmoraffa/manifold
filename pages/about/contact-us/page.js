import { mountContactForm } from '../../../js/ui/contact-form.js';

export default {
  init(root) {
    mountContactForm(root.querySelector('[data-contact-form]'), {
      feedbackBefore: root.querySelector('.lede'),
    });
    return { draw() {} };
  },
};
