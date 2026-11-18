/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
const loadStaticPage = () => Promise.resolve({ default: { init: () => ({ draw() {} }) } });
export default {
  id: 'about',
  title: 'About Manifold',
  isAbout: true,
  lessons: [
    {
      id: 'references',
      tab: 'References',
      title: 'References',
      isAboutPage: true,
      viewUrl: new URL('./references/page.html', import.meta.url),
      load: loadStaticPage,
    },
    {
      id: 'books',
      tab: 'Suggested Books',
      title: 'Suggested Books',
      isAboutPage: true,
      viewUrl: new URL('./books/page.html', import.meta.url),
      load: loadStaticPage,
    },
    {
      id: 'acknowledgements',
      tab: 'Acknowledgements',
      title: 'Acknowledgements',
      isAboutPage: true,
      viewUrl: new URL('./acknowledgements/page.html', import.meta.url),
      load: loadStaticPage,
    },
    {
      id: 'cite',
      tab: 'How to Cite',
      title: 'How to Cite Manifold',
      isAboutPage: true,
      viewUrl: new URL('./cite/page.html', import.meta.url),
      load: () => import('./cite/page.js'),
    },
    {
      id: 'contact-us',
      tab: 'Contact Us',
      title: 'Contact Us',
      isAboutPage: true,
      viewUrl: new URL('./contact-us/page.html', import.meta.url),
      load: () => import('./contact-us/page.js'),
    },
  ],
};
