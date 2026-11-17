/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
const loadStaticPage = () => Promise.resolve({ default: { init: () => ({ draw() {} }) } });
export default {
  id: 'about',
  t: 'About Manifold',
  about: true,
  lessons: [
    {
      id: 'references',
      tab: 'References',
      title: 'References',
      page: true,
      viewUrl: new URL('./references/page.html', import.meta.url),
      load: loadStaticPage,
    },
    {
      id: 'books',
      tab: 'Suggested Books',
      title: 'Suggested Books',
      page: true,
      viewUrl: new URL('./books/page.html', import.meta.url),
      load: loadStaticPage,
    },
    {
      id: 'acknowledgements',
      tab: 'Acknowledgements',
      title: 'Acknowledgements',
      page: true,
      viewUrl: new URL('./acknowledgements/page.html', import.meta.url),
      load: loadStaticPage,
    },
    {
      id: 'cite',
      tab: 'How to Cite',
      title: 'How to Cite Manifold',
      page: true,
      viewUrl: new URL('./cite/page.html', import.meta.url),
      load: () => import('./cite/page.js'),
    },
  ],
};
