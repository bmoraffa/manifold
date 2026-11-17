/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */

const AUTHORS = 'Bahman Moraffah and Dachun Sun';

const SITE_FULL = 'Manifold: Machine Learning You Can See Move';

const SITE_YEAR = '2026';

const SITE_URL = 'https://bmoraffa.github.io/manifold/';

const siteUrl = () => {
  if (SITE_URL) return SITE_URL;
  try {
    if (
      /^https?:$/.test(location.protocol) &&
      !/claude|anthropic|localhost|127\.0\.0\.1/.test(location.hostname)
    )
      return location.href.split('#')[0];
  } catch (e) {}
  return 'https://YOUR-ADDRESS-HERE/';
};

export { AUTHORS, SITE_FULL, SITE_YEAR, SITE_URL, siteUrl };
