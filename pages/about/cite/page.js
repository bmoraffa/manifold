/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $ } from '../../../js/core/shared.js';
import { AUTHOR, SITE_FULL, SITE_YEAR, siteUrl } from '../shared.js';

export default {
  init(root) {
    const box = $('#cite-box', root),
      url = siteUrl(),
      accessedAt = new Date(),
      monthNames = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
      ],
      isoDate = accessedAt.toISOString().slice(0, 10),
      longDate =
        monthNames[accessedAt.getMonth()] +
        ' ' +
        accessedAt.getDate() +
        ', ' +
        accessedAt.getFullYear(),
      shortDate =
        monthNames[accessedAt.getMonth()].slice(0, 3) +
        '. ' +
        accessedAt.getDate() +
        ', ' +
        accessedAt.getFullYear();
    const citationFormats = [
      [
        'BibTeX',
        `@misc{moraffah${SITE_YEAR}manifold,\n  author       = {Moraffah, Bahman},\n  title        = {${SITE_FULL}},\n  year         = {${SITE_YEAR}},\n  howpublished = {\\url{${url}}},\n  note         = {Interactive website. Accessed: ${isoDate}}\n}`,
      ],
      [
        'BibLaTeX',
        `@online{moraffah${SITE_YEAR}manifold,\n  author  = {Moraffah, Bahman},\n  title   = {${SITE_FULL}},\n  year    = {${SITE_YEAR}},\n  url     = {${url}},\n  urldate = {${isoDate}}\n}`,
      ],
      ['APA', `Moraffah, B. (${SITE_YEAR}). ${SITE_FULL} [Interactive website]. ${url}`],
      [
        'IEEE',
        `B. Moraffah, “${SITE_FULL}.” ${SITE_YEAR}. [Online]. Available: ${url} (accessed ${shortDate}).`,
      ],
      [
        'Chicago',
        `Moraffah, Bahman. ${SITE_YEAR}. “${SITE_FULL}.” Interactive website. Accessed ${longDate}. ${url}`,
      ],
      [
        'MLA',
        `Moraffah, Bahman. ${SITE_FULL}. ${SITE_YEAR}, ${url}. Accessed ${accessedAt.getDate()} ${monthNames[accessedAt.getMonth()].slice(0, 3)}. ${accessedAt.getFullYear()}.`,
      ],
      ['Plain text', `Bahman Moraffah, ${SITE_FULL}, ${SITE_YEAR}. ${url}`],
    ];
    $('#cite-main', root).textContent = `${AUTHOR} (${SITE_YEAR}). ${SITE_FULL}.`;
    const citationLink = $('#cite-link', root);
    citationLink.href = url;
    citationLink.textContent = url.replace('https://', '').replace(/\/$/, '');
    box.innerHTML = citationFormats
      .map(
        ([n], i) =>
          `<div class="cite"><div class="cite-h"><span>${n}</span><button type="button" data-i="${i}">Copy</button></div><pre></pre></div>`,
      )
      .join('');
    box.querySelectorAll('pre').forEach((p, i) => {
      p.textContent = citationFormats[i][1];
    });
    box.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-i]');
      if (!b) return;
      const citationText = citationFormats[+b.dataset.i][1],
        showCopied = () => {
          b.textContent = 'Copied';
          setTimeout(() => {
            b.textContent = 'Copy';
          }, 1400);
        };
      const copyViaSelection = () => {
        try {
          const r = document.createRange();
          r.selectNodeContents(b.closest('.cite').querySelector('pre'));
          const s = getSelection();
          s.removeAllRanges();
          s.addRange(r);
          document.execCommand('copy');
          showCopied();
        } catch (e) {
          b.textContent = 'Select and copy';
        }
      };
      try {
        navigator.clipboard.writeText(citationText).then(showCopied, copyViaSelection);
      } catch (e) {
        copyViaSelection();
      }
    });
    return { draw() {} };
  },
};
