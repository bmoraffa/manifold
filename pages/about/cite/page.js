/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import { $ } from '../../../js/core/shared.js';
import { AUTHOR, SITE_FULL, SITE_YEAR, siteUrl } from '../shared.js';

export default {
  init(root) {
    const box = $('#cite-box', root),
      u = siteUrl(),
      d = new Date(),
      mon = [
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
      iso = d.toISOString().slice(0, 10),
      long = mon[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear(),
      short = mon[d.getMonth()].slice(0, 3) + '. ' + d.getDate() + ', ' + d.getFullYear();
    const F = [
      [
        'BibTeX',
        `@misc{moraffah${SITE_YEAR}manifold,\n  author       = {Moraffah, Bahman},\n  title        = {${SITE_FULL}},\n  year         = {${SITE_YEAR}},\n  howpublished = {\\url{${u}}},\n  note         = {Interactive website. Accessed: ${iso}}\n}`,
      ],
      [
        'BibLaTeX',
        `@online{moraffah${SITE_YEAR}manifold,\n  author  = {Moraffah, Bahman},\n  title   = {${SITE_FULL}},\n  year    = {${SITE_YEAR}},\n  url     = {${u}},\n  urldate = {${iso}}\n}`,
      ],
      ['APA', `Moraffah, B. (${SITE_YEAR}). ${SITE_FULL} [Interactive website]. ${u}`],
      [
        'IEEE',
        `B. Moraffah, “${SITE_FULL}.” ${SITE_YEAR}. [Online]. Available: ${u} (accessed ${short}).`,
      ],
      [
        'Chicago',
        `Moraffah, Bahman. ${SITE_YEAR}. “${SITE_FULL}.” Interactive website. Accessed ${long}. ${u}`,
      ],
      [
        'MLA',
        `Moraffah, Bahman. ${SITE_FULL}. ${SITE_YEAR}, ${u}. Accessed ${d.getDate()} ${mon[d.getMonth()].slice(0, 3)}. ${d.getFullYear()}.`,
      ],
      ['Plain text', `Bahman Moraffah, ${SITE_FULL}, ${SITE_YEAR}. ${u}`],
    ];
    $('#cite-main', root).textContent = `${AUTHOR} (${SITE_YEAR}). ${SITE_FULL}.`;
    const lk = $('#cite-link', root);
    lk.href = u;
    lk.textContent = u.replace('https://', '').replace(/\/$/, '');
    box.innerHTML = F.map(
      ([n], i) =>
        `<div class="cite"><div class="cite-h"><span>${n}</span><button type="button" data-i="${i}">Copy</button></div><pre></pre></div>`,
    ).join('');
    box.querySelectorAll('pre').forEach((p, i) => {
      p.textContent = F[i][1];
    });
    box.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-i]');
      if (!b) return;
      const tx = F[+b.dataset.i][1],
        ok = () => {
          b.textContent = 'Copied';
          setTimeout(() => {
            b.textContent = 'Copy';
          }, 1400);
        };
      const fb = () => {
        try {
          const r = document.createRange();
          r.selectNodeContents(b.closest('.cite').querySelector('pre'));
          const s = getSelection();
          s.removeAllRanges();
          s.addRange(r);
          document.execCommand('copy');
          ok();
        } catch (e) {
          b.textContent = 'Select and copy';
        }
      };
      try {
        navigator.clipboard.writeText(tx).then(ok, fb);
      } catch (e) {
        fb();
      }
    });
    return { draw() {} };
  },
};
