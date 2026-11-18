import vm from 'node:vm';
import { readFile } from 'node:fs/promises';
import { parseHTML } from 'linkedom';
import { TOPICS } from '../../js/app/catalog.js';
import { renderLesson } from '../../js/app/lesson-view.js';
import { createLessonLoader } from '../../js/app/loader.js';
import {
  $,
  Anim,
  prefersReducedMotion,
  clamp,
  escapeHTML,
  f2,
  f3,
  lgam,
  letterData,
  playBtn,
  randnFrom,
  rng,
  attachRunLimit,
} from '../../js/core/shared.js';
import { UC, uView, uFrame, uData, jacobi } from '../../lessons/unsupervised/shared.js';
import { parseXY } from '../../lessons/regression/shared.js';
import { parseXgbCSV, pct, xgbBuildTree } from '../../lessons/ensembles/shared.js';

const entries = TOPICS.flatMap((topic) => topic.lessons);
const lessons = await Promise.all(
  entries.map(async ({ id, load }) => ({ id, ...(await load()).default })),
);

export async function renderSite() {
  const { document } = parseHTML('<html><body></body></html>');
  const loader = createLessonLoader({ fetchText: (url) => readFile(url, 'utf8') });
  return Promise.all(
    TOPICS.flatMap(({ lessons }) =>
      lessons.map(async (entry, i) => ({
        id: entry.id,
        root: await renderLesson(await loader.load(entry), {
          document,
          nextEntry: lessons[i + 1],
          loadText: loader.loadText,
        }),
      })),
    ),
  );
}

// A small VM isolates the original frame-callback probes and their fake scheduler.
export function loadSite({ reducedMotion = prefersReducedMotion, globals = {} } = {}) {
  const context = vm.createContext({
    $,
    clamp,
    escapeHTML,
    f2,
    f3,
    lgam,
    letterData,
    playBtn,
    randnFrom,
    rng,
    attachRunLimit,
    parseXY,
    parseXgbCSV,
    pct,
    xgbBuildTree,
    UC,
    uView,
    uFrame,
    uData,
    jacobi,
    prefersReducedMotion: reducedMotion,
    Anim: (tick) => Anim(tick, globals),
    document: {},
    window: {},
    performance,
    ...globals,
  });
  return { lessons, evaluate: (source) => vm.runInContext(source, context) };
}
