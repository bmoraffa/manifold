import { readFileSync } from 'node:fs';
import vm from 'node:vm';

export function loadSite({ reducedMotion = false, globals = {} } = {}) {
  const root = new URL('../../', import.meta.url);
  const context = vm.createContext({
    matchMedia: () => ({ matches: reducedMotion }),
    document: {},
    window: {},
    performance,
    ...globals,
  });
  const html = readFileSync(new URL('index.html', root), 'utf8');
  for (const [, src] of html.matchAll(/<script[^>]+src="(js\/[^"]+)"/g)) {
    if (src === 'js/app.js') continue;
    vm.runInContext(readFileSync(new URL(src, root), 'utf8'), context, { filename: src });
  }
  const evaluate = (source) => vm.runInContext(source, context);
  return {
    evaluate,
    lessons: evaluate(
      '[...LESSONS,...REG,...FND,...CLF,...ENS,...UNS,...NNL,...DLL,...GEN,...LML,...RLL,...PML,...ABOUT]',
    ),
  };
}
