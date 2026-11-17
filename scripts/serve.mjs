import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const types = { html: 'text/html', js: 'text/javascript', css: 'text/css' };
const port = Number(process.env.PORT || 4173);

createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/' || pathname === '/manifold') {
      response.writeHead(302, { Location: '/manifold/' }).end();
      return;
    }
    const relative = pathname.replace(/^\/manifold\//, '') || 'index.html';
    const entry =
      relative === 'index.html' ||
      /^(?:js|css|lessons|pages)\/(?:[a-z0-9-]+\/)*[a-z0-9-]+\.(?:html|js|css)$/.test(relative);
    if (!entry || !['GET', 'HEAD'].includes(request.method)) {
      response.writeHead(404).end();
      return;
    }
    const body = await readFile(new URL(relative, root));
    response.writeHead(200, {
      'Content-Type': types[relative.split('.').at(-1)] + '; charset=utf-8',
    });
    response.end(request.method === 'HEAD' ? undefined : body);
  } catch {
    response.writeHead(404).end();
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Manifold preview: http://127.0.0.1:${port}/manifold/`);
});
