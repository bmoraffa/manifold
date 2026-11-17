/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
import {
  $,
  Anim,
  sharedGaussianShape,
  Hill,
  readThemeColors,
  drawMap,
  eig,
  f2,
  f3,
  grab,
  playBtn,
  setReadout,
  shapeBind,
} from '../../../js/core/shared.js';
import { standardNormalPairs } from '../shared.js';

export default {
  init(root) {
    const c3 = $('#cov-c3', root),
      c2 = $('#cov-c2', root),
      st = $('#cov-status', root),
      ia = $('#cov-a', root),
      ib = $('#cov-b', root),
      mirroredCovarianceInput = $('#cov-b2', root),
      id = $('#cov-d', root);
    let draw,
      shapeTween = null,
      playback;
    const hill = Hill(c3, () => draw(), $('.viewbar', root));
    const fields = () => {
      const e = eig(sharedGaussianShape),
        ae = document.activeElement;
      if (ae !== ia) ia.value = f2(e.a);
      if (ae !== ib) ib.value = f2(e.b);
      mirroredCovarianceInput.value = f2(e.b);
      if (ae !== id) id.value = f2(e.d);
    };
    const sh = shapeBind(root, 'cov', sharedGaussianShape, () => {
        fields();
        draw();
      }),
      syncUI = () => {
        sh.sync();
        fields();
      };
    draw = () => {
      const colors = readThemeColors(),
        e = eig(sharedGaussianShape);
      drawMap(c2, colors, sharedGaussianShape, { pts: standardNormalPairs, eig: true });
      hill.draw(colors, sharedGaussianShape, {});
      setReadout(
        root,
        'shape',
        Math.abs(sharedGaussianShape.rho) < 0.03
          ? Math.abs(sharedGaussianShape.sx - sharedGaussianShape.sy) < 0.04
            ? 'Isotropic (circles)'
            : 'Diagonal (axis-aligned)'
          : 'Full (tilted)',
      );
      setReadout(root, 'l1', f3(e.l1));
      setReadout(root, 'l2', f3(e.l2));
      setReadout(root, 'th', ((e.th * 180) / Math.PI).toFixed(0) + '°');
      setReadout(root, 'det', f3(e.l1 * e.l2));
    };
    const tweenTo = (tg) => {
      const s = { ...sharedGaussianShape };
      let k = 0;
      if (shapeTween) shapeTween.stop();
      shapeTween = Anim(() => {
        k++;
        const u = Math.min(1, k / 40),
          e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u);
        sharedGaussianShape.sx = s.sx + (tg.sx - s.sx) * e;
        sharedGaussianShape.sy = s.sy + (tg.sy - s.sy) * e;
        sharedGaussianShape.rho = s.rho + (tg.rho - s.rho) * e;
        syncUI();
        draw();
        if (u >= 1) shapeTween.stop();
      });
      shapeTween.start();
    };
    const shapeFromCovariance = (a, b, d) => {
      if (!(isFinite(a) && isFinite(b) && isFinite(d)))
        return 'Fill in all three entries with numbers.';
      if (a < 0.2 || d < 0.2 || a > 4 || d > 4)
        return 'The two diagonal entries are variances. Keep each between 0.2 and 4.';
      const lim = Math.sqrt(a * d);
      if (Math.abs(b) >= lim)
        return (
          'Not a valid covariance matrix: the off-diagonal entry must be smaller in size than √(Σ₁₁·Σ₂₂) = ' +
          f2(lim) +
          '. Otherwise one direction would have negative variance.'
        );
      if (Math.abs(b) > 0.97 * lim)
        return (
          'That is valid but too close to a flat line to draw. Keep the off-diagonal entry under ' +
          f2(0.97 * lim) +
          ' in size.'
        );
      return { sx: Math.sqrt(a), sy: Math.sqrt(d), rho: b / lim };
    };
    const typed = () => {
      playback.stop();
      const r = shapeFromCovariance(
        parseFloat(ia.value),
        parseFloat(ib.value),
        parseFloat(id.value),
      );
      if (typeof r === 'string') {
        st.textContent = r;
        return;
      }
      st.textContent = '';
      tweenTo(r);
    };
    [ia, ib, id].forEach((i) => i.addEventListener('change', typed));
    const PRE = [...root.querySelectorAll('#cov-presets button')].map((b) =>
      b.dataset.p.split(',').map(Number),
    );
    let pi = 0,
      frameCount = 0;
    const tour = Anim(() => {
      if (frameCount++ % 130 === 0) {
        const p = PRE[pi++ % PRE.length];
        st.textContent = '';
        tweenTo(shapeFromCovariance(p[0], p[1], p[2]));
      }
    });
    playback = playBtn($('#cov-play', root), tour);
    $('#cov-presets', root).addEventListener('click', (e) => {
      const p = e.target.dataset && e.target.dataset.p;
      if (!p) return;
      playback.stop();
      st.textContent = '';
      const v = p.split(',').map(Number);
      tweenTo(shapeFromCovariance(v[0], v[1], v[2]));
    });
    sh.els.forEach((el) => grab(el, playback));
    return {
      draw,
      stop() {
        playback.stop();
        frameCount = 0;
        if (shapeTween) shapeTween.stop();
        hill.spin(false);
      },
      auto() {
        frameCount = 60;
        playback.start();
      },
      sync: syncUI,
    };
  },
};
