/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
export default {
  id: 'ensembles',
  t: 'Ensembles',
  lessons: [
    {
      id: 'bagging',
      tab: 'Bagging',
      title: 'Many unstable trees, one steady vote',
      viewUrl: new URL('./bagging/lesson.html', import.meta.url),
      load: () => import('./bagging/lesson.js'),
    },
    {
      id: 'forest',
      tab: 'Random forests',
      title: 'Make the trees disagree',
      viewUrl: new URL('./forest/lesson.html', import.meta.url),
      load: () => import('./forest/lesson.js'),
    },
    {
      id: 'adaboost',
      tab: 'AdaBoost',
      title: 'Each learner fixes the last one’s mistakes',
      viewUrl: new URL('./adaboost/lesson.html', import.meta.url),
      load: () => import('./adaboost/lesson.js'),
    },
    {
      id: 'gboost',
      tab: 'Gradient boosting',
      title: 'Fit what is left over, again and again',
      viewUrl: new URL('./gboost/lesson.html', import.meta.url),
      load: () => import('./gboost/lesson.js'),
    },
    {
      id: 'xgboost',
      tab: 'XGBoost',
      title: 'Gradient boosting, engineered',
      viewUrl: new URL('./xgboost/lesson.html', import.meta.url),
      load: () => import('./xgboost/lesson.js'),
    },
    {
      id: 'stacking',
      tab: 'Stacking',
      title: 'A model that decides whom to trust',
      viewUrl: new URL('./stacking/lesson.html', import.meta.url),
      load: () => import('./stacking/lesson.js'),
    },
  ],
};
