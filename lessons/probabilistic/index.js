/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
export default {
  id: 'probabilistic',
  title: 'Probabilistic methods',
  lessons: [
    {
      id: 'bayes-update',
      tab: 'Bayesian updating',
      title: 'Bayesian updating: change your mind by the right amount',
      viewUrl: new URL('./bayes-update/lesson.html', import.meta.url),
      load: () => import('./bayes-update/lesson.js'),
    },
    {
      id: 'mcmc',
      tab: 'MCMC',
      title: 'MCMC: explore a landscape you can only feel underfoot',
      viewUrl: new URL('./mcmc/lesson.html', import.meta.url),
      load: () => import('./mcmc/lesson.js'),
    },
    {
      id: 'variational-bayes',
      tab: 'Variational Bayes',
      title: 'Variational Bayes: replace a hard belief with the closest easy one',
      viewUrl: new URL('./variational-bayes/lesson.html', import.meta.url),
      load: () => import('./variational-bayes/lesson.js'),
    },
    {
      id: 'hmm',
      tab: 'Hidden Markov models',
      title: 'Hidden Markov models: guess what you cannot see from what you can',
      viewUrl: new URL('./hmm/lesson.html', import.meta.url),
      load: () => import('./hmm/lesson.js'),
    },
    {
      id: 'kalman',
      tab: 'Kalman filter',
      title: 'The Kalman filter: trust the forecast and the sensor, each as much as it deserves',
      viewUrl: new URL('./kalman/lesson.html', import.meta.url),
      load: () => import('./kalman/lesson.js'),
    },
    {
      id: 'recommender',
      tab: 'Recommender systems',
      title: 'Recommender systems: fill in the ratings nobody gave',
      viewUrl: new URL('./recommender/lesson.html', import.meta.url),
      load: () => import('./recommender/lesson.js'),
    },
    {
      id: 'time-series',
      tab: 'Time series',
      title: 'Time series: forecast tomorrow from the shape of yesterday',
      viewUrl: new URL('./time-series/lesson.html', import.meta.url),
      load: () => import('./time-series/lesson.js'),
    },
    {
      id: 'explainability',
      tab: 'Explainability',
      title: 'Explainability: ask a model why',
      viewUrl: new URL('./explainability/lesson.html', import.meta.url),
      load: () => import('./explainability/lesson.js'),
    },
  ],
};
