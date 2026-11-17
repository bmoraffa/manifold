/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
export default {
  id: 'neural-nets',
  t: 'Neural networks',
  lessons: [
    {
      id: 'mlp',
      tab: 'Multilayer perceptron',
      title: 'Watch a network learn a shape',
      viewUrl: new URL('./mlp/lesson.html', import.meta.url),
      load: () => import('./mlp/lesson.js'),
    },
    {
      id: 'backprop',
      tab: 'Backpropagation',
      title: 'Backpropagation: follow one example through, and back',
      viewUrl: new URL('./backprop/lesson.html', import.meta.url),
      load: () => import('./backprop/lesson.js'),
    },
    {
      id: 'activations',
      tab: 'Activation functions',
      title: 'The bend in every neuron',
      viewUrl: new URL('./activations/lesson.html', import.meta.url),
      load: () => import('./activations/lesson.js'),
    },
    {
      id: 'init',
      tab: 'Weight initialization',
      title: 'Where to start: the first weights decide whether it can learn',
      viewUrl: new URL('./init/lesson.html', import.meta.url),
      load: () => import('./init/lesson.js'),
    },
    {
      id: 'dropout',
      tab: 'Dropout',
      title: 'Dropout: train with half the team missing',
      viewUrl: new URL('./dropout/lesson.html', import.meta.url),
      load: () => import('./dropout/lesson.js'),
    },
    {
      id: 'mcdropout',
      tab: 'Bayesian dropout',
      title: 'Bayesian dropout: a network that knows when it is guessing',
      viewUrl: new URL('./mcdropout/lesson.html', import.meta.url),
      load: () => import('./mcdropout/lesson.js'),
    },
    {
      id: 'universal',
      tab: 'Universal approximation',
      title: 'Enough neurons can draw any curve',
      viewUrl: new URL('./universal/lesson.html', import.meta.url),
      load: () => import('./universal/lesson.js'),
    },
    {
      id: 'batch-norm',
      tab: 'Batch normalization',
      title: 'Batch normalization: keep every layer’s inputs in a sensible range',
      viewUrl: new URL('./batch-norm/lesson.html', import.meta.url),
      load: () => import('./batch-norm/lesson.js'),
    },
  ],
};
