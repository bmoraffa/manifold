/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
export default {
  id: 'reinforcement',
  t: 'Reinforcement learning',
  lessons: [
    {
      id: 'bandits',
      tab: 'Multi-armed bandits',
      title: 'Multi-armed bandits: try new things, or stick with what works?',
      viewUrl: new URL('./bandits/lesson.html', import.meta.url),
      load: () => import('./bandits/lesson.js'),
    },
    {
      id: 'value-iteration',
      tab: 'Value and policy iteration',
      title: 'Value and policy iteration: planning when you know the rules',
      viewUrl: new URL('./value-iteration/lesson.html', import.meta.url),
      load: () => import('./value-iteration/lesson.js'),
    },
    {
      id: 'td-learning',
      tab: 'Monte Carlo and TD learning',
      title: 'Monte Carlo and TD learning: wait for the end, or learn as you go?',
      viewUrl: new URL('./td-learning/lesson.html', import.meta.url),
      load: () => import('./td-learning/lesson.js'),
    },
    {
      id: 'q-learning',
      tab: 'Q-learning',
      title: 'Q-learning: fill in the table, one step at a time',
      viewUrl: new URL('./q-learning/lesson.html', import.meta.url),
      load: () => import('./q-learning/lesson.js'),
    },
    {
      id: 'sarsa',
      tab: 'SARSA and the cliff',
      title: 'SARSA and Q-learning: two robots on a cliff edge',
      viewUrl: new URL('./sarsa/lesson.html', import.meta.url),
      load: () => import('./sarsa/lesson.js'),
    },
    {
      id: 'policy-gradient',
      tab: 'Policy gradient',
      title: 'Policy gradient: adjust the habits directly',
      viewUrl: new URL('./policy-gradient/lesson.html', import.meta.url),
      load: () => import('./policy-gradient/lesson.js'),
    },
    {
      id: 'dqn',
      tab: 'Actor-critic and DQN',
      title: 'Actor-critic and DQN: when the table would never fit',
      viewUrl: new URL('./dqn/lesson.html', import.meta.url),
      load: () => import('./dqn/lesson.js'),
    },
  ],
};
