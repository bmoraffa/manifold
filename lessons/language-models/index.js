/* Manifold. © 2026 Bahman Moraffah. All rights reserved. */
export default {
  id: 'language-models',
  title: 'Language models',
  lessons: [
    {
      id: 'bpe',
      tab: 'Tokenization and BPE',
      title: 'Tokenization and BPE: how text becomes pieces',
      viewUrl: new URL('./bpe/lesson.html', import.meta.url),
      load: () => import('./bpe/lesson.js'),
    },
    {
      id: 'next-token',
      tab: 'Next-token sampling',
      title: 'Next-token sampling: temperature, top-k and top-p',
      viewUrl: new URL('./next-token/lesson.html', import.meta.url),
      load: () => import('./next-token/lesson.js'),
    },
    {
      id: 'rag',
      tab: 'Retrieval-augmented generation',
      title: 'Retrieval-augmented generation: look it up, then answer',
      viewUrl: new URL('./rag/lesson.html', import.meta.url),
      load: () => import('./rag/lesson.js'),
    },
    {
      id: 'lora',
      tab: 'LoRA and fine-tuning',
      title: 'LoRA and fine-tuning: teach an old network a new task cheaply',
      viewUrl: new URL('./lora/lesson.html', import.meta.url),
      load: () => import('./lora/lesson.js'),
    },
    {
      id: 'quantization',
      tab: 'Quantization',
      title: 'Quantization: the same network in fewer bits',
      viewUrl: new URL('./quantization/lesson.html', import.meta.url),
      load: () => import('./quantization/lesson.js'),
    },
    {
      id: 'moe',
      tab: 'Mixture of experts',
      title: 'Mixture of experts: many small specialists and a router',
      viewUrl: new URL('./moe/lesson.html', import.meta.url),
      load: () => import('./moe/lesson.js'),
    },
  ],
};
