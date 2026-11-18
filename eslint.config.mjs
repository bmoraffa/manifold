import js from '@eslint/js';
import globals from 'globals';

export default [
  { ignores: ['node_modules/**', '.codegraph/**', '.superpowers/**', 'coverage/**', 'dist/**'] },
  js.configs.recommended,
  { files: ['tests/**/*.mjs', 'scripts/**/*.mjs'], languageOptions: { globals: globals.node } },
  {
    files: ['js/**/*.js', 'lessons/**/*.js', 'pages/**/*.js'],
    languageOptions: { ecmaVersion: 2022, sourceType: 'module', globals: globals.browser },
    rules: { 'no-unused-vars': ['error', { args: 'after-used', caughtErrors: 'none' }] },
  },
];
