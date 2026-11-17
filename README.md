# Manifold

**Machine learning you can see move.**

**[Open the site → bmoraffa.github.io/manifold](https://bmoraffa.github.io/manifold/)**

Manifold is a free, interactive site for learning machine learning by handling it. Pick an algorithm, change
one thing, and watch what happens. Every lesson pairs a live animation with a plain-language explanation and
the full mathematics, so the picture, the intuition and the derivation sit side by side.

It was built for students who can follow a derivation on the board and still cannot say what the algorithm
will do when the data change.

## What every lesson gives you

- **A picture that moves.** The algorithm runs in front of you, one step at a time or all at once.
- **Controls that matter.** Sliders, presets and your own data, with the effect visible immediately.
- **The intuition first.** A short explanation in everyday terms before any formula.
- **The steps spelled out.** What the method does in order, and what just happened on screen.
- **The mathematics in full.** Typeset derivations under the animation, not instead of it.
- **A question to check yourself.** With a worked answer.

## What it covers

115 lessons across 12 topics:

|                 |                        |                       |
| --------------- | ---------------------- | --------------------- |
| The Gaussian    | Foundations            | Regression            |
| Classification  | Ensembles              | Unsupervised learning |
| Neural networks | Deep learning          | Generative AI         |
| Language models | Reinforcement learning | Probabilistic methods |

The range runs from the bell curve and least squares to diffusion models, transformers, Q-learning and
Hamiltonian Monte Carlo.

## Using it

Open the site in any modern browser. There is nothing to install and no account to create. Everything runs
on your own device, and nothing you do or upload is sent anywhere.

Teachers are welcome to use it in lectures, assignments and office hours. Please cite it when you do.

## How to cite

> Moraffah, B. (2026). _Manifold: Machine Learning You Can See Move_ [Interactive website].
> https://bmoraffa.github.io/manifold/

```bibtex
@misc{moraffah2026manifold,
  author       = {Moraffah, Bahman},
  title        = {Manifold: Machine Learning You Can See Move},
  year         = {2026},
  howpublished = {\url{https://bmoraffa.github.io/manifold/}},
  note         = {Interactive website}
}
```

BibLaTeX, APA, IEEE, Chicago and MLA versions are on the site's
[How to Cite](https://bmoraffa.github.io/manifold/#cite) page. GitHub's **Cite this repository** button uses
the same details from `CITATION.cff`.

## Acknowledgements

The idea came out of conversations about teaching machine learning with Professor Ethan Prihar at Worcester
Polytechnic Institute. Thanks also to Professor Raha Moraffah and Professor Dachun Sun, both at Worcester
Polytechnic Institute, for discussion and feedback throughout. The full note is on the site's
[Acknowledgements](https://bmoraffa.github.io/manifold/#acknowledgements) page.

## For maintainers

The site is plain HTML, CSS and JavaScript. There is no build step and no server code.

| Path                                     | Contents                                                    |
| ---------------------------------------- | ----------------------------------------------------------- |
| `index.html`                             | The page shell, and the scripts in the order they must load |
| `css/style.css`                          | All styling; light and dark themes follow the device        |
| `js/core.js`                             | Shared helpers: drawing, controls, animation, small maths   |
| `js/gaussian.js` … `js/probabilistic.js` | One file per topic                                          |
| `js/extra-lessons.js`                    | Lessons added later, appended to their topics               |
| `js/about.js`                            | References, Suggested Books, Acknowledgements, How to Cite  |
| `js/app.js`                              | Topic list, home-page animation, menu and routing           |
| `CITATION.cff`                           | Citation details for GitHub                                 |

To preview a change, open `index.html` in a browser. To publish it, commit to `main`; GitHub Pages updates
the live site within a minute or two.

The only outside resources are MathJax, for typesetting formulas, and three typefaces from Google Fonts.

### Formatting and linting

Development tooling requires Node.js 22.22 or newer on the 22.x release line, or Node.js 24.8 or newer.
Install the locked development dependencies with `npm ci`. The website itself still runs directly in
the browser without installing dependencies or running a build.

| Command                | Purpose                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `npm run format`       | Format supported project files with Prettier                   |
| `npm run format:check` | Check formatting without changing files                        |
| `npm run lint:js`      | Check JavaScript with ESLint's recommended rules               |
| `npm run lint:css`     | Check CSS with Stylelint's recommended rules                   |
| `npm run lint:html`    | Validate the HTML shell with HTML Validate's recommended rules |
| `npm run lint`         | Run the three linters, stopping at the first failure           |
| `npm run check`        | Check formatting, then run the linters                         |

Prettier owns formatting: two-space indentation, semicolons, single quotes in JavaScript, trailing
commas, a 100-column target, and LF line endings. `.editorconfig` and `.gitattributes` keep editor and
Git line endings consistent. Embedded-language formatting is disabled to preserve lesson strings,
and HTML whitespace uses strict sensitivity. HTML Validate's Prettier preset disables conflicting
formatting rules; CSS linting checks correctness rather than duplicating Prettier's style rules.

ESLint treats `js/` files as classic browser scripts, not modules. `eslint.config.mjs` records which
file owns each binding used by other scripts; update that map when adding a shared helper. Shared
bindings are read-only in consuming files. Unused local variables and parameters are checked, while
top-level bindings and caught exception parameters are allowed to be unused.

The initial formatting pass preserves existing code issues: ESLint reports 86 errors (76 unused
locals or parameters, six precision-loss warnings treated as errors, three empty blocks, and one
constant condition), and Stylelint reports seven descending-specificity errors. HTML validation
passes. Consequently `npm run check` currently fails on the existing lint debt. Fix those issues in
a separate behavior-reviewed change; `npm run format` never runs linter fixes or reorders CSS rules.

## Copyright

© 2026 Bahman Moraffah. All rights reserved.

You are welcome to use the site for teaching and study. Please reference it when you do. For any other use,
including copying or redistributing the code or content, ask first.
