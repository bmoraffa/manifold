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

| | | |
| --- | --- | --- |
| The Gaussian | Foundations | Regression |
| Classification | Ensembles | Unsupervised learning |
| Neural networks | Deep learning | Generative AI |
| Language models | Reinforcement learning | Probabilistic methods |

The range runs from the bell curve and least squares to diffusion models, transformers, Q-learning and
Hamiltonian Monte Carlo.

## Using it

Open the site in any modern browser. There is nothing to install and no account to create. Everything runs
on your own device, and nothing you do or upload is sent anywhere.

Teachers are welcome to use it in lectures, assignments and office hours. Please cite it when you do.

## How to cite

> Moraffah, B. (2026). *Manifold: Machine Learning You Can See Move* [Interactive website].
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

| Path | Contents |
| --- | --- |
| `index.html` | The page shell, and the scripts in the order they must load |
| `css/style.css` | All styling; light and dark themes follow the device |
| `js/core.js` | Shared helpers: drawing, controls, animation, small maths |
| `js/gaussian.js` … `js/probabilistic.js` | One file per topic |
| `js/extra-lessons.js` | Lessons added later, appended to their topics |
| `js/about.js` | References, Suggested Books, Acknowledgements, How to Cite |
| `js/app.js` | Topic list, home-page animation, menu and routing |
| `CITATION.cff` | Citation details for GitHub |

To preview a change, open `index.html` in a browser. To publish it, commit to `main`; GitHub Pages updates
the live site within a minute or two.

The only outside resources are MathJax, for typesetting formulas, and three typefaces from Google Fonts.

## Copyright

© 2026 Bahman Moraffah. All rights reserved.

You are welcome to use the site for teaching and study. Please reference it when you do. For any other use,
including copying or redistributing the code or content, ask first.
