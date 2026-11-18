# Browser verification for the lesson migration

Use `npm run dev` with the site mounted at `/manifold/`. The following checks were
performed during the migration on 2026-11-17.

| Check                                                              | Observed result                                                                                       |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Initialize and draw the full catalog in desktop Chrome             | 119 of 119 entries initialized and drew                                                               |
| Galton board → Bell Curve → Galton board                           | The edited rows slider retained its value                                                             |
| Logistic Regression and Softmax Regression                         | Shared dataset controls and the additional Class C control rendered, and prediction inputs worked     |
| Warm MathJax followed by rapid tab changes                         | Bell Curve retained its rendered equations, and the last selected route had no stale transition layer |
| Classification at 390px                                            | The navigation stayed pinned and the 64px fade appeared after scrolling with pointer events disabled  |
| Light and dark mode with synthetic CSV data                        | Simple Linear Regression loaded six rows and kept its upload status and DOM after a round trip        |
| Reduced-motion preference supplied to the app in a browser harness | No initial autoplay and no page-transition snapshot                                                   |

The separate HTML parity test compares all 119 authored entries against the captured
catalog and original DOM, allowing HTML entity encoding for literal less-than signs inside
existing TeX. Browser checks exercise drawing and behavior, and do not establish pixel
equality for every lesson or every browser.
