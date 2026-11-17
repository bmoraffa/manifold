# Manifold design

Manifold is a quiet, editorial place to learn machine learning. The canvas, explanation,
and equations describe the same idea. Space, type, and fine rules give the lesson structure.

The visual source of truth is `css/style.css`. It sets light and dark colors through
`--bg`, `--ink`, `--muted`, and the existing accent variables. Cormorant Garamond identifies
titles, Hanken Grotesk carries the lesson prose, and JetBrains Mono presents numerical readouts.
Shared native controls use the same classes and label associations across lessons.

Lesson navigation stays pinned during document scrolling. A 64px gradient matches the theme
background and appears once scrolled. It occupies no layout space and accepts no pointer
input. Keep keyboard focus below both the navigation and the gradient.

Later lessons in a topic enter from the right and earlier lessons enter from the left.
The shared transition uses a 30px shift and a 280ms fade. Topic, home, and About changes
use a 220ms fade. The outgoing visual fades over 180ms. Navigation remains stationary.
Reduced-motion preferences switch pages immediately and suppress lesson autoplay.
