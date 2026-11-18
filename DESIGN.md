# Manifold design

Manifold is a quiet, editorial place to learn machine learning. The canvas, explanation,
and equations describe the same idea. Space, type, and fine rules give the lesson structure.

The visual source of truth is `css/style.css`. It sets light and dark colors through
`--bg`, `--ink`, `--muted`, and the existing accent variables. Cormorant Garamond identifies
titles, Hanken Grotesk carries the lesson prose, and JetBrains Mono presents numerical readouts.
Shared native controls use the same classes and label associations across lessons.
The theme switch sits beside the Manifold wordmark on home and after the topic name in
lessons. The device preference applies until the learner chooses a theme. Switching themes
crossfades the page and turns the sun and moon icons when reduced motion is off.

Lesson navigation stays pinned during document scrolling. A 64px gradient matches the theme
background and appears once scrolled. It occupies no layout space and accepts no pointer
input. Keep keyboard focus below both the navigation and the gradient.

Later lessons in a topic enter from the right and earlier lessons enter from the left.
The shared transition uses a 30px shift and a 280ms fade. Topic, home, and About changes
use a 220ms fade. The outgoing visual fades over 180ms. Navigation remains stationary.
Reduced-motion preferences switch pages immediately and suppress lesson autoplay.

The round Contact Us button floats at the bottom-right with clearance for the home footer.
It is hidden on the Contact Us page. Its solid fill and envelope icon use the theme colors.
Hover deepens the shadow and lifts the button when reduced motion is off. An aurora ring
rotates and a few particles drift outward during hover or keyboard focus. The native
dialog fades and slides up when it opens and closes. The backdrop gradually dims and blurs
as the dialog opens and clears as it closes. Reduced motion removes those animations.
Both contact views lead with a highlighted feedback invitation marked by a lightbulb.
They share the Web3Forms form and underlined email link.
