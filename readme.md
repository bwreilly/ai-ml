##  What

Basic implementations of some machine learning/ai algorithms.

##  Why

Reference or examples/templates for other projects.

### Kalman Filter

[Annotated Source](http://bwreilly.github.com/ai-ml/docs/kalman.html)

A [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) attempts to determine state from noisy input data, typically paired with positional shifting. In this implementation, the predict step offers a method of convolution (state becomes less certain) typical of movement and measurement (in which the state is likely to become more certain). Most common [applications](https://en.wikipedia.org/wiki/Kalman_filter#Applications) are in reference, navigation, and guidance.

### Particle Filter

[Annotated Source](http://bwreilly.github.com/ai-ml/docs/pfilter.html)

A [particle filter](https://en.wikipedia.org/wiki/Particle_filter) also does state estimation, but has several advantages over Kalman filters and other methods. A pfilter can be used in continuous space and on multimodal distributions (Kalman filters only work on unimodal distributions). Implementation is also fairly easy. However, efficiency can suffer significantly in large state spaces. [Here](http://www.youtube.com/watch?v=4S-sx5_cmLU&feature=youtu.be&t=1m24s) is a good visualization of the process.

##  Credit

Some of the implementations have been derived from work done in the [Stanford Introduction to Artificial Intelligence](https://www.ai-class.com/) class and the [Udacity course CS373: Programming a Robot Car](http://www.udacity.com/overview/Course/cs373/), modified for clarity and reusability. I am deeply indebted to the efforts of those involved in the making of those classes (Sebastian Thrun, Peter Norvig, and the staffs at ai-class.com and udacity.com)