# #Particle Filter example
# This code demonstrates a simple particle filter in a two dimensional space. It can come in very handy for situations involving localization under uncertain conditions.
import math
import random


def example():
    """ Example use of particle filter.
    """
    mv = [(0, 0)] * 7
    expected = (0, 0)
    start = (1, 3)
    landmarks = [(13, 14), (11, 46), (47, 4), (34, 17), (26, 25), (23, 55), (10, 10)]

    d = 0.0
    W = 100
    particles = [((1, 1), 0), ((100, 100), 0)]
    x = 0
    while not (converged(particles, 0.001) or x > W):
        # Here is the actual call. It uses some dummy variables and an evaluate function based on distance.
        best, particles = particle_filter(start, mv, evaluate, 1000, landmarks)
        d += distance(best, expected)
        x += 1

    print d / W
    print best
    print x


def generate_particles(point, k, mdist=None):
    """ Generate k random particles around the point (x, y) with an optional max distance (mdist).
    """
    sd = mdist or 0.1
    particles = [point] * k

    def fuzz(x):
        return x + random.uniform(-1 * sd, sd)

    particles = [((fuzz(p[0]), fuzz(p[1])), 0) for p in particles]

    return particles


def normalize(particles):
    """ Transforms weighted particles into a bayesian distribution
            >>> normalize([(1, 2), 50, (3, 4), 20, (3, 3), 30])
            >   [(1, 2), 0.19354838709677416,
                 (3, 3), 0.3225806451612903,
                 (3, 4), 0.4838709677419355]
    """
    weights = []
    for p in particles:
        k = p[0]
        v = p[1]
        weights.append((k, (v) * 0.1))

    total = sum([x[1] for x in weights])  # total amount
    return [(n[0], n[1] / total) for n in weights]  # normalization


def sample(particles):
    """ Given some particles with probability values, pick new ones with replacement based on weights.

            particles = [((x, y), weight), ((x, y), weight)...]
    """
    p = []
    w = [(x[1]) for x in particles]
    N = len(particles)
    index = int(random.random() * N)
    beta = 0.0
    mw = max(w)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        p.append(particles[index])
    return p


def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def particle_filter(init, movement, evaluate, N, landmarks):
    """ Given:

        1. An initial guess at the state
        2. Some movement iterable
        3. A number of particles (N)
        4. an evaluate function that returns a non-normalized weight comparing two states

        ...this function will return the estimated state at the end of the movements.
    """
    def best(points):
        x = sum([p[0] for p in points])
        y = sum([p[1] for p in points])
        return x / len(points), y / len(points)

    position = init
    particles = generate_particles(position, N)
    for m in movement:

        # **Sense step**
        # This will choose new particles based on how they match up with the position state.
        particles = [(p[0], evaluate(p[0], position, landmarks)) for p in particles]
        particles = normalize(particles)
        particles = sample(particles)
        position = best([p[0] for p in particles])

        # **Move step**
        # This shifts everything in the direction of the move.
        particles = [(move(p[0], m), p[1]) for p in particles]
        position = move(position, m)

    return position, particles


def gaussian(mu, sig, x):
    """ Probability of x given the normal distribution of mu and sig (mean and variance)
    """
    try:
        g = math.exp(- ((mu - x) ** 2) / (sig ** 2) / 2.0) / math.sqrt(2.0 * math.pi * (sig ** 2))
    except ZeroDivisionError:
        # To handle cases where x is exactly the mean.
        return 1.0
    return g


def evaluate(a, b, landmarks):
    """ Compare two particles based on distance, returning a prior probability.
    """
    # Noise in this context is kind of the learning variable. It might be wise want to start higher, then gradually anneal.
    noise = 0.1
    prob = 1.0
    for l in landmarks:
        d1 = distance(a, l)
        d2 = distance(b, l)
        prob *= gaussian(d1, noise, d2)
    return prob


def move(init, m):
    return (init[0] + m[0], init[1] + m[1])


def converged(particles, tolerance):
    """ Returns true when converged, based on tolerance """
    if len(particles) <= 1:
        return True
    convg = all([distance(p[0], particles[0][0]) <= tolerance for p in particles])
    return convg
