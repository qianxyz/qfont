import numpy as np
import matplotlib.pyplot as plt


class BSpline:

    def __init__(self, control_points, order) -> None:
        self.control_points = control_points
        self.order = order
        self.knots = np.concatenate((
            np.zeros(order),
            np.linspace(0, 1, num=len(control_points)-order+1),
            np.ones(order)
        ))

    def _weight(self, i, p, x):
        t = self.knots
        if t[i+p] == t[i]:
            return 0
        return (x - t[i]) / (t[i+p] - t[i])

    def _basis(self, i, p, x):
        if p == 0:
            return int(self.knots[i] <= x < self.knots[i+1])
        return (self._weight(i, p, x) * self._basis(i, p-1, x)
                + (1 - self._weight(i+1, p, x)) * self._basis(i+1, p-1, x))

    def _point_on_curve(self, x):
        return sum(point * self._basis(i, self.order, x)
                   for i, point in enumerate(self.control_points))

    def curve(self, N=50):
        return [self._point_on_curve(x)
                for x in np.linspace(0, 1, num=N, endpoint=False)]


if __name__ == '__main__':
    test_control_points = np.array([
        (0.0, 0.0),
        (0.1, 0.6),
        (0.5, 0.9),
        (0.8, 0.2),
        (1.0, 0.7),
    ])
    bspline = BSpline(test_control_points, order=3)
    xys = bspline.curve()
    plt.plot(*zip(*test_control_points), 'o-')
    plt.plot(*zip(*xys))
    plt.show()
