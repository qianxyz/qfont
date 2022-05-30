import numpy as np
import matplotlib.pyplot as plt


class BSpline:

    def __init__(self, control_points, order=3) -> None:
        """Construct B-spline with given control points and order.

        Args:
            control_points: array of shape (d, 2).
            order: degree of the curve, default to 3 (cubic)
        """
        self.control_points = control_points
        self.order = order
        self.knots = np.concatenate((
            np.zeros(order),
            np.linspace(0, 1, num=len(control_points)-order+1),
            np.ones(order)
        ))

    def _weight(self, i, p, x):
        """Helper for calculating basis function."""
        t = self.knots
        if t[i+p] == t[i]:
            return 0
        return (x - t[i]) / (t[i+p] - t[i])

    def _basis(self, i, p, x):
        """Calculate the basis function B_{i,p}(x).

        Note that `p` is the degree of the curve: e.g.,
        B_{i,0} is a step function, B_{i,1} is piecewise linear, and so on.
        """
        if p == 0:
            return int(self.knots[i] <= x < self.knots[i+1])
        return (self._weight(i, p, x) * self._basis(i, p-1, x)
                + (1 - self._weight(i+1, p, x)) * self._basis(i+1, p-1, x))

    def _point_on_curve(self, x):
        """S(x) = sum(c_i * B_{i,order}(x)) where c_i are control points."""
        return sum(point * self._basis(i, self.order, x)
                   for i, point in enumerate(self.control_points))

    def curve(self, N=50):
        """The actual B-spline.

        Args:
            N: number of points taken equidistantly in [0, 1].

        Returns:
            A list of N points on the curve.
        """
        return [self._point_on_curve(x)
                for x in np.linspace(0, 1, num=N, endpoint=False)]


if __name__ == '__main__':
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.axis("off")

    test_control_points = np.array([
        (0.0, 0.0),
        (0.1, 0.6),
        (0.5, 0.9),
        (0.8, 0.2),
        (1.0, 0.7),
    ])
    bspline = BSpline(test_control_points)
    xys = bspline.curve()

    plt.plot(*zip(*test_control_points), 'o-')
    plt.plot(*zip(*xys))
    plt.savefig("images/bspline.png")
    plt.show()
