import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def lerp(t, x0, x1):
    """Linear interpolation."""
    return x0 + t * (x1 - x0)


class Bezier:

    def __init__(self, control_points) -> None:
        """Construct Bezier curve with the control points.

        Args:
            control_points: array of shape (d, 2).
        """
        self.control_points = control_points

    def _de_casteljau(self, t):
        """De Casteljau's algorithm at time t.

        Args:
            t: the time parameter of curve, in interval [0, 1].

        Returns:
            A list of control points xs, where
            xs[0] is the original control points (d points),
            xs[1] is the points generated in the first step (d-1 points),
            .. and xs[d-1] is a list of length 1,
            containing the actual point on the bezier curve.
        """
        xs = [[x for x in self.control_points]]
        last = xs[0]
        while len(last) > 1:
            last = [lerp(t, last[i], last[i+1])
                    for i in range(len(last)-1)]
            xs.append(last)
        return xs

    def curve(self, N=50):
        """The actual Bezier curve.

        Args:
            N: number of points taken equidistantly in [0, 1].

        Returns:
            A list of N points on the curve.
        """
        return [self._de_casteljau(t)[-1][0]
                for t in np.linspace(0, 1, num=N)]

    def _animate(self):
        fig, ax = plt.subplots()
        artists = []
        # line segments
        for _ in self._de_casteljau(t=0):
            line, = ax.plot([], [], 'o-')
            artists.append(line)
        # actual curve
        x_data, y_data = [], []
        curve, = ax.plot([], [], 'k', linewidth=5)
        artists.append(curve)

        def init():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_axis_off()
            # clear the curve data
            x_data.clear()
            y_data.clear()
            return artists

        def update(t):
            xs = self._de_casteljau(t)
            # update lines
            for line, ps in zip(artists, xs):
                line.set_data(*zip(*ps))
            # update curve
            x_data.append(xs[-1][0][0])
            y_data.append(xs[-1][0][1])
            artists[-1].set_data(x_data, y_data)
            return artists

        ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, num=50),
                            init_func=init, blit=True, interval=50)
        # ani.save("images/bezier-animation.gif", fps=30)
        plt.show()


if __name__ == '__main__':
    test_control_points = np.array([
        (0.0, 0.0),
        (0.1, 0.6),
        (0.5, 0.9),
        (0.8, 0.2),
        (1.0, 0.7),
    ])
    bez = Bezier(test_control_points)
    bez._animate()
