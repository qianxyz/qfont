import numpy as np
import matplotlib.pyplot as plt
from bezier import Bezier
from bspline import BSpline


BLACK = "#000000"
GREY  = "#808080"

UPPER_Q = [
    np.array([
        (.62, .37),
        (.50, .25),
        (.20, .55),
        (.50, .85),
        (.80, .55),
        (.66, .41),
    ]),
    np.array([
        (.62, .37),
        (.62, .39),
        (.54, .49),
    ]),
    np.array([
        (.54, .49),
        (.59, .49),
        (.67, .42),
    ]),
    np.array([
        (.66, .31),
        (.50, .15),
        (.10, .55),
        (.50, .95),
        (.90, .55),
        (.71, .36),
    ]),
    np.array([
        (.66, .31),
        (.70, .25),
        (.80, .25),
    ]),
    np.array([
        (.72, .37),
        (.73, .33),
        (.80, .25),
    ]),
]

LOWER_Q = [
    np.array([
        (.40, .51),
        (.31, .51),
        (.31, .69),
        (.49, .69),
        (.49, .51),
        (.40, .51),
    ]),
    np.array([
        (.50, .48),
        (.40, .43),
        (.29, .47),
        (.22, .60),
        (.28, .72),
        (.40, .78),
        (.55, .70),
        (.58, .60),
        (.53, .20),
        (.57, .16),
        (.60, .17),
        (.62, .40),
    ]),
    np.array([
        (.50, .48),
        (.44, .14),
        (.59, .07),
        (.65, .12),
        (.67, .30),
    ]),
    np.array([
        (.62, .36),
        (.68, .39),
        (.74, .36),
        (.78, .36),
    ]),
    np.array([
        (.67, .30),
        (.68, .34),
        (.78, .36),
    ]),
]


def plot_bspline(ctrl_point_list, clean=False, save=None):
    plt.figure(figsize=(8, 8))
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.axis("off")

    for control_points in ctrl_point_list:
        spline = BSpline(control_points, order=2)
        if not clean:  # plot control points
            plt.plot(*zip(*control_points), 'o-', color=GREY)
        plt.plot(*zip(*spline.curve(100)), color=BLACK, linewidth=5)

    if save is not None:
        plt.savefig(f"images/{save}.png")
    plt.show()



if __name__ == '__main__':
    plot_bspline(UPPER_Q, clean=True)
    # plot_bspline(UPPER_Q, save="upper")
    plot_bspline(LOWER_Q, clean=True)
