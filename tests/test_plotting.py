import matplotlib.pyplot as plt
from numpy import (sin, pi, linspace)
from numpy.testing import (assert_almost_equal)
from pymarine.utils.plotting import sub_plot_axis_to_2d


def test_sub_plot_axis_to_2d():
    def test_plot(nr, nc, x, y):
        # create a plot with nr x nc subplots
        fig, axis = plt.subplots(nrows=nr, ncols=nc)
        # turn the axis into a 2D list
        axis_2d = sub_plot_axis_to_2d(axis, n_rows=nr, n_cols=nc)
        # for all the plots, add a line and check if the plot line data equals the original data
        for i_row in range(nr):
            for j_col in range(nc):
                line, = axis_2d[i_row][j_col].plot(x, y)
                x_line, y_line = line.get_xydata().T
                assert_almost_equal(x_line, x)
                assert_almost_equal(y_line, y)

    x = linspace(0, 2 * pi, num=100)
    y = sin(x)
    # loop over a number of rows and columns ranging in between 1,2,3
    for nr in range(1, 4):
        for nc in range(1, 4):
            test_plot(nr, nc, x, y)
