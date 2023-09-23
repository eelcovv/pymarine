import os
import string

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from scipy.interpolate import interp1d
import netCDF4 as nc

from pymarine.utils.numerical import (
    get_nearest_index,
    find_idx_nearest_val,
    ecdf2percentile,
    get_column_with_max_cumulative_value,
    get_range_from_string,
    extrap1d,
    loadmat,
    print_mat_nested,
)

DATA_DIR = "data"
MATLAB_DATAFILE = "RAO_7.mat"


def test_get_nearest_index():
    data = np.linspace(0, 10, 27)
    index = get_nearest_index(data, 3)
    index_expected = 7

    assert_equal(index, index_expected)

    # check if an assertion error is raised
    data_array = np.array([3, 0, 2, 4, 2, 1])
    with assert_raises(AssertionError):
        get_nearest_index(data_array, value=2.1)


def test_find_idx_nearest_val():
    data = np.linspace(0, 10, 27)
    index = find_idx_nearest_val(data, 3)
    index_expected = 8

    assert_equal(index, index_expected)

    data_array = np.array([3, 0, 2, 4, 2, 1])
    a = find_idx_nearest_val(data_array, value=2.1)
    assert_equal(a, 4)

    data_array = np.array([3, 0, 2, 4, 2, 1, 2])
    a = find_idx_nearest_val(data_array, value=2.1)
    assert_equal(a, 6)

    data_array = np.array([3, 0, 2, 4, 2.11, 1, 2])
    a = find_idx_nearest_val(data_array, value=2.1)
    assert_equal(a, 4)


def test_ecdf2percentile():
    np.random.seed(0)
    number_of_observations = 100
    # generate random data variing in between 0 and 100
    x_data = 100 * np.random.rand(number_of_observations)
    # calculate the cumulative distribution function of this random data using statsmodel
    e_cdf = sm.distributions.empirical_distribution.ECDF(x_data)

    result = [
        ecdf2percentile(ecdf=e_cdf, percentile=x)
        for x in np.linspace(0, 1, 10, endpoint=False)
    ]
    result_expected = np.array(
        [
            0.46954761925470656,
            9.6098407893963067,
            14.335328740904639,
            26.538949093944542,
            38.344151882577769,
            46.865120164770161,
            57.594649555617927,
            65.632958946527339,
            77.423368943421664,
            89.177300078207978,
        ]
    )

    assert_almost_equal(result, result_expected)


def test_get_column_with_max_cumulative_value():
    np.random.seed(0)
    n_cols = 5
    n_rows = 10
    # create a 10 x 5 data frame with random values with columns named as A, B, C, etc
    data_frame = pd.DataFrame(
        np.random.random_sample((n_rows, n_cols)),
        columns=list(string.ascii_uppercase)[:n_cols],
    )
    # obtain the name of the column with the maximum cumulative value
    col1 = get_column_with_max_cumulative_value(data_frame)
    col2 = get_column_with_max_cumulative_value(data_frame, regular_expression="[ABC]")

    assert_equal(col1, "D")
    assert_equal(col2, "C")


def test_get_range_from_string():
    r1 = get_range_from_string("0:10:2")
    r1_expected = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert_almost_equal(r1, r1_expected)

    r2 = get_range_from_string("0:7")
    r2_expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert_almost_equal(r2, r2_expected)

    r3 = get_range_from_string("3:4:0.2")
    r3_expected = np.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
    assert_almost_equal(r3, r3_expected)


def test_extrap1d():
    # create
    xp = np.linspace(0, 2 * np.pi, 20)
    yp = np.sin(xp)
    f_inter = interp1d(xp, yp)

    f_extra = extrap1d(f_inter)

    xp_new = np.linspace(-0.1 * np.pi, 2.1 * np.pi, 20)
    yp_new = f_extra(xp_new)

    yp_exp = np.array(
        [
            -0.3084645,
            0.0487049,
            0.3970778,
            0.6922465,
            0.8966717,
            0.9843516,
            0.9440572,
            0.7807363,
            0.5149139,
            0.1801622,
            -0.1801622,
            -0.5149139,
            -0.7807363,
            -0.9440572,
            -0.9843516,
            -0.8966717,
            -0.6922465,
            -0.3970778,
            -0.0487049,
            0.3084645,
        ]
    )

    debug_plot = False
    if debug_plot:
        import matplotlib.pyplot as plt

        plt.plot(xp_new, yp_new, "-o")
        plt.plot(xp, yp, "x")
        plt.show()

    assert_almost_equal(yp_new, yp_exp)


def test_load_matlab():
    # construct the matlab and netcdf data file name
    file_name = os.path.join(DATA_DIR, MATLAB_DATAFILE)
    if not os.path.exists(file_name):
        file_name = os.path.join("..", file_name)
    file_name_nc = os.path.splitext(file_name)[0] + ".nc"
    # read the matlab data and make a reference to the RAO data, which is a 24 x 250 x 6 array
    data_ml = loadmat(filename=file_name)
    data = data_ml["RAO"]

    # read the netcdf data
    data_nc = nc.Dataset(file_name_nc)

    # loop over the  6 DOF components
    variable_base = "TowO_ACC"
    for i_rao, dof_name in enumerate(["AX", "AY", "AZ", "RXX", "RYY", "RZZ"]):
        # the matlab file contains complex rao component at the indices 0..5
        rao_2d = data[:, :, i_rao]

        # convert the complex values in 2 2D arrays with the magnitude and phase. Transpose as well
        rao_abs = abs(rao_2d).T
        rao_phase = np.angle(rao_2d).T

        # the net cdf file contains the separate magnitude and phase of the complex stored as named arrays
        rao_abs_nc = data_nc.variables["_".join([variable_base, dof_name, "abs"])]
        rao_phase_nc = data_nc.variables["_".join([variable_base, dof_name, "phase"])]

        # check if the 2D arrays are equal
        assert_almost_equal(rao_abs, rao_abs_nc)
        assert_almost_equal(rao_phase, rao_phase_nc)

    print_mat_nested(data)
    print(data_nc.description)
    print(data_nc.description)
