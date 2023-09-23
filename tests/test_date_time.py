import datetime
from time import strptime
from numpy import (datetime64, array)
from matplotlib.dates import (date2num)
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from hmc_utils.date_time import (matlabnum2date, _datetime2matlabdn, datenum2matlabnumdn,
                                 valid_date)
from argparse import ArgumentTypeError


def test_matlabnum2date():
    # test the scalar version of the matlab num 2 date function
    date_time = datetime64("2012-12-21T12:12:12")

    # the date_time above can be converted to a number using the 'datenum' function in matlab. It returns
    num_date_matlab = 7.352245084722222222222222e+05

    num_date_python = matlabnum2date(num_date_matlab)  # number of hours since 1 Jan 0001

    # convert number of hours to datetime object with seconds precision
    date_time_conv = array([num_date_python]).astype("datetime64[s]")[0]

    assert_equal(date_time, date_time_conv)


def test_matlabnum2date_vector():
    # test the vectorized version of the matlab num 2 date function
    date_time_array = array([datetime64("2012-12-21T12:12:12"), datetime64("1973-11-12T09:15:43")])

    # the date_time above can be converted to a number using the 'datenum' function in matlab. It returns
    num_date_matlab = array([7.352245084722222222222222e+05, 7.209403859143519e+05])

    num_date_python = matlabnum2date(num_date_matlab)

    # convert number of hours to datetime object with seconds precision
    date_time_array_conv = num_date_python.astype("datetime64[s]")

    assert_equal(date_time_array, date_time_array_conv)


def test_datenum2matlabdn():
    date_time = datetime.datetime(2012, 12, 21, 12, 12, 12)
    num_date_matlab = 7.352245084722222222222222e+05

    # the date_time above can be converted to a number using the 'datenum' function in matlab. It returns
    num_date_python = date2num(date_time)
    num_date_matlab_conv = datenum2matlabnumdn(num_date_python)
    assert_equal([num_date_matlab], [num_date_matlab_conv])


def test_datetime2matlabdn():
    ref_date_time = datetime.datetime(2012, 12, 21, 12, 12, 12)
    # this value can be obtained with Matlab using the function date2num
    num_date_matlab = 7.352245084722222222222222e+05
    num_date_matlab_conv = _datetime2matlabdn(ref_date_time)
    assert_almost_equal([num_date_matlab], [num_date_matlab_conv])


def test_valid_date():
    date_string = "1973-11-12"
    date = strptime(date_string, "%Y-%m-%d")
    date_2 = valid_date(date_string)
    assert_equal(date, date_2)
    # check if an argument type error is raised when a non-valid date is passed
    assert_raises(ArgumentTypeError, valid_date, "19731112")


def main():
    test_matlabnum2date()
    test_matlabnum2date_vector()
    test_datetime2matlabdn()
    test_datetime2matlabdn()


if __name__ == "__main__":
    main()
