import argparse
import datetime
from time import strptime

import numpy as np


def _from_matlab_to_python_num_days(x):
    # matlab epoch time start at the year 0 , python at the 1-1-0001. Both
    # formats differ 366 days. So to convert the matlab format to the python
    # format, convert it here
    return (
        datetime.datetime.fromordinal(int(x))
        + datetime.timedelta(days=x % 1)
        - datetime.timedelta(days=366)
    )


_from_matlab_to_python_num_days_vectorized = np.vectorize(
    _from_matlab_to_python_num_days
)


def matlabnum2date(x):
    """Convert a Matlab numerical date/time representation into a Python datetime object

    Parameters
    ----------

    Parameters
    ----------
    x : float or ndarrray

        Matlab numerical date/time representation giving the number of days since 0000 00 00:00:00

    Returns
    -------
    number_of_days : ndarray or scalar of type :class:`datetime`
        Date/time corresponding to the float value `x`

    Notes
    -----
    * In Matlab, the numerical date/time representation gives the number of days since
      Jan 00 0000 00:00:00
    * In Python, the numerical date/time representation gives the number of days since
      Jan 01 0001 00:00:00
    * To convert from matlab to python, 366 days need to be subtracted from the matlab numerical
      date/time representation to get the Python numerical data/time representation
      (note that matlab starts at Jan 0th!)

    Examples
    --------

    The following two dates in numerical representation are obtained using the Matlab `datenum`
    function

    * 2012-12-21T12:12:12 -> 7.352245084722222e+05 # days since 0000 00 00:00:00
    * 1973-11-12T09:15:43 -> 7.209403859143519e+05 # days since 0000 00 00:00:00

    >>> num_date_matlab = np.array([7.352245084722222222222222e+05, 7.209403859143519e+05])

    >>> matlabnum2date(num_date_matlab)
    array([datetime.datetime(2012, 12, 21, 12, 12, 12, 3),
           datetime.datetime(1973, 11, 12, 9, 15, 43, 5)], dtype=object)

    """
    try:
        number_of_days = _from_matlab_to_python_num_days(x)
    except TypeError:
        x = np.asarray(x)
        if not x.size:
            raise
        number_of_days = _from_matlab_to_python_num_days_vectorized(x)

    return number_of_days


def valid_date(s):
    """Check if supplied data *s* is a valid date for the format Year-Month-Day

    Parameters
    ----------
    s : str
        A valid date in the form of YYYY-MM-DD, so first the year, then the month, then the day

    Returns
    -------
    :class:`datetime`
        Date object with with the  year, month, day obtained from the valid string representation

    Raises
    ------
    argparse.ArgumentTypeError:

    Notes
    -----
    This is a helper function for the argument parser module `argparse` which allows you to check
    if the argument passed on the command line is a valid date.

    Examples
    --------

    This is the direct usage of `valid_date` to see if the date supplied is of format YYYY-MM-DD

    >>> try:
    ...     date = valid_date("1973-11-12")
    ... except argparse.ArgumentTypeError:
    ...     print("This date is invalid")
    ... else:
    ...     print("This date is valid")
    This date is valid

    In case an invalid date is supplied

    >>> try:
    ...     date = valid_date("1973-15-12")
    ... except argparse.ArgumentTypeError:
    ...     print("This date is invalid")
    ... else:
    ...     print("This date is valid")
    This date is invalid


    Here it is demonstrated how to add a '--startdate' command line option to the argparse parser
    which checks if a valid date is supplied

    >>> parser = argparse.ArgumentParser()
    >>> p = parser.add_argument("--startdate",
    ...                         help="The Start Date - format YYYY-MM-DD ",
    ...                         required=True,
    ...                         type=valid_date)

    References
    ----------

    https://stackoverflow.com/questions/25470844/specify-format-for-input-arguments-argparse-python
    """

    try:
        return strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.\nSupply date as YYYY-MM-DD".format(s)
        raise argparse.ArgumentTypeError(msg)
