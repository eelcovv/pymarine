"""
Some numerical utilities used in other hmc modules
"""
import re

import numpy as np
import scipy.io as spio


def ecdf2percentile(ecdf, percentile):
    """Calculate a percentile of an Empirical CDF function as returned by the `statsmodels` package

    Parameters
    ----------
    ecdf : :obj:`statsmodel.distributions.empirical_distribution.ECDF`
        Empirical CDF as a step function.
    percentile : float
        Percentile of the distribution

    Returns
    -------
    float:
       value where a fraction  of `percentile` will be lower

    Examples
    --------

    >>> import statsmodels.api as sm
    >>> np.random.seed(0)
    >>> number_of_observations = 100
    >>> # generate random data variing in between 0 and 100
    >>> x_data = 100 * np.random.rand(number_of_observations)
    >>> # calculate the cumulative distribution function of this random data using statsmodel
    >>> e_cdf = sm.distributions.empirical_distribution.ECDF(x_data)
    >>> # print some P values belonging to the percentiles
    >>> print("percentile {:2.1f} : ecdf ={:6.2f}".format(0.1, ecdf2percentile(ecdf=e_cdf, percentile=0.1)))
    percentile 0.1 : ecdf =  9.61
    >>> print("percentile {:2.1f} : ecdf ={:6.2f}".format(0.5, ecdf2percentile(ecdf=e_cdf, percentile=0.5)))
    percentile 0.5 : ecdf = 46.87
    >>> print("percentile {:2.1f} : ecdf ={:6.2f}".format(0.9, ecdf2percentile(ecdf=e_cdf, percentile=0.9)))
    percentile 0.9 : ecdf = 89.18
    """

    for i in range(ecdf.n):
        if ecdf.y[i] > percentile:
            return ecdf.x[i]


def get_parameter_list_key(parlist):
    """Utility for the qtgraphs Parameter Item

    Parameters
    ----------
    parlist :
        the parameter tree list (is an ordered list) contain all the values and current one

    Returns
    -------
    str
        The name belonging to the current value

    Notes
    -----
    The parameter tree widget has a field 'list' in which a list of values is given with
    corresponding integers. The current integer belonging to the parlist is obtained by
    parlist.value() however, to get the associated value of the key field is less straightforward
    In this routine it is retrieved
    """

    # the current value of the parlist
    value = parlist.value()

    # get the reverse list from the parlist
    reverselist = parlist.reverse

    # get the index belonging to the current value
    index = reverselist[0].index(value)

    # get the name of the key of this index
    keyname = reverselist[1][index]

    return keyname


def get_column_with_max_cumulative_value(data, regular_expression='.*'):
    """Find the column of a pandas DataFrame with the maximum cumulative value

    Parameters
    ----------
    data : DataFrame
        Data frame with the columns
    regular_expression : str
        Regular expression used to make a selection of columns to include. Default to '.*', which
        means that all columns are included

    Returns
    -------
    str or None:
        The name of the column with the maximum cumulative value or None if no columns were found

    Notes
    -----
    * Only the columns with a name obeying the regular expression are taken into account

    * An example of usage can be found in the fatigue monitoring software where we have data frames
      with damage over all the channels at a hot spots. If you want to obtained the channel with
      the maximum cumulative damage you can use this function

    Examples
    --------

    >>> import string
    >>> import pandas as pd
    >>> np.random.seed(0)
    >>> n_cols = 5
    >>> n_rows = 10

    Create a 10 x 5 data frame with random values with columns named as A, B, C, etc

    >>> data_frame = pd.DataFrame(np.random.random_sample((n_rows, n_cols)),
    ...                           columns=list(string.ascii_uppercase)[:n_cols])

    Obtain the name of the column with the maximum cumulative value

    >>> get_column_with_max_cumulative_value(data_frame)
    'D'

    Obtain the name of the column with the maximum cumulative value only including colums A, B and C

    >>> get_column_with_max_cumulative_value(data_frame, regular_expression="[ABC]")
    'C'

    """

    columns = list()
    for col in data.columns.values:
        if re.match(regular_expression, col):
            columns.append(col)
    if columns:
        name_of_maximum_column = data.fillna(0).sum()[columns[0]:columns[-1]].idxmax()
    else:
        name_of_maximum_column = None

    return name_of_maximum_column


def nans(shape, dtype=float):
    """Create an array filled with :class:`numpy.nan` values

    Parameters
    ----------
    shape : tuple
        Shape of the array of nans to create

    dtype : `numpy.dtype`, optional, default=float
        Type of the nan

    Returns
    -------
    ndarray
        Array filled with `numpy.nan` values

    Examples
    --------
    >>> nans((2, 3))
    array([[ nan,  nan,  nan],
           [ nan,  nan,  nan]])

    """
    # create an array of nans with give shape
    arr = np.empty(shape, dtype)
    arr.fill(np.nan)
    return arr


def get_nearest_index(data, value):
    """
    Find the index of the first occurrence of a value in a array with monotonically increasing
    values

    Parameters
    ----------
    data : array_like
        Array with monotonically increasing numbers
    value :
        Value of the point to give the index

    Returns
    -------
    int
        Index of floor cell where the values in the array first exceed the value

    Example
    -------
    >>> data_array = np.linspace(0, 10, 8, endpoint=False)
    >>> print(data_array)
    [ 0.    1.25  2.5   3.75  5.    6.25  7.5   8.75]
    >>> get_nearest_index(data_array, value=3)
    2
    >>> get_nearest_index(data_array, value=3.5)
    2
    >>> get_nearest_index(data_array, value=8.5)
    6
    >>> get_nearest_index(data_array, value=10)
    7
    >>> get_nearest_index(data_array, value=-1)
    0

    In case of a non-increasing array an AssertionError is raised

    >>> data_array = np.array([3, 0, 2, 4, 2.11, 1])
    >>> try:
    ...   get_nearest_index(data_array, value=2.1)
    ... except AssertionError as err:
    ...   print("An assertion error was raised")
    An assertion error was raised

    Raises
    ------
    AssertionError
        In case the array does not have monotonically increasing values

    Notes
    -----
    * Only arrays with monotonically increasing values are allowed.
    * In case the nearest index of an arbitrary array is needed, `find_inx_nearest_val`  should be
      used
    * In case the `value` a larger than any value in the `data` array, the maximum index is return,
      while in case the `value` is smaller than any value in the `data` array, a zero is return

    See Also
    --------
    find_idx_nearest_val: a function to get the index of the nearest value for an arbitraray array



    """
    data_array = np.asanyarray(data)
    if (np.diff(data_array) < 0).any():
        raise AssertionError("Only monotonic increasing arrays are allowed. "
                             "Use `find_inx_nearest_val` instead ")

    sgn = np.sign(np.asarray(data) - value)
    try:
        nearest_index = np.where(np.diff(sgn) > 0)[0][0]
    except IndexError:
        if sgn[0] > 0:
            nearest_index = 0
        else:
            nearest_index = data.size - 1

    return nearest_index


def find_idx_nearest_val(array, value):
    """ Find the nearest index of a value in a array.

    Parameters
    ----------
    array : array_like
        an array with values
    value : float
        the value for which we want the nearest index

    Returns
    -------
    int
        The index of the nearest value to 'value' in array.
          the number of items in the array

    Examples
    --------
    >>> data_array = np.linspace(0, 10, 8, endpoint=False)
    >>> print(data_array)
    [ 0.    1.25  2.5   3.75  5.    6.25  7.5   8.75]
    >>> find_idx_nearest_val(data_array, value=3)
    2
    >>> find_idx_nearest_val(data_array, value=3.5)
    3
    >>> find_idx_nearest_val(data_array, value=8.5)
    7
    >>> find_idx_nearest_val(data_array, value=10)
    7
    >>> find_idx_nearest_val(data_array, value=-1)
    0

    In case of a non-increasing array with equal candidates, the last one is returned

    >>> data_array = np.array([3, 0, 2, 4, 2, 1])
    >>> find_idx_nearest_val(data_array, value=2.1)
    4
    >>> data_array = np.array([3, 0, 2, 4, 2, 1, 2])
    >>> find_idx_nearest_val(data_array, value=2.1)
    6

    If we change the first candidate so it becomes the nearest, this one will be returned

    >>> data_array = np.array([3, 0, 2.09, 4, 2, 1])
    >>> find_idx_nearest_val(data_array, value=2.1)
    2

    Notes
    -----
    * In case that 2 or more items exist with the same distance from `value`, the *last* occurrence
      is returned
    * In case the value is outside the range of any value inside the array, either 0 or N-1 is
      return, with N the number of array elements of the input array

    See Also
    --------
    get_nearest_index:
        This function returns the first occurrence and only works for monotonically increasing
        arrays
    """
    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, [value], side="left")
    try:
        # in case search sorted returns a list of values take the first one
        idx = idx[0]
    except IndexError:
        # it was not a list, so do't do anything
        pass

    if idx >= len(array):
        idx_nearest = idx_sorted[len(array) - 1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    else:
        if abs(value - sorted_array[idx - 1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx - 1]
        else:
            idx_nearest = idx_sorted[idx]

    return idx_nearest


def get_range_from_string(range_string):
    """Analyse a range string to get the start, end and step and return a numpy array

    Parameters
    ----------
    range_string: str, start:end:[step]
        String representing the start, end and step size of a range

    Returns
    -------
    ndarray:
        1-D Array with points defined by range_string

    Examples
    --------
    >>> get_range_from_string("0:10:2")
    array([  0.,   2.,   4.,   6.,   8.,  10.])

    >>> get_range_from_string("0:7")
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])

    >>> get_range_from_string("3:4:0.2")
    array([ 3. ,  3.2,  3.4,  3.6,  3.8,  4. ])
    """
    values = range_string.split(':')
    try:
        start = values[0]
        stop = values[1]
        try:
            spacing = values[2]
        except IndexError:
            spacing = 1.0
    except IndexError:
        raise AssertionError("Usage supply string 'start:end:[step]'")

    try:
        start = float(start)
        stop = float(stop)
        spacing = float(spacing)
    except ValueError:
        raise AssertionError("Could not convert one of the value to a float from string {}"
                             "".format(range_string))

    n_points = int((stop - start) / spacing) + 1

    return np.linspace(start, stop, n_points)


def make_2d_array_cyclic(data_2d, axis=0, add_constant=0.0):
    """Makes a 2D array period by copying the first row toward the end

    Parameters
    ----------
    data_2d : ndarray
        NxM array carrying the data. It is assumed that the data along one of the axis is periodic
        and that we want to make the array cyclic by copying the first row or column to the end
    axis: int
        Axis to make periodic. Default = 0
    add_constant : float
        Add this constant to the copy row or column in order to allow to add 2*pi or 360

    Returns
    -------
    ndarray
        Same 2d data array with one extra column (N+1 x M for axis == 0) or one extra row (N x M + 1
        for axis == 1)


    Examples
    --------
    First make some 2D data array using mesh grid containing the direction we want to make periodic

    >>> directions = np.linspace(0, 360, 6, endpoint=False)
    >>> frequencies = np.linspace(0, 3, 3, endpoint=True)
    >>> dd, ff = np.meshgrid(directions, frequencies)
    >>> dd
    array([[   0.,   60.,  120.,  180.,  240.,  300.],
           [   0.,   60.,  120.,  180.,  240.,  300.],
           [   0.,   60.,  120.,  180.,  240.,  300.]])

    The array `dd` is periodic along the axis = 0 direction. So copy the first column to the end

    >>> dd_periodic = make_2d_array_cyclic(dd)
    >>> dd_periodic
    array([[   0.,   60.,  120.,  180.,  240.,  300.,    0.],
           [   0.,   60.,  120.,  180.,  240.,  300.,    0.],
           [   0.,   60.,  120.,  180.,  240.,  300.,    0.]])

    We can do the same in case we have the transposed version, only we have to use the axis = 1
    argument to pick the right axis

    >>> dd_tr = dd.T
    >>> dd_tr
    array([[   0.,    0.,    0.],
           [  60.,   60.,   60.],
           [ 120.,  120.,  120.],
           [ 180.,  180.,  180.],
           [ 240.,  240.,  240.],
           [ 300.,  300.,  300.]])
    >>> dd_tr_periodic = make_2d_array_cyclic(dd_tr, axis=1)
    >>> dd_tr_periodic
    array([[   0.,    0.,    0.],
           [  60.,   60.,   60.],
           [ 120.,  120.,  120.],
           [ 180.,  180.,  180.],
           [ 240.,  240.,  240.],
           [ 300.,  300.,  300.],
           [   0.,    0.,    0.]])

    For polar plotting it is required that the angle is increasing, which means that the last 0
    needs to be 360. This can be established by using the `add_constant` option

    >>> dd_periodic = make_2d_array_cyclic(dd, add_constant=360.)
    >>> dd_periodic
    array([[   0.,   60.,  120.,  180.,  240.,  300.,  360.],
           [   0.,   60.,  120.,  180.,  240.,  300.,  360.],
           [   0.,   60.,  120.,  180.,  240.,  300.,  360.]])

    Notes
    -----
    This function can be used to ensure that a 2D data array with directions at one axis can be
    easily made periodic into the direction axis. This is a requirement to make a polar plot

    """

    if axis == 0:
        # create a Mx1 array of the first column of the 2D data array
        first_col = data_2d[:, 0].reshape(data_2d.shape[0], 1) + add_constant
        data_2d_periodic = np.hstack((data_2d, first_col))
    elif axis == 1:
        # create a 1xN array of the first row of the 2D data array
        first_row = data_2d[0, :].reshape(1, data_2d.shape[1]) + add_constant
        data_2d_periodic = np.vstack((data_2d, first_row))
    else:
        raise AssertionError("Argument `axis` can only be 0 or 1. {} given.".format(axis))

    return data_2d_periodic


def extrap1d(interpolator):
    """
    Extrapolate the interp1d function outside the boundaries

    .. deprecated:: 0.3.4

    Deprecated function, the scipy *interp1d* now can extrapolate as well. This function is
    maintained for backward compatibility.

    Parameters
    ----------
    interpolator: interp1d
        The interpolator object created with scipy

    Returns
    -------
    function
        New interpolator object that extrapolates values outside the range

    Examples
    --------

    Assumed you have a x and y array you want to interpolate. You can use the scipy interp1d
    function for that

    >>> from scipy.interpolate import interp1d

    >>> xp = np.linspace(0, 3, 4)
    >>> yp = xp**2
    >>> print(np.vstack((xp, yp)))
    [[ 0.  1.  2.  3.]
     [ 0.  1.  4.  9.]]

    Use the data samples to create a Interpolator *f_inter*

    >>> f_inter = interp1d(xp, yp)

    Interpolation on a new mesh within the boundaries of the previous mesh can be done as

    >>> xp_new = np.linspace(0, 3, 6)
    >>> yp_new = f_inter(xp_new)
    >>> print(np.vstack((xp_new, yp_new)))
    [[ 0.   0.6  1.2  1.8  2.4  3. ]
     [ 0.   0.6  1.6  3.4  6.   9. ]]

    However, perhaps you want to extend the boundaries outside the initial boundaries. In that case
    you can use extrap1 in order to extrapolate the function outside the boundaries. First create
    the Extrapolator using the scipy Interpolator

    >>> f_extra = extrap1d(f_inter)

    Now we can also extrapolate outside the range

    >>> xp_new2 = np.linspace(0, 4, 9)
    >>> yp_new2 = f_extra(xp_new2)
    >>> print(np.vstack((xp_new2, yp_new2)))
    [[  0.    0.5   1.    1.5   2.    2.5   3.    3.5   4. ]
     [  0.    0.5   1.    2.5   4.    6.5   9.   11.5  14. ]]

    In the latest scipy version extrapolation is possible with the *interp1d* function as well  :

    >>> f_extra2 = interp1d(xp, yp, fill_value="extrapolate")
    >>> yp_new3 = f_extra2(xp_new2)
    >>> print(np.vstack((yp_new2, yp_new3)))
    [[  0.    0.5   1.    2.5   4.    6.5   9.   11.5  14. ]
     [  0.    0.5   1.    2.5   4.    6.5   9.   11.5  14. ]]

    As you can see, the result is the same as the *extrap1d* function. The native
    scipy.interp1d with *extrapolate* as *fill_values* is recommended; the *extrap1d* function is
    only kept for backward compatibility and may be dropped soon

    We can plot the results and compare it with the original data line

    .. plot::

    >>> import matplotlib.pyplot as plt
    >>> l = plt.plot(xp, yp, "o", label="samples")
    >>> l = plt.plot(xp_new2, yp_new2, "-x", label="extrap1d")
    >>> xp_new3 = np.linspace(0, 4, 50)
    >>> yp_new3 = xp_new3**2
    >>> l = plt.plot(xp_new3, yp_new3, "-", label="original")
    >>> l = plt.legend()
    >>> plt.ion()
    >>> plt.show()

    Notes
    -----

    * In case you want to use interpolate with values outside of the boundaries, this function
      allows to extrapolate outside the boundaries as described here interp_

    References
    ----------

    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
    https://stackoverflow.com/a/37172840/4515114

    .. _interp:
        http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
    """

    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike


def print_mat_nested(d, indent=0, nkeys=0):
    """
    Pretty print nested structures from .mat files

    Parameters
    ----------
    indent: int
        Number of spaces to indent
    nkeys: int
        Number of keys to show

    References
    ----------
    * http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python
    """

    # Subset dictionary to limit keys to print.  Only works on first level
    if nkeys > 0:
        d = {k: d[k] for k in
             sorted(d.keys())[:nkeys]}  # Dictionary comprehension: limit to first nkeys keys.

    if isinstance(d, dict):
        for key in sorted(d.keys()):  # loops through key, sort them
            value = d[key]
            print("{}{}{}".format("\t" * indent, "Key: ", str(key)))
            print_mat_nested(value, indent + 1)

    if isinstance(d,
                  np.ndarray) and d.dtype.names is not None:  # Note: and short-circuits by default
        for n in d.dtype.names:  # This means it's a struct, it's bit of a kludge test.
            print("{}{}{}".format("\t" * indent, "Field: ", str(key)))
            print_mat_nested(d[n], indent + 1)


def loadmat(filename):
    """
    Load a matlab data file with a complex data structure

    Parameters
    ----------
    filename: str
        Name of the matlab file to import

    Returns
    -------
    dict:
        Dictionary with the complex matlab data structure

    Examples
    --------

    In order to read a matlab file do

    >>> import os
    >>> file_name = os.path.join("..", "data", "RAO_7.mat")
    >>> data = loadmat(filename=file_name)

    Now have a look at the contents

    >>> print_mat_nested(data)
    Key: DirRange
    Key: FreqRange
    Key: RAO
    Key: __globals__
    Key: __header__
    Key: __version__


    The data stored in the Dir Range field can be accessed as

    >>> data["DirRange"]
    array([  0,  15,  30,  45,  60,  75,  90, 105, 120, 135, 150, 165, 180,
           195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345], dtype=uint16)



    Notes
    -----

    * This function should be called instead of direct spio.loadmat as it cures the problem of not
      properly recovering python dictionaries from mat files. It calls the function check keys to
      cure all entries which are still mat-objects

    References
    ----------
    * http://pyhogs.github.io/reading-mat-files.html
    * http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """

    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries

    Parameters
    ----------
    dict

    Returns
    -------
    dict
        Dictionary with the complex data structure

    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    Convert a matlab object to a dictionary

    Parameters
    ----------
    matobj:
        A matlab opject

    Returns
    -------
    dict
        Dictionary with a matlab object

    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


if __name__ == "__main":
    import doctest

    doctest.testmod()
