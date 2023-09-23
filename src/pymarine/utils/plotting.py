"""
Some functions to support with plotting in Python, mostly based on the *matplotlib* module
"""
import itertools
import re
import logging

_logger = logging.getLogger()


# definition of hmc colors
c_hmc = dict(
    blue="#0083D1",
    lightblue="#69ABD2",
    red="#DD2E1A",
    lightred="#DD877E",
    green="#00CC00",
    darkcyan="#2eb8b8"
)


def analyse_annotations(annotation):
    """Analyse the string `annotation` which compactly sets the properties of a label string such as
    position, size and color.

    Parameters
    ----------
    annotation : str
        label[@(xp, yp)[s10][a0][cRRGGBB]]

    Returns
    -------
    tuple of strings and integers
        text (str), x position (float) y position (float), color (str), size (int), axis (int)

    Notes
    -----
    * The annotation string can be just given a a label. This label optionally can be extended
      with a '@' sign to include more information, like the location (xp, yp), the size with s10,
      the color with c and the axis with a and a integer

    * The annotation format is mostly used as a compact way to provide information on the annotation
      via a user input file

    * When specifying a color with the c construct, make sure that you put the color at the end of
      the *annotation* string.

    * In the Parameter scription we used a hex formulation for the color like cFFAA00. However,
      also the color names like cblue or cred are allowed

    * In case you add chmc:red, the hmc definition of red will be used. Other hmc color
      definitions are:

         - blue
         - lightblue
         - red
         - lightred
         - green
         - darkcyan

    * Also we can add the xkcd_ color definitions as described in the matplotlib_ manual

    .. _xkcd:
        https://xkcd.com/color/rgb/

    .. _matplotlib:
        https://matplotlib.org/users/colors.html

    Examples
    --------

    Simple example of a label with all the properties set to default

    >>> analyse_annotations("simple_label")
    ('simple_label', 0.0, 0.0, 'black', 12, 0)

    place a label at position x=0.1, y=0.4

    >>> analyse_annotations("more_complex@0.1,0.4")
    ('more_complex', 0.1, 0.4, 'black', 12, 0)

    place a label at position x=0.8, y=0.9 with color red. Note that we need to add brackets
    around the location

    >>> analyse_annotations("colored_label@(0.8,0.9)cred")
    ('colored_label', 0.8, 0.9, 'red', 12, 0)

    place a label at position x=0.8, y=0.9 with color red. Note that we need to add brackets
    around the location

    >>> analyse_annotations("small_label@(0.8,0.9)s8")
    ('small_label', 0.8, 0.9, 'black', 8, 0)

    Place a label at position x=0.8, y=0.9 with color red and size 16. This time we need
    to add the color label at the end to extract it correcly

    >>> analyse_annotations("large_colored_label_in_axis_2@(0.8,0.9)s16a2cred")
    ('large_colored_label_in_axis_2', 0.8, 0.9, 'red', 16, 2)

    Finally lets show how you use more colours

    >>> analyse_annotations("label@(0.8,0.9)s16a2chmc:red")
    ('label', 0.8, 0.9, '#DD2E1A', 16, 2)

    As you can see, the hex code of HMC read is returned.

    To set the xkcd colors do

    >>> analyse_annotations("label@(0.8,0.9)s16a2cxkcd:red")
    ('label', 0.8, 0.9, 'xkcd:red', 16, 2)

    This color value 'xkcd:red' is understood by all matplotlib routines
    """
    lx = 0.0
    ly = 0.0
    color = "black"
    axis = 0
    size = 12
    pos = "({},{})".format(lx, ly)
    # first replace all white spaces from the string, as it is not allowed
    try:
        # see if there is an @ sign indicating that the position is specified
        text, rest = annotation.split('@')
        try:
            # after the @ sign we start with the position between brackets (,). Find it
            pos, rest = re.sub("[(|)]", " ", rest).split()

            # now the rest is only a size s18 and a color cred or c#FFAA00 (hexa code). First get
            # the size, then the color
            size_pattern = "s(\d+)"
            axis_pattern = "a(\d)"
            color_pattern = "c(.*)"
            m = re.search(size_pattern, rest)
            if bool(m):
                size = int(m.group(1))
                rest = re.sub(size_pattern, "", rest)
            m = re.match(axis_pattern, rest)
            if bool(m):
                try:
                    axis = int(m.group(1))
                except ValueError:
                    _logger.warning("axis must by integers. Set zero")
                rest = re.sub(axis_pattern, "", rest)
            m = re.match(color_pattern, rest)
            if bool(m):
                color = m.group(1)

        except ValueError:
            # in case of a value error we did not have a rest, so try again without split
            pos = re.sub("[(|)]", " ", rest)
        finally:
            lx, ly = pos.split(',')
            lx = float(lx)
            ly = float(ly)
    except ValueError:
        # there as no @ sign: just return the text value with all the rest the defaults
        text = annotation

    if re.match("^hmc:", color):
        hmc_color_name = color.split(":")[1]
        try:
            color = c_hmc[hmc_color_name]
        except KeyError:
            _logger.warning("color name not recognised: {}. Keeping black".format(color))
            color = "black"

    return text, lx, ly, color, size, axis


def clean_up_artists(axis, artist_list):
    """Remove all the artists stored in the `artist_list` belonging to the `axis`.


    Parameters
    ----------
    axis : :class:matplotlib.axes.Axes
        Clean Artists (ie. items added to a matplotlib plot) belonging to this axis
    artist_list : list
        List of artist to remove.

    Notes
    -----

    In case of animation of complex plots with contours and labels (such as a timer) we sometimes
    need to take care of removing all the Artists which are changing every time step.
    This function takes care of that. It also also ensured that we are not running out of memory
    when too many Artists are added

    Examples
    --------

    >>> from matplotlib.pyplot import subplots
    >>> from numpy.random import random_sample

    Create a list which we use to store all the artists which need to be cleaned

    >>> artist_list = list()

    Create a plot of some random data

    >>> fig, ax = subplots(ncols=1, nrows=1)
    >>> data = random_sample((20, 30))
    >>> cs = ax.contourf(data)

    Store the contour Artist in a list

    >>> artist_list.append(cs)

    Now clean it again

    >>> clean_up_artists(ax, artist_list)

    """

    for artist in artist_list:
        try:
            # fist attempt: try to remove collection of contours for instance
            while artist.collections:
                for col in artist.collections:
                    artist.collections.remove(col)
                    try:
                        axis.collections.remove(col)
                    except ValueError:
                        pass

                artist.collections = []
                axis.collections = []
        except AttributeError:
            pass

        # second attempt, try to remove the text
        try:
            artist.remove()
        except (AttributeError, ValueError):
            pass


def clean_up_plot(artist_list):
    """A small script to clean up all lines or other items of a matplot lib plot.


    Parameters
    ----------
    artist_list :
        a list of items to clean up

    Notes
    -----

    Necessary if you want to loop over multiple plot and maintain the axes and only update the data.
    Basically this does the same as `clean_up_artists`

    """
    n_cleaned = 0
    for ln in artist_list:
        n_cleaned += 1
        try:
            ln.pop(0).remove()
        except (IndexError, AttributeError):
            try:
                ln.remove()
            except (ValueError, TypeError, AttributeError):
                del ln
        else:
            n_cleaned -= 1
            _logger.debug("All clean up failed. ")
    artist_list = []
    return artist_list


def sub_plot_axis_to_2d(axis, n_rows=1, n_cols=1):
    """
    Turn the axis return value of matplotlibs subfigures in a 2D list, regardless of the number of
    rows and columns

    Parameters
    ----------
    axis : list or :class:matplotlib.axes.Axes
        Axes object or list of axis objects as return by the matplotlib subplots command
    n_rows : int, optional, default: 1
        The number of rows of the current plot
    n_cols : int, optional, default: 1
        number of columns of the current plot

    Returns
    -------
    type: list
        new 2D list of the axis  which can be referred to as `axis[i_row][j_col]`

    Notes
    -----

    The return axis of the *matplotlib.subplots* command is 2D in the case we have more than one row
    and more than one column. However, if there is only one row or col then the list will be 1D,
    for *n_col = 1* and *n_row = 1* the axis are directly returned. This scrips ensures that the
    list is always started as a 2D list such that the axis can be referred to with two indices:
    *axis[i_row][j_col]*, also if we only have on row or columns

    The practical usage of this function is to be able to write more generic code for which it is
    not known before hand how many rows and columns need to be generated (it may depend on the user
    input).

    Examples
    --------

    First create a *matplotlib* plot with only one row of two column

    >>> import matplotlib.pyplot as plt
    >>> from numpy import (sin, cos, pi, linspace)
    >>> fig, axis = plt.subplots(nrows=1, ncols=2)

    At this point, *axis* is a 1D list of 2 *matplotlib.axis.Axes* objects, one for each column.
    So we can plot like this

    >>> x = linspace(0, 2 * pi, num=100)
    >>> l0 = axis[0].plot(x, sin(x))
    >>> l1 = axis[1].plot(x, cos(x))

    Now we want to turn that into a 2D list

    >>> axis_2d = sub_plot_axis_to_2d(axis, n_cols=2)

    Now, axis2d is a 2D list of Axes for 2 columns which are referred as axis[0][0] and axis[0][1],
    so we can plot

    >>> l2d0 = axis_2d[0][0].plot(x, sin(x))
    >>> l2d1 = axis_2d[0][1].plot(x, cos(x))

    Note that we are referring to *axis_2d* as a 2D list with 2 indices in stead of one.
    This allows to create plots on the subplots returned axis in always the same way, which is handy
    if we don't know beforehand how many rows and columns we have.

    """
    new_axis = list()
    if n_cols == 1 and n_rows == 1:
        # we have only one column and one row. Make the new 2D list by appending a list of the axis
        # to the new axis
        new_axis.append([axis])
    elif n_cols == 1 and n_rows > 1:
        for i_r in range(n_rows):
            new_axis.append([axis[i_r]])
    elif n_cols > 1 and n_rows == 1:
        new_axis.append(axis)
    else:
        new_axis = axis

    return new_axis


def set_limits(axis, v_min=None, v_max=None, direction=0):
    """
     Set the x or y limits of the current axis and overwrite them if requested.

    Parameters
    ----------
     v_min: float
        The x or y limit to override if given
     v_max: float
        The x or y maximum to override ig given
    direction: int or str
        Set limit on this axis. 0=x, 1=y, or pass "x" or "y"

    Returns
    -------
    list
        [min, max] with the current x or y limits

    Notes
    -----
    If only the min or max is imposed, obtained the other limit from the current settings
    """
    lim = list()
    if direction in (0, "x"):
        current_limits = axis.get_xlim()
    elif direction in (1, "y"):
        current_limits = axis.get_ylim()
    else:
        raise ValueError("Invalid value for the axis parameter")

    if v_min is not None:
        lim.append(v_min)
    else:
        # the minimum is not given, so just return the default value
        lim.append(current_limits[0])
    if v_max is not None:
        lim.append(v_max)
    else:
        # the y maximum is not given, so just return the default value
        lim.append(current_limits[1])

    if direction in (0, "x"):
        axis.set_xlim(lim)
    else:
        axis.set_ylim(lim)

    return lim


def get_valid_legend_handles_labels(axis):
    """
    Function to get all the labels from an axis which are not set to "None" (as a string!)

    Parameters
    ----------
    axis: :obj:`SubPlotAxis`

    Returns
    -------
    tuple of list:
        handles, labels

    """
    handles_tpm, labels_tmp = axis.get_legend_handles_labels()
    handles = list()
    labels = list()
    for ii, label in enumerate(labels_tmp):
        if label != "None":
            handles.append(handles_tpm[ii])
            labels.append(label)
    return handles, labels


def flip(items, ncol):
    """
    Flip the legend list to get the lines in horizontal layout

    Parameters
    ----------
    items : list
        list of labels

    ncol : int
        Number of columns


    Returns
    -------
    list
        Reorganised list of labels

    """
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def clear_all_legends(axis):
    """
    Remove all the legends from the current axis

    Parameters
    ----------
    axis: list of :obj:AxesSubPlots
    """

    for ax in axis:
        try:
            ax.legend_.remove()
        except AttributeError:
            pass
