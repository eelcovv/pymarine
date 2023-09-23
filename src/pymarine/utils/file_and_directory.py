"""
Collection of functions dealing with the system file and directories
"""
import errno
import os

import pandas as pd

from hmc_utils.misc import (get_regex_pattern, get_logger, clear_path, get_time_stamp_from_string)

MSG_FORMAT = "{:30s} : {}"


def get_path_depth(path_name):
    """
    Get the depth of a path or file name

    Parameters
    ----------
    path_name : str
        Path name to get the depth from

    Returns
    -------
    int
        depth of the path


    Examples
    --------

    >>> get_path_depth("C:\Anaconda")
    1
    >>> get_path_depth("C:\Anaconda\share")
    2
    >>> get_path_depth("C:\Anaconda\share\pywafo")
    3
    >>> get_path_depth(".\imaginary\path\subdir\share")
    4
    """

    if os.path.isfile(path_name) and os.path.exists(path_name):
        current_path = os.path.split(path_name)[0]
    else:
        current_path = path_name
    depth = 0
    previous_path = current_path
    while current_path not in ("", "."):
        current_path = os.path.split(current_path)[0]
        if current_path == previous_path:
            # for a full path name we end at the root 'C:\'. Detect that be comparing with the
            # previous round
            break
        previous_path = current_path
        depth += 1

    return depth


def scan_base_directory(walk_dir=".",
                        supplied_file_list=None,
                        file_has_string_pattern="",
                        file_has_not_string_pattern="",
                        dir_has_string_pattern="",
                        dir_has_not_string_pattern="",
                        start_date_time=None,
                        end_date_time=None,
                        time_zone=None,
                        time_stamp_year_first=True,
                        time_stamp_day_first=False,
                        extension=None,
                        max_depth=None,
                        sort_file_base_names=False
                        ):
    """Recursively scan the directory `walk_dir` and get all files underneath obeying the search
    strings and/or date/time ranges

    Parameters
    ----------
    walk_dir : str, optional
        The base directory to start the import. Default = "."
    supplied_file_list: list, optional
        In case walk dir is not given we can explicitly pass a file list to analyse. Default = None
    dir_has_string_pattern : str, optional
        Requires the directory name to have this pattern (Default value = ""). This selection is
        only made on the first directory level below the walk_dir
    dir_has_not_string_pattern : str, optional
        Requires the directory name NOT to have this pattern (Default value = ""). This selection is
        only made on the first directory level below the walk_dir
    file_has_string_pattern : str, optional
        Requires the file name to have this pattern (Default value = "", i.e. matches all)
    file_has_not_string_pattern : str, optional
        Requires the file name NOT to have this pattern (Default value = "")
    extension : str or None, optional
        Extension of the file to match. If None, also matches. Default = None
    max_depth : int, optional
        Sets a maximum depth to which the search is carried out. Default = None, which does not
        limit the search depth. For deep file structures setting a limit to the search depth speeds
        up the search.
    sort_file_base_names: bool, option
        If True, sort the resulting file list alphabetically based on the file base name.
        Default = False
    start_date_time: DateTime or None, optional
        If given, get the date time from the current file name and only add the files with a
        date/time equal or large the *start_date_time*. Default is None
    end_date_time: DateTime or None, optional
        If given, get the date time from the current file name and only add the files with a
        date/time smaller than the *end_date_time*. Default is None
    time_zone:str or None, optional
        If given add this time zone to the file stamp. The start and end time should also have a
        time zone
    time_stamp_year_first: bool, optional
        Passed to the datetime parser. If true, the year is first in the date/time string.
        Default = True
    time_stamp_day_first: bool, optional
        Passed to the datetime parser. If true, the day is first in the date/time string.
        Default = False

    Returns
    -------
    list
        All the  file names found below the input directory `walk_dir` obeying all the search
        strings

    Examples
    --------

    Find all the python files under the share directory in the Anaconda installation folder

    >>> scan_dir = "C:\\Anaconda\\share"
    >>> file_list = scan_base_directory(scan_dir, extension='.py')

    Find all the python files under the share directory in the Anaconda installation folder
    belonging to the pywafo directory

    >>> file_list = scan_base_directory(scan_dir, extension='.py', dir_has_string_pattern="wafo")

    Note that wafo matches on the directory 'pywafo', which is the first directory level below the
    scan directory. However, if we would match on '^wafo' the returned list would be empty as the
    directory has to *start* with wafo.

    In order to get all the files with "test" in the name with a directory depth smaller than 3 do

    >>> file_list = scan_base_directory(scan_dir, extension='.py', dir_has_string_pattern="wafo",
    ...                                 file_has_string_pattern="test", max_depth=3)


    Test the date/time boundaries. First create a file list from 28 sep 2017 00:00 to 5:00 with a
    hour interval and convert it to a string list

    >>> file_names = ["AMS_{}.mdf".format(dt.strftime("%y%m%dT%H%M%S")) for dt in
    ...    pd.date_range("20170928T000000", "20170928T030000", freq="30min")]
    >>> for file_name in file_names:
    ...     print(file_name)
    AMS_170928T000000.mdf
    AMS_170928T003000.mdf
    AMS_170928T010000.mdf
    AMS_170928T013000.mdf
    AMS_170928T020000.mdf
    AMS_170928T023000.mdf
    AMS_170928T030000.mdf

    Use the scan_base_directory to get the files within a specific date/time range

    >>> file_selection = scan_base_directory(supplied_file_list=file_names,
    ...  start_date_time="20170928T010000", end_date_time="20170928T023000")

    >>> for file_name in file_selection:
    ...     print(file_name)
    AMS_170928T010000.mdf
    AMS_170928T013000.mdf
    AMS_170928T020000.mdf

    Note that the selected range run from 1 am until 2 am; the end_date_time of 2.30 am is not
    included

    """

    log = get_logger(__name__)

    # get the regular expression for the has_pattern and has_not_pattern of the files and
    # directories
    file_has_string = get_regex_pattern(file_has_string_pattern)
    file_has_not_string = get_regex_pattern(file_has_not_string_pattern)
    dir_has_string = get_regex_pattern(dir_has_string_pattern)
    dir_has_not_string = get_regex_pattern(dir_has_not_string_pattern)
    log.debug(MSG_FORMAT.format("file_has_string", file_has_string))
    log.debug(MSG_FORMAT.format("file_has_not_string", file_has_not_string))
    log.debug(MSG_FORMAT.format("dir_has_string", dir_has_string))
    log.debug(MSG_FORMAT.format("dir_has_not_string", dir_has_not_string))

    # use os.walk to recursively walk over all the file and directories
    top_directory = True
    file_list = list()
    log.debug("Scanning directory {}".format(walk_dir))
    for root, subdirs, files in os.walk(walk_dir, topdown=True):

        if supplied_file_list is not None:
            root = "."
            subdirs[:] = list()
            files = supplied_file_list

        log.debug("root={}  sub={} files={}".format(root, subdirs, files))
        log.debug(MSG_FORMAT.format("root", root))
        log.debug(MSG_FORMAT.format("sub dirs", subdirs))
        log.debug(MSG_FORMAT.format("files", files))
        # get the relative path towards the top directory (walk_dir)
        relative_path = os.path.relpath(root, walk_dir)

        depth = get_path_depth(relative_path)

        if root == walk_dir:
            top_directory = True
        else:
            top_directory = False

        # base on the first directory list we are going to make selection of directories to
        # process
        if top_directory:
            include_dirs = list()
            for subdir in subdirs:
                add_dir = False
                if dir_has_string is None or bool(dir_has_string.search(subdir)):
                    add_dir = True
                if add_dir and dir_has_not_string is not None:
                    if bool(dir_has_not_string.search(subdir)):
                        add_dir = False
                if add_dir:
                    include_dirs.append(subdir)
                # overrule the subdirectory list of os.walk:
                # http://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk
                log.debug("Overruling subdirs with {}".format(include_dirs))
                subdirs[:] = include_dirs

        for filename in files:

            (filebase, ext) = os.path.splitext(filename)
            if extension is None or extension == ext:

                add_file = False

                if file_has_string is None or bool(file_has_string.search(filebase)):
                    # if has_string is none, the search pattern was either empty or invalid (which
                    # happens during typing the regex in the edit_box). In this case, always add the
                    # file. If not none, filter on the regex, so only add the file if the search
                    # pattern is in the filename
                    add_file = True

                # do not add the file in case the has_not string edit has been set (!="") and if the
                # file contains the pattern
                if add_file and file_has_not_string is not None:
                    if bool(file_has_not_string.search(filebase)):
                        # in case we want to exclude the file, the has_not search pattern must be
                        # valid so may not be None
                        add_file = False

                if add_file and (start_date_time is not None or end_date_time is not None):
                    # we have supplied a start time or a end time. See if we can get a date time
                    # from the file name
                    file_time_stamp = get_time_stamp_from_string(
                        string_with_date_time=filebase, yearfirst=time_stamp_year_first,
                        dayfirst=time_stamp_day_first, timezone=time_zone)

                    if file_time_stamp is not None:
                        # we found a file time stamp. Compare it with the start time
                        if start_date_time is not None:
                            if isinstance(start_date_time, str):
                                # in case the start time was supplied as a string
                                start_date_time = get_time_stamp_from_string(
                                    string_with_date_time=start_date_time,
                                    yearfirst=time_stamp_year_first,
                                    dayfirst=time_stamp_day_first, timezone=time_zone)

                            if file_time_stamp < start_date_time:
                                # the file time stamp is smaller, so don't add it
                                add_file = False
                        # if a end time is supplied. Also compare it with the end time
                        if end_date_time is not None:
                            if isinstance(end_date_time, str):
                                end_date_time = get_time_stamp_from_string(
                                    string_with_date_time=end_date_time,
                                    yearfirst=time_stamp_year_first,
                                    dayfirst=time_stamp_day_first, timezone=time_zone)
                            if file_time_stamp >= end_date_time:
                                # the file time stamp is larger, so don't add it
                                add_file = False

                if dir_has_string is not None and top_directory:
                    # in case we have specified a directory name with a string search, exclude the
                    # top directory
                    add_file = False

                if max_depth is not None and depth > max_depth:
                    add_file = False

                # create the full base name file
                file_name_to_add = os.path.join(walk_dir, relative_path, filebase)

                # get the path to the stl relative to the selected scan directory
                if add_file:
                    log.debug("Adding file {}".format(filebase))
                    file_list.append(clear_path(file_name_to_add + ext))

    # sort on the file name. First split the file base from the path, because if the file are in
    # different directories, the first file is not necessarily the oldest
    if sort_file_base_names:
        df = pd.DataFrame(data=file_list, index=[os.path.split(f)[1] for f in file_list],
                          columns=["file_list"])
        df.sort_index(inplace=True)
        file_list = df.file_list.values

    return file_list


def make_directory(directory):
    """Create a directory in case it does not yet exist.

    Parameters
    ----------
    directory : str
        Name of the directory to create

    Notes
    -----
    This function is used to create directories without checking if it already exist. If the
    directory already exists, we can silently continue.

    Raises
    ------
    OSError
        The OSError is only raised if it is not an `EEXIST` error. This implies that the creation
        of the directory failed due to another reason than the directory already being present.
        It could be that the file system is full or that we may not have write permission

    """
    logger = get_logger(__name__)
    try:
        os.makedirs(directory)
        logger.debug("Created directory : {}".format(directory))
    except OSError as exc:
        # an OSError was raised, see what is the cause
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            # the output directory already exists, that is ok so just continue
            logger.debug(
                "Directory {} already exists. No problem, we just continue".format(directory))
        else:
            # something else was wrong. Raise an error
            logger.warning(
                "Failed to create the directory {} because raised:\n{}".format(directory, exc))
            raise
