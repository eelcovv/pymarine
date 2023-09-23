"""
Collection of functions dealing with geographical coordinates based on the *LatLon* module.
"""
import logging
import fastkml as kml
import numpy as np
import pandas as pd
from latloncalc.latlon import LatLon
from scipy.constants import nautical_mile


_logger = logging.getLogger(__name__)


class LocationCheck(object):
    """
    Class to store a location in lat/lon and check if the distance of another location is out of
    range

    Parameters
    ----------
    longitude : float
        Longitude of target location as a float
    latitude : float
        Latitude of target location as a float
    distance_deviation_allowed: float, optional
        Maximum allowed distance from initial location in km, Default = 10 km

    Notes
    -----
    * Use this class to store a reference location and check later if the distance of another
      location is out of range given by the maximum `distance_deviation_allowed` input argument.
    * Class uses the LatLon package so distances are all in km

    """

    def __init__(self, longitude, latitude, distance_deviation_allowed=10):
        self.target_location = LatLon(latitude, longitude)
        self.distance_deviation_allowed = float(distance_deviation_allowed)
        self.distance = None
        self.current_location = None

    def out_of_range(self, current_latitude, current_longitude):
        """Check if the current location is outside the given distance from the target location

        Parameters
        ----------
        current_latitude : float
            Latitude as a float of the current location
        current_longitude : float
            Longitude as a float of the current location

        Returns
        -------
        bool
            True in case the current location is outside the distance from the target given by
           `distance_deviation_allowed`

        Examples
        --------

        Initialise the location_check with a target location (52.37, 4.89). The we can use the
        method `out_of_range` to test if the current_location is out of range of the target location

        >>> location_check = LocationCheck(latitude=52.37, longitude=4.89,
        ...                                distance_deviation_allowed=10)
        >>> out_range = location_check.out_of_range(current_latitude=52.37, current_longitude=4.89)
        >>> print("Distance : {:.1f} km. Out of range : {}".format(location_check.distance, out_range))
        Distance : 0.0 km. Out of range : False
        >>> out_range = location_check.out_of_range(current_latitude=52.31, current_longitude=4.84)
        >>> print("Distance : {:.1f} km. Out of range : {}".format(location_check.distance, out_range))
        Distance : 7.5 km. Out of range : False
        >>> out_range = location_check.out_of_range(current_latitude=52.36, current_longitude=4.90)
        >>> print("Distance : {:.1f} km. Out of range : {}".format(location_check.distance, out_range))
        Distance : 1.3 km. Out of range : False
        >>> out_range = location_check.out_of_range(current_latitude=52.31, current_longitude=4.77)
        >>> print("Distance : {:.1f} km. Out of range : {}".format(location_check.distance, out_range))
        Distance : 10.6 km. Out of range : True

        Also check around the 0th and 180th meridians

        >>> greenwich = LocationCheck(latitude=51.48, longitude=-1,
        ...                              distance_deviation_allowed=10)
        >>> out_range = greenwich.out_of_range(current_latitude=51.48, current_longitude=1)
        >>> print("Distance : {:.1f} km. Out of range : {}".format(greenwich.distance, out_range))
        Distance : 138.9 km. Out of range : True
        >>> out_range = greenwich.out_of_range(current_latitude=51.48, current_longitude=359)
        >>> print("Distance : {:.1f} km. Out of range : {}".format(greenwich.distance, out_range))
        Distance : 0.0 km. Out of range : False

        >>> pacific_ocean = LocationCheck(latitude=0.0, longitude=179,
        ...                              distance_deviation_allowed=223)
        >>> out_range = pacific_ocean.out_of_range(current_latitude=0.0, current_longitude=-179)
        >>> print("Distance : {:.1f} km. Out of range : {}".format(pacific_ocean.distance, out_range))
        Distance : 222.6 km. Out of range : False
        >>> out_range = pacific_ocean.out_of_range(current_latitude=0.0, current_longitude=181)
        >>> print("Distance : {:.1f} km. Out of range : {}".format(pacific_ocean.distance, out_range))
        Distance : 222.6 km. Out of range : False

        Notes
        -----
        The outcome of this routine has been checked with the Google Earth ruler
        """

        self.current_location = LatLon(current_latitude, current_longitude)

        self.distance = self.current_location.distance(self.target_location)

        _logger.debug(
            "Distance forecast location {} with target location {}: {}"
            "".format(self.current_location, self.target_location, self.distance)
        )

        if self.distance > self.distance_deviation_allowed:
            _logger.debug(
                "Distance {} out of range of {} with {} km"
                "".format(self.distance, self.target_location, self.distance)
            )
            out_of_range = True
        else:
            out_of_range = False

        return out_of_range

    def make_report(self):

        msg = "{:20s} : {} {}"
        _logger.info(msg.format("Target location set at", self.target_location, ""))
        _logger.info(
            msg.format(
                "Maximum distance deviation allowed",
                self.distance_deviation_allowed,
                " km",
            )
        )


def get_speed_from_distance_and_time(
    trajectory,
    distance_name="travel_distance",
    speed_name="speed_sustained",
    datetime_name="DateTime",
    travel_time_name="travel_time",
    speed_max_clip=None,
):
    """Calculate the speed based on the travel distance and travel time.

    Parameters
    ----------
    trajectory : dataframe
        Trajectory
    distance_name: str, optional
        Name of the distance column, Default value = "travel_distance"
    speed_name : str, optional
        Name of the column with the sustained speed,  Default value = "speed_sustained"
    datetime_name:  str, optional
        Name of the column with the date time,  Default value = "DateTime"
    distance_name : str, optional
        Name of the column with the travel distance, Default value = "travel_distance
    travel_time_name : str, optional
        Name of the newly created column with the travel time, Default value = "travel_time"
    speed_max_clip : float, optional
        clip all values above this speed in knots. Default value = None, which means that the speed
        is not clipped

    Returns
    -------
    DataFrame
        Update of the input DataFrame `trajectory` with an extra column named `speed_name`
        containing the calculated speed

    Raises
    ------
    KeyError:
        Raised in case no travel time is available

    Examples
    --------

    First create a data frame with some travel distance at and interval of 3h over 12 hours.

    >>> data =  pd.DataFrame(index=pd.date_range(start="20160101", end="20160101T120000", freq="3h"))
    >>> data["travel_distance"] = np.linspace(start=0, stop=60, num=data.index.size, endpoint=True)

    Now calculate the speed based on the travel distance change between each time step

    >>> data = get_speed_from_distance_and_time(data)
    >>> data
                         travel_distance  travel_time  speed_sustained
    2016-01-01 00:00:00              0.0          0.0              5.0
    2016-01-01 03:00:00             15.0          3.0              5.0
    2016-01-01 06:00:00             30.0          6.0              5.0
    2016-01-01 09:00:00             45.0          9.0              5.0
    2016-01-01 12:00:00             60.0         12.0              5.0

    Notes
    -----
    * The output of `travel_distance_and_heading_from_coordinates` can be used as an input for
      `get_speed_from_distance_and_time`

    See Also
    --------
    travel_distance_and_heading_from coordinates : Calculate the travel distance from coordinates
    """

    delete_datetime_column_at_end = False
    try:
        date_time = trajectory[datetime_name]
    except KeyError:
        # we failed to get the datetime from the column. This means we have to copy it to a column
        # first
        trajectory[datetime_name] = trajectory.index
        date_time = trajectory[datetime_name]
        delete_datetime_column_at_end = True

    # make sure we are dealing with date/time, not strings
    date_time = pd.Series([pd.Timestamp(t) for t in date_time])

    time_in_hour = (date_time - date_time[0]) / pd.Timedelta("1 hour")
    delta_time = time_in_hour.diff().fillna(1).values
    trajectory[travel_time_name] = time_in_hour.values
    trajectory[speed_name] = trajectory[distance_name].diff().fillna(0) / delta_time

    # with the diff method we can not get the first position. Just assume it to be the same as the
    # next one
    trajectory.ix[0, speed_name] = trajectory.ix[1, speed_name]

    trajectory[speed_name].fillna(0, inplace=True)

    if speed_max_clip is not None:
        # clip the speed above the threshold value with nan and then interpolate between the missing
        # values
        trajectory.ix[trajectory[speed_name] > speed_max_clip, speed_name] = np.nan
        trajectory[speed_name].bfill(inplace=True)

    if delete_datetime_column_at_end:
        # only delete the date time column if we have created it internally
        trajectory.drop(datetime_name, inplace=True, axis=1)

    return trajectory


def travel_distance_and_heading_from_coordinates(
    db,
    latitude_name="GPS_LATITUDE",
    longitude_name="GPS_LONGITUDE",
    heading_name="HEADING",
    travel_distance_name="travel_distance",
):
    """
    Calculate the travel distance and optionally the heading based on the latitude and longitude
    columns in the DataFrame `db`

    Parameters
    ----------
    db : DataFrame
        Data base carrying all the data (including the latitude and longitude)
    latitude_name : str, optional
         Name of the latitude column (Default value = "GPS_LATITUDE")
    longitude_name : str, optional
         Name of the longitude column (Default value = "GPS_LONGITUDE")
    heading_name : str, optional
        The heading name. It is assumed that the heading entry already exist and that only
        the missing values are added based on the coordinate displacement (Default = "HEADING")
    travel_distance_name : str, optional
        name of the newly created column with the travel distance in nautical miles
        (Default value = "travel_distance")

    Returns
    -------
    DataFrame
        DataFrame with an extra column named `travel_distance_name` containing the travel distance
        in nautical miles and optionally an extra column named `heading_name` with the headings.

    Examples
    --------
    First create a data frame with some latitude longitude values at an 3 hour interval over 12 hour

    >>> data = pd.DataFrame(index=pd.date_range(start="20160101", end="20160101T120000", freq="3h"))
    >>> data["GPS_LATITUDE"] = np.linspace(start=55.4, stop=54.4, num=data.index.size)
    >>> data["GPS_LONGITUDE"] = np.linspace(start=3.34, stop=3.14, num=data.index.size)

    Now use the `travel_distance_and_heading_from_coordinates` function to add the travel distance
    and heading to the data frame. A new data frame will be copied to the output

    >>> data = travel_distance_and_heading_from_coordinates(data)
    >>> data
                         GPS_LATITUDE  GPS_LONGITUDE  travel_distance     HEADING
    2016-01-01 00:00:00         55.40           3.34         0.000000  186.534194
    2016-01-01 03:00:00         55.15           3.29        15.125795  186.574900
    2016-01-01 06:00:00         54.90           3.24        30.252197  186.615478
    2016-01-01 09:00:00         54.65           3.19        45.379211  186.655928
    2016-01-01 12:00:00         54.40           3.14        60.506836  186.655928

    Check the 180 meridian as well. Split `linspace` for the longitude in two parts in order to
    introduce the discontinuity in longitude when passing the 180-meridian

    >>> data =  pd.DataFrame(index=pd.date_range(start="20160101", end="20160101T120000", freq="3h"))
    >>> data["GPS_LATITUDE"] = np.linspace(start=0.0, stop=0, num=data.index.size)
    >>> data["GPS_LONGITUDE"] = np.append(np.linspace(start=-179.0, stop=-180, num=data.index.size//2,
    ...                                               endpoint=False),
    ...                                   np.linspace(start=180.0, stop=179, num=data.index.size//2+1))
    >>> data = travel_distance_and_heading_from_coordinates(data)
    >>> data
                         GPS_LATITUDE  GPS_LONGITUDE  travel_distance  HEADING
    2016-01-01 00:00:00           0.0         -179.0         0.000000    270.0
    2016-01-01 03:00:00           0.0         -179.5        30.053858    270.0
    2016-01-01 06:00:00           0.0          180.0        60.107716    270.0
    2016-01-01 09:00:00           0.0          179.5        90.161575    270.0
    2016-01-01 12:00:00           0.0          179.0       120.215433    270.0

    Notes
    -----
    * The DataFrame `db` must have at least two columns containing the latitude and longitude. In
      case more columns are present this is not a problem: all columns will be copied the the output
    * Based on the latitude and longitude values, the travel distance and heading is calculated
      using the LatLon package * In case the heading field already exists, only the missing
      values will be updated.
    * The column names of the latitude, longitude, distance and headding can be defined via the
      arguments of the function.

    See Also
    --------
    get_speed_from_distance_and_time : calculate the speed based on the distance and time
    import_way_points :  import a list of coordinates from a kml file generated in Google Earth

    """


    n_rows = db.index.size

    last_coordinates = None

    # create a list with the current distances (starting with 0 for the first location)
    delta_distance = [0]
    try:
        headings = db[heading_name].values
    except KeyError:
        headings = np.zeros(n_rows)
    else:
        # check if headings is not empty
        if True in db[heading_name].isnull():
            _logger.debug("empty heading. filling it with zeros")
            headings = np.zeros(n_rows)

    _logger.debug("Start loop {} ".format(LatLon))
    # loop over all the lines
    for i in range(n_rows):
        # get the coordinates from the GPS_LATITUDE and GPS_LONGITUDE columns and turn into a LatLon
        # object
        latitude = db[latitude_name].values[i]
        longitude = db[longitude_name].values[i]
        _logger.debug("Converting {} {} with {}".format(latitude, longitude, LatLon))
        coordinates = LatLon(latitude, longitude)

        if last_coordinates is None:
            # for the first row, the travel distance =0, just store the coordinates
            last_coordinates = coordinates
        else:
            # for all other row, calculate the travel distance wro the last position. Convert to
            # miles
            try:
                displacement = coordinates.distance(last_coordinates) / (
                    nautical_mile / 1000.0
                )
            except ValueError:
                displacement = 0

            # now store the displacement and the current position to last_position for the next row
            delta_distance.append(displacement)

            # heading from last coordinate to coordinate
            try:
                headings[i - 1] = last_coordinates.heading_initial(coordinates)
                headings[i] = headings[i - 1]
            except ValueError:
                headings[i] = np.nan

            # finally, update the last_coordinates with the current
            last_coordinates = coordinates

    # turn the displacement array into a cumulative sum to get the total travel distance
    db[travel_distance_name] = np.array(delta_distance).cumsum()
    db[heading_name] = headings
    db[heading_name].bfill(inplace=True)
    db[heading_name] = db[heading_name].mod(360)

    return db


def import_way_points(
    file_name,
    latitude_name="latitude",
    longitude_name="longitude",
    heading_name="heading",
    travel_distance_name="travel_distance",
    n_distance_points=2000,
    start_wp=None,
):
    """Import the way points of a Google Earth kml file.

    Parameters
    ----------
    file_name : str
        Name of the google earth kml file to import
    latitude_name : str, optional
        Name of the latitudes column (Default value = "latitude")
    longitude_name : str, optional
        Name of the longitudes column (Default value = "longitude")
    heading_name : str, optional
        Name of the heading column (Default value = "heading")
    travel_distance_name : str, optional
        Name of the travel distance column (Default value = "travel_distance")
    n_distance_points : int, optional
        Number of points to use to interpolate the kml data (Default value = 2000)
    start_wp : int, optional
        If given, the first `start_wp` pins are skipped (Default value = None)

    Returns
    -------
    DataFrame
        New data frame with the latitude and longitude coordinates. The travel distance is used for
        the index of the DataFrame

    Notes
    -----

    * The kml input data can be made in Google Earth by putting a list of way points (yellow pin
      points) to the map and store this into one directory. Then, with the right mouse button on
      this directory you can choose "Save Place As"
      and pick the kml format.
    * If the `n_distance_points` is set to None, only the coordinates of the  pin locations as found
      in the kml file are imported.
    * If `n_distance_points` is specified, an equidistant set of coordinates obtained from the
      interpolated values between the pin location in the kml file is returned.
    * In case that interpolation is used, make sure that there is at least one sample point in
      between the pin
      locations as defined in the kml file.

    Examples
    --------

    First import the pure coordinates as defined at the pin  point

    >>> data = import_way_points("../data/madeira_ivory.kml", n_distance_points=None)
    >>> data.info()
    <class 'pandas.core.frame.DataFrame'>
    Float64Index: 15 entries, 0.0 to 2519.33685128
    Data columns (total 3 columns):
    latitude     15 non-null float64
    longitude    15 non-null float64
    heading      15 non-null float64
    dtypes: float64(3)
    memory usage: 480.0 bytes

    >>> data["latitude"].head()
    travel_distance
    0.000000      32.400000
    51.353449     31.800000
    103.761006    31.090513
    280.375912    28.358815
    540.155420    24.033664
    Name: latitude, dtype: float64

    >>> data["longitude"].head()
    travel_distance
    0.000000     -17.180000
    51.353449    -17.900000
    103.761006   -18.498083
    280.375912   -19.776735
    540.155420   -20.206784
    Name: longitude, dtype: float64


    Do the same import but interpolate in between the pin locations on a regular grid

    >>> data = import_way_points("../data/madeira_ivory.kml", n_distance_points=2000)
    >>> data["latitude"].head()
    travel_distance
    0.000000    32.400000
    2.441572    32.371429
    3.662502    32.357143
    4.883529    32.342857
    6.104652    32.328571
    Name: latitude, dtype: float64

    >>> data["longitude"].head()
    travel_distance
    0.000000   -17.180000
    2.441572   -17.214286
    3.662502   -17.231429
    4.883529   -17.248571
    6.104652   -17.265714
    Name: longitude, dtype: float64

    """


    # open the google earth kml file. Do it as binary in order to avoid decoding issues
    with open(file_name, "rb") as fp:
        kml_string = fp.read()

    latitudes = list()
    longitudes = list()
    #
    k_object = kml.KML()
    k_object.from_string(kml_string)
    for k_level1 in k_object.features():
        for k_level2 in k_level1.features():
            for cnt, k_level3 in enumerate(k_level2.features()):
                point = k_level3.geometry
                if start_wp is not None and cnt < start_wp:
                    _logger.debug(
                        "Skipping Way point {}: lat = {} lon = {}"
                        "".format(cnt, point.x, point.y)
                    )
                    continue
                longitudes.append(point.x)
                latitudes.append(point.y)
                _logger.debug(
                    "Adding Way point {}: lat = {} lon = {}"
                    "".format(cnt, latitudes[-1], longitudes[-1])
                )

    # create data frame out of way point coordinates and return
    df = pd.DataFrame(
        data=np.vstack((latitudes, longitudes)).T,
        columns=[latitude_name, longitude_name],
    )

    # calculate the travel distance
    df = travel_distance_and_heading_from_coordinates(
        df,
        latitude_name=latitude_name,
        longitude_name=longitude_name,
        heading_name=heading_name,
        travel_distance_name=travel_distance_name,
    )

    if n_distance_points is not None:
        df.set_index(travel_distance_name, inplace=True, drop=False)
        distances = np.linspace(0, df.index[-1], n_distance_points, endpoint=True)
        df2 = pd.DataFrame(
            index=distances, columns=[latitude_name, longitude_name, heading_name]
        )
        df2[travel_distance_name] = distances
        df3 = pd.concat([df, df2]).sort_index().interpolate()
        df3 = (
            df3.reset_index()
            .drop_duplicates(keep="first", subset=travel_distance_name)
            .set_index(travel_distance_name)
        )
        df3 = df3.reindex(df2.index)
        df3.reset_index(inplace=True, drop=True)
        df3.drop("index", axis=1, inplace=True)

        df = travel_distance_and_heading_from_coordinates(
            df3,
            latitude_name=latitude_name,
            longitude_name=longitude_name,
            heading_name=heading_name,
            travel_distance_name=travel_distance_name,
        )

    df.set_index(travel_distance_name, inplace=True, drop=True)

    return df
