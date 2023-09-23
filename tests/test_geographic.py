import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_almost_equal
from pandas.testing import assert_frame_equal

from pymarine.utils.geographic import import_way_points, LocationCheck


def test_import_way_points():
    try:
        data = import_way_points("../data/madeira_ivory.kml", n_distance_points=None)
    except FileNotFoundError:
        data = import_way_points("data/madeira_ivory.kml", n_distance_points=None)

    # create the expected data frame
    data_expected = pd.DataFrame(
        index=np.array(
            [
                0.0,
                51.35344852,
                103.76100595,
                280.37591193,
                540.1554197,
                718.74267821,
                892.77067385,
                1121.19537128,
                1380.98042693,
                1590.32023099,
                1813.18621139,
                2002.15956558,
                2176.81400435,
                2400.01710602,
                2519.33685128,
            ]
        )
    )
    data_expected.index.name = "travel_distance"
    data_expected["latitude"] = np.array(
        [
            32.4,
            31.8,
            31.09051332,
            28.3588147,
            24.03366368,
            21.10140247,
            18.20281473,
            14.38076727,
            10.24088837,
            7.39676749,
            4.39631756,
            2.64584774,
            1.5183185,
            3.43848364,
            4.90171136,
        ]
    )
    data_expected["longitude"] = np.array(
        [
            -17.18,
            -17.9,
            -18.49808275,
            -19.77673541,
            -20.20678434,
            -19.59538317,
            -19.30659972,
            -19.38683736,
            -18.0310994,
            -15.97084304,
            -13.75368275,
            -11.12936662,
            -8.44639843,
            -5.25718259,
            -3.90157281,
        ]
    )
    data_expected["heading"] = np.array(
        [
            225.80012291,
            216.01231006,
            202.54248937,
            185.2219716,
            168.92178621,
            174.55887537,
            181.17301186,
            162.00194422,
            144.06214121,
            143.38056125,
            123.49090439,
            112.61847796,
            59.03325232,
            42.88156671,
            42.88156671,
        ]
    )

    assert_frame_equal(data, data_expected)


def test_import_way_points_interpolated():
    # the kml data without interpolation and select the first five points
    try:
        data_kml = import_way_points(
            "../data/madeira_ivory.kml", n_distance_points=None
        )
    except FileNotFoundError:
        data_kml = import_way_points("data/madeira_ivory.kml", n_distance_points=None)

    data_kml = data_kml.head(5)
    data_kml_expected = pd.DataFrame(
        index=np.array(
            [0.0, 51.3534485188, 103.761005947, 280.375911929, 540.155419698]
        )
    )
    data_kml_expected.index.name = "travel_distance"
    data_kml_expected["latitude"] = np.array(
        [32.4, 31.8, 31.0905133216, 28.3588147039, 24.0336636847]
    )
    data_kml_expected["longitude"] = np.array(
        [-17.18, -17.9, -18.4980827493, -19.7767354088, -20.206784336]
    )
    data_kml_expected["heading"] = np.array(
        [225.800122908, 216.012310055, 202.54248937, 185.221971597, 168.921786213]
    )
    assert_frame_equal(data_kml, data_kml_expected)

    # the kml data with interpolation and select the first five points
    try:
        data_int = import_way_points(
            "../data/madeira_ivory.kml", n_distance_points=2000
        )
    except FileNotFoundError:
        data_int = import_way_points("data/madeira_ivory.kml", n_distance_points=2000)

    # select only the first 5 rows for testing purpose and remove the empty column
    data_int = data_int.head(5)

    data_expected = pd.DataFrame(
        index=np.array([0.0, 2.44157154192, 3.6625019557, 4.88352875138, 6.10465189836])
    )
    data_expected.index.name = "travel_distance"
    data_expected["heading"] = np.array(
        [225.526534405, 225.528796643, 225.533362888, 225.537926615, 225.542487823]
    )
    data_expected["latitude"] = np.array(
        [32.4, 32.3714285714, 32.3571428571, 32.3428571429, 32.3285714286]
    )
    data_expected["longitude"] = np.array(
        [-17.18, -17.2142857143, -17.2314285714, -17.2485714286, -17.2657142857]
    )

    data_int = data_int[data_expected.columns]

    assert_frame_equal(data_int, data_expected)

    # check if the first location of the interpolated dataframe is the same as the pure kml file
    first_location_kml = [data_kml.loc[0, "latitude"], data_kml.loc[0, "longitude"]]
    first_location_dat = [data_int.loc[0, "latitude"], data_int.loc[0, "longitude"]]
    assert_almost_equal([first_location_kml], [first_location_dat])


def test_location_check():
    location_check = LocationCheck(
        latitude=52.37, longitude=4.89, distance_deviation_allowed=10
    )

    out_of_range = location_check.out_of_range(
        current_latitude=52.31, current_longitude=4.78
    )
    assert_equal(out_of_range, True)
    assert_almost_equal(location_check.distance, 10.03906752260248)

    out_of_range = location_check.out_of_range(
        current_latitude=52.36, current_longitude=4.902
    )
    assert_equal(out_of_range, False)
    assert_almost_equal(location_check.distance, 1.3807122835704022)


def main():
    test_import_way_points()
    test_import_way_points_interpolated()
    test_location_check()


if __name__ == "__main__":
    main()
