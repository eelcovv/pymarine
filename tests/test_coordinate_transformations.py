import numpy as np
from numpy import pi, sqrt, degrees
from numpy.testing import assert_almost_equal

from pymarine.utils.coordinate_transformations import (
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    cartesian_to_spherical,
    spherical_to_cartesian,
    latitude_to_colatitude,
    colatitude_to_latitude,
)


def test_cartensian_to_cylindrical():
    # a list of input argument (left) with the expected answer (right)
    input_arguments_and_expected_result = np.array(
        [
            ((0, 0, 0), (0, 0, 0)),
            ((1, 0, 0), (1, 0, 0)),
            ((0, 1, 0), (1, pi / 2, 0)),
            ((0, 0, 1), (0, 0, 1)),
            ((-1, 0, 0), (1, pi, 0)),
            ((0, -1, 0), (1, -pi / 2, 0)),
            ((0, 0, -1), (0, 0, -1)),
            ((-1, 1, 0), (sqrt(2), 3 * pi / 4, 0)),
            ((1, -1, 0), (sqrt(2), -pi / 4, 0)),
            ((0, 1, -1), (1, pi / 2, -1)),
        ]
    )

    # apply all tests as float
    for argument, expected_result in input_arguments_and_expected_result:
        x, y, z = argument[0], argument[1], argument[2]
        result = cartesian_to_cylindrical(x, y, z)
        assert_almost_equal([result], [expected_result])

        # test for degrees as well
        result = cartesian_to_cylindrical(x, y, z, degrees=True)
        expected_result_deg = [
            degrees(t) if i == 1 else t for i, t in enumerate(expected_result)
        ]
        assert_almost_equal([result], [expected_result_deg])

    # test array like result a well
    input_arguments = input_arguments_and_expected_result[
        :, 0, :
    ]  # select the input arguments
    output_arguments = input_arguments_and_expected_result[
        :, 1, :
    ]  # select the output arguments
    xx, yy, zz = input_arguments[:, 0], input_arguments[:, 1], input_arguments[:, 2]
    rr_e, tt_e, zzc_e = (
        output_arguments[:, 0],
        output_arguments[:, 1],
        output_arguments[:, 2],
    )
    rr, tt, zzc = cartesian_to_cylindrical(xx, yy, zz)
    assert_almost_equal([rr], [rr_e])
    assert_almost_equal([tt], [tt_e])
    assert_almost_equal([zzc], [zzc_e])

    rr, tt, zzc = cartesian_to_cylindrical(xx, yy, zz, degrees=True)
    assert_almost_equal([rr], [rr_e])
    assert_almost_equal([tt], [np.rad2deg(tt_e)])
    assert_almost_equal([zzc], [zzc_e])


def test_cylindrical_to_cartesian():
    # a list of input argument (left) with the expected answer (right)
    input_arguments_and_expected_result = np.array(
        [
            ((0, 0, 0), (0, 0, 0)),
            ((1, 0, 0), (1, 0, 0)),
            ((1, pi / 2, 0), (0, 1, 0)),
            ((0, 0, 1), (0, 0, 1)),
            ((1, pi, 0), (-1, 0, 0)),
            ((1, -pi / 2, 0), (0, -1, 0)),
            ((0, 0, -1), (0, 0, -1)),
            ((sqrt(2), 3 * pi / 4, 0), (-1, 1, 0)),
            ((sqrt(2), -pi / 4, 0), (1, -1, 0)),
            ((1, pi / 2, -1), (0, 1, -1)),
        ]
    )

    # apply all tests
    for argument, expected_result in input_arguments_and_expected_result:
        result = cylindrical_to_cartesian(argument[0], argument[1], argument[2])
        assert_almost_equal([result], [expected_result])

        # test for degrees as well
        argument_deg = [degrees(t) if i == 1 else t for i, t in enumerate(argument)]
        result = cylindrical_to_cartesian(
            argument_deg[0], argument_deg[1], argument_deg[2], degrees=True
        )
        assert_almost_equal([result], [expected_result])

    # test array like result a well
    input_arguments = input_arguments_and_expected_result[
        :, 0, :
    ]  # select the input arguments
    output_arguments = input_arguments_and_expected_result[
        :, 1, :
    ]  # select the output arguments
    rr, tt, zzc = input_arguments[:, 0], input_arguments[:, 1], input_arguments[:, 2]
    xx_e, yy_e, zz_e = (
        output_arguments[:, 0],
        output_arguments[:, 1],
        output_arguments[:, 2],
    )
    xx, yy, zz = cylindrical_to_cartesian(rr, tt, zzc)
    assert_almost_equal([xx], [xx_e])
    assert_almost_equal([yy], [yy_e])
    assert_almost_equal([zz], [zz_e])

    xx, yy, zz = cylindrical_to_cartesian(rr, np.rad2deg(tt), zzc, degrees=True)
    assert_almost_equal([xx], [xx_e])
    assert_almost_equal([yy], [yy_e])
    assert_almost_equal([zz], [zz_e])


def test_cartesian_to_spherical():
    # a list of input argument (left) with the expected answer (right)
    input_arguments_and_expected_result = np.array(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [1, pi / 2, 0]],
            [[0, 1, 0], [1, pi / 2, pi / 2]],
            [[0, 0, 1], [1, 0, 0]],
            [[-1, 0, 0], [1, pi / 2, pi]],
            [[0, -1, 0], [1, pi / 2, -pi / 2]],
            [[0, 0, -1], [1, pi, 0]],
            [[-1, 1, 0], [sqrt(2), pi / 2, 3 * pi / 4]],
            [[1, -1, 0], [sqrt(2), pi / 2, -pi / 4]],
            [[0, 1, -1], [sqrt(2), 3 * pi / 4, pi / 2]],
        ]
    )

    # apply all tests
    for argument, expected_result in input_arguments_and_expected_result:
        result = cartesian_to_spherical(argument[0], argument[1], argument[2])
        assert_almost_equal([result], [expected_result])

        # test for degrees as well
        result = cartesian_to_spherical(
            argument[0], argument[1], argument[2], degrees=True
        )
        expected_result_deg = [
            degrees(t) if i > 0 else t for i, t in enumerate(expected_result)
        ]
        assert_almost_equal([result], [expected_result_deg])

    # test array like result a well
    input_arguments = input_arguments_and_expected_result[
        :, 0, :
    ]  # select the input arguments
    output_arguments = input_arguments_and_expected_result[
        :, 1, :
    ]  # select the output arguments
    xx, yy, zz = input_arguments[:, 0], input_arguments[:, 1], input_arguments[:, 2]
    rr_e, tt_e, pp_e = (
        output_arguments[:, 0],
        output_arguments[:, 1],
        output_arguments[:, 2],
    )
    rr, tt, pp = cartesian_to_spherical(xx, yy, zz)
    assert_almost_equal([rr], [rr_e])
    assert_almost_equal([tt], [tt_e])
    assert_almost_equal([pp], [pp_e])

    rr, tt, pp = cartesian_to_spherical(xx, yy, zz, degrees=True)
    assert_almost_equal([rr], [rr_e])
    assert_almost_equal([tt], [np.rad2deg(tt_e)])
    assert_almost_equal([pp], [np.rad2deg(pp_e)])


def test_spherical_to_cartesian():
    # a list of input argument (left) with the expected answer (right)
    input_arguments_and_expected_result = np.array(
        [
            ((0, 0, 0), (0, 0, 0)),
            ((1, pi / 2, 0), (1, 0, 0)),
            ((1, pi / 2, pi / 2), (0, 1, 0)),
            ((1, 0, 0), (0, 0, 1)),
            ((1, pi / 2, pi), (-1, 0, 0)),
            ((1, pi / 2, -pi / 2), (0, -1, 0)),
            ((1, pi, 0), (0, 0, -1)),
            ((sqrt(2), pi / 2, 3 * pi / 4), (-1, 1, 0)),
            ((sqrt(2), pi / 2, -pi / 4), (1, -1, 0)),
            ((sqrt(2), 3 * pi / 4, pi / 2), (0, 1, -1)),
        ]
    )

    # apply all tests
    for argument, expected_result in input_arguments_and_expected_result:
        result = spherical_to_cartesian(argument[0], argument[1], argument[2])
        assert_almost_equal([result], [expected_result])

        # test for degrees as well
        argument_deg = [degrees(t) if i > 0 else t for i, t in enumerate(argument)]
        result = spherical_to_cartesian(
            argument_deg[0], argument_deg[1], argument_deg[2], degrees=True
        )
        assert_almost_equal([result], [expected_result])

    # test array like result a well
    input_arguments = input_arguments_and_expected_result[
        :, 0, :
    ]  # select the input arguments
    output_arguments = input_arguments_and_expected_result[
        :, 1, :
    ]  # select the output arguments
    rr, tt, pp = input_arguments[:, 0], input_arguments[:, 1], input_arguments[:, 2]
    xx_e, yy_e, zz_e = (
        output_arguments[:, 0],
        output_arguments[:, 1],
        output_arguments[:, 2],
    )
    xx, yy, zz = spherical_to_cartesian(rr, tt, pp)
    assert_almost_equal([xx], [xx_e])
    assert_almost_equal([yy], [yy_e])
    assert_almost_equal([zz], [zz_e])

    xx, yy, zz = spherical_to_cartesian(
        rr, np.rad2deg(tt), np.rad2deg(pp), degrees=True
    )
    assert_almost_equal([xx], [xx_e])
    assert_almost_equal([yy], [yy_e])
    assert_almost_equal([zz], [zz_e])


def test_latitude_to_colatitude():
    input_arguments_and_expected_result = np.array(
        [
            (0, pi / 2),
            (pi / 4, pi / 4),
            (pi / 2, 0),
        ]
    )

    for argument, expected_result in input_arguments_and_expected_result:
        result = latitude_to_colatitude(argument)
        assert_almost_equal([result], [expected_result])

        argument_deg = degrees(argument)
        result = latitude_to_colatitude(argument_deg, degrees=True)
        expected_result_deg = degrees(expected_result)
        assert_almost_equal([result], [expected_result_deg])

    # test the array input as well
    ll = input_arguments_and_expected_result[:, 0]
    cl_e = input_arguments_and_expected_result[:, 1]

    cl = latitude_to_colatitude(ll)
    assert_almost_equal([cl], [cl_e])

    cl = latitude_to_colatitude(np.rad2deg(ll), degrees=True)
    assert_almost_equal([cl], [np.rad2deg(cl_e)])


def test_colatitude_to_latitude():
    input_arguments_and_expected_result = np.array(
        [
            (0, pi / 2),
            (pi / 4, pi / 4),
            (pi / 2, 0),
        ]
    )

    for argument, expected_result in input_arguments_and_expected_result:
        result = colatitude_to_latitude(argument)
        assert_almost_equal([result], [expected_result])

        argument_deg = degrees(argument)
        result = colatitude_to_latitude(argument_deg, degrees=True)
        expected_result_deg = degrees(expected_result)
        assert_almost_equal([result], [expected_result_deg])

    # test the array input as well
    cl = input_arguments_and_expected_result[:, 0]
    ll_e = input_arguments_and_expected_result[:, 1]

    ll = latitude_to_colatitude(cl)
    assert_almost_equal([ll], [ll_e])

    ll = latitude_to_colatitude(np.rad2deg(cl), degrees=True)
    assert_almost_equal([ll], [np.rad2deg(ll_e)])
