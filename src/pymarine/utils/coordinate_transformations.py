"""
Definition of some common coordinate system conversions

Notes
-----
* The coordinate transformations defined in this section are a selection of the
  coordinate_transformation implemented in the astropysics model. The reason why this module is not
  installed by it self is that this module has not been updated for Python 3.
  It is therefore easier just to take the part of the module which is relevant for HMC

* The definitions of the spherical coordinates follow the physics convention as described here_

References
----------

* https://en.wikipedia.org/wiki/Spherical_coordinate_system

.. _here:
    https://en.wikipedia.org/wiki/Spherical_coordinate_system

"""

import numpy as np
from numpy import pi


# <--------------------Functional coordinate transforms------------------------->
def cartesian_to_polar(x, y, degrees=False):
    """Converts arrays in 2D rectangular Cartesian coordinates to polar
    coordinates.

    Parameters
    ----------
    x : float or array_like
        First cartesian coordinate
    y : float or array_like
        Second cartesian coordinate
    degrees : boolean
        If True, the output theta angle will be in degrees, otherwise radians. (Default = False)

    Returns
    -------
    tuple of 2 floats or ndarrays
        Polar coordinates (r, theta) where theta is measured from the +x axis increasing towards
        the +y axis

    """
    npx, npy = np.array(x), np.array(y)
    r = (npx * npx + npy * npy) ** 0.5
    t = np.arctan2(npy, npx)
    if degrees:
        t = np.degrees(t)

    return r, t


def polar_to_cartesian(r, t, degrees=False):
    """Converts arrays in 2D polar coordinates to rectangular cartesian
    coordinates.

    Parameters
    ----------
    r : float or array_like
        Radial coordinate
    t : float or array_like
        Azimuthal angle from +x-axis increasing towards +y-axis
    degrees : boolean
        If True, the input angles will be in degrees, otherwise radians. (Default value = False)

    Returns
    -------
    tuple of 2 floats or ndarrays
        Cartesian coordinates (x,y)

    """
    npt, npr = np.array(t), np.array(r)
    if degrees:
        npt = np.radians(npt)

    return npr * np.cos(npt), r * np.sin(npt)


def cartesian_to_spherical(x, y, z, degrees=False):
    """Converts three arrays in 3D rectangular cartesian coordinates to
    spherical polar coordinates.

    Note that the spherical coordinates are in *physicist* convention such that (1,pi/2,0) is
    x-axis.

    Parameters
    ----------
    x : float or array_like
        First cartesian coordinate
    y : float or array_like
        Second cartesian coordinate
    z : float or array_like
        Third cartesian coordinate
    degrees : boolean
        If True, the output theta angle will be in degrees, otherwise radians. (Default = False)

    Returns
    -------
    tuple of 3 floats or ndarrays
        Spherical coordinates (r,theta,phi)

    Examples
    --------
    Convert the coordinates (1, 0, 0)

    >>> radius, theta, phi = cartesian_to_spherical(1, 0, 0)
    >>> print("Radius = {:.4g}  Theta = {:.4g}  Phi = {:.4g}".format(radius, theta, phi))
    Radius = 1  Theta = 1.571  Phi = 0

    >>> radius, theta, phi = cartesian_to_spherical(1, 1, 1, degrees=True)
    >>> print("Radius = {:.4g}  Theta = {:.4g}  Phi = {:.4g}".format(radius, theta, phi))
    Radius = 1.732  Theta = 54.74  Phi = 45


    Convert array_like coordinates

    >>> xx, yy, zz = [0, 0, 1 ], [1, 0, 1 ], [0, 1, 1 ]
    >>> radii, thetas, phis = cartesian_to_spherical(xx, yy, zz)
    >>> for ii, radius in enumerate(radii):
    ...   print("({:.2f}, {:.2f} {:.2f}) -> ({:.4f}, {:.4f}, {:.4f})"
    ...         "".format(xx[ii], yy[ii], zz[ii], radius, thetas[ii], phis[ii]))
    (0.00, 1.00 0.00) -> (1.0000, 1.5708, 1.5708)
    (0.00, 0.00 1.00) -> (1.0000, 0.0000, 0.0000)
    (1.00, 1.00 1.00) -> (1.7321, 0.9553, 0.7854)

    References
    ----------
    https://en.wikipedia.org/wiki/Spherical_coordinate_system

    """
    npx, npy, npz = np.array(x), np.array(y), np.array(z)
    xsq, ysq, zsq = npx * npx, npy * npy, npz * npz
    r = (xsq + ysq + zsq) ** 0.5
    t = np.arctan2((xsq + ysq) ** 0.5, z)
    p = np.arctan2(npy, npx)
    if degrees:
        t, p = np.degrees(t), np.degrees(p)
    return r, t, p


def spherical_to_cartesian(r, t, p, degrees=False):
    """Converts arrays in 3D spherical polar coordinates to rectangular cartesian coordinates.

    Parameters
    ----------
    r : float or array like
        Radial coordinate
    t : float or array_like
        Colatitude (angle from z-axis)
    p : float or array_lie
        Azimuthal angle from +x-axis increasing towards +y-axis
    degrees : boolean
        If True, the input angles will be in degrees, otherwise radians. (Default value = False)

    Returns
    -------
    tuple of 3 floats or ndarrays
        Cartesian coordinates (x,y,z)

    Notes
    -----
    * The spherical coordinates are in *physicist* convention such that (1,pi/2,0) is the x-axis.

    Examples
    --------
    >>> xx, yy, zz = spherical_to_cartesian(1, 0, 0)
    >>> print("xx = {:.4g}  yy = {:.4g}  zz = {:.4g}".format(xx, yy, zz))
    xx = 0  yy = 0  zz = 1
    >>> xx, yy, zz = spherical_to_cartesian(1.7321, 54.736, 45, degrees=True)
    >>> print("xx = {:.4g}  yy = {:.4g}  zz = {:.4g}".format(xx, yy, zz))
    xx = 1  yy = 1  zz = 1
    >>> rr, tt, pp = np.array([1, 2, 3 ]), np.array([0, 45, 90 ]), np.array([0, -45, 45 ])
    >>> xx, yy, zz = spherical_to_cartesian(rr, tt, pp, degrees=True)
    >>> for ii, x in enumerate(xx):
    ...   print("({:6.2f}, {:6.2f} {:6.2f}) -> ({:6.2f}, {:6.2f}, {:6.2f})"
    ...         "".format(rr[ii], tt[ii], pp[ii], x, yy[ii], zz[ii]))
    (  1.00,   0.00   0.00) -> (  0.00,   0.00,   1.00)
    (  2.00,  45.00 -45.00) -> (  1.00,  -1.00,   1.41)
    (  3.00,  90.00  45.00) -> (  2.12,   2.12,   0.00)
    """
    npr, npt, npp = np.array(r), np.array(t), np.array(p)
    if degrees:
        npt, npp = np.radians(npt), np.radians(npp)
    x = npr * np.sin(npt) * np.cos(npp)
    y = npr * np.sin(npt) * np.sin(npp)
    z = npr * np.cos(npt)

    return x, y, z


def latitude_to_colatitude(lat, degrees=False):
    """Converts from latitude  (i.e. 0 at the equator) to colatitude/inclination
    (i.e. "theta" in physicist convention).

    Parameters
    ----------
    lat : float or array_like
        Latitude

    degrees : bool
         If True, use degrees (Default value = False)

    Returns
    -------
    float or ndarray
        Colatitude

    Examples
    --------
    >>> latitude_to_colatitude(10, degrees=True)
    80
    >>> phi_list = [np.pi / i for i in range(1,6)]
    >>> co_latitude = latitude_to_colatitude(phi_list)
    >>> for ii, phi in enumerate(phi_list):
    ...    print("{:6.4f} -> {:7.4f}".format(phi, co_latitude[ii]))
    3.1416 -> -1.5708
    1.5708 ->  0.0000
    1.0472 ->  0.5236
    0.7854 ->  0.7854
    0.6283 ->  0.9425
    """
    lat_array = np.array(lat)
    if degrees:
        return 90 - lat_array
    else:
        return pi / 2 - lat_array


def colatitude_to_latitude(theta, degrees=False):
    """Converts from colatitude/inclination (i.e. "theta" in physicist convention)
    to latitude (i.e. 0 at the equator).

    Parameters
    ----------
    theta : float or array-like
        input colatitude
    degrees : bool
        If True, the input is interpreted as degrees, otherwise radians. (Default value = False)

    Returns
    -------
    float or ndarray
        Latitude

    Examples
    --------
    >>> colatitude_to_latitude(15, degrees=True)
    75
    >>> phi_list = [np.pi / i for i in range(1,6)]
    >>> latitude = colatitude_to_latitude(phi_list)
    >>> for ii, phi in enumerate(phi_list):
    ...    print("{:6.4f} -> {:7.4f}".format(phi, latitude[ii]))
    3.1416 -> -1.5708
    1.5708 ->  0.0000
    1.0472 ->  0.5236
    0.7854 ->  0.7854
    0.6283 ->  0.9425


    """
    theta_array = np.array(theta)
    if degrees:
        return 90 - theta_array
    else:
        return pi / 2 - theta_array


def cartesian_to_cylindrical(x, y, z, degrees=False):
    """Converts three arrays in 3D rectangular Cartesian coordinates to cylindrical
    polar coordinates.

    Parameters
    ----------
    x : float or array-like
        x cartesian coordinate
    y : float or array-like
        y cartesian coordinate
    z : float or array-like
        z cartesian coordinate
    degrees : bool
        If True, the output angles will be in degrees, otherwise radians. (Default value = False)

    Returns
    -------
    tuple of 3 floats or ndarrays
        Cylindrical coordinates as a (rho,theta,z),  with theta increasing from +x to +y and
        theta = 0 at x-axis.

    """
    s, t = cartesian_to_polar(x, y, degrees)
    return s, t, z


def cylindrical_to_cartesian(s, t, z, degrees=False):
    """Converts three arrays in cylindrical polar coordinates to 3D rectangular
    Cartesian coordinates.

    Parameters
    ----------
    s : float or array-like
        radial polar coordinate
    t : float or array-like
        polar angle (increasing from +x to +y, 0 at x-axis)
    z : float or array-like
        z coordinate
    degrees : bool
        If True, the input angle will be in degrees, otherwise radians. (Default value = False)

    Returns
    -------
    tuple of 3 floats or ndarrays
        Cartesian coordinates as an (x,y,z) tuple.

    """
    x, y = polar_to_cartesian(s, t, degrees)
    return x, y, z
