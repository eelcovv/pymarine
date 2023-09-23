"""
HMC Marine Wave spectral distributions

Implementations of a series of standard wave spectra and distribution functions and other
utilities.
"""

import logging
from math import lgamma, gamma, sqrt

import numpy as np
import scipy as sp
from matplotlib.pyplot import figure, plot, ioff, show, title
from numpy.fft import fftshift, ifftshift
from scipy.constants import g as g0
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate

import pymarine.utils.coordinate_transformations as acf
from pymarine.utils.numerical import find_idx_nearest_val

logger = logging.getLogger()

TINY = np.exp(-53)
TP_MINIMUM = 1.0


def omega_peak_jonswap(wind_speed, fetch):
    """Calculate the angular peak frequency :math:`\omega_p` for a given wind speed and fetch length

    Parameters
    ----------
    wind_speed : float
        Wind speed at 10 m
    fetch : float
        Fetch length in m

    Returns
    -------
    float :
        Angular peak frequency

    Notes
    -----
    The following equation is implemented to calculate the angular peak frequency

    .. math::

        \omega_p  = 22 (\\frac{g_0 ^ 2 }{U_w F}) ^{1/3}

    with :math:`U_w`  the wind speed and :math:`F` the fetch length
    """

    return 22 * (g0 ** 2 / (wind_speed * fetch)) ** (1.0 / 3.0)


def alpha_jonswap(wind_speed, fetch):
    """Calculate  the alpha parameter for the Jonswap spectrum

    Parameters
    ----------
    wind_speed : :obj:`float`
        wind speed at 10 m
    fetch : :obj:`float`
        fetch length in m

    Returns
    -------
    float
        alpha factor

    Notes
    -----
    * The alpha factor is calculated as

    .. math::

        \\alpha_g = 0.076 \Big(\\frac{{U_w}^2}{F  g_0}\Big)^{0.22}

    Where :math:`U_w` is the wind velocity and :math:`F` is the Fetch length


    * Deprecated: it is better to define the Jonswap spectrum by explicitly specifying Hs and Tp

    """
    return 0.076 * (wind_speed ** 2 / (fetch * g0)) ** 0.22


# @profile
def spreading_function2(
    theta, theta0=0.0, s_spreading_factor=5, n_spreading_factor=None
):
    """Calculate the spreading function :math:`D_k(k,\\theta)` for polar coordinates

    Parameters
    ----------
    theta :
        frequency in rad/s
    theta0 :
        peak frequency in rad/s (Default value = 0.0)
    s_spreading_factor : :obj:`int`
        The spreading factor :math:`s`. Can be any integer > 0. (Default value = 5).
    n_spreading_factor : :obj:`int`
        The spreading factor :math:`n`. Can be any integer > 0. (Default value = None)
        In case this argument is given (not None) the s_spreading factor will be overruled as both
        are related according to s = 2 n + 1

    Returns
    -------
    ndarray
        Distribution function over the theta range

    Notes
    -----
      in which
    * This spreading function definition implements the version 3.5.8.6 of DNV,

      .. math::

        D(\\theta) = \\frac{\Gamma(1+s)}{2\sqrt{\pi}*\Gamma(0.5+s)}  \cos(\\frac{\\theta-\\theta_0}{2})^{2s}

      for which the input argument `spreading_factor` :math:`s` is given by

      .. math::
        s = 2 n + 1

      where :math:`n` is the spreading factor as used in the `spreading_function` implementation.

    * The recommended values for respectively :math:`n` and :math:`s` spreading factors is given as

    ================ ============ =============
    Spreading factor  Wind waves     Swell
    ================ ============ =============
      :math:`s`       5 ~ 9         >= 13
      :math:`n`       2 ~ 4         >= 6
    ================ ============ =============

    * Note the the *spreading_factor* and *spreading_factor2* are to alternative
      formulation given by DNV and should should yield almost identical results.
    * Formally the *spreading_factor* is only based on the n-spreading definition,
      and the *spreading_factor2* function only based on the s-spreading factor definition. In the
      API of the function,  however, you can pass for both functions either the n or s spreading
      factor, internally the spreading factor is converted to the n or s definition, respectively
    """
    if n_spreading_factor is not None:
        s_spreading_factor = 2 * n_spreading_factor + 1
    if s_spreading_factor is None:
        raise AssertionError(
            "At this point the s_spreading factor should be defined. Most like no spreading factor"
            "was given at all. Please check your input"
        )

    factor = (
        gamma(s_spreading_factor + 1)
        / gamma(s_spreading_factor + 0.5)
        / 2
        / sqrt(np.pi)
    )
    theta_prime = np.angle(np.exp(1j * (theta - theta0)))
    return factor * (np.cos(0.5 * theta_prime)) ** (2 * s_spreading_factor)


# @profile
def spreading_function(
    theta, theta0=0.0, s_spreading_factor=5, n_spreading_factor=None
):
    """Calculate the spreading function :math:`D_k(k,\\theta)` for polar coordinates according to
    the DNV definition.

    Parameters
    ----------

    theta : :class:`ndarray`
        The theta angle in radians running from 0 to 2pi
    theta0 : :obj:`float`, optional
        The main direction of the spreading function [rad] (Default value = 0.0 rad)
    s_spreading_factor : :obj:`int`
        The spreading factor :math:`s`. Can be any integer > 0. (Default value = 5).
    n_spreading_factor : :obj:`int`
        The spreading factor :math:`n`. Can be any integer > 0. (Default value = None)
        In case this argument is not None the s_spreading factor will be overruled as both are
        related as s = 2 n + 1

    Returns
    -------
    ndarray
        The spreading function as a function of theta

    Notes
    -----

    * The spreading function as defined in DNV 3.5.8.4 is calculated:

      .. math::

        D(\\theta)=\\frac{\Gamma(1+n/2)}{\sqrt{\pi}\Gamma(1/2+n/2)}\cos^n(\\theta-\\theta_0)

      with :math:`n` the spreading factor and :math:`\Gamma` is the standard Gamma function.

    * An alternative spreading factor :math:`s` can be found in the DNV spreading function
      definition 3.5.8.6. :

      .. math::

        D(\\theta)=\\frac{\Gamma(s+1)}{2\sqrt{\pi}\Gamma(s+1/2)}\cos(\\frac{1}{2}(\\theta-\\theta_0))^{(2s)}


      Where the spreading factor :math:`s` is related to the factor :math:`n`  as

      .. math::
        s = 2 n + 1

      By default this function except the s spreading factor via the `s_spreading_factor` argument,
      however, the spreading factor :math:`n` may be passed in stead via the `n_spreading_factor`
      argument. In that case the n_spreading_factor argument will be overwritten by the value as
      defined by :math:`n= (s-1) / 2`

    * The recommended values for respectively :math:`n` and :math:`s` spreading factors is given as

    ================ ============ =============
    Spreading factor  Wind waves     Swell
    ================ ============ =============
      :math:`s`       5 ~ 9         >= 13
      :math:`n`       2 ~ 4         >= 6
    ================ ============ =============


    * In order to prevent an overflow of the gamma function,  :math:`\Gamma`, the logarithm of the
      spreading function :math:`\log(D(\\theta))` is calculated and then converted to its normal
      value by returning the exponential of :math:`\log(D(\\theta))`

    """

    if n_spreading_factor is not None:
        # update the s spreading definition is the n value is given
        s_spreading_factor = 2 * n_spreading_factor + 1

    if s_spreading_factor is None:
        raise AssertionError(
            "At this point the s_spreading factor should be defined. Most like no spreading factor"
            "was given at all. Please check your input"
        )

    # derive the n spreading definition from the spreading definition
    n_spreading_factor = (s_spreading_factor - 1) / 2

    if n_spreading_factor <= 0:
        raise AssertionError(
            "s_spreading factor of {} leads to a zero or negative n_spreading factor"
            "".format(s_spreading_factor)
        )

    # note that in the implementation below we use the equation first formulation based on the
    # n-spreading factor and the cos(theta)^n term (not the cos(theta/2)^(s2)). We can still
    # supply an s spreading factor as this is converted to the n spreading factor above
    max_angle = np.pi / 2.0

    # convert the theta array into modulated array always in the range -pi - pi
    thetaprime = np.angle(np.exp(1j * (theta - theta0)))

    size = thetaprime.size

    # calculate the logarithm of the spreading function in order to prevent an overflow of the gamma
    # function
    loggam1 = lgamma(1 + n_spreading_factor / 2.0)
    loggam2 = lgamma(0.5 + n_spreading_factor / 2.0)
    logsqrtpi = np.log(np.sqrt(np.pi))
    logD0 = loggam1 - loggam2 - logsqrtpi

    logspread = np.where(
        abs(thetaprime) >= max_angle,
        np.full(size, -np.infty),
        np.full(
            size, logD0 + n_spreading_factor * np.log(np.cos(thetaprime).clip(TINY))
        ),
    )

    # return the exponential of the logarithm of the spreading function (i.e. the spreading function
    # itself)
    return np.exp(logspread)


# @profile
def spectrum_gauss(omega, Hs=1.0, Tp=10.0, sigma=0.0625, spectral_version="dnv"):
    """Calculate the gauss spectral density function

    Parameters
    ----------
    omega : ndarray
        Vector with angular frequencies [rad/s]
    Hs : float, optional
        Significant wave height in meter (Default value = 1.0 m)
    Tp : float, optional
        Peak period swell in seconds (Default value = 10.0 s)
    sigma : float, optional
        Width of the spectrum used for the spectral_version="dnv" only. In the "sim" spectral
        version, the width is related to Tp. For Hs=1 m and Tp=10 s, the default value of
        sigma=0.0625 results in the same width for the "sim" and "dnv" spectral_version.
    spectral_version : {"sim", "dnv"}
        Type of Gauss spectrum used, Default="sim"

        * "sim": Hs and Tp define energy of spectrum and peak location, respectively. The width of
          the gauss follows automatically from Tp. Larger Tp gives more narrow width of the spectrum
        * "dnv": Hs and Tp define energy of spectrum and peak location, respectively. The width of
          the gauss is explicitly defined by the sigma input argument and does not depened on Tp

    Notes
    -----
    * The benefit of the 'sim' spectral version is that the width does not need to be defined: it
      follows automatically from the Tp values (larger Tp leads to a more narrow width of the
      spectrum)
    * The 'dnv' definition allows to set the width explicitly via the *sigma* input argument. This
      allow better control over the width of the spectrum yourself. It can also be used to define a
      spectrum which has all its energy focused into one signal frequency bin by setting sigma to a
      very small value such as 1e-6.

    References
    ----------
    * The Orcina spectrum is defined here
      https://www.orcina.com/SoftwareProducts/OrcaFlex/Documentation/Help/Content/html/Waves,WaveSpectra.htm

    Returns
    -------
    ndarray:
        Vector with spectral densities equal in length to omega

    """

    try:
        omega_peak = 2 * np.pi / Tp
    except ZeroDivisionError:
        logger.info("Tp given yields a ZeroDivisionError. Return zero psd")
        psd = np.zeros(omega.shape)
    else:
        if spectral_version == "sim":
            psd = (
                (Hs / 4.0) ** 2
                * np.exp(-50 * (omega / omega_peak - 1) ** 2)
                / (0.1 * omega_peak * np.sqrt(2 * np.pi))
            )
        elif spectral_version == "dnv":
            if sigma is None:
                # in case sigma was passed as None also impose the default value
                sigma = 0.0625
            psd = (
                (Hs / 4.0) ** 2
                * np.exp(-((omega - omega_peak) ** 2) / (2 * sigma ** 2))
                / (sigma * np.sqrt(2 * np.pi))
            )
        else:
            raise AssertionError(
                "Spectral_version input argument is either 'dnv' or 'sim'. "
                "Found : {}".format(spectral_version)
            )

    return psd


def ag_taylor_expand(gamma):
    """Obtain the Ag parameter for the Jonswap model using a taylor expansion

    Parameters
    ----------
    gamma: float
        Peakness parameter of the Jonswap model

    Returns
    -------
    float
        Ag parameter

    """

    x = (float(gamma) - 4) / 3.0
    a_list = [0.60077816, -0.21413165, 0.08440066, -0.03609301, 0.03850485, -0.02472358]
    ag = 0
    for p, a in enumerate(a_list):
        ag += a * x ** p

    return ag


# @profile
def spectrum_jonswap(omega, Hs=1.0, Tp=10.0, gamma=3.3, spectral_version="dnv"):
    """
    Calculate the Jonswap Wave Spectral Density function vs the angular frequency

    Parameters
    ----------
    omega : :class:`numpy.ndarray`
        1-D array with the radial frequencies in rad/s
    Hs : float, optional
        Significant wave height in meter, Default=1.0 m
    Tp : float, optional
        Peak period in seconds, Default=10.0 s
    gamma : float, optional
        Peaking factor, Default=3.3
    spectral_version : {"sim", "dnv"}, optional
        Type of Jonswap spectrum used, either "sim" or "dnv" can b used. Default="sim". Note that
        the difference is is minor and can be neglected.

    Returns
    -------
    :class:`numpy.ndarray`
        1-D array with wave spectral density  distribution

    Notes
    -----

    *Definition Jonswap Spectrum*

    The coefficients defined in this function are

    .. math::

        \\alpha = 5.0 / 16.0

        \\beta = 5.0 / 4.0

        \omega_p = 2 \pi / T_p


        \omega < \omega_p : \sigma = 0.07

        \omega \ge \omega_p : \sigma = 0.09

    The  Pierson-Moskovitz spectrum is  calculated as

    .. math::

        S_P = \\alpha {H_s}^2 {\omega_p}^4 \omega^{-5} \exp\Big(-\\beta  (\omega_p / \omega)^4\Big)

    Extra parameters for Jonswap are

    .. math::

        r_g = \exp\Big(-\Big[\\frac{(\omega - \omega_p)^2}{(2 (\sigma \omega_p)^2}\Big])\Big)

    How the scaling factor :math:`A_g` is calculated depends on the "spectral_version" option

    * spectral_version: "dnv"

        .. math::

            A_g = 1 - 0.287 \log(\gamma)

    * spectral_version: "sim"


        .. math::

            A_g = \sum_{p=0}^{5} a_p  \Big(\\frac{\gamma - 4}{3}\Big)^p

    The difference in value of :math:`A_g` between both versions is numerically negligible

    And finally the Jonswap spectrum is calculated as

    .. math::

        S_J = A_g \gamma^{r_g} S_P

    *Coefficients Specification*

    The coefficient :math:`a_p` for the :math:`A_g` scaling factor in spectral_version="sim" are
    given as

    =============== ============ ============= ============ ============ =========== ==============
    Index p         0            1             2            3            4           5
    =============== ============ ============= ============ ============ =========== ==============
    :math:`a_p`     0.60077816   -0.21413165   0.08440066   -0.0360930   0.03850485  -0.02472358
    =============== ============ ============= ============ ============ =========== ==============


    References
    ----------

    * DNV-RP-C205 pg 33

    """
    # Constants
    alpha = 5.0 / 16.0
    beta = 5.0 / 4.0

    size = omega.size

    try:
        omega_peak = 2 * np.pi / Tp
    except ZeroDivisionError:
        logger.info("Tp given yields a ZeroDivisionError. Return zero psd")
        psd = np.zeros(size)
    else:

        om = np.where(abs(omega) < TINY, np.full(size, TINY), np.full(size, abs(omega)))

        # set JS sigma 0.07 for omega<omega_peak and 0.09 otherwise
        sigma = np.where(om <= omega_peak, np.full(size, 0.07), np.full(size, 0.09))

        # Pierson Moskowitz contribution
        SPM = (
            alpha
            * Hs ** 2
            * omega_peak ** 4
            * om ** (-5)
            * np.exp(-beta * (omega_peak / om) ** 4)
        )

        # Jonswap
        rg = np.exp(-((om - omega_peak) ** 2 / (2.0 * (sigma ** 2) * omega_peak ** 2)))

        if spectral_version == "sim":
            # the only difference in the matlab implementation is in the Ag factor
            Ag = ag_taylor_expand(gamma)
        elif spectral_version == "dnv":
            Ag = 1 - 0.287 * np.log(gamma)
        else:
            raise AssertionError(
                "spectral_version argument must be 'sim' or 'dnv'. Other versions are not "
                "implemented. We found {}".format(spectral_version)
            )

        SJS = Ag * gamma ** rg
        psd = SJS * SPM

    return psd


# @profile
def omega_deep_water(wave_number_vector):
    """Deep water wave dispersion relation

    Parameters
    ----------
    wave_number_vector : ndarray
        vector with the wave numbers in rad/m

    Returns
    -------
    ndarray:
        vector with omega for k. If k < 0, omega is also < 0

    Notes
    -----
    Implements the dispersion relation for deep water waves

    .. math ::

        \omega(k) = \sqrt{g_0 k}

    where :math:`g_0=9.81~m/s^2`. A sign is added such that waves traveling in negative direction
    have also a negative angular frequency.

    """
    size = wave_number_vector.size
    omega_d = np.sqrt(abs(wave_number_vector) * g0) * sp.where(
        wave_number_vector >= 0, np.full(size, 1.0), np.full(size, -1.0)
    )
    return omega_d


# @profile
def spectrum_wave_k_domain(
    k_waves,
    Hs=1.0,
    Tp=10.0,
    gamma=3.3,
    sigma=0.0625,
    spectrum_type="jonswap",
    spectral_version="sim",
):
    """Calculate a wave spectrum in wave-vector domain based on its definition in frequency domain

    Parameters
    ----------
    k_waves : ndarray
        Input array with the wave vector nodes in rad/m
    Hs : float, optional
        significant wave height (Default value = 1.0)
    Tp : float, optional
        Peak period (Default value = 10.0)
    gamma : float, optional
        peaking factor (Default value = 3.3)
    sigma : float, optional
        The width of the spectrum; only used for a Gauss spectrum (spectrum_type="gauss") using the
        spectral_version="dnv". Default =  0.064 rad/s
    spectrum_type : {"jonswap", "gauss"}
        type of  spectrum used. Either "jonswap" or "gauss". Default = "jonswap"
    spectral_version: {"sim", "dnv"}
        type of spectrum_type used. Default = "sim"
        * "sim": spectrum definition as used
        * "dnv": spectrum definition by DNV.

    Returns
    -------
    type
        spectrum : array with spectral components relating the spectrum to the wave number

    Notes
    -----
    * The  spectrum in omega domain, :math:`S(\omega)` can be converted to a spectrum in
      k domain, :math:`S(k)` using

      .. math ::

          S(k) = S(\omega) \\frac{d\omega(k)}{dk} =  S(\omega) \\frac{1}{2}\sqrt{\\frac{g_0}{k}}

      where the second step used the deep water dispersion relation.
    * In this function the either a Jonswap of a Gauss spectrum can be used to convert to the
      k-domain

    """

    try:
        size = k_waves.size
    except AttributeError:
        size = 1
    sign = sp.where(k_waves < 0, np.full(size, -1.0), np.full(size, 1.0))

    # replace zero k_waves for a TINY number to prevent a zero by division error
    kk = np.abs(np.where(abs(k_waves) < TINY, np.full(size, sign * TINY), k_waves))

    # calculate the omega from the wave numbers based on the deep water dispersion relation
    omega = abs(omega_deep_water(kk))

    # calculate the frequency domain spectrum
    if spectrum_type == "jonswap":
        spectrum = spectrum_jonswap(
            omega, Hs=Hs, Tp=Tp, gamma=gamma, spectral_version=spectral_version
        )
    elif spectrum_type == "gauss":
        spectrum = spectrum_gauss(
            omega, Hs=Hs, Tp=Tp, sigma=sigma, spectral_version=spectral_version
        )
    else:
        raise AssertionError(
            "spectrum_type must be either 'jonswap' or 'gauss'. Found {}"
            "".format(spectrum_type)
        )

    # The relation between psd in omega and k domain follows from the dk/domega deriviate. Use deep
    # water dispersion
    domegadk = 0.5 * np.sqrt(g0 / kk)

    spectrum_vs_k = domegadk * spectrum

    # multiply with the derivative to convert to the wave number domain
    return spectrum_vs_k


# @profile
def spectrum_jonswap_k_domain_2(
    k_waves, Hs=1.0, Tp=10.0, gamma=3.3, spectral_version="sim"
):
    """
    Explicit version of Jonswap spectrum in k domain obtained by substituting the derivative
    :math:`d\omega/dk` into the Jonswap equation.

    Parameters
    ----------
    k_waves :
        input array with the wave numbers
    Hs :
        significant wave height (Default value = 1.0)
    Tp :
        Peak period (Default value = 10.0)
    gamma :
        peaking factor (Default value = 3.3)
    spectral_version : {"sim", "dnv"}, optional
        Type of Jonswap spectrum used, either "sim" or "dnv" can b used. Default="sim". Note that
        the difference is is minor and can be neglected.


    Notes
    -----
    * This spectrum is the same as the spectrum_wave_k_domain for spectrum_type=jonswap
    * In this version the derivative between k/omega is explicitly calculated from the Jonswap
      definition

    Returns
    -------
    ndarray
        Spectrum : array with spectral components relating the spectrum to the wave number

    """

    # calculate the omega from the wave numbers based on the deep water dispersion relation

    alpha = 5.0 / 16.0
    beta = 5.0 / 4.0

    size = k_waves.size

    # replace the zero k with a TINY number to prevent overflow. Also take absolute value of k as
    # direction is not import for the spectral value. Copy the whole vector k_wave to kk in order to
    # prevent that the array is changed outside the routine (as it is passed by reference)
    sign = np.where(k_waves < 0, np.full(size, -1.0), np.full(size, 1.0))
    kk = sp.where(abs(k_waves) < TINY, sign * TINY, abs(k_waves))

    try:
        omega_peak = 2 * np.pi / Tp
    except ZeroDivisionError:
        logger.info("Tp given yields a ZeroDivisionError. Return zero psd")
        psd = np.zeros(kk.shape)
    else:

        # peak wave number based on deep water dispersion relation
        kp = omega_peak ** 2 / g0

        # set JS sigma 0.07 for omega<omega_peak and 0.09 otherwise. since kp=sqrt(om/g),
        # the same holds for kk
        sigma = sp.where(kk <= kp, np.full(size, 0.07), np.full(size, 0.09))

        # Pierson Moskowitz contribution
        SPM = alpha * Hs ** 2 * kp ** 2 / (2 * kk ** 3) * np.exp(-beta * (kp / kk) ** 2)

        # Jonswap
        rg = np.exp(-((np.sqrt(kk) - np.sqrt(kp)) ** 2 / (2.0 * (sigma ** 2) * kp)))
        if spectral_version == "sim":
            Ag = ag_taylor_expand(gamma)
        elif spectral_version == "dnv":
            Ag = 1 - 0.287 * np.log(gamma)
        else:
            raise AssertionError(
                "spectral_version argument must be 'sim' or 'dnv'. Other versions are not "
                "implemented"
            )

        SJS = Ag * gamma ** rg

        # power spectrum is multiplication of Pierson Moskowitz and jonswap correction
        psd = SPM * SJS

    return psd


# @profile
def spectrum_to_complex_amplitudes(omega, spectral_modulus, phase, mirror=False):
    """Turn a wave spectral density function into a set of complex amplitudes

    Parameters
    ----------
    omega : ndarray
        vector of frequency components. Can be used for both k and f or omega
    spectral_modulus :
        vector with spectral modulus components belonging to the current omega
    phase :
        array with the phase components
    mirror :
        mirror the spectral components (Default value = False) such that we obey S(-k) = S*(k)

    Returns
    -------
    ndarray
        array of size of spectrum with complex amplitudes

    Notes
    -----
    * If the `mirror` flag is to True the return complex amplitude are mirrored over the symmetry
      axis. This allows to feed the resulting complex amplitude to the fft routine

    """

    # determine the omega spacing and use this to scale the complex amplitudes with sqrt(2*S*delta)

    if not mirror:
        if omega.size > 1:
            delta = np.diff(omega)
            delta = np.hstack((delta, np.array([delta[-1]])))
        else:
            delta = np.array([1.0])
    else:
        # for the mirrored version (used for FFT) the delta spacing is uniform and and may have a
        # jump at the Nyquist frequency. Therefore, just construct the array from the first delta
        d0 = omega[1] - omega[0]
        delta = np.full(omega.shape, d0)

    # calculate the full spectrum of all the wave vectors
    Ak = np.sqrt(2 * spectral_modulus * delta) * np.exp(1j * phase)

    if mirror:
        # if you want to feed the amplitudes to an FFT, it needs to be mirrored over the nyquist
        # frequency with A(k)=A^*(-k)
        # see for the definition of the frequencies:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq
        # Important to remember: the zero frequency is at the first index and is never mirrored.
        # So the first half 1...M contains the positive frequencies upto the nyquist freq and the
        # second halve the mirrored version of the first half except the zero
        N = spectral_modulus.size
        if N % 2 == 0:
            # mirror the array with even number of indices
            M = int(N / 2) - 1
            S = 1

        else:
            # mirror the array with odd number of indices
            M = int((N - 1) / 2)
            S = 0

        Ak[M + 1 + S :] = Ak[M:0:-1].conjugate()

    return Ak


def spectrum_complex_amplitudes_on_k_mesh(
    kx_nodes,
    ky_nodes,
    Hs=3,
    Tp=10,
    theta_0=0,
    theta_s_spreading=5.0,
    gamma=3.3,
    sigma=0.0625,
    spectral_version="sim",
    spectrum_type="jonswap",
    seed=None,
):
    """
    Calculate the complex amplitudes based on a spectrum with a random phase on a wave vector mesh

    Parameters
    ----------
    kx_nodes : ndarray
        list of kx nodes. Should start at kx=0
    ky_nodes : ndarray
        list of ky nodes. Should start at ky=0
    Hs : float, optional
        significant wave height (Default value = 3.0)
    Tp : float, optional
        peak period (Default value = 10)
    gamma : float, optional
        peak enhancement factor (Default value = 3.3)
    sigma : float, optional
        Width of the spectrum. Only used when the spectrum_type=gauss spectrum is used
    theta_0 : float, optional
        mean direction theta (Default value = 0.0)
    theta_s_spreading : float, optional
        n factor to control spreading of waves (Default value = 5)
    spectrum_type : {"jonswap", "gauss"}
        type of  spectrum used. Either "jonswap" or "gauss". Default = "jonswap"
    spectral_version : {"sim", "dnv"}
        Which spectral distribution version to use. Default is "sim". For the Jonswap spectrum there
        is virtually no difference between the "sim"  and "dnv" version. For the Gauss spectrum, the
        "sim" version has a fixed spectral width related to the peak period, whereas the "dnv"
        version allows to set the width of the Gauss via the *sigma* parameter. See the function
        *spectrum_jonswap* and *spectrum_gauss* from the *wave_spectra* module for more details.
    seed :
        seed for the random phase (Default value = None)

    Returns
    -------
    ndarray:
        Array  with complex amplitudes on a 2D wave vector mesh

    """

    # get the size and resolution of the kx/ky mesh vectors passed
    nx = kx_nodes.size
    ny = ky_nodes.size
    delta_kx = kx_nodes[1]
    delta_ky = ky_nodes[1]
    dkxdky = delta_kx * delta_ky

    # initialise an array of complex amplitudes
    complex_amplitudes = np.zeros((nx, ny), complex)

    # initialise the random generator
    if seed is None:
        np.random.seed(0)
    else:
        np.random.seed(seed)

    # Calculate the spectrum for all the quadrants over kx. We are going to point mirror later
    for j in range(0, ny, 1):
        for i in range(0, nx, 1):
            (kk, theta) = acf.cartesian_to_polar(kx_nodes[i], ky_nodes[j])

            # the marine definition of theta puts the theta=0 towards the positive y axis and is
            # clockwise rotating
            theta = np.pi / 2 - theta

            if kk == 0.0:
                complex_amplitudes[i, j] = 0.0 + 0.0j
            else:
                # spreading factor
                # Divided by kk in order to convert the Dspread from polar to cartesian coordinates
                Dspread = (
                    spreading_function(
                        theta=theta,
                        theta0=theta_0,
                        s_spreading_factor=theta_s_spreading,
                    )
                    / kk
                )

                spectrum = spectrum_wave_k_domain(
                    kk,
                    Hs=Hs,
                    Tp=Tp,
                    gamma=gamma,
                    sigma=sigma,
                    spectral_version=spectral_version,
                    spectrum_type=spectrum_type,
                )

                # get random phase between 0~2pi.
                phase = 2 * np.pi * np.random.random()

                # create the complex wave amplitude for the current wave vector kx,ky
                amplitude = np.sqrt(2.0 * spectrum * Dspread * dkxdky) * np.exp(
                    1j * phase
                )
                # amplitude is a ndarray of size 1. Get the first value for the complex amplitude
                complex_amplitudes[i, j] = amplitude[0]

    return complex_amplitudes


# @profile
def spectrum2d_complex_amplitudes(
    kx_nodes,
    ky_nodes,
    Hs=1.0,
    Tp=10.0,
    gamma=3.3,
    sigma=0.0625,
    Theta_0=0.0,
    Theta_s_spread_kx=5,
    spectrum_type="jonswap",
    spectral_version="sim",
    mirror=True,
    seed=None,
):
    """
    Calculate the spectral complex amplitude on a cartesian wave vector mesh with a given
    spectral and directional distribution.

    Parameters
    ----------
    kx_nodes : ndarray
        list of kx nodes. Should start at kx=0
    ky_nodes : ndarray
        list of ky nodes. Should start at ky=0
    Hs : float, optional
        significant wave height (Default value = 3.0)
    Tp : float, optional
        peak period (Default value = 10)
    gamma : float, optional
        peak enhancement factor (Default value = 3.3)
    sigma : float, optional
        Width of the spectrum. Only used when the spectrum_type=gauss spectrum is used
    theta_0 : float, optional
        mean direction theta (Default value = 0.0)
    theta_s_spreading : float, optional
        n factor to control spreading of waves (Default value = 5)
    spectrum_type : {"jonswap", "gauss"}
        type of  spectrum used. Either "jonswap" or "gauss". Default = "jonswap"
    spectral_version : {"sim", "dnv"}
        Which spectral distribution version to use. Default is "sim". For the Jonswap spectrum there
        is virtually no difference between the "sim"  and "dnv" version. For the Gauss spectrum, the
        "sim" version has a fixed spectral width related to the peak period, whereas the "dnv"
        version allows to set the width of the Gauss via the *sigma* parameter. See the function
        *spectrum_jonswap* and *spectrum_gauss* from the *wave_spectra* module for more details.
    seed :
        seed for the random phase (Default value = None)

    Returns
    -------
    (c_ampl, s_ampl)
        * c_ampl: 2D ndarray with complex amplitudes
        * s_ampl: 2D ndarray with sign to apply on exp(j omega t) component which should be
          conjugated along with the c_ampl components. Check for instance the
          Wave2D.calculate_omega_dispersion how to use this.

    Notes
    -----

    * This function create a the complex spectral amplitudes on a wave vector mesh where the
      spectrum magnitude is based on a Jonswap of Gauss power spectral density and the phase is
      generated with a random generator.
    * The mesh is cartesian, therefore only the x and y axis of the wave vector space have to be
      passed.
    * The returned complex amplitudes have to be passed to the fft2d in order to calculate the
      wave height in the spatial domain. For that reason it is ensured that the complex amplitude
      are point symmetric around k=0 such that A(-k) = A^*(k) which is required to use the FFT
    * The k-mesh must be defined such that the positive frequency are stored in the first half of
      the array and the negative frequencies are stored swapped on the second half of the array
      See for the details:
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html

    Examples
    --------

    This example shows how to create the complex spetral amplitudes and how to use that to
    calculate a wave surface. First, let's create the complex amplitude on based on a jonswap
    spectrum

    >>> L = 1000
    >>> nx = 64
    >>> ny = 64
    >>> dx = L / nx
    >>> dy = L / ny
    >>> kx_nodes = 2 * np.pi * np.fft.fftfreq(nx, dx)
    >>> ky_nodes = 2 * np.pi * np.fft.fftfreq(ny, dy)
    >>> ak, os = spectrum2d_complex_amplitudes(kx_nodes=kx_nodes, ky_nodes=ky_nodes,
    ...                                        Theta_0=0.0, Hs=2.3, Tp=12.2, gamma=3.3,
    ...                                        Theta_s_spread_kx=5.0, seed=1)

    Use Parseval's theorem to get the variance of the calculated Fourier components and turn it into
    a wave height using hs = 4 sqrt(variance). Note that we divide by 2 because the k-domain stores
    the amplitude mirrored over k=0 so each component is stored twice

    >>> hs_estimate = 4 * np.sqrt(np.square(abs(ak)).sum()) / 2
    >>> print("Hs = {:.1f} m".format(hs_estimate))
    Hs = 2.3 m

    Indeed we can see that the hs obtained from the complex amplitudes correspond with our target Hs

    The complex amplitudes can be used to calculate the wave surface at a certain time t. To do
    that first you need to convert the wave vector numbers to a angular frequency using the deep
    water dispersion relation

    >>> k_mesh = np.meshgrid(kx_nodes, ky_nodes)
    >>> kk = np.sqrt(k_mesh[0] ** 2 + k_mesh[1] ** 2)
    >>> g0 = 9.81
    >>> omega_dispersion = np.sqrt(g0 * abs(kk)) * os

    Now we have omega belonging to each wave vector we can calculate the amplitude of the surface

    >>> n_nodes = int(ak.size / 2)
    >>> for time in np.linspace(start=0, stop=10, num=6, endpoint=True):
    ...    amplitudes = n_nodes * np.fft.ifft2(ak) * np.exp(1j * (-time * os))
    ...    hs_et2 = 4 * np.std(amplitudes)
    ...    print("time = {} s  Hs = {:.1f} m".format(time, hs_estimate))
    time = 0.0 s  Hs = 2.3 m
    time = 2.0 s  Hs = 2.3 m
    time = 4.0 s  Hs = 2.3 m
    time = 6.0 s  Hs = 2.3 m
    time = 8.0 s  Hs = 2.3 m
    time = 10.0 s  Hs = 2.3 m

    The wave surface over the domain of 1000 x 1000 m2 is stored in *amplitudes*. The wave height
    Hs estimated with 4*variance indeed is 2.3 for all time steps

    """

    # get the complex amplitudes for the requested spectrum
    c_ampl = spectrum_complex_amplitudes_on_k_mesh(
        kx_nodes=kx_nodes,
        ky_nodes=ky_nodes,
        Hs=Hs,
        Tp=Tp,
        theta_0=Theta_0,
        theta_s_spreading=Theta_s_spread_kx,
        gamma=gamma,
        sigma=sigma,
        spectrum_type=spectrum_type,
        spectral_version=spectral_version,
        seed=seed,
    )

    # get the size of the wave vector mesh
    nx = kx_nodes.size
    ny = ky_nodes.size

    # see if we are dealing with an even or odd mesh
    nx_is_even = bool(nx % 2 == 0)
    ny_is_even = bool(ny % 2 == 0)

    # create a mesh to store the sign of the fourier components. Required as later as next to the
    # complex conjugate of the c_ampl mesh (where c_ampl(-k) = c_ampl^*(k) we need the conjugates
    # of the exp(-j omega t) as well, so we need to keep which c_ampl's were the mirrored conjugated
    # values
    s_ampl = np.ones((nx, ny), int)

    # get the middle of the mesh depending if we are on an even of an odd mesh
    if nx_is_even:
        # mirror the array with even number of indices
        mx = int(nx / 2) - 1
        sx = 1
    else:
        # mirror the array with odd number of indices
        mx = int((nx - 1) / 2)
        sx = 0

    if ny_is_even:
        # mirror the array with even number of indices
        my = int(ny / 2) - 1
        Sy = 1
    else:
        # mirror the array with odd number of indices
        my = int((ny - 1) / 2)
        Sy = 0

    # make a real copy of the array into c_ampl_1 and c_ampl_2
    c_ampl_1 = c_ampl.copy()
    c_ampl_2 = c_ampl.copy()
    s_ampl_1 = s_ampl.copy()
    s_ampl_2 = s_ampl.copy()

    # Point mirror first the 1st and 2nd quandrant (i.e. kx>0) to the 3rd and fourth quadrant.
    # Then do the same with with the c_ampl_2, but this time we  point mirror the 3rd and 4th
    # quadrant to the to the 1st and 2nd quadrant.
    # After that we can construct to true point mirrored spectrum from c_ampl_1 and c_ampl_2

    # Constructing the other half (complex conjugates) ~ linear vectors first
    # note for even arrays sx = 1
    c_ampl_1[mx + sx + 1 :, 0] = c_ampl[mx:0:-1, 0].conjugate()  # along ky = 0
    c_ampl_1[0, my + Sy + 1 :] = c_ampl[0, my:0:-1].conjugate()  # along kx = 0

    # do the kx = Nx/2 and ny/2 axis only if we have an even number of row/cols
    if nx_is_even:
        # along kx = Nx/2
        c_ampl_1[mx + sx, my + Sy + 1 :] = c_ampl[mx + sx, my:0:-1].conjugate()
    if ny_is_even:
        # along ky = ny/2
        c_ampl_1[mx + sx + 1 :, my + Sy] = c_ampl[mx:0:-1, my + Sy].conjugate()
    #
    # Set the omega values negative for all conjugated complex amplitudes so that is can later
    # be used in the fft
    s_ampl_1[mx + sx + 1 :, 0] = -1
    s_ampl_1[0, my + Sy + 1 :] = -1
    if nx_is_even:
        s_ampl_1[mx + sx, my + Sy + 1 :] = -1
    if ny_is_even:
        s_ampl_1[mx + sx + 1 :, my + Sy] = -1

    # Now for the complex conjugates of the quadrants
    # copy the mirrored kx>0, ky<0 to the kx<0 and ky>0 quadrant
    # the odd/even differences is controlled with the S parameter. notice that the nyquist freq
    # at M+S is not mirrored for odd, whereas it is mirrored for N is even. Look at the
    # definition of the fftfreq to understand
    c_ampl_1[mx + sx + 1 :, 1 : my + 1] = c_ampl[mx:0:-1, ny - 1 : my + Sy : -1].conj()
    s_ampl_1[mx + sx + 1 :, 1 : my + 1] = -1

    # copy the mirrored kx>0, ky>0 to the kx<0,ky<0 quadrant
    c_ampl_1[mx + sx + 1 :, my + Sy + 1 :] = c_ampl[mx:0:-1, my:0:-1].conj()
    s_ampl_1[mx + sx + 1 :, my + Sy + 1 :] = -1

    # point mirror 3rd and 4th to 1st and 2nd quadrait
    # Constructing the other half (complex conjugates) ~ linear vectors first
    c_ampl_2[mx:0:-1, 0] = c_ampl[mx + sx + 1 :, 0].conjugate()  # along ky = 0
    c_ampl_2[0, my:0:-1] = c_ampl[0, my + Sy + 1 :].conjugate()  # along kx = 0
    if sx == 1:
        c_ampl_2[mx + sx, my:0:-1] = c_ampl[
            mx + sx, my + Sy + 1 :
        ].conjugate()  # along kx = Nx/2
    if Sy == 1:
        c_ampl_2[mx:0:-1, my + Sy] = c_ampl[
            mx + sx + 1 :, my + Sy
        ].conjugate()  # along ky = ny/2
    #
    # Set the omega values negative for all conjugated complex amplitudes so that is can later
    # be used in the fft
    s_ampl_2[mx:0:-1, 0] = -1
    s_ampl_2[0, my:0:-1] = -1
    if sx == 1:
        s_ampl_2[mx + sx, my:0:-1] = -1
    if Sy == 1:
        s_ampl_2[mx:0:-1, my + Sy] = -1

    # Now for the complex conjugates of the quadrants
    # copy the mirrored kx>0, ky<0 to the kx<0 and ky>0 quadrant
    # the odd/even differences is controlled with the S parameter. notice that the nyquist freq
    # at M+Sis not mirrored for odd, whereas it is mirrored for N is even. Look at the
    # definition of the fftfreq to understand
    c_ampl_2[mx:0:-1, ny - 1 : my + Sy : -1] = c_ampl[mx + sx + 1 :, 1 : my + 1].conj()
    s_ampl_2[mx:0:-1, ny - 1 : my + Sy : -1] = -1

    # copy the mirrored kx>0, ky>0 to the kx<0,ky<0 quadrant
    c_ampl_2[mx:0:-1, my:0:-1] = c_ampl[mx + sx + 1 :, my + Sy + 1 :].conj()
    s_ampl_2[mx:0:-1, my:0:-1] = -1

    # select the wave vector with the largest magnitude. This construct the full point mirrored
    # spectrum
    mAk1 = abs(c_ampl_1)
    mAk2 = abs(c_ampl_2)
    c_ampl = np.where(mAk1 > mAk2, c_ampl_1, c_ampl_2)
    s_ampl = np.where(mAk1 > mAk2, s_ampl_1, s_ampl_2)

    return c_ampl, s_ampl


def rotate_fft_2d(data2d, angle=0, pivot_x=0, pivot_y=0):
    """Rotate a 2D array around a pivot
    Parameters
    ----------
    data2d: ndarray
        A 2D array
    angle: float, optional
        Angle of rotation. default = 0, so no rotation
    pivot_x: int, optional
        pivot at x-axis, Default =0
    pivot_y: int, optional
        pivot at y-axis, Default =0
    Returns
    -------
    ndarray:
        Rotate 2d array of same shape


    """
    pad_x = [data2d.shape[1] - pivot_x, pivot_x]
    pad_y = [data2d.shape[0] - pivot_y, pivot_y]

    data2d_p = np.pad(fftshift(data2d), [pad_y, pad_x], "constant")
    data2d_r = rotate(data2d_p, angle=angle, reshape=False, order=0)
    data2d_new = data2d_r[pad_y[0] : -pad_y[1], pad_x[0] : -pad_x[1]]
    data2d_new = ifftshift(data2d_new)

    return data2d_new


# @profile
def omega_e_vs_omega(omega, u_monitor):
    """Calculate the the relation between the encountered and true frequency

    Parameters
    ----------
    omega : ndarray of float
        True frequency in rad/s
    u_monitor : float
        The velocity of the monitor point in the direction of the wave vector k in m/s

    Returns
    -------
    float or ndarray:

        Encountered frequency for *omega*

    Notes
    -----

    The following equation is used

    .. math ::

        \omega_e = \omega - k U = \omega - (\omega^2/g) U

    where

    .. math ::

        k = \omega^2/g_0

    and :math:`g_0=9.81~m/s^2` is the gravity constant

    """
    omega_e = omega - omega ** 2 * u_monitor / g0
    return omega_e


# @profile
def d_omega_e_prime(omega, velocity):
    """Calculate the gradient of the relation between the encountered and true frequency

    Parameters
    ----------
    omega : :class:`numpy.ndarray`
        1-D array with true frequency in rad/s
    velocity : :obj:`float`
        Velocity along the wave traveling direction.

    Returns
    -------
    ndarray:
        1-D array with the gradient of the relation between true and encountered frequency

    Notes
    -----

    For deep water waves the relation between the true and encountered frequency is

    .. math::

        \omega_e = \omega - k U = \omega - \\frac{\omega^2 U}{g_0}

    Where

    ============    ====================================================================
    Variable        Meaning
    ============    ====================================================================
    :math:`k`       Wave vector given by deep water dispersion relation :math:`\omega^2/g_0`
    :math:`g_0`     Gravity constant :math:`g_0=9.81` m/s
    :math:`U`       Velocity parallel to wave vector direction. U > 0 is traveling with the wave
    ============    ====================================================================

    The gradient as calculated in this function is therefore

    .. math::

        \\frac{d\omega_e}{d\omega} = 1 - 2 \omega \\frac{U}{g_0}
    """

    d_omega_e_over_d_omega = 1 - 2 * omega * velocity / g0
    return d_omega_e_over_d_omega


def omega_critical(velocity, omega=None):
    """Get the critical frequency where the encountered frequency reaches its maximum value

    Parameters
    ----------
    omega :
        true frequency (Default value = None)
    velocity :
        the velocity of the monitor point in the direction of the wave vector k
    omega : ndarray, optional
        array with true frequencies (Default value = None). If this array is passed, also the index
        of the location where the critical frequency is reached is returned

    Returns
    -------
    tuple
        (omega_c, i_c) Critical encountered frequency plus the location in the *omega* array of the
        critical encountered frequecy. If the *omega* vector was not given, None is returned for
        *i_c*

    Notes
    -----

    The encountered frequency based on the deep water dispersion relation is

    .. math ::

        \omega_e = \omega - k U = \omega - \\frac{\omega^2 U}{g_0}

    The critical frequency where the encountered frequency reaches it maximum is therefore located
    where the derivative of :math:`\omega_e`  is zero , i.e for:

    .. math ::

        \omega_c = \\frac{g_0}{2 U}

    """
    omega_c = g0 / (2 * velocity)

    if omega is not None:
        index_c = np.where(np.diff(np.sign(omega - omega_c)))[0]
    else:
        index_c = None

    return omega_c, index_c


def omega_vs_omega_e(omega, velocity):
    """Calculate the the relation between the true and encountered frequency

    Parameters
    ----------
    omega : :obj:`float`
        true frequency
    velocity : :obj:`float`
        the velocity of the monitor point in the direction of the wave vector k

    Returns
    -------
    tuple:
        (omega_1, omega_2) : 2 real frequencies belonging to the same encountered frequency or
        omega_1, None if only one solution is available or (None, None) if no solutions exist

    Notes
    -----
    The encountered frequency is given by

    .. math ::

        \omega_e = \omega - k U = \omega - (\omega^2/g) U = \omega (1 - \\frac{\omega}{2\omega_c})

    where we assume deep water waves such that

    .. math ::

        k = \omega^2/g_0

    and :math:`\omega_c` is the  critical frequency defined  as

    .. math ::

        \omega_c = g / (2 U)


    Using  the quadratic equation above we can write the true angular frequency as a function of the
    encountered frequency as

    .. math ::

        \omega = \omega_c ( 1 \pm \sqrt{D})

    with

    .. math ::

        D = 1 - \\frac{2\omega_e}{\omega_c}

    For :math:`\omega_e = \omega_c/2` there is exactly one solution: :math:`\omega=\omega_c`.
    For :math:`\omega_e < \omega_c/2` two solution are returned and
    for :math:`\omega_e > \omega_c/2` no solutions exist.

    """
    if abs(velocity) < TINY:
        # for a zero velocity the frequency and encountered frequency are the same and there is only
        # one solution
        omega1 = omega
        omega2 = None
    else:
        # for a positive velocity there may be zero, one or two solutions
        omega_c = g0 / (2 * velocity)
        determinant = 1 - 2 * omega / omega_c
        if determinant >= 0:
            det_sqrt = np.sqrt(determinant)
            omega1 = omega_c * (1 + det_sqrt)
            if abs(determinant) < TINY:
                # only one solution for determinant = 0
                omega2 = None
            else:
                # return second solutions for determinant > 0
                omega2 = omega_c * (1 - det_sqrt)
        else:
            # No solutions for determinant < 0
            omega1 = None
            omega2 = None

    return omega1, omega2


# @profile
def spectrum_to_spectrum_encountered(spectrum, frequencies, velocity, debug_plot=False):
    """Apply a velocity shift on a 1d spectrum density to obtain the encountered spectrum

    Parameters
    ----------
    spectrum : ndarray
        2d array with dimensions n_omega x n_directions
    frequencies : ndarray
        2D array with frequencies in rad/s (created with meshgrid)
    velocity : float
        velocity in direction of wave.
    debug_plot : bool, optional
        create some plots for debugging (Default value = False)

    Returns
    -------
    tuple (spectrum_e_int, info_matrix )
        *spectrum_e_int* is the new spectrum with shifted distribution belonging to a uniformly
        distributed frequency array *frequencies* and info_matrix is a 3 x N matrix with
        three vectors: omega_e (encountered frequencies which are not uniformly distributed),
        omega_e_swap (encountered frequencies swapped around the critical frequency) and
        spectrum_e (encountered spectrum)

    Notes
    -----
    * The encountered frequencies are

      .. math ::

            \omega_e = \omega - U \omega^2 / g_0 = \omega - \\frac{\omega^2}{2\omega_c}

      where :math:`\omega_c` is the critical frequency. It can be seen that the distribution of
      encountered frequencies is not uniform. To deal with this, the returned spectrum_e is put on
      a uniform sampled frequency array which is the same as *frequencies*.
      This done by first converting the encountered spectrum to a cumulative energy,
      then resample this function and turn the cumulative energy into a a density function again.

    * In order to remove the second branch of the encountered frequency, this solution is swapped
      over the critical frequency. This means that for :math:`\omega>\omega_c` the solution is
      not valid but is still returned.

    """

    # calculate the encountered frequencies based on the velocity which varies with the direction
    # over the 2d plane
    omega_e = omega_e_vs_omega(frequencies, velocity)

    # calculate the derivative of the encountered frequencies
    d_omega_e_d_omega = d_omega_e_prime(frequencies, velocity)

    # replace all zero derivatives with a tiny number to prevent an division by zero
    d_omega_e_d_omega = np.where(
        abs(d_omega_e_d_omega) < TINY,
        np.sign(d_omega_e_d_omega) * TINY,
        d_omega_e_d_omega,
    )

    # the critical omega is where the encountered waves show a peak in the spectrum, ie.
    # where d_omega_e_d_omega = 0
    omega_critical = g0 / (2 * velocity)

    # we swap the omega frequency around omega_critical to prevent double frequencies along a
    # direction axis
    omega_e_swap = np.where(d_omega_e_d_omega > 0, omega_e, omega_critical - omega_e)

    # get the gradient in the omega_e direction
    delta_omega_e_swap = np.gradient(omega_e_swap)

    # calculate the transformed spectrum and return
    spectrum_e = spectrum / abs(d_omega_e_d_omega)

    # the transformed spectrum is plotted vs a new frequency axis which is not uniform. Interpolate
    # back to the uniform mesh via the cumulative energy
    cum_sum_energy = np.cumsum(spectrum_e * abs(delta_omega_e_swap), axis=0)

    # the delta frequency adn delta direction of the original input
    df = np.diff(frequencies)[0]

    # interpolate the frequency axis to put the encountered frequency on the same mesh as the
    # original frequencies

    # make interpolation function
    f_inter = interp1d(
        omega_e_swap, cum_sum_energy, bounds_error=False, fill_value="extrapolate"
    )
    cum_sum_energy_new = f_inter(frequencies)

    # go from the cumulative distribution back to the spectral density by taking the gradient and
    # dividing by df
    spectrum_e_int = np.gradient(cum_sum_energy_new / df)

    if debug_plot:
        # use this to make some plots for debugging
        Hs1 = 4 * np.sqrt(spectrum.sum() * df)
        Hs2 = 4 * np.sqrt(spectrum_e_int.sum() * df)
        print("Equivalent Hs : {} {} {}".format(Hs1, Hs2, df))
        fig2 = figure()
        plot(frequencies, spectrum[:, 0], "-r")
        plot(omega_e_swap, spectrum_e[:, 0], "-b")
        plot(frequencies, spectrum_e_int[:, 0], "-og")
        title("Spectra at theta = 0, with Hs1/Hs2 {:.2f} / {:.2f}".format(Hs1, Hs2))
        ioff()
        show()

    return spectrum_e_int, np.vstack((omega_e, omega_e_swap, spectrum_e))


# @profile
def spectrum2d_to_spectrum2d_encountered(
    spectrum_2d, frequencies, directions, velocity, debug_plot=False
):
    """Apply a velocity shift on a 2d spectrum density. It

    Parameters
    ----------
    spectrum_2d : ndarray
        2d array with dimensions n_omega x n_directions
    frequencies : ndarray
        2d array with frequencies in rad/s (created with meshgrid)
    directions : ndarray
        2d array with directions in rad (created with mesh grid
    velocity : float
        velocity in direction of wave.
    debug_plot : bool, optional
        Pop up some debugging plots. (Default value = False)

    Returns
    -------
    ndarray:
        New spectrum with shifted distribution

    Notes
    -----

    It is assumed that the spectrum is defined with respect to the current heading of the ship
    (and not to the north). The velocity is therefore always into the theta=0 direction by
    definition. A peak in the spectrum at the theta=0 means that we have following waves (because k
    is into the positive x-axis and the u-velocity is in the positive x-axis direction)

    """

    # create 2d array of velocities parallel to the wave vector
    u_parallel = np.cos(directions) * velocity

    # calculate the encountered frequencies based on the velocity which varies with the direction
    # over the 2d plane
    omega_e = omega_e_vs_omega(frequencies, u_parallel)

    # calculate the derivative of the encountered frequencies
    d_omega_e_d_omega = d_omega_e_prime(frequencies, u_parallel)

    # replace all zero derivatives with a tiny number to prevent an division by zero
    d_omega_e_d_omega = np.where(
        abs(d_omega_e_d_omega) < TINY,
        np.sign(d_omega_e_d_omega) * TINY,
        d_omega_e_d_omega,
    )

    # the critical omega is where the encountered waves show a peak in the spectrum, ie. where
    # d_omega_e_d_omega = 0
    omega_critical = g0 / (2 * u_parallel)

    # we swap the omega frequency around omega_critical to prevent double frequencies along a
    # direction axis
    omega_e_swap = np.where(d_omega_e_d_omega > 0, omega_e, omega_critical - omega_e)

    # get the gradient in the omega_e direction
    delta_omega_e_swap, grad_y_dummy = np.gradient(omega_e_swap)

    # calculate the transformed spectrum and return
    spectrum_2d_e = spectrum_2d / abs(d_omega_e_d_omega)

    # the transformed spectrum is plotted vs a new frequency axis which is not uniform. Interpolate
    # back to the uniform
    # mesh via the cumulative energy
    cum_sum_energy = np.cumsum(spectrum_2d_e * abs(delta_omega_e_swap), axis=0)

    cum_sum_energy_new = np.zeros(spectrum_2d.shape)
    spectrum_2d_e_int = np.zeros(spectrum_2d.shape)

    # the delta frequency adn delta direction of the original input
    df = np.diff(frequencies[:, 0])[0]
    dd = np.diff(directions[0, :])[0]

    # loop over all the directions and interpolate the frequency axis to put the encountered
    # frequency on the same mesh as the original frequencies
    for cnt in range(directions.shape[1]):
        # make interpolation function
        f_inter = interp1d(
            omega_e_swap[:, cnt],
            cum_sum_energy[:, cnt],
            bounds_error=False,
            fill_value="extrapolate",
        )
        cum_sum_energy_new[:, cnt] = f_inter(frequencies[:, cnt])

        # go from the cumulative distribution back to the spectral density by taking the gradient
        # and dividing by df
        spectrum_2d_e_int[:, cnt] = np.gradient(cum_sum_energy_new[:, cnt] / df)

    if debug_plot:
        # use this to make some plots for debugging
        Hs1 = 4 * np.sqrt(spectrum_2d.sum() * df * dd)
        Hs2 = 4 * np.sqrt(spectrum_2d_e_int.sum() * df * dd)
        print("Equivalent Hs : {} {} {} {}".format(Hs1, Hs2, df, dd))
        fig2 = figure()
        plot(frequencies[:, 0], spectrum_2d[:, 0], "-r")
        plot(omega_e_swap[:, 0], spectrum_2d_e[:, 0], "-b")
        plot(frequencies[:, 0], spectrum_2d_e_int[:, 0], "-og")
        title("Spectra at theta = 0, with Hs1/Hs2 {:.2f} / {:.2f}".format(Hs1, Hs2))
        ioff()
        show()

    return spectrum_2d_e_int


# @profile
def mask_out_of_range(kx, kmin, kmax):
    """Create a mask array with the values in between kmin and kmax True and the rest False


    Parameters
    ----------
    kx : ndarray
        Wave vector array
    kmin : float
        Minimum wave value
    kmax : float
        Maximum wave value

    Returns
    -------
    ndarray
        array of length kx.size with booleans to mask the range

    Notes
    -----

    It is ensured that there is always an even amount of True values

    """

    mask = np.full(kx.shape, True, dtype=bool)
    mask[kx < kmin] = False
    mask[kx > kmax] = False
    if np.count_nonzero(mask) % 2 != 0:
        # and odd number of k vectors should be increased with one k value to get it even
        index = np.where(mask)  # a list with all the True values
        mask[
            index[0][-1] + 1
        ] = True  # the element after the last true index[-1] also to True

    return mask


# @profile
def initialize_phase(k_waves, seed=1):
    """
    Initialise a random phase in in the range :math:`0\sim 2\pi` for all the nodes of the input
    vector *k_waves*

    Parameters
    ----------
    k_waves :
        vector of wave components for which the phase need to be calculated
    seed : int, optional
        See the random generator with an integer such that every run you get the same random results
        for the same seed. Default value = 1. If seed = 0, no seed is used and this will result in
        a different result each random run

    Returns
    -------
    ndarray:
        Array of same length as *k_waves* with a random phase in the range :math:`0\sim 2\pi`

    """
    if seed > 0:
        np.random.seed(seed)
    else:
        # for seed is 0 use random seed
        np.random.seed()

    return 2 * np.pi * np.random.random_sample(k_waves.shape)


# @profile
def specspecs(frequency, amplitude, lowlim=0.01, higlim=0.9, mirror=False):
    """Calculate some specs from a spectrum: the peak frequency, total energy, etc.

    Parameters
    ----------
    frequency : array_like
        the frequency axis
    amplitude : array_like
        the amplitude of the spectral values
    lowlim : float, optional
        Find the point below which the energy is a fraction lowlim of the total. Default = 0.01
    higlim : float, optional
        Find the point below which the energy is a fraction higlim of the total. Default = 0.9
         (Default value = 0.9)
    mirror: bool
        Assumed that we have a mirrored spectral distribution so use half of the domain only

    Returns
    -------
    type
        a tuple with the following values:

        - f_min :  frequency where spectrum first exceeds the threshold (0.01 of the peak value)
        - f_peak : the location of the peak
        - f01 : the frequency below with there is only 1% of the energy
        - f96 : the frequency below with there is 96% of the energy. The interval f01-f96 contains
          95 percent !
        - variance : total energy below spectrum
        - a_peak :  the value at the peak location

    """

    if mirror:
        size = np.asarray(frequency).size // 2
        freq = np.asarray(frequency[:size])
        ampl = np.asarray(amplitude[:size])
    else:
        freq = np.asarray(frequency)
        ampl = np.asarray(amplitude)

    delta_f = np.diff(freq)
    # extend the array with one to get it at the same size
    delta_f = np.hstack((delta_f, np.array([delta_f[-1]])))

    variance = np.dot(ampl, delta_f)
    if variance < 0:
        raise AssertionError("Variance can not be negative")

    peak_index = np.argmax(ampl)

    f_peak = freq[peak_index]
    a_peak = ampl[peak_index]

    pdf = ampl / variance
    cdf = np.cumsum(pdf * delta_f)

    # the index where the amplitude exceeds the 0.5% of the total energy
    i_low = np.argmax(sp.where(cdf > lowlim, 1, 0))
    f_low = freq[i_low]

    # the index where the amplitude exceeds the 95.5% of the total energy
    i_high = np.argmax(sp.where(cdf > higlim, 1, 0))
    f_high = freq[i_high]

    # the interval between f01 and f96 contains 95 percent of the energy of the spectrum

    return i_low, i_high, peak_index, f_low, f_high, f_peak, a_peak, variance


# @profile
def thetaspreadspecs(theta, Dspread, areafraction=0.99):
    """Calculate some specs from a spectrum: the peak frequency, total energy, etc.

    Parameters
    ----------
    theta : ndarray
        Array with directions

    Dspread : ndarray
        Array with distribution function with the property that the integral is one

    areafraction :
         (Default value = 0.99)

    Returns
    -------
    tuple (i_low, i_high, theta_low, theta_high, theta_peak, D_peak, area, mask)
        A tuple contain the following values:

        * i_low :  Index corresponding to theta_low
        * i_high :  Index corresponding to theta_high
        * theta_low :  Frequency where spectrum first exceeds the threshold (0.01 of the peak value)
        * theta_high :  Frequency where spectrum again is below the threshold (0.01 of the peak
          value)
        * theta_peak : the location of the peak of the spreading function
        * D_peak : the spreading value at the peak location theta_peak
        * area : the area below the spreading function supplied. For check only, the area must be 1
    """

    # the spacing in theta domain
    dtheta = theta[1] - theta[0]

    # for check: calculate the integral below the Dspread pdf: should be one
    area = np.sum(Dspread) * dtheta

    # the index of the maximum of the PDF
    peak_index = np.argmax(Dspread)

    # the theta value of the peak index and its maximum
    theta_peak = theta[peak_index]
    D_peak = Dspread[peak_index]

    # shift the peak to the centre of the array
    imiddle = int(round(theta.size / 2))
    ishift = imiddle - peak_index

    # get the zero centre theta.
    thetaprime = np.roll(theta, ishift)
    Dsprime = np.roll(Dspread, ishift)

    # get the pdf of Dsprime. Actually, Dsprime is already a pdf because area should be 1. Anyway,
    # just to be sure
    pdf = Dsprime / area
    cdf = np.cumsum(pdf * dtheta)

    # the areafraction give the total fraction (close to one) which should be contained by the
    # theta_min , theta_max interval.
    lowlim = (1 - areafraction) / 2.0

    # the index where the amplitude exceeds the 0.5% of the total energy
    i_low = np.argmax(sp.where(cdf > lowlim, 1, 0))
    theta_low = thetaprime[i_low]

    # the index of the high boundary is symmetrically located at the other side of the middle (where
    # we moved the peak)
    i_high = np.mod(imiddle + (imiddle - i_low), thetaprime.size)
    theta_high = thetaprime[i_high]

    # create a mask which can later be used to extract the theta angles within the range
    # i_low, i_high
    mask = np.full(theta.shape, True, dtype=bool)
    mask[0 : i_low + 1] = False
    mask[i_high:] = False

    # roll the mask back to its orignal position
    mask = np.roll(mask, -ishift)

    i_low = (i_low - ishift) % theta.size
    i_high = (i_high - ishift) % theta.size

    return i_low, i_high, theta_low, theta_high, theta_peak, D_peak, area, mask


def rayleigh_pdf(omega, sigma):
    """The Raleigh  probability density function.

    Parameters
    ----------
    omega :
        array with the frequencies
    sigma :
        Standard deviation of signal sigma = sqrt(m_0), where m_0 is the zeroth moment

    Returns
    -------
    ndarray:
        Array of same size as *omega* with the Rayleigh distribution


    Notes
    -----

    The matlab uses the CDF, which is related

    """

    pdf = np.exp(-(omega ** 2) / (8 * sigma ** 2)) * omega / (4 * sigma ** 2)

    return pdf


def rayleigh_cdf(omega, sigma):
    """Calculate the Rayleigh cumulative distribution

    Parameters
    ----------
    omega : class:`numpy.ndarray`
        1-D array with the angular frequency in rad/s
    sigma : float
        A control parameter

    Returns
    -------
    class:`numpy.ndarray`
        1-D array with the cumulative distribution function

    Examples
    --------

    Create a array with some radial frequencies running from 0 to 2.5 rad/s

    >>> frequencies = np.linspace(0, 2.5, 10)

    Calculate the Rayleigh cumulative distribution with a width parameter sigma of 3.1


    >>> rayleigh_cdf(omega=frequencies, sigma=3.1)
    array([ 1.        ,  0.99899686,  0.99599345,  0.99100784,  0.98406987,
            0.97522096,  0.9645136 ,  0.95201092,  0.937786  ,  0.9219212 ])

    Notes
    -----

    The Rayleigh Cumulative Distribution is calculated as

    .. math::

       C_R  = \exp\Big[-2\Big(\\frac{\omega}{4\sigma}\Big)^2\Big]

    """

    cdf = np.exp(-2 * (0.25 / sigma) ** 2 * omega ** 2)

    return cdf


def set_heading(data, directions, heading, heading_reverse=True, direction_axis=2):
    """Rotate the data with the heading of the RAO

    Parameters
    ----------
    data : ndarray
        3D array representing the ROA's on a mesh of n_frequencies x n_direction (direction are on
        axis=1). Note however, that during design phase data was passed as a 3D array in which the
        axis=0 direction gave the RAO, so the direction is really on axis=2. In case you have only
        one RAO, then the direction axis should be set to one
    directions : ndarray
        1D array with the direction in the RAO
    heading : float
        Heading in degrees defined as where the bow is going to
    heading_reverse : bool, optional
        In liftdyn a heading of 0 actually corresponds with stern going forward, so with this
        flag we can swap the direction, .i.e. define a header like envy view as e.g. H=10 will
        become H=190 (Default value = True)
    direction_axis : int, optional
        Axis direction over which we roll the RAO. Default is 2

    Returns
    -------
    type
        the rolled RAO for the current heading

    """

    # the heading in lift dyn is just the opposite: a heading of 0 is stern to the front
    if heading_reverse:
        heading += 180

    # note that directions is in radians
    dir_in_range = np.angle(np.exp(1j * directions))
    heading_rad_in_range = np.angle(np.exp(1j * np.deg2rad(heading)))
    index_direction_deg = find_idx_nearest_val(dir_in_range, heading_rad_in_range)
    try:
        data_shifted = np.roll(data, index_direction_deg, axis=direction_axis)
    except ValueError:
        # in case it fails, we assume that we are passing a 2D array (in stead of the assumed 3D
        # array) and that the direction therefore is on one axis lower
        data_shifted = np.roll(data, index_direction_deg, axis=direction_axis - 1)

    return data_shifted


if __name__ == "__main":
    import doctest

    doctest.testmod()
