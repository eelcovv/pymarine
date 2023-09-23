"""

Implementation of the wave field solution of 1D and 2D waves. For the Wave1D and Wave2D class a
power spectral density distribution function :math:`S(k)` can be imposed on the wave nodes such that
we can simulate the propagation of waves obeying either a Jonswap or Gauss spectrum.

The theoretical background has the following contents

* Wave1D_ : Description of the Wave1D class
    - dft1d_ :  Discrete Fourier Transform implementation for 1D wave spectra
    - fft1d_ :  Fast Fourier Transform implementation for 1D waves spectra
* Wave2D_ : Description of the Wave1D class
    - dft2d_ :  Discrete Fourier Transform implementation for 2D wave spectra
    - fft2d_ :  Fast Fourier Transform implementation for 2D wave spectra
* References_ : List of literature used for this implementation


.. _Wave1D:

=============================================
Wave1D: Theoretical background on one-D waves
=============================================

.. _dft1d:

Wave equation with DFT
----------------------

Assume a spectrum :math:`S(k_i)`, with :math:`i=0, n_k`. The wave field :math:`\eta(\mathbf{x}, t)`
at time :math:`t` along spatial positions :math:`x_p` (where :math:`p=0, n_x`) is constructed using
a Discrete Fourier Transform (DFT) as follows

.. math ::

    \eta(x_p, t) = \sum_i^{n_k} a_i \exp(j (k_i x_p - \omega_i t + \phi_i))

where :math:`j^2\equiv -1`, the amplitude :math:`a_i` follow from the power spectral density
:math:`S(\mathbf{k})` as

.. math ::

    a_i = \sqrt{2 S(k_i) dk_i}

:math:`\phi_i` is the random phase corresponding to the wave node :math:`i` and  the angular
frequency :math:`\omega_i` is related to the wave vector via the deep water dispersion relation
according to

.. math ::

    \omega_i =  \sqrt{g  |k_i|}

with :math:`g=9.81 m/s^2` being the gravity constant.
We can rewrite the wave equation in matrix form:

.. math ::

    \eta(x_p, t) = \sum_i^{n_k} M_{ip}\exp(-j \omega_i t)

where the matrix :math:`M_{ip}` is

.. math ::

    M_{ip} = \hat{A}_i \exp(j k_i x_p)

and the complex  amplitude :math:`\hat{A}_i` follows from

.. math ::

    \hat{A}_{i} = a_i\exp(j\phi_i)

From the matrix form is can be seen that the wave equation can calculated  by a dot product
multiplication of the :math:`n_x \\times n_k` matrix :math:`M_{ip}` and the vector
:math:`\exp(-j\omega_i t)` of size :math:`n_k \\times 1`. The result is a :math:`1 \\times n_x`
vector of the wave along the x-axis :math:`\eta(x_p, t)`. This is how the DFT is implemented in the
code of Wave1D.

Note that we have described the wave equation in terms of the power spectral density (PSD) equation
in the wave vector domain :math:`k`, denoted here as :math:`S(k)`.  Normally the PSD is given in the
angular frequency domain :math:`\omega` as :math:`S^\prime(\omega)` We can obtain the version in
wave vector domain using the equality

.. math ::

    S(k) = S^\prime(\omega)\\frac{d\omega}{dk} = S^\prime(\sqrt{g k}) \\frac{g}{2 k}.


Using the PSD in the wave vector domain has the advantage that we only have to deal with the
periodicity of the discrete Fourier transform in the spatial domain x: a wave will be period after a
length :math:`2\pi/\Delta k`. To get rid of this periodicity is we can simply use a spatial domain
which larger than this repetition length. The waves in time domain, however, are never periodic in
this formulation, so we can run the simulated wave infinitely long without seeing the same signal
twice.

.. _fft1d:

Wave equation with FFT
----------------------

In case the number of wave numbers nodes is equal to the number of spatial nodes and we have ensured
that

.. math ::

    \hat{A}(-k_i)\exp(-j\omega t) = \hat{A}^\\ast(k_i)\exp(j\omega t)

we can use the FFT algorithm for calculating the wave equation at time :math:`t`.  This is because
the FFT uses the symmetry rule that for any *real* valued signal :math:`F(x)`, the Fourier transform
:math:`\hat{F}(k)`  by definition has the property that :math:`\hat{F}(-k) = \hat{F}^\\ast(k)`
(where the :math:`\\ast` indicates the complex conjugate). The FFT is a  much more efficient way to
obtain a solution for the wave equation: the calculation time for a DFT  of a signal with :math:`N`
nodes is  proportional to :math:`N^2`, while the calculation time of an FFT is proportional to
:math:`N\log(N)`. The FFT is used for the Wave1D solution when the *wave_construction*  field is
set to *FFT*.

.. _Wave2D:

=============================================
Wave2D: Theoretical background on two-D waves
=============================================

Wave Wave2D Class give a description of a 2D propagating wave with a spectral density distribution
given by either a Gauss or a Jonswap spectrum. The class extents the functionality of the Wave1D:
it obtains all the information related to the 1D wave in k-vector domain from the wave1D attribute
field which stores a Wave1D object. For all the information additionally required for the 2D
description (like spreading function), extra attribute fields are added.

Again, three ways to solve the wave field from the spectral components have been implemented:

* *DFTpolar*: A 2D wave field is constructed from the multiplication of the Spectral density S(k)
  and the directional distribution D(theta). A Discrete Fourier Transform (DFT) is used to calculate
  the wave field. This is a straightforward implementation, but again, DFT is `slow` to calculate,
  especially in 2D, so only use this for testing. You can speed up the calculation by selecting wave
  node with an amplitude larger than a certain threshold.
* *FFT*: if a symmetry in spectral k domain is imposed we can again use the FFT to calculate the
  wave field. Recommended
* *DFTcartesian*: The exact same symmetric spectral amplitudes as the FFT is used, but the wave is
  calculate with a DFT. Slow and no possibility of wave selection, so only used for testing purposes

.. _dft2d:

2D DFT implementation
---------------------

A 2D wave can be described by its spectral density distribution :math:`S(k)` and the directional
distribution :math:`D(\\theta)` as

.. math ::

    E_{k, \\theta}(k, \\theta) = S(k)D(\\theta)

where for :math:`S(k)` we can either take a Jonswap or a Gauss distribution as implemented in the
*wave_spectra* module in *spectrum_jonswap* and *spectrum_gauss*. Since we are working in the wave
vector domain :math:`k`,  we must first transform the spectrum from :math:`\omega` to :math:`k`
space, just as described at the 1D wave theory. For directional distribution the
*spreading_function* is recommended.

The *DFTpolar*  implementation constructs the complex amplitudes by multiplying the spectral density
:math:`E_{k, \\theta}` with its bin width :math:`dk d\\theta`:

.. math ::

    \hat{A}_{k, \\theta} = \sqrt{2 E_{k, \\theta}(k, \\theta) dk d\\theta} \exp({j\phi})

where the phase :math:`\phi` is a random number between :math:`0\sim 2\pi`.
The wave amplitude is then calculate by simply summing over all the wave components we have added,
similar to our 1D version

.. math ::

    \eta(\mathbf{x}, t) = \sum_p^{n_{k}}\sum_q^{n_{\\theta}} \hat{A}_{k_p, \\theta_q}
    \exp(j (\mathbf{k}_{p,q}\cdot\mathbf{x}- \omega(k_p) t))

where :math:`\mathbf{k} = (k_x, k_y)` is the 2D wave vector in cartesian coordinates belonging to
the sample wave vector in polar coordinates :math:`(k_p, \\theta_q)` with :math:`k=|\mathbf{k}|`.
The wave vector magnitude, :math:`\mathbf{x}` is the cartesian vector  in spatial domain.  Also we
are using the deep water dispersion relation again to calculate the angular wave frequency
:math:`\omega` for a given wave vector :math:`k`.

The DFT implementation can be sped up significantly by smartly selecting nodes, for instance based
on the components above a certain threshold, or we could make a non-uniform distribution. Both
methods have been implemented and can be selected by settings the *wave_selection* attribute field
to  *Subrange* or *EqualEnergyBins*, respectively.

.. _fft2d:

2D FFT implementation
---------------------

As already mentioned, the DFT implementation is very slow. We can speed up by selecting wave nodes
in a smart way, but this consequently also takes out information from our resulting wave field. On
top of that: it turns out that even after selecting only the most important spectral wave
components, the DFT still is not very fast.

The recommended way to construct a wave field is again by using the FFT setting for the
*wave_construction* argument. There is only one catch: the FFT algorithm requires the spectral
components described in the cartesian wave vector domain, whereas so far we have been working in the
polar wave vector domain. On top of that, we have to impose the symmetry required for a FFT:

.. math ::

    \hat{A}(-\mathbf{k})\exp(-j\omega t)= \hat{A}^\\ast(\mathbf{k})\exp(j\omega t)

This symmetry rule is the 2D version from the one we have seen in the 1D wave section.

For the FFT method we need a describtion of the complex amplitudes in the Cartisian domain, i.e. we
need to derive :math:`E_{k}(\mathbf{k})` from our starting point `E_{k, \\theta}(k, \\theta)`. It
can be shown that  we need to do

.. math ::

    E_{k}(\mathbf{k}) = \\frac{E_{k, \\theta}(k, \\theta)}{k}

This equation is used in the function *spectrum2d_complex_amplitudes* as we divide the directional
spreading function with the wave vector magnitude. For a more detailed derivation, please have a
look at the References_.

.. _References:

References
----------

* Det Norske Veritas: Environmental conditions and environmental loads, April 2007, DNV-RP-C205
* Jocely Fr\'echot: Realistic simulation of ocean surface using wave spectra. Proceedings of the
  First International Conference on Computer Graphics Theory and Applications (GRAPP, 2006),
  2006, Portugal, p 76--83 <hal-00307938>

=========================================
Start of the *wave_fields* implementation
=========================================

"""

import logging
from builtins import object
from builtins import range
from os.path import splitext

import colorcet as cc
import h5py
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from pymarine.utils.coordinate_transformations import polar_to_cartesian
from pymarine.utils.numerical import find_idx_nearest_val
from pymarine.utils.plotting import set_limits, clean_up_artists
from matplotlib.colors import LightSource
from numpy.fft import fftshift
from scipy.constants import g as g0  # gravity constant 9.81 m/s2

import pymarine.waves.wave_spectra as ms

sns.set(context="notebook")

logger = logging.getLogger(__name__)


class PlotProperties(object):
    """
    A class containing the properties of the line plot.

    Parameters
    ----------
    show: bool, optional,
        Show the current wae
    color: str, optional
        Store the color of the line plotting the wave. Default = "w"
    linewidth: int, optional
        Width of the line plotting the wave. Default = 1
    scattersize: int, optional
        Size of the scatter points plotted at the wave. Default = 0, i.e. no scatter points are
        plotted

    Notes
    -----
    * Only used by WaveSimulator to connect plot properties to each wave via the *plot* attribute
      field of the wave
    * Not critical and perhaps this class should be moved out of the *wave_fields* module later. For
      now, leave it as it is. It doesn't hurt me as well.
    """

    def __init__(self, show=True, color="w", linewidth=1, scattersize=0):
        self.show = show
        self.color = color
        self.linewidth = linewidth
        self.scattersize = scattersize
        self.save_image = False
        self.save_data = False


# @profile
def _energy_deficit(k, k0, E, Hs, Tp, gamma, spectral_version, spectrum_type, sigma):
    """
    Helper function for fsolve to calculate the integral of the spectrum between k0 and k1 and
    subtract E

    Parameters
    ----------
    k: ndarray
       Array with the wave vector  nodes
    k0: float
       Start integrating at wave vector k0
    E:
        Energy per bin
    Hs:
        Significant wave height
    Tp:
        Peak period
    gamma:
        Peakness of Jonswap
    spectral_version:
        Version of implementation used
    spectrum_type: {"jonswap", "gauss"}
        Type of wave spectrum. Default = jonswap
    sigma:
        Width of the Gauss

    Returns
    -------
    float:
        Energy dificit

    Notes
    -----

    This function is used a helper function for fsolve and should not be called directly. It is used
    to calculate the integrated energy of a bin between k and k+dk

    """

    # solve the integral of the spectrum between k0 and k1 and subtract E and returns the deficit
    # energy this function can be used to calculate the end border of an integration interval such
    # that the area of the integral results in a given value E
    (delta_e, err) = (
        sp.integrate.quad(
            ms.spectrum_wave_k_domain,
            k0,
            k,
            args=(Hs, Tp, gamma, sigma, spectrum_type, spectral_version),
        )
        - E
    )
    return delta_e


class Wave2D(object):
    """
    A class for linearized solutions of the 2D wave (deep water, linear). The Wave1D is used to
    describe the radial direction

    Parameters
    ----------
    wave1D: :obj:`Wave1D`
        The Wave1D field to describe the spectral properties along the radial direction in k-domain
        the 2D wave
    Lx: float, optional
        Length of the domain in x-direction in m, default = 200 m
    Ly: float, optional
        Length of the domain in y-direction in m, default = 200 m
    xmin: float, optional
        Starting coordinate of the x-domain. Default = 0 m
    ymin: float, optional
        Starting coordinate of the y-domain. Default = 0 m
    nx_points: int, optional
        Number of nodes in the x-domain. Default = 64
    ny_points: int, optional
        Number of nodes in the y-domain. Default = 64
    kx_min: float, optional
        Start of wave vector domain. Default = 0 rad/m
    kx_max: float, optional
        End of wave vector domain. Default = pi rad/m
    delta_kx: float, optional
        Bin spacing in wave vector domain. Default is pi/32
    n_kx_nodes: int, optional
        Number of bins in in x direction in k vector domain. Default = 64
    ky_min: float, optional
        Start of wave vector domain. Default = 0 rad/m
    ky_max: float, optional
        End of wave vector domain. Default = pi rad/m
    delta_ky: float, optional
        Bin spacing in wave vector domain. Default is pi/32
    n_ky_nodes: int, optional
        Number of bins in y direction in k vector domain. Default = 64
    theta_min: float, optional
         Minimum value of theta direction in polar domain. Default = 0 rad
    theta_max: float, optional
         Maximum value of theta direction in polar domain. Default = 2 pi rad
    n_theta_nodes: int, optional
         Number of nodes in theta direction. Default = 100
    Theta_0: float, optional
         Main wave direction in rad. Default = 0 rad. The valid range is: -pi/2 < T < pi
    Theta_s_spreading_factor: float, optional
        Spreading factor (s-definition) in theta domain. Default = 5 (Typical for wind waves, for
        swell use 13.0)
    """

    def __init__(
        self,
        wave1D,
        name=None,
        Lx=200,
        Ly=200,
        xmin=0,
        ymin=0,
        nx_points=64,
        ny_points=64,
        kx_min=0.0,
        kx_max=3.1415,
        delta_kx=0.0981,
        n_kx_nodes=32,
        ky_min=0.0,
        ky_max=3.1415,
        delta_ky=0.0981,
        n_ky_nodes=32,
        theta_min=0,
        theta_max=2 * np.pi,
        n_theta_nodes=100,
        Theta_0=0,
        Theta_s_spreading_factor=5,
    ):

        logger.info("Initialise JonSwap 1D wave field")

        if name is None:
            self.name = "_".join(
                [
                    "wave",
                    wave1D.spectrum_type,
                    wave1D.wave_construction,
                    wave1D.wave_selection,
                    "2d",
                ]
            )
        else:
            self.name = name

        self.wave1D = wave1D
        self.Lx = Lx
        self.Ly = Ly
        self.nx_points = nx_points
        self.ny_points = ny_points

        self.xmin = xmin
        self.Lx = Lx
        self.xmax = self.xmin + self.Lx
        self.xmid = self.xmin + self.Lx / 2.0
        self.xpoints = np.linspace(self.xmin, self.xmax, self.nx_points, endpoint=True)
        self.delta_x = self.xpoints[1] - self.xpoints[0]

        self.ymin = ymin
        self.Ly = Ly
        self.ymax = self.ymin + self.Ly
        self.ymid = self.ymin + self.Ly / 2.0
        self.ypoints = np.linspace(self.ymin, self.ymax, self.ny_points, endpoint=True)
        self.delta_y = self.ypoints[1] - self.ypoints[0]

        self.n_theta_nodes = n_theta_nodes
        self.Theta_0 = Theta_0
        self.Theta_s_spreading_factor = Theta_s_spreading_factor

        self.theta_points = None
        self.kx_nyquist = None
        self.kt = None
        self.delta_theta = None

        # theta=0 corresponds to north direction. Rotate clock wise
        self.theta_min = theta_min
        self.theta_max = theta_max

        self.kx_min = kx_min
        self.kx_max = kx_max
        self.delta_kx = delta_kx
        self.n_kx_nodes = n_kx_nodes
        self.k_polar_mesh = None
        self.k_polar_bin_area_over_kk = None
        self.omega_dispersion = None
        self.E_wave_density_polar = None

        self.ky_min = ky_min
        self.ky_max = ky_max
        self.delta_ky = delta_ky
        self.n_ky_nodes = n_ky_nodes
        self.k_xy_mesh = None

        self.kx_nodes = None
        self.ky_nodes = None

        self.k_cartesian_mesh = None
        self.kk = None
        self.E_wave_complex_amplitudes = None
        self.omega_sign = None

        self.xy_mesh = None

        self.amplitude = None

        self.phase = None

        self.theta_area_fraction = 1

        # use seed to update the random phase if required
        self.seed = 1
        self.update_phase = True

        self.update_x_k_theta_sample_space()
        self.D_spread = np.zeros(self.theta_points.shape)
        self.calculate_spreading_function()
        self.calculate_spectral_components()
        self.calculate_wave_surface()

    def make_report(self):
        """Make report of settings for this wave"""

        frm = "{:40s} : {}"

        print(frm.format("Name", self.name))
        print("----------- Domain settings --------")
        print(frm.format("Start x [m]", self.xmin))
        print(frm.format("Start y [m]", self.ymin))
        print(frm.format("End x [m]", self.xmax))
        print(frm.format("End y [m]", self.ymax))
        print(frm.format("Length x [m]", self.Lx))
        print(frm.format("Length y [m]", self.Ly))
        print("----------- Time settings --------")
        print(frm.format("Start t [s]", self.wave1D.t_start))
        print(frm.format("End t [s]", self.wave1D.t_end))
        print(frm.format("Length t [s]", self.wave1D.t_length))
        print(frm.format("Delta t [s]", self.wave1D.delta_t))

        n_x_points_total = self.xpoints.size * self.ypoints.size
        n_k_points_total = self.k_cartesian_mesh[0].size

        print("----------- Numerical resolutions --------")
        print(frm.format("Number x - nodes", self.xpoints.size))
        print(frm.format("Number y - nodes", self.ypoints.size))
        print(frm.format("Total number of spatial points", n_x_points_total))
        if self.kx_nodes is not None:
            print("# Cartesian mesh specifications")
            print(frm.format("Number kx - nodes", self.kx_nodes.size))
            print(frm.format("Number ky - nodes", self.ky_nodes.size))
            print(frm.format("Delta x [m]", self.delta_x))
            print(frm.format("Delta y [m]", self.delta_y))
            dkx = np.diff(self.kx_nodes)
            dky = np.diff(self.ky_nodes)
            print(frm.format("Delta kx min [rad/m]", dkx.min()))
            print(frm.format("Delta ky min [rad/m]", dky.min()))
            print(frm.format("Delta kx max [rad/m]", dkx.max()))
            print(frm.format("Delta ky max [rad/m]", dky.max()))
            print(frm.format("Delta kx first [rad/m]", self.delta_kx))
            print(frm.format("Delta ky first [rad/m]", self.delta_ky))
            print(frm.format("kx-min [rad/m]", self.kx_nodes.min()))
            print(frm.format("ky-min [rad/m]", self.ky_nodes.min()))
            print(frm.format("kx-max [rad/m]", self.kx_nodes.max()))
            print(frm.format("ky-max [rad/m]", self.ky_nodes.max()))
            print(frm.format("kx-nyq [rad/m]", self.kx_nyquist))
            print(frm.format("ky-nyq [rad/m]", self.ky_nyquist))
            print(frm.format("Lx_max [m]", 2 * np.pi / dkx[0]))
            print(frm.format("Ly_max [m]", 2 * np.pi / dky[0]))
        else:
            print("# Polar mesh specifications")
            print(frm.format("Number k_r - nodes", self.k_polar_mesh[1].shape[0]))
            print(frm.format("Number theta - nodes", self.k_polar_mesh[0].shape[1]))
            print(frm.format("Number total nodes", n_k_points_total))

        print("----------- Numerical methods --------")
        print(frm.format("Selection method", self.wave1D.wave_selection))
        print(frm.format("Construction method", self.wave1D.wave_construction))
        print(frm.format("DFT N x N", n_x_points_total * n_k_points_total))
        print(frm.format("FFT N x log(N)", n_x_points_total * np.log(n_x_points_total)))

    def update_k_polar_mesh(self):
        """Update the polar mesh and its bin area divided by k belonging to the current k/theta mesh

        Notes
        -----
        The bin area follows from the theta and k linear array as k x dtheta x dk
        However,  to calculate the complex wave amplitude we need to divede D(omega, kk) by kk
        in order to go the cartesian mesh. Here by not multiplying with kk to get the bin
        Therefore we calculate darea / kk = kk dtheta * dkk / kk = dtheta x dkk

        """
        # create polar mesh if direction x kk_magnitude and store in k_polar_mesh.
        # k_polar_mesh[0] are the directions over the 2D mesh, k_polar_mesh[1] are the wave
        # vector magnitudes over the 2D mesh
        self.k_polar_mesh = np.meshgrid(self.theta_points, self.wave1D.kx_nodes)

        # k_polar_mesh[0] is the theta over the 2D plane, so the delta theta is the gradient-y
        delta_theta = np.gradient(self.k_polar_mesh[0])[1]
        # k_polar_mesh[1] is the kr over the 2D plane, so the delta kr is the gradient-x
        delta_k_r = np.gradient(self.k_polar_mesh[1])[0]

        # the  area of a polar mesh is k * dtheta * dk and dived by k, so the kmagnitude drops here
        self.k_polar_bin_area_over_kk = delta_theta * delta_k_r

        # turn the 2D polar coordinates into cartesian coordinates convert the polar angle from
        # mathematical definition (theta=0 -> x-axis and counter-clock rotation) to naval
        # definition (theta=0 is y-axis and clock-wise rotation) theta_naval=pi/2-theta
        self.k_cartesian_mesh = np.array(
            polar_to_cartesian(self.k_polar_mesh[1], np.pi / 2 - self.k_polar_mesh[0])
        )
        self.kk = np.sqrt(self.k_cartesian_mesh[0] ** 2 + self.k_cartesian_mesh[1] ** 2)

    # @profile
    def update_x_k_theta_sample_space(self):

        self.theta_points = np.linspace(
            self.theta_min, self.theta_max, self.n_theta_nodes, endpoint=True
        )

        self.delta_theta = self.theta_points[1] - self.theta_points[0]

        self.xmax = self.xmin + self.Lx

        self.xmid = self.xmin + self.Lx / 2.0

        self.xpoints = np.linspace(self.xmin, self.xmax, self.nx_points, endpoint=True)

        self.delta_x = self.xpoints[1] - self.xpoints[0]

        self.ymax = self.ymin + self.Ly

        self.ymid = self.ymin + self.Ly / 2.0

        self.ypoints = np.linspace(self.ymin, self.ymax, self.ny_points, endpoint=True)

        self.delta_y = self.ypoints[1] - self.ypoints[0]

        self.xy_mesh = np.meshgrid(self.xpoints, self.ypoints, indexing="ij")

        self.amplitude = np.zeros(self.xy_mesh[0].shape)

        self.kx_nyquist = np.pi / self.delta_x
        self.ky_nyquist = np.pi / self.delta_y

        if self.wave1D.wave_construction == "DFTpolar":

            # take the (non)-uniform wave vectors from the 1D wave
            self.update_k_polar_mesh()

        else:
            # for the FFT the number wave vectors should be equal to the number of x points
            # for the DFT on the cartesian mesh we use the same mesh as the FFT, so we can compare
            # the speed of the algorithms
            self.kx_nodes = 2 * np.pi * np.fft.fftfreq(self.nx_points, self.delta_x)
            self.ky_nodes = 2 * np.pi * np.fft.fftfreq(self.ny_points, self.delta_y)

            self.delta_kx = self.kx_nodes[1] - self.kx_nodes[0]
            self.delta_ky = self.ky_nodes[1] - self.ky_nodes[0]

            # create the mesh [KX,KY]
            self.k_xy_mesh = np.meshgrid(self.kx_nodes, self.ky_nodes, indexing="ij")
            self.k_cartesian_mesh = self.k_xy_mesh
            self.kk = np.sqrt(
                self.k_cartesian_mesh[0] ** 2 + self.k_cartesian_mesh[1] ** 2
            )

    # @profile
    def calculate_spreading_function(self):

        self.D_spread = ms.spreading_function(
            theta=self.theta_points,
            theta0=self.Theta_0,
            s_spreading_factor=self.Theta_s_spreading_factor,
        )

        (
            self.itheta_low,
            self.itheta_high,
            self.theta_low,
            self.theta_high,
            self.theta_peak,
            self.D_peak,
            self.D_spread_area,
            self.theta_mask,
        ) = ms.thetaspreadspecs(
            self.theta_points, self.D_spread, self.theta_area_fraction
        )

        # check if we need to mask the points
        mask_points = False
        if self.wave1D.wave_selection == "Subrange":
            mask_points = True
        if (
            self.wave1D.wave_selection == "EqualEnergyBins"
            and self.wave1D.use_subrange_energy_limits
        ):
            mask_points = True

        if mask_points:
            # extra wave vectors based on the range k_low,k_high
            self.theta_points = np.extract(self.theta_mask, self.theta_points)
            self.D_spread = np.extract(self.theta_mask, self.D_spread)

            if self.wave1D.sample_every > 1:
                self.theta_points = self.theta_points[:: self.wave1D.sample_every]
                self.D_spread = self.D_spread[:: self.wave1D.sample_every]

            # we need to update the k_polar_mesh as well
            self.update_k_polar_mesh()

        if self.phase is not None:
            if (
                self.phase.shape[0] != self.wave1D.n_kx_nodes
                or self.phase.shape[1] != self.theta_points.size
            ):
                phase_shape_mismatch = True
            else:
                phase_shape_mismatch = False
        if self.phase is None or phase_shape_mismatch or self.update_phase:
            self.E_wave_density_polar = np.zeros(self.k_cartesian_mesh[0].shape)
            self.phase = ms.initialize_phase(
                self.E_wave_density_polar, self.wave1D.seed
            )
            self.update_phase = False

    # @profile
    def calculate_spectral_components(self):
        """
        Calculate the 2D wave density function from the current k-array and spreading angles
        """

        if self.wave1D.wave_construction == "DFTpolar":
            # the DFT based on a polar mesh is used. So calculate the polar coordinates and turn it
            # in cartesian values

            self.omega_dispersion = np.sqrt(
                self.wave1D.gravity0 * abs(self.k_polar_mesh[1])
            ) * sp.where(self.k_polar_mesh[1] >= 0, 1, -1)

            # the wave density function follows from Sk*Dspread. To get a matrix out of it, multiply
            # the transposed S^T with the D
            n_k = self.wave1D.spectrumK.size
            n_d = self.D_spread.size
            self.E_wave_density_polar = self.wave1D.spectrumK.reshape(
                n_k, 1
            ) * self.D_spread.reshape(1, n_d)

            # in polar domain, the integral multiplied with delta_theta*delta_kx give the complex
            # amplitude a_k = sqrt(2 S(k, theta) dk * dtheta
            self.E_wave_complex_amplitudes = np.sqrt(
                2 * self.E_wave_density_polar * self.k_polar_bin_area_over_kk
            ) * np.exp(1j * self.phase)
        else:
            # either a DFT (wave_construction==DFTcartesian) or a FFT (wave_construction==FFT)
            # based on a cartesian mesh is used.
            self.k_cartesian_mesh = self.k_xy_mesh
            self.kk = np.sqrt(self.k_xy_mesh[0] ** 2 + self.k_xy_mesh[1] ** 2)
            self.kk = np.where(
                abs(self.kk) < ms.TINY, ms.TINY * np.ones(self.kk.shape), self.kk
            )

            (
                self.E_wave_complex_amplitudes,
                self.omega_sign,
            ) = ms.spectrum2d_complex_amplitudes(
                kx_nodes=self.kx_nodes,
                ky_nodes=self.ky_nodes,
                Hs=self.wave1D.Hs,
                Tp=self.wave1D.Tp,
                gamma=self.wave1D.gamma,
                Theta_0=self.Theta_0,
                Theta_s_spread_kx=self.Theta_s_spreading_factor,
                spectrum_type=self.wave1D.spectrum_type,
                spectral_version=self.wave1D.spectral_version,
            )

            k_bin_area = self.delta_kx * self.delta_ky
            # area_polar = self.wave1D.kx_nodes * self.k_polar_bin_area_over_kk
            np_e_sq = np.square(abs(self.E_wave_complex_amplitudes))
            self.E_wave_density_polar = np_e_sq / (k_bin_area / 2) / self.kk

            # calculate the omega values belong the the wave vectors
            self.calculate_omega_dispersion()

    # @profile
    def calculate_omega_dispersion(self):
        # calculate the omega rrequency belonging to the wave vectors according to the deep water
        # dispersion relation.
        self.omega_dispersion = np.sqrt(g0 * abs(self.kk)) * self.omega_sign
        self.delta_omega = np.diff(self.omega_dispersion)

    # @profile
    def calculate_wave_surface(self):

        if not self.wave1D.wave_construction == "FFT":
            # For the DFT directly calculate the wave field from the spectral components
            self.amplitude = self.dft_complex_amplitudes(
                self.E_wave_complex_amplitudes, self.omega_dispersion, self.wave1D.time
            )
            if self.wave1D.wave_construction == "DFTcartesian":
                # scale the amplitude with a factor 2 because we used the two-side k-space
                self.amplitude *= 0.5
        else:
            # get the wave field using an FFT
            self.amplitude = self.fft_amplitude(
                self.E_wave_complex_amplitudes, self.omega_dispersion, self.wave1D.time
            )

        logger.debug("H_s of 2D surface {}  ".format(4 * np.std(self.amplitude)))

    # @profile
    def dft_complex_amplitudes(self, S_tilde, omega, time):
        """Calculate DFT of complex amplitudes at time 'time'

        Parameters
        ----------
        S_tilde: Complex amplitude obtained from
        omega: angular frequency for each node
        time: current time

        Returns
        -------
        ndarray:
            2D array with Discrete fourier transform

        Notes
        -----
        * The trick with the exponential matrix exp (j*x*k) does not work in 2D because you run out
          of memory too fast. Therefore, calculate the wave field with a loop over the wave vectors.
        * This algorithm is really slow, so you should use FFT for 2D waves!

        """

        dft = np.full(self.xy_mesh[0].shape, 0 + 0 * 1j, dtype=complex)
        KX = self.k_cartesian_mesh[0]
        KY = self.k_cartesian_mesh[1]
        for j in range(self.k_cartesian_mesh[0].shape[1]):
            for i in range(self.k_cartesian_mesh[0].shape[0]):
                s_theta = S_tilde[i, j]
                dft += s_theta * np.exp(
                    1j
                    * (
                        self.xy_mesh[0] * KX[i, j]
                        + self.xy_mesh[1] * KY[i, j]
                        - omega[i, j] * time
                    )
                )
        return np.real(dft)

    # @profile
    def fft_amplitude(self, S_tilde, omega, time):
        """
        Calculate Fourier transform of S using the FFT
        Parameters
        ----------
        S_tilde: ndarray, complex
            Complex array of the fouriern components
        omega: ndarray
            Angular freqyencies belong to the wave vectors
        time: float
            Current time

        Returns
        -------
        ndarray
            real array with the DFT of the complex amplitudes

        """
        N = int(S_tilde.size / 2)
        ampl = N * np.fft.ifft2(S_tilde * np.exp(1j * (-time * omega)))
        # ampl should be real already because S_tilde should by symmetrical around k=0 S(k)=S^*(-k)
        # to be sure, take the real value only
        return np.real(ampl)

    # @profile
    def export_complex_amplitudes(self, filename, exportAsHD5=True):
        """Export the calculate complex amplitudes to HDF 5 file

        Parameters
        ----------
        filename: str
            Name of the file
        exportAsHD5: bool, optional
            Export as HD5. Default = True. If false, the complex amplitudes are written to Ascii

        """
        if exportAsHD5:
            filebase, ext = splitext(filename)
            logger.debug("writing hd5 file to {}".format(filebase))
            with h5py.File(filebase + ".h5", "w") as hf:
                hf.create_dataset("Kx", data=self.k_cartesian_mesh[0])
                hf.create_dataset("Ky", data=self.k_cartesian_mesh[1])
                hf.create_dataset("Amodulus", data=abs(self.E_wave_complex_amplitudes))
                hf.create_dataset(
                    "Aphase", data=np.angle(self.E_wave_complex_amplitudes)
                )
                hf.create_dataset("omega", data=self.omega_dispersion)

        else:
            # write the complex amplitudes to the file filename in ascii format
            logger.debug("writing numpy ascii file to {}".format(filename))

            nx = self.E_wave_complex_amplitudes.shape[0]
            ny = self.E_wave_complex_amplitudes.shape[1]
            KX = self.k_cartesian_mesh[0]
            KY = self.k_cartesian_mesh[1]
            with open(filename, "w") as f:
                f.write(
                    "# complex amplitudes a at kx,ky mesh {}x{} kxtheta\n".format(
                        nx, ny
                    )
                )
                f.write(
                    "# {:>18s}{:>20s}{:>20s}{:>20s}{:>20s}\n".format(
                        "kx", "ky", "real(a)", "imag(a)", "omega"
                    )
                )
                for j in range(ny):
                    f.write("# j={}\n".format(j))
                    for i in range(nx):
                        ampl = self.E_wave_complex_amplitudes[i][j]
                        f.write(
                            "{:20.8g}{:20.8g}{:20.8g}{:20.8g}{:20.8g}\n".format(
                                KX[i, j],
                                KY[i, j],
                                float(np.real(ampl)),
                                float(np.imag(ampl)),
                                float(self.omega_dispersion[i, j]),
                            )
                        )
                    f.write("\n")

    def propagate_wave(self):
        """Increase the time and recalculate the surface"""

        # the time is stored in the wave1D object
        self.wave1D.next_time()
        self.calculate_wave_surface()

    def plot_wave(
        self,
        figsize=None,
        r_axis_type="frequency",
        plot_title=None,
        x_plot_title=0.05,
        y_plot_title=0.99,
        min_data_value=None,
        max_data_value=None,
        number_of_contour_levels=10,
        title_font_size=10,
        x_time_label=0.05,
        y_time_label=0.95,
        x_hs_label=0.05,
        y_hs_label=0.92,
        add_hs_estimate=True,
        color_map=cc.m_coolwarm,
        zorder=0,
        use_contourf=True,
    ):
        """Create a plot of the current wave

        Parameters
        ----------
        figsize: tuple
            x and y size of the total figure. Default is None, so figure size is not imposed
        r_axis_type: {"frequency", "period"}, optional
            quantity to use along the radial axis. Either frequency [Hz] or period [s]. Default =
            "frequency"
        plot_title: str
            Title to put in th figure. If None, use the titles found in the liftdyn file
        x_plot_title: float, optional
            x position (as a fraction of the sub plot index_title_axis). Default = 0.0
        y_plot_title: float, optional
            y position (as a fraction of the sub plot index_title_axis). Default = 1.1
        delta_y_titles: float, optional
            spacing between the lines of thet title lines. Default = 0.05
        max_title_lines: int or None
            Maxinmum number of title lines to plot. Default = None, ie. plot all
        title_font_size: int
            Size of the title font
        r_axis_lim: tuple or None
            The limits of the x axis. Default = None, so no limits are imposed
        zorder:  int, optional
            Position of the contour plot. Default = 0, meaning that the contours are placed at the
            bottom and the grid and labels are visible
        use_contourf: bool, optional
            If true, use contourf to make the contour plot. Slower. Default = False

        Returns
        -------
        tuple (fig, axis)
            Handle to the figure and the axis
        """
        if figsize is not None:
            size = figsize
        else:
            size = None

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=size)
        ax.set_aspect("equal", "datalim")

        x_label = "X [m]"
        y_label = "Y [m]"
        data_x_2d = self.xy_mesh[0]
        data_y_2d = self.xy_mesh[1]
        amplitude = self.amplitude

        if min_data_value is None:
            v_min = np.nanmin(amplitude)
        else:
            v_min = min_data_value

        if max_data_value is None:
            v_max = np.nanmax(amplitude)
        else:
            v_max = max_data_value

        # set the contour levels belonging to this subplot
        if min_data_value is None and max_data_value is None:
            # if both limits where not given, assume that we want matplotlib decide on the limits so
            # set levels=None
            levels = None
        else:
            levels = np.linspace(
                v_min, v_max, number_of_contour_levels + 1, endpoint=True
            )

        # finally create the contour plot wit the RAO magnitude
        if use_contourf:
            cs = ax.contourf(
                data_x_2d,
                data_y_2d,
                amplitude,
                cmap=color_map,
                levels=levels,
                zorder=zorder,
            )
        else:
            ls = LightSource(azdeg=315, altdeg=30)
            vert_eg = 1
            L_boundaries = (0, self.Lx, 0, self.Ly)
            rgb = ls.shade(
                amplitude.T, cmap=cm.ocean, vert_exag=vert_eg, blend_mode="hsv"
            )
            cs = plt.imshow(
                rgb,
                origin="lower",
                extent=L_boundaries,
                animated=True,
                vmin=v_min,
                vmax=v_max,
            )

        # if levels is not None:
        #    cs.cmap.set_under("k")
        #    cs.cmap.set_over("k")
        #    cs.set_clim(v_min, v_max)

        # set the axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if r_axis_type == "period":
            label_values = ax.get_yticks()
            label_strings = []
            for label in label_values:
                try:
                    label_strings.append("{:.1f}".format(1 / float(label)))
                except (ValueError, ZeroDivisionError):
                    pass

            ax.set_yticks(label_values)
            ax.set_yticklabels(label_strings)

        # create a contour bar
        cbar = fig.colorbar(cs, ax=ax)
        cbar.ax.set_ylabel("{} [{}]".format("Wave height", "m"))

        if add_hs_estimate:
            hs_estimate = 4 * amplitude.std()
            plt.figtext(
                x_hs_label,
                y_hs_label,
                "Hs={:.2f} m".format(hs_estimate),
                fontsize=title_font_size,
                verticalalignment="top",
            )

        if plot_title is not None:
            plt.figtext(
                x_plot_title,
                y_plot_title,
                plot_title,
                fontsize=title_font_size,
                verticalalignment="top",
            )
            window_title = plot_title
            fig.canvas.set_window_title(window_title)

        return fig, ax

    @staticmethod
    def _update_plot(
        frame_index=-1,
        self=None,
        fig=None,
        ax=None,
        changed_list=list(),
        min_data_value=0,
        max_data_value=None,
        number_of_contour_levels=10,
        color_map=cc.m_coolwarm,
        use_contourf=True,
        add_hs_estimate=True,
        title_font_size=10,
        x_time_label=0.05,
        y_time_label=0.95,
        x_hs_label=0.05,
        y_hs_label=0.92,
        zorder=0,
    ):

        if frame_index < 0:
            changed_list = list()
        else:
            clean_up_artists(axis=ax, artist_list=changed_list)
            self.wave1D.propagate_wave()
        self.calculate_wave_surface()
        amplitude = self.amplitude
        data_x_2d = self.xy_mesh[0]
        data_y_2d = self.xy_mesh[1]

        if min_data_value is None:
            v_min = np.nanmin(amplitude)
        else:
            v_min = min_data_value

        if max_data_value is None:
            v_max = np.nanmax(amplitude)
        else:
            v_max = max_data_value

        if min_data_value is not None and max_data_value is not None:
            # set the contour levels belonging to this subplot
            levels = np.linspace(
                v_min, v_max, number_of_contour_levels + 1, endpoint=True
            )
        else:
            levels = None

        # finally create the contour plot wit the RAO magnitude
        if use_contourf:
            cs = ax.contourf(
                data_x_2d,
                data_y_2d,
                amplitude,
                cmap=color_map,
                levels=levels,
                zorder=zorder,
            )
        else:
            ls = LightSource(azdeg=315, altdeg=30)
            vert_eg = 1
            L_boundaries = (0, self.Lx, 0, self.Ly)
            rgb = ls.shade(
                amplitude.T, cmap=cm.ocean, vert_exag=vert_eg, blend_mode="hsv"
            )
            cs = plt.imshow(
                rgb,
                origin="lower",
                extent=L_boundaries,
                animated=True,
                vmin=v_min,
                vmax=v_max,
            )

        changed_list.append(cs)

        # if levels is not None:
        #    cs.cmap.set_under("k")
        #    cs.cmap.set_over("k")
        #    cs.set_clim(v_min, v_max)

        if add_hs_estimate:
            hs_estimate = 4 * amplitude.std()
            txt = plt.figtext(
                x_hs_label,
                y_hs_label,
                "Hs={:.6f} m".format(hs_estimate),
                fontsize=title_font_size,
                verticalalignment="top",
            )
            changed_list.append(txt)

        if frame_index < 0:
            # create a contour bar
            cbar = fig.colorbar(cs, ax=ax)
            cbar.ax.set_ylabel("{} [{}]".format("Wave height", "m"))

        time_label = "{}".format(self.wave1D.time_delta)
        time_txt = plt.figtext(
            x_time_label,
            y_time_label,
            time_label,
            fontsize=title_font_size,
            verticalalignment="top",
        )
        changed_list.append(time_txt)

        return changed_list

    def animate_wave(
        self,
        figsize=None,
        plot_title=None,
        x_plot_title=0.05,
        y_plot_title=0.99,
        min_data_value=None,
        max_data_value=None,
        number_of_contour_levels=10,
        title_font_size=10,
        x_time_label=0.05,
        y_time_label=0.95,
        x_hs_label=0.05,
        y_hs_label=0.92,
        add_hs_estimate=True,
        color_map=cc.m_coolwarm,
        zorder=0,
        use_contourf=True,
        interval=1,
        title_horizontal_algnment="center",
    ):
        """Create a plot of the current wave

        Parameters
        ----------
        figsize: tuple
            x and y size of the total figure. Default is None, so figure size is not imposed
        plot_title: str
            Title to put in th figure. If None, use the titles found in the liftdyn file
        x_plot_title: float, optional
            x position (as a fraction of the sub plot index_title_axis). Default = 0.0
        y_plot_title: float, optional
            y position (as a fraction of the sub plot index_title_axis). Default = 1.1
        delta_y_titles: float, optional
            spacing between the lines of thet title lines. Default = 0.05
        max_title_lines: int or None
            Maxinmum number of title lines to plot. Default = None, ie. plot all
        title_font_size: int
            Size of the title font
        r_axis_lim: tuple or None
            The limits of the x axis. Default = None, so no limits are imposed
        zorder:  int, optional
            Position of the contour plot. Default = 0, meaning that the contours are placed at the
            bottom and the grid and labels are visible
        use_contourf: bool, optional
            If true, use contourf to make the contour plot. Slower. Default = False

        Returns
        -------
        tuple (fig, axis)
            Handle to the figure and the axis
        """
        if figsize is not None:
            size = figsize
        else:
            size = None

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=size)
        ax.set_aspect("equal", "datalim")

        x_label = "X [m]"
        y_label = "Y [m]"
        data_x_2d = self.xy_mesh[0]
        data_y_2d = self.xy_mesh[1]
        amplitude = self.amplitude

        if min_data_value is None:
            v_min = np.nanmin(amplitude)
        else:
            v_min = min_data_value

        if max_data_value is None:
            v_max = np.nanmax(amplitude)
        else:
            v_max = max_data_value

        # set the axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plot_title is not None:
            plt.figtext(
                x_plot_title,
                y_plot_title,
                plot_title,
                fontsize=title_font_size,
                verticalalignment="top",
                ha=title_horizontal_algnment,
            )
            window_title = plot_title
            fig.canvas.set_window_title(window_title)

        # set the contour levels belonging to this subplot
        if min_data_value is None and max_data_value is None:
            # if both limits where not given, assume that we want matplotlib decide on the limits so
            # set levels to None
            levels = None
        else:
            levels = np.linspace(
                v_min, v_max, number_of_contour_levels + 1, endpoint=True
            )

        changed_artists = list()

        # initial call to _update_plot to put of the axis. Return changed_artists with a list of all
        # items which need to be deleted every frame
        changed_artists = self._update_plot(
            frame_index=-1,
            self=self,
            fig=fig,
            ax=ax,
            changed_list=changed_artists,
            min_data_value=min_data_value,
            max_data_value=max_data_value,
            number_of_contour_levels=number_of_contour_levels,
            color_map=color_map,
            use_contourf=use_contourf,
            add_hs_estimate=add_hs_estimate,
            title_font_size=title_font_size,
            x_time_label=x_time_label,
            y_time_label=y_time_label,
            x_hs_label=x_hs_label,
            y_hs_label=y_hs_label,
            zorder=0,
        )

        # the fagrs list below must exactly match the arguments of the _update_plot function, except
        # for the first argument which is the frame_index
        ani = animation.FuncAnimation(
            fig,
            self._update_plot,
            frames=int(self.wave1D.nt_samples),
            fargs=(
                self,
                fig,
                ax,
                changed_artists,
                min_data_value,
                max_data_value,
                number_of_contour_levels,
                color_map,
                use_contourf,
                add_hs_estimate,
                title_font_size,
                x_time_label,
                y_time_label,
                x_hs_label,
                y_hs_label,
                zorder,
            ),
            interval=interval,
            blit=False,
            repeat=True,
        )

        return ani

    def plot_spectrum(
        self,
        figsize=(12, 6),
        r_axis_type="wave_number",
        plot_title=None,
        x_plot_title=0.05,
        y_plot_title=0.99,
        min_data_value=0,
        max_data_value=None,
        number_of_contour_levels=10,
        title_font_size=10,
        r_axis_lim=None,
        polar_projection=False,
        shift_origin=True,
        color_map=cc.m_rainbow,
        color_map_phase=cc.m_coolwarm,
        zorder=0,
        use_contourf=False,
        r_label_position=180,
        kx_min=None,
        kx_max=None,
        ky_min=None,
        ky_max=None,
    ):
        """Create a polar plot of the current spectral data

        Parameters
        ----------
        figsize: tuple
            x and y size of the total figure. Default is None, so figure size is not imposed
        r_axis_type: {"wave_number", "wave_length"}, optional
            quantity to use along the radial axis. Either wave_number [rad/m] or wave_length [m].
            Default = "wave_number"
        plot_title: str
            Title to put in th figure. If None, don't add a title
        x_plot_title: float, optional
            x position (as a fraction of the sub plot index_title_axis). Default = 0.0
        y_plot_title: float, optional
            y position (as a fraction of the sub plot index_title_axis). Default = 1.1
        title_font_size: int
            Size of the title font
        r_axis_lim: tuple or None
            The limits of the x axis. Default = None, so no limits are imposed
        zorder:  int, optional
            Position of the contour plot. Default = 0, meaning that the contours are placed at the
            bottom and the grid and labels are visible
        r_label_position: float, optional
            Angle along which we position the labels of the r-axis
        use_contourf: bool, optional
            Use the slower contourf to create a contour plot. Default = False, i.e. imshow is used
        kx_min: float, optional
            Minimum wave vector node in x direction for cartesian plot. Default = None
        kx_max: float, optional
            Maximum wave vector node in x direction for cartesian plot. Default = None
        ky_min: float, optional
            Minimum wave vector node in y direction for cartesian plot. Default = None
        ky_max: float, optional
            Maximum wave vector node in y direction for cartesian plot. Default = None

        Returns
        -------
        tuple (fig, axis)
            Handle to the figure and the axis
        """

        if figsize is not None:
            size = figsize
        else:
            size = None

        if polar_projection:
            fig, axis = plt.subplots(
                nrows=1, ncols=2, figsize=size, subplot_kw=dict(projection="polar")
            )
            for ax in axis:
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
        else:
            fig, axis = plt.subplots(nrows=1, ncols=2, figsize=size)
            for ax in axis:
                ax.set_aspect("equal", "datalim")

        if r_axis_lim is not None:
            # the radial limits are but on the ylim
            ax.set_ylim(r_axis_lim)

        if r_axis_type == "wave_number":
            x_label = "Wave number [rad/m]"
            y_label = ""
        elif r_axis_type == "wave_length":
            x_label = "Wave length [m]"
            y_label = ""
        else:
            raise AssertionError(
                "r_axis_type can only be 'wave_number' or 'wave_length'. Found {}"
                "".format(r_axis_type)
            )

        if polar_projection:
            data_x_2d = self.k_polar_mesh[0]
            data_y_2d = self.k_polar_mesh[1]
            psd_2d = abs(self.E_wave_density_polar)
            ang_2d = np.angle(self.E_wave_complex_amplitudes)
        else:
            x_label = "k_x [rad/m]"
            y_label = "k_y [rad/m]"
            if self.wave1D.mirror or shift_origin:
                data_x_2d = fftshift(self.k_cartesian_mesh[0])
                data_y_2d = fftshift(self.k_cartesian_mesh[1])
                psd_2d = fftshift(abs(self.E_wave_density_polar))
                ang_2d = fftshift(np.angle(self.E_wave_complex_amplitudes))
            else:
                data_x_2d = self.k_cartesian_mesh[0]
                data_y_2d = self.k_cartesian_mesh[1]
                psd_2d = abs(self.E_wave_density_polar)
                ang_2d = np.angle(self.E_wave_complex_amplitudes)

        if min_data_value is None:
            v_min = np.nanmin(psd_2d)
        else:
            v_min = min_data_value

        if max_data_value is None:
            v_max = np.nanmax(psd_2d)
        else:
            v_max = max_data_value

        # set the contour levels belonging to this subplot
        if min_data_value is None and min_data_value is None:
            # if both limits where not given, assume that we want matplotlib decide on the limits so
            # set levels=None
            levels = None
        else:
            levels = np.linspace(
                v_min, v_max, number_of_contour_levels + 1, endpoint=True
            )

        # finally create the contour plot wit the RAO magnitude
        if use_contourf:
            cs = axis[0].contourf(
                data_x_2d,
                data_y_2d,
                psd_2d,
                cmap=color_map,
                levels=levels,
                zorder=zorder,
            )
            cs2 = axis[1].contourf(
                data_x_2d,
                data_y_2d,
                ang_2d,
                cmap=color_map_phase,
                levels=None,
                zorder=zorder,
            )
        else:
            # k_boundaries = (data_x_2d.min(), data_x_2d.max(), data_y_2d.min(), data_y_2d.max())
            cs = axis[0].imshow(psd_2d.T, origin="lower", cmap=color_map)
            cs2 = axis[1].imshow(ang_2d.T, origin="lower", cmap=color_map_phase)

        # if levels is not None:
        # if levels is not None:
        #    cs.cmap.set_under("k")
        #    cs.cmap.set_over("k")
        #    cs.set_clim(v_min, v_max)
        # set the axis labels

        if r_axis_type == "wave_length":
            label_values = ax.get_yticks()
            label_strings = []
            for label in label_values:
                try:
                    label_strings.append("{:.1f}".format(2 * np.pi / float(label)))
                except (ValueError, ZeroDivisionError):
                    pass

            ax.set_yticks(label_values)
            ax.set_yticklabels(label_strings)

        # set the axis labels
        for ax in axis:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            if polar_projection:
                ax.set_rlabel_position(r_label_position)

                if r_axis_lim is not None:
                    # the radial limits are but on the ylim
                    ax.set_ylim(r_axis_lim)
            else:
                lim = set_limits(axis=ax, v_min=kx_min, v_max=kx_max, direction="x")
                lim = set_limits(axis=ax, v_min=ky_min, v_max=ky_max, direction="y")
                ax.set_aspect("equal", "datalim")

        # create a contour bar
        cbar = fig.colorbar(cs, ax=axis[0])
        cbar.ax.set_ylabel("{} [{}]".format("PSD", "m3"))

        cbar2 = fig.colorbar(cs2, ax=axis[1])
        cbar2.ax.set_ylabel("{} [{}]".format("Phase", "rad"))

        if plot_title is not None:
            plt.figtext(
                x_plot_title,
                y_plot_title,
                plot_title,
                fontsize=title_font_size,
                verticalalignment="top",
            )
            window_title = plot_title
            fig.canvas.set_window_title(window_title)

        return fig, axis


class Wave1D(object):
    """
    A class for linearized solutions of 1D wave vector equation

    .. math ::

        \eta(x_p, t) = \sum_i^{n_k} a_i \exp(j (k_i x_p - \omega_i t + \phi_i))

    where the amplitudes :math:`a_i` follow from the Jonswap or Gauss power spectral density
    function. See the top of the module for more information

    Parameters
    ----------
    xmin: float, optional
        Start of spatial domain. Default = 0
    Lx: float, optional
        Length of spatial domain. Default = 100
    t_start: float, optional
        Start of temporal domain. Default = 0
    t_length: float, optional
        Length of temporal domain. Default = 100
    nt_samples: int, optional
        Number of time samples. Default = 256
    nx_points: float, optional
        Number of space domain samples in x direction. Default = 256
    E_limit_low: float, optional
        Fraction of energy to cut at the low end of the spectrum.  Default = 0.01
    E_limit_high: float, optional
        Fraction of energy to keep at the high end of the spectrum.  Default = 0.9 (so 0.1 is cut
        off at the high end)
    kx_min: float, optional
        Minimal value of the wave vector domain. Default = 0
    kx_max: float, optional
        Maximum value of the wave vector domain. Default = pi/2
    delta_kx: float, optional
        Spacing in wave vector domain. Default = 0.06283 ((kx_max - kx_min) / n_kx_nodes)
    n_kx_nodes: int, optional
        Number of nodes in the wave vector domain. Default = 256
    n_bins_equal_energy: int, optional
        Number of nodes in the wave vector domain in case we are in the EqualEnergyBins mode. See
        `wave_selection` below. Default value = 100
    Hs: float, optional
        Significant wave height [m]. Default = 3 m
    Tp: float, optional
        Peak period [s]. Default = 10 s
    gamma: float, optional
        Peakness factor in case Jonswap spectrum is used. Default = 3.3
    sigma: float, optional
        Width of spectrum in case Gauss spectrum  with *spectral_version='dnv'* is used. Default = 0.0625
    random_phase: bool, optional
        Use random phase with each run. Default = False
    spectrum_type: {"jonswap", "gauss"}, optional
        Type of spectrum used. Default = "jonswap"
    spectral_version: {"sim", "dnv"}, optional
        Version of wave spectrum used. Default = "sim". For the Jonswap spectrum there is virtually
        no difference between the "sim" and "dnv" version. For the Gauss spectrum the "sim" version
        has a spectral width related to Tp which is fixed, wheras the "dnv" version has a width
        based on the *sigma* input argument.
    gravity0: float, optional
        Gravitation constant. Default = g0 = 9.81
    wave_construction: {"FFT", "DFTpolar", "DFTcartesian"}
        Method how the wave field is constructed from the spectrum. Default = "FFT". The options are

        * *DFTpolar*: A DFT (see top of module) is used to calculate the wave field. Slow, however,
          the location  of the wave vectors is not restricted. Therefore a smart selection of the
          wave nodes can be made. See *wave_selection*  for the possibilities.
        * *FFT* : The wave is constructed from the wave vector nodes using a Fast Fourier Transfor
          (FFT). This implies that the spectrum nodal poits are restricted: the number of wave nodes
          equals the number of spatial points x, and more over, the complex amplitude must obey the
          symmetry rule :math:`a(k)=a^*(-k)`. The implies that we can not make a wave selection.
          However, the FFT algorithm is so much faster than the DFT that we can still
          obtain faster calculations time using the FFT based on all wave nodes. The FFT wave
          construction is therefore recommended.
        * *DFTcartesian*: This choice is for validation purpose only. It assumes the same symmetric
          spectrum as used for the FFT option but then still the slow DFT  is used to calculate the
          wave field.
    wave_selection: {"All", "EqualEnergyBins", "Subrange"}
        For the DFTPolar wave construction mode we can make a selection of wave components in order
        to speed up the wave calculation. Three choices are possible

        * All: No selection is made so all the wave vectors as defined in the kx_nodes domain are
          used
        * Subrange: A Subrange of wave vectors based on the E_limit_low and E_limit_high values is
          made
        * EqualEnergyBins: The energy per bin is assumed equal
    use_subrange_energy_limits : bool
        If true the wave selection in the EqualEnergyBins mode is also limited by the Subrange
        settings. Default = True
    sample_every : int
        Make a wave selection by taking every 'sample_every' point in the wave vector domain. Only
        applicable when the wave_selection modes is *Subrange*

    Attributes
    ----------
    kx_nodes : ndarray
       Wave vector nodes of size :math:`n_k` used to build the spectrum and the wave field. The
       values depend on the wave_selection mode and wave_construction type. In case FFT is used, the
       kx_nodes are symmetrical around 0 to be able to define the symmetric spectrum.
    xpoints : ndarray
       Spatial position array of size :math:`n_x`
    amplitude : ndarray
       Wave height along x-direction at time t of size :math:`n_x`
    phase : ndarray
       Random phase array of size :math:`n_k`
    omega_dispersion : ndarray
       Angular frequency per wave node of size :math:`n_k` following from the deep water dispersion
       relation
    spectrumK : ndarray
       Power spectral density of the wave spectrum along the wave vector nodes k of size :math:`n_k`
    spectrumW : ndarray
       Power spectral density of the wave spectrum along the angular frequencies of size
       :math:`n_k`. Note that since the angular frequency are coupled to the wave vector nodes via
       the dispesion relation, the bin spacing of this vector is not equidistant
    complex_amplitudes : ndarray
        The complex amplitudes following from the power spectral density spectrumK and the phase
        vector as S_k * exp(j phase)
    delta_x :  float
        Spatial resolution of the x-domain
    kx_nyquist :  float
        Maximum wave vector following from the spatial resolution *delta_x* as  pi / delta_x

    Examples
    --------

    A wave can be setup a wave with all the default settings  and plot the spectrum and wave at time
    0 as follows

    >>> wave_dft = Wave1D()
    >>> fig, axis = wave_dft.plot_spectrum()
    >>> fig, axis = wave_dft.plot_wave()

    This will create the wave based on the DFT, hence it is slow. A wave construction based on the
    FFT can be picked as

    >>> wave_fft = Wave1D(wave_construction="FFT")

    An animation of the wave can be made as follows

    >>> movie = wave_fft.animate_wave()

    This will infinitely simulate the wave. We can put and finit time by setting the t_length to the
    wave

    >>> wave_fft.reset_time(t_length=100)

    For more examples, please have a look at the notebook linked via the Example section
    example_wave_fields
    """

    def __init__(
        self,
        name=None,
        xmin=0,
        Lx=None,
        xmax=None,
        t_start=0,
        delta_t=1,
        t_length=None,
        nt_samples=1000000,
        nx_points=512,
        E_limit_low=0.01,
        E_limit_high=0.90,
        kx_min=0.0,
        kx_max=1.5708,
        delta_kx=0.06283,
        n_kx_nodes=512,
        n_bins_equal_energy=64,
        use_subrange_energy_limits=False,
        sample_every=1,
        Hs=3.0,
        Tp=10.0,
        gamma=3.3,
        sigma=0.0625,
        random_phase=False,
        wave_construction="FFT",
        wave_selection="All",
        lock_nodes_to_wave_one=False,
        spectrum_type="jonswap",
        spectral_version="sim",
        gravity0=g0,
    ):

        self.spectrum_type = spectrum_type
        self.spectral_version = spectral_version

        logger.info("Initialise {} 1D wave field".format(self.spectrum_type))

        if name is None:
            self.name = "_".join(
                ["wave", spectrum_type, wave_construction, wave_selection, "1d"]
            )
        else:
            self.name = name

        # current time settings
        self.time = 0
        self.time_delta = pd.Timedelta(self.time, unit="s")
        self.t_index = 0
        self.repeat_movie = False
        self.playing_movie = False

        self.xmin = xmin
        if xmax is not None:
            self.Lx = xmax - self.xmin
            if Lx is not None:
                logger.warning(
                    "Both maximum x value xmax={} and length of domain Lx={} are "
                    "defined. Taking xmax leading so set Lx to {}"
                    "".format(xmax, Lx, self.Lx)
                )
        else:
            if Lx is not None:
                self.Lx = Lx
            else:
                self.Lx = 1000
        self.xmax = self.xmin + self.Lx

        self.t_start = None
        self.delta_t = None
        self.nt_samples = None
        self.t_length = None
        self.t_end = None
        self.reset_time(
            t_length=t_length, t_start=t_start, nt_samples=nt_samples, delta_t=delta_t
        )

        self.nx_points = nx_points
        self.nt_samples = nt_samples

        self.Hs = Hs
        self.Tp = Tp

        self.gravity0 = gravity0

        # depending on the wave type we need to define gamma or sigma (for Jonswap or Gauss, resp.)
        self.gamma = gamma
        self.sigma = sigma

        self.E_limit_low = E_limit_low
        self.E_limit_high = E_limit_high
        self.kx_min = kx_min
        self.kx_max = kx_max
        self.delta_kx = delta_kx
        self.delta_kx_non_uni = None
        self.n_kx_nodes = n_kx_nodes

        self.random_phase = random_phase

        # select which wave component are selected. Choices are: Subrange, EqualEnergyBins, All
        self.wave_selection = wave_selection

        # set the wave construction
        self.wave_construction = None
        self.mirror = False
        self.set_wave_construction(wave_construction)

        if self.wave_construction == "FFT" and self.wave_selection != "All":
            raise ValueError(
                "Can not combine 'FFT' wave_construction method with a "
                "wave_selecting other than 'All'"
            )

        self.lock_nodes_to_wave_one = lock_nodes_to_wave_one

        # use seed to update the random phase if required
        self.seed = 1
        self.update_phase = True

        self.pick_single_wave = False
        self.picked_wave_index = 0

        self.n_bins_equal_energy = n_bins_equal_energy
        self.use_subrange_energy_limits = use_subrange_energy_limits
        self.sample_every = sample_every

        self.xpoints = None
        self.phase = None
        self.kx = None

        self.amplitude = None
        self.delta_x = None
        self.kx_nyquist = None

        self.eta = 0

        self.ik_low = None
        self.ik_high = None
        self.ik_peak = None
        self.k_low = None
        self.k_high = None
        self.k_peak = None
        self.a_peakK = None
        self.varianceK = None
        self.iW_low = None
        self.iW_high = None
        self.iW_peak = None
        self.W_low = None
        self.W_high = None
        self.W_peak = None
        self.a_peakW = None
        self.varianceW = None
        self.spectrumK = None

        self.omega_dispersion = None
        self.delta_omega = None

        self.complex_amplitudes = None

        # the minimum and maximum wave
        self.global_wave_extremes = [0, 0]

        # add all the properties of the plot of the current wave
        self.plot = PlotProperties()

        self.update_x_k_t_sample_space()
        self.calculate_spectra_modulus()

    def reset_time(self, t_length=None, t_start=0, nt_samples=10000000, delta_t=1):
        """Reset all time properties and allow to recalculate"""
        self.t_start = t_start
        self.delta_t = delta_t
        self.nt_samples = nt_samples
        if t_length is None:
            self.t_length = self.nt_samples * self.delta_t
        else:
            self.t_length = t_length
            self.nt_samples = int(self.t_length / self.delta_t)
        self.t_end = self.t_start + self.t_length
        self.time = self.t_start
        self.t_index = 0
        self.time_delta = pd.Timedelta(self.time, unit="s")

    def make_report(self):
        """Make report of settings for this wave"""

        frm = "{:40s} : {}"

        logger.info(frm.format("Name", self.name))
        logger.info("----------- Domain settings --------")
        logger.info(frm.format("Start x [m]", self.xmin))
        logger.info(frm.format("End x [m]", self.xmax))
        logger.info(frm.format("Length x [m]", self.Lx))
        logger.info("----------- Time settings --------")
        logger.info(frm.format("Start t [s]", self.t_start))
        logger.info(frm.format("End t [s]", self.t_end))
        logger.info(frm.format("Length t [s]", self.t_length))
        logger.info(frm.format("Delta t [s]", self.delta_t))

        logger.info("----------- Numerical resolutions --------")
        logger.info(frm.format("Number x - nodes", self.xpoints.size))
        logger.info(frm.format("Number kx - nodes", self.kx_nodes.size))
        logger.info(frm.format("Delta x [m]", self.delta_x))
        dkx = np.diff(self.kx_nodes)
        logger.info(frm.format("Delta kx min [rad/m]", dkx.min()))
        logger.info(frm.format("Delta kx max [rad/m]", dkx.max()))
        logger.info(frm.format("Delta kx first [rad/m]", self.delta_kx))
        logger.info(frm.format("k-min [rad/m]", self.kx_nodes.min()))
        logger.info(frm.format("k-max [rad/m]", self.kx_nodes.max()))
        logger.info(frm.format("k-nyq [rad/m]", self.kx_nyquist))
        logger.info(frm.format("L_max [m]", 2 * np.pi / dkx[0]))

        logger.info("----------- Numerical methodds --------")
        logger.info(frm.format("Selection method", self.wave_selection))
        logger.info(frm.format("Construction method", self.wave_construction))

    def next_time(self):
        """Increase the time"""
        self.time += self.delta_t
        self.time_delta = pd.Timedelta(self.time, unit="s")
        self.t_index += 1
        if self.time > self.t_end:
            if self.repeat_movie:
                self.time = self.t_start
                self.t_index = 0
            else:
                self.playing_movie = False

    def propagate_wave(self):
        """Increase the time and recalculate the surface"""

        self.next_time()
        self.calculate_wave_surface()

    def animate_wave(
        self,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        figsize=(12, 4),
        interval=25,
        x_time_label=0.01,
        y_time_label=0.93,
        plot_title=None,
        x_plot_title=0.5,
        y_plot_title=0.97,
        title_font_size=12,
        title_horizontal_algnment="center",
    ):
        """Animate the 1D wave vs space and wave vs time

        Parameters
        ----------
        fig: :obj:`figure`
            Figure returned by previous call can be update
        ax: :obj:`Axes`
            Axes returned by previous call can be updated to add more waves
        x_min: float, optional
            Start of the x-range. Default = None
        x_max: float, optional
            End of the x-range. Default = None
        y_min: float, optional
            Start of the y-range. Default = None
        y_max: float, optional
            End of the y-range. Default = None
        figsize: tuple, optional
            Tuple with (with, height) in inches. Default = None
        interval: int, optional
            Time in ms between each frame. Default = 25
        x_time_label: float, optional
            X position relative to the current graph where to put the time label
        y_time_label: float, optional
            Y Position relative to the current graph where to put the time label

        Returns
        -------
        tuple
            (fig, ax) reference to the figure and its axis
        """

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        self.calculate_wave_surface()
        (line,) = ax.plot(self.xpoints, self.amplitude)
        ax.set_xlabel("x position [m]")
        ax.set_ylabel("wave amplitude [m]")

        time_label_string = "{:02d}:{:02d}:{:02d}"

        time_text = ax.text(
            x_time_label,
            y_time_label,
            time_label_string.format(
                self.time_delta.components.hours,
                self.time_delta.components.minutes,
                self.time_delta.components.seconds,
            ),
            transform=ax.transAxes,
        )

        if plot_title is not None:
            plt.figtext(
                x_plot_title,
                y_plot_title,
                plot_title,
                fontsize=title_font_size,
                verticalalignment="top",
                ha=title_horizontal_algnment,
            )
            window_title = plot_title
            fig.canvas.set_window_title(window_title)

        lim = set_limits(axis=ax, v_min=y_min, v_max=y_max, direction="y")

        self.playing_movie = True

        def next_time():
            # and iterator to proceed to the next wave and yield the index only if we want to keep
            # playing
            # obsolete, I have now added the proceed wave to the animate function and past nt_samples
            logger.debug("Updating wave for time {}".format(self.time))
            self.propagate_wave()
            if self.playing_movie:
                yield self.t_index

        def animate(i):
            logger.debug(
                "Plotting wave for time {:10.1f} s until {:10.1f} s :  {}".format(
                    self.time, self.t_end, self.time_delta
                )
            )
            self.propagate_wave()
            line.set_ydata(self.amplitude)
            time_text.set_text(
                time_label_string.format(
                    self.time_delta.components.hours,
                    self.time_delta.components.minutes,
                    self.time_delta.components.seconds,
                )
            )
            return line, time_text

        def init():
            # initialise the wave plot by first calculating the current wave  height and than set
            # the data
            line.set_ydata(self.amplitude)
            return (line,)

        ani = animation.FuncAnimation(
            fig,
            animate,
            self.nt_samples,
            init_func=init,
            interval=interval,
            blit=False,
            repeat=True,
        )
        return ani

    @staticmethod
    def limit_markers(
        ax, text, a_x, a_y, s_x=0, s_y=0, symbol="vy", arrow_color="black", fontsize=8
    ):

        ax.plot([a_x], [a_y], symbol)
        ax.annotate(
            text,
            xy=(a_x, a_y),
            xytext=(s_x, s_y),
            textcoords="offset points",
            horizontalalignment="left",
            va="top",
            arrowprops=dict(width=2, headwidth=5, shrink=0.1, color=arrow_color),
            fontsize=fontsize,
        )

    def plot_spectrum(
        self,
        fig=None,
        axis=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        figsize=None,
        add_limit_markers=False,
        add_hs_estimate=False,
        add_n_points=True,
        y_n_points_label=0.8,
        line_markers="o",
        markersize=4,
        linestyle="-",
        linecolor=None,
        plot_title=None,
        x_plot_title=0.5,
        y_plot_title=0.97,
        title_font_size=12,
        title_horizontal_algnment="center",
    ):
        """Plot the spectrum in k-spaco

        Parameters
        ----------
        fig: :obj:`Figure`, optional
            Reference to the figure to add the plot to
        axis: :obj:`Axis`, optional
            Reference to the axis
        x_min: float, optional
            set minimum x value, Default = None i.e take the default
        x_max: float, optional
            set maximum x value, Default = None i.e take the default
        y_min: float, optional
            set minimum y value, Default = None i.e take the default
        y_max: float, optional
            set maximum y value, Default = None i.e take the default
        figsize: tuple
            (width, height) of Figure. Default = None ie. take the default
        add_limit_markers: bool, optional
            Add markers to indicate the low and high limits of the energy in the spectrum
        add_hs_estimate=False, optional
            Add a label indicating the Hs belonging to the energy of the spectrum
        add_n_points: bool, optional
            Add a label with the number of points
        y_n_points_label: bool, optional
            Give the y coordinates  the n_point label, Default = 0.8
        line_markers: str, optional
            Give the marker of the plot. Default = "o". Set to "" to remove the markers
        linestyle: str, optional
            Give the linestyle, Default = "-"
        linecolor: str, optional
            Give the color of the line. Default = None, ie take the default
        markersize: int, optional
            Give the size of the markers. Default = 4

        Returns
        -------
        tuple
            (fig, axis), reference to the create fig and axis

        """
        if fig is None or axis is None:
            fig, axis = plt.subplots(nrows=2, ncols=1, figsize=figsize)
            plt.subplots_adjust(hspace=0.25)
            axis[0].set_ylabel("spectral amplitude [m3]")
            axis[0].set_xlabel("wave vector kx [rad/m]")

            axis[1].set_ylabel("spectral amplitdude [m2s]")
            axis[1].set_xlabel("omega [rad/s]")

        if self.mirror:
            kx_nodes = fftshift(self.kx_nodes)
            omega_disp = fftshift(self.omega_dispersion)
            spectrum_k = fftshift(self.spectrumK)
            spectrum_w = fftshift(self.spectrumW)
        else:
            kx_nodes = self.kx_nodes
            omega_disp = self.omega_dispersion
            spectrum_k = self.spectrumK
            spectrum_w = self.spectrumW

        (line0,) = axis[0].plot(
            kx_nodes,
            spectrum_k,
            linestyle=linestyle,
            marker=line_markers,
            markersize=markersize,
            color=linecolor,
        )

        (line1,) = axis[1].plot(
            omega_disp,
            spectrum_w,
            linestyle=linestyle,
            marker=line_markers,
            markersize=markersize,
            color=linecolor,
        )

        if add_limit_markers:
            self.limit_markers(
                ax=axis[0],
                a_x=self.k_low,
                a_y=self.spectrumK[self.ik_low],
                symbol="vy",
                text="E(k<{:.2g})/E\n={:.2g}".format(self.k_low, self.E_limit_low),
                s_x=-50,
                s_y=40,
            )
            self.limit_markers(
                ax=axis[0],
                a_x=self.k_high,
                a_y=self.spectrumK[self.ik_high],
                symbol="^g",
                text="E(k>{:.2g})/E\n={:.2g}".format(
                    self.k_high, 1 - self.E_limit_high
                ),
                s_x=10,
                s_y=40,
            )
            self.limit_markers(
                ax=axis[0],
                a_x=self.k_peak,
                a_y=self.spectrumK[self.ik_peak],
                symbol="or",
                text="k={:.2g} rad/m\nL={:.1f} m".format(
                    self.k_peak, 2 * np.pi / self.k_peak
                ),
                s_x=20,
                s_y=-30,
            )

            self.limit_markers(
                ax=axis[1],
                a_x=self.W_low,
                a_y=self.spectrumW[self.iW_low],
                symbol="vy",
                text="E(k<{:.2g})/E\n={:.2g}".format(self.W_low, self.E_limit_low),
                s_x=-50,
                s_y=40,
            )
            self.limit_markers(
                ax=axis[1],
                a_x=self.W_high,
                a_y=self.spectrumW[self.iW_high],
                symbol="^g",
                text="E(k>{:.2g})/E\n={:.2g}".format(
                    self.W_high, 1 - self.E_limit_high
                ),
                s_x=10,
                s_y=40,
            )
            self.limit_markers(
                ax=axis[1],
                a_x=self.W_peak,
                a_y=self.spectrumW[self.iW_peak],
                symbol="or",
                text="k={:.2g} rad/s\nT={:.1f} s".format(
                    self.W_peak, 2 * np.pi / self.W_peak
                ),
                s_x=20,
                s_y=-30,
            )

        if add_hs_estimate:
            var_k = np.sum(self.spectrumK * self.delta_kx)
            if self.wave_construction in ("FFT", "DFTcartesian"):
                var_k /= 2.0
            hs_estimate_k = 4 * np.sqrt(var_k)
            axis[0].text(
                0.85,
                0.92,
                "Hs={:.1f} m".format(hs_estimate_k),
                transform=axis[0].transAxes,
                va="top",
            )
            var_w = np.sum(self.spectrumW * self.delta_omega)
            if self.wave_construction in ("FFT", "DFTcartesian"):
                var_w /= 2.0
            hs_estimate_w = 4 * np.sqrt(var_w)
            axis[1].text(
                0.85,
                0.92,
                "Hs={:.1f} m".format(hs_estimate_w),
                transform=axis[1].transAxes,
            )
        if add_n_points:
            axis[0].text(
                0.85,
                y_n_points_label,
                "N={:d}".format(self.spectrumK.size),
                transform=axis[0].transAxes,
                va="top",
            )

        # for ax in axis:
        #    set_limits(ax, v_min=x_min, v_max=x_max, axis="x")
        #    set_limits(ax, v_min=y_min, v_max=y_max, axis="y")

        if plot_title is not None:
            plt.figtext(
                x_plot_title,
                y_plot_title,
                plot_title,
                fontsize=title_font_size,
                verticalalignment="top",
                ha=title_horizontal_algnment,
            )
            window_title = plot_title
            fig.canvas.set_window_title(window_title)

        return fig, axis

    def plot_wave(
        self,
        fig=None,
        ax=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        figsize=(12, 4),
        linestyle="-",
        line_markers=None,
        linecolor=None,
        markersize=5,
        label=None,
        plot_title=None,
        x_plot_title=0.5,
        y_plot_title=0.97,
        title_font_size=12,
        title_horizontal_algnment="center",
    ):
        """Plot the 1D wave vs space and wave vs time

        Parameters
        ----------
        fig: :obj:`figure`
            Figure returned by previous call can be update
        ax: :obj:`Axes`
            Axes returned by previous call can be updated to add more waves
        x_min: float, optional
            Start of the x-range. Default = None
        x_max: float, optional
            End of the x-range. Default = None
        y_min: float, optional
            Start of the y-range. Default = None
        y_max: float, optional
            End of the y-range. Default = None
        figsize: tuple, optional
            Tuple with (with, height) in inches. Default = None
        plot_title: str, optional
            Add a title to the plot. Default = None, ie. don't add a title
        x_plot_title: float, optional
            X Location of the plot title. Default = 0.5 wro the figurek
        y_plot_title: float, optional
            Y Location of the plot title. Default = 0.97 wro the figurek
        title_font_size: int, optional
            Font size of the title

        Returns
        -------
        tuple
            (fig, ax) reference to the figure and its axis
        """

        self.calculate_wave_surface()

        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        (line1,) = ax.plot(
            self.xpoints,
            self.amplitude,
            linestyle=linestyle,
            marker=line_markers,
            markersize=markersize,
            color=linecolor,
            label=label,
        )
        ax.set_xlabel("x position [m]")
        ax.set_ylabel("wave amplitude [m]")

        lim = set_limits(axis=ax, v_min=x_min, v_max=x_max, direction="x")
        lim = set_limits(axis=ax, v_min=y_min, v_max=y_max, direction="y")

        if plot_title is not None:
            plt.figtext(
                x_plot_title,
                y_plot_title,
                plot_title,
                fontsize=title_font_size,
                verticalalignment="top",
                horizontalalignment=title_horizontal_algnment,
            )
            window_title = plot_title
            fig.canvas.set_window_title(window_title)

        return fig, ax

    def set_wave_construction(self, mode):
        """
        Set the wave how the wave field is constructed from the power spectral density: via FFT or
        DFT

        Parameters
        ----------
        mode: {"FFT", "DFTpolar", "DFTcartesian"}
            Construction type of the wave field. In case FFT or DFTcartesian is chosen, the
            spectrum most be symmetric and therefore mirror must be True

        """
        if self.wave_selection != "All" and mode != "DFTpolar":
            logger.warning(
                "You have set a wave selection but want to use an FFT. Selecting waves is only "
                "possible for DFTpolar. Setting that now"
            )
            mode = "DFTpolar"

        self.wave_construction = mode
        if mode in ("FFT", "DFTcartesian"):
            self.mirror = True
        elif mode == "DFTpolar":
            self.mirror = False
        else:
            raise AssertionError(
                "Wave construction must be FFT, DFTcartesian, or DFTpolar. Found {}".format(
                    mode
                )
            )

    # @profile
    def update_x_k_t_sample_space(self):
        """
        After a change of number of x-points or wave vector nodes k has been made, call this routine
        to update all the arrays

        """

        self.xmax = self.xmin + self.Lx

        self.xpoints = np.linspace(self.xmin, self.xmax, self.nx_points, endpoint=True)

        self.amplitude = np.zeros(self.xpoints.shape)

        self.delta_x = self.xpoints[1] - self.xpoints[0]

        logger.debug(
            "XMIN {} XMAX {} nx {}  deltax {}".format(
                self.xmin, self.xmax, self.nx_points, self.delta_x
            )
        )
        self.kx_nyquist = np.pi / self.delta_x

        if self.wave_construction == "DFTpolar":
            # If a DDT is used for the wave field calculation, you can use any amount of wave
            # vectors
            if self.kx_max > self.kx_nyquist:
                logger.warning(
                    "kx_max larger than nyquist belongin to dx={}: clipping kx to {}".format(
                        self.delta_x, self.kx_nyquist
                    )
                )
                self.kx_max = self.kx_nyquist

            self.kx_nodes = np.linspace(
                self.kx_min, self.kx_max, self.n_kx_nodes, endpoint=True
            )

            self.delta_kx = self.kx_nodes[1] - self.kx_nodes[0]
        else:
            # for the FFT the number wave vectors should be equal to the number of x points
            # the DFT based on cartesian values in this case take the same mesh as FFT but then uses
            # the DFT algorith for comparison
            self.kx_nodes = 2 * np.pi * np.fft.fftfreq(self.nx_points, self.delta_x)
            self.delta_kx = self.kx_nodes[1] - self.kx_nodes[0]

            logger.debug(
                "kx_nodes minx/max {} {}".format(
                    np.amin(self.kx_nodes), np.amax(self.kx_nodes)
                )
            )

        if (
            self.phase is None
            or self.kx_nodes.shape != self.phase.shape
            or self.update_phase
        ):
            self.phase = ms.initialize_phase(self.kx_nodes, self.seed)
            self.update_phase = False

        self.t_end = self.t_start + self.t_length

        self.delta_t = self.t_length / self.nt_samples

    # @profile
    def calculate_omega_dispersion(self):
        """
        Calculate the omega frequency belonging to the wave vectors according to the deep water
        dispersion relation.
        """
        self.omega_dispersion = np.sqrt(self.gravity0 * abs(self.kx_nodes)) * sp.where(
            self.kx_nodes >= 0,
            np.ones(self.kx_nodes.shape),
            -np.ones(self.kx_nodes.shape),
        )
        delta_omega = np.diff(self.omega_dispersion)
        self.delta_omega = np.append(delta_omega, [delta_omega[-1]])

    # @profile
    def calculate_spectra_modulus(self):
        """
        this routine calcutes the Spectrum in omega and k domain. The spectrum can be either a
        Jonswap or a Gauss spectrum depending on the wave_type argument
        Based on the energy limit the minimum and maximum frequency/wave vector is obtained using
        the specspecs routine. Based on those limit, you may take a selection of wave vectors if the
        wave_selection variable is set to 'subrange' or 'EqualEnergyBins'
        """

        # self.phase = ms.initialize_phase(self.kx_nodes,self.seed)

        self.spectrumK = ms.spectrum_wave_k_domain(
            k_waves=self.kx_nodes,
            Hs=self.Hs,
            Tp=self.Tp,
            gamma=self.gamma,
            sigma=self.sigma,
            spectrum_type=self.spectrum_type,
            spectral_version=self.spectral_version,
        )

        (
            self.ik_low,
            self.ik_high,
            self.ik_peak,
            self.k_low,
            self.k_high,
            self.k_peak,
            self.a_peakK,
            self.varianceK,
        ) = ms.specspecs(
            self.kx_nodes,
            self.spectrumK,
            lowlim=self.E_limit_low,
            higlim=self.E_limit_high,
            mirror=self.mirror,
        )

        # select a sub range of the wave vectors. In case that fft is used, this is not possible.
        if self.wave_construction == "DFTpolar" and self.wave_selection == "Subrange":
            # extra wave vectors based on the range k_low,k_high

            # create a mask array to select the wave vectors within the subrange k_low k_high
            mask = ms.mask_out_of_range(self.kx_nodes, self.k_low, self.k_high)

            # make a selection of wave vectors: set value equal to zero outside range
            self.kx_nodes = np.extract([mask], [self.kx_nodes])
            self.spectrumK = np.extract([mask], [self.spectrumK])
            self.phase = np.extract([mask], [self.phase])

            # subsample the wave vectors and frequencies bin in case sample_every is larger than one
            if self.sample_every > 1:
                self.kx_nodes = self.kx_nodes[:: self.sample_every]
                self.spectrumK = self.spectrumK[:: self.sample_every]
                self.phase = self.phase[:: self.sample_every]

        elif (
            self.wave_construction == "DFTpolar"
            and self.wave_selection == "EqualEnergyBins"
        ):
            # if EqualEnergy bins is selected, make a selection of frequency bins such that the
            # energy per interval S*dk remains constant to Ebin (the mean energy per bin based on
            # the total energy and number of bins
            self.Ebin = self.varianceK / self.n_bins_equal_energy
            kx_nodes = []
            kk = self.kx_min
            if self.use_subrange_energy_limits:
                kk = self.k_low
            kx_nodes.append(kk)
            logger.debug("Start solving the euqal energy bins...")
            n_trail_max = 10
            while kk < self.kx_max:
                logger.debug("Solving bin for kk = {}".format(kk))
                # calculate the next wave vector such that S*dk (energy in this bin) equals Ebin
                # by solving the integral int_k0^knew Sdk. kk+delta_kx is just a first guess
                delta_kx = self.delta_kx
                n_trail = n_trail_max
                found_solution = False
                while not found_solution and n_trail > 0:
                    ans = sp.optimize.fsolve(
                        _energy_deficit,
                        kk + delta_kx,
                        args=(
                            kk,
                            self.Ebin,
                            self.Hs,
                            self.Tp,
                            self.gamma,
                            self.spectral_version,
                            self.spectrum_type,
                            self.sigma,
                        ),
                    )
                    if ans[0] > 0:
                        found_solution = True
                        kk = ans[0]
                    else:
                        delta_kx /= 2
                        n_trail -= 1
                        # we could not find a solution. Try again with a smaller offset
                        logger.debug(
                            "Failed to find a solution for kk. Try again for smaller"
                            " delta kx {} {}".format(delta_kx, n_trail)
                        )
                if n_trail < 0:
                    logger.warning(
                        "Tried {} times without succes. Stop solving, "
                        "hope for the best".format(n_trail)
                    )
                    break
                # check if kk is within subrange is requested and then add it to the list
                if kk < self.kx_max and not (
                    self.use_subrange_energy_limits
                    and (kk < self.k_low or kk > self.k_high)
                ):
                    logger.debug(
                        "Storing bin kk = {} with k_low = {} k_high ={}"
                        "".format(kk, self.k_low, self.k_high)
                    )
                    kx_nodes.append(kk)
            logger.debug("Done")

            # turn the create kx_node list into a nparray
            if self.lock_nodes_to_wave_one:
                # if lock_nodes_to_wave_one is true, the k values calculated above all all locked to
                # the wave vectors of the full wave domain by finding the nearest wave value
                mask = np.full(self.kx_nodes.shape, False, dtype=np.bool)
                for kk in kx_nodes:
                    kinx = find_idx_nearest_val(self.kx_nodes, kk)
                    mask[kinx] = True

                # select the  kx and phase at the nodes
                self.kx_nodes = np.extract(mask, self.kx_nodes)
                self.phase = np.extract(mask, self.phase)
            else:
                # in case the nodes do not have to be locked to the first wave, just convert the
                # created list of wave vectors into a numpy array, calculate a new phase and the
                # corresponding modulus
                self.kx_nodes = np.array(kx_nodes)
                self.phase = ms.initialize_phase(self.kx_nodes, self.seed)

            # based on the locked or non-locked kx_nodes with fixed and non-fixed phases, calculate
            # the spectrumK
            self.spectrumK = ms.spectrum_wave_k_domain(
                k_waves=self.kx_nodes,
                Hs=self.Hs,
                Tp=self.Tp,
                gamma=self.gamma,
                sigma=self.sigma,
                spectrum_type=self.spectrum_type,
                spectral_version=self.spectral_version,
            )

        elif self.wave_selection == "OneWave":
            logger.debug("selecting wave {}".format(self.picked_wave_index))
            # pick one wave only
            k = self.picked_wave_index
            if not self.wave_construction == "FFT":
                # only copy the single wave of the picked wave vector for the DFT algorithms
                self.kx_nodes = np.array([self.kx_nodes[k]])
            else:
                # keep all the wave vectors except set the value not equal to the picked wave to
                # zero
                mask = np.full(self.spectrumK.shape, True, dtype=bool)
                mask[self.picked_wave_index] = False
                self.spectrumK[mask] = 0.0
                self.phase[mask] = 0.0
        else:
            logger.debug("No wave selection has been made, so using all the k nodes")

        # based on the new omega values, calculate the omega spectrum as well
        self.calculate_omega_dispersion()
        if self.spectrum_type == "jonswap":
            self.spectrumW = ms.spectrum_jonswap(
                abs(self.omega_dispersion),
                Hs=self.Hs,
                Tp=self.Tp,
                gamma=self.gamma,
                spectral_version=self.spectral_version,
            )
        elif self.spectrum_type == "gauss":
            self.spectrumW = ms.spectrum_gauss(
                abs(self.omega_dispersion),
                Hs=self.Hs,
                Tp=self.Tp,
                sigma=self.sigma,
                spectral_version=self.spectral_version,
            )
        else:
            raise AssertionError(
                "spectrum type must either be 'jonswap' or 'gauss'. Found {}".format(
                    self.spectrum_type
                )
            )

        (
            self.iW_low,
            self.iW_high,
            self.iW_peak,
            self.W_low,
            self.W_high,
            self.W_peak,
            self.a_peakW,
            self.varianceW,
        ) = ms.specspecs(
            self.omega_dispersion,
            self.spectrumW,
            lowlim=self.E_limit_low,
            higlim=self.E_limit_high,
            mirror=self.mirror,
        )

        # calculate the complex amplitudes for the wave nodes and random phase vector
        self.complex_amplitudes = ms.spectrum_to_complex_amplitudes(
            self.kx_nodes,
            spectral_modulus=self.spectrumK,
            phase=self.phase,
            mirror=self.mirror,
        )

        if not self.wave_construction == "FFT":
            # create the maxtrix exp (j * kx_nodes * x_nodes) where kx_nodes * x_nodes is the matrix
            # following from the vector procuct of the vectos k^T and x
            # only calculate this for DFT
            self.exp_matrix_kx = np.exp(
                1j
                * self.kx_nodes.reshape(self.kx_nodes.size, 1)
                * self.xpoints.reshape((1, self.xpoints.size))
            )

    # @profile
    def calculate_wave_surface(self):
        """
        Calculate the wave surface for current time using either DFT or FFT

        Notes
        -----

        The *wave_construction* attribute determines which algorithm is used to calculate the the
        wave surface. The following options are possible

        * DFTpolar: a DFT (Discrete Fourier Transform) is used to calculate the wave surface. There
          are no requirement to the number and shape of the fourier nodes stored in the
          'complex_amplitudes'.
        * FFT: a FFT (Fast Fourier Transform) is used to calculate the wave surface. This implies
          that the spectral nodes must be symmetrical around zero as S(k)=S^*(-k)
        * DFTcartesian: A DFT is used to calculate the wave surface, however, still the same
          symmetric spectrum as used with the FFT is supplied. This allows to compare the DFT and
          DFT in calculation time and outcome with the exact same outcome (as the input nodes can be
          the same). The scaling with 0.5 due to the double spectrum (as it is symmetric) is taken
          care of here.
        """

        if self.wave_construction in ("DFTpolar", "DFTcartesian"):
            self.amplitude = self.dft_complex_amplitudes(
                self.complex_amplitudes,
                self.exp_matrix_kx,
                self.omega_dispersion,
                self.time,
            )
            if self.wave_construction == "DFTcartesian":
                # the DFTcartesian uses a symmetric spectrum, just as you do with the FFT.
                # Therefore you have to scale the energy with a half
                self.amplitude *= 0.5
        elif self.wave_construction == "FFT":
            # the fft is used
            self.amplitude = self.fft_amplitude(
                self.complex_amplitudes, self.omega_dispersion, self.time
            )
        else:
            raise (
                AssertionError(
                    "wave_construction should be either FFT, DFTpolar, or DFTcartesian. Found {}"
                    "".format(self.wave_construction)
                )
            )

        logger.debug("H_s of 1D surface {} ".format(4 * np.std(self.amplitude)))

    # @profile
    @staticmethod
    def dft_complex_amplitudes(S_tilde, exp_kx, omega, time):
        """
        Calculate the wave height from the complex amplitudes at given time

        Parameters
        ----------
        S_tilde: complex ndarray
            vector Nx1 with the complex amplitudes
        exp_kx: ndarray array
            NxM matrix with exp(k * x) where k is Nx1 k waves vectors and x is 1XM spatial nodes
        omega: ndarray
            N ndvector
        time: float
            Current time
        Returns
        -------
        ndarray array
            Mx1 real array with the wave amplitude for given time

        Notes
        -----
        Calculate the vector height(x0,x1,..xM) = real(sum_k A*exp(j(k*x-w*t))), where the MxN
        matrix exp(k*x) and the Nx1 vector A (with the complex amplitudes) have been pre-calculated
        and exp(j*w*t) is put here in a NxN diagonal matrix. Return the Mx1 vector with height for
        each position x
        """
        N = S_tilde.size
        omega_matrix = np.eye(N) * np.exp(-1j * omega * time)
        M = np.dot(S_tilde, omega_matrix)
        dft = np.dot(M, exp_kx)
        return np.real(dft)

    # @profile
    @staticmethod
    def fft_amplitude(S_tilde, omega, time):
        """Calculate the amplitude at time using the FFT

        Parameters
        ----------
        S_tilde: ndarray
            Nx1 array with the complex amplitudes of the wave spectrum
        omega: ndarray
            N x 1 array with the real angular frequency rad/s per wave vector k
        time:  float
            Current time

        Returns
        -------
        ndarray
            real Nx1 array with the wave amplitude at time t

        Notes
        -----
        Since the FFT is used, the  S_tilde should by symmetrical around k=0 such that S(k)=S^*(-k)

        """
        N = S_tilde.size / 2
        ampl = N * np.fft.ifft(S_tilde * np.exp(1j * (-time * omega)))
        # ampl should be real already because S_tilde should by symmetrical around k=0 S(k)=S^*(-k)
        # to be sure, take the real value only
        return np.real(ampl)


def new_function(arg1, arg2, option_arg=None):
    """

    Parameters
    ----------
    arg1
    arg2
    option_arg

    Returns
    -------

    """
