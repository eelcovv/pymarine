#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import (assert_almost_equal)

from pymarine.waves.wave_fields import (Wave1D, Wave2D)


def test_wave_1d():
    hs_in = 3.0

    for wave_construction in ("FFT", "DFTpolar", "DFTcartesian"):
        if wave_construction == "DFTpolar":
            factor = 1.0
        else:
            factor = 0.5

        wave1d = Wave1D(Hs=hs_in, n_kx_nodes=1024, Lx=2000, nx_points=1024,
                        wave_construction=wave_construction)

        # check the energy. Note that for FFT and DFTcartesian we have mirrored the spectrum, and we
        # have every bin double; therefore factor = 0.5 to take only half of the domain energy
        hs_out = 4 * np.sqrt(wave1d.delta_kx * np.sum(wave1d.spectrumK) * factor)
        assert_almost_equal(hs_in, hs_out, decimal=1)

        # bring the delta_omega inside as it is not equidistant anymore
        hs_out = 4 * np.sqrt(np.sum(wave1d.delta_omega * wave1d.spectrumW) * factor)
        assert_almost_equal(hs_in, hs_out, decimal=1)


def test_wave_2d():
    hs_in = 3.0

    for wave_construction in ("FFT", "DFTpolar", "DFTcartesian"):

        wave1d = Wave1D(Hs=hs_in, n_kx_nodes=128, Lx=1000, nx_points=128,
                        wave_construction=wave_construction)

        wave2d = Wave2D(wave1D=wave1d)

        # check the significant wave heigt from the amplitude
        hs_out = 4 * wave2d.amplitude.std()
        assert_almost_equal(hs_in, hs_out, decimal=1)
