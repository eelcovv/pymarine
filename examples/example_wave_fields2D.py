import logging

import matplotlib.pyplot as plt
from hmc_utils.misc import (create_logger)

import hmc_marine
import hmc_marine.wave_fields as wf

print(hmc_marine.__file__)

logger = create_logger(console_log_level=logging.INFO)

for wave_construction, polar_projection in zip(("DFTpolar", "DFTcartesian", "FFT"),
                                               (True, True, False)):
    if wave_construction != "DFTpolar":
        continue
    logger.info("Creating Wave1D")
    wave1D_1 = wf.Wave1D(n_kx_nodes=64, nx_points=251, Lx=500, n_bins_equal_energy=16,
                         wave_construction=wave_construction,
                         Tp=10, spectrum_type="jonswap", Hs=3, spectral_version="hmc", sigma=0.12,
                         wave_selection="EqualEnergyBins",
                         use_subrange_energy_limits=True)
    wave1D_1.make_report()
    wave1D_1.plot_spectrum(plot_title=wave_construction, add_hs_estimate=True)
    wave1D_1.plot_wave(plot_title=wave_construction, y_min=-2, y_max=2)
    continue
    # plt.show()
    # movie1d = wave1D_1.animate_wave(y_min=-2, y_max=2)

    logger.info("Creating Wave2D")
    with Timer(message=wave_construction) as t:
        wave2D_1 = wf.Wave2D(wave1D=wave1D_1,
                             nx_points=256,
                             ny_points=256,
                             n_theta_nodes=60,
                             Lx=500,
                             Ly=500,
                             Theta_0=np.deg2rad(215),
                             Theta_s_spreading_factor=13,
                             )
    wave2D_1.make_report()
    wave2D_1.plot_spectrum(plot_title=wave_construction, polar_projection=polar_projection,  #
                           use_contourf=True,
                           shift_origin=False,
                           r_axis_lim=(0, 0.2)
                           )
    continue
    #
    logger.info("Creating plot")
    wave2D_1.plot_wave(plot_title=wave_construction,
                       use_contourf=True,
                       min_data_value=-2,
                       max_data_value=2,
                       color_map=cc.m_coolwarm,
                       )

    logger.info("Creating movie")
    wave1D_1.reset_time(delta_t=0.5, nt_samples=1000)
    movie = wave2D_1.animate_wave(plot_title=wave_construction, min_data_value=-2.0,
                                  max_data_value=2.0,
                                  use_contourf=True,
                                  color_map=cc.m_coolwarm,
                                  add_hs_estimate=True)
plt.show()
