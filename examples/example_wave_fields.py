import logging

import matplotlib.pyplot as plt
from hmc_utils.misc import (create_logger, Timer)

import hmc_marine.wave_fields as wf

logger = create_logger(console_log_level=logging.INFO)

# spectrum_type = "gauss"
spectrum_type = "jonswap"

wave1 = wf.Wave1D(delta_t=1, spectrum_type=spectrum_type, xmax=3000, n_kx_nodes=512, kx_max=3.14,
                  nt_samples=1000, nx_points=512, wave_construction="DFTpolar")
# wave2 = wf.Wave1D(delta_t=1, spectrum_type=spectrum_type, xmax=3000, wave_selection="Subrange",
# n_kx_nodes=64)
wave2 = wf.Wave1D(delta_t=1, spectrum_type=spectrum_type, xmax=3000,
                  wave_selection="EqualEnergyBins",
                  lock_nodes_to_wave_one=False, use_subrange_energy_limits=False,
                  n_bins_equal_energy=20,
                  kx_max=3.14, nt_samples=1000, nx_points=512, wave_construction="DFTpolar",
                  spectral_version="hmc")

wave3 = wf.Wave1D(delta_t=1, spectrum_type=spectrum_type, xmax=3000, n_kx_nodes=1024, kx_max=3.14,
                  nt_samples=1000, nx_points=1024, wave_construction="FFT")

wave1.make_report()
wave2.make_report()
wave3.make_report()

fig, axis = wave1.plot_spectrum(add_limit_markers=True, add_hs_estimate=True, plot_title="Compare "
                                                                                         "Spectra")
fig, axis = wave2.plot_spectrum(fig=fig, axis=axis, linecolor="r", linestyle="--", line_markers="v",
                                y_n_points_label=0.7, markersize=10)

fig, axis = wave1.plot_wave(plot_title="Wave compare")
fig, axis = wave2.plot_wave(fig=fig, ax=axis, linestyle="--")
fig, axis = wave3.plot_wave(fig=fig, ax=axis, linestyle="-.")

with Timer(message=wave1.name) as t:
    while wave1.time < wave1.t_end:
        wave1.propagate_wave()

with Timer(message=wave2.name) as t:
    while wave2.time < wave2.t_end:
        wave2.propagate_wave()

with Timer(message=wave3.name) as t:
    while wave3.time < wave3.t_end:
        wave3.propagate_wave()

plt.show()

wave1.reset_time(nt_samples=100, delta_t=0.1)
# wave2.reset_time(nt_samples=1000, delta_t=0.1)
# wave3.reset_time(nt_samples=1000, delta_t=0.1)

# movie = wave1.animate_wave(interval=1, y_min=-1, y_max=1, figsize=(12,4))
# plt.show()

# movie = wave2.animate_wave(interval=1, y_min=-1, y_max=1, figsize=(12,4))
# plt.show()
#
# movie = wave3.animate_wave(interval=1, y_min=-1, y_max=1, figsize=(12,4))
# plt.show()
