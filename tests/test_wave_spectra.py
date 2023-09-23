#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

import numpy as np
from numpy import pi
from numpy.testing import assert_almost_equal, assert_equal

from pymarine.waves.wave_spectra import (
    omega_peak_jonswap,
    alpha_jonswap,
    spreading_function2,
    spreading_function,
    spectrum_gauss,
    spectrum_jonswap,
    spectrum_jonswap_k_domain_2,
    omega_deep_water,
    spectrum_wave_k_domain,
    spectrum_to_complex_amplitudes,
    initialize_phase,
    spectrum2d_complex_amplitudes,
    omega_e_vs_omega,
    d_omega_e_prime,
    omega_critical,
    omega_vs_omega_e,
    spectrum_to_spectrum_encountered,
    spectrum2d_to_spectrum2d_encountered,
    mask_out_of_range,
    specspecs,
    thetaspreadspecs,
    rayleigh_pdf,
    rayleigh_cdf,
    set_heading,
)


def test_omega_peak_jonswap():
    peak_expected = 4.678462267825839
    peak = omega_peak_jonswap(wind_speed=10, fetch=1000)
    assert_almost_equal([peak], [peak_expected])


def test_alpha_jonswap():
    alpha_expected = 0.0277127
    alpha = alpha_jonswap(wind_speed=10, fetch=1000)
    assert_almost_equal([alpha], [alpha_expected])


def test_spreading_function2():
    n_size = 10
    directions = np.linspace(0, 360, n_size, endpoint=False)
    result = spreading_function2(theta=directions, theta0=100, s_spreading_factor=5)
    result_expected = np.array(
        [
            4.5273054e-01,
            1.0557061e-01,
            1.4777023e-09,
            9.2072475e-03,
            6.4647155e-01,
            1.1260938e-02,
            3.7390244e-10,
            9.3699574e-02,
            4.7425908e-01,
            2.5123128e-04,
        ]
    )
    assert_almost_equal(result, result_expected)


def test_spreading_function():
    n_size = 10
    directions = np.linspace(0, 2 * pi, n_size, endpoint=False)
    result_expected = np.array(
        [0.6366198, 0.4166731, 0.0607918, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0607918, 0.4166731]
    )
    result = spreading_function(theta=directions, theta0=0, n_spreading_factor=2)
    assert_almost_equal(result, result_expected)


def test_spectrum_gauss():
    n_size = 10
    frequencies = np.linspace(0, 2.5, n_size)
    result = spectrum_gauss(frequencies, Hs=2, Tp=10, sigma=2.4, spectral_version="dnv")
    result_expected = np.array(
        [
            0.0401565,
            0.0411156,
            0.0415374,
            0.0414051,
            0.0407241,
            0.0395213,
            0.0378436,
            0.035755,
            0.0333321,
            0.0306599,
        ]
    )
    assert_almost_equal(result, result_expected)

    result = spectrum_gauss(frequencies, Hs=2, Tp=10, spectral_version="sim")
    result_expected = np.array(
        [
            3.0615835e-022,
            2.7659152e-007,
            8.1180934e-001,
            7.7409033e-003,
            2.3980138e-013,
            2.4134267e-032,
            7.8911262e-060,
            8.3823596e-096,
            2.8927840e-140,
            3.2433077e-193,
        ]
    )
    assert_almost_equal(result, result_expected)

    n_size = 10000
    frequencies = np.linspace(0, 10, n_size)
    delta_f = np.diff(frequencies)[0]
    hs_values = np.linspace(0, 10, num=21, endpoint=True)
    for hs_in in hs_values:
        # check version 0
        result = spectrum_gauss(
            omega=frequencies, Hs=hs_in, Tp=10, spectral_version="dnv"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in)

        # check version 1
        result = spectrum_gauss(
            omega=frequencies, Hs=hs_in, Tp=10, spectral_version="sim"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in)

    hs_in = 3
    tp_values = np.linspace(1.0, 20, num=21, endpoint=True)
    sigma_values = np.linspace(0.01, 0.1, num=5, endpoint=True)
    for tp in tp_values:
        # check version 0
        for sigma in sigma_values:
            result = spectrum_gauss(
                omega=frequencies, Hs=hs_in, Tp=tp, sigma=sigma, spectral_version="dnv"
            )
            hs_out = 4 * np.sqrt(result.sum() * delta_f)
            assert_almost_equal(hs_out, hs_in, decimal=1)

        # check version 1
        result = spectrum_gauss(
            omega=frequencies, Hs=hs_in, Tp=tp, spectral_version="sim"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in, decimal=3)


def test_spectrum_jonswap():
    n_size = 10
    frequencies = np.linspace(0, 2.5, n_size)
    result = spectrum_jonswap(
        omega=frequencies, Hs=3.0, Tp=11, gamma=3.5, spectral_version="dnv"
    )
    result_expected = np.array(
        [
            0.0000000e00,
            2.2801231e-08,
            2.8604450e00,
            3.6208292e-01,
            1.0375968e-01,
            3.5798406e-02,
            1.4655414e-02,
            6.8345663e-03,
            3.5190394e-03,
            1.9568243e-03,
        ]
    )
    assert_almost_equal(result, result_expected)

    # test the other version as well
    result = spectrum_jonswap(
        omega=frequencies, Hs=3, Tp=11, gamma=3.5, spectral_version="sim"
    )
    result_expected = np.array(
        [
            0.0000000e00,
            2.2749758e-08,
            2.8539876e00,
            3.6126553e-01,
            1.0352545e-01,
            3.5717592e-02,
            1.4622330e-02,
            6.8191375e-03,
            3.5110953e-03,
            1.9524068e-03,
        ]
    )
    assert_almost_equal(result, result_expected)

    # jonswap 2 should give the same result as jonswap with 'version=1' flag on
    result2 = spectrum_jonswap(
        omega=frequencies, Hs=3, Tp=11, gamma=3.5, spectral_version="sim"
    )
    assert_almost_equal(result2, result_expected)

    # test the Hs on a higher resolution
    n_size = 10000
    frequencies = np.linspace(0, 10, n_size)
    delta_f = np.diff(frequencies)[0]
    hs_values = np.linspace(0, 10, num=21, endpoint=True)
    for hs_in in hs_values:
        # check version 0
        result = spectrum_jonswap(
            omega=frequencies, Hs=hs_in, Tp=11, gamma=3.5, spectral_version="sim"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in, decimal=2)

        # check version 1
        result = spectrum_jonswap(
            omega=frequencies, Hs=hs_in, Tp=11, gamma=3.5, spectral_version="sim"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in, decimal=2)

    hs_in = 10.0
    gamma_values = np.linspace(1, 6, num=21, endpoint=True)
    for gamma in gamma_values:
        # check version 0
        result = spectrum_jonswap(
            omega=frequencies, Hs=hs_in, Tp=11, gamma=gamma, spectral_version="dnv"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in, decimal=1)

        # check version 1
        result = spectrum_jonswap(
            omega=frequencies, Hs=hs_in, Tp=11, gamma=gamma, spectral_version="sim"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in, decimal=1)

    tp_values = np.linspace(4, 20, num=21, endpoint=True)
    for tp in tp_values:
        # check version 0
        result = spectrum_jonswap(
            omega=frequencies, Hs=hs_in, Tp=tp, gamma=3.3, spectral_version="dnv"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in, decimal=2)

        # check version 1
        result = spectrum_jonswap(
            omega=frequencies, Hs=hs_in, Tp=tp, gamma=3.3, spectral_version="sim"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_f)
        assert_almost_equal(hs_out, hs_in, decimal=2)


def test_omega_deep_water():
    n_size = 10
    wave_numbers = np.linspace(0, 2 * np.pi / 100, n_size)
    result = omega_deep_water(wave_number_vector=wave_numbers)
    result_expected = np.array(
        [
            0.0,
            0.261655,
            0.370036,
            0.4531997,
            0.52331,
            0.5850783,
            0.6409212,
            0.692274,
            0.7400721,
            0.784965,
        ]
    )
    assert_almost_equal(result, result_expected)


def test_spectrum_jonswap_k_domain():
    n_size = 10
    wave_numbers = np.linspace(0, 2 * np.pi / 100, n_size)
    result = spectrum_wave_k_domain(
        k_waves=wave_numbers,
        Hs=10,
        Tp=12,
        gamma=3.5,
        spectrum_type="jonswap",
        spectral_version="dnv",
    )
    result_expected = np.array(
        [
            0.0000000e00,
            4.5328519e-05,
            1.9150280e01,
            1.1190881e02,
            3.5913257e02,
            1.4081188e02,
            6.4523656e01,
            4.4606775e01,
            3.2823016e01,
            2.4618044e01,
        ]
    )
    assert_almost_equal(result, result_expected, decimal=5)

    result = spectrum_wave_k_domain(
        k_waves=wave_numbers,
        Hs=10,
        Tp=12,
        gamma=3.5,
        spectrum_type="jonswap",
        spectral_version="sim",
    )
    result_expected = np.array(
        [
            0.0000000e00,
            4.5226191e-05,
            1.9107049e01,
            1.1165618e02,
            3.5832184e02,
            1.4049400e02,
            6.4377996e01,
            4.4506077e01,
            3.2748920e01,
            2.4562470e01,
        ]
    )
    assert_almost_equal(result, result_expected, decimal=5)

    n_size = 10000
    wave_numbers = np.linspace(0, 10, n_size)
    delta_k = np.diff(wave_numbers)[0]
    hs_values = np.linspace(0, 10, num=21, endpoint=True)
    for hs_in in hs_values:
        # check version 0
        result = spectrum_wave_k_domain(
            k_waves=wave_numbers,
            Hs=hs_in,
            Tp=11,
            gamma=3.5,
            spectrum_type="jonswap",
            spectral_version="dnv",
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_k)
        assert_almost_equal(hs_out, hs_in, decimal=2)

        # spectrum_jonswap_k_domain_2 is the explicit verison of the transormed jonswap spectrum. Should yield the
        # same results as the
        result2 = spectrum_jonswap_k_domain_2(
            k_waves=wave_numbers, Hs=hs_in, Tp=11, gamma=3.5, spectral_version="dnv"
        )
        assert_almost_equal(result, result2)

        result = spectrum_wave_k_domain(
            k_waves=wave_numbers, Hs=hs_in, Tp=11, gamma=3.5, spectrum_type="gauss"
        )
        hs_out = 4 * np.sqrt(result.sum() * delta_k)
        assert_almost_equal(hs_out, hs_in, decimal=4)


def test_initialize_phase():
    n_size = 10
    wave_numbers = np.linspace(0, 2 * np.pi / 100, n_size)
    result = initialize_phase(k_waves=wave_numbers, seed=1)
    result_expected = np.array(
        [
            2.6202265e00,
            4.5259323e00,
            7.1863817e-04,
            1.8996116e00,
            9.2209446e-01,
            5.8018050e-01,
            1.1703074e00,
            2.1712221e00,
            2.4929636e00,
            3.3854854e00,
        ]
    )
    assert_almost_equal(result, result_expected)


def test_spectrum_to_complex_amplitudes():
    n_size = 10
    wave_numbers = np.linspace(0, 2 * np.pi / 100, n_size)
    spectrum = spectrum_wave_k_domain(
        k_waves=wave_numbers,
        Hs=10,
        Tp=12,
        gamma=3.5,
        spectrum_type="jonswap",
        spectral_version="dnv",
    )
    phase = initialize_phase(k_waves=wave_numbers, seed=1)

    result = spectrum_to_complex_amplitudes(
        wave_numbers, spectral_modulus=spectrum, phase=phase, mirror=False
    )

    result_expected = [
        -0.00000000e00 + 0.00000000e00j,
        -1.47478283e-04 - 7.81764461e-04j,
        5.17095937e-01 + 3.71604943e-04j,
        -4.03657860e-01 + 1.18304777e00j,
        1.35287488e00 + 1.78442321e00j,
        1.17273127e00 + 7.68639107e-01j,
        3.70050868e-01 + 8.74060956e-01j,
        -0.445890050764 + 0.651160605446j,
        -0.539490341563 + 0.408957133181j,
        -0.568936261313 - 0.141577804j,
    ]

    assert_almost_equal(result, result_expected)


def test_spectrum_to_complex_amplitudes_mirror():
    n_size = 5
    wave_numbers = np.linspace(0, 2 * np.pi / 100, n_size)
    spectrum = spectrum_wave_k_domain(
        k_waves=wave_numbers,
        Hs=10,
        Tp=12,
        gamma=3.5,
        spectrum_type="jonswap",
        spectral_version="dnv",
    )
    phase = initialize_phase(k_waves=wave_numbers, seed=1)

    result = spectrum_to_complex_amplitudes(
        wave_numbers, spectral_modulus=spectrum, phase=phase, mirror=True
    )

    result_expected = np.array(
        [
            -0.00000000 + 0.0j,
            -0.20406898 - 1.08174486j,
            2.83300855 + 0.00203591j,
            2.83300855 - 0.00203591j,
            -0.20406898 + 1.08174486j,
        ]
    )

    assert_almost_equal(result, result_expected)


def test_jonswap2D_complex_amplitudes():
    n_size = 4
    wave_numbers_x = np.linspace(0, 2 * np.pi / 100.0, n_size)
    wave_numbers_y = np.linspace(0, 2 * np.pi / 100.0, n_size)

    ak, os = spectrum2d_complex_amplitudes(
        kx_nodes=wave_numbers_x,
        ky_nodes=wave_numbers_y,
        Theta_0=pi / 6,
        Hs=2.1,
        Tp=12.2,
        gamma=3.3,
        Theta_s_spread_kx=5.0,
        mirror=True,
        seed=1,
    )

    ak_exp_real = np.array(
        [
            [0.0, -0.1108242, -0.0937341, -0.1108242],
            [-0.1718166, 0.2862183, -0.1266911, 0.0244118],
            [-0.0177563, 0.1111589, -0.1007618, 0.1111589],
            [-0.1718166, 0.0244118, -0.1266911, 0.2862183],
        ]
    )

    ak_exp_imag = np.array(
        [
            [0.0, 0.3248055, 0.1368857, -0.3248055],
            [0.0986876, 0.377518, 0.0960374, 0.0829592],
            [-0.0941239, 0.0728565, -0.0250742, -0.0728565],
            [-0.0986876, -0.0829592, -0.0960374, -0.377518],
        ]
    )

    os_exp = np.array([[1, 1, 1, -1], [1, 1, 1, 1], [1, 1, 1, -1], [-1, -1, -1, -1]])

    assert_almost_equal(ak.real, ak_exp_real)
    assert_almost_equal(ak.imag, ak_exp_imag)
    assert_almost_equal(os, os_exp)

    def check_symmetry(nx, ny, theta_zero, hs_target=2.1, debug_output=False):
        L = 1000
        dx = L / nx
        dy = L / ny
        kx_nodes = 2 * np.pi * np.fft.fftfreq(nx, dx)
        ky_nodes = 2 * np.pi * np.fft.fftfreq(ny, dy)
        ak, os = spectrum2d_complex_amplitudes(
            kx_nodes=kx_nodes,
            ky_nodes=ky_nodes,
            Theta_0=theta_zero,
            Hs=hs_target,
            Tp=12.2,
            gamma=3.3,
            Theta_s_spread_kx=5.0,
            mirror=True,
            seed=1,
        )

        # Parseval's theorem to calculate the variance from the complex fourier coefficients
        # divide by 2 as you only use half of the domain
        hs_estimate = 4 * np.sqrt(np.square(abs(ak)).sum()) / 2
        assert_almost_equal(hs_target, hs_estimate, decimal=1)

        if debug_output:
            print("calculating {} {} {}".format(nx, ny, theta_zero))
            # print out hs. You can compare with amplitudes if you want
            N = int(os.size / 2)
            amplitudes = N * np.fft.ifft2(ak)
            hs_et2 = 4 * np.std(amplitudes)
            print(
                "nx={} ny={} theta={}  {:.2f}  {:.2f}".format(
                    nx, ny, theta_zero, hs_target, hs_estimate
                )
            )

            # for debugging only if you want to view the maxtix to see if it is symmetric
            print()
            for j1 in range(ny):
                line = ""
                for i1 in range(nx):
                    line += "({:8.5f},{:8.5f}) ".format(
                        ak[i1][j1].real, ak[i1][j1].imag
                    )
                print(line)
        # this is an explicit loop over the complex amplitudes to check if it is point mirrored
        for i1 in range(nx):
            for j1 in range(ny):
                if i1 == 0:
                    i2 = 0  # we are at the y-axis, which is mirrored to the - y-axis
                else:
                    i2 = nx - 1 - (i1 - 1)
                if j1 == 0:
                    j2 = 0  # we are at the x-axis, which is mirrored to the - x-axis
                else:
                    j2 = ny - 1 - (j1 - 1)
                ak_1 = ak[i1][j1]
                ak_2 = ak[i2][j2].conjugate()

                if i1 == i2 and j1 == j2:
                    # this is itself, skip as it it the same by definition and not a conjugate
                    continue

                # test if coefficient and its point mirrored partner are each others conjugate
                assert_almost_equal(ak_1, ak_2)

    n_size = 32
    # check both even and odd number of nodes in mesh and loop over a series of angles
    for i_off in [0, 1]:
        for j_off in [0, 1]:
            nx = n_size + i_off
            ny = 2 * n_size + j_off
            for theta_0 in np.linspace(0, 2 * pi, 7, endpoint=True):
                check_symmetry(nx, ny, theta_zero=theta_0)


def test_omega_e_vs_omega():
    n_size = 10
    frequencies = np.linspace(0, 2.5, n_size)
    result = omega_e_vs_omega(frequencies, u_monitor=1.2)

    result_expected = np.array(
        [
            0.0,
            0.268336,
            0.5177883,
            0.748357,
            0.960042,
            1.1528435,
            1.3267613,
            1.4817954,
            1.6179459,
            1.7352128,
        ]
    )

    assert_almost_equal(result, result_expected)


def test_d_omega_e_prime():
    n_size = 10
    frequencies = np.linspace(0, 2.5, n_size)
    result = d_omega_e_prime(frequencies, velocity=2.2)
    result_expected = np.array(
        [
            1.0,
            0.875368,
            0.750736,
            0.6261041,
            0.5014721,
            0.3768401,
            0.2522081,
            0.1275761,
            0.0029441,
            -0.1216878,
        ]
    )
    assert_almost_equal(result, result_expected)


def test_omega_critical():
    n_size = 10
    om_c, index_c = omega_critical(velocity=2)
    om_c_exp = 2.4516625
    index_c_exp = 4
    assert_almost_equal([om_c], [om_c_exp])
    assert_equal([index_c], [None])

    frequencies = np.linspace(0, 5.5, n_size)
    om_c, index_c = omega_critical(velocity=2, omega=frequencies)
    assert_almost_equal([om_c], [om_c_exp])
    assert_equal([index_c], [index_c_exp])


def test_omega_vs_omega_e():
    result = omega_vs_omega_e(omega=1.2, velocity=1.2)

    result_expected = np.array([6.7109084, 1.4612999])

    assert_almost_equal(result, result_expected)


def test_spectrum_to_spectrum_encountered():
    n_size = 4
    frequencies = np.linspace(0, 2.5, n_size)
    spectrum_js = spectrum_jonswap(
        omega=frequencies, Hs=1, Tp=11, gamma=3.4, spectral_version="dnv"
    )

    result, results2 = spectrum_to_spectrum_encountered(
        spectrum=spectrum_js, frequencies=frequencies, velocity=1
    )
    result_expected = np.array([0.0409422, 0.0212782, 0.0010317, 0.0004494])

    assert_almost_equal(result, result_expected)


def test_spectrum2d_to_spectrum2d_encountered():
    n_size = 4
    frequencies = np.linspace(0, 2.5, n_size)
    directions = np.linspace(0, 2 * np.pi, n_size, endpoint=False)
    spectrum_js = spectrum_jonswap(
        omega=frequencies, Hs=1, Tp=11, gamma=3.4, spectral_version="dnv"
    )
    directional_dist = spreading_function(
        theta=directions, theta0=34, n_spreading_factor=2.3
    )
    ff_2d, dd_2d = np.meshgrid(frequencies, directions, indexing="ij")
    ss_om, dd_sp = np.meshgrid(spectrum_js, directional_dist, indexing="ij")
    s_2d = ss_om * dd_sp

    result = spectrum2d_to_spectrum2d_encountered(
        spectrum_2d=s_2d, frequencies=ff_2d, directions=dd_2d, velocity=0.1
    )
    result_expected = np.array(
        [
            [0.0000000e00, 6.3386893e-03, 1.8629551e-02, 0.0000000e00],
            [0.0000000e00, 3.2976248e-03, 9.7615486e-03, 0.0000000e00],
            [0.0000000e00, 1.4540844e-04, 5.0603899e-04, 0.0000000e00],
            [0.0000000e00, 3.4256523e-05, 1.1853199e-04, 0.0000000e00],
        ]
    )

    assert_almost_equal(result, result_expected)


def test_mask_out_of_range():
    n_size = 20
    wave_numbers = np.linspace(0, 2 * np.pi / 100, n_size)
    result = mask_out_of_range(kx=wave_numbers, kmin=0.01, kmax=0.03)

    result_expected = np.array(
        [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        dtype=bool,
    )

    assert_equal(result, result_expected)


def test_specspecs():
    n_size = 20
    frequencies = np.linspace(0, 2.5, n_size)
    spectrum_js = spectrum_jonswap(
        omega=frequencies, Hs=3, Tp=11, gamma=3.5, spectral_version="dnv"
    )
    result = specspecs(
        frequency=frequencies, amplitude=spectrum_js, lowlim=0.01, higlim=0.9
    )

    result_expected = (
        3,
        7,
        4,
        0.39473684210526316,
        0.92105263157894735,
        0.52631578947368418,
        1.6336106158889372,
        0.4975741649385097,
    )

    assert_almost_equal(result, result_expected)


def test_thetaspreadspecs():
    n_size = 20
    directions = np.linspace(0, 2 * np.pi, n_size, endpoint=False)
    dd_spread = spreading_function2(theta=directions, theta0=100, s_spreading_factor=5)
    result = thetaspreadspecs(theta=directions, Dspread=dd_spread, areafraction=0.9)

    result_p1 = result[:7]
    result_p2 = result[7]

    result_expected_p1 = (
        15,
        1,
        4.7123889803846897,
        0.31415926535897931,
        5.6548667764616276,
        0.63910521149421706,
        0.99999999999999811,
    )

    result_expected_p2 = np.array(
        [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        dtype=bool,
    )

    assert_almost_equal(result_p1, result_expected_p1)
    assert_equal(result_p2, result_expected_p2)


def test_rayleigh_pdf():
    n_size = 10
    frequencies = np.linspace(0, 2.5, n_size)
    result = rayleigh_pdf(omega=frequencies, sigma=3.1)
    result_expected = np.array(
        [
            0.0,
            0.007219,
            0.0143946,
            0.0214839,
            0.0284446,
            0.035236,
            0.041819,
            0.0481564,
            0.0542136,
            0.0599585,
        ]
    )
    assert_almost_equal(result, result_expected)


def test_rayleigh_cdf():
    n_size = 10
    frequencies = np.linspace(0, 2.5, n_size)
    result = rayleigh_cdf(omega=frequencies, sigma=3.1)
    result_expected = np.array(
        [
            1.0,
            0.9989969,
            0.9959935,
            0.9910078,
            0.9840699,
            0.975221,
            0.9645136,
            0.9520109,
            0.937786,
            0.9219212,
        ]
    )
    assert_almost_equal(result, result_expected)


def test_set_heading():
    n_size = 4
    np.random.seed(0)
    rao_data = np.random.rand(n_size, n_size)
    directions = np.linspace(0, 2 * np.pi, n_size, endpoint=False)
    result = set_heading(
        data=rao_data, directions=directions, heading=232, direction_axis=1
    )

    result_expected = np.array(
        [
            [0.5448832, 0.5488135, 0.7151894, 0.6027634],
            [0.891773, 0.4236548, 0.6458941, 0.4375872],
            [0.5288949, 0.9636628, 0.3834415, 0.791725],
            [0.0871293, 0.5680446, 0.9255966, 0.0710361],
        ]
    )
    assert_almost_equal(result, result_expected)


def main():
    # run all the unit test here. This is only needed if the test_wave_spectra script is executed as
    # a python script for debuggin purpose only. In case that the setup test suit is run, this is
    # not needed at all
    test_omega_peak_jonswap()
    test_alpha_jonswap()
    test_spreading_function2()
    test_spreading_function()
    test_spectrum_gauss()
    test_spectrum_jonswap()
    test_omega_deep_water()
    test_spectrum_jonswap_k_domain()
    test_spectrum_to_complex_amplitudes()
    test_jonswap2D_complex_amplitudes()
    test_omega_e_vs_omega()
    test_d_omega_e_prime()
    test_omega_critical()
    test_omega_vs_omega_e()
    test_spectrum_to_spectrum_encountered()
    test_spectrum2d_to_spectrum2d_encountered()
    test_mask_out_of_range()
    test_specspecs()
    test_thetaspreadspecs()
    test_rayleigh_pdf()
    test_rayleigh_cdf()
    test_set_heading()


if __name__ == "__main__":
    main()
