import copy

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import test_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results

# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters['pressures'],  #
        line_species=test_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(test_parameters['cloud_parameters']['cloud_species'].keys()),
        wavelength_boundaries=test_parameters['spectrum_parameters']['wavelength_range_correlated_k'],
        line_opacity_mode='c-k'
    )

    return atmosphere

atmosphere_ck = init_radtrans_correlated_k()

def make_cloud_particle_size_grid():
    """
    Create a grid of cloud particle sizes for the test.
    """

    pressures = test_parameters['pressures']
    particles_radii = np.logspace(-3, 5, 130) * 1e-4
    X, Y = np.meshgrid(pressures, particles_radii / 1e-4)

    return cloud_particle_sizes

def test_correlated_k_transmission_contribution_arbitrary_cloud_particle_size_grid():
    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = (
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=test_parameters['temperature_guillot_2010'] * np.ones_like(test_parameters['pressures']),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        return_contribution=True,
        frequencies_to_wavelengths=False
    )