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

    mean_pressure = 2e-3
    mean_radius = 1e0
    std_pressure = 0.2
    std_radius = 0.6
    weight = 0.001
    rotation_angle = 45

    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    grid_points = np.vstack([np.log10(X).ravel() - np.log10(mean_pressure),
                             np.log10(Y).ravel() - np.log10(mean_radius)])
    rotated_grid = rotation_matrix @ grid_points

    rotated_X, rotated_Y = rotated_grid[0], rotated_grid[1]
    Z_component = np.exp(
        -((rotated_X / std_pressure) ** 2 + (rotated_Y / std_radius) ** 2)
    ).reshape(X.shape)

    Z += weight * Z_component

    Z_min = 0
    Z_max = 1e1
    Z = (Z / Z.max()) * (Z_max - Z_min) + Z_min

    threshold = 1e-4
    Z[Z < threshold] = 0

    cloud_particle_size_distrib = np.expand_dims(Z, axis=-1)

    return cloud_particle_size_distrib

def test_correlated_k_transmission_arbitrary_cloud_particle_size_grid():
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
        cloud_particle_number_density_grid=make_cloud_particle_size_grid(),
        frequencies_to_wavelengths=True
    )