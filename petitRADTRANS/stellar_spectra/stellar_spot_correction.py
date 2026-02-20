"""Stellar spot correction
"""
import numpy as np

# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
import petitRADTRANS
import petitRADTRANS.physical_constants as cst


def rackham_stellar_spot_correction(prt_object, star_effective_temperature: float,
                                    star_spot_effective_temperature: float, spot_coverage: float, stellar_model=None,
                                    spot_model=None) -> np.ndarray:
    """Function to compute the stellar spot correction following Rackham et al. 2018 Eq. 2

    Args:
        prt_object: Radtrans object
            The Radtrans object containing the wavelength grid.
        star_effective_temperature: float
            The temperature of the unspotted stellar photosphere in K.
        star_spot_effective_temperature: float
            The temperature of the spots in K.
        spot_coverage: float
            The fraction of the stellar surface covered by spots (between 0 and 1).
        stellar_model: numpy.ndarray, optional
            A 2D array containing the wavelengths (in cm) and fluxes (in erg/cm^2/s/Hz) of the stellar model.
            If None, a Phoenix model will be used.
        spot_model: numpy.ndarray, optional
            A 2D array containing the wavelengths (in cm) and fluxes (in erg/cm^2/s/Hz) of the spot model.
            If None, a Phoenix model will be used.

    Returns:
        correction_factor: numpy.ndarray
            The correction factor to be applied to the transit depth in the dimension of the wavelength.

    Reference:
    - Rackham et al. (2018). The Transit Light Source Effect: False Spectral Features and Incorrect Densities for
      M-dwarf Transiting Planets.
      The Astrophysical Journal, Volume 853, Issue 2, article id. 122, 18 pp.
      https://ui.adsabs.harvard.edu/link_gateway/2018ApJ...853..122R/doi:10.3847/1538-4357/aaa08c
    """

    if stellar_model is not None:
        if spot_model is None:
            raise ValueError("if an individual stellar model should be used, a spot model must also be provided.")

    wavelength_bin_edges = cst.c / prt_object._frequency_bins_edges * 1e4  # in micron

    wavelengths = np.array((prt_object._frequency_bins_edges[1:] + prt_object._frequency_bins_edges[:-1]) * 0.5)
    wavelengths = cst.c / wavelengths * 1e4  # in micron
    wavelength_bin_widths = np.diff(wavelength_bin_edges)

    # Compute the stellar spectra
    if stellar_model is None:
        star = petitRADTRANS.stellar_spectra.phoenix.phoenix_star_table
        star_data, _ = star.compute_spectrum(star_effective_temperature)

    else:
        star_data = stellar_model

    wavelengths_star = star_data[:, 0] * 1e4  # cm to um
    star_flux = star_data[:, 1]

    # Rebin the stellar spectrum to the prt_object resolution
    bin_flux_star = frebin.rebin_spectrum_bin(
                                              wavelengths_star,
                                              star_flux,
                                              wavelengths,
                                              wavelength_bin_widths
    )

    # Compute the spot spectrum
    if spot_model is None:
        star = petitRADTRANS.stellar_spectra.phoenix.phoenix_star_table
        spot_data, _ = star.compute_spectrum(star_spot_effective_temperature)

    else:
        spot_data = spot_model

    wavelengths_spots = spot_data[:, 0] * 1e4  # cm to um
    star_spot_flux = spot_data[:, 1]

    # rebin the spot spectrum to the prt_object resolution
    bin_flux_spot = frebin.rebin_spectrum_bin(
                                              wavelengths_spots,
                                              star_spot_flux,
                                              wavelengths,
                                              wavelength_bin_widths
    )

    # Compute the correction factor
    correction_factor = 1 - spot_coverage * (1 - bin_flux_spot / bin_flux_star)

    return correction_factor
