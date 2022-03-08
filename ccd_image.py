"""
This module contains the CCDImage class, which simplifies the process of peak
tracking, clustering and data reduction.
"""

# pylint: disable=invalid-name

from dataclasses import dataclass

import numpy as np
import pywt
from scipy.ndimage import uniform_filter, gaussian_filter


@dataclass
class Metadata:
    """
    Simple dataclass storing some important scan metadata.
    """
    beam_centre_x: int
    beam_centre_y: int


class CCDImage:
    """
    Class for analysing scientific CCD images containing magnetic diffraction
    data.

    Attrs:
        raw_data:
            A numpy array containing the raw CCD data.
    """

    def __init__(self, raw_data: np.ndarray, bkg: np.ndarray,
                 significance_mask: np.ndarray, metadata: Metadata) -> None:
        # Save the image as it was stored on disc.
        self.raw = raw_data
        # Save a version of the data that we'll manipulate. We can reset this
        # to self.raw by running self.reset()
        self.data = np.copy(self.raw)
        # Save the rest of the arguments.
        self.bkg = bkg
        self.significance_mask = significance_mask
        self.metadata = metadata

        # We should initialize this later using self.init_significant_pixels().
        self.significant_pixels = None

    @property
    def significant_pixels_DBSCAN_compatible(self):
        """
        Returns the significant pixels in a scikit-learn compatible form.
        """
        # Get all of the coordinates of the significant pixels.
        pixels_y, pixels_x = np.where(self.significant_pixels == 1)

        # Massage these pixels into the form that sklearn wants to see.
        pixel_coords = np.zeros((len(pixels_x), 2))
        pixel_coords[:, 0] = pixels_x
        pixel_coords[:, 1] = pixels_y
        return pixel_coords

    def reset(self):
        """
        Reset the state of this instance of CCDImage.
        """
        self.data = np.copy(self.raw)

    def subtract_bkg(self):
        """
        Returns the background subtracted image. Clips any pixels that were
        below the background.
        """
        self.data -= self.bkg
        # Kill any pixels that were below the background.
        self.data = np.clip(self.data, 0, np.inf)

    def wavelet_denoise(self,
                        cutoff_factor: float = 0.2,
                        wavelet_choice: str = "sym4",
                        high_freqs_to_kill: int = 3) -> None:
        """
        Runs some wavelet denoising on the image. Without arguments, will run
        default denoising.

        Args:
            cutoff_factor:
                If any wavelet coefficient is less than cutoff_factor*(maximum
                wavelet coefficient), then set it to zero. The idea is that
                small coefficients are required to represent noise; meaningful
                data, as long as it is large compared to background, will
                require large coefficients to be constructed in the wavelet
                representation.
            wavelet_choice:
                Fairly arbitrary. Defaults to sym4. Look at
                http://wavelets.pybytes.com/ for more info.
            high_freqs_to_kill:
                We will completely remove high frequency wavelets; beyond a
                cutoff, they aren't necessary at all to represent out data. They
                are, on the other hand, essential for the representation of
                noise. The number of high frequency bands that should be killed
                depends on:
                    1) Your wavelet choice.
                    2) The pixel length-scale of your data. Larger length scales
                       -> you should kill more frequency bands, and vice versa.
        """
        coeffs = pywt.wavedec(self.data, wavelet_choice)
        new_coeffs = [arr for arr in coeffs]
        for i in range(high_freqs_to_kill):
            new_coeffs[-(i+1)] = np.zeros_like(new_coeffs[-(i+1)])

        # Work out the largest wavelet coefficient.
        max_coeff = 0
        for arr in new_coeffs:
            max_coeff = np.max(arr) if np.max(arr) > max_coeff else max_coeff

        # Get min_coeff from the arguments to this method.
        min_coeff = max_coeff*cutoff_factor

        # Apply the decimation.
        new_coeffs = [np.where(
            ((arr > min_coeff).any() or (arr < -min_coeff).any()).any(), arr, 0
        ) for arr in new_coeffs]

        # Invert the wavelet transformation.
        self.data = pywt.waverec(new_coeffs, wavelet_choice)

    def init_significant_pixels(self, signal_length_scale: int,
                                bkg_length_scale: int) -> None:
        """
        Returns a significance map of the pixels in self.data.

        Args:
            signal_length_scale:
                The length scale over which signal is present. This is usually
                just a few pixels for typical magnetic diffraction data.
            bkg_length_scale:
                The length scale over which background level varies in a CCD
                image. If your CCD is perfect, you can set this to the number
                of pixels in a detector, but larger numbers will run more
                slowly. Typically something like 1/10th of the number of pixels
                in your detector is probably sensible.
        """
        # Compute local statistics.
        local_signal = gaussian_filter(self.data, signal_length_scale)
        local_bkg_levels = uniform_filter(local_signal, bkg_length_scale)
        local_deviations = local_signal - local_bkg_levels

        # Compute significance; return masked significance.
        significant_pixels = np.where(local_signal > 2*local_deviations, 0, 1)
        self.significant_pixels = significant_pixels*self.significance_mask

    @property
    def mean_signal_radius(self):
        """
        Computes the significant pixels array. Returns the mean distance from
        the beam centre to the significant pixels. This will return nonsense
        if self.data hasn't been properly background subtracted, denoised etc..
        """
        peak_pixels = np.where(self.significant_pixels == 1)

        dx_sq = np.square(peak_pixels[1] - self.metadata.beam_centre_x)
        dy_sq = np.square(peak_pixels[0] - self.metadata.beam_centre_y)

        radii_sq = dx_sq + dy_sq
        radii = np.sqrt(radii_sq)
        return np.mean(radii)
