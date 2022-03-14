"""
This module contains the CCDImage class, which simplifies the process of peak
tracking, clustering and data reduction.
"""

# pylint: disable=invalid-name

from dataclasses import dataclass
from typing import List

import numpy as np
import pywt
from scipy.ndimage import uniform_filter, gaussian_filter
from sklearn.cluster import DBSCAN

from cluster import Cluster


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
                 significance_mask: np.ndarray, metadata: Metadata,
                 store_raw=False) -> None:
        # Save a version of the data that we'll manipulate. We can reset this
        # to self.raw by running self.reset()
        self.data = raw_data

        # Save the image as it was stored on disc, if requested.
        if store_raw:
            self.raw = np.copy(raw_data)
        else:
            self.raw = None

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

    @property
    def _pixel_dx(self):
        """
        Returns the horizontal distance between each pixel and the beamstop.
        """
        horizontal_x = np.arange(0, self.data.shape[1])
        horizontal_dx = horizontal_x - self.metadata.beam_centre_x
        pixel_dx = np.zeros_like(self.data)
        for col in range(self.data.shape[1]):
            pixel_dx[:, col] = horizontal_dx[col]

        return pixel_dx

    @property
    def _pixel_dy(self):
        """
        Returns the vertical distance between each pixel and the beamstop.
        """
        vertical_y = np.arange(self.data.shape[0]-1, -1, -1)
        vertical_dy = vertical_y - (
            self.data.shape[0] - 1 - self.metadata.beam_centre_y
        )
        pixel_dy = np.zeros_like(self.data)
        for row in range(self.data.shape[0]):
            pixel_dy[row, :] = vertical_dy[row]

        return pixel_dy

    @property
    def pixel_radius(self):
        """
        Returns each pixel's radial distance from the beam centre, in units of
        pixels.
        """
        return np.sqrt(np.square(self._pixel_dx) + np.square(self._pixel_dy))

    def radial_mask(self, lower_bound: float, upper_bound: float) -> np.ndarray:
        """
        Returns a mask array which is true when self.pixel_radius is less than
        mask_radius.

        Args:
            lower_bound:
                The radius below which we want to mask pixels.
            upper_bound:
                The radius above which we want to mask pixels.

        Returns:
            2D boolean mask array. False -> unmasked; True -> masked.
        """
        inner_mask = np.where(self.pixel_radius <= lower_bound, True, False)
        outer_mask = np.where(self.pixel_radius >= upper_bound, True, False)
        return np.ma.mask_or(inner_mask, outer_mask)

    @property
    def theta(self):
        """
        Returns each pixel's theta value for a polar coordinate mapping.
        """
        return np.arctan2(self._pixel_dx, self._pixel_dy)

    def significance_levels(self, signal_length_scale: int,
                            bkg_length_scale: int) -> np.ndarray:
        """
        Returns an image of the local significance level of every pixel in the
        image.

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

        Returns:
            Array of standard deviations between the mean and each pixel.
        """
        # Compute local statistics.
        local_signal = gaussian_filter(self.data, int(signal_length_scale/3))
        local_bkg_levels = uniform_filter(local_signal, bkg_length_scale)
        total_deviaiton = np.std(local_signal)

        return np.abs((local_signal - local_bkg_levels)/total_deviaiton)

    def init_significant_pixels(self, signal_length_scale: int,
                                bkg_length_scale: int,
                                n_sigma: float = 4) -> None:
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
            n_sigma:
                The number of standard deviations above the mean that a pixel
                needs to be to be considered significant.
        """
        # Compute significance; return masked significance. Significant iff
        # pixel is more than 4stddevs larger than the local average.
        significant_pixels = np.where(self.significance_levels(
            signal_length_scale, bkg_length_scale) > n_sigma, 1, 0)
        self.significant_pixels = significant_pixels*self.significance_mask

    def cluster_significant_pixels(self, signal_length_scale: int,
                                   bkg_length_scale: int,
                                   n_sigma: float = 4) -> List[Cluster]:
        """
        Returns the clustered significant pixels. Does significance calculations
        if they haven't already been done.

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
            n_sigma:
                The number of standard deviations above the mean that a pixel
                needs to be to be considered significant.
        """
        if self.significant_pixels is None:
            self.init_significant_pixels(signal_length_scale, bkg_length_scale,
                                         n_sigma)
        pixels_y, pixels_x = np.where(self.significant_pixels == 1)

        # Massage these pixels into the form that sklearn wants to see.
        pixel_coords = np.zeros((len(pixels_x), 2))
        pixel_coords[:, 0] = pixels_x
        pixel_coords[:, 1] = pixels_y

        # Don't try to run DBSCAN on nothing!
        if len(pixel_coords) == 0:
            return []

        # Run the DBSCAN algorithm, setting eps and min_samples according to our
        # expected signal_length_scale.
        dbscan = DBSCAN(
            eps=signal_length_scale, min_samples=signal_length_scale**2
        ).fit(pixel_coords)

        return Cluster.from_DBSCAN(pixel_coords, dbscan.labels_)

    def mask_from_clusters(self, clusters: List[Cluster]) -> np.ndarray:
        """
        Generates a mask array from clusters.

        Args:
            clusters:
                A list of the cluster objects that we'll use to generate our
                mask.

        Returns:
            A boolean numpy mask array.
        """
        # Make an array of zeros; every pixel is a mask by default.
        mask = np.zeros_like(self.data)

        for cluster in clusters:
            mask[cluster.pixel_indices[1], cluster.pixel_indices[0]] = 1

        return mask

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
