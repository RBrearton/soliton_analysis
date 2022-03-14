"""
Take the necessary utility functions from the playground notebook.
"""

# pylint: disable=invalid-name

import os
import re


from typing import Tuple
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image

import ccd_image as ccd

# First construct a path to where the data is stored on my machine.
LOCAL_DATA_DIR = "/Users/richard/Data/i10/CSL_Feb_2022/azimuthal_scans/"

LOCAL_20K_DIR = "/Users/richard/Data/i10/CSL_Feb_2022/azimuthal_20K/"

# Get all the nexus files and scan directories in the data directory.
NEXUS_FILES = sorted(
    [f for f in os.listdir(LOCAL_DATA_DIR) if f.endswith('.nxs')]
)
SCAN_DIRS = sorted(
    [d for d in os.listdir(LOCAL_DATA_DIR) if d.endswith('files')]
)

SCAN_DIRS_20K = sorted(
    [d for d in os.listdir(LOCAL_20K_DIR) if d.endswith('files')]
)

PONI_PATH = "saxs_calib.poni"

# The first scan number in the script.
FIRST_SCAN = 687537

# The field values in the field sweeps.
field_values = list(range(31))

# Calculated manually
BEAMSTOP_TOP = 1082
BEAMSTOP_BOTTOM = 1284
BEAMSTOP_LEFT = 948
BEAMSTOP_RIGHT = 1145

BEAM_CENTRE_X = 1045
BEAM_CENTRE_Y = 1175

# Fit in the playground notebook.
FITTING_CONSTANT = 61393.75028770998

LOWER_RADIAL_BOUND = (BEAMSTOP_BOTTOM - BEAMSTOP_TOP)*np.sqrt(2)
UPPER_RADIAL_BOUND = 2000 - BEAM_CENTRE_Y

METADATA = ccd.Metadata(BEAM_CENTRE_X, BEAM_CENTRE_Y)

ANGLES_20K = list(range(0, 180, 15))
ANGLES_20K.extend(list(range(5, 155, 15)))


def name_to_scan_number(dir_or_file_name: str) -> int:
    """
    Takes the name of a scan directory or nexus file. Outputs the scan number.

    Args:
        dir_or_file_name:
            The name of the directory or file.

    Returns:
        The scan number.
    """
    split_name = re.split('\W', dir_or_file_name)
    for maybe_scan_number in split_name:
        try:
            return int(maybe_scan_number)
        except ValueError:
            continue


def scan_to_angle(dir_or_file_name: str) -> float:
    """
    Takes the name of a scan directory or nexus file name. Outputs the azimuthal
    angle at which the field was applied in the scan.

    Args:
        dir_or_file_name:
            The name of the directory or file.

    Returns:
        The scan's corresponding azimuthal angle.
    """
    scan_number = name_to_scan_number(dir_or_file_name)
    return (scan_number - FIRST_SCAN)*1.5


def _angle_to_scan_no(angle: float) -> int:
    """
    Converts input angle to integer nth scan number.
    """
    if float(angle) not in np.arange(0, 180, 1.5):
        raise ValueError("Angle was not scanned.")
    return int(angle/1.5)


def get_path(scan_dir: str, field_magnitude: int, t_20K=False):
    """
    Each scan has several .tiff files, one for each field magnitude in
    range(31).

    Args:
        scan_dir:
            The name of a scan directory or file.
        field_magnitude:
            The magnitude of the field of interest

    Returns:
        The path to the corresponding .tiff file.
    """
    local_tiff_name = f"pixis-{field_magnitude}.tiff"
    if t_20K:
        return os.path.join(LOCAL_20K_DIR, scan_dir, local_tiff_name)
    return os.path.join(LOCAL_DATA_DIR, scan_dir, local_tiff_name)


def get_tiff(scan_dir: str, field_magnitude: int, t_20K=False) -> np.ndarray:
    """
    Each scan has several .tiff files, one for each field magnitude in
    range(31).

    Args:
        scan_dir:
            The name of a scan directory or file.
        field_magnitude:
            The magnitude of the field of interest
        t_20K:
            A boolean representing whether or not this .tiff should be from
            the 20 K dataset.

    Returns:
        A numpy array representing the .tiff file.
    """
    full_tiff_path = get_path(scan_dir, field_magnitude, t_20K)
    return np.array(Image.open(full_tiff_path)).astype(np.float64)


def get_tiff_angle_field(angle: float, field_magnitude: int, t_20K=False):
    """
    Returns the tiff image at a certain angle and field.
    """
    if t_20K:
        # It's a bit of a more complicated ordeal with this (incomplete) data.
        if angle not in ANGLES_20K:
            raise ValueError("Bad angle provided for 20 K scan.")
        # Get the scan's index in the directory.
        idx = ANGLES_20K.index(angle)
        scan_dir = SCAN_DIRS_20K[idx]
    else:
        scan_dir = SCAN_DIRS[int(angle/1.5)]
    return get_tiff(scan_dir, field_magnitude, t_20K)


def get_rough_background(scan_dir: str, t_20K=False) -> np.ndarray:
    """
    Assume that we can model the background as an image taken in the field
    polarized state. Assume that we field polarize at the maximum field value
    of 30 mT. In this case, the background is just the 30 mT image; return it.

    Args:
        scan_dir:
            The scan directory of interest.
        t_20K:
            True if we want background for a 20K scan, False if the high T
            dataset.

    Returns:
        A simple estimate of the background for the images in that scan
        directory.
    """
    if t_20K:
        return get_tiff(scan_dir, 70, t_20K)
    return get_tiff(scan_dir, 30)


def get_rough_background_angle(field_angle, t_20K=False) -> np.ndarray:
    """
    Assume that we can model the background as an image taken in the field
    polarized state. Assume that we field polarize at the maximum field value
    of 30 mT. In this case, the background is just the 30 mT image; return it.

    If t_20K is True, we take the background to be the 70mT image.
    """
    if t_20K:
        idx = ANGLES_20K.index(field_angle)
        scan_dir = SCAN_DIRS_20K[idx]
        return get_tiff(scan_dir, 70, t_20K)

    scan_dir = SCAN_DIRS[int(field_angle/1.5)]
    return get_tiff(scan_dir, 30)


def imshow(img: np.ndarray, figsize: Tuple[int] = (20, 20),
           cmap: str = 'jet', title="", **kwargs) -> None:
    """
    Imshow, but with pretty colours and a big size.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    picture = ax.imshow(img, cmap=cmap, **kwargs)
    fig.colorbar(picture, ax=ax)

    plt.title(title)

    fig.show()


def plotly_imshow(img: np.ndarray, **kwargs):
    """
    Imshow, plotly version.
    """
    px.imshow(img, color_continuous_scale="jet", **kwargs).show()


def gauss(x, a, x0, sigma, const):
    """
    A general Gaussian function.
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + const


def gauss_at_origin(x, a, sigma, const):
    """
    A Gaussian that's fixed to start at the origin.
    """
    return gauss(x, a, 0, sigma, const)


def summed_gauss(x, a1, sigma1, const1, a2, sigma2, offset2, const2):
    """
    Sum of a gaussian fixed at the origin and a gaussian peak profile.
    """
    return gauss_at_origin(x, a1, sigma1, const1) + \
        gauss(x, a2, offset2, sigma2, const2)


# Prepare a mask.
SIGNAL_LENGTH_SCALE = 20
BKG_LENGTH_SCALE = 100

OPEN_MASK = get_rough_background(SCAN_DIRS[0]) > np.inf
MASK = np.ones_like(get_rough_background(SCAN_DIRS[0]))
MASK[(BEAMSTOP_TOP-BKG_LENGTH_SCALE):(BEAMSTOP_BOTTOM+BKG_LENGTH_SCALE),
     (BEAMSTOP_LEFT-BKG_LENGTH_SCALE):(BEAMSTOP_RIGHT+BKG_LENGTH_SCALE)] = 0
