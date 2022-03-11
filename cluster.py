"""
This module contains the Cluster class. This is a convenient way to handle the
output of scikit-learn's DBSCAN algorithm.
"""

# pylint: disable=invalid-name

from typing import List
import numpy as np


class Cluster:
    """
    For tracking clusters of pixels.
    """

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    @property
    def mean(self):
        """
        Returns the mean pixel value.
        """
        mean_x = np.mean(self._arr[:, 0])
        mean_y = np.mean(self._arr[:, 1])
        return np.array([mean_x, mean_y])

    @property
    def size(self):
        """
        Returns the number of pixels in the cluster.
        """
        return len(self._arr)

    @property
    def pixel_indices(self):
        """
        Returns pixel indices in a np.where compatible format.
        """
        return (self._arr[:, 0].astype(np.int64),
                self._arr[:, 1].astype(np.int64))

    @classmethod
    def from_DBSCAN(cls, X: np.ndarray, labels_) -> List["Cluster"]:
        """
        Takes the input and output of scikit-learn's DBSCAN algorithm. Returns
        a list of cluster objects. Note that, because of naive use of lists,
        this will slow down if len(labels_) is around 10^5-10^7-ish.
        """
        # Work out how many cluster instances we're going to need.
        num_clusters = np.max(labels_) + 1

        # Prepare a list to store the raw data.
        cluster_arrays = [[] for _ in range(num_clusters)]

        # Populate the list of arrays
        for i, label in enumerate(labels_):
            if label == -1:
                continue
            cluster_arrays[label].append((X[i][0], X[i][1]))

        return [cls(np.array(arr)) for arr in cluster_arrays]

    def intensity(self, arr: np.ndarray) -> float:
        """
        Returns the sum of this cluster's pixels in arr.

        Args:
            arr:
                The image to sum over.

        Returns:
            The area underneath this cluster in the image array passed as an
            argument.
        """
        return np.sum(arr[self.pixel_indices])
