"""Fitting the GAP potentials with scikit-learn"""
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise
from dscribe.descriptors import SOAP
import numpy as np


def scalable_kernel(mol_a: np.ndarray, mol_b: np.ndarray, gamma: float = 1.0) -> float:
    """Compute the scalable kernel between molecules

    Args:
        mol_a (ndarray): Represnetation of a molecule
        mol_b (ndarray): Representation of another molecule
        gamma (float): Kernel parameter
    Returns:
        (float) Similarity between molecules
    """
    # Fixing n_jobs=1 because these are small calculations (typically less than ~500 pairwise measurements)
    return np.exp(-1 * pairwise.pairwise_distances(mol_a, mol_b, 'sqeuclidean', n_jobs=1) / gamma).sum()


class SOAPConverter(BaseEstimator, TransformerMixin):
    """Compute the SOAP descriptors for molecules"""

    def __init__(self, rcut: float = 6, nmax: int = 8, lmax: int = 6, species=frozenset({'C', 'O', 'H', 'N', 'F'})):
        """Initialize the converter
        
        Args:
            rcut (float); Cutoff radius
            nmax (int):
            lmax (int):
            species (Iterable): List of elements to include in potential
        """
        super().__init__()
        self.soap = SOAP(rcut=rcut, nmax=nmax, lmax=lmax, species=sorted(species))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.soap.create(x) for x in X]


class ScaleFeatures(BaseEstimator, TransformerMixin):
    """Scales 3D arrays of features to [0, 1]"""

    def __init__(self, eps=1e-6):
        """
        Args:
             eps (float): Tolerance value
        """
        super().__init__()
        self.eps = eps
        self.min_values = self.range = None

    def fit(self, X, y=None):
        max_values = np.max(X, axis=0).max(axis=0)
        self.min_values = np.min(X, axis=0).min(axis=0)
        self.range = np.clip(max_values - self.min_values, self.eps, np.inf)
        return self

    def transform(self, X, y=None):
        return np.subtract(X, self.min_values) / self.range


class ScalableKernel(BaseEstimator, TransformerMixin):
    """Class for computing a scalable atomistic kernel

    This kernel computes the pairwise similarities between each atom in both molecules.
    The total similarity molecules is then computed as the sum of these points
    """

    def __init__(self, max_points: Optional[int] = None, gamma: float = 1.0):
        super(ScalableKernel, self).__init__()
        self.train_points = None
        self.max_points = max_points
        self.gamma = gamma

    def fit(self, X, y=None):
        if self.max_points is None:
            # Store the training set
            self.train_points = np.array(X)
        else:
            inds = np.random.choice(len(X), size=(self.max_points))
            self.train_points = X[inds]
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        K = np.zeros((len(X), len(self.train_points)))
        for i, x in enumerate(X):
            for j, tx in enumerate(self.train_points):
                K[i, j] = scalable_kernel(x, tx, self.gamma)
        return K
