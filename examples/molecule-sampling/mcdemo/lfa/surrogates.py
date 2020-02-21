"""Classes that define the inference engines"""
from typing import Optional

import numpy as np
from ase import Atoms
from matminer.featurizers.structure import CoulombMatrix
from proxima.data import BaseDataSource
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from proxima.inference import ScikitLearnInferenceEngine, BaseInferenceEngine

from mcdemo.lfa.gap.skl import SOAPConverter, ScalableKernel


class CoulombMatrixKNNSurrogate(ScikitLearnInferenceEngine):
    """A very fast implementation for a model: Coulomb Matrix via Matminer plus a KNN surrogate model"""

    def __init__(self, n_neighbors: int = 5):
        """
        Args:
             n_neighbors (int): Number of neighboring points to use for the NN model
        """
        cm = CoulombMatrix(flatten=True)
        cm.set_n_jobs(1)
        model = Pipeline([
            ('featurizer', cm),
            ('scaler', RobustScaler()),
            ('model', KNeighborsRegressor(n_neighbors))
        ])
        super().__init__(model)

    def infer(self, X: Atoms) -> float:
        # Convert to pymatgen format needed by matminer
        strc = AseAtomsAdaptor.get_molecule(X[0])
        return self.model.predict([strc])[0]


class GAPSurrogate(BaseInferenceEngine):
    """Inference engine using the Gaussian Approximation Potentials

    Uses SOAP to compute molecular representation, a scalable kernel
    to compute molecular symmetries, and BayesianRidge regression to
    fit model parameters
    """

    def __init__(self, max_kernel_size: int = 256, soap_settings: Optional[dict] = None,
                 gamma: float = 1.0):
        """
        Args:
            max_kernel_size (int): Maximum number of training entries to use in the GAP KRR model
                Larger values lead to higher accuracies at a greater computational cost
            soap_settings (dict): Settings that define the SOAP representation for a molecule
            gamma (float): Width parameter for the kernels
        """
        super().__init__()
        self.max_kernel_size = max_kernel_size
        if soap_settings is None:
            soap_settings = dict()

        # Build the model
        self.model = Pipeline([
            ('soap', SOAPConverter(**soap_settings)),
            ('kernel', ScalableKernel(max_points=max_kernel_size, gamma=gamma)),
            ('model', BayesianRidge(fit_intercept=True))
        ])

    def infer(self, X: Atoms) -> float:
        return self.model.predict([X])[0]

    def retrain(self, data: BaseDataSource):
        X, y = data.get_all_data()

        # Fit the model
        # TODO (wardlt): Consider adding some hyperparameter optimization
        self.model.fit(X, y)
