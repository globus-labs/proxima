"""Various implementations of surrogate models for QC calculations"""
from typing import Union, Tuple

from proxima.uq import BaseUQEngine
from pymatgen import Molecule

from proxima.data import InMemoryDataStorage, BaseDataSource

from proxima.inference import ScikitLearnInferenceEngine, BaseInferenceEngine

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from matminer.featurizers.structure import CoulombMatrix


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
            ('model', KNeighborsRegressor(n_neighbors))
        ])
        super().__init__(model)

    def infer(self, X: Atoms) -> float:
        # Compute the features
        strc = AseAtomsAdaptor.get_molecule(X[0])
        return self.model.predict([strc])[0]


class ASEDataStore(InMemoryDataStorage):
    """Stores data from an ASE simulation as Pymatgen objects because they work better with matminer"""

    def __init__(self, convert_to_pmg=True):
        """
        Args:
             convert_to_pmg (bool): Whether to convert the Atoms object to PMG before storing
        """
        self.convert_to_pmg = convert_to_pmg
        super().__init__()

    def transform(self, args: Tuple[Atoms]) -> Union[Atoms, Molecule]:
        atoms = args[0]
        if self.convert_to_pmg:
            return AseAtomsAdaptor.get_molecule(atoms)
        return atoms

    def add_pair(self, inputs, outputs):
        inputs = self.transform(inputs)
        super().add_pair(inputs, outputs)

    def add_pairs(self, inputs, outputs):
        inputs = [self.transform(i) for i in inputs]
        return super().add_pairs(inputs, outputs)


class DistanceBasedUQWithFeaturization(BaseUQEngine):
    def __init__(self, threshold, min_entries=10, metric='minkowski', k=1, n_jobs=None):
        """Initialize the metric
        Args:
            threshold (float): Maximum distance for a prediction to be "trustable"
            min_entries (int): Minimum number of training entries before
            metric (string): Distance metric to use
            k (int): Number of nearest neighbors to consider
            n_jobs (int): Number of threads to use when computing distances
        """
        super().__init__()
        self.cm = CoulombMatrix(flatten=True)
        self.cm.set_n_jobs(1)
        self.threshold = threshold
        self.min_entries = 10
        self.metric = metric
        self.k = k
        self.n_jobs = n_jobs
        self.nn_ = None

    def is_supported(self, model: BaseInferenceEngine,
                     training_data: BaseDataSource, X: Tuple[Atoms]):
        # Check total count
        if training_data.count() < self.min_entries:
            return False

        # Get the training points
        train_X, _ = training_data.get_all_data()
        train_X = self.cm.fit_transform(train_X)

        # Make a distance computer
        # TODO (wardlt): Here is where we could benefit from checking if training set is updated
        nn = NearestNeighbors(n_neighbors=self.k, n_jobs=self.n_jobs,
                              metric=self.metric).fit(train_X)

        # Get the distance
        strc = AseAtomsAdaptor.get_molecule(X[0])
        features = self.cm.featurize(strc)
        dists, _ = nn.kneighbors([features])
        return np.mean(dists, axis=1) < self.threshold
