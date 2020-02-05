"""Classes that define the inference engines"""

from ase import Atoms
from matminer.featurizers.structure import CoulombMatrix
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from proxima.inference import ScikitLearnInferenceEngine


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
        # Compute the features
        strc = AseAtomsAdaptor.get_molecule(X[0])
        return self.model.predict([strc])[0]
