"""UQ implementations specific to the """

from typing import Tuple

import numpy as np
from ase import Atoms
from matminer.featurizers.structure import CoulombMatrix
from mcdemo.lfa.data import ASEDataStore
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from proxima.data import BaseDataSource
from proxima.inference import BaseInferenceEngine
from proxima.uq import BaseUQEngine


class DistanceBasedUQWithFeaturization(BaseUQEngine):
    """An implementation of the distance based UQ that measures distances
    based on a featurized version of the structure"""

    def __init__(self, threshold,perturb, temp,retrain_interval,mol, min_entries=10, metric='minkowski', k=1, n_jobs=1):
        """Initialize the metric
        Args:
            threshold (float): Maximum distance for a prediction to be "trustable"
            min_entries (int): Minimum number of training entries before surrogate can be evaluated
            metric (string): Distance metric to use
            k (int): Number of nearest neighbors to consider
            n_jobs (int): Number of threads to use when computing distances
        """
        super().__init__()

        # Make the featurizer
        # TODO (wardlt): This code is duplicated in the inference engine. Maybe we should let "featurizer" be a param
        cm = CoulombMatrix(flatten=True)
        cm.set_n_jobs(1)
        self.cm = Pipeline([
            ('featurizer', cm),
            ('scaler', RobustScaler())
        ])

        # Save the other things
        self.threshold = threshold
        self.min_entries = min_entries
        self.metric = metric
        self.k = k
        self.n_jobs = n_jobs
        self.nn_ = None
        self.perturb = perturb
        self.temp = temp
        self.retrain_interval = retrain_interval
        self.mol = mol
        
        

    def is_supported(self, model: BaseInferenceEngine,
                     training_data: ASEDataStore,
                     X: Tuple[Atoms],
                     return_metric=False):
        # Check total count
        insufficient_data = training_data.count() < self.min_entries
        if insufficient_data: # and return_metric is False:
            if return_metric is False:
                return False
            elif return_metric is True and training_data.count() < 1:
                return False, None

        #elif training_data.count() < 1 and return_metric is True:
        #    return False, None

        # Get the training points
        train_X, _ = training_data.get_all_data()
        if not training_data.convert_to_pmg:
            train_X = [AseAtomsAdaptor.get_molecule(x) for x in train_X]
        train_X = self.cm.fit_transform(train_X)

        # Make a distance computer
        # TODO (wardlt): Here is where we could benefit from checking if training set is updated
        nn = NearestNeighbors(n_neighbors=self.k, n_jobs=self.n_jobs,
                              metric=self.metric).fit(train_X)

        # Get the distance
        strc = AseAtomsAdaptor.get_molecule(X[0])
        features = self.cm.transform([strc])
        dists, _ = nn.kneighbors(features)
        distance = np.mean(dists, axis=1)[0]
        """with open (file_name,'a') as f:ß
            f.write("%f\n" % (distance) )"""

        print("UQ:", distance < self.threshold, distance, self.threshold)

        result = False if insufficient_data else distance < self.threshold
        if return_metric:
            return result, distance
        return result
