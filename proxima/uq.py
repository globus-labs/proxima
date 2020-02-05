"""Engine that performs decisions about whether to employ a surrogate"""

from proxima.inference import BaseInferenceEngine, ScikitLearnInferenceEngine
from proxima.data import BaseDataSource
import numpy as np

from sklearn.neighbors import NearestNeighbors

# TODO (wardlt): Provide some mechanism for checking if UQ tool needs to be updated
#  Perhaps having the data store track the last time it was updated?


class BaseUQEngine:
    """Base class for tools that decide whether to invoke a surrogate model"""

    def is_supported(self, model: BaseInferenceEngine,
                     training_data: BaseDataSource, X):
        """Decide whether a prediction is supported by the data

        Args:
            model (BaseInferenceEngine): Model execution engine
            training_data (BaseDataSource): Data source
        """
        raise NotImplementedError


class SklearnUncertainty(BaseUQEngine):
    """Tool that uses uncertainty estimates from a scikit-learn model to determine applicability"""

    def __init__(self, tolerance):
        """

        Args:
             tolerance (float): Maximum estimated standard deviation
        """
        self.tolerance = tolerance

    def is_supported(self, model: ScikitLearnInferenceEngine,
                     training_data: BaseDataSource, X):
        _, y_std = model.model.predict([X], return_std=True)
        return y_std < self.tolerance


class DistanceBasedUQ(BaseUQEngine):
    def __init__(self, threshold, metric='mahalanobis', k=1, n_jobs=None):
        """Initialize the metric
        Args:
            threshold (float): Maximum distance for a prediction to be "trustable"
            metric (string): Distance metric to use
            k (int): Number of nearest neighbors to consider
            n_jobs (int): Number of threads to use when computing distances
        """
        super().__init__()
        self.threshold = threshold
        self.metric = metric
        self.k = k
        self.n_jobs = n_jobs
        self.nn_ = None

    def is_supported(self, model: BaseInferenceEngine,
                     training_data: BaseDataSource, X):
        # Get the training points
        train_X, _ = training_data.get_all_data()

        # Make a distance computer
        # TODO (wardlt): Here is where we could benefit from checking if training set is updated
        nn = NearestNeighbors(n_neighbors=self.k, n_jobs=self.n_jobs,
                              metric=self.metric).fit(train_X)

        # Get the distance
        dists, _ = nn.kneighbors([X])
        return np.mean(dists, axis=1) < self.threshold
