"""Engines for performing inference"""

from proxima.data import BaseDataSource

# TODO (wardlt): Refactor so sklearn is not required
from sklearn.base import BaseEstimator

# TODO (wardlt): Figure out if this supports online training


class BaseInferenceEngine:
    """Abstract interface to surrogate models

    Supplies both an inter

    # TODO (wardlt): Move notes to sphinx docs
    Notes:
        - Model assumed to be trained already
    """

    def infer(self, X):
        """Perform inference to estimate the output for the models

        Args:
            X: Inputs to the function
        Returns:
            Estimated output of the function
        """
        raise NotImplementedError

    def retrain(self, data: BaseDataSource):
        """Retrain the machine learning model

        Args:
            data (BaseDataSource): Connection to the store holding the training data
        """
        raise NotImplementedError


class ScikitLearnInferenceEngine(BaseInferenceEngine):
    """Inference engine built on top of a scikit-learn estimator"""

    def __init__(self, model: BaseEstimator):
        """
        Args:
             model (BaseEstimator): Model used for inference
        """
        self.model = model

    def infer(self, X):
        return self.model.predict([X])

    def retrain(self, data: BaseDataSource):
        X, y = data.get_all_data()
        self.model.fit(X, y)
