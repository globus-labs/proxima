"""Engine that performs decisions about whether to employ a surrogate"""

from proxima.inference import BaseInferenceEngine, ScikitLearnInferenceEngine
from proxima.data import BaseDataSource


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
        _, y_std = model.model.predict([X], reutrn_std=True)
        return y_std < self.tolerance
