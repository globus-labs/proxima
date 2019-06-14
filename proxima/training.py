"""Engine that performs the model re-training"""

from proxima.data import BaseDataSource
from proxima.inference import BaseInferenceEngine

# TODO (wardlt): Be consistent about what I call the model/surrogate/LFA/learner


class TrainingEngine:
    """Basic model training engine, always retrains the learner"""

    def request_update(self, learner: BaseInferenceEngine, data_source: BaseDataSource) -> bool:
        """Request for the learner to be updated

        Args:
            learner (BaseInferenceEngine): Learned accelerator to be updated
            data_source (BaseDataSource): Link to the data source
        Returns:
            (bool): Whether the model was updated
        """
        learner.retrain(data_source)
        return True


class NeverRetrain(TrainingEngine):
    """Training engine that never re-trains the model"""

    def request_update(self, learner: BaseInferenceEngine, data_source: BaseDataSource):
        return False
