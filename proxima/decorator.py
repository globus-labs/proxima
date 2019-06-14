"""Defines the decorator used by Proxima to mark a function to be learned"""


from proxima.inference import BaseInferenceEngine
from proxima.training import TrainingEngine
from proxima.data import BaseDataSource
from proxima.uq import BaseUQEngine

from functools import update_wrapper
from warnings import warn

# TODO (wardlt): Make training/uq policies configurable
# TODO (wardlt): Add preprocessing class for transforming function inputs


def lfa(learner: BaseInferenceEngine, uq_engine: BaseUQEngine,
        data_source: BaseDataSource, train_engine: TrainingEngine):
    """Make a wrapper that augments a function with a learned fnuction accelerator

    Args:
        learner (BaseEstimator): Link to the inference engine
        uq_engine (BaseUQEngine): Link to the UQ engine
        data_source (BaseDataSource): Link to the data source
        train_engine (TrainingEngine): Link to the retraining engine
    """

    def decorating_function(target_function):
        """Function that applies the decorator function"""

        def decorator(*args, **kwargs):
            """Wrapper function for the target function"""

            # Make inputs from the position args only
            if len(kwargs) > 0:
                warn('Keyword arguments are currently being ignored')
            inputs = args  # TODO (lw): Just concatenate

            # Execute the approximation
            if uq_engine.is_supported(learner, data_source, inputs):
                return learner.infer(inputs)
            else:
                # Run the original function
                outputs = target_function(*args, **kwargs)

                # Store the data
                data_source.add_pair(inputs, outputs)

                # Retrain the model
                train_engine.request_update(learner, data_source)
                return outputs

        return update_wrapper(decorator, target_function)

    return decorating_function
