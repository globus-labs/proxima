"""Defines the decorator used by Proxima to mark a function to be learned"""


from proxima.inference import BaseInferenceEngine
from proxima.training import TrainingEngine
from proxima.data import BaseDataSource
from proxima.uq import BaseUQEngine

from functools import update_wrapper
from collections import namedtuple
from time import perf_counter
from warnings import warn

# TODO (wardlt): Make training/uq policies configurable
# TODO (wardlt): Add preprocessing class for transforming function inputs
# TODO (wardlt): Add statistics class

_perf_info = namedtuple('AccelStats', ['lfa_runs', 'lfa_time', 'uq_time', 'train_time',
                                        'target_runs', 'target_time'])


def lfa(inference_engine: BaseInferenceEngine, uq_engine: BaseUQEngine,
        data_source: BaseDataSource, train_engine: TrainingEngine):
    """Make a wrapper that augments a function with a learned fnuction accelerator

    Args:
        inference_engine (BaseEstimator): Link to the inference engine
        uq_engine (BaseUQEngine): Link to the UQ engine
        data_source (BaseDataSource): Link to the data source
        train_engine (TrainingEngine): Link to the retraining engine
    """

    def decorating_function(target_function):
        """Function that applies the decorator function"""

        # Initialize the storage for performance counters
        lfa_runs = lfa_time = uq_time = target_runs = target_time = train_time = 0

        # Make the decorator
        def decorator(*args, **kwargs):
            """Wrapper function for the target function"""

            nonlocal lfa_runs, lfa_time, uq_time, target_runs, target_time, train_time

            # Make inputs from the position args only
            # TODO (lw): Replace with better strategy
            if len(kwargs) > 0:
                warn('Keyword arguments are currently being ignored')
            inputs = args

            # Execute the approximation
            start_time = perf_counter()
            is_supported = uq_engine.is_supported(inference_engine, data_source, inputs)
            uq_time += perf_counter() - start_time

            if is_supported:
                start_time = perf_counter()
                try:
                    return inference_engine.infer(inputs)
                finally:
                    lfa_runs += 1
                    lfa_time += perf_counter() - start_time
            else:
                # Run the original function
                try:
                    start_time = perf_counter()
                    outputs = target_function(*args, **kwargs)
                finally:
                    target_runs += 1
                    target_time += perf_counter() - start_time

                # Store the data
                data_source.add_pair(inputs, outputs)

                # Retrain the model
                start_time = perf_counter()
                try:
                    train_engine.request_update(inference_engine, data_source)
                finally:
                    train_time += perf_counter() - start_time
                return outputs

        # Attach the stats to the decorator
        def _get_perf_info():
            return _perf_info(lfa_runs, lfa_time, uq_time, train_time, target_runs, target_time)
        decorator.get_performance_info = _get_perf_info

        # Attach the engines
        decorator.inference_engine = inference_engine
        decorator.train_engine = train_engine
        decorator.uq_engine = uq_engine
        decorator.data_source = data_source

        return update_wrapper(decorator, target_function)

    return decorating_function
