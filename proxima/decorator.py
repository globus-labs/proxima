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
    """Make a wrapper that augments a function with a learned function accelerator

    Args:
        inference_engine (BaseEstimator): Link to the inference engine
        uq_engine (BaseUQEngine): Link to the UQ engine
        data_source (BaseDataSource): Link to the data source
        train_engine (TrainingEngine): Link to the retraining engine
    """

    def decorating_function(target_function):
        """Function that applies the decorator function"""
        engine = LFAEngine(target_function, inference_engine, uq_engine, data_source, train_engine)
        return update_wrapper(engine, target_function)

    return decorating_function


class LFAEngine:
    """Class that manages the interposition of functions with a learned function accelerator"""

    def __init__(self, target_function, inference_engine: BaseInferenceEngine,
                 uq_engine: BaseUQEngine, data_source: BaseDataSource,
                 train_engine: TrainingEngine):
        """
        Args:
            target_function: Function to augment with an accelerator
            inference_engine (BaseEstimator): Link to the inference engine
            uq_engine (BaseUQEngine): Link to the UQ engine
            data_source (BaseDataSource): Link to the data source
            train_engine (TrainingEngine): Link to the retraining engine
        """

        # Store the function information
        self.target_function = target_function
        self.inference_engine = inference_engine
        self.uq_engine = uq_engine
        self.data_source = data_source
        self.train_engine = train_engine

        # Initialize performance stats
        self._lfa_runs = self._lfa_time = 0
        self._uq_time = self._train_time = 0
        self._target_runs = self._target_time = 0
        self._used_surrogate = None

    # TODO (wardlt): Make a batch version (build batch versions into "Engines" first)
    def __call__(self, *args, **kwargs):
        # Make inputs from the position args only
        # TODO (lw): Replace with better strategy for going from args -> inputs
        #  For example, do we need to support kwargs as well. Should we have a tool to convert
        #   inputs into a more versatile representation?
        if len(kwargs) > 0:
            warn('Keyword arguments are currently being ignored')
        inputs = args

        # Execute the approximation
        start_time = perf_counter()
        is_supported = self.uq_engine.is_supported(self.inference_engine, self.data_source, inputs)
        self._uq_time += perf_counter() - start_time

        # Saved if the surrogate was used
        self._used_surrogate = is_supported

        if is_supported:
            start_time = perf_counter()
            try:
                return self.inference_engine.infer(inputs)
            finally:
                self._lfa_runs += 1
                self._lfa_time += perf_counter() - start_time
        else:
            # Run the original function
            # TODO (wardlt): Support apps where we cannot run LFA on same inputs (e.g., AIMD)
            try:
                start_time = perf_counter()
                outputs = self.target_function(*args, **kwargs)
            finally:
                self._target_runs += 1
                self._target_time += perf_counter() - start_time

            # Store the data
            self.data_source.add_pair(inputs, outputs)

            # Retrain the model
            start_time = perf_counter()
            try:
                self.train_engine.request_update(self.inference_engine, self.data_source)
            finally:
                self._train_time += perf_counter() - start_time
            return outputs

    def did_last_call_use_surrogate(self) -> bool:
        """Check if the last call used the surrogate"""
        # TODO (wardlt): Should I have similar functions of other statistics of last call (e.g., runtimes)
        return self._used_surrogate

    def get_performance_info(self) -> _perf_info:
        """Get measurements of the performance of the LFA"""
        return _perf_info(self._lfa_runs, self._lfa_time, self._uq_time, self._train_time,
                          self._target_runs, self._target_time)
