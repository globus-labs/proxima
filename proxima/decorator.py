"""Defines the decorator used by Proxima to mark a function to be learned"""

import warnings

from proxima.inference import BaseInferenceEngine
from proxima.training import TrainingEngine
from proxima.data import BaseDataSource
from proxima.uq import BaseUQEngine

from functools import update_wrapper
from collections import namedtuple
from time import perf_counter
from warnings import warn
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import pdb
import sys
import os
from scipy import stats
# TODO (wardlt): Make training/uq policies configurable
# TODO (wardlt): Add preprocessing class for transforming function inputs
# TODO (wardlt): Add statistics class


DFT_MCSTEP_RUNTIME = 0.542


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
                 train_engine: TrainingEngine, reference_dict,out_dir):
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

        self.needs_retrain = True
        self.lazy_retrain = True
        if self.lazy_retrain and hasattr(train_engine, "interval") and train_engine.interval > 1:
            warnings.warn("Using lazy retraining with a retrain interval >1!!!")

        # Reference trajectory
        self.ref_data = None
        self.hf_times = None
        if reference_dict:
            ref_path = reference_dict["data_path"]
            ref_times = reference_dict["runtimes"]
            atom_file = open(ref_path,"rb")
            #atom_file = open("training_set_-1_T_1000_M_1","rb")
            self.ref_data = pickle.load(atom_file)
            if isinstance(ref_times, str):
                self.hf_times = pd.read_csv(ref_times)["times"].to_list()
            else:
                self.hf_times = [
                    ref_times for i in range(len(self.ref_data[0]))
                ]


        #control system variables
        self.surrogate_count = 0 # keeping track of surrogate streak
        self.outstanding_audit = None  # whether we have a HF calculation to collect from another process
        self.compare_true = []
        self.compare_surrogate = []
        self.compare_uq = []
        self.mean_error_log = []
        self.uq_log = []
        self.size_compare = 10#  10
        self.audit_threshold = 1050 #when to trigger an extra HF calculation - surrogate streak max, 25 to check streak
        self.audit_step_delay = 2 #how long to wait before collecting result
        self.mae_threshold = 0.002
        self.uq_t_min = 0.1#0.1
        self.uq_t_max = 0.7
        self.step_num = 0
        self.tune_uq = True
        self.surg_data = 0
        self.add_data_count = 0
        self.first_slope = 1
        self.uq_slope_vals = []
        self.me_vals = []
        self.alpha_log =[]
        self.alpha = None
        self.prev_alpha = None
        self.alpha_calc = False ## use precalculated Alpha
        self.hf_streak = 0
        self.alpha_time = 0
        self.out_dir = out_dir
        self.to_print = False

    def _uq(self, inputs):
        start_time = perf_counter()
        is_supported, dist = self.uq_engine.is_supported(self.inference_engine, self.data_source, inputs, return_metric=True)
        self._uq_time += perf_counter() - start_time
        return is_supported, dist
    
    def _target(self, *args, record_stats=True, ref_step=None, **kwargs):
        if ref_step is not None:
            outputs = self.ref_data[1][ref_step]
            if record_stats:
                self._target_runs += 1
                #self._target_time += DFT_MCSTEP_RUNTIME
                self._target_time += self.hf_times[ref_step]
                #atoms = self.ref_data[0][ref_step]
                #outputs = calc.get_potential_energy(atoms, target=True)
        else:
            if record_stats:
                start_time = perf_counter()
            try:
                outputs = self.target_function(*args, **kwargs)
            finally:
                if record_stats:
                    self._target_runs += 1
                    self._target_time += perf_counter() - start_time
        return outputs

    def _prep_lists(self, uq_max=None):
        tlist = []; slist = []; uqlist = []
        uq_max = uq_max or max(self.compare_uq)
        for t, s, u in zip(self.compare_true, self.compare_surrogate, self.compare_uq):
            if u is not None and u <= uq_max:
                tlist.append(t)
                slist.append(s)
                uqlist.append(u)
        len_tocompare = int(len(tlist)/2)
        return tlist, slist, uqlist, len_tocompare

    def _update_uq_alpha(self, mean_error, window_size=None):
        tlist, slist, uqlist, len_tocompare = self._prep_lists(uq_max=None)
        if len(uqlist) < 2:
            return
        if len(self.me_vals) < 2:
            pt1 = mean_absolute_error(tlist[:len_tocompare], slist[:len_tocompare])
            uq1 = max(uqlist[:len_tocompare])
            pt2 = mean_absolute_error(tlist[len_tocompare:], slist[len_tocompare:])
            uq2 = max(uqlist[len_tocompare:])
            for u, m in zip([uq1, uq2], [pt1, pt2]):
                self.uq_slope_vals.append(u)
                self.me_vals.append(m)
        else:
            if window_size is None:
                window_size = len_tocompare
            else:
                window_size = min(window_size, len(tlist))
            pt1 = mean_absolute_error(tlist[window_size:], slist[window_size:])
            uq1 = max(uqlist[window_size:])
            self.uq_slope_vals.append(uq1)
            self.me_vals.append(pt1)
        if self.alpha_calc:
            self.alpha = 0.00457
        else:
            self.alpha = stats.linregress(self.uq_slope_vals, self.me_vals).slope
            self.alpha = max(self.alpha, 1e-6)
            if self.to_print: 
                print("ALPHA = ",self.alpha)
            self.prev_alpha = self.alpha
            ####self.alpha_log.append(self.alpha)
        """if len(self.me_vals) > 200 and len(self.me_vals)%10 == 0:
            _df = pd.DataFrame({'uq':self.uq_slope_vals,'mae':self.me_vals})
            _df.to_csv('slope_mae_check.csv')
            del _df"""


    def _tune_uq_tolerance(self, legacy=False):
        if self.tune_uq:
            if len(self.compare_true) >= self.size_compare:

                # Get reference MAE
                mean_error = mean_absolute_error(self.compare_true, self.compare_surrogate)

                # Update slope (alpha)
                if legacy:
                    self._update_uq_alpha_legacy(mean_error)
                else:
                    self._update_uq_alpha(mean_error)

                if self.alpha:
                
                    # Evict oldest reference data
                    self.compare_true.pop(0)
                    self.compare_surrogate.pop(0)
                    self.compare_uq.pop(0)
                    self.surrogate_count = 0

                    # Calculate UQ tolerence change
                    #alpha = 0.00413 #0.00304
                    try:
                        rho = 0
                        _change = - ((1-rho)/ self.alpha) * (mean_error - self.mae_threshold)
                        #_rt_change = ((1-rho)/ self.train_engine.interval) *1000* (mean_error - self.mae_threshold)
                    except:
                        import pdb; pdb.set_trace()
                    
                    # Update UQ tolerence
                    #bounding to +- 0.1
                    _change_lim = 0.6
                    _change = min(_change, _change_lim)
                    _change = max(_change, -_change_lim)
                    _new_uq  = self.uq_engine.threshold + _change
                    
                    ##bound RT change
                    """_changeRT_lim = 100
                    _changeRT_lim = min(_changeRT_lim,_rt_change)
                    _changeRT_lim = max(_changeRT_lim, -_rt_change)
                    _new_RT = self.train_engine.interval + _changeRT_lim"""
                    
                    #import pdb; pdb.set_trace()
                    """_new_uq = max(_new_uq, self.uq_t_min)
                    _new_uq = min(_new_uq, self.uq_t_max)"""
                    _new_uq = max(_new_uq, self.uq_t_min)
                    _new_uq = min(_new_uq, self.uq_t_max)
                    if self.to_print:
                        print("___________\n")
                        #print("Retrain Interval Change", self.train_engine.interval, _new_RT, mean_error,_changeRT_lim)
                        print("UQ-Change", self.uq_engine.threshold, _new_uq, mean_error,_change)
                        print("\n___________\n")
                    #self.train_engine.interval = _new_RT
                    self.uq_engine.threshold = _new_uq
        #val_mae_compare = len(self.me_vals)
        #self.alpha_log.append(val_mae_compare )
        #if self.alpha:
        """if True:
            with open(os.path.join(self.out_dir,'alpha.txt'), 'a') as f:
                for item in self.alpha_log:
                    f.write("%s\n" % item)
                    self.alpha_log = []"""





    # TODO (wardlt): Make a batch version (build batch versions into "Engines" first)
    def __call__(self, *args, **kwargs):
        # Make inputs from the position args only
        # TODO (lw): Replace with better strategy for going from args -> inputs
        #  For example, do we need to support kwargs as well. Should we have a tool to convert
        #   inputs into a more versatile representation?

        # forced = kwargs.pop("forced", None)    # Force "high_fidelity" or "surrogate"
        # target = kwargs.pop("target", False)   # Just call target "high_fidelity" and return
        ref_step = kwargs.pop("ref_step", None)  # MC step for reference trajectory

        if len(kwargs) > 0:
            warn('Keyword arguments are currently being ignored')
        
        # [Debug Code]
        debug_measure = False
        if debug_measure:
            # This code just measures the time needed to train a model
            # with `count` datapoints. Results:
            #
            # count  time
            # 1      0.02 s
            # 10     0.414 s
            # 100    1.65 s
            # 500    41.6 s
            # 1000   164.0 s
            #
            count = 250
            x = self.ref_data[0][:count]
            y = self.ref_data[1][:count]
            for _x, _y in zip(x, y):
                self.data_source.add_pair([_x], _y)
            start_time = perf_counter()
            self.train_engine.request_update(self.inference_engine, self.data_source)
            self._train_time += perf_counter() - start_time
            if self.to_print: 
                print(self._train_time)
            #import pdb; pdb.set_trace()
        self.alpha_log.append(self.alpha)
        if ref_step is not None:
            #import pdb; pdb.set_trace()
            args = [self.ref_data[0][ref_step]]
        inputs = args

        # Check if we should use the surrogate
        is_supported, dist = self._uq(inputs)
        self._used_surrogate = is_supported
        #import pdb; pdb.set_trace()
        # Get output from lfa model or target function
        #is_supported = False
        if is_supported:
            # Retrain the model 
            if self.lazy_retrain and self.needs_retrain:
                start_time = perf_counter()         
                try:
                    self.train_engine.request_update(self.inference_engine, self.data_source)
                finally:
                    self._train_time += perf_counter() - start_time
                    self.needs_retrain = False
            self.hf_streak -= 1
            self.hf_streak = max(self.hf_streak,0)
            #import pdb; pdb.set_trace()
            self.surrogate_count +=1
            self.surg_data += 1
            start_time = perf_counter()
            try:
                surrogate_energy = self.inference_engine.infer(inputs)
                out = surrogate_energy
            finally:
                self._lfa_runs += 1
                self._lfa_time += perf_counter() - start_time  
            true_energy = self._target(*args, record_stats=False, ref_step=ref_step, **kwargs) #for final comparison
        else:
            try:
                #start_time = perf_counter()
                self.hf_streak += 1
                true_energy = self._target(*args, record_stats=True, ref_step=ref_step, **kwargs)
                out = true_energy           
                self.surrogate_count = 0
            finally:
               #self._target_runs += 1
               #self.data_source.add_pair(inputs, true_energy)
               #self.add_data_count += 1
               x =1
               #self._target_time += perf_counter() - start_time
            
            if self.uq_engine.threshold > 0:
                try:
                    start_time = perf_counter()
                    surrogate_energy = self.inference_engine.infer(inputs)
                    self._lfa_time += perf_counter() - start_time 
                    self._lfa_runs += 1
                    self.compare_true.append(true_energy)
                    self.compare_surrogate.append(surrogate_energy)
                    self.compare_uq.append(dist)
                except:
                    # Store the data
                    surrogate_energy = None
                    #import pdb; pdb.set_trace()
                    #finally:
                    """self.data_source.add_pair(inputs, true_energy)
                    self.add_data_count += 1"""
                    #self.data_source.fifo_evict()
                finally:
                    self.data_source.add_pair(inputs, true_energy)
                    self.add_data_count += 1
                    if not self.lazy_retrain:
                        start_time = perf_counter()         
                        try:
                            self.train_engine.request_update(self.inference_engine, self.data_source)
                        finally:
                            self._train_time += perf_counter() - start_time
                    self.needs_retrain = True
                    # Retrain the model 
                    #if self.hf_streak < 5 or self.hf_streak > 25:
                    """start_time = perf_counter()         
                    try:
                        self.train_engine.request_update(self.inference_engine, self.data_source)
                    finally:
                        self._train_time += perf_counter() - start_time"""
                    if self.to_print:
                        print("ref_step", ref_step)
                        print("surrogate use number", self.surg_data )
                        print("Training Data Size", self.add_data_count )
                    assert len(self.data_source.inputs) == self.add_data_count
        
        if hasattr(self.inference_engine, "global_update"):
            self.inference_engine.global_update()
            
        self.uq_log.append(self.uq_engine.threshold)
        ##changing UQ
        #start_time = datetime.utcnow()
        ##### to change interval -- self.train_engine.interval
        # Update the UQ tolerance
        if self.tune_uq:
            start_time = perf_counter()
            if self.surrogate_count >= self.audit_threshold:
                self.compare_true.append(self._target(*args, record_stats=True, ref_step=ref_step, **kwargs))
                self.compare_surrogate.append(surrogate_energy)
                self.compare_uq.append(dist)
                self.surrogate_count = 0
        self._tune_uq_tolerance()
        #self.alpha_time += perf_counter() - start_time
        #print("THSSISISI IS SURROGATE", surrogate_energy)
        #print("alpha time", self.alpha_time)
        return out, true_energy, surrogate_energy
        """if self.uq_engine.threshold > 0:
            return out, true_energy, surrogate_energy
        else:
            return out, true_energy"""
        """self.step_num += 1
         # Use ref data if ref_steps is available
        if ref_step is not None:
        args = [self.ref_data[0][ref_step]]
        inputs = args

        # Check if we are just returning "target" value
        if target:
        #saving true new energy
        true_new_energy = self._target(ref_step, target, *args, **kwargs)
        return true_new_energy 

        # Check if we should use the surrogate
        is_supported = self._uq(inputs, forced)

        # Record if the surrogate will be used
        if not forced == "surrogate":
        self._used_surrogate = is_supported

        if is_supported:
        self.surrogate_count += 1
        start_time = perf_counter()
        try:
            #saving surrogate energy
            out = self.inference_engine.infer(inputs)
            self.compare_surrogate.append(out)
        finally:
            self._lfa_runs += 1
            self._lfa_time += perf_counter() - start_time
        return out
        else:
        # Run the original function
        # TODO (wardlt): Support apps where we cannot run LFA on same inputs (e.g., AIMD)
        outputs = self._target(ref_step, target, *args, **kwargs)
        self.compare_true.append(outputs)
        self.surrogate_count = 0
        # Store the data
        self.data_source.add_pair(inputs, outputs)

        if not forced == "high_fidelity":
            # Retrain the model
            start_time = perf_counter()
            try:
                self.train_engine.request_update(self.inference_engine, self.data_source)
            finally:
                self._train_time += perf_counter() - start_time
        return outputs"""

    def did_last_call_use_surrogate(self) -> bool:
        """Check if the last call used the surrogate"""
        # TODO (wardlt): Should I have similar functions of other statistics of last call (e.g., runtimes)
        return self._used_surrogate

    def get_performance_info(self) -> _perf_info:
        """Get measurements of the performance of the LFA"""
        return _perf_info(self._lfa_runs, self._lfa_time, self._uq_time, self._train_time,
                          self._target_runs, self._target_time)
