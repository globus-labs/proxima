"""Classes that define the inference engines"""
from typing import Optional

import numpy as np
import copy

from ase import Atoms
from matminer.featurizers.structure import CoulombMatrix
from proxima.data import BaseDataSource
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from proxima.inference import ScikitLearnInferenceEngine, BaseInferenceEngine
from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import Sequential
from keras.layers import Dense

import torch
import torchani

from mcdemo.lfa.gap.skl import SOAPConverter, ScalableKernel


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
        # Convert to pymatgen format needed by matminer
        strc = AseAtomsAdaptor.get_molecule(X[0])
        return self.model.predict([strc])[0]


class GAPSurrogate(BaseInferenceEngine):
    """Inference engine using the Gaussian Approximation Potentials

    Uses SOAP to compute molecular representation, a scalable kernel
    to compute molecular symmetries, and BayesianRidge regression to
    fit model parameters
    """

    def __init__(self, max_kernel_size: int = 256, soap_settings: Optional[dict] = None,
                 gamma: float = 1.0):
        """
        Args:
            max_kernel_size (int): Maximum number of training entries to use in the GAP KRR model
                Larger values lead to higher accuracies at a greater computational cost
            soap_settings (dict): Settings that define the SOAP representation for a molecule
            gamma (float): Width parameter for the kernels
        """
        super().__init__()
        self.max_kernel_size = max_kernel_size
        if soap_settings is None:
            soap_settings = dict()

        # Build the model
        self.model = Pipeline([
            ('soap', SOAPConverter(**soap_settings)),
            ('kernel', ScalableKernel(max_points=max_kernel_size, gamma=gamma)),
            ('model', BayesianRidge(fit_intercept=True))
        ])

    def infer(self, X: Atoms) -> float:
        return self.model.predict([X])[0]

    def retrain(self, data: BaseDataSource):
        X, y = data.get_all_data()

        # Fit the model
        # TODO (wardlt): Consider adding some hyperparameter optimization
        self.model.fit(X, y)

class RFPipeline:
    def __init__(self, soap_settings, max_kernel_size, gamma, nestimators, random_state):
        self.model = Pipeline([
            ('soap', SOAPConverter(**soap_settings)),
            ('kernel', ScalableKernel(max_points=max_kernel_size, gamma=gamma)),
            ('model', RandomForestRegressor(nestimators, random_state=random_state))
        ])
        self.hasbeenfit = False
    def fit(self, X, y):
        self.model.fit(X, y)
        self.hasbeenfit = True

    def predict(self, X):
        return self.model.predict(X)

    def merge_model(self, other_model):
        
        self.model[2].estimators_ += other_model.model[2].estimators_
        self.model[2].n_estimators = len(self.model[2].estimators_)

    @staticmethod
    def aggregate_models(models: list):
        #import pdb; pdb.set_trace()
        if models and all([m.hasbeenfit for m in models]):
            model = copy.deepcopy(models[0])
            for other in models[1:]:
                model.merge_model(other)
            return model
        return None

def create_model(optimizer='adagrad',
                kernel_initializer='glorot_uniform'):
    model = Sequential()
    model.add(Dense(3000,input_shape=(5,3780),activation='relu',kernel_initializer=kernel_initializer)) #
    model.add(Dense(3000,activation='sigmoid',kernel_initializer=kernel_initializer))

    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    return model

class GAPSurrogateParallel(BaseInferenceEngine):
    """Inference engine using the Gaussian Approximation Potentials

    Uses SOAP to compute molecular representation, a scalable kernel
    to compute molecular symmetries, and BayesianRidge regression to
    fit model parameters
    """

    def __init__(self, max_kernel_size: int = 256, soap_settings: Optional[dict] = None,
                 gamma: float = 1.0, nestimators=None, random_state=None, comm=None):
        """
        Args:
            max_kernel_size (int): Maximum number of training entries to use in the GAP KRR model
                Larger values lead to higher accuracies at a greater computational cost
            soap_settings (dict): Settings that define the SOAP representation for a molecule
            gamma (float): Width parameter for the kernels
        """
        super().__init__()
        self.max_kernel_size = max_kernel_size
        if soap_settings is None:
            soap_settings = dict()

        # Get rank and comm size
        self.comm = comm
        rank = self.comm.Get_rank() if self.comm else 0
        size = self.comm.Get_size() if self.comm else 1
        random_state = random_state or rank
        self.nestimators = (nestimators or 1000) // size

        # Build the model
        # self._model = Local Model
        # self.model  = Global Model
        ##self._model = RFPipeline(soap_settings, max_kernel_size, gamma, self.nestimators, random_state)
        self._model = NNPipeline(soap_settings, max_kernel_size, gamma, self.nestimators, random_state)
        # Initialize global model with local model.
        # This global model will be updated in `global_update`
        self.model = self._model


    def global_update(self):
        if self.comm:
            model_list = self.comm.allgather(self._model)
            new = RFPipeline.aggregate_models(model_list)
            self.model = new if new is not None else self._model
        else:
            # Not a parallel run - just use local model
            self.model = self._model

    def infer(self, X: Atoms) -> float:
        return self.model.predict([X])[0]

    def retrain(self, data: BaseDataSource):
        X, y = data.get_all_data()

        # Fit the model
        # TODO (wardlt): Consider adding some hyperparameter optimization
        self.model.fit(X, y)
