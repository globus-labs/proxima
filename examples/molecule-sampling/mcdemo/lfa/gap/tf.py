"""Fitting GAP with a tensorflow

Warning: This is both slow and inaccurate. I'm not sure why"""
from typing import List

from ase import Atoms
from dscribe.descriptors import SOAP
from tensorflow_probability import positive_semidefinite_kernels as tfk
import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm


class GAPModel(tf.keras.Model):
    """Implements the Gaussian Approximation Potentials as a Keras model"""

    def __init__(self, n_features: int, n_points: int, **kwargs):
        """
        Args:
             n_features (int): Number of features describing each point
             n_points (int): Number of Gaussians
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_points = n_points

        # Create the weights
        self.observed_points = self.add_weight(
            shape=(self.n_points, self.n_features),
            trainable=True,
            name='observed_points'
        )
        self.observed_values = self.add_weight(
            shape=(self.n_points,),
            trainable=True,
            name='observed_values'
        )
        self.amplitude = tf.Variable(1.0, name='amplitude')
        self.length_scale = tf.Variable(1.0, name='length_scale')

    def preseed_weights(self, observed_points, observed_values):
        """Define initial points and observed values for the Guassians"""
        self.observed_points.assign(observed_points)
        self.observed_values.assign(observed_values)

    def call(self, inputs, training=None, mask=None):
        # Define the kernel
        kernel = tfk.ExponentiatedQuadratic(amplitude=self.amplitude, length_scale=self.length_scale)

        # Define the GPR model
        self.gpr = tfp.distributions.GaussianProcessRegressionModel(
            kernel=kernel,
            index_points=inputs,
            observation_index_points=self.observed_points,  # Observations define position of Gaussians
            observations=self.observed_values  # Used to determine position of Gaussian
        )

        # Compute the output value and uncertainties
        per_atom_energy = self.gpr.mean()
        total_energy = tf.reduce_sum(per_atom_energy, name='total_energy')
        energy_var = tf.reduce_sum(tf.multiply(per_atom_energy,
                                               tf.tensordot(self.gpr.covariance(), per_atom_energy, axes=[1, 0])),
                                   name='total_energy_var')

        return total_energy, energy_var


def neg_log_likelihood(y_obs, y_mean, y_var):
    """Negative log likelihood of a certain observation"""
    pi2 = 2 * np.pi

    # Compute the log likelihood for each sample
    error = y_mean - y_obs
    neg_log_likli = tf.math.log(pi2 * y_var) + error * error / y_var
    return neg_log_likli


def train_loop(model: GAPModel, soap: SOAP, mols: List[Atoms], energies: List[float],
               epochs: int, verbose=True):
    """Training procedure for GAP model"""
    # Make the optimizer
    opt = tf.keras.optimizers.Adam()

    # Compute the features for each atom
    features = np.array([soap.create(x) for x in mols])

    # Make an list of indices
    inds = np.arange(len(mols))

    # Run the prescribed number of epochs
    epoch_data = []
    for e in range(epochs):
        # Shuffle data before each epoch
        np.random.shuffle(inds)

        # Update gradients at each step
        full_loss = 0
        for ind in tqdm(inds, total=len(inds), desc=f'epoch {e}', disable=not verbose, leave=False):
            with tf.GradientTape() as t:
                energy_mean, energy_var = model(features[ind])
                loss = neg_log_likelihood(energies[ind], energy_mean, energy_var)
            full_loss += loss

            # Make the update to the weights
            grads = t.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_data.append({'epoch': e, 'loss': full_loss})
    return epoch_data

