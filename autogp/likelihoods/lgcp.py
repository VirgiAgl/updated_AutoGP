import numpy as np
import tensorflow as tf

import likelihood

# Implementation of a Log Gaussian Cox process

# p(y|f) = (lambda)^y exp(-lambda) / y!
# y is the number of events (points) within an area?

# lambda = f + offset



class LGCP(likelihood.Likelihood):
    def __init__(self, offset=1.0):
        self.offset = tf.Variable(offset)

    def log_cond_prob(self, outputs, latent):
        log_lambda = (latent + self.offset)
        return (outputs * log_lambda - tf.exp(log_lambda) - tf.lgamma(outputs + 1))

    def get_params(self):
        return [self.offset]

    def predict(self, latent_means, latent_vars):
        sigma = tf.sqrt(latent_vars)
        pred_means = tf.exp(latent_means + sigma / 2) * tf.exp(self.offset)
        pred_vars = (tf.exp(sigma) - 1) * tf.exp(2 * latent_means + sigma) * tf.exp(2 * self.offset)
        return pred_means, pred_vars
