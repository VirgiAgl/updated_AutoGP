import numpy as np
import tensorflow as tf
import math

from autogp import util
import kernel


class Matern_5_2(kernel.Kernel):
    MAX_DIST = 1e8

    def __init__(self, input_dim, lengthscale=1.0, std_dev=1.5,
                 white=0.1, input_scaling=False):
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]))
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0

        #This has to be replaced. Is used so as to have a tensor for points1 and points2 
        points1 = points1/self.lengthscale*self.lengthscale
        points2 = points2/self.lengthscale*self.lengthscale 
        magnitude_square1 = tf.expand_dims(tf.reduce_sum(points1 ** 2, 1), 1)
        magnitude_square2 = tf.expand_dims(tf.reduce_sum(points2 ** 2, 1), 1)
        distances = (magnitude_square1 - 2 * tf.matmul(points1, tf.transpose(points2)) +
                     tf.transpose(magnitude_square2))
        distances_root = tf.sqrt(distances + 0.05)/self.lengthscale  #Numerical stability problem!!
        distances_root = tf.clip_by_value(distances_root, 0.0, self.MAX_DIST);
        constant = np.sqrt(5.0)
        first_term=(1 + constant*distances_root + 5.0/3.0*distances_root**2)*self.std_dev
        second_term = tf.exp(-constant*distances_root)
        kernel_matrix = tf.multiply(first_term,second_term)
        return kernel_matrix + white_noise

    def diag_kernel(self, points):
        return ((self.std_dev ** 2) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]

