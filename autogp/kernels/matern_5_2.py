import numpy as np
import tensorflow as tf

from autogp import util
import kernel


class Matern(kernel.Kernel):
    MAX_DIST = 1e8

    def __init__(self, input_dim, lengthscale=1.0, std_dev=1.0,
                 white=0.01, input_scaling=False):
        #if input_scaling:
        #    self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]))
        #else:
        self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.input_dim = input_dim
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0

        kern_matrix = np.zeros((tf.shape(points1)[0],tf.shape(points1)[0]))

        for i in range(tf.shape(points1)[0]):
            for j in range(tf.shape(points1)[0]):
            distances = util.euclidean(points1[i,:], points2[j,:])
            distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST); #V_this is to limit the distances between these two values
            first_term = self.std_dev*(1 + sqrt(3)*distances)
            second_term = exp(-sqrt(3)/self.lengthscale*distances)
            kern[i,j] = first_term*second_term

        return kern + white_noise

    def diag_kernel(self, points):
        return ((self.std_dev ** 2) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]

