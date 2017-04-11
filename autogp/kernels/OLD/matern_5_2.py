import numpy as np
import tensorflow as tf

from autogp import util
import kernel


class Matern_5_2(kernel.Kernel):
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

        kern_matrix = tf.zeros((len(points1),len(points2)))
        output_list = []
        for i in range(kern_matrix.shape[0]):
            for j in range(kern_matrix.shape[1]):
                distances = util.euclidean(points1[i,:], points2[j,:])
                distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST)
                #V_this is to limit the distances between these two values
                first_term = 
                first_term = tf.scalar_mul(self.std_dev, 1.0 + tf.scalar_mul(math.sqrt(5.0)/self.lengthscale,distances) + tf.scalar_mul(5.0/(3.0*self.lengthscale), distances**2))
                second_term = tf.exp(tf.scalar_mul(-math.sqrt(5.0)/self.lengthscale, distances))
                output_list.append(tf.multiply(first_term,second_term))
        kernel_matrix = tf.transpose(tf.reshape((tf.stack(output_list)), [points2.shape[0],points1.shape[0]]))
        return kernel_matrix + white_noise

    def diag_kernel(self, points):
        return ((self.std_dev ** 2) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]

