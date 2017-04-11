import numpy as np
import tensorflow as tf
import math

from autogp import util
import kernel


class Polynomial(kernel.Kernel):

    def __init__(self, input_dim, std_dev=1.5, degree = 3.0, offset = 1.0,
                 white=0.1, input_scaling=False):
        if input_scaling:
            self.std_dev = tf.Variable(std_dev * tf.ones([input_dim]))
        else:
            self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.white = white

        self.offset = offset
        self.degree = degree

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0

        #This has to be replaced. Is used so as to have a tensor for points1 and points2 
        points1 = points1/self.std_dev*self.std_dev
        points2 = points2/self.std_dev*self.std_dev 

        variance = self.std_dev**2
        kernel_matrix = (tf.matmul(points1 * variance, tf.transpose(points2)) + self.offset)**self.degree
        return kernel_matrix + white_noise

    def diag_kernel(self, points):
        variance = self.std_dev**2
        diag = tf.reduce_sum(tf.square(points) * variance, 1)
        return (((diag + self.offset)**self.degree) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.std_dev]

