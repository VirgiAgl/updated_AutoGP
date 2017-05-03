import autogp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.metrics.pairwise as sk
import time
import scipy 
import seaborn as sns
import random

from kerpy.Kernel import Kernel

from kerpy.MaternKernel import MaternKernel

random.seed(1200)

# Generate synthetic data. N_all = total number of observations, N = training points.
N_all = 5

inputs = np.linspace(0, 1, num=N_all)[:, np.newaxis]

print('inputs', inputs)
print('shape 0', inputs.shape[0])
print('shape 1', inputs.shape[1])

sigma_rbf = sk.rbf_kernel(inputs, inputs)

print('rbf kernel', sigma_rbf)
#sigma = sk.rbf_kernel(inputs, inputs, gamma = 50)
#print('shape of sigma', sigma.shape)
cholesky_rbf = np.linalg.cholesky(sigma_rbf)
print('cholesky rbf', cholesky_rbf)

sigma_matern = MaternKernel(nu=1.5).kernel(inputs,inputs)


print('Matern nu=0.5', sigma_matern)

cholesky_matern = np.linalg.cholesky(sigma_matern)
print('cholesky matern', cholesky_matern)
