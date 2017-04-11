import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matern_3_2_2 import Matern_3_2_2
import kernel


# Generate synthetic data.
N_all = 200
N =  50
inputs = 5 * np.linspace(0, 1, num=N_all)[:, np.newaxis]
outputs = np.sin(inputs)

# selects training and test
idx = np.arange(N_all)
np.random.shuffle(idx)
xtrain = inputs[idx[:N]]
ytrain = outputs[idx[:N]]

prova_kernel = Matern_3_2(1)
values = prova_kernel.kernel(points1 = xtrain, points2 = xtrain)
print(values)
kernel_chol = tf.cholesky(values)
print(kernel_chol)
inverse_ch = tf.matrix_inverse(kernel_chol)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("This is the kernel")
print values.eval(session=sess)
print("This is cholesky")
print kernel_chol.eval(session=sess)
print("This is the inverse of Cholesky")
print inverse_ch.eval(session=sess)


