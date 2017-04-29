import autogp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Generate synthetic data.
N_all = 100
N =  50

#inputs = 5 * np.linspace(0, 1, num=N_all)[:, np.newaxis]
#outputs = np.sin(inputs)
inputs = np.linspace(0, 1, num=N_all)[:, np.newaxis]
sigma = autogp.kernels.RadialBasis(1).kernel(inputs)

n_samples =1
process_values = rep(np.random.normal(mean=rep(0,N_all), sigma=sigma),n_samples)
print('process values', process_values.shape)

sample_intensity = np.exp(process_values)
print('sample intensity', sample_intensity)
#outputs = np.random.poisson()

#outputs1 = np.random.poisson(lam=1.0, size = N_all/3)[:, np.newaxis]
#outputs2 = np.random.poisson(lam=2.0, size = N_all/3)[:, np.newaxis]
#outputs3 = np.random.poisson(lam=3.0, size = N_all/3)[:, np.newaxis]
#outputs = np.vstack((outputs1,outputs2, outputs3))

#print(outputs.shape)

# selects training and test
idx = np.arange(N_all)
np.random.shuffle(idx)
xtrain = inputs[idx[:N]]
ytrain = outputs[idx[:N]]
data = autogp.datasets.DataSet(xtrain, ytrain)
xtest = inputs[idx[N:]]
ytest = outputs[idx[N:]]

# Initialize the Gaussian process.
likelihood = autogp.likelihoods.LGCP()
kernel = [autogp.kernels.RadialBasis(1)]
inducing_inputs = xtrain
model = autogp.GaussianProcess(likelihood, kernel, inducing_inputs)

# Train the model.
optimizer = tf.train.RMSPropOptimizer(0.005)
model.fit(data, optimizer, loo_steps=0, var_steps=50, epochs=100)

# Predict new inputs.
ypred, post_var = model.predict(xtest) #V_the command predict gives back the predicted mean and the predicted variance corresponding to the xtest
plt.plot(xtrain, ytrain, '.', mew=2) #V_plot the points used to train the model
plt.plot(xtest, ytest, 'o', mew=2) #V_plot the points used to trest the model
plt.plot(xtest, ypred, 'x', mew=2) #V_plot the tested x with the predicted y
plt.show()

