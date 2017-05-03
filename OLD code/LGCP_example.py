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
from kerpy.GaussianKernel import GaussianKernel


# This code does the following:
# generate the values for f
# generate the values for exp pf f = intensity
# generate with this intensity some observations (thus each obs will correspond to a different intensity)
# use the inputs and the outputs in the GP model
# look at the posterior distribution for the intenisty to see if the true values of lambda are close to the posterior means

## NB. When using the Matern Kernel there is still a problem of numerical stability for the inversion the Cholesky matrix! I had to increase the jitter.
## Similar problem when computing the RBF kernel
np.random.seed(1200)

# Generate synthetic data. N_all = total number of observations, N = training points.
N_all = 300
N =  150

#Set data parameters
offset_data = 1.0
lengthscale_data = 1.0/3.0  #fixed to be a third od the range of the dataset
sigma_data = 1.0
random_noise = np.random.normal(loc=0.0, scale=1.0, size=None)

#Set initial  parameters
#gamma = 1. #to check
#lengthscale_initial = np.sqrt(1/(2*gamma)) + random_noise

lengthscale_initial = lengthscale_data + random_noise
offset_initial = offset_data + random_noise
sigma_initial = sigma_data + random_noise

inputs = np.linspace(0, 1, num=N_all)[:, np.newaxis]
sigma = GaussianKernel(sigma = lengthscale_data).kernel(inputs,inputs) #lenghtscale in kerpy is called sigma
#np.savetxt("../../Workspace/updated_AutoGP/R_plots/inputs.csv", inputs, header="inputs", delimiter=",")

# There is a problem of numerical precision
#pert = np.zeros((N_all,N_all))
#np.fill_diagonal(pert, 0.001)
#print('perturbation',pert)
#print('this is the covariance used to generate the data', sigma)
#print('this is the covariance shape', sigma.shape)
#print('its cholesky', np.linalg.cholesky(sigma+pert))


#sigma = MaternKernel(width = lengthscale_data, nu = 1.5, sigma = sigma_data).kernel(inputs,inputs) #Matern 3_2
#sigma = MaternKernel(width = lengthscale_data, nu = 2.5, sigma = sigma_data).kernel(inputs,inputs) #Matern 5_2

#sigma = sk.rbf_kernel(inputs, inputs)
#sigma = sk.rbf_kernel(inputs, inputs, gamma = 50)
#print('shape of sigma', sigma.shape)

n_samples = 1  # num of realisations for the GP
process_values = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma)
process_values = np.reshape(process_values, (N_all,n_samples))
print('process_values', process_values)

sample_intensity = np.exp(process_values + offset_data)

outputs = np.ones((N_all,n_samples))
for i in range(N_all):
	for j in range(n_samples):
		outputs[i,j] = np.random.poisson(lam=sample_intensity[i,j])


# selects training and test
idx = np.arange(N_all)
np.random.shuffle(idx) #Maybe to be fixed?
print('idx', idx)
xtrain = inputs[idx[:N]]
ytrain = outputs[idx[:N]]
data = autogp.datasets.DataSet(xtrain, ytrain)
xtest = inputs[idx[N:]]
ytest = outputs[idx[N:]]


# Initialize the Gaussian process.
likelihood = autogp.likelihoods.LGCP(offset = offset_initial)
kernel = [autogp.kernels.RadialBasis(1, lengthscale= lengthscale_initial, std_dev = sigma_initial)]
#kernel = [autogp.kernels.Matern_3_2(1, lengthscale= lengthscale_initial, std_dev = sigma_initial)]
#kernel = [autogp.kernels.Matern_5_2(1, lengthscale= lengthscale_initial, std_dev = sigma_initial)]

# Define the sparse approximation
sparsity_factor = 1.
inducing_number = int(sparsity_factor*N)
id_sparse = np.arange(N)
np.random.shuffle(id_sparse)
inducing_inputs = xtrain[id_sparse[:inducing_number]] 


# Define the model
model = autogp.GaussianProcess(likelihood, kernel, inducing_inputs, num_components=1, diag_post=False)

# Define the optimizer
optimizer = tf.train.RMSPropOptimizer(0.005)

# Train the model
start = time.time()
print("Start the training")
model.fit(data, optimizer, loo_steps=0, var_steps=60, epochs=300)
end = time.time()
time_elapsed = end-start
print("Execution time in seconds", time_elapsed)


# Predict new inputs.
ypred, _ = model.predict(xtest) #V_the command predict gives back the predicted mean and the predicted variance corresponding to the xtest
_, post_var = model.predict(xtest) #V_the command predict gives back the predicted mean and the predicted variance corresponding to the xtest

ypred_np = np.asarray(ypred)
post_var_np = np.asarray(post_var)


# Save the data to export to R
np.savetxt("../../Workspace/updated_AutoGP/R_plots/data_inputs.csv", inputs, header='inputs',  delimiter=",")
np.savetxt("../../Workspace/updated_AutoGP/R_plots/data_outputs.csv", outputs,  header='outputs', delimiter=",")
np.savetxt("../../Workspace/updated_AutoGP/R_plots/xtest.csv", xtest,  header='xtest', delimiter=",")
np.savetxt("../../Workspace/updated_AutoGP/R_plots/ytest.csv", ytest,  header='ytest', delimiter=",")
np.savetxt("../../Workspace/updated_AutoGP/R_plots/xtrain.csv", xtrain,  header='xtrain', delimiter=",")
np.savetxt("../../Workspace/updated_AutoGP/R_plots/ytrain.csv", ytrain,  header='ytrain', delimiter=",")
np.savetxt("../../Workspace/updated_AutoGP/R_plots/sample_intensity_test.csv", sample_intensity[idx[N:]],  header='sample_intensity_test', delimiter=",")
np.savetxt("../../Workspace/updated_AutoGP/R_plots/total_results_ypred.csv", ypred_np, header='ypred', delimiter =",")
np.savetxt("../../Workspace/updated_AutoGP/R_plots/total_results_postvar.csv", post_var_np, header='post_var', delimiter =",")


# Plot the training set, the set test and the posterior mean for the intesity which is equal to the E[y].
first_line, = plt.plot(xtrain, ytrain, '.', mew=2, label = "a") #V_plot the points used to train the model
second_line, = plt.plot(xtest, ytest, 'o', mew=2, label = "b") #V_plot the points used to trest the model
third_line, =  plt.plot(xtest, ypred, 'x', mew=2,  label = "c") #V_plot the tested x with the predicted y
plt.ylabel('Value of the process')
plt.xlabel('x')
plt.legend([first_line, second_line, third_line], ['Training set', 'Test set', 'Predicted y values'])
plt.show()


#Plot posterior intensity together with the the intensity used to generate the model
first_line, = plt.plot(xtest, ypred, 'x', mew=2) #plotting the ypred which is the pposterior mean of the intensity
second_line, = plt.plot(xtest, sample_intensity[idx[N:]], 'o', mew=2) #plotting the intensity used to generate ytest
plt.ylabel('Intensity')
plt.xlabel('x')
plt.legend([first_line, second_line], ['Posterior mean', 'True mean'])
plt.show()


# Plotting the histograms for the true intensity and posterior mean intensity
#count, bins, ignored = plt.hist(sample_intensity, normed=True)
#count, bins, ignored = plt.hist(ypred, normed=True)
sns.distplot(sample_intensity, label='Sample intensity')
sns.distplot(ypred, label = 'Posterior mean intensity')
plt.ylabel('Frequency of the intensity')
plt.xlabel('x')
plt.legend()
plt.show()

