import autogp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.metrics.pairwise as sk
import time
import scipy 

# This code does the following:
# generate the values for f
# generate the values for exp pf f = intensity
# generate with this intensity some observations (thus each obs will correspond to a different intensity)
# use the inputs and the outputs in the GP model
# look at the posterior distribution for the intenisty to see if the true values of lambda are in the interval


# Generate synthetic data. N_all = total number of observations, N = training points.
N_all = 300
N =  200

inputs = np.linspace(0, 1, num=N_all)[:, np.newaxis]
sigma = sk.rbf_kernel(inputs, inputs)
#print('shape of sigma', sigma.shape)

n_samples = 1  # num of realisations for the GP
process_values = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma)
#print('process values', process_values.shape)
process_values = np.reshape(process_values, (N_all,n_samples))
#print('values after reshape', process_values.shape)

sample_intensity = np.exp(process_values)
#print('sample intensity', sample_intensity.shape)

outputs = np.ones((N_all,n_samples))
for i in range(N_all):
	for j in range(n_samples):
		outputs[i,j] = np.random.poisson(lam=sample_intensity[i,j])

#print('size of the outputs',outputs.shape)
#print('observations',outputs.shape)


# selects training and test
idx = np.arange(N_all)
np.random.shuffle(idx)
xtrain = inputs[idx[:N]]
ytrain = outputs[idx[:N]]
data = autogp.datasets.DataSet(xtrain, ytrain)
xtest = inputs[idx[N:]]
ytest = outputs[idx[N:]]


# Initialize the Gaussian process.
likelihood = autogp.likelihoods.LGCP(offset=0.25)
kernel = [autogp.kernels.Matern_3_2(1, lengthscale=0.1)]


# Define the sparse approximation
sparsity_factor = 0.3
inducing_number = int(sparsity_factor*N)
#print ('number of inducing inputs', inducing_number)
id_sparse = np.arange(N)
np.random.shuffle(id_sparse)
inducing_inputs = xtrain[id_sparse[:inducing_number]] 


# Define the model
model = autogp.GaussianProcess(likelihood, kernel, inducing_inputs)

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
np.savetxt("data_inputs.csv", inputs, header='inputs',  delimiter=",")
np.savetxt("data_outputs.csv", outputs,  header='outputs', delimiter=",")
np.savetxt("xtest.csv", xtest,  header='outputs', delimiter=",")
np.savetxt("total_results_ypred.csv", ypred_np, header='ypred', delimiter =",")
np.savetxt("total_results_postvar.csv", post_var_np, header='post_var', delimiter =",")


# Plot the training set, the set test and the posterior mean for the intesity which is equal to the E[y].
plt.plot(xtrain, ytrain, '.', mew=2) #V_plot the points used to train the model
plt.plot(xtest, ytest, 'o', mew=2) #V_plot the points used to trest the model
plt.plot(xtest, ypred, 'x', mew=2) #V_plot the tested x with the predicted y
plt.show()


#Plot posterior intensity together with the the intensity used to generate the model
plt.plot(xtest, ypred, 'x', mew=2) #plotting the ypred which is the pposterior mean of the intensity
plt.plot(xtest, sample_intensity[idx[N:]], 'o', mew=2) #plotting the intensity used to generate ytest
plt.show()


# Plotting the histograms for the true intensity and posterior mean intensity
count, bins, ignored = plt.hist(sample_intensity, normed=True)
count, bins, ignored = plt.hist(ypred, normed=True)
plt.show()

