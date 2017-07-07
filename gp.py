from math import exp
from math import pi as Pi
from numpy.random import normal, multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import sys
import argparse
import kernels as k
import visualization as viz

def parse_inputs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-kernel", help="number of data samples",
		type=str)
	parser.add_argument("--length_scale", help="length scale of SE kernel",
		type=float)
	parser.add_argument("--sigma", help="variance of SE kernel",
		type=float)
	parser.add_argument("--num_x_inputs", help="number of data samples",
		type=int)
	parser.add_argument("--period", help="number of data samples",
		type=float)
	config = parser.parse_args()
	return config 

class GaussianProcess:
	def __init__(self, kernel):
		if not isinstance(kernel, k.Kernel):
			raise ValueError("Gaussian Process must be instantiated with a valid Kernel object.")
		self.kernel = kernel
		self.x = np.sort(np.random.uniform(-5,5,200))
		self.mean = np.zeros(len(self.x))
		self.covariance = k.expand_kernel(self.kernel, self.x, self.x)

		# placeholders for fitted data
		self.is_fitted = False
		self.x_observed = None
		self.y_observed = None
		self.x_test = None

	def sample(self, num_samples):
		samples = []
		for i in range(0,num_samples):
			g =  multivariate_normal(self.mean, self.covariance) 
			samples.append(g)
		return samples

	def fit(self, x_observed, y_observed):
		self.x_test = np.linspace(min(x_observed),max(x_observed),200)
		self.x_observed = x_observed
		self.y_observed = y_observed

		# Joint prior covariance is a block matrix with these four components
		# Eq (2.18)
		X_obs_obs = k.expand_kernel(self.kernel, self.x_observed, self.x_observed)
		X_obs_test = k.expand_kernel(self.kernel, self.x_observed, self.x_test)
		X_test_obs = k.expand_kernel(self.kernel, self.x_test, self.x_observed)
		X_test_test = k.expand_kernel(self.kernel, self.x_test, self.x_test)

		# Eq (2.19)
		self.mean = X_test_obs.dot(np.linalg.inv(X_obs_obs)).dot(y_observed)
		self.covariance = X_test_test - X_test_obs.dot( np.linalg.inv(X_obs_obs) ).dot(X_obs_test)
		self.is_fitted = True

	def predict(self):
		variances = np.sqrt(np.diag(self.covariance))
		upper_conf = self.mean + 2 * variances
		lower_conf = self.mean - 2 * variances
		return (self.mean, upper_conf, lower_conf)


if __name__ == '__main__':
	config = parse_inputs()
	kernel = k.kernel_factory(config)
	gp = GaussianProcess(kernel)
	print("Instantiated GP with kernel: " + kernel.name())
	print("Visualizing GP prior distribution")
	viz.prior_viz(gp)

	print("Fitting model")
	x_observed = np.sort(np.random.uniform(-15,15,20))
	y_observed = np.sin((Pi * (x_observed))/5)
	gp.fit(x_observed, y_observed)

	print("Visualizing GP posterior distribution")
	viz.posterior_viz(gp)
