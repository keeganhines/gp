from math import exp
from math import pi as Pi
from numpy.random import normal, multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import sys
import argparse
from kernels import kernel_factory

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

def expand_kernel(kernel, X_1, X_2):
	Sigma = np.zeros((len(X_1), len(X_2)))
	for i in range(0, len(X_1)):
		for j in range(0, len(X_2)):
			Sigma[i,j] = kernel(X_1[i], X_2[j])
	return Sigma

def prior_sample(config):
	linestyles = ['-', '--', '-.', ':']
	kernel = kernel_factory(config)
	print("Instantiated prior with kernel: " + kernel.name())
	x = np.sort(np.random.uniform(-5,5,config.num_x_inputs))
	Sigma = expand_kernel(kernel, x, x)

	fig=plt.figure()
	gs=GridSpec(2,2)
	ax1=fig.add_subplot(gs[0,0])
	x_dist = np.linspace(0, max(x) + 5, num=50)
	kern_value = [kernel(0, thing) for thing in x_dist]
	plt.plot(x_dist, kern_value, '-', linewidth=4)

	ax2=fig.add_subplot(gs[0,1])
	plt.imshow(Sigma, extent=[min(x), max(x), min(x), max(x)])

	ax1=fig.add_subplot(gs[1,:])
	plt.ylim(-3,3)
	for i in range(0,3):
		g =  multivariate_normal(np.zeros(len(x)), Sigma) 
		plt.plot(x,g,linestyles[i], linewidth=3, color = cm.Paired(i*30))
	plt.savefig('gp_prior.png')
	plt.close()


def posterior_sample(config):
	kernel = kernel_factory(config)
	print("Instantiated posterior with kernel: " + kernel.name())
	x_observed = np.sort(np.random.uniform(-15,15,20))
	y_observed = np.sin((Pi * (x_observed))/5)
	x_test = np.linspace(-15,15,200)

	# Joint prior covariance is a block matrix with these four components
	# Eq (2.18)
	X_obs_obs = expand_kernel(kernel, x_observed, x_observed)
	X_obs_test = expand_kernel(kernel, x_observed, x_test)
	X_test_obs = expand_kernel(kernel, x_test,x_observed)
	X_test_test = expand_kernel(kernel, x_test, x_test)

	# Eq (2.19)
	posterior_mean = X_test_obs.dot(np.linalg.inv(X_obs_obs)).dot(y_observed)
	posterior_covariance = X_test_test - X_test_obs.dot( np.linalg.inv(X_obs_obs) ).dot(X_obs_test)

	fig=plt.figure()
	gs=GridSpec(2,1)
	ax1=fig.add_subplot(gs[0,0])
	plt.plot(x_observed,y_observed,'k+',markersize=15)
	for i in range(0,5):
		p =  multivariate_normal(posterior_mean, posterior_covariance) 
		plt.plot(x_test,p,'--', color = cm.Paired(i*30),linewidth=1)


	ax1=fig.add_subplot(gs[1,0])
	plt.plot(x_observed,y_observed,'k+',markersize=15)
	plt.xlim([min(x_test), max(x_test)])
	plt.plot(x_test, posterior_mean, 'k-',linewidth=1, alpha=.5)
	variances = np.sqrt(np.diag(posterior_covariance))
	upper_conf = posterior_mean + 2 * variances
	lower_conf = posterior_mean - 2 * variances
	ax1.fill_between(x_test, upper_conf , lower_conf, alpha=.5, color="#3690C0", linewidth=0)
	plt.savefig('gp_posterior.png')


if __name__ == '__main__':
	config = parse_inputs()
	prior_sample(config)
	posterior_sample(config)



