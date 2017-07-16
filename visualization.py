
"""
Utilities for visualizing Gaussian Processes.
"""
#from gp import GaussianProcess
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import numpy as np

def prior_viz(gp):
	"""
	Visualization and exploration of prior distribution. Really, this 
	utility is just for visually exploring and understanding various kernels.

	Visualizes (1) the kernel function itself and (2) the implied covariance 
	matrix (finitely sampled) and (3) a few draws from a GaussianProcess prior
	with the kernel as configured by the input.

	Output plot is saved in current directory. 


	Parameters
	----------
	gp: GaussianProcess

	"""

	linestyles = ['-', '--', '-.', ':']

	fig=plt.figure()
	gs=GridSpec(2,2)
	ax1=fig.add_subplot(gs[0,0])
	x_dist = np.linspace(0, max(gp.x) + 5, num=50)
	kern_value = [gp.kernel(0, thing) for thing in x_dist]
	plt.plot(x_dist, kern_value, '-', linewidth=4)

	ax2=fig.add_subplot(gs[0,1])
	plt.imshow(gp.covariance, extent=[min(gp.x), max(gp.x), min(gp.x), max(gp.x)])

	ax1=fig.add_subplot(gs[1,:])
	plt.ylim(-3,3)
	for i in range(0,3):
		f = gp.sample(1)[0]
		plt.plot(gp.x,f,linestyles[i], linewidth=3, color = cm.Paired(i*30))
	plt.savefig('gp_prior.png')
	plt.close()


def posterior_viz(gp):
	"""
	Visualization and exploration of posterior distribution, after model has
	been fit to observations. 

	Visualizes (1) the observed data along with (2) several draws from the learned posterior
	distribution and (3) the full poster confidence intervals. 

	Output plot is saved in current directory. 

	Parameters
	----------
	gp: GaussianProcess

	"""

	if not gp.is_fitted:
		raise ValueError("""
			You probably want to fit this model to some observations.
			Otherwise, the visualizations won't be very interesting.
			""")

	fig=plt.figure()
	gs=GridSpec(2,1)
	ax1=fig.add_subplot(gs[0,0])
	plt.plot(gp.x_observed,gp.y_observed,'k+',markersize=15)
	for i in range(0,5):
		f = gp.sample(1)[0] 
		plt.plot(gp.x_test,f,'--', color = cm.Paired(i*30),linewidth=1)

	ax1=fig.add_subplot(gs[1,0])
	plt.plot(gp.x_observed,gp.y_observed,'k+',markersize=15)
	plt.xlim([min(gp.x_test), max(gp.x_test)])
	(mean, upper_conf, lower_conf) = gp.predict()
	plt.plot(gp.x_test, mean, 'k-',linewidth=1, alpha=.5)
	ax1.fill_between(gp.x_test, upper_conf , lower_conf, alpha=.5, color="#3690C0", linewidth=0)
	plt.savefig('gp_posterior.png')
