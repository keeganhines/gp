import numpy as np
from math import pi as Pi
from abc import ABCMeta, abstractmethod, abstractproperty


class Kernel:
	__metaclass__ = ABCMeta

	@abstractmethod
	def __call__(self, x, xprime):
		return

	@abstractmethod
	def name(self):
		pass

class SE(Kernel):
    def __init__(self, length_scale, sigma):
        self.length_scale = length_scale
        self.sigma = sigma

    def name(self):
    	return "Squared Exponential Kernel"
        
    def __call__(self, x, xprime):
        return self.sigma**2 * np.exp(-((x - xprime)**2)/ (2*self.length_scale**2))
   
class Periodic(Kernel):
	def __init__(self, length_scale, sigma, period):
		self.length_scale = length_scale
		self.sigma = sigma
		self.period = period

	def name(self):
		return "Periodic Kernel"

	def __call__(self, x, xprime):
		return self.sigma**2 * np.exp(-(2*np.sin((Pi * (x - xprime))/self.period)**2)/ (self.length_scale**2))

class LocallyPeriodic(Kernel):
	def __init__(self, length_scale, sigma, period):
		self.length_scale = length_scale
		self.sigma = sigma
		self.period = period
		self.se = SE(self.length_scale, self.sigma)
		self.periodic = Periodic(self.length_scale, self.sigma, self.period)

	def name(self):
		return "Locally Periodic Kernel"
	def __call__(self, x, xprime):
		return self.se(x,xprime) * self.periodic(x, xprime)

def kernel_factory(config):
	if config.kernel.lower() == "se":
		return SE(config.length_scale,config.sigma)
	elif config.kernel.lower() == "periodic":
		return Periodic(config.length_scale,config.sigma, config.period)
	elif config.kernel.lower() == "locally_periodic":
		return LocallyPeriodic(config.length_scale,config.sigma, config.period)
	else:
		raise ValueError("Invalid Kernel choice")
