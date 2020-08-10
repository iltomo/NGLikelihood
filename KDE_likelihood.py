#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from scipy.interpolate import interp1d
#from distutils.version import LooseVersion
#from scipy.stats import norm
#from sklearn.neighbors import KernelDensity
#from astropy.io import fits
#import glob

from cosmosis.datablock import names, SectionOptions

class KDELikelihood (object):

	like_name = "KDE"

	def __init__ (self, options):
		self.options=options

	def do_likelihood (self, block):

			like = 0.1
			block[names.likelihoods, self.like_name+"_LIKE"] = like
	
	def cleanup (self):		
			pass

	@classmethod
	def build_module (cls):

		def setup (options):
			options=SectionOptions(options)
			likelihoodCalculator=cls(options)
			return likelihoodCalculator

		def execute (block, config):
			likelihoodCalculator=config
			likelihoodCalculator.do_likelihood(block)
			return 0

		def cleanup (config):
			likelihoodCalculator=config
			likelihoodCalculator.cleanup()

		return setup, execute, cleanup

setup, execute, cleanup = KDELikelihood.build_module ()
