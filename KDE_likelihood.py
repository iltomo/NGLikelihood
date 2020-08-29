import numpy as np
from cosmosis.datablock import names, SectionOptions, option_section
from multi_twopoint_cosmosis import theory_names, type_table
#from twopoint_cosmosis import theory_names, type_table
import twopoint
from spec_tools import SpectrumInterp

class KDELikelihood (object):

	like_name = "KDE"

	def __init__ (self, options):
		self.options=options
		self.data_x, self.data_y = self.build_data()
		self.ndraw = self.options.get_int ("ndraw", default=100) 

	def build_data (self):
		filename = self.options.get_string('data_file')
		covmat_name = self.options.get_string("covmat_name", "COVMAT")

		suffix = self.options.get_string('suffix', default="")
		if suffix:
			self.suffix = "_" + suffix
		else:
			self.suffix = suffix

		# This is the main work - read data in from the file
		self.two_point_data = twopoint.TwoPointFile.from_fits(filename, covmat_name)

		# All the names of two-points measurements that were found in the data file
		all_names = [spectrum.name for spectrum in self.two_point_data.spectra]

		# We may not want to use all the likelihoods in the file.
		# We can set an option to only use some of them
		data_sets = self.options.get_string("data_sets", default="all")
		if data_sets != "all":
			data_sets = data_sets.split()
			self.two_point_data.choose_data_sets(data_sets)
		
		 # The ones we actually used.
		self.used_names = [spectrum.name for spectrum in self.two_point_data.spectra]

		# Check for scale cuts. In general, this is a minimum and maximum angle for
    # each spectrum, for each redshift bin combination. Which is clearly a massive pain...
    # but what can you do?
		scale_cuts = {}
		for name in self.used_names:
			s = self.two_point_data.get_spectrum(name)
			for b1, b2 in s.bin_pairs:
				option_name = "angle_range_{}_{}_{}".format(name, b1, b2)
				if self.options.has_value(option_name):
					r = self.options.get_double_array_1d(option_name)
					scale_cuts[(name, b1, b2)] = r

		# Now check for completely cut bins
    # example:
    # cut_wtheta = 1,2  1,3  2,3
		bin_cuts = []
		for name in self.used_names:
			s = self.two_point_data.get_spectrum(name)
			option_name = "cut_{}".format(name)
			if self.options.has_value(option_name):
				cuts = self.options[option_name].split()
				cuts = [eval(cut) for cut in cuts]
				for b1, b2 in cuts:
					bin_cuts.append((name, b1, b2))


		if scale_cuts or bin_cuts:
			self.two_point_data.mask_scales(scale_cuts, bin_cuts)
		else:
			print("No scale cuts mentioned in ini file.")

		# Info on which likelihoods we do and do not use
		print("Found these data sets in the file:")
		total_data_points = 0
		final_names = [spectrum.name for spectrum in self.two_point_data.spectra]
		for name in all_names:
			if name in final_names:
				data_points = len(self.two_point_data.get_spectrum(name))
			else:
				data_points = 0
			if name in self.used_names:
				print("    - {}  {} data points after cuts {}".format(name,  data_points, "  [using in likelihood]"))
				total_data_points += data_points
			else:
				 print("    - {}  {} data points after cuts {}".format(name, data_points, "  [not using in likelihood]"))
		print("Total data points used = {}".format(total_data_points))

		# Convert all units to radians.  The units in cosmosis are all
		# in radians, so this is the easiest way to compare them.
		for spectrum in self.two_point_data.spectra:
			if spectrum.is_real_space():
				spectrum.convert_angular_units("rad")


		# build up the data vector from all the separate vectors.
		# Just concatenation
		data_vector = np.concatenate([spectrum.value for spectrum in self.two_point_data.spectra])
		
		# Make sure
		if len(data_vector) == 0:
			raise ValueError("No data was chosen to be used from 2-point data file {0}. It was either not selectedin data_sets or cut out".format(filename))

		# The x data is not especially useful here, so return None.
		# We will access the self.two_point_data directly later to
		# determine ell/theta values
		return None, data_vector

	def extract_theory_points (self, block):
		theory = []
		# We may want to save these splines for the covariance matrix later
		self.theory_splines = {}

		# We have a collection of data vectors, one for each spectrum
		# that we include. We concatenate them all into one long vector,
		# so we do the same for our theory data so that they match

		# We will also save angles and bin indices for plotting convenience,
		# although these are not actually used in the likelihood
		angle = []
		bin1 = []
		bin2 = []
		dataset_name = []

		# Now we actually loop through our data sets
		theory_n = []
		for spectrum in self.two_point_data.spectra:
			theory_vector, angle_vector, bin1_vector, bin2_vector = self.extract_spectrum_prediction(block, spectrum, 1)
			theory_n.append(theory_vector)
			angle.append(angle_vector)
			bin1.append(bin1_vector)
			bin2.append(bin2_vector)
		theory_n = np.concatenate (theory_n)
		theory.append (theory_n)

		for num_of_draw in range (2, self.ndraw+1):
			theory_n = []
			for spectrum in self.two_point_data.spectra:
				theory_vector, angle_vector, bin1_vector, bin2_vector = self.extract_spectrum_prediction(block, spectrum, num_of_draw)
				theory_n.append(theory_vector)
			theory_n = np.concatenate (theory_n)
			theory.append (theory_n)

		# We also collect the ell or theta values.
		# The gaussian likelihood code itself is not expecting these,
		# so we just save them here for convenience.
		angle = np.concatenate(angle)
		bin1 = np.concatenate(bin1)
		bin2 = np.concatenate(bin2)
		# dataset_name = np.concatenate(dataset_name)
		block[names.data_vector, self.like_name + "_angle"] = angle
		block[names.data_vector, self.like_name + "_bin1"] = bin1
		block[names.data_vector, self.like_name + "_bin2"] = bin2
		# block[names.data_vector, self.like_name+"_name"] = dataset_name

## the thing it does want is the theory vector, for comparison with
## the data vector
#theory = np.concatenate(theory)
		
		return theory

	def extract_spectrum_prediction (self, block, spectrum, num_of_draw):
		# We may need theory predictions for multiple different
		# types of spectra: e.g. shear-shear, pos-pos, shear-pos.
		# So first we find out from the spectrum where in the data
		# block we expect to find these - mapping spectrum types
		# to block names
		section, x_name, y_name = theory_names(spectrum)

		# To handle multiple different data sets we allow a suffix
		# to be applied to the section names, so that we can look up
		# e.g. "shear_cl_des" instead of just "shear_cl".
		section += self.suffix

		# We need the angle (ell or theta depending on the spectrum)
		# for the theory spline points - we will be interpolating
		# between these to get the data points
		angle_theory = block[section, x_name]

		# Now loop through the data points that we have.
		# For each one we have a pairs of bins and an angular value.
		# This assumes that we can take a single sample point from
		# each theory vector rather than integrating with a window function
		# over the theory to get the data prediction - this will need updating soon.
		bin_data = {}
		theory_vector = []

		# For convenience we will also return the bin and angle (ell or theta)
		# vectors for this bin too.
		angle_vector = []
		bin1_vector = []
		bin2_vector = []

		for (b1, b2, angle) in zip(spectrum.bin1, spectrum.bin2, spectrum.angle):
			# We are going to be making splines for each pair of values that we need.
      # We make splines of these and cache them so we don't re-make them for every
      # different theta/ell data point
			if (b1, b2) in bin_data:
				# either use the cached spline
				theory_spline = bin_data[(b1, b2)]
			else:
				# or make a new cache value
				# load from the data block and make a spline
				# and save it
				if block.has_value(section, y_name.format(b1,b2,num_of_draw)):
					theory = block[section, y_name.format(b1,b2,num_of_draw)]
				# It is okay to swap if the spectrum types are the same - symmetrical
				elif block.has_value(section, y_name.format(b1,b2,num_of_draw)) and spectrum.type1 == spectrum.type2:
					theory = block[section, y_name.format(b1,b2,num_of_draw)]
				else:
					raise ValueError("Could not find theory prediction {} in section {}".format(y_name.format(b1,b2,num_of_draw), section))
				#theory_spline = interp1d(angle_theory, theory)
				theory_spline = SpectrumInterp(angle_theory, theory)
				bin_data[(b1, b2)] = theory_spline
				# This is a bit silly, and is a hack because the
				# book-keeping is very hard.
				bin_data[y_name.format(b1,b2,num_of_draw)] = theory_spline


			# use our spline - interpolate to this ell or theta value
			# and add to our list
			try:
				theory = theory_spline(angle)
			except ValueError:
				raise ValueError ("""Tried to get theory prediction for {} {}, but ell or theta value ({}) was out of range.
           "Maybe increase the range when computing/projecting or check units?""".format(section, y_name.format(b1,b2,num_of_draw), angle))
			theory_vector.append(theory)
			angle_vector.append(angle)
			bin1_vector.append(b1)
			bin2_vector.append(b2)

		# We are saving the theory splines as we may need them
		# to calculate covariances later
		self.theory_splines[section] = bin_data

		# Return the whole collection as an array
		theory_vector = np.array(theory_vector)

		# For convenience we also save the angle vector (ell or theta)
		# and bin indices
		angle_vector = np.array(angle_vector)
		bin1_vector = np.array(bin1_vector, dtype=int)
		bin2_vector = np.array(bin2_vector, dtype=int)

		return theory_vector, angle_vector, bin1_vector, bin2_vector

	def do_likelihood (self, block):

		#get data x by interpolation
		x = np.atleast_1d(self.extract_theory_points(block))
		mu = np.atleast_1d(self.data_y)

		print ("Per iniziare, la teoria")
		print (x)
		print (x[0])
		print (x[1])
		print ("E adesso i dati")
		print (mu)

		like = sum (mu+x[0]+x[1])
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
