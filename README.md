# some_codes
Sample codes  for surface wave processing, okada green functions and inversion of geodetic data 


This repository contains two folders with sample code snippets I have generated within my bachelor and master degrees.

1) seismic_cycle:
seismic_cycle.py: class for generating okada green functions from subduction geometry obtained from Hayes et al., (2018) Slab2 MODEL. It is configured for generating coseismic green functions for South-Central Chile margin but it can be adapted for other phases and tectonic settings.

bayesian_inversion.py: A collection of functions for performing GNSS+InSAR inversion with a bayesian methodology. Since the solution is semianalytical, it computes a solution and uncertainties in reasonable time (although I definitely can improve some stuff!) and calculates the Bayesian Evidence for model selection.

2) swave_processing:
cc_processing.py: A collection of functions for obtaining a dispersion curve from ambient noise recordings. Based on Menke and Jin (2015) and Olivar-Castaño et al., (2020) work.
model3D_functions.py: collection of functions for generating a 3D velocity model of a site from individual 1D ground profiles derived from dispersion curves. 


swprepost is a very useful python package for surface wave inversion Pre- and Post-Processing
created by jpvantassel: https://github.com/jpvantassel/swprepost


