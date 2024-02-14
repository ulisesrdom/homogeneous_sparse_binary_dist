# -*- coding: utf-8 -*-
import numpy as np
import argparse
import main_plots as m_plots


# --------------------------------------------------------------------------------
# Construct the argument parser and parse the arguments
# -------------------------------------------------------
ap = argparse.ArgumentParser()

# ---------Parameters to select a condition type ---
ap.add_argument("-OPT"       ,  "--OPTION", required = True, help="Probability density function plots (1); Histogram plots from simulated samples (2); Numerical convergence visualization plots (3); Data fitting and visualization of fitted model (4).")
ap.add_argument("-SAMP_T"    ,  "--SAMPLING_TYPE", required = True, help="Continuous population rate distributions (1); N-dimensional binary neurons distributions (2).")
ap.add_argument("-VIS_C_T"   ,  "--VISUAL_CONVERGENCE_TEST", required = True, help="Polynomial Q_N(r_N; THETA_N) (1); Probability mass functions P(r_N; THETA_N) (2).")
ap.add_argument("-DIST_T"    ,  "--DISTRIBUTION_TYPE", required = True, help="Polylogarithmic exponential (1); Shifted-geometric exponential (2); Independent exponential (3); Second-order exponential (4); All models comparison (5).")
ap.add_argument("-N_SAMP"    ,  "--N_SAMPLES", required = True, help="Number of samples to draw for OPTION=2 and OPTION=4.")
ap.add_argument("-N_P"       ,  "--N_P", required = True, help="Number of bins for the histograms for OPTION=2 and OPTION=4; or maximum number of points per row for OPTION=3.")
ap.add_argument("-N"         ,  "--N", required = True, help="Number of neurons in the population.")
ap.add_argument("-BASE_PAR"  , "--BASELINE_PARAMETERS", required=True, help="Comma separated baseline parameter values for each distribution in each option type.")
# ---------Parameters for visualization and other options ----
ap.add_argument("-IN_F",     "--INPUT_FOLDER", required = True, help="Input folder where input spiking data is stored for data fitting.")
ap.add_argument("-IN_FI",    "--INPUT_FILES", required = True, help="Comma separated input file names with the spiking data pickle files.")
ap.add_argument("-OUT_F",    "--OUTPUT_FOLDER", required = True, help="Output folder to store variables values and results.")
ap.add_argument("-COLOR_L",  "--COLOR_LIST", required = True, help="Comma separated colors for each curve in the 1D plots or for the color ranges of the 2D plots.")
ap.add_argument("-LSTY",     "--LINE_STYLES", required = True, help="Comma separated line styles for the curves in the 1D plots.")
ap.add_argument("-DPI",      "--DPI", required = True, help="Dots per inch for the quality of the plots to generate.")

args = vars(ap.parse_args())

# Read parameters from the input arguments----------
# --------------------------------------------------
OPTION           = int(args['OPTION'])
SAMPLING_TYPE    = int(args['SAMPLING_TYPE'])
VISUAL_CONV_TEST = int(args['VISUAL_CONVERGENCE_TEST'])
DISTR_TYPE       = int(args['DISTRIBUTION_TYPE'])
N_SAMP           = int(args['N_SAMPLES'])
N_P              = int(args['N_P'])
N                = int(args['N'])
BASE_PARAMETERS  = str(args['BASELINE_PARAMETERS'])

IN_FOLDER        = str(args['INPUT_FOLDER'])
IN_FILES         = str(args['INPUT_FILES']).split(',')
OUT_FOLDER       = str(args['OUTPUT_FOLDER'])
COLOR_LIST       = str(args['COLOR_LIST'])
LINE_STYLES      = str(args['LINE_STYLES'])
DPI              = int(args['DPI'])
PLOT_PARAM_LIST  = [COLOR_LIST,LINE_STYLES,DPI]


# Main plotting function calls ---------------------
# --------------------------------------------------
if OPTION == 1 :
   # Function call to plot the probability density functions for the continuous population rate
   m_plots.plot_PDF( DISTR_TYPE, N, BASE_PARAMETERS,\
                     OUT_FOLDER, PLOT_PARAM_LIST )
elif OPTION == 2 :
   # Function call to plot histograms after simulation of random variables from a selected distribution
   m_plots.plot_histogram( SAMPLING_TYPE, DISTR_TYPE, N_SAMP,N_P,N, BASE_PARAMETERS,\
                           OUT_FOLDER, PLOT_PARAM_LIST )
elif OPTION == 3:
   # Function call to plot the absolute difference between continuous functions in the limit
   # and their discrete approximation as the number of neurons N vary
   m_plots.plot_visual_convergence_test( VISUAL_CONV_TEST, DISTR_TYPE, N_P, BASE_PARAMETERS,\
                                         OUT_FOLDER, PLOT_PARAM_LIST )
else:
   # Function call to fit the model parameters to a given dataset under the maximum likelihood principle
   m_plots.plot_model_fit( DISTR_TYPE, N_SAMP,N_P, BASE_PARAMETERS, IN_FOLDER, IN_FILES, OUT_FOLDER, PLOT_PARAM_LIST )
