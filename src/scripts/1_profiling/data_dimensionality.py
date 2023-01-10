#!/usr/bin/env python3

import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data

from data_profiling.dimensionality import plot_dim, plot_missing, plot_types 

if __name__ == "__main__":

    data_diabetic, data_drought = read_data()

    plot_dim(data_diabetic, 'dims_diabetic_nr_rec_nr_vars')
    plot_dim(data_drought, 'dims_drought_nr_rec_nr_vars')

    plot_types(data_diabetic, 'dims_diabetic_var_type')
    plot_types(data_drought, 'dims_drought_var_type')

    plot_missing(data_diabetic, 'dims_diabetic_missing')

    # Don't has missing values
    # plot_missing(data_drought, 'dims_drought_missing')