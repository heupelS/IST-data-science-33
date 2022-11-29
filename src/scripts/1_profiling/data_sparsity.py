#!/usr/bin/env python3

import sys, os
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show, title
from ds_charts import get_variable_types, HEIGHT
from seaborn import heatmap

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )

from general_utils import get_plot_folder_path
from load_data import read_data

###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
###########################################

def plot_scatter(data, name):
    register_matplotlib_converters()

    vars = get_variable_types(data)
    if [] == vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = len(vars)-1, len(vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1 = vars[i]
        for j in range(i+1, len(vars)):
            var2 = vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])

    savefig( os.path.join(get_plot_folder_path(), f'{name}') )

    if SHOW_PLOTS:
        show()

def plot_correlation(data, name):
    savefig( os.path.join(get_plot_folder_path(), name) )

    if SHOW_PLOTS:
        show()

if __name__ == "__main__":

    data_diabetic, data_drought = read_data()

