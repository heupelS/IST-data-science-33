
import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )
from general_utils import get_plot_folder_path
from load_data import read_data

from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

def data_granularity(data, name, var_type):

    variables = get_variable_types(data)[var_type]
    if [] == variables:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(variables))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT* 1.5, rows*HEIGHT * 1.5), squeeze=False)
    i, j = 0, 0

    for n in range(len(variables)):
        axs[i, j].set_title('Histogram for %s'%variables[n])
        axs[i, j].set_xlabel(variables[n])
        axs[i, j].set_ylabel('nr records')
        axs[i, j].hist(data[variables[n]].values, bins=100)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)

    savefig( os.path.join(get_plot_folder_path(), f'{name}_{var_type}') )
