
import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data
import pandas as pd
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
###########################################

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

    if SHOW_PLOTS:
        show()

if __name__ == "__main__":
    data_diabetic, data_drought = read_data()

    # convert date into pandas datetime64, need that for last plot
    data_drought['date'] = pd.to_datetime(data_drought['date'])

    data_granularity(data_diabetic, 'gran_diabetic', 'Numeric')
    data_granularity(data_drought, 'gran_drought', 'Numeric')

    data_granularity(data_diabetic, 'gran_diabetic', 'Symbolic')
    data_granularity(data_drought, 'gran_drought', 'Symbolic')

    # No date variables in diabetic
    #data_granularity(data_diabetic, 'gran_diabetic', 'Date')
    data_granularity(data_drought, 'gran_drought', 'Date')
