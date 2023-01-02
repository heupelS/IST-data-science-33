
import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data
from data_profiling.granuality import data_granularity

from ds_charts import get_variable_types, choose_grid, HEIGHT

import pandas as pd
from matplotlib.pyplot import subplots, savefig, show

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
