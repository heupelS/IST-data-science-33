#!/usr/bin/env python3

import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data

from ds_charts import bar_chart
import pandas as pd
from matplotlib.pyplot import savefig, show, figure

def plot(data, name):
    figure(figsize=(4,2))
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
    savefig(  os.path.join(get_plot_folder_path(), name) )
    show()

if __name__ == "__main__":
    data_diabetic, data_drought = read_data()
    plot(data_diabetic, 'diabetic_records_variables')
    plot(data_drought, 'drought_records_variables')