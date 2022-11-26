#!/usr/bin/env python3

import sys
sys.path.append('../../utils')

from general_utils import get_plot_folder_path
from load_data import load_diabetic_data
# import ds_charts

import pandas as pd

def data_dimemsionality():
    data = load_diabetic_data()

    print('Dim of data: %s' % str(data.shape))

    # plot(data)

def plot(data):
    figure(figsize=(4,2))
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
    savefig(get_plot_folder_path())

if __name__ == "__main__":
    data_dimemsionality()
