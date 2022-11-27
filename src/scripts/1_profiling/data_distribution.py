#!/usr/bin/env python3

import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data

from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from matplotlib.pyplot import savefig, show

def data_distribution(data):
    register_matplotlib_converters()
    summary5 = data.describe()

    print(summary5)

def plot(data, name):
    data.boxplot(rot=45)
    savefig( os.path.join(get_plot_folder_path(), name) )
    show()

if __name__ == "__main__":

    data_diabetic, data_drought = read_data()

    data_distribution(data_diabetic)
    print('#############################')
    data_distribution(data_drought)

    plot(data_diabetic, 'diabetic_data_box' )
    plot(data_drought, 'drought_data_box' )
