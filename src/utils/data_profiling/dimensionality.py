import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data

from ds_charts import bar_chart, get_variable_types
import pandas as pd
from matplotlib.pyplot import savefig, show, figure


def plot_dim(data, name):
    figure(figsize=(4,2))
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
    savefig(  os.path.join(get_plot_folder_path(), name) )


def plot_types(data, name):
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4,2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    savefig(  os.path.join(get_plot_folder_path(), name) )


def plot_missing(data, name):
    mv = {}
    for var in data:
        nr = 0
        for el in data[var]:
            if el == '?' or el == None:
                nr = nr + 1

        if nr > 0:
            mv[var] = nr

    figure()
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
                xlabel='variables', ylabel='nr missing values', rotation=True)
    savefig(  os.path.join(get_plot_folder_path(), name) )
