import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )
from general_utils import get_plot_folder_path

from ts_functions import plot_series, HEIGHT

from numpy import ones

import pandas as pd
from pandas import Series, DataFrame

from matplotlib.pyplot import figure, xticks, savefig, subplots, show


def plot_aggregate_multi(data, agg_type: str, index_multi: str, targets: list, name: str, y_label='No Label'):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_multi_df = aggregate_multi(data, index_multi, agg_type)

    plot_series(agg_multi_df[targets[0]], title='Application - %s_%s' % (name, agg_type), x_label=agg_type, y_label=y_label)

    for i in range(1, len(targets)):
        plot_series(agg_multi_df[targets[i]])
        
    xticks(rotation = 45)
    savefig(  os.path.join(get_plot_folder_path(), '%s_aggregation_%s' % (name, agg_type) ) )


def plot_smoothing(data: DataFrame, targets: list, name: str, win_size: int=10, x_label: str='No Label', y_label: str='No Label'):
    smooth_df_multi = smoothing(data, win_size)
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(smooth_df_multi[targets[0]], 
        title=f'Appliances - Smoothing (win_size={win_size})', 
        x_label=x_label, 
        y_label=y_label)
    
    for i in range(1, len(targets)):
        plot_series(smooth_df_multi[targets[i]])

    xticks(rotation = 45)
    savefig(  os.path.join(get_plot_folder_path(), '%s_smoothing_win_%s' % (name, win_size) ) )


def plot_differention(data: DataFrame, targets: list, name: str, win_size: int=10, x_label: str='No Label', y_label: str='No Label'):
    diff_df_multi = differention(data)
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(diff_df_multi[targets[0]], title='Appliances - Differentiation', x_label=x_label, y_label=y_label)
    
    for i in range(1, len(targets)):
        plot_series(diff_df_multi[targets[i]])

    xticks(rotation = 45)
    savefig(  os.path.join(get_plot_folder_path(), '%s_differentation_%s' % (name, win_size) ) )


# Core functions

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

def aggregate_multi(data, index_multi, agg_type):
    return aggregate_by(data, index_multi, agg_type)

def differention(data):
    return data.diff()

def smoothing(data, win_size):
    rolling_multi = data.rolling(window=win_size)
    return rolling_multi.mean()