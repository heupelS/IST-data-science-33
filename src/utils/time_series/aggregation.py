import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )
from general_utils import get_plot_folder_path
from load_data import read_data

from ts_functions import plot_series, HEIGHT

import pandas as pd
from pandas import Series

from matplotlib.pyplot import figure, xticks, savefig


def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df


def aggregate_multi(data, agg_type: str, index_multi: str, targets: list, name: str, y_label='No Label'):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_multi_df = aggregate_by(data, index_multi, agg_type)

    plot_series(agg_multi_df[targets[0]], title='Application - %s_%s' % (name, agg_type), x_label=agg_type, y_label=y_label)

    for i in range(1, len(targets)):
        plot_series(agg_multi_df[targets[i]])
        pass
    xticks(rotation = 45)
    savefig(  os.path.join(get_plot_folder_path(), '%s_%s' % (name, agg_type) ) )


def aggregate_uni(data, agg_type, name):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', agg_type)
    plot_series(agg_df, title='Daily consumptions', x_label='timestamp', y_label='consumption')
    xticks(rotation = 45)
    savefig(  os.path.join(get_plot_folder_path(), '%s_%s' % (name, agg_type) ) )
