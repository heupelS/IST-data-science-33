import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )
from general_utils import get_plot_folder_path

from ts_functions import plot_series, HEIGHT

from numpy import ones

import pandas as pd
from pandas import Series, DataFrame

from matplotlib.pyplot import figure, xticks, savefig, subplots, show


def smoothing(data: DataFrame, targets: list, name: str, win_size: int=10, x_label: str='No Label', y_label: str='No Label'):
    rolling_multi = data.rolling(window=win_size)
    smooth_df_multi = rolling_multi.mean()
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(smooth_df_multi[targets[0]], 
        title=f'Appliances - Smoothing (win_size={win_size})', 
        x_label=x_label, 
        y_label=y_label)
    
    for i in range(1, len(targets)):
        plot_series(smooth_df_multi[targets[i]])

    xticks(rotation = 45)
    savefig(  os.path.join(get_plot_folder_path(), '%s_smoothing_win_%s' % (name, win_size) ) )


def differention(data: DataFrame, targets: list, name: str, win_size: int=10, x_label: str='No Label', y_label: str='No Label'):
    diff_df_multi = data.diff()
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(diff_df_multi[targets[0]], title='Appliances - Differentiation', x_label=x_label, y_label=y_label)
    
    for i in range(1, len(targets)):
        plot_series(diff_df_multi[targets[i]])

    xticks(rotation = 45)
    savefig(  os.path.join(get_plot_folder_path(), '%s_differentation_%s' % (name, win_size) ) )
