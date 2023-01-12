import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )
from general_utils import get_plot_folder_path

from ts_functions import plot_series, HEIGHT

from numpy import ones

import pandas as pd
from pandas import Series

from matplotlib.pyplot import figure, xticks, savefig, subplots, show


def box_plot(data, target_index, name):
    index = data.index.to_period('W')
    week_df = data.copy().groupby(index).sum()
    week_df[target_index] = index.drop_duplicates().to_timestamp()
    week_df.set_index(target_index, drop=True, inplace=True)
    _, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT/2))
    axs[0].grid(False)
    axs[0].set_axis_off()
    axs[0].set_title('HOURLY', fontweight="bold")
    axs[0].text(0, 0, str(data.describe()))
    axs[1].grid(False)
    axs[1].set_axis_off()
    axs[1].set_title('WEEKLY', fontweight="bold")
    axs[1].text(0, 0, str(week_df.describe()))
    savefig(  os.path.join(get_plot_folder_path(), '%s_%s_descriptions' % (name, target_index) ) )

    _, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
    data.boxplot(ax=axs[0])
    week_df.boxplot(ax=axs[1])
    savefig(  os.path.join(get_plot_folder_path(), '%s_%s_boxplot' % (name, target_index) ) )


def var_distribution(data, target_index, target, name):
    bins = (10, 25, 50)
    _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
    for j in range(len(bins)):
        axs[j].set_title(f'Histogram for hourly {target} {bins[j]} bins')
        axs[j].set_xlabel(target)
        axs[j].set_ylabel('Nr records')
        axs[j].hist(data.values, bins=bins[j])
    savefig(  os.path.join(get_plot_folder_path(), '%s_%s_var_distribution' % (name, target_index) ) )


def data_stationary(data, target_var, name):
    dt_series = Series(data[target_var])

    mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
    series = {'ashrae': dt_series, 'mean': mean_line}
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(series, x_label='timestamp', y_label='consumption', title='Stationary study', show_std=True)

    savefig(  os.path.join(get_plot_folder_path(), '%s_target_%s_stationary_study' % (name, target_var) ) )


def data_stationary_2(data, target_var, name):
    BINS = 10
    line = []
    n = len(data)

    dt_series = Series(data[target_var])

    for i in range(BINS):
        b = dt_series[i*n//BINS:(i+1)*n//BINS]
        mean = [b.mean()] * (n//BINS)
        line += mean
    line += [line[-1]] * (n - len(line))
    mean_line = Series(line, index=dt_series.index)
    series = {'ashrae': dt_series, 'mean': mean_line}
    figure(figsize=(3*HEIGHT, HEIGHT))

    plot_series(series, x_label='time', y_label='consumptions', title='Stationary study', show_std=True)
    savefig(  os.path.join(get_plot_folder_path(), '%s_target_%s_stationary_study_2' % (name, target_var) ) )




