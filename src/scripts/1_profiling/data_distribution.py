#!/usr/bin/env python3

import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data

from ds_charts import get_variable_types, choose_grid, HEIGHT, multiple_bar_chart, multiple_line_chart, bar_chart

from numpy import log

from seaborn import distplot

from pandas import read_csv, Series
from pandas.plotting import register_matplotlib_converters

from scipy.stats import norm, expon, lognorm
from matplotlib.pyplot import savefig, show, subplots, figure, Axes

###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
###########################################


def plot_global(data, name):

    data.boxplot(rot=45)
    savefig( os.path.join(get_plot_folder_path(), name) )

    if SHOW_PLOTS:
        show()


def plot_numeric(data, name):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(  os.path.join(get_plot_folder_path(), name) )

    if SHOW_PLOTS:
        show()


def plot_outliers(data, name):
    
    NR_STDEV: int = 2

    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    outliers_iqr = []
    outliers_stdev = []
    summary5 = data.describe(include='number')

    for var in numeric_vars:
        iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
        outliers_iqr += [
            data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
            data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
        std = NR_STDEV * summary5[var]['std']
        outliers_stdev += [
            data[data[var] > summary5[var]['mean'] + std].count()[var] +
            data[data[var] < summary5[var]['mean'] - std].count()[var]]

    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
    
    savefig(  os.path.join(get_plot_folder_path(), name) )

    if SHOW_PLOTS:
        show()


def plot_numeric_hist(data, name):

    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    
    savefig(  os.path.join(get_plot_folder_path(), name) )

    if SHOW_PLOTS:
        show()


def plot_displot(data, name): 
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
        distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    
    savefig(  os.path.join(get_plot_folder_path(), name) )

    if SHOW_PLOTS:
        show()


def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

def plot_bestfit(data, name):

    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    print(len(numeric_vars))
    for n in range(len(numeric_vars)):
        print(n)
        histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    
    savefig(  os.path.join(get_plot_folder_path(), name) )

    if SHOW_PLOTS:
        show()


def plot_symbolic_vals(data, name):
    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = data[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    
    savefig(  os.path.join(get_plot_folder_path(), name) )

    if SHOW_PLOTS:
        show()


if __name__ == "__main__":

    data_diabetic, data_drought = read_data()

    """ plot_global(data_diabetic, 'dist_diabetic_global_box' )
    plot_numeric(data_diabetic, 'dist_diabetic_numeric_box' )
    plot_outliers(data_diabetic, 'dist_diabetic_outlier' )
    plot_numeric_hist(data_diabetic, 'dist_diabetic_numeric_hist' )
    plot_displot(data_diabetic, 'dist_diabetic_distplot' )
    plot_bestfit(data_diabetic, 'dist_diabetic_bestfit' ) 
    plot_symbolic_vals(data_diabetic, 'dist_diabetic_symbol_vals' ) """

    print("diabetic finished")

    # plot_global(data_drought, 'dist_drought_global_box' )
    plot_numeric(data_drought, 'dist_drought_numeric_box' )
    plot_outliers(data_drought, 'dist_drought_outlier' )
    plot_numeric_hist(data_drought, 'dist_drought_numeric_hist' )
    plot_displot(data_drought, 'dist_drought_distplot' )
    plot_bestfit(data_drought, 'dist_drought_bestfit' )
    plot_symbolic_vals(data_drought, 'dist_drought_symbol_vals' )

    print("drought finished")

