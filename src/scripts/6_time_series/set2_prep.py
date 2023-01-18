import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv

from data_profiling.granuality import data_granularity
from data_profiling.dimensionality import plot_dim

from time_series.ts_profiling import box_plot, var_distribution, data_stationary, data_stationary_2
from time_series.ts_transformation import plot_smoothing, plot_differention, plot_aggregate_multi

from set_procedures import set_forecast, agg_forecast, smoothing_forecast, diff_forecast, test_regressor


def final_set_forecasting_drought():
    data_set2 = read_time_series_by_filename('drought.forecasting_dataset.csv', 'date')
    data_set2_dropna = data_set2.dropna(axis=0)
    data_set2_sorted = data_set2_dropna.sort_values(by=['date'])
    return data_set2_sorted


def set2_profiling(data_set2):

    # targets = ['PRECTOT', 'PS', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'QV2M']
    targets = ['QV2M']
    agg_types = ['H', 'D', 'W', 'M']

    for agg in agg_types:
        plot_aggregate_multi(
            data=data_set2, 
            agg_type=agg, 
            index_multi='date', 
            targets=targets, 
            name='drought', 
            y_label='QV2M')
    
    box_plot(data_set2, 'date', 'drought')
    var_distribution(data_set2, 'date', 'QV2M', 'drought')
    data_stationary(data_set2, 'QV2M', 'drought')
    data_stationary_2(data_set2, 'QV2M', 'drought')


def set2_transformation(data_set2):
    targets = ['PRECTOT', 'PS', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'QV2M']

    plot_smoothing(data_set2.copy(), targets, 'glucose', 100, 'Date', 'Consumption')
    plot_differention(data_set2.copy(), targets, 'glucose', 100, 'Date', 'Consumption')


def set2_forecast(data_set2):

    show_in_plots = ['QV2M']
    target = 'QV2M' 
    target_index = 'date' 
    agg_types = ['D', 'W', 'M']
    win_sizes = [10, 20, 50, 60, 70, 80, 90, 100, 200, 250, 300, 350]
    filename = 'drought'

    variant = 'persistence'

    # Set full set overview - Just for test purpose -
    # set_forecast(data_set2, target, target_index, agg_types, win_sizes, show_in_plots, filename, 
    #   variant=variant)
    
    test_regressor(data_set2, target, target_index, filename, show_in_plots,
      variant=variant)

    # Sets with condition (winsize/agg)
    agg_forecast(data_set2, target, target_index, agg_types, show_in_plots, filename, 
      variant=variant)

    best_agg = 'M'
    smoothing_forecast(data_set2, target, target_index, best_agg, win_sizes, show_in_plots, filename, 
      variant=variant)

    best_win = 60
    derivative = 1
    diff_forecast(data_set2, target, target_index, best_agg, best_win, show_in_plots, filename, 
        derivative=derivative, variant=variant)

if __name__ == '__main__':
    data_set2 = final_set_forecasting_drought()[['QV2M']]

    set2_profiling(data_set2)
    set2_transformation(data_set2)
    set2_forecast(data_set2)
   


