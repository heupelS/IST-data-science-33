import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv

from data_profiling.granuality import data_granularity
from data_profiling.dimensionality import plot_dim

from time_series.ts_profiling import  box_plot, var_distribution, data_stationary, data_stationary_2
from time_series.ts_transformation import plot_smoothing, plot_differention, plot_aggregate_multi

from set_procedures import set_forecast, agg_forecast, smoothing_forecast, diff_forecast, test_regressor

def final_set_forecasting_glucose():
    data_set1 = read_time_series_by_filename('glucose.csv', 'Date')
    data_set1_dropna = data_set1.dropna(axis=0)
    data_set1_sorted = data_set1_dropna.sort_values(by=['Date'])
    return data_set1_sorted


def set1_profiling(data_set1):

    targets = ['Glucose']
    agg_types = ['D', 'W', 'H']

    for agg in agg_types:
        plot_aggregate_multi(
            data=data_set1, 
            agg_type=agg, 
            index_multi='Date', 
            targets=targets, 
            name='glucose', 
            y_label='Glucose')

    box_plot(data_set1, 'Date', 'glucose')
    var_distribution(data_set1, 'Date', 'Glucose', 'glucose')
    data_stationary(data_set1, 'Glucose', 'glucose')
    data_stationary_2(data_set1, 'Glucose', 'glucose')


def set1_transformation(data_set1):
    targets = ['Glucose', 'Insulin']

    plot_smoothing(data_set1.copy(), targets, 'glucose', 100, x_label='Date', y_label='Consumption')
    plot_differention(data_set1.copy(), targets, 'glucose', x_label='Date', y_label='Consumption')


def set1_forecast(data_set1):

    # Setup for forecasting
    show_in_plots = ['Glucose']
    target = 'Glucose' 
    target_index = 'Date' 
    agg_types = ['H', 'D', 'W', 'M', 'Q']
    # win_sizes = []
    win_sizes = [2, 5, 7, 10, 15, 20, 30, 50, 50, 70, 80, 90, 100, 120, 150, 170, 200, 250 ]
    filename = 'glucose'

    variant = 'persistence'

    # Set full set overview
    # set_forecast(data_set1, target, target_index, agg_types, win_sizes, show_in_plots, filename, 
    #   variant=variant)
    
    test_regressor(data_set1, target, target_index, filename, show_in_plots,
      variant=variant)

    # Sets with condition (winsize/agg)
    agg_forecast(data_set1, target, target_index, agg_types, show_in_plots, filename, 
      variant=variant)

    best_agg = 'H'
    smoothing_forecast(data_set1, target, target_index, best_agg, win_sizes, show_in_plots, filename, 
      variant=variant)

    best_win = 150
    derivative = 1
    diff_forecast(data_set1, target, target_index, best_agg, best_win, show_in_plots, filename, 
        derivative=derivative, variant=variant)


if __name__ == '__main__':
    data_set1 = final_set_forecasting_glucose()[['Glucose']]

    # Profiling of dataset
    set1_profiling(data_set1)

    # Testwise transformation
    set1_transformation(data_set1)
    
    # Forecast
    set1_forecast(data_set1)

    

    


