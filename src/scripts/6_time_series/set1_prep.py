import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv

from data_profiling.granuality import data_granularity

from time_series.ts_profiling import  box_plot, var_distribution, data_stationary
from time_series.ts_transformation import plot_smoothing, plot_differention, plot_aggregate_multi

from set_procedures import set_forecast


def final_set_forecasting_glucose():
    data_set1 = read_time_series_by_filename('glucose.csv', 'Date')
    data_set1_dropna = data_set1.dropna(axis=0)
    data_set1_sorted = data_set1_dropna.sort_values(by=['Date'])
    return data_set1_sorted


def set1_profiling(data_set1):
    data_granularity(data_set1, 'glucose_gran', 'Numeric')

    targets = ['Glucose', 'Insulin']
    agg_types = ['H', 'D', 'W', 'M', 'Q']

    for agg in agg_types:
        plot_aggregate_multi(
            data=data_set1, 
            agg_type=agg, 
            index_multi='Date', 
            targets=targets, 
            name='glucose', 
            y_label='Consumption')

    box_plot(data_set1, 'Date', 'glucose')
    var_distribution(data_set1, 'Date', 'glucose')
    data_stationary(data_set1, 'Glucose', 'glucose')


def set1_transformation(data_set1):
    targets = ['Glucose', 'Insulin']

    plot_smoothing(data_set1.copy(), targets, 'glucose', 100, x_label='Date', y_label='Consumption')
    plot_differention(data_set1.copy(), targets, 'glucose', x_label='Date', y_label='Consumption')


def set1_forecast(data_set1):

    show_in_plots = ['Glucose']
    target = 'Glucose' 
    target_index = 'Date' 
    agg_types = ['H', 'D', 'W', 'M', 'Q']
    win_sizes = [10, 20, 50, 100]
    filename = 'glucose'

    set_forecast(data_set1, target, target_index, agg_types, win_sizes, show_in_plots, filename)


if __name__ == '__main__':
    data_set1 = final_set_forecasting_glucose()

    set1_profiling(data_set1)
    set1_transformation(data_set1)
    set1_forecast(data_set1)

    

    


