import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv

from data_profiling.granuality import data_granularity

from time_series.ts_profiling import aggregate_multi, box_plot, var_distribution, data_stationary
from time_series.ts_transformation import smoothing, differention
from time_series.ts_forecasting import forecast

def final_set_forecasting_drought():
    data_set2 = read_time_series_by_filename('drought.forecasting_dataset.csv', 'date')
    data_set2_dropna = data_set2.dropna(axis=0)
    data_set2_sorted = data_set2_dropna.sort_values(by=['date'])
    return data_set2_sorted


def set1_profiling(data_set2):

    data_granularity(data_set2, 'drought_gran', 'Numeric')

    targets = ['PRECTOT', 'PS', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'QV2M']
    agg_types = ['S', 'H', 'D', 'W', 'M', 'Q']

    for agg in agg_types:
        aggregate_multi(
            data=data_set2, 
            agg_type=agg, 
            index_multi='date', 
            targets=targets, 
            name='drought', 
            y_label='Consumption')
    
    box_plot(data_set2, 'date', 'drought')
    var_distribution(data_set2, 'date', 'drought')
    data_stationary(data_set2, 'QV2M', 'drought')


def set1_transformation(data_set2):
    targets = ['PRECTOT', 'PS', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'QV2M']

    smoothing(data_set2.copy(), targets, 'glucose', 100, 'Date', 'Consumption')
    differention(data_set2.copy(), targets, 'glucose', 100, 'Date', 'Consumption')


if __name__ == '__main__':
    data_set2 = final_set_forecasting_drought()

    set1_profiling(data_set2)
    set1_transformation(data_set2)
   


