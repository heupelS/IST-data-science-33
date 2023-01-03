import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv

from data_preperation.outlier_handling import drop_outliers
from data_preperation.scaling import std_scaler_minmax
from data_profiling.granuality import data_granularity
from time_series.aggregation import aggregate_multi

def final_set_forecasting_drought():
    data_set2 = read_time_series_by_filename('drought.forecasting_dataset.csv', 'date')
    data_set2_dropna = data_set2.dropna(axis=0)
    data_set2_sorted = data_set2_dropna.sort_values(by=['date'])
    return data_set2_sorted

if __name__ == '__main__':
    data_set2 = final_set_forecasting_drought()

    data_granularity(data_set2, 'drought_gran', 'Numeric')

    targets = ['PRECTOT', 'PS', 'T2M', 'T2MDEW', 'T2MWET', 'TS', 'QV2M']
    agg_types = ['D', 'W', 'M']

    for agg in agg_types:
        aggregate_multi(
            data=data_set2, 
            agg_type=agg, 
            index_multi='date', 
            targets=targets, 
            name='drought', 
            y_label='Consumption')

    


