import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv

from data_preperation.outlier_handling import drop_outliers
from data_preperation.scaling import std_scaler_minmax
from data_profiling.granuality import data_granularity
from time_series.aggregation import aggregate_multi

def final_set_forecasting_glucose():
    data_set1 = read_time_series_by_filename('glucose.csv', 'Date')
    print(data_set1.info())
    drop_outliers(data_set1.copy(), 'glucose_prep.csv')

    return data_set1

if __name__ == '__main__':
    data_set1 = final_set_forecasting_glucose()

    data_granularity(data_set1, 'glucose_gran', 'Numeric')

    targets = ['Glucose', 'Insulin']
    agg_types = ['D', 'W', 'M']

    for agg in agg_types:

        aggregate_multi(
            data=data_set1, 
            agg_type=agg, 
            index_multi='Date', 
            targets=targets, 
            name='glucose', 
            y_label='Consumption')

    


