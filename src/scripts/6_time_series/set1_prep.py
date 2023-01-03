import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv

from data_preperation.missing_values import drop_missing_values_cols
from data_profiling.granuality import data_granularity
from time_series.aggregation import aggregate_multi

def final_set_forecasting_glucose():
    data_set1 = read_time_series_by_filename('glucose.csv', 'Date')
    data_set1_dropna = data_set1.dropna(axis=0)
    data_set1_sorted = data_set1_dropna.sort_values(by=['Date'])
    return data_set1_sorted

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

    

