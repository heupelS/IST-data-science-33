import sys, os

from pandas import DataFrame

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from time_series.ts_transformation import smoothing, differention, aggregate_multi
from time_series.ts_forecasting import forecast, calculate_fc_with_plot


''' 
Forecast function for evaluation all 
approaches for one timeseries

@show_in_plots: Variables for forecasting
@win_size: for smoothing window
@aggregates: for aggregation evaluations window
@target_var: target variable in dataset
@target_index: target index (date/timestamp) in time series
@name: save file name of data
'''
def set_forecast(
    data_set: DataFrame, 
    target_var: str, 
    target_index: str,
    aggregates: list,
    win_sizes: list,
    show_in_plots: list, 
    name: str
    ):

    # Test all regressor
    calculate_fc_with_plot(data_set.copy()[show_in_plots], 
        target_var, target_index, f'{name}_without_tf', 'simple_avg')

    calculate_fc_with_plot(data_set.copy()[show_in_plots], 
        target_var, target_index, f'{name}_without_tf', 'persistence')

    calculate_fc_with_plot(data_set.copy()[show_in_plots], 
        target_var, target_index, f'{name}_without_tf', 'rolling_mean')

    # Test all aggregation methods
    for agg in aggregates:
        df_agg = aggregate_multi(data_set.copy(), target_index, agg)
        df_agg = df_agg.dropna(axis=0)
        calculate_fc_with_plot(df_agg.copy()[show_in_plots], target_var, target_index, 
            f'{name}_agg_{agg}')

    # Test all window sizes
    for ws in win_sizes:
        df_sm = smoothing(data_set.copy(), ws)
        df_sm = df_sm.dropna(axis=0)
        calculate_fc_with_plot(df_sm.copy()[show_in_plots], target_var, target_index, 
            f'{name}_sm_win_{ws}')

    # calculate differention
    df_diff = differention(data_set.copy())
    df_diff = df_diff.dropna(axis=0)
    calculate_fc_with_plot(df_diff.copy()[show_in_plots], target_var, target_index, 
        f'{name}_diff')