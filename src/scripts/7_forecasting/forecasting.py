import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv

from time_series.ts_forecasting import split_dataframe
from time_series.ts_arima import arima_forecast, find_arima_parameter
from time_series.ts_lstm import ltsm_train, lsmt_forecast

from pandas import DataFrame


def final_set_forecasting_glucose():
    data_set1 = read_time_series_by_filename('glucose.csv', 'Date')
    data_set1_dropna = data_set1.dropna(axis=0)
    data_set1_sorted = data_set1_dropna.sort_values(by=['Date'])
    return data_set1_sorted


def final_set_forecasting_drought():
    data_set2 = read_time_series_by_filename('drought.forecasting_dataset.csv', 'date')
    data_set2_dropna = data_set2.dropna(axis=0)
    data_set2_sorted = data_set2_dropna.sort_values(by=['date'])
    return data_set2_sorted


def arima_forecast(
    dataset: DataFrame, 
    index: str,
    target: str,
    freq:str,
    file_tag: str):

    train, test = split_dataframe(dataset, trn_pct=0.75)

    order = (2, 2, 2)
    order = find_arima_parameter(train, test, index, target, freq, file_tag)

    arima_forecast(train, index, target, freq, order, file_tag)


def lstm_forecast(
    dataset: DataFrame, 
    index: str,
    target: str,
    freq:str,
    file_tag: str):

    train, test = split_dataframe(dataset, trn_pct=0.75)

    nr_features = len(dataset.columns)

    order = (2, 2, 2)
    order = ltsm_train(train, test, index, target, nr_features, file_tag)
    lstm_forecast(train, index, target, nr_features, order, file_tag)


if __name__ == '__main__':
    data_set1 = final_set_forecasting_glucose()[['Glucose']]
    data_set2 = final_set_forecasting_drought()[['QV2M']]

    arima_forecast(data_set1, 'Date', 'glucose', 'H', 'glucose')
    arima_forecast(data_set2, 'date', 'QV2M', 'H', 'drought')
    
    lstm_forecast(data_set1, 'Date', 'glucose', 'H', 'glucose')
    lstm_forecast(data_set2, 'date', 'QV2M', 'H', 'drought')

    

    


