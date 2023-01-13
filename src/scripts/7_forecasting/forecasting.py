import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_time_series_by_filename, save_new_csv
from general_utils import get_plot_folder_path

from time_series.ts_forecasting import split_dataframe, calculate_fc_with_plot
from time_series.ts_arima import arima_forecast, find_arima_parameter, arima_plot_diagnostics
from time_series.ts_lstm import lstm_train, lstm_forecast, lstm_plot_diagnostics

from time_series.ts_transformation import smoothing, differention, aggregate_multi

from matplotlib.pyplot import show, savefig

from statsmodels.tsa.arima.model import ARIMA

from pandas import DataFrame

from dsLSTM import DS_LSTM
from torch import save as save_model
from torch import load as load_model

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


def final_arima_forecast(
    dataset: DataFrame, 
    index: str,
    target: str,
    freq:str,
    file_tag: str,
    exe_training: bool=True):

    train, test = split_dataframe(dataset, trn_pct=0.75)

    if exe_training:
        order, model = find_arima_parameter(train, test, index, target, freq, file_tag)
        savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_arima_params' ) )
        # save_model(model.state_dict(), f'./model_{file_tag}_arima.model')
    else:
        order = (2, 2, 2)
        # model = ARIMA(train, order=order)
        # model.load_state_dict(load_model(f'./model_{file_tag}_arima.model'))
    
    arima_plot_diagnostics(train, order)
    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_arima_diagnostic' ) )

    arima_forecast(train, test, index, target, freq, model, file_tag)
    savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_arima_forecast' ) )



def final_lstm_forecast(
    dataset: DataFrame, 
    index: str,
    target: str,
    freq:str,
    file_tag: str,
    exe_training: bool=True):

    train, test = split_dataframe(dataset, trn_pct=0.75)

    nr_features = len(dataset.columns)
    length=20
    learning_rate = 0.001
    hidden_units = 32 

    if exe_training:
        lenght, model = lstm_train(train, test, index, target, nr_features, file_tag)
        savefig( os.path.join(get_plot_folder_path(), f'{file_tag}_lstm_params' ) )
        # save_model(model.state_dict(), f'./model_{file_tag}_lstm.model')
    else:
        try:
            # take from last eval
            model = DS_LSTM(input_size=nr_features, hidden_size=hidden_units, learning_rate=learning_rate)
            model.load_state_dict(load_model(f'./model_{file_tag}_lstm.model'))
        except Exception as e:
            print(e)
            raise Exception('No model exists. Pleas run with train=True !')

    lstm_plot_diagnostics(nr_features, hidden_units, learning_rate, file_tag)

    lstm_forecast(train, test, index, target, length, model, file_tag)


def rollong_mean(data_set, target_var, target_index):
    calculate_fc_with_plot(data_set.copy(), 
        target_var, target_index, f'{target_var}', 'rolling_mean')


def simple_avg(data_set, target_var, target_index):
    calculate_fc_with_plot(data_set.copy(), 
        target_var, target_index, f'{target_var}', 'simple_avg')


if __name__ == '__main__':
    data_set1 = final_set_forecasting_glucose()
    data_set2 = final_set_forecasting_drought()

    # rollong_mean(data_set1, 'Glucose', 'Date')
    # rollong_mean(data_set2, 'QV2M', 'date')

    # simple_avg(data_set1, 'Glucose', 'Date')
    # simple_avg(data_set2, 'QV2M', 'date')

    # final_arima_forecast(data_set1, 'Date', 'glucose', 'H', 'glucose', exe_training=True)
    # final_arima_forecast(data_set2, 'date', 'QV2M', 'H', 'drought', exe_training=True)
    
    for column in data_set1:
        final_lstm_forecast(data_set1, 'Date', 'glucose', 'H', 'glucose', exe_training=True)

    # for column in data_set2:
    #     final_lstm_forecast(data_set2, 'date', 'QV2M', 'H', 'drought', exe_training=True)


    


    


