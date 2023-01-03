import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )
from general_utils import get_plot_folder_path

import pandas as pd
from pandas import Series, DataFrame

from matplotlib.pyplot import figure, xticks, savefig, subplots, show

from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series


class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd =  len(X) * [self.mean]
        return prd


def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test


def forecast(data, target, index_target, name, measure='R2', flag_pct=False):

    train, test = split_dataframe(data, trn_pct=0.75)

    fr_mod = SimpleAvgRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train)
    prd_tst = fr_mod.predict(test)

    eval_results = {}
    eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)

    return train, test, prd_trn, prd_tst


def plot_forecasting(train, test, prd_trn, prd_tst, target, index_target, name):

    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, 
        f'{name}_forecast_simpleAvg_plot.png')

    savefig(  os.path.join(get_plot_folder_path(), f'{name}_forecast_simpleAvg_plot' ) )

    plot_forecasting_series(train, test, prd_trn, prd_tst, 
        f'{name}_forecast_simpleAvg_eval.png', 
        x_label=index_target, y_label=target)

    savefig( os.path.join(get_plot_folder_path(), f'{name}_forecast_simpleAvg_eval' ) )


def calculate_fc_with_plot(data, target, index_target, name, measure='R2', flag_pct=False):
    train, test, prd_trn, prd_tst = forecast(data, target, index_target, name, measure=measure, flag_pct=flag_pct)

    plot_forecasting(
        train, test, prd_trn, prd_tst, target, index_target, name,)