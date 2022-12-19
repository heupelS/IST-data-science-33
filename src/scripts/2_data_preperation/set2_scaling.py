#!/usr/bin/env python3

import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data_by_filename, save_new_csv

from data_preperation.scaling import std_scaler_z_score, std_scaler_minmax, scale_boxplot

from evaluation.knn import KNN, knn_plot_save
from evaluation.naive_bayes import NB


###########################################
## Select if plots show up of just saved ##
FILENAME = 'drought_replace_outliers.csv'
filename = FILENAME.split(".")[0]
RUN_EVALUATION = True
###########################################


def scaling(df):
    zscore = std_scaler_z_score(df.copy(), 'drought', '%s_std_scaler_z_score.csv' % filename)
    minmax = std_scaler_minmax(df.copy(), 'drought', '%s_std_scaler_z_minmax.csv' % filename)

    scale_boxplot(df, zscore, minmax, '%s_std_scaler_comparison'% filename, False)

    if RUN_EVALUATION:

        # Evaluation
        NB(zscore.copy(), 'drought', '%s_nb_scale_zscore'% filename)
        NB(minmax.copy(), 'drought', '%s_nb_scale_minmax'% filename)
        
        predictions_dict, best = KNN(zscore.copy(), 'drought', '%s_knn_scale_zscore'% filename)
        knn_plot_save(zscore.copy(), 'drought', '%s_knn_scale_zscore'% filename, predictions_dict, best)
        predictions_dict, best = KNN(minmax.copy(), 'drought', '%s_knn_scale_minmax'% filename)
        knn_plot_save(minmax.copy(), 'drought', '%s_knn_scale_minmax'% filename, predictions_dict, best)


if __name__ == "__main__":

    data_drought = read_data_by_filename(FILENAME)

    # TODO: WTF is this, where do they came from ?? -> I Solve this problem puting parameter Index = False on to_csv.
    #data_drought.drop(columns=['Unnamed: 0'], inplace=True)
    #data_drought.drop(columns=['Unnamed: 0.1'], inplace=True)

    # Scaling encoded data with handled missing values and outliers
    scaling(data_drought)
