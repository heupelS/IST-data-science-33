#!/usr/bin/env python3

import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data_by_filename, save_new_csv

from scaling import std_scaler_z_score, std_scaler_minmax, scale_boxplot

from evaluation.knn import KNN
from evaluation.naive_bayes import NB


###########################################
## Select if plots show up of just saved ##
FILENAME = 'drought_drop_outliers.csv'
RUN_EVALUATION = True
###########################################


def scaling(df):
    zscore = std_scaler_z_score(df, 'std_scaler_z_score.csv')
    minmax = std_scaler_minmax(df, 'std_scaler_z_minmax.csv')

    scale_boxplot(df, zscore, minmax, 'std_scaler_comparison', False)

    if RUN_EVALUATION:

        # Evaluation
        NB(zscore.copy(), 'drought', 'drought_nb_scale_zscore')
        NB(minmax.copy(), 'drought', 'drought_nb_scale_minmax')
        
        KNN(zscore.copy(), 'drought', 'drought_knn_scale_zscore')
        KNN(minmax.copy(), 'drought', 'drought_knn_scale_minmax')


if __name__ == "__main__":

    data_drought = read_data_by_filename(FILENAME)

    # TODO: WTF is this, where do they came from ??
    data_drought.drop(columns=['Unnamed: 0'], inplace=True)
    data_drought.drop(columns=['Unnamed: 0.1'], inplace=True)

    # Scaling encoded data with handled missing values and outliers
    scaling(data_drought)
