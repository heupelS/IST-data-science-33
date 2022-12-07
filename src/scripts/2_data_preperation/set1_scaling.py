#!/usr/bin/env python3

import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data_by_filename, save_new_csv

from data_preperation.scaling import std_scaler_z_score, std_scaler_minmax, scale_boxplot

from data_preperation.outlier_handling import replace_outliers, drop_outliers, truncate_outliers

from evaluation.knn import KNN
from evaluation.naive_bayes import NB


###########################################
## Select if plots show up of just saved ##
FILENAME_LOAD = 'diabetic_mv_deleted_rows_encoded.csv'
FILENAME = 'diabetic_drop_outliers.csv'
RUN_EVALUATION = False
###########################################


def scaling(df):
    zscore = std_scaler_z_score(df.copy(), 'readmitted', 'diabetic_std_scaler_z_score.csv')
    minmax = std_scaler_minmax(df.copy(), 'readmitted', 'diabetic_std_scaler_z_minmax.csv')

    # scale_boxplot(df, zscore, minmax, 'std_scaler_comparison', False)

    if RUN_EVALUATION:

        # Evaluation
        NB(zscore.copy(), 'readmitted', 'diabetic_nb_scale_zscore')
        NB(minmax.copy(), 'readmitted', 'diabetic_nb_scale_minmax')
        
        KNN(zscore.copy(), 'readmitted', 'diabetic_knn_scale_zscore')
        KNN(minmax.copy(), 'readmitted', 'diabetic_knn_scale_minmax')


def handle_outliers(data):
    drop_outliers(data.copy(), FILENAME)


if __name__ == "__main__":

    data_drought = read_data_by_filename(FILENAME_LOAD)
    handle_outliers(data_drought)

    data_drought_out = read_data_by_filename(FILENAME)

    data_drought.drop(columns=['Unnamed: 0'], inplace=True)

    # Scaling encoded data with handled missing values and outliers
    scaling(data_drought_out)
