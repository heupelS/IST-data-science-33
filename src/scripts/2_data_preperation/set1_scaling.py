#!/usr/bin/env python3

import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data_by_filename, save_new_csv

from scaling import std_scaler_z_score, std_scaler_minmax, scale_boxplot

from outlier_handling import replace_outliers, drop_outliers, truncate_outliers

from knn import KNN
from naive_bayes import NB


###########################################
## Select if plots show up of just saved ##
FILENAME_LOAD = 'diabetic_mv_deleted_rows.csv'
FILENAME = 'diabetic_drop_outliers.csv'
RUN_EVALUATION = True
###########################################


def scaling(df):
    zscore = std_scaler_z_score(df, 'diabetic_std_scaler_z_score.csv')
    minmax = std_scaler_minmax(df, 'diabetic_std_scaler_z_minmax.csv')

    # scale_boxplot(df, zscore, minmax, 'std_scaler_comparison', False)

    if RUN_EVALUATION:

        #  [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        nvalues = [17]
        # ['manhattan', 'euclidean', 'chebyshev']
        dist = ['manhattan']
        
        # Evaluation
        NB(zscore.copy(), 'readmitted', 'diabetic_nb_scale_zscore')
        NB(minmax.copy(), 'readmitted', 'diabetic_nb_scale_minmax')
        
        KNN(zscore.copy(), 'readmitted', 'diabetic_knn_scale_zscore', nvalues=nvalues, dist=dist)
        KNN(minmax.copy(), 'readmitted', 'diabetic_knn_scale_minmax', nvalues=nvalues, dist=dist)


def handle_outliers(data):
    drop_outliers(data.copy(), FILENAME)


if __name__ == "__main__":

    data_drought = read_data_by_filename(FILENAME_LOAD)
    handle_outliers(data_drought)

    data_drought_out = read_data_by_filename(FILENAME_LOAD)

    # Scaling encoded data with handled missing values and outliers
    scaling(data_drought_out)
