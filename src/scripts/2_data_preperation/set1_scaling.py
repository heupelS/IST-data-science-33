#!/usr/bin/env python3

import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data_by_filename, save_new_csv

from data_preperation.scaling import std_scaler_z_score, std_scaler_minmax, scale_boxplot

from data_preperation.outlier_handling import replace_outliers, drop_outliers, truncate_outliers

from evaluation.knn import KNN, knn_plot_save
from evaluation.naive_bayes import NB


###########################################
## Select if plots show up of just saved ##
FILENAME_LOAD = 'diabetic_mv_deleted_rows_encoded.csv'
FILENAME = FILENAME_LOAD.split(".")[0]
RUN_EVALUATION = True
###########################################


def scaling(df,outliers_mode):
    zscore = std_scaler_z_score(df.copy(), 'readmitted', '%s_std_scaler_z_score.csv'%(FILENAME+outliers_mode))
    minmax = std_scaler_minmax(df.copy(), 'readmitted', '%s_std_scaler_z_minmax.csv'%(FILENAME+outliers_mode))

    # scale_boxplot(df, zscore, minmax, 'std_scaler_comparison', False)

    if RUN_EVALUATION:

        # Evaluation
        NB(zscore.copy(), 'readmitted', '%s_nb_scale_zscore'%(FILENAME+outliers_mode))
        NB(minmax.copy(), 'readmitted', '%s_diabetic_nb_scale_minmax'%(FILENAME+outliers_mode))
        
        predictions_dict, best = KNN(zscore.copy(), 'readmitted', '%s_knn_scale_zscore'%(FILENAME+outliers_mode))
        knn_plot_save(zscore.copy(), 'readmitted', '%s_knn_scale_zscore'%(FILENAME+outliers_mode), predictions_dict, best)
        predictions_dict, best = KNN(minmax.copy(), 'readmitted', '%s_knn_scale_minmax'%(FILENAME+outliers_mode))
        knn_plot_save(minmax.copy(), 'readmitted', '%s_knn_scale_minmax'%(FILENAME+outliers_mode), predictions_dict, best)


def handle_outliers(data):
    drop_outliers(data.copy(), "%s_drop_outliers.csv" % FILENAME)
    replace_outliers(data.copy(), "%s_replace_outliers.csv" % FILENAME)
    truncate_outliers(data.copy(), "%s_truncate_outliers.csv" % FILENAME)


if __name__ == "__main__":

    data_drought = read_data_by_filename(FILENAME_LOAD)
    handle_outliers(data_drought)

    data_drought_out = read_data_by_filename('%s_drop_outliers.csv'% FILENAME)
    data_drought_replace_outliers = read_data_by_filename('%s_replace_outliers.csv'% FILENAME)
    data_drought_truncate_outliers = read_data_by_filename('%s_truncate_outliers.csv'% FILENAME)

    #Problem already solved
    #data_drought.drop(columns=['Unnamed: 0'], inplace=True)

    # Scaling encoded data with handled missing values and outliers
    scaling(data_drought_out,'_drop_outliers')
    scaling(data_drought_replace_outliers,'_replace_outliers')
    scaling(data_drought_truncate_outliers,'_truncate_outliers')
