#!/usr/bin/env python3

import sys, os

import pandas as pd
import numpy as np

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data
from knn import KNN
from naive_bayes import NB

from ds_charts import get_variable_types
from matplotlib.pyplot import savefig, show, figure


###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
###########################################


def encode_time(df):
    # Transform to unix timestamp
    df['date'] = pd.to_datetime(df['date']).astype(int) / 10**9


def evaluate_drought(df):
    df['drought'] = df['class']
    df.drop(['class'], axis=1)


if __name__ == "__main__":

    _, data_drought = read_data()

    # Encoding
    encode_time(data_drought)
    evaluate_drought(data_drought)

    #  [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    nvalues = [17]
    # ['manhattan', 'euclidean', 'chebyshev']
    dist = ['manhattan']

    # Evaluation
    NB(data_drought.copy(), 'drought', 'drought_nb_best_res')
    KNN(data_drought.copy(), 'drought', 'drought_knn_best_res', nvalues=nvalues, dist=dist)
