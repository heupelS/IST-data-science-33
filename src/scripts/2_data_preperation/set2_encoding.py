#!/usr/bin/env python3

import sys, os

import pandas as pd
import numpy as np

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data, save_new_csv

from knn import KNN
from naive_bayes import NB


###########################################
## Select if plots show up of just saved ##
SAVE_FILENAME = 'drought_encoding.csv'
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

    save_new_csv(data_drought.copy(), SAVE_FILENAME) 
 
    #  [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    nvalues = [17]
    # ['manhattan', 'euclidean', 'chebyshev']
    dist = ['manhattan']

    # Evaluation
    NB(data_drought.copy(), 'drought', 'drought_nb_best_res')
    KNN(data_drought.copy(), 'drought', 'drought_knn_best_res', nvalues=nvalues, dist=dist)
