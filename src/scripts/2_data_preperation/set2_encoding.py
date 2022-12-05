#!/usr/bin/env python3

import sys, os

import pandas as pd
import numpy as np

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data
from knn import KNN

from ds_charts import get_variable_types
from matplotlib.pyplot import savefig, show, figure


def encode_time(df):
    # Transform to unix timestamp
    df['date'] = pd.to_datetime(df['date']).astype(int) / 10**9

def evaluate_drought(df):
    df['drought'] = df['class']
    df.drop('class')

if __name__ == "__main__":

    _, data_drought = read_data()

    encode_time(data_drought)

    # evaluate_drought(data_drought)

    nvalues = [17]
    dist = ['manhattan']
    KNN(data_drought, 'class', 'drought_knn_best_res', nvalues=nvalues, dist=dist)


