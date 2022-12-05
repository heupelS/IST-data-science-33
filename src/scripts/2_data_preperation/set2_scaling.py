#!/usr/bin/env python3

import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data_by_filename, save_new_csv

###########################################
## Select if plots show up of just saved ##
FILENAME = 'drought_truncate_outliers.csv'
###########################################


def scaling(df):
    pass


if __name__ == "__main__":

    data_drought = read_data_by_filename(FILENAME)

    print(data_drought.shape)

    # Encoding
    scaling(data_drought)
