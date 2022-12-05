#!/usr/bin/env python3

import sys, os

from pandas import DataFrame
import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import save_new_csv, read_data_by_filename

from outlier_handling import replace_outliers, drop_outliers, truncate_outliers


###########################################
## Select if plots show up of just saved ##
FILENAME = 'drought_encoding.csv'
###########################################


def handle_outliers(data):
    drop_outliers(data.copy(), 'drought_drop_outliers.csv')
    replace_outliers(data.copy(), 'drought_replace_outliers.csv')
    truncate_outliers(data.copy(), 'drought_truncate_outliers.csv')


if __name__ == "__main__":


    data_drought = read_data_by_filename(FILENAME)

    # Encoding
    handle_outliers(data_drought)
