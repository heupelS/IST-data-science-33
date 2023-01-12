import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )

from load_data import read_data_by_filename, save_new_csv

from evaluation.Decision_tree_natan import DT
from evaluation.Random_forest_natan import RF

def evaluating_dataset(data,target):

    DT(data.copy(), target, '%s_DT'% target)
    RF(data.copy(), target, '%s_RF'% target)


if __name__ == "__main__":
    data1 = read_data_by_filename('diabetic_dataset_oversampled.csv')
    data2 = read_data_by_filename('drought_dataset_undersampled.csv')

    # Evaluating dataset
    evaluating_dataset(data1, 'readmitted')
    evaluating_dataset(data2, 'drought')