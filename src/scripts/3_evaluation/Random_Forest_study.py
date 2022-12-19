import sys, os

import pandas as pd

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )

from load_data import read_data_by_filename, save_new_csv

from evaluation.Random_Forest import Random_Forest

###########################################
## Select if plots show up of just saved ##
FILENAME = 'drought_dataset_undersampled.csv'
main_name =FILENAME.split("_")[0]
filename = FILENAME.split(".")[0]
###########################################


def evaluating_dataset(data,target):

    Random_Forest(data.copy(), target, '{filename}_DT')


if __name__ == "__main__":
    data = read_data_by_filename(FILENAME)

    if main_name == 'drought':
        target = 'drought'
    elif main_name == 'diabetic':
        target = 'readmitted'
    else:
        print("Your dataset doesnt match with diabetic or drought..Try again with onde of them!")
        exit(0)

    # Evaluating dataset
    evaluating_dataset(data,target)