import sys,os

import pandas as pd
sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )

from data_preperation.data_balancing import plot_dataset_balance, undersample_dataset, oversample_dataset, smote_dataset

from load_data import read_data_by_filename



###########################################
## Select if plots show up or just saved ##
FILENAME = 'drought_replace_outliers_std_scaler_z_minmax.csv'
filename = FILENAME.split("_")[0]+'_dataset'
###########################################


def balancing_dataset(data, target):

    plot_dataset_balance(data, target, filename)

    # I think for this dataset is the better approach because we have 60000 records and the data is close to be total balanced. 29382 vs 30047
    undersample_dataset(data, target, filename)

    #oversample_dataset(data, target, filename)
    #smote_dataset(data, target, filename)


if __name__ == "__main__":
    data_drought = read_data_by_filename(FILENAME)

    # Balancing
    balancing_dataset(data_drought,'drought')
    
