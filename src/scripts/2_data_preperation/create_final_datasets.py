import sys, os
sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data
from data_preperation.outlier_handling import drop_outliers
from set2_encoding import encode_time, evaluate_drought
from data_preperation.scaling import std_scaler_minmax


def final_set2():
    _, data_drought = read_data()

    # Encoding
    encode_time(data_drought)
    evaluate_drought(data_drought)

    data_drought = drop_outliers(data_drought.copy(), 'drought_drop_outliers.csv')

    data_drought = std_scaler_minmax(data_drought.copy(), 'drought', 'drought_prep.csv')

if __name__ == '__main__':
    final_set2()
