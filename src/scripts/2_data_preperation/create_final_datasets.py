import sys, os
sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data
from data_preperation.outlier_handling import drop_outliers
from set2_encoding import encode_time, evaluate_drought
from data_preperation.scaling import std_scaler_minmax, std_scaler_z_score
from set1_encoding_evaluate import *


def final_set1():
    data, _ = read_data()

    data_diabetic = drop_id_cols(data)

    data_diabetic = replace_questionmarks(data_diabetic)

    data_diabetic_mv_deleted_rows = drop_missing_records(data_diabetic.copy(), 'data_diabetic_mv_deleted_rows.csv') 

    final_df = encode(data_diabetic_mv_deleted_rows, 'diabetic_mv_deleted_rows_encoded')

    drop_outliers(final_df, 'diabetic_drop_outliers.csv')

    std_scaler_minmax(final_df.copy(), 'readmitted', 'diabetic_minmax.csv')
    std_scaler_z_score(final_df.copy(), 'readmitted', 'diabetic_zscore.csv')

def final_set2():
    _, data_drought = read_data()

    # Encoding
    encode_time(data_drought)
    evaluate_drought(data_drought)

    drop_outliers(data_drought, 'drought_drop_outliers.csv')

    std_scaler_minmax(data_drought, 'drought', 'drought_prep.csv')

if __name__ == '__main__':
    final_set1()
