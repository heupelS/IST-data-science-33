import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import read_data, save_new_csv
from ds_charts import bar_chart, get_variable_types
from missing_values import *

from knn import KNN
from naive_bayes import NB

import pandas as pd
from pandas import DataFrame, read_csv, unique, concat

from matplotlib.pyplot import savefig, show, figure
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import numpy as np
from numpy import number


def replace_questionmarks(df):
    df = df.replace(['?'], np.nan)
    return df


def load_diabetic_data():
    """Just for Erik"""
    filename = '/Users/erikspieler/Desktop/NTNU_MASTER/Utveksling/Data science/Project/PythonProject/Classification/Data preperation/dataset/diabetic_data.csv'
    data = read_csv(filename)
    return data   


def drop_id_cols(df):
    df = df.drop(columns = ['encounter_id','patient_nbr'])
    return df


def change_to_categorical(df):
    cat_vars = df.select_dtypes(include='object')
    df[cat_vars.columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))


def encode_binary_variables(df):
    variables = get_variable_types(df)
    binary = variables['Binary']

    df_bin = pd.DataFrame()
    binary_encoder = LabelEncoder()
    for var in binary:
        binary_encoder.fit(df[var])
        df_bin[var] = binary_encoder.transform(df[var])

    return df_bin


def encode_symbolic_variables(df):
    variables = get_variable_types(df)
    symbolic = variables['Symbolic']
    symbolic_encoder = LabelEncoder()

    df_sym = pd.DataFrame()
    for var in symbolic:
        symbolic_encoder.fit(df[var])
        df_sym[var] = symbolic_encoder.transform(df[var])

    return df_sym


def get_numeric_data(df):
    variables = get_variable_types(df)
    numeric = variables['Numeric']
    numeric_data = df[numeric]
    return numeric_data


def concat_encoded_data(binary, symbolic, numeric):
    final_df = concat([binary, symbolic, numeric], axis = 1)
    return final_df


def encode_and_evaluate(df, filename):

    change_to_categorical(df)

    binary_encoded_data = encode_binary_variables(df)
    symbolic_encoded_data = encode_symbolic_variables(df)
    numeric_data = get_numeric_data(df)

    final_df = concat_encoded_data(binary_encoded_data, symbolic_encoded_data, numeric_data)

    save_new_csv(final_df, filename)

    #  [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    nvalues = [17]
    # ['manhattan', 'euclidean', 'chebyshev']
    dist = ['manhattan']

    # Evaluation
    # NB(final_df.copy(), 'readmitted', '%s_nb_best_res' % filename)
    KNN(final_df.copy(), 'readmitted', '%s_knn_best_res' % filename, nvalues=nvalues, dist=dist)


if __name__ == "__main__":

    data_diabetic, _ = read_data() 
    data_diabetic = drop_id_cols(data_diabetic)

    data_diabetic = replace_questionmarks(data_diabetic)
    
    data_diabetic_mv_filled = filling_missing_value_most_frequent(data_diabetic.copy(),'data_diabetic_mv_most_frequent')
    data_diabetic_mv_deleted_rows = drop_missing_records(data_diabetic.copy(), 'data_diabetic_mv_deleted_rows')

    encode_and_evaluate(data_diabetic_mv_filled, 'diabetic_mv_most_frequent')
    encode_and_evaluate(data_diabetic_mv_deleted_rows, 'diabetic_mv_deleted_rows')
    