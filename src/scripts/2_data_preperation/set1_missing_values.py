import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data

from ds_charts import bar_chart, get_variable_types
import pandas as pd
from matplotlib.pyplot import savefig, show, figure
from pandas import DataFrame, read_csv, unique
import numpy as np
from sklearn.preprocessing import LabelBinarizer


from pandas import DataFrame, concat
from ds_charts import get_variable_types
from sklearn.preprocessing import LabelEncoder
from numpy import number

###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
###########################################


def replace_questionmarks(df):
    df.replace(['?'], [None])

def load_diabetic_data():
    """Just for Erik"""
    filename = '/Users/erikspieler/Desktop/NTNU_MASTER/Utveksling/Data science/Project/PythonProject/Classification/Data preperation/dataset/diabetic_data.csv'
    data = read_csv(filename)
    return data   

def change_to_categorical(df):
    cat_vars = df.select_dtypes(include='object')
    df[cat_vars.columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))


def one_hot_encode_target(df):
    #Pop target values from dataset
    target = 'readmitted'
    trnY: np.ndarray = df.pop(target).values

    #One hot encode target values
    lb = LabelBinarizer()
    lb.fit(trnY)
    one_hot_y = lb.transform(trnY)
    #one_hot_y = pd.DataFrame(one_hot_y)
    return one_hot_y


def encode_binary_variables(df):
    variables = get_variable_types(df)
    binary = variables['Binary']

    binary_encoder = LabelEncoder()
    new_binary_variables = []
    for var in binary:
        binary_encoder.fit(df[var])
        new_binary_variables.append(binary_encoder.transform(df[var]))

    #Change order of the class ['change']
    i = 0
    for entry in new_binary_variables[7]:
        if entry == 0:
            new_binary_variables[7][i] = 1
            i += 1
        else:
            new_binary_variables[7][i] = 0
            i += 1

    new_binary_variables_data = pd.DataFrame(new_binary_variables)

    new_binary_variables_data = new_binary_variables_data.T

    i = 0
    while i  < 9:
        new_binary_variables_data.rename(columns = {i: binary[i]}, inplace = True)
        i+=1

    return new_binary_variables_data

def encode_symbolic_variables(df):
    variables = get_variable_types(df)
    symbolic = variables['Symbolic']
    symbolic_encoder = LabelEncoder()
    new_symbolic_variables = []
    for var in symbolic:
        symbolic_encoder.fit(df[var])
        new_symbolic_variables.append(symbolic_encoder.transform(df[var]))

    new_symbolic_variables_data = pd.DataFrame(new_symbolic_variables)
    new_symbolic_variables_data = new_symbolic_variables_data.T

    i = 0
    while i  < 27:
        new_symbolic_variables_data.rename(columns = {i: symbolic[i]}, inplace = True)
        i+=1
        
    return new_symbolic_variables_data


def concat_encoded_data(binary,symbolic, numeric):
    trn_X = concat([binary, symbolic, numeric], axis = 1)
    return trn_X

def get_numeric_data(df):
    variables = get_variable_types(df)
    numeric = variables['Numeric']
    numeric_data = df[numeric]
    return numeric_data



if __name__ == "__main__":

    data_diabetic = load_diabetic_data()
    replace_questionmarks(data_diabetic)
    change_to_categorical(data_diabetic)
    trn_y = one_hot_encode_target(data_diabetic)
    binary_encoded_data = encode_binary_variables(data_diabetic)
    symbolic_encoded_data = encode_symbolic_variables(data_diabetic)
    numeric_data = get_numeric_data(data_diabetic)
    trn_X = concat_encoded_data(binary_encoded_data,symbolic_encoded_data, numeric_data)
    trn_X.to_csv('diabetic_X.csv')
    #trn_y.to_csv('diabetic_y.csv')






