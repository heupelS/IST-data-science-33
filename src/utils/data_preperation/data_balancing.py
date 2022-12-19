import sys,os

import pandas as pd
from pandas import DataFrame
from pandas import concat, DataFrame
from pandas import read_csv
from pandas import Series
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from matplotlib.pyplot import figure, savefig, show


sys.path.append( os.path.join(os.path.dirname(__file__), '..') )
from load_data import save_new_csv, read_data, load_diabetic_data, load_drought_data
from ds_charts import bar_chart

from general_utils import get_plot_folder_path


###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = False
RANDOM_STATE = 42
###########################################


def plot_dataset_balance(df: DataFrame, target_var: str, save_file_name: str):
    class_var = target_var
    target_count = df[class_var].value_counts()
    print("Target Count:",target_count)
    # Natan here
    majority_class = target_count.idxmax()
    #print("Majority_class:", majority_class)
    #ind_positive_class = target_count.index.get_loc(positive_class)

    values = {'Original': []}
    for index in target_count.index:
        
        print('class=', index, ':', target_count[index])
        print('Proportion:', round(target_count[index] / target_count[majority_class], 2), ': 1')
        values['Original'].append(target_count[index])
    
    #print(values)

    figure()
    bar_chart(target_count.index, target_count.values, title='Class balance')
    savefig(os.path.join(get_plot_folder_path(), '%s_Class_balance' % save_file_name) )
    print("------------- FINAL BALANCE -------------------")
    
    if SHOW_PLOTS:
        show()

def undersample_dataset(df: DataFrame, target_var: str, new_file_name: str):
    class_var = target_var

    target_count = df[class_var].value_counts()

    minority_class = target_count.idxmin()

    values = {'Original': []}
    df_simple = {}
    for index in target_count.index:
        values['Original'].append(target_count[index])
        df_simple[index] = df[df[class_var] == index]

    df_neg_sample= {}
    values['UnderSample'] = []
    for index in target_count.index:
        df_neg_sample[index] = DataFrame(df_simple[index].sample(target_count[minority_class]))
        values['UnderSample'].append(len(df_neg_sample[index]))
        #print('Proportion:', round(target_count[minority_class] / len(df_neg_sample[index]), 2), ': 1')
    
    df_under = pd.concat(df_neg_sample.values(), ignore_index=True)

    save_new_csv(df_under, '%s_undersampled.csv' % new_file_name)
    print("Original dataset shape:",df.shape)
    print("New dataset shape:",df_under.shape)

    plot_dataset_balance(df_under, target_var, '%s_undersampled'% new_file_name)


def oversample_dataset(df: DataFrame, target_var: str, new_file_name: str):
    class_var = target_var

    target_count = df[class_var].value_counts()

    majority_class = target_count.idxmax()

    values = {'Original': []}
    df_simple = {}
    for index in target_count.index:
        values['Original'].append(target_count[index])
        df_simple[index] = df[df[class_var] == index]

    df_pos_sample= {}
    values['OverSample'] = []
    for index in target_count.index:
        df_pos_sample[index] = DataFrame(df_simple[index].sample(target_count[majority_class], replace = True))
        values['OverSample'].append(len(df_pos_sample[index]))
        #print('Proportion:', round(target_count[majority_class] / len(df_pos_sample[index]), 2), ': 1')
    
    df_over = pd.concat(df_pos_sample.values(), ignore_index=True)

    print("Original dataset shape:",df.shape)
    print("New dataset shape:",df_over.shape)
   
    save_new_csv(df_over, '%s_oversampled.csv' % new_file_name)

    plot_dataset_balance(df_over, target_var, '%s_oversampled'% new_file_name)


def smote_dataset(df: DataFrame, target_var: str, new_file_name: str):
    class_var = target_var

    target_count = df[class_var].value_counts()

    majority_class = target_count.idxmax()

    values = {'Original': []}
    df_simple = {}
    for index in target_count.index:
        values['Original'].append(target_count[index])
        df_simple[index] = df[df[class_var] == index]
    
    
    smote = SMOTE(sampling_strategy='not majority', random_state=RANDOM_STATE)
    y = df.pop(class_var).values
    X = df.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(df.columns) + [class_var]

    print("Original dataset shape:",df.shape)
    print("New dataset shape:",df_smote.shape)

    save_new_csv(df_smote, '%s_smoted.csv' % new_file_name)

    smote_target_count = Series(smote_y).value_counts()

    values = {'SMOTE': []}
    for index in smote_target_count.index:
        values['SMOTE'].append(target_count[index])

    plot_dataset_balance(df_smote, target_var, '%s_smoted'% new_file_name)

def random_undersample_dataset(df: DataFrame, target_var: str, new_file_name: str):
    print('Class variables started at:', df[target_var].value_counts())
    target = df.pop(target_var)
    ros = RandomUnderSampler(random_state=0)
    x, y = ros.fit_resample(df, target)
    df_under = concat([x,y], axis = 1)
    print('Class variables undersampled to:', df_under[target_var].value_counts())
    save_new_csv(df_under, '%s_undersampled.csv' % new_file_name)
    return df_under

def random_oversample_dataset(df: DataFrame, target_var: str, new_file_name: str):
    print('Class variables started at:', df[target_var].value_counts())
    target = df.pop(target_var)
    ros = RandomOverSampler(random_state=0)
    x, y = ros.fit_resample(df, target)
    df_over = concat([x,y], axis = 1)
    print('Class variables oversampled to:', df_over[target_var].value_counts())
    save_new_csv(df_over, '%s_undersampled.csv' % new_file_name)
    return df_over