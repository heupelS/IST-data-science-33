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
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    figure()
    bar_chart(target_count.index, target_count.values, title='Class balance')
    savefig(os.path.join(get_plot_folder_path(), '%s_Class_balance' % save_file_name) )
    
    if SHOW_PLOTS:
        show()




def undersample_dataset(df: DataFrame, target_var: str, new_file_name: str):
    class_var = target_var

    target_count = df[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = df[df[class_var] == positive_class]
    df_negatives = df[df[class_var] == negative_class]

    values = {'Original': [target_count[positive_class], target_count[negative_class]]}
    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_under = concat([df_positives, df_neg_sample], axis=0)

    save_new_csv(df_under, '%s_undersampled.csv' % new_file_name)

    values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
    print('Minority class=', positive_class, ':', len(df_positives))
    print('Majority class=', negative_class, ':', len(df_neg_sample))
    print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')

    plot_dataset_balance(df_under, target_var, '%s_undersampled'% new_file_name)


def oversample_dataset(df: DataFrame, target_var: str, new_file_name: str):
    class_var = target_var

    target_count = df[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = df[df[class_var] == positive_class]
    df_negatives = df[df[class_var] == negative_class]

    values = {'Original': [target_count[positive_class], target_count[negative_class]]}
    df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives], axis=0)

    save_new_csv(df_over, '%s_oversampled.csv' % new_file_name)

    values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
    print('Minority class=', positive_class, ':', len(df_pos_sample))
    print('Majority class=', negative_class, ':', len(df_negatives))
    print('Proportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': 1')

def smote_dataset(df: DataFrame, target_var: str, new_file_name: str):
    class_var = target_var

    target_count = df[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = df[df[class_var] == positive_class]
    df_negatives = df[df[class_var] == negative_class]

    values = {'Original': [target_count[positive_class], target_count[negative_class]]}
    
    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = original.pop(class_var).values
    X = original.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(original.columns) + [class_var]

    save_new_csv(df_smote, '%s_smoted.csv' % new_file_name)

    smote_target_count = Series(smote_y).value_counts()

    values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
    print('Minority class=', positive_class, ':', smote_target_count[positive_class])
    print('Majority class=', negative_class, ':', smote_target_count[negative_class])
    print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')


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