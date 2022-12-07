import sys,os

import pandas as pd
from pandas import DataFrame
from pandas import concat, DataFrame
from pandas import read_csv

from matplotlib.pyplot import figure, savefig, show


sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import save_new_csv, read_data, load_diabetic_data, load_drought_data
from ds_charts import bar_chart

from general_utils import get_plot_folder_path


###########################################
## Select if plots show up of just saved ##
SHOW_PLOTS = True
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

    save_new_csv(df_under, new_file_name)

    values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
    print('Minority class=', positive_class, ':', len(df_positives))
    print('Majority class=', negative_class, ':', len(df_neg_sample))
    print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')


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

    save_new_csv(df_over, new_file_name)

    values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
    print('Minority class=', positive_class, ':', len(df_pos_sample))
    print('Majority class=', negative_class, ':', len(df_negatives))
    print('Proportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': 1')


if __name__ == "__main__":
    diabeteic_data, _ = read_data()
    target = 'readmitted'
    plot_dataset_balance(diabeteic_data, target, 'diabetic_dataset')
    