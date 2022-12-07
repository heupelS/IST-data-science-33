import sys, os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame, concat

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import save_new_csv

from ds_charts import get_variable_types

from matplotlib.pyplot import subplots, show, savefig


def std_scaler_z_score(df, filename):

    variables = get_variable_types(df)

    numeric = variables['Numeric']
    binary = variables['Binary']

    numeric_data = df[numeric]
    binary_data = df[binary]

    target = numeric_data.pop('readmitted') 

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(numeric_data)
    scaled_data = DataFrame(transf.transform(numeric_data), index=numeric_data.index, columns=numeric_data.columns)

    norm_data_zscore = concat([scaled_data, binary_data, target], axis = 1)

    save_new_csv(norm_data_zscore, filename)
    print(norm_data_zscore.describe())

    return norm_data_zscore


def std_scaler_minmax(df, filename):

    variables = get_variable_types(df)

    numeric = variables['Numeric']
    binary = variables['Binary']

    numeric_data = df[numeric]
    target = numeric_data.pop('readmitted') 

    binary_data = df[binary]

    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(numeric_data)
    scaled_values = DataFrame(transf.transform(numeric_data), index=numeric_data.index, columns=numeric_data.columns)
    
    norm_data_minmax = concat([scaled_values, binary_data, target], axis = 1)

    save_new_csv(norm_data_minmax, filename)
    print(norm_data_minmax.describe())

    return norm_data_minmax


def scale_boxplot(df, df_zscore, df_minmax, filename, showplot=False):
    
    fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
    
    axs[0, 0].set_title('Original data')
    df.boxplot(ax=axs[0, 0])
    
    axs[0, 1].set_title('Z-score normalization')
    df_zscore.boxplot(ax=axs[0, 1])
    
    axs[0, 2].set_title('MinMax normalization')
    df_minmax.boxplot(ax=axs[0, 2])
    
    savefig( os.path.join(get_plot_folder_path(), filename) )

    if showplot:
        show()

