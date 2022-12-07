from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from numpy import nan
import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import save_new_csv

from ds_charts import get_variable_types


def get_na_distribution_dict(df):
    mv = {}
    for var in df:
        nr = df[var].isna().sum()
        if nr > 0:
            mv[var] = nr
    return mv

def filling_missing_value_most_frequent(df, filename):

    variables = get_variable_types(df)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']

    tmp_nr, tmp_sb, tmp_bool = None, None, None
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(df[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_sb = DataFrame(imp.fit_transform(df[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(df[binary_vars]), columns=binary_vars)

    df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)  # type: ignore
    df.index = df.index

    save_new_csv(df, filename)
    print(df.describe())
    return df


def drop_missing_records(df, filename):
    # defines the number of variables to discard entire records
    threshold = df.shape[1] * 0.50

    df = df.dropna(thresh=threshold, inplace=False)

    save_new_csv(df, filename)
    print(df.shape)
    return df


def drop_missing_values_cols(df, filename):
    # defines the number of records to discard entire columns
    threshold = df.shape[0] * 0.90
    mv = get_na_distribution_dict(df)
    missings = [c for c in mv.keys() if mv[c]>threshold]
    df = df.drop(columns=missings, inplace=False)

    save_new_csv(df, filename)
    print('Dropped variables', missings)

