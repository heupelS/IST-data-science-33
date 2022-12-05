from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from ds_charts import get_variable_types
from numpy import nan

# Data should be encoded and inlcude only numeric variables!
# strategy: mean / median 
def filling_missing_value(df, filename, strategy):

    imp = SimpleImputer(strategy=strategy, fill_value=False, missing_values=nan, copy=True)
    data = imp.fit_transform(df)

    save_new_csv(df, filename)
    print(df.describe())


def drop_missing_records(df, filename):
    # defines the number of variables to discard entire records
    threshold = data.shape[1] * 0.50

    df = data.dropna(thresh=threshold, inplace=False)

    save_new_csv(df, filename)
    print(df.shape)


def drop_missing_values_cols(df, filename):
    # defines the number of records to discard entire columns
    threshold = data.shape[0] * 0.90

    missings = [c for c in mv.keys() if mv[c]>threshold]
    df = data.drop(columns=missings, inplace=False)

    save_new_csv(df, filename)
    print('Dropped variables', missings)

