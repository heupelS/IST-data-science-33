from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame, concat

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from load_data import save_new_csv


def std_scaler_z_score(df, filename):
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df)
    norm_data_zscore = transf.transform(df)

    save_new_csv(norm_data_zscore, filename)
    print(norm_data_zscore.describe())


def std_scaler_z_score(df, filename):
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df)
    norm_data_minmax = transf.transform(df_nr)
    
    save_new_csv(norm_data_minmax, filename)
    print(norm_data_minmax.describe())

