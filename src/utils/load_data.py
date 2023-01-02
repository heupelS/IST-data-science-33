
import os
import pandas as pd

# Leave Standalone, maybe we have to adjust some parameter 
# for the loading!
def load_diabetic_data():
    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path_ws,'..', 'data','diabetic_data.csv')
        csv = pd.read_csv(data_path)
        return csv

    except Exception as e:
        print(e)

# Leave Standalone, maybe we have to adjust some parameter 
# for the loading!
def load_drought_data():
    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path_ws,'..', 'data','drought.csv')
        csv = pd.read_csv(data_path, parse_dates=['date'], dayfirst=True)
        return csv

    except Exception as e:
        print(e)

def read_data():
    return load_diabetic_data(), load_drought_data()


def read_time_series_by_filename(filename, index_col):
    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path_ws,'..', 'data', filename)
        csv = pd.read_csv(data_path, \
            index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
        return csv

    except Exception as e:
        print(e)


def read_data_by_filename(filename):
    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path_ws,'..', 'data', filename)
        csv = pd.read_csv(data_path)
        return csv

    except Exception as e:
        print(e)


def save_new_csv(data, filename):

    if filename == 'drought.csv' or filename == 'diabetic_data.csv':
        print(f'''Don't overwrite basic data!''')
        return

    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path_ws,'..', 'data', filename)
        csv = data.to_csv(data_path, index=False)
        return csv

    except Exception as e:
        print(e)
