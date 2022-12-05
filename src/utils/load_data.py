
import os
import pandas as pd

def load_diabetic_data():
    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path_ws,'..', 'data','diabetic_data.csv')
        csv = pd.read_csv(data_path, parse_dates=['date'])
        return csv

    except Exception as e:
        print(e)


def load_drought_data():
    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path_ws,'..', 'data','drought.csv')
        csv = pd.read_csv(data_path, parse_dates=['date'])
        return csv

    except Exception as e:
        print(e)

def read_data():
    return load_diabetic_data(), load_drought_data()