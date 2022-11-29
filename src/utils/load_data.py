
import csv
import os
import pandas as pd

def load_diabetic_data():
    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        csv = pd.read_csv('%s/../data/diabetic_data.csv' % path_ws)
        return csv

    except Exception as e:
        print(e)


def load_drought_data():
    try:
        path_ws = os.path.dirname(os.path.abspath(__file__))
        csv = pd.read_csv('%s/../data/drought.csv' % path_ws, parse_dates=True)
        return csv

    except Exception as e:
        print(e)

def read_data():
    return load_diabetic_data(), load_drought_data()