#!/usr/bin/env python3

import sys, os

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data


def data_sparity():
    pass


if __name__ == "__main__":

    data_diabetic, data_drought = read_data()