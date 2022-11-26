import os

def get_plot_folder_path():
    path_ws = os.path.dirname(os.path.abspath(__file__))
    return '%s/../../plots/' % path_ws
