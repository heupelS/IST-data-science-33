import os

def get_plot_folder_path():
    path_ws = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(path_ws,'docs','plots')
