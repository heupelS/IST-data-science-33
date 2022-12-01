from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from general_utils import get_plot_folder_path
from load_data import read_data

def dummification(data):
    # Drop out all records with missing values
    data.dropna(inplace=True)

if __name__ == "__main__":
    data_diabetic, data_drought = read_data()

    