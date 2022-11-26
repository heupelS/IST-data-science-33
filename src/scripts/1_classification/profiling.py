import sys
sys.path.append('../../utils')
from load_data import load_diabetic_data

import pandas as pd

def profiling():
    data = load_diabetic_data()

if __name__ == "__main__":
    profiling()
