import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.create_validation_set import create_validation_set
from preprocessing.data_names import COLS
from preprocessing.data_exploration import view_data_distributions
from preprocessing.transformation import data_transformation


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # separate validation dataset
    create_validation_set(pd.read_csv("data/raw/full_dataset.csv", na_values="nan"), 0.2)

    # read training set
    df_train_orig = pd.read_csv("data/input/df_train.csv", na_values="nan")

    # EXPLORE DATA
    view_data_distributions(df_train_orig.copy())

    # TRANSFORM DATA
    df_train = data_transformation(df_train_orig.copy())

    # SAVE TO OUTPUT
    df_train.to_csv("data/output/df_train_processed.csv", index=False)

    # END
    print("END")



