import pandas as pd
from preprocessing.preparing_datasets import create_full_dataset
from preprocessing.data_exploration import view_data_distributions
from preprocessing.data_exploration import view_conditional_mean
from preprocessing.transformation import deal_with_missing
from preprocessing.transformation import data_transformation
from modelling.model_prep import prepare_model,fit_random_forest


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # COMBINE TEST AND TRAIN SET
    create_full_dataset()

    # READ DATASET
    df_full = pd.read_csv("data/processing/input/train_and_prediction_set.csv")

    # DEAL WITH MISSING
    df_full = deal_with_missing(df_full)

    # EXPLORE DATA
    view_data_distributions(df_full, plot_bool=False)

    # VIEW DISTRIBUTIONS
    view_conditional_mean(df_full, df_full.columns, plot_bool=False)

    # TRANSFORM DATA
    df_full = data_transformation(df_full)

    # SAVE TO OUTPUT
    df_full.to_csv("data/processing/output/df_processed.csv", index=False)

    # PREPARE MODELING
    X_train, X_holdout, y_train, y_holdout = prepare_model(df_full)

    # Random forest
    fit_random_forest(X_train, y_train, X_holdout, y_holdout, eval_on_holdout=True, feature_importance=True, make_prediction=True)


    # END
    print("END")
