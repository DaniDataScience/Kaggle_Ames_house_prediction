import pandas as pd

def create_full_dataset():
    """
    combine test and train dataset so data processing is done on both sets the same way
    """
    print("...create_full_dataset: combining test and train set so  processing is done on both sets...")

    df_full_train = pd.read_csv("data/raw/full_train_dataset.csv")
    df_prediction = pd.read_csv("data/raw/prediction_set.csv")

    df_prediction["Dataset"] = "Prediction set"
    df_full_train["Dataset"] = "Train set"

    df_train_and_prediction = pd.concat([df_prediction, df_full_train])
    print(df_train_and_prediction.describe())
    df_train_and_prediction.to_csv("data/processing/input/train_and_prediction_set.csv", index=False)

    print("raw test and train datasets combined for preprocessing")


