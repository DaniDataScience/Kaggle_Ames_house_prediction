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


def split_into_tran_valid_prediction_set(df:'pd.DataFrame', ratio:float=0.20):
    """
    Create a validation data set from the original raw data set before feature engineering
    :param df: the raw dataset with all the data
    :param ratio: ratio of the validation dataset size compared to the total dataset size
    :return: validation and train dataset
    """
    df_predict = df[df["Dataset"]=="Prediction set"]
    df_predict.to_csv("data/predict/input/prediction_set.csv")

    df = df[df["Dataset"]=="Train set"]

    df_valid = df.sample(frac=ratio, random_state=1)
    df_train = df.drop(df_valid.index)

    df_valid.to_csv("data/model/validate/df_valid.csv", index=False)
    df_train.to_csv("data/model/train/df_train.csv", index=False)

    print("dataset slit into prediction, validation and test set with ratio of {}".format(ratio))
    print("prediction set shape: {}".format(df_predict.shape))
    print("validation set shape: {}".format(df_valid.shape))
    print("train set shape: {}".format(df_train.shape))
