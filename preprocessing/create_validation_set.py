import pandas as pd


def create_validation_set(df:'pd.DataFrame', ratio:float):
    """
    Create a validation data set from the original raw data set before feature engineering
    :param df: the raw dataset with all the data
    :param ratio: ratio of the validation dataset size compared to the total dataset size
    :return: validation and train dataset
    """
    df_valid = df.sample(frac=ratio, random_state=1)
    df_train = df.drop(df_valid.index)

    df_valid.to_csv("data/df_valid.csv")
    df_train.to_csv("data/df_train.csv")

    print("dataset slit into validation and test set with ratio of {}".format(ratio))
