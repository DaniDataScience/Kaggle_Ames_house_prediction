import matplotlib.pyplot as plt


def is_missing(df):
    """
    check columns that have missing values
    :param df: dataframe to check
    :return: number of missing
    """
    if df.isna().sum().sum() == 0:
        print("No missing values")
    else:
        return df.isna().sum()[df.isna().sum() >0]