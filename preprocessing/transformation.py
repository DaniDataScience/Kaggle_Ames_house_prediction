import numpy as np

from preprocessing.data_names import COLS
from preprocessing.data_exploration import is_missing
import pandas as pd


def drop_columns(df: "pd.DataFrame", columns: list):
    """
    drop not needed columns
    """
    df_drop = df.copy()
    for column in columns:
        df_drop = df.drop(column, axis=1)
        print("dropped column {}".format(column))
    return df_drop


def age_transformer(df: "pd.DataFrame", columns: list, base_year: str):
    """
    subtract variables containing dates from date sold, to calculate age, and rou to decimals to create bins
    """
    print("...applying function {} ...".format(__name__))
    df_age = df.copy()
    for column in columns:
        df_age[column + "_DECADE_BIN"] = df[base_year] - df[column]
        df_age[column + "_DECADE_BIN"] = df_age[column + "_DECADE_BIN"].apply(lambda x:
                                                                              round(x, -1) / 10
                                                                              )
        COLS.NOMINAL_COLS.append(column + "_DECADE_BIN")  # adding new columns to list of column names
        print("Created column {}".format(column + "_AGE"))
    return df_age


def one_hot_encoding(df: "pd.DataFrame", columns):
    df_one_hot = df.copy()
    df_one_hot = pd.get_dummies(df_one_hot, columns=columns)
    return df_one_hot


def refactor_ordinals(df: "pd.DataFrame"):
    """
    Refactor ordinal categorical variables into numerical variables
    """
    df_refactor = df.copy()

    df_refactor["ExterQual"] = df_refactor["ExterQual"].apply(lambda x:
                                                              1 if x == "Po" else (
                                                                  2 if x == "Fa" else (
                                                                      3 if x == "TA" else (
                                                                          4 if x == "Gd" else (
                                                                              5 if x == "Ex" else -1)))))
    df_refactor["ExterCond"] = df_refactor["ExterCond"].apply(lambda x:
                                                              1 if x == "Po" else (
                                                                  2 if x == "Fa" else (
                                                                      3 if x == "TA" else (
                                                                          4 if x == "Gd" else (
                                                                              5 if x == "Ex" else -1)))))

    df_refactor["BsmtQual"] = df_refactor["BsmtQual"].apply(lambda x:
                                                            1 if x == "Po" else (
                                                                2 if x == "Fa" else (
                                                                    3 if x == "TA" else (
                                                                        4 if x == "Gd" else (
                                                                            5 if x == "Ex" else (
                                                                                0 if np.NaN else -1))))))
    df_refactor["BsmtCond"] = df_refactor["BsmtCond"].apply(lambda x:
                                                            1 if x == "Po" else (
                                                                2 if x == "Fa" else (
                                                                    3 if x == "TA" else (
                                                                        4 if x == "Gd" else (
                                                                            5 if x == "Ex" else (
                                                                                0 if np.NaN else -1))))))
    df_refactor["BsmtExposure"] = df_refactor["BsmtExposure"].apply(lambda x:
                                                                    1 if x == "Po" else (
                                                                        2 if x == "Fa" else (
                                                                            3 if x == "TA" else (
                                                                                4 if x == "Gd" else (
                                                                                    0 if np.NaN else -1)))))

    df_refactor["BsmtFinType1"] = df_refactor["BsmtFinType1"].apply(lambda x:
                                                                    1 if x == "GLQ" else (
                                                                        2 if x == "ALQ" else (
                                                                            3 if x == "BLQ" else (
                                                                                4 if x == "Rec" else (
                                                                                    5 if x == "LwQ" else (
                                                                                        6 if x == "Unf" else (
                                                                                            0 if np.NaN else -1)))))))
    df_refactor["BsmtFinType2"] = df_refactor["BsmtFinType2"].apply(lambda x:
                                                                    1 if x == "GLQ" else (
                                                                        2 if x == "ALQ" else (
                                                                            3 if x == "BLQ" else (
                                                                                4 if x == "Rec" else (
                                                                                    5 if x == "LwQ" else (
                                                                                        6 if x == "Unf" else (
                                                                                            0 if np.NaN else -1)))))))
    df_refactor["HeatingQC"] = df_refactor["HeatingQC"].apply(lambda x:
                                                              1 if x == "Po" else (
                                                                  2 if x == "Fa" else (
                                                                      3 if x == "TA" else (
                                                                          4 if x == "Gd" else (
                                                                              5 if x == "Ex" else -1)))))
    df_refactor["KitchenQual"] = df_refactor["KitchenQual"].apply(lambda x:
                                                                  1 if x == "Po" else (
                                                                      2 if x == "Fa" else (
                                                                          3 if x == "TA" else (
                                                                              4 if x == "Gd" else (
                                                                                  5 if x == "Ex" else -1)))))
    df_refactor["Functional"] = df_refactor["Functional"].apply(lambda x:
                                                                1 if x == "Typ" else (
                                                                    2 if x == "Min1" else (
                                                                        3 if x == "Min2" else (
                                                                            4 if x == "Mod" else (
                                                                                5 if x == "Maj1" else (
                                                                                    6 if x == "Maj2" else (
                                                                                        7 if x == "Sev" else (
                                                                                            8 if x == "Sal" else -1)))))
                                                                    )))
    df_refactor["FireplaceQu"] = df_refactor["FireplaceQu"].apply(lambda x:
                                                                  1 if x == "Po" else (
                                                                      2 if x == "Fa" else (
                                                                          3 if x == "TA" else (
                                                                              4 if x == "Gd" else (
                                                                                  5 if x == "Ex" else (
                                                                                      0 if np.NaN else -1))))))
    df_refactor["GarageFinish"] = df_refactor["GarageFinish"].apply(lambda x:
                                                                    1 if x == "Fin" else (
                                                                        2 if x == "RFn" else (
                                                                            3 if x == "unf" else (
                                                                                0 if x == np.NaN else -1))))
    df_refactor["GarageQual"] = df_refactor["GarageQual"].apply(lambda x:
                                                                1 if x == "Po" else (
                                                                    2 if x == "Fa" else (
                                                                        3 if x == "TA" else (
                                                                            4 if x == "Gd" else (
                                                                                5 if x == "Ex" else (
                                                                                    0 if x == np.NaN else -1))))))
    df_refactor["GarageCond"] = df_refactor["GarageCond"].apply(lambda x:
                                                                1 if x == "Po" else (
                                                                    2 if x == "Fa" else (
                                                                        3 if x == "TA" else (
                                                                            4 if x == "Gd" else (
                                                                                5 if x == "Ex" else (
                                                                                    0 if x == np.NaN else -1))))))
    df_refactor["PoolQC"] = df_refactor["PoolQC"].apply(lambda x:
                                                        1 if x == "Fa" else (
                                                            2 if x == "TA" else (
                                                                3 if x == "Gd" else (
                                                                    4 if x == "Ex" else (
                                                                        0 if x == np.NaN else -1)))))
    df_refactor["Fence"] = df_refactor["Fence"].apply(lambda x:
                                                      1 if x == "MnWw" else (
                                                          2 if x == "GdW" else (
                                                              3 if x == "MnPrv" else (
                                                                  4 if x == "GdPrv" else (
                                                                      0 if x == np.NaN else -1)))))

    return df_refactor


# --------------------------------------------------------------------------------------------------------------------

def data_transformation(df: "pd.DataFrame"):
    # drop ID column
    df = drop_columns(df, [COLS.ID])

    # transform years to age, relative to year of sales
    df = age_transformer(df, COLS.DATE_COLS, COLS.BASE_YEAR)

    # Drop year columns
    df = drop_columns(df, COLS.DATE_COLS)

    # one hot encoding
    df = one_hot_encoding(df, COLS.NOMINAL_COLS)

    # refactor ordinals
    df = refactor_ordinals(df)

    # dealing with missing

    # winsorizing

    # normalizing

    # flagging zeros

    # checking missing again after transformations
    is_missing(df)

    print("--- end of data transformation ---")

    return df
