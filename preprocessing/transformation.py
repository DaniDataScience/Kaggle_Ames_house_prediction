import numpy as np

from preprocessing.data_names import COLS
from preprocessing.data_exploration import is_missing
import pandas as pd


def drop_columns(df: "pd.DataFrame", columns: list):
    """
    drop not needed columns
    """
    print("...drop_columns: applying function to drop columns {}...".format(columns))
    df_drop = df.copy()
    for column in columns:
        df_drop = df.drop(column, axis=1)
        print("dropped column {}".format(column))
    return df_drop


def age_transformer(df: "pd.DataFrame", columns: list, base_year: str):
    """
    subtract variables containing dates from date sold, to calculate age, and rou to decimals to create bins
    """
    print("...age_transformer: applying function to ttransform years to age relative to year sold ...")
    df_age = df.copy()
    for column in columns:
        df_age[column + "_DECADE_BIN"] = df[base_year] - df[column]
        df_age[column + "_DECADE_BIN"] = df_age[column + "_DECADE_BIN"].apply(lambda x:
                                                                              round(x, -1) / 10
                                                                              )
        COLS.NUMERIC_COLS.append(column + "_DECADE_BIN")  # adding new columns to list of column names
        print("Created column {}".format(column + "_DECADE_BIN"))
        df_age = df_age.drop(column, axis=1)
        print("Dropped column {}".format(column))

    return df_age


def refactor_ordinals(df: "pd.DataFrame"):
    """
    Refactor ordinal categorical variables into numerical variables
    """
    print("...refactor_ordinals: applying function for refactoring ordinal columns {}...".format(COLS.ORDINAL_COLS))
    df_refactor = df.copy()

    df_refactor.drop(["ExterQual"], axis=1)
    df_refactor["ExterQual"] = df["ExterQual"].apply(lambda x:
                                                              1 if x == "Po" else (
                                                                  2 if x == "Fa" else (
                                                                      3 if x == "TA" else (
                                                                          4 if x == "Gd" else (
                                                                              5 if x == "Ex" else -1)))))
    df_refactor.drop(["ExterCond"], axis=1)
    df_refactor["ExterCond"] = df["ExterCond"].apply(lambda x:
                                                              1 if x == "Po" else (
                                                                  2 if x == "Fa" else (
                                                                      3 if x == "TA" else (
                                                                          4 if x == "Gd" else (
                                                                              5 if x == "Ex" else -1)))))
    df["BsmtQual"].fillna("missing", inplace=True)
    df_refactor.drop(["BsmtQual"], axis=1)
    df_refactor["BsmtQual"] = df["BsmtQual"].apply(lambda x:
                                                            1 if x == "Po" else (
                                                                2 if x == "Fa" else (
                                                                    3 if x == "TA" else (
                                                                        4 if x == "Gd" else (
                                                                            5 if x == "Ex" else (
                                                                                np.nan if "missing" else -1))))))
    df["BsmtCond"].fillna("missing", inplace=True)
    df_refactor.drop(["BsmtCond"], axis=1)
    df_refactor["BsmtCond"] = df["BsmtCond"].apply(lambda x:
                                                            1 if x == "Po" else (
                                                                2 if x == "Fa" else (
                                                                    3 if x == "TA" else (
                                                                        4 if x == "Gd" else (
                                                                            5 if x == "Ex" else (
                                                                                np.nan if "missing" else -1))))))
    df["BsmtExposure"].fillna("missing", inplace=True)
    df_refactor.drop(["BsmtExposure"], axis=1)
    df_refactor["BsmtExposure"] = df["BsmtExposure"].apply(lambda x:
                                                                    1 if x == "No" else (
                                                                        2 if x == "Mn" else (
                                                                            3 if x == "Av" else (
                                                                                4 if x == "Gd" else (
                                                                                    np.nan if "missing" else -1)))))
    df["BsmtFinType1"].fillna("missing", inplace=True)
    df_refactor.drop(["BsmtFinType1"], axis=1)
    df_refactor["BsmtFinType1"] = df["BsmtFinType1"].apply(lambda x:
                                                                    1 if x == "GLQ" else (
                                                                        2 if x == "ALQ" else (
                                                                            3 if x == "BLQ" else (
                                                                                4 if x == "Rec" else (
                                                                                    5 if x == "LwQ" else (
                                                                                        6 if x == "Unf" else (
                                                                                            np.nan if "missing" else -1)))))))
    df["BsmtFinType2"].fillna("missing", inplace=True)
    df_refactor.drop(["BsmtFinType2"], axis=1)
    df_refactor["BsmtFinType2"] = df["BsmtFinType2"].apply(lambda x:
                                                                    1 if x == "GLQ" else (
                                                                        2 if x == "ALQ" else (
                                                                            3 if x == "BLQ" else (
                                                                                4 if x == "Rec" else (
                                                                                    5 if x == "LwQ" else (
                                                                                        6 if x == "Unf" else (
                                                                                            np.nan if "missing" else -1)))))))
    df_refactor.drop(["HeatingQC"], axis=1)
    df_refactor["HeatingQC"] = df["HeatingQC"].apply(lambda x:
                                                              1 if x == "Po" else (
                                                                  2 if x == "Fa" else (
                                                                      3 if x == "TA" else (
                                                                          4 if x == "Gd" else (
                                                                              5 if x == "Ex" else -1)))))
    df_refactor.drop(["KitchenQual"], axis=1)
    df_refactor["KitchenQual"] = df["KitchenQual"].apply(lambda x:
                                                                  1 if x == "Po" else (
                                                                      2 if x == "Fa" else (
                                                                          3 if x == "TA" else (
                                                                              4 if x == "Gd" else (
                                                                                  5 if x == "Ex" else -1)))))
    df_refactor.drop(["Functional"], axis=1)
    df_refactor["Functional"] = df["Functional"].apply(lambda x:
                                                                1 if x == "Typ" else (
                                                                    2 if x == "Min1" else (
                                                                        3 if x == "Min2" else (
                                                                            4 if x == "Mod" else (
                                                                                5 if x == "Maj1" else (
                                                                                    6 if x == "Maj2" else (
                                                                                        7 if x == "Sev" else (
                                                                                            8 if x == "Sal" else -1)))))
                                                                    )))
    df["FireplaceQu"].fillna("missing", inplace=True)
    df_refactor.drop(["FireplaceQu"], axis=1)
    df_refactor["FireplaceQu"] = df["FireplaceQu"].apply(lambda x:
                                                                  1 if x == "Po" else (
                                                                      2 if x == "Fa" else (
                                                                          3 if x == "TA" else (
                                                                              4 if x == "Gd" else (
                                                                                  5 if x == "Ex" else (
                                                                                      np.nan if "missing" else -1))))))
    df["GarageFinish"].fillna("missing", inplace=True)
    df_refactor.drop(["GarageFinish"], axis=1)
    df_refactor["GarageFinish"] = df["GarageFinish"].apply(lambda x:
                                                                    1 if x == "Fin" else (
                                                                        2 if x == "RFn" else (
                                                                            3 if x == "Unf" else (
                                                                                np.nan if x == "missing" else -1))))
    df["GarageQual"].fillna("missing", inplace=True)
    df_refactor.drop(["GarageQual"], axis=1)
    df_refactor["GarageQual"] = df["GarageQual"].apply(lambda x:
                                                                1 if x == "Po" else (
                                                                    2 if x == "Fa" else (
                                                                        3 if x == "TA" else (
                                                                            4 if x == "Gd" else (
                                                                                5 if x == "Ex" else (
                                                                                    np.nan if x == "missing" else -1))))))
    df["GarageCond"].fillna("missing", inplace=True)
    df_refactor.drop(["GarageCond"], axis=1)
    df_refactor["GarageCond"] = df["GarageCond"].apply(lambda x:
                                                                1 if x == "Po" else (
                                                                    2 if x == "Fa" else (
                                                                        3 if x == "TA" else (
                                                                            4 if x == "Gd" else (
                                                                                5 if x == "Ex" else (
                                                                                    np.nan if x == "missing" else -1))))))
    df["PoolQC"].fillna("missing", inplace=True)
    df_refactor.drop(["PoolQC"], axis=1)
    df_refactor["PoolQC"] = df["PoolQC"].apply(lambda x:
                                                        1 if x == "Fa" else (
                                                            2 if x == "TA" else (
                                                                3 if x == "Gd" else (
                                                                    4 if x == "Ex" else (
                                                                        np.nan if x == "missing" else -1)))))
    df["Fence"].fillna("missing", inplace=True)
    df_refactor.drop(["Fence"], axis=1)
    df_refactor["Fence"] = df["Fence"].apply(lambda x:
                                                      1 if x == "MnWw" else (
                                                          2 if x == "GdW" else (
                                                              3 if x == "MnPrv" else (
                                                                  4 if x == "GdPrv" else (
                                                                      np.nan if x == "missing" else -1)))))

    return df_refactor


def flag_missing_and_impute(df: "pd.DataFrame", columns: list, impute):
    print("...flag_missing_and_impute: applying function for flagging and imputing columns {}...".format(columns))
    df_flag = df.copy()
    for column in columns:
        df_flag.drop(column, axis=1, inplace=True)
        df[column].fillna(impute, inplace=True)
        df_flag[column] = df[column]
        df_flag[column + "_missing_flag"] = df_flag[column].apply(lambda x: 1 if x == impute else 0)
        print("created columns {}".format(column + "_missing_flag"))
        df_flag[column] = df_flag[column].astype(int)
    return df_flag


def create_impute_nominal_missing(df: "pd.DataFrame", columns: list, impute):
    print("...create_impute_nominal_missing: applying function for imputing missing as a category for nominals {}...".format(columns))
    df_missing_cat = df.copy()
    df_missing_cat.drop(columns, axis=1, inplace=True)
    for column in columns:
        df[column].fillna(impute, inplace=True)
    df_missing_cat[columns] = df[columns]


def one_hot_encoding(df: "pd.DataFrame", columns):
    print("...one_hot_encoding: applying function for one hot encoding of columns {}...".format(columns))
    df_one_hot = pd.get_dummies(df, columns=columns)
    return df_one_hot


# --------------------------------------------------------------------------------------------------------------------

def data_transformation(df: "pd.DataFrame"):
    # drop ID column
    df = drop_columns(df, [COLS.ID])

    # DEALING WITH DATES

    # transform years to age, relative to year of sales
    df = age_transformer(df, COLS.DATE_COLS, COLS.BASE_YEAR)

    # DEALING WITH ORDINALS

    # refactor ordinals
    df = refactor_ordinals(df)

    # deal with missing ordinals
    df = flag_missing_and_impute(df, COLS.ORDINAL_COLS, 0)

    # DEALING WITH NOMINALS

    # create category from missing in nominals
    #df = create_impute_nominal_missing(df, COLS.NOMINAL_COLS, "No")

    # one hot encoding
    #df = one_hot_encoding(df, COLS.NOMINAL_COLS)

    # winsorizing

    # normalizing

    # checking missing again after transformations
    is_missing(df)

    print("--- end of data transformation ---")

    return df
