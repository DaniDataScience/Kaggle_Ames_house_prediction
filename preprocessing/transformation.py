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


def flag_missing_and_impute(df: "pd.DataFrame", columns: list, impute):
    print("...flag_missing_and_impute: applying function for flagging and imputing columns {}...".format(columns))
    df_flag = df.copy()
    for column in columns:
        df_flag.drop(column, axis=1, inplace=True)
        df[column].fillna(impute, inplace=True)
        df_flag[column] = df[column]
        df_flag[column + "_missing_flag"] = df_flag[column].apply(lambda x: 1 if x == impute else 0)
        print("created columns {}".format(column + "_missing_flag"))
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
