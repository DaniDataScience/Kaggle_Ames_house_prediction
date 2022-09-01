import numpy as np

from preprocessing.data_names import COLS
from preprocessing.data_exploration import is_missing
import pandas as pd



def refactor_NaN_and_flag(df: "pd.DataFrame", categorical_refactor: str, numeric_refactor: float):
    """
    refactor missing either into categorical value for missing or into zero
    """
    print("...refactor_NaN: refactoring NaN...")
    df_temp = df.copy()
    for column in df.columns:

        # numeric refactor
        if column in ["LotFrontage", "MasVnrArea"]:
            df_temp[column].fillna(numeric_refactor, inplace=True)
            print("refactored Nan values in column {} to 0".format(column))
            df_temp[column + "_missing_flag"] = df_temp[column].apply(lambda x: 1 if x == numeric_refactor else 0)
            print("created columns {}".format(column + "_missing_flag"))
            df_temp[column + "_missing_flag"] = df_temp[column + "_missing_flag"].astype(int)

        # caregorical refactor
        if column in ["MasVnrType", "Alley", "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                      "BsmtFinType2","Electrical","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageQual",
                      "GarageCond","PoolQC","Fence","MiscFeature"]:
            df_temp[column].fillna(categorical_refactor, inplace=True)
            print("refactored column {} to None".format(categorical_refactor))
            df_temp[column + "_missing_flag"] = df_temp[column].apply(lambda x: 1 if x == categorical_refactor else 0)
            print("created columns {}".format(column + "_missing_flag"))
            df_temp[column + "_missing_flag"] = df_temp[column + "_missing_flag"].astype(int)

        # for missing values in prediction set
        if column in ["MSZoning","Utilities","Exterior1st","Exterior2nd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
                      "TotalBsmtSF","BsmtFullBath","BsmtHalfBath","KitchenQual","Functional","GarageCars",
                      "GarageArea","SaleType"]:
            df_temp[column].fillna(df[column].mode()[0],inplace=True)
            print("imputed missing for {}".format(column))
    return df_temp


def GarageYrBlt_refactoring(df: "pd.DataFrame"):
    """
    Transform garage year built to a flag variable determining if garage was built with house or later"
    """
    print("...GarageYrBlt_refactoring: applying function to transform GarageYrBlt ...")
    df_temp = df.copy()
    df_temp['Garage_BwH'] = np.where(df["YearBuilt"] == df["GarageYrBlt"], 1, 0)
    df_temp = df_temp.drop("GarageYrBlt", axis=1)
    print("GarageYrBlt dropped, Garage_BwH created")
    return df_temp


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
    print("...age_transformer: applying function to transform years to age relative to year sold ...")
    df_age = df.copy()
    for column in columns:
        try:
            df_age[column + "_DECADE_BIN"] = df[base_year] - df[column]
            df_age[column + "_DECADE_BIN"] = df_age[column + "_DECADE_BIN"].apply(lambda x:
                                                                                  round(x, -1) / 10
                                                                                  )
            COLS.NUMERIC_COLS.append(column + "_DECADE_BIN")  # adding new columns to list of column names
            print("Created column {}".format(column + "_DECADE_BIN"))
            df_age = df_age.drop(column, axis=1)
            print("Dropped column {}".format(column))
            df_age[column + "_DECADE_BIN"] = df_age[column + "_DECADE_BIN"].astype(int)
        except KeyError:
            print("Columns {} has already been dropped or missing".format(column))
    df_age = drop_columns(df_age, [COLS.BASE_YEAR])

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
    df_refactor.drop(["BsmtQual"], axis=1)
    df_refactor["BsmtQual"] = df["BsmtQual"].apply(lambda x:
                                                   1 if x == "Po" else (
                                                       2 if x == "Fa" else (
                                                           3 if x == "TA" else (
                                                               4 if x == "Gd" else (
                                                                   5 if x == "Ex" else (
                                                                       0 if "None" else -1))))))
    df_refactor.drop(["BsmtCond"], axis=1)
    df_refactor["BsmtCond"] = df["BsmtCond"].apply(lambda x:
                                                   1 if x == "Po" else (
                                                       2 if x == "Fa" else (
                                                           3 if x == "TA" else (
                                                               4 if x == "Gd" else (
                                                                   5 if x == "Ex" else (
                                                                       0 if "None" else -1))))))
    df_refactor.drop(["BsmtExposure"], axis=1)
    df_refactor["BsmtExposure"] = df["BsmtExposure"].apply(lambda x:
                                                           1 if x == "No" else (
                                                               2 if x == "Mn" else (
                                                                   3 if x == "Av" else (
                                                                       4 if x == "Gd" else (
                                                                           0 if "None" else -1)))))
    df_refactor.drop(["BsmtFinType1"], axis=1)
    df_refactor["BsmtFinType1"] = df["BsmtFinType1"].apply(lambda x:
                                                           1 if x == "GLQ" else (
                                                               2 if x == "ALQ" else (
                                                                   3 if x == "BLQ" else (
                                                                       4 if x == "Rec" else (
                                                                           5 if x == "LwQ" else (
                                                                               6 if x == "Unf" else (
                                                                                   0 if "None" else -1)))))))
    df_refactor.drop(["BsmtFinType2"], axis=1)
    df_refactor["BsmtFinType2"] = df["BsmtFinType2"].apply(lambda x:
                                                           1 if x == "GLQ" else (
                                                               2 if x == "ALQ" else (
                                                                   3 if x == "BLQ" else (
                                                                       4 if x == "Rec" else (
                                                                           5 if x == "LwQ" else (
                                                                               6 if x == "Unf" else (
                                                                                   0 if "None" else -1)))))))
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
    df_refactor.drop(["FireplaceQu"], axis=1)
    df_refactor["FireplaceQu"] = df["FireplaceQu"].apply(lambda x:
                                                         1 if x == "Po" else (
                                                             2 if x == "Fa" else (
                                                                 3 if x == "TA" else (
                                                                     4 if x == "Gd" else (
                                                                         5 if x == "Ex" else (
                                                                             0 if "None" else -1))))))
    df_refactor.drop(["GarageFinish"], axis=1)
    df_refactor["GarageFinish"] = df["GarageFinish"].apply(lambda x:
                                                           1 if x == "Fin" else (
                                                               2 if x == "RFn" else (
                                                                   3 if x == "Unf" else (
                                                                       0 if x == "None" else -1))))
    df_refactor.drop(["GarageQual"], axis=1)
    df_refactor["GarageQual"] = df["GarageQual"].apply(lambda x:
                                                       1 if x == "Po" else (
                                                           2 if x == "Fa" else (
                                                               3 if x == "TA" else (
                                                                   4 if x == "Gd" else (
                                                                       5 if x == "Ex" else (
                                                                           0 if x == "None" else -1))))))
    df_refactor.drop(["GarageCond"], axis=1)
    df_refactor["GarageCond"] = df["GarageCond"].apply(lambda x:
                                                       1 if x == "Po" else (
                                                           2 if x == "Fa" else (
                                                               3 if x == "TA" else (
                                                                   4 if x == "Gd" else (
                                                                       5 if x == "Ex" else (
                                                                           0 if x == "None" else -1))))))
    df_refactor.drop(["PoolQC"], axis=1)
    df_refactor["PoolQC"] = df["PoolQC"].apply(lambda x:
                                               1 if x == "Fa" else (
                                                   2 if x == "TA" else (
                                                       3 if x == "Gd" else (
                                                           4 if x == "Ex" else (
                                                               0 if x == "None" else -1)))))
    df_refactor.drop(["Fence"], axis=1)
    df_refactor["Fence"] = df["Fence"].apply(lambda x:
                                             1 if x == "MnWw" else (
                                                 2 if x == "GdW" else (
                                                     3 if x == "MnPrv" else (
                                                         4 if x == "GdPrv" else (
                                                             0 if x == "None" else -1)))))

    return df_refactor


def one_hot_encoding(df: "pd.DataFrame", columns:list):
    print("...one_hot_encoding: applying function for one hot encoding of columns {}...".format(columns))

    for column in columns:
        df = pd.get_dummies(df, columns=[column])
        print("one hot encoding done for {}".format(column))
    return df


def winsorizing(df, threshold:int=20):
        """
        Cutting the min and max outlier values by limiting htem
        :param threshold: how many sigma distances should there be alowed for outliers
        """
        print("...winsorizing: applying function to limit min and max outliers for all columns")
        df_win = df.copy()
        for column in COLS.NUMERIC_COLS:
            mean_1, std_1 = np.mean(df[column]), np.std(df[column])
            if (np.abs((df[column] - mean_1) / std_1) > threshold).sum() > 0:
                print(f"column winsorized:{column} number winsorized:",
                      (df[column] > threshold * std_1 + mean_1).sum())
                df_win[column] = df[column].apply(lambda x: threshold * std_1 + mean_1
                if np.abs((x - mean_1) / std_1) > threshold else x)
        return df_win


def normalizing(df):
    """
    Normalizing numeric variables
    """
    print("...normalizing: applying function to normalize numerical variables")
    df_norm = df.copy()
    for column in COLS.NUMERIC_COLS+COLS.ORDINAL_COLS:
            series_min, series_max = df[column].min(), df[column].max()
            df_norm[column] = df[column].apply(lambda x:
                                               (x-series_min)/(series_max-series_min))
    return df_norm

# --------------------------------------------------------------------------------------------------------------------

def deal_with_missing(df: "pd.DataFrame"):
    print("--- START OF DEAL WITH MISSING ---")

    # refactor missing
    df = refactor_NaN_and_flag(df, "None", 0)

    # transform garage year built (deal with mix of years and no garage)
    df = GarageYrBlt_refactoring(df)

    # check remaining missing
    is_missing(df)

    print("--- END OF DEAL WITH MISSING ---")

    return df


def data_transformation(df: "pd.DataFrame"):
    print("--- START OF DATA TRANSFORMATION ---")

    # DEALING WITH DATES

    # transform years to age, relative to year of sales
    df = age_transformer(df, COLS.DATE_COLS, COLS.BASE_YEAR)

    # DEALING WITH ORDINALS

    # refactor ordinals
    df = refactor_ordinals(df)

    # winsorizing
    df = winsorizing(df)

    # normalizing
    df = normalizing(df)

    # one hot encoding
    df = one_hot_encoding(df, COLS.NOMINAL_COLS)

    # checking missing again after transformations
    is_missing(df)

    print("--- END OF DATA TRANSFORMATION ---")

    return df
