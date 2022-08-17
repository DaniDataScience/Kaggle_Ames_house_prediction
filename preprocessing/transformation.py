from preprocessing.data_names import COLS
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
                                                                              round(x, -1)/10
                                                                              )
        print("Created column {}".format(column + "_AGE"))
    return df_age


# --------------------------------------------------------------------------------------------------------------------

def data_transformation(df: "pd.DataFrame"):
    # drop ID column
    df = drop_columns(df, [COLS.ID])

    # transform years to age, relative to year of sales
    df = age_transformer(df, COLS.DATE_COLS, COLS.BASE_YEAR)

    # Drop year columns
    df = drop_columns(df, COLS.DATE_COLS)

    print("--- end of data transformation ---")

    return df
