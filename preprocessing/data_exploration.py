import matplotlib.pyplot as plt
from preprocessing.data_names import COLS



def is_missing(df: "pd.DataFrame"):
    """
    check columns that have missing values
    :param df: dataframe to check
    :return: number of missing
    """
    print("...is_missing: applying function to check missing...")
    if df.isna().sum().sum() == 0:
        print("No missing values")
    else:
        return print(df.isna().sum()[df.isna().sum() > 0])


def numerical_distribution_hist(df: "pd.DataFrame", columns: list, plot_bool: bool):
    """
    check the distribution of any numeric variable with a histogram
    """
    print("...numerical_distribution_hist: applying function to check histograms for {}...".format(columns))
    if plot_bool == True:
        for column in columns:
            try:
                columnSeriesObj = df[column]
                columnSeriesObj.hist(bins=50)
                plt.title(columnSeriesObj.name)
                plt.show()
            except KeyError:
                print("{} column not in dataset".format(column))
    else:
        print("Plotting turned off")


def categorical_distribution_bar(df: "pd.DataFrame", columns: list, plot_bool: bool):
    """
    check the bar plot of any categorical variable
    """
    print("...categorical_distribution_bar: applying function to check bar plots")
    if plot_bool == True:
        for column in columns:
            try:
                columnSeriesObj = df[column]
                columnSeriesObj.value_counts().plot(kind="bar")
                plt.title(columnSeriesObj.name)
                if plot_bool == True:
                    plt.show()
                else:
                    print("Plotting turned off")
            except KeyError:
                print("{} column not in dataset".format(column))
    else:
        print("Plotting turne off")


# --------------------------------------------------------------------------------------------------------------------


def view_data_distributions(df: "pd.DataFrame"):
    # show basic info
    print("...describe dataset..")
    print(df.describe)

    # show columns
    print("...check column types...")
    print(df.info())

    # check missing
    is_missing(df)

    # check target variable hist
    numerical_distribution_hist(df, [COLS.TARGET], plot_bool=False)

    # check numerical variable hist
    numerical_distribution_hist(df, COLS.NUMERIC_COLS, plot_bool=False)

    # check categorical variable bar plot
    categorical_distribution_bar(df, COLS.CATEGORICAL_COLS, plot_bool=False)

    print("--- end of data exploration ---")
