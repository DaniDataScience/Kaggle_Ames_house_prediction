import sklearn.metrics
from sklearn.model_selection import train_test_split
from preprocessing.data_names import COLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error


def split_prediction_and_train(df:'pd.DataFrame'):
    """
    Create a validation data set from the original raw data set before feature engineering
    :param df: the raw dataset with all the data
    :param ratio: ratio of the validation dataset size compared to the total dataset size
    :return: validation and train dataset
    """
    print("...split_prediction_and_train: splitting whole dataset into prediction and train set...")

    df_predict = df[df["Dataset"]=="Prediction set"]
    df_predict.to_csv("data/predict/input/prediction_set.csv")
    print("prediction set shape: {}".format(df_predict.shape))

    df_train = df[df["Dataset"]=="Train set"]
    df_train.to_csv("data/model/df_train_full.csv", index=False)
    print("train set shape: {}".format(df_train.shape))

    return df_train


# --------------------------------------------------------------------------------------------------------------------

def prepare_model(df: "pd.DataFrame", target_col:str=COLS.TARGET):
    print("--- START OF MODEL PREPARATION ---")

    # splitting train and prediction set
    df_train = split_prediction_and_train(df)

    # train test split
    X = df_train.drop(target_col, axis=1)
    y = df_train[[target_col]].iloc[:,0]
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=42)

    print("X_train shape:{}".format(X_train.shape))
    print("X_holdout shape:{}".format(X_holdout.shape))
    print("y_train shape:{}".format(y_train.shape))
    print("y_holdout shape:{}".format(y_holdout.shape))

    print("--- END OF MODEL PREPARATION ---")

    return X_train, X_holdout, y_train, y_holdout


class random_forest():
    def __init__(self):
        self.grid_search = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid= {
                'bootstrap': [True, False],
                'n_estimators': [5, 10],
                'criterion': ['gini', 'entropy'],
                'min_samples_leaf': list(range(1, 12, 4)),
                'max_features': ['sqrt', 'log2']
            },
            cv=StratifiedKFold(n_splits=3),
            verbose=1,
            scoring="mean_squared_error",
            refit=True)
        self.trained_model = None

    def train(self, X_train:"pd.DataFrame", y_train:"np.ndarray"):

        self.trained_model = self.grid_search
        self.trained_model.fit(X_train, y_train)

        # best grid
        best_grid = self.trained_model.best_params_
        print("Best grid is {}".format(best_grid))

        # Best params and AUC
        print("Best parameters are:{}".format(self.trained_model.best_params_))
        print("best score is {}".format(self.trained_model.best_score_))

    def predict(self, X_test:"pd.DataFrame")->"np.ndarray":
        return self.trained_model.predict_proba(X_test)[:,1]

    def evaluate(self, X_test:"pd.DataFrame", y_test:"np.ndarray"):
        price = self.predict(X_test)
        score = sklearn.metrics.mean_squared_error(y_test, price)
        print("RMSE on test set: {}".format(score))








