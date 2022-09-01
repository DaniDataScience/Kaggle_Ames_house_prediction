import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from preprocessing.data_names import COLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import math
from sklearn.inspection import permutation_importance


def split_prediction_and_train(df: "pd.DataFrame"):
    """
    Create a validation data set from the original raw data set before feature engineering
    :param df: the raw dataset with all the data
    :param ratio: ratio of the validation dataset size compared to the total dataset size
    :return: validation and train dataset
    """
    print("...split_prediction_and_train: splitting whole dataset into prediction and train set...")

    df_predict = df[df["Dataset"] == "Prediction set"]
    df_predict = df_predict.drop("Dataset", axis=1)
    df_predict = df_predict.drop("SalePrice", axis=1)
    df_predict.to_csv("data/predict/input/prediction_set.csv", index=False)
    print("prediction set shape: {}".format(df_predict.shape))

    df_train = df[df["Dataset"] == "Train set"]
    df_train = df_train.drop("Dataset", axis=1)
    df_train.to_csv("data/model/df_train_full.csv", index=False)
    print("train set shape: {}".format(df_train.shape))

    return df_train


# --------------------------------------------------------------------------------------------------------------------

def prepare_model(df: "pd.DataFrame", target_col: str = COLS.TARGET):
    print("--- START OF MODEL PREPARATION ---")

    # splitting train and prediction set
    df = split_prediction_and_train(df)

    # drop ID
    df = df.drop([COLS.ID], axis=1)

    # train test split
    X = df.drop(target_col, axis=1)
    y = df[[target_col]].iloc[:, 0]
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=42)

    print("X_train shape:{}".format(X_train.shape))
    print("X_holdout shape:{}".format(X_holdout.shape))
    print("y_train shape:{}".format(y_train.shape))
    print("y_holdout shape:{}".format(y_holdout.shape))

    print("--- END OF MODEL PREPARATION ---")

    return X_train, X_holdout, y_train, y_holdout


def fit_random_forest(X_train, y_train, X_holdout, y_holdout, feature_importance: bool, eval_on_holdout: bool,
                      make_prediction: bool):
    # CREATING MODEL
    print("...fitting model...")

    grid_search_rf = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'bootstrap': [True, False],
            'n_estimators': [10, 50, 100, 150, 200, 250],
            'min_samples_leaf': [1,5,10],
            'max_features': ['sqrt', 'log2']
        },
        cv=3,
        verbose=1,
        scoring="r2",
        refit="r2")

    grid_search_rf.fit(X_train, y_train)

    print("Random Forest best parameters are:", grid_search_rf.best_params_)
    print("Random Forest CV R2 score is: ", grid_search_rf.best_score_)

    # EVALUATE ON HOLDOUT
    if eval_on_holdout == True:
        print("...evaluate_on_holdout: evaluating on holdout...")

        y_pred = grid_search_rf.predict(X_holdout)

        score_r2 = r2_score(y_holdout, y_pred)
        score_RMSE = math.sqrt(mean_squared_error(y_holdout, y_pred))

        print("R2 on holdout set is: {}".format(score_r2))
        print("RMSE on holdout set is: {}".format(score_RMSE))

    # FEATURE IMPORTANCE REFIT
    if feature_importance == True:

        print("...calculating feature importance...")

        # creating model
        forest_ft_imp = RandomForestRegressor(random_state=0)

        # fitting
        forest_ft_imp.fit(X_train, y_train)

        # getting feature names
        feature_names = forest_ft_imp.feature_names_in_

        # calculating permutation importance
        result = permutation_importance(
            forest_ft_imp, X_holdout, y_holdout, n_repeats=10, random_state=42, n_jobs=2)

        # storing results in df
        forest_importances_dict = {"features": feature_names,
                                   "importance": result.importances_mean}

        forest_importances = pd.DataFrame(forest_importances_dict).sort_values("importance", ascending=False)

        # refitting model with top features
        feature_number_dict = []
        r2_per_feature_num_dict = []
        feature_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(10, X_train.shape[1], 20))
        for feature_num in feature_numbers:
            selected_features = forest_importances["features"][0:feature_num]
            forest_i = RandomForestRegressor(random_state=0).fit(X_train[selected_features], y_train)

            # calculating on holdout set
            y_pred = forest_i.predict(X_holdout[selected_features])
            score_r2_i = r2_score(y_holdout, y_pred)
            feature_number_dict.append(feature_num)
            r2_per_feature_num_dict.append(score_r2_i)

        r2_per_feature_num_dict = pd.DataFrame({"feature number": feature_number_dict,
                                                "r2 score": r2_per_feature_num_dict})

        fig = px.line(r2_per_feature_num_dict, x="feature number", y="r2 score",
                      title='R2 score per number of features used')
        fig.show()

    # MAKE PREDICTION
    if make_prediction == True:
        print("...making prediction on kaggle test set...")

        kaggle_x = pd.read_csv("data/predict/input/prediction_set.csv")

        kaggle_y_pred = grid_search_rf.predict(kaggle_x.loc[:, kaggle_x.columns != COLS.ID])

        submission = kaggle_x[[COLS.ID]]
        submission["SalePrice"] = kaggle_y_pred

        print(submission)

        submission.to_csv("data/predict/output/kaggle_y_pred.csv", index=False)


