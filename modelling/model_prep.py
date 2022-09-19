import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from preprocessing.data_names import COLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import math
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import os.path
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone



def split_prediction_and_train(df: "pd.DataFrame"):
    """
    Create a validation data set from the original raw data set before feature engineering
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


def calculate_feature_importance(X_train, y_train, X_holdout, y_holdout):
    print("...calculating feature importance...")

    # creating model
    forest_ft_imp = RandomForestRegressor(random_state=0)

    # fitting
    forest_ft_imp.fit(X_train, y_train)

    # getting feature names
    feature_names = forest_ft_imp.feature_names_in_

    # calculating permutation importance
    result = permutation_importance(
        forest_ft_imp, X_holdout, y_holdout, n_repeats=10, random_state=42, n_jobs=5)

    # storing results in df
    forest_importance = pd.DataFrame(
        {"features": feature_names,
         "importance": result.importances_mean}).sort_values("importance", ascending=False)

    # refitting model with top features
    feature_number_dict = []
    r2_per_feature_num_dict = []
    selected_feature_list = []
    feature_numbers = list(range(1, X_train.shape[1], 1))
    for feature_num in feature_numbers:
        selected_features = forest_importance["features"][0:feature_num]
        forest_i = forest_ft_imp.fit(X_train[selected_features], y_train)

        # calculating on holdout set
        y_pred = forest_i.predict(X_holdout[selected_features])
        score_r2_i = r2_score(y_holdout, y_pred)
        feature_number_dict.append(feature_num)
        r2_per_feature_num_dict.append(score_r2_i)
        selected_feature_list.append(forest_importance["features"][feature_num])

    # collecting in df
    r2_per_feature_num_dict = pd.DataFrame({"feature number": feature_number_dict,
                                            "added feature": selected_feature_list,
                                            "r2 score": r2_per_feature_num_dict})

    # save best features in csv
    r2_per_feature_num_dict.to_csv("data/processing/input/feature_importance.csv", index=False)

    # plotting
    fig = px.line(r2_per_feature_num_dict, x="feature number", y="r2 score",
                  title='R2 score per number of features used')

    fig.show()

    # evaluate best parameter
    print("refitting model based on selected features")
    best_feature_num = 40
    best_selected_features = forest_importance["features"][0:best_feature_num]

    # defining model
    grid_search_rf = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'bootstrap': [True, False],
            'n_estimators': [200,250,300],
            'min_samples_leaf': [1, 3, 5, 10],
            'max_features': ['sqrt', 'log2']
        },
        cv=3,
        verbose=1,
        scoring="neg_mean_squared_error",
        refit="neg_mean_squared_error")

    return best_selected_features


def getRMSLE(model, X, y):
    """
    Return the average RMSLE over all folds of training data.
    """
    # Set KFold to shuffle data before the split
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42)

    # Get RMSLE score
    rmse = np.sqrt(-cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=kf))

    return rmse.mean()


# --------------------------------------------------------------------------------------------------------------------


def fit_random_forest(X_train, y_train, X_holdout, y_holdout,
                      calc_feature_importance: bool, do_gridsearch_fitting: bool,
                      use_feature_importance: bool, eval_on_holdout: bool,
                      visualize_holdout: bool,
                      make_final_prediction: bool):

    # CALCULATING FEATURE IMPORTANCE
    if calc_feature_importance == True:
        calculate_feature_importance(X_train, y_train, X_holdout, y_holdout)
    else:
        if os.path.exists("data/processing/input/feature_importance.csv"):
            print("using existing feature importance calculation")
        else:
            print("no feature importance found. please run feature importance calculation")


    # SELECT FEATURES (FT IMP OR ALL)
    print("...fitting RF...")

    if use_feature_importance == True:
        best_feature_num = 40
        selected_features = pd.read_csv("data/processing/input/feature_importance.csv")["added feature"][
                            0:best_feature_num]
    else:
        selected_features = X_train.columns

    # GRID SEARCH MODEL
    if do_gridsearch_fitting == True:

        print("...fitting model with gridsearch CV...")

        grid_search_rf = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'bootstrap': [True, False],
                'n_estimators': [100, 200, 300],
                'min_samples_leaf': [1, 3, 5, 10, 20, 30],
                'max_features': ['sqrt', 'log2']
            },
            cv=3,
            verbose=1,
            scoring="neg_mean_squared_error",
            refit="neg_mean_squared_error")

        grid_search_rf.fit(X_train[selected_features], y_train)

        print("Random Forest best parameters are:", grid_search_rf.best_params_)
        print("Random Forest CV RMSE score is: ", math.sqrt(-grid_search_rf.best_score_))

    # EVALUATE ON HOLDOUT
    if eval_on_holdout == True:
        print("...evaluate_on_holdout: evaluating on holdout...")

        y_pred = grid_search_rf.predict(X_holdout[selected_features])

        score_r2 = r2_score(y_holdout, y_pred)
        score_RMSE = math.sqrt(mean_squared_error(y_holdout, y_pred))

        print("R2 on holdout set is: {}".format(score_r2))
        print("RMSE on holdout set is: {}".format(score_RMSE))

    # VISUALIZE ON HOLDOUT
    if visualize_holdout == True:
        print("...visualizing prediction vs actual of the holdout set...")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted(y_holdout),
            y=sorted(y_holdout),
            name='Actual y in holdout'
        ))

        fig.add_trace(go.Scatter(
            x=sorted(y_holdout),
            y=sorted(y_pred),
            name='Predicted y in holdout'
        ))

        fig.update_layout(title='Actual and predicted y on holdout set',
                          xaxis_title='Actual y',
                          yaxis_title='Y')

        fig.show()

    # MAKE PREDICTION
    if make_final_prediction == True:
        print("...making prediction on kaggle test set...")

        # loading kaggle test set
        X_prediction = pd.read_csv("data/predict/input/prediction_set.csv")
        id_col = X_prediction[[COLS.ID]]

        # create whole dataframe
        X_full = pd.concat([X_holdout[selected_features], X_train[selected_features]])
        y_full = pd.concat([y_holdout, y_train])

        # fit
        grid_search_rf_final = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'bootstrap': [False],
                'n_estimators': [500],
                'min_samples_leaf': [1],
                'max_features': ['sqrt']
            },
            cv=3,
            verbose=1,
            scoring="neg_mean_squared_error",
            refit="neg_mean_squared_error")

        grid_search_rf_final.fit(X_full[selected_features], y_full)

        # predicting
        kaggle_x = X_prediction[selected_features]
        kaggle_y_pred = grid_search_rf_final.predict(kaggle_x)

        submission = id_col
        submission["SalePrice"] = kaggle_y_pred
        print(submission)

        submission.to_csv("data/predict/output/kaggle_y_pred_RF.csv", index=False)



def fit_XGBoost(X_train, y_train, X_holdout, y_holdout,
                use_feature_importance: bool,
                eval_on_holdout: bool,
                visualize_holdout:bool,
                manual_correction:bool,
                make_final_prediction: bool):


    if use_feature_importance == True:
        best_feature_num = 40
        selected_features = pd.read_csv("data/processing/input/feature_importance.csv")["added feature"][
                            0:best_feature_num]
    else:
        selected_features = X_train.columns


    if eval_on_holdout == True:
        print("...evaluate_on_holdout: evaluating on holdout...")

        # fit model
        XGB_model = xgb.XGBRegressor(
            n_estimators=1000,
            early_stopping_rounds=50,
            learning_rate=0.01
        )

        XGB_model.fit(X_train[selected_features], y_train,
                      eval_set=[(X_train[selected_features], y_train), (X_holdout[selected_features], y_holdout)],
                      verbose=50
                      )

        y_pred = XGB_model.predict(X_holdout[selected_features])

        score_r2 = r2_score(y_holdout, y_pred)
        score_RMSE = math.sqrt(mean_squared_error(y_holdout, y_pred))

        print("R2 on holdout set is (XGBoost): {}".format(score_r2))
        print("RMSE on holdout set is (XGBoost): {}".format(score_RMSE))


    # VISUALIZE ON HOLDOUT
    if visualize_holdout == True:
        print("...visualizing prediction vs actual of the holdout set...")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted(y_holdout),
            y=sorted(y_holdout),
            name='Actual y in holdout'
        ))

        fig.add_trace(go.Scatter(
            x=sorted(y_holdout),
            y=sorted(y_pred),
            name='Predicted y in holdout'
        ))

        fig.update_layout(title='Actual and predicted y on holdout set (XGBoost)',
                          xaxis_title='Actual y',
                          yaxis_title='Y')

        fig.show()

        if manual_correction == True:
            print("...manually adjusting y_pred...")

            y_pred_adj = [i*0.9 if i < 100000 else i*1.1 if i>350000 else i for i in y_pred]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sorted(y_holdout),
                y=sorted(y_holdout),
                name='Actual y in holdout'
            ))

            fig.add_trace(go.Scatter(
                x=sorted(y_holdout),
                y=sorted(y_pred_adj),
                name='Predicted y in holdout'
            ))

            fig.update_layout(title='Actual and predicted adjusted y on holdout set (XGBoost)',
                              xaxis_title='Actual y',
                              yaxis_title='Y')

            fig.show()

            score_r2 = r2_score(y_holdout, y_pred_adj)
            score_RMSE = math.sqrt(mean_squared_error(y_holdout, y_pred_adj))

            print("R2 on holdout set is (XGBoost, adjusted): {}".format(score_r2))
            print("RMSE on holdout set is (XGBoost, adjusted): {}".format(score_RMSE))


    # PREDICTING ON KAGGLE DATA
    if make_final_prediction == True:

        print("...making prediction on kaggle test set...")

        XGB_model = xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.01
        )

        # loading kaggle test set
        X_prediction = pd.read_csv("data/predict/input/prediction_set.csv")

        # create whole dataframe
        X_full = pd.concat([X_holdout, X_train])
        y_full = pd.concat([y_holdout, y_train])

        # fitting
        XGB_model.fit(X_full[selected_features], y_full)

        # predicting
        kaggle_x = X_prediction[selected_features]
        kaggle_y_pred = XGB_model.predict(kaggle_x)

        if manual_correction == True:
            print("...manually adjusting y_pred...")

            y_pred = [i*0.9 if i < 100000 else i*1.1 if i>350000 else i for i in kaggle_y_pred]

        else:
            y_pred = kaggle_y_pred

        id_col = X_prediction[[COLS.ID]]
        submission = id_col
        submission["SalePrice"] = y_pred
        print(submission)

        submission.to_csv("data/predict/output/kaggle_y_pred_XGBoost.csv", index=False)

def averaging_model():

    class AveragingModel(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, models):
            self.models = models

        def fit(self, X, y):
            # Create clone models
            self.models_ = [clone(x) for x in self.models]

            # Train cloned models
            for model in self.models_:
                model.fit(X, y)

            return self

        def predict(self, X):
            # Get predictions from trained clone models
            predictions = np.column_stack(
                [model.predict(X) for model in self.models_])

            # Return average predictions
            return np.mean(predictions, axis=1)