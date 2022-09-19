# VISUALIZE OVERFITTING
    print("...visualizing overfitting of min_samples_split parameter...")

    rf = RandomForestRegressor(min_samples_leaf=2)
    min_samples_split_i, train_loss, test_loss = [], [], []
    for iter in range(100):
        # fit
        rf.fit(X_train, y_train)

        # predict
        y_train_predicted = rf.predict(X_train)
        y_test_predicted = rf.predict(X_holdout)

        # evaluate
        score_r2_train = r2_score(y_train, y_train_predicted)
        adj_r2_train = 1 - (1-score_r2_train)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)

        score_r2_holdout = r2_score(y_holdout, y_test_predicted)
        adj_r2_holdout = 1 - (1-score_r2_train)*(len(y_holdout)-1)/(len(y_holdout)-X_train.shape[1]-1)

        min_samples_split_i.append(rf.min_samples_split)
        train_loss.append(adj_r2_train)
        test_loss.append(adj_r2_holdout)
        rf.min_samples_split += 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted(min_samples_split_i),
        y=sorted(train_loss),
        name="MSE on Train data"
    ))

    fig.add_trace(go.Scatter(
        x=sorted(min_samples_split_i),
        y=sorted(test_loss),
        name="MSE on Holdout data"
    ))

    fig.update_layout(title='Adj R2 on holdout and train set',
                      xaxis_title='min_samples_leaf',
                      yaxis_title='Adj. R2')

    fig.show()