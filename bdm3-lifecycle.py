import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_tracking_uri("http://training.itu.dk:5000/")
mlflow.set_experiment("yuji-BDM3")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="GBRT"):
    # TODO: Insert path to dataset
    path = "new_data.json"
    data = pd.read_json(path)
    X = data[["Speed","Direction"]]
    y = data["Total"]
    # TODO: Handle missing data
    numerical_cols = [cname for cname in X.columns 
                      if X[cname].dtype in ['int64', 'float64']]

    categorical_cols = [cname for cname in X.columns 
                        if X[cname].dtype not in ['int64', 'float64']]
    
    #handling missing data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('sc', StandardScaler())
    ])

#altering wind direction — OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    params = {'n_estimators': 4000, 'max_depth': 4, 'min_samples_split': 3,
              'learning_rate': 0.01, 'loss': 'squared_error'}
    gbrt_pipel = Pipeline(steps=[('preprocessor', preprocessor),
                              ('rf', GradientBoostingRegressor(**params))])
    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    metrics = [
        ("MAE", mean_absolute_error, [],[]),
        ("MSE", mean_squared_error, [],[]),
        ("R2", r2_score, [],[]),
    ]

    number_of_splits = 5

    #TODO: Log your parameters. What parameters are important to log?
    mlflow.log_param("data_file",path)
    mlflow.log_param("number_of_splits",number_of_splits)
    mlflow.log_param('n_estimators',4000)
    mlflow.log_param('max_depth', 4)
    mlflow.log_param('learning_rate', 0.01)
    mlflow.log_param('min_samples_split', 3)
    
    #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        gbrt_pipel.fit(X.iloc[train],y.iloc[train])
        predictions_train = gbrt_pipel.predict(X.iloc[train])
        predictions_test = gbrt_pipel.predict(X.iloc[test])
        truth_train = y.iloc[train]
        truth_test = y.iloc[test]

        from matplotlib import pyplot as plt 
        plt.plot(truth_test.index, truth_test.values, label="Truth")
        plt.plot(truth_test.index, predictions_test, label="Predictions")
        plt.show()
        
        # Calculate and save the metrics for this fold
        for name, func, scores_train, scores_test in metrics:
            score_train = func(truth_train, predictions_train)
            scores_train.append(score_train)
            score_test = func(truth_test, predictions_test)
            scores_test.append(score_test)
    
    # Log model
    #mlflow.sklearn.log_model(gbrt_pipel,'model')
    
    # Log a summary of the metrics
    for name, _, scores_train, scores_test in metrics:
            # NOTE: Here we just log the mean of the scores. 
            # Are there other summarizations that could be interesting?
            mean_score_train = sum(scores_train)/number_of_splits
            mean_score_test = sum(scores_test)/number_of_splits
            mlflow.log_metric(f"train_mean_{name}", mean_score_train)
            mlflow.log_metric(f"test_mean_{name}", mean_score_test)