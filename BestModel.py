import mlflow
from data import get_client, set_to_dataframe, query_power, query_wind
from model import polynomial,randomfor,SVMRegression,NeuralNet
from model import join_df, handle_missing, filter_data,x_preprocess
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, explained_variance_score
import numpy as np

client = get_client()
power_set = query_power(client, days=90)
wind_set = query_wind(client, days=90)
power_df = set_to_dataframe(power_set)
wind_df = set_to_dataframe(wind_set)

joined_dfs = filter_data(handle_missing(join_df(power_df,wind_df)))


# Define models and parameters
models = {
    "PolynomialRegression": polynomial,
    # "RandomForest": randomfor,
    "SVR":SVMRegression,
    "Neural Networks":NeuralNet
}
parameters = {
    "PolynomialRegression": [{'degree': 2}, {'degree': 3}, {'degree': 4}],
    # "RandomForest":[{'n_estimators':100,'max_features':1.0},{'n_estimators':200,'max_features':1.0},
                   # {'n_estimators':100,'max_features':'sqrt'},{'n_estimators':200,'max_features':'sqrt'},
                   # {'n_estimators':100,'max_features':'log2'},{'n_estimators':200,'max_features':'log2'}],
    "SVR":[{'C':0.1},{'C':1},{'C':2},{'C':5}],
    "Neural Networks":[{'optimizer':'sgd'},{'optimizer':'adam'}]
}

X = joined_dfs.drop(['Total','time'],axis=1)
y = joined_dfs['Total'].values.reshape(-1,1)


def main():
    # Start an MLflow run
    mlflow.sklearn.autolog() # This is to help us track scikit learn metrics.
    mlflow.set_tracking_uri("http://127.0.0.1:5000") # We set the MLFlow UI to display in our local host.

    for model_name, model_func in models.items():
        experiment_name = f"{model_name}_Experiment"
        mlflow.set_experiment(experiment_name)

        param_str = '_'.join([f'{key}{value}' for key, value in parameters.items()])
        run_name = f"{model_name}_Run_{param_str}"
        for param in parameters[model_name]:
            with mlflow.start_run(run_name=run_name):
                # Log model name and parameters
                mlflow.log_param("model", model_name)
                for key, value in param.items():
                    mlflow.log_param(key, value)

                mse_score = []
                variance_score = []
                tscv = TimeSeriesSplit(n_splits=5)
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # Initialize and train the model
                    model = model_func(**param)
                    pipeline = Pipeline([
                                ('preprocessor',x_preprocess()),
                                ('model', model)
                            ])
                    pipeline.fit(X_train, y_train)

                    # Predict and evaluate
                    y_pred = pipeline.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred, squared=True)
                    explained_variance = explained_variance_score(y_test, y_pred)
                    mse_score.append(mse)
                    variance_score.append(explained_variance)

                # Calculate the average MSE over all folds
                average_mse = np.mean(mse_score)
                average_variance = np.mean(variance_score)

                # Log metrics
                mlflow.log_metric("Average MSE", average_mse)
                mlflow.log_metric("Average Explained Variance", average_variance)

                # End the run
                mlflow.end_run()


    
if __name__ == "__main__":
    main()
