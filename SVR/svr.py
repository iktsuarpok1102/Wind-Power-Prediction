import mlflow
from data import get_client, set_to_dataframe, query_power, query_wind
from model import join_df, handle_missing, filter_data,x_preprocess
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.svm import SVR
import numpy as np

client = get_client()
power_set = query_power(client, days=90)
wind_set = query_wind(client, days=90)
power_df = set_to_dataframe(power_set)
wind_df = set_to_dataframe(wind_set)

joined_dfs = filter_data(handle_missing(join_df(power_df,wind_df)))


# Define models and parameters
model = SVR(C=5)

X = joined_dfs.drop(['Total','time'],axis=1)
y = joined_dfs['Total'].values.reshape(-1,1)


def main():
    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run():
        mse_score = []
        variance_score = []
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Initialize and train the model
            pipeline = Pipeline([
                ('preprocessor',x_preprocess()),
                ('model', model)])
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

        print(average_mse,average_variance)

        # Log parameters
        mlflow.log_param("C", 5)

        # Log metrics
        mlflow.log_metric("average_mse", average_mse)
        mlflow.log_metric("average_variance", average_variance)

        # Log the entire pipeline
        # mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="SVR")

    
if __name__ == "__main__":
    main()