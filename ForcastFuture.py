import mlflow
from data import get_client, set_to_dataframe, query_power, query_wind,query_future_wind,query_future_power
from model import polynomial,randomfor,SVMRegression,NeuralNet
from model import join_df, handle_missing, filter_data,x_preprocess
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, explained_variance_score
import pandas as pd

start_date = "2023-11-20 00:00:00"
end_date = "2023-11-25 00:00:00"

client = get_client()
forecasts = set_to_dataframe(query_future_wind(client,start_date=start_date,end_date=end_date))
max_source_time_per_group = forecasts.groupby('time')['Source_time'].transform('max')

# Filter the DataFrame to only include rows where 'Source_time' is equal to the max 'Source_time' in its group
newest_forecasts = forecasts[forecasts['Source_time'] == max_source_time_per_group].copy()

power_test = set_to_dataframe(query_future_power(client,start_date=start_date,end_date=end_date))
power_test = newest_forecasts.join(power_test, how="inner")['Total']

# Specify the model
logged_model = 'runs:/ad22b1f91f8c41a8bb1ce4e30601d95a/model'


def main(power_test):
    mlflow.sklearn.autolog() # This is to help us track scikit learn metrics.
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 

    # Use MLFlow to load the saved model
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Use the saved model to predict the power generation
    predictions = loaded_model.predict(pd.DataFrame(newest_forecasts))

    # We can transform the predictions array to a Pandas DataFrame
    predictions = pd.DataFrame(predictions, index=newest_forecasts.index, columns=["Power"])
    predictions = predictions.join(power_test,how='inner')["Power"]

    mse = mean_squared_error(power_test, predictions, squared=True)
    explained_variance = explained_variance_score(power_test,predictions)

    print(mse,explained_variance)


    
if __name__ == "__main__":
    main(power_test=power_test)