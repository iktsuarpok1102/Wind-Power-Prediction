from influxdb import InfluxDBClient
import pandas as pd


# Create an InfluxDB Client instance and select the orkney database
def get_client():
    settings = {
    'host': 'influxus.itu.dk',
    'port': 8086,
    'username': 'lsda',
    'password': 'icanonlyread'
    }
    client = InfluxDBClient(host=settings['host'], port=settings['port'], username=settings['username'], password=settings['password'])
    client.switch_database('orkney')
    return client

## Function to tranform the InfluxDB resulting set into a Dataframe
def set_to_dataframe(resulting_set):
    values = resulting_set.raw["series"][0]["values"]
    columns = resulting_set.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index

    return df

def query_power(client, days):
    power_set = client.query(
        "SELECT * FROM Generation where time > now()-"+str(days)+"d"
        ) # Query written in InfluxQL. We are retrieving all generation data from 90 days back.
    return power_set

    # Get the last 90 days of weather forecasts with the shortest lead time
def query_wind(client,days):
    wind_set  = client.query(
        "SELECT * FROM MetForecasts where time > now()-"+str(days)+"d and time <= now() and Lead_hours = '1'"
        ) # Query written in InfluxQL. We are retrieving all weather forecast data from 90 days back and with 1 lead hour.
    return wind_set

def query_future_power(client,start_date,end_date):
    more_power = client.query(
        f"SELECT * FROM Generation WHERE time >= '{start_date}' AND time < '{end_date}'"
        ) # Query written in InfluxQL. We are retrieving all generation data from 90 days back.
    return more_power

def query_future_wind(client,start_date,end_date):
    wind_forecasts  = client.query(
        f"SELECT * FROM MetForecasts WHERE time >= '{start_date}' AND time < '{end_date}'") # Query written in InfluxQL

    return wind_forecasts
