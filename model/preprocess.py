import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def join_df(power_df,wind_df):
    joined_dfs = power_df.join(wind_df, how="left")
    joined_dfs = joined_dfs.drop(['ANM','Non-ANM','Lead_hours','Source_time'],axis=1)
    joined_dfs = joined_dfs.reset_index()
    return joined_dfs

def handle_missing(joined_dfs):
    joined_dfs['Speed'] = joined_dfs['Speed'].interpolate(method='linear')
    joined_dfs['Direction'] = joined_dfs['Direction'].fillna(method='ffill')  # Forward fill
    joined_dfs = joined_dfs.dropna(subset=['Direction'])
    return joined_dfs

def filter_data(joined_dfs):
    joined_dfs['time'] = pd.to_datetime(joined_dfs['time'])
    joined_dfs = joined_dfs[joined_dfs['time'].dt.minute.isin([0, 30])]
    return joined_dfs

def x_preprocess():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Speed']),
            ('cat', OneHotEncoder(), ['Direction'])
        ])
    return preprocessor