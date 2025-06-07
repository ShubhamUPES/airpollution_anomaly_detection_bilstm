import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_prepare_data(path):
    """
    Loads data from CSV, parses dates, removes rows with missing PM2.5,
    and sorts data by City and Datetime.
    """
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.dropna(subset=['PM2.5'])
    df = df.sort_values(['City', 'Datetime'])
    return df

def get_normal_data(df_city):
    """
    Filters data points within 3 standard deviations of the PM2.5 mean,
    for outlier removal.
    """
    mu, sigma = df_city['PM2.5'].mean(), df_city['PM2.5'].std()
    return df_city[(df_city['PM2.5'] >= mu - 3*sigma) & (df_city['PM2.5'] <= mu + 3*sigma)]

def window_data(series, window_size=20):
    """
    Converts a time series into windowed samples for ML:
    X contains sequences of length window_size,
    y contains the next value after each sequence.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def prepare_city_data(df_city, window_size=20):
    """
    Scales PM2.5 values to [0,1] range and generates windowed data for modeling.
    Returns X, y, and the scaler for inverse transformations.
    """
    scaler = MinMaxScaler()
    values = scaler.fit_transform(df_city[['PM2.5']].values)
    X, y = window_data(values, window_size)
    return X, y, scaler

# Example usage:
# path = 'your_data.csv'
# df = load_and_prepare_data(path)
# city_name = 'Delhi'
# df_city = df[df['City'] == city_name]
# df_city_normal = get_normal_data(df_city)
# X, y, scaler = prepare_city_data(df_city_normal)
