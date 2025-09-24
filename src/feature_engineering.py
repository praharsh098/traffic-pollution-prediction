from typing import Iterable


def add_time_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    return df


def add_lag_features(df, cols: Iterable[str], lags: Iterable[int] = (1, 2, 3)):
    for col in cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df.dropna(inplace=True)
    return df
