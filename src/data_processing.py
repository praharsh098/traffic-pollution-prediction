import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Normalize datetime column name
    if 'datetime' not in df.columns:
        if 'DateTime' in df.columns:
            df = df.rename(columns={'DateTime': 'datetime'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})

    # Normalize traffic volume column if present
    if 'traffic_volume' not in df.columns:
        if 'Vehicles' in df.columns:
            df = df.rename(columns={'Vehicles': 'traffic_volume'})
        elif 'traffic' in df.columns:
            df = df.rename(columns={'traffic': 'traffic_volume'})

    # Normalize pollution columns if present
    if 'pm25' not in df.columns:
        for candidate in ['PM2.5', 'PM2_5', 'PM25', 'PM2.5 Emissions', 'PM2.5 Emission', 'PM2.5_Emissions']:
            if candidate in df.columns:
                df = df.rename(columns={candidate: 'pm25'})
                break

    # Parse or synthesize datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        # If it's likely an emissions file (has pm25), try aligning to traffic datetime
        if 'pm25' in df.columns:
            data_dir = os.path.dirname(os.path.abspath(file_path))
            candidate_names = [
                'delhi_traffic.csv',
                'traffic.csv',
                'traffic_data.csv',
            ]
            traffic_path = None
            for name in candidate_names:
                path = os.path.join(data_dir, name)
                if os.path.exists(path):
                    traffic_path = path
                    break
            if traffic_path is not None:
                traffic_df = pd.read_csv(traffic_path)
                # Normalize and parse traffic datetime
                if 'datetime' not in traffic_df.columns:
                    if 'DateTime' in traffic_df.columns:
                        traffic_df = traffic_df.rename(columns={'DateTime': 'datetime'})
                if 'datetime' in traffic_df.columns:
                    traffic_df['datetime'] = pd.to_datetime(traffic_df['datetime'])
                    traffic_df.sort_values('datetime', inplace=True)
                    traffic_df.reset_index(drop=True, inplace=True)
                    length = min(len(df), len(traffic_df))
                    df = df.iloc[:length].copy()
                    df['datetime'] = traffic_df['datetime'].iloc[:length].values
        # If still no datetime, leave as-is; downstream merge will fail with clear error

    # Final sort/reset if datetime exists
    if 'datetime' in df.columns:
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Use forward fill without deprecated argument
    df.ffill(inplace=True)
    return df


def load_and_merge_two(traffic_path: str, pollution_path: str) -> pd.DataFrame:
    """Load traffic and pollution CSVs and merge on datetime (inner join).

    If the pollution data lacks a datetime column but contains a pm25 column,
    synthesize a datetime series aligned to the traffic timestamps (truncated to
    the shorter length).
    """
    traffic = load_data(traffic_path)
    pollution = load_data(pollution_path)

    if 'datetime' not in traffic.columns:
        raise KeyError("Traffic data must include a datetime column after normalization.")

    # If pollution has no datetime but has pm25, align to traffic timeline
    if 'datetime' not in pollution.columns and 'pm25' in pollution.columns:
        length = min(len(traffic), len(pollution))
        pollution = pollution.iloc[:length].copy()
        pollution['datetime'] = traffic['datetime'].iloc[:length].values
    elif 'datetime' not in pollution.columns:
        raise KeyError("Pollution data must include a datetime column or a recognizable pm25 column to align.")

    df = pd.merge(traffic, pollution, on='datetime', how='inner')
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
