
import pandas as pd
from typing import List, Tuple
import numpy as np
import os

def inject_unmetered_load(df: pd.DataFrame, feeders: List[str], intensity_levels: List[float], time_range: Tuple[pd.Timestamp, pd.Timestamp] = None) -> pd.DataFrame:
    """
    Increases total_consumption_active_import by a given percentage without changing aggregated_device_count_active.
    Applies uniformly across the selected time range.
    """
    df_copy = df.copy()
    for feeder in feeders:
        feeder_mask = df_copy['lv_feeder_unique_id'] == feeder
        if time_range:
            time_mask = (df_copy['data_collection_log_timestamp'] >= time_range[0]) & (df_copy['data_collection_log_timestamp'] <= time_range[1])
            target_mask = feeder_mask & time_mask
        else:
            target_mask = feeder_mask

        for intensity in intensity_levels:
            anomaly_id = f'{feeder}_unmetered_load_{intensity:.2f}'
            df_copy.loc[target_mask, 'total_consumption_active_import'] *= (1 + intensity)
            df_copy.loc[target_mask, 'anomaly_type'] = 'unmetered_load'
            df_copy.loc[target_mask, 'anomaly_intensity'] = intensity
            df_copy.loc[target_mask, 'anomaly_id'] = anomaly_id
    return df_copy

def inject_offpeak_exploitation(df: pd.DataFrame, feeders: List[str], intensity_levels: List[float], time_range: Tuple[pd.Timestamp, pd.Timestamp] = None) -> pd.DataFrame:
    """
    Within the hours 1:00 AM to 4:00 AM, increases total_consumption_active_import by the given intensity level.
    Other hours should remain untouched.
    """
    df_copy = df.copy()
    for feeder in feeders:
        feeder_mask = df_copy['lv_feeder_unique_id'] == feeder
        if time_range:
            time_mask = (df_copy['data_collection_log_timestamp'] >= time_range[0]) & (df_copy['data_collection_log_timestamp'] <= time_range[1])
            target_mask_base = feeder_mask & time_mask
        else:
            target_mask_base = feeder_mask

        # Ensure timestamp is datetime type for hour extraction
        df_copy['data_collection_log_timestamp'] = pd.to_datetime(df_copy['data_collection_log_timestamp'])
        offpeak_hours_mask = (df_copy['data_collection_log_timestamp'].dt.hour >= 1) & (df_copy['data_collection_log_timestamp'].dt.hour < 4)
        target_mask = target_mask_base & offpeak_hours_mask

        for intensity in intensity_levels:
            anomaly_id = f'{feeder}_offpeak_exploitation_{intensity:.2f}'
            df_copy.loc[target_mask, 'total_consumption_active_import'] *= (1 + intensity)
            df_copy.loc[target_mask, 'anomaly_type'] = 'offpeak_exploitation'
            df_copy.loc[target_mask, 'anomaly_intensity'] = intensity
            df_copy.loc[target_mask, 'anomaly_id'] = anomaly_id
    return df_copy


def inject_gradual_drift(df: pd.DataFrame, feeders: List[str], intensity_levels: List[float], time_range: Tuple[pd.Timestamp, pd.Timestamp] = None) -> pd.DataFrame:
    """
    Applies a linear upward slope to total_consumption_active_import over time.
    Intensity controls the total % increase over the period.
    """
    df_copy = df.copy()
    for feeder in feeders:
        feeder_mask = df_copy['lv_feeder_unique_id'] == feeder
        if time_range:
            time_mask = (df_copy['data_collection_log_timestamp'] >= time_range[0]) & (df_copy['data_collection_log_timestamp'] <= time_range[1])
            target_mask = feeder_mask & time_mask
        else:
            target_mask = feeder_mask

        for intensity in intensity_levels:
            anomaly_id = f'{feeder}_gradual_drift_{intensity:.2f}'
            # Calculate total number of points in the target mask for linear interpolation
            num_points = target_mask.sum()
            if num_points > 1:
                # Create a linear array from 0 to intensity over the target period
                linear_increase = np.linspace(0, intensity, num_points)
                # Apply the linear increase to the target values
                df_copy.loc[target_mask, 'total_consumption_active_import'] *= (1 + linear_increase)
                df_copy.loc[target_mask, 'anomaly_type'] = 'gradual_drift'
                df_copy.loc[target_mask, 'anomaly_intensity'] = intensity
                df_copy.loc[target_mask, 'anomaly_id'] = anomaly_id
            elif num_points == 1:
                 # If only one point, just apply the full intensity
                df_copy.loc[target_mask, 'total_consumption_active_import'] *= (1 + intensity)
                df_copy.loc[target_mask, 'anomaly_type'] = 'gradual_drift'
                df_copy.loc[target_mask, 'anomaly_intensity'] = intensity
                df_copy.loc[target_mask, 'anomaly_id'] = anomaly_id

    return df_copy


def inject_smoothed_profile(df: pd.DataFrame, feeders: List[str], intensity_levels: List[float], time_range: Tuple[pd.Timestamp, pd.Timestamp] = None) -> pd.DataFrame:
    """
    Adds a 24-hour sine wave to the active power profile, with amplitude based on intensity level.
    The sine wave should smooth out peaks and valleys while preserving overall load shape.
    Intensity controls the amplitude of the sine wave as a percentage of the local average.
    """
    df_copy = df.copy()
    for feeder in feeders:
        feeder_mask = df_copy['lv_feeder_unique_id'] == feeder
        if time_range:
            time_mask = (df_copy['data_collection_log_timestamp'] >= time_range[0]) & (df_copy['data_collection_log_timestamp'] <= time_range[1])
            target_mask = feeder_mask & time_mask
        else:
            target_mask = feeder_mask

        # Ensure timestamp is datetime type for hour extraction
        df_copy['data_collection_log_timestamp'] = pd.to_datetime(df_copy['data_collection_log_timestamp'])

        for intensity in intensity_levels:
            anomaly_id = f'{feeder}_smoothed_profile_{intensity:.2f}'
            # Calculate the local average for scaling the sine wave amplitude
            local_avg = df_copy.loc[target_mask, 'total_consumption_active_import'].mean()
            amplitude = local_avg * intensity

            # Generate a sine wave based on the time within the target range
            target_df = df_copy.loc[target_mask].copy()
            if not target_df.empty:
                start_time = target_df['data_collection_log_timestamp'].min()
                time_diff = (target_df['data_collection_log_timestamp'] - start_time).dt.total_seconds()
                # 24 hours in seconds = 24 * 3600
                sine_wave = amplitude * np.sin(2 * np.pi * time_diff / (24 * 3600))

                df_copy.loc[target_mask, 'total_consumption_active_import'] += sine_wave
                df_copy.loc[target_mask, 'anomaly_type'] = 'smoothed_profile'
                df_copy.loc[target_mask, 'anomaly_intensity'] = intensity
                df_copy.loc[target_mask, 'anomaly_id'] = anomaly_id
    return df_copy


def inject_flatline_consumption(df: pd.DataFrame, feeders: List[str], durations: List[int], time_range: Tuple[pd.Timestamp, pd.Timestamp] = None) -> pd.DataFrame:
    """
    Replaces total_consumption_active_import with a constant value (e.g., the day’s median or average) for each day.
    Uses durations (in days) to define how many consecutive days to flatten.
    """
    df_copy = df.copy()
    for feeder in feeders:
        feeder_mask = df_copy['lv_feeder_unique_id'] == feeder
        if time_range:
            time_mask = (df_copy['data_collection_log_timestamp'] >= time_range[0]) & (df_copy['data_collection_log_timestamp'] <= time_range[1])
            target_mask_base = feeder_mask & time_mask
        else:
            target_mask_base = feeder_mask

        # Ensure timestamp is datetime type for date extraction
        df_copy['data_collection_log_timestamp'] = pd.to_datetime(df_copy['data_collection_log_timestamp'])

        for duration in durations:
            anomaly_id = f'{feeder}_flatline_{duration}_days'
            target_df = df_copy.loc[target_mask_base].copy()

            if not target_df.empty:
                # Group by day to calculate daily median/average
                target_df['date'] = target_df['data_collection_log_timestamp'].dt.date
                daily_stats = target_df.groupby('date')['total_consumption_active_import'].median() # Using median as example

                dates_to_flatten = daily_stats.sample(n=min(duration, len(daily_stats))).index # Select random days to flatline for the duration

                for date_to_flatten in dates_to_flatten:
                    date_mask = df_copy['data_collection_log_timestamp'].dt.date == date_to_flatten
                    flatten_mask = target_mask_base & date_mask

                    if not df_copy.loc[flatten_mask].empty:
                        # Get the median value for the day to flatten
                        median_value = daily_stats.loc[date_to_flatten]
                        df_copy.loc[flatten_mask, 'total_consumption_active_import'] = median_value
                        df_copy.loc[flatten_mask, 'anomaly_type'] = 'flatline_consumption'
                        df_copy.loc[flatten_mask, 'anomaly_intensity'] = f'{duration} days'
                        df_copy.loc[flatten_mask, 'anomaly_id'] = anomaly_id

    return df_copy


if __name__ == "__main__":
    print(f"Anomaly simulation functions script saved to: {output_path}")
