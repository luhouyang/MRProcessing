import pandas as pd
import numpy as np


def normalize(vectors):
    """Normalizes a batch of vectors. Input shape (N, 3)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1e-9
    return vectors / norms


def vectors_to_angles_deg(vectors):
    """Converts (x, y, z) vectors to (azimuth, elevation) in degrees."""
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    # Note: Using arctan2(x, z) for azimuth assumes Z is forward, X is right.
    azimuth = np.rad2deg(np.arctan2(x, z))
    # elevation = np.rad2deg(np.arcsin(y))
    elevation = np.rad2deg(np.arctan2(y, z))
    return azimuth, elevation


# Core Time Calculation Functions


def calculate_dt(df_in):
    """
    Calculates time delta (dt) in seconds and ISI in milliseconds.
    Returns a DataFrame with 'dt' and 'isi_ms'.
    """
    if df_in.empty:
        return pd.DataFrame()

    df_time = df_in.copy()

    # Ensure timestamp is sorted
    df_time = df_time.sort_values(by='timestamp')

    # Calculate time delta (dt) in seconds (main unit)
    df_time['dt'] = df_time['timestamp'].diff()
    df_time['dt'] = df_time['dt'].replace(0, np.nan)

    # Calculate ISI in milliseconds (requested unit)
    df_time['isi_ms'] = df_time['dt'] * 1000

    return df_time


def calculate_frequency_metrics(df_in):
    """
    Calculates the instantaneous data frequency (Hz) and ISI (ms).
    Returns a DataFrame with 'dt', 'isi_ms', and 'frequency_hz'.
    """
    # Reuse the central time calculation
    df_freq = calculate_dt(df_in)

    if df_freq.empty:
        return pd.DataFrame()

    # Calculate instantaneous frequency (Hz = 1 / dt)
    df_freq['frequency_hz'] = 1.0 / df_freq['dt']

    return df_freq


# Core Metric Calculation Functions


def calculate_head_movement_metrics(df_in):
    """
    Calculates head positional and angular velocity from the entire dataframe.
    Returns a DataFrame with new columns for these metrics.
    """
    # Use the central time calculation
    df_head = calculate_dt(df_in)

    if df_head.empty:
        return pd.DataFrame()

    # 1. Positional Velocity (m/s)
    df_head['head_dx'] = df_head['headX'].diff()
    df_head['head_dy'] = df_head['headY'].diff()
    df_head['head_dz'] = df_head['headZ'].diff()

    df_head['head_distance_m'] = np.sqrt(df_head['head_dx']**2 +
                                         df_head['head_dy']**2 +
                                         df_head['head_dz']**2)

    # 3D Magnitude Velocity
    df_head[
        'head_pos_velocity_mps'] = df_head['head_distance_m'] / df_head['dt']

    # 2D Velocity components
    df_head['head_vel_x_mps'] = df_head['head_dx'] / df_head['dt']
    df_head['head_vel_y_mps'] = df_head['head_dy'] / df_head['dt']

    # 2D Magnitude Velocity (XY Plane)
    df_head['head_pos_velocity_xy_mps'] = np.sqrt(
        df_head['head_vel_x_mps']**2 + df_head['head_vel_y_mps']**2)

    # 2. Angular Velocity (deg/s)
    head_fwd_vectors = normalize(
        df_head[['headForwardX', 'headForwardY', 'headForwardZ']].values)
    head_fwd_vectors_shifted = df_head[[
        'headForwardX', 'headForwardY', 'headForwardZ'
    ]].shift(1).bfill()
    head_fwd_vectors_shifted = normalize(head_fwd_vectors_shifted.values)

    dot_prod = np.sum(head_fwd_vectors * head_fwd_vectors_shifted, axis=1)
    angle_deg = np.rad2deg(np.arccos(np.clip(dot_prod, -1.0, 1.0)))

    df_head['head_ang_velocity_dps'] = angle_deg / df_head['dt']

    return df_head


def calculate_eye_to_gaze_metrics(df_in):
    """
    Calculates the 3D distance and vector components from the eye origin to the gaze hit point.
    Returns a DataFrame with new columns for these metrics.
    """
    if df_in.empty:
        return pd.DataFrame()

    df_dist = df_in.copy()

    # Calculate the vector components (gaze - eye)
    df_dist['eye_gaze_vec_X'] = df_dist['globalX'] - df_dist['eyeOriginX']
    df_dist['eye_gaze_vec_Y'] = df_dist['globalY'] - df_dist['eyeOriginY']
    df_dist['eye_gaze_vec_Z'] = df_dist['globalZ'] - df_dist['eyeOriginZ']

    # Calculate the 3D distance magnitude
    df_dist['eye_gaze_dist_3D'] = np.sqrt(df_dist['eye_gaze_vec_X']**2 +
                                          df_dist['eye_gaze_vec_Y']**2 +
                                          df_dist['eye_gaze_vec_Z']**2)

    return df_dist


def calculate_gaze_error_metrics(df_in):
    """
    Calculates per-point 3D angular error and 2D spatial error.
    Returns a DataFrame with new columns for these metrics.
    """
    if df_in.empty:
        return pd.DataFrame()

    df_gaze = df_in.copy()

    # 3D Angular Error (Visual Angle)
    # The raw data should be used for calculation.
    gaze_vectors = df_gaze[['globalX', 'globalY', 'globalZ']].values - \
                   df_gaze[['eyeOriginX', 'eyeOriginY', 'eyeOriginZ']].values
    gaze_vectors_norm = normalize(gaze_vectors)

    target_vectors = df_gaze[['globaltargetX', 'globaltargetY', 'globaltargetZ']].values - \
                     df_gaze[['eyeOriginX', 'eyeOriginY', 'eyeOriginZ']].values
    target_vectors_norm = normalize(target_vectors)

    df_gaze['cosine_similarity'] = np.clip(
        np.sum(gaze_vectors_norm * target_vectors_norm, axis=1), -1.0, 1.0)
    df_gaze['angular_error_deg'] = np.rad2deg(
        np.arccos(df_gaze['cosine_similarity']))

    # 2D Distance Error (Spatial Units)
    df_gaze['distance_2d'] = np.sqrt(
        (df_gaze['globalX'] - df_gaze['globaltargetX'])**2 + \
        (df_gaze['globalY'] - df_gaze['globaltargetY'])**2
    )

    # Block ID
    # A new block is created every time the targetName changes
    df_gaze['block_id'] = (df_gaze['targetName']
                           != df_gaze['targetName'].shift()).cumsum()

    return df_gaze


def summarize_metrics(df, metrics):
    """
    Calculates mean, std, median, min, max, and 95th percentile for a list of metrics.
    Returns a dictionary of the results.
    """
    stats = {}
    if df.empty:
        return stats

    for metric in metrics:
        if metric in df.columns:
            data = df[metric].dropna()
            if not data.empty:
                stats[f'{metric}_mean'] = data.mean()
                stats[f'{metric}_std'] = data.std()
                stats[f'{metric}_median'] = data.median()
                stats[f'{metric}_min'] = data.min()
                stats[f'{metric}_max'] = data.max()
                stats[f'{metric}_q95'] = data.quantile(0.95)
    return stats


# Stable Window & Precision Functions


def find_stable_windows(df_gaze_with_errors, error_metric_col, window_size):
    """
    Finds the most stable window (lowest mean error) *within each block*.
    Returns a single concatenated DataFrame of all best windows.
    """
    if df_gaze_with_errors.empty or error_metric_col not in df_gaze_with_errors.columns:
        return pd.DataFrame()

    all_best_windows_dfs = []

    for block_id, block_df in df_gaze_with_errors.groupby('block_id'):
        if len(block_df) < window_size:
            continue

        rolling_mean_error = block_df[error_metric_col].rolling(
            window=window_size).mean()

        if rolling_mean_error.empty or rolling_mean_error.isnull().all():
            continue

        best_end_index = rolling_mean_error.idxmin()
        best_start_index = best_end_index - window_size + 1

        if pd.isna(best_end_index) or best_start_index < 0:
            continue

        selected_df = block_df.loc[best_start_index:best_end_index].copy()
        all_best_windows_dfs.append(selected_df)

    if not all_best_windows_dfs:
        return pd.DataFrame()

    return pd.concat(all_best_windows_dfs)


def calculate_spatial_precision_metrics(df_stable_windows_all_blocks):
    """
    Calculates angular precision (RMS and STD from centroid) for each stable window (block).
    Precision is calculated in degrees (visual angle).
    
    Returns a DataFrame with one row per block and columns for precision metrics.
    """
    if df_stable_windows_all_blocks.empty or 'block_id' not in df_stable_windows_all_blocks.columns:
        return pd.DataFrame()

    all_precision_stats = []

    for block_id, block_df in df_stable_windows_all_blocks.groupby('block_id'):
        if block_df.empty or len(
                block_df) < 2:  # Need at least 2 points to calculate std
            continue

        # 3D Angular Precision
        eye_origins = block_df[['eyeOriginX', 'eyeOriginY',
                                'eyeOriginZ']].values
        gaze_points = block_df[['globalX', 'globalY', 'globalZ']].values
        gaze_vectors = gaze_points - eye_origins

        # Filter out zero-length vectors (e.g., if gaze point == eye origin)
        norms = np.linalg.norm(gaze_vectors, axis=1)
        valid_indices = norms > 1e-9
        if np.sum(valid_indices) < 2:  # Need at least 2 valid vectors
            continue

        valid_gaze_vectors = gaze_vectors[valid_indices]
        gaze_vectors_norm = normalize(valid_gaze_vectors)

        # Calculate 3D angular centroid
        mean_gaze_vector = np.mean(gaze_vectors_norm, axis=0)
        centroid_gaze_vector = normalize(mean_gaze_vector.reshape(1, 3))[0]

        # Calculate angular distance from each point to the centroid
        dot_products = np.sum(gaze_vectors_norm * centroid_gaze_vector, axis=1)
        angular_distances_3d_deg = np.rad2deg(
            np.arccos(np.clip(dot_products, -1.0, 1.0)))

        # 2D Angular Precision (Azimuth/Elevation)
        # Convert normalized vectors to angles
        azimuths_deg, elevations_deg = vectors_to_angles_deg(gaze_vectors_norm)

        # Calculate 2D angular centroid (simple mean, assumes non-wrapping)
        centroid_az_deg = np.mean(azimuths_deg)
        centroid_el_deg = np.mean(elevations_deg)

        # Calculate 2D angular distance from each point to the 2D centroid
        # Handle azimuth wrap-around for distance calculation (e.g., -179 to +179 is 2 deg, not 358)
        delta_az = azimuths_deg - centroid_az_deg
        delta_az = (delta_az + 180) % 360 - 180
        delta_el = elevations_deg - centroid_el_deg

        angular_distances_2d_deg = np.sqrt(delta_az**2 + delta_el**2)

        all_precision_stats.append({
            'block_id':
            block_id,
            'precision_rms_3d':
            np.sqrt(np.mean(angular_distances_3d_deg**2)),
            'precision_std_3d':
            np.std(angular_distances_3d_deg),
            'precision_rms_2d':
            np.sqrt(np.mean(angular_distances_2d_deg**2)),
            'precision_std_2d':
            np.std(angular_distances_2d_deg)
        })

    if not all_precision_stats:
        return pd.DataFrame()

    return pd.DataFrame(all_precision_stats)


# Timeseries Helpers


def calculate_all_timeseries_metrics(df_in, has_valid_target):
    """
    Helper function to calculate all 4 key metrics for timeseries plots.
    Returns a single DataFrame.
    """
    if df_in.empty:
        return pd.DataFrame()

    df_plot = df_in.copy()

    # Calculate metrics
    # Head movement calc now includes 'dt' and 'isi_ms'
    df_head = calculate_head_movement_metrics(df_plot)
    df_eye_gaze = calculate_eye_to_gaze_metrics(df_head)

    if has_valid_target:
        df_gaze_error = calculate_gaze_error_metrics(df_eye_gaze)
        return df_gaze_error
    else:
        # Add empty columns so they exist for plotting
        df_eye_gaze['angular_error_deg'] = np.nan
        df_eye_gaze['distance_2d'] = np.nan
        return df_eye_gaze


def resample_timeseries_data(data_list, metrics, n_points=1000):
    """
    Resamples all timeseries data to a fixed length (n_points) for averaging.
    
    Args:
        data_list (list): List of dictionaries, e.g., [{'dataframe': df1, 'has_valid_target': True}, ...]
        metrics (list): List of column names to resample, e.g., ['angular_error_deg', 'head_pos_velocity_mps', ...]
        n_points (int): The number of points to resample to.

    Returns:
        tuple: (plot_stats, overall_stats)
            plot_stats (dict): Dictionary with mean/std for resampled data. e.g., {'angular_error_deg': {'mean': [..], 'std': [..]}}
            overall_stats (dict): Dictionary with single mean/std for *all* data. e.g., {'angular_error_deg': {'mean': 0.5, 'std': 0.1}}
    """

    resampled_data = {metric: [] for metric in metrics}
    all_data_raw = {metric: [] for metric in metrics}

    for item in data_list:
        df_raw = item['dataframe']
        has_valid_target = item['has_valid_target']

        # 1. Calculate all metrics for this df
        df_metrics = calculate_all_timeseries_metrics(df_raw, has_valid_target)
        if df_metrics.empty:
            continue

        # 2. Resample each metric
        df_metrics['time_sec'] = df_metrics['timestamp'] - df_metrics[
            'timestamp'].iloc[0]
        df_metrics = df_metrics.set_index('time_sec')

        # Create a new index from 0 to max_time with n_points
        max_time = df_metrics.index.max()
        if max_time == 0: continue  # Skip if no duration

        new_index = np.linspace(0, max_time, n_points)
        df_resampled = df_metrics.reindex(df_metrics.index.union(
            new_index)).interpolate('index').loc[new_index]

        # 3. Store results
        for metric in metrics:
            if metric in df_resampled.columns:
                metric_data_resampled = df_resampled[metric].values
                if pd.notna(metric_data_resampled).any():
                    resampled_data[metric].append(metric_data_resampled)

            if metric in df_metrics.columns:
                metric_data_raw = df_metrics[metric].dropna()
                if not metric_data_raw.empty:
                    all_data_raw[metric].append(metric_data_raw)

    # 4. Calculate summary stats from resampled data (for plots)
    plot_stats = {}
    for metric, data_arrays in resampled_data.items():
        if data_arrays:
            # Stack all arrays (n_sessions, n_points)
            stacked_data = np.stack(data_arrays)
            # Calculate mean/std along the session axis (axis=0)
            plot_stats[metric] = {
                'mean': np.nanmean(stacked_data, axis=0),
                'std': np.nanstd(stacked_data, axis=0)
            }

    # 5. Calculate overall summary stats from *all* raw data
    overall_stats = {}
    for metric, data_series_list in all_data_raw.items():
        if data_series_list:
            combined_series = pd.concat(data_series_list)
            if not combined_series.empty:
                overall_stats[metric] = {
                    'mean': combined_series.mean(),
                    'std': combined_series.std()
                }

    return plot_stats, overall_stats
