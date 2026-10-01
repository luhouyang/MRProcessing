import pandas as pd
import numpy as np


def normalize(vectors):
    """Normalizes a batch of vectors. Input shape (N, 3)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vectors / norms


def vectors_to_angles_deg(vectors):
    """Converts (x, y, z) vectors to (azimuth, elevation) in degrees."""
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    azimuth = np.rad2deg(np.arctan2(x, z))
    elevation = np.rad2deg(np.arctan2(y, z))
    return azimuth, elevation


# ────────────────────────────────────────────────────────────
# Core Time Calculation Functions
# ────────────────────────────────────────────────────────────

def calculate_dt(df_in):
    """
    Calculates time delta (dt) in seconds and ISI in milliseconds.
    Returns a DataFrame with 'dt' and 'isi_ms'.
    """
    if df_in.empty:
        return pd.DataFrame()

    df_time = df_in.copy()
    df_time = df_time.sort_values(by='timestamp')
    df_time['dt'] = df_time['timestamp'].diff()
    df_time['dt'] = df_time['dt'].replace(0, np.nan)
    df_time.loc[df_time['dt'] < 0.001, 'dt'] = np.nan
    df_time['isi_ms'] = df_time['dt'] * 1000

    return df_time


def calculate_frequency_metrics(df_in):
    """
    Calculates the instantaneous data frequency (Hz) and ISI (ms).
    Returns a DataFrame with 'dt', 'isi_ms', and 'frequency_hz'.
    """
    df_freq = calculate_dt(df_in)

    if df_freq.empty:
        return pd.DataFrame()

    df_freq['frequency_hz'] = 1.0 / df_freq['dt']

    return df_freq


def filter_by_isi(df_in, threshold_ms=30):
    """
    Filters a DataFrame based on Inter-Sample Interval (ISI) in milliseconds.
    Keeps rows where ISI is <= threshold_ms.
    """
    if df_in.empty:
        return df_in

    df_calc = calculate_dt(df_in)

    if 'isi_ms' not in df_calc.columns:
        return df_in

    mask = (df_calc['isi_ms'] <= threshold_ms) | (df_calc['isi_ms'].isna())
    filtered_df = df_calc[mask].copy()

    return filtered_df


# ────────────────────────────────────────────────────────────
# Core Metric Calculation Functions
# ────────────────────────────────────────────────────────────

def calculate_head_movement_metrics(df_in):
    """
    Calculates head positional and angular velocity from the entire dataframe.
    """
    df_head = calculate_dt(df_in)

    if df_head.empty:
        return pd.DataFrame()

    df_head['head_dx'] = df_head['headX'].diff()
    df_head['head_dy'] = df_head['headY'].diff()
    df_head['head_dz'] = df_head['headZ'].diff()

    df_head['head_distance_m'] = np.sqrt(df_head['head_dx']**2 +
                                         df_head['head_dy']**2 +
                                         df_head['head_dz']**2)

    df_head['head_pos_velocity_mps'] = df_head['head_distance_m'] / df_head['dt']

    df_head['head_vel_x_mps'] = df_head['head_dx'] / df_head['dt']
    df_head['head_vel_y_mps'] = df_head['head_dy'] / df_head['dt']

    df_head['head_pos_velocity_xy_mps'] = np.sqrt(
        df_head['head_vel_x_mps']**2 + df_head['head_vel_y_mps']**2)

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
    Calculates the 3D distance and vector components from the eye origin
    to the gaze hit point.
    """
    if df_in.empty:
        return pd.DataFrame()

    df_dist = df_in.copy()

    df_dist['eye_gaze_vec_X'] = df_dist['globalX'] - df_dist['eyeOriginX']
    df_dist['eye_gaze_vec_Y'] = df_dist['globalY'] - df_dist['eyeOriginY']
    df_dist['eye_gaze_vec_Z'] = df_dist['globalZ'] - df_dist['eyeOriginZ']

    df_dist['eye_gaze_dist_3D'] = np.sqrt(
        df_dist['eye_gaze_vec_X']**2 +
        df_dist['eye_gaze_vec_Y']**2 +
        df_dist['eye_gaze_vec_Z']**2)

    return df_dist


def calculate_gaze_error_metrics(df_in):
    """
    Calculates per-point 3D angular error and 2D spatial error.
    """
    if df_in.empty:
        return pd.DataFrame()

    df_gaze = df_in.copy()

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

    df_gaze['distance_2d'] = np.sqrt(
        (df_gaze['globalX'] - df_gaze['globaltargetX'])**2 +
        (df_gaze['globalY'] - df_gaze['globaltargetY'])**2
    )

    df_gaze['block_id'] = (df_gaze['targetName']
                           != df_gaze['targetName'].shift()).cumsum()

    return df_gaze


def summarize_metrics(df, metrics):
    """
    Calculates mean, std, median, min, max, and 95th percentile
    for a list of metrics.
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


# ────────────────────────────────────────────────────────────
# Stable Window & Precision Functions
# ────────────────────────────────────────────────────────────

def find_stable_windows(df_gaze_with_errors, error_metric_col,
                        window_duration_ms=500):
    """
    Finds the most stable window (lowest mean error) within each block.
    Window size is calculated dynamically from timestamps to match
    the requested duration in milliseconds (default: 500 ms).

    This ensures the same temporal duration is analysed regardless
    of the eye tracker's sampling rate (e.g., 90 Hz vs 60 Hz).

    Args:
        df_gaze_with_errors: DataFrame with error metrics and timestamps.
        error_metric_col: Column name to minimise (e.g., 'angular_error_deg').
        window_duration_ms: Desired window duration in milliseconds.

    Returns:
        Concatenated DataFrame of all best windows (one per block).
    """
    if df_gaze_with_errors.empty:
        return pd.DataFrame()
    if error_metric_col not in df_gaze_with_errors.columns:
        return pd.DataFrame()

    all_best_windows_dfs = []

    for block_id, block_df in df_gaze_with_errors.groupby('block_id'):
        # ── Calculate dynamic window size from actual timestamps ──
        if 'timestamp' in block_df.columns and len(block_df) >= 2:
            dt_ms = block_df['timestamp'].diff().dropna() * 1000.0
            # Remove spurious values (debounced < 1 ms or > 1000 ms)
            dt_ms = dt_ms[(dt_ms >= 1.0) & (dt_ms <= 1000.0)]
            if len(dt_ms) > 0:
                median_dt_ms = dt_ms.median()
                window_size = max(2, int(round(
                    window_duration_ms / median_dt_ms)))
            else:
                window_size = len(block_df)
        else:
            window_size = len(block_df)

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
        # Store the actual window size used for this block
        selected_df['window_size_used'] = window_size
        all_best_windows_dfs.append(selected_df)

    if not all_best_windows_dfs:
        return pd.DataFrame()

    return pd.concat(all_best_windows_dfs)


def calculate_spatial_precision_metrics(df_stable_windows_all_blocks):
    """
    Calculates angular precision (RMS and STD from centroid) for each
    stable window (block). Precision is calculated in degrees.
    """
    if df_stable_windows_all_blocks.empty:
        return pd.DataFrame()
    if 'block_id' not in df_stable_windows_all_blocks.columns:
        return pd.DataFrame()

    all_precision_stats = []

    for block_id, block_df in df_stable_windows_all_blocks.groupby('block_id'):
        if block_df.empty or len(block_df) < 2:
            continue

        eye_origins = block_df[['eyeOriginX', 'eyeOriginY',
                                'eyeOriginZ']].values
        gaze_points = block_df[['globalX', 'globalY', 'globalZ']].values
        gaze_vectors = gaze_points - eye_origins

        norms = np.linalg.norm(gaze_vectors, axis=1)
        valid_indices = norms > 1e-9
        if np.sum(valid_indices) < 2:
            continue

        valid_gaze_vectors = gaze_vectors[valid_indices]
        gaze_vectors_norm = normalize(valid_gaze_vectors)

        mean_gaze_vector = np.mean(gaze_vectors_norm, axis=0)
        centroid_gaze_vector = normalize(mean_gaze_vector.reshape(1, 3))[0]

        dot_products = np.sum(gaze_vectors_norm * centroid_gaze_vector, axis=1)
        angular_distances_3d_deg = np.rad2deg(
            np.arccos(np.clip(dot_products, -1.0, 1.0)))

        azimuths_deg, elevations_deg = vectors_to_angles_deg(gaze_vectors_norm)

        centroid_az_deg = np.mean(azimuths_deg)
        centroid_el_deg = np.mean(elevations_deg)

        delta_az = azimuths_deg - centroid_az_deg
        delta_az = (delta_az + 180) % 360 - 180
        delta_el = elevations_deg - centroid_el_deg

        angular_distances_2d_deg = np.sqrt(delta_az**2 + delta_el**2)

        all_precision_stats.append({
            'block_id': block_id,
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


# ────────────────────────────────────────────────────────────
# Timeseries Helpers
# ────────────────────────────────────────────────────────────

def calculate_all_timeseries_metrics(df_in, has_valid_target):
    """
    Helper function to calculate all 4 key metrics for timeseries plots.
    """
    if df_in.empty:
        return pd.DataFrame()

    df_plot = df_in.copy()

    df_head = calculate_head_movement_metrics(df_plot)
    df_eye_gaze = calculate_eye_to_gaze_metrics(df_head)

    if has_valid_target:
        df_gaze_error = calculate_gaze_error_metrics(df_eye_gaze)
        return df_gaze_error
    else:
        df_eye_gaze['angular_error_deg'] = np.nan
        df_eye_gaze['distance_2d'] = np.nan
        return df_eye_gaze


def resample_timeseries_data(data_list, metrics, n_points=1000):
    """
    Resamples all timeseries data to a fixed length (n_points) for averaging.
    """
    resampled_data = {metric: [] for metric in metrics}
    all_data_raw = {metric: [] for metric in metrics}

    for item in data_list:
        df_raw = item['dataframe']
        has_valid_target = item['has_valid_target']

        df_metrics = calculate_all_timeseries_metrics(df_raw, has_valid_target)
        if df_metrics.empty:
            continue

        df_metrics['time_sec'] = df_metrics['timestamp'] - df_metrics[
            'timestamp'].iloc[0]

        df_metrics = df_metrics.drop_duplicates(subset='time_sec', keep='first')

        cols_to_keep = ['time_sec'] + [m for m in metrics
                                       if m in df_metrics.columns]
        df_metrics = df_metrics[cols_to_keep]
        df_metrics = df_metrics.set_index('time_sec')

        max_time = df_metrics.index.max()
        if max_time == 0:
            continue

        new_index = np.linspace(0, max_time, n_points)

        df_resampled = (df_metrics
                        .reindex(df_metrics.index.union(new_index))
                        .interpolate(method='index')
                        .loc[new_index])

        for metric in metrics:
            if metric in df_resampled.columns:
                metric_data_resampled = df_resampled[metric].values
                if pd.notna(metric_data_resampled).any():
                    resampled_data[metric].append(metric_data_resampled)

            if metric in df_metrics.columns:
                metric_data_raw = df_metrics[metric].dropna()
                if not metric_data_raw.empty:
                    all_data_raw[metric].append(metric_data_raw)

    plot_stats = {}
    for metric, data_arrays in resampled_data.items():
        if data_arrays:
            stacked_data = np.stack(data_arrays)
            plot_stats[metric] = {
                'mean': np.nanmean(stacked_data, axis=0),
                'std': np.nanstd(stacked_data, axis=0)
            }

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