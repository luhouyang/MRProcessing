import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import shutil
import open3d as o3d
# Import for statistical analysis
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
from statsmodels.sandbox.stats.multicomp import multipletests

import precision_toolkit as ptk

# Configuration
# SESSIONS_ROOT_DIR = r'C:\Users\luhou\Desktop\python\MRProcessing\data\experiment'
SESSIONS_ROOT_DIR = r"C:\Users\luhou\Desktop\python\MRProcessing\data\ARCHIVE\TEST6"

RESULTS_DIR = os.path.join(os.path.dirname(SESSIONS_ROOT_DIR), 'RESULTS')

# Folders to process within each session
FOLDERS_TO_ANALYZE = [
    'ObjectCone(1)', 'ObjectCone(3)', 'PlaneClose', 'PlaneMiddle', 'PlaneFar',
    'UD0411(83)', 'PlaneWSW'
]

# Window size (in samples) to use for finding the most stable fixation
# For a 1-1.5s trial, 25 samples (e.g., 20ms per sample ~ 500ms)
WINDOW_SIZE = 45

# DATA FREQUENCY ANALYSIS


def analyze_data_frequency(df_in, viz_output_folder):
    """
    Calculates and reports data acquisition frequency (Hz) and ISI (ms), 
    and generates a corresponding boxplot.
    """
    required_cols = ['timestamp']
    if not all(col in df_in.columns for col in required_cols):
        print(
            f"Warning: Skipping frequency analysis. Missing required column: {required_cols}",
            file=sys.stderr)
        return None

    # 1. Calculate Metrics (includes 'isi_ms' and 'frequency_hz')
    df_freq = ptk.calculate_frequency_metrics(df_in)
    if df_freq.empty:
        print("Warning: Frequency calculation returned no data.")
        return None

    # 2. Get Statistics for Report
    freq_data = df_freq['frequency_hz'].dropna()
    isi_data = df_freq['isi_ms'].dropna()

    if freq_data.empty:
        print("No frequency data to analyze.")
        return None

    stats = ptk.summarize_metrics(df_freq, ['frequency_hz', 'isi_ms'])

    # Calculate the theoretical mean frequency and ISI from the entire duration
    duration = df_freq['timestamp'].iloc[-1] - df_freq['timestamp'].iloc[0]
    total_samples = len(df_freq)

    samples_used = total_samples - 1  # The time delta calculation yields N-1 non-NaN values
    if duration > 0 and samples_used > 0:
        theoretical_dt = duration / samples_used
        theoretical_isi_ms = theoretical_dt * 1000
        theoretical_freq_hz = 1.0 / theoretical_dt
    else:
        theoretical_isi_ms = np.nan
        theoretical_freq_hz = np.nan

    # 3. Save Statistics Report
    report = f"Data Acquisition Frequency Analysis Report ({os.path.basename(viz_output_folder)})\n" \
             f"Total Samples: {total_samples}\n" \
             f"Duration: {duration:.2f} seconds\n" \
             f"Samples Used for Rate Calc: {samples_used}\n\n" \
             f"Inter-Sample Interval (ISI) (milliseconds)\n" \
             f"Mean (Theoretical): {theoretical_isi_ms:.3f}\n" \
             f"Mean (Actual):  {stats.get('isi_ms_mean', np.nan):.3f}\n" \
             f"StdDev: {stats.get('isi_ms_std', np.nan):.3f}\n" \
             f"Median: {stats.get('isi_ms_median', np.nan):.3f}\n" \
             f"Min:    {stats.get('isi_ms_min', np.nan):.3f}\n" \
             f"Max:    {stats.get('isi_ms_max', np.nan):.3f}\n" \
             f"95th %: {stats.get('isi_ms_q95', np.nan):.3f}\n\n" \
             f"Instantaneous Frequency (Hz)\n" \
             f"Mean (Theoretical): {theoretical_freq_hz:.2f}\n" \
             f"Mean (Actual):  {stats.get('frequency_hz_mean', np.nan):.2f}\n" \
             f"StdDev: {stats.get('frequency_hz_std', np.nan):.2f}\n" \
             f"Median: {stats.get('frequency_hz_median', np.nan):.2f}\n" \
             f"Max:    {stats.get('frequency_hz_max', np.nan):.2f}\n" \
             f"Min:    {stats.get('frequency_hz_min', np.nan):.2f}\n" \
             f"95th %: {stats.get('frequency_hz_q95', np.nan):.2f}\n"

    report_filename = os.path.join(viz_output_folder,
                                   'analysis_report_frequency.txt')
    try:
        with open(report_filename, 'w') as f:
            f.write(report)
    except Exception as e:
        print(f"Error saving frequency report: {e}", file=sys.stderr)

    # 4. Save Box Plot (ISI in ms, Frequency in Hz)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(
        f'Data Acquisition Rate Analysis - {os.path.basename(viz_output_folder)}',
        y=1.02)

    # Plot 1: Inter-Sample Interval (ISI) in milliseconds
    if not isi_data.empty:
        ax1.boxplot(isi_data,
                    vert=False,
                    whis=[5, 95],
                    showfliers=False,
                    showmeans=True)
        mean_val = np.nanmean(isi_data)
        ax1.text(mean_val,
                 1.1,
                 f'Mean: {mean_val:.2f} ms',
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 color='blue',
                 fontsize=9,
                 fontweight='bold')
        ax1.set_title('Inter-Sample Interval (ISI) [5th-95th Percentile]')
        ax1.set_xlabel('Time (milliseconds)'), ax1.set_yticklabels(
            ['ISI (ms)'])
    else:
        ax1.set_title('No ISI Data')

    # Plot 2: Instantaneous Frequency (Hz)
    if not freq_data.empty:
        ax2.boxplot(freq_data,
                    vert=False,
                    whis=[5, 95],
                    showfliers=False,
                    showmeans=True)
        mean_val = np.nanmean(freq_data)
        ax2.text(mean_val,
                 1.1,
                 f'Mean: {mean_val:.1f} Hz',
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 color='blue',
                 fontsize=9,
                 fontweight='bold')
        ax2.set_title('Instantaneous Frequency [5th-95th Percentile]')
        ax2.set_xlabel('Rate (Hz)'), ax2.set_yticklabels(['Frequency (Hz)'])
    else:
        ax2.set_title('No Frequency Data')

    plt.tight_layout()
    plot_filename = os.path.join(viz_output_folder,
                                 'analysis_plot_frequency_boxplot.png')
    try:
        plt.savefig(plot_filename)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving frequency box plot: {e}", file=sys.stderr)

    return {
        'type': 'frequency',
        'freq_mean': stats.get('frequency_hz_mean'),
        'freq_std': stats.get('frequency_hz_std'),
        'isi_mean': stats.get('isi_ms_mean'),
        'isi_std': stats.get('isi_ms_std')
    }


# 3D PLOTTING FUNCTIONS


def generate_open3d_geometries(selected_df, block_id, viz_output_folder):
    """
    Generates and saves Open3D geometries for a single block's best window.
    Files are named with the block_id.
    Uses GLOBAL coordinates.
    """
    # Get the target name for a cleaner message
    target_name = selected_df['targetName'].iloc[0]

    # Get average positions from the selected window
    # Use global target coordinates
    avg_origin = selected_df[['eyeOriginX', 'eyeOriginY',
                              'eyeOriginZ']].mean().values
    avg_target = selected_df[[
        'globaltargetX', 'globaltargetY', 'globaltargetZ'
    ]].mean().values
    target_dist = np.linalg.norm(avg_target - avg_origin)

    if target_dist == 0:
        print(
            f"Warning: Target distance is 0 for Block {block_id}. Using 1.0.")
        target_dist = 1.0  # Use a fallback distance

    # Define a small radius for spheres, relative to the target distance
    sphere_radius = target_dist * 0.02  # 2% of distance
    if sphere_radius == 0: sphere_radius = 0.01  # Failsafe

    # 1. Create Gaze Cone (LineSet)
    # Use eyeDirection from data
    gaze_vec_sel = ptk.normalize(
        selected_df[['eyeDirectionX', 'eyeDirectionY',
                     'eyeDirectionZ']].values)
    gaze_endpoints = [avg_origin + vec * target_dist for vec in gaze_vec_sel]
    gaze_points = np.vstack([avg_origin] + gaze_endpoints)
    gaze_lines_indices = [[0, i + 1] for i in range(len(gaze_endpoints))]
    gaze_colors = [[0, 0.8, 0]
                   for _ in range(len(gaze_lines_indices))]  # Green

    gaze_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(gaze_points),
        lines=o3d.utility.Vector2iVector(gaze_lines_indices))
    gaze_lineset.colors = o3d.utility.Vector3dVector(gaze_colors)
    o3d.io.write_line_set(
        os.path.join(viz_output_folder, f"gaze_cone_block_{block_id}.ply"),
        gaze_lineset)

    # 2. Create Target Vector (LineSet)
    target_points_viz = np.vstack([avg_origin, avg_target])
    target_lines_indices = [[0, 1]]
    target_colors = [[1, 0, 0]]  # Red

    target_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(target_points_viz),
        lines=o3d.utility.Vector2iVector(target_lines_indices))
    target_lineset.colors = o3d.utility.Vector3dVector(target_colors)
    o3d.io.write_line_set(
        os.path.join(viz_output_folder, f"target_vector_block_{block_id}.ply"),
        target_lineset)

    # 3. Create Eye/Target Spheres (TriangleMesh)
    eye_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    eye_sphere.paint_uniform_color([0, 0, 1])  # Blue
    eye_sphere.translate(avg_origin)
    o3d.io.write_triangle_mesh(
        os.path.join(viz_output_folder, f"eye_origin_block_{block_id}.ply"),
        eye_sphere)

    target_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=sphere_radius)
    target_sphere.paint_uniform_color([1, 0, 0])  # Red
    target_sphere.translate(avg_target)
    o3d.io.write_triangle_mesh(
        os.path.join(viz_output_folder, f"target_point_block_{block_id}.ply"),
        target_sphere)

    return gaze_lineset


# ANALYSIS & PLOTTING FUNCTIONS


def analyze_head_movement(df_in, viz_output_folder):
    """
    Calculates head movement stats using the toolkit, then generates reports and plots.
    """
    required_cols = [
        'headX', 'headY', 'headZ', 'headForwardX', 'headForwardY',
        'headForwardZ', 'timestamp'
    ]
    if not all(col in df_in.columns for col in required_cols):
        print(
            f"Warning: Skipping head movement analysis. Missing one or more columns: {required_cols}",
            file=sys.stderr)
        return None

    # 1. Calculate Metrics
    df_head = ptk.calculate_head_movement_metrics(df_in)
    if df_head.empty:
        print("Warning: Head movement calculation returned no data.")
        return None

    # 2. Get Statistics for Report
    pos_vel_data = df_head['head_pos_velocity_mps'].dropna()
    ang_vel_data = df_head['head_ang_velocity_dps'].dropna()
    vel_xy_data = df_head['head_pos_velocity_xy_mps'].dropna()

    if pos_vel_data.empty and ang_vel_data.empty:
        print("No head velocity data to analyze.")
        return None

    stats_3d = ptk.summarize_metrics(df_head, ['head_pos_velocity_mps'])
    stats_2d = ptk.summarize_metrics(df_head, ['head_pos_velocity_xy_mps'])
    stats_ang = ptk.summarize_metrics(df_head, ['head_ang_velocity_dps'])

    # 3. Save Statistics Report
    report = f"Head Movement Analysis Report ({os.path.basename(viz_output_folder)})\n" \
             f"Total Samples: {len(df_head)}\n" \
             f"Duration: {df_head['timestamp'].iloc[-1] - df_head['timestamp'].iloc[0]:.2f} seconds\n\n" \
             f"Head Positional Velocity (m/s) (3D Magnitude)\n" \
             f"Mean:   {stats_3d.get('head_pos_velocity_mps_mean', np.nan):.4f}\n" \
             f"StdDev: {stats_3d.get('head_pos_velocity_mps_std', np.nan):.4f}\n" \
             f"Median: {stats_3d.get('head_pos_velocity_mps_median', np.nan):.4f}\n" \
             f"Max:    {stats_3d.get('head_pos_velocity_mps_max', np.nan):.4f}\n" \
             f"95th %: {stats_3d.get('head_pos_velocity_mps_q95', np.nan):.4f}\n\n" \
             f"Head Positional Velocity (m/s) (2D XY Magnitude - Overall)\n" \
             f"Mean:   {stats_2d.get('head_pos_velocity_xy_mps_mean', np.nan):.4f}\n" \
             f"StdDev: {stats_2d.get('head_pos_velocity_xy_mps_std', np.nan):.4f}\n\n" \
             f"Head Angular Velocity (deg/s)\n" \
             f"Mean:   {stats_ang.get('head_ang_velocity_dps_mean', np.nan):.4f}\n" \
             f"StdDev: {stats_ang.get('head_ang_velocity_dps_std', np.nan):.4f}\n" \
             f"Median: {stats_ang.get('head_ang_velocity_dps_median', np.nan):.4f}\n" \
             f"Max:    {stats_ang.get('head_ang_velocity_dps_max', np.nan):.4f}\n" \
             f"95th %: {stats_ang.get('head_ang_velocity_dps_q95', np.nan):.4f}\n"

    report_filename = os.path.join(viz_output_folder,
                                   'analysis_report_head_movement.txt')
    try:
        with open(report_filename, 'w') as f:
            f.write(report)
    except Exception as e:
        print(f"Error saving head movement report: {e}", file=sys.stderr)

    # 4. Save Box Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(
        f'Head Movement Velocity - {os.path.basename(viz_output_folder)}',
        y=1.02)

    if not pos_vel_data.empty:
        ax1.boxplot(pos_vel_data,
                    vert=False,
                    whis=[5, 95],
                    showfliers=False,
                    showmeans=True)
        mean_val = np.nanmean(pos_vel_data)
        ax1.text(mean_val,
                 1.1,
                 f'Mean: {mean_val:.3f}',
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 color='blue',
                 fontsize=9,
                 fontweight='bold')
        ax1.set_title('Head Positional Velocity (m/s) [5th-95th Percentile]')
        ax1.set_xlabel('Velocity (m/s)')
        ax1.set_yticklabels(['Positional (3D)'])
    else:
        ax1.set_title('No Positional Velocity Data')

    if not ang_vel_data.empty:
        ax2.boxplot(ang_vel_data,
                    vert=False,
                    whis=[5, 95],
                    showfliers=False,
                    showmeans=True)
        mean_val = np.nanmean(ang_vel_data)
        ax2.text(mean_val,
                 1.1,
                 f'Mean: {mean_val:.3f}',
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 color='blue',
                 fontsize=9,
                 fontweight='bold')
        ax2.set_title('Head Angular Velocity (deg/s) [5th-95th Percentile]')
        ax2.set_xlabel('Velocity (deg/s)')
        ax2.set_yticklabels(['Angular'])
    else:
        ax2.set_title('No Angular Velocity Data')

    plt.tight_layout()
    plot_filename = os.path.join(viz_output_folder,
                                 'analysis_plot_head_movement_boxplot.png')
    try:
        plt.savefig(plot_filename)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving head movement box plot: {e}", file=sys.stderr)

    # 5. Head Drift Path Plot
    try:
        CHUNK_SIZE = 10
        path_chunks = [
        ]  # Store (x_start, y_start, z_start, x_end, y_end, z_end, velocity)

        for i in range(0, len(df_head) - CHUNK_SIZE, CHUNK_SIZE):
            start_index = i
            end_index = i + CHUNK_SIZE - 1
            start_row = df_head.iloc[start_index]
            end_row = df_head.iloc[end_index]
            dt = end_row['timestamp'] - start_row['timestamp']

            if pd.isna(dt) or dt == 0: continue

            x_start, y_start, z_start = start_row['headX'], start_row[
                'headY'], start_row['headZ']
            x_end, y_end, z_end = end_row['headX'], end_row['headY'], end_row[
                'headZ']
            distance = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2 +
                               (z_end - z_start)**2)
            velocity = distance / dt

            if pd.notna(velocity):
                path_chunks.append(
                    (x_start, y_start, z_start, x_end, y_end, z_end, velocity))

        if not path_chunks:
            print("Warning: No data for head path plot.")
            return {
                'type': 'head',
                'pos_vel_mean': stats_3d.get('head_pos_velocity_mps_mean'),
                'pos_vel_std': stats_3d.get('head_pos_velocity_mps_std'),
                'pos_vel_median': stats_3d.get('head_pos_velocity_mps_median'),
                'ang_vel_mean': stats_ang.get('head_ang_velocity_dps_mean'),
                'ang_vel_std': stats_ang.get('head_ang_velocity_dps_std'),
                'ang_vel_median': stats_ang.get('head_ang_velocity_dps_median')
            }

        all_x = [p[0] for p in path_chunks] + [p[3] for p in path_chunks]
        all_y = [p[1] for p in path_chunks] + [p[4] for p in path_chunks]
        all_z = [p[2] for p in path_chunks] + [p[5] for p in path_chunks]
        vel_vals = [p[6] for p in path_chunks]
        min_vel, max_vel = np.min(vel_vals), np.max(vel_vals)

        fig_drift = plt.figure(figsize=(10, 10))
        ax_drift = fig_drift.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=min_vel, vmax=max_vel)
        ax_drift.view_init(elev=20, azim=60, vertical_axis='y')

        for x_start, y_start, z_start, x_end, y_end, z_end, vel in path_chunks:
            color = cmap(norm(vel))
            ax_drift.plot([x_start, x_end], [y_start, y_end], [z_start, z_end],
                          color=color,
                          alpha=0.7,
                          linewidth=2)

        first_x, first_y, first_z = path_chunks[0][0], path_chunks[0][
            1], path_chunks[0][2]
        ax_drift.plot(
            [first_x], [first_y], [first_z],
            'r+',
            markersize=15,
            label=
            f'Start Position ({first_x:.2f}, {first_y:.2f}, {first_z:.2f})')

        fig_drift.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                           ax=ax_drift,
                           label='Average Velocity (m/s)',
                           shrink=0.8)

        ax_drift.set_title(
            f'Head Movement Path 3D ({CHUNK_SIZE}-sample chunks) - {os.path.basename(viz_output_folder)}'
        )
        ax_drift.set_xlabel('Head X Position (m)'), ax_drift.set_ylabel(
            'Head Y Position (m)'), ax_drift.set_zlabel('Head Z Position (m)')
        ax_drift.legend()

        min_x, max_x = np.min(all_x), np.max(all_x)
        min_y, max_y = np.min(all_y), np.max(all_y)
        min_z, max_z = np.min(all_z), np.max(all_z)
        max_range = np.array([max_x - min_x, max_y - min_y,
                              max_z - min_z]).max()
        if max_range == 0: max_range = 1.0
        max_range /= 2.0
        mid_x, mid_y, mid_z = (max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (
            max_z + min_z) / 2.0

        ax_drift.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_drift.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_drift.set_zlim(mid_z - max_range, mid_z + max_range)

        plot_drift_filename = os.path.join(
            viz_output_folder, 'analysis_plot_head_movement_path_3D.png')
        plt.savefig(plot_drift_filename)
        plt.close(fig_drift)

    except Exception as e:
        print(f"Error generating head movement 3D path plot: {e}",
              file=sys.stderr)

    # Return key statistics
    return {
        'type': 'head',
        'pos_vel_mean': stats_3d.get('head_pos_velocity_mps_mean'),
        'pos_vel_std': stats_3d.get('head_pos_velocity_mps_std'),
        'pos_vel_median': stats_3d.get('head_pos_velocity_mps_median'),
        'ang_vel_mean': stats_ang.get('head_ang_velocity_dps_mean'),
        'ang_vel_std': stats_ang.get('head_ang_velocity_dps_std'),
        'ang_vel_median': stats_ang.get('head_ang_velocity_dps_median')
    }


def analyze_eye_to_gaze_distance(df_in, viz_output_folder):
    """
    Calculates eye-to-gaze stats using the toolkit, then generates reports and plots.
    """
    required_cols = [
        'eyeOriginX', 'eyeOriginY', 'eyeOriginZ', 'globalX', 'globalY',
        'globalZ'
    ]
    if not all(col in df_in.columns for col in required_cols):
        print(
            f"Warning: Skipping eye-to-gaze-hit analysis. Missing one or more columns: {required_cols}",
            file=sys.stderr)
        return None

    # 1. Calculate Metrics
    df_dist = ptk.calculate_eye_to_gaze_metrics(df_in)
    if df_dist.empty:
        print("Warning: Eye-to-gaze calculation returned no data.")
        return None

    # 2. Get Statistics for Report
    data_dist_3d = df_dist['eye_gaze_dist_3D'].dropna()
    if data_dist_3d.empty:
        print("Warning: No valid eye-to-gaze-hit distance data found.")
        return None

    stats = ptk.summarize_metrics(df_dist, ['eye_gaze_dist_3D'])

    # 3. Save Statistics Report
    report = f"Eye-to-Gaze-Hit Distance Analysis Report ({os.path.basename(viz_output_folder)})\n" \
             f"Total Samples: {len(data_dist_3d)}\n\n" \
             f"3D Distance (Gaze Hit - Eye Origin) (meters):\n" \
             f"Mean:   {stats.get('eye_gaze_dist_3D_mean', np.nan):.4f}\n" \
             f"StdDev: {stats.get('eye_gaze_dist_3D_std', np.nan):.4f}\n" \
             f"Median: {stats.get('eye_gaze_dist_3D_median', np.nan):.4f}\n" \
             f"Min:    {stats.get('eye_gaze_dist_3D_min', np.nan):.4f}\n" \
             f"Max:    {stats.get('eye_gaze_dist_3D_max', np.nan):.4f}\n"

    report_filename = os.path.join(viz_output_folder,
                                   'analysis_report_eye_gaze_distance.txt')
    try:
        with open(report_filename, 'w') as f:
            f.write(report)
    except Exception as e:
        print(f"Error saving eye-to-gaze-hit report: {e}", file=sys.stderr)

    # 4. Generate Bullseye + Scatter Plot
    try:
        data_x = df_dist['eye_gaze_vec_X'].dropna()
        data_y = df_dist['eye_gaze_vec_Y'].dropna()
        mean_dist_3d = stats.get('eye_gaze_dist_3D_mean', 0)
        std_dist_3d = stats.get('eye_gaze_dist_3D_std', 0)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(data_x,
                   data_y,
                   alpha=0.1,
                   s=10,
                   label='Eye-to-Gaze-Hit Vector (XY Plane)')
        ax.plot(0, 0, 'r+', markersize=10, label='Eye Origin (0,0)')

        circle_mean = plt.Circle((0, 0),
                                 mean_dist_3d,
                                 color='r',
                                 fill=False,
                                 linewidth=2,
                                 label=f'Mean 3D Dist: {mean_dist_3d:.2f}m')
        circle_std_plus = plt.Circle((0, 0),
                                     mean_dist_3d + std_dist_3d,
                                     color='k',
                                     fill=False,
                                     linestyle='--',
                                     alpha=0.7,
                                     label=f'Mean +/- 1 STD')
        ax.add_patch(circle_mean)
        ax.add_patch(circle_std_plus)
        if (mean_dist_3d - std_dist_3d) > 0:
            circle_std_minus = plt.Circle((0, 0),
                                          mean_dist_3d - std_dist_3d,
                                          color='k',
                                          fill=False,
                                          linestyle='--',
                                          alpha=0.7)
            ax.add_patch(circle_std_minus)

        ax.set_title(
            f'Eye-to-Gaze-Hit Vector (XY) & 3D Distance - {os.path.basename(viz_output_folder)}'
        )
        ax.set_xlabel('X Distance (Gaze - Eye Origin) (m)'), ax.set_ylabel(
            'Y Distance (Gaze - Eye Origin) (m)')
        ax.axis('equal'), ax.grid(True, linestyle='--', alpha=0.6), ax.legend()

        plot_filename = os.path.join(viz_output_folder,
                                     'analysis_plot_eye_gaze_bullseye.png')
        plt.savefig(plot_filename)
        plt.close(fig)

    except Exception as e:
        print(f"Error generating eye-to-gaze-hit bullseye plot: {e}",
              file=sys.stderr)

    # Return key statistics
    return {
        'type': 'eye_gaze_dist',
        'eye_gaze_dist_mean': stats.get('eye_gaze_dist_3D_mean'),
        'eye_gaze_dist_std': stats.get('eye_gaze_dist_3D_std'),
        'eye_gaze_dist_median': stats.get('eye_gaze_dist_3D_median')
    }


def analyze_gaze_data_3D(df_gaze_in,
                         output_path,
                         model_obj_path,
                         viz_output_folder,
                         window_size=30):
    """
    Performs 3D analysis for saccade tasks.
    Uses toolkit to calculate metrics and find windows, then generates plots/reports.
    """
    folder_basename = os.path.basename(output_path)

    # 1. Calculate Per-Point Metrics
    df_gaze = ptk.calculate_gaze_error_metrics(df_gaze_in)

    # 2. Find Best Window *PER BLOCK*
    overall_best_df = ptk.find_stable_windows(df_gaze, 'angular_error_deg',
                                              window_size)

    if overall_best_df.empty:
        print(
            f"Error: Could not find any valid windows for {folder_basename}. Aborting analysis.",
            file=sys.stderr)
        return None

    n_windows = overall_best_df['block_id'].nunique()

    # Calculate spatial precision for each block
    df_precision = ptk.calculate_spatial_precision_metrics(overall_best_df)

    # 3. Generate Open3D files
    all_gaze_linesets = []
    overall_best_df_viz = overall_best_df.copy()

    for block_id, block_df in overall_best_df_viz.groupby('block_id'):
        gaze_lineset = generate_open3d_geometries(block_df, block_id,
                                                  viz_output_folder)
        if gaze_lineset:
            all_gaze_linesets.append(gaze_lineset)

    if all_gaze_linesets:
        combined_lineset = o3d.geometry.LineSet()
        for lineset in all_gaze_linesets:
            combined_lineset += lineset
        o3d.io.write_line_set(
            os.path.join(viz_output_folder, "gaze_cone_ALL_COMBINED.ply"),
            combined_lineset)

    # 4. Matplotlib Plots & Report
    # 4a. Get Stats & Report (Based on all best windows)
    stats = ptk.summarize_metrics(
        overall_best_df,
        ['cosine_similarity', 'angular_error_deg', 'distance_2d'])
    # Get stats for precision (mean of all blocks' precision)
    if df_precision is not None and not df_precision.empty:
        precision_stats = ptk.summarize_metrics(df_precision, [
            'precision_rms_3d', 'precision_std_3d', 'precision_rms_2d',
            'precision_std_2d'
        ])
    else:
        precision_stats = {}  # Create an empty dict if df_precision is None
    stats.update(precision_stats)  # Add precision stats

    report = f"3D Gaze Analysis Report ({folder_basename})\n" \
             f"Task Mode: Random Saccades (Per-Block Analysis)\n" \
             f"Coordinates: GLOBAL\n" \
             f"Windows Found: {n_windows} stable windows (one per block)\n" \
             f"Window Size: {window_size} samples\n" \
             f"Total Points in Windows: {len(overall_best_df)}\n\n" \
             f"METRICS (from all stable windows)\n" \
             f"Cosine Similarity:\n" \
             f"   Mean:   {stats.get('cosine_similarity_mean', np.nan):.4f}\n" \
             f"   Median: {stats.get('cosine_similarity_median', np.nan):.4f}\n" \
             f"   Min:    {stats.get('cosine_similarity_min', np.nan):.4f}\n" \
             f"   Max:    {stats.get('cosine_similarity_max', np.nan):.4f}\n\n" \
             f"Angular Error (degrees):\n" \
             f"   Mean:   {stats.get('angular_error_deg_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('angular_error_deg_std', np.nan):.4f}\n" \
             f"   Median: {stats.get('angular_error_deg_median', np.nan):.4f}\n" \
             f"   Min:    {stats.get('angular_error_deg_min', np.nan):.4f}\n" \
             f"   Max:    {stats.get('angular_error_deg_max', np.nan):.4f}\n\n" \
             f"2D Distance Error (units):\n" \
             f"   Mean:   {stats.get('distance_2d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('distance_2d_std', np.nan):.4f}\n" \
             f"   Median: {stats.get('distance_2d_median', np.nan):.4f}\n" \
             f"   Min:    {stats.get('distance_2d_min', np.nan):.4f}\n" \
             f"   Max:    {stats.get('distance_2d_max', np.nan):.4f}\n\n" \
             f"Angular Precision 3D (RMS from centroid, mean of {n_windows} blocks) (deg):\n" \
             f"   Mean:   {stats.get('precision_rms_3d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('precision_rms_3d_std', np.nan):.4f}\n\n" \
             f"Angular Precision 3D (STD from centroid, mean of {n_windows} blocks) (deg):\n" \
             f"   Mean:   {stats.get('precision_std_3d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('precision_std_3d_std', np.nan):.4f}\n"

    report_filename = os.path.join(viz_output_folder, 'analysis_report_3D.txt')
    with open(report_filename, 'w') as f:
        f.write(report)

    # 4b. Gaze Error Box Plots (5-panel)
    fig_box, (ax_box1, ax_box2, ax_box3,
              ax_box4, ax_box5) = plt.subplots(5,
                                               1,
                                               figsize=(10, 18),
                                               sharex=False)  # Changed to 5, 1
    fig_box.suptitle(
        f'3D Gaze Error & Precision (Best Windows) - {folder_basename}',
        y=1.02)

    ang_err_data = overall_best_df['angular_error_deg'].dropna()
    cos_sim_data = overall_best_df['cosine_similarity'].dropna()
    dist_2d_data = overall_best_df['distance_2d'].dropna()
    precision_rms_3d_data = df_precision['precision_rms_3d'].dropna(
    ) if df_precision is not None else pd.Series(dtype='float64')
    precision_std_3d_data = df_precision['precision_std_3d'].dropna(
    ) if df_precision is not None else pd.Series(dtype='float64')

    if not ang_err_data.empty:
        ax_box1.boxplot(ang_err_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(ang_err_data)
        ax_box1.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box1.set_title('Angular Error (deg) [5th-95th Percentile]')
        ax_box1.set_xlabel('Error (degrees)'), ax_box1.set_yticklabels(
            ['Angular Error'])
    else:
        ax_box1.set_title('No Angular Error Data')

    if not cos_sim_data.empty:
        ax_box2.boxplot(cos_sim_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(cos_sim_data)
        ax_box2.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box2.set_title('Cosine Similarity [5th-95th Percentile]')
        ax_box2.set_xlabel('Similarity'), ax_box2.set_yticklabels(
            ['Cosine Sim.'])
    else:
        ax_box2.set_title('No Cosine Similarity Data')

    if not dist_2d_data.empty:
        ax_box3.boxplot(dist_2d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(dist_2d_data)
        ax_box3.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box3.set_title('2D Distance Error (units) [5th-95th Percentile]')
        ax_box3.set_xlabel('Error (units)'), ax_box3.set_yticklabels(
            ['2D Dist. Error'])
    else:
        ax_box3.set_title('No 2D Distance Data')

    # Precision Plots
    if not precision_rms_3d_data.empty:
        ax_box4.boxplot(precision_rms_3d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(precision_rms_3d_data)
        ax_box4.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box4.set_title(
            'Angular Precision 3D (RMS from Centroid) [5th-95th Percentile of blocks] (deg)'
        )
        ax_box4.set_xlabel('Precision (deg)'), ax_box4.set_yticklabels(
            ['RMS 3D (deg)'])
    else:
        ax_box4.set_title('No 3D RMS Precision Data')

    if not precision_std_3d_data.empty:
        ax_box5.boxplot(precision_std_3d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(precision_std_3d_data)
        ax_box5.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box5.set_title(
            'Angular Precision 3D (STD from Centroid) [5th-95th Percentile of blocks] (deg)'
        )
        ax_box5.set_xlabel('Precision (deg)'), ax_box5.set_yticklabels(
            ['STD 3D (deg)'])
    else:
        ax_box5.set_title('No 3D STD Precision Data')

    plt.tight_layout()
    plot_box_filename = os.path.join(viz_output_folder,
                                     'analysis_plot_3D_error_boxplot.png')
    plt.savefig(plot_box_filename)
    plt.close(fig_box)

    # 4c. 2D UD Model Plot (Azimuth/Elevation). Error from all best windows
    df_plot_ud = overall_best_df.copy()

    gaze_vec_sel = ptk.normalize(
        df_plot_ud[['globalX', 'globalY', 'globalZ']].values -
        df_plot_ud[['eyeOriginX', 'eyeOriginY', 'eyeOriginZ']].values)
    target_vec_sel = ptk.normalize(
        df_plot_ud[['globaltargetX', 'globaltargetY', 'globaltargetZ']].values - \
        df_plot_ud[['eyeOriginX', 'eyeOriginY', 'eyeOriginZ']].values
    )
    gaze_az, gaze_el = ptk.vectors_to_angles_deg(gaze_vec_sel)
    target_az, target_el = ptk.vectors_to_angles_deg(target_vec_sel)
    error_az = gaze_az - target_az
    error_el = gaze_el - target_el
    error_az[error_az > 180] -= 360
    error_az[error_az < -180] += 360

    plt.figure(figsize=(10, 8))
    plt.scatter(error_az,
                error_el,
                alpha=0.7,
                label=f'Gaze Point Error (All {n_windows} Best Windows)')
    plt.plot(0, 0, 'r+', markersize=10, label='Target (0, 0)')
    plt.title(f'2D Gaze Error (UD Model) - {folder_basename}')
    plt.xlabel('Azimuth Error (degrees)'), plt.ylabel(
        'Elevation Error (degrees)')
    plt.grid(True, linestyle='--', alpha=0.5), plt.legend(), plt.axis('equal')
    plot_2d_filename = os.path.join(viz_output_folder,
                                    'analysis_plot_2D_UD_model.png')
    plt.savefig(plot_2d_filename)
    plt.close()

    # 4d. 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=60, vertical_axis='y')

    unique_targets = df_gaze[[
        'globaltargetX', 'globaltargetY', 'globaltargetZ'
    ]].drop_duplicates()
    ax.scatter(unique_targets['globaltargetX'],
               unique_targets['globaltargetY'],
               unique_targets['globaltargetZ'],
               c='r',
               marker='x',
               s=100,
               label='All Targets (from pointcloud.csv)')

    ax.scatter(overall_best_df_viz['eyeOriginX'],
               overall_best_df_viz['eyeOriginY'],
               overall_best_df_viz['eyeOriginZ'],
               c='b',
               marker='.',
               s=10,
               alpha=0.3,
               label='Eye Origins (Best Windows)')

    for block_id, df_window in overall_best_df_viz.groupby('block_id'):
        gaze_hit_points = df_window[['globalX', 'globalY', 'globalZ']].values
        ax.scatter(gaze_hit_points[:, 0],
                   gaze_hit_points[:, 1],
                   gaze_hit_points[:, 2],
                   c='g',
                   marker='o',
                   s=10,
                   alpha=0.3)

    ax.scatter([], [], [],
               c='g',
               marker='.',
               s=10,
               label='Gaze Hit Points (Best Windows)')
    ax.set_xlabel('X Coordinate (Global)'), ax.set_ylabel(
        'Y Coordinate (Global)'), ax.set_zlabel('Z Coordinate (Global)')
    ax.set_title(f'3D Gaze Analysis - {folder_basename}'), ax.legend()

    plot_3d_filename = os.path.join(viz_output_folder,
                                    'analysis_plot_3D_scatter.png')
    plt.savefig(plot_3d_filename)
    plt.close(fig)

    # 4e. Copy the model.obj
    if os.path.exists(model_obj_path):
        shutil.copy(model_obj_path, os.path.join(viz_output_folder,
                                                 "model.obj"))
    else:
        print(f"Warning: model.obj not found at {model_obj_path}",
              file=sys.stderr)

    # Return key statistics
    return {
        'type': '3D',
        'ang_err_mean': stats.get('angular_error_deg_mean'),
        'ang_err_std': stats.get('angular_error_deg_std'),
        'ang_err_median': stats.get('angular_error_deg_median'),
        'cos_sim_mean': stats.get('cosine_similarity_mean'),
        'cos_sim_median': stats.get('cosine_similarity_median'),
        'dist_2d_mean': stats.get('distance_2d_mean'),
        'dist_2d_std': stats.get('distance_2d_std'),
        'dist_2d_median': stats.get('distance_2d_median'),
        'precision_rms_3d_mean': stats.get('precision_rms_3d_mean'),
        'precision_rms_3d_std': stats.get('precision_rms_3d_std'),
        'precision_std_3d_mean': stats.get('precision_std_3d_mean'),
        'precision_std_3d_std': stats.get('precision_std_3d_std'),
        'precision_rms_2d_mean': stats.get('precision_rms_2d_mean'),
        'precision_rms_2d_std': stats.get('precision_rms_2d_std'),
        'precision_std_2d_mean': stats.get('precision_std_2d_mean'),
        'precision_std_2d_std': stats.get('precision_std_2d_std')
    }


# FUNCTION FOR 3D TRACKING
def analyze_gaze_data_3D_TRACKING(df_gaze_in, output_path, model_obj_path,
                                  viz_output_folder):
    """
    Performs 3D analysis for CONTINUOUS TRACKING tasks.
    Uses toolkit to calculate metrics over the entire dataframe (all blocks).
    """
    folder_basename = os.path.basename(output_path)

    # 1. Calculate Per-Point Metrics
    df_gaze = ptk.calculate_gaze_error_metrics(df_gaze_in)

    # 2. Use the whole DataFrame for analysis (no stable window finding)
    overall_best_df = df_gaze  # This is the key change

    if overall_best_df.empty:
        print(
            f"Error: No valid data for {folder_basename}. Aborting analysis.",
            file=sys.stderr)
        return None

    n_windows = overall_best_df['block_id'].nunique()

    # Calculate spatial precision for each block (using the full block data)
    # For tracking, precision (spread from a single centroid) is not a meaningful metric,
    # as the gaze is supposed to move. We are interested in angular error (accuracy),
    # which is already calculated. We create an empty df here to skip the misleading plots.
    df_precision = pd.DataFrame()  # Create empty dataframe

    # Create the viz dataframe just for the 3D scatter plot
    overall_best_df_viz = overall_best_df.copy()

    # 3. Matplotlib Plots & Report

    # 3a. Get Stats & Report (Based on entire dataframe)
    stats = ptk.summarize_metrics(
        overall_best_df,
        ['cosine_similarity', 'angular_error_deg', 'distance_2d'])
    # Get stats for precision (mean of all blocks precision)
    if df_precision is not None and not df_precision.empty:
        precision_stats = ptk.summarize_metrics(df_precision, [
            'precision_rms_3d', 'precision_std_3d', 'precision_rms_2d',
            'precision_std_2d'
        ])
    else:
        precision_stats = {}  # Create an empty dict if df_precision is None
    stats.update(precision_stats)  # Add precision stats

    report = f"3D Gaze Analysis Report ({folder_basename})\n" \
             f"Task Mode: Continuous Tracking (Full Data Analysis)\n" \
             f"Coordinates: GLOBAL\n" \
             f"Blocks Found: {n_windows}\n" \
             f"Total Points Analyzed: {len(overall_best_df)}\n\n" \
             f"METRICS (from all data)\n" \
             f"Cosine Similarity:\n" \
             f"   Mean:   {stats.get('cosine_similarity_mean', np.nan):.4f}\n" \
             f"   Median: {stats.get('cosine_similarity_median', np.nan):.4f}\n" \
             f"   Min:    {stats.get('cosine_similarity_min', np.nan):.4f}\n" \
             f"   Max:    {stats.get('cosine_similarity_max', np.nan):.4f}\n\n" \
             f"Angular Error (degrees):\n" \
             f"   Mean:   {stats.get('angular_error_deg_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('angular_error_deg_std', np.nan):.4f}\n" \
             f"   Median: {stats.get('angular_error_deg_median', np.nan):.4f}\n" \
             f"   Min:    {stats.get('angular_error_deg_min', np.nan):.4f}\n" \
             f"   Max:    {stats.get('angular_error_deg_max', np.nan):.4f}\n\n" \
             f"2D Distance Error (units):\n" \
             f"   Mean:   {stats.get('distance_2d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('distance_2d_std', np.nan):.4f}\n" \
             f"   Median: {stats.get('distance_2d_median', np.nan):.4f}\n" \
             f"   Min:    {stats.get('distance_2d_min', np.nan):.4f}\n" \
             f"   Max:    {stats.get('distance_2d_max', np.nan):.4f}\n\n" \
             f"Angular Precision 3D (RMS from centroid, mean of {n_windows} blocks) (deg):\n" \
             f"   Mean:   {stats.get('precision_rms_3d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('precision_rms_3d_std', np.nan):.4f}\n\n" \
             f"Angular Precision 3D (STD from centroid, mean of {n_windows} blocks) (deg):\n" \
             f"   Mean:   {stats.get('precision_std_3d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('precision_std_3d_std', np.nan):.4f}\n"

    report_filename = os.path.join(viz_output_folder, 'analysis_report_3D.txt')
    with open(report_filename, 'w') as f:
        f.write(report)

    # 3b. Gaze Error Box Plots (5-panel)
    fig_box, (ax_box1, ax_box2, ax_box3,
              ax_box4, ax_box5) = plt.subplots(5,
                                               1,
                                               figsize=(10, 18),
                                               sharex=False)  # Changed to 5, 1
    fig_box.suptitle(
        f'3D Gaze Error & Precision (Full Tracking Task) - {folder_basename}',
        y=1.02)

    ang_err_data = overall_best_df['angular_error_deg'].dropna()
    cos_sim_data = overall_best_df['cosine_similarity'].dropna()
    dist_2d_data = overall_best_df['distance_2d'].dropna()
    precision_rms_3d_data = df_precision['precision_rms_3d'].dropna(
    ) if df_precision is not None and not df_precision.empty else pd.Series(
        dtype='float64')
    precision_std_3d_data = df_precision['precision_std_3d'].dropna(
    ) if df_precision is not None and not df_precision.empty else pd.Series(
        dtype='float64')

    if not ang_err_data.empty:
        ax_box1.boxplot(ang_err_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(ang_err_data)
        ax_box1.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box1.set_title('Angular Error (deg) [5th-95th Percentile]')
        ax_box1.set_xlabel('Error (degrees)'), ax_box1.set_yticklabels(
            ['Angular Error'])
    else:
        ax_box1.set_title('No Angular Error Data')

    if not cos_sim_data.empty:
        ax_box2.boxplot(cos_sim_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(cos_sim_data)
        ax_box2.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box2.set_title('Cosine Similarity [5th-95th Percentile]')
        ax_box2.set_xlabel('Similarity'), ax_box2.set_yticklabels(
            ['Cosine Sim.'])
    else:
        ax_box2.set_title('No Cosine Similarity Data')

    if not dist_2d_data.empty:
        ax_box3.boxplot(dist_2d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(dist_2d_data)
        ax_box3.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box3.set_title('2D Distance Error (units) [5th-95th Percentile]')
        ax_box3.set_xlabel('Error (units)'), ax_box3.set_yticklabels(
            ['2D Dist. Error'])
    else:
        ax_box3.set_title('No 2D Distance Data')

    # Precision Plots (These will be empty for tracking and show "No Data")
    if not precision_rms_3d_data.empty:
        ax_box4.boxplot(precision_rms_3d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(precision_rms_3d_data)
        ax_box4.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box4.set_title(
            'Angular Precision 3D (RMS from Centroid) [5th-95th Percentile of blocks] (deg)'
        )
        ax_box4.set_xlabel('Precision (deg)'), ax_box4.set_yticklabels(
            ['RMS 3D (deg)'])
    else:
        ax_box4.set_title(
            'No 3D RMS Precision Data (Not applicable for Tracking)')

    if not precision_std_3d_data.empty:
        ax_box5.boxplot(precision_std_3d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(precision_std_3d_data)
        ax_box5.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box5.set_title(
            'Angular Precision 3D (STD from Centroid) [5th-95th Percentile of blocks] (deg)'
        )
        ax_box5.set_xlabel('Precision (deg)'), ax_box5.set_yticklabels(
            ['STD 3D (deg)'])
    else:
        ax_box5.set_title(
            'No 3D STD Precision Data (Not applicable for Tracking)')

    plt.tight_layout()
    plot_box_filename = os.path.join(viz_output_folder,
                                     'analysis_plot_3D_error_boxplot.png')
    plt.savefig(plot_box_filename)
    plt.close(fig_box)

    # 4c. 2D UD Model Plot (Azimuth/Elevation). Error from all data
    df_plot_ud = overall_best_df.copy()

    gaze_vec_sel = ptk.normalize(
        df_plot_ud[['globalX', 'globalY', 'globalZ']].values -
        df_plot_ud[['eyeOriginX', 'eyeOriginY', 'eyeOriginZ']].values)
    target_vec_sel = ptk.normalize(
        df_plot_ud[['globaltargetX', 'globaltargetY', 'globaltargetZ']].values - \
        df_plot_ud[['eyeOriginX', 'eyeOriginY', 'eyeOriginZ']].values
    )
    gaze_az, gaze_el = ptk.vectors_to_angles_deg(gaze_vec_sel)
    target_az, target_el = ptk.vectors_to_angles_deg(target_vec_sel)
    error_az = gaze_az - target_az
    error_el = gaze_el - target_el
    error_az[error_az > 180] -= 360
    error_az[error_az < -180] += 360

    plt.figure(figsize=(10, 8))
    plt.scatter(error_az,
                error_el,
                alpha=0.7,
                s=5,
                label=f'Gaze Point Error (All {n_windows} Blocks)'
                )  # Reduced scatter size
    plt.plot(0, 0, 'r+', markersize=10, label='Target (0, 0)')
    plt.title(f'2D Gaze Error (UD Model) - {folder_basename}')
    plt.xlabel('Azimuth Error (degrees)'), plt.ylabel(
        'Elevation Error (degrees)')
    plt.grid(True, linestyle='--', alpha=0.5), plt.legend(), plt.axis('equal')
    plot_2d_filename = os.path.join(viz_output_folder,
                                    'analysis_plot_2D_UD_model.png')
    plt.savefig(plot_2d_filename)
    plt.close()

    # 4d. 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=60, vertical_axis='y')

    unique_targets = df_gaze[[
        'globaltargetX', 'globaltargetY', 'globaltargetZ'
    ]].drop_duplicates()
    ax.scatter(unique_targets['globaltargetX'],
               unique_targets['globaltargetY'],
               unique_targets['globaltargetZ'],
               c='r',
               marker='x',
               s=100,
               label='All Targets (from pointcloud.csv)')

    ax.scatter(overall_best_df_viz['eyeOriginX'],
               overall_best_df_viz['eyeOriginY'],
               overall_best_df_viz['eyeOriginZ'],
               c='b',
               marker='.',
               s=10,
               alpha=0.3,
               label='Eye Origins (All Data)')

    for block_id, df_window in overall_best_df_viz.groupby('block_id'):
        gaze_hit_points = df_window[['globalX', 'globalY', 'globalZ']].values
        ax.scatter(gaze_hit_points[:, 0],
                   gaze_hit_points[:, 1],
                   gaze_hit_points[:, 2],
                   c='g',
                   marker='o',
                   s=10,
                   alpha=0.3)

    ax.scatter([], [], [],
               c='g',
               marker='.',
               s=10,
               label='Gaze Hit Points (All Data)')
    ax.set_xlabel('X Coordinate (Global)'), ax.set_ylabel(
        'Y Coordinate (Global)'), ax.set_zlabel('Z Coordinate (Global)')
    ax.set_title(f'3D Gaze Analysis - {folder_basename}'), ax.legend()

    plot_3d_filename = os.path.join(viz_output_folder,
                                    'analysis_plot_3D_scatter.png')
    plt.savefig(plot_3d_filename)
    plt.close(fig)

    # 4e. Copy the model.obj
    if os.path.exists(model_obj_path):
        shutil.copy(model_obj_path, os.path.join(viz_output_folder,
                                                 "model.obj"))
    else:
        print(f"Warning: model.obj not found at {model_obj_path}",
              file=sys.stderr)

    # Return key statistics
    return {
        'type': '3D_Tracking',  # New type
        'ang_err_mean': stats.get('angular_error_deg_mean'),
        'ang_err_std': stats.get('angular_error_deg_std'),
        'ang_err_median': stats.get('angular_error_deg_median'),
        'cos_sim_mean': stats.get('cosine_similarity_mean'),
        'cos_sim_median': stats.get('cosine_similarity_median'),
        'dist_2d_mean': stats.get('distance_2d_mean'),
        'dist_2d_std': stats.get('distance_2d_std'),
        'dist_2d_median': stats.get('distance_2d_median'),
        'precision_rms_3d_mean': stats.get('precision_rms_3d_mean'),
        'precision_rms_3d_std': stats.get('precision_rms_3d_std'),
        'precision_std_3d_mean': stats.get('precision_std_3d_mean'),
        'precision_std_3d_std': stats.get('precision_std_3d_std'),
        'precision_rms_2d_mean': stats.get('precision_rms_2d_mean'),
        'precision_rms_2d_std': stats.get('precision_rms_2d_std'),
        'precision_std_2d_mean': stats.get('precision_std_2d_mean'),
        'precision_std_2d_std': stats.get('precision_std_2d_std')
    }


def analyze_gaze_data_2D(df_gaze_in,
                         output_path,
                         viz_output_folder,
                         window_size=30):
    """
    Performs 2D analysis for static target tasks.
    Uses toolkit to calculate metrics and find windows, then generates plots/reports.
    """
    folder_basename = os.path.basename(output_path)

    # 1. Calculate Per-Point Metrics
    df_gaze = ptk.calculate_gaze_error_metrics(df_gaze_in)

    # 2. Find Best Window PER BLOCK
    # Use 'angular_error_deg' as it's the more robust metric
    overall_best_df = ptk.find_stable_windows(df_gaze, 'angular_error_deg',
                                              window_size)

    if overall_best_df.empty:
        print(
            f"Error: Could not find any valid windows for {folder_basename}. Aborting analysis.",
            file=sys.stderr)
        return None

    n_windows = overall_best_df['block_id'].nunique()

    # Calculate spatial precision for each block
    df_precision = ptk.calculate_spatial_precision_metrics(overall_best_df)

    # 3. Report stats (Based on all best windows)
    stats = ptk.summarize_metrics(overall_best_df,
                                  ['distance_2d', 'angular_error_deg'])
    # Get stats for precision (mean of all blocks precision)
    if df_precision is not None and not df_precision.empty:
        precision_stats = ptk.summarize_metrics(df_precision, [
            'precision_rms_3d', 'precision_std_3d', 'precision_rms_2d',
            'precision_std_2d'
        ])
    else:
        precision_stats = {}  # Create an empty dict if df_precision is None
    stats.update(precision_stats)

    report = f"2D Gaze Analysis Report ({folder_basename})\n" \
             f"Task Mode: Per-Block Analysis)\n" \
             f"Total Points in Windows: {len(overall_best_df)}\n\n" \
             f"METRICS (from all stable windows)\n" \
             f"2D Distance Error (units):\n" \
             f"   Mean:   {stats.get('distance_2d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('distance_2d_std', np.nan):.4f}\n" \
             f"   Median: {stats.get('distance_2d_median', np.nan):.4f}\n" \
             f"   Min:    {stats.get('distance_2d_min', np.nan):.4f}\n" \
             f"   Max:    {stats.get('distance_2d_max', np.nan):.4f}\n\n" \
             f"Angular Error (degrees):\n" \
             f"   Mean:   {stats.get('angular_error_deg_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('angular_error_deg_std', np.nan):.4f}\n" \
             f"   Median: {stats.get('angular_error_deg_median', np.nan):.4f}\n" \
             f"   Min:    {stats.get('angular_error_deg_min', np.nan):.4f}\n" \
             f"   Max:    {stats.get('angular_error_deg_max', np.nan):.4f}\n\n" \
             f"Angular Precision 2D (RMS from centroid, mean of {n_windows} blocks) (deg):\n" \
             f"   Mean:   {stats.get('precision_rms_2d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('precision_rms_2d_std', np.nan):.4f}\n\n" \
             f"Angular Precision 2D (STD from centroid, mean of {n_windows} blocks) (deg):\n" \
             f"   Mean:   {stats.get('precision_std_2d_mean', np.nan):.4f}\n" \
             f"   StdDev: {stats.get('precision_std_2d_std', np.nan):.4f}\n"

    report_filename = os.path.join(viz_output_folder, 'analysis_report_2D.txt')
    with open(report_filename, 'w') as f:
        f.write(report)

    # 4. Gaze Error Box Plots (4-panel)
    fig_box_2d, (ax_box1, ax_box2, ax_box3,
                 ax_box4) = plt.subplots(4, 1, figsize=(10, 15),
                                         sharex=False)  # Changed to 4, 1
    fig_box_2d.suptitle(
        f'Gaze Error & Precision (Best Windows) - {folder_basename}', y=1.02)

    dist_2d_data = overall_best_df['distance_2d'].dropna()
    ang_err_data = overall_best_df['angular_error_deg'].dropna()
    # Precision data
    precision_rms_2d_data = df_precision['precision_rms_2d'].dropna(
    ) if df_precision is not None else pd.Series(dtype='float64')
    precision_std_2d_data = df_precision['precision_std_2d'].dropna(
    ) if df_precision is not None else pd.Series(dtype='float64')

    if not dist_2d_data.empty:
        ax_box1.boxplot(dist_2d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(dist_2d_data)
        ax_box1.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box1.set_title('2D Distance Error (units) [5th-95th Percentile]')
        ax_box1.set_xlabel('Error (units)'), ax_box1.set_yticklabels(
            ['2D Dist. Error'])
    else:
        ax_box1.set_title('No 2D Distance Data')

    if not ang_err_data.empty:
        ax_box2.boxplot(ang_err_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(ang_err_data)
        ax_box2.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box2.set_title('Angular Error (deg) [5th-95th Percentile]')
        ax_box2.set_xlabel('Error (degrees)'), ax_box2.set_yticklabels(
            ['Angular Error'])
    else:
        ax_box2.set_title('No Angular Error Data')

    # Precision Plots
    if not precision_rms_2d_data.empty:
        ax_box3.boxplot(precision_rms_2d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(precision_rms_2d_data)
        ax_box3.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box3.set_title(
            'Angular Precision 2D (RMS from Centroid) [5th-95th Percentile of blocks] (deg)'
        )
        ax_box3.set_xlabel('Precision (deg)'), ax_box3.set_yticklabels(
            ['RMS 2D (deg)'])
    else:
        ax_box3.set_title('No 2D RMS Precision Data')

    if not precision_std_2d_data.empty:
        ax_box4.boxplot(precision_std_2d_data,
                        vert=False,
                        whis=[5, 95],
                        showfliers=False,
                        showmeans=True)
        mean_val = np.nanmean(precision_std_2d_data)
        ax_box4.text(mean_val,
                     1.1,
                     f'Mean: {mean_val:.3f}',
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     color='blue',
                     fontsize=9,
                     fontweight='bold')
        ax_box4.set_title(
            'Angular Precision 2D (STD from Centroid) [5th-95th Percentile of blocks] (deg)'
        )
        ax_box4.set_xlabel('Precision (deg)'), ax_box4.set_yticklabels(
            ['STD 2D (deg)'])
    else:
        ax_box4.set_title('No 2D STD Precision Data')

    plt.tight_layout()
    plot_box_filename_2d = os.path.join(viz_output_folder,
                                        'analysis_plot_2D_error_boxplot.png')
    plt.savefig(plot_box_filename_2d)
    plt.close(fig_box_2d)

    # 5. Finalize and Save 2D Scatter Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(df_gaze['globalX'],
                df_gaze['globalY'],
                label='All Gaze Points (Global)',
                alpha=0.1,
                s=10,
                c='cyan')
    unique_targets = df_gaze[['globaltargetX',
                              'globaltargetY']].drop_duplicates()
    plt.scatter(unique_targets['globaltargetX'],
                unique_targets['globaltargetY'],
                label='Targets (from pointcloud.csv)',
                marker='x',
                c='red',
                s=100)
    plt.scatter(overall_best_df['globalX'],
                overall_best_df['globalY'],
                label=f'Best Windows ({n_windows} found)',
                alpha=0.8,
                s=20,
                c='blue')

    plt.title(f'2D Gaze Plot - {folder_basename}')
    plt.xlabel('X Coordinate (Global)'), plt.ylabel('Y Coordinate (Global)')
    plt.legend(), plt.grid(True, linestyle='--', alpha=0.5), plt.axis('equal')

    plot_filename = os.path.join(viz_output_folder,
                                 'analysis_plot_2D_scatter.png')
    plt.savefig(plot_filename)
    plt.close()

    # Return key statistics
    return {
        'type': '2D',
        'dist_2d_mean': stats.get('distance_2d_mean'),
        'dist_2d_std': stats.get('distance_2d_std'),
        'dist_2d_median': stats.get('distance_2d_median'),
        'ang_err_mean': stats.get('angular_error_deg_mean'),
        'ang_err_std': stats.get('angular_error_deg_std'),
        'ang_err_median': stats.get('angular_error_deg_median'),
        'precision_rms_3d_mean': stats.get('precision_rms_3d_mean'),
        'precision_rms_3d_std': stats.get('precision_rms_3d_std'),
        'precision_std_3d_mean': stats.get('precision_std_3d_mean'),
        'precision_std_3d_std': stats.get('precision_std_3d_std'),
        'precision_rms_2d_mean': stats.get('precision_rms_2d_mean'),
        'precision_rms_2d_std': stats.get('precision_rms_2d_std'),
        'precision_std_2d_mean': stats.get('precision_std_2d_mean'),
        'precision_std_2d_std': stats.get('precision_std_2d_std')
    }


# SUMMARY & TIMESERIES PLOTTING FUNCTIONS


def run_summary_analysis(all_results_data, summary_output_dir):
    """
    Generates summary plots and CSV from all processed sessions.
    """

    if not all_results_data:
        print("No data to summarize.")
        return

    df_summary = pd.DataFrame(all_results_data)

    try:
        summary_csv_path = os.path.join(summary_output_dir,
                                        'summary_all_results.csv')
        df_summary.to_csv(summary_csv_path, index=False)
    except Exception as e:
        print(f"Error saving summary CSV: {e}", file=sys.stderr)

    plot_definitions = [
        ('freq_mean', 'Data Acquisition Rate MEAN (All Sessions)',
         'Frequency (Hz)'),
        ('freq_std', 'Data Acquisition Rate STDEV (All Sessions)',
         'Frequency (Hz)'),
        ('isi_mean', 'Inter-Sample Interval MEAN (All Sessions)', 'Time (ms)'),
        ('isi_std', 'Inter-Sample Interval STDEV (All Sessions)', 'Time (ms)'),
        ('eye_gaze_dist_mean',
         'Eye-to-Gaze-Hit 3D Distance MEAN (All Sessions)', 'Distance (m)'),
        ('eye_gaze_dist_std',
         'Eye-to-Gaze-Hit 3D Distance STDEV (All Sessions)', 'Distance (m)'),
        ('ang_err_std', '3D Angular Error STDEV (All Sessions)',
         'Angular Error (deg)'),
        ('ang_err_mean', '3D Angular Error MEAN (All Sessions)',
         'Angular Error (deg)'),
        ('dist_2d_std', '2D Distance Error STDEV (All Sessions)',
         'Distance Error (units)'),
        ('dist_2d_mean', '2D Distance Error MEAN (All Sessions)',
         'Distance Error (units)'),
        ('pos_vel_std', 'Head Positional Velocity STDEV (All Sessions)',
         'Velocity (m/s)'),
        ('pos_vel_mean', 'Head Positional Velocity MEAN (All Sessions)',
         'Velocity (m/s)'),
        ('ang_vel_std', 'Head Angular Velocity STDEV (All Sessions)',
         'Velocity (deg/s)'),
        ('ang_vel_mean', 'Head Angular Velocity MEAN (All Sessions)',
         'Velocity (deg/s)'),
        ('precision_rms_3d_mean',
         'Spatial Precision 3D (RMS) MEAN (All Sessions)', 'Precision (deg)'),
        ('precision_std_3d_mean',
         'Spatial Precision 3D (STD) MEAN (All Sessions)', 'Precision (deg)'),
        ('precision_rms_2d_mean',
         'Spatial Precision 2D (RMS) MEAN (All Sessions)', 'Precision (deg)'),
        ('precision_std_2d_mean',
         'Spatial Precision 2D (STD) MEAN (All Sessions)', 'Precision (deg)'),
    ]

    for col, title, ylabel in plot_definitions:
        if col not in df_summary.columns:
            continue

        df_plot = df_summary[['analysis_folder', col]].dropna()
        if df_plot.empty:
            print(f"Skipping plot for '{col}' (no data).")
            continue

        folders = df_plot['analysis_folder'].unique()
        folders.sort()
        data_to_plot = [
            df_plot[df_plot['analysis_folder'] == f][col].values
            for f in folders
        ]
        if not data_to_plot:
            continue

        fig, ax = plt.subplots(figsize=(max(12, len(folders) * 1.5), 7))
        ax.boxplot(data_to_plot,
                   labels=folders,
                   whis=[5, 95],
                   showmeans=True,
                   showfliers=True)

        # Add text labels for mean values
        for i, data in enumerate(data_to_plot):
            if data.size > 0:
                mean_val = np.nanmean(data)
                ax.text(i + 1.05,
                        mean_val,
                        f' {mean_val:.3f}',
                        verticalalignment='center',
                        horizontalalignment='left',
                        color='blue',
                        fontsize=9,
                        fontweight='bold')

        ax.set_title(title), ax.set_ylabel(ylabel), ax.set_xlabel('Task')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6, axis='y'), plt.tight_layout()

        plot_filename = f"summary_plot_{col}.png"
        plot_path = os.path.join(summary_output_dir, plot_filename)
        try:
            plt.savefig(plot_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving summary plot '{plot_filename}': {e}",
                  file=sys.stderr)


def run_summary_timeseries_analysis(all_session_dataframes,
                                    summary_output_dir):
    """
    Generates summary timeseries plots using the toolkit for calculations.
    """
    grouped_dfs = {}
    for data in all_session_dataframes:
        folder_name = data['analysis_folder']
        if folder_name not in grouped_dfs:
            grouped_dfs[folder_name] = []
        grouped_dfs[folder_name].append({
            'dataframe':
            data['dataframe'],
            'has_valid_target':
            data['has_valid_target']
        })

    for folder_name, data_list in grouped_dfs.items():

        metrics = [
            'angular_error_deg', 'head_pos_velocity_mps',
            'head_ang_velocity_dps', 'eye_gaze_dist_3D'
        ]

        # 1. Calculate Resampled Stats
        plot_stats, overall_stats = ptk.resample_timeseries_data(data_list,
                                                                 metrics,
                                                                 n_points=1000)

        if not plot_stats or not overall_stats:
            print(f"No valid data to plot for {folder_name}, skipping.")
            continue

        has_valid_target_for_group = any(d['has_valid_target']
                                         for d in data_list)
        N_POINTS = 1000  # Must match the resample
        time_base = np.linspace(0.0, 1.0, N_POINTS)

        # 2. Create the 4-panel plot
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,
                                                     1,
                                                     figsize=(20, 15),
                                                     sharex=True)
            fig.suptitle(
                f'Summary Timeseries Overview - {folder_name} (n={len(data_list)} sessions)',
                fontsize=16,
                y=1.02)

            # Plot 1: Angular Error
            if has_valid_target_for_group and 'angular_error_deg' in plot_stats:
                mean = plot_stats['angular_error_deg']['mean']
                std = plot_stats['angular_error_deg']['std']
                ax1.plot(time_base,
                         mean,
                         label='Mean Error',
                         color='red',
                         linewidth=2)
                ax1.fill_between(time_base,
                                 mean - std,
                                 mean + std,
                                 color='red',
                                 alpha=0.2,
                                 label='Mean +/- 1 STD (time)')

                overall_mean = overall_stats['angular_error_deg']['mean']
                overall_std = overall_stats['angular_error_deg']['std']
                if pd.notna(overall_mean):
                    ax1.axhline(overall_mean,
                                color='k',
                                linestyle='-',
                                linewidth=2,
                                label=f'Overall Mean: {overall_mean:.2f}')
                if pd.notna(overall_std):
                    ax1.axhline(overall_mean + overall_std,
                                color='k',
                                linestyle='--',
                                linewidth=1,
                                label=f'Overall +/- 1 STD')
                    ax1.axhline(overall_mean - overall_std,
                                color='k',
                                linestyle='--',
                                linewidth=1)
                ax1.set_ylabel('Gaze Ang. Error (deg)'), ax1.set_title(
                    'Gaze Angular Error vs. Time')
            else:
                ax1.text(0.5,
                         0.5,
                         'No Target Data Available',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax1.transAxes,
                         fontsize=15,
                         color='gray',
                         alpha=0.6)
                ax1.set_title('Gaze Angular Error vs. Time (No Target)'
                              ), ax1.set_ylabel('Gaze Ang. Error (deg)')
            ax1.grid(True, linestyle='--',
                     alpha=0.5), ax1.legend(loc='upper right')

            # Plot 2: Head Positional Velocity
            if 'head_pos_velocity_mps' in plot_stats:
                mean = plot_stats['head_pos_velocity_mps']['mean']
                std = plot_stats['head_pos_velocity_mps']['std']
                ax2.plot(time_base,
                         mean,
                         label='Mean Velocity',
                         color='blue',
                         linewidth=2)
                ax2.fill_between(time_base,
                                 mean - std,
                                 mean + std,
                                 color='blue',
                                 alpha=0.2,
                                 label='Mean +/- 1 STD (time)')
                overall_mean = overall_stats['head_pos_velocity_mps']['mean']
                overall_std = overall_stats['head_pos_velocity_mps']['std']
                if pd.notna(overall_mean):
                    ax2.axhline(overall_mean,
                                color='k',
                                linestyle='-',
                                linewidth=2,
                                label=f'Overall Mean: {overall_mean:.2f}')
                if pd.notna(overall_std):
                    ax2.axhline(overall_mean + overall_std,
                                color='k',
                                linestyle='--',
                                linewidth=1,
                                label=f'Overall +/- 1 STD')
                    ax2.axhline(overall_mean - overall_std,
                                color='k',
                                linestyle='--',
                                linewidth=1)
                ax2.set_ylabel('Head Pos. Vel. (m/s)'), ax2.set_title(
                    'Head Positional Velocity vs. Time')
            ax2.grid(True, linestyle='--',
                     alpha=0.5), ax2.legend(loc='upper right')

            # Plot 3: Head Angular Velocity
            if 'head_ang_velocity_dps' in plot_stats:
                mean = plot_stats['head_ang_velocity_dps']['mean']
                std = plot_stats['head_ang_velocity_dps']['std']
                ax3.plot(time_base,
                         mean,
                         label='Mean Velocity',
                         color='green',
                         linewidth=2)
                ax3.fill_between(time_base,
                                 mean - std,
                                 mean + std,
                                 color='green',
                                 alpha=0.2,
                                 label='Mean +/- 1 STD (time)')
                overall_mean = overall_stats['head_ang_velocity_dps']['mean']
                overall_std = overall_stats['head_ang_velocity_dps']['std']
                if pd.notna(overall_mean):
                    ax3.axhline(overall_mean,
                                color='k',
                                linestyle='-',
                                linewidth=2,
                                label=f'Overall Mean: {overall_mean:.2f}')
                if pd.notna(overall_std):
                    ax3.axhline(overall_mean + overall_std,
                                color='k',
                                linestyle='--',
                                linewidth=1,
                                label=f'Overall +/- 1 STD')
                    ax3.axhline(overall_mean - overall_std,
                                color='k',
                                linestyle='--',
                                linewidth=1)
                ax3.set_ylabel('Head Ang. Vel. (deg/s)'), ax3.set_title(
                    'Head Angular Velocity vs. Time')
            ax3.grid(True, linestyle='--',
                     alpha=0.5), ax3.legend(loc='upper right')

            # Plot 4: Eye-to-Gaze Distance
            if 'eye_gaze_dist_3D' in plot_stats:
                mean = plot_stats['eye_gaze_dist_3D']['mean']
                std = plot_stats['eye_gaze_dist_3D']['std']
                ax4.plot(time_base,
                         mean,
                         label='Mean Eye-Gaze Dist',
                         color='purple',
                         linewidth=2)
                ax4.fill_between(time_base,
                                 mean - std,
                                 mean + std,
                                 color='purple',
                                 alpha=0.2,
                                 label='Mean +/- 1 STD (time)')
                overall_mean = overall_stats['eye_gaze_dist_3D']['mean']
                overall_std = overall_stats['eye_gaze_dist_3D']['std']
                if pd.notna(overall_mean):
                    ax4.axhline(overall_mean,
                                color='k',
                                linestyle='-',
                                linewidth=2,
                                label=f'Overall Mean: {overall_mean:.2f}')
                if pd.notna(overall_std):
                    ax4.axhline(overall_mean + overall_std,
                                color='k',
                                linestyle='--',
                                linewidth=1,
                                label=f'Overall +/- 1 STD')
                    ax4.axhline(overall_mean - overall_std,
                                color='k',
                                linestyle='--',
                                linewidth=1)
                ax4.set_ylabel('Distance (m)'), ax4.set_title(
                    'Eye-to-Gaze-Hit Distance vs. Time')
            ax4.set_xlabel(
                'Normalized Time (0.0 = start, 1.0 = end)'), ax4.grid(
                    True, linestyle='--',
                    alpha=0.5), ax4.legend(loc='upper right')

            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plot_filename = os.path.join(
                summary_output_dir, f'summary_timeseries_{folder_name}.png')
            plt.savefig(plot_filename)
            plt.close(fig)

        except Exception as e:
            print(
                f"Error generating summary timeseries plot for {folder_name}: {e}",
                file=sys.stderr)


def generate_folder_timeseries_plot(df_gaze_in, folder_name, viz_output_folder,
                                    has_valid_target):
    """
    Generates a 4-panel plot for a folder using the toolkit for calculations.
    """
    try:
        # 1. Calculate all 4 metrics
        df_plot = ptk.calculate_all_timeseries_metrics(df_gaze_in,
                                                       has_valid_target)
        if df_plot.empty:
            print("Warning: Timeseries calculation returned no data.",
                  file=sys.stderr)
            return

        df_plot[
            'time_sec'] = df_plot['timestamp'] - df_plot['timestamp'].iloc[0]

        # 2. Create the Plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,
                                                 1,
                                                 figsize=(20, 15),
                                                 sharex=True)
        fig.suptitle(f'Timeseries Overview - {folder_name}',
                     fontsize=16,
                     y=1.02)

        # Plot 1: Angular Error
        if has_valid_target:
            metric_data = df_plot['angular_error_deg'].dropna()
            ax1.plot(df_plot['time_sec'],
                     df_plot['angular_error_deg'],
                     label='Angular Error',
                     color='red',
                     alpha=0.7,
                     linewidth=0.8)
            if not metric_data.empty:
                mean_val, std_val = metric_data.mean(), metric_data.std()
                ax1.axhline(mean_val,
                            color='red',
                            linestyle='-',
                            linewidth=2,
                            label=f'Mean: {mean_val:.2f}')
                ax1.axhline(mean_val + std_val,
                            color='red',
                            linestyle='--',
                            linewidth=1,
                            label=f'Std: {std_val:.2f}')
                ax1.axhline(mean_val - std_val,
                            color='red',
                            linestyle='--',
                            linewidth=1)
            ax1.set_ylabel('Gaze Ang. Error (deg)'), ax1.set_title(
                'Gaze Angular Error vs. Time')
        else:
            ax1.text(0.5,
                     0.5,
                     'No Target Data Available',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax1.transAxes,
                     fontsize=15,
                     color='gray',
                     alpha=0.6)
            ax1.set_title('Gaze Angular Error vs. Time (No Target)'
                          ), ax1.set_ylabel('Gaze Ang. Error (deg)')
        ax1.grid(True, linestyle='--',
                 alpha=0.5), ax1.legend(loc='upper right')

        # Plot 2: Head Positional Velocity
        metric_data = df_plot['head_pos_velocity_mps'].dropna()
        ax2.plot(df_plot['time_sec'],
                 df_plot['head_pos_velocity_mps'],
                 label='Positional Velocity',
                 color='blue',
                 alpha=0.7,
                 linewidth=0.8)
        if not metric_data.empty:
            mean_val, std_val = metric_data.mean(), metric_data.std()
            ax2.axhline(mean_val,
                        color='blue',
                        linestyle='-',
                        linewidth=2,
                        label=f'Mean: {mean_val:.2f}')
            ax2.axhline(mean_val + std_val,
                        color='blue',
                        linestyle='--',
                        linewidth=1,
                        label=f'Std: {std_val:.2f}')
            ax2.axhline(mean_val - std_val,
                        color='blue',
                        linestyle='--',
                        linewidth=1)
        ax2.set_ylabel('Head Pos. Vel. (m/s)'), ax2.set_title(
            'Head Positional Velocity vs. Time')
        ax2.grid(True, linestyle='--',
                 alpha=0.5), ax2.legend(loc='upper right')

        # Plot 3: Head Angular Velocity
        metric_data = df_plot['head_ang_velocity_dps'].dropna()
        ax3.plot(df_plot['time_sec'],
                 df_plot['head_ang_velocity_dps'],
                 label='Angular Velocity',
                 color='green',
                 alpha=0.7,
                 linewidth=0.8)
        if not metric_data.empty:
            mean_val, std_val = metric_data.mean(), metric_data.std()
            ax3.axhline(mean_val,
                        color='green',
                        linestyle='-',
                        linewidth=2,
                        label=f'Mean: {mean_val:.2f}')
            ax3.axhline(mean_val + std_val,
                        color='green',
                        linestyle='--',
                        linewidth=1,
                        label=f'Std: {std_val:.2f}')
            ax3.axhline(mean_val - std_val,
                        color='green',
                        linestyle='--',
                        linewidth=1)
        ax3.set_ylabel('Head Ang. Vel. (deg/s)'), ax3.set_title(
            'Head Angular Velocity vs. Time')
        ax3.grid(True, linestyle='--',
                 alpha=0.5), ax3.legend(loc='upper right')

        # Plot 4: Eye-to-Gaze Distance
        metric_data = df_plot['eye_gaze_dist_3D'].dropna()
        ax4.plot(df_plot['time_sec'],
                 df_plot['eye_gaze_dist_3D'],
                 label='Eye-Gaze Distance',
                 color='purple',
                 alpha=0.7,
                 linewidth=0.8)
        if not metric_data.empty:
            mean_val, std_val = metric_data.mean(), metric_data.std()
            ax4.axhline(mean_val,
                        color='purple',
                        linestyle='-',
                        linewidth=2,
                        label=f'Mean: {mean_val:.2f}')
            ax4.axhline(mean_val + std_val,
                        color='purple',
                        linestyle='--',
                        linewidth=1,
                        label=f'Std: {std_val:.2f}')
            ax4.axhline(mean_val - std_val,
                        color='purple',
                        linestyle='--',
                        linewidth=1)
        ax4.set_ylabel('Distance (m)'), ax4.set_title(
            'Eye-to-Gaze-Hit Distance vs. Time')
        ax4.set_xlabel('Time (seconds)'), ax4.grid(
            True, linestyle='--', alpha=0.5), ax4.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        plot_filename = os.path.join(viz_output_folder,
                                     'analysis_plot_timeseries_overview.png')
        plt.savefig(plot_filename)
        plt.close(fig)

    except Exception as e:
        print(f"Error generating folder-wide timeseries plot: {e}",
              file=sys.stderr)


# NEW STATISTICAL ANALYSIS FUNCTION
def run_statistical_analysis(df_summary, summary_output_dir):
    """
    Performs non-parametric statistical tests (Friedman, Wilcoxon, Spearman) 
    to test the hypotheses H1 and H2.
    Saves the results to a structured CSV table.
    """
    table_rows = []

    # -------------------------------------------------------------------------
    # Step 1: Combine results from different 'type' rows into a single row per session/folder
    combined_data = []
    for (session,
         folder), group in df_summary.groupby(['session', 'analysis_folder']):
        row = {'session': session, 'analysis_folder': folder}
        for _, series in group.iterrows():
            row.update(
                series.drop(['session', 'analysis_folder',
                             'type']).dropna().to_dict())
        combined_data.append(row)

    df_stats = pd.DataFrame(combined_data)
    df_stats = df_stats.dropna(axis=1,
                               how='all')  # Drop columns that are entirely NaN

    # Define significance helper
    def get_sig_text(p_val):
        if np.isnan(p_val): return "N/A"
        if p_val < 0.001: return "***"
        if p_val < 0.01: return "**"
        if p_val < 0.05: return "*"
        return "ns"

    def get_explanation(test_name, p_val, comparisons=""):
        sig = get_sig_text(p_val)
        if sig == "ns" or sig == "N/A":
            return f"No significant effect found."
        else:
            return f"Significant effect found ({sig}). {comparisons}"

    # -------------------------------------------------------------------------
    # H1: Accuracy and precision is affected by distance of stimuli
    # Conditions: PlaneClose (Near), PlaneMiddle (Mid), PlaneFar (Far)
    # -------------------------------------------------------------------------
    distance_folders = ['PlaneClose', 'PlaneMiddle', 'PlaneFar']
    df_distance = df_stats[df_stats['analysis_folder'].isin(
        distance_folders)].copy()

    # Expanded metrics list for H1
    metrics_h1 = [
        'ang_err_mean',  # Accuracy (Angular Error)
        'precision_rms_3d_mean',  # Precision (RMS 3D)
        'eye_gaze_dist_mean',  # Distance Validation (Eye to Gaze)
        'eye_gaze_dist_std',  # Standard Deviation of Distance
        'pos_vel_mean',  # Head Positional Velocity
        'ang_vel_mean'  # Head Angular Velocity
    ]

    if len(df_distance['session'].unique()) >= 2:
        for metric_col in metrics_h1:
            if metric_col not in df_distance.columns: continue

            # Pivot to get paired data
            df_pivot = df_distance.pivot(index='session',
                                         columns='analysis_folder',
                                         values=metric_col).dropna()

            if len(df_pivot) < 2 or not all(col in df_pivot.columns
                                            for col in distance_folders):
                continue

            # Friedman Test
            try:
                data = [df_pivot[col].values for col in distance_folders]
                stat, p_friedman = friedmanchisquare(*data)

                table_rows.append({
                    'Hypothesis':
                    'H1: Effect of Distance',
                    'Test':
                    'Friedman Test',
                    'Comparison':
                    'PlaneClose vs PlaneMiddle vs PlaneFar',
                    'Metric':
                    metric_col,
                    'Statistic':
                    f"{stat:.4f}",
                    'P-Value':
                    f"{p_friedman:.4f}",
                    'Significance':
                    get_sig_text(p_friedman),
                    'Explanation':
                    get_explanation('Friedman', p_friedman)
                })

                # Post-Hoc Wilcoxon (Bonferroni)
                if p_friedman < 0.05:
                    pairs = [('PlaneClose', 'PlaneMiddle'),
                             ('PlaneClose', 'PlaneFar'),
                             ('PlaneMiddle', 'PlaneFar')]
                    p_vals = []
                    stats = []

                    for n1, n2 in pairs:
                        s, p = wilcoxon(df_pivot[n1], df_pivot[n2])
                        p_vals.append(p)
                        stats.append(s)

                    reject, p_corrected, _, _ = multipletests(
                        p_vals, alpha=0.05, method='bonferroni')

                    for i, (n1, n2) in enumerate(pairs):
                        table_rows.append({
                            'Hypothesis':
                            'H1: Effect of Distance (Post-Hoc)',
                            'Test':
                            'Wilcoxon Signed-Rank',
                            'Comparison':
                            f"{n1} vs {n2}",
                            'Metric':
                            metric_col,
                            'Statistic':
                            f"{stats[i]:.4f}",
                            'P-Value':
                            f"{p_vals[i]:.4f}",  # Uncorrected
                            'Corrected P-Value':
                            f"{p_corrected[i]:.4f}",
                            'Significance':
                            get_sig_text(p_corrected[i]),
                            'Explanation':
                            get_explanation('Wilcoxon', p_corrected[i])
                        })

            except Exception as e:
                print(f"Error in H1 Friedman for {metric_col}: {e}")

    # Extra H1: Compare ObjectCone(1) vs ObjectCone(3) if they exist (Direct Distance Comparison)
    cone_folders = ['ObjectCone(1)', 'ObjectCone(3)']
    df_cone = df_stats[df_stats['analysis_folder'].isin(cone_folders)].copy()
    if len(df_cone['session'].unique()) >= 2:
        for metric_col in metrics_h1:
            if metric_col not in df_cone.columns: continue
            df_pivot = df_cone.pivot(index='session',
                                     columns='analysis_folder',
                                     values=metric_col).dropna()
            if len(df_pivot) < 2 or not all(col in df_pivot.columns
                                            for col in cone_folders):
                continue

            try:
                s, p = wilcoxon(df_pivot['ObjectCone(1)'],
                                df_pivot['ObjectCone(3)'])
                table_rows.append({
                    'Hypothesis':
                    'H1: Effect of Distance (3D Tasks)',
                    'Test':
                    'Wilcoxon Signed-Rank',
                    'Comparison':
                    'ObjectCone(1) vs ObjectCone(3)',
                    'Metric':
                    metric_col,
                    'Statistic':
                    f"{s:.4f}",
                    'P-Value':
                    f"{p:.4f}",
                    'Significance':
                    get_sig_text(p),
                    'Explanation':
                    get_explanation('Wilcoxon', p, "1m vs 3m")
                })
            except:
                pass

    # -------------------------------------------------------------------------
    # H2: Accuracy and precision is affected by positional velocity and angular velocity
    # Test: Spearman's Rank Correlation (Non-Parametric)
    # -------------------------------------------------------------------------
    velocity_cols = ['pos_vel_mean', 'ang_vel_mean']
    gaze_cols = ['ang_err_mean', 'precision_rms_3d_mean', 'distance_2d_mean']

    if len(df_stats) >= 2:
        for vel_col in velocity_cols:
            for gaze_col in gaze_cols:
                if vel_col not in df_stats.columns or gaze_col not in df_stats.columns:
                    continue

                df_corr = df_stats[[
                    'session', 'analysis_folder', vel_col, gaze_col
                ]].dropna()
                if len(df_corr) < 5: continue  # Need decent N for correlation

                try:
                    rho, p_spearman = spearmanr(df_corr[vel_col],
                                                df_corr[gaze_col])
                    direction = "positive" if rho > 0 else "negative"

                    table_rows.append({
                        'Hypothesis':
                        'H2: Velocity vs Accuracy/Precision',
                        'Test':
                        'Spearman Correlation',
                        'Comparison':
                        'Correlation',
                        'Metric':
                        f"{vel_col} vs {gaze_col}",
                        'Statistic':
                        f"{rho:.4f}",  # Rho
                        'P-Value':
                        f"{p_spearman:.4f}",
                        'Significance':
                        get_sig_text(p_spearman),
                        'Explanation':
                        f"{'Significant' if p_spearman < 0.05 else 'Non-significant'} {direction} correlation."
                    })
                except Exception as e:
                    print(f"Error in H2 Correlation: {e}")

    # -------------------------------------------------------------------------
    # Save Results Table
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(table_rows)
    if not df_results.empty:
        # Reorder columns for readability
        cols = [
            'Hypothesis', 'Comparison', 'Metric', 'Test', 'Statistic',
            'P-Value', 'Corrected P-Value', 'Significance', 'Explanation'
        ]
        # Only include columns that exist
        cols = [c for c in cols if c in df_results.columns]
        df_results = df_results[cols]

        output_path = os.path.join(summary_output_dir,
                                   'statistical_results_table.csv')
        df_results.to_csv(output_path, index=False)
        print(f"\nStatistical Results Table saved to: {output_path}")
    else:
        print("No statistical results generated (insufficient data).")


# MAIN EXECUTION
def main():
    if not os.path.isdir(SESSIONS_ROOT_DIR):
        print(f"Error: Sessions root directory not found: {SESSIONS_ROOT_DIR}",
              file=sys.stderr)
        print(
            "Please update the 'SESSIONS_ROOT_DIR' variable at the top of the script.",
            file=sys.stderr)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Saving all results to: {RESULTS_DIR}")

    all_results_data = []  # For summary stats
    all_session_dataframes = []  # For summary timeseries

    for session_name in os.listdir(SESSIONS_ROOT_DIR):
        session_path = os.path.join(SESSIONS_ROOT_DIR, session_name)

        if not os.path.isdir(session_path):
            continue
        if not session_name.startswith('20'):  # Filter for session folders
            print(f"\nSkipping non-session folder: {session_name}")
            continue

        print(f"\nProcessing Session: {session_name}")

        session_results_dir = os.path.join(RESULTS_DIR, session_name)
        os.makedirs(session_results_dir, exist_ok=True)

        for folder_name in FOLDERS_TO_ANALYZE:
            folder_path = os.path.join(session_path, folder_name)
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder not found, skipping: {folder_path}")
                continue

            print(f"\nProcessing folder: {folder_name}")

            pointcloud_csv = os.path.join(folder_path, 'pointcloud.csv')
            model_obj_path = os.path.join(folder_path, 'model.obj')

            if not os.path.exists(pointcloud_csv):
                print(
                    f"Warning: Missing 'pointcloud.csv' in {folder_name}, skipping."
                )
                continue

            viz_output_folder = os.path.join(session_results_dir, folder_name)
            os.makedirs(viz_output_folder, exist_ok=True)

            try:
                df_gaze = pd.read_csv(pointcloud_csv)
                df_gaze.columns = df_gaze.columns.str.strip()

                if df_gaze.empty:
                    print(
                        f"Error: 'pointcloud.csv' in {folder_name} is empty. Skipping."
                    )
                    continue

                has_valid_target = True
                if 'targetName' not in df_gaze.columns or df_gaze[
                        'targetName'].isnull().all() or (df_gaze['targetName']
                                                         == 'null').all():
                    has_valid_target = False
                    print(
                        f"Info: No valid targets found for {folder_name} (targetName is null/NaN). Skipping target-based gaze analysis."
                    )

                all_session_dataframes.append({
                    'session':
                    session_name,
                    'analysis_folder':
                    folder_name,
                    'dataframe':
                    df_gaze.copy(),
                    'has_valid_target':
                    has_valid_target
                })

                required_cols = [
                    'targetName', 'timestamp', 'headX', 'headY', 'headZ',
                    'headForwardX', 'headForwardY', 'headForwardZ', 'globalX',
                    'globalY', 'globalZ', 'globaltargetX', 'globaltargetY',
                    'globaltargetZ', 'eyeOriginX', 'eyeOriginY', 'eyeOriginZ',
                    'eyeDirectionX', 'eyeDirectionY', 'eyeDirectionZ'
                ]
                missing_cols = [
                    col for col in required_cols if col not in df_gaze.columns
                ]
                if missing_cols:
                    print(
                        f"Error: 'pointcloud.csv' in {folder_name} is missing required columns: {missing_cols}. Skipping."
                    )
                    print(
                        "Please re-export your data with all global and head coordinate columns."
                    )
                    continue

            except Exception as e:
                print(
                    f"Error loading pointcloud.csv in {folder_name}: {e}. Skipping."
                )
                continue

            # Frequency Analysis
            freq_results = analyze_data_frequency(df_gaze.copy(),
                                                  viz_output_folder)
            if freq_results:
                freq_results['session'] = session_name
                freq_results['analysis_folder'] = folder_name
                all_results_data.append(freq_results)

            head_results = analyze_head_movement(df_gaze.copy(),
                                                 viz_output_folder)
            if head_results:
                head_results['session'] = session_name
                head_results['analysis_folder'] = folder_name
                all_results_data.append(head_results)

            eg_results = analyze_eye_to_gaze_distance(df_gaze.copy(),
                                                      viz_output_folder)
            if eg_results:
                eg_results['session'] = session_name
                eg_results['analysis_folder'] = folder_name
                all_results_data.append(eg_results)

            gaze_results = None
            if has_valid_target:
                if folder_name in ['PlaneClose', 'PlaneMiddle', 'PlaneFar']:
                    gaze_results = analyze_gaze_data_2D(
                        df_gaze.copy(),
                        output_path=folder_path,
                        viz_output_folder=viz_output_folder,
                        window_size=WINDOW_SIZE)
                elif folder_name in ['PlaneWSW']:
                    gaze_results = analyze_gaze_data_3D_TRACKING(
                        df_gaze.copy(),
                        output_path=folder_path,
                        model_obj_path=model_obj_path,
                        viz_output_folder=viz_output_folder)
                else:
                    gaze_results = analyze_gaze_data_3D(
                        df_gaze.copy(),
                        output_path=folder_path,
                        model_obj_path=model_obj_path,
                        viz_output_folder=viz_output_folder,
                        window_size=WINDOW_SIZE)
            if gaze_results:
                gaze_results['session'] = session_name
                gaze_results['analysis_folder'] = folder_name
                all_results_data.append(gaze_results)

            generate_folder_timeseries_plot(df_gaze.copy(), folder_name,
                                            viz_output_folder,
                                            has_valid_target)

    if all_results_data:
        summary_output_dir = os.path.join(RESULTS_DIR, 'summary')
        os.makedirs(summary_output_dir, exist_ok=True)
        print(
            f"\nGenerating Summary Analysis & Saving to: {summary_output_dir})"
        )

        run_summary_analysis(all_results_data, summary_output_dir)
        run_summary_timeseries_analysis(all_session_dataframes,
                                        summary_output_dir)

        run_statistical_analysis(pd.DataFrame(all_results_data),
                                 summary_output_dir)

    else:
        print("\nNo results were collected, skipping summary analysis.")


if __name__ == "__main__":
    main()
