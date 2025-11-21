import pandas as pd
import numpy as np
from . import aggregation 
import config

def calculate_velocity_3d(x, y, z, time):
    """Calculate 3D velocity (units/sec) if time is sec."""
    dx = np.diff(x); dy = np.diff(y); dz = np.diff(z); dt = np.diff(time)
    dt[dt == 0] = 1e-6 
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return np.concatenate(([0], dist / dt))

def process_fixations(gaze_data, algorithm=None, params=None):
    if gaze_data is None or gaze_data.empty: return pd.DataFrame()
    if algorithm is None: algorithm = config.FIXATION_ALGORITHM
    if params is None: params = config.FIXATION_PARAMS
    
    if 'z' not in gaze_data.columns: gaze_data['z'] = 0.0
    
    # --- Time Unit Detection ---
    # If max timestamp is small (< 1 day in seconds), assume Seconds.
    # Duration Threshold is in MS.
    is_seconds = False
    if gaze_data['timestamp'].max() < 86400: # 24 hours
        is_seconds = True
    
    # --- 1. Aggregation ---
    if algorithm == 'aggregation':
        return aggregation.aggregate_fixations(gaze_data)

    # --- 2. I-VT (3D Velocity) ---
    elif algorithm == 'velocity':
        threshold = params.get('velocity_threshold', 1.0)
        min_dur_ms = params.get('duration_threshold', 100)
        min_points = params.get('min_points', 2)
        
        df = gaze_data.copy()
        df['velocity'] = calculate_velocity_3d(df['x'], df['y'], df['z'], df['timestamp'])
        df['is_fixation'] = df['velocity'] < threshold
        df['group'] = (df['is_fixation'] != df['is_fixation'].shift()).cumsum()
        
        fixations = []
        for _, group in df[df['is_fixation']].groupby('group'):
            if len(group) < min_points: continue
            
            dur = group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]
            # Convert duration to MS for comparison if needed
            dur_ms = dur * 1000 if is_seconds else dur
            
            if dur_ms >= min_dur_ms:
                fixations.append({
                    'start_time': group['timestamp'].iloc[0],
                    'end_time': group['timestamp'].iloc[-1],
                    'duration_ms': dur_ms,
                    'x': group['x'].mean(), 'y': group['y'].mean(), 'z': group['z'].mean(),
                    'count': len(group)
                })
        return pd.DataFrame(fixations)

    # --- 3. I-DT (3D Dispersion) ---
    elif algorithm == 'dispersion':
        threshold = params.get('dispersion_threshold', 0.1)
        min_dur_ms = params.get('duration_threshold', 100)
        
        points = gaze_data[['x', 'y', 'z', 'timestamp']].to_dict('records')
        fixations = []
        n = len(points); window_size = 5; i = 0
        
        while i < n - window_size:
            window = points[i : i+window_size]
            xs = [p['x'] for p in window]; ys = [p['y'] for p in window]; zs = [p['z'] for p in window]
            
            disp = (max(xs)-min(xs)) + (max(ys)-min(ys)) + (max(zs)-min(zs))
            
            if disp <= threshold:
                while i + window_size < n:
                    nw = points[i : i+window_size+1]
                    nxs = [p['x'] for p in nw]; nys = [p['y'] for p in nw]; nzs = [p['z'] for p in nw]
                    ndisp = (max(nxs)-min(nxs)) + (max(nys)-min(nys)) + (max(nzs)-min(nzs))
                    if ndisp <= threshold: window_size += 1
                    else: break
                
                start_t = points[i]['timestamp']
                end_t = points[i+window_size-1]['timestamp']
                dur = end_t - start_t
                dur_ms = dur * 1000 if is_seconds else dur
                
                if dur_ms >= min_dur_ms:
                    fixations.append({
                        'start_time': start_t, 'end_time': end_t, 'duration_ms': dur_ms,
                        'x': np.mean([p['x'] for p in points[i:i+window_size]]),
                        'y': np.mean([p['y'] for p in points[i:i+window_size]]),
                        'z': np.mean([p['z'] for p in points[i:i+window_size]])
                    })
                i += window_size; window_size = 5 
            else: i += 1
        return pd.DataFrame(fixations)
        
    return pd.DataFrame()