"""
Author: Lu Hou Yang
Updated: Fix Config Attribute Error
"""

import os
import shutil
import threading
import queue
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from tqdm import tqdm

from dataset.processing.affective_state import process_questionnaire_answers_fast, process_questionnaire_answers_markers
from dataset.processing.fixation import process_fixations
from dataset.processing.pottery import voxelize_model
from dataset.processing.sanity_check import analyze_and_plot_point_cloud, generate_original_pointcloud
from dataset.processing.aggregation import generate_gaze_pointcloud_heatmap, generate_voxel_from_mesh, generate_voxel_from_mesh_rgb
from dataset.processing.voice import process_voice_data
import config

# Font
font_path = Path(__file__).parent / "processing" / "ipaexg.ttf"
if font_path.exists():
    try:
        fm.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = 'IPAexGothic'
    except:
        pass


# Helpers
def increment_error(key, path, errors: dict):
    if errors.get(key) == None:
        errors[key] = {'count': 1, 'paths': set([path])}
    else:
        errors[key]['count'] += 1
        errors[key]['paths'].add(path)
    return errors


def check_file_not_empty(file_path, min_size_bytes=10):
    if not os.path.exists(file_path): return False
    try:
        file_size = os.path.getsize(file_path)
        if file_size < min_size_bytes: return False
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return len(content) > 0
    except:
        return False


def save_geometry_threaded(save_path, geometry, error_queue):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _save(path, geom, errq):
        try:
            if isinstance(geom, o3d.geometry.PointCloud):
                o3d.io.write_point_cloud(str(path), geom, write_ascii=True)
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                o3d.io.write_triangle_mesh(str(path), geom, write_ascii=True)
        except Exception as e:
            errq.put({'Save error': str(path)})

    t = threading.Thread(target=_save, args=(save_path, geometry, error_queue))
    t.daemon = True
    t.start()
    return t


def save_plot_threaded(path, fig, error_queue):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _save(p, f, eq):
        try:
            f.savefig(str(p), dpi=300, bbox_inches='tight')
            plt.close(f)
        except:
            eq.put({'Save plot error': str(p)})

    t = threading.Thread(target=_save, args=(path, fig, error_queue))
    t.daemon = True
    t.start()
    return t


def filter_qna_by_emotion_count_and_type(data, min_c, max_c, types):
    """
    Filters data list based on unique answers in the QA file.
    """
    filtered = []
    for d in tqdm(data, desc="Filtering QnA Count"):
        if 'qa' not in d: continue
        try:
            df = pd.read_csv(d['qa'])
            if 'answer' in df.columns:
                cnt = df['answer'].nunique()
                if min_c <= cnt <= max_c: filtered.append(d)
        except:
            pass
    return filtered


def load_and_standardize_dataframe(file_path):
    """
    Loads CSV and extracts Local XYZ coordinates based on Config indices.
    """
    try:
        df = pd.read_csv(file_path, encoding=config.CSV_ENCODING)
        if len(df.columns) >= max(config.CSV_XYZ_INDICES) + 1:
            new_df = pd.DataFrame()
            new_df['estX'] = df.iloc[:, config.CSV_XYZ_INDICES[0]]
            new_df['estY'] = df.iloc[:, config.CSV_XYZ_INDICES[1]]
            new_df['estZ'] = df.iloc[:, config.CSV_XYZ_INDICES[2]]
            ts_idx = config.CSV_TIMESTAMP_INDEX
            if len(df.columns) > ts_idx:
                new_df['timestamp'] = df.iloc[:, ts_idx]
            elif 'timestamp' in df.columns:
                new_df['timestamp'] = df['timestamp']
            else:
                new_df['timestamp'] = 0
            if 'answer' in df.columns: new_df['answer'] = df['answer']
            for c in ['estX', 'estY', 'estZ', 'timestamp']:
                new_df[c] = pd.to_numeric(new_df[c], errors='coerce')
            return new_df
        return df
    except:
        return pd.DataFrame()


# --- Main Function ---
def filter_data_on_condition(
        root="",
        models_path="",
        preprocess=True,
        mode=0,
        hololens_2_spatial_error=config.DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
        target_voxel_resolution=config.DEFAULT_TARGET_VOXEL_RESOLUTION,
        qna_answer_color_map=config.DEFAULT_QNA_ANSWER_COLOR_MAP,
        base_color=config.DEFAULT_BASE_COLOR,
        cmap=config.DEFAULT_CMAP,
        groups=[],
        session_ids=[],
        model_ids=[],
        min_pointcloud_size=0.0,
        min_qa_size=0.0,
        min_voice_quality=0.1,
        min_emotion_count=0,
        max_emotion_count=5,
        emotion_type=[],
        use_cache=True,
        from_tracking_sheet=False,
        tracking_sheet_path="",
        generate_report=True,
        generate_pc_hm_voxel=True,
        generate_qna=True,
        generate_voice=True,
        generate_model_voxel=True,
        generate_sanity_check=False,
        limit=9,
        generate_fixation=False,
        qna_marker=False,
        generate_voxel=True,
        generate_mesh=True,
        generate_pointcloud=True,
        generate_transcript=True,
        language='japan'):
    # Use Universal Maps from Config (Fix for AttributeError)
    label_map = config.UNIVERSAL_LABEL_MAP
    symbol_map = config.UNIVERSAL_SYMBOL_MAP
    shape_map = config.QNA_SHAPE_KEY_MAP

    active_threads = []
    errors = {}
    error_queue = queue.Queue()
    data = []
    model_id_to_path = {}
    unique_model_voxel = set()
    GAUSSIAN_DENOMINATOR = 2 * (hololens_2_spatial_error**2)

    if not Path(root).exists(): raise ValueError(f"Root {root} not found")
    proc_dir = Path(root).parent / 'processed'
    proc_mod_dir = proc_dir / "voxel_models"
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(proc_mod_dir, exist_ok=True)

    # 1. Map Models
    model_all = [f"{k}({v})" for k, v in config.ASSIGNED_NUMBERS_DICT.items()]
    if Path(models_path).exists():
        avail = os.listdir(models_path)
        avail_ids = [p.split('.')[0] for p in avail]
        for p in model_all:
            if p in avail_ids:
                model_id_to_path[p] = Path(models_path) / avail[
                    avail_ids.index(p)]
            else:
                model_id_to_path[p] = ""
            # Note: Original code counted missing pottery errors here

    limit_track = {k: 0 for k in model_all}

    print("\nCHECKING RAW DATA PATHS")
    # 2. Crawl Directories
    # We traverse everything first, then apply filter logic on the collected list

    # Gather all potential paths first
    potential_groups = os.listdir(root)
    unique_group_keys = set(potential_groups)

    # Iterating over file system
    for g in potential_groups:
        g_path = Path(root) / g
        pg_path = proc_dir / g
        if not g_path.is_dir(): continue

        session_keys = os.listdir(g_path)
        for s in tqdm(session_keys, desc=g):
            s_path = g_path / s
            ps_path = pg_path / s
            if not s_path.is_dir(): continue

            model_keys = os.listdir(s_path)
            for p in model_keys:
                if p.endswith('.txt') or p == 'gender.txt': continue

                obj_path = s_path / p
                proc_path = ps_path / p
                if not obj_path.is_dir(): continue

                # File Paths
                pc_path = obj_path / "pointcloud.csv"
                qa_path = obj_path / "qa_corrected.csv"
                mod_path = obj_path / "model.obj"
                bkp_mod_path = Path(
                    models_path) / f"{p}{config.BACKUP_MODEL_EXTENSION}"
                voi_path = obj_path / "session_audio_45s.mp3"
                tsc_path = obj_path / "final_transcript.txt"

                # Output Paths
                d = {
                    'GROUP':
                    g,
                    'SESSION_ID':
                    s,
                    'ID':
                    p,
                    'eye_gaze_intensity_pc':
                    str(proc_path / f"eye_gaze_intensity_pc.ply"),
                    'eye_gaze_intensity_hm_rgb':
                    str(proc_path / f"eye_gaze_intensity_hm_rgb.ply"),
                    'eye_gaze_voxel':
                    str(proc_path / f"eye_gaze_voxel.ply"),
                    'original_pointcloud':
                    str(proc_path / f"original_pointcloud.ply"),
                    'pointcloud_occurrence_plot':
                    str(proc_path / f"pointcloud_occurrence_plot.png"),
                    'qa_pc':
                    str(proc_path / f"qa_pc.ply"),
                    'combined_qa_mesh':
                    str(proc_path / f"combined_qa_mesh.ply"),
                    'qa_segmented_mesh':
                    str(proc_path / "qa_segmented_mesh"),
                    'processed_voice':
                    str(proc_path / f"processed_voice.mp3"),
                    'final_transcript':
                    str(proc_path / f"final_transcript.txt"),
                    'PROCESSED_DATA':
                    str(proc_path)
                }

                # --- VALIDATION LOGIC (From Original) ---
                hm_error = False
                qna_error = False
                voice_error = False

                # Model Check
                if mod_path.exists(): d['model'] = str(mod_path)
                elif bkp_mod_path.exists(): d['model'] = str(bkp_mod_path)
                else:
                    hm_error = True
                    qna_error = True
                    increment_error('Model Missing', str(mod_path), errors)

                # Pointcloud Check
                if pc_path.exists() and os.path.getsize(pc_path) > 0:
                    d['pointcloud'] = str(pc_path)
                    d['POINTCLOUD_SIZE_KB'] = os.path.getsize(pc_path) / 1024
                else:
                    hm_error = True
                    increment_error('PC Missing/Empty', str(pc_path), errors)

                # QA Check
                if qa_path.exists():
                    d['qa'] = str(qa_path)
                    d['QA_SIZE_KB'] = os.path.getsize(qa_path) / 1024
                else:
                    d['QA_SIZE_KB'] = 0
                    qna_error = True
                    increment_error('QA Missing', str(qa_path), errors)

                # Voice Check
                if voi_path.exists(): d['voice'] = str(voi_path)
                else:
                    voice_error = True
                    increment_error('Voice Missing', str(voi_path), errors)

                if check_file_not_empty(tsc_path):
                    d['transcript'] = str(tsc_path)
                else:
                    voice_error = True
                    increment_error('Transcript Missing', str(tsc_path),
                                    errors)

                # --- MODE-BASED ERROR HANDLING (Restored) ---
                if hm_error: continue
                elif qna_error and (mode == 0 or mode == 1): continue
                elif voice_error and (mode == 0 or mode == 2): continue

                # Model Mapping Check
                if p in model_id_to_path and model_id_to_path[p] != "":
                    mf_path = model_id_to_path[p]
                    d['processed_model_path'] = str(
                        proc_mod_dir / f"{mf_path.name.split('.')[0]}.ply")
                    unique_model_voxel.add(str(mf_path))
                else:
                    continue  # Skip if no base model found in library

                if limit_track.get(p, 0) < limit:
                    limit_track[p] += 1
                    data.append(d)

    # 3. FILTERING PHASE
    n_valid_data = len(data)

    # Tracking Sheet Filtering
    if from_tracking_sheet:
        if not Path(tracking_sheet_path).exists():
            raise ValueError("Tracking sheet not found")
        ts_df = pd.read_csv(tracking_sheet_path)
        if 'VOICE_QUALITY_0_TO_5' in ts_df.columns:
            ts_df = ts_df[ts_df['VOICE_QUALITY_0_TO_5'] >= min_voice_quality]

        # Filter data list based on tracking sheet matches
        valid_combos = set(
            zip(ts_df['GROUP'], ts_df['SESSION_ID'], ts_df['ID']))
        # Adjust ID mapping if needed (TS might use raw ID)
        data = [
            x for x in data
            if (x['GROUP'], x['SESSION_ID'], x['ID']) in valid_combos
        ]

    # Arguments Filtering
    # Logic: Keep if it matches criteria
    if groups: unique_group_keys = list(set(unique_group_keys) & set(groups))
    keep_indices = []

    for i, d in enumerate(data):
        is_valid = True
        if groups and d['GROUP'] not in groups: is_valid = False
        if session_ids and d['SESSION_ID'] not in session_ids: is_valid = False
        if model_ids and d['ID'] not in model_ids: is_valid = False
        if d.get('POINTCLOUD_SIZE_KB', 0) < min_pointcloud_size:
            is_valid = False
        if d.get('QA_SIZE_KB', 0) < min_qa_size: is_valid = False

        if is_valid: keep_indices.append(i)

    data = [data[i] for i in keep_indices]

    # QnA Emotion Count Filtering (Based on Mode)
    if mode == 0 or mode == 1:
        if min_emotion_count > 0:
            data = filter_qna_by_emotion_count_and_type(
                data, min_emotion_count, max_emotion_count, emotion_type)

    # 4. PROCESSING PHASE

    # A. Generate Model Voxels (Once per unique model)
    if generate_model_voxel:
        for dp in tqdm(list(unique_model_voxel),
                       desc="Generating Model Voxels"):
            sp = proc_mod_dir / f"{Path(dp).name.split('.')[0]}.ply"
            if use_cache and sp.exists(): pass
            else:
                try:
                    active_threads.append(
                        save_geometry_threaded(
                            str(sp), voxelize_model(dp,
                                                    target_voxel_resolution),
                            error_queue))
                except:
                    pass

    # B. Per-Session Generation
    if preprocess:
        print(f"\nPREPROCESSING {len(data)} items")
        for d in tqdm(data, desc="Processing"):
            os.makedirs(d['PROCESSED_DATA'], exist_ok=True)

            # Sanity Check
            if generate_sanity_check:
                try:
                    plt_obj = analyze_and_plot_point_cloud(d['pointcloud'])
                    active_threads.append(
                        save_plot_threaded(d['pointcloud_occurrence_plot'],
                                           plt_obj, error_queue))
                    if generate_pointcloud:
                        active_threads.append(
                            save_geometry_threaded(
                                d['original_pointcloud'],
                                generate_original_pointcloud(d['pointcloud']),
                                error_queue))
                except:
                    increment_error('Sanity Fail', d['pointcloud'], errors)

            # Aggregation (PC/HM/Voxel)
            if generate_pc_hm_voxel:
                if not (use_cache
                        and Path(d['eye_gaze_intensity_pc']).exists()):
                    try:
                        pc, hm, vert, mesh = generate_gaze_pointcloud_heatmap(
                            d['pointcloud'], d['model'], cmap, base_color,
                            hololens_2_spatial_error, GAUSSIAN_DENOMINATOR)
                        if generate_pointcloud:
                            active_threads.append(
                                save_geometry_threaded(
                                    d['eye_gaze_intensity_pc'], pc,
                                    error_queue))
                        if generate_mesh:
                            active_threads.append(
                                save_geometry_threaded(
                                    d['eye_gaze_intensity_hm_rgb'], hm,
                                    error_queue))

                        if generate_voxel and d['processed_model_path']:
                            vox = generate_voxel_from_mesh(
                                mesh, vert, target_voxel_resolution, cmap,
                                base_color, d['processed_model_path'])
                            active_threads.append(
                                save_geometry_threaded(d['eye_gaze_voxel'],
                                                       vox, error_queue))
                    except:
                        increment_error('Aggregation Fail', d['pointcloud'],
                                        errors)

            # QnA
            if generate_qna and 'qa' in d:
                # Cache Check
                if use_cache and Path(d['qa_pc']).exists() and Path(
                        d['combined_qa_mesh']).exists():
                    pass
                else:
                    try:
                        qa_df = load_and_standardize_dataframe(d['qa'])
                        if qna_marker:
                            res = process_questionnaire_answers_markers(
                                qa_df, d['model'], base_color,
                                qna_answer_color_map, hololens_2_spatial_error,
                                GAUSSIAN_DENOMINATOR, label_map, symbol_map,
                                shape_map, generate_sanity_check)
                        else:
                            res = process_questionnaire_answers_fast(
                                qa_df, d['model'], base_color,
                                qna_answer_color_map, hololens_2_spatial_error,
                                GAUSSIAN_DENOMINATOR, label_map, symbol_map,
                                generate_sanity_check)

                        qa_pc, seg, comb, fig, table = res
                        if generate_pointcloud:
                            active_threads.append(
                                save_geometry_threaded(d['qa_pc'], qa_pc,
                                                       error_queue))
                        if generate_mesh:
                            active_threads.append(
                                save_geometry_threaded(d['combined_qa_mesh'],
                                                       comb, error_queue))

                        os.makedirs(d['qa_segmented_mesh'], exist_ok=True)
                        for n, (m, c) in seg.items():
                            if generate_mesh:
                                active_threads.append(
                                    save_geometry_threaded(
                                        str(
                                            Path(d['qa_segmented_mesh']) /
                                            f"{n}.ply"), m, error_queue))
                            if generate_voxel and d['processed_model_path']:
                                v = generate_voxel_from_mesh_rgb(
                                    m, c, target_voxel_resolution,
                                    d['processed_model_path'])
                                active_threads.append(
                                    save_geometry_threaded(
                                        str(
                                            Path(d['qa_segmented_mesh']) /
                                            f"{n}_voxel.ply"), v, error_queue))

                        if generate_sanity_check:
                            active_threads.append(
                                save_plot_threaded(
                                    str(
                                        Path(d['PROCESSED_DATA']) /
                                        'qa_timeline.png'), fig, error_queue))
                            table.to_csv(Path(d['PROCESSED_DATA']) /
                                         'average_data.csv',
                                         index=False,
                                         encoding='utf-8-sig')
                    except Exception as e:
                        increment_error('QNA Fail', str(d['qa']), errors)

            # Fixations
            if generate_fixation:
                fix_out = Path(d['PROCESSED_DATA']) / "fixations.csv"
                if not (use_cache and fix_out.exists()):
                    try:
                        gz = load_and_standardize_dataframe(d['pointcloud'])
                        gz = gz.rename(columns={
                            'estX': 'x',
                            'estY': 'y',
                            'estZ': 'z'
                        })
                        fix = process_fixations(gz, config.FIXATION_ALGORITHM,
                                                config.FIXATION_PARAMS)
                        fix.to_csv(fix_out, index=False)
                    except:
                        increment_error('Fixation Fail', d['pointcloud'],
                                        errors)

            # Voice
            if generate_voice and 'voice' in d:
                try:
                    shutil.copy(d['voice'], d['processed_voice'])
                except:
                    pass
            if generate_transcript and 'transcript' in d:
                try:
                    shutil.copy(d['transcript'], d['final_transcript'])
                except:
                    pass

    for t in active_threads:
        t.join()

    # Process Errors Queue
    while not error_queue.empty():
        err_item = error_queue.get()
        for k, v in err_item.items():
            increment_error(k, v, errors)

    return data, errors
