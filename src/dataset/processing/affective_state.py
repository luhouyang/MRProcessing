import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

# --- Geometry Helpers ---
def _create_base_dot_geometry(size):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=size/1.5); m.compute_vertex_normals(); return m
def _create_base_square_geometry(size):
    m = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size); m.compute_vertex_normals(); return m
def _create_base_triangle_geometry(size):
    m = o3d.geometry.TriangleMesh.create_tetrahedron(radius=size); m.compute_vertex_normals(); return m
def _create_diamond_geometry(size):
    m = o3d.geometry.TriangleMesh.create_octahedron(radius=size); m.compute_vertex_normals(); return m
def _create_base_x_geometry(size):
    t = size/4.0
    a1 = o3d.geometry.TriangleMesh.create_box(width=size, height=t, depth=t)
    a2 = deepcopy(a1)
    a1.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0,0,np.pi/4)), center=(0,0,0))
    a2.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0,0,-np.pi/4)), center=(0,0,0))
    m = a1+a2; m2 = deepcopy(m)
    m2.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0,0,np.pi)), center=(0,0,0))
    m2.translate((0, t, 0)); m += m2; m.compute_vertex_normals(); return m

# --- Main Logic ---
def process_questionnaire_answers_markers(
    df_or_path, model_file, base_color, qna_answer_color_map,
    hololens_2_spatial_error, gaussian_denominator,
    label_map, symbol_map, shape_map, sanity_check=False,
):
    # 1. Load
    if isinstance(df_or_path, str): df = pd.read_csv(df_or_path)
    else: df = df_or_path.copy()

    for c in ["estX", "estY", "estZ", "timestamp"]: 
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "answer" in df.columns: df["answer"] = df["answer"].astype(str).str.strip()
    df.dropna(subset=["estX", "estY", "estZ", "answer", "timestamp"], inplace=True)
    
    # Map Labels
    df['answer_short'] = df['answer'].map(label_map).fillna(df['answer'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2. Mesh
    mesh = o3d.io.read_triangle_mesh(model_file)
    if not mesh.has_vertices(): raise ValueError("No Vertices")
    marker_size = np.max(mesh.get_max_bound() - mesh.get_min_bound()) / 55

    # 3. Place Markers
    geom_cache = {
        'dot': _create_base_dot_geometry(marker_size),
        'square': _create_base_square_geometry(marker_size),
        'triangle': _create_base_triangle_geometry(marker_size),
        'diamond': _create_diamond_geometry(marker_size),
        'x': _create_base_x_geometry(marker_size),
    }
    
    marker_meshes = []
    for answer, grp in df.groupby("answer"):
        if answer not in qna_answer_color_map: continue
        shape_key = shape_map.get(answer, 'dot')
        tmpl = geom_cache.get(shape_key, geom_cache['dot'])
        col = np.array(qna_answer_color_map[answer]["rgb"]) / 255.0
        pts = grp[["estX", "estY", "estZ"]].values
        for i in range(0, len(pts), 3):
            inst = deepcopy(tmpl); inst.paint_uniform_color(col)
            inst.translate(pts[i], relative=False)
            marker_meshes.append(inst)
    
    shape_mesh = o3d.geometry.TriangleMesh()
    for m in marker_meshes: shape_mesh += m

    # 4. Gaussian
    verts = np.asarray(mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    
    final_c = np.zeros((verts.shape[0], 3))
    weights = np.zeros(verts.shape[0])
    qa_seg = {}
    
    for answer in df["answer"].unique():
        if answer not in qna_answer_color_map: continue
        pts = df[df["answer"]==answer][["estX", "estY", "estZ"]].values
        q_pts = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
        ids = scene.compute_closest_points(q_pts)["primitive_ids"].numpy()
        hits = set()
        tris = np.asarray(mesh.triangles)
        for fid in ids:
            if fid!=o3d.t.geometry.RaycastingScene.INVALID_ID: hits.update(tris[fid])
        
        col = np.array(qna_answer_color_map[answer]["rgb"]) / 255.0
        seg_c = np.tile(np.array(base_color), (verts.shape[0], 1))
        aff = set()
        
        for v in list(hits):
            [k, idxs, dists] = kdtree.search_radius_vector_3d(verts[v], hololens_2_spatial_error)
            if k>0:
                w = np.exp(-np.asarray(dists)/gaussian_denominator)
                final_c[idxs] += col * w[:,None]; weights[idxs] += w
                aff.update(idxs)
        
        seg_c[list(aff)] = col
        sm = o3d.geometry.TriangleMesh(mesh); sm.vertex_colors = o3d.utility.Vector3dVector(seg_c)
        qa_seg[qna_answer_color_map[answer]["name"]] = [sm, seg_c]

    mask = weights > 1e-9
    final_c[mask] /= weights[mask,None]; final_c[~mask] = base_color
    comb_mesh = o3d.geometry.TriangleMesh(mesh)
    comb_mesh.vertex_colors = o3d.utility.Vector3dVector(final_c)

    # 5. Timeline & Table
    df['grp'] = ((df['answer']!=df['answer'].shift())|(df['timestamp'].diff()>0.05)).cumsum()
    blocks = df.groupby('grp').agg(start=('timestamp','min'), end=('timestamp','max'), ans=('answer','first')).reset_index()
    blocks['dur'] = blocks['end'] - blocks['start']
    
    fig, ax = plt.subplots(figsize=(15,2))
    cols = [np.array(qna_answer_color_map.get(a,{'rgb':[128,128,128]})['rgb'])/255.0 for a in blocks['ans']]
    ax.barh([0]*len(blocks), blocks['dur'], left=blocks['start'], color=cols, height=1)
    patches = [mpatches.Patch(color=np.array(v['rgb'])/255.0, label=k) for k,v in qna_answer_color_map.items()]
    ax.legend(handles=patches, bbox_to_anchor=(1.02,1))
    
    final_table = pd.DataFrame()
    if sanity_check:
        sums = blocks.groupby('ans')['dur'].sum().reset_index()
        tot = sums['dur'].sum()
        rows = []
        for _, r in sums.iterrows():
            ans_raw = r['ans']
            ans_clean = str(ans_raw).strip()
            
            # -- Robust Symbol Search Strategy --
            # 1. Try exact match
            sym = symbol_map.get(ans_clean)
            # 2. Try mapped short label (e.g., "Interesting..." -> "Interesting")
            if not sym:
                short = label_map.get(ans_clean)
                if short: sym = symbol_map.get(short)
            # 3. Last Resort: First Character (e.g. "I" for Interesting)
            if not sym and len(ans_clean) > 0:
                sym = ans_clean[0].upper()
            
            rows.append({
                'Symbol': sym,
                'Class': label_map.get(ans_clean, ans_clean),
                'Time (ms)': round(r['dur']*1000, 0),
                '%': round(r['dur']/tot*100, 1) if tot>0 else 0
            })
        final_table = pd.DataFrame(rows)

    return shape_mesh, qa_seg, comb_mesh, fig, final_table

def process_questionnaire_answers_fast(df, model, base, cmap, err, gauss, lmap, smap, check=False):
    dmap = {k: 'dot' for k in cmap.keys()}
    return process_questionnaire_answers_markers(df, model, base, cmap, err, gauss, lmap, smap, dmap, check)