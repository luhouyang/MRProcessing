from copy import deepcopy
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

# yapf: disable
def _calculate_smoothed_vertex_intensities(
    gaze_points_np,
    mesh,
    hololens_2_spatial_error,
    gaussian_denominator,
):
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]

    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query_points = o3d.core.Tensor(gaze_points_np, dtype=o3d.core.Dtype.Float32)
    closest_geometry = mesh_scene.compute_closest_points(query_points)
    closest_face_indices = closest_geometry['primitive_ids'].numpy()

    raw_hit_counts = np.zeros(n_vertices, dtype=np.float64)
    point_to_face_map = np.empty(gaze_points_np.shape[0], dtype=int)

    for i, closest_face_idx in tqdm(enumerate(closest_face_indices), desc="Raycasting Gaze Points", leave=False):
        if closest_face_idx != o3d.t.geometry.RaycastingScene.INVALID_ID:
            point_to_face_map[i] = closest_face_idx
            for v_idx in mesh_triangles_np[closest_face_idx]:
                raw_hit_counts[v_idx] += 1

    raw_hit_counts = np.log1p(raw_hit_counts)

    kdtree = o3d.geometry.KDTreeFlann(mesh)
    interpolated_heatmap_values = np.copy(raw_hit_counts)
    hit_vertices_indices = np.where(raw_hit_counts > 0)[0]

    for start_node_idx in tqdm(hit_vertices_indices, desc="Applying Gaussian Spread", leave=False):
        hit_value = raw_hit_counts[start_node_idx]
        [k, indices, euclidean_dist] = kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
        if k > 1:
            gaussian_weights = np.exp(-np.asarray(euclidean_dist)**2 / gaussian_denominator)
            for i, neighbor_idx in enumerate(indices):
                if neighbor_idx != start_node_idx:
                    interpolated_heatmap_values[neighbor_idx] += hit_value * gaussian_weights[i]

    return interpolated_heatmap_values, point_to_face_map
# yapf: enable


# yapf: disable
def generate_gaze_pointcloud_heatmap(
    input_file,
    model_file,
    cmap,
    base_color,
    hololens_2_spatial_error,
    gaussian_denominator,
):
    if isinstance(input_file, str):
        df = pd.read_csv(input_file)
    else:
        df = input_file.copy()

    if len(df.columns) >= 9:
        gaze_points_np = df.iloc[:, [0, 1, 2]].to_numpy()
    elif 'estX' in df.columns:
        gaze_points_np = df[['estX', 'estY', 'estZ']].to_numpy()
    else:
        gaze_points_np = df.iloc[:, :3].to_numpy()

    mesh = o3d.io.read_triangle_mesh(model_file)
    if not mesh.has_vertices():
        raise ValueError(f"Mesh file '{model_file}' contains no vertices.")
    mesh.compute_vertex_normals()

    final_vertex_intensities, point_to_face_map = _calculate_smoothed_vertex_intensities(
        gaze_points_np=gaze_points_np,
        mesh=mesh,
        hololens_2_spatial_error=hololens_2_spatial_error,
        gaussian_denominator=gaussian_denominator,
    )

    max_mesh = np.max(final_vertex_intensities)
    normalized_vertex_intensities = final_vertex_intensities / max_mesh if max_mesh > 0 else np.zeros_like(final_vertex_intensities)

    mesh_vertex_colors = cmap(normalized_vertex_intensities)[:, :3]
    mesh_vertex_colors[final_vertex_intensities < 1e-9] = base_color
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)

    mesh_greyscale = deepcopy(mesh)
    greyscale_colors = np.repeat(normalized_vertex_intensities[:, np.newaxis], 3, axis=1)
    mesh_greyscale.vertex_colors = o3d.utility.Vector3dVector(greyscale_colors)

    mesh_triangles_np = np.asarray(mesh.triangles)
    final_point_intensities = []
    
    for face_idx in tqdm(point_to_face_map, desc="Making Intensity Point Cloud", leave=False):
        if face_idx < len(mesh_triangles_np):
            final_point_intensities.append(np.mean(final_vertex_intensities[mesh_triangles_np[face_idx]]))
        else:
            final_point_intensities.append(0.0)
            
    final_point_intensities = np.array(final_point_intensities)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gaze_points_np))
    max_pc = np.max(final_point_intensities)
    normalized_point_intensities = final_point_intensities / max_pc if max_pc > 0 else np.zeros_like(final_point_intensities)

    pc_colors = cmap(normalized_point_intensities)[:, :3]
    pc_colors[final_point_intensities < 1e-9] = base_color
    pcd.colors = o3d.utility.Vector3dVector(pc_colors)

    return pcd, mesh, final_vertex_intensities, mesh_greyscale
# yapf: enable


# yapf: disable
def generate_voxel_from_mesh(
    mesh,
    vertex_intensities,
    target_voxel_resolution,
    cmap,
    base_color,
    base_model_pcd=None, # Renamed from base_pottery_pcd
):
    if mesh is None or vertex_intensities is None:
        raise ValueError("Skipping voxel heatmap: Missing mesh or intensity data.")

    if not mesh.has_triangles() or not mesh.has_vertices():
        raise ValueError("Skipping voxel heatmap: Mesh has no triangles or vertices.")

    if base_model_pcd is not None:
        if isinstance(base_model_pcd, str):
             base_model_pcd = o3d.io.read_point_cloud(base_model_pcd)

        if len(base_model_pcd.points) == 0:
            return o3d.geometry.PointCloud()

        if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()

        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh_t)

        query_points = o3d.core.Tensor(np.asarray(base_model_pcd.points), dtype=o3d.core.Dtype.Float32)
        closest_points_ans = scene.compute_closest_points(query_points)

        triangle_ids = closest_points_ans['primitive_ids'].numpy()
        closest_surface_points = closest_points_ans['points'].numpy()

        mesh_vertices = np.asarray(mesh.vertices)
        mesh_triangles = np.asarray(mesh.triangles)
        hit_tri_vertices = mesh_vertices[mesh_triangles[triangle_ids]]
        a, b, c = hit_tri_vertices[:, 0], hit_tri_vertices[:, 1], hit_tri_vertices[:, 2]

        v0, v1, v2 = b - a, c - a, closest_surface_points - a
        d00 = np.einsum('ij,ij->i', v0, v0)
        d01 = np.einsum('ij,ij->i', v0, v1)
        d11 = np.einsum('ij,ij->i', v1, v1)
        d20 = np.einsum('ij,ij->i', v2, v0)
        d21 = np.einsum('ij,ij->i', v2, v1)

        denom = d00 * d11 - d01 * d01
        denom[np.abs(denom) < 1e-9] = 1e-9

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        bary_coords = np.vstack([u, v, w]).T

        tri_intensities = vertex_intensities[mesh_triangles[triangle_ids]]
        final_intensities = np.einsum('ij,ij->i', bary_coords, tri_intensities)

        max_val = np.max(final_intensities) if len(final_intensities) > 0 else 0
        normalized_intensities = final_intensities / max_val if max_val > 1e-9 else np.zeros_like(final_intensities)

        colors = np.repeat(normalized_intensities[:, np.newaxis], 3, axis=1)
        colors[normalized_intensities < 1e-9] = base_color

        heatmap_pcd = o3d.geometry.PointCloud()
        heatmap_pcd.points = base_model_pcd.points
        heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)
        return heatmap_pcd

    else:
        # Standard Voxel Generation
        mesh_vertices_np = np.asarray(mesh.vertices)
        mesh_triangles_np = np.asarray(mesh.triangles)
        min_bound = mesh.get_min_bound()
        max_range = np.max(mesh.get_max_bound() - min_bound)
        voxel_size = max_range / (target_voxel_resolution - 1)
        voxel_size_sq = voxel_size**2

        tri_vertices = mesh_vertices_np[mesh_triangles_np]
        tri_intensities = vertex_intensities[mesh_triangles_np]

        v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
        edge1, edge2 = v1 - v0, v2 - v0
        triangle_areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)

        num_samples_per_triangle = np.ceil(triangle_areas / voxel_size_sq).astype(int) + 10
        total_samples = np.sum(num_samples_per_triangle)
        
        if total_samples <= 0: return o3d.geometry.PointCloud()

        triangle_indices = np.repeat(np.arange(len(mesh_triangles_np)), num_samples_per_triangle)
        rand_points = np.random.rand(total_samples, 2)
        rand_points_sum = np.sum(rand_points, axis=1)
        rand_points[rand_points_sum > 1] = 1 - rand_points[rand_points_sum > 1]

        bary_coords = np.zeros((total_samples, 3))
        bary_coords[:, 0] = 1 - rand_points[:, 0] - rand_points[:, 1]
        bary_coords[:, 1] = rand_points[:, 0]
        bary_coords[:, 2] = rand_points[:, 1]

        all_sample_points = np.einsum('ij,ijk->ik', bary_coords, tri_vertices[triangle_indices])
        all_interpolated_intensities = np.einsum('ij,ij->i', bary_coords, tri_intensities[triangle_indices])

        voxel_coords_all = np.floor((all_sample_points - min_bound) / voxel_size).astype(int)

        df = pd.DataFrame(voxel_coords_all, columns=['x', 'y', 'z'])
        df['intensity'] = all_interpolated_intensities

        voxel_data_df = df.groupby(['x', 'y', 'z'])['intensity'].max()

        final_coords_np = np.array(voxel_data_df.index.to_list())
        final_intensities_np = voxel_data_df.to_numpy()
        voxel_points = min_bound + (final_coords_np + 0.5) * voxel_size

        max_val = np.max(final_intensities_np)
        normalized_intensities = final_intensities_np / max_val if max_val > 1e-9 else np.zeros_like(final_intensities_np)

        colors = cmap(normalized_intensities)[:, :3]
        colors[normalized_intensities < 1e-9] = base_color

        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
        voxel_pcd.colors = o3d.utility.Vector3dVector(colors)
        return voxel_pcd
# yapf: enable

# yapf: disable
def generate_voxel_from_mesh_rgb(
    mesh,
    vertex_colors,
    target_voxel_resolution,
    base_model_pcd=None, # Renamed
):
    if mesh is None or vertex_colors is None: raise ValueError("Missing data")
    vertex_colors = np.asarray(vertex_colors)

    if base_model_pcd is not None:
        if isinstance(base_model_pcd, str): base_model_pcd = o3d.io.read_point_cloud(base_model_pcd)
        if len(base_model_pcd.points) == 0: return o3d.geometry.PointCloud()
        if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()

        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh_t)

        query_points = o3d.core.Tensor(np.asarray(base_model_pcd.points), dtype=o3d.core.Dtype.Float32)
        closest = scene.compute_closest_points(query_points)
        
        tri_ids = closest['primitive_ids'].numpy()
        surf_pts = closest['points'].numpy()
        if len(tri_ids) == 0: return o3d.geometry.PointCloud()

        mesh_verts = np.asarray(mesh.vertices)
        mesh_tris = np.asarray(mesh.triangles)
        hit_verts = mesh_verts[mesh_tris[tri_ids]]
        a, b, c = hit_verts[:, 0], hit_verts[:, 1], hit_verts[:, 2]

        v0, v1, v2 = b - a, c - a, surf_pts - a
        d00 = np.einsum('ij,ij->i', v0, v0)
        d01 = np.einsum('ij,ij->i', v0, v1)
        d11 = np.einsum('ij,ij->i', v1, v1)
        d20 = np.einsum('ij,ij->i', v2, v0)
        d21 = np.einsum('ij,ij->i', v2, v1)
        denom = d00 * d11 - d01 * d01
        denom[np.abs(denom) < 1e-9] = 1e-9

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        bary = np.vstack([u, v, w]).T

        # Interpolate RGB
        tri_cols = vertex_colors[mesh_tris[tri_ids]]
        final_cols = np.einsum('ij,ijk->ik', bary, tri_cols)

        heatmap_pcd = o3d.geometry.PointCloud()
        heatmap_pcd.points = base_model_pcd.points
        heatmap_pcd.colors = o3d.utility.Vector3dVector(final_cols)
        return heatmap_pcd

    else:
        mesh_vertices_np = np.asarray(mesh.vertices)
        mesh_triangles_np = np.asarray(mesh.triangles)
        min_bound = mesh.get_min_bound()
        max_range = np.max(mesh.get_max_bound() - min_bound)
        voxel_size = max_range / (target_voxel_resolution - 1)
        voxel_size_sq = voxel_size**2

        tri_vertices = mesh_vertices_np[mesh_triangles_np]
        tri_vertex_colors = vertex_colors[mesh_triangles_np]

        v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
        edge1, edge2 = v1 - v0, v2 - v0
        triangle_areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)

        num_samples_per_triangle = np.ceil(triangle_areas / voxel_size_sq).astype(int) + 10
        total_samples = np.sum(num_samples_per_triangle)
        
        if total_samples <= 0: return o3d.geometry.PointCloud()

        triangle_indices = np.repeat(np.arange(len(mesh_triangles_np)), num_samples_per_triangle)
        rand_points = np.random.rand(total_samples, 2)
        rand_points_sum = np.sum(rand_points, axis=1)
        rand_points[rand_points_sum > 1] = 1 - rand_points[rand_points_sum > 1]

        bary_coords = np.zeros((total_samples, 3))
        bary_coords[:, 0] = 1 - rand_points[:, 0] - rand_points[:, 1]
        bary_coords[:, 1] = rand_points[:, 0]
        bary_coords[:, 2] = rand_points[:, 1]

        all_sample_points = np.einsum('ij,ijk->ik', bary_coords, tri_vertices[triangle_indices])
        all_sample_colors = np.einsum('ij,ijk->ik', bary_coords, tri_vertex_colors[triangle_indices])

        voxel_coords_all = np.floor((all_sample_points - min_bound) / voxel_size).astype(int)

        df = pd.DataFrame(voxel_coords_all, columns=['x', 'y', 'z'])
        df[['r', 'g', 'b']] = all_sample_colors

        # Mean Aggregation for RGB
        voxel_data_df = df.groupby(['x', 'y', 'z'])[['r', 'g', 'b']].mean()

        final_coords_np = np.array(voxel_data_df.index.to_list())
        final_colors = voxel_data_df.to_numpy()
        voxel_points = min_bound + (final_coords_np + 0.5) * voxel_size

        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
        voxel_pcd.colors = o3d.utility.Vector3dVector(final_colors)

        return voxel_pcd
# yapf: enable