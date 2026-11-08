"""
Custom Cattle Input Processing Module
=====================================
Handles arbitrary / random multi-camera RGB-D image uploads,
rigid extrinsic transformations, U²-Net matting, DBSCAN/SOR filtering,
volume integration, and live weight estimation.
"""

from pathlib import Path
import cv2
import numpy as np
import open3d as o3d


# Default Kinect v2 camera intrinsics (424x512 depth / registered RGB)
DEFAULT_FX = 365.1
DEFAULT_FY = -359.5
DEFAULT_CX = 256.0
DEFAULT_CY = 212.0

# Standard calibrated transformation matrices (from reference multi-camera chute setup)
DEFAULT_LEFT_TO_TOP = np.array([
    [ 1.0000,  0.0000,  0.0000,  0.4900],
    [ 0.0000,  0.1200,  0.9900, -2.4500],
    [ 0.0000, -0.9900,  0.1200,  1.8400],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float64)

DEFAULT_RIGHT_TO_TOP = np.array([
    [-1.0000, -0.0100,  0.0900,  0.0200],
    [-0.0900,  0.0000, -1.0000,  2.2200],
    [ 0.0200, -1.0000,  0.0000,  2.1700],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float64)


def colorize_depth(depth_img: np.ndarray) -> np.ndarray:
    """Creates a false-color Viridis visualization for depth maps."""
    if depth_img is None or depth_img.size == 0:
        return np.zeros((424, 512, 3), dtype=np.uint8)

    valid = (depth_img > 400) & (depth_img < 7000)
    if not np.any(valid):
        valid = depth_img > 0

    if not np.any(valid):
        return np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)

    min_val, max_val = float(depth_img[valid].min()), float(depth_img[valid].max())
    d_norm = np.zeros_like(depth_img, dtype=np.float32)
    d_norm[valid] = (depth_img[valid] - min_val) / max(1.0, max_val - min_val) * 255.0

    colorized = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colorized[~valid] = [25, 25, 25]
    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)


def parse_matrix_from_text(mat_str: str) -> np.ndarray:
    """Parses a 4x4 SE(3) matrix from multiline string."""
    rows = []
    for line in mat_str.strip().splitlines():
        parts = [float(v) for v in line.replace(",", " ").split() if v.strip()]
        if len(parts) == 4:
            rows.append(parts)
    if len(rows) == 4:
        return np.array(rows, dtype=np.float64)
    raise ValueError(f"Expected 4 rows of 4 numbers, got {len(rows)} rows.")


def parse_matrices_file(content: str) -> tuple[np.ndarray, np.ndarray]:
    """Parses Left -> Top and Right -> Top matrices from a .matrices file string."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    left_idx = next((i + 1 for i, line in enumerate(lines) if "Left -> Top" in line), -1)
    right_idx = next((i + 1 for i, line in enumerate(lines) if "Right -> Top" in line), -1)

    if left_idx == -1 or right_idx == -1:
        raise ValueError("Matrix file must contain 'Left -> Top' and 'Right -> Top' sections.")

    left_m = np.array([[float(v) for v in lines[left_idx + r].split()] for r in range(4)], dtype=np.float64)
    right_m = np.array([[float(v) for v in lines[right_idx + r].split()] for r in range(4)], dtype=np.float64)
    return left_m, right_m


def format_matrix_to_text(mat: np.ndarray) -> str:
    """Formats a 4x4 matrix into a clean multiline string."""
    return "\n".join("  ".join(f"{val:9.4f}" for val in row) for row in mat)


def segment_view_rgbd(
    rgb_bgr: np.ndarray,
    depth_img: np.ndarray,
    session,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
):
    """Executes salient U²-Net matting and projects valid pixels into 3D camera coordinates."""
    if rgb_bgr.shape[:2] != (424, 512):
        rgb_bgr = cv2.resize(rgb_bgr, (512, 424), interpolation=cv2.INTER_LINEAR)
    if depth_img.shape[:2] != (424, 512):
        depth_img = cv2.resize(depth_img, (512, 424), interpolation=cv2.INTER_NEAREST)

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # 1. U²-Net Matting & Morphological Closing
    import rembg
    alpha = rembg.remove(rgb, session=session, only_mask=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = (closed_alpha >= 180).astype(np.uint8)

    # 2. Keep components >= 3% area or >= 300 px
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    total_fg = stats[1:, cv2.CC_STAT_AREA].sum() if num_labels > 1 else 0
    min_keep_area = max(300, int(0.03 * total_fg))

    final_mask = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_keep_area:
            final_mask[labels == lbl] = 255

    # 3. Project RGB-D into metric 3D point cloud
    z = depth_img.astype(np.float32)
    if depth_img.dtype == np.uint16 or np.max(z) > 50:
        z /= 1000.0  # mm to meters

    u, v = np.meshgrid(np.arange(512), np.arange(424))
    valid = (final_mask > 0) & (z > 0.4) & (z < 6.0) & (~np.isnan(z))

    z_pts = z[valid]
    x_pts = (u[valid] - cx) * z_pts / fx
    y_pts = (v[valid] - cy) * z_pts / fy
    pts = np.column_stack([x_pts, y_pts, z_pts])
    colors = rgb[valid].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    seg_rgb = cv2.bitwise_and(rgb, rgb, mask=final_mask)
    return pcd, rgb, final_mask, seg_rgb


def merge_and_filter_custom_clouds(
    top_pcd: o3d.geometry.PointCloud,
    left_pcd: o3d.geometry.PointCloud,
    right_pcd: o3d.geometry.PointCloud,
    left_to_top: np.ndarray,
    right_to_top: np.ndarray,
    clean_out_path: Path,
):
    """Transforms Left and Right clouds into Top coordinates and applies DBSCAN + SOR filtering."""
    l_pcd = o3d.geometry.PointCloud(left_pcd)
    r_pcd = o3d.geometry.PointCloud(right_pcd)
    t_pcd = o3d.geometry.PointCloud(top_pcd)

    l_pcd.transform(left_to_top)
    r_pcd.transform(right_to_top)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(t_pcd.points), np.asarray(l_pcd.points), np.asarray(r_pcd.points)])
    )
    merged_pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(t_pcd.colors), np.asarray(l_pcd.colors), np.asarray(r_pcd.colors)])
    )
    raw_count = len(merged_pcd.points)

    # DBSCAN
    labels = np.array(merged_pcd.cluster_dbscan(eps=0.08, min_points=50, print_progress=False))
    if len(labels) > 0 and (labels >= 0).any():
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        cluster_pcd = merged_pcd.select_by_index(np.where(labels == unique_labels[np.argmax(counts)])[0])
    else:
        cluster_pcd = merged_pcd

    # SOR
    clean_pcd, _ = cluster_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    clean_out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(clean_out_path), clean_pcd)

    return clean_pcd, raw_count, len(clean_pcd.points)
