"""
Multi-View Point Cloud Merge
============================
Transforms Left and Right segmented point clouds into the Top camera coordinate
system using calibrated SE(3) transformation matrices, then merges into a single PLY.
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def load_matrix(matrix_file: Path, header: str) -> np.ndarray:
    """Extracts a 4x4 matrix following the specified header in a .matrices file."""
    with open(matrix_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    idx = next(i for i, line in enumerate(lines) if header in line) + 1
    return np.array([[float(v) for v in lines[idx + r].split()] for r in range(4)], dtype=np.float64)


def merge_views(cow_id: int, base_dir: Path) -> Path:
    """Merges Left, Right, and Top clouds for cow_id and saves to aligned_output/."""
    seg_dir = base_dir / "segmented_output" / f"cow_{cow_id}"
    matrix_file = base_dir / "data" / "Matrices" / f"{cow_id}.matrices"
    output_dir = base_dir / "aligned_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    left_pcd = o3d.io.read_point_cloud(str(seg_dir / "cloud_segmented_left.ply"))
    right_pcd = o3d.io.read_point_cloud(str(seg_dir / "cloud_segmented_right.ply"))
    top_pcd = o3d.io.read_point_cloud(str(seg_dir / "cloud_segmented_top.ply"))

    left_pcd.transform(load_matrix(matrix_file, "Left -> Top"))
    right_pcd.transform(load_matrix(matrix_file, "Right -> Top"))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(top_pcd.points), np.asarray(left_pcd.points), np.asarray(right_pcd.points)])
    )
    merged_pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(top_pcd.colors), np.asarray(left_pcd.colors), np.asarray(right_pcd.colors)])
    )

    out_path = output_dir / f"cow_{cow_id}_merged.ply"
    o3d.io.write_point_cloud(str(out_path), merged_pcd)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Multi-View Point Cloud Merge")
    parser.add_argument("--cow", type=int, default=1, help="Cow ID (default: 1)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_path = merge_views(args.cow, base_dir)
    print(f"Cow #{args.cow}: Saved merged point cloud to {out_path}")


if __name__ == "__main__":
    main()
