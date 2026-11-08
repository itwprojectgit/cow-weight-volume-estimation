"""
Point Cloud Refinement: DBSCAN Density Clustering & SOR Outlier Removal
======================================================================
1. DBSCAN: Isolates the primary connected animal body cluster.
2. Statistical Outlier Removal (SOR): Eliminates sparse boundary noise.
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def remove_outer_noise(
    input_ply: Path,
    output_ply: Path,
    cluster_eps_m: float = 0.08,
    min_points: int = 50,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
):
    """Filters outer enclosure artifacts and noise from merged cattle point clouds."""
    pcd = o3d.io.read_point_cloud(str(input_ply))

    # 1. DBSCAN Density Clustering
    labels = np.array(pcd.cluster_dbscan(eps=cluster_eps_m, min_points=min_points, print_progress=False))
    valid_labels = labels[labels >= 0]

    if len(valid_labels) > 0:
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        primary_cluster = unique_labels[np.argmax(counts)]
        body_pcd = pcd.select_by_index(np.where(labels == primary_cluster)[0])
    else:
        body_pcd = pcd

    # 2. Statistical Outlier Removal (SOR)
    clean_pcd, _ = body_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply), clean_pcd)
    return clean_pcd


def main():
    parser = argparse.ArgumentParser(description="Point Cloud Refinement")
    parser.add_argument("--cow", type=int, default=1, help="Cow ID (default: 1)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    in_path = base_dir / "aligned_output" / f"cow_{args.cow}_merged.ply"
    out_path = base_dir / "aligned_output" / f"cow_{args.cow}_clean.ply"
    clean_pcd = remove_outer_noise(in_path, out_path)
    print(f"Cow #{args.cow}: Refined clean cloud saved to {out_path} ({len(clean_pcd.points):,} points)")


if __name__ == "__main__":
    main()
