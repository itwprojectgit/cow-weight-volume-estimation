"""
Batch Processing Pipeline for All 103 Cattle
=============================================
Clears output directories and executes the complete 3-view segmentation,
metric SE(3) transformation, and DBSCAN/SOR filtering pipeline for all 103 cattle.
"""

from pathlib import Path
import time
import numpy as np
import open3d as o3d
import rembg

from segmentation import CAMERAS, segment_view
from merge_clouds import load_matrix
from refine_cloud import remove_outer_noise


def clear_folder_contents(folder: Path):
    """Deletes all files and subdirectories inside folder without unlinking root."""
    if folder.exists():
        for item in folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                clear_folder_contents(item)
                item.rmdir()


def process_single_cow(cow_id: int, data_dir: Path, seg_root: Path, aligned_root: Path, session):
    """Executes segmentation, SE(3) cloud merging, and DBSCAN/SOR filtering for one cow."""
    cow_seg_dir = seg_root / f"cow_{cow_id}"
    cow_seg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Segment Top, Left, and Right camera views
    for view in CAMERAS:
        segment_view(view, data_dir, cow_seg_dir, session, cow_id=cow_id)

    # 2. Load segmented point clouds
    left_pcd = o3d.io.read_point_cloud(str(cow_seg_dir / "cloud_segmented_left.ply"))
    right_pcd = o3d.io.read_point_cloud(str(cow_seg_dir / "cloud_segmented_right.ply"))
    top_pcd = o3d.io.read_point_cloud(str(cow_seg_dir / "cloud_segmented_top.ply"))

    # 3. Transform Left and Right into Top reference frame
    matrix_file = data_dir / "Matrices" / f"{cow_id}.matrices"
    left_pcd.transform(load_matrix(matrix_file, "Left -> Top"))
    right_pcd.transform(load_matrix(matrix_file, "Right -> Top"))

    # 4. Merge point clouds
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(top_pcd.points), np.asarray(left_pcd.points), np.asarray(right_pcd.points)])
    )
    merged_pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(top_pcd.colors), np.asarray(left_pcd.colors), np.asarray(right_pcd.colors)])
    )

    merged_path = aligned_root / f"cow_{cow_id}_merged.ply"
    clean_path = aligned_root / f"cow_{cow_id}_clean.ply"
    o3d.io.write_point_cloud(str(merged_path), merged_pcd)

    # 5. Filter noise with DBSCAN & SOR
    clean_pcd = remove_outer_noise(merged_path, clean_path, cluster_eps_m=0.08, min_points=50, nb_neighbors=20, std_ratio=2.0)
    return len(merged_pcd.points), len(clean_pcd.points)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    seg_root = base_dir / "segmented_output"
    aligned_root = base_dir / "aligned_output"

    seg_root.mkdir(parents=True, exist_ok=True)
    aligned_root.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("BATCH PROCESSING PIPELINE FOR ALL 103 CATTLE")
    print("=" * 65)

    print("[1/3] Clearing previous outputs...")
    clear_folder_contents(seg_root)
    clear_folder_contents(aligned_root)
    print("      Directories cleared.")

    print("[2/3] Loading U²-Net segmentation session...")
    t0 = time.time()
    session = rembg.new_session("u2net")
    print(f"      Model loaded in {time.time() - t0:.1f}s.")

    print("[3/3] Processing cattle records 1 to 103...")
    print("-" * 65)

    start_time = time.time()
    for cow_id in range(1, 104):
        t_cow = time.time()
        raw_pts, clean_pts = process_single_cow(cow_id, data_dir, seg_root, aligned_root, session)
        dt = time.time() - t_cow
        print(f"  [Cow #{cow_id:03d}/103] Merged: {raw_pts:,} -> Clean: {clean_pts:,} points ({dt:.1f}s)")

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"ALL 103 CATTLE PROCESSED IN {total_time/60:.1f} MINUTES ({total_time:.1f}s)")
    print(f"Clean PLY Models: {aligned_root}")
    print("=" * 65)


if __name__ == "__main__":
    main()
