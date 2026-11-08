"""
3-Axis Sliced Convex Hull Volume Integration
============================================
Calculates volume along all 3 coordinate axes using 2D Convex Hull cross-sections:
  - Longitudinal (X): ∫ Ax(x) dx
  - Lateral (Y):      ∫ Ay(y) dy
  - Vertical (Z):     ∫ Az(z) dz
  - Average Volume:   ( ∫ Ax dx + ∫ Ay dy + ∫ Az dz ) / 3
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull


def integrate_axis_slices(points: np.ndarray, axis: int, step_m: float = 0.02) -> float:
    """Slices point cloud along axis at step_m intervals and integrates 2D hull areas."""
    coord = points[:, axis]
    other_axes = [i for i in range(3) if i != axis]
    slices = np.arange(coord.min(), coord.max(), step_m)
    volume_m3 = 0.0

    for s in slices:
        mask = (coord >= s) & (coord < s + step_m)
        sp = points[mask][:, other_axes]
        if len(sp) >= 3:
            try:
                hull_2d = ConvexHull(sp)
                volume_m3 += hull_2d.volume * step_m
            except Exception:
                pass

    return volume_m3


def compute_sliced_hull_volume(ply_path: Path, step_m: float = 0.02) -> dict:
    """Computes sliced convex hull volume along X, Y, Z axes and their average."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)

    vol_x = integrate_axis_slices(points, axis=0, step_m=step_m)
    vol_y = integrate_axis_slices(points, axis=1, step_m=step_m)
    vol_z = integrate_axis_slices(points, axis=2, step_m=step_m)
    vol_avg = (vol_x + vol_y + vol_z) / 3.0

    return {
        "vol_x_m3": vol_x,
        "vol_y_m3": vol_y,
        "vol_z_m3": vol_z,
        "vol_avg_m3": vol_avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Sliced Convex Hull Volume Estimation")
    parser.add_argument("--cow", type=int, default=1, help="Cow ID (default: 1)")
    parser.add_argument("--step", type=float, default=0.02, help="Slice step in meters (default: 0.02)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    ply_path = base_dir / "aligned_output" / f"cow_{args.cow}_clean.ply"
    res = compute_sliced_hull_volume(ply_path, step_m=args.step)
    print(f"Cow #{args.cow} Volume: {res['vol_avg_m3']:.4f} m³ ({res['vol_avg_m3']*1000.0:.1f} Liters)")


if __name__ == "__main__":
    main()
