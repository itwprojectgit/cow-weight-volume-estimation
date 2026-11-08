"""
Live Weight Prediction from 3D Model
====================================
1. Extracts 9 anatomical body measurements directly from the 3D point cloud.
2. Predicts live cattle weight (kg) via regularized Lasso L1 model.
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import ConvexHull


def extract_measurements_from_3d(ply_path: Path) -> dict:
    """
    Extracts the 9 anatomical body measurements (in cm) directly from the 3D point cloud:
      - X axis: Longitudinal axis (tail at min X, head at max X)
      - Y axis: Lateral width
      - Z axis: Overhead depth from camera (min Z = spine, max Z = hooves)
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    length_span = x_max - x_min
    floor_z = np.percentile(pts[:, 2], 99.8)

    # 1. Anterior / Chest & Withers Region (55% to 70% from tail)
    chest_region = pts[(pts[:, 0] >= x_min + 0.55 * length_span) & (pts[:, 0] <= x_min + 0.70 * length_span)]
    spine_z_withers = np.percentile(chest_region[:, 2], 0.5)
    withers_height_cm = (floor_z - spine_z_withers) * 100.0

    torso_chest = chest_region[chest_region[:, 2] <= spine_z_withers + 0.62]
    chest_depth_cm = (torso_chest[:, 2].max() - spine_z_withers) * 100.0
    chest_width_cm = (np.percentile(torso_chest[:, 1], 95) - np.percentile(torso_chest[:, 1], 5)) * 100.0

    hull = ConvexHull(torso_chest[:, [1, 2]])
    heart_girth_cm = hull.area * 100.0 * 0.90

    # 2. Posterior / Pelvic & Hip Region (15% to 30% from tail)
    hip_region = pts[(pts[:, 0] >= x_min + 0.15 * length_span) & (pts[:, 0] <= x_min + 0.30 * length_span)]
    spine_z_hip = np.percentile(hip_region[:, 2], 0.5)
    hip_height_cm = (floor_z - spine_z_hip) * 100.0

    ilium_pts = hip_region[hip_region[:, 2] <= spine_z_hip + 0.15]
    ilium_width_cm = (np.percentile(ilium_pts[:, 1], 95) - np.percentile(ilium_pts[:, 1], 5)) * 100.0
    hip_joint_width_cm = ilium_width_cm * 0.98

    # 3. Length Proportions
    oblique_body_length_cm = length_span * 0.71 * 100.0
    hip_length_cm = oblique_body_length_cm * 0.30

    return {
        "withers height": withers_height_cm,
        "hip height": hip_height_cm,
        "chest depth": chest_depth_cm,
        "chest width": chest_width_cm,
        "ilium width": ilium_width_cm,
        "hip joint width": hip_joint_width_cm,
        "oblique body length": oblique_body_length_cm,
        "hip length": hip_length_cm,
        "heart girth": heart_girth_cm,
    }


def predict_weight(meas: dict, model_path: Path):
    """Predicts live weight in kg using the serialized model."""
    saved = joblib.load(model_path)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    x = np.array([[meas[c] for c in feature_cols]])
    return float(model.predict(x)[0]), feature_cols


def main():
    parser = argparse.ArgumentParser(description="Live Weight Prediction from 3D Model")
    parser.add_argument("--cow", type=int, default=1, help="Cow ID (default: 1)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    ply_path = base_dir / "aligned_output" / f"cow_{args.cow}_clean.ply"
    model_path = base_dir / "weight_model.joblib"
    excel_path = base_dir / "data" / "Measurements.xlsx"

    meas = extract_measurements_from_3d(ply_path)
    pred_wt, _ = predict_weight(meas, model_path)

    df = pd.read_excel(excel_path)
    actual_wt = float(df[df["N"] == args.cow]["live weithg"].iloc[0])
    error = abs(pred_wt - actual_wt)

    print(f"Cow #{args.cow} Scale Weight: {actual_wt:.1f} kg | 3D Predicted: {pred_wt:.1f} kg (Error: {error:.1f} kg / {error/actual_wt*100:.1f}%)")


if __name__ == "__main__":
    main()
