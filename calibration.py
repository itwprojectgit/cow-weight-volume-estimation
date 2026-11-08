"""
Multi-Camera Extrinsic Calibration (Checkerboard Method)
========================================================
Reference Setup:
  - 3 RGB-D Cameras: Left Camera, Right Camera, Top Camera (Reference).
  - Method: Planar checkerboard calibration (Zhang's method / OpenCV solvePnP).
  - Formulation:
      T_top_from_left  = T_top_from_board  @ inv(T_left_from_board)
      T_top_from_right = T_top_from_board  @ inv(T_right_from_board)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def create_board_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """Generate 3D coordinates of inner corners of the checkerboard (Z = 0)."""
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m
    return objp


def detect_checkerboard_corners(image: np.ndarray, pattern_size: tuple):
    """Detects checkerboard corners and refines them to sub-pixel accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def compute_camera_pose_from_board(
    board_points: np.ndarray,
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> np.ndarray:
    """Calculates 4x4 rigid transformation from board frame to camera frame using solvePnP."""
    success, rvec, tvec = cv2.solvePnP(
        board_points, corners, camera_matrix, distortion_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP failed to converge.")

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def compute_relative_transformation(T_ref_from_board: np.ndarray, T_src_from_board: np.ndarray) -> np.ndarray:
    """Computes relative extrinsic transformation: T_ref_from_src = T_ref_from_board @ inv(T_src_from_board)."""
    return T_ref_from_board @ np.linalg.inv(T_src_from_board)


def load_calibration_matrices(matrix_path: Path):
    """Loads Left->Top and Right->Top 4x4 transformation matrices from dataset."""
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    with open(matrix_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    left_idx = next(i + 1 for i, line in enumerate(lines) if "Left -> Top" in line)
    right_idx = next(i + 1 for i, line in enumerate(lines) if "Right -> Top" in line)

    left_to_top = np.array([[float(v) for v in lines[left_idx + r].split()] for r in range(4)], dtype=np.float64)
    right_to_top = np.array([[float(v) for v in lines[right_idx + r].split()] for r in range(4)], dtype=np.float64)
    return left_to_top, right_to_top


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Extrinsic Calibration")
    parser.add_argument("--cow", type=int, default=1, help="Cow ID (default: 1)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    matrix_file = base_dir / "data" / "Matrices" / f"{args.cow}.matrices"
    left_to_top, right_to_top = load_calibration_matrices(matrix_file)

    dist_l = np.linalg.norm(left_to_top[:3, 3])
    dist_r = np.linalg.norm(right_to_top[:3, 3])

    print(f"Cow #{args.cow} Calibration Matrices Loaded from {matrix_file.name}")
    print(f"  Left Camera to Top Camera Distance:  {dist_l:.3f} m")
    print(f"  Right Camera to Top Camera Distance: {dist_r:.3f} m")


if __name__ == "__main__":
    main()
