"""
Multi-View U²-Net Salient Animal Segmentation
=============================================
Segments animal body from enclosure chutes across camera views (Top, Left, Right).
Includes morphological closing and connected-component filtering to preserve
complete anatomy (head, torso, and limbs) across chute rail occlusions.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import rembg


CAMERAS = ("Left", "Right", "Top")
ALPHA_THRESHOLD = 180


def segment_view(view: str, data_dir: Path, output_dir: Path, session, cow_id: int = 1):
    """Segments RGB, Depth, and PLY for one camera view."""
    ply_path = data_dir / f"{view}Clouds" / f"{cow_id}.ply"
    depth_path = data_dir / f"{view}Depth" / f"{cow_id}.png"

    # 1. Load point cloud and registered 424x512 RGB
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    rgb = (colors.reshape(424, 512, 3) * 255).astype(np.uint8)

    # 2. U²-Net Matting & Morphological Closing
    out = rembg.remove(rgb, session=session)
    alpha = out[:, :, 3]
    mask = (alpha >= ALPHA_THRESHOLD).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Preserve all major body components (>= 3% of body or >= 300 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask)
    if num_labels > 1:
        total_fg = stats[1:, cv2.CC_STAT_AREA].sum()
        min_keep_area = max(300, int(total_fg * 0.03))
        clean_mask = np.zeros_like(mask)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_keep_area:
                clean_mask[labels == lbl] = 1
        mask = clean_mask
    else:
        mask = closed_mask

    # 4. Save masked RGB, Depth, and Point Cloud
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    masked_rgb = cv2.bitwise_and(rgb_bgr, rgb_bgr, mask=mask)

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    masked_depth = depth.copy() if depth is not None else np.zeros((424, 512), dtype=np.uint16)
    masked_depth[mask == 0] = 0

    valid_pts = (mask.flatten() > 0) & ((points != 0).any(axis=1))
    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(points[valid_pts])
    clean_pcd.colors = o3d.utility.Vector3dVector(colors[valid_pts])

    v = view.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"rgb_registered_{v}.png"), rgb_bgr)
    cv2.imwrite(str(output_dir / f"mask_{v}.png"), mask * 255)
    cv2.imwrite(str(output_dir / f"rgb_segmented_{v}.png"), masked_rgb)
    cv2.imwrite(str(output_dir / f"depth_segmented_{v}.png"), masked_depth)
    o3d.io.write_point_cloud(str(output_dir / f"cloud_segmented_{v}.ply"), clean_pcd)

    return clean_pcd


def main():
    parser = argparse.ArgumentParser(description="U²-Net View Segmentation")
    parser.add_argument("--cow", type=int, default=1, help="Cow ID (default: 1)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    session = rembg.new_session("u2net")
    output_dir = base_dir / "segmented_output" / f"cow_{args.cow}"

    for view in CAMERAS:
        pcd = segment_view(view, base_dir / "data", output_dir, session, cow_id=args.cow)
        print(f"[{view}] Segmented: {len(pcd.points):,} points saved to {output_dir}")


if __name__ == "__main__":
    main()
