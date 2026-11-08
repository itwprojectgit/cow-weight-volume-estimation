"""
Multi-Camera 3D Cattle Reconstruction, Volume & Weight Estimation
================================================================
A Streamlit web application with a warm, minimalist luxury editorial aesthetic
(inspired by Söl Haus: cream/linen background, serif typography, elegant cards).
"""

from pathlib import Path
import cv2
import joblib
import numpy as np
import open3d as o3d
import pandas as pd
import streamlit as st

from custom_processor import (
    DEFAULT_FX,
    DEFAULT_FY,
    DEFAULT_CX,
    DEFAULT_CY,
    DEFAULT_LEFT_TO_TOP,
    DEFAULT_RIGHT_TO_TOP,
    colorize_depth,
    parse_matrix_from_text,
    parse_matrices_file,
    format_matrix_to_text,
    segment_view_rgbd,
    merge_and_filter_custom_clouds,
)
from volume_sliced_hull import compute_sliced_hull_volume
from predict_weight import extract_measurements_from_3d, predict_weight

@st.cache_data
def get_pcd_spatial_info(clean_ply_path_str: str, merged_ply_path_str: str):
    clean_p = Path(clean_ply_path_str)
    merged_p = Path(merged_ply_path_str)
    pcd = o3d.io.read_point_cloud(str(clean_p))
    pts = np.asarray(pcd.points)
    raw_pts = len(o3d.io.read_point_cloud(str(merged_p)).points) if merged_p.exists() else len(pts)
    return raw_pts, len(pts), float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), float(np.ptp(pts[:, 2]))

@st.cache_data
def get_cached_volume(clean_ply_path_str: str):
    return compute_sliced_hull_volume(Path(clean_ply_path_str), step_m=0.02)

@st.cache_data
def get_cached_measurements(clean_ply_path_str: str):
    return extract_measurements_from_3d(Path(clean_ply_path_str))

@st.cache_resource
def get_rembg_session():
    import rembg
    return rembg.new_session("u2net")

@st.cache_resource
def get_weight_model(model_path_str: str):
    return joblib.load(model_path_str)

@st.cache_data
def get_ply_bytes(ply_path_str: str) -> bytes:
    with open(ply_path_str, "rb") as f:
        return f.read()

# -----------------------------------------------------------------------------
# Page Configuration & Warm Luxury Styling (Söl Haus Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cattle 3D Reconstruction, Volume & Weight Estimation",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-warm: #FAF8F5;
    --card-bg: #FFFFFF;
    --card-border: rgba(195, 180, 160, 0.28);
    --text-primary: #1F1E1D;
    --text-secondary: #6B665F;
    --accent-cognac: #8C6239;
    --accent-sand: #F3EFEA;
    --radius-lg: 18px;
    --radius-md: 12px;
    --radius-pill: 40px;
    --shadow-soft: 0 10px 30px rgba(40, 30, 20, 0.04);
}

html, body, [data-testid="stAppViewContainer"], .main {
    background-color: var(--bg-warm) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stSidebar"] { display: none; }

h1, h2, h3, .serif-font {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    letter-spacing: -0.01em;
    font-weight: 600;
    color: var(--text-primary);
}

.editorial-hero {
    background: linear-gradient(180deg, #FFFFFF 0%, #FAF8F5 100%);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 30px 38px;
    margin-bottom: 22px;
    box-shadow: var(--shadow-soft);
    position: relative;
}
.editorial-hero::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #8C6239 0%, #D4B996 50%, #8C6239 100%);
}
.hero-subtitle {
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent-cognac);
    margin-bottom: 4px;
}
.hero-title {
    font-size: 2.2rem;
    margin: 4px 0 10px 0;
}
.hero-description {
    font-size: 0.94rem;
    color: var(--text-secondary);
    line-height: 1.55;
    max-width: 900px;
}

.sol-card {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: var(--shadow-soft);
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--accent-cognac);
    margin-bottom: 3px;
}
.card-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 6px;
}

.metric-tile {
    background: #FFFFFF;
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    box-shadow: var(--shadow-soft);
    height: 100%;
}
.metric-desc {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    margin-bottom: 4px;
}
.metric-number {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
}
.metric-unit {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 3px;
}

.sol-badge {
    background: #F5EFE6;
    color: #735129;
    padding: 3px 12px;
    border-radius: var(--radius-pill);
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

.team-grid-card {
    background: #FFFFFF;
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 22px 16px;
    text-align: center;
    box-shadow: var(--shadow-soft);
}
.team-avatar {
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: linear-gradient(135deg, #FAF8F5 0%, #E8E2D8 100%);
    border: 1.5px solid rgba(140, 98, 57, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 12px auto;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--accent-cognac);
}
.team-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.team-id {
    font-size: 0.82rem;
    color: var(--accent-cognac);
    font-weight: 500;
}

.reference-card {
    background: #FFFFFF;
    border: 1px solid var(--card-border);
    border-left: 3.5px solid var(--accent-cognac);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.ref-authors { font-size: 0.88rem; font-weight: 600; color: var(--text-primary); }
.ref-title { font-style: italic; color: #3A3530; margin: 3px 0; font-size: 0.92rem; }
.ref-source { font-size: 0.82rem; color: var(--text-secondary); }

/* Horizontal Tabs Styling */
div[data-testid="stTabs"] button[role="tab"] {
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-pill) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 20px !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
}
div[data-testid="stTabs"] div[role="tablist"] {
    gap: 8px;
    border-bottom: none !important;
    margin-bottom: 18px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Base Directories & State Initialization
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SEG_DIR = BASE_DIR / "segmented_output"
ALIGNED_DIR = BASE_DIR / "aligned_output"
MODEL_PATH = BASE_DIR / "weight_model.joblib"
EXCEL_PATH = DATA_DIR / "Measurements.xlsx"
CUSTOM_DIR = BASE_DIR / "custom_output"
CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

if "selected_cow" not in st.session_state:
    st.session_state.selected_cow = 1
if "custom_left_mat_text" not in st.session_state:
    st.session_state.custom_left_mat_text = format_matrix_to_text(DEFAULT_LEFT_TO_TOP)
if "custom_right_mat_text" not in st.session_state:
    st.session_state.custom_right_mat_text = format_matrix_to_text(DEFAULT_RIGHT_TO_TOP)
if "custom_sample_id" not in st.session_state:
    st.session_state.custom_sample_id = None
if "custom_processed" not in st.session_state:
    st.session_state.custom_processed = False

df_measurements = pd.read_excel(EXCEL_PATH) if EXCEL_PATH.exists() else pd.DataFrame()


# -----------------------------------------------------------------------------
# Reusable UI Rendering Helpers (Eliminates Duplicate Markup & Waste Lines)
# -----------------------------------------------------------------------------
def metric_card(title: str, value: str, unit: str, border_left: str = None) -> str:
    border_css = f' style="border-left: 4px solid {border_left};"' if border_left else ""
    return f"""
    <div class="metric-tile"{border_css}>
        <div class="metric-desc">{title}</div>
        <div class="metric-number">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>
    """


def render_spatial_summary(clean_ply: Path, raw_pts: int, clean_pts: int, sx: float, sy: float, sz: float, dl_filename: str):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(metric_card("Clean Surface Points", f"{clean_pts:,}", "Animal Body Points"), unsafe_allow_html=True)
    with p2:
        st.markdown(metric_card("Noise Eliminated", f"{max(0, raw_pts - clean_pts):,}", "Outliers Filtered"), unsafe_allow_html=True)
    with p3:
        st.markdown(metric_card("Bounding Length (X)", f"{sx:.2f}", "Meters (Head to Tail)"), unsafe_allow_html=True)
    with p4:
        st.markdown(metric_card("Width & Height (Y×Z)", f"{sy:.2f}×{sz:.2f}", "Meters (Width × Height)"), unsafe_allow_html=True)

    st.write("")
    if clean_ply.exists():
        ply_bytes = get_ply_bytes(str(clean_ply))
        col_dl, _ = st.columns([1.6, 2.4])
        with col_dl:
            st.download_button(
                label="⬇ Download 3D Point Cloud (.PLY)",
                data=ply_bytes,
                file_name=dl_filename,
                mime="application/octet-stream",
            )


def render_volume_metrics(vol: dict):
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.markdown(metric_card("Average 3-Axis Volume", f"{vol['vol_avg_m3']:.3f}", f"m<sup>3</sup> ({vol['vol_avg_m3']*1000.0:.1f} Liters)"), unsafe_allow_html=True)
    with v2:
        st.markdown(metric_card("Longitudinal (X) Integral", f"{vol['vol_x_m3']:.3f}", f"m<sup>3</sup> ({vol['vol_x_m3']*1000.0:.1f} L)"), unsafe_allow_html=True)
    with v3:
        st.markdown(metric_card("Lateral (Y) Integral", f"{vol['vol_y_m3']:.3f}", f"m<sup>3</sup> ({vol['vol_y_m3']*1000.0:.1f} L)"), unsafe_allow_html=True)
    with v4:
        st.markdown(metric_card("Vertical (Z) Integral", f"{vol['vol_z_m3']:.3f}", f"m<sup>3</sup> ({vol['vol_z_m3']*1000.0:.1f} L)"), unsafe_allow_html=True)


def render_biometrics_table(meas_3d: dict, feature_cols: list, gt_row=None):
    rows = []
    for col in feature_cols:
        val_3d = meas_3d[col]
        if gt_row is not None and col in gt_row:
            val_gt = float(gt_row[col])
            diff = val_3d - val_gt
            rows.append({
                "Anatomical Measurement": col.title(),
                "3D Measurement (cm)": f"{val_3d:.1f}",
                "Manual Measurement (cm)": f"{val_gt:.1f}",
                "Difference (cm)": f"{diff:+.1f}",
                "Status": "Aligned" if abs(diff) <= 6.0 else "Variance",
            })
        else:
            rows.append({
                "Anatomical Measurement": col.title(),
                "3D Measurement (cm)": f"{val_3d:.1f}",
                "Status": "Extracted from 3D Model",
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_weight_cards(pred_wt: float, actual_wt: float = None, pred_wt_manual: float = None):
    if actual_wt is not None:
        err_3d = abs(pred_wt - actual_wt)
        err_pct = (err_3d / actual_wt) * 100.0
        cols = st.columns(3 if pred_wt_manual is not None else 2)
        with cols[0]:
            st.markdown(metric_card("Scale Weight", f"{actual_wt:.1f}", "kg (Platform Ground Truth)", "#8C6239"), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(metric_card("Predicted Weight (3D Model)", f"{pred_wt:.1f}", f"kg (Error: {err_3d:.1f} kg / {err_pct:.1f}%)", "#4E684B"), unsafe_allow_html=True)
        if pred_wt_manual is not None and len(cols) > 2:
            err_m = abs(pred_wt_manual - actual_wt)
            with cols[2]:
                st.markdown(metric_card("Predicted Weight (Manual Tape)", f"{pred_wt_manual:.1f}", f"kg (Error: {err_m:.1f} kg / {err_m/actual_wt*100:.1f}%)", "#A07248"), unsafe_allow_html=True)
    else:
        w1, w2 = st.columns(2)
        with w1:
            st.markdown(metric_card("Predicted Live Weight (3D Model)", f"{pred_wt:.1f}", "kg", "#4E684B"), unsafe_allow_html=True)
        with w2:
            st.markdown(metric_card("Regression Methodology", "Lasso L1", "Regularized Anatomical Estimator", "#8C6239"), unsafe_allow_html=True)


def render_view_tabs(reg_paths: dict, mask_paths: dict, seg_paths: dict, hd_paths: dict = None):
    views = ["Top", "Left", "Right"]
    tabs = st.tabs(["Top View", "Left View", "Right View"])
    for i, v in enumerate(views):
        with tabs[i]:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Registered {v} Frame**")
                p = reg_paths.get(v)
                if p and Path(p).exists():
                    st.image(str(p))
                else:
                    st.info(f"No registered {v} frame.")
            with c2:
                st.markdown("**Segmentation Mask**")
                p = mask_paths.get(v)
                if p and Path(p).exists():
                    st.image(str(p))
                else:
                    st.info(f"No {v} mask.")
            with c3:
                st.markdown(f"**Segmented {v} Image**")
                p = seg_paths.get(v)
                if p and Path(p).exists():
                    st.image(str(p))
                else:
                    st.info(f"No segmented {v} image.")
            if hd_paths and hd_paths.get(v) and Path(hd_paths[v]).exists():
                with st.expander(f"View Full HD {v} Sensor Image"):
                    st.image(str(hd_paths[v]), caption=f"Raw 1080p frame from {v} sensor")


# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="editorial-hero">
        <div class="hero-subtitle">Multi-Camera Computer Vision & Livestock Phenotyping Framework</div>
        <h1 class="hero-title">Multi-Camera 3D Cattle Reconstruction, Volume & Weight Estimation</h1>
        <p class="hero-description">
            A non-invasive phenotyping system for livestock that synchronizes multi-view RGB-D sensors 
            (Top, Left, Right), executes salient U²-Net animal segmentation with chute-bar bridging, 
            merges spatial coordinates via metric rigid SE(3) transformations, integrates 3-axis Sliced Convex Hulls, 
            and predicts live weight using regularized anatomical regression models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Horizontal Navigation Tabs
# -----------------------------------------------------------------------------
tab_main, tab_team, tab_refs = st.tabs(
    [
        "Cattle Analysis & Estimation",
        "Team Members",
        "Academic References",
    ]
)

# =============================================================================
# TAB 1: CATTLE ANALYSIS & ESTIMATION
# =============================================================================
with tab_main:
    st.markdown('<div class="card-label">Input Source</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Choose Analysis Input Mode:",
        options=["Dataset Cattle Records (1 to 103)", "Custom & Random Sensor Input (RGB-D & Calibration)"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_input_mode",
    )
    st.write("")

    # -------------------------------------------------------------------------
    # MODE A: DATASET CATTLE (1 TO 103)
    # -------------------------------------------------------------------------
    if input_mode == "Dataset Cattle Records (1 to 103)":
        st.markdown('<div class="card-label">Animal Selection</div>', unsafe_allow_html=True)
        c_sel1, c_sel2 = st.columns([1.5, 2.5])

        all_cows = list(range(1, 104))
        cur_idx = all_cows.index(st.session_state.selected_cow) if st.session_state.selected_cow in all_cows else 0

        with c_sel1:
            chosen = st.selectbox(
                "Select Cattle Sample:",
                options=all_cows,
                index=cur_idx,
                format_func=lambda x: f"Cow #{x:03d}",
                key="cow_select_unified",
            )
            if chosen != st.session_state.selected_cow:
                st.session_state.selected_cow = chosen
                st.rerun()

        cow_id = st.session_state.selected_cow

        with c_sel2:
            if not df_measurements.empty and cow_id in df_measurements["N"].values:
                gt_info = df_measurements[df_measurements["N"] == cow_id].iloc[0]
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid var(--card-border); border-radius: 12px; padding: 14px 20px;">
                        <span class="sol-badge">Animal Record &bull; Cow #{cow_id:03d}</span>
                        <div style="font-size: 0.92rem; color: #2B2927; margin-top: 6px;">
                            Scale Weight: <strong>{float(gt_info['live weithg']):.1f} kg</strong> &bull; 
                            Withers Height: <strong>{float(gt_info['withers height']):.1f} cm</strong> &bull; 
                            Heart Girth: <strong>{float(gt_info['heart girth']):.1f} cm</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")

        # 1. Multi-View Segmentation
        seg_sub = SEG_DIR / f"cow_{cow_id}"
        st.markdown(
            f"""
            <div class="sol-card">
                <div class="card-label">Multi-Camera Segmentation</div>
                <div class="card-title">Synchronized Sensor Acquisition & U²-Net Matting &bull; Cow #{cow_id:03d}</div>
                <p style="color: var(--text-secondary); font-size: 0.94rem;">
                    Salient animal body matting across Top (Reference), Left, and Right views with chute-bar morphological bridging.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        reg_dict = {v: seg_sub / f"rgb_registered_{v.lower()}.png" for v in ["Top", "Left", "Right"]}
        mask_dict = {v: seg_sub / f"mask_{v.lower()}.png" for v in ["Top", "Left", "Right"]}
        seg_dict = {v: seg_sub / f"rgb_segmented_{v.lower()}.png" for v in ["Top", "Left", "Right"]}
        hd_dict = {v: DATA_DIR / f"{v}Rgb" / f"{cow_id}.png" for v in ["Top", "Left", "Right"]}

        render_view_tabs(reg_dict, mask_dict, seg_dict, hd_dict)

        st.write("")

        # 2. 3D Model Spatial Reconstruction & PLY Download
        clean_ply_path = ALIGNED_DIR / f"cow_{cow_id}_clean.ply"
        merged_ply_path = ALIGNED_DIR / f"cow_{cow_id}_merged.ply"

        st.markdown(
            f"""
            <div class="sol-card">
                <div class="card-label">Spatial Fusion & Model Export</div>
                <div class="card-title">Reconstructed 3D Point Cloud & Downloadable Model &bull; Cow #{cow_id:03d}</div>
                <p style="color: var(--text-secondary); font-size: 0.94rem;">
                    Left and Right point clouds mapped into Top coordinates via Euclidean SE(3) transformation matrices, 
                    refined with DBSCAN density clustering and Statistical Outlier Removal.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if clean_ply_path.exists():
            raw_pts, clean_pts, sx, sy, sz = get_pcd_spatial_info(str(clean_ply_path), str(merged_ply_path))

            render_spatial_summary(
                clean_ply_path,
                raw_pts,
                clean_pts,
                sx,
                sy,
                sz,
                f"cow_{cow_id:03d}_clean.ply",
            )

        st.write("")

        # 3. Volume & Live Weight Estimation
        st.markdown(
            f"""
            <div class="sol-card">
                <div class="card-label">Volume & Biometric Estimation</div>
                <div class="card-title">Volume Computation & Live Weight Estimation &bull; Cow #{cow_id:03d}</div>
                <p style="color: var(--text-secondary); font-size: 0.94rem;">
                    3-axis Sliced Convex Hull volume integration and live weight predicted from 9 anatomical biometric landmarks.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if clean_ply_path.exists():
            vol_data = get_cached_volume(str(clean_ply_path))
            meas_3d = get_cached_measurements(str(clean_ply_path))

            pred_wt_3d, feature_cols = predict_weight(meas_3d, MODEL_PATH)

            gt_row = df_measurements[df_measurements["N"] == cow_id].iloc[0] if not df_measurements.empty else None
            actual_wt = float(gt_row["live weithg"]) if gt_row is not None else None

            pred_wt_gt = None
            if gt_row is not None:
                saved = joblib.load(MODEL_PATH)
                x_gt = np.array([[float(gt_row[c]) for c in feature_cols]])
                pred_wt_gt = float(saved["model"].predict(x_gt)[0])

            st.markdown('<div class="card-title">Sliced Convex Hull Volume Integration</div>', unsafe_allow_html=True)
            render_volume_metrics(vol_data)

            st.write("")
            st.markdown('<div class="card-title">Anatomical Biometric Measurements</div>', unsafe_allow_html=True)
            render_biometrics_table(meas_3d, feature_cols, gt_row)

            st.write("")
            st.markdown('<div class="card-title">Live Weight Prediction</div>', unsafe_allow_html=True)
            render_weight_cards(pred_wt_3d, actual_wt, pred_wt_gt)

    # -------------------------------------------------------------------------
    # MODE B: CUSTOM & RANDOM SENSOR INPUT
    # -------------------------------------------------------------------------
    else:
        st.markdown(
            """
            <div class="sol-card">
                <div class="card-label">Multi-Sensor Acquisition & Custom Ingestion</div>
                <div class="card-title">Arbitrary RGB-D & Rigid Calibration Ingestion Pipeline</div>
                <p style="color: var(--text-secondary); font-size: 0.94rem;">
                    Ingest multi-view RGB and depth images (Top, Left, Right) along with SE(3) transformation matrices, 
                    or generate random cattle sensor inputs from the dataset with one click.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Action Bar
        col_rand1, col_rand2, col_rand3 = st.columns([2, 1.4, 2.6])
        with col_rand1:
            if st.button("🎲 Load Random Cattle Sample from Dataset"):
                rand_id = int(np.random.randint(1, 104))
                st.session_state.custom_sample_id = rand_id
                mat_file = DATA_DIR / "Matrices" / f"{rand_id}.matrices"
                if mat_file.exists():
                    try:
                        with open(mat_file, "r", encoding="utf-8") as f:
                            parsed_l, parsed_r = parse_matrices_file(f.read())
                            st.session_state.custom_left_mat_text = format_matrix_to_text(parsed_l)
                            st.session_state.custom_right_mat_text = format_matrix_to_text(parsed_r)
                    except Exception:
                        pass
                st.session_state.custom_processed = False
                st.rerun()

        with col_rand2:
            if st.button("🔄 Reset Custom Inputs"):
                st.session_state.custom_sample_id = None
                st.session_state.custom_left_mat_text = format_matrix_to_text(DEFAULT_LEFT_TO_TOP)
                st.session_state.custom_right_mat_text = format_matrix_to_text(DEFAULT_RIGHT_TO_TOP)
                st.session_state.custom_processed = False
                st.rerun()

        with col_rand3:
            if st.session_state.get("custom_sample_id"):
                sid = st.session_state.custom_sample_id
                st.markdown(
                    f"""
                    <div style="background: #F5EFE6; border: 1px solid rgba(140, 98, 57, 0.25); border-radius: 8px; padding: 8px 14px; font-size: 0.86rem; color: #8C6239;">
                        Active Sample: <strong>Cow #{sid:03d}</strong> &bull; RGB-D frames & matrices preloaded.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")

        # 1. Multi-Camera Inputs
        st.markdown('<div class="card-label">Multi-Camera Sensor Inputs (RGB & Depth Maps)</div>', unsafe_allow_html=True)
        cam_cols = st.columns(3)
        views = [("Top", "Reference Overhead View"), ("Left", "Left Lateral View"), ("Right", "Right Lateral View")]

        active_rgb = {}
        active_depth = {}

        for idx, (view, desc) in enumerate(views):
            with cam_cols[idx]:
                st.markdown(f"**{view} Camera** <span style='font-size:0.8rem; color:#888;'>({desc})</span>", unsafe_allow_html=True)
                up_rgb = st.file_uploader(f"{view} RGB Image", type=["png", "jpg", "jpeg"], key=f"c_rgb_{view}")
                up_depth = st.file_uploader(f"{view} Depth Map", type=["png", "tiff", "npy"], key=f"c_depth_{view}")

                img_rgb = None
                img_depth = None

                if up_rgb is not None:
                    img_rgb = cv2.imdecode(np.asarray(bytearray(up_rgb.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                elif st.session_state.get("custom_sample_id"):
                    sid = st.session_state.custom_sample_id
                    p = DATA_DIR / f"{view}Rgb" / f"{sid}.png"
                    if p.exists():
                        img_rgb = cv2.imread(str(p))

                if up_depth is not None:
                    img_depth = cv2.imdecode(np.asarray(bytearray(up_depth.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                elif st.session_state.get("custom_sample_id"):
                    sid = st.session_state.custom_sample_id
                    p = DATA_DIR / f"{view}Depth" / f"{sid}.png"
                    if p.exists():
                        img_depth = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

                active_rgb[view] = img_rgb
                active_depth[view] = img_depth

                p1, p2 = st.columns(2)
                with p1:
                    if img_rgb is not None:
                        st.image(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), caption=f"{view} RGB")
                    else:
                        st.caption(f"No {view} RGB.")
                with p2:
                    if img_depth is not None:
                        st.image(colorize_depth(img_depth), caption=f"{view} Depth (Viridis)")
                    else:
                        st.caption(f"No {view} Depth.")

        st.write("")

        # 2. Calibration Inputs
        st.markdown('<div class="card-label">Multi-Camera Calibration Inputs</div>', unsafe_allow_html=True)
        up_mat = st.file_uploader("Upload .matrices Calibration File (Optional)", type=["matrices", "txt", "dat"], key="c_mat_file")
        if up_mat is not None:
            try:
                l_m, r_m = parse_matrices_file(up_mat.read().decode("utf-8"))
                st.session_state.custom_left_mat_text = format_matrix_to_text(l_m)
                st.session_state.custom_right_mat_text = format_matrix_to_text(r_m)
                st.success("Calibration matrices successfully loaded!")
            except Exception as e:
                st.error(f"Error parsing .matrices: {e}")

        m1, m2 = st.columns(2)
        with m1:
            left_text = st.text_area("Left -> Top Transformation Matrix (4×4 SE(3)):", value=st.session_state.get("custom_left_mat_text", format_matrix_to_text(DEFAULT_LEFT_TO_TOP)), height=100)
        with m2:
            right_text = st.text_area("Right -> Top Transformation Matrix (4×4 SE(3)):", value=st.session_state.get("custom_right_mat_text", format_matrix_to_text(DEFAULT_RIGHT_TO_TOP)), height=100)

        with st.expander("Camera Intrinsics Calibration (Optional)"):
            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1: p_fx = st.number_input("fx", value=float(DEFAULT_FX), step=0.1)
            with ic2: p_fy = st.number_input("fy", value=float(DEFAULT_FY), step=0.1)
            with ic3: p_cx = st.number_input("cx", value=float(DEFAULT_CX), step=0.1)
            with ic4: p_cy = st.number_input("cy", value=float(DEFAULT_CY), step=0.1)

        st.write("")

        # 3. Process Button
        if st.button("Process Custom Cattle Input", type="primary"):
            missing = [f"{v} {kind}" for v in ["Top", "Left", "Right"] for kind, d in [("RGB", active_rgb), ("Depth", active_depth)] if d[v] is None]
            if missing:
                st.error(f"Cannot process: missing {', '.join(missing)}. Click 'Load Random Cattle Sample' above to test immediately.")
            else:
                try:
                    left_mat = parse_matrix_from_text(left_text)
                    right_mat = parse_matrix_from_text(right_text)
                except Exception as e:
                    st.error(f"Matrix syntax error: {e}")
                    st.stop()

                with st.spinner("Processing custom segmentation, 3D projection, and biometric estimation..."):
                    session = get_rembg_session()
                    saved_model = get_weight_model(str(MODEL_PATH))
                    clouds = {}
                    for v in ["Top", "Left", "Right"]:
                        pcd_v, rgb_reg, mask_v, seg_rgb = segment_view_rgbd(active_rgb[v], active_depth[v], session, fx=p_fx, fy=p_fy, cx=p_cx, cy=p_cy)
                        clouds[v] = pcd_v
                        vl = v.lower()
                        cv2.imwrite(str(CUSTOM_DIR / f"rgb_registered_{vl}.png"), cv2.cvtColor(rgb_reg, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(CUSTOM_DIR / f"mask_{vl}.png"), mask_v)
                        cv2.imwrite(str(CUSTOM_DIR / f"rgb_segmented_{vl}.png"), cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR))

                    custom_clean = CUSTOM_DIR / "custom_cow_clean.ply"
                    clean_pcd, raw_c, clean_c = merge_and_filter_custom_clouds(clouds["Top"], clouds["Left"], clouds["Right"], left_mat, right_mat, custom_clean)

                    vol_data = compute_sliced_hull_volume(custom_clean, step_m=0.02)
                    meas_3d = extract_measurements_from_3d(custom_clean)
                    feature_cols = saved_model["feature_cols"]
                    x_feat = np.array([[meas_3d[c] for c in feature_cols]])
                    pred_wt = float(saved_model["model"].predict(x_feat)[0])

                    pts = np.asarray(clean_pcd.points)
                    st.session_state.custom_results = {
                        "raw_count": raw_c,
                        "clean_count": clean_c,
                        "span_x": float(np.ptp(pts[:, 0])),
                        "span_y": float(np.ptp(pts[:, 1])),
                        "span_z": float(np.ptp(pts[:, 2])),
                        "vol_data": vol_data,
                        "meas_3d": meas_3d,
                        "pred_wt": pred_wt,
                        "feature_cols": feature_cols,
                    }
                    st.session_state.custom_processed = True
                    st.success("Custom cattle input successfully processed!")
                    st.rerun()

        # 4. Display Custom Results
        custom_clean_ply = CUSTOM_DIR / "custom_cow_clean.ply"
        if st.session_state.get("custom_processed") and custom_clean_ply.exists():
            res = st.session_state.get("custom_results")
            if res is None:
                pcd = o3d.io.read_point_cloud(str(custom_clean_ply))
                pts = np.asarray(pcd.points)
                vol_data = compute_sliced_hull_volume(custom_clean_ply, step_m=0.02)
                meas_3d = extract_measurements_from_3d(custom_clean_ply)
                pred_wt, feature_cols = predict_weight(meas_3d, MODEL_PATH)
                res = {
                    "raw_count": len(pts), "clean_count": len(pts),
                    "span_x": float(np.ptp(pts[:, 0])), "span_y": float(np.ptp(pts[:, 1])), "span_z": float(np.ptp(pts[:, 2])),
                    "vol_data": vol_data, "meas_3d": meas_3d, "pred_wt": pred_wt, "feature_cols": feature_cols,
                }
                st.session_state.custom_results = res

            st.write("")
            st.markdown(
                """
                <div class="sol-card">
                    <div class="card-label">Segmentation Output</div>
                    <div class="card-title">Multi-View Salient U²-Net Matting Results</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c_reg = {v: CUSTOM_DIR / f"rgb_registered_{v.lower()}.png" for v in ["Top", "Left", "Right"]}
            c_mask = {v: CUSTOM_DIR / f"mask_{v.lower()}.png" for v in ["Top", "Left", "Right"]}
            c_seg = {v: CUSTOM_DIR / f"rgb_segmented_{v.lower()}.png" for v in ["Top", "Left", "Right"]}
            render_view_tabs(c_reg, c_mask, c_seg)

            st.write("")
            st.markdown(
                """
                <div class="sol-card">
                    <div class="card-label">Spatial Fusion & Model Export</div>
                    <div class="card-title">Reconstructed 3D Point Cloud & Downloadable Model</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_spatial_summary(custom_clean_ply, res["raw_count"], res["clean_count"], res["span_x"], res["span_y"], res["span_z"], "custom_cattle_clean.ply")

            st.write("")
            st.markdown(
                """
                <div class="sol-card">
                    <div class="card-label">Volume & Biometric Estimation</div>
                    <div class="card-title">Volume Computation & Live Weight Estimation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="card-title">Sliced Convex Hull Volume Integration</div>', unsafe_allow_html=True)
            render_volume_metrics(res["vol_data"])

            st.write("")
            st.markdown('<div class="card-title">Anatomical Biometric Measurements</div>', unsafe_allow_html=True)
            gt_row = None
            if st.session_state.get("custom_sample_id"):
                sid = st.session_state.custom_sample_id
                if not df_measurements.empty and sid in df_measurements["N"].values:
                    gt_row = df_measurements[df_measurements["N"] == sid].iloc[0]
            render_biometrics_table(res["meas_3d"], res["feature_cols"], gt_row)

            st.write("")
            st.markdown('<div class="card-title">Live Weight Prediction</div>', unsafe_allow_html=True)
            actual_wt = float(gt_row["live weithg"]) if gt_row is not None else None
            render_weight_cards(res["pred_wt"], actual_wt)


# =============================================================================
# TAB 2: TEAM MEMBERS
# =============================================================================
with tab_team:
    st.markdown(
        """
        <div class="sol-card">
            <div class="card-title">Project Team</div>
            <p style="color: var(--text-secondary); font-size: 0.95rem;">
                Automated 3D Multi-Camera Livestock Phenotyping, Volume Computation, and Allometric Live Weight Estimation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    members = [
        {"name": "M. Harshavardhan Reddy", "id": "25075073", "initials": "HR"},
        {"name": "PVK Sathwic", "id": "25075093", "initials": "PS"},
        {"name": "B Anjana Akhileshwar", "id": "25075035", "initials": "AA"},
        {"name": "YV Suresh Kumar Reddy", "id": "25075152", "initials": "SR"},
        {"name": "J Shashank Rahul", "id": "25075065", "initials": "SR"},
    ]

    t_cols = st.columns(5)
    for i, mem in enumerate(members):
        with t_cols[i]:
            st.markdown(
                f"""
                <div class="team-grid-card">
                    <div class="team-avatar">{mem['initials']}</div>
                    <div class="team-name">{mem['name']}</div>
                    <div class="team-id">ID: {mem['id']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# TAB 3: ACADEMIC REFERENCES
# =============================================================================
with tab_refs:
    st.markdown(
        """
        <div class="sol-card">
            <div class="card-label">Scholarly Foundation</div>
            <div class="card-title">Key Academic Literature & Peer-Reviewed References</div>
            <p style="color: var(--text-secondary); font-size: 0.95rem;">
                Curated research papers, mathematical treatises, and computer vision literature informing this framework.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    refs = [
        {
            "authors": "Ruchay, A., Kober, V., Dorofeev, K., Kolpakov, V., & Dzhulamanov, K. (2020)",
            "title": "Cattle body volume estimation using depth cameras.",
            "source": "Computers and Electronics in Agriculture, 175, 105578. DOI: 10.1016/j.compag.2020.105578",
            "desc": "Published the CowDatabase dataset of 103 Hereford cows, establishing multi-camera Kinect RGB-D chute setup and planar checkerboard calibration.",
        },
        {
            "authors": "Rozendo, C., et al. (2025)",
            "title": "Predicting live weight of beef cattle from body measurements and 3D computer vision.",
            "source": "Computers and Electronics in Agriculture, 228, 109520.",
            "desc": "Investigates regularized linear and ensemble models predicting beef cattle weight from non-invasive 3D point cloud anatomical biometric landmarks.",
        },
        {
            "authors": "Cavalieri, Bonaventura (1635)",
            "title": "Geometria Indivisibilibus Continuorum Nova Quadam Ratione Promota.",
            "source": "Bologna: Ferronius.",
            "desc": "Foundational mathematical theorem stating that solids having equal cross-sectional areas at all parallel slicing planes have identical volumes (applied in our longitudinal slice integration).",
        },
        {
            "authors": "Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020)",
            "title": "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection.",
            "source": "Pattern Recognition, 106, 107404.",
            "desc": "Two-level nested U-structure deep learning architecture providing high-resolution alpha boundary segmentation for livestock animals in noisy farm environments.",
        },
        {
            "authors": "Salau, J., Haas, J. H., Junge, W., & Krieter, J. (2014)",
            "title": "Extrinsic calibration of a multi-Kinect camera setup and automated body condition scoring.",
            "source": "Journal of Dairy Science, 97(5), 2736-2746.",
            "desc": "Details multi-camera rigid alignment, depth map distortion correction, and automated dorsal livestock landmark detection.",
        },
        {
            "authors": "Tasdemir, S., Urkmez, A., & Inal, S. (2011)",
            "title": "Determination of body measurements on Holstein cows using digital image analysis and estimation of body weight with regression analysis.",
            "source": "Computers and Electronics in Agriculture, 76(2), 189-197.",
            "desc": "Examines morphometric relationships between heart girth, withers height, oblique body length, and actual cattle scale weight.",
        },
        {
            "authors": "Zhou, Q.-Y., Park, J., & Koltun, V. (2018)",
            "title": "Open3D: A Modern Library for 3D Data Processing.",
            "source": "arXiv:1801.09847.",
            "desc": "Open-source 3D library utilized for Euclidean transformations, DBSCAN density-based noise clustering, and Statistical Outlier Removal (SOR).",
        },
    ]

    for ref in refs:
        st.markdown(
            f"""
            <div class="reference-card">
                <div class="ref-authors">{ref['authors']}</div>
                <div class="ref-title">"{ref['title']}"</div>
                <div class="ref-source">{ref['source']}</div>
                <div style="font-size: 0.85rem; color: #555; margin-top: 6px;">{ref['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-top: 48px; padding: 24px; border-top: 1px solid rgba(195, 180, 160, 0.25); color: #8C6239; font-size: 0.82rem; letter-spacing: 0.8px;">
        MULTI-CAMERA CATTLE VOLUME & LIVE WEIGHT ESTIMATION SYSTEM &bull; RESEARCH & ENGINEERING FRAMEWORK
    </div>
    """,
    unsafe_allow_html=True,
)
