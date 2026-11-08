# Multi-Camera 3D Cattle Reconstruction, Volume & Live Weight Estimation

A non-invasive computer vision framework for livestock phenotyping. The system synchronizes multi-view RGB-D sensors (Top, Left, Right), performs salient $U^2$-Net animal segmentation with morphological chute-bar bridging, aligns multi-view point clouds via metric $SE(3)$ rigid transformations, applies DBSCAN density clustering and Statistical Outlier Removal (SOR), computes 3-axis Sliced Convex Hull volumes, extracts 9 anatomical biometric measurements, and predicts live cattle weight using regularized Lasso L1 regression.

---

## System Pipeline Architecture

```
                 +-------------------------------------------------+
                 |  Multi-Camera Sensor Acquisition (Kinect v2)   |
                 |      Top (Reference), Left, and Right Views     |
                 +-----------------------+-------------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |      Salient U²-Net Matting & Bar Bridging      |
                 |      Morphological Closing & Area Filtering     |
                 +-----------------------+-------------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |    Metric SE(3) Rigid Coordinate Alignment     |
                 |   Left -> Top and Right -> Top Transformations  |
                 +-----------------------+-------------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |      DBSCAN Density Clustering & SOR Filter     |
                 |    (eps = 0.08 m, min_pts = 50, nb = 20, 2.0σ)   |
                 +-----------------------+-------------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |        Export Clean 3D Point Cloud (.PLY)       |
                 +-----------------------+-------------------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
+---------------------------------------+ +---------------------------------------+
|  3-Axis Sliced Convex Hull Volume     | |  9 Anatomical Biometric Measurements  |
|  - Longitudinal Integral: ∫ Ax(x) dx  | |  - Withers Height    - Hip Height     |
|  - Lateral Integral:      ∫ Ay(y) dy  | |  - Chest Depth       - Chest Width    |
|  - Vertical Integral:     ∫ Az(z) dz  | |  - Ilium Width       - Hip Joint W    |
|  - Mean Volume (m³ & Liters)          | |  - Oblique Length    - Hip Length     |
|                                       | |  - Heart Girth                        |
+---------------------------------------+ +-------------------+-------------------+
                                                              |
                                                              v
                                          +---------------------------------------+
                                          |   Lasso L1 Live Weight Prediction     |
                                          |      Predicted Weight in Kilograms    |
                                          +---------------------------------------+
```

---

## Directory Organization

```
animal_wt_volume/
├── app.py                   # Streamlit web application dashboard (Söl Haus editorial design)
├── custom_processor.py      # Custom & random sensor input processing engine
├── batch_process_all.py     # End-to-end batch processing pipeline for all 103 cattle
├── segmentation.py          # U²-Net segmentation with morphological chute-bar bridging
├── calibration.py           # Extrinsic SE(3) checkerboard calibration utilities
├── merge_clouds.py          # Multi-view point cloud rigid alignment and spatial fusion
├── refine_cloud.py          # DBSCAN density clustering and Statistical Outlier Removal
├── volume_sliced_hull.py    # 3-Axis Sliced Convex Hull volume integration
├── predict_weight.py        # 3D anatomical biometric landmark extraction & weight prediction
├── train_weight_model.py    # Lasso regression model training script
├── weight_model.joblib      # Serialized trained model weights and feature configuration
├── requirements.txt         # Pinned Python package dependencies
├── run_app.bat              # Windows batch launch shortcut
├── run_app.ps1              # PowerShell launch script
│
├── data/                    # Dataset of 103 Hereford cows
│   ├── LeftClouds/          # Left sensor colored point clouds (.ply)
│   ├── RightClouds/         # Right sensor colored point clouds (.ply)
│   ├── TopClouds/           # Top reference sensor point clouds (.ply)
│   ├── LeftDepth/           # Left 16-bit depth maps (424x512)
│   ├── RightDepth/          # Right 16-bit depth maps (424x512)
│   ├── TopDepth/            # Top 16-bit depth maps (424x512)
│   ├── LeftRgb/             # Left HD RGB sensor images
│   ├── RightRgb/            # Right HD RGB sensor images
│   ├── TopRgb/              # Top HD RGB sensor images
│   ├── Matrices/            # Extrinsic calibration files (.matrices)
│   └── Measurements.xlsx    # Ground truth tape measurements and platform scale weights
│
├── segmented_output/        # U²-Net alpha masks and segmented foreground images
├── aligned_output/          # Merged and clean filtered 3D models (cow_{id}_clean.ply)
└── custom_output/           # Artifacts generated from custom/random sensor input
```

---

## Installation & Setup

### 1. Environment Preparation

Ensure Python 3.10+ is installed. Create and activate a virtual environment:

```powershell
Set-Location "C:\Users\mhars\Downloads\animal_wt_volume"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Running the Application

### Launch Streamlit Dashboard

Run the preconfigured launch script:

```powershell
.\run_app.ps1
```

Or execute directly with Streamlit:

```powershell
.\venv\Scripts\streamlit.exe run app.py --server.port 8501
```

Open your browser to: **`http://localhost:8501`**

---

## Operating Modes

### Mode 1: Cattle Database Records (1 to 103)
- Dropdown selector covering all 103 cattle records with instant lookup.
- Synchronized multi-view segmentation results with **Top View** as the primary active tab.
- Reconstructed 3D point cloud metrics with one-click **`Download 3D Point Cloud (.PLY)`** export.
- 3-Axis Sliced Convex Hull Volume integration ($m^3$ and Liters).
- 9 anatomical biometric measurements compared against certified manual tape values.
- Regularized Lasso L1 Live Weight prediction ($\text{kg}$) compared with scale ground truth.

### Mode 2: Custom & Random Sensor Input
- **Quick Test**: Click `🎲 Load Random Cattle Sample from Dataset` to randomly load a cattle record into the custom fields for immediate testing.
- **Custom Uploads**: Upload arbitrary RGB images (`.png`, `.jpg`) and Depth maps (`.png`, `.tiff`, `.npy`) across Top, Left, and Right views with live false-color Viridis depth previews.
- **Calibration**: Upload custom `.matrices` calibration files or edit the $4 \times 4$ $SE(3)$ transformation matrices directly.
- **Processing**: Click `Process Custom Cattle Input` to execute full segmentation, 3D fusion, volume computation, biometric extraction, and live weight estimation.

---

## Standalone Pipeline Execution

Individual modules can be executed independently from the command line:

```powershell
# 1. Run batch processing for all 103 cattle
python batch_process_all.py

# 2. Segment a single cow view
python segmentation.py --view Top --cow 1

# 3. Merge multi-view point clouds for a cow
python merge_clouds.py --cow 1

# 4. Refine merged cloud with DBSCAN and SOR
python refine_cloud.py --cow 1

# 5. Compute 3-axis Sliced Convex Hull volume
python volume_sliced_hull.py --cow 1

# 6. Extract 3D biometrics and predict live weight
python predict_weight.py --cow 1
```

---

## Project Team

- **M. Harshavardhan Reddy** (ID: 25075073)
- **PVK Sathwic** (ID: 25075093)
- **B Anjana Akhileshwar** (ID: 25075035)
- **YV Suresh Kumar Reddy** (ID: 25075152)
- **J Shashank Rahul** (ID: 25075065)

---

## Academic References

1. **Ruchay, A., Kober, V., Dorofeev, K., Kolpakov, V., & Dzhulamanov, K. (2020)**. Cattle body volume estimation using depth cameras. *Computers and Electronics in Agriculture*, 175, 105578.
2. **Rozendo, C., et al. (2025)**. Predicting live weight of beef cattle from body measurements and 3D computer vision. *Computers and Electronics in Agriculture*, 228, 109520.
3. **Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020)**. $U^2$-Net: Going Deeper with Nested U-Structure for Salient Object Detection. *Pattern Recognition*, 106, 107404.
4. **Cavalieri, Bonaventura (1635)**. *Geometria Indivisibilibus Continuorum Nova Quadam Ratione Promota*. Bologna: Ferronius.
5. **Salau, J., Haas, J. H., Junge, W., & Krieter, J. (2014)**. Extrinsic calibration of a multi-Kinect camera setup and automated body condition scoring. *Journal of Dairy Science*, 97(5), 2736-2746.
6. **Tasdemir, S., Urkmez, A., & Inal, S. (2011)**. Determination of body measurements on Holstein cows using digital image analysis and estimation of body weight with regression analysis. *Computers and Electronics in Agriculture*, 76(2), 189-197.
7. **Zhou, Q.-Y., Park, J., & Koltun, V. (2018)**. Open3D: A Modern Library for 3D Data Processing. *arXiv:1801.09847*.
