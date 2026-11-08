@echo off
title Cattle 3D Reconstruction, Volume & Weight Estimation
cd /d "%~dp0"
echo Starting Streamlit Application...
call .\venv\Scripts\streamlit.exe run app.py --server.port 8501
pause
