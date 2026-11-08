# Run Streamlit Application
Set-Location -Path $PSScriptRoot
Write-Host "Launching Cattle 3D Reconstruction, Volume & Weight Estimation Web Application..." -ForegroundColor Cyan
& ".\venv\Scripts\streamlit.exe" run app.py --server.port 8501
