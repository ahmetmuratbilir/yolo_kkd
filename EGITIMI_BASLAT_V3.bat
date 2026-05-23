@echo off
title YOLOv8 V3 Egitim - RTX 4050 GPU
echo.
echo  ==========================================
echo   YOLOv8 V3 Egitim Basliyor...
echo   62.328 resim - best.pt'den devam
echo   Tahmini sure: 8-12 saat
echo  ==========================================
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python train_v3.py
echo.
echo  EGITIM TAMAMLANDI!
echo  Model: runs\detect\custom_ppe_v3\weights\best.pt
pause
