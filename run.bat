@echo off
chcp 65001 >nul
echo === Khoi dong ung dung Phan Tich Cam Xuc ===
call venv\Scripts\activate.bat
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
