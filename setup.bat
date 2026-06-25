@echo off
chcp 65001 >nul
echo =====================================================
echo   Setup: Phan Tich Cam Xuc Binh Luan Khach Hang
echo =====================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Python chua duoc cai dat!
    echo       Tai Python tai: https://python.org/downloads
    pause
    exit /b 1
)

echo [1/4] Tao virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [LOI] Khong the tao virtual environment
    pause
    exit /b 1
)

echo [2/4] Kich hoat virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Cai dat cac thu vien...
python -m pip install --upgrade pip --quiet
echo     Dang cai PyTorch (CPU-only, ~280 MB)...
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo [CANH BAO] Cai torch CPU that bai, thu ban mac dinh...
    pip install torch --quiet
)
echo     Dang cai cac thu vien con lai...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [LOI] Cai dat thu vien that bai
    pause
    exit /b 1
)

echo [4/4] Tai du lieu NLTK (VADER lexicon)...
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); print('NLTK OK')"

echo.
echo =====================================================
echo   Setup hoan tat thanh cong!
echo   Chay ung dung: run.bat
echo =====================================================
pause
