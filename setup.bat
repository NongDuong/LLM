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

echo [1/5] Tao virtual environment...
if exist venv (
    echo     Xoa venv cu...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo [LOI] Khong the tao virtual environment
    pause
    exit /b 1
)

echo [2/5] Kich hoat virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Nang cap pip...
python -m pip install --upgrade pip --quiet

echo [4/5] Cai dat cac thu vien (streamlit, transformers, ...)...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [LOI] Cai dat thu vien that bai
    pause
    exit /b 1
)

echo [5/5] Cai de PyTorch CPU-only (ghi de ban CUDA neu co)...
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --quiet
if errorlevel 1 (
    echo [LOI] Cai dat PyTorch CPU that bai. Kiem tra ket noi internet.
    pause
    exit /b 1
)

echo     Tai du lieu NLTK (VADER lexicon)...
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); print('NLTK OK')"

echo     Kiem tra PyTorch...
python -c "import torch; v=torch.__version__; print('PyTorch:', v); exit(0 if '+cpu' in v else 1)"
if errorlevel 1 (
    echo [CANH BAO] PyTorch co the chua phai ban CPU. Chay lai setup.bat.
    pause
    exit /b 1
)

echo.
echo =====================================================
echo   Setup hoan tat! Chay ung dung: run.bat
echo =====================================================
pause
