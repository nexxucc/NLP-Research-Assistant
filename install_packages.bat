@echo off
echo Installing NLP Research Assistant dependencies...
echo.
echo WARNING: Python 3.13 is very new. Some packages may not be available.
echo Consider using Python 3.11 or 3.12 if you encounter issues.
echo.

REM Core utilities (should work)
echo [1/12] Installing pypdf...
pip install pypdf
echo.

echo [2/12] Installing numpy...
pip install numpy
echo.

echo [3/12] Installing tqdm...
pip install tqdm
echo.

echo [4/12] Installing python-dotenv...
pip install python-dotenv
echo.

echo [5/12] Installing pandas...
pip install pandas
echo.

REM Google API (might work)
echo [6/12] Installing google-generativeai...
pip install google-generativeai
echo.

REM ML packages (might have issues with Python 3.13)
echo [7/12] Installing torch...
pip install torch
echo.

echo [8/12] Installing transformers...
pip install transformers
echo.

echo [9/12] Installing sentence-transformers...
pip install sentence-transformers
echo.

echo [10/12] Installing faiss-cpu...
pip install faiss-cpu
echo.

echo [11/12] Installing rank-bm25...
pip install rank-bm25
echo.

echo [12/12] Installing scikit-learn...
pip install scikit-learn
echo.

REM Optional packages
echo Installing optional packages...
pip install streamlit rouge-score nltk
echo.

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo If any packages failed, you may need to:
echo 1. Use Python 3.11 or 3.12 instead of 3.13
echo 2. Or wait for package updates to support Python 3.13
echo.
pause
