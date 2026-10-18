@echo off
chcp 65001 >nul
cd /d "%~dp0.."

echo ========================================
echo   PAA-DA 黏附预测 - 独立试用版
echo ========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 找不到 python。
    echo 请先安装 Python，并在安装时勾选 "Add Python to PATH"。
    pause
    exit /b 1
)

python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo 首次运行，正在安装依赖，请稍候...
    python -m pip install -r pilot_adhesion\requirements.txt
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败，请检查网络或 Python 环境。
        pause
        exit /b 1
    )
)

echo 正在启动，浏览器会自动打开...
echo 如果没有自动打开，请访问 http://localhost:8501
echo 关闭本窗口即可停止服务。
echo.

python -m streamlit run pilot_adhesion\app.py

if errorlevel 1 (
    echo.
    echo 启动失败，请把上面的错误信息发给开发者。
    pause
)
