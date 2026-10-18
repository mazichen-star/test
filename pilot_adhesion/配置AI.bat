@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".streamlit" mkdir ".streamlit"

if not exist ".streamlit\secrets.toml" (
    copy ".streamlit\secrets.toml.example" ".streamlit\secrets.toml" >nul
    echo 已创建配置文件：pilot_adhesion\.streamlit\secrets.toml
) else (
    echo 正在打开已有配置文件...
)

echo.
echo 请填写 DEEPSEEK_API_KEY 和 DEEPSEEK_MODEL，保存后关闭记事本。
echo 也可以填写 ARK_API_KEY 和 ARK_MODEL。
echo 不要把 key 发到聊天或提交到 Git。
echo.

notepad ".streamlit\secrets.toml"

echo.
echo 配置完成后，关闭本窗口，双击“启动试用版.bat”。
pause
