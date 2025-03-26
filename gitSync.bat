@echo off
setlocal enabledelayedexpansion

:: 检查是否提供了两个参数
if "%~2"=="" (
    echo Usage format: %0 "Component" "Description"
    exit /b 1
)

:: 设置时间变量和描述变量
set component=%~1
set description=%~2



:: 执行Git命令
git add ./*.md ./*.png ./*.jpg ./*.html ./*.cpp  ./*.h ./*.json ./*.bat
git commit -m "!component!: !description!"
git push

:: 结束脚本
endlocal