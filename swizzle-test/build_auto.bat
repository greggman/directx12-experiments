@echo off
echo Building DirectX 12 Textured Triangle Example...

REM Try to find Visual Studio installation
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo Visual Studio not found in common locations.
    echo Please run this from a Visual Studio Developer Command Prompt, or modify the paths in this script.
    goto :error
)

echo Compiling with Visual Studio compiler...
cl.exe textured_triangle_dx12.cpp /Fe:textured_triangle_dx12.exe /EHsc /I"%WindowsSdkDir%Include\%WindowsSDKVersion%\um" /I"%WindowsSdkDir%Include\%WindowsSDKVersion%\shared" /link user32.lib gdi32.lib /SUBSYSTEM:CONSOLE

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Run textured_triangle_dx12.exe to see the textured triangle.
) else (
    echo Build failed.
    goto :error
)
goto :end

:error
echo.
echo Alternative: Run from Visual Studio Developer Command Prompt and use:
echo cl.exe textured_triangle_dx12.cpp /Fe:textured_triangle_dx12.exe /EHsc
echo.

:end
