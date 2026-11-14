@echo off
rem Double-click to open Neutron Sim on Windows. Nothing to install if Python 3.10+ is present.
setlocal
cd /d "%~dp0"
set "PYCHECK=import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"

rem 1) the "py" launcher that python.org installs
where pyw >nul 2>nul && (
  py -3 -c "%PYCHECK%" >nul 2>nul && ( start "" pyw -3 "%~dp0launcher.py" & exit /b 0 )
)
rem 2) python / pythonw on PATH
where pythonw >nul 2>nul && (
  python -c "%PYCHECK%" >nul 2>nul && ( start "" pythonw "%~dp0launcher.py" & exit /b 0 )
)
rem 3) the usual per-user install folders
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
  if exist "%%D\pythonw.exe" (
    "%%D\python.exe" -c "%PYCHECK%" >nul 2>nul && ( start "" "%%D\pythonw.exe" "%~dp0launcher.py" & exit /b 0 )
  )
)

echo.
echo  Neutron Sim needs Python 3.10 or newer, and it was not found on this PC.
echo.
echo  1. A download page will open. Download and run the Python installer.
echo  2. On the first screen, TICK "Add python.exe to PATH", then click Install.
echo  3. Double-click "Neutron Sim.bat" again.
echo.
pause
start "" "https://www.python.org/downloads/windows/"
