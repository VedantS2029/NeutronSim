"""Double-click launcher for neutronsim (Windows, macOS, Linux). No commands, no installation.

Normally you start it through "Neutron Sim.bat" (Windows) or "Neutron Sim.command" (Mac), which find
Python and run this file. Double-clicking this file works too if Python is installed.

It checks that Python and tkinter are good enough, opens the setup wizard, and if anything is wrong it
shows a plain-language message box (not a console that vanishes) and writes neutronsim-error.log.
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
TITLE = "Neutron Sim"


def _log_path() -> Path:
    for base in (HERE, Path(tempfile.gettempdir())):
        try:
            p = base / "neutronsim-error.log"
            with open(p, "a", encoding="utf-8"):
                pass
            return p
        except OSError:
            continue
    return Path("neutronsim-error.log")


def show_message(text: str, error: bool = True) -> None:
    """Message box that works even when tkinter does not; falls back to printing."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        (messagebox.showerror if error else messagebox.showwarning)(TITLE, text)
        root.destroy()
        return
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, TITLE, 0x10 if error else 0x30)
            return
        if sys.platform == "darwin":
            import subprocess
            esc = text.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(["osascript", "-e", f'display dialog "{esc}" with title "{TITLE}" buttons {{"OK"}}'],
                           check=False)
            return
    except Exception:
        pass
    print(text, file=sys.stderr if sys.stderr else sys.stdout)


def _fix_missing_streams() -> None:
    """pythonw.exe (no console) has sys.stdout/sys.stderr = None; send them to the log file instead."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        try:
            sys.stderr = open(_log_path(), "a", encoding="utf-8")
        except OSError:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _sharp_on_windows() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on high-DPI screens
        except Exception:
            pass


def main() -> int:
    _fix_missing_streams()
    if sys.version_info < (3, 10):
        show_message(f"Neutron Sim needs Python 3.10 or newer; this is Python {sys.version.split()[0]}.\n\n"
                     "Install the latest Python from https://www.python.org/downloads/ and double-click again.")
        return 1
    try:
        import tkinter
    except ImportError:
        hint = {"win32": "Reinstall Python from python.org and keep \"tcl/tk and IDLE\" ticked.",
                "darwin": "Install Python from https://www.python.org/downloads/ (its installer includes tkinter)."}.get(
            sys.platform, "On Linux run: sudo apt install python3-tk")
        show_message("This Python was installed without tkinter, which the window needs.\n\n" + hint)
        return 1
    if sys.platform == "darwin" and tkinter.TkVersion < 8.6:
        show_message("This Mac's built-in Python uses an old, unreliable Tk (8.5); the window may look broken "
                     "or crash.\n\nFor best results install Python from https://www.python.org/downloads/ "
                     "and double-click again. Continuing anyway.", error=False)
    _sharp_on_windows()
    try:
        from neutronsim import cli
        code = cli.main(["gui"])
    except Exception:
        tb = traceback.format_exc()
        log = _log_path()
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(tb + "\n")
        except OSError:
            pass
        show_message(f"Neutron Sim hit an unexpected error.\n\n{tb.strip().splitlines()[-1]}\n\nDetails: {log}")
        return 1
    if code:
        show_message("The window could not be opened (no display available?).\n"
                     f"Details, if any: {_log_path()}")
    return code or 0


if __name__ == "__main__":
    sys.exit(main())
