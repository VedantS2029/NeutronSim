"""Small helpers that keep neutronsim windows inside the screen and let their contents resize.

No tkinter import here: every function takes the widget it should act on, so the module is safe to
load on headless installs and under the test stub. All of them swallow display errors and fall back
to the caller's own defaults, because a layout nicety must never stop the program from opening.
"""
from __future__ import annotations

SCREEN_FRACTION = 0.92   # never ask for more than this share of the screen
TASKBAR_ALLOWANCE = 48   # pixels reserved for panels/docks/title bars


def fit_scale(avail_w: float, avail_h: float, base_w: float, base_h: float,
              min_k: float = 0.4, max_k: float = 1.0) -> float:
    """Largest uniform scale (<= max_k) at which a base_w x base_h drawing fits in avail_w x avail_h."""
    k = min(avail_w / base_w, avail_h / base_h)
    return max(min_k, min(max_k, k))


def screen_size(win) -> tuple[int, int] | None:
    try:
        return int(win.winfo_screenwidth()), int(win.winfo_screenheight())
    except Exception:  # no display, or the headless test stub
        return None


def fit_window(win, want_w: int, want_h: int, min_w: int = 200, min_h: int = 150) -> None:
    """Open `win` at want_w x want_h, shrunk to fit the screen, centred, and freely resizable."""
    try:
        size = screen_size(win)
        if size is None:
            win.geometry(f"{int(want_w)}x{int(want_h)}")
            win.minsize(int(min_w), int(min_h))
            return
        sw, sh = size
        w = int(min(want_w, sw * SCREEN_FRACTION))
        h = int(min(want_h, sh * SCREEN_FRACTION - TASKBAR_ALLOWANCE))
        x, y = max(0, (sw - w) // 2), max(0, (sh - h - TASKBAR_ALLOWANCE) // 2)
        win.geometry(f"{w}x{h}+{x}+{y}")
        win.minsize(min(int(min_w), w), min(int(min_h), h))
        win.resizable(True, True)
    except Exception:
        pass


def autowrap(label, pad: int = 8, min_wrap: int = 120) -> None:
    """Make a label re-wrap its text to its own width. The label must be packed with fill="x"
    (or gridded sticky="ew") and live in a window whose size is fixed, not derived from its contents."""
    def _fit(event):
        try:
            label.configure(wraplength=max(min_wrap, event.width - pad))
        except Exception:
            pass
    label.bind("<Configure>", _fit)
