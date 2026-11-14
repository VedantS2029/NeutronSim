"""A stand-in for tkinter so GUI code can be exercised headlessly (no display, no Tk install).

It cannot judge appearance; it catches wiring errors on any machine, including CI.
"""
import contextlib
import importlib.util
import sys
import types


class Var:
    def __init__(self, master=None, value=None):
        self.v, self.cbs = value, []

    def get(self):
        return self.v

    def set(self, x):
        self.v = x
        for cb in self.cbs:
            cb()

    def trace_add(self, _mode, cb):
        self.cbs.append(cb)


class W:
    """Accepts any widget call. Tracks children, text content and disabled state."""
    def __init__(self, master=None, *a, **k):
        self.kids, self.text, self.disabled, self.master, self.calls = [], "", False, master, []
        if isinstance(master, W):
            master.kids.append(self)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def call(*a, **k):
            self.calls.append((name, a, k))
        return call

    def winfo_children(self):
        return list(self.kids)

    def destroy(self):
        self.dead = True
        if self.master is not None and self in getattr(self.master, "kids", []):
            self.master.kids.remove(self)

    def curselection(self):
        return (0,)

    def instate(self, _s):
        return not self.disabled

    def state(self, s):
        self.disabled = "disabled" in s and "!disabled" not in s

    def insert(self, _i, text, *_t):
        self.text = text if not self.text else self.text + text

    def delete(self, *_):
        self.text = ""
        self.calls = []

    def count(self, name):
        return sum(1 for c in self.calls if c[0] == name)


class Root(W):
    def __init__(self, *a, **k):
        super().__init__(None)
        self.pending = []

    def after(self, _ms, fn=None, *args):
        self.pending.append(fn)

    def destroy(self):
        self.closed = True


class FontObj:
    def __init__(self, size=12, **_k):
        self.size = size

    def cget(self, _k):
        return self.size

    def actual(self, _k):
        return "Helvetica"

    def copy(self):
        return FontObj(self.size)

    def configure(self, **kw):
        self.size = kw.get("size", self.size)


@contextlib.contextmanager
def stubbed_tk():
    names = ("tkinter", "tkinter.ttk", "tkinter.font", "tkinter.filedialog", "tkinter.messagebox")
    mods = {n: types.ModuleType(n) for n in names}
    tk, ttk, font, fd, mb = (mods[n] for n in names)
    tk.Tk, tk.Toplevel, tk.Text, tk.Listbox, tk.Menu, tk.Canvas = Root, W, W, W, W, W
    tk.StringVar = tk.BooleanVar = Var
    tk.TclError = type("TclError", (Exception,), {})
    for n in ("Frame", "Label", "Button", "Entry", "Combobox", "Checkbutton", "Radiobutton", "Separator",
              "Scrollbar", "Scale", "Notebook"):
        setattr(ttk, n, W)
    font.nametofont = lambda _n: FontObj()
    font.Font = FontObj
    tk.ttk, tk.font, tk.filedialog, tk.messagebox = ttk, font, fd, mb
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules.update(mods)
    try:
        yield tk
    finally:
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v


def load_source(modname, path):
    """Import a source file under a throwaway name (call inside stubbed_tk())."""
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclasses looks the module up by name
    spec.loader.exec_module(mod)
    return mod
