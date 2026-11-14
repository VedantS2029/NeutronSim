"""Open the GUI with one command, no installation needed:  python run_gui.py
(the wizard: pick isotope and energy, run; it has the periodic-table explorer and, on the results page, the visualizer)
Box test window:  python run_gui.py boxgui [ISOTOPE]   (also Tools > Box test... in the wizard)
Other windows:  python run_gui.py explore [ISOTOPE]   opens the periodic-table explorer on its own."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from neutronsim import cli  # noqa: E402

sys.exit(cli.main(sys.argv[1:] or ["gui"]))
