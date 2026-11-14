"""Run everything with one command, no installation needed:  python run_all.py   (add -v for full output)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from neutronsim import cli  # noqa: E402

sys.exit(cli.main(["all", *sys.argv[1:]]))
