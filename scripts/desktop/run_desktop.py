"""Launch the local D2 browser; defaults to an empty workspace."""

from pathlib import Path
import sys

if not getattr(sys, "frozen", False):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT/"src"))

from cmvd_demo.desktop.launch import main


if __name__ == "__main__":
    raise SystemExit(main())
