"""Launch the local D2 browser; defaults to an empty workspace."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case-dir", type=Path, help="one explicit unpacked case directory")
    group.add_argument("--synthetic", action="store_true", help="open an entirely synthetic orientation phantom")
    args = parser.parse_args()
    from PySide6.QtCore import QTimer,QCoreApplication,QEvent
    from PySide6.QtWidgets import QApplication
    from cmvd_demo.desktop.window import BrowserWindow
    app = QApplication([sys.argv[0]])
    app.setApplicationName("CMVD Research Browser")
    app.setStyle("Fusion")
    window = BrowserWindow()
    window.show()
    if args.synthetic:
        QTimer.singleShot(0, window.open_synthetic)
    elif args.case_dir:
        QTimer.singleShot(0, lambda: window.open_case(args.case_dir))
    code = app.exec()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
