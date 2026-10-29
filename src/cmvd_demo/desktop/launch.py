"""Shared source/frozen entry point; mutable work stays outside the bundle."""

import argparse
from datetime import datetime, timezone
import faulthandler
import os
from pathlib import Path
import sys
import uuid

from cmvd_demo.paths import configure_workspace
from .presentation import PRODUCT_NAME, PRODUCT_VERSION, error_message


def main(argv=None):
    parser = argparse.ArgumentParser(description=PRODUCT_NAME)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case-dir", type=Path, help="one explicit unpacked case directory")
    group.add_argument("--synthetic", action="store_true", help="open synthetic orientation phantom")
    parser.add_argument("--workspace", type=Path, help="explicit local writable work directory")
    parser.add_argument("--self-check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--check-resume", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_check and (args.workspace is None or args.check_output is None):
        parser.error("self-check requires explicit workspace and output")
    if args.check_resume is not None and not args.self_check:
        parser.error("check-resume requires self-check")
    workspace = configure_workspace(args.workspace)
    logs = workspace/"outputs/runtime_logs"
    logs.mkdir(parents=True, exist_ok=True)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    # PyInstaller windowed executables deliberately have no console streams.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    from PySide6.QtCore import QTimer, QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication, QMessageBox
    from cmvd_demo.desktop.window import BrowserWindow
    app = QApplication([sys.argv[0]])
    app.setApplicationName(PRODUCT_NAME)
    app.setApplicationVersion(PRODUCT_VERSION)
    app.setStyle("Fusion")
    with (logs/(run+"-native.log")).open("x", encoding="utf-8") as native:
        faulthandler.enable(native)
        previous_hook = sys.excepthook
        def exception_hook(kind, error, tb):
            from cmvd_demo.io import DicomReadError
            code = error.code if isinstance(error, DicomReadError) else kind.__name__
            with (logs/(run+"-errors.log")).open("a", encoding="utf-8") as stream:
                stream.write(code+"\n")
            QMessageBox.critical(None, PRODUCT_NAME+" · 操作未完成", error_message(code))
        sys.excepthook = exception_hook
        try:
            window = BrowserWindow()
            window.show()
            if args.self_check:
                app.setQuitOnLastWindowClosed(False)
                from .release_check import run_checks
                return run_checks(app, window, args.check_output, case_dir=args.case_dir, resume_record=args.check_resume)
            if args.synthetic:
                QTimer.singleShot(0, window.open_synthetic)
            elif args.case_dir:
                QTimer.singleShot(0, lambda: window.open_case(args.case_dir))
            code = app.exec()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            return code
        finally:
            sys.excepthook = previous_hook
            faulthandler.disable()
