"""Validate the actual exe and ZIP in relocated paths with a clean environment."""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.perfusion.storage import load_json, write_json
from cmvd_demo.desktop.presentation import PRODUCT_NAME, PRODUCT_VERSION


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def clean_environment():
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("PYTHON", "CONDA", "QT_", "QML"))}
    windows = env.get("SystemRoot", "C:/Windows")
    env["PATH"] = os.pathsep.join((str(Path(windows)/"System32"), windows))
    return env


def check_tree(package, manifest):
    files = {p.relative_to(package).as_posix(): p for p in package.rglob("*") if p.is_file()}
    if set(files) != set(manifest["files_sha256"]):
        raise ValueError("release_payload_file_set_changed")
    for relative, expected in manifest["files_sha256"].items():
        if sha(files[relative]) != expected:
            raise ValueError("release_payload_bytes_changed")


def self_check(exe, cwd, workspace, output, *, case=None, resume=None):
    cmd = [str(exe), "--self-check", "--workspace", str(workspace), "--check-output", str(output)]
    if case is not None:
        cmd += ["--case-dir", str(case)]
    if resume is not None:
        cmd += ["--check-resume",str(resume)]
    process = subprocess.Popen(cmd, cwd=cwd, env=clean_environment(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic()+300
    while process.poll() is None:
        if startup_error(process.pid):
            close_window(process.pid)
            process.wait(timeout=30)
            raise ValueError("release_frozen_bootloader_error")
        if time.monotonic() >= deadline:
            close_window(process.pid)
            raise ValueError("release_frozen_check_timeout")
        time.sleep(.2)
    code = process.returncode
    if code or not (output/"bundle.json").is_file():
        raise ValueError("release_frozen_check_failed")
    report = load_json(output/"report.json")
    if not report["passed"] or report["frozen"] is not True or report["qt_platform"] != "windows":
        raise ValueError("release_frozen_check_not_native_or_passed")
    return report


def app_windows(pid):
    windows = []
    user = ctypes.windll.user32
    user.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user.IsWindowVisible.argtypes = (wintypes.HWND,)
    user.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    @callback_type
    def callback(hwnd, _):
        owner = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user.IsWindowVisible(hwnd):
            title = ctypes.create_unicode_buffer(512)
            user.GetWindowTextW(hwnd, title, 512)
            if PRODUCT_NAME in title.value or "CMVD" in title.value or title.value in ("Unhandled exception in script", "读取病例"):
                windows.append(hwnd)
        return True
    user.EnumWindows(callback, 0)
    return windows


def startup_error(pid):
    user = ctypes.windll.user32
    for hwnd in app_windows(pid):
        title = ctypes.create_unicode_buffer(512)
        user.GetWindowTextW(hwnd, title, 512)
        if title.value == "Unhandled exception in script":
            return True
    return False


def close_window(pid):
    user = ctypes.windll.user32
    user.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    for hwnd in app_windows(pid):
        user.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE; no process termination.


def normal_launch(exe, cwd):
    process = subprocess.Popen([str(exe)], cwd=cwd, env=clean_environment(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic()+40
    seen = False
    while time.monotonic() < deadline and process.poll() is None:
        if startup_error(process.pid):
            break
        if app_windows(process.pid):
            seen = True
            break
        time.sleep(.2)
    # A fresh QProgressDialog can force-show several seconds after the main
    # window appears. Observing only the first window misses this regression.
    unexpected_progress = False
    observation_start = time.monotonic()
    while seen and process.poll() is None and time.monotonic()-observation_start < 8:
        for hwnd in app_windows(process.pid):
            title = ctypes.create_unicode_buffer(512)
            ctypes.windll.user32.GetWindowTextW(hwnd,title,512)
            if title.value == "读取病例":
                unexpected_progress = True
        if unexpected_progress or startup_error(process.pid):
            break
        time.sleep(.2)
    observed_full_idle = seen and process.poll() is None and time.monotonic()-observation_start >= 8
    close_window(process.pid)
    try:
        code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        raise ValueError("release_normal_close_timeout") from None
    if not seen or code:
        raise ValueError("release_normal_launch_or_exit_failed")
    if unexpected_progress:
        raise ValueError("release_idle_start_showed_case_progress")
    if not observed_full_idle:
        raise ValueError("release_normal_idle_closed_early")
    return {"normal_no_argument_window_seen": True, "normal_exit_code": code,
            "python_conda_qt_environment_removed": True, "path_only_system_windows": True,
            "idle_observation_seconds": 8, "unexpected_case_progress_seen": False}


def validate(run, with_sample):
    run = run.resolve()
    manifest = load_json(run/"build_manifest.json")
    package = run/"application/CMVDBrowser"
    check_tree(package, manifest)
    output = run/"validation"
    output.mkdir(exist_ok=False)
    report = {"completed": False, "passed": False, "scope": "native Windows source-independent frozen executable and relocated ZIP",
              "different_computer_tested": False, "clinical_validated": False, "failures": []}
    write_json(output/"attempt.json", report)
    try:
        case = None
        if with_sample:
            cases = [p for p in (ROOT/"data/阳性").iterdir() if p.is_dir() and not p.is_symlink()]
            if len(cases) != 1:
                raise ValueError("release_one_development_sample_required")
            case = cases[0]
        # New unrelated cwd for original exe; the supplied workspace is not source.
        cwd = output/"unrelated-cwd"
        cwd.mkdir()
        print("checking_frozen_original", flush=True)
        report["original_exe"] = self_check(package/"CMVDBrowser.exe", cwd, output/"original-workspace", output/"original-check", case=case)
        check_tree(package, manifest)
        archive = run/f"CMVDBrowser-Windows-x64-{PRODUCT_VERSION}.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for name in sorted(manifest["files_sha256"]):
                zipped.write(package/name, "CMVDBrowser/"+name)
        relocation = Path("E:/CMVD_Release_Check")/run.name/"中文 空格目录"
        relocation.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("release_zip_crc_failed")
            zipped.extractall(relocation)
        relocated = relocation/"CMVDBrowser"
        check_tree(relocated, manifest)
        print("checking_relocated_zip_without_python_or_conda_path", flush=True)
        report["relocated_exe"] = self_check(relocated/"CMVDBrowser.exe", relocation,
            relocation/"研究工作目录", output/"relocated-check",
            resume=output/"original-workspace"/report["original_exe"]["saved_work_record"])
        report["normal_launch"] = normal_launch(relocated/"CMVDBrowser.exe", relocation)
        check_tree(package, manifest)
        check_tree(relocated, manifest)
        report.update(completed=True, passed=True, archive_name=archive.name,
                      archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                      payload_files=manifest["file_count"], payload_bytes=manifest["bytes"],
                      payload_patient_data_files=0, code_signed=False)
        write_json(output/"report.json", report)
        (run/"SHA256.txt").write_text(report["archive_sha256"]+"  "+archive.name+"\n", encoding="utf-8")
        write_json(run/"bundle.json", {"completed": True, "kind": "validated_windows_browser_release",
            "files_sha256": {"build_manifest.json": sha(run/"build_manifest.json"), archive.name: sha(archive),
                             "validation/report.json": sha(output/"report.json"), "SHA256.txt": sha(run/"SHA256.txt")},
            "clinical_validated": False, "different_computer_tested": False})
        print("release_validated="+run.relative_to(ROOT).as_posix(), flush=True)
    except Exception as error:
        report["failures"].append(str(error) if isinstance(error, ValueError) else type(error).__name__)
        write_json(output/"failure.json", report)
        raise


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--with-sample", action="store_true")
    args = parser.parse_args()
    try:
        validate(args.release, args.with_sample)
        return 0
    except Exception as error:
        print("not_completed="+(str(error) if isinstance(error, ValueError) else type(error).__name__), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
