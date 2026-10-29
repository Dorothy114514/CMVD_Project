"""Build an append-only, patient-data-free Windows x64 onedir distribution."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.paths import RESOURCE_FILES
from cmvd_demo.desktop.presentation import PRODUCT_NAME, PRODUCT_VERSION
from cmvd_demo.perfusion.storage import write_json, load_json

README = """心灌影像工作站 · PET/MR 心肌灌注研究 · Windows x64 便携版

打开方法
1. 先将整个ZIP解压到本机普通文件夹，例如 E:\\CMVD_Browser。
2. 双击 CMVDBrowser.exe；保留旁边的 _internal 文件夹，不单独拷贝exe。
3. 点击“打开病例目录”选择单个已解压病例目录。
   练习操作请从顶部“合成演示 → 影像浏览与轮廓练习（完全合成）”进入。
   在“轮廓与曲线”选心肌/血池，选择“拖出椭圆”或“逐点勾画”，再点“在图上勾画”。
   椭圆按住左键拖出范围，松开完成；逐点用Enter/双击/点回起点闭合，Backspace撤一点。
   Esc或右键取消草稿；完成后可拖点细调，心肌可单独编辑外膜/内膜。
   同时创建的内膜只是几何起点，请手动核对；单帧修正只作用于所选帧。
4. “文件 → 打开工作目录”可查看ROI、曲线、特征、队列输出和运行日志。
5. “灌注分析 → 保存当前MR分析”可独立保存一个状态，不需要静息/负荷配对。
   计算原始曲线后，曲线上方选“定位影像帧”可点击定位；MR可选基线/首过后两次点选区间。
   绿色基线、金色首过色带可拖动；帧号双向同步，按真实采样时间吸附。
   修改区间后原分析和窗口复核失效，请重新计算；不会自动选择窗口或自动审核。
6. 顶部“保存工作”保存轮廓、曲线、保留分析、参数及浏览位置。
   重开程序后先打开原病例目录，再点“恢复工作”选择保存目录内的 work.json。
   “文件 → 打开最近工作记录目录”可直接查看本次保存或恢复的文件。
   记录不保存患者源路径；原件发生变化或复算不一致时停止恢复。
7. 左侧序列带读取状态，可筛选“仅索引”“有问题 / 提示”等。
   点击“序列状态与问题清单”查看原因、建议；选中问题后“定位 / 重试序列”。
   可记录“需医院核源/补导出”等本地处置，点击“导出状态与问题”单独保存。
   既有逐包JSON检查报告或已导出的 review.json 可关联为历史提示，需保留完整目录。
   历史报告不会把当前序列标成已通过；已预览也只表示读过部分位置。
8. 左侧“静息／负荷对照查看”并排浏览两侧影像；序列与层面须人工选择核对。
   默认独立选帧，可选相对时间联动；按最近采样选择，不插值或自动配准。
   先在主窗口计算并保留两状态结果，再查看对应曲线与MR测量。
9. “研究记录导出与核对”导出当前病例摘要、测量/曲线/序列/问题CSV与可恢复工作记录。
   输出在工作目录 outputs\\research_records；summary.html可本机浏览，完整目录须保留。
   可核对既有导出文件，再先打开原病例，从记录恢复工作；问题处置与对照位置只是快照。
   关闭任务窗口会请求取消；失败目录不作为完成结果。帮助菜单提供操作核对步骤；
   完整清单在工作目录 docs/USAGE_CHECKLIST.md，可用记事本查看。

首次运行的工作目录为 %LOCALAPPDATA%\\CMVDResearchBrowser\\workspace-v1。
包内资源只读；工作目录保存研究输出，状态与问题记录在 outputs\\sequence_reviews。
迁移/备份研究结果时请同时备份此目录。
如需指定位置，可在命令行使用 CMVDBrowser.exe --workspace "E:\\CMVD_Workspace"。
在多个版本间复用工作目录时，既有模板不会被覆盖；合同升级需另行审核。

本包自带Python及Qt运行库，无需安装Conda、Python或启动浏览器网页。
患者影像、身份映射、真实临床表、模型及旧研究输出均不包含在包内。
首次打开为空工作区，真实影像仍需从本机外部目录选择；不自动读取既有病例。
目前用于影像研究预览、手工ROI、探索性特征和队列整理；未配置真实诊断模型。
合成演示不可作为临床验证。不同内容的重复SOP等问题仍按严格读取器规则报告。
本exe未作发布者数字签名；本机构建验证不能替代其他电脑/医院终端的兼容性测试。
"""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def check_resources():
    for relative in RESOURCE_FILES:
        path = ROOT/relative
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            if len(rows) != 1:
                raise ValueError("release_template_must_be_header_only")
        elif "cohort_project" in path.name:
            project = load_json(path)
            if project["mode"] != "research" or any(v.startswith(("/", "\\")) or ":" in v for v in project["inputs"].values()):
                raise ValueError("release_template_contains_local_paths")
            if any(v is True for k, v in project["policy"].items() if k.endswith("_reviewed")):
                raise ValueError("release_template_contains_reviewed_project")


def copy_notices(package):
    licenses = package/"licenses"
    licenses.mkdir()
    dependencies = ("numpy", "pydicom", "pyqtgraph", "PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6", "colorama")
    records = []
    for name in dependencies:
        dist = importlib.metadata.distribution(name)
        copied = []
        for entry in dist.files or ():
            parts = Path(str(entry)).parts
            if not any(p.lower() in ("license", "licenses", "copying") or p.lower().startswith(("license.", "licence", "copying.", "copyright")) for p in parts):
                continue
            source = Path(dist.locate_file(entry))
            if not source.is_file():
                continue
            # Reproduce only upstream license paths, not machine/site-packages paths.
            safe = [p for p in parts if p not in ("..", ".")]
            target = licenses/name/Path(*safe)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(target.relative_to(package).as_posix())
        records.append({"name": name, "version": dist.version,
                        "license_expression": dist.metadata.get("License-Expression"),
                        "license": dist.metadata.get("License"), "license_files": copied,
                        "project_urls": dist.metadata.get_all("Project-URL", [])})
    # CPython license is part of the runtime distribution, not a Python wheel.
    pylicense = Path(sys.base_prefix)/"LICENSE.txt"
    if pylicense.is_file():
        shutil.copyfile(pylicense, licenses/"PYTHON-LICENSE.txt")
    write_json(package/"THIRD_PARTY_NOTICES.json", {"dependencies": records,
        "python_version": sys.version.split()[0], "notes": ["Qt/PySide shared libraries are in _internal; retain license files when sharing the complete folder.",
        "PyInstaller bootloader license and exception: https://pyinstaller.org/en/stable/license.html"]})


def build(tools, root):
    check_resources()
    if not (tools/"PyInstaller/__main__.py").is_file():
        raise ValueError("release_pinned_build_tools_missing")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    run = root/run_id
    run.mkdir(parents=True, exist_ok=False)
    work = Path("E:/Conda/cache/cmvd-build")/run_id
    if work.exists():
        raise ValueError("release_build_work_exists")
    work.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tools), str(ROOT/"src")))
    # Conda keeps CPython's ffi/bz2/lzma/expat dependencies in Library/bin.
    # Let the build-time binary scanner resolve them; do not change global PATH.
    env["PATH"] = os.pathsep.join((str(Path(sys.base_prefix)/"Library/bin"),
                                   str(Path(sys.base_prefix)/"DLLs"), env.get("PATH", "")))
    env["PYINSTALLER_CONFIG_DIR"] = str(work/"cache")
    version = subprocess.check_output([sys.executable, "-m", "PyInstaller", "--version"], env=env, text=True).strip()
    if version != "6.22.3":
        raise ValueError("release_build_tool_version_mismatch")
    cmd = [sys.executable, "-m", "PyInstaller", "--name", "CMVDBrowser", "--onedir", "--windowed", "--noupx",
           "--noconfirm", "--distpath", str(run/"application"), "--workpath", str(work/"build"),
           "--specpath", str(work/"spec"), "--paths", str(ROOT/"src"),
           "--additional-hooks-dir", str(ROOT/"scripts/release/hooks"),
           "--runtime-hook", str(ROOT/"scripts/release/runtime_hook.py"),
           "--collect-submodules", "pydicom.pixels",
           "--collect-submodules", "pydicom.pixel_data_handlers"]
    for name in ("pandas", "scipy", "sklearn", "matplotlib", "tkinter", "IPython", "PyQt5", "PyQt6", "cmvd_demo.training", "cmvd_demo.preparation"):
        cmd += ["--exclude-module", name]
    for relative in RESOURCE_FILES:
        target = "resources/"+Path(relative).parent.as_posix()
        cmd += ["--add-data", str(ROOT/relative)+os.pathsep+target]
    cmd.append(str(ROOT/"scripts/desktop/run_desktop.py"))
    print("release_run="+run.relative_to(ROOT).as_posix(), flush=True)
    write_json(run/"attempt.json", {"completed": False, "run_id": run_id, "pyinstaller": version,
        "runtime_dependencies_changed": False, "patient_data_included": False,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT/"src/cmvd_demo").rglob("*.py"))},
        "release_script_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT/"scripts/release").rglob("*.py"))},
        "resource_sha256": {name: sha(ROOT/name) for name in RESOURCE_FILES}})
    with (run/"build.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                print("packaging_in_progress", flush=True)
    if code:
        write_json(run/"failure.json", {"completed": False, "stage": "pyinstaller", "exit_code": code})
        raise ValueError("release_build_failed")
    package = run/"application/CMVDBrowser"
    (package/"使用说明.txt").write_text(README, encoding="utf-8-sig")
    copy_notices(package)
    files = sorted(p for p in package.rglob("*") if p.is_file())
    forbidden = {"data", "deidentified", "metadata", "audit", "outputs", "models", "tests", "test_files"}
    for path in files:
        relative = path.relative_to(package)
        if set(relative.parts) & forbidden or path.name == "runtime.local.json" or path.suffix.lower() in (".dcm", ".ima", ".7z", ".pkl", ".joblib"):
            write_json(run/"failure.json", {"completed": False, "stage": "payload_audit", "code": "release_forbidden_payload_file"})
            raise ValueError("release_forbidden_payload_file")
    # Recheck that all inputs stayed unchanged during the build.
    attempt = load_json(run/"attempt.json")
    for name, expected in (attempt["source_sha256"] | attempt["resource_sha256"] | attempt["release_script_sha256"]).items():
        if sha(ROOT/name) != expected:
            write_json(run/"failure.json", {"completed": False, "stage": "source_guard", "code": "release_source_changed_during_build"})
            raise ValueError("release_source_changed_during_build")
    write_json(run/"build_manifest.json", {"completed": True, "run_id": run_id,
        "product_name": PRODUCT_NAME, "product_version": PRODUCT_VERSION,
        "python": sys.version.split()[0], "pyinstaller": version,
        "file_count": len(files), "bytes": sum(p.stat().st_size for p in files),
        "files_sha256": {p.relative_to(package).as_posix(): sha(p) for p in files},
        "patient_data_included": False, "real_model_included": False, "code_signed": False,
        "runtime_validation_completed": False})
    print("build_completed_requires_artifact_validation", flush=True)
    return run


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-tools", type=Path, default=Path("E:/Conda/tools/pyinstaller/6.22.3"))
    args = parser.parse_args()
    try:
        build(args.build_tools, ROOT/"outputs/releases")
        return 0
    except Exception as error:
        print("not_completed="+(str(error) if isinstance(error, ValueError) else type(error).__name__), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
