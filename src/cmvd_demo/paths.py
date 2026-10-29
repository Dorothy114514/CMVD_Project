"""Source and frozen desktop paths. No patient directory is a resource."""

import os
from pathlib import Path
import shutil
import sys


RESOURCE_FILES = (
    "configs/cohort_contract.json", "configs/cohort_project.template.json",
    "configs/cohort_project.v2.template.json", "configs/clinical_candidate_fields.json",
    "examples/templates/clinical_features.csv", "examples/templates/diagnosis_labels.csv",
    "examples/templates/patient_associations.csv", "examples/templates/patient_partitions.csv",
    "examples/templates/perfusion_registry.csv", "examples/templates/clinical_missingness.csv",
    "docs/USAGE_CHECKLIST.md",
)
_workspace_override = None


def frozen():
    return bool(getattr(sys, "frozen", False))


def resource_root():
    return Path(sys._MEIPASS)/"resources" if frozen() else Path(__file__).resolve().parents[2]


def workspace_root():
    if _workspace_override is not None:
        return _workspace_override
    if frozen():
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home()/".local/share"
        return base/"CMVDResearchBrowser"/"workspace-v1"
    return resource_root()


def configure_workspace(path=None):
    """Copy only missing public templates; never overwrite any existing file."""
    global _workspace_override
    _workspace_override = Path(path).resolve() if path is not None else None
    workspace = workspace_root()
    if frozen() or path is not None:
        workspace.mkdir(parents=True, exist_ok=True)
        for relative in RESOURCE_FILES:
            source, destination = resource_root()/relative, workspace/relative
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with source.open("rb") as src, destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                except FileExistsError:
                    pass  # Another local instance published the template first.
    return workspace
