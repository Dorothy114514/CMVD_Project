"""Verify local CPU/Qt dependencies with synthetic data; no clinical model."""

from importlib.metadata import version
from io import BytesIO
import json
import platform
import sys


def main():
    import numpy as np
    import pandas as pd
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from PySide6.QtCore import qVersion
    from PySide6.QtGui import QImage
    import pyqtgraph

    # Qt DLL/import check without opening a window during environment setup.
    image = QImage(2, 2, QImage.Format.Format_RGB32)
    image.fill(0)
    if image.isNull():
        raise RuntimeError("Qt image smoke check failed")

    # Small CPU fit verifies compiled numerical dependencies. These are arbitrary numbers.
    table = pd.DataFrame({"x1": [0, 1, 0, 1, 2, 3, 2, 3], "x2": [1, 0, 2, 1, 3, 2, 4, 3]})
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    model.fit(table, targets)
    probabilities = model.predict_proba(table)
    if probabilities.shape != (8, 2) or not np.isfinite(probabilities).all():
        raise RuntimeError("Numerical dependency smoke check failed")

    # Round trip a synthetic DICOM in memory without reading patient files or pixels.
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientID = "SYNTH-ENV-CHECK"
    dataset.PatientName = "SYNTHETIC"
    dataset.Modality = "OT"
    buffer = BytesIO()
    pydicom.dcmwrite(buffer, dataset, enforce_file_format=True)
    buffer.seek(0)
    restored = pydicom.dcmread(buffer)
    if restored.PatientID != "SYNTH-ENV-CHECK":
        raise RuntimeError("DICOM dependency smoke check failed")

    print(json.dumps({
        "status": "passed",
        "scope": "synthetic CPU fit, in-memory DICOM and Qt image/import checks; no GUI acceptance",
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "executable": sys.executable,
        "dependencies": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "pydicom",
            "PySide6", "PySide6_Addons", "PySide6_Essentials", "shiboken6", "pyqtgraph", "colorama"
        )},
        "qt_runtime": qVersion(),
        "patient_data_accessed": False,
        "clinical_model_trained": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
