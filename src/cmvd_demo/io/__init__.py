"""Read-only DICOM indexing and lazy spatial/temporal assembly."""

from .dicom import (
    DicomReadError, ReadCancelled, ScanResult, Series, ImageGrid,
    scan_case, build_grid, build_cine_stack, read_pixels,
)

__all__ = [
    "DicomReadError", "ReadCancelled", "ScanResult", "Series", "ImageGrid",
    "scan_case", "build_grid", "build_cine_stack", "read_pixels",
]
