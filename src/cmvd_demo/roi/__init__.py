"""Manual ROI geometry and original signal curves, independent of Qt."""

from .model import RoiError, binding_for, new_document, validate_document, document_hash, contour_mask, set_track
from .curves import extract_curves
from .storage import save_snapshot, load_document
