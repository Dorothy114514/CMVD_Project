"""Original per-plane signal means, no temporal interpolation or MBF model."""

import numpy as np
from collections import OrderedDict

from cmvd_demo.io import ReadCancelled
from .model import RoiError, binding_for, validate_document, document_hash, contour_mask, contour_at


def extract_curves(grid, document, slice_index, *, read_plane=None, progress=None, cancel=None):
    def check_cancel():
        if cancel is not None and cancel():
            raise ReadCancelled()
    check_cancel()
    binding = binding_for(grid)
    validate_document(document,binding)
    tracks = [r for r in document["tracks"] if r["slice_index"]==slice_index]
    if not tracks:
        raise RoiError("roi_no_tracks_on_plane")
    if grid.times_s is None or grid.shape[0]<2:
        raise RoiError("roi_time_series_required")
    times = np.asarray(grid.times_s[slice_index],dtype=float)
    if len(times)!=grid.shape[0] or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise RoiError("roi_invalid_time_axis")
    if len(binding["units"])!=1:
        raise RoiError("roi_inconsistent_units")
    if grid.frame_durations_s is not None:
        durations = np.asarray(grid.frame_durations_s[slice_index],dtype=float)
        if len(durations)!=len(times) or not np.isfinite(durations).all() or np.any(durations<=0):
            raise RoiError("roi_invalid_frame_duration")
    curves, masks, mask_bytes = [], OrderedDict(), 0
    area = float(np.prod(grid.pixel_spacing_mm))
    if not np.isfinite(area) or area<=0:
        raise RoiError("roi_invalid_pixel_area")
    for track in tracks:
        curves.append({"role":track["role"],"slice_index":slice_index,"time_s":times.tolist(),
                       "frame_duration_s":None if grid.frame_durations_s is None else grid.frame_durations_s[slice_index].tolist(),
                       "units":binding["units"][0],"mean":[],"std":[],"min":[],"max":[],"pixel_count":[],"area_mm2":[]})
    for t in range(grid.shape[0]):
        check_cancel()
        pixels = np.asarray(read_plane(t,slice_index) if read_plane else grid.read_plane(t,slice_index))
        if pixels.shape!=tuple(grid.shape[-2:]) or not np.isfinite(pixels).all():
            raise RoiError("roi_invalid_pixels")
        frame_masks = []
        for track in tracks:
            contour = contour_at(track,t)
            key = document_hash(contour)
            if key not in masks:
                masks[key] = contour_mask(contour,grid.shape[-2:])
                mask_bytes += masks[key].nbytes
                while mask_bytes > 16*1024**2 and len(masks)>1:
                    _,removed = masks.popitem(last=False)
                    mask_bytes -= removed.nbytes
            masks.move_to_end(key)
            frame_masks.append(masks[key])
        if len(frame_masks)==2 and np.any(frame_masks[0]&frame_masks[1]):
            raise RoiError("roi_myocardium_blood_overlap")
        for curve,mask in zip(curves,frame_masks):
            values = pixels[mask].astype(np.float64)
            for name,value in (("mean",values.mean()),("std",values.std()),("min",values.min()),("max",values.max())):
                curve[name].append(float(value))
            curve["pixel_count"].append(int(mask.sum()))
            curve["area_mm2"].append(float(mask.sum()*area))
        if progress:
            progress("roi_curves",t+1,grid.shape[0])
    check_cancel()
    warnings = []
    if document["purpose"]=="engineering_fixture":
        warnings.append("engineering_fixture_not_anatomical_annotation")
    if len(tracks)!=2:
        warnings.append("myocardium_blood_pair_incomplete")
    for key in ("orientation_and_coverage","motion_and_contours"):
        if not document["reviews"][key]:
            warnings.append(key+"_not_reviewed")
    if grid.kind not in ("pet_dynamic","mr_perfusion"):
        warnings.append("not_a_dynamic_perfusion_branch")
    if grid.kind=="pet_dynamic" and binding["units"]!=["BQML"]:
        warnings.append("pet_units_require_review")
    return {"schema_version":1,"method":"pixel_centre_polygon_mean_v1","document_sha256":document_hash(document),
            "slice_index":slice_index,"timing_basis":grid.timing_basis,"curves":curves,
            "qc":{"warnings":warnings,"feature_input_ready":not warnings,"clinical_validated":False,
                  "motion_method":"fixed_reference_with_explicit_manual_frame_overrides",
                  "temporal_interpolation":False,"image_resampling":False}}
