"""Pixel-centre polygons with optional endocardial exclusion and frame edits.

Coordinates are [row,column], where integer values denote pixel centres.
No interpolation between overrides; unedited frames use the reference contour.
"""

import copy
import hashlib
import json

import numpy as np

from cmvd_demo.io import DicomReadError


class RoiError(DicomReadError):
    pass


def document_hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def binding_for(grid):
    if hasattr(grid, "cells"):
        source = [(h.study_uid, h.series_uid, h.sop_uid, h.fingerprint, h.slope, h.intercept,
                   h.units, h.decay_correction, h.corrected_image, h.half_life_s,
                   h.injected_dose_bq, h.injection_start) for row in grid.cells for h in row]
        units = sorted({h.units or "MR_SIGNAL_AU" for row in grid.cells for h in row})
    else:
        if not getattr(grid,"synthetic_id",""):
            raise RoiError("roi_source_identity_required")
        source, units = ["synthetic",grid.synthetic_id], ["SYNTHETIC_AU"]
    return {"source_signature": document_hash(source), "shape": list(grid.shape), "kind": grid.kind,
            "pixel_spacing_mm": list(grid.pixel_spacing_mm), "orientation_lps": list(grid.orientation_lps),
            "origins_lps_mm": [grid.pixel_to_lps(z, 0, 0).tolist() for z in range(grid.shape[1])],
            "timing_basis": grid.timing_basis,
            "times_s_by_slice": None if grid.times_s is None else grid.times_s.tolist(),
            "frame_durations_s_by_slice": None if grid.frame_durations_s is None else grid.frame_durations_s.tolist(),
            "units": units}


def new_document(grid):
    return {"schema_version": 1, "purpose":"research_draft", "binding": binding_for(grid), "tracks": [],
            "reviews": {"orientation_and_coverage": False, "motion_and_contours": False}}


def _fail_unless(condition, code):
    if not condition:
        raise RoiError(code)


def _cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _intersects(a, b, c, d):
    ab, cd = b-a, d-c
    u, v = _cross(ab, c-a), _cross(ab, d-a)
    w, x = _cross(cd, a-c), _cross(cd, b-c)
    if u*v < -1e-12 and w*x < -1e-12:
        return True
    for cross, point, lo, hi in ((u,c,a,b),(v,d,a,b),(w,a,c,d),(x,b,c,d)):
        if abs(cross) <= 1e-9 and np.all(point >= np.minimum(lo,hi)-1e-9) and np.all(point <= np.maximum(lo,hi)+1e-9):
            return True
    return False


def _polygon(vertices, shape):
    try:
        p = np.asarray(vertices, dtype=np.float64)
    except (ValueError, TypeError):
        raise RoiError("roi_invalid_vertices") from None
    _fail_unless(p.ndim == 2 and p.shape[1:] == (2,) and 3 <= len(p) <= 128 and np.isfinite(p).all(), "roi_invalid_vertices")
    _fail_unless(np.all(p >= -.5) and np.all(p[:,0] <= shape[0]-.5) and np.all(p[:,1] <= shape[1]-.5), "roi_outside_image")
    _fail_unless(len(np.unique(p, axis=0)) == len(p), "roi_duplicate_vertex")
    area = abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2
    _fail_unless(area > 1e-6, "roi_zero_area")
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if j == i+1 or (i == 0 and j == len(p)-1):
                continue
            _fail_unless(not _intersects(p[i],p[(i+1)%len(p)],p[j],p[(j+1)%len(p)]), "roi_self_intersection")
    return p


def _inside(row, column, polygon):
    shape = np.broadcast_shapes(np.shape(row), np.shape(column))
    inside, boundary = np.zeros(shape,dtype=bool), np.zeros(shape,dtype=bool)
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        dr, dc = b-a
        cross = (row-a[0])*dc-(column-a[1])*dr
        boundary |= ((np.abs(cross) <= 1e-9) & (row >= min(a[0],b[0])-1e-9) &
                     (row <= max(a[0],b[0])+1e-9) & (column >= min(a[1],b[1])-1e-9) &
                     (column <= max(a[1],b[1])+1e-9))
        if abs(dc) > 1e-12:
            inside ^= ((a[1] > column) != (b[1] > column)) & (row < a[0]+dr*(column-a[1])/dc)
    return inside | boundary


def contour_mask(contour, shape):
    _fail_unless(isinstance(contour,dict) and set(contour) == {"outer_rc","inner_rc"}, "roi_invalid_contour")
    outer = _polygon(contour["outer_rc"],shape)
    row, col = np.ogrid[:shape[0],:shape[1]]
    mask = _inside(row,col,outer)
    if contour["inner_rc"] is not None:
        inner = _polygon(contour["inner_rc"],shape)
        _fail_unless(_inside(inner[:,0],inner[:,1],outer).all(), "roi_hole_outside_outer")
        for a,b in zip(outer,np.roll(outer,-1,axis=0)):
            for c,d in zip(inner,np.roll(inner,-1,axis=0)):
                _fail_unless(not _intersects(a,b,c,d), "roi_hole_touches_outer")
        mask &= ~_inside(row,col,inner)
    _fail_unless(mask.any(), "roi_empty_mask")
    return mask


def validate_document(document, expected_binding=None):
    _fail_unless(isinstance(document,dict) and set(document)=={"schema_version","purpose","binding","tracks","reviews"}, "roi_invalid_document")
    _fail_unless(document["schema_version"] == 1, "roi_schema_version")
    _fail_unless(document["purpose"] in ("research_draft","engineering_fixture"), "roi_invalid_purpose")
    binding = document["binding"]
    keys = {"source_signature","shape","kind","pixel_spacing_mm","orientation_lps","origins_lps_mm",
            "timing_basis","times_s_by_slice","frame_durations_s_by_slice","units"}
    _fail_unless(isinstance(binding,dict) and set(binding)==keys and isinstance(binding.get("shape"),list)
                 and len(binding["shape"])==4, "roi_invalid_binding")
    signature = binding["source_signature"]
    _fail_unless(isinstance(signature,str) and len(signature)==64 and all(c in "0123456789abcdef" for c in signature), "roi_invalid_signature")
    _fail_unless(expected_binding is None or binding == expected_binding, "roi_source_binding_mismatch")
    shape = binding["shape"]
    _fail_unless(all(type(v) is int and v > 0 for v in shape) and shape[2]*shape[3] <= 4*1024**2, "roi_invalid_shape")
    reviews = document["reviews"]
    _fail_unless(isinstance(reviews,dict) and set(reviews)=={"orientation_and_coverage","motion_and_contours"}
                 and all(type(v) is bool for v in reviews.values()), "roi_invalid_reviews")
    tracks = document["tracks"]
    _fail_unless(isinstance(tracks,list) and len(tracks)<=1024, "roi_invalid_tracks")
    seen = set()
    for track in tracks:
        _fail_unless(isinstance(track,dict) and set(track)=={"role","slice_index","reference_frame","contour","overrides"}, "roi_invalid_track")
        role,z,t = track["role"],track["slice_index"],track["reference_frame"]
        _fail_unless(role in ("myocardium","blood_pool") and type(z) is int and 0<=z<shape[1]
                     and type(t) is int and 0<=t<shape[0], "roi_invalid_track_index")
        _fail_unless((role,z) not in seen, "roi_duplicate_track")
        seen.add((role,z))
        contour_mask(track["contour"],shape[-2:])
        overrides = track["overrides"]
        _fail_unless(isinstance(overrides,list) and len(overrides)<=shape[0], "roi_invalid_overrides")
        frames = set()
        for item in overrides:
            _fail_unless(isinstance(item,dict) and set(item)=={"frame_index","contour"}, "roi_invalid_override")
            frame = item["frame_index"]
            _fail_unless(type(frame) is int and 0<=frame<shape[0] and frame not in frames, "roi_duplicate_or_invalid_frame")
            frames.add(frame)
            contour_mask(item["contour"],shape[-2:])
    return document


def set_track(document, role, z, t, contour, *, frame_only=False):
    updated = copy.deepcopy(document)
    track = next((r for r in updated["tracks"] if r["role"]==role and r["slice_index"]==z),None)
    if frame_only:
        _fail_unless(track is not None, "roi_reference_required")
        track["overrides"] = [r for r in track["overrides"] if r["frame_index"]!=t]
        track["overrides"].append({"frame_index":t,"contour":copy.deepcopy(contour)})
    elif track is None:
        updated["tracks"].append({"role":role,"slice_index":z,"reference_frame":t,
                                  "contour":copy.deepcopy(contour),"overrides":[]})
    else:
        track["contour"],track["reference_frame"] = copy.deepcopy(contour),t
    updated["reviews"] = {key:False for key in updated["reviews"]}
    return validate_document(updated)


def contour_at(track, frame):
    return next((r["contour"] for r in track["overrides"] if r["frame_index"]==frame),track["contour"])
