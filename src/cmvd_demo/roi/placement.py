"""Manual placement geometry in pixel-centre RC coordinates; no image inference."""
import copy

import numpy as np

from .model import RoiError, contour_at, contour_mask


def ellipse_vertices(start, end):
    bounds = np.asarray([start, end], dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(np.abs(bounds[1]-bounds[0]) < 2):
        raise RoiError("roi_drawing_too_small")
    center, radius = bounds.mean(axis=0), np.abs(bounds[1]-bounds[0])/2
    angles = np.arange(16)*2*np.pi/16
    return (center+np.column_stack([np.sin(angles), np.cos(angles)])*radius).tolist()


def placed_contour(document, role, z, t, vertices, boundary, include_inner):
    track = next((r for r in document['tracks'] if r['role']==role and r['slice_index']==z), None)
    previous = copy.deepcopy(contour_at(track, t)) if track else None
    if boundary == 'inner' and role == 'myocardium':
        if previous is None:
            raise RoiError('roi_draw_outer_first')
        contour = dict(previous, inner_rc=copy.deepcopy(vertices))
    else:
        inner = None
        if role == 'myocardium':
            if boundary == 'outer' and previous:
                inner = previous['inner_rc']
            elif include_inner:
                outer = np.asarray(vertices, dtype=float)
                # An explicit manual starting shape, never image segmentation.
                inner = (outer.mean(axis=0)+(outer-outer.mean(axis=0))*.55).tolist()
        contour = {'outer_rc':copy.deepcopy(vertices), 'inner_rc':inner}
    contour_mask(contour, document['binding']['shape'][-2:])
    return contour
