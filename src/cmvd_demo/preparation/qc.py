"""Safe branch evidence; counts do not establish clinical suitability."""

from collections import Counter


def review_status(inspection, branches, duplicates):
    mainline_clear = inspection["inspection_passed"] and not branches["zero_slope_objects_all_indexed_series"]
    return {"mainline_checks_passed_in_transient_view": bool(mainline_clear),
            "engineering_clear_in_transient_view": bool(mainline_clear and not duplicates["conflicting_groups"]),
            "clinical_or_training_ready": False}


def branch_inventory(scan, inspection):
    passed = {g["grid_token"] for g in inspection["grids"]}
    failures = {g["grid_token"]: g["code"] for g in inspection["grid_failures"]}
    rows = []
    for series in scan.series:
        headers = series.objects
        kind = series.kind
        time_field, declared_field = {
            "pet_dynamic": ("frame_reference_s", "declared_time_slices"),
            "pet_gated": ("trigger_s", "declared_time_slots"),
            "mr_cine": ("trigger_s", "declared_cardiac_images"),
            "mr_perfusion": ("acquisition_time", "declared_temporal_positions"),
        }.get(kind, (None, None))
        actual_times = {getattr(h, time_field) for h in headers} if time_field else set()
        declared = sorted({getattr(h, declared_field) for h in headers
                           if getattr(h, declared_field) is not None}) if declared_field else []
        zero_slope = sum(h.slope == 0 for h in headers)
        code = failures.get(series.token)
        rows.append({"series_token": series.token, "kind": kind,
                     "condition_candidate": series.condition_candidate,
                     "objects_in_selected_index": len(headers),
                     "unique_sop_count": len({h.sop_uid for h in headers}),
                     "observed_time_count": len(actual_times-{None}) if time_field else None,
                     "missing_time_values": sum(getattr(h, time_field) is None for h in headers) if time_field else None,
                     "declared_time_counts": declared,
                     "zero_rescale_slope_objects": zero_slope,
                     "grid_check": "passed_first_last_only" if series.token in passed else (code or "not_mainline"),
                     "quantitative_header_blocked": bool(zero_slope),
                     "anatomical_or_clinical_reviewed": False})
    return {"series": rows, "series_by_kind": dict(sorted(Counter(r["kind"] for r in rows).items())),
            "zero_slope_objects_all_indexed_series": sum(r["zero_rescale_slope_objects"] for r in rows),
            "zero_slope_series_all_indexed_series": sum(bool(r["zero_rescale_slope_objects"]) for r in rows),
            "note": "All indexed slope fields checked; pixel reads remain first/last only. Conditions are unreviewed candidates."}
