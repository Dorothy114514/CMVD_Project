"""Explicit, transient duplicate-export selection; never changes source files.

The default reader remains strict. Only a whole repeated SOP group whose files
all have the same full SHA-256 may be collapsed in this engineering QC view.
No saved selection is accepted as a future reading permission.
"""

from collections import defaultdict
from dataclasses import dataclass, field, replace
import hashlib

from cmvd_demo.io.dicom import DicomReadError, ReadCancelled


def _cancel(cancel):
    if cancel is not None and cancel():
        raise ReadCancelled()


def _unchanged(header):
    try:
        stat = header.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != header.fingerprint:
            raise DicomReadError("duplicate_source_changed", header.file_token)
    except OSError:
        raise DicomReadError("duplicate_source_unavailable", header.file_token) from None


def _digest(header, cancel):
    _cancel(cancel)
    _unchanged(header)
    digest = hashlib.sha256()
    try:
        with header.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4*1024**2), b""):
                _cancel(cancel)
                digest.update(chunk)
    except OSError:
        raise DicomReadError("duplicate_source_unavailable", header.file_token) from None
    _unchanged(header)
    return digest.hexdigest()


@dataclass(frozen=True)
class DuplicateSelection:
    scan: object = field(repr=False)
    groups: tuple = field(repr=False)

    def safe_summary(self):
        return {"policy": "whole_sop_group_full_sha256_equal_v1",
                "duplicate_groups": len(self.groups),
                "identical_groups": sum(g["collapsed"] for g in self.groups),
                "conflicting_groups": sum(not g["collapsed"] for g in self.groups),
                "hashed_files": sum(len(g["members"]) for g in self.groups),
                "excluded_from_transient_index": sum(len(g["members"])-1 for g in self.groups if g["collapsed"]),
                "source_files_removed_or_modified": 0,
                "clinical_reviewed": False,
                "persistent_selection_reuse_supported": False}


def select_identical_sops(scan, *, progress=None, cancel=None):
    """Keep conflicts intact, retain scan issues, and fail on changed sources.

    Returns an ephemeral index and restricted provenance (paths, fingerprints,
    hashes). Callers must keep provenance out of public logs. Patient identity,
    labels, geometry, time, scaling and nonduplicate objects are not inferred.
    """
    _cancel(cancel)
    jobs = []
    for series in scan.series:
        groups = defaultdict(list)
        for header in series.objects:
            groups[(header.study_uid, header.series_uid, header.sop_uid)].append(header)
        for members in groups.values():
            if len(members) > 1:
                jobs.append((series.token, sorted(members, key=lambda h: (h.file_token, str(h.path)))))
    jobs.sort(key=lambda job: (job[0], job[1][0].file_token))
    total = sum(len(members) for _, members in jobs)
    done, excluded, records = 0, set(), []
    for series_token, members in jobs:
        evidence = []
        for header in members:
            digest = _digest(header, cancel)
            evidence.append({"file_token": header.file_token, "path": str(header.path),
                             "size": header.fingerprint[0], "mtime_ns": header.fingerprint[1],
                             "sha256": digest})
            done += 1
            if progress is not None:
                progress("duplicate_hash", done, total)
        identical = len({m["sha256"] for m in evidence}) == 1
        if identical:
            excluded.update(h.path for h in members[1:])
        records.append({"series_token": series_token, "group_token": members[0].file_token,
                        "collapsed": identical, "kept_file_token": members[0].file_token if identical else None,
                        "members": evidence})
    # Catch mutations of an earlier group while later groups were being hashed.
    for _, members in jobs:
        for header in members:
            _cancel(cancel)
            _unchanged(header)
    selected = tuple(replace(s, objects=tuple(h for h in s.objects if h.path not in excluded)) for s in scan.series)
    _cancel(cancel)
    return DuplicateSelection(replace(scan, series=selected), tuple(records))
