"""Safe business failure logging: no exception strings, paths, UIDs or patient values."""
from datetime import datetime, timezone
import errno
import json
import os
import re
from threading import Lock
import uuid
from cmvd_demo.paths import workspace_root
from cmvd_demo.io import DicomReadError

_lock = Lock()
_run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]


def error_code(error):
    if isinstance(error, DicomReadError):
        return error.code
    if isinstance(error, OSError) and (error.errno == errno.ENOSPC or getattr(error, 'winerror', None) == 112):
        return 'output_disk_full'
    if isinstance(error, PermissionError):
        return 'filesystem_permission_denied'
    if isinstance(error, (FileExistsError, NotADirectoryError, IsADirectoryError)):
        return 'output_path_unavailable'
    if isinstance(error, FileNotFoundError):
        return 'file_unavailable'
    return 'unexpected_'+type(error).__name__


def record_error(code, stage):
    clean = lambda value: value if isinstance(value,str) and re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,95}',value) else 'unclassified'
    value = {'runtime_utc':datetime.now(timezone.utc).isoformat(), 'stage':clean(stage), 'code':clean(code)}
    try:
        with _lock:
            root=workspace_root()/'outputs/runtime_logs'
            root.mkdir(parents=True,exist_ok=True)
            with (root/(_run+'-events.jsonl')).open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(value,sort_keys=True)+'\n')
        return True
    except OSError:
        # A full/unwritable workspace must not turn error handling into another crash.
        return False
