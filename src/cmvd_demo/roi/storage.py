"""Immutable local ROI snapshots; no source writes, path lists or raw UIDs."""

import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import uuid

from .model import RoiError,validate_document,document_hash


def load_document(path, expected_binding):
    path = Path(path)
    if path.stat().st_size > 4*1024**2:
        raise RoiError("roi_file_too_large")
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        return validate_document(document,expected_binding)
    except RoiError:
        raise
    except (ValueError,TypeError,KeyError):
        raise RoiError("roi_invalid_json") from None


def save_snapshot(output_root, document, result=None):
    validate_document(document)
    if result is not None and result.get("document_sha256") != document_hash(document):
        raise RoiError("roi_stale_curve_result")
    serialized = json.dumps(document,ensure_ascii=True,allow_nan=False,sort_keys=True,separators=(",",":"))
    if len(serialized.encode("utf-8")) > 4*1024**2:
        raise RoiError("roi_file_too_large")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    output = Path(output_root)/run_id
    output.mkdir(parents=True,exist_ok=False)
    def write(name,data):
        with (output/name).open("x",encoding="utf-8") as stream:
            json.dump(data,stream,ensure_ascii=True,allow_nan=False,indent=2)
    with (output/"roi.json").open("x",encoding="utf-8",newline="") as stream:
        stream.write(serialized)
    if result is not None:
        write("curves.json",result)
        fields = ["role","slice_index","time_index","time_s","frame_duration_s","units","pixel_count","area_mm2","mean","std","min","max"]
        with (output/"curves.csv").open("x",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for curve in result["curves"]:
                for t in range(len(curve["time_s"])):
                    row = {key:curve[key] for key in ("role","slice_index","units")}
                    row.update({key:curve[key][t] for key in ("time_s","pixel_count","area_mm2","mean","std","min","max")})
                    row.update(time_index=t,frame_duration_s="" if curve["frame_duration_s"] is None else curve["frame_duration_s"][t])
                    writer.writerow(row)
    write("bundle.json",{"completed":True,"schema_version":1,"document_sha256":document_hash(document),
                         "curves_included":result is not None,"clinical_validated":False})
    return output
