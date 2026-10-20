"""Browser choices, bounded plane cache and an explicitly synthetic phantom."""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np

from cmvd_demo.io import DicomReadError, ReadCancelled, scan_case, build_grid, build_cine_stack

KINDS = {"pet_dynamic": "PET 动态", "pet_gated": "PET 门控", "pet_static": "PET 静态",
         "mr_perfusion": "MR 灌注", "mr_cine": "MR cine", "cine_stack": "短轴 cine 组合",
         "mr_other": "其他 MR（仅索引）", "secondary_capture": "截图（仅索引）",
         "unsupported_sop_class": "其他对象（仅索引）", "unsupported_multiframe": "多帧（暂不支持）"}
SUPPORTED = {"pet_dynamic", "pet_gated", "pet_static", "mr_perfusion", "mr_cine", "cine_stack"}
CONDITIONS = {"rest": "静息候选", "stress": "负荷候选", "unknown": "采集状态待核对"}


@dataclass(frozen=True)
class Choice:
    key: str
    label: str
    kind: str
    condition: str
    count: int
    source: object = field(repr=False)


def choices_from_scan(scan):
    choices, sax = [], defaultdict(list)
    for series in scan.series:
        first = series.objects[0]
        label = f"S{first.series_number if first.series_number is not None else '?'} · {KINDS.get(series.kind, series.kind)}"
        choices.append(Choice(series.token, label, series.kind, series.condition_candidate, len(series.objects), series))
        if series.kind == "mr_cine" and "sax" in series.tokens:
            key = (first.study_uid, first.frame_of_reference_uid,
                   tuple(round(v, 5) for v in (first.orientation or ())))
            sax[key].append(series)
    for index, group in enumerate(sax.values(), 1):
        if len(group) > 1:
            choices.append(Choice(f"cine-stack-{index}", f"短轴 cine 组合候选 {index}", "cine_stack", "unknown",
                                  sum(len(s.objects) for s in group), tuple(group)))
    return choices


class ReaderBackend:
    def index(self, path, progress, cancel):
        scan = scan_case(path, progress=progress, cancel=cancel)
        return choices_from_scan(scan), scan.safe_summary()

    def grid(self, choice, progress, cancel):
        if cancel():
            raise ReadCancelled()
        if isinstance(choice.source, PhantomGrid):
            return choice.source
        grid = build_cine_stack(choice.source) if choice.kind == "cine_stack" else build_grid(choice.source)
        if cancel():
            raise ReadCancelled()
        return grid

    def plane(self, grid, t, z, cached, progress, cancel):
        if cancel():
            raise ReadCancelled()
        if grid.shape[-2]*grid.shape[-1]*4 > 16*1024**2:
            raise DicomReadError("desktop_plane_memory_budget_exceeded")
        if isinstance(grid, PhantomGrid):
            return cached if cached is not None else grid.read_plane(t, z)
        header = grid.cells[t][z]
        if cached is not None:
            stat = header.path.stat()
            if (stat.st_size, stat.st_mtime_ns) != header.fingerprint:
                raise DicomReadError("source_changed_since_index")
            return cached
        return grid.read_plane(t, z)


class PlaneCache:
    """Main-thread LRU; only accepted current results enter the cache."""
    def __init__(self, max_bytes=64*1024**2):
        self.max_bytes = max_bytes
        self.bytes_used = 0
        self._items = OrderedDict()

    def get(self, key):
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key, value):
        old = self._items.pop(key, None)
        if old is not None:
            self.bytes_used -= old.nbytes
        if value.nbytes > self.max_bytes:
            return
        value.setflags(write=False)
        self._items[key] = value
        self.bytes_used += value.nbytes
        while self.bytes_used > self.max_bytes:
            _, removed = self._items.popitem(last=False)
            self.bytes_used -= removed.nbytes

    def clear(self):
        self._items.clear()
        self.bytes_used = 0


class PhantomGrid:
    """Known asymmetric synthetic anatomy for UI checks; never a real case."""
    shape = (12, 5, 96, 128)
    kind = "synthetic"
    orientation_lps = (1, 0, 0, 0, 1, 0)
    pixel_spacing_mm = (2., 1.)
    timing_basis = "synthetic_relative_acquisition"
    frame_durations_s = None
    qc_warnings = ("synthetic_only",)
    photometric = "MONOCHROME2"
    synthetic_id = "SYNTHETIC_PHANTOM_V1"

    def __init__(self):
        self.times_s = np.tile(np.array([0, .7, 1.5, 2.4, 3.4, 4.5, 5.7, 7, 8.4, 9.9, 11.5, 13.2]), (5, 1))

    def read_plane(self, t, z):
        y, x = np.mgrid[:96, :128]
        radius = np.sqrt(((x-64)/1.0)**2+((y-48)*1.4)**2)
        image = 12+100*np.exp(-((radius-(22+z))/5)**2)*(1+.35*np.sin(t/12*2*np.pi))
        image[8:16, 12:24] = 250  # asymmetric top-left orientation marker
        image[72:76, 100:116] = 180
        return image.astype(np.float32)

    def pixel_to_lps(self, z, row, column):
        return np.array([column, row*2, z*10], dtype=float)


def synthetic_choices():
    return [Choice("synthetic-phantom", "合成方向与时间体模", "synthetic", "unknown", 60, PhantomGrid())]


class PerfusionPhantomGrid(PhantomGrid):
    """Known nonuniform first-pass signals, explicitly synthetic per condition."""
    shape = (7,1,96,128)
    kind = "mr_perfusion"

    def __init__(self,condition):
        self.condition = condition
        self.synthetic_id = "D4_SYNTHETIC_FIRST_PASS_"+condition
        self.times_s = np.array([[0.,1.,2.,4.,7.,8.,10.]])

    def read_plane(self,t,z):
        r,c = np.mgrid[:96,:128]
        radius = ((r-47.5)/21)**2+((c-63.5)/28)**2
        enhancement = np.array([0,0,2,10,16,12,6])[t]*(2 if self.condition=="stress" else 1)
        blood = 20+np.array([0,0,8,24,28,20,8])[t]
        pixels = np.full((96,128),5.,dtype=np.float32)
        pixels[(radius>=.30)&(radius<=1.05)] = 10+enhancement
        pixels[radius<.25] = blood
        pixels[8:16,12:24] = 250
        pixels[72:76,100:116] = 180
        return pixels


def perfusion_synthetic_choices():
    return [Choice("synthetic-perfusion-"+c,"完全合成首过 · "+("静息" if c=="rest" else "负荷"),
                   "mr_perfusion",c,7,PerfusionPhantomGrid(c)) for c in ("rest","stress")]


def direction_label(vector):
    """Major and minor LPS components, strongest first (supports oblique planes)."""
    vector = np.asarray(vector)
    return "".join(("LPS" if vector[i] >= 0 else "RAI")[i]
                   for i in np.argsort(-np.abs(vector)) if abs(vector[i]) > 1e-3)


def auto_levels(pixels):
    low, high = np.percentile(pixels, [1, 99])
    if not high > low:
        low, high = float(pixels.min()), float(pixels.max())
    if not high > low:
        high = low+1
    return float(low), float(high)
