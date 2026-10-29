"""Read-only comparison rules; no registration, interpolation or clinical pairing."""
import numpy as np
from cmvd_demo.io import DicomReadError
from cmvd_demo.perfusion.windows import nearest_frame,sample_times


def relative_target(source_times,target_times,index):
    source,target=sample_times(source_times),sample_times(target_times)
    if not 0<=index<len(source):raise DicomReadError('compare_invalid_position')
    relative=float(source[index]-source[0])
    if relative<0 or relative>target[-1]-target[0]+1e-9:
        raise DicomReadError('compare_time_outside')
    return nearest_frame(target-target[0],relative)


def link_allowed(left,right):
    if left is None or right is None:return False
    return (left['kind']==right['kind'] and left['units']==right['units']
            and left['kind'] in ('mr_perfusion','pet_dynamic','synthetic')
            and left['times_s_by_slice'] is not None and right['times_s_by_slice'] is not None
            and left['shape'][0]>1 and right['shape'][0]>1
            and not any('cardiac' in b['timing_basis'] for b in (left,right)))


def plane_time(binding,t,z):
    times=binding['times_s_by_slice']
    return None if times is None else float(times[z][t])
