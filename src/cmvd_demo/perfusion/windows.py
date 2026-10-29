"""Frame selection on actual sample times, without interpolation or signal fitting."""
import numpy as np

from .core import FeatureError


def sample_times(times):
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or len(values)<2 or not np.isfinite(values).all() or np.any(np.diff(values)<=0):
        raise FeatureError('feature_invalid_curve')
    return values


def nearest_frame(times, position):
    times = sample_times(times)
    if not np.isfinite(position):
        raise FeatureError('feature_invalid_curve')
    # Stable ties select the earlier acquired sample.
    return int(np.argmin(np.abs(times-float(position))))


def span_frames(times, span):
    times = sample_times(times)
    a,b = sorted(nearest_frame(times, float(x)) for x in span)
    if a==b:
        raise FeatureError('feature_window_two_frames_required')
    return a,b


def analysis_frames(n, baseline_start, baseline_frames, start, end):
    """UI inputs are one-based; return inclusive zero-based sample intervals."""
    if not (all(type(x) is int for x in (n,baseline_start,baseline_frames,start,end))
            and 1<=baseline_start and baseline_frames>=2
            and baseline_start+baseline_frames-1<start<end<=n):
        raise FeatureError('feature_invalid_analysis_window')
    return baseline_start-1,baseline_start+baseline_frames-2,start-1,end-1
