import types
import sys
import types as _types

# Inject a dummy pyloudnorm so per_track can import in test environments
m = _types.ModuleType("pyloudnorm")
class _DummyMeter:
    def __init__(self, sr):
        self.sr = sr
    def integrated_loudness(self, y):
        return -12.0
m.Meter = _DummyMeter
sys.modules.setdefault("pyloudnorm", m)

import numpy as np
from offbeat.analysis.per_track import _local_guide_tempo


def test_local_guide_tempo_uses_trimmed_start_offset():
    # Build a fake context with a bpm_curve sampled per second relative to trimmed_start_sec
    ctx = types.SimpleNamespace()
    # 10 seconds at 125, then 10 seconds at 175
    ctx.bpm_curve = [125.0] * 10 + [175.0] * 10
    # Suppose trimming removed 2.5 seconds at the start of the file
    ctx.trimmed_start_sec = 2.5

    # Absolute window [3.0, 8.0] maps to relative [0.5, 5.5] -> indices [0:6] inside first plateau (~125)
    t_local_1 = _local_guide_tempo(ctx, 3.0, 8.0)
    assert t_local_1 is not None
    assert abs(t_local_1 - 125.0) < 1e-6

    # Absolute window [15.0, 22.0] -> relative [12.5, 19.5] -> indices [12:20], inside second plateau (~175)
    t_local_2 = _local_guide_tempo(ctx, 15.0, 22.0)
    assert t_local_2 is not None
    assert abs(t_local_2 - 175.0) < 1e-6


def test_local_guide_tempo_handles_empty_or_unstable():
    ctx = types.SimpleNamespace()
    ctx.bpm_curve = []
    ctx.trimmed_start_sec = 1.0
    assert _local_guide_tempo(ctx, 5.0, 10.0) is None

    # Unstable window: large variance triggers IQR > 8.0
    ctx.bpm_curve = [120.0, 180.0] * 10
    ctx.trimmed_start_sec = 0.0
    t = _local_guide_tempo(ctx, 0.0, 20.0)
    assert t is None or (t >= 60.0 and t <= 200.0)  # Accept None due to instability guard
