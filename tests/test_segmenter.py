import numpy as np

from pipeline.live.segmenter import EnergySegmenter

SR = 16000


def tone(seconds, db=-30.0, f=220.0):
    t = np.arange(int(seconds * SR)) / SR
    amp = np.sqrt(2) * 10 ** (db / 20)  # sine RMS = amp / sqrt(2)
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds, db=-75.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 10 ** (db / 20)).astype(np.float32)


def run(seg, x, chunk=320):
    out = []
    for i in range(0, len(x), chunk):
        out += seg.push(x[i:i + chunk])
    return out + seg.flush()


def test_two_bursts_two_segments_with_boundaries():
    x = np.concatenate([silence(0.5), tone(1.0), silence(0.8), tone(0.6), silence(0.5)])
    segs = run(EnergySegmenter(), x)
    assert len(segs) == 2
    assert abs(segs[0].start - int(0.5 * SR)) <= 320 and abs(segs[0].end - int(1.5 * SR)) <= 320
    assert abs(segs[1].start - int(2.3 * SR)) <= 320 and abs(segs[1].end - int(2.9 * SR)) <= 320
    # the end is only decided after the hangover (0.3 s) of silence
    assert abs((segs[0].closed_at - segs[0].end) - int(0.3 * SR)) <= 320
    np.testing.assert_array_equal(segs[0].audio, x[segs[0].start:segs[0].end])


def test_short_pause_inside_utterance_is_not_a_split():
    x = np.concatenate([silence(0.3), tone(0.5), silence(0.2), tone(0.5), silence(0.6)])
    assert len(run(EnergySegmenter(hangover_s=0.3), x)) == 1


def test_short_click_is_dropped():
    x = np.concatenate([silence(0.3), tone(0.06), silence(0.6)])
    assert run(EnergySegmenter(min_speech_s=0.2), x) == []


def test_quiet_speech_above_threshold_is_kept():
    x = np.concatenate([silence(0.3), tone(0.8, db=-50.0), silence(0.6)])
    assert len(run(EnergySegmenter(threshold_db=-55.0), x)) == 1


def test_max_segment_forces_close():
    x = np.concatenate([tone(2.5), silence(0.5)])
    segs = run(EnergySegmenter(max_segment_s=1.0), x)
    assert segs[0].forced and len(segs[0].audio) == SR


def test_chunk_size_does_not_change_result():
    x = np.concatenate([silence(0.4), tone(0.7), silence(0.7), tone(0.9), silence(0.4)])
    ref = [(s.start, s.end) for s in run(EnergySegmenter(), x, chunk=320)]
    for chunk in (1, 100, 333, 1600, 16000):
        assert [(s.start, s.end) for s in run(EnergySegmenter(), x, chunk=chunk)] == ref


def test_open_segment_closed_by_flush():
    x = np.concatenate([silence(0.2), tone(0.5)])
    segs = run(EnergySegmenter(), x)
    assert len(segs) == 1 and not segs[0].forced


# ---- adaptive noise floor (live microphone mode) ----

def test_adaptive_off_by_default_keeps_fixed_threshold():
    s = EnergySegmenter(threshold_db=-55.0)
    run(s, np.concatenate([silence(1.0, db=-40.0), tone(0.5)]))
    assert s.current_threshold_db() == -55.0


def test_fixed_threshold_fails_in_noisy_room():
    # room noise at -40 dBFS is above the fixed -55 threshold: everything looks like speech
    x = np.concatenate([silence(1.0, db=-40.0), tone(0.8, db=-20.0), silence(1.0, db=-40.0), tone(0.8, db=-20.0),
                        silence(1.0, db=-40.0)])
    assert len(run(EnergySegmenter(threshold_db=-55.0), x)) == 1  # one giant segment


def test_adaptive_finds_utterances_in_noisy_room():
    x = np.concatenate([silence(1.0, db=-40.0), tone(0.8, db=-20.0), silence(1.0, db=-40.0, seed=1),
                        tone(0.8, db=-20.0), silence(1.0, db=-40.0, seed=2)])
    s = EnergySegmenter(threshold_db=-55.0, adaptive=True, margin_db=12.0)
    assert abs(s.calibrate(silence(1.0, db=-40.0, seed=3)) - (-40.0)) < 1.0
    segs = run(s, x)
    assert len(segs) == 2
    assert abs(segs[0].start - SR) <= 320 and abs(segs[0].end - int(1.8 * SR)) <= 320


def test_adaptive_floor_tracks_slow_noise_change_and_is_capped():
    s = EnergySegmenter(threshold_db=-55.0, adaptive=True, floor_window_s=1.0, max_floor_db=-30.0)
    s.calibrate(silence(0.5, db=-60.0))
    run(s, silence(3.0, db=-45.0))          # room gets louder, no speech
    assert abs(s.noise_floor_db - (-45.0)) < 2.0
    run(s, silence(3.0, db=-10.0))          # very loud: floor must stop at the cap
    assert s.noise_floor_db <= -30.0 + 1e-9
