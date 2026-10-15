from argparse import Namespace

import numpy as np
import pytest

from wearablepermed_utils.core import (
    apply_windowing_WPM_segmented_data,
    extract_features,
    extract_features_from_stack,
    load_concat_window_stack,
    resample_signal,
)

FS = 25
WINDOW_SIZE = 250  # 10 s a 25 Hz
FREQUENCIES = [25, 20, 15, 10, 5, 2, 1, 0.5]


def synthetic_gait(duration_s=120, seed=0):
    """Marcha sintética: cadencia de 1.8 Hz en el eje vertical con la gravedad en z."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / FS)
    signal = np.zeros((t.size, 7))
    signal[:, 0] = t * 1000  # timestamp [ms]
    signal[:, 1] = 0.10 * np.sin(2 * np.pi * 0.9 * t)
    signal[:, 2] = 0.05 * np.sin(2 * np.pi * 1.8 * t + 1)
    signal[:, 3] = 1.0 + 0.30 * np.sin(2 * np.pi * 1.8 * t) + 0.08 * np.sin(2 * np.pi * 3.6 * t)
    signal[:, 4:] = 40 * np.sin(2 * np.pi * 0.9 * t[:, None] + np.arange(3))
    signal[:, 1:] += 0.02 * rng.standard_normal((t.size, 6))
    return signal


def iqr(x):
    return np.subtract(*np.percentile(x, [75, 25]))


def build_stack(tmp_path, fs_downsampling):
    npz_file = tmp_path / "PMP0000_W1_PI_segmented.npz"
    np.savez(npz_file, **{
        "CAMINAR USUAL SPEED": synthetic_gait(seed=0),
        "SENTADO LEYENDO": synthetic_gait(seed=1)[:, :] * [1, 0, 0, 1, 0, 0, 0],
        "YOGA": synthetic_gait(seed=2),
        "TAPIZ RODANTE": synthetic_gait(seed=3),
    })
    stack_file = tmp_path / f"stack_{fs_downsampling}.npz"
    args = Namespace(fs_downsampling=fs_downsampling, include_not_estructure_data=True)
    data, labels, _ = load_concat_window_stack(
        args, "PMP0000", str(npz_file), slice(1, 7), WINDOW_SIZE, save_file_name=str(stack_file)
    )
    return data, labels, stack_file


def test_resample_signal_preserves_dynamics():
    gait = synthetic_gait()[:, 1:]
    for fs_new in (20, 15, 10, 5):
        resampled = resample_signal(gait, fs_new, FS)
        assert resampled.shape == (int(np.ceil(gait.shape[0] * fs_new / FS)), 6)
        # La amplitud de la marcha (1.8 Hz < Nyquist) se conserva
        assert iqr(resampled[:, 2]) == pytest.approx(iqr(gait[:, 2]), rel=0.1)
        # La gravedad se conserva en los bordes (sin transitorios por relleno con ceros)
        assert abs(resampled[0, 2] - gait[0, 2]) < 0.2
        assert abs(resampled[-1, 2] - gait[-1, 2]) < 0.2


def test_resample_signal_identity_and_limits():
    gait = synthetic_gait()[:, 1:]
    np.testing.assert_array_equal(resample_signal(gait, 25, FS), gait)
    with pytest.raises(ValueError):
        resample_signal(gait, 50, FS)


@pytest.mark.parametrize("fs_new", FREQUENCIES)
def test_stack_and_features_at_each_frequency(tmp_path, fs_new):
    data, labels, stack_file = build_stack(tmp_path, fs_new)

    # La ventana conserva su duración (10 s)
    assert data.shape[1:] == (6, int(round(10 * fs_new)))
    # Por encima de ~4 Hz la marcha de 1.8 Hz sobrevive al diezmado
    walking = data[np.array(labels) == "CAMINAR USUAL SPEED"]
    if fs_new >= 5:
        assert np.median([iqr(w[2]) for w in walking]) > 0.3

    features, _, _ = extract_features_from_stack(str(stack_file))
    assert np.all(np.isfinite(features[np.array(labels) == "CAMINAR USUAL SPEED"]))


def test_features_at_25hz_match_original_definition(tmp_path):
    data, _, stack_file = build_stack(tmp_path, None)
    stack_25, _, _ = build_stack(tmp_path, 25)
    np.testing.assert_array_equal(data, stack_25)

    features, _, _ = extract_features_from_stack(str(stack_file))
    np.testing.assert_allclose(features, extract_features(data, fs=25))
    # Stacks antiguos sin WINDOW_FS se interpretan como 25 Hz
    np.testing.assert_allclose(features, extract_features(data))


def test_spectral_feature_reports_hz():
    gait = synthetic_gait()[:, 1:]
    for fs_new in (25, 10, 5):
        windows = apply_windowing_WPM_segmented_data(
            {"W": resample_signal(gait, fs_new, FS)}, int(10 * fs_new), None
        )["W"]
        features = extract_features(windows, fs=fs_new)
        # Columnas 30..59: 5 características espectrales por canal; f1 del eje z (canal 2)
        assert np.median(features[:, 30 + 2 * 5]) == pytest.approx(1.8, abs=0.11)
