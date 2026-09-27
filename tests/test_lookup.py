from pathlib import Path

import numpy as np
import pytest

from pimf import IMF, SmoothAbs


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data" / "lookup_reference.npz") as data:
        return dict(data)


@pytest.mark.parametrize("lookup", [None, 0, 1, 0.5, "yes", [], np.nan])
def test_invalid_lookup_raises(lookup):
    with pytest.raises(ValueError, match="lookup must be a boolean"):
        SmoothAbs(0.4, lookup=lookup)


def test_lookup_is_default():
    assert SmoothAbs(0.4).lookup is True
    assert SmoothAbs(0.4, lookup=np.bool_(False)).lookup is False


@pytest.mark.parametrize("H", [0.02, 0.4, 2.0])
def test_lookup_matches_research_score_and_loss(reference, H):
    r = reference["z"] * H
    contrast = SmoothAbs(H)
    np.testing.assert_allclose(contrast.psi(r), reference["score"], rtol=0, atol=2e-15)
    np.testing.assert_allclose(contrast(r), reference["rho"] * H, rtol=0, atol=4e-14)
    np.testing.assert_allclose(contrast.psi(r), reference["direct_score"], rtol=0, atol=1.1e-6)
    np.testing.assert_allclose(contrast(r) / H, reference["direct_rho"], rtol=0, atol=1.6e-6)


def test_direct_evaluation_matches_research(reference):
    contrast = SmoothAbs(1.0, lookup=False)
    # The saved fixture crosses CPU architectures and NumPy math implementations.
    np.testing.assert_allclose(
        contrast.psi(reference["z"]), reference["direct_score"], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        contrast(reference["z"]), reference["direct_rho"], rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("workers", [1, 4])
def test_lookup_components_match_research(reference, workers):
    result = IMF(SmoothAbs(0.4), workers=workers).decompose(
        reference["y"], window_sizes=reference["window_sizes"]
    )
    tolerance = 1e-12 if workers == 1 else 2e-6
    np.testing.assert_allclose(result.imfs, reference["imfs"], rtol=0, atol=tolerance)
    np.testing.assert_allclose(result.residual, reference["residual"], rtol=0, atol=tolerance)
    np.testing.assert_allclose(result.reconstruction, reference["y"], rtol=0, atol=1e-12)
    assert all(stage.converged for stage in result.stages)
    if workers == 1:
        assert [stage.iterations for stage in result.stages] == reference["iterations"].tolist()


def test_lookup_preserves_shapes_and_inputs(reference):
    contrast = SmoothAbs(0.4)
    r = np.stack([reference["z"][:100], reference["z"][100:200]])[:, ::3] * 0.4
    r.setflags(write=False)
    original = r.copy()
    for method in [contrast, contrast.psi]:
        result = method(r)
        assert result.shape == r.shape
        np.testing.assert_allclose(result.ravel(), method(r.ravel()), rtol=0, atol=1e-15)
        np.testing.assert_allclose(method(float(r[0, 0])), result[0, 0], rtol=0, atol=1e-15)
        assert method(np.empty((2, 0))).shape == (2, 0)
    np.testing.assert_allclose(r, original, rtol=0, atol=0)


def test_lookup_handles_nan_and_infinite_residuals():
    contrast = SmoothAbs(0.4)
    r = np.array([-np.inf, np.nan, np.inf])
    with np.errstate(all="raise"):
        np.testing.assert_allclose(contrast.psi(r), [-1, np.nan, 1], rtol=0, atol=2e-15)
        np.testing.assert_allclose(contrast(r), [np.inf, np.nan, np.inf], rtol=0, atol=0)
        assert np.isnan(contrast.psi(np.nan))
        assert np.isnan(contrast(np.nan))


def test_lookup_saturates_when_standardization_overflows():
    with np.errstate(all="raise"):
        contrast = SmoothAbs(np.finfo(float).tiny)
        r = np.array([-1e300, 0, 1e300])
        np.testing.assert_allclose(contrast.psi(r), [-1, 0, 1], rtol=0, atol=2e-15)
