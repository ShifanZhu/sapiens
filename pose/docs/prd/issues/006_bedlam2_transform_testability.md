# Issue 006: Injectable Cache Providers + Fixture Factory for BEDLAM2 Transforms

**Type:** Refactor RFC
**Status:** Open
**Generated:** 2026-03-21

---

## Summary

`bedlam2_transforms.py` contains 5 sequential transforms (`LoadBedlamLabels → NoisyBBoxTransform → CropPersonRGBD → SubtractRootJoint → PackBedlamInputs`) that are currently untestable in isolation. The blocker is three module-level global dicts (`_label_cache`, `_depth_mmap`, `_depth_npz_cache`) that `LoadBedlamLabels` writes to and reads from at module scope — tests cannot inject synthetic data without touching real NPZ/NPY/JPEG files on disk.

This RFC proposes the minimum change to make every transform independently unit-testable: encapsulate the globals into injected cache providers, and add a synthetic fixture factory for tests.

---

## Problem

- `_label_cache`, `_depth_mmap`, `_depth_npz_cache` are module-level globals → no injection point for tests
- Cannot test `SubtractRootJoint` or `PackBedlamInputs` without first running `LoadBedlamLabels` on real files
- Currently **zero unit tests** for any of the 5 transforms
- Silent `KeyError` (not a clear message) if transforms are applied out of order

---

## Proposed Change

### 1. Extract cache globals into singleton classes

```python
# mmpose/datasets/transforms/bedlam2_transforms.py

from typing import Optional, Protocol
import numpy as np

class LabelCacheProvider(Protocol):
    def get(self, label_path: str) -> dict: ...

class DepthCacheProvider(Protocol):
    def get(self, npy_path: str, npz_path: str, frame_idx: int) -> Optional[np.ndarray]: ...

class _GlobalLabelCache:
    """Encapsulates current _label_cache dict. One instance per worker."""
    def __init__(self):
        self._cache: dict[str, dict] = {}
    def get(self, label_path: str) -> dict:
        # identical logic to current LoadBedlamLabels label loading

class _GlobalDepthCache:
    """Encapsulates current _depth_mmap + _depth_npz_cache dicts."""
    def __init__(self):
        self._mmap: dict = {}
        self._npz: OrderedDict = OrderedDict()
    def get(self, npy_path, npz_path, frame_idx) -> Optional[np.ndarray]:
        # identical logic to current _read_depth()

# Module-level singletons — per-worker after fork, same isolation as today
_DEFAULT_LABEL_CACHE = _GlobalLabelCache()
_DEFAULT_DEPTH_CACHE = _GlobalDepthCache()
```

### 2. Add injectable params to `LoadBedlamLabels`

```python
@TRANSFORMS.register_module()
class LoadBedlamLabels(BaseTransform):
    def __init__(
        self,
        depth_required: bool = True,
        filter_invalid: bool = True,
        # Underscore prefix = internal/testing only; defaults to production singletons
        _label_cache: Optional[LabelCacheProvider] = None,
        _depth_cache: Optional[DepthCacheProvider] = None,
    ):
        self._label_cache = _label_cache or _DEFAULT_LABEL_CACHE
        self._depth_cache = _depth_cache or _DEFAULT_DEPTH_CACHE
        ...
```

No config change required — existing configs pass only `depth_required` and `filter_invalid`.

### 3. Add `make_bedlam2_results` fixture factory

```python
# mmpose/datasets/transforms/testing_utils.py

def make_bedlam2_results(
    *,
    img_h: int = 640,
    img_w: int = 384,
    after_stage: str = 'load',   # 'load' | 'crop' | 'subtract_root'
    joints_cam: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    ...
) -> dict:
    """Return a synthetic results dict matching each transform's input contract.

    after_stage='load'          -> keys written by LoadBedlamLabels
    after_stage='crop'          -> above + img resized to (img_h, img_w)
    after_stage='subtract_root' -> above + pelvis_abs, pelvis_depth, pelvis_uv
    """
```

### 4. Unit tests (≥6 covering all 5 transforms)

```python
# tests/test_datasets/test_bedlam2_transforms.py

def test_subtract_root_pelvis_at_origin():
    results = make_bedlam2_results(after_stage='crop')
    out = SubtractRootJoint().transform(results)
    assert np.allclose(out['joints_cam'][0], 0.0, atol=1e-6)

def test_pack_output_tensor_shape():
    results = make_bedlam2_results(after_stage='subtract_root', img_h=640, img_w=384)
    out = PackBedlamInputs().transform(results)
    assert out['inputs'].shape == (4, 640, 384)

def test_load_with_fake_cache(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, 'imread', lambda p: np.zeros((480, 640, 3), np.uint8))
    t = LoadBedlamLabels(_label_cache=FakeLabelCache(), _depth_cache=FakeDepthCache())
    out = t.transform(_make_load_input())
    assert out['joints_cam'].shape == (70, 3)
```

---

## Scope

- **1 production class changed:** `LoadBedlamLabels.__init__` (add 2 optional params)
- **1 new file:** `mmpose/datasets/transforms/testing_utils.py` (fixture factory)
- **1 new test file:** `tests/test_datasets/test_bedlam2_transforms.py`
- **0 config changes**
- **0 behavior changes** — production path identical (defaults to module-global singletons)

## Out of Scope

- Order-enforcement / key-contract validation between transforms (separate issue)
- Typed schema (`Bedlam2Sample` dataclass)
- Pelvis geometry deduplication (separate issue)

---

## Acceptance Criteria

- [ ] `LoadBedlamLabels` accepts `_label_cache` and `_depth_cache` constructor params
- [ ] Module-level globals encapsulated in `_GlobalLabelCache` / `_GlobalDepthCache`
- [ ] `make_bedlam2_results(after_stage=...)` in `mmpose/datasets/transforms/testing_utils.py`
- [ ] `pytest tests/test_datasets/test_bedlam2_transforms.py` passes with ≥6 tests, zero real file I/O
- [ ] Existing configs unchanged and training smoke-test still passes

---

## Design Alternatives Considered

**A. Single `Bedlam2Pipeline` class (minimal interface):** Wraps all 5 transforms into one registered class; configs shrink to 1 entry. Rejected: requires a `_transform_from_loaded` backdoor for tests, loses per-transform granularity, makes adding new augmentations harder.

**B. Protocol layer + `Bedlam2Sample` dataclass + `ValidatingCompose` (flexible):** Full typed schema, order validation, swappable loaders. Rejected: the dataclass requires a dict↔dataclass adapter around MMEngine's evaluator; `ValidatingCompose` is a framework on top of a framework. Over-engineered for a 5-step pipeline.

**C (chosen) + Protocol types from B:** Design C's injectable singletons give 80% of the testability benefit at minimal disruption. The `LabelCacheProvider`/`DepthCacheProvider` Protocol types from B are included as documentation (5 lines each, zero runtime cost).
