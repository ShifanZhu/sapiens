"""Train / val / test split utilities for BEDLAM2.

Splits are always performed at the **sequence level** to prevent frame-level
data leakage between sets.  Sequences from the same folder can appear in
different splits (folder-level split is a future option if needed).
"""

import os
import random
from pathlib import Path


def get_seq_paths(
    overview_path: str,
    single_body_only: bool = True,
    skip_missing_body: bool = True,
    depth_required: bool = True,
    mp4_required: bool = True,
    frames_root: str | None = None,
) -> list[str]:
    """Return a list of relative sequence paths that pass all filters.

    Each entry has the form ``"folder_name/seq_name.npz"`` and can be used
    directly as a key into ``data/label/``.

    Args:
        overview_path: Absolute path to ``data/overview.txt``.
        single_body_only: Skip sequences with more than one person.
        skip_missing_body: Skip sequences where a body annotation is absent.
        depth_required: Skip sequences without a depth npz file.
        mp4_required: Skip sequences without an mp4 video file.
        frames_root: If given (path to ``data/frames/``), skip sequences whose
            frame directory does not exist under this root.

    Returns:
        Filtered list of ``"folder/seq.npz"`` strings.
    """
    seq_paths = []
    with open(overview_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, conditions = line.split(":", 1)
            if single_body_only and "not_single_body=True" in conditions:
                continue
            if skip_missing_body and "missing_body=True" in conditions:
                continue
            if depth_required and "no_depth=True" in conditions:
                continue
            if mp4_required and "no_mp4=True" in conditions:
                continue
            folder, seq = name.strip().split("/")
            if frames_root is not None:
                frames_dir = Path(frames_root) / folder / seq
                if not frames_dir.is_dir():
                    continue
            seq_paths.append(f"{folder}/{seq}.npz")
    return seq_paths


def split_sequences(
    seq_paths: list[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2026,
) -> tuple[list[str], list[str], list[str]]:
    """Randomly split sequence paths into train / val / test.

    Args:
        seq_paths: List of relative sequence paths from :func:`get_seq_paths`.
        val_ratio: Fraction of sequences for validation.
        test_ratio: Fraction of sequences for test.
        seed: Random seed for reproducibility.

    Returns:
        ``(train_paths, val_paths, test_paths)`` — three lists of sequence paths.
    """
    rng = random.Random(seed)
    paths = list(seq_paths)
    rng.shuffle(paths)

    n = len(paths)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))

    test_paths = paths[:n_test]
    val_paths = paths[n_test: n_test + n_val]
    train_paths = paths[n_test + n_val:]
    return train_paths, val_paths, test_paths


def get_splits(
    overview_path: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2026,
    **filter_kwargs,
) -> tuple[list[str], list[str], list[str]]:
    """Convenience wrapper: filter sequences then split.

    Returns:
        ``(train_paths, val_paths, test_paths)``
    """
    seq_paths = get_seq_paths(overview_path, **filter_kwargs)
    return split_sequences(seq_paths, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
