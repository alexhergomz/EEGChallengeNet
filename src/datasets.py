import re
from typing import Iterable, Iterator, List, Optional, Tuple, Dict, Any

import fsspec
import numpy as np
import torch
from torch.utils.data import IterableDataset


_SUB_RE = re.compile(r"sub-([A-Za-z0-9_]+)")
_TASK_RE = re.compile(r"task-([A-Za-z0-9_]+)")


def _extract_labels_from_path(path: str) -> Tuple[int, int]:
    sub_match = _SUB_RE.search(path)
    task_match = _TASK_RE.search(path)
    # Fallback to hash buckets if missing
    sub_raw = sub_match.group(1) if sub_match else "unknown_sub"
    task_raw = task_match.group(1) if task_match else "unknown_task"
    # Map to stable ints
    sub_id = abs(hash(("sub", sub_raw))) % (10 ** 6)
    task_id = abs(hash(("task", task_raw))) % (10 ** 6)
    return sub_id, task_id


def _load_array(fs: fsspec.AbstractFileSystem, path: str) -> Optional[np.ndarray]:
    if path.endswith('.npy') or path.endswith('.npz'):
        with fs.open(path, 'rb') as f:
            return np.load(f)
    if path.endswith('.csv') or path.endswith('.txt'):
        with fs.open(path, 'rb') as f:
            return np.loadtxt(f, delimiter=',')
    return None


class S3EEGIterableDataset(IterableDataset):
    """
    Streams EEG arrays from S3 under a given prefix. Supports .npy/.npz/.csv/.txt numeric matrices.
    Produces sliding windows of shape (L, C) with subject/task ids extracted from BIDS-like paths.
    """

    def __init__(
        self,
        s3_uri: str,
        window_length: int,
        stride: int,
        max_files: Optional[int] = None,
        allowed_exts: Optional[List[str]] = None,
        channels: Optional[int] = None,
        s3_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.s3_uri = s3_uri.rstrip('/')
        self.window_length = window_length
        self.stride = stride
        self.max_files = max_files
        self.allowed_exts = allowed_exts or ['.npy', '.npz', '.csv', '.txt']
        self.channels = channels
        # Initialize filesystem with options (e.g., anon=True)
        self.fs = fsspec.filesystem('s3', **(s3_options or {}))
        self.prefix = self.s3_uri

    def _list_files(self) -> List[str]:
        fs = self.fs
        prefix = self.prefix
        files = []
        for p, _, fnames in fs.walk(prefix):
            for name in fnames:
                if any(name.endswith(ext) for ext in self.allowed_exts):
                    files.append(f"{p}/{name}")
                    if self.max_files and len(files) >= self.max_files:
                        return files
        return files

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, int]]:
        fs = self.fs
        for path in self._list_files():
            try:
                arr = _load_array(fs, path)
                if arr is None:
                    continue
                if arr.ndim == 1:
                    arr = arr[:, None]
                # Ensure float32
                X = arr.astype(np.float32)
                # Optionally trim channels
                if self.channels is not None:
                    X = X[:, : self.channels]
                T, C = X.shape
                sub_id, task_id = _extract_labels_from_path(path)
                # Sliding windows
                for t0 in range(0, max(1, T - self.window_length + 1), self.stride):
                    window = X[t0 : t0 + self.window_length]
                    if window.shape[0] < self.window_length:
                        continue
                    yield torch.from_numpy(window), sub_id, task_id
            except Exception:
                # Skip unreadable files
                continue


