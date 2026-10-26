"""Shared file/tensor identities and bounded private-memory sampling."""

import hashlib
from pathlib import Path
import threading

import psutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(items):
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


class ResourceSampler:
    def __init__(self):
        self.stop = threading.Event()
        self.max_uss = self.max_rss = 0
        self.errors = []
        self.over_limit = False
        self.proc = psutil.Process()

    def measure(self):
        try:
            m = self.proc.memory_full_info()
            self.max_uss = max(self.max_uss, m.uss)
            self.max_rss = max(self.max_rss, m.rss)
            self.over_limit |= m.uss > 48 * 1024**3
        except psutil.Error as exc:
            self.errors.append(type(exc).__name__)

    def loop(self):
        while not self.stop.wait(0.05):
            self.measure()

    def __enter__(self):
        self.measure()
        if self.errors:
            raise RuntimeError(
                "Private-memory measurement unavailable; use read-only process permission"
            )
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.measure()
