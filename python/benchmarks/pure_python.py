"""Pure-Python reference implementations of the same structures.

These exist for two reasons: as an honest, dependency-free baseline for the
benchmarks, and as *independent* implementations the test suite cross-checks
against the native core. They mirror the algorithms in ``src/`` but are written
straightforwardly in Python (and are, predictably, far slower).
"""

from __future__ import annotations

import hashlib
import math

_MASK64 = (1 << 64) - 1
_LN2 = math.log(2)


def _hash_pair(data: bytes, seed: int) -> tuple[int, int]:
    """Two independent 64-bit hashes via BLAKE2b (deterministic, unlike the
    process-salted built-in ``hash``)."""
    h = hashlib.blake2b(seed.to_bytes(8, "little") + data, digest_size=16).digest()
    return int.from_bytes(h[:8], "little"), int.from_bytes(h[8:], "little") | 1


def _hash64(data: bytes, seed: int) -> int:
    return _hash_pair(data, seed)[0]


class PyBloomFilter:
    def __init__(self, capacity: int, error_rate: float = 0.01, seed: int = 0):
        m = math.ceil(-capacity * math.log(error_rate) / (_LN2 * _LN2))
        m = max(8, (m + 7) & ~7)  # whole bytes
        self.m = m
        self.k = max(1, round(m / capacity * _LN2))
        self.seed = seed
        self.bits = bytearray(m // 8)

    def add(self, data: bytes) -> None:
        h1, h2 = _hash_pair(data, self.seed)
        for i in range(self.k):
            pos = (h1 + i * h2) % self.m
            self.bits[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, data: bytes) -> bool:
        h1, h2 = _hash_pair(data, self.seed)
        for i in range(self.k):
            pos = (h1 + i * h2) % self.m
            if not (self.bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True


class PyHyperLogLog:
    def __init__(self, precision: int = 14, seed: int = 0):
        self.p = precision
        self.m = 1 << precision
        self.seed = seed
        self.registers = bytearray(self.m)

    def add(self, data: bytes) -> None:
        x = _hash64(data, self.seed)
        idx = x >> (64 - self.p)
        w = (x << self.p) & _MASK64
        clz = 64 - w.bit_length()
        rank = min(clz + 1, 64 - self.p + 1)
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def _alpha(self) -> float:
        m = self.m
        if m == 16:
            return 0.673
        if m == 32:
            return 0.697
        if m == 64:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / m)

    def estimate(self) -> float:
        s = 0.0
        zeros = 0
        for r in self.registers:
            s += 2.0 ** -r
            if r == 0:
                zeros += 1
        raw = self._alpha() * self.m * self.m / s
        if raw <= 2.5 * self.m and zeros != 0:
            return self.m * math.log(self.m / zeros)
        return raw

    def __len__(self) -> int:
        return round(self.estimate())


class PyCountMinSketch:
    def __init__(self, epsilon: float = 0.001, delta: float = 0.001, seed: int = 0):
        self.width = max(1, math.ceil(math.e / epsilon))
        self.depth = max(1, math.ceil(math.log(1.0 / delta)))
        self.seed = seed
        self.counters = [[0] * self.width for _ in range(self.depth)]
        self.total = 0

    def add(self, data: bytes, count: int = 1) -> None:
        h1, h2 = _hash_pair(data, self.seed)
        for r in range(self.depth):
            col = (h1 + r * h2) % self.width
            self.counters[r][col] += count
        self.total += count

    def estimate(self, data: bytes) -> int:
        h1, h2 = _hash_pair(data, self.seed)
        return min(
            self.counters[r][(h1 + r * h2) % self.width] for r in range(self.depth)
        )
