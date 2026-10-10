"""Resource limits for research: execution and safety always come first.

Research may start a job only while memory and CPU leave room; it stops a running job (which is queued again)
when memory gets close to the limit. Memory is read the way the kernel enforces it: the container's own limit
(cgroup v2 ``memory.max``/``memory.current``, or v1) when there is one, and the machine's memory as well (the
database and the dashboard share a small server) — the higher of the two percentages counts. Heavy jobs need
an extra margin.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from quantpulse.config import Settings

HEAVY_MARGIN = 10.0  # percentage points of extra memory headroom a heavy job needs


@dataclass(frozen=True)
class Snapshot:
    memory_pct: float  # the higher of the container's and the machine's memory use
    memory_source: str  # cgroup | host
    load_per_cpu: float
    rss_mb: float  # this process


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cgroup_pct() -> float | None:
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        limit, usage = _read(limit_path), _read(usage_path)
        if limit is None or usage is None or limit == "max":
            continue
        try:
            lim, use = int(limit), int(usage)
        except ValueError:
            continue
        if 0 < lim < 1 << 60:  # cgroup v1 reports "no limit" as a huge number
            return 100.0 * use / lim
    return None


def _host_pct() -> float:
    info: dict[str, float] = {}
    text = _read("/proc/meminfo") or ""
    for line in text.splitlines():
        if ":" in line:
            key, rest = line.split(":", 1)
            parts = rest.split()
            if parts:
                info[key] = float(parts[0])
    total = info.get("MemTotal", 0.0)
    if not total:
        return 0.0
    return 100.0 * (total - info.get("MemAvailable", total)) / total


def _rss_mb() -> float:
    for line in (_read("/proc/self/status") or "").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024
    return 0.0


def measure() -> Snapshot:
    host = _host_pct()
    group = _cgroup_pct()
    memory, source = (group, "cgroup") if group is not None and group >= host else (host, "host")
    try:
        load = os.getloadavg()[0] / (os.cpu_count() or 1)
    except OSError:
        load = 0.0
    return Snapshot(round(memory, 1), source, round(load, 3), round(_rss_mb(), 1))


class ResourceGovernor:
    def __init__(self, settings: Settings, read: Callable[[], Snapshot] = measure) -> None:
        self._s = settings
        self._read = read

    def snapshot(self) -> Snapshot:
        return self._read()

    def may_start(self, cost: str, snap: Snapshot | None = None) -> tuple[bool, str]:
        snap = snap or self._read()
        limit = self._s.research_max_memory_pct - (HEAVY_MARGIN if cost == "heavy" else 0.0)
        if snap.memory_pct >= limit:
            return (
                False,
                f"memory {snap.memory_pct:.0f}% ≥ {limit:.0f}% ({snap.memory_source}): research waits",
            )
        if snap.load_per_cpu >= self._s.research_max_load:
            return (
                False,
                f"CPU load {snap.load_per_cpu:.2f}/core ≥ {self._s.research_max_load:.2f}: research waits",
            )
        return True, "room to run"

    def must_stop(self, snap: Snapshot | None = None) -> tuple[bool, str]:
        snap = snap or self._read()
        if snap.memory_pct >= self._s.research_abort_memory_pct:
            return True, (
                f"memory {snap.memory_pct:.0f}% ≥ {self._s.research_abort_memory_pct:.0f}% "
                f"({snap.memory_source}): research stopped to protect execution"
            )
        return False, ""
