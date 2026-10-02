"""Which deployment, version and instance this process is — for the lease holder, logs and the cloud status.

Render sets ``RENDER_GIT_COMMIT``, ``RENDER_GIT_BRANCH``, ``RENDER_SERVICE_NAME`` and ``RENDER_INSTANCE_ID`` on
every instance; elsewhere ``QP_GIT_COMMIT`` can be set, and the host name stands in for the instance. None of
these is a secret.
"""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass
from typing import Any

from quantpulse import __version__


@dataclass(frozen=True, slots=True)
class Runtime:
    version: str
    commit: str | None
    branch: str | None
    service: str | None
    instance: str
    platform: str

    @property
    def short_commit(self) -> str | None:
        return self.commit[:7] if self.commit else None

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "short_commit": self.short_commit}


def current() -> Runtime:
    env = os.environ
    on_render = bool(env.get("RENDER") or env.get("RENDER_SERVICE_ID"))
    return Runtime(
        version=__version__,
        commit=env.get("RENDER_GIT_COMMIT") or env.get("QP_GIT_COMMIT") or None,
        branch=env.get("RENDER_GIT_BRANCH") or env.get("QP_GIT_BRANCH") or None,
        service=env.get("RENDER_SERVICE_NAME") or None,
        instance=(env.get("RENDER_INSTANCE_ID") or socket.gethostname())[:60],
        platform="render" if on_render else "self-hosted",
    )
