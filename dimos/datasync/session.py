# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Session management for multi-sensor recording."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dimos.memory.timeseries.sqlite import SqliteStore
from dimos.utils.data import get_data_dir


@dataclass
class SessionMeta:
    """Metadata for a recording session."""

    session_id: str
    created_at: float
    robot_type: str = ""
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


class Session:
    """Groups per-topic SqliteStores into a named recording session.

    Disk layout::

        data/sessions/{session_id}/
            meta.json
            stores/{topic_key}.db
    """

    def __init__(self, meta: SessionMeta, base_dir: Path) -> None:
        self._meta = meta
        self._base_dir = base_dir
        self._stores: dict[str, SqliteStore] = {}

    @property
    def meta(self) -> SessionMeta:
        return self._meta

    @property
    def session_id(self) -> str:
        return self._meta.session_id

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @classmethod
    def create(
        cls,
        session_id: str | None = None,
        robot_type: str = "",
        tags: list[str] | None = None,
    ) -> Session:
        """Create a new recording session."""
        sid = session_id or uuid.uuid4().hex[:12]
        meta = SessionMeta(
            session_id=sid,
            created_at=time.time(),
            robot_type=robot_type,
            tags=tags or [],
        )
        base_dir = get_data_dir(f"sessions/{sid}")
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "stores").mkdir(exist_ok=True)
        cls._write_meta(base_dir, meta)
        return cls(meta, base_dir)

    @classmethod
    def open(cls, path: str | Path) -> Session:
        """Open an existing session from disk."""
        p = Path(path)
        if not p.is_absolute():
            p = get_data_dir(f"sessions/{path}")
        meta_file = p / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Session metadata not found: {meta_file}")
        with open(meta_file) as f:
            meta = SessionMeta(**json.load(f))
        return cls(meta, p)

    def get_store(self, topic_key: str) -> SqliteStore:
        """Get or create a SqliteStore for the given topic key."""
        if topic_key not in self._stores:
            db_path = self._base_dir / "stores" / f"{topic_key}.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            store: SqliteStore = SqliteStore(str(db_path))
            self._stores[topic_key] = store
            if topic_key not in self._meta.topics:
                self._meta.topics.append(topic_key)
        return self._stores[topic_key]

    @property
    def topic_keys(self) -> list[str]:
        return list(self._meta.topics)

    def save_meta(self) -> None:
        """Persist current metadata to disk."""
        self._write_meta(self._base_dir, self._meta)

    def close(self) -> None:
        """Close all stores and save metadata."""
        self.save_meta()
        for store in self._stores.values():
            store.close()
        self._stores.clear()

    @staticmethod
    def _write_meta(base_dir: Path, meta: SessionMeta) -> None:
        with open(base_dir / "meta.json", "w") as f:
            json.dump(asdict(meta), f, indent=2)
