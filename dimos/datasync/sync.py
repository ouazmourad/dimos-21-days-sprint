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
"""SyncTransformer — resample multiple TimeSeriesStores onto a uniform time grid."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from typing import Any

from dimos.memory.timeseries.base import TimeSeriesStore


class SyncPolicy(Enum):
    """Policy for handling missing data at a grid tick."""

    HOLD = "hold"
    ASOF = "asof"
    DROP = "drop"


class SyncTransformer:
    """Resample multiple TimeSeriesStores onto a uniform time grid."""

    def __init__(
        self,
        stores: dict[str, TimeSeriesStore],
        target_hz: float = 10.0,
        policy: SyncPolicy = SyncPolicy.HOLD,
        staleness_sec: float = 1.0,
    ) -> None:
        if target_hz <= 0:
            raise ValueError("target_hz must be positive")
        self._stores = stores
        self._target_hz = target_hz
        self._policy = policy
        self._staleness_sec = staleness_sec

    @property
    def stores(self) -> dict[str, TimeSeriesStore]:
        return self._stores

    @property
    def target_hz(self) -> float:
        return self._target_hz

    @property
    def policy(self) -> SyncPolicy:
        return self._policy

    @classmethod
    def from_session(
        cls,
        session: Any,
        target_hz: float = 10.0,
        policy: SyncPolicy = SyncPolicy.HOLD,
        staleness_sec: float = 1.0,
        topic_keys: list[str] | None = None,
    ) -> SyncTransformer:
        """Create a SyncTransformer from a Session's stores."""
        keys = topic_keys or session.topic_keys
        stores = {k: session.get_store(k) for k in keys}
        return cls(stores, target_hz, policy, staleness_sec)

    def _compute_overlap(
        self,
        start: float | None = None,
        end: float | None = None,
    ) -> tuple[float, float] | None:
        starts: list[float] = []
        ends: list[float] = []
        for store in self._stores.values():
            s = store.start_ts
            e = store.end_ts
            if s is not None and e is not None:
                starts.append(s)
                ends.append(e)
        if not starts:
            return None
        overlap_start = max(starts) if start is None else max(max(starts), start)
        overlap_end = min(ends) if end is None else min(min(ends), end)
        if overlap_start > overlap_end:
            return None
        return (overlap_start, overlap_end)

    def iterate_synced(
        self,
        start: float | None = None,
        end: float | None = None,
    ) -> Iterator[tuple[float, dict[str, Any | None]]]:
        """Yield ``(grid_ts, {topic_key: data | None})`` at uniform intervals."""
        overlap = self._compute_overlap(start, end)
        if overlap is None:
            return
        grid_start, grid_end = overlap
        interval = 1.0 / self._target_hz

        cursors: dict[str, _Cursor] = {}
        for key, store in self._stores.items():
            it = store._iter_items(start=grid_start)
            cursors[key] = _Cursor(it)

        tick = grid_start
        while tick <= grid_end:
            row: dict[str, Any | None] = {}
            all_present = True
            for key, cursor in cursors.items():
                cursor.advance_to(tick)
                value = cursor.value_at(tick, self._policy, self._staleness_sec)
                row[key] = value
                if value is None:
                    all_present = False
            if self._policy == SyncPolicy.DROP and not all_present:
                tick += interval
                continue
            yield (tick, row)
            tick += interval


class _Cursor:
    def __init__(self, iterator: Iterator[tuple[float, Any]]) -> None:
        self._iterator = iterator
        self._prev_ts: float | None = None
        self._prev_data: Any | None = None
        self._curr_ts: float | None = None
        self._curr_data: Any | None = None
        self._exhausted = False
        self._advance_once()

    def _advance_once(self) -> bool:
        if self._exhausted:
            return False
        try:
            ts, data = next(self._iterator)
            self._prev_ts = self._curr_ts
            self._prev_data = self._curr_data
            self._curr_ts = ts
            self._curr_data = data
            return True
        except StopIteration:
            self._exhausted = True
            return False

    def advance_to(self, tick: float) -> None:
        while (
            not self._exhausted
            and self._curr_ts is not None
            and self._curr_ts <= tick
        ):
            next_ts = self._peek_next_ts()
            if next_ts is not None and next_ts <= tick:
                self._advance_once()
            else:
                break

    def _peek_next_ts(self) -> float | None:
        if self._exhausted:
            return None
        if self._advance_once():
            return self._curr_ts
        return None

    def value_at(self, tick: float, policy: SyncPolicy, staleness_sec: float) -> Any | None:
        best_ts: float | None = None
        best_data: Any | None = None
        if self._curr_ts is not None and self._curr_ts <= tick:
            best_ts = self._curr_ts
            best_data = self._curr_data
        elif self._prev_ts is not None and self._prev_ts <= tick:
            best_ts = self._prev_ts
            best_data = self._prev_data
        if best_ts is None:
            return None
        if policy == SyncPolicy.HOLD:
            return best_data
        elif policy == SyncPolicy.ASOF:
            if (tick - best_ts) <= staleness_sec:
                return best_data
            return None
        elif policy == SyncPolicy.DROP:
            return best_data
        return None
