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
"""Tests for SyncTransformer and DataFrameExporter."""

from __future__ import annotations

import pytest

from dimos.datasync.export import DataFrameExporter, flatten_generic, flatten_odometry, get_flattener
from dimos.datasync.sync import SyncPolicy, SyncTransformer
from dimos.memory.timeseries.inmemory import InMemoryStore
from dimos.types.timestamped import Timestamped


class FakeSensor(Timestamped):
    def __init__(self, ts: float, value: float = 0.0, label: str = "") -> None:
        super().__init__(ts)
        self.value = value
        self.label = label


class FakeOdometry(Timestamped):
    msg_name = "nav_msgs.Odometry"
    def __init__(self, ts: float, x: float = 0, y: float = 0, z: float = 0) -> None:
        super().__init__(ts)
        self.x = x; self.y = y; self.z = z
        self.roll = self.pitch = self.yaw = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.wx = self.wy = self.wz = 0.0


def _make_store(hz: float, duration: float, start: float = 100.0) -> InMemoryStore:
    store: InMemoryStore = InMemoryStore()
    interval = 1.0 / hz
    t = start
    i = 0
    while t <= start + duration:
        store.save(FakeSensor(ts=t, value=float(i)))
        t += interval
        i += 1
    return store


class TestSyncTransformer:
    def test_basic_sync(self) -> None:
        fast = _make_store(hz=30, duration=1.0, start=100.0)
        slow = _make_store(hz=10, duration=1.0, start=100.0)
        sync = SyncTransformer(stores={"fast": fast, "slow": slow}, target_hz=5.0)
        rows = list(sync.iterate_synced())
        assert len(rows) > 0
        for i in range(1, len(rows)):
            assert abs((rows[i][0] - rows[i - 1][0]) - 0.2) < 1e-6
        for _, row in rows:
            assert row["fast"] is not None
            assert row["slow"] is not None

    def test_hold_policy_carries_forward(self) -> None:
        store_a: InMemoryStore = InMemoryStore()
        store_b: InMemoryStore = InMemoryStore()
        store_a.save(FakeSensor(ts=0.0, value=10.0))
        store_a.save(FakeSensor(ts=1.0, value=20.0))
        store_b.save(FakeSensor(ts=0.0, value=100.0))
        store_b.save(FakeSensor(ts=1.0, value=200.0))
        sync = SyncTransformer(stores={"a": store_a, "b": store_b}, target_hz=2.0, policy=SyncPolicy.HOLD)
        rows = list(sync.iterate_synced())
        assert len(rows) >= 2
        assert rows[0][1]["a"] is not None
        assert rows[0][1]["b"] is not None

    def test_drop_policy_filters_incomplete(self) -> None:
        store_a: InMemoryStore = InMemoryStore()
        store_b: InMemoryStore = InMemoryStore()
        for i in range(21):
            store_a.save(FakeSensor(ts=float(i) * 0.1, value=float(i)))
        for i in range(11):
            store_b.save(FakeSensor(ts=0.5 + float(i) * 0.1, value=float(i)))
        sync = SyncTransformer(stores={"a": store_a, "b": store_b}, target_hz=5.0, policy=SyncPolicy.DROP)
        for _, row in sync.iterate_synced():
            assert row["a"] is not None
            assert row["b"] is not None

    def test_asof_policy_staleness(self) -> None:
        store_a: InMemoryStore = InMemoryStore()
        store_b: InMemoryStore = InMemoryStore()
        for i in range(11):
            store_a.save(FakeSensor(ts=float(i) * 0.1, value=float(i)))
        store_b.save(FakeSensor(ts=0.0, value=99.0))
        store_b.save(FakeSensor(ts=1.0, value=100.0))
        sync = SyncTransformer(stores={"a": store_a, "b": store_b}, target_hz=5.0, policy=SyncPolicy.ASOF, staleness_sec=0.3)
        has_none = any(row["b"] is None for _, row in sync.iterate_synced())
        assert has_none

    def test_empty_stores(self) -> None:
        assert list(SyncTransformer(stores={"a": InMemoryStore()}, target_hz=10.0).iterate_synced()) == []

    def test_single_item_stores(self) -> None:
        store: InMemoryStore = InMemoryStore()
        store.save(FakeSensor(ts=5.0, value=42.0))
        rows = list(SyncTransformer(stores={"s": store}, target_hz=10.0).iterate_synced())
        assert len(rows) == 1 and rows[0][1]["s"].value == 42.0

    def test_non_overlapping_stores(self) -> None:
        a: InMemoryStore = InMemoryStore()
        b: InMemoryStore = InMemoryStore()
        a.save(FakeSensor(ts=0.0)); a.save(FakeSensor(ts=1.0))
        b.save(FakeSensor(ts=5.0)); b.save(FakeSensor(ts=6.0))
        assert list(SyncTransformer(stores={"a": a, "b": b}, target_hz=10.0).iterate_synced()) == []

    def test_custom_start_end(self) -> None:
        store = _make_store(hz=10, duration=5.0, start=0.0)
        for ts, _ in SyncTransformer(stores={"s": store}, target_hz=5.0).iterate_synced(start=1.0, end=3.0):
            assert 1.0 <= ts <= 3.0

    def test_invalid_hz_raises(self) -> None:
        with pytest.raises(ValueError):
            SyncTransformer(stores={}, target_hz=0)

    def test_from_session(self) -> None:
        class MockSession:
            topic_keys = ["a", "b"]
            def get_store(self, key: str) -> InMemoryStore:
                return _make_store(hz=10, duration=1.0, start=0.0)
        assert set(SyncTransformer.from_session(MockSession(), target_hz=5.0).stores.keys()) == {"a", "b"}


class TestDataFrameExporter:
    def test_to_dataframe_columns(self) -> None:
        pd = pytest.importorskip("pandas")
        a: InMemoryStore = InMemoryStore()
        b: InMemoryStore = InMemoryStore()
        for i in range(11):
            t = float(i) * 0.1
            a.save(FakeSensor(ts=t, value=float(i), label="A"))
            b.save(FakeSensor(ts=t, value=float(i) * 10, label="B"))
        df = DataFrameExporter(SyncTransformer(stores={"sa": a, "sb": b}, target_hz=5.0)).to_dataframe()
        assert not df.empty and "sa.value" in df.columns and "sb.value" in df.columns

    def test_to_dataframe_with_odometry_flattener(self) -> None:
        pytest.importorskip("pandas")
        store: InMemoryStore = InMemoryStore()
        for i in range(11):
            store.save(FakeOdometry(ts=float(i) * 0.1, x=float(i)))
        df = DataFrameExporter(SyncTransformer(stores={"odom": store}, target_hz=5.0)).to_dataframe()
        assert "odom.x" in df.columns and "odom.vx" in df.columns

    def test_chunked_export(self) -> None:
        pytest.importorskip("pandas")
        chunks = list(DataFrameExporter(SyncTransformer(stores={"s": _make_store(hz=10, duration=3.0, start=0.0)}, target_hz=5.0)).to_dataframes(chunk_duration=1.0))
        assert len(chunks) >= 2

    def test_empty_dataframe(self) -> None:
        pytest.importorskip("pandas")
        assert DataFrameExporter(SyncTransformer(stores={"s": InMemoryStore()}, target_hz=10.0)).to_dataframe().empty

    def test_custom_flattener(self) -> None:
        pytest.importorskip("pandas")
        store: InMemoryStore = InMemoryStore()
        for i in range(11):
            store.save(FakeSensor(ts=float(i) * 0.1, value=float(i)))
        df = DataFrameExporter(SyncTransformer(stores={"s": store}, target_hz=5.0), flatteners={"s": lambda m: {"cf": m.value * 2}}).to_dataframe()
        assert "s.cf" in df.columns and "s.value" not in df.columns


class TestFlatteners:
    def test_flatten_generic(self) -> None:
        assert flatten_generic(FakeSensor(ts=1.0, value=42.0, label="test"))["value"] == 42.0

    def test_get_flattener_by_msg_name(self) -> None:
        assert get_flattener(FakeOdometry(ts=1.0)) is flatten_odometry

    def test_get_flattener_fallback(self) -> None:
        assert get_flattener(FakeSensor(ts=1.0, value=1.0)) is flatten_generic
