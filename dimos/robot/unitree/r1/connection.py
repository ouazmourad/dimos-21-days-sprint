# Copyright 2026 Dimensional Inc.
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

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.unitree.connection import UnitreeWebRTCConnection
from dimos.robot.unitree.r1.effectors.high_level.dds_sdk import R1HighLevelDdsSdk
from dimos.robot.unitree.r1.effectors.high_level.loco_proxy import R1LocoProxy
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class R1Config(ModuleConfig):
    ip: str = Field(default_factory=lambda m: m["g"].robot_ip)
    # TODO(R1): confirm against R1 SDK/docs — currently mirrors G1 ("eth0").
    network_interface: str = "eth0"
    connection_type: str = Field(default_factory=lambda m: m["g"].unitree_connection_type)
    # Real-hardware control backend:
    #   "pc1"    — proxy loco commands to PC1's native r1_loco_client over SSH.
    #              The working path on a real R1: PC1's unitree_sdk2 matches the
    #              firmware, so commands reach the robot.
    #   "dds"    — native unitree_sdk2py on this host (needs the [unitree-dds]
    #              extra AND an SDK whose IDL matches the R1 firmware; pip 1.0.3
    #              does NOT — XTypes type-consistency rejects all endpoints).
    #   "webrtc" — Go2/G1 data-channel path; the R1 brokers WebRTC over DDS with
    #              no HTTP /offer server, so this does not apply to the R1.
    backend: str = "pc1"
    # PC1 (onboard Jetson) SSH-proxy settings for the "pc1" backend. The password
    # is read from the R1_PC1_PASS env var by R1LocoProxy (Unitree default "123").
    pc1_host: str = "192.168.123.164"
    pc1_user: str = "unitree"
    pc1_interface: str = "eth10"


class R1ConnectionBase(Module, ABC):
    """Abstract base for R1 connections (real hardware and simulation).

    Modules that depend on R1 connection RPC methods should reference this
    base class so the blueprint wiring works regardless of which concrete
    connection is deployed.
    """

    config: ModuleConfig

    @rpc
    @abstractmethod
    def start(self) -> None:
        super().start()

    @rpc
    @abstractmethod
    def stop(self) -> None:
        super().stop()

    @rpc
    @abstractmethod
    def move(self, twist: Twist, duration: float = 0.0) -> None: ...

    @rpc
    @abstractmethod
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]: ...


class R1Connection(R1ConnectionBase):
    config: R1Config
    cmd_vel: In[Twist]
    connection: UnitreeWebRTCConnection | R1HighLevelDdsSdk | R1LocoProxy | None = None

    def __init__(self, *args: Any, g: GlobalConfig = global_config, **kwargs: Any) -> None:
        super().__init__(*args, g=g, **kwargs)
        self._global_config = g

    @rpc
    def start(self) -> None:
        super().start()

        if self.config.connection_type == "replay":
            raise NotImplementedError("Replay connection not implemented for R1 robot")
        if self.config.connection_type == "mujoco":
            raise NotImplementedError(
                "This module does not support simulation, use R1SimConnection instead"
            )

        # Real hardware — pick the control backend (see R1Config.backend).
        match self.config.backend:
            case "pc1":
                # Proxy loco commands to PC1's native r1_loco_client over SSH.
                # PC1's unitree_sdk2 matches the R1 firmware, so commands reach
                # the robot where this host's pip SDK cannot (type mismatch).
                self.connection = R1LocoProxy(
                    host=self.config.pc1_host,
                    user=self.config.pc1_user,
                    interface=self.config.pc1_interface,
                )
            case "webrtc":
                # WebRTC variant: the same data-channel path DimOS uses in
                # production for the Go2 and G1. Needs the [unitree] extra
                # (unitree-webrtc-connect-leshy), no DDS/SDK install required.
                self.connection = UnitreeWebRTCConnection(self.config.ip)
            case "dds":
                # DDS variant: native unitree_sdk2py LocoClient (needs
                # [unitree-dds]). SDK imports live inside R1HighLevelDdsSdk.start(),
                # so this module stays import-safe without the extra.
                self.connection = R1HighLevelDdsSdk(
                    g=self._global_config,
                    network_interface=self.config.network_interface,
                )
            case _:
                raise ValueError(
                    f"Unknown R1 control backend: {self.config.backend!r} (use 'webrtc' or 'dds')"
                )

        assert self.connection is not None
        self.connection.start()

        self.register_disposable(Disposable(self.cmd_vel.subscribe(self.move)))

    @rpc
    def stop(self) -> None:
        assert self.connection is not None
        self.connection.stop()
        super().stop()

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> None:
        assert self.connection is not None
        self.connection.move(twist, duration)

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        logger.info(f"Publishing request to topic: {topic} with data: {data}")
        assert self.connection is not None
        return self.connection.publish_request(topic, data)  # type: ignore[no-any-return]

    @rpc
    def stand_up(self) -> bool:
        assert self.connection is not None
        if not hasattr(self.connection, "stand_up"):
            logger.warning("stand_up() is a DDS-backend helper; on WebRTC use a mode command")
            return False
        return self.connection.stand_up()

    @rpc
    def lie_down(self) -> bool:
        assert self.connection is not None
        if not hasattr(self.connection, "lie_down"):
            logger.warning("lie_down() is a DDS-backend helper; on WebRTC use a mode command")
            return False
        return self.connection.lie_down()

    @rpc
    def get_state(self) -> str:
        assert self.connection is not None
        if not hasattr(self.connection, "get_state"):
            return "unknown"
        return self.connection.get_state()


__all__ = ["R1Config", "R1Connection", "R1ConnectionBase"]
