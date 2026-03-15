"""Base Gymnasium wrapper for DimOS MuJoCo simulations."""

from __future__ import annotations

from abc import abstractmethod

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces


class DimOSMuJoCoEnv(gymnasium.Env):
    """Base Gymnasium environment wrapping a DimOS MuJoCo simulation.

    Subclasses must:
      1. Load ``self.model`` and ``self.data`` in ``__init__``
      2. Call ``self._finish_init()`` once model/data are ready
      3. Implement ``_get_obs``, ``_get_reward``, ``_is_terminated``
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sim_dt: float,
        ctrl_dt: float,
        episode_length: int = 1000,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
    ) -> None:
        super().__init__()
        self._sim_dt = sim_dt
        self._ctrl_dt = ctrl_dt
        self._n_substeps = round(ctrl_dt / sim_dt)
        self._episode_length = episode_length
        self.render_mode = render_mode
        self._render_width = render_width
        self._render_height = render_height

        self._step_count = 0
        self._last_action: np.ndarray | None = None

        # Must be set by subclass before calling _finish_init()
        self.model: mujoco.MjModel = None  # type: ignore[assignment]
        self.data: mujoco.MjData = None  # type: ignore[assignment]
        self._renderer: mujoco.Renderer | None = None

    # ------------------------------------------------------------------
    # Initialisation helper (called by subclass after model is loaded)
    # ------------------------------------------------------------------

    def _finish_init(self) -> None:
        """Finalise spaces and renderer.  Call once model/data are set."""
        self.model.opt.timestep = self._sim_dt

        # Observation space (inferred from a single observation)
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32,
        )

        # Action space
        act_low, act_high = self._action_bounds()
        self.action_space = spaces.Box(
            low=act_low, high=act_high, dtype=np.float32,
        )

        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        if self.render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(
                self.model, self._render_height, self._render_width,
            )

    # ------------------------------------------------------------------
    # Interface for subclasses
    # ------------------------------------------------------------------

    def _action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)`` for the action space."""
        low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        return low, high

    def _keyframe_id(self) -> int:
        """Index of the MuJoCo keyframe used on reset."""
        return 0

    def _reset_noise(self, np_random: np.random.Generator) -> None:
        """Add randomisation after keyframe reset (domain randomisation)."""

    def _apply_action(self, action: np.ndarray) -> None:
        """Write ``action`` into ``self.data.ctrl``.  Override for scaling."""
        self.data.ctrl[:] = action

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        ...

    @abstractmethod
    def _get_reward(self, action: np.ndarray) -> float:
        ...

    @abstractmethod
    def _is_terminated(self) -> bool:
        ...

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._keyframe_id())
        self._reset_noise(self.np_random)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._apply_action(action)
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward = self._get_reward(action)
        terminated = self._is_terminated()
        truncated = self._step_count >= self._episode_length
        self._last_action = action.copy()
        return obs, reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
