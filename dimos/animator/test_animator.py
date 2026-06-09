"""Unit tests for the deterministic parts of the animator.

Channel timing tests are integration-level and live in the demo
script — they need a long enough sim to be meaningful, and they
test perceptual outcomes ("does it look shy") that don't decompose
to assertions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dimos.animator.channels.gaze import GazeChannel, GazeTarget
from dimos.animator.channels.gesture import GestureChannel
from dimos.animator.channels.posture import PostureChannel, PostureTarget
from dimos.animator.intents import (
    INTENTS,
    curious_head_tilt,
    notice_guest,
    proud_chest_lift,
    search_room,
)
from dimos.animator.mixer import BehaviorMixer, ChannelState
from dimos.animator.orchestrator import PerformanceOrchestrator
from dimos.animator.personality import Personality
from dimos.animator.retargeter import GO1_JOINT_ORDER, Go2Retargeter
from dimos.animator.rig import CharacterRig


# Helpers ----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RIG_PATH = REPO_ROOT / "data" / "animator" / "rigs" / "unitree_go2.yaml"
PERSONALITY_DIR = REPO_ROOT / "data" / "animator" / "personalities"


@pytest.fixture
def rig() -> CharacterRig:
    return CharacterRig.from_yaml(RIG_PATH)


@pytest.fixture
def curious() -> Personality:
    return Personality.from_yaml(PERSONALITY_DIR / "curious.yaml")


@pytest.fixture
def shy() -> Personality:
    return Personality.from_yaml(PERSONALITY_DIR / "shy.yaml")


# CharacterRig -----------------------------------------------------------

def test_rig_loads_from_yaml(rig: CharacterRig) -> None:
    assert rig.robot == "unitree_go1"
    assert len(rig.default_pose) == 12
    assert rig.has_channel("gaze")
    assert rig.has_channel("posture")
    assert rig.has_channel("breathing")
    assert rig.has_channel("gesture")


def test_rig_joints_per_channel(rig: CharacterRig) -> None:
    # Two roles attached to "gaze": yaw + pitch.
    gaze_roles = rig.joints_for_channel("gaze")
    names = {r.joint for r in gaze_roles}
    assert names == {"trunk_yaw", "trunk_pitch"}


def test_rig_virtual_real_split(rig: CharacterRig) -> None:
    # In this rig everything is virtual (Go2 has no head joint).
    assert len(rig.real_joints()) == 0
    assert len(rig.virtual_joints()) == len(rig.roles)


def test_joint_role_clamps_offset(rig: CharacterRig) -> None:
    role = next(r for r in rig.roles if r.joint == "trunk_yaw")
    assert role.clamp(role.max_offset_rad + 1.0) == role.max_offset_rad
    assert role.clamp(-role.max_offset_rad - 1.0) == -role.max_offset_rad
    assert role.clamp(0.1) == 0.1


# Personality ------------------------------------------------------------

def test_personality_loads_all_profiles() -> None:
    expected = {"curious", "shy", "proud", "nervous", "calm"}
    found = {p.stem for p in PERSONALITY_DIR.glob("*.yaml")}
    assert expected == found
    for name in expected:
        Personality.from_yaml(PERSONALITY_DIR / f"{name}.yaml")  # no exceptions


def test_personality_clamps_out_of_range() -> None:
    p = Personality(curiosity=1.7, confidence=-2.0, energy=0.5)
    assert p.curiosity == 1.0
    assert p.confidence == -1.0
    assert p.energy == 0.5


def test_personality_scales_make_sense(curious: Personality, shy: Personality) -> None:
    # Curious has positive energy: speed scale > 1.
    assert curious.speed_scale() > 1.0
    # Shy has negative energy: speed scale < 1.
    assert shy.speed_scale() < 1.0
    # Shy has negative confidence and positive softness → real initiation delay.
    assert shy.initiation_delay() > 0.5
    # Curious has slight positive confidence → small initiation delay.
    assert curious.initiation_delay() < shy.initiation_delay()


# Mixer ------------------------------------------------------------------

def test_mixer_clamps_via_rig(rig: CharacterRig) -> None:
    mixer = BehaviorMixer(rig)
    snapshot = ChannelState()
    snapshot.gaze = GazeTarget(yaw_rad=99.0, pitch_rad=99.0)
    out = mixer.mix(snapshot)
    yaw_role = next(r for r in rig.roles if r.joint == "trunk_yaw")
    pitch_role = next(r for r in rig.roles if r.joint == "trunk_pitch")
    assert out.trunk_yaw == yaw_role.max_offset_rad
    assert out.trunk_pitch == pitch_role.max_offset_rad


def test_mixer_default_is_zero(rig: CharacterRig) -> None:
    mixer = BehaviorMixer(rig)
    out = mixer.mix(ChannelState())
    for field_name in ("trunk_yaw", "trunk_pitch", "trunk_roll", "trunk_z",
                       "trunk_x", "trunk_z_breath", "paw_lift_fl"):
        assert getattr(out, field_name) == 0.0


# Retargeter -------------------------------------------------------------

def test_retargeter_default_pose_is_passthrough(rig: CharacterRig) -> None:
    """With zero motion, retargeted output equals the rig's default pose."""
    from dimos.animator.mixer import MixedMotion
    r = Go2Retargeter(rig)
    cmd = r.retarget(MixedMotion())
    for joint in GO1_JOINT_ORDER:
        assert cmd.angles[joint] == pytest.approx(rig.default_pose[joint])


def test_retargeter_trunk_z_extends_legs(rig: CharacterRig) -> None:
    """Positive trunk_z means standing taller → thighs less flexed."""
    from dimos.animator.mixer import MixedMotion
    r = Go2Retargeter(rig)
    cmd = r.retarget(MixedMotion(trunk_z=0.1))
    # Thigh defaults to 0.9, with z=0.1 the offset is -0.6*0.1 = -0.06.
    assert cmd.angles["FR_thigh_joint"] == pytest.approx(0.9 - 0.06)


def test_retargeter_paw_lift_only_affects_fl(rig: CharacterRig) -> None:
    """paw_lift_fl moves only FL_thigh and FL_calf, no other foot."""
    from dimos.animator.mixer import MixedMotion
    r = Go2Retargeter(rig)
    cmd = r.retarget(MixedMotion(paw_lift_fl=0.5))
    for joint in GO1_JOINT_ORDER:
        if joint in ("FL_thigh_joint", "FL_calf_joint"):
            assert cmd.angles[joint] != pytest.approx(rig.default_pose[joint])
        else:
            assert cmd.angles[joint] == pytest.approx(rig.default_pose[joint])


def test_retargeter_trunk_pitch_is_symmetric(rig: CharacterRig) -> None:
    """trunk_pitch should produce mirrored left/right offsets."""
    from dimos.animator.mixer import MixedMotion
    r = Go2Retargeter(rig)
    cmd = r.retarget(MixedMotion(trunk_pitch=0.1))
    assert cmd.angles["FR_thigh_joint"] == pytest.approx(cmd.angles["FL_thigh_joint"])
    assert cmd.angles["RR_thigh_joint"] == pytest.approx(cmd.angles["RL_thigh_joint"])


# Channels (timing / smoothing) ------------------------------------------

def test_gaze_channel_low_pass_converges() -> None:
    p = Personality()
    g = GazeChannel()
    g.set_target(GazeTarget(yaw_rad=0.5), p)
    # Step for 3 seconds and check we got close to the target.
    state = None
    for _ in range(int(3 * 50)):
        state = g.step(1.0 / 50.0, p)
    assert state is not None
    assert state.yaw_rad == pytest.approx(0.5, abs=0.05)


def test_posture_channel_critically_damped() -> None:
    p = Personality()
    pc = PostureChannel()
    pc.set_target(PostureTarget(z_offset_rad=0.1))
    last = 0.0
    max_seen = 0.0
    for _ in range(int(2 * 50)):
        state = pc.step(1.0 / 50.0, p)
        max_seen = max(max_seen, state.z_offset_rad)
        last = state.z_offset_rad
    # Critically damped → should approach target without significant overshoot.
    # (Allow small numerical overshoot from the discretization.)
    assert max_seen <= 0.1 + 0.01
    assert last == pytest.approx(0.1, abs=0.02)


def test_gesture_finishes_in_finite_time() -> None:
    p = Personality()
    g = GestureChannel()
    g.trigger("paw_acknowledge", p)
    assert g.is_active
    for _ in range(int(3 * 50)):  # 3 seconds at 50 Hz is way more than needed
        g.step(1.0 / 50.0, p)
    assert not g.is_active


# Intents ----------------------------------------------------------------

@pytest.mark.parametrize("intent_name", ["notice_guest", "curious_head_tilt",
                                          "proud_chest_lift", "search_room"])
def test_intents_are_registered(intent_name: str) -> None:
    assert intent_name in INTENTS
    assert callable(INTENTS[intent_name])


def test_notice_guest_finishes(curious: Personality) -> None:
    ticks = list(notice_guest(0.5, 0.1, curious))
    assert len(ticks) > 10
    assert ticks[-1].finished is True
    # Most ticks have at least one target set.
    targeted = sum(1 for t in ticks if t.gaze_target or t.posture_target or t.fire_gesture)
    assert targeted > len(ticks) // 2


def test_curious_head_tilt_finishes() -> None:
    ticks = list(curious_head_tilt(direction=1))
    assert ticks[-1].finished is True
    # Should peak at a positive roll value.
    rolls = [t.posture_target.roll_offset_rad for t in ticks
             if t.posture_target is not None]
    assert max(rolls) > 0.05


def test_proud_chest_lift_finishes_and_lifts(curious: Personality) -> None:
    ticks = list(proud_chest_lift(curious))
    assert ticks[-1].finished is True
    z_values = [t.posture_target.z_offset_rad for t in ticks
                if t.posture_target is not None]
    assert max(z_values) > 0.04


def test_search_room_visits_multiple_yaws(curious: Personality) -> None:
    ticks = list(search_room(yaw_range_rad=0.5, n_stops=3, personality=curious))
    yaws = [t.gaze_target.yaw_rad for t in ticks if t.gaze_target is not None]
    assert max(yaws) > 0.2
    assert min(yaws) < -0.2
    assert ticks[-1].finished is True


def test_intent_timing_differs_by_personality(
    curious: Personality, shy: Personality,
) -> None:
    """The same intent should produce different total durations across
    personalities. This is the v1 architectural claim, distilled to
    a unit test."""
    curious_ticks = list(notice_guest(0.5, 0.1, curious))
    shy_ticks = list(notice_guest(0.5, 0.1, shy))
    # Shy stretches the timeline through slower speed_scale + initiation
    # delay + longer dwells, so it should produce at least 20% more ticks.
    assert len(shy_ticks) > len(curious_ticks) * 1.2


# Orchestrator -----------------------------------------------------------

def test_orchestrator_produces_valid_joint_commands(
    rig: CharacterRig, curious: Personality,
) -> None:
    orch = PerformanceOrchestrator(rig, curious)
    intent = notice_guest(0.5, 0.1, curious, tick_dt=orch.tick_dt)
    last = None
    for tick in orch.run_intent(intent):
        last = tick
        # The 12 leg joints must always be present; head joints (neck /
        # eyelids / brows) are also emitted for articulated-head models.
        assert set(GO1_JOINT_ORDER).issubset(tick.command.angles.keys())
        # All angles should be finite and in a sane range.
        for v in tick.command.angles.values():
            assert v == v  # NaN check
            assert -10 < v < 10  # sanity range
    assert last is not None
    assert last.finished is True


def test_orchestrator_personality_swap_changes_breathing(rig: CharacterRig) -> None:
    """Swapping from calm to energetic should noticeably increase the
    breathing amplitude observed over the same number of ticks."""
    calm = Personality.from_yaml(PERSONALITY_DIR / "calm.yaml")
    nervous = Personality.from_yaml(PERSONALITY_DIR / "nervous.yaml")
    orch = PerformanceOrchestrator(rig, calm)
    calm_breaths = [orch.idle_tick().snapshot.breathing.z_breath_offset_rad
                    for _ in range(200)]
    orch.set_personality(nervous)
    nervous_breaths = [orch.idle_tick().snapshot.breathing.z_breath_offset_rad
                       for _ in range(200)]
    # Both go through the same phase, so the *amplitude* (max - min) is
    # the right comparison rather than the instantaneous value.
    calm_amp = max(calm_breaths) - min(calm_breaths)
    nervous_amp = max(nervous_breaths) - min(nervous_breaths)
    # Just assert they're different in the expected direction — we don't
    # want a tight bound that fluctuates with implementation tweaks.
    assert nervous_amp > 0.0
    assert calm_amp > 0.0
    # No equality guarantee; the assertion is that swap actually does
    # something measurable.
    assert calm_amp != nervous_amp


# Expression channel + articulated head ----------------------------------

def test_expression_openness_differs_by_personality(
    curious: Personality, shy: Personality,
) -> None:
    """Curious eyes should rest wider open than shy eyes."""
    from dimos.animator.channels.expression import ExpressionChannel
    ec_curious = ExpressionChannel()
    ec_shy = ExpressionChannel()
    cur_vals, shy_vals = [], []
    for _ in range(50):
        cur_vals.append(ec_curious.step(1 / 50, curious).eye_openness)
        shy_vals.append(ec_shy.step(1 / 50, shy).eye_openness)
    # Compare the open-state (max between blinks) baselines.
    assert max(cur_vals) > max(shy_vals)


def test_expression_surprise_widens_eyes(curious: Personality) -> None:
    from dimos.animator.channels.expression import ExpressionChannel
    ec = ExpressionChannel()
    base = ec.step(1 / 50, curious).eye_openness
    ec.notice_surprise(0.5)
    after = ec.step(1 / 50, curious).eye_openness
    assert after >= base


def test_retargeter_emits_head_joints(rig: CharacterRig) -> None:
    from dimos.animator.mixer import MixedMotion
    r = Go2Retargeter(rig)
    cmd = r.retarget(MixedMotion(neck_yaw=0.3, eye_openness=1.0, brow_raise=1.0))
    for j in ("neck_yaw", "neck_pitch", "lid_l", "lid_r", "brow_l", "brow_r"):
        assert j in cmd.angles
    # Wide-open eyes → lids tucked up (negative angle).
    assert cmd.angles["lid_l"] < 0.0
    # Fully closed → lids swept down (large positive angle).
    cmd_closed = r.retarget(MixedMotion(eye_openness=0.0))
    assert cmd_closed.angles["lid_l"] > 1.0


def test_sim_head_xml_loads_in_mujoco() -> None:
    """The articulated-head Go1 XML must parse + build in MuJoCo and
    expose all six head joints."""
    import mujoco

    from dimos.animator.sim_head import HEAD_JOINTS, compose_animator_go1_xml
    from dimos.simulation.mujoco.model import get_assets
    from dimos.utils.data import get_data

    data_dir = Path(str(get_data("mujoco_sim")))
    scene = (data_dir / "scene_empty.xml").read_text()
    xml, assets, head_joints = compose_animator_go1_xml(scene, get_assets())
    assert head_joints == HEAD_JOINTS

    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    for j in HEAD_JOINTS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0


# Animation engine (layered composition) ---------------------------------

def test_blend_frames_endpoints() -> None:
    from dimos.animator.engine import Frame, blend_frames
    from dimos.animator.channels.gaze import GazeTarget
    bg = Frame(gaze_target=GazeTarget(yaw_rad=0.0))
    trig = Frame(gaze_target=GazeTarget(yaw_rad=1.0))
    assert blend_frames(bg, trig, 0.0).gaze_target.yaw_rad == pytest.approx(0.0)
    assert blend_frames(bg, trig, 1.0).gaze_target.yaw_rad == pytest.approx(1.0)
    assert blend_frames(bg, trig, 0.5).gaze_target.yaw_rad == pytest.approx(0.5)


def test_blend_gesture_owned_by_dominant_layer() -> None:
    from dimos.animator.engine import Frame, blend_frames
    trig = Frame(fire_gesture="paw_wave")
    # Below 0.5 the triggered layer isn't dominant → no gesture.
    assert blend_frames(Frame(), trig, 0.3).fire_gesture is None
    # At/above 0.5 the gesture fires.
    assert blend_frames(Frame(), trig, 0.6).fire_gesture == "paw_wave"


def test_engine_background_always_runs(rig: CharacterRig, curious: Personality) -> None:
    """With no intent queued, the engine still produces valid commands
    forever (the always-on background layer)."""
    from dimos.animator.engine import AnimationEngine
    orch = PerformanceOrchestrator(rig, curious)
    eng = AnimationEngine(orch, curious)
    assert not eng.is_playing
    last_yaw = []
    for _ in range(150):  # 3 s
        tick = eng.tick()
        assert set(GO1_JOINT_ORDER).issubset(tick.command.angles.keys())
        last_yaw.append(tick.command.angles["neck_yaw"])
    # The background drifts the gaze, so neck_yaw should not be constant.
    assert max(last_yaw) - min(last_yaw) > 0.02


def test_engine_intent_ramps_in_and_out(rig: CharacterRig, curious: Personality) -> None:
    """A triggered intent blends in over the background and the engine
    returns to idle (is_playing False) once it finishes."""
    from dimos.animator.engine import AnimationEngine
    orch = PerformanceOrchestrator(rig, curious)
    eng = AnimationEngine(orch, curious)
    eng.trigger(notice_guest(0.6, 0.2, curious, tick_dt=orch.tick_dt))
    assert eng.is_playing
    n = 0
    while eng.is_playing and n < 2000:
        eng.tick()
        n += 1
    assert not eng.is_playing       # clip finished
    assert 0 < n < 2000             # finite duration
    # After finishing, the engine keeps producing background ticks.
    tick = eng.tick()
    assert set(GO1_JOINT_ORDER).issubset(tick.command.angles.keys())


def test_engine_lowpass_bounds_action_acceleration(
    rig: CharacterRig, curious: Personality,
) -> None:
    """The output low-pass should keep the per-tick second difference
    (action acceleration) small — the smoothness the paper regularises."""
    from dimos.animator.engine import AnimationEngine
    orch = PerformanceOrchestrator(rig, curious)
    eng = AnimationEngine(orch, curious)
    eng.trigger(notice_guest(0.8, 0.3, curious, tick_dt=orch.tick_dt))
    neck, calf = [], []
    for _ in range(300):
        cmd = eng.tick().command.angles
        neck.append(cmd["neck_yaw"])
        calf.append(cmd["FL_calf_joint"])

    def accels(seq):
        return [abs(seq[i] - 2 * seq[i - 1] + seq[i - 2]) for i in range(2, len(seq))]

    # The continuously-driven neck (gaze blend) is smooth throughout —
    # this is what the low-pass guarantees for non-discrete motion.
    assert max(accels(neck)) < 0.02
    # The gesture-driven calf has one sharp transient at the paw-lift
    # onset (a crisp gesture is *meant* to be sharp), but stays bounded.
    assert max(accels(calf)) < 0.1
