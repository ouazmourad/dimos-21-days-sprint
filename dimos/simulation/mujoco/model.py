#!/usr/bin/env python3

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


from pathlib import Path
import xml.etree.ElementTree as ET

from etils import epath
import mujoco
from mujoco_playground._src import mjx_env
import numpy as np

from dimos.core.global_config import GlobalConfig
from dimos.mapping.occupancy.extrude_occupancy import generate_mujoco_scene
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.simulation.mujoco.input_controller import InputController
from dimos.simulation.mujoco.policy import G1OnnxController, Go1OnnxController, OnnxController
from dimos.utils.data import get_data


def _get_data_dir() -> epath.Path:
    return epath.Path(str(get_data("mujoco_sim")))


# Wooden toy skin — hides Go1 visual meshes entirely and inserts
# procedural sphere/capsule geoms arranged to match a wooden-toy-dog
# silhouette (rounded head with eyes, barrel body, tail, tapered legs).
# Physics / collision / control are untouched.
_WOOD_MAT_BODY = "wood_skin"
_WOOD_MAT_FEET = "wood_skin_dark"
_WOOD_MAT_EYE = "wood_skin_eye"
_WOOD_TEX_NAME = "wood_oak_tex"
_WOOD_TEX_FILE = "wood_oak.png"
_WOODEN_GO1_XML_NAME = "unitree_go1_wooden.xml"


def _inject_wooden_assets(root: ET.Element) -> None:
    """Add the wood texture + three oak materials to the scene XML asset
    block. One light-oak material for the body, one darker oak for the
    feet, one black for the eyes.
    """
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    if asset.find(f"texture[@name='{_WOOD_TEX_NAME}']") is None:
        ET.SubElement(
            asset,
            "texture",
            name=_WOOD_TEX_NAME,
            type="2d",
            file=_WOOD_TEX_FILE,
        )
    if asset.find(f"material[@name='{_WOOD_MAT_BODY}']") is None:
        ET.SubElement(
            asset,
            "material",
            name=_WOOD_MAT_BODY,
            texture=_WOOD_TEX_NAME,
            texrepeat="2 2",
            texuniform="true",
            specular="0.2",
            shininess="0.4",
            reflectance="0.04",
            rgba="1 0.85 0.6 1",
        )
    if asset.find(f"material[@name='{_WOOD_MAT_FEET}']") is None:
        ET.SubElement(
            asset,
            "material",
            name=_WOOD_MAT_FEET,
            texture=_WOOD_TEX_NAME,
            texrepeat="1 1",
            texuniform="true",
            specular="0.15",
            shininess="0.35",
            reflectance="0.03",
            rgba="0.55 0.35 0.18 1",
        )
    if asset.find(f"material[@name='{_WOOD_MAT_EYE}']") is None:
        ET.SubElement(
            asset,
            "material",
            name=_WOOD_MAT_EYE,
            rgba="0.02 0.02 0.02 1",
            specular="0.4",
            shininess="0.6",
        )


def _find_body_named(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _add_geom(
    parent: ET.Element, **attrs: str,
) -> ET.Element:
    """Add a visual-only (mass=0, no collision) geom to a body."""
    base = {
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
        "mass": "0",
        "density": "0",
    }
    base.update(attrs)
    return ET.SubElement(parent, "geom", base)


def _decorate_trunk(trunk: ET.Element) -> None:
    """Add wooden-toy body parts (barrel body, shoulders, head, eyes,
    tail) to the Go1 trunk body. Positions are in the trunk's local
    frame — +x is forward, +z is up.

    Proportions tuned to match the Kay Bojesen-style wooden toy dog:
    large rounded head with prominent dark eyes, short barrel body,
    visible shoulder/hip joints, small upright tail.
    """
    # Main barrel body (horizontal capsule, trunk forward-axis).
    # Slightly shorter than the Go1 for a more compact toy look.
    _add_geom(
        trunk,
        type="capsule",
        fromto="-0.13 0 0.005 0.10 0 0.005",
        size="0.088",
        material=_WOOD_MAT_BODY,
    )
    # Front shoulder flare — wide cylinder where the front legs attach.
    _add_geom(
        trunk,
        type="cylinder",
        fromto="0.12 -0.055 0 0.12 0.055 0",
        size="0.102",
        material=_WOOD_MAT_BODY,
    )
    # Rear hip flare.
    _add_geom(
        trunk,
        type="cylinder",
        fromto="-0.14 -0.055 0 -0.14 0.055 0",
        size="0.102",
        material=_WOOD_MAT_BODY,
    )
    # Short neck capsule angling up to the head.
    _add_geom(
        trunk,
        type="capsule",
        fromto="0.14 0 0.03 0.19 0 0.12",
        size="0.048",
        material=_WOOD_MAT_BODY,
    )
    # Head — large rounded sphere (dominant feature, ~45% of body length).
    _add_geom(
        trunk,
        type="sphere",
        pos="0.24 0 0.17",
        size="0.14",
        material=_WOOD_MAT_BODY,
    )
    # Eye sockets — slightly raised darker-oak discs behind each eye
    # (the reference toy has distinct eye regions framing the black pupils).
    _add_geom(
        trunk,
        type="sphere",
        pos="0.31 0.09 0.21",
        size="0.045",
        material=_WOOD_MAT_FEET,
    )
    _add_geom(
        trunk,
        type="sphere",
        pos="0.31 -0.09 0.21",
        size="0.045",
        material=_WOOD_MAT_FEET,
    )
    # Eyes — prominent dark pupils on the front of the head.
    _add_geom(
        trunk,
        type="sphere",
        pos="0.338 0.092 0.215",
        size="0.028",
        material=_WOOD_MAT_EYE,
    )
    _add_geom(
        trunk,
        type="sphere",
        pos="0.338 -0.092 0.215",
        size="0.028",
        material=_WOOD_MAT_EYE,
    )
    # Nose — tiny dark sphere at the front of the snout.
    _add_geom(
        trunk,
        type="sphere",
        pos="0.375 0 0.16",
        size="0.012",
        material=_WOOD_MAT_EYE,
    )
    # Tail — short tapered capsule with a ball on top, angled back-up.
    _add_geom(
        trunk,
        type="capsule",
        fromto="-0.175 0 0.06 -0.215 0 0.16",
        size="0.018",
        material=_WOOD_MAT_BODY,
    )
    _add_geom(
        trunk,
        type="sphere",
        pos="-0.222 0 0.18",
        size="0.034",
        material=_WOOD_MAT_BODY,
    )


def _decorate_hip(hip: ET.Element) -> None:
    """Shoulder/hip ball at each leg-to-body joint."""
    _add_geom(
        hip,
        type="sphere",
        pos="0 0 0",
        size="0.055",
        material=_WOOD_MAT_BODY,
    )


def _decorate_thigh(thigh: ET.Element) -> None:
    """Upper leg — tapered capsule along the thigh's local -z axis."""
    _add_geom(
        thigh,
        type="capsule",
        fromto="0 0 0 0 0 -0.2",
        size="0.038",
        material=_WOOD_MAT_BODY,
    )
    # Knee ball at the bottom of the thigh
    _add_geom(
        thigh,
        type="sphere",
        pos="0 0 -0.21",
        size="0.042",
        material=_WOOD_MAT_BODY,
    )


def _decorate_calf(calf: ET.Element) -> None:
    """Lower leg — tapered capsule plus a darker foot disk at the tip."""
    _add_geom(
        calf,
        type="capsule",
        fromto="0 0 0 0 0 -0.2",
        size="0.03",
        material=_WOOD_MAT_BODY,
    )
    # Foot — slightly oversized flat disk in darker oak
    _add_geom(
        calf,
        type="cylinder",
        fromto="0 0 -0.207 0 0 -0.222",
        size="0.04",
        material=_WOOD_MAT_FEET,
    )


def _build_wooden_toy_go1_xml() -> bytes:
    """Read the Unitree Go1 XML, strip every visual mesh geom, and add
    procedural wooden-toy-dog geoms to the trunk + each leg body. The
    result is a self-contained XML document that MuJoCo can load via
    ``<include file="unitree_go1_wooden.xml"/>`` — the file is served
    from the assets dict, so no file is written to disk.

    Collision geoms are preserved untouched, so physics and the walking
    policy continue to work as before.
    """
    go1_path = _get_data_dir() / "unitree_go1.xml"
    root = ET.fromstring(go1_path.read_text())
    root.set("model", "unitree_go1_wooden")

    # Hide every visual-class mesh geom — we'll replace them with
    # procedural wooden toy shapes below.
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("class") == "visual":
                body.remove(geom)

    # Decorate the trunk with the main body + head + eyes + tail.
    trunk = _find_body_named(root, "trunk")
    if trunk is not None:
        _decorate_trunk(trunk)

    # Decorate each leg with shoulder ball, thigh capsule, calf capsule,
    # and a foot disk.
    for leg in ("FR", "FL", "RR", "RL"):
        hip = _find_body_named(root, f"{leg}_hip")
        thigh = _find_body_named(root, f"{leg}_thigh")
        calf = _find_body_named(root, f"{leg}_calf")
        if hip is not None:
            _decorate_hip(hip)
        if thigh is not None:
            _decorate_thigh(thigh)
        if calf is not None:
            _decorate_calf(calf)

    return ET.tostring(root, encoding="utf-8", xml_declaration=False)


def _apply_wooden_skin(model: mujoco.MjModel) -> None:
    """No-op retained for API compatibility — the wooden skin is now
    applied at XML composition time via :func:`_build_wooden_toy_go1_xml`
    and :func:`_inject_wooden_assets`, so nothing needs to happen after
    the model is loaded.
    """
    _ = model


def get_assets() -> dict[str, bytes]:
    data_dir = _get_data_dir()
    assets: dict[str, bytes] = {}

    # Assets used from https://sketchfab.com/3d-models/mersus-office-8714be387bcd406898b2615f7dae3a47
    # Created by Ryan Cassidy and Coleman Costello
    mjx_env.update_assets(assets, data_dir, "*.xml")
    mjx_env.update_assets(assets, data_dir / "scene_office1/textures", "*.png")
    mjx_env.update_assets(assets, data_dir / "scene_office1/office_split", "*.obj")
    mjx_env.update_assets(assets, mjx_env.MENAGERIE_PATH / "unitree_go1" / "assets")
    mjx_env.update_assets(assets, mjx_env.MENAGERIE_PATH / "unitree_g1" / "assets")

    # Wooden-toy skin texture (used when mujoco_wooden_skin is enabled).
    wood_tex_path = data_dir / _WOOD_TEX_FILE
    if wood_tex_path.exists():
        assets[_WOOD_TEX_FILE] = wood_tex_path.read_bytes()

    # Wooden-toy Go1 variant: Go1 XML with meshes hidden and wooden-toy
    # geoms added to each body. Generated on the fly so it never needs
    # to live on disk.
    if (data_dir / "unitree_go1.xml").exists():
        assets[_WOODEN_GO1_XML_NAME] = _build_wooden_toy_go1_xml()

    # From: https://sketchfab.com/3d-models/jeong-seun-34-42956ca979404a038b8e0d3e496160fd
    person_dir = epath.Path(str(get_data("person")))
    mjx_env.update_assets(assets, person_dir, "*.obj")
    mjx_env.update_assets(assets, person_dir, "*.png")

    return assets


def load_model(
    input_device: InputController,
    robot: str,
    scene_xml: str,
    config: GlobalConfig | None = None,
    wooden_skin: bool = True,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    mujoco.set_mjcb_control(None)

    # Honour a caller-level wooden_skin=False even if config says True.
    effective_config = config
    if not wooden_skin and (config is None or config.mujoco_wooden_skin):
        effective_config = GlobalConfig() if config is None else config.model_copy()
        effective_config.mujoco_wooden_skin = False

    xml_string = get_model_xml(robot, scene_xml, config=effective_config)
    model = mujoco.MjModel.from_xml_string(xml_string, assets=get_assets())
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    match robot:
        case "unitree_g1":
            sim_dt = 0.002
        case _:
            sim_dt = 0.005

    ctrl_dt = 0.02
    n_substeps = round(ctrl_dt / sim_dt)
    model.opt.timestep = sim_dt

    params = {
        "policy_path": (_get_data_dir() / f"{robot}_policy.onnx").as_posix(),
        "default_angles": np.array(model.keyframe("home").qpos[7:]),
        "n_substeps": n_substeps,
        "action_scale": 0.5,
        "input_controller": input_device,
        "ctrl_dt": ctrl_dt,
    }

    match robot:
        case "unitree_go1":
            policy: OnnxController = Go1OnnxController(**params)
        case "unitree_g1":
            policy = G1OnnxController(**params, drift_compensation=[-0.18, 0.0, -0.09])
        case _:
            raise ValueError(f"Unknown robot policy: {robot}")

    mujoco.set_mjcb_control(policy.get_control)

    return model, data


def get_model_xml(robot: str, scene_xml: str, config: GlobalConfig | None = None) -> str:
    root = ET.fromstring(scene_xml)
    root.set("model", f"{robot}_scene")

    # If the wooden skin is enabled for the Go1, include the generated
    # wooden-toy variant of the Go1 XML instead of the stock one. The
    # variant has the visual meshes stripped and wooden-toy geoms added
    # (head, eyes, tail, rounded body/legs) — see get_assets().
    wood_enabled = (config is None or config.mujoco_wooden_skin) and robot == "unitree_go1"
    include_file = _WOODEN_GO1_XML_NAME if wood_enabled else f"{robot}.xml"
    root.insert(0, ET.Element("include", file=include_file))

    # Ensure visual/map element exists with znear and zfar
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    map_elem = visual.find("map")
    if map_elem is None:
        map_elem = ET.SubElement(visual, "map")
    map_elem.set("znear", "0.01")
    map_elem.set("zfar", "10000")

    # Inject shadowsize from config
    if config is not None:
        quality_elem = visual.find("quality")
        if quality_elem is None:
            quality_elem = ET.SubElement(visual, "quality")
        quality_elem.set("shadowsize", str(config.mujoco_shadowsize))

    if config is None or config.mujoco_person:
        _add_person_object(root)

    # Inject wooden skin texture + materials into the scene asset block.
    if wood_enabled:
        _inject_wooden_assets(root)

    return ET.tostring(root, encoding="unicode")


def _add_person_object(root: ET.Element) -> None:
    asset = root.find("asset")

    if asset is None:
        asset = ET.SubElement(root, "asset")

    ET.SubElement(asset, "mesh", name="person_mesh", file="jeong_seun_34.obj")
    ET.SubElement(asset, "texture", name="person_texture", file="material_0.png", type="2d")
    ET.SubElement(asset, "material", name="person_material", texture="person_texture")

    worldbody = root.find("worldbody")

    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    person_body = ET.SubElement(worldbody, "body", name="person", pos="0 0 0", mocap="true")

    ET.SubElement(
        person_body,
        "geom",
        type="mesh",
        mesh="person_mesh",
        material="person_material",
        euler="1.5708 0 0",
    )


def load_scene_xml(config: GlobalConfig) -> str:
    if config.mujoco_room_from_occupancy:
        path = Path(config.mujoco_room_from_occupancy)
        return generate_mujoco_scene(OccupancyGrid.from_path(path))

    mujoco_room = config.mujoco_room or "office1"
    xml_file = (_get_data_dir() / f"scene_{mujoco_room}.xml").as_posix()
    with open(xml_file) as f:
        return f.read()
