"""Articulated expressive head for the sim Go2 (option B).

A real Go2 has no face. To give the animator somewhere to put
expression, we replace the wooden-toy's *static* head with an
*articulated* one: a neck (yaw + pitch) so the head turns
independently of the body, plus per-eye hinged eyelids and hinged
brows. These are near-zero-mass, collision-free visual joints driven
kinematically by the orchestrator — they never touch physics or the
walking policy.

This is honest only if the *real* Bingo-class hardware also has an
expressive head (actuated eyes / lids / neck). It is a sim stand-in
for that hardware, not a claim that a stock Go2 can do this.

Public API:
    compose_animator_go1_xml(scene_xml, base_assets)
        -> (xml_string, assets, head_joints)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# Head joints, in the order the retargeter emits them.
HEAD_JOINTS: tuple[str, ...] = (
    "neck_yaw",
    "neck_pitch",
    "lid_l",
    "lid_r",
    "brow_l",
    "brow_r",
)

# Static head geoms added by model._decorate_trunk that we strip out,
# matched by sphere size — these values are fixed in that function.
# (head 0.14, sockets 0.045, pupils 0.028, nose 0.012). The tail ball
# is 0.034, not in this set, so it survives.
_STATIC_HEAD_SIZES = {"0.14", "0.045", "0.028", "0.012"}

# Eye centre in head-local frame (+x forward, +y left, +z up).
_EYE_X, _EYE_Z, _EYE_R = 0.116, 0.028, 0.050


def _white_material(asset: ET.Element) -> None:
    if asset.find("material[@name='eye_white']") is None:
        ET.SubElement(
            asset, "material", name="eye_white",
            rgba="0.96 0.96 0.93 1", specular="0.35", shininess="0.55",
        )


def _strip_static_head(trunk: ET.Element) -> None:
    for geom in list(trunk.findall("geom")):
        if geom.get("type") != "sphere":
            continue
        size = (geom.get("size") or "").split()[0] if geom.get("size") else ""
        if size in _STATIC_HEAD_SIZES:
            trunk.remove(geom)


def _tiny_inertial(body: ET.Element, mass: str = "0.0001") -> None:
    """Kinematically-driven bodies still need mass > mjMINVAL to compile."""
    ET.SubElement(
        body, "inertial", pos="0 0 0", mass=mass,
        diaginertia="1e-7 1e-7 1e-7",
    )


def _add_eye(head: ET.Element, side: str, y: float) -> None:
    """Sclera + pupil + hinged eyelid + hinged brow for one eye."""
    ex, ez, er = _EYE_X, _EYE_Z, _EYE_R

    # Sclera (white) — static, set into the front of the head.
    ET.SubElement(
        head, "geom", type="sphere", pos=f"{ex} {y} {ez}", size=f"{er}",
        material="eye_white", contype="0", conaffinity="0", group="1", mass="0",
    )
    # Pupil — dark sphere slightly forward of the sclera.
    ET.SubElement(
        head, "geom", type="sphere", pos=f"{ex + 0.028} {y} {ez}", size="0.024",
        material="wood_skin_eye", contype="0", conaffinity="0", group="1", mass="0",
    )

    # Upper eyelid. Pivot just above + behind the eye centre. At the rest
    # (open) angle the lid is tucked up behind the eye; rotating +pitch
    # sweeps it forward and down over the sclera.
    lid = ET.SubElement(
        head, "body", name=f"eyelid_{side}", pos=f"{ex - 0.005} {y} {ez + 0.052}",
    )
    _tiny_inertial(lid)
    ET.SubElement(
        lid, "joint", name=f"lid_{side}", type="hinge", axis="0 1 0",
        range="-0.5 1.6", damping="0", stiffness="0", limited="true",
    )
    # Lid cap offset forward + down so a positive rotation covers the eye.
    ET.SubElement(
        lid, "geom", type="ellipsoid", pos="0.03 0 -0.005", size="0.054 0.052 0.018",
        material="wood_skin", contype="0", conaffinity="0", group="1", mass="0",
    )

    # Brow — a flat wood bar sitting just above the eye, hugging the head
    # surface. Pivot above the eye; +pitch raises the outer end.
    brow = ET.SubElement(
        head, "body", name=f"brow_{side}", pos=f"{ex + 0.01} {y} {ez + 0.062}",
    )
    _tiny_inertial(brow)
    ET.SubElement(
        brow, "joint", name=f"brow_{side}", type="hinge", axis="0 1 0",
        range="-0.6 0.6", damping="0", stiffness="0", limited="true",
    )
    ET.SubElement(
        brow, "geom", type="box", pos="0.012 0 0", size="0.042 0.013 0.009",
        material="wood_skin_dark", contype="0", conaffinity="0", group="1", mass="0",
    )


def _build_head_body() -> ET.Element:
    head = ET.Element("body", name="head", pos="0.24 0 0.17")
    _tiny_inertial(head, mass="0.001")
    ET.SubElement(
        head, "joint", name="neck_yaw", type="hinge", axis="0 0 1",
        range="-0.9 0.9", damping="0", stiffness="0", limited="true",
    )
    ET.SubElement(
        head, "joint", name="neck_pitch", type="hinge", axis="0 1 0",
        range="-0.6 0.6", damping="0", stiffness="0", limited="true",
    )
    # Head sphere.
    ET.SubElement(
        head, "geom", type="sphere", pos="0 0 0", size="0.14",
        material="wood_skin", contype="0", conaffinity="0", group="1", mass="0",
    )
    # Snout / nose at the front.
    ET.SubElement(
        head, "geom", type="sphere", pos="0.142 0 -0.02", size="0.013",
        material="wood_skin_eye", contype="0", conaffinity="0", group="1", mass="0",
    )
    _add_eye(head, "l", 0.062)
    _add_eye(head, "r", -0.062)
    return head


def compose_animator_go1_xml(
    scene_xml: str,
    base_assets: dict[str, bytes],
) -> tuple[str, dict[str, bytes], tuple[str, ...]]:
    """Build the animator's Go1 scene with an articulated expressive head."""
    wooden_xml_bytes = base_assets.get("unitree_go1_wooden.xml")
    if wooden_xml_bytes is None:
        raise RuntimeError(
            "unitree_go1_wooden.xml not found in assets — wooden skin must "
            "be enabled (it's generated by model.get_assets())."
        )

    wooden_root = ET.fromstring(wooden_xml_bytes.decode("utf-8"))
    trunk = next((b for b in wooden_root.iter("body") if b.get("name") == "trunk"), None)
    if trunk is None:
        raise RuntimeError("trunk body not found in wooden Go1 XML")
    _strip_static_head(trunk)
    trunk.append(_build_head_body())

    animator_xml_name = "unitree_go1_animator.xml"
    assets = dict(base_assets)
    assets[animator_xml_name] = ET.tostring(wooden_root, encoding="utf-8")

    scene_root = ET.fromstring(scene_xml)
    scene_root.set("model", "animator_scene")
    scene_root.insert(0, ET.Element("include", file=animator_xml_name))

    # Inject wood texture + materials (reuses the production helper), then
    # add the eye-white material.
    from dimos.simulation.mujoco.model import _inject_wooden_assets
    _inject_wooden_assets(scene_root)
    _white_material(scene_root.find("asset"))

    visual = scene_root.find("visual")
    if visual is None:
        visual = ET.SubElement(scene_root, "visual")
    g = visual.find("global")
    if g is None:
        g = ET.SubElement(visual, "global")
    g.set("offwidth", "1280")
    g.set("offheight", "960")

    return ET.tostring(scene_root, encoding="unicode"), assets, HEAD_JOINTS
