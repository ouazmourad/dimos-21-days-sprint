"""Articulated expressive head + appendages for the sim Go2 (option B).

A real Go2 has no face. To give the animator somewhere to put
expression, we replace the wooden-toy's *static* head with a fully
articulated character rig:

  * neck yaw + pitch          — head turns independently of the body
  * per-eye eyeballs          — pupils rotate inside the sclera
                                (eyes lead, head follows — saccades)
  * hinged eyelids            — blinks, sleepy half-lids
  * hinged brows              — raise (curious) / furrow (focused)
  * hinged ears               — perk when alert, droop when shy,
                                flop with head motion
  * jointed tail (yaw+pitch)  — wags when happy, droops when timid

All near-zero-mass, collision-free visual joints driven kinematically
by the orchestrator — they never touch physics or the walking gait.

This is honest only if the *real* Bingo-class hardware also has an
expressive head. It is a sim stand-in for that hardware, not a claim
that a stock Go2 can do this.

Public API:
    compose_animator_go1_xml(scene_xml, base_assets)
        -> (xml_string, assets, head_joints)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# All expressive joints, in the order the retargeter emits them.
HEAD_JOINTS: tuple[str, ...] = (
    "neck_yaw",
    "neck_pitch",
    "lid_l",
    "lid_r",
    "brow_l",
    "brow_r",
    "eye_l_yaw",
    "eye_l_pitch",
    "eye_r_yaw",
    "eye_r_pitch",
    "ear_l",
    "ear_r",
    "tail_yaw",
    "tail_pitch",
)

# Static geoms added by model._decorate_trunk that we strip and replace,
# matched by their (fixed) sizes. Spheres: head 0.14, sockets 0.045,
# pupils 0.028, nose 0.012, tail ball 0.034. Capsule: tail 0.018.
_STATIC_SPHERE_SIZES = {"0.14", "0.045", "0.028", "0.012", "0.034"}
_STATIC_CAPSULE_SIZES = {"0.018"}

# Eye centre in head-local frame (+x forward, +y left, +z up).
_EYE_X, _EYE_Z, _EYE_R = 0.116, 0.028, 0.050


def _white_material(asset: ET.Element) -> None:
    if asset.find("material[@name='eye_white']") is None:
        ET.SubElement(
            asset, "material", name="eye_white",
            rgba="0.96 0.96 0.93 1", specular="0.35", shininess="0.55",
        )


def _strip_static_decorations(trunk: ET.Element) -> None:
    """Remove the static head spheres AND the static tail capsule+ball."""
    for geom in list(trunk.findall("geom")):
        gtype = geom.get("type")
        size = (geom.get("size") or "").split()[0] if geom.get("size") else ""
        if gtype == "sphere" and size in _STATIC_SPHERE_SIZES:
            trunk.remove(geom)
        elif gtype == "capsule" and size in _STATIC_CAPSULE_SIZES:
            trunk.remove(geom)


def _tiny_inertial(body: ET.Element, mass: str = "0.0001") -> None:
    """Kinematically-driven bodies still need mass > mjMINVAL to compile."""
    ET.SubElement(
        body, "inertial", pos="0 0 0", mass=mass,
        diaginertia="1e-7 1e-7 1e-7",
    )


def _vis_geom(parent: ET.Element, **attrs: str) -> None:
    base = {"contype": "0", "conaffinity": "0", "group": "1", "mass": "0"}
    base.update(attrs)
    ET.SubElement(parent, "geom", base)


def _add_eye(head: ET.Element, side: str, y: float) -> None:
    """Sclera + rotating eyeball + hinged eyelid + hinged brow."""
    ex, ez, er = _EYE_X, _EYE_Z, _EYE_R

    # Sclera (white) — static, set into the front of the head.
    _vis_geom(head, type="sphere", pos=f"{ex} {y} {ez}", size=f"{er}",
              material="eye_white")

    # Eyeball — a body pivoting at the sclera centre. The pupil geom is
    # offset forward, so yaw/pitch sweeps it across the sclera surface
    # like a real eye. This is THE channel that makes gaze read as alive.
    eye = ET.SubElement(head, "body", name=f"eye_{side}",
                        pos=f"{ex} {y} {ez}")
    _tiny_inertial(eye)
    ET.SubElement(eye, "joint", name=f"eye_{side}_yaw", type="hinge",
                  axis="0 0 1", range="-0.55 0.55", limited="true",
                  damping="0", stiffness="0")
    ET.SubElement(eye, "joint", name=f"eye_{side}_pitch", type="hinge",
                  axis="0 1 0", range="-0.45 0.45", limited="true",
                  damping="0", stiffness="0")
    _vis_geom(eye, type="sphere", pos="0.040 0 0", size="0.0235",
              material="wood_skin_eye")

    # Upper eyelid — hinged wood cap that sweeps down over the eye.
    lid = ET.SubElement(head, "body", name=f"eyelid_{side}",
                        pos=f"{ex - 0.005} {y} {ez + 0.052}")
    _tiny_inertial(lid)
    ET.SubElement(lid, "joint", name=f"lid_{side}", type="hinge",
                  axis="0 1 0", range="-0.5 1.6", limited="true",
                  damping="0", stiffness="0")
    _vis_geom(lid, type="ellipsoid", pos="0.03 0 -0.005",
              size="0.054 0.052 0.018", material="wood_skin")

    # Brow — flat wood bar above the eye, on the front of the head.
    brow = ET.SubElement(head, "body", name=f"brow_{side}",
                         pos=f"{ex + 0.02} {y} {ez + 0.062}")
    _tiny_inertial(brow)
    ET.SubElement(brow, "joint", name=f"brow_{side}", type="hinge",
                  axis="0 1 0", range="-0.6 0.6", limited="true",
                  damping="0", stiffness="0")
    _vis_geom(brow, type="box", pos="0 0 0", size="0.016 0.042 0.008",
              material="wood_skin_dark")


def _add_ear(head: ET.Element, side: str, y: float) -> None:
    """A floppy ear on top of the head. Hinge axis y: rotating forward
    (negative) perks the ear up-forward; positive flops it back-down."""
    ear = ET.SubElement(head, "body", name=f"ear_{side}",
                        pos=f"0.015 {y} 0.118")
    _tiny_inertial(ear)
    ET.SubElement(ear, "joint", name=f"ear_{side}", type="hinge",
                  axis="0 1 0", range="-0.5 1.4", limited="true",
                  damping="0", stiffness="0")
    # Ear shape: capsule pointing up with a slight outward cant.
    out = 0.018 if side == "l" else -0.018
    _vis_geom(ear, type="capsule",
              fromto=f"0 0 0 -0.012 {out} 0.085", size="0.024",
              material="wood_skin_dark")
    _vis_geom(ear, type="sphere", pos=f"-0.012 {out} 0.09", size="0.027",
              material="wood_skin")


def _build_tail_body() -> ET.Element:
    """Jointed tail mounted at the rear of the trunk: yaw wags, pitch
    sets carriage (up = proud, tucked-down = timid)."""
    tail = ET.Element("body", name="tail", pos="-0.19 0 0.06")
    _tiny_inertial(tail)
    ET.SubElement(tail, "joint", name="tail_yaw", type="hinge",
                  axis="0 0 1", range="-0.8 0.8", limited="true",
                  damping="0", stiffness="0")
    ET.SubElement(tail, "joint", name="tail_pitch", type="hinge",
                  axis="0 1 0", range="-0.5 1.0", limited="true",
                  damping="0", stiffness="0")
    _vis_geom(tail, type="capsule",
              fromto="0 0 0 -0.10 0 0.10", size="0.018",
              material="wood_skin")
    _vis_geom(tail, type="sphere", pos="-0.115 0 0.115", size="0.034",
              material="wood_skin")
    return tail


def _build_head_body() -> ET.Element:
    head = ET.Element("body", name="head", pos="0.24 0 0.17")
    _tiny_inertial(head, mass="0.001")
    ET.SubElement(head, "joint", name="neck_yaw", type="hinge",
                  axis="0 0 1", range="-0.9 0.9", limited="true",
                  damping="0", stiffness="0")
    ET.SubElement(head, "joint", name="neck_pitch", type="hinge",
                  axis="0 1 0", range="-0.6 0.6", limited="true",
                  damping="0", stiffness="0")
    _vis_geom(head, type="sphere", pos="0 0 0", size="0.14",
              material="wood_skin")
    # Snout / nose.
    _vis_geom(head, type="sphere", pos="0.142 0 -0.02", size="0.013",
              material="wood_skin_eye")
    _add_eye(head, "l", 0.062)
    _add_eye(head, "r", -0.062)
    _add_ear(head, "l", 0.085)
    _add_ear(head, "r", -0.085)
    return head


def compose_animator_go1_xml(
    scene_xml: str,
    base_assets: dict[str, bytes],
) -> tuple[str, dict[str, bytes], tuple[str, ...]]:
    """Build the animator's Go1 scene with the articulated character rig."""
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
    _strip_static_decorations(trunk)
    trunk.append(_build_head_body())
    trunk.append(_build_tail_body())

    animator_xml_name = "unitree_go1_animator.xml"
    assets = dict(base_assets)
    assets[animator_xml_name] = ET.tostring(wooden_root, encoding="utf-8")

    scene_root = ET.fromstring(scene_xml)
    scene_root.set("model", "animator_scene")
    scene_root.insert(0, ET.Element("include", file=animator_xml_name))

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
