"""Multi-robot fleet orchestration for DIMOS.

Provides a thin layer over :func:`autoconnect` and :meth:`Blueprint.remappings`
that automatically creates per-robot module namespaces, eliminating the need to
hand-write subclasses and stream remappings for every robot.

Usage::

    from dimos.core.fleet import fleet, RobotConfig, SharedModule

    bp = fleet(
        robots=[
            RobotConfig("alpha", G1SimConnection, modules=[VLMAgent, NavSkill]),
            RobotConfig("bravo", G1SimConnection, modules=[VLMAgent, NavSkill]),
        ],
        shared=[SharedModule(RadioBridge)],
    )
    coordinator = bp.build()

Under the hood this:
1. Creates unique subclasses per robot (``Alpha_G1SimConnection``, etc.)
2. Generates stream remappings with robot-name prefixes (``odom`` → ``alpha/odom``)
3. Wires everything through the standard :func:`autoconnect` pipeline
"""

from __future__ import annotations

import copyreg
import sys
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from dimos.core.blueprints import Blueprint, autoconnect
from dimos.core.module import Module
from dimos.core.stream import In, Out


# ─── Public dataclasses ───────────────────────────────────────────


@dataclass(frozen=True)
class RobotConfig:
    """Configuration for a single robot in a fleet.

    Args:
        name: Unique robot identifier (e.g. ``"alpha"``). Used as the
            stream-name prefix.
        connection: The connection module class (e.g. ``G1SimConnection``).
        modules: Per-robot module classes that will be namespaced alongside
            the connection (e.g. ``[VLMAgent, NavSkill, RadioSkill]``).
        args: Positional arguments forwarded to each module's blueprint.
        kwargs: Keyword arguments forwarded to the *connection* blueprint.
    """

    name: str
    connection: type[Module]
    modules: list[type[Module]] = field(default_factory=list)
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SharedModule:
    """A module shared across the entire fleet (not namespaced).

    Args:
        module: The module class (e.g. ``MissionCoordinator``).
        args: Positional args for the blueprint.
        kwargs: Keyword args for the blueprint.
    """

    module: type[Module]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


# ─── Internals ────────────────────────────────────────────────────


# Cache to avoid recreating the same subclass.
_fleet_cache: dict[tuple[str, type], type[Module]] = {}

# Track which metaclasses have been registered with copyreg.
_registered_metas: set[type] = set()


def _reduce_fleet_class(cls: type) -> tuple:
    """copyreg reducer — tells pickle how to reconstruct a fleet class."""
    return (_namespace_class, (cls._fleet_robot_name, cls._fleet_base))


def _namespace_class(robot_name: str, base: type[Module]) -> type[Module]:
    """Create a uniquely-named subclass of *base* for *robot_name*.

    Uses a custom metaclass + ``copyreg`` so the class can be pickled
    across worker-process boundaries.  When unpickling, pickle calls
    ``_namespace_class(robot_name, base)`` which recreates the class.
    """
    key = (robot_name, base)
    if key in _fleet_cache:
        return _fleet_cache[key]

    cls_name = f"{robot_name.capitalize()}_{base.__name__}"

    # Build a metaclass compatible with the base's metaclass
    base_meta = type(base)

    class _FleetMeta(base_meta):
        pass

    _FleetMeta.__name__ = f"_FleetMeta_{cls_name}"
    _FleetMeta.__qualname__ = f"_FleetMeta_{cls_name}"

    # Create the dynamic subclass
    new_cls: type[Module] = _FleetMeta(  # type: ignore[assignment]
        cls_name,
        (base,),
        {
            "_fleet_robot_name": robot_name,
            "_fleet_base": base,
        },
    )
    new_cls.__module__ = base.__module__

    # Register copyreg reducer for this metaclass so pickle calls
    # _reduce_fleet_class instead of doing module+name lookup.
    if _FleetMeta not in _registered_metas:
        copyreg.pickle(_FleetMeta, _reduce_fleet_class)
        _registered_metas.add(_FleetMeta)

    _fleet_cache[key] = new_cls
    return new_cls


def _stream_names(module_cls: type[Module]) -> list[str]:
    """Return the names of all ``In[T]`` / ``Out[T]`` streams on *module_cls*."""
    names: list[str] = []

    # Walk MRO to collect annotations (same logic as _BlueprintAtom.create).
    globalns: dict[str, Any] = {}
    for c in reversed(module_cls.__mro__):
        if c.__module__ in sys.modules:
            globalns.update(sys.modules[c.__module__].__dict__)

    try:
        from typing import get_type_hints
        all_annotations = get_type_hints(module_cls, globalns=globalns)
    except Exception:
        all_annotations = {}
        for base in reversed(module_cls.__mro__):
            if hasattr(base, "__annotations__"):
                all_annotations.update(base.__annotations__)

    for name, ann in all_annotations.items():
        origin = get_origin(ann)
        if origin in (In, Out):
            names.append(name)

    return names


def _make_remappings(
    robot_name: str,
    namespaced_cls: type[Module],
    original_cls: type[Module],
) -> list[tuple[type[Module], str, str]]:
    """Generate ``(NamespacedClass, stream, "robot/stream")`` remappings."""
    return [
        (namespaced_cls, s, f"{robot_name}/{s}")
        for s in _stream_names(original_cls)
    ]


# ─── Public API ───────────────────────────────────────────────────


def fleet(
    robots: list[RobotConfig],
    shared: list[SharedModule] | None = None,
) -> Blueprint:
    """Build a multi-robot :class:`Blueprint` with automatic namespacing.

    For each robot in *robots*, every module (connection + per-robot modules)
    is given a unique dynamic subclass and its streams are prefixed with the
    robot's name.  Shared modules are included as-is with no prefix.

    Args:
        robots: One :class:`RobotConfig` per robot.
        shared: Optional list of :class:`SharedModule` instances that are
            wired globally (not namespaced).

    Returns:
        A standard :class:`Blueprint` ready for ``.build()``.

    Raises:
        ValueError: If robot names are not unique.
    """
    # ── Validate ──
    names = [r.name for r in robots]
    if len(names) != len(set(names)):
        dupes = [n for n in names if names.count(n) > 1]
        raise ValueError(f"Duplicate robot names: {set(dupes)}")

    all_blueprints: list[Blueprint] = []
    all_remappings: list[tuple[type[Module], str, str]] = []

    # ── Per-robot modules ──
    for robot in robots:
        # Namespace the connection class
        ns_conn = _namespace_class(robot.name, robot.connection)
        bp = ns_conn.blueprint(*robot.args, **robot.kwargs)  # type: ignore[attr-defined]
        all_blueprints.append(bp)
        all_remappings.extend(
            _make_remappings(robot.name, ns_conn, robot.connection)
        )

        # Namespace each per-robot module
        for mod_cls in robot.modules:
            ns_mod = _namespace_class(robot.name, mod_cls)
            bp_mod = ns_mod.blueprint()  # type: ignore[attr-defined]
            all_blueprints.append(bp_mod)
            all_remappings.extend(
                _make_remappings(robot.name, ns_mod, mod_cls)
            )

    # ── Shared modules (no namespace prefix) ──
    for sm in shared or []:
        bp_shared = sm.module.blueprint(*sm.args, **sm.kwargs)  # type: ignore[attr-defined]
        all_blueprints.append(bp_shared)

    # ── Assemble ──
    combined = autoconnect(*all_blueprints)
    if all_remappings:
        combined = combined.remappings(all_remappings)

    return combined


__all__ = ["RobotConfig", "SharedModule", "fleet"]
