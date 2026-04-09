"""Hermes Agent <-> DIMOS integration.

Exposes a single Go1 quadruped (in MuJoCo simulation) and its skills
via DIMOS's built-in MCP server. Hermes connects over HTTP and uses
its own LLM to drive the robot — DIMOS provides the embodiment.
"""
