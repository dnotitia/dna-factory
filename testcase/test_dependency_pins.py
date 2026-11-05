"""
Test cases for the dependency pins in pyproject.toml

`[tool.uv] override-dependencies` re-pins some core packages so the optional `unsloth` extra (which caps
them lower) resolves. An override silently wins over the core pin in `uv lock`, so bumping only the core pin
leaves the lock on the old version with no error. These tests keep the two in sync without a GPU or network.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def find_override_mismatches(pyproject):
    """Return (package, core specifier, override specifier) for every override that differs from its core pin"""
    core = {}
    for dep in pyproject.get("project", {}).get("dependencies", []):
        req = Requirement(dep)
        core[canonicalize_name(req.name)] = req.specifier

    mismatches = []
    for dep in pyproject.get("tool", {}).get("uv", {}).get("override-dependencies", []):
        req = Requirement(dep)
        name = canonicalize_name(req.name)
        # Overrides for packages outside the core dependencies (e.g. unsloth-only) are allowed
        if name in core and set(core[name]) != set(req.specifier):
            mismatches.append((name, str(core[name]), str(req.specifier)))
    return mismatches


def test_overrides_match_core_pins():
    """Every override of a core dependency in the real pyproject.toml must use the identical specifier"""
    with open(PYPROJECT, "rb") as f:
        mismatches = find_override_mismatches(tomllib.load(f))

    details = "\n".join(
        f"  {name}: core '{core}' vs override '{override}'"
        for name, core, override in mismatches
    )
    assert not mismatches, (
        "Core pins and `[tool.uv] override-dependencies` disagree:\n"
        f"{details}\n"
        "Keep `[tool.uv] override-dependencies` in sync with the core pins "
        "(an override silently wins over the core pin in `uv lock`)."
    )


def test_find_override_mismatches_detects_drift():
    """A bumped core pin with a stale override is reported; non-core overrides and no overrides are ignored"""
    pyproject = {
        "project": {"dependencies": ["transformers==5.18.0", "Datasets == 5.1.0"]},
        "tool": {
            "uv": {
                "override-dependencies": [
                    "transformers==5.19.0",
                    "datasets==5.1.0",
                    "unsloth==1.0",
                ]
            }
        },
    }
    assert find_override_mismatches(pyproject) == [
        ("transformers", "==5.18.0", "==5.19.0")
    ]

    del pyproject["tool"]["uv"]["override-dependencies"]
    assert find_override_mismatches(pyproject) == []


# The overrides exist for the optional unsloth extra; its own caps are the ones they lift.
OVERRIDE_ONLY_FOR = {"unsloth", "unsloth-zoo"}

# Caps of other packages the overrides deliberately lift, as (dependent, package): each was validated by hand and
# must be dropped once the dependent's cap covers the pin again (test_intentional_lifts_are_still_needed).
# vllm 0.32.0 caps transformers <5.20.0; transformers 5.22.1 was checked with vllm offline generation (Qwen3-0.6B).
INTENTIONALLY_LIFTED = {("vllm", "transformers")}


def find_overridden_requirements(pyproject, distributions):
    """Return (dependent, requirement, installed version) for every installed package other than unsloth whose
    requirement on an overridden package the installed version breaks, i.e. that an override silently lifted"""
    overridden = {
        canonicalize_name(Requirement(dep).name)
        for dep in pyproject.get("tool", {})
        .get("uv", {})
        .get("override-dependencies", [])
    }
    installed = {
        canonicalize_name(d.metadata["Name"]): d.version for d in distributions
    }
    broken = []
    for dist in distributions:
        dependent = canonicalize_name(dist.metadata["Name"])
        if dependent in OVERRIDE_ONLY_FOR:
            continue
        for raw in dist.requires or []:
            req = Requirement(raw)
            name = canonicalize_name(req.name)
            if name not in overridden or name not in installed:
                continue
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            if not req.specifier.contains(installed[name], prereleases=True):
                broken.append((dependent, str(req), installed[name]))
    return broken


def _installed_lifts():
    from importlib.metadata import distributions

    with open(PYPROJECT, "rb") as f:
        return find_overridden_requirements(tomllib.load(f), list(distributions()))


def test_overrides_lift_only_allowed_caps():
    """The core pins must satisfy every installed package's requirements except unsloth's and the ones listed in
    INTENTIONALLY_LIFTED: an override must not silently force, say, a transformers version past a new cap"""
    broken = [
        (dep, req, ver)
        for dep, req, ver in _installed_lifts()
        if (dep, canonicalize_name(Requirement(req).name)) not in INTENTIONALLY_LIFTED
    ]

    details = "\n".join(
        f"  {dep} requires {req}, installed {ver}" for dep, req, ver in broken
    )
    assert not broken, (
        "`[tool.uv] override-dependencies` lifts a cap that is neither unsloth's nor in INTENTIONALLY_LIFTED:\n"
        f"{details}\n"
        "Pick core pins that the other packages accept, or validate the combination and add it to "
        "INTENTIONALLY_LIFTED."
    )


def test_intentional_lifts_are_still_needed():
    """Every INTENTIONALLY_LIFTED entry must still lift a cap; a stale one would hide a future unvalidated lift"""
    lifted = {
        (dep, canonicalize_name(Requirement(req).name))
        for dep, req, _ in _installed_lifts()
    }
    stale = INTENTIONALLY_LIFTED - lifted
    assert not stale, (
        f"INTENTIONALLY_LIFTED entries no longer lift any cap: {sorted(stale)}. Remove them."
    )


def test_find_overridden_requirements_detects_lifted_cap():
    """A non-unsloth cap broken by the installed version is reported; unsloth's caps and extras are ignored"""

    class Dist:
        def __init__(self, name, version, requires=()):
            self.metadata, self.version, self.requires = (
                {"Name": name},
                version,
                list(requires),
            )

    pyproject = {"tool": {"uv": {"override-dependencies": ["transformers==5.22.1"]}}}
    dists = [
        Dist("transformers", "5.22.1"),
        Dist("vllm", "0.32.0", ["transformers<5.20.0,>=5.16.1"]),
        Dist("unsloth", "2026.11.2", ["transformers<=5.17.0"]),
        Dist("peft", "0.21.2", ["transformers<5.0; extra == 'dev'", "accelerate>=1.0"]),
    ]
    assert find_overridden_requirements(pyproject, dists) == [
        ("vllm", "transformers<5.20.0,>=5.16.1", "5.22.1")
    ]
    dists[0].version = "5.19.0"
    assert find_overridden_requirements(pyproject, dists) == []
