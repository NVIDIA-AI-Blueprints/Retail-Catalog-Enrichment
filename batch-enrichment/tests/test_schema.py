"""The reference schema, and the validator that turns it into the goodput metric.

These are not style checks. The validator IS the goodput number, so a permissive schema
would inflate every measurement in the report rather than fail visibly.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from catalog_bench.schema_check import inspect


@pytest.fixture
def validator(schema):
    return Draft202012Validator(schema)


def test_schema_is_well_formed(schema):
    Draft202012Validator.check_schema(schema)


def test_every_property_is_required(schema):
    """Optional keys make the emitted structure vary per product, which widens output
    length and breaks the byte-identical prefix that prefix caching depends on."""
    assert set(schema["required"]) == set(schema["properties"])


def test_additional_properties_closed(schema):
    assert schema["additionalProperties"] is False


def test_every_array_is_bounded(schema):
    unbounded = [name for name, spec in schema["properties"].items()
                 if spec.get("type") == "array" and "maxItems" not in spec]
    assert not unbounded, f"unbounded arrays make output length unbounded: {unbounded}"


def test_valid_instance_validates(validator, instance):
    assert validator.is_valid(instance)


@pytest.mark.parametrize("mutate, why", [
    (lambda d: d.update(category="not_a_real_category"), "out-of-enum value"),
    (lambda d: d.pop("brand"), "missing required field"),
    (lambda d: d.update(hallucinated_field="x"), "invented field"),
    (lambda d: d.update(secondary_colors=["a", "b", "c", "d"]), "over-long array"),
])
def test_invalid_instances_are_rejected(validator, instance, mutate, why):
    mutate(instance)
    assert not validator.is_valid(instance), f"{why} should not validate"


def test_reference_schema_passes_its_own_check(schema):
    """The schema we ship must pass the linter we hand integrators."""
    findings = inspect(schema)
    assert findings.ok, findings.errors


def test_check_flags_an_unbounded_array():
    findings = inspect({
        "type": "object",
        "additionalProperties": False,
        "required": ["tags"],
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    })
    assert not findings.ok
    assert findings.unbounded_arrays == ["$.tags"]


def test_check_flags_open_objects():
    findings = inspect({"type": "object", "properties": {}, "required": []})
    assert any("additionalProperties" in e for e in findings.errors)
