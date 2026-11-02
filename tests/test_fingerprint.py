import json
from pathlib import Path

import pytest

from complyee_agent_guard import canonical_tool_call, tool_call_payload_hash

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures/conformance-vectors.json").read_text("utf-8")
)


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda v: v["name"])
def test_matches_acr_conformance_vectors(vector):
    assert canonical_tool_call(vector["tool"], vector["arguments"]) == vector["canonical"]
    assert tool_call_payload_hash(vector["tool"], vector["arguments"]) == vector["payloadHash"]


def test_python_native_values_hash_like_javascript():
    assert canonical_tool_call("n", {"a": 1.0, "b": -0.0, "c": (1, 2), "d": 1e16}) == (
        '{"arguments":{"a":1,"b":0,"c":[1,2],"d":10000000000000000},"tool":"n"}'
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "x"}, object(), {"s": {1, 2}}])
def test_unhashable_arguments_are_refused(bad):
    with pytest.raises(ValueError):
        tool_call_payload_hash("t", {"v": bad})
