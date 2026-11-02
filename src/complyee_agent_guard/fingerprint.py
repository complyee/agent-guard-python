"""sha256-canonical-json-v1: the tool-call fingerprint shared with ACR and the
TypeScript MCP guard. Byte-identical to JavaScript's JSON.stringify over
{"tool": name, "arguments": args} with object keys sorted by UTF-16 code unit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal
from typing import Any, Mapping

TOOL_CALL_HASH_ALGORITHM = "sha256-canonical-json-v1"

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _js_number(value: float) -> str:
    """Format a number exactly as JavaScript's Number#toString does."""
    if not math.isfinite(value):
        raise ValueError("non-finite number in tool arguments")
    if value == 0:
        return "0"  # also -0, as JavaScript prints it
    sign = "-" if value < 0 else ""
    # repr gives the shortest round-trip digits, the same digits JavaScript picks.
    _, digits_t, exponent = Decimal(repr(abs(value))).as_tuple()
    digits = "".join(map(str, digits_t)).rstrip("0") or "0"
    stripped = len("".join(map(str, digits_t))) - len(digits)
    k = len(digits)
    n = k + exponent + stripped  # value = 0.digits * 10**n
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
        out = f"{mantissa}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + out


def _js_string(value: str) -> str:
    out = json.dumps(value, ensure_ascii=False)
    return _LONE_SURROGATE.sub(lambda m: f"\\u{ord(m.group()):04x}", out)


def _utf16_key(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


def canonical_json(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _js_string(value)
    if isinstance(value, (int, float)):
        return _js_number(float(value))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    if isinstance(value, Mapping):
        for k in value:
            if not isinstance(k, str):
                raise ValueError("tool argument keys must be strings")
        keys = sorted(value.keys(), key=_utf16_key)
        return "{" + ",".join(f"{_js_string(k)}:{canonical_json(value[k])}" for k in keys) + "}"
    raise ValueError(f"unsupported value in tool arguments: {type(value).__name__}")


def canonical_tool_call(tool_name: str, tool_arguments: Mapping[str, Any]) -> str:
    return canonical_json({"tool": tool_name, "arguments": tool_arguments})


def tool_call_payload_hash(tool_name: str, tool_arguments: Mapping[str, Any]) -> str:
    data = canonical_tool_call(tool_name, tool_arguments).encode("utf-8", "surrogatepass")
    return hashlib.sha256(data).hexdigest()
