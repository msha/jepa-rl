from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


class SimpleYamlError(ValueError):
    """Raised when the dependency-free YAML subset parser cannot parse a file."""


def load_simple_yaml(path: str | Path) -> Any:
    """Load the small YAML subset used by the starter configs.

    This fallback exists so the scaffold can validate configs before optional
    dependencies are installed. It intentionally supports only mappings, nested
    mappings, block lists, inline lists, strings, numbers, booleans, and null.
    Install ``PyYAML`` for full YAML support.
    """

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for line_no, original in enumerate(lines, start=1):
        stripped = _strip_comment(original).rstrip()
        if not stripped:
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        if indent % 2 != 0:
            raise SimpleYamlError(f"{path}:{line_no}: indentation must use two-space steps")
        content = stripped.lstrip(" ")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise SimpleYamlError(f"{path}:{line_no}: invalid indentation")

        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise SimpleYamlError(f"{path}:{line_no}: list item is not under a list")
            parent.append(_parse_scalar(content[2:].strip()))
            continue

        if ":" not in content:
            raise SimpleYamlError(f"{path}:{line_no}: expected key: value")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise SimpleYamlError(f"{path}:{line_no}: empty key")
        if not isinstance(parent, dict):
            raise SimpleYamlError(f"{path}:{line_no}: mapping entry is not under a mapping")

        if raw_value == "":
            next_container = _infer_child_container(lines, line_no, indent)
            parent[key] = next_container
            stack.append((indent, next_container))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def dump_simple_yaml(data: Any) -> str:
    return "\n".join(_dump_lines(data, 0)) + "\n"


def _dump_lines(data: Any, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_lines(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(value)}")
        return lines
    if isinstance(data, list):
        lines = []
        for value in data:
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_dump_lines(value, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(value)}")
        return lines
    return [f"{prefix}{_format_scalar(data)}"]


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _infer_child_container(lines: list[str], current_line_no: int, current_indent: int) -> Any:
    for future in lines[current_line_no:]:
        stripped = _strip_comment(future).rstrip()
        if not stripped:
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        if indent <= current_indent:
            return {}
        return [] if stripped.lstrip(" ").startswith("- ") else {}
    return {}


def _parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") or value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise SimpleYamlError(f"could not parse inline value: {value}") from exc
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
