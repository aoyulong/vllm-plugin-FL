# Copyright (c) 2026 BAAI. All rights reserved.

"""
Shared utilities for IO inspector and IO dumper.

Provides common formatting, filtering, feature detection,
re-entrancy guard, execution order tracking, and YAML config
parsing used by both io_inspector.py and io_dumper.py.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Set, Tuple

import torch

# ── Feature detection (done once at import) ──

try:
    from torch.overrides import TorchFunctionMode
    HAS_TORCH_FUNC_MODE = True
except ImportError:
    TorchFunctionMode = None  # type: ignore[misc,assignment]
    HAS_TORCH_FUNC_MODE = False

try:
    from torch.nn.modules.module import (
        register_module_forward_pre_hook,
        register_module_forward_hook,
    )
    HAS_GLOBAL_MODULE_HOOKS = True
except ImportError:
    HAS_GLOBAL_MODULE_HOOKS = False

# ── Re-entrancy guard (shared, per-thread) ──

_in_hook = threading.local()


def guard_active() -> bool:
    """Check if we are already inside an IO hook (prevents recursion)."""
    return getattr(_in_hook, "active", False)


def set_guard(active: bool) -> None:
    """Set the re-entrancy guard."""
    _in_hook.active = active


# ── Torch function filtering constants ──

SKIP_TORCH_FUNCS = frozenset({
    "size", "dim", "is_contiguous", "contiguous", "stride",
    "storage_offset", "numel", "element_size", "is_floating_point",
    "is_complex", "requires_grad_", "data_ptr", "device",
    "dtype", "shape", "ndim", "is_cuda", "is_cpu",
})


# ── Common formatting ──


def format_value(value: Any) -> str:
    """Format a single value for display."""
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={list(value.shape)}, "
            f"dtype={value.dtype}, device={value.device})"
        )
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        type_name = type(value).__name__
        if len(value) <= 4:
            items = ", ".join(format_value(v) for v in value)
            return f"{type_name}([{items}])"
        return f"{type_name}(len={len(value)})"
    if isinstance(value, torch.nn.Module):
        return f"{type(value).__name__}(...)"
    return f"{type(value).__name__}(...)"


def format_result(result: Any) -> str:
    """Format operator output for display."""
    if isinstance(result, tuple):
        lines = []
        for i, v in enumerate(result):
            lines.append(f"  result[{i}]: {format_value(v)}")
        return "\n".join(lines)
    return f"  result: {format_value(result)}"


def get_module_class_name(args: tuple) -> Optional[str]:
    """Extract module class name from args[0] if it is an nn.Module."""
    if args and isinstance(args[0], torch.nn.Module):
        return type(args[0]).__name__
    return None


def get_torch_func_name(func) -> str:
    """Extract short name from a torch function."""
    return getattr(func, "__name__", str(func))


# ── Common parsing ──


def parse_torch_funcs_config(value: str) -> Tuple[bool, Set[str]]:
    """
    Parse a torch funcs env var value (e.g. "1", "matmul,softmax").

    Returns:
        (enabled, func_filter) — func_filter empty means "all"
    """
    value = value.strip()
    if not value or value == "0":
        return False, set()
    if value == "1":
        return True, set()
    funcs = {t.strip() for t in value.split(",") if t.strip()}
    return True, funcs


def should_inspect_torch_func(
    func_name: str,
    torch_funcs_enabled: bool,
    torch_func_filter: Set[str],
    match_all: bool,
    op_filter: Set[str],
) -> bool:
    """Check if a torch function should be intercepted."""
    if not torch_funcs_enabled:
        return False
    if torch_func_filter:
        return func_name in torch_func_filter
    # "all" mode: skip dunder and trivial ops
    if func_name.startswith("_"):
        return False
    if func_name in SKIP_TORCH_FUNCS:
        return False
    # When specific ops are set (not match_all), only match those
    if op_filter and func_name not in op_filter:
        return False
    return True


# ── Execution order tracking ──

_exec_order_counter: int = 0
_exec_order_lock = threading.Lock()


def next_exec_order() -> int:
    """Get the next global execution order number (thread-safe).

    This provides a monotonically increasing counter across all
    intercepted ops, allowing users to align output with model
    definition order.
    """
    global _exec_order_counter
    with _exec_order_lock:
        _exec_order_counter += 1
        return _exec_order_counter


def reset_exec_order() -> None:
    """Reset the execution order counter (e.g. at step boundaries)."""
    global _exec_order_counter
    with _exec_order_lock:
        _exec_order_counter = 0


def get_exec_order() -> int:
    """Get the current execution order counter value (without incrementing)."""
    return _exec_order_counter


# ── YAML config parsing ──


def parse_io_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Parse IO inspect/dump settings from a YAML config file.

    Expected YAML structure (all fields optional)::

        io_inspect:
          enabled: true           # or "1", or "rms_norm,silu_and_mul"
          ops:
            - rms_norm
            - silu_and_mul
          modules:
            - Linear
            - RMSNormFL
          torch_funcs: true       # or list ["matmul", "softmax"]

        io_dump:
          dir: /tmp/io_dump
          ops:
            - rms_norm
          modules:
            - Linear
          max_calls: 100
          step_range: [5, 15]
          torch_funcs: true

    Returns:
        Dict with keys "io_inspect" and/or "io_dump", each a dict
        of parsed settings.  Returns empty dict if file not found
        or has no io_inspect/io_dump sections.
    """
    if not os.path.isfile(config_path):
        return {}

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return {}

    result: Dict[str, Any] = {}

    inspect_cfg = config.get("io_inspect")
    if isinstance(inspect_cfg, dict):
        result["io_inspect"] = _parse_inspect_section(inspect_cfg)

    dump_cfg = config.get("io_dump")
    if isinstance(dump_cfg, dict):
        result["io_dump"] = _parse_dump_section(dump_cfg)

    return result


def _parse_inspect_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the io_inspect section of a YAML config."""
    parsed: Dict[str, Any] = {}

    enabled = cfg.get("enabled", False)
    if isinstance(enabled, bool):
        parsed["enabled"] = enabled
    elif isinstance(enabled, str):
        parsed["enabled"] = enabled.strip() not in ("", "0", "false", "False")
    elif isinstance(enabled, int):
        parsed["enabled"] = bool(enabled)
    else:
        parsed["enabled"] = False

    parsed["ops"] = _parse_string_list(cfg.get("ops"))
    parsed["modules"] = _parse_string_list(cfg.get("modules"))
    parsed["torch_funcs"] = _parse_torch_funcs_yaml(cfg.get("torch_funcs"))

    return parsed


def _parse_dump_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the io_dump section of a YAML config."""
    parsed: Dict[str, Any] = {}

    dump_dir = cfg.get("dir", "")
    parsed["dir"] = str(dump_dir).strip() if dump_dir else ""

    parsed["ops"] = _parse_string_list(cfg.get("ops"))
    parsed["modules"] = _parse_string_list(cfg.get("modules"))

    max_calls = cfg.get("max_calls", 0)
    try:
        parsed["max_calls"] = int(max_calls)
    except (ValueError, TypeError):
        parsed["max_calls"] = 0

    step_range = cfg.get("step_range")
    if isinstance(step_range, (list, tuple)) and len(step_range) == 2:
        try:
            parsed["step_range"] = (int(step_range[0]), int(step_range[1]))
        except (ValueError, TypeError):
            parsed["step_range"] = None
    else:
        parsed["step_range"] = None

    parsed["torch_funcs"] = _parse_torch_funcs_yaml(cfg.get("torch_funcs"))

    return parsed


def _parse_string_list(value: Any) -> Set[str]:
    """Parse a YAML value into a set of strings.

    Accepts:
      - list of strings: ["rms_norm", "silu_and_mul"]
      - comma-separated string: "rms_norm,silu_and_mul"
      - None / empty → empty set
    """
    if not value:
        return set()
    if isinstance(value, (list, tuple)):
        return {str(v).strip() for v in value if v}
    if isinstance(value, str):
        return {t.strip() for t in value.split(",") if t.strip()}
    return set()


def _parse_torch_funcs_yaml(value: Any) -> Tuple[bool, Set[str]]:
    """Parse torch_funcs from YAML config.

    Accepts:
      - true/false: enable/disable all
      - list: ["matmul", "softmax"] — enable specific
      - string: "matmul,softmax" or "1"
    """
    if value is None or value is False:
        return False, set()
    if value is True:
        return True, set()
    if isinstance(value, (list, tuple)):
        funcs = {str(v).strip() for v in value if v}
        return bool(funcs), funcs
    if isinstance(value, str):
        return parse_torch_funcs_config(value)
    return False, set()
