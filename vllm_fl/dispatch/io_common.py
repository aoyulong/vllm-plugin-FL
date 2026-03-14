# Copyright (c) 2026 BAAI. All rights reserved.

"""
Shared utilities for IO inspector and IO dumper.

Provides common formatting, filtering, feature detection,
re-entrancy guard, execution order tracking, and YAML config
parsing used by both io_inspector.py and io_dumper.py.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

_logger = logging.getLogger(__name__)

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

# ── Distributed rank ──

_rank: Optional[int] = None  # cached after first call


def get_rank() -> int:
    """Get the current distributed rank.

    Resolution order:
    1. ``torch.distributed.get_rank()`` if the default process group is initialized
    2. ``RANK`` environment variable
    3. ``LOCAL_RANK`` environment variable
    4. 0 (single-process fallback)

    The result is cached once a definitive source (dist or env var) is found.
    The fallback ``0`` is never cached so that later calls can pick up a
    process group that was initialized after the first call.
    """
    global _rank
    if _rank is not None:
        return _rank
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            _rank = dist.get_rank()
            return _rank
    except Exception:
        pass
    for var in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            try:
                _rank = int(val)
                return _rank
            except ValueError:
                pass
    return 0


def reset_rank() -> None:
    """Reset cached rank (for testing only)."""
    global _rank
    _rank = None


def parse_rank_filter(value: str) -> Optional[Set[int]]:
    """Parse a rank filter string.

    Args:
        value: ``"all"`` or ``""`` for all ranks, or comma-separated
            rank numbers like ``"0"`` or ``"0,2,4"``.

    Returns:
        None for all ranks, or a set of rank integers.
    """
    value = value.strip().lower()
    if not value or value == "all":
        return None
    ranks: Set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if token:
            try:
                ranks.add(int(token))
            except ValueError:
                pass
    return ranks if ranks else None


# ── Re-entrancy guard (per-caller, per-thread) ──
#
# Each subsystem (inspector, dumper) creates its own guard via
# ``make_guard()`` so they do not block each other.  The guard is
# thread-local to prevent cross-thread interference.


def make_guard():
    """Create an independent re-entrancy guard (thread-local).

    Returns ``(guard_active, set_guard)`` functions.  Each caller gets
    its own guard so the inspector and dumper do not interfere.
    """
    _tls = threading.local()

    def guard_active() -> bool:
        return getattr(_tls, "active", False)

    def set_guard(active: bool) -> None:
        _tls.active = active

    return guard_active, set_guard


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


# ── Per-step counters ──
#
# module counter: each unique module class name gets a monotonic index (0,1,2,…)
# and a per-class call count, both reset each step.
# op counter: same for ops/torch funcs.
# seen sets: track which modules/ops were encountered this step (for summary).

_module_type_index: Dict[str, int] = {}
_module_type_count: Dict[str, int] = {}
_module_next_idx: int = 0

_op_type_index: Dict[str, int] = {}
_op_type_count: Dict[str, int] = {}
_op_next_idx: int = 0

_seen_modules: Set[str] = set()
_seen_ops: Set[str] = set()

_counter_lock = threading.Lock()


def next_module_counter(cls_name: str) -> Tuple[int, int]:
    """Get (type_index, call_count) for a module class this step, incrementing call count."""
    global _module_next_idx
    with _counter_lock:
        if cls_name not in _module_type_index:
            _module_type_index[cls_name] = _module_next_idx
            _module_next_idx += 1
            _module_type_count[cls_name] = 0
        _module_type_count[cls_name] += 1
        return _module_type_index[cls_name], _module_type_count[cls_name]


def next_op_counter(op_name: str) -> Tuple[int, int]:
    """Get (type_index, call_count) for an op this step, incrementing call count."""
    global _op_next_idx
    with _counter_lock:
        if op_name not in _op_type_index:
            _op_type_index[op_name] = _op_next_idx
            _op_next_idx += 1
            _op_type_count[op_name] = 0
        _op_type_count[op_name] += 1
        return _op_type_index[op_name], _op_type_count[op_name]


def _reset_per_step_counters() -> Tuple[Set[str], Set[str]]:
    """Reset module/op counters and seen sets for a new step.

    Returns:
        (seen_modules, seen_ops) from the completed step.
    """
    global _module_next_idx, _op_next_idx
    with _counter_lock:
        seen_modules = _seen_modules.copy()
        seen_ops = _seen_ops.copy()
        _seen_modules.clear()
        _seen_ops.clear()
        _module_type_index.clear()
        _module_type_count.clear()
        _module_next_idx = 0
        _op_type_index.clear()
        _op_type_count.clear()
        _op_next_idx = 0
    return seen_modules, seen_ops


# ── Step tracking (shared by inspector and dumper) ──

_step_counter: int = 0  # first forward pass runs at step 0
_step_lock = threading.Lock()
_step_callbacks: List[Any] = []  # called on each step advance


def get_step() -> int:
    """Get the current step counter value."""
    return _step_counter


def advance_step() -> int:
    """Increment the step counter and reset execution order (thread-safe).

    Returns the new step number.  Also resets per-step counters and
    fires registered step callbacks with the summary of the completed step.
    """
    global _step_counter
    with _step_lock:
        prev_step = _step_counter
        _step_counter += 1
        result = _step_counter
        callbacks = list(_step_callbacks)
    reset_exec_order()
    seen_modules, seen_ops = _reset_per_step_counters()
    for cb in callbacks:
        try:
            cb(prev_step, seen_modules, seen_ops)
        except Exception:
            _logger.warning("Step callback %s failed", cb.__name__, exc_info=True)
    return result


def reset_step() -> None:
    """Reset the step counter to 0 (for testing)."""
    global _step_counter
    with _step_lock:
        _step_counter = 0
    _reset_per_step_counters()


def register_step_callback(cb) -> None:
    """Register a callback to be called on each step advancement.

    Callbacks receive (step, seen_modules, seen_ops) for the completed step.
    """
    with _step_lock:
        if cb not in _step_callbacks:
            _step_callbacks.append(cb)


def unregister_step_callback(cb) -> None:
    """Remove a previously registered step callback."""
    with _step_lock:
        try:
            _step_callbacks.remove(cb)
        except ValueError:
            pass


def parse_step_range(value) -> Optional[Tuple[int, int]]:
    """Parse a step range string into a half-open tuple for internal use.

    Accepted formats (all inclusive):
    - ``"start-end"``: ``"0-2"`` → ``(0, 3)`` (steps 0, 1, 2)
    - Bare integer: ``"5"`` → ``(5, 6)`` (step 5 only)

    Returns ``None`` if the value is ``None``, empty, or cannot be parsed.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if "-" in value:
        parts = value.split("-", 1)
        try:
            return (int(parts[0].strip()), int(parts[1].strip()) + 1)
        except ValueError:
            return None
    elif value.isdigit():
        s = int(value)
        return (s, s + 1)
    return None


def parse_step_range_env(*env_vars: str) -> Optional[Tuple[int, int]]:
    """Parse a step range from the first non-empty env var.

    Checks each env var in order; returns the first valid ``(start, end)``
    half-open tuple or ``None``.  Accepts ``"start-end"`` inclusive dash
    format (e.g. ``"0-1"`` → ``(0, 2)``) or a bare integer (e.g. ``"0"``
    → ``(0, 1)``).
    """
    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if not value:
            continue
        result = parse_step_range(value)
        if result is not None:
            return result
    return None


# ── Module path registry ──
#
# Maps ``id(module)`` → full dotted path (e.g. ``"model.layers.0.self_attn"``).
# Populated by ``register_module_paths(model)`` after the model is loaded.
# Used by ``push_module_context`` and ``layer_path_matches`` for layer filtering.

_module_path_map: Dict[int, str] = {}
_module_path_lock = threading.Lock()


def register_module_paths(model: torch.nn.Module) -> None:
    """Build ``id(module) → path`` mapping from a model's named_modules.

    Call after the model is loaded (e.g. in ``load_model()``) so that
    global module hooks can look up each module's full path.
    """
    with _module_path_lock:
        for name, mod in model.named_modules():
            _module_path_map[id(mod)] = name


def get_module_path(module: torch.nn.Module) -> str:
    """Look up the full dotted path of a module, or empty string."""
    return _module_path_map.get(id(module), "")


# ── Module context tracking (per-thread) ──
#
# When module filtering is active, global module hooks push/pop module
# class names onto a thread-local stack.  This lets op hooks and
# TorchFunctionMode know which module(s) are currently executing,
# enabling "only log ops/torch_funcs within these modules" semantics.
#
# The stack stores (cls_name, module_type_idx, module_call_count) tuples
# so that op hooks can include the enclosing module's counter in output.

_module_context = threading.local()


def push_module_context(cls_name: str, module: Optional[torch.nn.Module] = None) -> None:
    """Push a module class name onto the current-module stack.

    Also tracks per-step module type index and call count, and the
    module's full path if registered via :func:`register_module_paths`.
    """
    stack = getattr(_module_context, "stack", None)
    if stack is None:
        _module_context.stack = []
        stack = _module_context.stack
    path = get_module_path(module) if module is not None else ""
    # Track module counter for this step
    mod_idx, mod_count = next_module_counter(cls_name)
    stack.append((cls_name, mod_idx, mod_count, path))


def pop_module_context() -> None:
    """Pop the most recent module class name from the stack."""
    stack = getattr(_module_context, "stack", None)
    if stack:
        stack.pop()


def get_current_module() -> Optional[str]:
    """Get the innermost (most recent) module class name, or None."""
    stack = getattr(_module_context, "stack", None)
    if stack:
        return stack[-1][0]
    return None


def get_current_module_counter() -> Optional[Tuple[int, int]]:
    """Get (module_type_idx, call_count) for the innermost module, or None."""
    stack = getattr(_module_context, "stack", None)
    if stack:
        return (stack[-1][1], stack[-1][2])
    return None


def module_context_matches(filter_set: Set[str]) -> bool:
    """Check if any module in the current call stack matches the filter."""
    stack = getattr(_module_context, "stack", None)
    if not stack:
        return False
    # Check from innermost to outermost for fast match
    for entry in reversed(stack):
        if entry[0] in filter_set:
            return True
    return False


def get_current_module_path() -> str:
    """Get the full path of the innermost module, or empty string."""
    stack = getattr(_module_context, "stack", None)
    if stack:
        return stack[-1][3]
    return ""


# ── Layer filtering ──

# Range pattern for layer specs like "0-3", "10-15"
_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")

# Default prefix for integer shorthand expansion
DEFAULT_LAYER_PREFIX = "model.layers."


def _is_glob(s: str) -> bool:
    """Check if a string contains glob wildcard characters."""
    return "*" in s or "?" in s


def expand_layer_specs(
    specs: Set[str],
    prefix: str = DEFAULT_LAYER_PREFIX,
) -> Set[str]:
    """Expand user-friendly layer specifications to full paths.

    Accepts several shorthand forms in addition to full paths:

    - **Integer**: ``"0"`` → ``"model.layers.0"``
    - **Range**: ``"0-3"`` → ``"model.layers.0"``, ..., ``"model.layers.3"``
    - **Full path**: ``"model.layers.0.self_attn"`` → kept as-is
    - **Glob pattern**: ``"model.layers.*.self_attn"`` → kept as-is

    Args:
        specs: Raw layer specifications from the user.
        prefix: Prefix for integer/range expansion.
            Defaults to ``"model.layers."``.

    Returns:
        Expanded set of layer path prefixes and/or glob patterns.
    """
    result: Set[str] = set()
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        # Pure integer → single layer
        if spec.isdigit():
            result.add(prefix + spec)
            continue
        # Range like "0-3"
        m = _RANGE_RE.match(spec)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            for i in range(start, end + 1):
                result.add(prefix + str(i))
            continue
        # Full path or glob pattern — keep as-is
        result.add(spec)
    return result


def list_model_layers(
    model: torch.nn.Module,
    max_depth: Optional[int] = None,
) -> List[str]:
    """List all module paths in a model, useful for discovering layer names.

    Logs the paths at INFO level and returns them as a list.

    Args:
        model: The model to inspect.
        max_depth: Maximum nesting depth to show.  ``None`` = unlimited.
            Depth 1 shows direct children, depth 2 shows grandchildren, etc.

    Returns:
        List of ``(name, class_name)`` strings like
        ``"model.layers.0 (Qwen3DecoderLayer)"``.

    Example::

        from vllm_fl.dispatch.io_common import list_model_layers
        list_model_layers(model, max_depth=2)
    """
    paths: List[str] = []
    for name, mod in model.named_modules():
        if not name:
            continue  # skip root
        if max_depth is not None:
            depth = name.count(".") + 1
            if depth > max_depth:
                continue
        cls = type(mod).__name__
        paths.append(f"{name} ({cls})")
    if paths:
        _logger.info(
            "Model layer paths (%d modules):\n  %s",
            len(paths),
            "\n  ".join(paths),
        )
    return paths


def layer_path_matches(filter_set: Set[str]) -> bool:
    """Check if any module in the current call stack has a matching layer path.

    Supports three matching modes (determined per filter entry):
    - **Exact/prefix**: ``"model.layers.0"`` matches ``"model.layers.0"``
      and ``"model.layers.0.self_attn"`` but NOT ``"model.layers.00"``.
    - **Glob/wildcard**: entries containing ``*`` or ``?`` are matched
      with :func:`fnmatch.fnmatch`.  E.g. ``"model.layers.*.self_attn"``
      matches ``"model.layers.0.self_attn"``, ``"model.layers.1.self_attn"``.
    """
    stack = getattr(_module_context, "stack", None)
    if not stack:
        return False
    for entry in reversed(stack):
        path = entry[3]
        if path:
            for pattern in filter_set:
                if _is_glob(pattern):
                    if fnmatch.fnmatch(path, pattern):
                        return True
                else:
                    if path == pattern or path.startswith(pattern + "."):
                        return True
    return False


def parse_layers_env(*env_vars: str) -> Set[str]:
    """Parse layer path filters from the first non-empty env var.

    Checks each env var in order; returns the first valid set of
    layer specs after expansion (integers, ranges, globs, full paths).
    """
    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if value:
            raw = {t.strip() for t in value.split(",") if t.strip()}
            return expand_layer_specs(raw)
    return set()


# ── Torch function filtering ──

SKIP_TORCH_FUNCS = frozenset({
    "size", "dim", "is_contiguous", "contiguous", "stride",
    "storage_offset", "numel", "element_size", "is_floating_point",
    "is_complex", "requires_grad_", "data_ptr", "device",
    "dtype", "shape", "ndim", "is_cuda", "is_cpu",
})


def get_torch_func_name(func) -> str:
    """Extract short name from a torch function."""
    return getattr(func, "__name__", str(func))


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


# ── Tensor statistics (shared by inspector and dumper) ──
#
# Extensible registry: users can add custom stats via register_tensor_stat().


def _stat_min(t: torch.Tensor) -> float:
    return t.min().item()


def _stat_max(t: torch.Tensor) -> float:
    return t.max().item()


def _stat_mean(t: torch.Tensor) -> float:
    return t.mean().item()


def _stat_std(t: torch.Tensor) -> float:
    return t.std().item()


# Each entry: (name, fn, float_only).
# float_only=True → fn receives t.detach().float(); skipped for integer tensors.
# float_only=False → fn receives the original tensor (int or float).
_TENSOR_STAT_REGISTRY: List[Tuple[str, Callable[[torch.Tensor], Any], bool]] = [
    ("min", _stat_min, False),
    ("max", _stat_max, False),
    ("mean", _stat_mean, True),
    ("std", _stat_std, True),
]
_tensor_stat_lock = threading.Lock()


def register_tensor_stat(
    name: str,
    fn: Callable[[torch.Tensor], Any],
    float_only: bool = True,
) -> None:
    """Register a custom tensor statistic for both inspector and dumper.

    The function receives a tensor and should return a JSON-serializable
    value — scalar, list, or dict.  Scalars are displayed inline;
    lists are shown truncated (e.g. ``top_k=[1.2, 0.8, ...]``).

    For ``float_only=True`` (default), the tensor is ``t.detach().float()``;
    for ``float_only=False``, the original tensor is passed.

    If a stat with the same *name* already exists, it is replaced.
    Raises ``ValueError`` if *name* is empty.

    Example::

        from vllm_fl.dispatch import register_tensor_stat
        # Scalar stats
        register_tensor_stat("l2_norm", lambda t: t.norm(2).item())
        register_tensor_stat("sparsity",
                             lambda t: (t == 0).float().mean().item())
        # List stats (for future top-k, first-n, last-n, etc.)
        register_tensor_stat("top_5",
                             lambda t: t.flatten().topk(5).values.tolist(),
                             float_only=False)
        register_tensor_stat("first_4",
                             lambda t: t.flatten()[:4].tolist(),
                             float_only=False)
    """
    if not name or not name.strip():
        raise ValueError("Tensor stat name must be a non-empty string")
    with _tensor_stat_lock:
        for i, (existing_name, _, _) in enumerate(_TENSOR_STAT_REGISTRY):
            if existing_name == name:
                _TENSOR_STAT_REGISTRY[i] = (name, fn, float_only)
                return
        _TENSOR_STAT_REGISTRY.append((name, fn, float_only))


def tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    """Compute shape, dtype, device, and all registered statistics for a tensor.

    Used by both the inspector (text formatting) and the dumper (JSON metadata).
    Returns a dict like ``{"shape": [...], "dtype": "...", "min": ..., ...}``.
    """
    meta: Dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    if t.numel() == 0 or t.is_complex():
        return meta
    is_fp = t.is_floating_point()
    fp_tensor: Optional[torch.Tensor] = None  # lazy
    for name, fn, float_only in _TENSOR_STAT_REGISTRY:
        if float_only and not is_fp:
            continue
        try:
            if float_only:
                if fp_tensor is None:
                    fp_tensor = t.detach().float()
                meta[name] = fn(fp_tensor)
            else:
                meta[name] = fn(t)
        except Exception:
            pass
    return meta


# ── Formatting ──

_STAT_LIST_DISPLAY_LIMIT = 4  # max items shown inline for list-valued stats


def _format_stat_value(v: Any) -> str:
    """Format a stat value for inline display.

    Scalars: ``0.123456``.  Lists: ``[1.2, 0.8, ...]`` (truncated).
    """
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        limit = _STAT_LIST_DISPLAY_LIMIT
        formatted = [f"{x:.6g}" if isinstance(x, float) else repr(x)
                     for x in v[:limit]]
        if len(v) > limit:
            formatted.append("...")
        return f"[{', '.join(formatted)}]"
    return repr(v)


def format_value(value: Any) -> str:
    """Format a single value for display, including tensor statistics."""
    if isinstance(value, torch.Tensor):
        stats = tensor_stats(value)
        parts = [
            f"shape={stats['shape']}",
            f"dtype={stats['dtype']}",
            f"device={stats['device']}",
        ]
        for name, _, _ in _TENSOR_STAT_REGISTRY:
            if name in stats:
                v = stats[name]
                parts.append(f"{name}={_format_stat_value(v)}")
        return f"Tensor({', '.join(parts)})"
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


def make_module_tag() -> str:
    """Build ``[module=X,Y]`` from current module context, or empty string."""
    counter = get_current_module_counter()
    if counter is not None:
        return f"[module={counter[0]},{counter[1]}]"
    return ""


def make_op_tag(op_name: str) -> str:
    """Build ``[op=X,Y]``, incrementing the per-step op counter."""
    idx, count = next_op_counter(op_name)
    return f"[op={idx},{count}]"


def make_label(op_name: str, args: tuple = ()) -> str:
    """Build a display label like ``rms_norm (module=Linear, layer=model.layers.0)``."""
    module_name = get_module_class_name(args) or get_current_module()
    path = get_current_module_path()
    parts = [f"module={module_name or ''}"]
    if path:
        parts.append(f"layer={path}")
    return f"{op_name} ({', '.join(parts)})"


def record_seen(op_name: str, args: tuple = ()) -> None:
    """Record an op and its enclosing module as seen this step (for summary).

    Detects the module name from args[0] (if nn.Module) or the module context.
    """
    module_name = get_module_class_name(args) or get_current_module()
    with _counter_lock:
        _seen_ops.add(op_name)
        if module_name:
            _seen_modules.add(module_name)


# ── Torch func tag management ──
#
# When both inspector and dumper TorchFunctionMode handlers are active,
# they are stacked: the outer handler calls func() which enters the inner.
# Without coordination, both would call make_op_tag() / next_exec_order(),
# double-incrementing the shared counters.  These helpers ensure that tags
# and exec_order are generated once (by the first handler) and reused by
# the nested handler.

_tf_tags = threading.local()


def acquire_torch_func_tags(op_name: str) -> Tuple[str, str, int]:
    """Get or generate (module_tag, op_tag, exec_order) for a torch function call.

    The first caller (outermost TorchFunctionMode) generates and caches
    the values.  Nested callers reuse the cached values, preventing
    double-increment of shared counters.

    Must be paired with :func:`release_torch_func_tags`.
    """
    depth = getattr(_tf_tags, "depth", 0)
    _tf_tags.depth = depth + 1

    if depth == 0:
        # First (outermost) caller: generate and cache
        module_tag = make_module_tag()
        op_tag = make_op_tag(op_name)
        order = next_exec_order()
        _tf_tags.module_tag = module_tag
        _tf_tags.op_tag = op_tag
        _tf_tags.exec_order = order
    else:
        # Nested caller: reuse cached values
        module_tag = getattr(_tf_tags, "module_tag", "")
        op_tag = getattr(_tf_tags, "op_tag", "")
        order = getattr(_tf_tags, "exec_order", 0)

    return module_tag, op_tag, order


def release_torch_func_tags() -> None:
    """Release torch function tags.  Clears the cache when the outermost caller exits."""
    depth = getattr(_tf_tags, "depth", 0)
    if depth > 0:
        _tf_tags.depth = depth - 1
    if depth <= 1:
        _tf_tags.module_tag = None
        _tf_tags.op_tag = None
        _tf_tags.exec_order = None


# ── Global module hooks (context tracking only) ──
#
# Shared by inspector and dumper.  These hooks push/pop the module class
# name onto a thread-local context stack so op hooks and TorchFunctionMode
# know which module(s) are currently executing.  They do NOT log or dump.
#
# Refcounted: when both inspector and dumper are active, hooks are
# registered only once and removed when the last user releases.

_global_hook_lock = threading.Lock()
_global_hook_refcount = 0
_global_hook_handles: List[Any] = []


def _global_forward_pre_hook(module, args):
    """Push module onto the context stack before forward."""
    if torch.compiler.is_compiling():
        return
    push_module_context(type(module).__name__, module)


def _global_forward_post_hook(module, args, output):
    """Pop module from the context stack after forward."""
    if torch.compiler.is_compiling():
        return
    pop_module_context()


def acquire_global_module_hooks() -> None:
    """Register global module hooks (refcounted, safe to call multiple times)."""
    global _global_hook_refcount
    with _global_hook_lock:
        _global_hook_refcount += 1
        if _global_hook_refcount == 1 and HAS_GLOBAL_MODULE_HOOKS:
            h1 = register_module_forward_pre_hook(_global_forward_pre_hook)
            h2 = register_module_forward_hook(_global_forward_post_hook)
            _global_hook_handles.extend([h1, h2])


def release_global_module_hooks() -> None:
    """Unregister global module hooks when the last user releases."""
    global _global_hook_refcount
    with _global_hook_lock:
        _global_hook_refcount = max(0, _global_hook_refcount - 1)
        if _global_hook_refcount == 0:
            for h in _global_hook_handles:
                h.remove()
            _global_hook_handles.clear()


# ── YAML config parsing ──


def parse_io_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Parse IO inspect/dump settings from a YAML config file.

    Expected YAML structure (all fields optional)::

        io_inspect:
          enabled: true           # boolean: true/false
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
          step_range: "5-15"      # inclusive "start-end"
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


def _parse_step_range_yaml(cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Parse ``step_range`` from a YAML section.

    Accepts a dash-string like ``"0-2"`` (inclusive, same format as env vars).
    Returns a half-open ``(start, end+1)`` tuple for internal use,
    or ``None`` if unset / invalid.
    """
    step_range = cfg.get("step_range")
    if step_range is None:
        return None
    # Convert int/list to string for uniform handling
    if isinstance(step_range, int):
        step_range = str(step_range)
    elif isinstance(step_range, (list, tuple)):
        # Legacy [start, end] format — convert to "start-end" string
        if len(step_range) == 2:
            step_range = f"{step_range[0]}-{step_range[1]}"
        else:
            return None
    return parse_step_range(step_range)


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
    parsed["layers"] = expand_layer_specs(_parse_string_list(cfg.get("layers")))
    parsed["step_range"] = _parse_step_range_yaml(cfg)

    # Only include torch_funcs when explicitly set in YAML so that
    # _init_from_env can apply the match-all default for absent keys.
    if "torch_funcs" in cfg:
        parsed["torch_funcs"] = _parse_torch_funcs_yaml(cfg["torch_funcs"])
    parsed["ranks"] = _parse_ranks_yaml(cfg.get("ranks"))

    return parsed


def _parse_dump_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the io_dump section of a YAML config."""
    parsed: Dict[str, Any] = {}

    dump_dir = cfg.get("dir", "")
    parsed["dir"] = str(dump_dir).strip() if dump_dir else ""

    parsed["ops"] = _parse_string_list(cfg.get("ops"))
    parsed["modules"] = _parse_string_list(cfg.get("modules"))
    parsed["layers"] = expand_layer_specs(_parse_string_list(cfg.get("layers")))

    max_calls = cfg.get("max_calls", 0)
    try:
        parsed["max_calls"] = int(max_calls)
    except (ValueError, TypeError):
        parsed["max_calls"] = 0

    parsed["step_range"] = _parse_step_range_yaml(cfg)

    # Only include torch_funcs when explicitly set in YAML so that
    # _init_from_env can apply the match-all default for absent keys.
    if "torch_funcs" in cfg:
        parsed["torch_funcs"] = _parse_torch_funcs_yaml(cfg["torch_funcs"])
    parsed["ranks"] = _parse_ranks_yaml(cfg.get("ranks"))
    parsed["meta_only"] = bool(cfg.get("meta_only", True))

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


def _parse_ranks_yaml(value: Any) -> Optional[Set[int]]:
    """Parse ``ranks`` from YAML config.

    Accepts:
      - None / "all" → None (all ranks)
      - list of ints: [0, 1, 2]
      - single int: 0
      - string: "0,2,4" or "all"
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        ranks = set()
        for v in value:
            try:
                ranks.add(int(v))
            except (ValueError, TypeError):
                pass
        return ranks if ranks else None
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        return parse_rank_filter(value)
    return None
