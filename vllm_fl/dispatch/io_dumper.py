# Copyright (c) 2026 BAAI. All rights reserved.

"""
IO Dumper for vllm-plugin-FL dispatch.

Saves operator input/output tensors to disk as PyTorch .pt files.

Supports three interception mechanisms:
1. Dispatch-managed ops: Automatic via OpManager.call() hooks
2. nn.Module forward passes: Automatic via global module hooks (when
   modules filter is set or dump-all mode)
3. Bare torch functions: Via TorchFunctionMode (opt-in)

Configuration (priority order):
    1. YAML config file (via VLLM_FL_CONFIG):
        io_dump:
          dir: /tmp/io_dump
          ops: [rms_norm, silu_and_mul]
          modules: [Linear]
          max_calls: 100
          step_range: [5, 15]
          torch_funcs: true
    2. Environment variables:
        VLLM_FL_IO_DUMP              - Directory path (enables dumping)
        VLLM_FL_IO_DUMP_OPS          - Comma-separated op names
        VLLM_FL_IO_DUMP_MODULES      - Comma-separated module class names
        VLLM_FL_IO_DUMP_MAX_CALLS    - Max calls per op (0 = unlimited)
        VLLM_FL_IO_DUMP_STEP_RANGE   - "start,end" half-open range
        VLLM_FL_IO_DUMP_TORCH_FUNCS  - "1" or "matmul,softmax"

File layout:
    dump_dir/step_0005/rms_norm/order_000001_call_0001_{input,output}.pt
    dump_dir/step_0005/torch.matmul/order_000002_call_0001_{input,output}.pt
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ._io_common import (
    HAS_GLOBAL_MODULE_HOOKS,
    HAS_TORCH_FUNC_MODE,
    TorchFunctionMode,
    get_module_class_name,
    get_torch_func_name,
    guard_active,
    next_exec_order,
    parse_io_config_from_yaml,
    parse_torch_funcs_config,
    reset_exec_order,
    set_guard,
    should_inspect_torch_func,
)
from .logger_manager import get_logger

if HAS_GLOBAL_MODULE_HOOKS:
    from ._io_common import register_module_forward_hook, register_module_forward_pre_hook

logger = get_logger("vllm_fl.dispatch.io_dump")

# ── Module-level state ──

_enabled: bool = False
_dump_dir: str = ""
_match_all: bool = False
_op_filter: Set[str] = set()
_module_filter: Set[str] = set()
_max_calls: int = 0  # 0 = unlimited
_step_range: Optional[Tuple[int, int]] = None

_step_counter: int = 0
_call_counters: Dict[str, int] = {}
_lock = threading.Lock()

# Thread-local storage for pairing dump_before → dump_after.
# Each thread stores a dict of op_name → list of (call_num, exec_order, op_dir).
_dump_pairing = threading.local()

_torch_funcs_enabled: bool = False
_torch_func_filter: Set[str] = set()

# Hook handles
_hook_handles: List[Any] = []
_global_hook_handles: List[Any] = []
_torch_func_mode_instance: Optional[Any] = None


# ── Filtering ──


def _check_limits(op_name: str) -> bool:
    """Check step_range and max_calls limits (no filter logic)."""
    if _step_range is not None:
        if _step_counter < _step_range[0] or _step_counter >= _step_range[1]:
            return False
    if _max_calls > 0:
        if _call_counters.get(op_name, 0) >= _max_calls:
            return False
    return True


def _should_dump(op_name: str, args: tuple) -> bool:
    """Check if this op call should be dumped."""
    if not _check_limits(op_name):
        return False
    if _match_all:
        return True
    if op_name in _op_filter:
        return True
    if _module_filter:
        cls = get_module_class_name(args)
        if cls and cls in _module_filter:
            return True
    return False


def _should_dump_module(cls_name: str) -> bool:
    """Check if a module should be dumped (for global hooks)."""
    if _match_all:
        return True
    return cls_name in _module_filter


def _should_dump_torch_func(func_name: str) -> bool:
    """Check if a torch function should be dumped."""
    return should_inspect_torch_func(
        func_name, _torch_funcs_enabled, _torch_func_filter,
        _match_all, _op_filter,
    )


# ── Serialization ──


def _serialize_value(value: Any) -> Any:
    """Prepare a value for torch.save(). Tensors -> CPU, Modules -> string."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, torch.nn.Module):
        return f"<module:{type(value).__name__}>"
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_serialize_value(v) for v in value)
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    return f"<{type(value).__name__}>"


def _build_input_dict(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Build a serializable dict from operator inputs."""
    data: Dict[str, Any] = {}
    for i, arg in enumerate(args):
        data[f"arg_{i}"] = _serialize_value(arg)
    for k, v in kwargs.items():
        data[f"kwarg_{k}"] = _serialize_value(v)
    return data


def _build_output_dict(result: Any) -> Dict[str, Any]:
    """Build a serializable dict from operator output."""
    if isinstance(result, tuple):
        return {f"result_{i}": _serialize_value(v) for i, v in enumerate(result)}
    return {"result": _serialize_value(result)}


def _sanitize_path_component(name: str) -> str:
    """Sanitize a name for safe use as a single path component.

    Replaces path separators and '..' to prevent directory traversal.
    """
    # Replace OS path separators with underscores
    safe = name.replace(os.sep, "_")
    if os.altsep:
        safe = safe.replace(os.altsep, "_")
    # Collapse any remaining '..' sequences
    safe = safe.replace("..", "__")
    # Strip leading/trailing whitespace and dots
    safe = safe.strip(". ")
    return safe or "_unnamed_"


def _push_pairing(op_name: str, call_num: int, exec_order: int, op_dir: str) -> None:
    """Store pairing info in thread-local for dump_after to consume."""
    stack = getattr(_dump_pairing, "stack", None)
    if stack is None:
        _dump_pairing.stack = {}
        stack = _dump_pairing.stack
    stack.setdefault(op_name, []).append((call_num, exec_order, op_dir))


def _pop_pairing(op_name: str) -> Optional[Tuple[int, int, str]]:
    """Retrieve pairing info stored by the most recent dump_before for this op."""
    stack = getattr(_dump_pairing, "stack", None)
    if stack is None:
        return None
    entries = stack.get(op_name)
    if entries:
        return entries.pop()
    return None


def _get_call_dir(op_name: str, exec_order: Optional[int] = None) -> Tuple[str, int, int]:
    """Get dump directory and increment call counter.

    Returns (dir_path, call_number, exec_order).
    Stores pairing info in thread-local so dump_after can find the
    matching call_num and exec_order without shared mutable state.

    Args:
        exec_order: Pre-allocated execution order.  If *None*, one is
            allocated internally via ``next_exec_order()``.
    """
    order = exec_order if exec_order is not None else next_exec_order()
    safe_name = _sanitize_path_component(op_name)
    with _lock:
        count = _call_counters.get(op_name, 0) + 1
        _call_counters[op_name] = count
    step_dir = os.path.join(_dump_dir, f"step_{_step_counter:04d}")
    op_dir = os.path.join(step_dir, safe_name)
    os.makedirs(op_dir, exist_ok=True)
    _push_pairing(op_name, count, order, op_dir)
    return op_dir, count, order


# ── Dump helpers ──


def _dump_input(op_name: str, args: tuple, kwargs: dict,
                exec_order: Optional[int] = None) -> None:
    """Save operator inputs to disk.

    Args:
        exec_order: Pre-allocated execution order passed through to
            ``_get_call_dir``.  *None* means allocate internally.
    """
    try:
        op_dir, call_num, order = _get_call_dir(op_name, exec_order=exec_order)
        path = os.path.join(op_dir, f"order_{order:06d}_call_{call_num:04d}_input.pt")
        data = _build_input_dict(args, kwargs)
        data["__meta__"] = {"op_name": op_name, "exec_order": order, "call_num": call_num}
        torch.save(data, path)
        logger.debug(f"Dumped input: {path}")
    except Exception as e:
        logger.warning(f"Failed to dump input for '{op_name}': {e}")


def _dump_output(op_name: str, result: Any) -> None:
    """Save operator outputs to disk."""
    try:
        pairing = _pop_pairing(op_name)
        if pairing is None:
            logger.warning(f"No pairing info for dump output '{op_name}', skipping")
            return
        call_num, order, op_dir = pairing
        path = os.path.join(op_dir, f"order_{order:06d}_call_{call_num:04d}_output.pt")
        data = _build_output_dict(result)
        data["__meta__"] = {"op_name": op_name, "exec_order": order, "call_num": call_num}
        torch.save(data, path)
        logger.debug(f"Dumped output: {path}")
    except Exception as e:
        logger.warning(f"Failed to dump output for '{op_name}': {e}")


# ── Public API ──


def is_dump_enabled() -> bool:
    """Check if IO dumping is enabled (fast path)."""
    return _enabled


def dump_before(op_name: str, args: tuple, kwargs: dict,
                exec_order: Optional[int] = None) -> None:
    """Dump operator inputs (called from OpManager).

    Args:
        exec_order: Pre-allocated execution order shared with the inspector.
    """
    if guard_active():
        return
    if not _should_dump(op_name, args):
        return
    set_guard(True)
    try:
        _dump_input(op_name, args, kwargs, exec_order=exec_order)
    finally:
        set_guard(False)


def dump_after(op_name: str, args: tuple, result: Any) -> None:
    """Dump operator outputs (called from OpManager)."""
    if guard_active():
        return
    if not _should_dump(op_name, args):
        return
    set_guard(True)
    try:
        _dump_output(op_name, result)
    finally:
        set_guard(False)


def dump_cleanup(op_name: str) -> None:
    """Pop stale pairing left by dump_before when the op raises.

    Called from ``_call_with_hooks`` (and TorchFunctionMode) when the
    actual operator execution fails so that the pairing stack stays
    clean for subsequent calls.
    """
    _pop_pairing(op_name)


def io_dump_step() -> int:
    """Increment step counter and reset per-op call counters."""
    global _step_counter
    with _lock:
        _step_counter += 1
        _call_counters.clear()
    reset_exec_order()
    return _step_counter


def enable_io_dump(
    dump_dir: str,
    ops: Optional[Set[str]] = None,
    modules: Optional[Set[str]] = None,
    max_calls: int = 0,
    step_range: Optional[Tuple[int, int]] = None,
    torch_funcs: bool = False,
) -> None:
    """
    Programmatically enable IO dumping.

    Automatically registers global module hooks when modules are being
    dumped (match-all or specific module filter).

    Args:
        dump_dir: Directory to save dump files.
        ops: Dispatch-managed op names to dump. None = all.
        modules: nn.Module class names to dump. None = all.
        max_calls: Max calls per op to dump (0 = unlimited).
        step_range: (start, end) half-open range for step filtering.
        torch_funcs: Also intercept bare torch functional ops.
    """
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter
    global _max_calls, _step_range, _torch_funcs_enabled, _torch_func_filter

    _dump_dir = dump_dir
    os.makedirs(_dump_dir, exist_ok=True)

    if ops is None and modules is None:
        _match_all = True
        _op_filter = set()
        _module_filter = set()
    else:
        _match_all = False
        _op_filter = set(ops) if ops else set()
        _module_filter = set(modules) if modules else set()
    _max_calls = max_calls
    _step_range = step_range
    _torch_funcs_enabled = torch_funcs
    _torch_func_filter = set()
    _enabled = True

    _activate_hooks()

    logger.info(
        f"IO Dump enabled: dir={_dump_dir}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"max_calls={_max_calls}, step_range={_step_range}, "
        f"torch_funcs={_torch_funcs_enabled}"
    )


def disable_io_dump() -> None:
    """Programmatically disable IO dumping and remove all hooks."""
    _reset_state()
    _deactivate_hooks()
    remove_dump_hooks()


# ── Global Module Hooks (auto-enabled when modules are dumped) ──


def _global_forward_pre_hook(module, args):
    """Global pre-hook: dump inputs for matching modules."""
    if not _enabled or guard_active():
        return
    cls_name = type(module).__name__
    if not _should_dump_module(cls_name):
        return
    if not _check_limits(cls_name):
        return
    set_guard(True)
    try:
        _dump_input(cls_name, args, {})
    finally:
        set_guard(False)


def _global_forward_post_hook(module, args, output):
    """Global post-hook: dump outputs for matching modules."""
    if not _enabled or guard_active():
        return
    cls_name = type(module).__name__
    if not _should_dump_module(cls_name):
        return
    if not _check_limits(cls_name):
        return
    set_guard(True)
    try:
        _dump_output(cls_name, output)
    finally:
        set_guard(False)


# ── TorchFunctionMode (opt-in for bare torch ops) ──


if HAS_TORCH_FUNC_MODE:
    class _DumpTorchFuncMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if not _enabled or guard_active():
                return func(*args, **kwargs)

            func_name = get_torch_func_name(func)
            if not _should_dump_torch_func(func_name):
                return func(*args, **kwargs)

            op_name = f"torch.{func_name}"
            if not _check_limits(op_name):
                return func(*args, **kwargs)

            set_guard(True)
            try:
                _dump_input(op_name, args, kwargs)
            finally:
                set_guard(False)

            try:
                result = func(*args, **kwargs)
            except Exception:
                # Clean up stale pairing pushed by _dump_input
                _pop_pairing(op_name)
                raise

            set_guard(True)
            try:
                _dump_output(op_name, result)
            finally:
                set_guard(False)

            return result


# ── Hook Lifecycle ──


def _activate_hooks():
    """Register global module hooks and/or TorchFunctionMode as needed."""
    global _torch_func_mode_instance

    needs_module_hooks = _match_all or bool(_module_filter)
    if needs_module_hooks and HAS_GLOBAL_MODULE_HOOKS and not _global_hook_handles:
        h1 = register_module_forward_pre_hook(_global_forward_pre_hook)
        h2 = register_module_forward_hook(_global_forward_post_hook)
        _global_hook_handles.extend([h1, h2])

    if _torch_funcs_enabled and HAS_TORCH_FUNC_MODE and _torch_func_mode_instance is None:
        _torch_func_mode_instance = _DumpTorchFuncMode()
        _torch_func_mode_instance.__enter__()


def _deactivate_hooks():
    """Remove all global hooks and exit TorchFunctionMode."""
    global _torch_func_mode_instance

    for h in _global_hook_handles:
        h.remove()
    _global_hook_handles.clear()

    if _torch_func_mode_instance is not None:
        _torch_func_mode_instance.__exit__(None, None, None)
        _torch_func_mode_instance = None


# ── Per-Model Hook Support (backward compatible) ──


def _should_hook_module(module: torch.nn.Module, name: str) -> bool:
    """Check if a specific module instance should have hooks attached."""
    if _match_all:
        return True
    cls_name = type(module).__name__
    if cls_name in _module_filter:
        return True
    if name in _op_filter:
        return True
    return False


def _make_dump_pre_hook(label: str):
    def hook(module, args):
        if guard_active():
            return
        if not _check_limits(label):
            return
        set_guard(True)
        try:
            _dump_input(label, args, {})
        finally:
            set_guard(False)
    return hook


def _make_dump_post_hook(label: str):
    def hook(module, args, output):
        if guard_active():
            return
        if not _check_limits(label):
            return
        set_guard(True)
        try:
            _dump_output(label, output)
        finally:
            set_guard(False)
    return hook


def attach_dump_hooks(model: torch.nn.Module) -> int:
    """
    Attach IO dump hooks to specific submodules of a model.

    When using enable_io_dump() with a modules filter, global hooks
    are registered automatically. Use this for targeted per-model hooks.
    """
    if not _enabled:
        return 0
    count = 0
    for name, module in model.named_modules():
        if not _should_hook_module(module, name):
            continue
        label = name if name else type(module).__name__
        h1 = module.register_forward_pre_hook(_make_dump_pre_hook(label))
        h2 = module.register_forward_hook(_make_dump_post_hook(label))
        _hook_handles.extend([h1, h2])
        count += 1
    if count > 0:
        logger.info(f"[IO_DUMP] Attached dump hooks to {count} modules")
    return count


def remove_dump_hooks() -> None:
    """Remove all per-model IO dump hooks."""
    for h in _hook_handles:
        h.remove()
    _hook_handles.clear()


# ── Environment Initialization ──


def _reset_state() -> None:
    """Reset all module-level state to defaults."""
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter
    global _max_calls, _step_range, _step_counter
    global _torch_funcs_enabled, _torch_func_filter

    _enabled = False
    _dump_dir = ""
    _match_all = False
    _op_filter = set()
    _module_filter = set()
    _max_calls = 0
    _step_range = None
    _torch_funcs_enabled = False
    _torch_func_filter = set()
    with _lock:
        _step_counter = 0
        _call_counters.clear()


def _init_from_env() -> None:
    """Initialize from VLLM_FL_IO_DUMP* environment variables or YAML config."""
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter
    global _max_calls, _step_range, _torch_funcs_enabled, _torch_func_filter

    _deactivate_hooks()

    # Priority 1: YAML config via VLLM_FL_CONFIG
    config_path = os.environ.get("VLLM_FL_CONFIG", "").strip()
    if config_path:
        io_cfg = parse_io_config_from_yaml(config_path).get("io_dump")
        if io_cfg is not None:
            # YAML config is authoritative — if section exists, use it
            if not io_cfg.get("dir"):
                _reset_state()
                return
            _dump_dir = io_cfg["dir"]
            os.makedirs(_dump_dir, exist_ok=True)

            ops = io_cfg.get("ops", set())
            modules = io_cfg.get("modules", set())
            if not ops and not modules:
                _match_all = True
                _op_filter = set()
                _module_filter = set()
            else:
                _match_all = False
                _op_filter = set(ops)
                _module_filter = set(modules)

            _max_calls = io_cfg.get("max_calls", 0)
            _step_range = io_cfg.get("step_range")

            tf_enabled, tf_filter = io_cfg.get("torch_funcs", (False, set()))
            _torch_funcs_enabled = tf_enabled
            _torch_func_filter = tf_filter

            _enabled = True
            _activate_hooks()

            logger.info(
                f"IO Dump enabled (YAML): dir={_dump_dir}, "
                f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
                f"max_calls={_max_calls}, step_range={_step_range}, "
                f"torch_funcs={_torch_funcs_enabled}"
            )
            return

    # Priority 2: Environment variables
    dump_dir = os.environ.get("VLLM_FL_IO_DUMP", "").strip()
    if not dump_dir:
        _reset_state()
        return

    _dump_dir = dump_dir
    os.makedirs(_dump_dir, exist_ok=True)

    ops_str = os.environ.get("VLLM_FL_IO_DUMP_OPS", "").strip()
    modules_str = os.environ.get("VLLM_FL_IO_DUMP_MODULES", "").strip()
    _op_filter = {t.strip() for t in ops_str.split(",") if t.strip()} if ops_str else set()
    _module_filter = {t.strip() for t in modules_str.split(",") if t.strip()} if modules_str else set()
    _match_all = not _op_filter and not _module_filter

    max_calls_str = os.environ.get("VLLM_FL_IO_DUMP_MAX_CALLS", "0").strip()
    try:
        _max_calls = int(max_calls_str)
    except ValueError:
        _max_calls = 0

    step_range_str = os.environ.get("VLLM_FL_IO_DUMP_STEP_RANGE", "").strip()
    if step_range_str:
        parts = step_range_str.split(",")
        if len(parts) == 2:
            try:
                _step_range = (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                _step_range = None
        else:
            _step_range = None
    else:
        _step_range = None

    torch_funcs_val = os.environ.get("VLLM_FL_IO_DUMP_TORCH_FUNCS", "")
    if torch_funcs_val:
        _torch_funcs_enabled, _torch_func_filter = parse_torch_funcs_config(
            torch_funcs_val
        )
    else:
        _torch_funcs_enabled = False
        _torch_func_filter = set()

    _enabled = True
    _activate_hooks()

    logger.info(
        f"IO Dump enabled: dir={_dump_dir}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"max_calls={_max_calls}, step_range={_step_range}, "
        f"torch_funcs={_torch_funcs_enabled}"
    )


_init_from_env()
