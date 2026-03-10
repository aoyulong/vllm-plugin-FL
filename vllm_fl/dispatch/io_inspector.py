# Copyright (c) 2026 BAAI. All rights reserved.

"""
IO Inspector for vllm-plugin-FL dispatch.

Prints operator input/output metadata (tensor shapes, dtypes, devices)
before and after each operator call.

Supports three interception mechanisms:
1. Dispatch-managed ops: Automatic via OpManager.call() hooks
2. nn.Module forward passes: Automatic via global module hooks (when
   modules filter is set or inspect-all mode)
3. Bare torch functions: Via TorchFunctionMode (opt-in)

Configuration (priority order):
    1. YAML config file (via VLLM_FL_CONFIG):
        io_inspect:
          enabled: true
          ops: [rms_norm, silu_and_mul]
          modules: [Linear, RMSNormFL]
          torch_funcs: true
    2. Environment variables:
        VLLM_FL_IO_INSPECT:
            "1"                     - Inspect all operators
            "op1,op2"               - Inspect specific dispatch-managed operators
            "module:ClassA"         - Inspect by module class name
            "op1,module:ClassA"     - Mix op names and module names
        VLLM_FL_IO_INSPECT_TORCH_FUNCS:
            "1"                     - Inspect all torch functional ops
            "matmul,softmax"        - Inspect specific torch functions
        VLLM_FL_IO_RANK:
            "all"                   - Inspect on all ranks (default)
            "0"                     - Inspect only on rank 0
            "0,2,4"                 - Inspect only on ranks 0, 2, 4
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Set, Tuple

import torch

from ._io_common import (
    HAS_GLOBAL_MODULE_HOOKS,
    HAS_TORCH_FUNC_MODE,
    TorchFunctionMode,
    format_result,
    format_value,
    get_module_class_name,
    get_rank,
    get_torch_func_name,
    guard_active,
    next_exec_order,
    parse_io_config_from_yaml,
    parse_rank_filter,
    parse_torch_funcs_config,
    rank_enabled,
    set_guard,
    set_rank_filter,
    should_inspect_torch_func,
)
from .logger_manager import get_logger

if HAS_GLOBAL_MODULE_HOOKS:
    from ._io_common import register_module_forward_hook, register_module_forward_pre_hook

logger = get_logger("vllm_fl.dispatch.io_inspect")

# ── Module-level state ──

_enabled: bool = False
_match_all: bool = False
_op_filter: Set[str] = set()
_module_filter: Set[str] = set()
_torch_funcs_enabled: bool = False
_torch_func_filter: Set[str] = set()

# Hook handles
_hook_handles: List[Any] = []          # Per-model hooks (attach_io_hooks)
_global_hook_handles: List[Any] = []   # Global module hooks
_torch_func_mode_instance: Optional[Any] = None


# ── Filtering ──


def _should_inspect(op_name: str, args: tuple) -> bool:
    """Check if this dispatch-managed op call should be inspected."""
    if _match_all:
        return True
    if op_name in _op_filter:
        return True
    if _module_filter:
        cls = get_module_class_name(args)
        if cls and cls in _module_filter:
            return True
    return False


def _should_inspect_module(cls_name: str) -> bool:
    """Check if a module should be inspected (for global hooks)."""
    if _match_all:
        return True
    return cls_name in _module_filter


def _should_inspect_torch_func(func_name: str) -> bool:
    """Check if a torch function should be inspected."""
    return should_inspect_torch_func(
        func_name, _torch_funcs_enabled, _torch_func_filter,
        _match_all, _op_filter,
    )


def _parse_config(value: str) -> Tuple[bool, Set[str], Set[str]]:
    """Parse VLLM_FL_IO_INSPECT value into (match_all, op_filter, module_filter)."""
    value = value.strip()
    if not value or value == "0":
        return False, set(), set()
    if value == "1":
        return True, set(), set()
    ops: Set[str] = set()
    modules: Set[str] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("module:"):
            modules.add(token[7:])
        else:
            ops.add(token)
    return False, ops, modules


# ── Logging helpers ──


def _log_inputs(label: str, args: tuple, kwargs: dict,
                skip_module_arg: bool = False,
                exec_order: Optional[int] = None) -> None:
    """Log operator/module inputs."""
    rank = get_rank()
    order_str = f" #{exec_order}" if exec_order is not None else ""
    lines = [f"[IO_INSPECT][rank={rank}]{order_str} {label} INPUTS:"]
    for i, arg in enumerate(args):
        if skip_module_arg and i == 0 and isinstance(arg, torch.nn.Module):
            continue
        lines.append(f"  arg[{i}]: {format_value(arg)}")
    for k, v in kwargs.items():
        lines.append(f"  {k}: {format_value(v)}")
    logger.info("\n".join(lines))


def _log_outputs(label: str, result: Any,
                 exec_order: Optional[int] = None) -> None:
    """Log operator/module outputs."""
    rank = get_rank()
    order_str = f" #{exec_order}" if exec_order is not None else ""
    logger.info(f"[IO_INSPECT][rank={rank}]{order_str} {label} OUTPUTS:\n{format_result(result)}")


# ── Public API ──


def is_inspect_enabled() -> bool:
    """Check if IO inspection is enabled (fast path)."""
    return _enabled


def inspect_before(op_name: str, args: tuple, kwargs: dict,
                   exec_order: Optional[int] = None) -> None:
    """Log operator inputs before execution (called from OpManager).

    Args:
        exec_order: Pre-allocated execution order number.  When called from
            ``_call_with_hooks`` a single order is shared with the dumper so
            that log lines and dump files can be correlated.  If *None*, an
            order is allocated internally (standalone usage).
    """
    if guard_active():
        return
    if not rank_enabled():
        return
    if not _should_inspect(op_name, args):
        return
    order = exec_order if exec_order is not None else next_exec_order()
    module_name = get_module_class_name(args)
    module_str = f" (module={module_name})" if module_name else ""
    set_guard(True)
    try:
        _log_inputs(f"Op '{op_name}'{module_str}", args, kwargs,
                    skip_module_arg=True, exec_order=order)
    finally:
        set_guard(False)


def inspect_after(op_name: str, args: tuple, result: Any,
                  exec_order: Optional[int] = None) -> None:
    """Log operator outputs after execution (called from OpManager).

    Args:
        exec_order: Pre-allocated execution order (for log correlation).
    """
    if guard_active():
        return
    if not rank_enabled():
        return
    if not _should_inspect(op_name, args):
        return
    module_name = get_module_class_name(args)
    module_str = f" (module={module_name})" if module_name else ""
    set_guard(True)
    try:
        _log_outputs(f"Op '{op_name}'{module_str}", result, exec_order=exec_order)
    finally:
        set_guard(False)


def enable_io_inspect(
    ops: Optional[Set[str]] = None,
    modules: Optional[Set[str]] = None,
    torch_funcs: bool = False,
    ranks: Optional[Set[int]] = None,
) -> None:
    """
    Programmatically enable IO inspection.

    Automatically registers global module hooks when modules are being
    inspected (match-all or specific module filter). Does NOT register
    global hooks for ops-only filtering to avoid unnecessary overhead.

    Args:
        ops: Dispatch-managed op names to inspect. None = all.
        modules: nn.Module class names to inspect. None = all.
        torch_funcs: Also intercept bare torch functional ops.
        ranks: Set of ranks to inspect on. None = all ranks.
    """
    global _enabled, _match_all, _op_filter, _module_filter
    global _torch_funcs_enabled, _torch_func_filter

    if ops is None and modules is None:
        _match_all = True
        _op_filter = set()
        _module_filter = set()
    else:
        _match_all = False
        _op_filter = set(ops) if ops else set()
        _module_filter = set(modules) if modules else set()

    _torch_funcs_enabled = torch_funcs
    _torch_func_filter = set()
    _enabled = True

    set_rank_filter(ranks)
    _activate_hooks()

    logger.info(
        f"IO Inspect enabled: rank={get_rank()}, "
        f"rank_filter={ranks or 'all'}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"torch_funcs={_torch_funcs_enabled}"
    )


def disable_io_inspect() -> None:
    """Programmatically disable IO inspection and remove all hooks."""
    _reset_state()
    _deactivate_hooks()
    remove_io_hooks()


# ── Global Module Hooks (auto-enabled when modules are inspected) ──


def _global_forward_pre_hook(module, args):
    """Global pre-hook: log inputs for matching modules."""
    if not _enabled or guard_active() or not rank_enabled():
        return
    cls_name = type(module).__name__
    if not _should_inspect_module(cls_name):
        return
    order = next_exec_order()
    set_guard(True)
    try:
        _log_inputs(f"Module '{cls_name}'", args, {}, exec_order=order)
    finally:
        set_guard(False)


def _global_forward_post_hook(module, args, output):
    """Global post-hook: log outputs for matching modules."""
    if not _enabled or guard_active() or not rank_enabled():
        return
    cls_name = type(module).__name__
    if not _should_inspect_module(cls_name):
        return
    set_guard(True)
    try:
        _log_outputs(f"Module '{cls_name}'", output)
    finally:
        set_guard(False)


# ── TorchFunctionMode (opt-in for bare torch ops) ──


if HAS_TORCH_FUNC_MODE:
    class _InspectTorchFuncMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if not _enabled or guard_active() or not rank_enabled():
                return func(*args, **kwargs)

            func_name = get_torch_func_name(func)
            if not _should_inspect_torch_func(func_name):
                return func(*args, **kwargs)

            label = f"torch.{func_name}"
            order = next_exec_order()
            set_guard(True)
            try:
                _log_inputs(label, args, kwargs, exec_order=order)
            finally:
                set_guard(False)

            result = func(*args, **kwargs)

            set_guard(True)
            try:
                _log_outputs(label, result, exec_order=order)
            finally:
                set_guard(False)

            return result


# ── Hook Lifecycle ──


def _activate_hooks():
    """Register global module hooks and/or TorchFunctionMode as needed."""
    global _torch_func_mode_instance

    # Only register global module hooks when modules need inspection.
    # Skip for ops-only mode to avoid per-forward overhead.
    needs_module_hooks = _match_all or bool(_module_filter)
    if needs_module_hooks and HAS_GLOBAL_MODULE_HOOKS and not _global_hook_handles:
        h1 = register_module_forward_pre_hook(_global_forward_pre_hook)
        h2 = register_module_forward_hook(_global_forward_post_hook)
        _global_hook_handles.extend([h1, h2])

    if _torch_funcs_enabled and HAS_TORCH_FUNC_MODE and _torch_func_mode_instance is None:
        _torch_func_mode_instance = _InspectTorchFuncMode()
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


def _make_pre_hook(name: str):
    def hook(module, args):
        if guard_active():
            return
        cls_name = type(module).__name__
        label = f"{name} ({cls_name})" if name else cls_name
        order = next_exec_order()
        set_guard(True)
        try:
            _log_inputs(f"Module '{label}'", args, {}, exec_order=order)
        finally:
            set_guard(False)
    return hook


def _make_post_hook(name: str):
    def hook(module, args, output):
        if guard_active():
            return
        cls_name = type(module).__name__
        label = f"{name} ({cls_name})" if name else cls_name
        set_guard(True)
        try:
            _log_outputs(f"Module '{label}'", output)
        finally:
            set_guard(False)
    return hook


def attach_io_hooks(model: torch.nn.Module) -> int:
    """
    Attach IO inspect hooks to specific submodules of a model.

    When using enable_io_inspect() with a modules filter, global hooks
    are registered automatically. Use this function only when you need
    hooks on specific named submodules.
    """
    if not _enabled:
        return 0
    count = 0
    for name, module in model.named_modules():
        if not _should_hook_module(module, name):
            continue
        h1 = module.register_forward_pre_hook(_make_pre_hook(name))
        h2 = module.register_forward_hook(_make_post_hook(name))
        _hook_handles.extend([h1, h2])
        count += 1
    if count > 0:
        logger.info(f"[IO_INSPECT] Attached hooks to {count} modules")
    return count


def remove_io_hooks() -> None:
    """Remove all per-model IO inspect hooks."""
    for h in _hook_handles:
        h.remove()
    _hook_handles.clear()


def _reset_state() -> None:
    """Reset all module-level state to defaults."""
    global _enabled, _match_all, _op_filter, _module_filter
    global _torch_funcs_enabled, _torch_func_filter

    _enabled = False
    _match_all = False
    _op_filter = set()
    _module_filter = set()
    _torch_funcs_enabled = False
    _torch_func_filter = set()


# ── Environment Initialization ──


def _init_from_env() -> None:
    """Initialize from VLLM_FL_IO_INSPECT* environment variables or YAML config."""
    global _enabled, _match_all, _op_filter, _module_filter
    global _torch_funcs_enabled, _torch_func_filter

    # Reset state first
    _deactivate_hooks()

    # Priority 1: YAML config via VLLM_FL_CONFIG
    config_path = os.environ.get("VLLM_FL_CONFIG", "").strip()
    if config_path:
        io_cfg = parse_io_config_from_yaml(config_path).get("io_inspect")
        if io_cfg is not None:
            # YAML config is authoritative — if section exists, use it
            if not io_cfg.get("enabled", False):
                _reset_state()
                return
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
            tf_enabled, tf_filter = io_cfg.get("torch_funcs", (False, set()))
            _torch_funcs_enabled = tf_enabled
            _torch_func_filter = tf_filter
            _enabled = True

            set_rank_filter(io_cfg.get("ranks"))
            _activate_hooks()

            logger.info(
                f"IO Inspect enabled (YAML): rank={get_rank()}, "
                f"rank_filter={io_cfg.get('ranks') or 'all'}, "
                f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
                f"torch_funcs={_torch_funcs_enabled}"
            )
            return

    # Priority 2: Environment variables
    env_val = os.environ.get("VLLM_FL_IO_INSPECT", "")
    if not env_val or env_val == "0":
        _reset_state()
        return

    _match_all, _op_filter, _module_filter = _parse_config(env_val)
    _enabled = _match_all or bool(_op_filter) or bool(_module_filter)

    torch_funcs_val = os.environ.get("VLLM_FL_IO_INSPECT_TORCH_FUNCS", "")
    if torch_funcs_val:
        _torch_funcs_enabled, _torch_func_filter = parse_torch_funcs_config(
            torch_funcs_val
        )
    else:
        _torch_funcs_enabled = False
        _torch_func_filter = set()

    if _enabled:
        # Parse rank filter from env var (shared by inspector and dumper)
        rank_env = os.environ.get("VLLM_FL_IO_RANK", "")
        if rank_env:
            set_rank_filter(parse_rank_filter(rank_env))

        _activate_hooks()

        logger.info(
            f"IO Inspect enabled: rank={get_rank()}, "
            f"rank_filter={parse_rank_filter(rank_env) if rank_env else 'all'}, "
            f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
            f"torch_funcs={_torch_funcs_enabled}"
        )


_init_from_env()
