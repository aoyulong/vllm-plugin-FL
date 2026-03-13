# Copyright (c) 2026 BAAI. All rights reserved.

"""
IO Inspector for vllm-plugin-FL dispatch.

Prints operator input/output metadata (tensor shapes, dtypes, statistics)
before and after each operator call.

Only logs at the **op level** (dispatch-managed ops and bare torch functions).
Module-level logging is not produced; instead, module names are appended
to op log entries so users can see which module invoked each op.

When a ``modules`` filter is set (e.g., ``modules={"RMSNormFL"}``), global
module hooks track which module is executing.  Only ops and torch functions
that run inside the specified modules are logged.

Supports two interception mechanisms:
1. Dispatch-managed ops: Automatic via OpManager.call() hooks
2. Bare torch functions: Via TorchFunctionMode (opt-in)

Configuration (priority order):
    1. Python API: enable_io_inspect(ops=..., step_range=..., layers=...)
    2. YAML config file (via VLLM_FL_CONFIG):
        io_inspect:
          enabled: true
          ops: [rms_norm, silu_and_mul]
          modules: [RMSNormFL, Qwen3Attention]
          layers: [0, 1-3, model.layers.*.self_attn]
          torch_funcs: true
          step_range: [5, 15]     # half-open [start, end)
    3. Environment variables:
        VLLM_FL_IO_INSPECT:
            "1"                     - Inspect all operators
            "op1,op2"               - Inspect specific dispatch-managed operators
            "module:ClassA"         - Inspect ops within module ClassA
            "op1,module:ClassA"     - Mix op names and module scoping
        VLLM_FL_IO_INSPECT_TORCH_FUNCS:
            "1"                     - Inspect all torch functional ops
            "matmul,softmax"        - Inspect specific torch functions
        VLLM_FL_IO_INSPECT_STEP_RANGE:
            "5-14"                  - Only inspect during steps 5 to 14 (inclusive)
            "0"                     - Only inspect step 0
        VLLM_FL_IO_STEP_RANGE:
            "5-14"                  - Shared step range (applies to both inspector and dumper)
        VLLM_FL_IO_INSPECT_LAYERS:
            "0,1-3,model.layers.*.self_attn"
                                    - Layer specs: integers, ranges, globs, full paths
        VLLM_FL_IO_LAYERS:
            "model.layers.0"       - Shared layer filter (applies to both inspector and dumper)
        VLLM_FL_IO_INSPECT_RANK:
            "all"                   - Inspect on all ranks (default)
            "0"                     - Inspect only on rank 0
            "0,2,4"                 - Inspect only on ranks 0, 2, 4
        VLLM_FL_IO_RANK:
            Shared rank filter (fallback if VLLM_FL_IO_INSPECT_RANK is unset)

    Note: Setting any filter env var (step_range or layers) auto-enables
    inspection for all ops, even without VLLM_FL_IO_INSPECT=1.

    Quick start (env-var only, no Python API needed):
        VLLM_FL_IO_STEP_RANGE=0-2 VLLM_FL_IO_LAYERS=1-3 python script.py

All filter dimensions are composable (AND logic): step_range, layers, modules,
and ops are orthogonal gates.  When multiple filters are set, an op must pass
ALL of them.  Unset filters are pass-through.
"""

from __future__ import annotations

import os
import threading
from typing import Any, List, Optional, Set, Tuple

import torch

from ._io_common import (
    HAS_TORCH_FUNC_MODE,
    TorchFunctionMode,
    acquire_global_module_hooks,
    acquire_torch_func_tags,
    advance_step,
    expand_layer_specs,
    format_result,
    format_value,
    get_module_class_name,
    get_rank,
    get_step,
    get_torch_func_name,
    layer_path_matches,
    make_guard,
    make_label,
    make_module_tag,
    make_op_tag,
    module_context_matches,
    next_exec_order,
    parse_io_config_from_yaml,
    parse_layers_env,
    parse_rank_filter,
    parse_step_range_env,
    parse_torch_funcs_config,
    pop_module_context,
    push_module_context,
    rank_enabled,
    record_seen,
    register_step_callback,
    release_global_module_hooks,
    release_torch_func_tags,
    set_rank_filter,
    should_inspect_torch_func,
    unregister_step_callback,
)
from .logger_manager import get_logger

logger = get_logger("vllm_fl.dispatch.io_inspect")

# ── Module-level state ──

# Independent re-entrancy guard so inspector doesn't block the dumper
guard_active, set_guard = make_guard()

_enabled: bool = False
_match_all: bool = False
_op_filter: Set[str] = set()
_module_filter: Set[str] = set()
_layer_filter: Set[str] = set()
_step_range: Optional[Tuple[int, int]] = None
_torch_funcs_enabled: bool = False
_torch_func_filter: Set[str] = set()

# Hook handles
_hook_handles: List[Any] = []          # Per-model hooks (attach_io_hooks)
_torch_func_mode_instance: Optional[Any] = None
_owns_global_hooks: bool = False       # Whether we acquired global module hooks

# Thread-local storage for pairing inspect_before → inspect_after
_inspect_pairing = threading.local()


# ── Filtering ──


def _should_inspect(op_name: str, args: tuple) -> bool:
    """Check if this dispatch-managed op call should be inspected.

    Filters are composable AND gates — each active filter must pass:
    - ``_step_range``: step must be in range
    - ``_layer_filter``: current layer path must match (prefix)
    - ``_module_filter``: must be inside a matching module
    - ``_op_filter``: op name must be in the filter set
    """
    if _step_range is not None:
        step = get_step()
        if step < _step_range[0] or step >= _step_range[1]:
            return False
    if _layer_filter and not layer_path_matches(_layer_filter):
        return False
    if _match_all:
        return True
    if _module_filter:
        cls = get_module_class_name(args)
        if not (cls and cls in _module_filter) and not module_context_matches(_module_filter):
            return False
    if _op_filter:
        if op_name not in _op_filter:
            return False
    return bool(_op_filter) or bool(_module_filter)


def _should_inspect_torch_func(func_name: str) -> bool:
    """Check if a torch function should be inspected.

    Layer and module filters are AND gates (same as ``_should_inspect``).
    """
    if _step_range is not None:
        step = get_step()
        if step < _step_range[0] or step >= _step_range[1]:
            return False
    if _layer_filter and not layer_path_matches(_layer_filter):
        return False
    if not should_inspect_torch_func(
        func_name, _torch_funcs_enabled, _torch_func_filter,
        _match_all, _op_filter,
    ):
        return False
    # When module filter is active, only match torch funcs inside those modules
    if _module_filter and not _match_all:
        return module_context_matches(_module_filter)
    return True


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


def _format_inputs(args: tuple, kwargs: dict,
                   skip_module_arg: bool = False) -> List[str]:
    """Format operator/module inputs into lines."""
    lines = []
    for i, arg in enumerate(args):
        if skip_module_arg and i == 0 and isinstance(arg, torch.nn.Module):
            continue
        lines.append(f"  arg[{i}]: {format_value(arg)}")
    for k, v in kwargs.items():
        lines.append(f"  {k}: {format_value(v)}")
    return lines


def _log_combined(label: str, input_lines: List[str], result: Any,
                  module_tag: str = "",
                  op_tag: str = "") -> None:
    """Log a consolidated INPUTS + OUTPUTS block for an op call."""
    rank = get_rank()
    step = get_step()
    sep = "-" * 60
    parts = [
        f"\n{sep}",
        f"[IO_INSPECT][rank={rank}][step={step}]{module_tag}{op_tag} {label}",
        f"{sep}",
        "  INPUTS:",
    ]
    parts.extend(f"  {line}" for line in input_lines)
    parts.append("  OUTPUTS:")
    parts.append(f"  {format_result(result)}")
    parts.append(sep)
    logger.info("\n".join(parts))


def _log_outputs_only(label: str, result: Any,
                      module_tag: str = "",
                      op_tag: str = "") -> None:
    """Log operator/module outputs only (fallback when no pairing)."""
    rank = get_rank()
    step = get_step()
    logger.info(f"[IO_INSPECT][rank={rank}][step={step}]{module_tag}{op_tag} {label} OUTPUTS:\n{format_result(result)}")


# ── Public API ──


def is_inspect_enabled() -> bool:
    """Check if IO inspection is enabled (fast path)."""
    return _enabled


def inspect_before(op_name: str, args: tuple, kwargs: dict,
                   exec_order: Optional[int] = None,
                   module_tag: Optional[str] = None,
                   op_tag: Optional[str] = None) -> None:
    """Capture operator inputs before execution (called from OpManager).

    Inputs are stored in thread-local and combined with outputs in
    ``inspect_after`` to produce a single consolidated log block.

    Args:
        exec_order: Pre-allocated execution order number.  When called from
            ``_call_with_hooks`` a single order is shared with the dumper so
            that log lines and dump files can be correlated.  If *None*, an
            order is allocated internally (standalone usage).
        module_tag: Pre-computed module counter tag (e.g. ``[module=0,1]``).
            When *None*, generated internally.
        op_tag: Pre-computed op counter tag (e.g. ``[op=3,2]``).
            When *None*, generated internally.
    """
    if guard_active():
        return
    if not rank_enabled():
        return
    if not _should_inspect(op_name, args):
        return
    order = exec_order if exec_order is not None else next_exec_order()
    label = make_label(f"Op '{op_name}'", args)
    _module_tag = module_tag if module_tag is not None else make_module_tag()
    _op_tag = op_tag if op_tag is not None else make_op_tag(op_name)
    record_seen(op_name, args)
    set_guard(True)
    try:
        input_lines = _format_inputs(args, kwargs, skip_module_arg=True)
    finally:
        set_guard(False)
    # Store for inspect_after to combine
    stack = getattr(_inspect_pairing, "stack", None)
    if stack is None:
        _inspect_pairing.stack = {}
        stack = _inspect_pairing.stack
    stack.setdefault(op_name, []).append((label, order, input_lines, _module_tag, _op_tag))


def inspect_after(op_name: str, args: tuple, result: Any,
                  exec_order: Optional[int] = None) -> None:
    """Log consolidated INPUTS+OUTPUTS block (called from OpManager).

    Retrieves inputs captured by ``inspect_before`` and combines them
    with the result into a single log message.

    Args:
        exec_order: Pre-allocated execution order (for log correlation).
    """
    if guard_active():
        return
    if not rank_enabled():
        return
    if not _should_inspect(op_name, args):
        return
    # Retrieve stored inputs from inspect_before
    pairing = None
    stack = getattr(_inspect_pairing, "stack", None)
    if stack:
        entries = stack.get(op_name)
        if entries:
            pairing = entries.pop()
    set_guard(True)
    try:
        if pairing:
            label, order, input_lines, module_tag, op_tag = pairing
            _log_combined(label, input_lines, result,
                          module_tag=module_tag, op_tag=op_tag)
        else:
            # Fallback: no pairing (shouldn't happen normally)
            module_tag = make_module_tag()
            op_tag = make_op_tag(op_name)
            _log_outputs_only(make_label(f"Op '{op_name}'", args), result,
                              module_tag=module_tag, op_tag=op_tag)
    finally:
        set_guard(False)


def enable_io_inspect(
    ops: Optional[Set[str]] = None,
    modules: Optional[Set[str]] = None,
    layers: Optional[Set[str]] = None,
    torch_funcs: bool = True,
    ranks: Optional[Set[int]] = None,
    step_range: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Programmatically enable IO inspection.

    All filter dimensions are composable (AND logic): when multiple
    filters are set, an op must satisfy ALL of them to be logged.

    Args:
        ops: Dispatch-managed op names to inspect. None = all.
        modules: nn.Module class names to scope inspection to.
            When set, only ops/torch_funcs executing inside these modules
            are logged.  None = no module scoping (inspect everywhere).
        layers: Layer specifications to scope inspection to.  Supports
            integer shorthand (``"0"`` → ``"model.layers.0"``),
            ranges (``"0-3"``), glob patterns (``"model.layers.*.self_attn"``),
            and full paths.  None = no layer scoping.
        torch_funcs: Intercept bare torch functional ops. Default True
            (all intercepted). Set False to skip torch functions.
        ranks: Set of ranks to inspect on. None = all ranks.
        step_range: (start, end) half-open step range [start, end).
            Only log ops whose step falls in this range. None = all steps.
    """
    global _enabled, _match_all, _op_filter, _module_filter, _layer_filter
    global _torch_funcs_enabled, _torch_func_filter, _step_range

    if ops is None and modules is None:
        _match_all = True
        _op_filter = set()
        _module_filter = set()
    else:
        _match_all = False
        _op_filter = set(ops) if ops else set()
        _module_filter = set(modules) if modules else set()

    # If no layers given explicitly, check env vars (already expanded)
    if layers is None:
        layers = parse_layers_env(
            "VLLM_FL_IO_INSPECT_LAYERS", "VLLM_FL_IO_LAYERS"
        )
    else:
        layers = expand_layer_specs(layers)
    _layer_filter = set(layers) if layers else set()

    _torch_funcs_enabled = torch_funcs
    _torch_func_filter = set()
    # If no step_range given explicitly, check env vars
    if step_range is None:
        step_range = parse_step_range_env(
            "VLLM_FL_IO_INSPECT_STEP_RANGE", "VLLM_FL_IO_STEP_RANGE"
        )
    _step_range = step_range
    _enabled = True

    set_rank_filter(ranks)
    _activate_hooks()

    # Propagate to env vars so child processes (e.g. vLLM EngineCore workers)
    # pick up the config via _init_from_env() on module import.
    _set_env_vars(ops, modules, _layer_filter, torch_funcs, ranks, step_range)

    logger.info(
        f"IO Inspect enabled: rank={get_rank()}, "
        f"rank_filter={ranks or 'all'}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"layers={_layer_filter or 'all'}, "
        f"torch_funcs={_torch_funcs_enabled}, step_range={_step_range}"
    )


def disable_io_inspect() -> None:
    """Programmatically disable IO inspection and remove all hooks."""
    _reset_state()
    _deactivate_hooks()
    remove_io_hooks()
    _clear_env_vars()


# ── Env-var propagation for child processes ──


def _set_env_vars(
    ops: Optional[Set[str]],
    modules: Optional[Set[str]],
    layers: Set[str],
    torch_funcs: bool,
    ranks: Optional[Set[int]],
    step_range: Optional[Tuple[int, int]],
) -> None:
    """Set VLLM_FL_IO_INSPECT* env vars so child processes inherit the config."""
    if ops is None and modules is None:
        os.environ["VLLM_FL_IO_INSPECT"] = "1"
    else:
        tokens = []
        if ops:
            tokens.extend(sorted(ops))
        if modules:
            tokens.extend(f"module:{m}" for m in sorted(modules))
        os.environ["VLLM_FL_IO_INSPECT"] = ",".join(tokens) if tokens else "1"

    if torch_funcs:
        os.environ["VLLM_FL_IO_INSPECT_TORCH_FUNCS"] = "1"
    else:
        os.environ.pop("VLLM_FL_IO_INSPECT_TORCH_FUNCS", None)

    if ranks is not None:
        os.environ["VLLM_FL_IO_INSPECT_RANK"] = ",".join(str(r) for r in sorted(ranks))
    else:
        os.environ.pop("VLLM_FL_IO_INSPECT_RANK", None)

    if step_range is not None:
        os.environ["VLLM_FL_IO_INSPECT_STEP_RANGE"] = f"{step_range[0]}-{step_range[1] - 1}"
    else:
        os.environ.pop("VLLM_FL_IO_INSPECT_STEP_RANGE", None)

    if layers:
        os.environ["VLLM_FL_IO_INSPECT_LAYERS"] = ",".join(sorted(layers))
    else:
        os.environ.pop("VLLM_FL_IO_INSPECT_LAYERS", None)


def _clear_env_vars() -> None:
    """Remove VLLM_FL_IO_INSPECT* env vars."""
    os.environ.pop("VLLM_FL_IO_INSPECT", None)
    os.environ.pop("VLLM_FL_IO_INSPECT_TORCH_FUNCS", None)
    os.environ.pop("VLLM_FL_IO_INSPECT_RANK", None)
    os.environ.pop("VLLM_FL_IO_INSPECT_STEP_RANGE", None)
    os.environ.pop("VLLM_FL_IO_INSPECT_LAYERS", None)


# ── TorchFunctionMode (opt-in for bare torch ops) ──


if HAS_TORCH_FUNC_MODE:
    class _InspectTorchFuncMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if torch.compiler.is_compiling() or not _enabled or guard_active() or not rank_enabled():
                return func(*args, **kwargs)

            func_name = get_torch_func_name(func)
            if not _should_inspect_torch_func(func_name):
                return func(*args, **kwargs)

            # Include enclosing module context in label
            full_name = f"torch.{func_name}"
            label = make_label(full_name)
            module_tag, op_tag, order = acquire_torch_func_tags(full_name)
            record_seen(full_name)
            set_guard(True)
            try:
                input_lines = _format_inputs(args, kwargs)
            finally:
                set_guard(False)

            try:
                result = func(*args, **kwargs)
            finally:
                release_torch_func_tags()

            set_guard(True)
            try:
                _log_combined(label, input_lines, result,
                              module_tag=module_tag, op_tag=op_tag)
            finally:
                set_guard(False)

            return result


# ── Step summary callback ──


def _on_step_summary(step: int, seen_modules: Set[str], seen_ops: Set[str]) -> None:
    """Log a summary of modules and ops seen during the completed step."""
    if not seen_modules and not seen_ops:
        return
    rank = get_rank()
    logger.info(
        f"[IO_INSPECT][rank={rank}][step={step}] Step summary: "
        f"modules={sorted(seen_modules)}, ops={sorted(seen_ops)}"
    )


# ── Hook Lifecycle ──


def _activate_hooks():
    """Register global module hooks and/or TorchFunctionMode as needed."""
    global _torch_func_mode_instance, _owns_global_hooks

    # Register global module hooks to track module context.
    # In match-all mode, this provides module annotations on op/torch_func
    # log entries (e.g. "Op 'rms_norm' (module=RMSNormFL)").
    # When module filter or layer filter is set, this also enables
    # scope-based filtering (layer filter needs paths from the hook).
    needs_module_hooks = _match_all or bool(_module_filter) or bool(_layer_filter)
    if needs_module_hooks and not _owns_global_hooks:
        acquire_global_module_hooks()
        _owns_global_hooks = True

    if _torch_funcs_enabled and HAS_TORCH_FUNC_MODE and _torch_func_mode_instance is None:
        _torch_func_mode_instance = _InspectTorchFuncMode()
        _torch_func_mode_instance.__enter__()

    register_step_callback(_on_step_summary)


def _deactivate_hooks():
    """Remove all global hooks and exit TorchFunctionMode."""
    global _torch_func_mode_instance, _owns_global_hooks

    unregister_step_callback(_on_step_summary)

    if _owns_global_hooks:
        release_global_module_hooks()
        _owns_global_hooks = False

    if _torch_func_mode_instance is not None:
        _torch_func_mode_instance.__exit__(None, None, None)
        _torch_func_mode_instance = None


# ── Per-Model Hook Support (backward compatible) ──
#
# These hooks push/pop module context for specific named submodules,
# mirroring the global hooks but on a per-model basis.


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


def _make_pre_hook():
    def hook(module, args):
        cls_name = type(module).__name__
        push_module_context(cls_name, module)
    return hook


def _make_post_hook():
    def hook(module, args, output):
        pop_module_context()
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
        h1 = module.register_forward_pre_hook(_make_pre_hook())
        h2 = module.register_forward_hook(_make_post_hook())
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
    global _enabled, _match_all, _op_filter, _module_filter, _layer_filter
    global _torch_funcs_enabled, _torch_func_filter, _step_range

    _enabled = False
    _match_all = False
    _op_filter = set()
    _module_filter = set()
    _layer_filter = set()
    _step_range = None
    _torch_funcs_enabled = False
    _torch_func_filter = set()


# ── Environment Initialization ──


def _init_from_env() -> None:
    """Initialize from VLLM_FL_IO_INSPECT* environment variables or YAML config."""
    global _enabled, _match_all, _op_filter, _module_filter, _layer_filter
    global _torch_funcs_enabled, _torch_func_filter, _step_range

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
            tf_default = (True, set()) if (not ops and not modules) else (False, set())
            tf_enabled, tf_filter = io_cfg.get("torch_funcs", tf_default)
            _torch_funcs_enabled = tf_enabled
            _torch_func_filter = tf_filter
            _step_range = io_cfg.get("step_range")
            _layer_filter = set(io_cfg.get("layers", set()))
            _enabled = True

            set_rank_filter(io_cfg.get("ranks"))
            _activate_hooks()

            logger.info(
                f"IO Inspect enabled (YAML): rank={get_rank()}, "
                f"rank_filter={io_cfg.get('ranks') or 'all'}, "
                f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
                f"layers={_layer_filter or 'all'}, "
                f"torch_funcs={_torch_funcs_enabled}, step_range={_step_range}"
            )
            return

    # Priority 2: Environment variables
    env_val = os.environ.get("VLLM_FL_IO_INSPECT", "")
    if env_val == "0":
        # Explicit disable — never auto-enable.
        _reset_state()
        return
    if not env_val:
        # Auto-enable when shared or inspector-specific filter env vars are
        # set — the user clearly intends to use IO inspection.
        _has_filters = any(
            os.environ.get(v, "").strip()
            for v in (
                "VLLM_FL_IO_INSPECT_STEP_RANGE", "VLLM_FL_IO_INSPECT_LAYERS",
                "VLLM_FL_IO_STEP_RANGE", "VLLM_FL_IO_LAYERS",
            )
        )
        if not _has_filters:
            _reset_state()
            return
        env_val = "1"  # enable all ops

    _match_all, _op_filter, _module_filter = _parse_config(env_val)
    _enabled = _match_all or bool(_op_filter) or bool(_module_filter)

    torch_funcs_val = os.environ.get("VLLM_FL_IO_INSPECT_TORCH_FUNCS", "")
    if torch_funcs_val:
        _torch_funcs_enabled, _torch_func_filter = parse_torch_funcs_config(
            torch_funcs_val
        )
    else:
        # Default: enable torch_funcs when inspecting all (match_all mode)
        _torch_funcs_enabled = _match_all
        _torch_func_filter = set()

    # Parse step range (inspector-specific → shared fallback)
    _step_range = parse_step_range_env(
        "VLLM_FL_IO_INSPECT_STEP_RANGE", "VLLM_FL_IO_STEP_RANGE"
    )

    # Parse layer filter (inspector-specific → shared fallback)
    _layer_filter = parse_layers_env(
        "VLLM_FL_IO_INSPECT_LAYERS", "VLLM_FL_IO_LAYERS"
    )

    if _enabled:
        # Parse rank filter (inspector-specific → shared fallback)
        rank_env = os.environ.get("VLLM_FL_IO_INSPECT_RANK", "") or os.environ.get("VLLM_FL_IO_RANK", "")
        if rank_env:
            set_rank_filter(parse_rank_filter(rank_env))

        _activate_hooks()

        logger.info(
            f"IO Inspect enabled: rank={get_rank()}, "
            f"rank_filter={parse_rank_filter(rank_env) if rank_env else 'all'}, "
            f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
            f"layers={_layer_filter or 'all'}, "
            f"torch_funcs={_torch_funcs_enabled}, step_range={_step_range}"
        )


_init_from_env()
