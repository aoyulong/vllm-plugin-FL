# Copyright (c) 2026 BAAI. All rights reserved.

"""
IO Inspector for vllm-plugin-FL dispatch.

Prints operator input/output metadata (tensor shapes, dtypes, statistics)
before and after each operator call.

Only logs at the **op level** (dispatch-managed ops and bare torch functions).
Module-level logging is not produced; instead, module names are appended
to op log entries so users can see which module invoked each op.

When a ``modules`` filter is set (e.g., ``modules={"RMSNormFL"}``), module
context is derived from the Python call stack.  Only ops that run inside
the specified modules are logged.

Supports three interception mechanisms:
1. Dispatch-managed ops: Automatic via OpManager.call() hooks
2. ATen-level ops: Via TorchDispatchMode (default, works in both eager and compile modes)
3. Bare torch functions: Via TorchFunctionMode (opt-in, eager mode only)

When initialized via environment variables (``_init_from_env``), config is parsed
during ``load_model()`` but dispatch mode activation is deferred until
``maybe_activate_hooks()`` is called from ``execute_model()``.  This prevents
the dispatch mode from interfering with vLLM's ``determine_available_memory()``
phase.  The programmatic API (``enable_io_inspect()``) activates hooks immediately.

Configuration (priority order):
    1. Python API: enable_io_inspect(ops=..., step_range=..., layers=...)
    2. YAML config file (via VLLM_FL_CONFIG):
        io_inspect:
          enabled: true
          ops: [rms_norm, silu_and_mul]
          modules: [RMSNormFL, Qwen3Attention]
          layers: [0, 1-3, model.layers.*.self_attn]
          torch_funcs: true
          step_range: "5-15"      # inclusive "start-end"
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
from typing import Any, List, Optional, Set, Tuple, Union

import torch

from .io_common import (
    HAS_TORCH_DISPATCH_MODE,
    HAS_TORCH_FUNC_MODE,
    TorchDispatchMode,
    TorchFunctionMode,
    acquire_torch_func_tags,
    expand_layer_specs,
    format_result,
    format_value,
    get_dispatch_keys,
    get_dispatch_op_name,
    get_dispatch_op_namespace,
    get_module_class_name,
    get_module_context_from_stack,
    get_rank,
    get_step,
    get_torch_func_name,
    layer_path_matches,
    layer_path_matches_from_stack,
    make_guard,
    make_label,
    make_module_tag,
    make_op_tag,
    module_context_matches,
    module_context_matches_from_stack,
    next_exec_order,
    parse_io_config_from_yaml,
    parse_layers_env,
    parse_rank_filter,
    parse_step_range,
    parse_step_range_env,
    parse_torch_funcs_config,
    record_seen,
    register_step_callback,
    release_torch_func_tags,
    should_inspect_dispatch_op,
    should_inspect_torch_func,
    unregister_step_callback,
    warn_if_not_eager,
    _is_eager_mode,
)
from .logger_manager import get_logger

logger = get_logger("vllm_fl.dispatch.io_inspect")

# Sentinel for "not set by caller — inherit from env/YAML"
_UNSET = object()

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
_rank_filter: Optional[Set[int]] = None  # None = all ranks


def _rank_ok() -> bool:
    """Check if the current rank passes this subsystem's rank filter."""
    if _rank_filter is None:
        return True
    return get_rank() in _rank_filter


# Hook handles
_torch_func_mode_instance: Optional[Any] = None
_dispatch_mode_instance: Optional[Any] = None
_hooks_activated: bool = False  # True once dispatch/func modes have been entered

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


def _should_inspect_torch_func(func_name: str, module_ctx=None) -> bool:
    """Check if a torch function should be inspected.

    Layer and module filters are AND gates (same as ``_should_inspect``).

    Args:
        func_name: Torch function name.
        module_ctx: Stack-derived module context (from ``get_module_context_from_stack``).
            If provided, used for layer/module filtering instead of global hook context.
    """
    if _step_range is not None:
        step = get_step()
        if step < _step_range[0] or step >= _step_range[1]:
            return False
    if _layer_filter:
        if module_ctx is not None:
            if not layer_path_matches_from_stack(_layer_filter, module_ctx):
                return False
        elif not layer_path_matches(_layer_filter):
            return False
    if not should_inspect_torch_func(
        func_name, _torch_funcs_enabled, _torch_func_filter,
        _match_all, _op_filter,
    ):
        return False
    # When module filter is active, only match torch funcs inside those modules
    if _module_filter and not _match_all:
        if module_ctx is not None:
            return module_context_matches_from_stack(_module_filter, module_ctx)
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


# ── Formatting ──


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
                  op_tag: str = "",
                  exec_order: int = 0) -> None:
    """Log a consolidated INPUTS + OUTPUTS block for an op call."""
    rank = get_rank()
    step = get_step()
    sep = "-" * 60
    parts = [
        f"\n{sep}",
        f"[IO_INSPECT][rank={rank}][step={step}][exec_order={exec_order}]{module_tag}{op_tag} {label}",
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
                      op_tag: str = "",
                      exec_order: int = 0) -> None:
    """Log operator/module outputs only (fallback when no pairing)."""
    rank = get_rank()
    step = get_step()
    logger.info(f"[IO_INSPECT][rank={rank}][step={step}][exec_order={exec_order}]{module_tag}{op_tag} {label} OUTPUTS:\n{format_result(result)}")


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
    if not _rank_ok():
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


def inspect_after(op_name: str, args: tuple, result: Any) -> None:
    """Log consolidated INPUTS+OUTPUTS block (called from OpManager).

    Retrieves inputs captured by ``inspect_before`` and combines them
    with the result into a single log message.
    """
    if guard_active():
        return
    if not _rank_ok():
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
                          module_tag=module_tag, op_tag=op_tag,
                          exec_order=order)
        else:
            # Fallback: no pairing (shouldn't happen normally)
            module_tag = make_module_tag()
            op_tag = make_op_tag(op_name)
            _log_outputs_only(make_label(f"Op '{op_name}'", args), result,
                              module_tag=module_tag, op_tag=op_tag)
    finally:
        set_guard(False)


def inspect_cleanup(op_name: str) -> None:
    """Pop stale pairing left by inspect_before when the op raises.

    Called from ``_call_with_hooks`` when the actual operator execution
    fails so that the pairing stack stays clean for subsequent calls.
    """
    stack = getattr(_inspect_pairing, "stack", None)
    if stack:
        entries = stack.get(op_name)
        if entries:
            entries.pop()


def enable_io_inspect(
    ops: Union[Optional[Set[str]], object] = _UNSET,
    modules: Union[Optional[Set[str]], object] = _UNSET,
    layers=_UNSET,
    torch_funcs: Optional[bool] = None,
    ranks: Union[Optional[Set[int]], object] = _UNSET,
    step_range: Optional[str] = None,
) -> None:
    """
    Programmatically enable IO inspection.

    All filter dimensions are composable (AND logic): when multiple
    filters are set, an op must satisfy ALL of them to be logged.

    Config sources are merged with priority: API > YAML > env.
    Fields not explicitly set via the API fall through to env vars / YAML.

    By default, uses TorchDispatchMode which intercepts ops at the ATen
    dispatcher level and works in both eager and compiled modes.

    Args:
        ops: Dispatch-managed op names to inspect. ``None`` = all ops.
            Unset = inherit from env var ``VLLM_FL_IO_INSPECT``.
        modules: nn.Module class names to scope inspection to.
            ``None`` = no module scoping (inspect everywhere).
            Unset = inherit from env var ``VLLM_FL_IO_INSPECT``.
        layers: Layer specifications to scope inspection to.  Supports
            integer shorthand (``"0"`` → ``"model.layers.0"``),
            ranges (``"0-3"``), glob patterns (``"model.layers.*.self_attn"``),
            and full paths.  ``None`` = no layer scoping.
            Unset = inherit from env var.
        torch_funcs: Also intercept bare torch functional ops via
            TorchFunctionMode (eager mode only).  Default ``None`` (inherit
            from env var ``VLLM_FL_IO_INSPECT_TORCH_FUNCS``).
            Set ``True``/``False`` to explicitly enable/disable.
        ranks: Set of ranks to inspect on. ``None`` = all ranks.
            Unset = inherit from env var ``VLLM_FL_IO_INSPECT_RANK``.
        step_range: Inclusive step range string.  ``"0-2"`` means
            steps 0, 1, 2.  A bare integer ``"5"`` means step 5 only.
            ``None`` = inherit from env var.
    """
    global _enabled, _match_all, _op_filter, _module_filter, _layer_filter
    global _torch_funcs_enabled, _torch_func_filter, _step_range, _rank_filter

    # ── ops / modules: API > env fallback ──
    if ops is _UNSET and modules is _UNSET:
        # Neither set via API — try env var
        env_val = os.environ.get("VLLM_FL_IO_INSPECT", "").strip()
        if env_val and env_val != "0":
            _match_all, _op_filter, _module_filter = _parse_config(env_val)
        else:
            _match_all = True
            _op_filter = set()
            _module_filter = set()
    elif ops is _UNSET or modules is _UNSET:
        # One set, one not — resolve the unset one from env, then merge
        env_val = os.environ.get("VLLM_FL_IO_INSPECT", "").strip()
        _, env_ops, env_modules = _parse_config(env_val) if (env_val and env_val != "0") else (True, set(), set())
        resolved_ops = env_ops if ops is _UNSET else (set(ops) if ops else set())
        resolved_modules = env_modules if modules is _UNSET else (set(modules) if modules else set())
        if not resolved_ops and not resolved_modules:
            _match_all = True
            _op_filter = set()
            _module_filter = set()
        else:
            _match_all = False
            _op_filter = resolved_ops
            _module_filter = resolved_modules
    else:
        # Both explicitly set via API
        if ops is None and modules is None:
            _match_all = True
            _op_filter = set()
            _module_filter = set()
        else:
            _match_all = False
            _op_filter = set(ops) if ops else set()
            _module_filter = set(modules) if modules else set()

    # ── layers: API > env fallback ──
    if layers is _UNSET or layers is None:
        layers = parse_layers_env(
            "VLLM_FL_IO_INSPECT_LAYERS", "VLLM_FL_IO_LAYERS"
        )
    else:
        if isinstance(layers, str):
            layers = {layers}
        layers = expand_layer_specs(layers)
    _layer_filter = set(layers) if layers else set()

    # ── torch_funcs: API > env fallback ──
    if torch_funcs is not None:
        _torch_funcs_enabled = torch_funcs
        _torch_func_filter = set()
    else:
        tf_val = os.environ.get("VLLM_FL_IO_INSPECT_TORCH_FUNCS", "")
        if tf_val:
            _torch_funcs_enabled, _torch_func_filter = parse_torch_funcs_config(tf_val)
        else:
            _torch_funcs_enabled = False
            _torch_func_filter = set()

    # ── step_range: API > env fallback ──
    if step_range is not None:
        _step_range = parse_step_range(step_range)
    else:
        _step_range = parse_step_range_env(
            "VLLM_FL_IO_INSPECT_STEP_RANGE", "VLLM_FL_IO_STEP_RANGE"
        )
    _enabled = True

    # ── ranks: API > env fallback ──
    if ranks is not _UNSET:
        _rank_filter = ranks
    else:
        rank_env = os.environ.get("VLLM_FL_IO_INSPECT_RANK", "") or os.environ.get("VLLM_FL_IO_RANK", "")
        if rank_env:
            _rank_filter = parse_rank_filter(rank_env)
        else:
            _rank_filter = None
    _activate_hooks()

    # Propagate resolved config to env vars so child processes
    # (e.g. vLLM EngineCore workers) inherit via _init_from_env().
    _resolved_ops = _op_filter if not _match_all else None
    _resolved_modules = _module_filter if not _match_all else None
    _set_env_vars(_resolved_ops, _resolved_modules, _layer_filter,
                  _torch_funcs_enabled, _rank_filter, _step_range)

    logger.info(
        f"IO Inspect enabled: rank={get_rank()}, "
        f"rank_filter={_rank_filter or 'all'}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"layers={_layer_filter or 'all'}, "
        f"torch_funcs={_torch_funcs_enabled}, step_range={_step_range}"
    )
    warn_if_not_eager("IO_INSPECT")


def disable_io_inspect() -> None:
    """Programmatically disable IO inspection and remove all hooks."""
    _reset_state()
    _deactivate_hooks()
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
    """Set VLLM_FL_IO_INSPECT* env vars so child processes inherit the resolved config."""
    if ops is None and modules is None:
        os.environ["VLLM_FL_IO_INSPECT"] = "1"
    else:
        tokens = []
        if ops:
            tokens.extend(sorted(ops))
        if modules:
            tokens.extend(f"module:{m}" for m in sorted(modules))
        os.environ["VLLM_FL_IO_INSPECT"] = ",".join(tokens) if tokens else "1"

    os.environ["VLLM_FL_IO_INSPECT_TORCH_FUNCS"] = "1" if torch_funcs else "0"

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


# ── TorchDispatchMode (default — works in both eager and compile modes) ──


if HAS_TORCH_DISPATCH_MODE:
    class _InspectDispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if not _enabled or guard_active() or not _rank_ok():
                return func(*args, **kwargs)

            op_name = get_dispatch_op_name(func)
            if not should_inspect_dispatch_op(op_name, _match_all, _op_filter):
                return func(*args, **kwargs)

            # Step range check
            if _step_range is not None:
                step = get_step()
                if step < _step_range[0] or step >= _step_range[1]:
                    return func(*args, **kwargs)

            # Derive module context from call stack (works in compile mode)
            module_ctx = get_module_context_from_stack()

            # Layer filter check
            if _layer_filter and not layer_path_matches_from_stack(
                _layer_filter, module_ctx
            ):
                return func(*args, **kwargs)

            # Module filter check
            if _module_filter and not _match_all:
                if not module_context_matches_from_stack(
                    _module_filter, module_ctx
                ):
                    return func(*args, **kwargs)

            ns = get_dispatch_op_namespace(func)
            full_name = f"{ns}.{op_name}"
            dispatch_keys = get_dispatch_keys(func, args, kwargs)
            # Extract module name and layer path from stack context
            mod_name = module_ctx[0][0] if module_ctx else ""
            mod_path = module_ctx[0][1] if module_ctx else ""
            label = make_label(full_name, module_name=mod_name or None,
                               layer_path=mod_path or None,
                               dispatch_keys=dispatch_keys)
            order = next_exec_order()
            op_tag = make_op_tag(full_name)
            record_seen(full_name, module_name=mod_name or None)

            set_guard(True)
            try:
                input_lines = _format_inputs(args, kwargs)
            finally:
                set_guard(False)

            result = func(*args, **kwargs)

            set_guard(True)
            try:
                _log_combined(label, input_lines, result,
                              op_tag=op_tag, exec_order=order)
            finally:
                set_guard(False)

            return result


# ── TorchFunctionMode (opt-in for bare torch ops) ──


if HAS_TORCH_FUNC_MODE:
    class _InspectTorchFuncMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if torch.compiler.is_compiling() or not _enabled or guard_active() or not _rank_ok():
                return func(*args, **kwargs)

            func_name = get_torch_func_name(func)
            module_ctx = get_module_context_from_stack()
            if not _should_inspect_torch_func(func_name, module_ctx):
                return func(*args, **kwargs)

            # Include enclosing module context in label
            full_name = f"torch.{func_name}"
            mod_name = module_ctx[0][0] if module_ctx else ""
            mod_path = module_ctx[0][1] if module_ctx else ""
            label = make_label(full_name, module_name=mod_name or None,
                               layer_path=mod_path or None)
            _mt, op_tag, order = acquire_torch_func_tags(full_name)
            record_seen(full_name, module_name=mod_name or None)
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
                              op_tag=op_tag, exec_order=order)
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


# ── Hook lifecycle ──


def _activate_hooks():
    """Register TorchDispatchMode (default) and optionally TorchFunctionMode."""
    global _torch_func_mode_instance, _dispatch_mode_instance, _hooks_activated

    # TorchDispatchMode is the default — works in both eager and compile modes.
    if HAS_TORCH_DISPATCH_MODE and _dispatch_mode_instance is None:
        _dispatch_mode_instance = _InspectDispatchMode()
        _dispatch_mode_instance.__enter__()

    # TorchFunctionMode is opt-in (torch_funcs=True) and eager-mode only.
    eager = _is_eager_mode()
    if eager and _torch_funcs_enabled and HAS_TORCH_FUNC_MODE and _torch_func_mode_instance is None:
        _torch_func_mode_instance = _InspectTorchFuncMode()
        _torch_func_mode_instance.__enter__()

    register_step_callback(_on_step_summary)
    _hooks_activated = True


def maybe_activate_hooks() -> None:
    """Activate hooks if IO inspect is enabled but hooks are not yet entered.

    Called from model_runner at the start of the first ``execute_model()``
    to defer dispatch mode activation past the memory profiling phase.
    When the programmatic API (``enable_io_inspect()``) is used, hooks are
    activated immediately and this is a no-op.
    """
    if _enabled and not _hooks_activated:
        _activate_hooks()


def _deactivate_hooks():
    """Remove all active hooks."""
    global _torch_func_mode_instance, _dispatch_mode_instance, _hooks_activated

    unregister_step_callback(_on_step_summary)

    if _dispatch_mode_instance is not None:
        _dispatch_mode_instance.__exit__(None, None, None)
        _dispatch_mode_instance = None

    if _torch_func_mode_instance is not None:
        _torch_func_mode_instance.__exit__(None, None, None)
        _torch_func_mode_instance = None

    _hooks_activated = False


# ── State management ──


def _reset_state() -> None:
    """Reset all module-level state to defaults."""
    global _enabled, _match_all, _op_filter, _module_filter, _layer_filter
    global _torch_funcs_enabled, _torch_func_filter, _step_range, _rank_filter

    _enabled = False
    _match_all = False
    _op_filter = set()
    _module_filter = set()
    _layer_filter = set()
    _step_range = None
    _torch_funcs_enabled = False
    _torch_func_filter = set()
    _rank_filter = None


# ── Env-var / YAML initialization ──


def _init_from_env() -> None:
    """Initialize from VLLM_FL_IO_INSPECT* environment variables or YAML config.

    Skipped when the programmatic API (``enable_io_inspect``) has already been
    called — the Python API has the highest priority.
    """
    global _enabled, _match_all, _op_filter, _module_filter, _layer_filter
    global _torch_funcs_enabled, _torch_func_filter, _step_range, _rank_filter

    if _enabled:
        return

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
            tf_default = (False, set())
            tf_enabled, tf_filter = io_cfg.get("torch_funcs", tf_default)
            _torch_funcs_enabled = tf_enabled
            _torch_func_filter = tf_filter
            _step_range = io_cfg.get("step_range")
            _layer_filter = set(io_cfg.get("layers", set()))
            _enabled = True

            _rank_filter = io_cfg.get("ranks")
            # Register step callback but defer dispatch mode activation
            # to maybe_activate_hooks() (called from model_runner before the
            # first execute_model).  This avoids interfering with vLLM's
            # memory profiling which runs between load_model and execute_model.
            register_step_callback(_on_step_summary)

            logger.info(
                f"IO Inspect enabled (YAML): rank={get_rank()}, "
                f"rank_filter={_rank_filter or 'all'}, "
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
        # Default: torch_funcs is opt-in (TorchDispatchMode is the default)
        _torch_funcs_enabled = False
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
            _rank_filter = parse_rank_filter(rank_env)

        # Register step callback but defer dispatch mode activation
        # to maybe_activate_hooks() (called from model_runner before the
        # first execute_model).
        register_step_callback(_on_step_summary)

        logger.info(
            f"IO Inspect enabled: rank={get_rank()}, "
            f"rank_filter={_rank_filter or 'all'}, "
            f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
            f"layers={_layer_filter or 'all'}, "
            f"torch_funcs={_torch_funcs_enabled}, step_range={_step_range}"
        )
