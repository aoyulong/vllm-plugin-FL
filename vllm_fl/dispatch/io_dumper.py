# Copyright (c) 2026 BAAI. All rights reserved.

"""
IO Dumper for vllm-plugin-FL dispatch.

Saves operator input/output tensors to disk as PyTorch .pt files.

Only dumps at the **op level** (dispatch-managed ops and bare torch functions).
Module-level dumping is not produced; instead, module names are appended
to op/torch-func labels so users can see which module invoked each op.

When a ``modules`` filter is set (e.g., ``modules={"RMSNormFL"}``), global
module hooks track which module is executing.  Only ops and torch functions
that run inside the specified modules are dumped.

Supports two interception mechanisms:
1. Dispatch-managed ops: Automatic via OpManager.call() hooks
2. Bare torch functions: Via TorchFunctionMode (enabled by default)

Configuration (priority order):
    1. Python API: enable_io_dump(dump_dir=..., ops=..., step_range=..., layers=...)
    2. YAML config file (via VLLM_FL_CONFIG):
        io_dump:
          dir: /tmp/io_dump
          ops: [rms_norm, silu_and_mul]
          modules: [Linear]
          layers: [0, 1-3, model.layers.*.self_attn]
          max_calls: 100
          step_range: "5-15"      # inclusive "start-end"
          torch_funcs: true
          meta_only: true          # default; set false to dump .pt tensors
    3. Environment variables:
        VLLM_FL_IO_DUMP              - Directory path or "1" for ./io_dump
        VLLM_FL_IO_DUMP_OPS          - Comma-separated op names
        VLLM_FL_IO_DUMP_MODULES      - Comma-separated module class names
        VLLM_FL_IO_DUMP_LAYERS       - Layer specs: integers, ranges, globs, paths
        VLLM_FL_IO_LAYERS            - Shared layer filter (inspector + dumper)
        VLLM_FL_IO_DUMP_MAX_CALLS    - Max calls per op (0 = unlimited)
        VLLM_FL_IO_DUMP_STEP_RANGE   - "start-end" inclusive range (e.g. "0-4")
        VLLM_FL_IO_STEP_RANGE        - Shared step range (inspector + dumper)
        VLLM_FL_IO_DUMP_TORCH_FUNCS  - "1" or "matmul,softmax"
        VLLM_FL_IO_DUMP_META_ONLY    - "0"/"false" to dump .pt tensors (default: meta only)
        VLLM_FL_IO_DUMP_RANK         - Rank filter: "all", "0", "0,2,4"
        VLLM_FL_IO_RANK              - Shared rank filter (fallback if VLLM_FL_IO_DUMP_RANK is unset)

    Note: Setting any filter env var (step_range or layers) auto-enables
    dumping for all ops, even without VLLM_FL_IO_DUMP=1.

    Quick start (env-var only, no Python API needed):
        VLLM_FL_IO_STEP_RANGE=0-2 VLLM_FL_IO_LAYERS=1-3 python script.py

All filter dimensions are composable (AND logic): step_range, layers, modules,
and ops are orthogonal gates.  When multiple filters are set, an op must pass
ALL of them.  Unset filters are pass-through.

File layout:
    dump_dir/rank_0000/step_0005/rms_norm/
        input.json                 # merged metadata for all calls' inputs
        output.json                # merged metadata for all calls' outputs
        call_1_input.pt            # tensor data for call 1 inputs (meta_only=False)
        call_1_output.pt           # tensor data for call 1 outputs
        call_2_input.pt
        call_2_output.pt
    dump_dir/rank_0000/step_0005/torch.matmul/
        input.json
        output.json
        ...

    JSON keys (``call_1``, ``call_2``, ...) match the PT file names for
    easy cross-reference.  When ``meta_only=True`` (default), only JSON
    files are written.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ._io_common import (
    HAS_TORCH_FUNC_MODE,
    TorchFunctionMode,
    acquire_global_module_hooks,
    acquire_torch_func_tags,
    advance_step,
    attach_per_model_hooks,
    expand_layer_specs,
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
    parse_step_range,
    parse_step_range_env,
    parse_torch_funcs_config,
    record_seen,
    register_step_callback,
    release_global_module_hooks,
    release_torch_func_tags,
    remove_per_model_hooks,
    should_inspect_torch_func,
    tensor_stats,
    unregister_step_callback,
)
from .logger_manager import get_logger

logger = get_logger("vllm_fl.dispatch.io_dump")

# ── Module-level state ──

# Independent re-entrancy guard so dumper doesn't block the inspector
guard_active, set_guard = make_guard()

_enabled: bool = False
_dump_dir: str = ""
_match_all: bool = False
_op_filter: Set[str] = set()
_module_filter: Set[str] = set()
_layer_filter: Set[str] = set()
_max_calls: int = 0  # 0 = unlimited
_step_range: Optional[Tuple[int, int]] = None
_meta_only: bool = True

_call_counters: Dict[str, int] = {}
_lock = threading.Lock()

# Per-file locks for _append_to_json to prevent concurrent read-modify-write corruption
_json_file_locks: Dict[str, threading.Lock] = {}
_json_file_locks_guard = threading.Lock()

# Thread-local storage for pairing dump_before → dump_after.
# Each thread stores a dict of op_name → list of (call_num, exec_order, op_dir).
_dump_pairing = threading.local()

_torch_funcs_enabled: bool = False
_torch_func_filter: Set[str] = set()
_rank_filter: Optional[Set[int]] = None  # None = all ranks


def _rank_ok() -> bool:
    """Check if the current rank passes this subsystem's rank filter."""
    if _rank_filter is None:
        return True
    return get_rank() in _rank_filter


# Hook handles
_hook_handles: List[Any] = []
_torch_func_mode_instance: Optional[Any] = None
_owns_global_hooks: bool = False


def _on_step_advance(step: int, seen_modules: Set[str], seen_ops: Set[str]) -> None:
    """Callback to clear per-op call counters and log summary on step advance."""
    with _lock:
        _call_counters.clear()
    if seen_modules or seen_ops:
        rank = get_rank()
        logger.info(
            f"[IO_DUMP][rank={rank}][step={step}] Step summary: "
            f"modules={sorted(seen_modules)}, ops={sorted(seen_ops)}"
        )


# ── Filtering ──


def _check_limits(op_name: str) -> bool:
    """Check step_range and max_calls limits (no filter logic)."""
    if _step_range is not None:
        step = get_step()
        if step < _step_range[0] or step >= _step_range[1]:
            return False
    if _max_calls > 0:
        with _lock:
            if _call_counters.get(op_name, 0) >= _max_calls:
                return False
    return True


def _should_dump(op_name: str, args: tuple) -> bool:
    """Check if this dispatch-managed op call should be dumped.

    Filters are composable AND gates — each active filter must pass:
    - ``_step_range`` / ``_max_calls``: checked via ``_check_limits``
    - ``_layer_filter``: current layer path must match (prefix)
    - ``_module_filter``: must be inside a matching module
    - ``_op_filter``: op name must be in the filter set
    """
    if not _check_limits(op_name):
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


def _should_dump_torch_func(func_name: str) -> bool:
    """Check if a torch function should be dumped.

    Layer and module filters are AND gates (same as ``_should_dump``).
    """
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


def _build_meta(args: tuple, kwargs: dict, *, is_output: bool = False) -> Dict[str, Any]:
    """Build tensor metadata dict for JSON.

    For inputs: keys are ``arg_0``, ``arg_1``, ..., ``kwarg_<name>``.
    For outputs (when *is_output* is True and *args* is a tuple result):
    keys are ``result_0``, ``result_1``, ... or just ``result``.
    """
    meta: Dict[str, Any] = {}
    if is_output:
        # args is actually (result,) and kwargs is empty
        result = args[0] if args else None
        if isinstance(result, tuple):
            for i, v in enumerate(result):
                if isinstance(v, torch.Tensor):
                    meta[f"result_{i}"] = tensor_stats(v)
        elif isinstance(result, torch.Tensor):
            meta["result"] = tensor_stats(result)
    else:
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                meta[f"arg_{i}"] = tensor_stats(arg)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                meta[f"kwarg_{k}"] = tensor_stats(v)
    return meta


def _build_data(args: tuple, kwargs: dict, *, is_output: bool = False) -> Dict[str, Any]:
    """Build tensor data dict for torch.save (PT file).

    Keys match those produced by ``_build_meta`` so users can cross-reference.
    """
    data: Dict[str, Any] = {}
    if is_output:
        result = args[0] if args else None
        if isinstance(result, tuple):
            for i, v in enumerate(result):
                data[f"result_{i}"] = _serialize_value(v)
        else:
            data["result"] = _serialize_value(result)
    else:
        for i, arg in enumerate(args):
            data[f"arg_{i}"] = _serialize_value(arg)
        for k, v in kwargs.items():
            data[f"kwarg_{k}"] = _serialize_value(v)
    return data


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


def _push_pairing(op_name: str, call_num: int, exec_order: int, op_dir: str,
                   label: Optional[str] = None,
                   module_tag: str = "",
                   op_tag: str = "") -> None:
    """Store pairing info in thread-local for dump_after to consume."""
    stack = getattr(_dump_pairing, "stack", None)
    if stack is None:
        _dump_pairing.stack = {}
        stack = _dump_pairing.stack
    stack.setdefault(op_name, []).append((call_num, exec_order, op_dir, label or op_name, module_tag, op_tag))


def _pop_pairing(op_name: str) -> Optional[Tuple[int, int, str, str, str, str]]:
    """Retrieve pairing info stored by the most recent dump_before for this op."""
    stack = getattr(_dump_pairing, "stack", None)
    if stack is None:
        return None
    entries = stack.get(op_name)
    if entries:
        return entries.pop()
    return None


def _get_op_dir(op_name: str) -> str:
    """Build the per-op directory path: dump_dir/rank_XXXX/step_XXXX/op_name."""
    safe_name = _sanitize_path_component(op_name)
    rank = get_rank()
    rank_dir = os.path.join(_dump_dir, f"rank_{rank:04d}")
    step_dir = os.path.join(rank_dir, f"step_{get_step():04d}")
    return os.path.join(step_dir, safe_name)


def _next_call_num(op_name: str) -> int:
    """Increment and return the per-op call counter (thread-safe)."""
    with _lock:
        count = _call_counters.get(op_name, 0) + 1
        _call_counters[op_name] = count
        return count


def _get_json_lock(json_path: str) -> threading.Lock:
    """Get or create a per-file lock for JSON read-modify-write."""
    with _json_file_locks_guard:
        lock = _json_file_locks.get(json_path)
        if lock is None:
            lock = threading.Lock()
            _json_file_locks[json_path] = lock
        return lock


def _append_to_json(json_path: str, key: str, value: dict) -> None:
    """Add a key to a JSON file (read-modify-write). Creates file if needed.

    Uses a per-file lock to prevent concurrent updates from corrupting the file.
    If the existing file is corrupted or unreadable, it is treated as empty so
    that subsequent dumps are not permanently broken.
    """
    lock = _get_json_lock(json_path)
    with lock:
        data: Dict[str, Any] = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    f"Corrupt or unreadable JSON '{json_path}': {exc}. "
                    "Starting fresh."
                )
                data = {}
        data[key] = value
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, json_path)


# ── Dump helpers ──


def _dump_input(op_name: str, args: tuple, kwargs: dict,
                exec_order: Optional[int] = None,
                label: Optional[str] = None,
                module_tag: str = "",
                op_tag: str = "") -> None:
    """Save operator inputs to disk.

    Appends an entry to the merged ``input.json`` (keyed by call number)
    and optionally writes a per-call ``call_<N>_input.pt``.

    Args:
        op_name: Raw op name, used as pairing key.
        exec_order: Pre-allocated execution order.  *None* → allocate internally.
        label: Display label (may include module info) for metadata.
        module_tag: Module counter tag string, e.g. ``[module=0,1]``.
        op_tag: Op counter tag string, e.g. ``[op=3,2]``.
    """
    display = label or op_name
    try:
        order = exec_order if exec_order is not None else next_exec_order()
        call_num = _next_call_num(op_name)
        op_dir = _get_op_dir(op_name)
        os.makedirs(op_dir, exist_ok=True)

        _push_pairing(op_name, call_num, order, op_dir, label=display,
                      module_tag=module_tag, op_tag=op_tag)

        call_key = f"call_{call_num}"
        meta_entry = {
            "op_name": display,
            "exec_order": order,
            "call_num": call_num,
            "step": get_step(),
            "rank": get_rank(),
            "module_tag": module_tag,
            "op_tag": op_tag,
            "tensors": _build_meta(args, kwargs),
        }
        _append_to_json(os.path.join(op_dir, "input.json"), call_key, meta_entry)

        if not _meta_only:
            pt_path = os.path.join(op_dir, f"call_{call_num}_input.pt")
            torch.save(_build_data(args, kwargs), pt_path)

        logger.debug(f"Dumped input: {op_dir} [{call_key}]")
    except Exception as e:
        logger.warning(f"Failed to dump input for '{display}': {e}")


def _dump_output(op_name: str, result: Any) -> None:
    """Save operator outputs to disk.

    Appends an entry to the merged ``output.json`` (keyed by call number)
    and optionally writes a per-call ``call_<N>_output.pt``.
    """
    try:
        pairing = _pop_pairing(op_name)
        if pairing is None:
            logger.warning(f"No pairing info for dump output '{op_name}', skipping")
            return
        call_num, order, op_dir, label, module_tag, op_tag = pairing

        call_key = f"call_{call_num}"
        meta_entry = {
            "op_name": label,
            "exec_order": order,
            "call_num": call_num,
            "step": get_step(),
            "rank": get_rank(),
            "module_tag": module_tag,
            "op_tag": op_tag,
            "tensors": _build_meta((result,), {}, is_output=True),
        }
        _append_to_json(os.path.join(op_dir, "output.json"), call_key, meta_entry)

        if not _meta_only:
            pt_path = os.path.join(op_dir, f"call_{call_num}_output.pt")
            torch.save(_build_data((result,), {}, is_output=True), pt_path)

        logger.debug(f"Dumped output: {op_dir} [{call_key}]")
    except Exception as e:
        logger.warning(f"Failed to dump output for '{op_name}': {e}")


# ── Public API ──


def is_dump_enabled() -> bool:
    """Check if IO dumping is enabled (fast path)."""
    return _enabled


def dump_before(op_name: str, args: tuple, kwargs: dict,
                exec_order: Optional[int] = None,
                module_tag: Optional[str] = None,
                op_tag: Optional[str] = None) -> None:
    """Dump operator inputs (called from OpManager).

    Args:
        exec_order: Pre-allocated execution order shared with the inspector.
        module_tag: Pre-computed module counter tag (e.g. ``[module=0,1]``).
            When *None*, generated internally.
        op_tag: Pre-computed op counter tag (e.g. ``[op=3,2]``).
            When *None*, generated internally.
    """
    if guard_active():
        return
    if not _rank_ok():
        return
    if not _should_dump(op_name, args):
        return
    label = make_label(op_name, args)
    _module_tag = module_tag if module_tag is not None else make_module_tag()
    _op_tag = op_tag if op_tag is not None else make_op_tag(op_name)
    record_seen(op_name, args)
    set_guard(True)
    try:
        _dump_input(op_name, args, kwargs, exec_order=exec_order, label=label,
                    module_tag=_module_tag, op_tag=_op_tag)
    finally:
        set_guard(False)


def dump_after(op_name: str, args: tuple, result: Any) -> None:
    """Dump operator outputs (called from OpManager)."""
    if guard_active():
        return
    if not _rank_ok():
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
    """Increment step counter and reset per-op call counters.

    Uses the shared step counter from _io_common so the inspector
    and dumper stay in sync.  The ``_on_step_advance`` callback
    registered by ``_activate_hooks`` clears per-op call counters.
    """
    return advance_step()


def enable_io_dump(
    dump_dir: str = "",
    ops: Optional[Set[str]] = None,
    modules: Optional[Set[str]] = None,
    layers: Optional[Set[str]] = None,
    max_calls: int = 0,
    step_range: Optional[str] = None,
    torch_funcs: bool = True,
    ranks: Optional[Set[int]] = None,
    meta_only: bool = True,
) -> None:
    """
    Programmatically enable IO dumping.

    All filter dimensions are composable (AND logic): when multiple
    filters are set, an op must satisfy ALL of them to be dumped.

    Args:
        dump_dir: Directory to save dump files. Defaults to ``./io_dump``
            under the current working directory.
        ops: Dispatch-managed op names to dump. None = all.
        modules: nn.Module class names to scope dumping to.
            When set, only ops/torch_funcs executing inside these modules
            are dumped.  None = no module scoping (dump everywhere).
        layers: Layer specifications to scope dumping to.  Supports
            integer shorthand (``"0"`` → ``"model.layers.0"``),
            ranges (``"0-3"``), glob patterns (``"model.layers.*.self_attn"``),
            and full paths.  None = no layer scoping.
        max_calls: Max calls per op to dump (0 = unlimited).
        step_range: Inclusive step range string.  ``"0-2"`` means
            steps 0, 1, 2.  A bare integer ``"5"`` means step 5 only.
            None = all steps.
        torch_funcs: Intercept bare torch functional ops. Default True
            (all intercepted). Set False to skip torch functions.
        ranks: Set of ranks to dump on. None = all ranks.
        meta_only: If True (default), only write ``.json`` metadata files,
            skip ``.pt`` tensor data files.  Set to False to dump full
            tensor data (may use significant disk space).
    """
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter, _layer_filter
    global _max_calls, _step_range, _torch_funcs_enabled, _torch_func_filter, _meta_only
    global _rank_filter

    _dump_dir = dump_dir if dump_dir else os.path.join(os.getcwd(), "io_dump")
    os.makedirs(_dump_dir, exist_ok=True)

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
            "VLLM_FL_IO_DUMP_LAYERS", "VLLM_FL_IO_LAYERS"
        )
    else:
        layers = expand_layer_specs(layers)
    _layer_filter = set(layers) if layers else set()

    _max_calls = max_calls
    # Parse step_range: accepts string "0-2" or None.
    # parse_step_range handles inclusive→half-open conversion.
    if step_range is not None:
        _step_range = parse_step_range(step_range)
    else:
        _step_range = parse_step_range_env(
            "VLLM_FL_IO_DUMP_STEP_RANGE", "VLLM_FL_IO_STEP_RANGE"
        )
    _torch_funcs_enabled = torch_funcs
    _torch_func_filter = set()
    _meta_only = meta_only
    _enabled = True

    _rank_filter = ranks
    _activate_hooks()

    # Propagate to env vars so child processes (e.g. vLLM EngineCore workers)
    # pick up the config via _init_from_env() on module import.
    _set_env_vars(_dump_dir, ops, modules, _layer_filter, max_calls, _step_range,
                  torch_funcs, ranks, _meta_only)

    logger.info(
        f"IO Dump enabled: rank={get_rank()}, "
        f"rank_filter={ranks or 'all'}, dir={_dump_dir}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"layers={_layer_filter or 'all'}, "
        f"max_calls={_max_calls}, step_range={_step_range}, "
        f"torch_funcs={_torch_funcs_enabled}, meta_only={_meta_only}"
    )


def disable_io_dump() -> None:
    """Programmatically disable IO dumping and remove all hooks."""
    _reset_state()
    _deactivate_hooks()
    remove_dump_hooks()
    _clear_env_vars()


# ── Env-var propagation for child processes ──


def _set_env_vars(
    dump_dir: str,
    ops: Optional[Set[str]],
    modules: Optional[Set[str]],
    layers: Set[str],
    max_calls: int,
    step_range: Optional[Tuple[int, int]],
    torch_funcs: bool,
    ranks: Optional[Set[int]],
    meta_only: bool,
) -> None:
    """Set VLLM_FL_IO_DUMP* env vars so child processes inherit the config."""
    os.environ["VLLM_FL_IO_DUMP"] = dump_dir

    if ops:
        os.environ["VLLM_FL_IO_DUMP_OPS"] = ",".join(sorted(ops))
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_OPS", None)

    if modules:
        os.environ["VLLM_FL_IO_DUMP_MODULES"] = ",".join(sorted(modules))
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_MODULES", None)

    if layers:
        os.environ["VLLM_FL_IO_DUMP_LAYERS"] = ",".join(sorted(layers))
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_LAYERS", None)

    if max_calls > 0:
        os.environ["VLLM_FL_IO_DUMP_MAX_CALLS"] = str(max_calls)
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_MAX_CALLS", None)

    if step_range is not None:
        os.environ["VLLM_FL_IO_DUMP_STEP_RANGE"] = f"{step_range[0]}-{step_range[1] - 1}"
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_STEP_RANGE", None)

    if torch_funcs:
        os.environ["VLLM_FL_IO_DUMP_TORCH_FUNCS"] = "1"
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_TORCH_FUNCS", None)

    if ranks is not None:
        os.environ["VLLM_FL_IO_DUMP_RANK"] = ",".join(str(r) for r in sorted(ranks))
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_RANK", None)

    if not meta_only:
        os.environ["VLLM_FL_IO_DUMP_META_ONLY"] = "0"
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_META_ONLY", None)


def _clear_env_vars() -> None:
    """Remove VLLM_FL_IO_DUMP* env vars."""
    for key in [
        "VLLM_FL_IO_DUMP", "VLLM_FL_IO_DUMP_OPS", "VLLM_FL_IO_DUMP_MODULES",
        "VLLM_FL_IO_DUMP_LAYERS", "VLLM_FL_IO_DUMP_MAX_CALLS",
        "VLLM_FL_IO_DUMP_STEP_RANGE", "VLLM_FL_IO_DUMP_TORCH_FUNCS",
        "VLLM_FL_IO_DUMP_META_ONLY", "VLLM_FL_IO_DUMP_RANK",
    ]:
        os.environ.pop(key, None)


# ── TorchFunctionMode (opt-in for bare torch ops) ──


if HAS_TORCH_FUNC_MODE:
    class _DumpTorchFuncMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if torch.compiler.is_compiling() or not _enabled or guard_active() or not _rank_ok():
                return func(*args, **kwargs)

            func_name = get_torch_func_name(func)
            if not _should_dump_torch_func(func_name):
                return func(*args, **kwargs)

            # Use raw name for pairing/limits, annotated label for metadata
            raw_name = f"torch.{func_name}"
            if not _check_limits(raw_name):
                return func(*args, **kwargs)

            label = make_label(raw_name)
            module_tag, op_tag, order = acquire_torch_func_tags(raw_name)
            record_seen(raw_name)

            set_guard(True)
            try:
                _dump_input(raw_name, args, kwargs, exec_order=order,
                            label=label, module_tag=module_tag, op_tag=op_tag)
            finally:
                set_guard(False)

            try:
                result = func(*args, **kwargs)
            except Exception:
                # Clean up stale pairing pushed by _dump_input
                _pop_pairing(raw_name)
                release_torch_func_tags()
                raise

            set_guard(True)
            try:
                _dump_output(raw_name, result)
            finally:
                set_guard(False)

            release_torch_func_tags()
            return result


# ── Hook Lifecycle ──


def _activate_hooks():
    """Register global module hooks and/or TorchFunctionMode as needed."""
    global _torch_func_mode_instance, _owns_global_hooks

    # Register global module hooks to track module context.
    # In match-all mode, this provides module annotations on dump file
    # labels (e.g. "rms_norm (module=RMSNormFL)").
    # When module filter or layer filter is set, this also enables
    # scope-based filtering (layer filter needs paths from the hook).
    needs_module_hooks = _match_all or bool(_module_filter) or bool(_layer_filter)
    if needs_module_hooks and not _owns_global_hooks:
        acquire_global_module_hooks()
        _owns_global_hooks = True

    if _torch_funcs_enabled and HAS_TORCH_FUNC_MODE and _torch_func_mode_instance is None:
        _torch_func_mode_instance = _DumpTorchFuncMode()
        _torch_func_mode_instance.__enter__()

    # Register callback to clear call counters on each step advance
    register_step_callback(_on_step_advance)


def _deactivate_hooks():
    """Remove all global hooks and exit TorchFunctionMode."""
    global _torch_func_mode_instance, _owns_global_hooks

    if _owns_global_hooks:
        release_global_module_hooks()
        _owns_global_hooks = False

    unregister_step_callback(_on_step_advance)

    if _torch_func_mode_instance is not None:
        _torch_func_mode_instance.__exit__(None, None, None)
        _torch_func_mode_instance = None


# ── Per-Model Hook Support ──


def attach_dump_hooks(model: torch.nn.Module) -> int:
    """Attach IO dump hooks to specific submodules of a model.

    When using enable_io_dump() with a modules filter, global hooks
    are registered automatically. Use this for targeted per-model hooks.
    """
    handles = attach_per_model_hooks(
        model,
        enabled=_enabled,
        match_all=_match_all,
        op_filter=_op_filter,
        module_filter=_module_filter,
    )
    count = len(handles) // 2
    _hook_handles.extend(handles)
    if count > 0:
        logger.info(f"[IO_DUMP] Attached dump hooks to {count} modules")
    return count


def remove_dump_hooks() -> None:
    """Remove all per-model IO dump hooks."""
    remove_per_model_hooks(_hook_handles)


# ── Environment Initialization ──


def _reset_state() -> None:
    """Reset all module-level state to defaults."""
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter, _layer_filter
    global _max_calls, _step_range, _meta_only
    global _torch_funcs_enabled, _torch_func_filter, _rank_filter

    _enabled = False
    _dump_dir = ""
    _match_all = False
    _op_filter = set()
    _module_filter = set()
    _layer_filter = set()
    _max_calls = 0
    _step_range = None
    _meta_only = True
    _torch_funcs_enabled = False
    _torch_func_filter = set()
    _rank_filter = None
    with _lock:
        _call_counters.clear()


def _init_from_env() -> None:
    """Initialize from VLLM_FL_IO_DUMP* environment variables or YAML config."""
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter, _layer_filter
    global _max_calls, _step_range, _torch_funcs_enabled, _torch_func_filter, _meta_only
    global _rank_filter

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
            _dump_dir = io_cfg["dir"] if io_cfg.get("dir") else os.path.join(os.getcwd(), "io_dump")
            try:
                os.makedirs(_dump_dir, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    f"Cannot create dump directory '{_dump_dir}': {exc}. "
                    "IO dumping disabled."
                )
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

            _max_calls = io_cfg.get("max_calls", 0)
            _step_range = io_cfg.get("step_range")
            _layer_filter = set(io_cfg.get("layers", set()))
            _meta_only = io_cfg.get("meta_only", True)

            tf_default = (True, set()) if (not ops and not modules) else (False, set())
            tf_enabled, tf_filter = io_cfg.get("torch_funcs", tf_default)
            _torch_funcs_enabled = tf_enabled
            _torch_func_filter = tf_filter

            _enabled = True

            _rank_filter = io_cfg.get("ranks")
            _activate_hooks()

            logger.info(
                f"IO Dump enabled (YAML): rank={get_rank()}, "
                f"rank_filter={_rank_filter or 'all'}, dir={_dump_dir}, "
                f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
                f"layers={_layer_filter or 'all'}, "
                f"max_calls={_max_calls}, step_range={_step_range}, "
                f"torch_funcs={_torch_funcs_enabled}, meta_only={_meta_only}"
            )
            return

    # Priority 2: Environment variables
    dump_dir = os.environ.get("VLLM_FL_IO_DUMP", "").strip()
    if dump_dir == "0":
        # Explicit disable — never auto-enable.
        _reset_state()
        return
    if not dump_dir:
        # Auto-enable when shared or dumper-specific filter env vars are
        # set — the user clearly intends to use IO dumping.
        _has_filters = any(
            os.environ.get(v, "").strip()
            for v in (
                "VLLM_FL_IO_DUMP_STEP_RANGE", "VLLM_FL_IO_DUMP_LAYERS",
                "VLLM_FL_IO_STEP_RANGE", "VLLM_FL_IO_LAYERS",
            )
        )
        if not _has_filters:
            _reset_state()
            return
        dump_dir = "1"  # default directory

    # "1" means enable with default directory
    if dump_dir == "1":
        dump_dir = os.path.join(os.getcwd(), "io_dump")

    _dump_dir = dump_dir
    try:
        os.makedirs(_dump_dir, exist_ok=True)
    except OSError as exc:
        logger.warning(
            f"Cannot create dump directory '{_dump_dir}': {exc}. "
            "IO dumping disabled."
        )
        _reset_state()
        return

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

    # Parse step range (dumper-specific → shared fallback)
    _step_range = parse_step_range_env(
        "VLLM_FL_IO_DUMP_STEP_RANGE", "VLLM_FL_IO_STEP_RANGE"
    )

    # Parse layer filter (dumper-specific → shared fallback)
    _layer_filter = parse_layers_env(
        "VLLM_FL_IO_DUMP_LAYERS", "VLLM_FL_IO_LAYERS"
    )

    # Parse meta_only flag (default True; set "0" or "false" to disable)
    meta_only_str = os.environ.get("VLLM_FL_IO_DUMP_META_ONLY", "").strip().lower()
    if meta_only_str in ("0", "false"):
        _meta_only = False
    else:
        _meta_only = True  # default

    torch_funcs_val = os.environ.get("VLLM_FL_IO_DUMP_TORCH_FUNCS", "")
    if torch_funcs_val:
        _torch_funcs_enabled, _torch_func_filter = parse_torch_funcs_config(
            torch_funcs_val
        )
    else:
        # Default: enable torch_funcs when dumping all (match_all mode)
        _torch_funcs_enabled = _match_all
        _torch_func_filter = set()

    _enabled = True

    # Parse rank filter (dumper-specific → shared fallback)
    rank_env = os.environ.get("VLLM_FL_IO_DUMP_RANK", "") or os.environ.get("VLLM_FL_IO_RANK", "")
    if rank_env:
        _rank_filter = parse_rank_filter(rank_env)

    _activate_hooks()

    logger.info(
        f"IO Dump enabled: rank={get_rank()}, "
        f"rank_filter={_rank_filter or 'all'}, "
        f"dir={_dump_dir}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"layers={_layer_filter or 'all'}, "
        f"max_calls={_max_calls}, step_range={_step_range}, "
        f"torch_funcs={_torch_funcs_enabled}, meta_only={_meta_only}"
    )


_init_from_env()
