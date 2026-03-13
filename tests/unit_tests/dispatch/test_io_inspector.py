# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tests for IO Inspector module.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
import torch

from vllm_fl.dispatch import io_inspector
from vllm_fl.dispatch._io_common import (
    HAS_GLOBAL_MODULE_HOOKS,
    HAS_TORCH_FUNC_MODE,
    advance_step,
    format_result,
    format_value,
    get_module_class_name,
    get_rank,
    layer_path_matches,
    next_exec_order,
    parse_io_config_from_yaml,
    parse_rank_filter,
    parse_torch_funcs_config,
    pop_module_context,
    push_module_context,
    register_module_paths,
    reset_exec_order,
    reset_rank,
    reset_step,
)
from vllm_fl.dispatch.io_inspector import (
    _parse_config,
    _should_inspect,
    _should_inspect_torch_func,
    attach_io_hooks,
    disable_io_inspect,
    enable_io_inspect,
    inspect_after,
    inspect_before,
    is_inspect_enabled,
    remove_io_hooks,
)


class TestParseConfig:
    """Test _parse_config parsing logic."""

    def test_empty_string(self):
        inspect_all, ops, modules = _parse_config("")
        assert not inspect_all
        assert ops == set()
        assert modules == set()

    def test_zero_disables(self):
        inspect_all, ops, modules = _parse_config("0")
        assert not inspect_all
        assert ops == set()
        assert modules == set()

    def test_one_enables_all(self):
        inspect_all, ops, modules = _parse_config("1")
        assert inspect_all
        assert ops == set()
        assert modules == set()

    def test_op_names(self):
        inspect_all, ops, modules = _parse_config("silu_and_mul,rms_norm")
        assert not inspect_all
        assert ops == {"silu_and_mul", "rms_norm"}
        assert modules == set()

    def test_module_names(self):
        inspect_all, ops, modules = _parse_config(
            "module:RMSNormFL,module:SiluAndMulFL"
        )
        assert not inspect_all
        assert ops == set()
        assert modules == {"RMSNormFL", "SiluAndMulFL"}

    def test_mixed(self):
        inspect_all, ops, modules = _parse_config("rms_norm,module:RotaryEmbeddingFL")
        assert not inspect_all
        assert ops == {"rms_norm"}
        assert modules == {"RotaryEmbeddingFL"}

    def test_whitespace_handling(self):
        inspect_all, ops, modules = _parse_config(" silu_and_mul , module:RMSNormFL ")
        assert ops == {"silu_and_mul"}
        assert modules == {"RMSNormFL"}


class TestFormatValue:
    """Test format_value formatting."""

    def test_tensor(self):
        t = torch.zeros(4, 512, dtype=torch.float16)
        result = format_value(t)
        assert "shape=[4, 512]" in result
        assert "float16" in result
        assert "min=" in result
        assert "max=" in result

    def test_none(self):
        assert format_value(None) == "None"

    def test_int(self):
        assert format_value(42) == "42"

    def test_float(self):
        assert format_value(1.5) == "1.5"

    def test_bool(self):
        assert format_value(True) == "True"

    def test_small_tuple(self):
        result = format_value((1, 2, 3))
        assert "tuple" in result

    def test_large_tuple(self):
        result = format_value((1, 2, 3, 4, 5))
        assert "len=5" in result

    def test_module(self):
        m = torch.nn.Linear(10, 10)
        result = format_value(m)
        assert "Linear" in result


class TestFormatResult:
    """Test format_result formatting."""

    def test_single_tensor(self):
        t = torch.zeros(2, 3)
        result = format_result(t)
        assert "result:" in result
        assert "shape=[2, 3]" in result

    def test_tuple_result(self):
        t1 = torch.zeros(2, 3)
        t2 = torch.ones(4, 5)
        result = format_result((t1, t2))
        assert "result[0]:" in result
        assert "result[1]:" in result


class TestGetModuleName:
    """Test get_module_class_name extraction."""

    def test_with_module(self):
        m = torch.nn.Linear(10, 10)
        assert get_module_class_name((m, torch.zeros(2))) == "Linear"

    def test_without_module(self):
        assert get_module_class_name((torch.zeros(2),)) is None

    def test_empty_args(self):
        assert get_module_class_name(()) is None


class TestShouldInspect:
    """Test _should_inspect filtering logic."""

    def setup_method(self):
        disable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    def test_disabled_returns_false(self):
        assert not _should_inspect("rms_norm", ())

    def test_inspect_all(self):
        enable_io_inspect()
        assert _should_inspect("rms_norm", ())
        assert _should_inspect("silu_and_mul", ())

    def test_op_filter(self):
        enable_io_inspect(ops={"rms_norm"})
        assert _should_inspect("rms_norm", ())
        assert not _should_inspect("silu_and_mul", ())

    def test_module_filter(self):
        enable_io_inspect(modules={"Linear"})
        m = torch.nn.Linear(10, 10)
        assert _should_inspect("any_op", (m,))
        assert not _should_inspect("any_op", (torch.zeros(2),))


class TestProgrammaticAPI:
    """Test enable/disable programmatic API."""

    def setup_method(self):
        disable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    def test_enable_all(self):
        assert not is_inspect_enabled()
        enable_io_inspect()
        assert is_inspect_enabled()

    def test_disable(self):
        enable_io_inspect()
        assert is_inspect_enabled()
        disable_io_inspect()
        assert not is_inspect_enabled()

    def test_enable_with_ops(self):
        enable_io_inspect(ops={"rms_norm"})
        assert is_inspect_enabled()
        assert _should_inspect("rms_norm", ())
        assert not _should_inspect("silu_and_mul", ())

    def test_enable_with_modules(self):
        enable_io_inspect(modules={"Linear"})
        assert is_inspect_enabled()


class TestEnvVarInit:
    """Test initialization from environment variable."""

    def teardown_method(self):
        disable_io_inspect()

    @patch.dict(os.environ, {"VLLM_FL_IO_INSPECT": "1"})
    def test_env_all(self):
        io_inspector._init_from_env()
        assert is_inspect_enabled()
        assert io_inspector._match_all

    @patch.dict(os.environ, {"VLLM_FL_IO_INSPECT": "rms_norm,silu_and_mul"})
    def test_env_ops(self):
        io_inspector._init_from_env()
        assert is_inspect_enabled()
        assert io_inspector._op_filter == {"rms_norm", "silu_and_mul"}

    @patch.dict(os.environ, {"VLLM_FL_IO_INSPECT": "module:RMSNormFL"})
    def test_env_modules(self):
        io_inspector._init_from_env()
        assert is_inspect_enabled()
        assert io_inspector._module_filter == {"RMSNormFL"}

    @patch.dict(os.environ, {"VLLM_FL_IO_INSPECT": "0"})
    def test_env_disabled(self):
        io_inspector._init_from_env()
        assert not is_inspect_enabled()

    @patch.dict(os.environ, {}, clear=False)
    def test_env_unset(self):
        os.environ.pop("VLLM_FL_IO_INSPECT", None)
        io_inspector._init_from_env()
        assert not is_inspect_enabled()


class TestInspectBeforeAfter:
    """Test inspect_before and inspect_after don't crash."""

    def setup_method(self):
        enable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    def test_inspect_before_with_tensors(self):
        t = torch.zeros(2, 3, dtype=torch.float32)
        # Should not raise (stores inputs, no log yet)
        inspect_before("test_op", (t,), {})
        # Complete the pairing
        inspect_after("test_op", (t,), t)

    def test_inspect_before_with_module(self):
        m = torch.nn.Linear(10, 10)
        t = torch.zeros(2, 10)
        inspect_before("test_op", (m, t), {"epsilon": 1e-6})
        inspect_after("test_op", (m, t), t)

    def test_inspect_after_with_tensor(self):
        t = torch.zeros(2, 3)
        # Without inspect_before, falls back to outputs-only log
        inspect_after("test_op", (), t)

    def test_inspect_after_with_tuple(self):
        t1 = torch.zeros(2, 3)
        t2 = torch.ones(4, 5)
        inspect_after("test_op", (), (t1, t2))

    def test_inspect_before_with_none_args(self):
        inspect_before("test_op", (None,), {})
        inspect_after("test_op", (None,), None)

    def test_inspect_skips_when_filtered(self):
        disable_io_inspect()
        enable_io_inspect(ops={"other_op"})
        # Should be a no-op (not matching filter)
        inspect_before("test_op", (torch.zeros(2),), {})
        inspect_after("test_op", (torch.zeros(2),), torch.zeros(2))


class TestForwardHooks:
    """Test nn.Module forward hook attachment."""

    def setup_method(self):
        disable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    def test_attach_hooks_all(self):
        enable_io_inspect()
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        count = attach_io_hooks(model)
        # Sequential + 3 children + submodules = at least 4
        assert count >= 4
        remove_io_hooks()

    def test_attach_hooks_by_module_filter(self):
        enable_io_inspect(modules={"Linear"})
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        count = attach_io_hooks(model)
        assert count == 2  # Only the two Linear modules
        remove_io_hooks()

    def test_hooks_fire_on_forward(self, caplog):
        import logging

        enable_io_inspect(modules={"Linear"})
        model = torch.nn.Linear(4, 3)
        attach_io_hooks(model)

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            x = torch.randn(2, 4)
            model(x)

        # Check that both input and output were logged
        assert any("INPUTS" in r.message for r in caplog.records)
        assert any("OUTPUTS" in r.message for r in caplog.records)
        remove_io_hooks()

    def test_remove_hooks(self):
        enable_io_inspect()
        model = torch.nn.Linear(4, 3)
        attach_io_hooks(model)
        remove_io_hooks()
        # After removal, hooks should not fire (no crash at least)
        x = torch.randn(2, 4)
        model(x)

    def test_attach_disabled_returns_zero(self):
        # When disabled, attach should be a no-op
        model = torch.nn.Linear(4, 3)
        count = attach_io_hooks(model)
        assert count == 0

    def test_disable_removes_hooks(self):
        enable_io_inspect()
        model = torch.nn.Linear(4, 3)
        attach_io_hooks(model)
        disable_io_inspect()
        # Hooks should be removed
        from vllm_fl.dispatch.io_inspector import _hook_handles

        assert len(_hook_handles) == 0


class TestParseTorchFuncsConfig:
    """Test parse_torch_funcs_config parsing logic."""

    def test_empty_string(self):
        enabled, funcs = parse_torch_funcs_config("")
        assert not enabled
        assert funcs == set()

    def test_zero_disables(self):
        enabled, funcs = parse_torch_funcs_config("0")
        assert not enabled
        assert funcs == set()

    def test_one_enables_all(self):
        enabled, funcs = parse_torch_funcs_config("1")
        assert enabled
        assert funcs == set()

    def test_specific_funcs(self):
        enabled, funcs = parse_torch_funcs_config("matmul,softmax")
        assert enabled
        assert funcs == {"matmul", "softmax"}

    def test_whitespace(self):
        enabled, funcs = parse_torch_funcs_config(" matmul , linear ")
        assert enabled
        assert funcs == {"matmul", "linear"}


class TestShouldInspectTorchFunc:
    """Test _should_inspect_torch_func filtering logic."""

    def setup_method(self):
        disable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    def test_disabled_returns_false(self):
        assert not _should_inspect_torch_func("matmul")

    def test_enabled_all(self):
        enable_io_inspect(torch_funcs=True)
        assert _should_inspect_torch_func("matmul")
        assert _should_inspect_torch_func("softmax")

    def test_skips_dunder(self):
        enable_io_inspect(torch_funcs=True)
        assert not _should_inspect_torch_func("__add__")
        assert not _should_inspect_torch_func("_internal_op")

    def test_skips_trivial_ops(self):
        enable_io_inspect(torch_funcs=True)
        assert not _should_inspect_torch_func("size")
        assert not _should_inspect_torch_func("dim")
        assert not _should_inspect_torch_func("is_contiguous")

    def test_op_filter_match(self):
        enable_io_inspect(ops={"matmul"}, torch_funcs=True)
        assert _should_inspect_torch_func("matmul")
        assert not _should_inspect_torch_func("softmax")


class TestGlobalModuleHooks:
    """Test automatic global module hook registration."""

    def setup_method(self):
        disable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    @pytest.mark.skipif(
        not HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_auto_registered(self):
        enable_io_inspect(modules={"Linear"})
        from vllm_fl.dispatch.io_inspector import _owns_global_hooks

        assert _owns_global_hooks is True

    @pytest.mark.skipif(
        not HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_fire_without_attach(self, caplog):
        """Global hooks should fire without calling attach_io_hooks()."""
        import logging

        enable_io_inspect(modules={"Linear"})
        model = torch.nn.Linear(4, 3)

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            x = torch.randn(2, 4)
            model(x)

        assert any("INPUTS" in r.message for r in caplog.records)
        assert any("OUTPUTS" in r.message for r in caplog.records)

    @pytest.mark.skipif(
        not HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_removed_on_disable(self):
        enable_io_inspect(modules={"Linear"})
        from vllm_fl.dispatch.io_inspector import _owns_global_hooks

        assert _owns_global_hooks is True
        disable_io_inspect()
        from vllm_fl.dispatch import io_inspector

        assert io_inspector._owns_global_hooks is False

    @pytest.mark.skipif(
        not HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_registered_in_match_all(self):
        """In match-all mode, global module hooks ARE registered
        to provide module context annotations on op/torch_func entries."""
        enable_io_inspect()
        from vllm_fl.dispatch.io_inspector import _owns_global_hooks

        assert _owns_global_hooks is True

    @pytest.mark.skipif(
        not HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_filter_by_module(self, caplog):
        """Only filtered modules should be logged."""
        import logging

        enable_io_inspect(modules={"Linear"})
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
        )

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            x = torch.randn(2, 4)
            model(x)

        # Linear should appear, ReLU should not
        messages = " ".join(r.message for r in caplog.records)
        assert "Linear" in messages
        # ReLU should not appear in module-specific logs
        assert "ReLU" not in messages


class TestTorchFunctionMode:
    """Test TorchFunctionMode for bare torch functional ops."""

    def setup_method(self):
        disable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_activated(self):
        enable_io_inspect(torch_funcs=True)
        from vllm_fl.dispatch.io_inspector import _torch_func_mode_instance

        assert _torch_func_mode_instance is not None

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_deactivated_on_disable(self):
        enable_io_inspect(torch_funcs=True)
        disable_io_inspect()
        from vllm_fl.dispatch.io_inspector import _torch_func_mode_instance

        assert _torch_func_mode_instance is None

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_captures_matmul(self, caplog):
        import logging

        enable_io_inspect(ops={"matmul"}, torch_funcs=True)

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            a = torch.randn(2, 3)
            b = torch.randn(3, 4)
            torch.matmul(a, b)

        messages = " ".join(r.message for r in caplog.records)
        assert "matmul" in messages
        assert "INPUTS" in messages
        assert "OUTPUTS" in messages

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_activated_by_default(self):
        enable_io_inspect()  # torch_funcs=True by default
        from vllm_fl.dispatch.io_inspector import _torch_func_mode_instance

        assert _torch_func_mode_instance is not None

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_not_activated_when_disabled(self):
        enable_io_inspect(torch_funcs=False)
        from vllm_fl.dispatch.io_inspector import _torch_func_mode_instance

        assert _torch_func_mode_instance is None

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_no_infinite_recursion(self):
        """Ensure re-entrancy guard prevents infinite recursion."""
        enable_io_inspect(torch_funcs=True)
        # This should not hang or crash
        a = torch.randn(3, 3)
        b = torch.randn(3, 3)
        result = torch.matmul(a, b)
        assert result.shape == (3, 3)


class TestEnvVarTorchFuncs:
    """Test torch funcs env var initialization."""

    def teardown_method(self):
        disable_io_inspect()

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_INSPECT": "1",
            "VLLM_FL_IO_INSPECT_TORCH_FUNCS": "1",
        },
        clear=False,
    )
    def test_env_torch_funcs_all(self):
        io_inspector._init_from_env()
        assert io_inspector._torch_funcs_enabled
        assert io_inspector._torch_func_filter == set()

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_INSPECT": "1",
            "VLLM_FL_IO_INSPECT_TORCH_FUNCS": "matmul,softmax",
        },
        clear=False,
    )
    def test_env_torch_funcs_specific(self):
        io_inspector._init_from_env()
        assert io_inspector._torch_funcs_enabled
        assert io_inspector._torch_func_filter == {"matmul", "softmax"}

    @patch.dict(os.environ, {"VLLM_FL_IO_INSPECT": "1"}, clear=False)
    def test_env_torch_funcs_unset_match_all(self):
        os.environ.pop("VLLM_FL_IO_INSPECT_TORCH_FUNCS", None)
        io_inspector._init_from_env()
        # torch_funcs defaults to True in match-all mode
        assert io_inspector._torch_funcs_enabled

    @patch.dict(os.environ, {"VLLM_FL_IO_INSPECT": "rms_norm"}, clear=False)
    def test_env_torch_funcs_unset_filtered(self):
        os.environ.pop("VLLM_FL_IO_INSPECT_TORCH_FUNCS", None)
        io_inspector._init_from_env()
        # torch_funcs defaults to False when specific ops are filtered
        assert not io_inspector._torch_funcs_enabled


class TestExecOrder:
    """Test execution order tracking in inspect output."""

    def setup_method(self):
        disable_io_inspect()
        reset_exec_order()

    def teardown_method(self):
        disable_io_inspect()
        reset_exec_order()

    def test_op_tag_in_log(self, caplog):
        import logging

        enable_io_inspect(torch_funcs=False)
        reset_exec_order()

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            t = torch.zeros(2, 3)
            inspect_before("test_op", (t,), {})
            inspect_after("test_op", (t,), t)

        # Check that op counter tag appears in consolidated log
        assert any("[op=" in r.message for r in caplog.records)

    def test_exec_order_increments_across_ops(self):
        reset_exec_order()
        o1 = next_exec_order()
        o2 = next_exec_order()
        o3 = next_exec_order()
        assert o1 == 1
        assert o2 == 2
        assert o3 == 3


class TestYamlConfig:
    """Test YAML config parsing for IO inspect."""

    def setup_method(self):
        disable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    def test_parse_io_config_inspect_section(self):
        cfg_content = """
io_inspect:
  enabled: true
  ops:
    - rms_norm
  modules:
    - Linear
  torch_funcs: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            inspect_cfg = result["io_inspect"]
            assert inspect_cfg["enabled"] is True
            assert inspect_cfg["ops"] == {"rms_norm"}
            assert inspect_cfg["modules"] == {"Linear"}
            tf_enabled, tf_filter = inspect_cfg["torch_funcs"]
            assert tf_enabled is True
        finally:
            os.unlink(cfg_path)

    def test_parse_io_config_both_sections(self):
        cfg_content = """
io_inspect:
  enabled: true
  ops:
    - rms_norm
io_dump:
  dir: /tmp/test
  ops:
    - silu_and_mul
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            assert "io_inspect" in result
            assert "io_dump" in result
            assert result["io_inspect"]["ops"] == {"rms_norm"}
            assert result["io_dump"]["ops"] == {"silu_and_mul"}
        finally:
            os.unlink(cfg_path)

    @patch.dict(os.environ, {}, clear=False)
    def test_init_from_yaml_config(self):
        cfg_content = """
io_inspect:
  enabled: true
  ops:
    - rms_norm
  modules:
    - Linear
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            os.environ.pop("VLLM_FL_IO_INSPECT", None)
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": cfg_path}, clear=False):
                io_inspector._init_from_env()
            assert is_inspect_enabled()
            assert io_inspector._op_filter == {"rms_norm"}
            assert io_inspector._module_filter == {"Linear"}
        finally:
            os.unlink(cfg_path)

    @patch.dict(os.environ, {}, clear=False)
    def test_yaml_config_disabled(self):
        cfg_content = """
io_inspect:
  enabled: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            os.environ.pop("VLLM_FL_IO_INSPECT", None)
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": cfg_path}, clear=False):
                io_inspector._init_from_env()
            assert not is_inspect_enabled()
        finally:
            os.unlink(cfg_path)


class TestExecOrderParam:
    """Test that pre-allocated exec_order is respected."""

    def setup_method(self):
        disable_io_inspect()
        reset_exec_order()

    def teardown_method(self):
        disable_io_inspect()
        reset_exec_order()

    def test_inspect_before_uses_op_tag(self, caplog):
        import logging

        enable_io_inspect()

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            inspect_before("op", (torch.zeros(2),), {}, exec_order=42)
            inspect_after("op", (torch.zeros(2),), torch.zeros(2))

        assert any("[op=" in r.message for r in caplog.records)

    def test_inspect_after_uses_op_tag(self, caplog):
        import logging

        enable_io_inspect()

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            # Without prior inspect_before, falls back to outputs-only
            inspect_after("op", (), torch.zeros(2))

        assert any("[op=" in r.message for r in caplog.records)

    def test_exec_order_none_allocates_internally(self):
        enable_io_inspect(torch_funcs=False)
        reset_exec_order()
        t = torch.zeros(2)
        inspect_before("op", (t,), {})
        inspect_after("op", (t,), t)
        from vllm_fl.dispatch._io_common import get_exec_order

        assert get_exec_order() >= 1


class TestRankCommon:
    """Test rank detection and filtering utilities in _io_common."""

    def setup_method(self):
        reset_rank()
        io_inspector._rank_filter = None

    def teardown_method(self):
        reset_rank()
        io_inspector._rank_filter = None
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)

    def test_get_rank_default(self):
        assert get_rank() == 0

    @patch.dict(os.environ, {"RANK": "3"}, clear=False)
    def test_get_rank_from_env(self):
        reset_rank()
        assert get_rank() == 3

    @patch.dict(os.environ, {"LOCAL_RANK": "5"}, clear=False)
    def test_get_rank_from_local_rank(self):
        os.environ.pop("RANK", None)
        reset_rank()
        assert get_rank() == 5

    def test_rank_cached(self):
        reset_rank()
        r1 = get_rank()
        r2 = get_rank()
        assert r1 == r2

    def test_rank_filter_all(self):
        io_inspector._rank_filter = None
        assert io_inspector._rank_ok()

    def test_rank_filter_matching(self):
        io_inspector._rank_filter = {0}
        reset_rank()
        assert io_inspector._rank_ok()

    @patch.dict(os.environ, {"RANK": "1"}, clear=False)
    def test_rank_filter_not_matching(self):
        reset_rank()
        io_inspector._rank_filter = {0}
        assert not io_inspector._rank_ok()

    def test_parse_rank_filter_all(self):
        assert parse_rank_filter("all") is None
        assert parse_rank_filter("") is None

    def test_parse_rank_filter_single(self):
        assert parse_rank_filter("0") == {0}

    def test_parse_rank_filter_multiple(self):
        assert parse_rank_filter("0,2,4") == {0, 2, 4}

    def test_parse_rank_filter_whitespace(self):
        assert parse_rank_filter(" 0 , 1 ") == {0, 1}

    def test_parse_rank_filter_invalid(self):
        assert parse_rank_filter("abc") is None


class TestRankInInspectorLogs:
    """Test that rank appears in inspector log output."""

    def setup_method(self):
        disable_io_inspect()
        reset_rank()
        io_inspector._rank_filter = None

    def teardown_method(self):
        disable_io_inspect()
        reset_rank()
        io_inspector._rank_filter = None

    def test_rank_in_input_log(self, caplog):
        import logging

        enable_io_inspect()

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            t = torch.zeros(2)
            inspect_before("op", (t,), {})
            inspect_after("op", (t,), t)

        assert any("[rank=0]" in r.message for r in caplog.records)

    def test_rank_in_output_log(self, caplog):
        import logging

        enable_io_inspect()

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            inspect_after("op", (), torch.zeros(2))

        assert any("[rank=0]" in r.message for r in caplog.records)

    @patch.dict(os.environ, {"RANK": "7"}, clear=False)
    def test_rank_in_log_nonzero(self, caplog):
        import logging

        reset_rank()
        enable_io_inspect()

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            t = torch.zeros(2)
            inspect_before("op", (t,), {})
            inspect_after("op", (t,), t)

        assert any("[rank=7]" in r.message for r in caplog.records)


class TestRankFilterInInspector:
    """Test rank filtering prevents inspection on non-matching ranks."""

    def setup_method(self):
        disable_io_inspect()
        reset_rank()
        io_inspector._rank_filter = None

    def teardown_method(self):
        disable_io_inspect()
        reset_rank()
        io_inspector._rank_filter = None
        os.environ.pop("RANK", None)

    def test_rank_filter_blocks_inspection(self, caplog):
        import logging

        enable_io_inspect(ranks={1, 2})  # current rank is 0
        reset_rank()

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            t = torch.zeros(2)
            inspect_before("op", (t,), {})
            inspect_after("op", (t,), t)

        assert not any("INPUTS" in r.message for r in caplog.records)

    def test_rank_filter_allows_matching(self, caplog):
        import logging

        reset_rank()
        enable_io_inspect(ranks={0})

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            t = torch.zeros(2)
            inspect_before("op", (t,), {})
            inspect_after("op", (t,), t)

        assert any("INPUTS" in r.message for r in caplog.records)

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_INSPECT": "1", "VLLM_FL_IO_RANK": "0"},
        clear=False,
    )
    def test_env_rank_filter(self):
        reset_rank()
        io_inspector._init_from_env()
        assert is_inspect_enabled()
        assert io_inspector._rank_ok()

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_INSPECT": "1", "VLLM_FL_IO_RANK": "3"},
        clear=False,
    )
    def test_env_rank_filter_blocks(self):
        reset_rank()
        io_inspector._init_from_env()
        assert is_inspect_enabled()
        assert not io_inspector._rank_ok()


class TestYamlRanksConfig:
    """Test YAML ranks field parsing."""

    def setup_method(self):
        disable_io_inspect()
        reset_rank()
        io_inspector._rank_filter = None

    def teardown_method(self):
        disable_io_inspect()
        reset_rank()
        io_inspector._rank_filter = None

    def test_yaml_ranks_list(self):
        cfg_content = """
io_inspect:
  enabled: true
  ranks: [0, 1]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            assert result["io_inspect"]["ranks"] == {0, 1}
        finally:
            os.unlink(cfg_path)

    def test_yaml_ranks_single(self):
        cfg_content = """
io_inspect:
  enabled: true
  ranks: 0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            assert result["io_inspect"]["ranks"] == {0}
        finally:
            os.unlink(cfg_path)

    @patch.dict(os.environ, {}, clear=False)
    def test_init_from_yaml_with_ranks(self):
        reset_rank()
        cfg_content = """
io_inspect:
  enabled: true
  ranks: [0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            os.environ.pop("VLLM_FL_IO_INSPECT", None)
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": cfg_path}, clear=False):
                io_inspector._init_from_env()
            assert is_inspect_enabled()
            assert io_inspector._rank_ok()
        finally:
            os.unlink(cfg_path)
            io_inspector._rank_filter = None


class TestStepSummary:
    """Test that advance_step logs a summary of seen modules and ops."""

    def test_step_advance_logs_summary(self, caplog):
        """After inspect_before/inspect_after, advance_step should log a summary."""
        enable_io_inspect()
        try:
            t = torch.zeros(2)
            inspect_before("test_op", (t,), {})
            inspect_after("test_op", (t,), t)
            caplog.clear()
            with caplog.at_level("INFO"):
                advance_step()
            summary_msgs = [
                r.message for r in caplog.records if "Step summary" in r.message
            ]
            assert len(summary_msgs) == 1
            assert "test_op" in summary_msgs[0]
        finally:
            disable_io_inspect()
            reset_step()

    def test_no_summary_when_no_ops(self, caplog):
        """If no ops were inspected, no summary should be logged."""
        enable_io_inspect()
        try:
            caplog.clear()
            with caplog.at_level("INFO"):
                advance_step()
            summary_msgs = [
                r.message for r in caplog.records if "Step summary" in r.message
            ]
            assert len(summary_msgs) == 0
        finally:
            disable_io_inspect()
            reset_step()


class TestLayerFilter:
    """Test layer path filtering for IO inspector."""

    def setup_method(self):
        disable_io_inspect()
        # Build a simple model and register its paths
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        register_module_paths(self.model)

    def teardown_method(self):
        disable_io_inspect()

    def test_layer_filter_state(self):
        enable_io_inspect(layers={"model.layers.0"})
        assert io_inspector._layer_filter == {"model.layers.0"}

    def test_layer_filter_resets(self):
        enable_io_inspect(layers={"model.layers.0"})
        disable_io_inspect()
        assert io_inspector._layer_filter == set()

    def test_layer_path_matches_exact(self):
        """Test exact path matching."""
        push_module_context("Linear", self.model[0])
        try:
            assert layer_path_matches({"0"})
        finally:
            pop_module_context()

    def test_layer_path_matches_prefix(self):
        """Test prefix matching with dot boundary."""
        # "0" is the path for self.model[0] (Sequential names children 0, 1, 2)
        push_module_context("Linear", self.model[0])
        try:
            # "0" is the exact path, so it should match
            assert layer_path_matches({"0"})
        finally:
            pop_module_context()

    def test_layer_path_no_match(self):
        """Test that non-matching prefix is rejected."""
        push_module_context("Linear", self.model[0])
        try:
            assert not layer_path_matches({"model.layers.99"})
        finally:
            pop_module_context()

    def test_layer_path_no_partial_segment_match(self):
        """Prefix 'model.layers.0' should NOT match 'model.layers.00'."""
        push_module_context("Linear", self.model[0])
        try:
            # Path for model[0] is "0", should not match prefix "00"
            assert not layer_path_matches({"00"})
        finally:
            pop_module_context()

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_INSPECT": "1", "VLLM_FL_IO_INSPECT_LAYERS": "0"},
        clear=False,
    )
    def test_env_layer_filter(self):
        io_inspector._init_from_env()
        assert io_inspector._layer_filter == {"model.layers.0"}

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_INSPECT": "1",
            "VLLM_FL_IO_LAYERS": "model.layers.0,model.layers.1",
        },
        clear=False,
    )
    def test_env_shared_layer_filter(self):
        os.environ.pop("VLLM_FL_IO_INSPECT_LAYERS", None)
        io_inspector._init_from_env()
        assert io_inspector._layer_filter == {"model.layers.0", "model.layers.1"}

    @pytest.mark.skipif(
        not HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_layer_filter_acquires_module_hooks(self):
        """Layer filter needs module hooks for path tracking."""
        enable_io_inspect(ops={"rms_norm"}, layers={"model.layers.0"})
        assert io_inspector._owns_global_hooks is True

    def test_layer_shorthand_expansion(self):
        """Integer shorthand and ranges should be expanded."""
        enable_io_inspect(layers={"0", "2-4"})
        assert "model.layers.0" in io_inspector._layer_filter
        assert "model.layers.2" in io_inspector._layer_filter
        assert "model.layers.3" in io_inspector._layer_filter
        assert "model.layers.4" in io_inspector._layer_filter

    def test_layer_glob_pattern(self):
        """Glob patterns should be kept as-is."""
        enable_io_inspect(layers={"model.layers.*.self_attn"})
        assert "model.layers.*.self_attn" in io_inspector._layer_filter

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_INSPECT": "1", "VLLM_FL_IO_INSPECT_LAYERS": "0,1-2"},
        clear=False,
    )
    def test_env_layer_expansion(self):
        """Env var layer specs should be expanded."""
        io_inspector._init_from_env()
        assert "model.layers.0" in io_inspector._layer_filter
        assert "model.layers.1" in io_inspector._layer_filter
        assert "model.layers.2" in io_inspector._layer_filter


class TestComposableFilters:
    """Test AND-based composable filter logic."""

    def setup_method(self):
        disable_io_inspect()

    def teardown_method(self):
        disable_io_inspect()

    def test_op_and_module_filter(self):
        """When both op and module filter are set, both must match (AND)."""
        enable_io_inspect(ops={"rms_norm"}, modules={"Linear"})
        m = torch.nn.Linear(10, 10)
        # rms_norm inside Linear → True
        assert _should_inspect("rms_norm", (m,))
        # silu_and_mul inside Linear → False (op doesn't match)
        assert not _should_inspect("silu_and_mul", (m,))
        # rms_norm outside Linear → False (module doesn't match)
        assert not _should_inspect("rms_norm", (torch.zeros(2),))

    def test_op_filter_only(self):
        """When only op filter is set, only op must match."""
        enable_io_inspect(ops={"rms_norm"})
        assert _should_inspect("rms_norm", ())
        assert not _should_inspect("silu_and_mul", ())

    def test_module_filter_only(self):
        """When only module filter is set, only module must match."""
        enable_io_inspect(modules={"Linear"})
        m = torch.nn.Linear(10, 10)
        assert _should_inspect("any_op", (m,))
        assert not _should_inspect("any_op", (torch.zeros(2),))
