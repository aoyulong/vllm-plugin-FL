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
from vllm_fl.dispatch.io_inspector import (
    _format_result,
    _format_value,
    _get_module_name,
    _parse_config,
    _parse_torch_funcs_config,
    _should_inspect,
    _should_inspect_torch_func,
    _HAS_GLOBAL_MODULE_HOOKS,
    _HAS_TORCH_FUNC_MODE,
    attach_io_hooks,
    disable_io_inspect,
    enable_io_inspect,
    inspect_after,
    inspect_before,
    is_inspect_enabled,
    remove_io_hooks,
)
from vllm_fl.dispatch._io_common import (
    get_exec_order,
    next_exec_order,
    parse_io_config_from_yaml,
    reset_exec_order,
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
        inspect_all, ops, modules = _parse_config(
            "rms_norm,module:RotaryEmbeddingFL"
        )
        assert not inspect_all
        assert ops == {"rms_norm"}
        assert modules == {"RotaryEmbeddingFL"}

    def test_whitespace_handling(self):
        inspect_all, ops, modules = _parse_config(
            " silu_and_mul , module:RMSNormFL "
        )
        assert ops == {"silu_and_mul"}
        assert modules == {"RMSNormFL"}


class TestFormatValue:
    """Test _format_value formatting."""

    def test_tensor(self):
        t = torch.zeros(4, 512, dtype=torch.float16)
        result = _format_value(t)
        assert "shape=[4, 512]" in result
        assert "float16" in result

    def test_none(self):
        assert _format_value(None) == "None"

    def test_int(self):
        assert _format_value(42) == "42"

    def test_float(self):
        assert _format_value(1.5) == "1.5"

    def test_bool(self):
        assert _format_value(True) == "True"

    def test_small_tuple(self):
        result = _format_value((1, 2, 3))
        assert "tuple" in result

    def test_large_tuple(self):
        result = _format_value((1, 2, 3, 4, 5))
        assert "len=5" in result

    def test_module(self):
        m = torch.nn.Linear(10, 10)
        result = _format_value(m)
        assert "Linear" in result


class TestFormatResult:
    """Test _format_result formatting."""

    def test_single_tensor(self):
        t = torch.zeros(2, 3)
        result = _format_result(t)
        assert "result:" in result
        assert "shape=[2, 3]" in result

    def test_tuple_result(self):
        t1 = torch.zeros(2, 3)
        t2 = torch.ones(4, 5)
        result = _format_result((t1, t2))
        assert "result[0]:" in result
        assert "result[1]:" in result


class TestGetModuleName:
    """Test _get_module_name extraction."""

    def test_with_module(self):
        m = torch.nn.Linear(10, 10)
        assert _get_module_name((m, torch.zeros(2))) == "Linear"

    def test_without_module(self):
        assert _get_module_name((torch.zeros(2),)) is None

    def test_empty_args(self):
        assert _get_module_name(()) is None


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
        assert io_inspector._inspect_all

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
        # Should not raise
        inspect_before("test_op", (t,), {})

    def test_inspect_before_with_module(self):
        m = torch.nn.Linear(10, 10)
        t = torch.zeros(2, 10)
        inspect_before("test_op", (m, t), {"epsilon": 1e-6})

    def test_inspect_after_with_tensor(self):
        t = torch.zeros(2, 3)
        inspect_after("test_op", (), t)

    def test_inspect_after_with_tuple(self):
        t1 = torch.zeros(2, 3)
        t2 = torch.ones(4, 5)
        inspect_after("test_op", (), (t1, t2))

    def test_inspect_before_with_none_args(self):
        inspect_before("test_op", (None,), {})

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
    """Test _parse_torch_funcs_config parsing logic."""

    def test_empty_string(self):
        enabled, funcs = _parse_torch_funcs_config("")
        assert not enabled
        assert funcs == set()

    def test_zero_disables(self):
        enabled, funcs = _parse_torch_funcs_config("0")
        assert not enabled
        assert funcs == set()

    def test_one_enables_all(self):
        enabled, funcs = _parse_torch_funcs_config("1")
        assert enabled
        assert funcs == set()

    def test_specific_funcs(self):
        enabled, funcs = _parse_torch_funcs_config("matmul,softmax")
        assert enabled
        assert funcs == {"matmul", "softmax"}

    def test_whitespace(self):
        enabled, funcs = _parse_torch_funcs_config(" matmul , linear ")
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
        not _HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_auto_registered(self):
        enable_io_inspect(modules={"Linear"})
        from vllm_fl.dispatch.io_inspector import _global_hook_handles
        assert len(_global_hook_handles) == 2  # pre + post

    @pytest.mark.skipif(
        not _HAS_GLOBAL_MODULE_HOOKS,
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
        not _HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_removed_on_disable(self):
        enable_io_inspect()
        from vllm_fl.dispatch.io_inspector import _global_hook_handles
        assert len(_global_hook_handles) == 2
        disable_io_inspect()
        assert len(_global_hook_handles) == 0

    @pytest.mark.skipif(
        not _HAS_GLOBAL_MODULE_HOOKS,
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
        not _HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_activated(self):
        enable_io_inspect(torch_funcs=True)
        from vllm_fl.dispatch.io_inspector import _torch_func_mode_instance
        assert _torch_func_mode_instance is not None

    @pytest.mark.skipif(
        not _HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_deactivated_on_disable(self):
        enable_io_inspect(torch_funcs=True)
        disable_io_inspect()
        from vllm_fl.dispatch.io_inspector import _torch_func_mode_instance
        assert _torch_func_mode_instance is None

    @pytest.mark.skipif(
        not _HAS_TORCH_FUNC_MODE,
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
        not _HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_not_activated_by_default(self):
        enable_io_inspect()  # torch_funcs=False by default
        from vllm_fl.dispatch.io_inspector import _torch_func_mode_instance
        assert _torch_func_mode_instance is None

    @pytest.mark.skipif(
        not _HAS_TORCH_FUNC_MODE,
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
    def test_env_torch_funcs_unset(self):
        os.environ.pop("VLLM_FL_IO_INSPECT_TORCH_FUNCS", None)
        io_inspector._init_from_env()
        assert not io_inspector._torch_funcs_enabled


class TestExecOrder:
    """Test execution order tracking in inspect output."""

    def setup_method(self):
        disable_io_inspect()
        reset_exec_order()

    def teardown_method(self):
        disable_io_inspect()
        reset_exec_order()

    def test_exec_order_in_log(self, caplog):
        import logging

        enable_io_inspect()
        reset_exec_order()

        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_inspect"):
            t = torch.zeros(2, 3)
            inspect_before("test_op", (t,), {})

        # Check that exec order number appears in log
        assert any("#" in r.message for r in caplog.records)

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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            os.environ.pop("VLLM_FL_IO_INSPECT", None)
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": cfg_path}, clear=False):
                io_inspector._init_from_env()
            assert not is_inspect_enabled()
        finally:
            os.unlink(cfg_path)
