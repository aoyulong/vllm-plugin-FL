# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tests for IO Dumper module.
"""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
import torch

from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch._io_common import (
    get_exec_order,
    next_exec_order,
    parse_io_config_from_yaml,
    reset_exec_order,
)
from vllm_fl.dispatch.io_dumper import (
    _HAS_GLOBAL_MODULE_HOOKS,
    _HAS_TORCH_FUNC_MODE,
    _build_input_dict,
    _build_output_dict,
    _parse_torch_funcs_config,
    _serialize_value,
    _should_dump,
    _should_dump_torch_func,
    attach_dump_hooks,
    disable_io_dump,
    dump_after,
    dump_before,
    enable_io_dump,
    io_dump_step,
    is_dump_enabled,
    remove_dump_hooks,
)


@pytest.fixture
def dump_dir():
    """Create a temporary directory for dump files."""
    d = tempfile.mkdtemp(prefix="vllm_fl_dump_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestSerializeValue:
    """Test _serialize_value conversion."""

    def test_tensor_to_cpu(self):
        t = torch.zeros(2, 3, dtype=torch.float32)
        result = _serialize_value(t)
        assert isinstance(result, torch.Tensor)
        assert result.device == torch.device("cpu")
        assert result.shape == (2, 3)

    def test_none(self):
        assert _serialize_value(None) is None

    def test_scalar(self):
        assert _serialize_value(42) == 42
        assert _serialize_value(1.5) == 1.5

    def test_bool(self):
        assert _serialize_value(True) is True

    def test_module_to_string(self):
        m = torch.nn.Linear(10, 10)
        result = _serialize_value(m)
        assert isinstance(result, str)
        assert "Linear" in result

    def test_tuple(self):
        t = torch.zeros(2)
        result = _serialize_value((t, None, 42))
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], torch.Tensor)
        assert result[1] is None
        assert result[2] == 42

    def test_list(self):
        result = _serialize_value([1, 2, 3])
        assert isinstance(result, list)
        assert result == [1, 2, 3]


class TestBuildDicts:
    """Test _build_input_dict and _build_output_dict."""

    def test_input_dict_args(self):
        t = torch.zeros(2, 3)
        d = _build_input_dict((t, None, 1.0), {})
        assert "arg_0" in d
        assert isinstance(d["arg_0"], torch.Tensor)
        assert d["arg_1"] is None
        assert d["arg_2"] == 1.0

    def test_input_dict_kwargs(self):
        d = _build_input_dict((), {"epsilon": 1e-6, "inplace": True})
        assert d["kwarg_epsilon"] == 1e-6
        assert d["kwarg_inplace"] is True

    def test_output_dict_single(self):
        t = torch.ones(4, 5)
        d = _build_output_dict(t)
        assert "result" in d
        assert isinstance(d["result"], torch.Tensor)

    def test_output_dict_tuple(self):
        t1 = torch.zeros(2)
        t2 = torch.ones(3)
        d = _build_output_dict((t1, t2))
        assert "result_0" in d
        assert "result_1" in d


class TestShouldDump:
    """Test _should_dump filtering logic."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_disabled_returns_false(self):
        assert not _should_dump("rms_norm", ())

    def test_dump_all(self, dump_dir):
        enable_io_dump(dump_dir)
        assert _should_dump("rms_norm", ())
        assert _should_dump("silu_and_mul", ())

    def test_op_filter(self, dump_dir):
        enable_io_dump(dump_dir, ops={"rms_norm"})
        assert _should_dump("rms_norm", ())
        assert not _should_dump("silu_and_mul", ())

    def test_module_filter(self, dump_dir):
        enable_io_dump(dump_dir, modules={"Linear"})
        m = torch.nn.Linear(10, 10)
        assert _should_dump("any_op", (m,))
        assert not _should_dump("any_op", (torch.zeros(2),))

    def test_max_calls_limit(self, dump_dir):
        enable_io_dump(dump_dir, max_calls=2)
        # Simulate call counts
        io_dumper._call_counters["test_op"] = 2
        assert not _should_dump("test_op", ())
        # Different op should still work
        assert _should_dump("other_op", ())

    def test_step_range(self, dump_dir):
        enable_io_dump(dump_dir, step_range=(5, 10))
        io_dumper._step_counter = 3
        assert not _should_dump("test_op", ())
        io_dumper._step_counter = 5
        assert _should_dump("test_op", ())
        io_dumper._step_counter = 9
        assert _should_dump("test_op", ())
        io_dumper._step_counter = 10
        assert not _should_dump("test_op", ())


class TestDumpBeforeAfter:
    """Test dump_before and dump_after file creation."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_dump_creates_input_file(self, dump_dir):
        enable_io_dump(dump_dir)
        reset_exec_order()
        t = torch.zeros(2, 3)
        dump_before("test_op", (t,), {"epsilon": 1e-6})

        # Find the input file (filename includes exec order)
        step_dir = os.path.join(dump_dir, "step_0000", "test_op")
        assert os.path.isdir(step_dir)
        input_files = [f for f in os.listdir(step_dir) if f.endswith("_input.pt")]
        assert len(input_files) == 1
        assert "call_0001_input.pt" in input_files[0]

        data = torch.load(os.path.join(step_dir, input_files[0]), weights_only=False)
        assert "arg_0" in data
        assert isinstance(data["arg_0"], torch.Tensor)
        assert data["kwarg_epsilon"] == 1e-6
        # Check metadata
        assert "__meta__" in data
        assert data["__meta__"]["op_name"] == "test_op"
        assert data["__meta__"]["exec_order"] >= 1

    def test_dump_creates_output_file(self, dump_dir):
        enable_io_dump(dump_dir)
        reset_exec_order()
        t_in = torch.zeros(2, 3)
        # Call dump_before first to set call counter
        dump_before("test_op", (t_in,), {})

        t_out = torch.ones(2, 3)
        dump_after("test_op", (t_in,), t_out)

        step_dir = os.path.join(dump_dir, "step_0000", "test_op")
        output_files = [f for f in os.listdir(step_dir) if f.endswith("_output.pt")]
        assert len(output_files) == 1

        data = torch.load(os.path.join(step_dir, output_files[0]), weights_only=False)
        assert "result" in data
        assert "__meta__" in data

    def test_dump_tuple_output(self, dump_dir):
        enable_io_dump(dump_dir)
        reset_exec_order()
        t_in = torch.zeros(2)
        dump_before("test_op", (t_in,), {})

        t1 = torch.zeros(2)
        t2 = torch.ones(3)
        dump_after("test_op", (t_in,), (t1, t2))

        step_dir = os.path.join(dump_dir, "step_0000", "test_op")
        output_files = [f for f in os.listdir(step_dir) if f.endswith("_output.pt")]
        assert len(output_files) == 1
        data = torch.load(os.path.join(step_dir, output_files[0]), weights_only=False)
        assert "result_0" in data
        assert "result_1" in data

    def test_dump_skips_when_filtered(self, dump_dir):
        enable_io_dump(dump_dir, ops={"other_op"})
        t = torch.zeros(2, 3)
        dump_before("test_op", (t,), {})

        step_dir = os.path.join(dump_dir, "step_0000", "test_op")
        assert not os.path.exists(step_dir)


class TestIoDumpStep:
    """Test io_dump_step step counter management."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_step_increments(self, dump_dir):
        enable_io_dump(dump_dir)
        assert io_dumper._step_counter == 0
        new_step = io_dump_step()
        assert new_step == 1
        assert io_dumper._step_counter == 1

    def test_step_resets_call_counters(self, dump_dir):
        enable_io_dump(dump_dir)
        io_dumper._call_counters["test_op"] = 5
        io_dump_step()
        assert io_dumper._call_counters == {}

    def test_dump_files_in_different_steps(self, dump_dir):
        enable_io_dump(dump_dir)
        reset_exec_order()
        t = torch.zeros(2)

        # Step 0
        dump_before("test_op", (t,), {})
        step0_dir = os.path.join(dump_dir, "step_0000", "test_op")
        assert os.path.isdir(step0_dir)
        assert any(f.endswith("_input.pt") for f in os.listdir(step0_dir))

        # Step 1
        io_dump_step()
        dump_before("test_op", (t,), {})
        step1_dir = os.path.join(dump_dir, "step_0001", "test_op")
        assert os.path.isdir(step1_dir)
        assert any(f.endswith("_input.pt") for f in os.listdir(step1_dir))


class TestProgrammaticAPI:
    """Test enable/disable programmatic API."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_enable_disable(self, dump_dir):
        assert not is_dump_enabled()
        enable_io_dump(dump_dir)
        assert is_dump_enabled()
        disable_io_dump()
        assert not is_dump_enabled()

    def test_enable_with_filters(self, dump_dir):
        enable_io_dump(
            dump_dir,
            ops={"rms_norm"},
            modules={"Linear"},
            max_calls=10,
            step_range=(0, 100),
        )
        assert is_dump_enabled()
        assert io_dumper._op_filter == {"rms_norm"}
        assert io_dumper._module_filter == {"Linear"}
        assert io_dumper._max_calls == 10
        assert io_dumper._step_range == (0, 100)

    def test_disable_resets_everything(self, dump_dir):
        enable_io_dump(dump_dir, ops={"rms_norm"}, max_calls=5)
        io_dump_step()
        disable_io_dump()

        assert not is_dump_enabled()
        assert io_dumper._dump_dir == ""
        assert io_dumper._op_filter == set()
        assert io_dumper._max_calls == 0
        assert io_dumper._step_counter == 0


class TestEnvVarInit:
    """Test initialization from environment variables."""

    def teardown_method(self):
        disable_io_dump()

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump"},
        clear=False,
    )
    def test_env_basic(self):
        io_dumper._init_from_env()
        assert is_dump_enabled()
        assert io_dumper._dump_dir == "/tmp/test_dump"
        assert io_dumper._dump_all

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_OPS": "rms_norm,silu_and_mul",
        },
        clear=False,
    )
    def test_env_ops(self):
        io_dumper._init_from_env()
        assert io_dumper._op_filter == {"rms_norm", "silu_and_mul"}
        assert not io_dumper._dump_all

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_MODULES": "RMSNormFL",
        },
        clear=False,
    )
    def test_env_modules(self):
        io_dumper._init_from_env()
        assert io_dumper._module_filter == {"RMSNormFL"}

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_MAX_CALLS": "10",
        },
        clear=False,
    )
    def test_env_max_calls(self):
        io_dumper._init_from_env()
        assert io_dumper._max_calls == 10

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_STEP_RANGE": "5,15",
        },
        clear=False,
    )
    def test_env_step_range(self):
        io_dumper._init_from_env()
        assert io_dumper._step_range == (5, 15)

    @patch.dict(os.environ, {}, clear=False)
    def test_env_unset(self):
        os.environ.pop("VLLM_FL_IO_DUMP", None)
        os.environ.pop("VLLM_FL_IO_DUMP_OPS", None)
        os.environ.pop("VLLM_FL_IO_DUMP_MODULES", None)
        os.environ.pop("VLLM_FL_IO_DUMP_MAX_CALLS", None)
        os.environ.pop("VLLM_FL_IO_DUMP_STEP_RANGE", None)
        io_dumper._init_from_env()
        assert not is_dump_enabled()


class TestForwardDumpHooks:
    """Test nn.Module forward hook attachment for dumping."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_attach_hooks_all(self, dump_dir):
        enable_io_dump(dump_dir)
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        count = attach_dump_hooks(model)
        assert count >= 4  # Sequential + 3 children
        remove_dump_hooks()

    def test_attach_hooks_by_module_filter(self, dump_dir):
        enable_io_dump(dump_dir, modules={"Linear"})
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        count = attach_dump_hooks(model)
        assert count == 2  # Only the two Linear modules
        remove_dump_hooks()

    def test_hooks_create_dump_files(self, dump_dir):
        enable_io_dump(dump_dir, modules={"Linear"})
        model = torch.nn.Linear(4, 3)
        attach_dump_hooks(model)

        x = torch.randn(2, 4)
        model(x)

        # Check files were created
        step_dir = os.path.join(dump_dir, "step_0000")
        assert os.path.isdir(step_dir)
        # The label should be "Linear." (root module has empty name)
        # Find any .pt files
        found_pt = False
        for root, dirs, files in os.walk(step_dir):
            for f in files:
                if f.endswith(".pt"):
                    found_pt = True
        assert found_pt
        remove_dump_hooks()

    def test_attach_disabled_returns_zero(self):
        model = torch.nn.Linear(4, 3)
        count = attach_dump_hooks(model)
        assert count == 0

    def test_disable_removes_hooks(self, dump_dir):
        enable_io_dump(dump_dir)
        model = torch.nn.Linear(4, 3)
        attach_dump_hooks(model)
        disable_io_dump()
        from vllm_fl.dispatch.io_dumper import _hook_handles

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


class TestShouldDumpTorchFunc:
    """Test _should_dump_torch_func filtering logic."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_disabled_returns_false(self):
        assert not _should_dump_torch_func("matmul")

    def test_enabled_all(self, dump_dir):
        enable_io_dump(dump_dir, torch_funcs=True)
        assert _should_dump_torch_func("matmul")
        assert _should_dump_torch_func("softmax")

    def test_skips_dunder(self, dump_dir):
        enable_io_dump(dump_dir, torch_funcs=True)
        assert not _should_dump_torch_func("__add__")

    def test_skips_trivial_ops(self, dump_dir):
        enable_io_dump(dump_dir, torch_funcs=True)
        assert not _should_dump_torch_func("size")
        assert not _should_dump_torch_func("dim")

    def test_op_filter_match(self, dump_dir):
        enable_io_dump(dump_dir, ops={"matmul"}, torch_funcs=True)
        assert _should_dump_torch_func("matmul")
        assert not _should_dump_torch_func("softmax")


class TestGlobalModuleHooks:
    """Test automatic global module hook registration for dumping."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    @pytest.mark.skipif(
        not _HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_auto_registered(self, dump_dir):
        enable_io_dump(dump_dir, modules={"Linear"})
        from vllm_fl.dispatch.io_dumper import _global_hook_handles

        assert len(_global_hook_handles) == 2  # pre + post

    @pytest.mark.skipif(
        not _HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_create_files_without_attach(self, dump_dir):
        """Global hooks should dump files without calling attach_dump_hooks()."""
        enable_io_dump(dump_dir, modules={"Linear"})
        model = torch.nn.Linear(4, 3)

        x = torch.randn(2, 4)
        model(x)

        # Check files were created
        step_dir = os.path.join(dump_dir, "step_0000")
        assert os.path.isdir(step_dir)
        found_pt = False
        for root, dirs, files in os.walk(step_dir):
            for f in files:
                if f.endswith(".pt"):
                    found_pt = True
        assert found_pt

    @pytest.mark.skipif(
        not _HAS_GLOBAL_MODULE_HOOKS,
        reason="Global module hooks not available in this PyTorch version",
    )
    def test_global_hooks_removed_on_disable(self, dump_dir):
        enable_io_dump(dump_dir)
        from vllm_fl.dispatch.io_dumper import _global_hook_handles

        assert len(_global_hook_handles) == 2
        disable_io_dump()
        assert len(_global_hook_handles) == 0


class TestDumpTorchFunctionMode:
    """Test TorchFunctionMode for bare torch functional ops dumping."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    @pytest.mark.skipif(
        not _HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_activated(self, dump_dir):
        enable_io_dump(dump_dir, torch_funcs=True)
        from vllm_fl.dispatch.io_dumper import _torch_func_mode_instance

        assert _torch_func_mode_instance is not None

    @pytest.mark.skipif(
        not _HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_deactivated_on_disable(self, dump_dir):
        enable_io_dump(dump_dir, torch_funcs=True)
        disable_io_dump()
        from vllm_fl.dispatch.io_dumper import _torch_func_mode_instance

        assert _torch_func_mode_instance is None

    @pytest.mark.skipif(
        not _HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_dumps_matmul(self, dump_dir):
        enable_io_dump(dump_dir, ops={"matmul"}, torch_funcs=True)

        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        torch.matmul(a, b)

        # Check that files were created under torch.matmul directory
        step_dir = os.path.join(dump_dir, "step_0000")
        assert os.path.isdir(step_dir)
        found_pt = False
        for root, dirs, files in os.walk(step_dir):
            for f in files:
                if f.endswith(".pt"):
                    found_pt = True
        assert found_pt

    @pytest.mark.skipif(
        not _HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_not_activated_by_default(self, dump_dir):
        enable_io_dump(dump_dir)  # torch_funcs=False by default
        from vllm_fl.dispatch.io_dumper import _torch_func_mode_instance

        assert _torch_func_mode_instance is None

    @pytest.mark.skipif(
        not _HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_no_infinite_recursion(self, dump_dir):
        """Ensure re-entrancy guard prevents infinite recursion."""
        enable_io_dump(dump_dir, torch_funcs=True)
        a = torch.randn(3, 3)
        b = torch.randn(3, 3)
        result = torch.matmul(a, b)
        assert result.shape == (3, 3)


class TestEnvVarTorchFuncs:
    """Test torch funcs env var initialization for dumper."""

    def teardown_method(self):
        disable_io_dump()

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_TORCH_FUNCS": "1",
        },
        clear=False,
    )
    def test_env_torch_funcs_all(self):
        io_dumper._init_from_env()
        assert io_dumper._torch_funcs_enabled
        assert io_dumper._torch_func_filter == set()

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_TORCH_FUNCS": "matmul,softmax",
        },
        clear=False,
    )
    def test_env_torch_funcs_specific(self):
        io_dumper._init_from_env()
        assert io_dumper._torch_funcs_enabled
        assert io_dumper._torch_func_filter == {"matmul", "softmax"}

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump"},
        clear=False,
    )
    def test_env_torch_funcs_unset(self):
        os.environ.pop("VLLM_FL_IO_DUMP_TORCH_FUNCS", None)
        io_dumper._init_from_env()
        assert not io_dumper._torch_funcs_enabled


class TestExecOrder:
    """Test execution order tracking in dump files."""

    def setup_method(self):
        disable_io_dump()
        reset_exec_order()

    def teardown_method(self):
        disable_io_dump()
        reset_exec_order()

    def test_exec_order_increments(self):
        o1 = next_exec_order()
        o2 = next_exec_order()
        assert o2 == o1 + 1

    def test_exec_order_resets(self):
        next_exec_order()
        next_exec_order()
        reset_exec_order()
        assert get_exec_order() == 0

    def test_dump_files_contain_exec_order(self, dump_dir):
        enable_io_dump(dump_dir)
        reset_exec_order()
        t = torch.zeros(2, 3)
        dump_before("op_a", (t,), {})
        dump_before("op_b", (t,), {})

        # Check file names include order prefix
        for op in ["op_a", "op_b"]:
            op_dir = os.path.join(dump_dir, "step_0000", op)
            files = os.listdir(op_dir)
            assert any("order_" in f for f in files)

    def test_dump_metadata_has_exec_order(self, dump_dir):
        enable_io_dump(dump_dir)
        reset_exec_order()
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})

        step_dir = os.path.join(dump_dir, "step_0000", "test_op")
        input_files = [f for f in os.listdir(step_dir) if f.endswith("_input.pt")]
        data = torch.load(os.path.join(step_dir, input_files[0]), weights_only=False)
        assert data["__meta__"]["exec_order"] == 1

    def test_io_dump_step_resets_exec_order(self, dump_dir):
        enable_io_dump(dump_dir)
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})
        assert get_exec_order() >= 1
        io_dump_step()
        assert get_exec_order() == 0


class TestYamlConfig:
    """Test YAML config parsing for IO dump."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_parse_io_config_dump_section(self):
        cfg_content = """
io_dump:
  dir: /tmp/yaml_dump
  ops:
    - rms_norm
    - silu_and_mul
  modules:
    - Linear
  max_calls: 50
  step_range: [2, 10]
  torch_funcs: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            dump_cfg = result["io_dump"]
            assert dump_cfg["dir"] == "/tmp/yaml_dump"
            assert dump_cfg["ops"] == {"rms_norm", "silu_and_mul"}
            assert dump_cfg["modules"] == {"Linear"}
            assert dump_cfg["max_calls"] == 50
            assert dump_cfg["step_range"] == (2, 10)
            tf_enabled, tf_filter = dump_cfg["torch_funcs"]
            assert tf_enabled is True
            assert tf_filter == set()
        finally:
            os.unlink(cfg_path)

    def test_parse_io_config_missing_file(self):
        result = parse_io_config_from_yaml("/nonexistent/path.yaml")
        assert result == {}

    def test_parse_io_config_torch_funcs_list(self):
        cfg_content = """
io_dump:
  dir: /tmp/yaml_dump
  torch_funcs:
    - matmul
    - softmax
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            dump_cfg = result["io_dump"]
            tf_enabled, tf_filter = dump_cfg["torch_funcs"]
            assert tf_enabled is True
            assert tf_filter == {"matmul", "softmax"}
        finally:
            os.unlink(cfg_path)

    @patch.dict(os.environ, {}, clear=False)
    def test_init_from_yaml_config(self, dump_dir):
        cfg_content = f"""
io_dump:
  dir: {dump_dir}
  ops:
    - rms_norm
  max_calls: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            os.environ.pop("VLLM_FL_IO_DUMP", None)
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": cfg_path}, clear=False):
                io_dumper._init_from_env()
            assert is_dump_enabled()
            assert io_dumper._dump_dir == dump_dir
            assert io_dumper._op_filter == {"rms_norm"}
            assert io_dumper._max_calls == 10
        finally:
            os.unlink(cfg_path)
