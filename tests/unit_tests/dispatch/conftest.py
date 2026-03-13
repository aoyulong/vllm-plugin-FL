# Copyright (c) 2026 BAAI. All rights reserved.

"""
Shared fixtures for dispatch unit tests.

- Pre-mocks ``vllm_fl.utils`` to avoid ``flag_gems → triton → CUDA``
  dependency when running on CPU-only environments.
- Attaches caplog to dispatch loggers that have ``propagate=False``
  so that ``caplog``-based assertions work in IO inspector/dumper tests.
"""

import logging
import os
import sys
import types

import pytest

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

if "vllm_fl.utils" not in sys.modules:
    mock_utils = types.ModuleType("vllm_fl.utils")
    mock_utils.get_op_config = lambda: None  # type: ignore[attr-defined]
    sys.modules["vllm_fl.utils"] = mock_utils

if "vllm_fl" not in sys.modules:
    vllm_fl_mod = types.ModuleType("vllm_fl")
    vllm_fl_mod.__path__ = [os.path.join(_project_root, "vllm_fl")]  # type: ignore[attr-defined]
    vllm_fl_mod.__package__ = "vllm_fl"
    sys.modules["vllm_fl"] = vllm_fl_mod


@pytest.fixture(autouse=True)
def _attach_caplog_to_dispatch_loggers(caplog):
    """Attach caplog handler to dispatch loggers that have propagate=False."""
    logger_names = [
        "vllm_fl.dispatch.io_inspect",
        "vllm_fl.dispatch.io_dump",
    ]
    loggers = []
    for name in logger_names:
        lgr = logging.getLogger(name)
        if not lgr.propagate:
            lgr.addHandler(caplog.handler)
            loggers.append(lgr)
    yield
    for lgr in loggers:
        lgr.removeHandler(caplog.handler)
