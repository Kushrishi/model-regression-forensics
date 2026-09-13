from __future__ import annotations

import os
import platform
from types import SimpleNamespace

from model_forensics.training import _runtime_execution_metadata


class _FakeMPS:
    @staticmethod
    def is_built() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return True


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    __version__ = "test-torch"
    backends = SimpleNamespace(mps=_FakeMPS())
    cuda = _FakeCuda()

    @staticmethod
    def get_num_threads() -> int:
        return 8

    @staticmethod
    def get_num_interop_threads() -> int:
        return 16


def test_runtime_execution_metadata_is_hardware_neutral() -> None:
    metadata = _runtime_execution_metadata(
        _FakeTorch(),
        device="mps",
    )

    assert metadata["device"] == "mps"
    assert metadata["platform_system"] == platform.system()
    assert metadata["platform_release"] == platform.release()
    assert metadata["platform_machine"] == platform.machine()
    assert metadata["python_version"] == platform.python_version()
    assert metadata["cpu_count"] == os.cpu_count()

    assert metadata["torch_num_threads"] == 8
    assert metadata["torch_num_interop_threads"] == 16

    assert metadata["mps_built"] is True
    assert metadata["mps_available"] is True
    assert metadata["cuda_available"] is False
    assert metadata["torch"] == "test-torch"
