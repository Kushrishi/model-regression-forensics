from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from model_forensics.exp009_m4_authorization import (
    PROTOCOL_PATHS,
    REQUEST_PATH,
    verify_authorization,
)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def authorized_repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "test")
    git(tmp_path, "config", "user.email", "test@example.com")
    hashes = {}
    for relative in (*PROTOCOL_PATHS, "src/implementation.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen\n")
        if relative in PROTOCOL_PATHS:
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "source")
    source = git(tmp_path, "rev-parse", "HEAD")
    request = {
        "schema_version": 1,
        "milestone": "M4_LOCALIZATION",
        "source_git_sha": source,
        "official_test_authorized": False,
        "restoration_authorized": False,
        "trajectories": [0, 1, 2],
        "worlds": [0, 1],
        "protocol_sha256": hashes,
        "successful_source_ci": [
            {"name": name, "head_sha": source, "conclusion": "success"}
            for name in ("CI", "Research CI")
        ],
    }
    (tmp_path / REQUEST_PATH).write_text(json.dumps(request))
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "authorize")
    return request


def test_source_authorization_accepts_request_only_commit(tmp_path):
    request = authorized_repository(tmp_path)
    assert verify_authorization(tmp_path) == request


def test_source_authorization_rejects_changed_code_even_with_valid_protocols(tmp_path):
    authorized_repository(tmp_path)
    (tmp_path / "src/implementation.py").write_text("changed\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "change")
    with pytest.raises(ValueError, match="differs from source pin"):
        verify_authorization(tmp_path)


def test_source_authorization_rejects_dirty_tracked_execution(tmp_path):
    authorized_repository(tmp_path)
    (tmp_path / "src/implementation.py").write_text("dirty\n")
    with pytest.raises(ValueError, match="dirty"):
        verify_authorization(tmp_path)


@pytest.mark.parametrize("field", ["official_test_authorized", "restoration_authorized"])
def test_authorization_rejects_scope_expansion(tmp_path, field):
    request = authorized_repository(tmp_path)
    request[field] = True
    (tmp_path / REQUEST_PATH).write_text(json.dumps(request))
    with pytest.raises(ValueError, match="unauthorized"):
        verify_authorization(tmp_path)
