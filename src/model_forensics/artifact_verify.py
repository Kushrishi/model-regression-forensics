"""Explicit, bounded local payload checks; no execution or authenticity claims."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_forensics.incidents import Ledger, Truth

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_MAP_BYTES = 8 * 1024 * 1024


def references(ledger: Ledger, truth: Truth | None = None) -> dict[str, str]:
    incident = ledger.incident
    artifacts = [
        incident.good_release,
        incident.bad_release,
        incident.environment,
        incident.evaluation,
        *(c.visible_diff for c in incident.candidates),
        *incident.engineer_visible_evidence,
        *(r.artifact for r in ledger.runs),
        ledger.decision.policy,
    ]
    if truth is not None:
        artifacts.append(truth.provenance)
    result = {}
    for artifact in artifacts:
        previous = result.setdefault(artifact.identity, artifact.sha256)
        if previous != artifact.sha256:
            raise ValueError("conflicting hashes for one artifact identity")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate artifact-map key")
        result[key] = value
    return result


def read_map(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        raw = source.read(MAX_MAP_BYTES + 1)
    if len(raw) > MAX_MAP_BYTES:
        raise ValueError("artifact map exceeds input bound")
    result = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(result, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in result.items()
    ):
        raise ValueError("artifact map must contain identity-to-relative-path strings")
    return result


def _parts(path: str) -> list[str]:
    if not isinstance(path, str):
        raise ValueError("artifact path must be a string")
    parts = path.split("/")
    if "\\" in path or ":" in path or any(p in ("", ".", "..") for p in parts):
        raise ValueError("artifact paths must be canonical relative POSIX paths")
    return parts


def _hash_file(root_fd: int, parts: list[str], remaining: int) -> tuple[str, int]:
    directory = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("artifact must be a regular file")
            limit = min(MAX_FILE_BYTES, remaining)
            if info.st_size > limit:
                raise ValueError("artifact exceeds byte budget")
            hasher = hashlib.sha256()
            count = 0
            while chunk := source.read(min(1024 * 1024, limit - count + 1)):
                count += len(chunk)
                if count > limit:
                    raise ValueError("artifact exceeds byte budget")
                hasher.update(chunk)
            after = os.fstat(source.fileno())
            if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or count != info.st_size:
                raise ValueError("artifact changed during verification")
            return hasher.hexdigest(), count
    finally:
        os.close(directory)


def verify_payloads(
    ledger: Ledger, root: Path, paths: dict[str, str], truth: Truth | None = None
) -> dict:
    """Verify only explicitly mapped evidence, without printing payloads or paths.

    The root's parent directories are trusted. Descriptor traversal rejects
    symlinks inside the root. This is not a hostile-filesystem sandbox.
    """
    expected = references(ledger, truth)
    if set(paths) != set(expected):
        raise ValueError("artifact map must exactly cover referenced identities")
    parts = {identity: _parts(path) for identity, path in paths.items()}
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise ValueError("safe descriptor-based payload verification is unsupported")
    total = 0
    records = []
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for identity in sorted(expected):
                sha, size = _hash_file(root_fd, parts[identity], MAX_TOTAL_BYTES - total)
                if sha != expected[identity]:
                    raise ValueError("artifact SHA-256 mismatch")
                total += size
                records.append({"identity": identity, "sha256": sha, "bytes": size})
        finally:
            os.close(root_fd)
    except OSError as error:
        # Never echo local paths from OS exceptions into a potentially public report.
        raise ValueError("artifact inaccessible, nonregular, or symlinked") from error
    return {
        "schema_version": "payload-verification/0.1",
        "claim": "referenced_byte_identity_only",
        "total_bytes": total,
        "artifacts": records,
    }
