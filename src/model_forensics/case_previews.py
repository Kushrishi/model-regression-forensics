"""Render bounded text or grayscale input previews without loading external assets."""

import base64
import html
import json
import math
import struct
import zlib
from pathlib import Path


def previews(directory: Path, case_ids: set[str]) -> dict[str, str]:
    path = directory / "case_inputs.json"
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        data = stream.read(8_000_001)
    if len(data) > 8_000_000:
        raise ValueError("case previews exceed 8 MB")
    records = json.loads(data)
    if not isinstance(records, dict) or not records.keys() <= case_ids:
        raise ValueError("preview contains unknown case IDs")
    rendered = {}
    for key, value in records.items():
        if not isinstance(value, dict):
            raise ValueError("preview must be an object")
        if set(value) == {"text"} and isinstance(value["text"], str):
            if len(value["text"]) > 10000:
                raise ValueError("text preview too long")
            rendered[key] = f"<span>{html.escape(value['text'])}</span>"
        elif set(value) == {"width", "height", "pixels"}:
            w, h, pixels = value["width"], value["height"], value["pixels"]
            if (
                type(w) is not int
                or type(h) is not int
                or not 1 <= w <= 64
                or not 1 <= h <= 64
                or not isinstance(pixels, list)
                or len(pixels) != w * h
                or any(
                    type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 255
                    for p in pixels
                )
            ):
                raise ValueError("invalid grayscale preview")

            def chunk(kind, payload):
                return (
                    struct.pack(">I", len(payload))
                    + kind
                    + payload
                    + struct.pack(">I", zlib.crc32(kind + payload))
                )

            raw = b"".join(
                b"\0" + bytes(round(p) for p in pixels[y * w : (y + 1) * w]) for y in range(h)
            )
            png = (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw))
                + chunk(b"IEND", b"")
            )
            encoded = base64.b64encode(png).decode("ascii")
            rendered[key] = (
                f'<img alt="Grayscale input preview" width="80" height="80" '
                f'style="image-rendering:pixelated" src="data:image/png;base64,{encoded}">'
            )
        else:
            raise ValueError("unsupported preview format")
    return rendered
