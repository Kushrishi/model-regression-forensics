from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from model_forensics.config import load_experiment_config
from model_forensics.exp007 import EXP007_FROZEN_MANIFEST_SHA256


def _load(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate(payload: dict[str, Any], dose: int) -> None:
    if payload.get("experiment_id") != "exp007":
        raise ValueError("dose summary is not from Experiment 007")
    if payload.get("gate") != "calibration_target_dose":
        raise ValueError("unexpected calibration dose summary")
    if payload.get("target_dose") != dose:
        raise ValueError("calibration dose mismatch")
    if not isinstance(payload.get("qualified"), bool):
        raise ValueError("dose summary lacks qualification result")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the prospective Experiment 007 calibration selection."
    )
    parser.add_argument("--config", default="configs/exp007.yaml")
    parser.add_argument("--dose9-summary", required=True)
    parser.add_argument("--dose18-summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_experiment_config(config_path)

    if config.experiment_id != "exp007" or config.calibration is None:
        raise ValueError("selection freeze requires Experiment 007 config")

    dose9 = _load(args.dose9_summary)
    _validate(dose9, 9)

    observed = [9]
    selected: int | None = None
    dose18: dict[str, Any] | None = None

    if dose9["qualified"] is True:
        if args.dose18_summary is not None:
            raise ValueError("dose 18 must remain uninspected after dose 9 qualifies")
        selected = 9
    else:
        if args.dose18_summary is None:
            raise ValueError("dose 18 summary is required after dose 9 fails calibration")

        dose18 = _load(args.dose18_summary)
        _validate(dose18, 18)
        observed.append(18)

        if dose18["qualified"] is True:
            selected = 18

    payload = {
        "experiment_id": "exp007",
        "selection_rule": config.calibration.selection_rule,
        "observed_target_doses": observed,
        "selected_target_dose": selected,
        "certification_authorized": selected is not None,
        "manifest_sha256": EXP007_FROZEN_MANIFEST_SHA256,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "calibration_results": {
            "9": {
                "qualified": dose9["qualified"],
                "passing_worlds": dose9.get("passing_worlds"),
            },
            **(
                {
                    "18": {
                        "qualified": dose18["qualified"],
                        "passing_worlds": dose18.get("passing_worlds"),
                    }
                }
                if dose18 is not None
                else {}
            ),
        },
    }

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
