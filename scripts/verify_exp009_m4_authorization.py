import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from model_forensics.exp009_m4_authorization import verify_authorization

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-hosted-ci", action="store_true")
    args = parser.parse_args()
    request = verify_authorization(Path.cwd())
    if args.verify_hosted_ci:
        for evidence in request["successful_source_ci"]:
            run_id = int(evidence["run_id"])
            url = (
                "https://api.github.com/repos/Kushrishi/model-regression-forensics/"
                f"actions/runs/{run_id}"
            )
            api_request = Request(
                url,
                headers={
                    "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urlopen(api_request, timeout=30) as response:
                run = json.load(response)
            if (
                run["head_sha"] != request["source_git_sha"]
                or run["name"] != evidence["name"]
                or run["conclusion"] != "success"
                or run["event"] != "push"
            ):
                raise ValueError("live source CI identity or outcome mismatch")
    print(f"M4_SOURCE_GIT_SHA={request['source_git_sha']}")
    print("M4_AUTHORIZATION=PASS official_test_authorized=False restoration_authorized=False")
