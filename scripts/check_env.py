from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")

    try:
        import torch
    except ImportError:
        print("torch=not-installed (install research extra before Experiment 000)")
        return

    print(f"torch={torch.__version__}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    print(f"cuda_available={torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
