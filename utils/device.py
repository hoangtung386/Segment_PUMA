"""Device setup utilities.

IMPORTANT: early_device_setup() must be called BEFORE importing torch,
because CUDA_VISIBLE_DEVICES is only read by torch at import time.

Usage in scripts:
    from utils.device import early_device_setup
    early_device_setup()   # sets CUDA_VISIBLE_DEVICES before torch import

    import torch  # now sees the correct devices
"""
import os
import argparse


def early_device_setup():
    """Parse --devices from CLI and set CUDA_VISIBLE_DEVICES.

    Must be called before `import torch` to take effect.
    Uses a partial parser so it doesn't interfere with the main argparse.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--devices', type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
