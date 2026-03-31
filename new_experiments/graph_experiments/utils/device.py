"""
Device selection utility with CUDA, MPS (Apple Silicon), and CPU support.
"""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Select the best available device.

    Args:
        preference: "auto" (best available), "cuda", "mps", or "cpu".

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)
