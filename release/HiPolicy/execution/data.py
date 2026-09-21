"""Publish converted datasets only after the original converter succeeds."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


def validate_dataset(path, policy, episodes):
    """Check store metadata without loading or changing demonstration arrays."""
    import numpy as np
    import zarr

    store = zarr.open(str(path), mode="r")
    ends = store["meta/episode_ends"][:]
    if (ends.shape != (episodes,) or not np.issubdtype(ends.dtype, np.integer)
            or np.any(ends <= 0) or np.any(np.diff(ends) <= 0)):
        raise ValueError("Converted episode boundaries are incomplete or invalid")
    length = int(ends[-1])
    for key in ("state", "action"):
        if store[f"data/{key}"].shape != (length, 14):
            raise ValueError(f"Expected {length} frames of 14D {key}")
    key = "head_camera" if policy == "dp" else "point_cloud"
    shape = store[f"data/{key}"].shape
    if shape[0] != length or len(shape) != (4 if policy == "dp" else 3):
        raise ValueError(f"Converted {key} does not align with state/action frames")


@contextmanager
def conversion_output(destination, policy, episodes):
    """Stage on the same filesystem and remove partial output after failure."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Dataset already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        previous = os.environ.get("HIPOLICY_CONVERT_OUTPUT")
        os.environ["HIPOLICY_CONVERT_OUTPUT"] = str(staged)
        try:
            yield staged
            validate_dataset(staged, policy, episodes)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Dataset already exists: {destination}")
            staged.rename(destination)
        finally:
            if previous is None:
                os.environ.pop("HIPOLICY_CONVERT_OUTPUT", None)
            else:
                os.environ["HIPOLICY_CONVERT_OUTPUT"] = previous
