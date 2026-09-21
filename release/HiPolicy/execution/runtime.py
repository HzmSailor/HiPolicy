"""Filesystem-only overrides for the original benchmark scripts."""
import json
import os
from pathlib import Path


def usr_variant():
    return os.environ["HIPOLICY_POLICY"]


def collection_config(config):
    config["save_path"] = os.environ["HIPOLICY_RAW_ROOT"]
    config["episode_num"] = int(os.environ["HIPOLICY_EPISODES"])
    camera = os.environ["HIPOLICY_CAMERA"]
    if "camera" in config:
        config["camera"]["head_camera_type"] = camera
    else:
        config["head_camera_type"] = camera
    # DP3 needs the existing point-cloud producer; its crop/color/FPS stay intact.
    if usr_variant() == "dp3":
        config["data_type"]["pointcloud"] = True
    output = Path(os.environ["HIPOLICY_RUN_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "collection_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def seed_file(task):
    destination = Path(os.environ["HIPOLICY_RAW_ROOT"]) / "seeds" / (task + ".txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Retain preselected seeds when a task explicitly requests them.
    original = Path("task_config/seeds") / (task + ".txt")
    if not destination.exists() and original.is_file():
        destination.write_bytes(original.read_bytes())
    return str(destination)


def conversion_paths():
    return os.environ["HIPOLICY_CONVERT_INPUT"], os.environ["HIPOLICY_CONVERT_OUTPUT"]
