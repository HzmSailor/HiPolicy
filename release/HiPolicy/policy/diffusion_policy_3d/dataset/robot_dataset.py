"""Stable import path for original checkpoint configurations."""
from importlib import import_module
from execution.profile import benchmark

RobotDataset = getattr(import_module("hipolicy_compat." + benchmark() + ".dp3_dataset"), "RobotDataset")
