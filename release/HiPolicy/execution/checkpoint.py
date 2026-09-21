"""Load original policy weights using their saved architecture and normalization."""


def load_training_payload(path, cfg):
    """Reject incompatible resumption before constructing or mutating a trainer."""
    import dill
    import torch
    from omegaconf import OmegaConf

    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    if OmegaConf.to_container(payload["cfg"].policy, resolve=True) != OmegaConf.to_container(cfg.policy, resolve=True):
        raise ValueError("Resume policy configuration differs; use the checkpoint's architecture and scheduler")
    return payload


def load_policy(path, device="cuda:0", approved_scheduler=True):
    import dill
    import hydra
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from omegaconf import OmegaConf

    # Checkpoints are Python pickle files. Load only trusted experiment outputs.
    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    policy = hydra.utils.instantiate(cfg.policy)
    state_name = "ema_model" if cfg.training.use_ema else "model"
    if state_name not in payload["state_dicts"]:
        raise KeyError(f"Checkpoint requests {state_name}, but its state is absent")
    state = {}
    for key, value in payload["state_dicts"][state_name].items():
        # DDP/torch.compile wrappers do not change the underlying architecture.
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in state:
            raise ValueError(f"Checkpoint wrapper-prefix collision: {key}")
        state[key] = value
    # Old V1 checkpoints may predate registered, constant ImageNet buffers.
    # Fill only these exact constants, then use strict loading for everything.
    constants = {key: value for key, value in policy.state_dict().items()
                 if key.startswith("obs_encoder.key_transform_map.") and key.endswith((".2.mean", ".2.std"))}
    for key, value in constants.items():
        state.setdefault(key, value)
    policy.load_state_dict(state, strict=True)
    if not any(key.startswith("normalizer.params_dict.") for key in state):
        raise ValueError("Checkpoint does not contain the training normalizer")
    required = {"action", *cfg.shape_meta.obs.keys()}
    missing = required - set(policy.normalizer.params_dict.keys())
    if missing:
        raise ValueError(f"Checkpoint normalizer is missing fields: {sorted(missing)}")
    if approved_scheduler:
        # Preserve prediction_type and beta/clip settings in the original cfg.
        policy.noise_scheduler = DDPMScheduler.from_config(policy.noise_scheduler.config)
        policy.num_inference_steps = 100
    policy.to(device)
    policy.eval()
    return policy, cfg
