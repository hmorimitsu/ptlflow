import torch
def build_flowformer(cfg):
    name = cfg.transformer 
    if name == "percostformer3":
        from .PerCostFormer3.transformer import FlowFormer
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid optimizer!")

    return FlowFormer(cfg[name])
