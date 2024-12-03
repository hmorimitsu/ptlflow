import torch
from torch.optim.lr_scheduler import OneCycleLR


def fetch_optimizer(
    model,
    optimizer_name,
    scheduler_name,
    canonical_lr,
    twins_lr_factor,
    adam_decay,
    adamw_decay,
    epsilon,
    num_steps,
    anneal_strategy,
):
    """Create the optimizer and learning rate scheduler"""
    optimizer = build_optimizer(
        model,
        optimizer_name,
        canonical_lr,
        twins_lr_factor,
        adam_decay,
        adamw_decay,
        epsilon,
    )
    scheduler = build_scheduler(
        scheduler_name,
        canonical_lr,
        twins_lr_factor,
        num_steps,
        anneal_strategy,
        optimizer,
    )

    return optimizer, scheduler


def build_optimizer(
    model,
    optimizer_name,
    canonical_lr,
    twins_lr_factor,
    adam_decay,
    adamw_decay,
    epsilon,
):
    name = optimizer_name
    lr = canonical_lr

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=adam_decay,
            eps=epsilon,
        )
    elif name == "adamw":
        if twins_lr_factor is not None:
            factor = twins_lr_factor
            print("[Decrease lr of pre-trained model by factor {}]".format(factor))
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "feat_encoder" not in n
                        and "context_encoder" not in n
                        and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if ("feat_encoder" in n or "context_encoder" in n)
                        and p.requires_grad
                    ],
                    "lr": lr * factor,
                },
            ]
            full = [n for n, _ in model.named_parameters()]
            return torch.optim.AdamW(
                param_dicts, lr=lr, weight_decay=adamw_decay, eps=epsilon
            )
        else:
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=adamw_decay,
                eps=epsilon,
            )
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(
    scheduler_name, canonical_lr, twins_lr_factor, num_steps, anneal_strategy, optimizer
):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
        }
    """
    name = scheduler_name
    lr = canonical_lr

    if name == "OneCycleLR":
        if twins_lr_factor is not None:
            factor = twins_lr_factor
            scheduler = OneCycleLR(
                optimizer,
                [lr, lr * factor],
                num_steps + 100,
                pct_start=0.05,
                cycle_momentum=False,
                anneal_strategy=anneal_strategy,
            )
        else:
            scheduler = OneCycleLR(
                optimizer,
                lr,
                num_steps + 100,
                pct_start=0.05,
                cycle_momentum=False,
                anneal_strategy=anneal_strategy,
            )
    else:
        raise NotImplementedError()

    return scheduler
