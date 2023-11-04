def torch_init_model(model, total_dict, key, rank=0):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix="")

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print("unexpected keys:{}".format(unexpected_keys))
        print("error msgs:{}".format(error_msgs))
