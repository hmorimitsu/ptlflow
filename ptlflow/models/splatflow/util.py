import torch.nn.functional as F


class InputPadder:
    def __init__(self, dims, base=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // base) + 1) * base - self.ht) % base
        pad_wd = (((self.wd // base) + 1) * base - self.wd) % base
        self._pad = [0, pad_wd, 0, pad_ht]

    def pad(self, *inputs):
        outputs = []

        for x in inputs:
            bhw_mode = 0

            if len(x.shape) == 3:
                bhw_mode = 1
                x = x.unsqueeze(1)

            x = F.pad(x, self._pad, mode="replicate")
            if bhw_mode:
                x = x.squeeze(1)
            outputs.append(x)

        return outputs

    def unpad(self, *inputs):
        ht, wd = inputs[0].shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]

        return [x[..., c[0] : c[1], c[2] : c[3]] for x in inputs]
