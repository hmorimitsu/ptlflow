import torch
from .kv_memory_store import KeyValueMemoryStore
from .MemFlowNet.memory_util import *


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """

    def __init__(self, config):
        self.cfg = config
        self.enable_long_term = config.enable_long_term
        self.enable_long_term_usage = config.enable_long_term_count_usage
        # top_k for softmax
        self.top_k = config.top_k
        self.max_mt_frames = config.max_mid_term_frames
        self.min_mt_frames = config.min_mid_term_frames

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        self.work_mem = KeyValueMemoryStore(count_usage=self.enable_long_term)
        self.reset_config = True

    def _readout(self, affinity, v):
        # this function is for a single object group
        return v @ affinity

    def match_memory(self, query_key, current_key, current_value, scale):
        # query_key: B x C^k x H x W
        h, w = query_key.shape[-2:]

        query_key = query_key.flatten(start_dim=2)
        if current_key is not None and current_value is not None:
            current_key = current_key.flatten(start_dim=2)
            current_value = current_value.flatten(start_dim=2)

        """
        Memory readout using keys
        """
        if self.work_mem.engaged():
            # No long-term memory
            if current_key is not None and current_value is not None:
                memory_key = torch.cat([self.work_mem.key, current_key], -1)
            else:
                memory_key = self.work_mem.key

            scale = scale * math.log(memory_key.shape[-1], self.cfg.train_avg_length)
            similarity = (
                torch.einsum("b c l, b c t -> b t l", query_key, memory_key) * scale
            )

            affinity, usage = do_softmax(
                similarity, inplace=True, top_k=self.top_k, return_usage=True
            )

            if current_key is not None and current_value is not None:
                all_memory_value = torch.cat([self.work_mem.value, current_value], -1)
                work_usage = usage[:, : -h * w]
            else:
                all_memory_value = self.work_mem.value
                work_usage = usage
            # Record memory usage for working memory
            self.work_mem.update_usage(work_usage.flatten())
            # else:
            #     raise NotImplementedError
        else:
            # No working-term memory
            if current_key is not None and current_value is not None:
                memory_key = current_key
                scale = scale * math.log(
                    memory_key.shape[-1], self.cfg.train_avg_length
                )
                similarity = (
                    torch.einsum("b c l, b c t -> b t l", query_key, memory_key) * scale
                )
                affinity = do_softmax(
                    similarity, inplace=True, top_k=None, return_usage=False
                )
                all_memory_value = current_value
            else:
                return 0

        # Shared affinity within each group
        all_readout_mem = self._readout(affinity, all_memory_value)

        return all_readout_mem.view(all_readout_mem.shape[0], -1, h, w)

    def add_memory(self, key, value):
        # key: 1*C*H*W
        # value: 1*C*H*W
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H * self.W
            self.min_work_elements = self.min_mt_frames * self.HW
            self.max_work_elements = self.max_mt_frames * self.HW

        # key:   1*C*N
        # value: 1*C*N
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        self.work_mem.add(key, value)

        # Do memory compressed if needed
        if self.work_mem.size >= self.max_work_elements:
            # Remove obsolete features if needed
            self.compress_features()

    def compress_features(self):
        # remove consolidated working memory
        self.work_mem.sieve_by_range(0, -self.min_work_elements)
