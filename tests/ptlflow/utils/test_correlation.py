# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

try:
    from spatial_correlation_sampler import spatial_correlation_sample
    import torch

    from ptlflow.utils.correlation import iter_spatial_correlation_sample

    def test_correlation() -> None:
        i1 = torch.arange(200000).view(2, 10, 100, 100).float() / 10000
        if torch.cuda.is_available():
            i1 = i1.cuda()
        i2 = i1.clone()

        test_params = [
            {'patch_size': (1, 1), 'stride': (1, 1), 'padding': (0, 0), 'dilation_patch': (1, 1)},
            {'patch_size': (3, 3), 'stride': (1, 1), 'padding': (0, 0), 'dilation_patch': (1, 1)},
            {'patch_size': (1, 1), 'stride': (2, 2), 'padding': (0, 0), 'dilation_patch': (1, 1)},
            {'patch_size': (1, 1), 'stride': (1, 1), 'padding': (1, 1), 'dilation_patch': (1, 1)},
            {'patch_size': (1, 1), 'stride': (1, 1), 'padding': (0, 0), 'dilation_patch': (2, 2)},
            {'patch_size': (5, 5), 'stride': (3, 3), 'padding': (0, 0), 'dilation_patch': (1, 1)},
            {'patch_size': (7, 7), 'stride': (1, 1), 'padding': (2, 2), 'dilation_patch': (1, 1)},
            {'patch_size': (9, 9), 'stride': (1, 1), 'padding': (0, 0), 'dilation_patch': (3, 3)},
            {'patch_size': (1, 1), 'stride': (2, 2), 'padding': (3, 3), 'dilation_patch': (1, 1)},
            {'patch_size': (1, 1), 'stride': (4, 4), 'padding': (0, 0), 'dilation_patch': (4, 4)},
            {'patch_size': (1, 1), 'stride': (1, 1), 'padding': (4, 4), 'dilation_patch': (5, 5)},
            {'patch_size': (3, 3), 'stride': (2, 2), 'padding': (1, 1), 'dilation_patch': (1, 1)},
            {'patch_size': (5, 5), 'stride': (2, 2), 'padding': (0, 0), 'dilation_patch': (2, 2)},
            {'patch_size': (7, 7), 'stride': (1, 1), 'padding': (1, 1), 'dilation_patch': (2, 2)},
            {'patch_size': (1, 1), 'stride': (2, 2), 'padding': (1, 1), 'dilation_patch': (2, 2)},
            {'patch_size': (9, 9), 'stride': (3, 3), 'padding': (5, 5), 'dilation_patch': (4, 4)}
        ]
        for p in test_params:
            cref = spatial_correlation_sample(i1, i2, **p)
            ctest = iter_spatial_correlation_sample(i1, i2, **p)
            diff = torch.abs(cref - ctest).max().item()
            assert diff < 10
except ModuleNotFoundError:
    pass
