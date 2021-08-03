========================
List of available models
========================

Below is a list and a brief explanation about the models currently available on PTLFlow.

List of models
==============

DICL-Flow
---------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/dicl>`__

- Paper: **Displacement-Invariant Matching Cost Learning for Accurate Optical Flow Estimation** - `https://arxiv.org/abs/2010.14851 <https://arxiv.org/abs/2010.14851>`_

- Reference code: `https://github.com/jytime/DICL-Flow <https://github.com/jytime/DICL-Flow>`_

- Model names: ``dicl``

Flownet
-------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/flownet>`__

- Papers:

  - **FlowNet: Learning Optical Flow with Convolutional Networks** - `https://arxiv.org/abs/1504.06852 <https://arxiv.org/abs/1504.06852>`_

  - **FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks** - `https://arxiv.org/abs/1612.01925 <https://arxiv.org/abs/1612.01925>`_

- Reference code: `https://github.com/NVIDIA/flownet2-pytorch <https://github.com/NVIDIA/flownet2-pytorch>`_

- Model names: ``flownets``, ``flownetc``, ``flownet2``, ``flownetcs``, ``flownetcss``, ``flownetsd``

HD3
---

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/hd3>`__

- Paper: **Hierarchical Discrete Distribution Decomposition for Match Density Estimation** - `https://arxiv.org/abs/1812.06264 <https://arxiv.org/abs/1812.06264>`_

- Reference code: `https://github.com/ucbdrive/hd3 <https://github.com/ucbdrive/hd3>`_

- Model names: ``hd3``, ``hd3_ctxt``


IRR
---

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/irr>`__

- Paper: **Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation** - `https://arxiv.org/abs/1904.05290 <https://arxiv.org/abs/1904.05290>`_

- Reference code: `https://github.com/visinf/irr <https://github.com/visinf/irr>`_

- Model names: ``irr_pwc``, ``irr_pwcnet``, ``irr_pwcnet_irr``


LCV
---

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/lcv>`__

- Paper: **Learnable Cost Volume Using the Cayley Representation** - `https://arxiv.org/abs/2007.11431 <https://arxiv.org/abs/2007.11431>`_

- Reference code: `https://github.com/Prinsphield/LCV <https://github.com/Prinsphield/LCV>`_

- Model names: ``lcv_raft``, ``lcv_raft_small``

LiteFlowNet
-----------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/liteflownet>`__

- Paper: **LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation** - `https://arxiv.org/abs/1805.07036 <https://arxiv.org/abs/1805.07036>`_

- Reference code: `https://github.com/twhui/LiteFlowNet <https://github.com/twhui/LiteFlowNet>`__

- Model name: ``liteflownet``

LiteFlowNet2
------------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/liteflownet>`__

- Paper: **A Lightweight Optical Flow CNN - Revisiting Data Fidelity and Regularization** - `https://ieeexplore.ieee.org/document/9018073 <https://ieeexplore.ieee.org/document/9018073>`_

- Reference code: `https://github.com/twhui/LiteFlowNet2 <https://github.com/twhui/LiteFlowNet2>`__

- Model names: ``liteflownet2``, ``liteflownet2_pseudoreg``

LiteFlowNet3
------------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/liteflownet>`__

- Paper: **LiteFlowNet3: Resolving Correspondence Ambiguity for More Accurate Optical Flow Estimation** - `https://arxiv.org/abs/2007.09319 <https://arxiv.org/abs/2007.09319>`_

- Reference code: `https://github.com/twhui/LiteFlowNet3 <https://github.com/twhui/LiteFlowNet3>`__

- Model names: ``liteflownet3``, ``liteflownet3_pseudoreg``, ``liteflownet3s``, ``liteflownet3s_pseudoreg``

MaskFlownet
-----------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/maskflownet>`__

- Paper: **MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask** - `https://arxiv.org/abs/2003.10955 <https://arxiv.org/abs/2003.10955>`_

- Reference code: `https://github.com/cattaneod/MaskFlownet-Pytorch <https://github.com/cattaneod/MaskFlownet-Pytorch>`__

- Model names: ``maskflownet``, ``maskflownet_s``

PWCNet
------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/pwcnet>`__

- Paper: **PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume** - `https://arxiv.org/abs/1709.02371 <https://arxiv.org/abs/1709.02371>`_

- Reference code: `https://github.com/NVlabs/PWC-Net <https://github.com/NVlabs/PWC-Net>`_

- Model names: ``pwcnet``, ``pwcdcnet``

RAFT
----

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/raft>`__

- Paper: **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow** - `https://arxiv.org/abs/2003.12039 <https://arxiv.org/abs/2003.12039>`_

- Reference code: `https://github.com/princeton-vl/RAFT <https://github.com/princeton-vl/RAFT>`_

- Model names: ``raft``, ``raft_small``

ScopeFlow
---------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/scopeflow>`__

- Paper: **ScopeFlow: Dynamic Scene Scoping for Optical Flow** - `https://arxiv.org/abs/2002.10770 <https://arxiv.org/abs/2002.10770>`_

- Reference code: `https://github.com/avirambh/ScopeFlow <https://github.com/avirambh/ScopeFlow>`_

- Model names: ``scopeflow``

STaRFlow
--------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/starflow>`__

- Paper: **STaRFlow: A SpatioTemporal Recurrent Cell for Lightweight Multi-Frame Optical Flow Estimation** - `https://arxiv.org/abs/2007.05481 <https://arxiv.org/abs/2007.05481>`_

- Reference code: `https://github.com/pgodet/star_flow <https://github.com/pgodet/star_flow>`_

- Model names: ``starflow``

VCN
---

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/vcn>`__

- Paper: **Volumetric Correspondence Networks for Optical Flow** - `https://papers.nips.cc/paper/2019/file/bbf94b34eb32268ada57a3be5062fe7d-Paper.pdf <https://papers.nips.cc/paper/2019/file/bbf94b34eb32268ada57a3be5062fe7d-Paper.pdf>`_

- Reference code: `https://github.com/gengshan-y/VCN <https://github.com/gengshan-y/VCN>`_

- Model names: ``vcn``, ``vcn_small``